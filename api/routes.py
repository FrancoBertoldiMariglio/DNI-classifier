import os
import base64
import io
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException, Depends, FastAPI
import logging
from PIL import Image
import torch
import numpy as np

from api.MachineLearningModel import DNIAnomalyDetector
from api.schemas import PredictionRequest, PredictionResponse, HealthCheckResponse
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Configuración
YOLO_PATH = "best.pt"
AUTOENCODER_PATH = "dni_anomaly_detector.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lista blanca de labels válidos
VALID_LABELS = ['FrenteValido', 'DorsoValido']


class DNIValidator:
    def __init__(self):
        self.yolo_model = YOLO(YOLO_PATH)
        self.autoencoder = DNIAnomalyDetector(device=DEVICE)
        self.autoencoder.load_model(AUTOENCODER_PATH)

    def crop_image(self, image: Image.Image, bbox) -> Image.Image:
        """Recorta la imagen usando el bbox del YOLO."""
        x1, y1, x2, y2 = map(int, bbox)
        return image.crop((x1, y1, x2, y2))

    def validate(self, image_path: str) -> tuple[float, str]:
        """
        Realiza la validación doble: YOLO + Autoencoder
        Returns: (confidence, message)
        """
        # Primera validación: YOLO
        results = self.yolo_model(image_path)[0]

        # Si no hay detecciones
        if len(results.boxes) == 0:
            return 0.0, "No object detected"

        # Obtener la detección con mayor confianza
        confidences = results.boxes.conf.cpu().numpy()
        best_idx = confidences.argmax()
        box = results.boxes[best_idx]

        cls_id = box.cls.item()
        cls_name = results.names[int(cls_id)]
        yolo_confidence = box.conf.item()

        # Si la clase no está en la lista blanca
        if cls_name not in VALID_LABELS:
            return 0.0, f"Invalid object detected: {cls_name}"

        # Recortar imagen usando el bbox
        original_image = Image.open(image_path)
        bbox = box.xyxy[0].cpu().numpy()
        cropped_image = self.crop_image(original_image, bbox)

        # Guardar temporalmente la imagen recortada
        temp_crop_path = "temp_crop.jpg"
        cropped_image.save(temp_crop_path)

        # Segunda validación: Autoencoder
        try:
            autoencoder_confidence = self.autoencoder.predict(temp_crop_path)
        finally:
            os.remove(temp_crop_path)

        # Combinar confidencias
        final_confidence = autoencoder_confidence
        return final_confidence, f"Valid DNI detected: {cls_name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global validator
    try:
        validator = DNIValidator()
        logger.info("DNI Validator initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize validator: {str(e)}")
        raise
    finally:
        if validator:
            pass


def get_validator():
    if validator is None:
        raise HTTPException(
            status_code=503,
            detail="Validator service not initialized"
        )
    return validator


def create_router() -> APIRouter:
    router = APIRouter()

    @router.post("/predict", response_model=PredictionResponse)
    async def predict(
            request: PredictionRequest,
            validator: DNIValidator = Depends(get_validator)
    ) -> PredictionResponse:
        try:
            # Convertir base64 a imagen
            image_bytes = base64.b64decode(request.base64_image)
            temp_path = "temp_input.jpg"

            with open(temp_path, "wb") as f:
                f.write(image_bytes)

            try:
                confidence, message = validator.validate(temp_path)
                logger.info(f"Validation result: {message} with confidence {confidence}")
                return PredictionResponse(confidence=float(confidence))
            finally:
                os.remove(temp_path)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get("/healthcheck", response_model=HealthCheckResponse)
    async def healthcheck(validator: DNIValidator = Depends(get_validator)) -> HealthCheckResponse:
        status = "healthy"
        details = {
            "validator": {
                "status": "healthy",
                "message": "Services initialized successfully"
            },
            "resources": {
                "status": "healthy",
                "message": "All models loaded"
            }
        }

        # Verificar modelos
        if not os.path.exists(YOLO_PATH):
            status = "unhealthy"
            details["resources"]["status"] = "unhealthy"
            details["resources"]["message"] = f"YOLO model not found at {YOLO_PATH}"

        if not os.path.exists(AUTOENCODER_PATH):
            status = "unhealthy"
            details["resources"]["status"] = "unhealthy"
            details["resources"]["message"] = f"Autoencoder not found at {AUTOENCODER_PATH}"

        if status == "unhealthy":
            raise HTTPException(status_code=503, detail=details)

        return HealthCheckResponse(status=status, details=details)

    return router