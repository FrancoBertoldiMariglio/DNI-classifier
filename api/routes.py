import os
import base64
import io
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException, Depends, FastAPI
import logging
from PIL import Image
from schemas import PredictionRequest, PredictionResponse, HealthCheckResponse
from MachineLearningModel import AnomalyDetectionModel
import tensorflow as tf

logger = logging.getLogger(__name__)

# Configuración
AUTOENCODER_PATH = "best_model_cropped_original.keras"
SVM_PATH = "svm_classifier.joblib"


class DNIValidator:
    def __init__(self):
        """Inicializa el detector de anomalías"""
        try:
            logger.info("Initializing anomaly detector...")

            # Forzar el uso de CPU para todas las operaciones
            with tf.device('/CPU:0'):
                self.anomaly_detector = AnomalyDetectionModel(
                    autoencoder_path=AUTOENCODER_PATH,
                    svm_path=SVM_PATH
                )
            logger.info("Anomaly detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {str(e)}")
            raise

    def validate(self, image: Image.Image) -> dict:
        """
        Valida una imagen usando el detector de anomalías
        Args:
            image: Imagen PIL a validar
        Returns:
            Dict con resultados de la predicción
        """
        try:
            # Forzar CPU para la predicción
            with tf.device('/CPU:0'):
                result = self.anomaly_detector.predict_image(image)
            return result
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            raise


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
            image = Image.open(io.BytesIO(image_bytes))

            # Realizar validación
            result = validator.validate(image)

            logger.info(f"Validation result: {result}")

            return PredictionResponse(
                prediction=result["prediction"],
                confidence=result["confidence"],
                probabilities=result["probabilities"]
            )

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
                "message": "Service initialized successfully"
            },
            "resources": {
                "status": "healthy",
                "message": "All models loaded"
            }
        }

        # Verificar modelos
        if not os.path.exists(AUTOENCODER_PATH):
            status = "unhealthy"
            details["resources"]["status"] = "unhealthy"
            details["resources"]["message"] = f"Autoencoder not found at {AUTOENCODER_PATH}"

        if not os.path.exists(SVM_PATH):
            status = "unhealthy"
            details["resources"]["status"] = "unhealthy"
            details["resources"]["message"] = f"SVM model not found at {SVM_PATH}"

        if status == "unhealthy":
            raise HTTPException(status_code=503, detail=details)

        return HealthCheckResponse(status=status, details=details)

    return router