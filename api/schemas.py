# Esquemas Pydantic actualizados
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    base64_image: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    details: dict