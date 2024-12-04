from typing import Dict

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    base64_image: str = Field(..., description="Base64 encoded image string")

class PredictionResponse(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = Field(..., description="Predicted label")

class HealthCheckResponse(BaseModel):
    status: str
    details: Dict[str, Dict[str, str]]