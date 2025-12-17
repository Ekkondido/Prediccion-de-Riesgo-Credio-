from pydantic import BaseModel

class ApplicantData(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    probability: float
    decision: str
