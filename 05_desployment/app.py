from fastapi import FastAPI
from schemas import ApplicantData, PredictionResponse
from service import evaluate_risk

app = FastAPI(
    title="Credit Risk API",
    description="API para evaluar riesgo de incumplimiento de crédito",
    version="1.0"
)

@app.get("/")
def root():
    return {"status": "API funcionando correctamente"}

@app.post("/evaluate_risk", response_model=PredictionResponse)
def evaluate(data: ApplicantData):
    return evaluate_risk(data)
