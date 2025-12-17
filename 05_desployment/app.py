# app.py

from fastapi import FastAPI
from schemas import Applicant
from service import evaluate_risk

app = FastAPI(
    title="Credit Risk Evaluation API",
    description="API para evaluar riesgo de incumplimiento de crédito",
    version="1.0"
)

@app.post("/evaluate_risk")
def evaluate(applicant: Applicant):
    """
    Evalúa el riesgo crediticio de un solicitante
    """
    result = evaluate_risk(applicant.dict())
    return result
