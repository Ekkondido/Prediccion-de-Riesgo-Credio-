from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "artifacts" / "model.pkl")
features = joblib.load(BASE_DIR / "artifacts" / "features.pkl")


# Umbrales de decisión
THRESHOLD_APPROVE = 0.30
THRESHOLD_REVIEW = 0.55


def evaluate_risk(applicant_data: dict):
    """
    Recibe datos del solicitante y retorna probabilidad y decisión
    """
    df = pd.DataFrame([applicant_data])

    prob_default = model.predict_proba(df)[:, 1][0]

    if prob_default < THRESHOLD_APPROVE:
        decision = "APROBAR"
    elif prob_default < THRESHOLD_REVIEW:
        decision = "REVISION_MANUAL"
    else:
        decision = "RECHAZAR"

    return {
        "prob_default": round(float(prob_default), 4),
        "decision": decision
    }
