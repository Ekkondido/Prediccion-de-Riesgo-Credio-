import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# =========================
# Rutas
# =========================
DATA_PATH = Path("Data/processed/dataset_merged.parquet")
ARTIFACTS_PATH = Path("artifacts")

# =========================
# Cargar modelo y datos
# =========================
model = joblib.load(ARTIFACTS_PATH / "model.pkl")
features = joblib.load(ARTIFACTS_PATH / "features.pkl")

df = pd.read_parquet(DATA_PATH)

X = df[features]
y = df["TARGET"]

# =========================
# Predicciones
# =========================
y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# =========================
# Métricas
# =========================
auc = roc_auc_score(y, y_proba)
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print("\nAUC-ROC:", auc)
print("\nMatriz de Confusión:")
print(cm)
print("\nReporte de Clasificación:")
print(report)
