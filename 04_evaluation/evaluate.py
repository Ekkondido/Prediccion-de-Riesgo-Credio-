# evaluate.py

import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

from config import TARGET, NUMERIC_FEATURES

# =========================
# 1. Cargar dataset y modelo
# =========================

df = pd.read_parquet("Data/processed/dataset_merged.parquet")
model = joblib.load("artifacts/model.pkl")

X = df[NUMERIC_FEATURES]
y = df[TARGET]

# Split para evaluación final
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 2. Predicciones
# =========================

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# =========================
# 3. Métricas
# =========================

print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred, digits=4))
