import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import roc_curve, confusion_matrix

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

y_proba = model.predict_proba(X)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

# =========================
# Curva ROC
# =========================
fpr, tpr, _ = roc_curve(y, y_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()

# =========================
# Matriz de Confusión
# =========================
cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# =========================
# Importancia de variables
# =========================
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    fi = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8,5))
    sns.barplot(x="importance", y="feature", data=fi)
    plt.title("Top 15 Variables Más Importantes")
    plt.show()
