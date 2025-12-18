import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# Rutas
# =========================
DATA_PATH = Path("Data/processed/dataset_merged.parquet")
ARTIFACTS_PATH = Path("artifacts")
ARTIFACTS_PATH.mkdir(exist_ok=True)

# =========================
# Cargar dataset
# =========================
df = pd.read_parquet(DATA_PATH)

print("Dataset cargado:", df.shape)

# =========================
# Separar X / y
# =========================
TARGET = "TARGET"

y = df[TARGET]
X = df.drop(columns=[TARGET, "SK_ID_CURR"])

# Mantener solo columnas numéricas
X = X.select_dtypes(include=[np.number])

print("Features:", X.shape)
print("Target balance:")
print(y.value_counts(normalize=True))

# =========================
# Modelos a evaluar
# =========================
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )
}

# =========================
# Validación cruzada
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    aucs = []
    print(f"\nEntrenando modelo: {name}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

        print(f"  Fold {fold} AUC: {auc:.4f}")

    results[name] = np.mean(aucs)
    print(f"  AUC promedio {name}: {results[name]:.4f}")

# =========================
# Seleccionar mejor modelo
# =========================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nModelo campeón:", best_model_name)
print("AUC final:", results[best_model_name])

# =========================
# Entrenar modelo final
# =========================
best_model.fit(X, y)

# =========================
# Guardar artefactos
# =========================
joblib.dump(best_model, ARTIFACTS_PATH / "model.pkl")
joblib.dump(list(X.columns), ARTIFACTS_PATH / "features.pkl")

print("\nModelo y features guardados en /artifacts")
