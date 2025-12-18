import pandas as pd
from pathlib import Path

# =========================
# Rutas
# =========================
RAW_PATH = Path("Data/Raw")
PROCESSED_PATH = Path("Data/processed")
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# Cargar datasets
# =========================
application = pd.read_parquet(RAW_PATH / "application.parquet")
bureau = pd.read_parquet(RAW_PATH / "bureau.parquet")
previous = pd.read_parquet(RAW_PATH / "previous_application.parquet")



print("Datasets cargados:")
print(application.shape, bureau.shape, previous.shape)

# =========================
# Features desde bureau
# =========================
bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    bureau_credit_count=("SK_ID_BUREAU", "count"),
    bureau_credit_active=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
    bureau_credit_closed=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
    bureau_debt_total=("AMT_CREDIT_SUM_DEBT", "sum"),
    bureau_credit_sum=("AMT_CREDIT_SUM", "sum")
).reset_index()

# =========================
# Features desde previous_application
# =========================
prev_agg = previous.groupby("SK_ID_CURR").agg(
    prev_app_count=("SK_ID_PREV", "count"),
    prev_app_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
    prev_app_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
    prev_app_credit_mean=("AMT_CREDIT", "mean")
).reset_index()

# =========================
# Merge final
# =========================
df = application.merge(bureau_agg, on="SK_ID_CURR", how="left")
df = df.merge(prev_agg, on="SK_ID_CURR", how="left")

# =========================
# Rellenar nulos correctamente
# =========================
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(0)



# =========================
# Guardar dataset final
# =========================
output_path = PROCESSED_PATH / "dataset_merged.parquet"
df.to_parquet(output_path)

print("Dataset final guardado en:", output_path)
print("Shape final:", df.shape)
