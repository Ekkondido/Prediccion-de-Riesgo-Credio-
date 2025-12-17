# feature_utils.py

import pandas as pd

def aggregate_bureau(bureau_df):
    """
    Agrega información del bureau a nivel de cliente (SK_ID_CURR)
    """
    agg = bureau_df.groupby("SK_ID_CURR").agg(
        bureau_credit_count=("SK_ID_BUREAU", "count"),
        bureau_credit_active=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        bureau_credit_closed=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        bureau_debt_total=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_credit_sum=("AMT_CREDIT_SUM", "sum")
    ).reset_index()

    return agg


def aggregate_previous_app(prev_df):
    """
    Agrega información de solicitudes previas
    """
    agg = prev_df.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_app_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_app_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_app_credit_mean=("AMT_CREDIT", "mean")
    ).reset_index()

    return agg
