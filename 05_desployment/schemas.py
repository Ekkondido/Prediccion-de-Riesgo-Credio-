# schemas.py

from pydantic import BaseModel

class Applicant(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: float
    DAYS_EMPLOYED: float
    bureau_credit_count: float
    bureau_credit_active: float
    bureau_credit_closed: float
    bureau_debt_total: float
    bureau_credit_sum: float
    prev_app_count: float
    prev_app_approved: float
    prev_app_refused: float
    prev_app_credit_mean: float
