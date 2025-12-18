# models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_logistic_model():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )
