# src/anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .config import IF_RANDOM_STATE, IF_CONTAMINATION, FEATURE_COLUMNS


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model = IsolationForest(
        n_estimators=300,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    df["anomaly_if"] = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    # Higher scores = more normal. Convert so lower = more anomalous for intuition.
    df["score_if_raw"] = model.decision_function(X)
    df["score_if"] = -df["score_if_raw"]

    return df