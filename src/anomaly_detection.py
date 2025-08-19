# src/anomaly_detection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from .config import IF_RANDOM_STATE, IF_CONTAMINATION, FEATURE_COLUMNS, OP_tags


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    # Assume FEATURE_COLUMNS contains both ion and convectron features, e.g., "pressure_ion", "pressure_convectron"
    ion_feats = [c for c in FEATURE_COLUMNS if "ion" in c and c in df.columns]
    conv_feats = [c for c in FEATURE_COLUMNS if "conv" in c and c in df.columns]

    # Detect anomalies separately for ion and convectron
    if ion_feats:
        X_ion = df[ion_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model_ion = IsolationForest(
            n_estimators=300,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        df["anomaly_if_ion"] = model_ion.fit_predict(X_ion)
        df["score_if_raw_ion"] = model_ion.decision_function(X_ion)
        df["score_if_ion"] = -df["score_if_raw_ion"]
    else:
        df["anomaly_if_ion"] = 1
        df["score_if_raw_ion"] = 0.0
        df["score_if_ion"] = 0.0

    if conv_feats:
        X_conv = df[conv_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        model_conv = IsolationForest(
            n_estimators=300,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        df["anomaly_if_conv"] = model_conv.fit_predict(X_conv)
        df["score_if_raw_conv"] = model_conv.decision_function(X_conv)
        df["score_if_conv"] = -df["score_if_raw_conv"]
    else:
        df["anomaly_if_conv"] = 1
        df["score_if_raw_conv"] = 0.0
        df["score_if_conv"] = 0.0

    # Only mark as anomaly if both ion and convectron agree
    df["anomaly_if"] = np.where(
        (df["anomaly_if_ion"] == -1) | (df["anomaly_if_conv"] == -1), -1, 1
    )
    # For score, you could take the min (most anomalous) or mean
    df["score_if_raw"] = df[["score_if_raw_ion", "score_if_raw_conv"]].min(axis=1)
    df["score_if"] = -df["score_if_raw"]

    return df


def tag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_op_tag"] = df[OP_tags].any(axis=1)

    # Classify ion anomalies
    def classify_ion(row):
        if row["anomaly_if_ion"] == -1 and row["has_op_tag"]:
            return "operational"
        elif row["anomaly_if_ion"] == -1 and not row["has_op_tag"]:
            return "unexpected"
        else:
            return "normal"

    # Classify convectron anomalies
    def classify_conv(row):
        if row["anomaly_if_conv"] == -1 and not row["has_op_tag"]:
            return "unexpected"
        elif row["anomaly_if_conv"] == -1 and row["has_op_tag"]:
            return "operational"
        else:
            return "normal"

    df["anomaly_ion"] = df.apply(classify_ion, axis=1)
    df["anomaly_conv"] = df.apply(classify_conv, axis=1)


    return df
