# src/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import NUMERIC_COLS, ROLL_WINDOWS, SLOPE_WINDOW

def drop_nonessential_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ion_analog" not in df.columns or "conv_analog" not in df.columns:
        return df
    cols_to_drop = ["conv_analog", "ion_analog"]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df

# Build datetime column from date and time columns
def build_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "datetime" in df.columns:
        return df
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str)
    )
    df.drop(columns=["date", "time"], inplace=True, errors="ignore")
    df = df[["datetime"]].join(df.drop(columns=["datetime"]))
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = build_datetime(df)
    df = drop_nonessential_cols(df)
    return df

# Calculate rolling features like mean, std, min, max for specified windows
def _rolling_features(s: pd.Series, name: str, windows: list[int]) -> pd.DataFrame:
    out = {}
    for w in windows:
        roll = s.rolling(window=w, min_periods=max(2, w//2))
        out[f"roll{w}_mean_{name}"] = roll.mean()
        out[f"roll{w}_std_{name}"] = roll.std(ddof=0)
        out[f"roll{w}_min_{name}"] = roll.min()
        out[f"roll{w}_max_{name}"] = roll.max()
    return pd.DataFrame(out)

# Calculate rolling slope (linear trend) using a simple linear regression approach
def _rolling_slope(y: pd.Series, window: int) -> pd.Series:
    # Linear regression slope per rolling window using simple formula
    # slope = cov(x,y)/var(x) with x = [0..w-1]
    w = window
    if w < 2:
        return pd.Series(np.nan, index=y.index)
    x = np.arange(w)
    x = (x - x.mean())  # center to avoid overflow
    denom = (x**2).sum()

    def slope_window(vals):
        if np.isnan(vals).any():
            vals = vals.astype(float)
        return np.dot(vals - np.nanmean(vals), x) / denom

    return y.rolling(w, min_periods=w).apply(slope_window, raw=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Differences (first derivative)
    df["delta_ion"] = df["pressure_ion"].diff()
    df["delta_conv"] = df["pressure_conv"].diff()

    # Log pressures + deltas (protect against non-positive values)
    df["log_pressure_ion"] = np.log(df["pressure_ion"].where(df["pressure_ion"] > 0))
    df["log_pressure_conv"] = np.log(df["pressure_conv"].where(df["pressure_conv"] > 0))
    df["delta_log_ion"] = df["log_pressure_ion"].diff()
    df["delta_log_conv"] = df["log_pressure_conv"].diff()

    # Rolling stats
    roll_ion = _rolling_features(df["pressure_ion"], "ion", ROLL_WINDOWS)
    roll_conv = _rolling_features(df["pressure_conv"], "conv", ROLL_WINDOWS)
    df = pd.concat([df, roll_ion, roll_conv], axis=1)

    # Rolling slopes (trend per window)
    df["slope_ion"] = _rolling_slope(df["pressure_ion"], SLOPE_WINDOW)
    df["slope_conv"] = _rolling_slope(df["pressure_conv"], SLOPE_WINDOW)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    df = engineer_features(df)
    return df