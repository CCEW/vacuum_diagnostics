from pathlib import Path

DATA_RAW = Path("../data/raw")
DATA_PROCESSED = Path("../data/processed")
OUTPUT_PLOTS = Path("../outputs/plots")
OUTPUT_TABLES = Path("../outputs/tables")

NUMERIC_COLS = [
    "ion_analog","voltage_ion","pressure_ion",
    "conv_analog","voltage_conv","pressure_conv",
]

IG_TAGS = ["IG on","IG off","IG fail","IG turn on","IG turn off", "IG slow on"]
CG_TAGS = ["CG on", "CG off", "CG turn off", "CG turn on"]
CH_TAGS = ["gate manipulation", "RP on", "chamber open", "venting"]

# === Feature settings ===
ROLL_WINDOWS = [5, 15]  # in samples (adapt to your sampling cadence)
SLOPE_WINDOW = 15       # samples for rolling slope (linear trend)

# === ML settings ===
# Isolation Forest parameters:
# - IF_RANDOM_STATE: For reproducibility. Usually keep fixed.
# - IF_CONTAMINATION: Expected fraction of anomalies. Tune based on domain knowledge or by inspecting results.
#   Start with a small value (e.g., 0.01), then adjust up/down depending on how many anomalies you expect.
#   You can use visual inspection or cross-validation with labeled data (if available) to guide this.
IF_RANDOM_STATE = 42
IF_CONTAMINATION = 0.01  # expected fraction of anomalies

# To tune these parameters:
# 1. Start with default values.
# 2. Run the model and inspect detected anomalies.
# 3. If too many/too few anomalies are detected, adjust IF_CONTAMINATION.
# 4. If you have labeled data, use grid search or cross-validation to optimize.
# 5. For rolling windows, try different values and see which best captures the trends/disturbances in your data.

# Features used by the anomaly model (must exist after feature engineering)
FEATURE_COLUMNS = [
    "pressure_ion", "pressure_conv",
    "delta_ion", "delta_conv",
    "log_pressure_ion", "log_pressure_conv",
    "delta_log_ion", "delta_log_conv",
    "roll5_mean_ion", "roll5_std_ion", "roll15_mean_ion", "roll15_std_ion",
    "roll5_mean_conv", "roll5_std_conv", "roll15_mean_conv", "roll15_std_conv",
    "slope_ion", "slope_conv",
]