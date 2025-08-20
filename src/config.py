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
OP_tags = ["tag_gate_manipulation", "tag_RP_on", "tag_chamber_open", "tag_venting"]

ION_THRESHOLD = 1e-7  # Threshold for ion pressure to classify as unexpected
CONV_THRESHOLD = 1e-3  # Threshold for convectron pressure to classify
ROLL_WINDOWS = [3, 15]  # in samples (adapt to your sampling cadence)
SLOPE_WINDOW = 5       # samples for rolling slope (linear trend)

IF_RANDOM_STATE = 42
IF_CONTAMINATION = 0.01  # expected fraction of anomalies


FEATURE_COLUMNS = [
    "pressure_ion", "pressure_conv",
    "delta_ion", "delta_conv",
    "log_pressure_ion", "log_pressure_conv",
    "delta_log_ion", "delta_log_conv",
    "roll3_mean_ion", "roll3_std_ion", "roll15_mean_ion", "roll15_std_ion",
    "roll3_mean_conv", "roll3_std_conv", "roll15_mean_conv", "roll15_std_conv",
    "slope_ion", "slope_conv", "tag_IG_fail", "tag_gate_manipulation", "tag_RP_on", "tag_chamber_open", "tag_venting"
]