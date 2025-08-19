# src/data_loader.py
import pandas as pd
from .config import DATA_RAW, DATA_PROCESSED
from pathlib import Path
folder = DATA_RAW
def load_all_csv() -> pd.DataFrame:
    files = sorted(folder.glob("*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    save_path = DATA_PROCESSED / "merged_all_raw_data.csv"
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(save_path, index=False)
    print(f"Combined data saved to {save_path}")

    return pd.concat(dfs, ignore_index=True)