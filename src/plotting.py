# src/plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import pandas as pd
from .config import IG_TAGS, CG_TAGS, CH_TAGS, NUMERIC_COLS
def _format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

def plot_time_with_events(df: pd.DataFrame, savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)
    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_title("Ion and Convectron Pressure Over Time")
    _format_time_axis(ax)
    ax.set_xlabel("Datetime")
    ax.legend(loc="upper right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_time_with_unplugged_events(df: pd.DataFrame, savepath: Path | None = None): 
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron")
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion")

    # IC markers
    
    '''ic_turn_off = df[df["IG_state"] == "IG turn off"]
    ax.scatter(ic_turn_off["datetime"], 
               ic_turn_off["pressure_ion"],
               label="IC Turn Off", marker="x", color="red", s=60, zorder=5)'''
    ic_unplugged = df[df["IC_unplugged"]]
    ax.scatter(ic_unplugged["datetime"], 
               [1e-7] * len(ic_unplugged),
               label="IC Unplugged", marker="o", color="yellow", s=60, zorder=5)
    '''ic_off = df[df["IG_state"] == "IG off"]
    ax.scatter(ic_off["datetime"], 
                ic_off["pressure_ion"],
                label="IC Off", marker="o", color="yellow", s=60, zorder=5)'''
    
    # CC markers
    '''cc_turn_off = df[df["CG_state"] == "CG turn off"]
    ax.scatter(cc_turn_off["datetime"], 
               cc_turn_off["pressure_conv"],
               label="CC Turn Off", marker="x", color="red", s=60, zorder=5)'''
    cc_unplugged = df[df["CC_unplugged"]]
    ax.scatter(cc_unplugged["datetime"], 
               [1e-3] * len(cc_unplugged),
               label="CC Unplugged", marker="o", color="purple", s=60, zorder=5)
    '''cc_off = df[df["CG_state"] == "CG off"]
    ax.scatter(cc_off["datetime"], 
               cc_off["pressure_conv"],
               label="CC Off", marker="o", color="purple", s=60, zorder=5)'''
    
    

    ax.set_yscale("log")
    ax.set_title("Time Series with Unplugged Events")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Pressure (Torr)")
    _format_time_axis(ax)
    ax.legend(loc="upper right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()




def _contiguous_runs(df: pd.DataFrame, state_col: str = "IG_state"):
    states = df[state_col].fillna("IG unknown").to_numpy()
    if len(states) == 0:
        return []
    runs = []
    start = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            runs.append((df["datetime"].iloc[start], df["datetime"].iloc[i - 1], states[i - 1]))
            start = i
    runs.append((df["datetime"].iloc[start], df["datetime"].iloc[len(states) - 1], states[-1]))
    return runs


def plot_time_with_state_bands(df: pd.DataFrame, title: str, savepath: Path | None = None):
    state_colors = {
        "IG on": "#b2df8a",
        "IG turn on": "#a6cee3",
        "IG slow on": "#fb9a99",
        "IG off": "#fdbf6f",
        "IG fail": "#e31a1c",
        "IG unknown": "#dddddd",
    }

    fig, ax = plt.subplots(figsize=(12, 5))

    # Background bands for IG_state
    for t0, t1, st in _contiguous_runs(df, "IG_state"):
        ax.axvspan(t0, t1, color=state_colors.get(st, "#dddddd"), alpha=0.25, lw=0)

    # Pressure lines
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)

    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_xlabel("Datetime")
    ax.set_title(title)
    _format_time_axis(ax)

    # Legends
    ax.legend(loc="upper right")
    patches = [Patch(facecolor=c, alpha=0.25, label=s) for s, c in state_colors.items()]
    ax.legend(handles=patches, title="IG_state", loc="upper right")
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()



def plot_time_with_tag_markers(df: pd.DataFrame, tags_to_mark=CH_TAGS, title: str = "Pressures with tag markers", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)

    exploded = df[["datetime", "pressure_conv", "pressure_ion", "tag_list"]].explode("tag_list")
    exploded = exploded[exploded["tag_list"].isin(tags_to_mark)]

    for tag in tags_to_mark:
        sub = exploded[exploded["tag_list"] == tag]
        ax.scatter(sub["datetime"], sub["pressure_conv"], label=f"{tag} (conv)", s=30, marker="o")
        ax.scatter(sub["datetime"], sub["pressure_ion"], label=f"{tag} (ion)", s=30, marker="x")

    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_xlabel("Datetime")
    ax.set_title(title)
    _format_time_axis(ax)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_anomalies(df: pd.DataFrame, title: str = "Anomaly overlay", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)

    # Highlight anomalies ion 
    ion_anomalies = df[df["anomaly_if_ion"] == -1]
    ax.scatter(ion_anomalies["datetime"], ion_anomalies["pressure_ion"],
               label="Ion Anomaly", color="green", marker="o", s=50, alpha=0.7)
    # Highlight anomalies convectron
    conv_anomalies = df[df["anomaly_if_conv"] == -1]
    ax.scatter(conv_anomalies["datetime"], conv_anomalies["pressure_conv"],
               label="Convectron Anomaly", color="red", marker="D", s=50, alpha=0.7)
    
    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_xlabel("Datetime")
    ax.set_title(title)
    _format_time_axis(ax)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()


def plot_tag_anomalies(df: pd.DataFrame, title: str = "Tagged Anomalies", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2, alpha=0.8)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.8)

    # Map anomaly categories to colors/markers
    anomaly_styles = {
        "operational": {"color": "green", "marker": "s"},
        "unexpected": {"color": "red", "marker": "x"},
        "normal": None,
    }

    # Plot ion anomalies
    for category, style in anomaly_styles.items():
        if style:
            mask = df["anomaly_ion"] == category
            ax.scatter(df.loc[mask, "datetime"], df.loc[mask, "pressure_ion"],
                       label=f"Ion {category}", s=50, **style)

    # Plot convectron anomalies
    for category, style in anomaly_styles.items():
        if style:
            mask = df["anomaly_conv"] == category
            ax.scatter(df.loc[mask, "datetime"], df.loc[mask, "pressure_conv"],
                       label=f"Convectron {category}", s=50, **style)

    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_xlabel("Datetime")
    ax.set_title(title)
    _format_time_axis(ax)
    plt.grid()
    ax.legend(loc="upper right")
    
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")

    plt.show()
