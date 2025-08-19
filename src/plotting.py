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
    # legend over graph
    ax.legend(loc="upper left")
    
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
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
    ic_off = df[df["IG_state"] == "IG off"]
    ax.scatter(ic_off["datetime"], 
                ic_off["pressure_ion"],
                label="IC Off", marker="o", color="yellow", s=60, zorder=5)
    
    # CC markers
    '''cc_turn_off = df[df["CG_state"] == "CG turn off"]
    ax.scatter(cc_turn_off["datetime"], 
               cc_turn_off["pressure_conv"],
               label="CC Turn Off", marker="x", color="red", s=60, zorder=5)'''
    cc_unplugged = df[df["CC_unplugged"]]
    ax.scatter(cc_unplugged["datetime"], 
               [1e-3] * len(cc_unplugged),
               label="CC Unplugged", marker="o", color="purple", s=60, zorder=5)
    cc_off = df[df["CG_state"] == "CG off"]
    ax.scatter(cc_off["datetime"], 
               cc_off["pressure_conv"],
               label="CC Off", marker="o", color="purple", s=60, zorder=5)
    
    

    ax.set_yscale("log")
    ax.set_title("Time Series with Unplugged Events")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (Torr)")
    _format_time_axis(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()



def scatter_ion_vs_conv_by_IG_CG_state(df: pd.DataFrame, savepath: Path | None = None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="pressure_ion", y="pressure_conv", hue="IG_state", style="CG_state", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Pressure Ion vs Convectron by State")
    plt.xlabel("Pressure Ion (Torr)")
    plt.ylabel("Pressure Convectron (Torr)")
    plt.legend(title="IG State / CG State")
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

def scatter_ion_vs_conv_by_CH_state(df: pd.DataFrame, savepath: Path| None = None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="pressure_ion", y="pressure_conv", hue="CH_state", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Pressure Ion vs Convectron by Chamber State")
    plt.xlabel("Pressure Ion (Torr)")
    plt.ylabel("Pressure Convectron (Torr)")
    plt.legend(title="CH State")
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

####



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
    ax.set_title(title)
    _format_time_axis(ax)

    # Legends
    ax.legend(loc="upper right")
    patches = [Patch(facecolor=c, alpha=0.25, label=s) for s, c in state_colors.items()]
    leg2 = ax.legend(handles=patches, title="IG_state", loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.add_artist(leg2)

    plt.tight_layout()
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
    ax.set_title(title)
    _format_time_axis(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_anomalies(df: pd.DataFrame, title: str = "Anomaly overlay", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)

    # Highlight anomalies (Isolation Forest: -1 is anomaly)
    mask = df.get("anomaly_if", pd.Series(index=df.index, dtype=int)) == -1
    ax.scatter(df.loc[mask, "datetime"], df.loc[mask, "pressure_conv"], label="Anomaly (conv)", s=50, color="red", marker="D")
    ax.scatter(df.loc[mask, "datetime"], df.loc[mask, "pressure_ion"], label="Anomaly (ion)", s=50, color="green", marker="o")

    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_title(title)
    _format_time_axis(ax)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_tag_anomalies(df: pd.DataFrame, title: str = "Anomalies with Tags", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["datetime"], df["pressure_conv"], label="Convectron", lw=2)
    ax.plot(df["datetime"], df["pressure_ion"], label="Ion", lw=2, alpha=0.9)
    
    # plot anomaly_ion and anomaly_conv as scatter points
    mask_ion = df.get("anomaly_if_ion", pd.Series(index=df.index, dtype=int)) == -1
    mask_conv = df.get("anomaly_if_conv", pd.Series(index=df.index, dtype=int)) == -1
    ax.scatter(df.loc[mask_ion, "datetime"], df.loc[mask_ion, "pressure_ion"], 
               label="Anomaly Ion", s=50, color="red", marker="D")
    ax.scatter(df.loc[mask_conv, "datetime"], df.loc[mask_conv, "pressure_conv"], 
               label="Anomaly Convectron", s=50, color="blue", marker="o")
     
    '''
        # Add tags as vertical lines
        for tag in CH_TAGS:
            tag_mask = df[tag]
            if tag_mask.any():
                ax.axvline(x=df.loc[tag_mask, "datetime"].iloc[0], color='orange', linestyle='--', label=f"{tag} event")'''

    ax.set_yscale("log")
    ax.set_ylabel("Pressure (Torr, log)")
    ax.set_title(title)
    _format_time_axis(ax)
    plt.grid()
    ax.legend(bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()