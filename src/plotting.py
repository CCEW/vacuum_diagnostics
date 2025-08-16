# src/plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import pandas as pd
from .config import IG_TAGS, CG_TAGS, NUMERIC_COLS

def plot_time_with_events(df: pd.DataFrame, savepath: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    df["pressure_ion"].plot(ax=ax, label="Pressure Ion", color="blue")
    df["pressure_conv"].plot(ax=ax, label="Pressure Conv", color="orange")
    
    for tag in IG_TAGS + CG_TAGS:
        if f"tag_{tag.replace(' ', '_')}" in df.columns:
            events = df[df[f"tag_{tag.replace(' ', '_')}"] == 1]
            ax.scatter(events.index, events["pressure_ion"], label=tag, marker='x')
    ax.set_yscale("log")
    ax.set_title("Time Series with Event Markers")
    _format_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (Torr)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def scatter_ion_vs_conv_by_state(df: pd.DataFrame, savepath: Path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="pressure_ion", y="pressure_conv", hue="IG_state", style="CG_state", alpha=0.7)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Pressure Ion vs Convectron by State")
    plt.xlabel("Pressure Ion (Torr)")
    plt.ylabel("Pressure Convectron (Torr)")
    plt.legend(title="IG State / CG State")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()



####
def _format_time_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


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



def plot_time_with_tag_markers(df: pd.DataFrame, tags_to_mark=("IG fail", "IG turn on", "IG slow on"), title: str = "Pressures with tag markers", savepath: Path | None = None):
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