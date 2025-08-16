# main.py
from src.data_loader import load_all_csv
from src.config import DATA_RAW, DATA_PROCESSED, OUTPUT_PLOTS
from src.plotting import plot_time_with_events, scatter_ion_vs_conv_by_state, plot_time_with_tag_markers, plot_time_with_state_bands, plot_anomalies
from src.anomaly_detection import detect_anomalies
from src.preprocessing import preprocess
from src.tags import tag_events, tag_frequencies, combo_counts

def main():
    df = load_all_csv()
    df = preprocess(df)
    df = tag_events(df)
    df = detect_anomalies(df)
    df.to_csv(DATA_PROCESSED / "processed.csv", index=False)

    tag_freqs = tag_frequencies(df)
    combo_counts_df = combo_counts(df)
    tag_freqs.to_csv(DATA_PROCESSED / "tag_frequencies.csv", index=False)
    combo_counts_df.to_csv(DATA_PROCESSED / "tag_combinations.csv", index=False)
    print("Saved tag stats.")

    plot_time_with_state_bands(df, savepath=OUTPUT_PLOTS / "time_series_state_bands.png")
    plot_time_with_events(df, savepath=OUTPUT_PLOTS / "time_series_events.png")
    scatter_ion_vs_conv_by_state(df, savepath=OUTPUT_PLOTS / "scatter_by_state.png")
    plot_time_with_tag_markers(df, savepath=OUTPUT_PLOTS / "time_series_tags.png")

    # High-level plots
    plot_time_with_state_bands(df, title="Pressures with IG_state bands", savepath=OUTPUT_PLOTS  / "state_bands.png")
    plot_time_with_tag_markers(df, title="Tag markers over time", savepath=OUTPUT_PLOTS / "tag_markers.png")
    plot_anomalies(df, title="Anomaly overlay (IsolationForest)", savepath=OUTPUT_PLOTS / "anomalies_if.png")


if __name__ == "__main__":
    main()