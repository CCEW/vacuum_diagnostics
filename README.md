# Vacuum Diagnostics — Dynamic Data Collection & ML Anomaly Detection

This project demonstrates a low-cost data-driven method to analyze vacuum system behavior. It prioritizes and detects system disturbances (sudden manipulations, pump instabilities, potential leaks) by correlating pressure trends and event tags over time.

## Pipeline
1. **Ingest**: merge CSV logs from `data/raw/`.
2. **Preprocess**: build timestamps, drop nonessential columns, engineer time-series features (derivatives, rolling stats, slopes).
3. **Tag & State**: parse human tags, build binary tag columns, derive IG/CG states.
4. **Anomaly Detection**: Isolation Forest over engineered features.
5. **Visualize**: time-series with IG state bands, tag markers, anomaly overlays.

## Usage
```bash
python -m src.main
```
Outputs:
- Processed data → `data/processed/processed.csv`
- Plots → `outputs/plots/`