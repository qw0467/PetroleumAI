# PetroleumAI - Predictive Maintenance Dashboard

A predictive maintenance MVP for petroleum plant equipment using multivariate time-series analysis and machine learning anomaly detection.

## Overview

PetroleumAI monitors sensor data from petroleum equipment (pressure, temperature, flow, vibration) and uses Isolation Forest anomaly detection to identify potential equipment issues before they cause failures.

## Features

- **Real-time KPI Monitoring**: Track equipment health status, anomaly scores, and risk levels
- **Anomaly Detection**: Isolation Forest algorithm identifies abnormal patterns in sensor data
- **Feature Engineering**: Automated computation of rolling statistics, trend analysis, spike and drift detection
- **Interactive Visualizations**: Toggle overlays for anomalies, drift regions, and spikes
- **Data Export**: Download processed data with all computed features

## Project Structure

```
petroleumai/
├── data/           # Data storage
├── src/
│   ├── simulate.py  # Sensor data simulation
│   ├── features.py  # Feature engineering
│   ├── model.py     # Anomaly detection model
│   ├── kpis.py      # KPI calculations
│   └── visualize.py # Visualization functions
├── app.py          # Main Streamlit dashboard
└── README.md       # This file
```

## Usage

### Running the Dashboard

The application runs on port 5000:

```bash
streamlit run app.py --server.port 5000
```

### Data Input Options

1. **Simulated Data**: Generate synthetic sensor readings with configurable parameters
2. **Upload CSV**: Upload your own data with columns: `timestamp`, `pressure`, `temperature`, `flow`, `vibration`

### Key Metrics

- **Equipment Health**: Overall health status (Normal/Warning/Critical)
- **Anomaly Score**: Current anomaly score (0-1 scale)
- **Irregular Events**: Count of detected anomalies
- **Risk Level**: Computed risk based on health, anomalies, and drift
- **Drift Detection**: Per-sensor drift monitoring

## Technical Details

### Anomaly Detection

Uses scikit-learn's Isolation Forest algorithm:
- Contamination parameter controls expected anomaly rate
- Features normalized using StandardScaler
- Scores normalized to 0-1 range

### Feature Engineering

For each sensor:
- Rolling mean and standard deviation (20-point window)
- Slope/trend calculation
- Spike detection using z-score method
- Drift detection comparing rolling to overall mean
- Cross-correlation between sensor pairs
- Instability score based on variance changes

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
