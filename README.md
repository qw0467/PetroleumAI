# PetroleumAI - Predictive Maintenance Dashboard

A predictive maintenance MVP for petroleum plant equipment using multivariate time-series analysis and machine learning anomaly detection.

## Overview

PetroleumAI monitors sensor data from petroleum equipment (pressure, temperature, flow, vibration) and uses Isolation Forest anomaly detection to identify potential equipment issues before they cause failures.

## Features

- **Real-time KPI Monitoring**: Track equipment health status, anomaly scores, and risk levels
- **Anomaly Detection**: Isolation Forest algorithm identifies abnormal patterns in sensor data
- **Feature Engineering**: Automated computation of rolling statistics, trend analysis, spike and drift detection
- **Root Cause Analysis**: Identify what went wrong, when abnormalities started, and generate engineering-style diagnostic reports
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
│   ├── visualize.py # Visualization functions
│   └── root_cause.py # Root cause analysis module
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

## Root Cause Analysis

The Root Cause Analysis module performs gap analysis to identify what caused equipment anomalies and potential shutdowns.

### What It Analyzes

1. **First Abnormality Detection**: Identifies which sensor first showed abnormal behavior and the exact timestamp
2. **Warning Lead Time**: Calculates how far before the shutdown event the first warning appeared
3. **Sensor Contribution Ranking**: Ranks sensors by their contribution to anomaly scores
4. **Baseline Comparison**: Compares abnormal regions to normal operation:
   - Mean deviation percentage
   - Variability increase
   - Trend reversals
5. **Correlation Breakdown**: Detects loss of normal operational relationships between sensors

### Using Root Cause Analysis

1. Load data (simulated or uploaded)
2. Navigate to the "Root Cause Analysis" tab
3. Configure shutdown event selection:
   - **Auto-detect**: Automatically finds shutdown events in the data
   - **Select manually**: Choose a specific timestamp
   - **Use last timestamp**: Use the last data point as reference
4. Click "Run Root Cause Analysis"

### Output

- **Key Findings**: First abnormality time, warning lead time, primary causal sensor
- **Sensor Contribution Ranking**: Table showing each sensor's contribution score
- **Baseline Comparison**: Table with deviation and variability metrics
- **Pattern Summary**: Detected abnormal patterns (drift, spikes, trend reversals)
- **Root Cause Explanation**: Engineering-style narrative explaining the failure sequence
- **Full Engineering Report**: Downloadable detailed diagnostic report

### Example Output

> "The equipment issue was first detected at 2024-01-01 14:32:00, approximately 45 minutes before the shutdown event. The initial drift was observed in the pressure sensor. The primary contributing factor was pressure, which deviated 35% above baseline with 28% increased variability. Additional contributing factors included: flow, temperature. Trend reversals were observed in pressure, suggesting a fundamental change in equipment behavior."

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
