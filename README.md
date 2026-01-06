# PetroleumAI — Real-Time Anomaly Detection & Predictive Maintenance System

A real-time anomaly detection and predictive maintenance system for mission-critical industrial and scientific infrastructure using multivariate time-series analysis and machine learning.

---

## Overview

PetroleumAI is an operator-facing monitoring and diagnostics platform designed for environments that require **continuous 24/7 uptime**, rapid fault detection, and clear engineering-style explanations.

The system ingests multivariate sensor data (pressure, temperature, flow, vibration), performs feature engineering on streaming signals, and applies machine-learning-based anomaly detection to identify early warning signs of equipment degradation — before failures occur.

While inspired by industrial petroleum systems, the architecture and methodology are directly applicable to **large-scale scientific and operational facilities**, including power systems, cooling plants, cryogenics, and compute infrastructure.

---

## Why This Project

Large technical facilities operate under strict uptime and safety requirements.  
In these environments, failures are rarely sudden — they are preceded by subtle signal changes:

- slow sensor drift  
- increasing variance  
- abnormal correlations  
- early trend reversals  

This project was built to prototype an **early-warning and diagnostic system** that focuses on:

- Continuous monitoring  
- Rapid fault identification  
- Interpretable results for operators  
- Engineering-style explanations, not just model scores  

The goal is not only to detect anomalies, but to **explain what changed, when it changed, and why it matters.**

---

## Key Capabilities

- **Real-Time KPI Monitoring**  
  Track equipment health, anomaly scores, risk levels, and system stability.

- **Machine-Learning Anomaly Detection**  
  Isolation Forest detects abnormal behavior in high-dimensional sensor data.

- **Advanced Feature Engineering**  
  Automated computation of:
  - rolling statistics  
  - trend slopes  
  - spike detection  
  - drift indicators  
  - cross-sensor correlations  

- **Root Cause Analysis**  
  Engineering-style diagnostics to identify:
  - first abnormal signal  
  - warning lead time  
  - dominant causal sensors  
  - breakdowns in normal operating relationships  

- **Interactive Visualization**  
  Operator dashboard with overlays for:
  - anomalies  
  - drift regions  
  - spikes  
  - risk zones  

- **Exportable Engineering Reports**  
  Generate structured diagnostic summaries for incident review and post-mortems.

---

## System Architecture
petroleumai/
├── data/ # Data storage
├── src/
│ ├── simulate.py # Sensor data simulation
│ ├── features.py # Feature engineering pipeline
│ ├── model.py # Anomaly detection (Isolation Forest)
│ ├── kpis.py # Health & risk metric computation
│ ├── visualize.py # Plotting and dashboard utilities
│ └── root_cause.py # Diagnostic & root cause analysis
├── app.py # Streamlit operator dashboard
└── README.md

---

## Running the System

### 1. Install dependencies

pip install -r requirements.txt
### 2. Launch the dashboard
streamlit run app.py --server.port 5000
The dashboard will open automatically in your browser.
---
## Data Input Options

### 1.Simulated Sensor Data
Generate synthetic operational data with configurable noise, drift, and fault scenarios.
### 2.CSV Upload
Upload your own dataset with the following columns:
timestamp, pressure, temperature, flow, vibration
---
## Operational Metrics

### Equipment Health
Overall system status: Normal / Warning / Critical
### Anomaly Score
Normalized 0–1 scale representing abnormality level
### Irregular Events
Count of detected anomaly points
### Risk Level
Composite risk derived from:
health status
anomaly density
drift persistence

### Drift Detection
Continuous per-sensor monitoring for slow degradation patterns
---
## Root Cause Analysis
The Root Cause Analysis module performs structured diagnostics to explain how and why anomalies emerged.

### What It Determines
### 1. First Abnormality Detection
Identifies the earliest sensor deviation and its timestamp.

### 2. Warning Lead Time
Measures how far in advance early indicators appeared before a shutdown-level event.

### 3. Sensor Contribution Ranking
Quantifies each sensor’s influence on anomaly scores.

### 4. Baseline Comparison
Compares abnormal operation to normal conditions:
mean deviation
variability increase
trend reversals

### 5. Correlation Breakdown
Detects loss of normal relationships between sensors.
---
## Root Cause Workflow
### 1. Load data (simulated or uploaded)
### 2. Navigate to Root Cause Analysis tab
### 3. Select shutdown reference:
Auto-detect
Manual timestamp
Last datapoint
### 4. Run analysis
--- 

## Example Diagnostic Output
“The first abnormal behavior was detected at 2024-01-01 14:32:00, approximately 45 minutes before the shutdown event.
Initial drift was observed in the pressure sensor. Pressure deviated 35% above baseline with a 28% increase in variability, indicating progressive system instability.
Secondary contributing factors included flow and temperature.
Trend reversals in pressure suggest a fundamental change in equipment behavior prior to failure.”
--- 
## Technical Approach
### Anomaly Detection
Algorithm: Isolation Forest
Feature scaling: StandardScaler
Scores normalized to a 0–1 operational scale
Contamination parameter controls expected anomaly rate
--- 
## Feature Engineering
### For each sensor stream:
Rolling mean and standard deviation
Trend slope estimation
Spike detection via z-score
Drift detection using rolling vs global baselines
Cross-correlation between sensor pairs
Instability scoring based on variance changes
--- 
## Design Philosophy
This system prioritizes:
Interpretability over black-box predictions
Early warning over post-failure analysis
Operator usability over raw model complexity
System thinking over isolated metrics
The focus is on delivering actionable operational insight, not just machine-learning output.
--- 
## Dependencies
streamlit
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
--- 
## Future Work
Streaming ingestion (Kafka / WebSockets)
Model drift monitoring
Multi-model ensemble detection
Fault classification
Deployment via Docker for edge environments

## Author
## Muhammad Qasim Ayyaz
## Applied AI Engineer — Real-Time Systems & Operational Analytics