# PetroleumAI - Predictive Maintenance Dashboard

## Overview

PetroleumAI is a predictive maintenance MVP for petroleum plant equipment that uses multivariate time-series analysis and machine learning for anomaly detection. The system monitors sensor data (pressure, temperature, flow, vibration) from petroleum equipment and employs Isolation Forest algorithms to identify potential equipment issues before they cause failures.

The application is built as a Streamlit dashboard that provides real-time KPI monitoring, anomaly detection visualization, and data export capabilities. It supports both simulated sensor data generation and custom CSV data uploads.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
**Decision**: Streamlit-based web application  
**Rationale**: Streamlit provides rapid prototyping for data science applications with built-in support for interactive visualizations and minimal frontend code. Perfect for MVP deployment in Replit environment.  
**Port Configuration**: Runs on port 5000 (configured in streamlit command)

### Modular Code Organization
**Decision**: Separation of concerns into specialized modules under `petroleumai/src/`  
**Rationale**: Each module handles a specific domain (simulation, features, model, KPIs, visualization), making the codebase maintainable and testable.

**Module Breakdown**:
- `simulate.py` - Sensor data generation with realistic patterns (drift, spikes, shutdowns)
- `features.py` - Feature engineering (rolling statistics, trends, spike/drift detection)
- `model.py` - Isolation Forest anomaly detection implementation
- `kpis.py` - Equipment health metrics and status calculations
- `visualize.py` - Matplotlib/Seaborn chart generation with interactive toggles
- `root_cause.py` - Gap analysis and root cause identification for equipment failures

### Machine Learning Architecture
**Decision**: Isolation Forest for unsupervised anomaly detection  
**Rationale**: Isolation Forest excels at detecting anomalies in multivariate time-series data without requiring labeled training data. Ideal for predictive maintenance where failure patterns may be unknown.

**Implementation Details**:
- Uses sklearn's IsolationForest implementation
- StandardScaler for feature normalization
- Configurable contamination parameter (default 0.1)
- Feature matrix preparation with NaN handling

### Feature Engineering Pipeline
**Decision**: Time-series feature extraction with statistical and trend analysis  
**Rationale**: Raw sensor readings alone are insufficient for detecting subtle equipment degradation. Engineered features capture temporal patterns and anomalies.

**Computed Features**:
- Rolling mean and standard deviation (window-based)
- Slope/trend analysis using linear regression
- Spike detection (deviation from rolling statistics)
- Drift detection (gradual baseline shifts)
- Cross-correlation between sensors (optional)
- Instability scores

### State Management
**Decision**: Streamlit session state for data persistence  
**Rationale**: Maintains processed data, KPIs, and thresholds across user interactions without re-computation.

**Session State Keys**:
- `data_loaded` - Boolean flag for data availability
- `raw_data` - Original sensor readings
- `processed_data` - Data with engineered features and anomaly scores
- `kpis` - Computed key performance indicators
- `thresholds` - Anomaly detection thresholds
- `root_cause_results` - Results from root cause analysis

### Data Input Strategy
**Decision**: Dual-mode data ingestion (simulated + CSV upload)  
**Rationale**: Simulated data enables immediate demonstration and testing, while CSV upload supports real-world sensor data integration.

**Expected CSV Schema**:
- Required columns: `timestamp`, `pressure`, `temperature`, `flow`, `vibration`
- Timestamp format: Flexible (pandas datetime parsing)

### Visualization Strategy
**Decision**: Matplotlib/Seaborn with seaborn-v0_8-whitegrid style  
**Rationale**: Provides publication-quality plots with consistent styling. Interactive toggles for anomalies, drift regions, and spikes enhance exploratory analysis.

**Chart Types**:
- Multi-panel sensor timeseries
- Anomaly timeline visualization
- Drift period highlighting
- KPI metric displays with color-coded health status

### Health Status Classification
**Decision**: Three-tier equipment health system (Normal/Warning/Critical)  
**Rationale**: Provides actionable insights for maintenance teams using threshold-based classification.

**Thresholds**:
- Critical: max_score ≥ 95th percentile OR avg_score ≥ 90th percentile
- Warning: max_score ≥ 90th percentile OR avg_score ≥ 75th percentile
- Normal: Below warning thresholds

## External Dependencies

### Python Libraries
- **streamlit** - Web application framework
- **pandas** - Data manipulation and CSV I/O
- **numpy** - Numerical computations
- **scikit-learn** - Isolation Forest model and StandardScaler
- **matplotlib** - Core plotting library
- **seaborn** - Statistical visualization styling
- **scipy** - Statistical functions (used in feature engineering)

### Runtime Environment
- **Python 3** - Core runtime
- **pip** - Package management
- **Replit** - Deployment platform

### Data Storage
- **Local filesystem** - CSV data storage in `data/` directory
- **In-memory** - Session state for processed data (no persistent database)

### Notable Absence
The application currently does not integrate external databases, authentication systems, or third-party APIs. All data processing occurs in-memory with optional CSV export for persistence.