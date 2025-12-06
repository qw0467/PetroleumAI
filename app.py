"""
PetroleumAI - Predictive Maintenance Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import io

from petroleumai.src.simulate import generate_simulated_data, load_data
from petroleumai.src.features import engineer_features
from petroleumai.src.model import run_anomaly_detection
from petroleumai.src.kpis import compute_all_kpis, format_kpi_for_display
from petroleumai.src.visualize import (
    setup_plot_style,
    plot_all_sensors,
    plot_anomaly_timeline,
    plot_drift_chart
)

st.set_page_config(
    page_title="PetroleumAI - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

setup_plot_style()

if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'kpis' not in st.session_state:
    st.session_state['kpis'] = None
if 'thresholds' not in st.session_state:
    st.session_state['thresholds'] = None

st.title("⚙️ PetroleumAI - Predictive Maintenance Dashboard")
st.markdown("**Multivariate Time-Series Anomaly Detection for Petroleum Equipment**")

with st.sidebar:
    st.header("Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use Simulated Data", "Upload CSV File"]
    )
    
    if data_source == "Use Simulated Data":
        n_points = st.slider("Number of data points", 100, 2000, 1000, step=100)
        if st.button("Generate Data", type="primary"):
            st.session_state['raw_data'] = generate_simulated_data(n_points=n_points)
            st.session_state['data_loaded'] = True
            st.success(f"Generated {n_points} data points!")
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                st.session_state['raw_data'] = df
                st.session_state['data_loaded'] = True
                st.success(f"Loaded {len(df)} rows from file!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    st.divider()
    st.header("Visualization Options")
    show_anomalies = st.checkbox("Show Anomaly Overlay", value=True)
    show_drift = st.checkbox("Show Drift Regions", value=False)
    show_spikes = st.checkbox("Show Spikes", value=False)
    
    st.divider()
    st.header("Model Settings")
    contamination = st.slider(
        "Anomaly Contamination",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
        help="Expected proportion of anomalies in the data"
    )

if st.session_state['data_loaded'] and st.session_state['raw_data'] is not None:
    
    with st.spinner("Processing data..."):
        raw_df = st.session_state['raw_data']
        
        featured_df = engineer_features(raw_df)
        
        processed_df, thresholds, model, scaler = run_anomaly_detection(
            featured_df, 
            contamination=contamination
        )
        
        kpis = compute_all_kpis(processed_df, thresholds)
        formatted_kpis = format_kpi_for_display(kpis)
        
        st.session_state['processed_data'] = processed_df
        st.session_state['kpis'] = kpis
        st.session_state['thresholds'] = thresholds
    
    st.header("📊 Key Performance Indicators")
    
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        health_status = kpis['health_status']
        health_color = kpis['health_color']
        st.metric(
            label="Equipment Health",
            value=health_status,
        )
        st.markdown(
            f"<div style='width:100%;height:10px;background-color:{health_color};border-radius:5px;'></div>",
            unsafe_allow_html=True
        )
    
    with kpi_cols[1]:
        st.metric(
            label="Current Anomaly Score",
            value=formatted_kpis['anomaly_score']
        )
    
    with kpi_cols[2]:
        st.metric(
            label="Irregular Events",
            value=formatted_kpis['irregular_events']
        )
    
    with kpi_cols[3]:
        st.metric(
            label="Risk Level",
            value=formatted_kpis['risk']
        )
    
    with kpi_cols[4]:
        st.metric(
            label="Anomaly Rate",
            value=formatted_kpis['anomaly_rate']
        )
    
    drift_cols = st.columns(4)
    drift_status = kpis['drift_detected']
    sensors = ['pressure', 'temperature', 'flow', 'vibration']
    for i, sensor in enumerate(sensors):
        with drift_cols[i]:
            has_drift = drift_status.get(sensor, False)
            status_icon = "🔴" if has_drift else "🟢"
            st.markdown(f"**{sensor.capitalize()}** {status_icon}")
            st.caption("Drift Detected" if has_drift else "Stable")
    
    st.divider()
    
    st.header("📈 Sensor Time Series")
    
    processed_df = st.session_state['processed_data']
    
    fig_sensors = plot_all_sensors(
        processed_df,
        show_anomalies=show_anomalies,
        show_drift=show_drift,
        show_spikes=show_spikes
    )
    st.pyplot(fig_sensors)
    
    st.divider()
    
    st.header("🔍 Anomaly Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Anomaly Score Timeline")
        fig_anomaly = plot_anomaly_timeline(processed_df)
        st.pyplot(fig_anomaly)
    
    with col2:
        st.subheader("Drift Detection by Sensor")
        fig_drift = plot_drift_chart(processed_df)
        st.pyplot(fig_drift)
    
    st.divider()
    
    st.header("📥 Download Processed Data")
    
    csv_buffer = io.StringIO()
    processed_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="processed_sensor_data.csv",
            mime="text/csv",
            type="primary"
        )
    with col2:
        st.caption(f"Processed data contains {len(processed_df)} rows and {len(processed_df.columns)} columns")
    
    with st.expander("View Data Preview"):
        st.dataframe(processed_df.head(100), use_container_width=True)

else:
    st.info("👈 Use the sidebar to load data. You can generate simulated sensor data or upload your own CSV file.")
    
    st.markdown("""
    ### Getting Started
    
    1. **Generate Simulated Data**: Click "Generate Data" in the sidebar to create synthetic sensor readings
    2. **Upload Your Data**: Or upload a CSV file with columns: `timestamp`, `pressure`, `temperature`, `flow`, `vibration`
    
    ### Features
    
    - **Real-time KPI Monitoring**: Track equipment health, anomaly scores, and risk levels
    - **Anomaly Detection**: Isolation Forest algorithm identifies abnormal patterns
    - **Feature Engineering**: Rolling statistics, trend analysis, spike and drift detection
    - **Interactive Visualizations**: Toggle overlays for anomalies, drift regions, and spikes
    - **Data Export**: Download processed data with all computed features
    """)

st.sidebar.divider()
st.sidebar.caption("PetroleumAI v1.0 | Predictive Maintenance MVP")
