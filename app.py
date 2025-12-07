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
from petroleumai.src.root_cause import (
    identify_root_cause,
    compare_to_baseline,
    generate_engineer_report
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
if 'root_cause_results' not in st.session_state:
    st.session_state['root_cause_results'] = None

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
            st.session_state['root_cause_results'] = None
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
                st.session_state['root_cause_results'] = None
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
    
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Root Cause Analysis", "📥 Data Export"])
    
    with tab1:
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
    
    with tab2:
        st.header("🔍 Root Cause Analysis")
        st.markdown("Identify what went wrong and when the first abnormality occurred.")
        
        processed_df = st.session_state['processed_data']
        
        st.subheader("Analysis Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            shutdown_option = st.radio(
                "Shutdown Event Selection:",
                ["Auto-detect from data", "Select manually", "Use last timestamp"]
            )
        
        shutdown_time = None
        
        with col2:
            if shutdown_option == "Select manually":
                if 'timestamp' in processed_df.columns:
                    min_time = processed_df['timestamp'].min()
                    max_time = processed_df['timestamp'].max()
                    
                    if hasattr(min_time, 'to_pydatetime'):
                        min_time = min_time.to_pydatetime()
                        max_time = max_time.to_pydatetime()
                    
                    shutdown_date = st.date_input(
                        "Shutdown Date",
                        value=max_time.date() if hasattr(max_time, 'date') else max_time
                    )
                    shutdown_hour = st.time_input(
                        "Shutdown Time",
                        value=max_time.time() if hasattr(max_time, 'time') else None
                    )
                    
                    if shutdown_date and shutdown_hour:
                        shutdown_time = pd.Timestamp.combine(shutdown_date, shutdown_hour)
            elif shutdown_option == "Use last timestamp":
                shutdown_time = processed_df['timestamp'].iloc[-1]
                st.info(f"Using last timestamp: {shutdown_time}")
            else:
                if 'shutdown_event' in processed_df.columns and processed_df['shutdown_event'].sum() > 0:
                    shutdown_events = processed_df[processed_df['shutdown_event'] == 1]
                    shutdown_time = shutdown_events['timestamp'].iloc[0]
                    st.info(f"Auto-detected shutdown at: {shutdown_time}")
                else:
                    shutdown_time = processed_df['timestamp'].iloc[-1]
                    st.info(f"No shutdown events detected. Using last timestamp: {shutdown_time}")
        
        if st.button("Run Root Cause Analysis", type="primary"):
            with st.spinner("Analyzing root causes..."):
                try:
                    results = identify_root_cause(
                        processed_df,
                        shutdown_time=shutdown_time
                    )
                    st.session_state['root_cause_results'] = results
                    st.success("Analysis complete!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state['root_cause_results'] = None
        
        if st.session_state['root_cause_results'] is not None:
            results = st.session_state['root_cause_results']
            
            st.divider()
            
            st.subheader("Key Findings")
            
            findings_cols = st.columns(3)
            
            with findings_cols[0]:
                first_time = results.get('first_abnormal_time')
                if first_time is not None:
                    time_str = first_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(first_time, 'strftime') else str(first_time)
                else:
                    time_str = "Not detected"
                st.metric(
                    label="First Abnormality",
                    value=time_str[:19] if len(time_str) > 19 else time_str
                )
            
            with findings_cols[1]:
                time_before = results.get('time_before_shutdown_minutes')
                if time_before is not None:
                    st.metric(
                        label="Warning Lead Time",
                        value=f"{time_before:.0f} min"
                    )
                else:
                    st.metric(
                        label="Warning Lead Time",
                        value="N/A"
                    )
            
            with findings_cols[2]:
                primary_sensor = results.get('primary_causal_sensor')
                st.metric(
                    label="Primary Causal Sensor",
                    value=primary_sensor.upper() if primary_sensor else "Unknown"
                )
            
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sensor Contribution Ranking")
                
                contributions = results.get('sensor_contributions', [])
                if contributions:
                    contrib_df = pd.DataFrame(contributions, columns=['Sensor', 'Score'])
                    contrib_df['Sensor'] = contrib_df['Sensor'].str.capitalize()
                    contrib_df['Score'] = contrib_df['Score'].round(3)
                    contrib_df['Rank'] = range(1, len(contrib_df) + 1)
                    contrib_df = contrib_df[['Rank', 'Sensor', 'Score']]
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No contribution data available")
            
            with col2:
                st.subheader("Baseline Comparison")
                
                baseline_comp = results.get('differences_from_baseline', {})
                if baseline_comp:
                    comparison_data = []
                    for sensor, metrics in baseline_comp.items():
                        comparison_data.append({
                            'Sensor': sensor.capitalize(),
                            'Baseline Mean': metrics.get('baseline_mean', 'N/A'),
                            'Abnormal Mean': metrics.get('abnormal_mean', 'N/A'),
                            'Deviation %': f"{metrics.get('mean_deviation_pct', 0):.1f}%",
                            'Variability +%': f"{metrics.get('variability_increase_pct', 0):.1f}%"
                        })
                    
                    comp_df = pd.DataFrame(comparison_data)
                    st.dataframe(comp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No baseline comparison data available")
            
            st.divider()
            
            st.subheader("Pattern Summary")
            patterns = results.get('abnormal_patterns_summary', 'No patterns detected')
            st.info(patterns)
            
            corr_info = results.get('correlation_breakdown', {})
            if corr_info.get('correlation_breakdown'):
                st.warning("Correlation breakdown detected between sensors")
                breakdown_pairs = corr_info.get('breakdown_pairs', [])
                for pair in breakdown_pairs:
                    st.write(f"- **{pair['sensors']}**: Correlation changed from {pair['early_correlation']:.3f} to {pair['late_correlation']:.3f}")
            
            st.divider()
            
            st.subheader("Root Cause Explanation")
            explanation = results.get('final_explanation', 'Unable to determine root cause.')
            st.markdown(f"> {explanation}")
            
            st.divider()
            
            with st.expander("📋 Full Engineering Report", expanded=False):
                full_report = generate_engineer_report(results)
                st.markdown(full_report)
                
                report_buffer = io.StringIO()
                report_buffer.write(full_report)
                report_data = report_buffer.getvalue()
                
                st.download_button(
                    label="Download Report as Markdown",
                    data=report_data,
                    file_name="root_cause_report.md",
                    mime="text/markdown"
                )
        else:
            st.info("Click 'Run Root Cause Analysis' to identify the root cause of equipment issues.")
    
    with tab3:
        st.header("📥 Download Processed Data")
        
        processed_df = st.session_state['processed_data']
        
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
    - **Root Cause Analysis**: Identify what went wrong and generate engineering reports
    - **Data Export**: Download processed data with all computed features
    """)

st.sidebar.divider()
st.sidebar.caption("PetroleumAI v1.1 | Predictive Maintenance with Root Cause Analysis")
