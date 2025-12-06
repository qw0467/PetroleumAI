"""
Visualization Module for PetroleumAI

This module creates charts and visualizations for sensor data,
anomaly detection results, and KPI displays.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style():
    """
    Set up consistent plot styling for all visualizations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


def plot_sensor_timeseries(df, sensor_column, show_anomalies=True, 
                            show_drift=False, show_spikes=False, ax=None):
    """
    Plot a single sensor time series with optional overlays.
    
    Args:
        df: DataFrame with sensor data and analysis results
        sensor_column: Name of the sensor column to plot
        show_anomalies: Whether to highlight anomaly points
        show_drift: Whether to highlight drift periods
        show_spikes: Whether to highlight spike points
        ax: Matplotlib axes object (optional)
        
    Returns:
        Matplotlib figure and axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.get_figure()
    
    if 'timestamp' in df.columns:
        x_data = pd.to_datetime(df['timestamp'])
    else:
        x_data = df.index
    
    ax.plot(x_data, df[sensor_column], 'b-', linewidth=1, 
            label=sensor_column.capitalize(), alpha=0.8)
    
    if show_anomalies and 'is_anomaly' in df.columns:
        anomaly_mask = df['is_anomaly'] == 1
        if anomaly_mask.any():
            ax.scatter(x_data[anomaly_mask], df.loc[anomaly_mask, sensor_column],
                      c='red', s=50, marker='o', label='Anomaly', zorder=5, alpha=0.7)
    
    drift_col = f'{sensor_column}_drift'
    if show_drift and drift_col in df.columns:
        drift_mask = df[drift_col] == 1
        if drift_mask.any():
            drift_regions = []
            in_drift = False
            start_idx = 0
            
            for i, is_drift in enumerate(drift_mask):
                if is_drift and not in_drift:
                    start_idx = i
                    in_drift = True
                elif not is_drift and in_drift:
                    drift_regions.append((start_idx, i-1))
                    in_drift = False
            
            if in_drift:
                drift_regions.append((start_idx, len(df)-1))
            
            for start, end in drift_regions:
                ax.axvspan(x_data.iloc[start], x_data.iloc[end], 
                          alpha=0.2, color='orange', label='Drift' if start == drift_regions[0][0] else '')
    
    spike_col = f'{sensor_column}_spike'
    if show_spikes and spike_col in df.columns:
        spike_mask = df[spike_col] == 1
        if spike_mask.any():
            ax.scatter(x_data[spike_mask], df.loc[spike_mask, sensor_column],
                      c='purple', s=80, marker='^', label='Spike', zorder=6, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(sensor_column.capitalize())
    ax.set_title(f'{sensor_column.capitalize()} Over Time')
    ax.legend(loc='upper right')
    
    if isinstance(x_data.iloc[0], pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig, ax


def plot_all_sensors(df, sensor_columns=None, show_anomalies=True,
                      show_drift=False, show_spikes=False):
    """
    Plot all sensor time series in a grid layout.
    
    Args:
        df: DataFrame with sensor data
        sensor_columns: List of sensor columns to plot
        show_anomalies: Whether to show anomaly overlay
        show_drift: Whether to show drift overlay
        show_spikes: Whether to show spike overlay
        
    Returns:
        Matplotlib figure
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    available_sensors = [col for col in sensor_columns if col in df.columns]
    n_sensors = len(available_sensors)
    
    if n_sensors == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No sensor data available', ha='center', va='center')
        return fig
    
    fig, axes = plt.subplots(n_sensors, 1, figsize=(12, 3*n_sensors), sharex=True)
    
    if n_sensors == 1:
        axes = [axes]
    
    for ax, sensor in zip(axes, available_sensors):
        plot_sensor_timeseries(df, sensor, show_anomalies, show_drift, show_spikes, ax)
    
    plt.tight_layout()
    return fig


def plot_anomaly_timeline(df):
    """
    Plot anomaly score over time with threshold lines.
    
    Args:
        df: DataFrame with anomaly scores
        
    Returns:
        Matplotlib figure
    """
    if 'anomaly_score' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No anomaly score data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if 'timestamp' in df.columns:
        x_data = pd.to_datetime(df['timestamp'])
    else:
        x_data = df.index
    
    colors = plt.cm.RdYlGn_r(df['anomaly_score'])
    
    ax.scatter(x_data, df['anomaly_score'], c=df['anomaly_score'], 
               cmap='RdYlGn_r', s=20, alpha=0.6)
    ax.plot(x_data, df['anomaly_score'], 'k-', alpha=0.3, linewidth=0.5)
    
    threshold_75 = np.percentile(df['anomaly_score'], 75)
    threshold_90 = np.percentile(df['anomaly_score'], 90)
    threshold_95 = np.percentile(df['anomaly_score'], 95)
    
    ax.axhline(y=threshold_75, color='yellow', linestyle='--', 
               label=f'75th percentile ({threshold_75:.3f})', alpha=0.7)
    ax.axhline(y=threshold_90, color='orange', linestyle='--', 
               label=f'90th percentile ({threshold_90:.3f})', alpha=0.7)
    ax.axhline(y=threshold_95, color='red', linestyle='--', 
               label=f'95th percentile ({threshold_95:.3f})', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Score Timeline')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    
    if isinstance(x_data.iloc[0], pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_drift_chart(df, sensor_columns=None):
    """
    Plot drift detection status for all sensors.
    
    Args:
        df: DataFrame with drift detection results
        sensor_columns: List of sensor columns
        
    Returns:
        Matplotlib figure
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    drift_cols = [f'{col}_drift' for col in sensor_columns if f'{col}_drift' in df.columns]
    
    if not drift_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No drift detection data', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    if 'timestamp' in df.columns:
        x_data = pd.to_datetime(df['timestamp'])
    else:
        x_data = df.index
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, col in enumerate(drift_cols):
        sensor_name = col.replace('_drift', '').capitalize()
        drift_values = df[col].values * (i + 1)
        ax.fill_between(x_data, i, i + df[col].values, 
                       alpha=0.5, color=colors[i % len(colors)], 
                       label=sensor_name)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Sensor')
    ax.set_title('Drift Detection by Sensor')
    ax.set_yticks(range(len(drift_cols)))
    ax.set_yticklabels([col.replace('_drift', '').capitalize() for col in drift_cols])
    ax.legend(loc='upper right')
    
    if isinstance(x_data.iloc[0], pd.Timestamp):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def plot_kpi_summary(kpis):
    """
    Create a visual summary of KPIs.
    
    Args:
        kpis: Dictionary of KPI values
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for ax in axes:
        ax.axis('off')
    
    health_colors = {'Normal': '#28a745', 'Warning': '#ffc107', 'Critical': '#dc3545', 'Unknown': '#6c757d'}
    health = kpis.get('health_status', 'Unknown')
    axes[0].add_patch(plt.Circle((0.5, 0.5), 0.4, color=health_colors.get(health, '#6c757d')))
    axes[0].text(0.5, 0.5, health, ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    axes[0].set_title('Equipment Health', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    score = kpis.get('current_anomaly_score', 0)
    axes[1].barh(['Score'], [score], color='#17a2b8', height=0.3)
    axes[1].barh(['Score'], [1-score], left=[score], color='#e9ecef', height=0.3)
    axes[1].text(0.5, 0, f'{score:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
    axes[1].set_xlim(0, 1)
    axes[1].set_title('Current Anomaly Score', fontsize=14, fontweight='bold')
    
    irregular = kpis.get('irregular_events_count', 0)
    axes[2].text(0.5, 0.5, str(irregular), ha='center', va='center', fontsize=40, fontweight='bold', color='#dc3545')
    axes[2].set_title('Irregular Events', fontsize=14, fontweight='bold')
    
    risk_level = kpis.get('risk_level', 'Unknown')
    risk_score = kpis.get('risk_score', 0)
    risk_colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
    axes[3].text(0.5, 0.6, risk_level, ha='center', va='center', fontsize=24, fontweight='bold', 
                color=risk_colors.get(risk_level, '#6c757d'))
    axes[3].text(0.5, 0.3, f'{risk_score}%', ha='center', va='center', fontsize=18)
    axes[3].set_title('Risk Level', fontsize=14, fontweight='bold')
    
    drift_detected = kpis.get('drift_detected', {})
    drift_text = '\n'.join([f"{k}: {'Yes' if v else 'No'}" for k, v in drift_detected.items()])
    if not drift_text:
        drift_text = 'No data'
    axes[4].text(0.5, 0.5, drift_text, ha='center', va='center', fontsize=12)
    axes[4].set_title('Drift Detection', fontsize=14, fontweight='bold')
    
    first_abnormal = kpis.get('first_abnormal_timestamp', 'N/A')
    axes[5].text(0.5, 0.5, str(first_abnormal)[:19] if first_abnormal else 'N/A', 
                ha='center', va='center', fontsize=12)
    axes[5].set_title('First Abnormal Event', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    setup_plot_style()
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'pressure': np.random.normal(100, 5, 100),
        'temperature': np.random.normal(75, 3, 100),
        'flow': np.random.normal(50, 2, 100),
        'vibration': np.random.normal(2.5, 0.5, 100),
        'anomaly_score': np.random.uniform(0, 1, 100),
        'is_anomaly': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'pressure_drift': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'pressure_spike': np.random.choice([0, 1], 100, p=[0.95, 0.05])
    })
    
    fig = plot_sensor_timeseries(test_df, 'pressure', show_anomalies=True)
    plt.savefig('test_sensor_plot.png')
    print("Test plot saved")
