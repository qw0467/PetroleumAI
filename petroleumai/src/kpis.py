"""
KPI Calculation Module for PetroleumAI

This module computes key performance indicators for equipment health
monitoring based on anomaly detection and feature analysis results.
"""

import numpy as np
import pandas as pd


def compute_equipment_health(anomaly_scores, thresholds):
    """
    Determine overall equipment health status.
    
    Args:
        anomaly_scores: Array of recent anomaly scores
        thresholds: Dictionary of threshold values
        
    Returns:
        Health status string and color code
    """
    if len(anomaly_scores) == 0:
        return 'Unknown', '#808080'
    
    recent_scores = anomaly_scores[-20:] if len(anomaly_scores) > 20 else anomaly_scores
    avg_score = np.mean(recent_scores)
    max_score = np.max(recent_scores)
    
    threshold_95 = thresholds.get('threshold_95', 0.95)
    threshold_90 = thresholds.get('threshold_90', 0.90)
    threshold_75 = thresholds.get('threshold_75', 0.75)
    
    if max_score >= threshold_95 or avg_score >= threshold_90:
        return 'Critical', '#FF0000'
    elif max_score >= threshold_90 or avg_score >= threshold_75:
        return 'Warning', '#FFA500'
    else:
        return 'Normal', '#00FF00'


def compute_current_anomaly_score(anomaly_scores):
    """
    Get the most recent anomaly score.
    
    Args:
        anomaly_scores: Array of anomaly scores
        
    Returns:
        Current (most recent) anomaly score
    """
    if len(anomaly_scores) == 0:
        return 0.0
    return float(anomaly_scores[-1])


def count_irregular_events(df):
    """
    Count the number of irregular/anomalous events detected.
    
    Args:
        df: DataFrame with anomaly detection results
        
    Returns:
        Count of irregular events
    """
    if 'is_anomaly' not in df.columns:
        return 0
    return int(df['is_anomaly'].sum())


def detect_sensor_drift(df, sensor_columns=None):
    """
    Detect which sensors have drift patterns.
    
    Args:
        df: DataFrame with drift detection results
        sensor_columns: List of sensor names to check
        
    Returns:
        Dictionary of sensors with drift flags
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    drift_status = {}
    for sensor in sensor_columns:
        drift_col = f'{sensor}_drift'
        if drift_col in df.columns:
            drift_detected = df[drift_col].sum() > len(df) * 0.1
            drift_status[sensor] = drift_detected
    
    return drift_status


def find_first_abnormal_timestamp(df):
    """
    Find the timestamp of the first detected anomaly.
    
    Args:
        df: DataFrame with anomaly detection results
        
    Returns:
        Timestamp of first anomaly or None
    """
    if 'is_anomaly' not in df.columns or 'timestamp' not in df.columns:
        return None
    
    anomalies = df[df['is_anomaly'] == 1]
    if len(anomalies) == 0:
        return None
    
    return anomalies['timestamp'].iloc[0]


def compute_overall_risk(health_status, irregular_count, drift_count, total_points):
    """
    Compute an overall risk indicator based on multiple factors.
    
    Args:
        health_status: Current health status string
        irregular_count: Number of irregular events
        drift_count: Number of sensors with drift
        total_points: Total number of data points
        
    Returns:
        Risk level string and numeric score (0-100)
    """
    risk_score = 0
    
    if health_status == 'Critical':
        risk_score += 50
    elif health_status == 'Warning':
        risk_score += 25
    
    if total_points > 0:
        anomaly_rate = irregular_count / total_points
        risk_score += min(30, anomaly_rate * 300)
    
    risk_score += drift_count * 5
    
    risk_score = min(100, risk_score)
    
    if risk_score >= 70:
        risk_level = 'High'
    elif risk_score >= 40:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return risk_level, round(risk_score, 1)


def compute_all_kpis(df, thresholds, sensor_columns=None):
    """
    Compute all KPIs from the processed data.
    
    Args:
        df: DataFrame with anomaly detection and feature results
        thresholds: Dictionary of threshold values
        sensor_columns: List of sensor column names
        
    Returns:
        Dictionary containing all KPI values
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    available_sensors = [col for col in sensor_columns if col in df.columns]
    
    anomaly_scores = df['anomaly_score'].values if 'anomaly_score' in df.columns else np.array([])
    
    health_status, health_color = compute_equipment_health(anomaly_scores, thresholds)
    current_score = compute_current_anomaly_score(anomaly_scores)
    irregular_count = count_irregular_events(df)
    drift_status = detect_sensor_drift(df, available_sensors)
    first_abnormal = find_first_abnormal_timestamp(df)
    
    drift_count = sum(drift_status.values())
    risk_level, risk_score = compute_overall_risk(
        health_status, irregular_count, drift_count, len(df)
    )
    
    kpis = {
        'health_status': health_status,
        'health_color': health_color,
        'current_anomaly_score': current_score,
        'irregular_events_count': irregular_count,
        'drift_detected': drift_status,
        'drift_count': drift_count,
        'first_abnormal_timestamp': first_abnormal,
        'risk_level': risk_level,
        'risk_score': risk_score,
        'total_data_points': len(df),
        'anomaly_rate': round(irregular_count / max(len(df), 1) * 100, 2)
    }
    
    return kpis


def format_kpi_for_display(kpis):
    """
    Format KPIs for display in the UI.
    
    Args:
        kpis: Dictionary of KPI values
        
    Returns:
        Dictionary of formatted KPI strings
    """
    formatted = {}
    
    formatted['health'] = f"{kpis['health_status']}"
    formatted['anomaly_score'] = f"{kpis['current_anomaly_score']:.3f}"
    formatted['irregular_events'] = f"{kpis['irregular_events_count']}"
    formatted['risk'] = f"{kpis['risk_level']} ({kpis['risk_score']}%)"
    formatted['anomaly_rate'] = f"{kpis['anomaly_rate']}%"
    
    drift_sensors = [sensor for sensor, drifted in kpis['drift_detected'].items() if drifted]
    formatted['drift_sensors'] = ', '.join(drift_sensors) if drift_sensors else 'None'
    
    if kpis['first_abnormal_timestamp'] is not None:
        formatted['first_abnormal'] = str(kpis['first_abnormal_timestamp'])
    else:
        formatted['first_abnormal'] = 'N/A'
    
    return formatted


if __name__ == "__main__":
    np.random.seed(42)
    
    test_thresholds = {'threshold_75': 0.5, 'threshold_90': 0.7, 'threshold_95': 0.85}
    
    test_df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'pressure': np.random.normal(100, 5, 100),
        'temperature': np.random.normal(75, 3, 100),
        'flow': np.random.normal(50, 2, 100),
        'vibration': np.random.normal(2.5, 0.5, 100),
        'anomaly_score': np.random.uniform(0, 1, 100),
        'is_anomaly': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'pressure_drift': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
        'temperature_drift': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'flow_drift': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
        'vibration_drift': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })
    
    kpis = compute_all_kpis(test_df, test_thresholds)
    formatted = format_kpi_for_display(kpis)
    
    print("KPIs:")
    for key, value in formatted.items():
        print(f"  {key}: {value}")
