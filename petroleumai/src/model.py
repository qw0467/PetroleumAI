"""
Anomaly Detection Module for PetroleumAI

This module implements Isolation Forest-based anomaly detection
for identifying abnormal patterns in petroleum equipment sensor data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def prepare_features_for_model(df, feature_columns=None):
    """
    Prepare feature matrix for the anomaly detection model.
    
    Args:
        df: DataFrame with engineered features
        feature_columns: Specific columns to use (optional)
        
    Returns:
        Feature matrix (numpy array), column names used
    """
    if feature_columns is None:
        exclude_cols = ['timestamp', 'shutdown_event']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_cols if col not in exclude_cols]
    
    available_cols = [col for col in feature_columns if col in df.columns]
    
    if not available_cols:
        raise ValueError("No valid feature columns found")
    
    X = df[available_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, available_cols


def train_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of anomalies
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model, scaler used for normalization
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        bootstrap=False,
        n_jobs=-1
    )
    
    model.fit(X_scaled)
    
    return model, scaler


def compute_anomaly_scores(model, scaler, X):
    """
    Compute anomaly scores for each data point.
    
    Args:
        model: Trained Isolation Forest model
        scaler: Fitted StandardScaler
        X: Feature matrix
        
    Returns:
        Anomaly scores (higher = more anomalous)
    """
    X_scaled = scaler.transform(X)
    raw_scores = model.decision_function(X_scaled)
    anomaly_scores = -raw_scores
    
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    if max_score - min_score > 0:
        normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.zeros_like(anomaly_scores)
    
    return normalized_scores


def compute_anomaly_labels(model, scaler, X):
    """
    Compute binary anomaly labels for each data point.
    
    Args:
        model: Trained Isolation Forest model
        scaler: Fitted StandardScaler
        X: Feature matrix
        
    Returns:
        Binary labels (1 = normal, -1 = anomaly)
    """
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def compute_thresholds(anomaly_scores, percentiles=[75, 90, 95]):
    """
    Compute threshold values at various percentiles.
    
    Args:
        anomaly_scores: Array of anomaly scores
        percentiles: List of percentile values to compute
        
    Returns:
        Dictionary of threshold values
    """
    thresholds = {}
    for p in percentiles:
        thresholds[f'threshold_{p}'] = np.percentile(anomaly_scores, p)
    return thresholds


def run_anomaly_detection(df, contamination=0.1):
    """
    Run complete anomaly detection pipeline on a dataframe.
    
    Args:
        df: DataFrame with engineered features
        contamination: Expected proportion of anomalies
        
    Returns:
        DataFrame with anomaly scores and labels added,
        thresholds dictionary, model, scaler
    """
    X, feature_cols = prepare_features_for_model(df)
    
    model, scaler = train_isolation_forest(X, contamination=contamination)
    
    anomaly_scores = compute_anomaly_scores(model, scaler, X)
    anomaly_labels = compute_anomaly_labels(model, scaler, X)
    
    thresholds = compute_thresholds(anomaly_scores)
    
    result_df = df.copy()
    result_df['anomaly_score'] = anomaly_scores
    result_df['anomaly_label'] = anomaly_labels
    result_df['is_anomaly'] = (anomaly_labels == -1).astype(int)
    
    return result_df, thresholds, model, scaler


def classify_severity(anomaly_score, thresholds):
    """
    Classify anomaly severity based on score and thresholds.
    
    Args:
        anomaly_score: Single anomaly score value
        thresholds: Dictionary of threshold values
        
    Returns:
        Severity level string
    """
    if anomaly_score >= thresholds.get('threshold_95', 0.95):
        return 'Critical'
    elif anomaly_score >= thresholds.get('threshold_90', 0.90):
        return 'High'
    elif anomaly_score >= thresholds.get('threshold_75', 0.75):
        return 'Medium'
    else:
        return 'Low'


if __name__ == "__main__":
    from simulate import generate_simulated_data
    from features import engineer_features
    
    df = generate_simulated_data(n_points=500)
    df_features = engineer_features(df)
    df_anomalies, thresholds, model, scaler = run_anomaly_detection(df_features)
    
    print("Thresholds:", thresholds)
    print("\nAnomaly distribution:")
    print(df_anomalies['is_anomaly'].value_counts())
    print("\nSample anomalies:")
    print(df_anomalies[df_anomalies['is_anomaly'] == 1][['timestamp', 'anomaly_score']].head())
