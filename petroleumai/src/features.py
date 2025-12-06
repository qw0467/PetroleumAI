"""
Feature Engineering Module for PetroleumAI

This module computes derived features from raw sensor data including
rolling statistics, trend analysis, spike detection, drift detection,
cross-correlation, and instability scores.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_rolling_mean(series, window=20):
    """
    Compute rolling mean for a time series.
    
    Args:
        series: pandas Series or numpy array
        window: Rolling window size
        
    Returns:
        Rolling mean values
    """
    return pd.Series(series).rolling(window=window, min_periods=1).mean()


def compute_rolling_std(series, window=20):
    """
    Compute rolling standard deviation for a time series.
    
    Args:
        series: pandas Series or numpy array
        window: Rolling window size
        
    Returns:
        Rolling standard deviation values
    """
    return pd.Series(series).rolling(window=window, min_periods=1).std().fillna(0)


def compute_slope(series, window=10):
    """
    Compute rolling slope/trend using linear regression.
    
    Args:
        series: pandas Series or numpy array
        window: Window size for slope calculation
        
    Returns:
        Rolling slope values
    """
    series = pd.Series(series)
    slopes = []
    
    for i in range(len(series)):
        if i < window - 1:
            slopes.append(0)
        else:
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            if len(y) == window and not np.any(np.isnan(y)):
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(0)
    
    return pd.Series(slopes, index=series.index)


def detect_spikes(series, threshold=3.0):
    """
    Detect spikes using z-score method.
    
    Args:
        series: pandas Series or numpy array
        threshold: Z-score threshold for spike detection
        
    Returns:
        Boolean series indicating spike locations
    """
    series = pd.Series(series)
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def detect_drift(series, window=50, threshold=0.1):
    """
    Detect drift by comparing rolling mean to overall mean.
    
    Args:
        series: pandas Series or numpy array
        window: Window size for rolling mean
        threshold: Threshold for drift detection (fraction of mean)
        
    Returns:
        Boolean series indicating drift presence
    """
    series = pd.Series(series)
    overall_mean = series.mean()
    rolling_mean = compute_rolling_mean(series, window)
    
    if overall_mean == 0:
        return pd.Series([False] * len(series), index=series.index)
    
    deviation = np.abs(rolling_mean - overall_mean) / np.abs(overall_mean)
    return deviation > threshold


def compute_cross_correlation(series1, series2, max_lag=10):
    """
    Compute cross-correlation between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        max_lag: Maximum lag to consider
        
    Returns:
        Maximum cross-correlation value
    """
    s1 = pd.Series(series1).dropna()
    s2 = pd.Series(series2).dropna()
    
    min_len = min(len(s1), len(s2))
    s1 = s1.iloc[:min_len]
    s2 = s2.iloc[:min_len]
    
    if len(s1) < 2 or s1.std() == 0 or s2.std() == 0:
        return 0
    
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = s1.iloc[:lag].corr(s2.iloc[-lag:])
        elif lag > 0:
            corr = s1.iloc[lag:].corr(s2.iloc[:-lag])
        else:
            corr = s1.corr(s2)
        
        if not np.isnan(corr):
            correlations.append(abs(corr))
    
    return max(correlations) if correlations else 0


def compute_instability_score(series, window=20):
    """
    Compute instability score based on rolling variance changes.
    
    Args:
        series: pandas Series or numpy array
        window: Window size for calculation
        
    Returns:
        Instability score series (0-1 normalized)
    """
    series = pd.Series(series)
    rolling_std = compute_rolling_std(series, window)
    overall_std = series.std()
    
    if overall_std == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    instability = rolling_std / overall_std
    instability_normalized = (instability - instability.min()) / (instability.max() - instability.min() + 1e-10)
    
    return instability_normalized


def engineer_features(df, sensor_columns=None):
    """
    Apply all feature engineering to a dataframe.
    
    Args:
        df: Input DataFrame with sensor data
        sensor_columns: List of sensor column names to process
        
    Returns:
        DataFrame with original data and engineered features
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    available_sensors = [col for col in sensor_columns if col in df.columns]
    
    if not available_sensors:
        raise ValueError("No valid sensor columns found in the dataframe")
    
    result_df = df.copy()
    
    for col in available_sensors:
        result_df[f'{col}_rolling_mean'] = compute_rolling_mean(df[col])
        result_df[f'{col}_rolling_std'] = compute_rolling_std(df[col])
        result_df[f'{col}_slope'] = compute_slope(df[col])
        result_df[f'{col}_spike'] = detect_spikes(df[col]).astype(int)
        result_df[f'{col}_drift'] = detect_drift(df[col]).astype(int)
        result_df[f'{col}_instability'] = compute_instability_score(df[col])
    
    cross_corr_features = {}
    for i, col1 in enumerate(available_sensors):
        for col2 in available_sensors[i+1:]:
            corr_name = f'{col1}_{col2}_xcorr'
            cross_corr_features[corr_name] = compute_cross_correlation(df[col1], df[col2])
    
    for name, value in cross_corr_features.items():
        result_df[name] = value
    
    instability_cols = [f'{col}_instability' for col in available_sensors]
    result_df['overall_instability'] = result_df[instability_cols].mean(axis=1)
    
    return result_df


if __name__ == "__main__":
    np.random.seed(42)
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
        'pressure': np.random.normal(100, 5, 100),
        'temperature': np.random.normal(75, 3, 100),
        'flow': np.random.normal(50, 2, 100),
        'vibration': np.random.normal(2.5, 0.5, 100)
    })
    
    featured_data = engineer_features(test_data)
    print("Feature columns:", [col for col in featured_data.columns if col not in test_data.columns])
    print(featured_data.head())
