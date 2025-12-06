"""
Sensor Data Simulation Module for PetroleumAI

This module generates synthetic sensor data for petroleum plant equipment,
including pressure, temperature, flow, and vibration readings with realistic
patterns like drift, spikes, and shutdown events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_base_signal(n_points, base_value, noise_level=0.05):
    """
    Generate a base signal with random noise.
    
    Args:
        n_points: Number of data points to generate
        base_value: The baseline value for the signal
        noise_level: Standard deviation of noise as fraction of base value
        
    Returns:
        numpy array with the base signal
    """
    noise = np.random.normal(0, base_value * noise_level, n_points)
    return base_value + noise


def add_drift(signal, drift_start, drift_rate=0.001):
    """
    Add gradual drift to a signal starting from a specific point.
    
    Args:
        signal: The input signal array
        drift_start: Index where drift begins
        drift_rate: Rate of drift per time step
        
    Returns:
        Signal with drift added
    """
    drifted_signal = signal.copy()
    n_points = len(signal)
    for i in range(drift_start, n_points):
        drifted_signal[i] += (i - drift_start) * drift_rate * signal[drift_start]
    return drifted_signal


def add_spikes(signal, n_spikes=5, spike_magnitude=2.0):
    """
    Add random spikes to the signal.
    
    Args:
        signal: The input signal array
        n_spikes: Number of spikes to add
        spike_magnitude: Magnitude of spikes as multiplier
        
    Returns:
        Signal with spikes, list of spike indices
    """
    spiked_signal = signal.copy()
    spike_indices = np.random.choice(len(signal), n_spikes, replace=False)
    for idx in spike_indices:
        direction = np.random.choice([-1, 1])
        spiked_signal[idx] += direction * spike_magnitude * np.std(signal)
    return spiked_signal, spike_indices.tolist()


def add_instability(signal, instability_start, instability_end, factor=3.0):
    """
    Add a period of instability (increased variance) to the signal.
    
    Args:
        signal: The input signal array
        instability_start: Start index of instability period
        instability_end: End index of instability period
        factor: Multiplier for noise during instability
        
    Returns:
        Signal with instability period
    """
    unstable_signal = signal.copy()
    base_std = np.std(signal)
    for i in range(instability_start, min(instability_end, len(signal))):
        unstable_signal[i] += np.random.normal(0, base_std * factor)
    return unstable_signal


def add_shutdown_event(signal, shutdown_start, shutdown_duration=20, recovery_time=30):
    """
    Simulate a shutdown event where values drop and then recover.
    
    Args:
        signal: The input signal array
        shutdown_start: Index where shutdown begins
        shutdown_duration: Number of points during shutdown
        recovery_time: Number of points for recovery
        
    Returns:
        Signal with shutdown event
    """
    shutdown_signal = signal.copy()
    n_points = len(signal)
    
    shutdown_end = min(shutdown_start + shutdown_duration, n_points)
    for i in range(shutdown_start, shutdown_end):
        progress = (i - shutdown_start) / shutdown_duration
        shutdown_signal[i] = signal[i] * (1 - 0.8 * progress)
    
    recovery_end = min(shutdown_end + recovery_time, n_points)
    for i in range(shutdown_end, recovery_end):
        progress = (i - shutdown_end) / recovery_time
        shutdown_signal[i] = signal[i] * (0.2 + 0.8 * progress)
    
    return shutdown_signal


def generate_simulated_data(n_points=1000, start_date=None, save_path=None):
    """
    Generate a complete simulated dataset with all sensor readings.
    
    Args:
        n_points: Number of data points to generate
        start_date: Starting datetime for the time series
        save_path: Path to save the CSV file (optional)
        
    Returns:
        pandas DataFrame with simulated sensor data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(hours=n_points)
    
    timestamps = [start_date + timedelta(hours=i) for i in range(n_points)]
    
    pressure = generate_base_signal(n_points, base_value=100, noise_level=0.03)
    pressure = add_drift(pressure, drift_start=int(n_points * 0.6), drift_rate=0.002)
    pressure, pressure_spikes = add_spikes(pressure, n_spikes=8, spike_magnitude=2.5)
    
    temperature = generate_base_signal(n_points, base_value=75, noise_level=0.04)
    temperature = add_instability(temperature, 
                                   instability_start=int(n_points * 0.3),
                                   instability_end=int(n_points * 0.4),
                                   factor=2.5)
    temperature, temp_spikes = add_spikes(temperature, n_spikes=5, spike_magnitude=2.0)
    
    flow = generate_base_signal(n_points, base_value=50, noise_level=0.05)
    flow = add_drift(flow, drift_start=int(n_points * 0.7), drift_rate=-0.001)
    shutdown_idx = int(n_points * 0.5)
    flow = add_shutdown_event(flow, shutdown_start=shutdown_idx, 
                              shutdown_duration=15, recovery_time=25)
    
    vibration = generate_base_signal(n_points, base_value=2.5, noise_level=0.1)
    vibration = add_instability(vibration,
                                 instability_start=int(n_points * 0.45),
                                 instability_end=int(n_points * 0.55),
                                 factor=4.0)
    vibration, vib_spikes = add_spikes(vibration, n_spikes=10, spike_magnitude=3.0)
    
    shutdown_flag = np.zeros(n_points)
    shutdown_flag[shutdown_idx:shutdown_idx + 40] = 1
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'pressure': pressure,
        'temperature': temperature,
        'flow': flow,
        'vibration': vibration,
        'shutdown_event': shutdown_flag.astype(int)
    })
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    
    return df


def load_data(file_path):
    """
    Load sensor data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pandas DataFrame with the loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'simulated_data.csv')
    df = generate_simulated_data(n_points=1000, save_path=data_path)
    print(f"Generated {len(df)} data points")
    print(df.head())
