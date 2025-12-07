"""
Root Cause Analysis Module for PetroleumAI

This module performs gap analysis and root cause analysis to identify
what caused equipment anomalies and potential shutdowns in petroleum plants.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


def find_first_abnormality(df, sensor_columns, anomaly_col='is_anomaly', 
                           drift_suffix='_drift', spike_suffix='_spike'):
    """
    Find the first timestamp where abnormal behavior was detected.
    
    Args:
        df: DataFrame with processed sensor data
        sensor_columns: List of sensor column names
        anomaly_col: Column name for anomaly labels
        drift_suffix: Suffix for drift detection columns
        spike_suffix: Suffix for spike detection columns
        
    Returns:
        Dictionary with first abnormal time and triggering sensor
    """
    first_abnormal = None
    first_sensor = None
    abnormality_type = None
    
    for sensor in sensor_columns:
        drift_col = f'{sensor}{drift_suffix}'
        spike_col = f'{sensor}{spike_suffix}'
        
        if drift_col in df.columns:
            drift_events = df[df[drift_col] == 1]
            if len(drift_events) > 0:
                drift_time = drift_events['timestamp'].iloc[0]
                if first_abnormal is None or drift_time < first_abnormal:
                    first_abnormal = drift_time
                    first_sensor = sensor
                    abnormality_type = 'drift'
        
        if spike_col in df.columns:
            spike_events = df[df[spike_col] == 1]
            if len(spike_events) > 0:
                spike_time = spike_events['timestamp'].iloc[0]
                if first_abnormal is None or spike_time < first_abnormal:
                    first_abnormal = spike_time
                    first_sensor = sensor
                    abnormality_type = 'spike'
    
    if anomaly_col in df.columns and first_abnormal is None:
        anomaly_events = df[df[anomaly_col] == 1]
        if len(anomaly_events) > 0:
            first_abnormal = anomaly_events['timestamp'].iloc[0]
            first_sensor = 'multiple'
            abnormality_type = 'anomaly'
    
    return {
        'first_abnormal_time': first_abnormal,
        'first_sensor': first_sensor,
        'abnormality_type': abnormality_type
    }


def rank_sensor_contribution(df, sensor_columns, anomaly_score_col='anomaly_score'):
    """
    Rank sensors by their contribution to anomaly scores.
    
    Args:
        df: DataFrame with processed sensor data
        sensor_columns: List of sensor column names
        anomaly_score_col: Column name for anomaly scores
        
    Returns:
        List of tuples (sensor, contribution_score) sorted by contribution
    """
    contributions = []
    
    anomalous_df = df[df.get('is_anomaly', pd.Series([0]*len(df))) == 1]
    normal_df = df[df.get('is_anomaly', pd.Series([1]*len(df))) == 0]
    
    if len(anomalous_df) == 0 or len(normal_df) == 0:
        for sensor in sensor_columns:
            if sensor in df.columns:
                std = df[sensor].std()
                contributions.append((sensor, std if std > 0 else 0))
        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions
    
    for sensor in sensor_columns:
        if sensor not in df.columns:
            continue
            
        normal_mean = normal_df[sensor].mean()
        normal_std = normal_df[sensor].std()
        anomalous_mean = anomalous_df[sensor].mean()
        
        if normal_std > 0:
            deviation = abs(anomalous_mean - normal_mean) / normal_std
        else:
            deviation = abs(anomalous_mean - normal_mean)
        
        instability_col = f'{sensor}_instability'
        if instability_col in df.columns:
            instability_increase = (
                anomalous_df[instability_col].mean() - 
                normal_df[instability_col].mean()
            )
            deviation += max(0, instability_increase) * 2
        
        drift_col = f'{sensor}_drift'
        if drift_col in df.columns:
            drift_rate = anomalous_df[drift_col].mean()
            deviation += drift_rate * 1.5
        
        spike_col = f'{sensor}_spike'
        if spike_col in df.columns:
            spike_rate = anomalous_df[spike_col].mean()
            deviation += spike_rate * 2
        
        contributions.append((sensor, deviation))
    
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions


def compare_to_baseline(df, abnormal_start_time=None, abnormal_end_time=None,
                        sensor_columns=None, baseline_fraction=0.3):
    """
    Compare abnormal region to baseline (normal) behavior.
    
    Args:
        df: DataFrame with processed sensor data
        abnormal_start_time: Start of abnormal window
        abnormal_end_time: End of abnormal window
        sensor_columns: List of sensor column names
        baseline_fraction: Fraction of data to use as baseline
        
    Returns:
        Dictionary with baseline vs abnormal comparison metrics
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    sensor_columns = [col for col in sensor_columns if col in df.columns]
    
    if abnormal_start_time is None:
        if 'is_anomaly' in df.columns:
            anomalies = df[df['is_anomaly'] == 1]
            if len(anomalies) > 0:
                abnormal_start_time = anomalies['timestamp'].iloc[0]
    
    if abnormal_start_time is None:
        n_baseline = int(len(df) * baseline_fraction)
        baseline_df = df.iloc[:n_baseline]
        abnormal_df = df.iloc[n_baseline:]
    else:
        baseline_df = df[df['timestamp'] < abnormal_start_time]
        abnormal_df = df[df['timestamp'] >= abnormal_start_time]
        
        if abnormal_end_time is not None:
            abnormal_df = abnormal_df[abnormal_df['timestamp'] <= abnormal_end_time]
    
    if len(baseline_df) == 0:
        baseline_df = df.iloc[:int(len(df) * baseline_fraction)]
    if len(abnormal_df) == 0:
        abnormal_df = df.iloc[int(len(df) * baseline_fraction):]
    
    comparison = {}
    
    for sensor in sensor_columns:
        baseline_mean = baseline_df[sensor].mean()
        baseline_std = baseline_df[sensor].std()
        abnormal_mean = abnormal_df[sensor].mean()
        abnormal_std = abnormal_df[sensor].std()
        
        if baseline_mean != 0:
            mean_deviation_pct = ((abnormal_mean - baseline_mean) / abs(baseline_mean)) * 100
        else:
            mean_deviation_pct = 0
        
        if baseline_std != 0:
            variability_increase = ((abnormal_std - baseline_std) / baseline_std) * 100
        else:
            variability_increase = 0
        
        slope_col = f'{sensor}_slope'
        trend_reversal = False
        if slope_col in df.columns:
            baseline_slope = baseline_df[slope_col].mean()
            abnormal_slope = abnormal_df[slope_col].mean()
            if (baseline_slope > 0 and abnormal_slope < 0) or (baseline_slope < 0 and abnormal_slope > 0):
                trend_reversal = True
        
        comparison[sensor] = {
            'baseline_mean': round(baseline_mean, 3),
            'baseline_std': round(baseline_std, 3),
            'abnormal_mean': round(abnormal_mean, 3),
            'abnormal_std': round(abnormal_std, 3),
            'mean_deviation_pct': round(mean_deviation_pct, 2),
            'variability_increase_pct': round(variability_increase, 2),
            'trend_reversal': trend_reversal
        }
    
    return comparison


def detect_correlation_breakdown(df, sensor_columns=None, window=50):
    """
    Detect if sensor correlations broke down during anomalous periods.
    
    Args:
        df: DataFrame with processed sensor data
        sensor_columns: List of sensor column names
        window: Window size for correlation calculation
        
    Returns:
        Dictionary with correlation breakdown information
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    sensor_columns = [col for col in sensor_columns if col in df.columns]
    
    if len(sensor_columns) < 2:
        return {'correlation_breakdown': False, 'details': 'Insufficient sensors'}
    
    mid_point = len(df) // 2
    early_df = df.iloc[:mid_point]
    late_df = df.iloc[mid_point:]
    
    breakdowns = []
    
    for i, s1 in enumerate(sensor_columns):
        for s2 in sensor_columns[i+1:]:
            early_corr = early_df[s1].corr(early_df[s2])
            late_corr = late_df[s1].corr(late_df[s2])
            
            if pd.isna(early_corr) or pd.isna(late_corr):
                continue
            
            corr_change = abs(late_corr - early_corr)
            
            if corr_change > 0.3:
                breakdowns.append({
                    'sensors': f'{s1}-{s2}',
                    'early_correlation': round(early_corr, 3),
                    'late_correlation': round(late_corr, 3),
                    'change': round(corr_change, 3)
                })
    
    return {
        'correlation_breakdown': len(breakdowns) > 0,
        'breakdown_pairs': breakdowns
    }


def identify_root_cause(df, anomaly_col='is_anomaly', drift_cols=None, 
                        spike_cols=None, shutdown_time=None,
                        sensor_columns=None):
    """
    Perform comprehensive root cause analysis.
    
    Args:
        df: DataFrame with processed sensor data and anomaly labels
        anomaly_col: Column name for anomaly labels
        drift_cols: List of drift detection column names
        spike_cols: List of spike detection column names
        shutdown_time: Timestamp of shutdown event (optional)
        sensor_columns: List of sensor column names
        
    Returns:
        Dictionary with complete root cause analysis results
    """
    if sensor_columns is None:
        sensor_columns = ['pressure', 'temperature', 'flow', 'vibration']
    
    sensor_columns = [col for col in sensor_columns if col in df.columns]
    
    if len(df) == 0:
        return {
            'error': 'Empty dataset',
            'first_abnormal_time': None,
            'time_before_shutdown_minutes': None,
            'primary_causal_sensor': None,
            'contributing_sensors': [],
            'abnormal_patterns_summary': 'No data available',
            'differences_from_baseline': {},
            'final_explanation': 'Unable to perform analysis on empty dataset.'
        }
    
    first_abnormal_info = find_first_abnormality(
        df, sensor_columns, anomaly_col
    )
    first_abnormal_time = first_abnormal_info['first_abnormal_time']
    
    if shutdown_time is None:
        if 'shutdown_event' in df.columns and df['shutdown_event'].sum() > 0:
            shutdown_events = df[df['shutdown_event'] == 1]
            shutdown_time = shutdown_events['timestamp'].iloc[0]
        else:
            shutdown_time = df['timestamp'].iloc[-1]
    
    time_before_shutdown = None
    if first_abnormal_time is not None and shutdown_time is not None:
        time_diff = shutdown_time - first_abnormal_time
        if isinstance(time_diff, timedelta):
            time_before_shutdown = time_diff.total_seconds() / 60
        else:
            time_before_shutdown = time_diff / np.timedelta64(1, 'm')
    
    sensor_contributions = rank_sensor_contribution(df, sensor_columns)
    primary_sensor = sensor_contributions[0][0] if sensor_contributions else None
    contributing_sensors = [s[0] for s in sensor_contributions[:3]] if len(sensor_contributions) >= 3 else [s[0] for s in sensor_contributions]
    
    baseline_comparison = compare_to_baseline(
        df, 
        abnormal_start_time=first_abnormal_time,
        sensor_columns=sensor_columns
    )
    
    correlation_info = detect_correlation_breakdown(df, sensor_columns)
    
    patterns = []
    for sensor in contributing_sensors:
        comp = baseline_comparison.get(sensor, {})
        deviation = comp.get('mean_deviation_pct', 0)
        var_increase = comp.get('variability_increase_pct', 0)
        
        if abs(deviation) > 10:
            direction = 'increased' if deviation > 0 else 'decreased'
            patterns.append(f"{sensor.capitalize()} {direction} by {abs(deviation):.1f}%")
        
        if var_increase > 20:
            patterns.append(f"{sensor.capitalize()} showed {var_increase:.1f}% variability increase")
        
        if comp.get('trend_reversal', False):
            patterns.append(f"{sensor.capitalize()} exhibited trend reversal")
    
    if correlation_info['correlation_breakdown']:
        for breakdown in correlation_info.get('breakdown_pairs', [])[:2]:
            patterns.append(f"Correlation breakdown between {breakdown['sensors']}")
    
    abnormal_patterns_summary = "; ".join(patterns) if patterns else "No significant patterns detected"
    
    final_explanation = generate_explanation(
        first_abnormal_time=first_abnormal_time,
        time_before_shutdown=time_before_shutdown,
        primary_sensor=primary_sensor,
        contributing_sensors=contributing_sensors,
        baseline_comparison=baseline_comparison,
        correlation_info=correlation_info,
        first_abnormal_info=first_abnormal_info
    )
    
    return {
        'first_abnormal_time': first_abnormal_time,
        'time_before_shutdown_minutes': round(time_before_shutdown, 1) if time_before_shutdown else None,
        'primary_causal_sensor': primary_sensor,
        'contributing_sensors': contributing_sensors,
        'sensor_contributions': sensor_contributions,
        'abnormal_patterns_summary': abnormal_patterns_summary,
        'differences_from_baseline': baseline_comparison,
        'correlation_breakdown': correlation_info,
        'final_explanation': final_explanation
    }


def generate_explanation(first_abnormal_time, time_before_shutdown, 
                         primary_sensor, contributing_sensors,
                         baseline_comparison, correlation_info,
                         first_abnormal_info):
    """
    Generate a human-readable engineering explanation.
    
    Returns:
        String with engineering-style diagnostic explanation
    """
    explanation_parts = []
    
    if first_abnormal_time is not None:
        time_str = first_abnormal_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(first_abnormal_time, 'strftime') else str(first_abnormal_time)
        abnormal_type = first_abnormal_info.get('abnormality_type', 'abnormality')
        
        if time_before_shutdown is not None and time_before_shutdown > 0:
            explanation_parts.append(
                f"The equipment issue was first detected at {time_str}, "
                f"approximately {time_before_shutdown:.0f} minutes before the shutdown event."
            )
        else:
            explanation_parts.append(
                f"The equipment issue was first detected at {time_str}."
            )
        
        if first_abnormal_info.get('first_sensor'):
            first_sensor = first_abnormal_info['first_sensor']
            explanation_parts.append(
                f"The initial {abnormal_type} was observed in the {first_sensor} sensor."
            )
    
    if primary_sensor and baseline_comparison.get(primary_sensor):
        comp = baseline_comparison[primary_sensor]
        deviation = comp.get('mean_deviation_pct', 0)
        var_increase = comp.get('variability_increase_pct', 0)
        
        primary_desc = f"The primary contributing factor was {primary_sensor}"
        if abs(deviation) > 5:
            direction = 'above' if deviation > 0 else 'below'
            primary_desc += f", which deviated {abs(deviation):.1f}% {direction} baseline"
        if var_increase > 10:
            primary_desc += f" with {var_increase:.1f}% increased variability"
        primary_desc += "."
        explanation_parts.append(primary_desc)
    
    if len(contributing_sensors) > 1:
        other_sensors = [s for s in contributing_sensors if s != primary_sensor]
        if other_sensors:
            explanation_parts.append(
                f"Additional contributing factors included: {', '.join(other_sensors)}."
            )
    
    if correlation_info.get('correlation_breakdown'):
        breakdowns = correlation_info.get('breakdown_pairs', [])
        if breakdowns:
            pairs = [b['sensors'] for b in breakdowns[:2]]
            explanation_parts.append(
                f"Correlation breakdown was detected between {' and '.join(pairs)}, "
                f"indicating loss of normal operational relationships between sensors."
            )
    
    trend_reversals = []
    for sensor, comp in baseline_comparison.items():
        if comp.get('trend_reversal', False):
            trend_reversals.append(sensor)
    
    if trend_reversals:
        explanation_parts.append(
            f"Trend reversals were observed in {', '.join(trend_reversals)}, "
            f"suggesting a fundamental change in equipment behavior."
        )
    
    explanation_parts.append(
        "Recommended actions: Inspect the identified sensors for physical damage, "
        "calibration issues, or environmental factors. Review maintenance logs for "
        "recent changes. Consider implementing tighter monitoring thresholds for early warning."
    )
    
    return " ".join(explanation_parts)


def generate_engineer_report(results_dict):
    """
    Generate a comprehensive multi-paragraph engineering diagnostic report.
    
    Args:
        results_dict: Dictionary from identify_root_cause()
        
    Returns:
        String with full engineering report
    """
    if results_dict.get('error'):
        return f"**Error**: {results_dict['error']}\n\nUnable to generate diagnostic report."
    
    report_sections = []
    
    report_sections.append("# Root Cause Analysis Report\n")
    report_sections.append("## Executive Summary\n")
    
    first_time = results_dict.get('first_abnormal_time')
    time_before = results_dict.get('time_before_shutdown_minutes')
    primary = results_dict.get('primary_causal_sensor')
    
    if first_time:
        time_str = first_time.strftime('%Y-%m-%d %H:%M:%S') if hasattr(first_time, 'strftime') else str(first_time)
        summary = f"Analysis identified the first anomalous behavior at **{time_str}**"
        if time_before:
            summary += f", occurring **{time_before:.0f} minutes** before the shutdown event"
        summary += "."
        report_sections.append(summary + "\n")
    else:
        report_sections.append("No specific anomaly timestamp could be identified.\n")
    
    if primary:
        report_sections.append(f"The primary causal sensor was identified as **{primary.upper()}**.\n")
    
    report_sections.append("\n## Timeline of Instability\n")
    
    if first_time:
        time_str = first_time.strftime('%H:%M:%S') if hasattr(first_time, 'strftime') else str(first_time)
        report_sections.append(f"- **Initial Detection**: {time_str}\n")
    
    if time_before:
        report_sections.append(f"- **Warning Lead Time**: {time_before:.0f} minutes\n")
    
    report_sections.append(f"- **Pattern Summary**: {results_dict.get('abnormal_patterns_summary', 'N/A')}\n")
    
    report_sections.append("\n## Parameter Deviations\n")
    
    baseline_comp = results_dict.get('differences_from_baseline', {})
    if baseline_comp:
        report_sections.append("| Sensor | Baseline Mean | Abnormal Mean | Deviation % | Variability Increase % |\n")
        report_sections.append("|--------|---------------|---------------|-------------|------------------------|\n")
        
        for sensor, metrics in baseline_comp.items():
            report_sections.append(
                f"| {sensor.capitalize()} | "
                f"{metrics.get('baseline_mean', 'N/A')} | "
                f"{metrics.get('abnormal_mean', 'N/A')} | "
                f"{metrics.get('mean_deviation_pct', 0):.1f}% | "
                f"{metrics.get('variability_increase_pct', 0):.1f}% |\n"
            )
    else:
        report_sections.append("No baseline comparison data available.\n")
    
    report_sections.append("\n## Correlation Analysis\n")
    
    corr_info = results_dict.get('correlation_breakdown', {})
    if corr_info.get('correlation_breakdown'):
        report_sections.append("**Correlation breakdown detected** between sensor pairs:\n\n")
        for breakdown in corr_info.get('breakdown_pairs', []):
            report_sections.append(
                f"- **{breakdown['sensors']}**: Correlation changed from "
                f"{breakdown['early_correlation']:.3f} to {breakdown['late_correlation']:.3f} "
                f"(Δ = {breakdown['change']:.3f})\n"
            )
    else:
        report_sections.append("No significant correlation breakdowns detected.\n")
    
    report_sections.append("\n## Sensor Contribution Ranking\n")
    
    contributions = results_dict.get('sensor_contributions', [])
    if contributions:
        for i, (sensor, score) in enumerate(contributions, 1):
            report_sections.append(f"{i}. **{sensor.capitalize()}** - Contribution Score: {score:.3f}\n")
    
    report_sections.append("\n## Probable Root Cause\n")
    report_sections.append(results_dict.get('final_explanation', 'Unable to determine root cause.') + "\n")
    
    report_sections.append("\n## Recommended Actions\n")
    report_sections.append("1. **Immediate**: Conduct physical inspection of primary causal sensor\n")
    report_sections.append("2. **Short-term**: Review calibration records and recent maintenance activities\n")
    report_sections.append("3. **Medium-term**: Implement enhanced monitoring with tighter anomaly thresholds\n")
    report_sections.append("4. **Long-term**: Consider predictive maintenance scheduling based on detected patterns\n")
    
    return "".join(report_sections)


if __name__ == "__main__":
    from simulate import generate_simulated_data
    from features import engineer_features
    from model import run_anomaly_detection
    
    print("Generating test data...")
    df = generate_simulated_data(n_points=500)
    df_features = engineer_features(df)
    df_processed, thresholds, model, scaler = run_anomaly_detection(df_features)
    
    print("\nRunning root cause analysis...")
    results = identify_root_cause(df_processed)
    
    print("\n" + "="*60)
    print("ROOT CAUSE ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nFirst Abnormal Time: {results['first_abnormal_time']}")
    print(f"Time Before Shutdown: {results['time_before_shutdown_minutes']} minutes")
    print(f"Primary Causal Sensor: {results['primary_causal_sensor']}")
    print(f"Contributing Sensors: {results['contributing_sensors']}")
    print(f"\nPattern Summary: {results['abnormal_patterns_summary']}")
    
    print("\n" + "-"*60)
    print("ENGINEERING EXPLANATION")
    print("-"*60)
    print(results['final_explanation'])
    
    print("\n" + "="*60)
    print("FULL ENGINEERING REPORT")
    print("="*60)
    report = generate_engineer_report(results)
    print(report)
