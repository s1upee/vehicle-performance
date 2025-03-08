import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def detect_anomalies(df):
    """Detects anomalies in acceleration, braking, and steering data."""
    ACCELERATION_THRESHOLD = 5  # m/s² (sudden acceleration)
    BRAKE_FORCE_THRESHOLD = 15  # High brake force
    STEERING_ANGLE_THRESHOLD = 30  # Degrees (sharp turns)

    df['hard_braking'] = df['brake_force'] >= BRAKE_FORCE_THRESHOLD
    df['sudden_acceleration'] = df['acceleration'] > ACCELERATION_THRESHOLD
    df['sharp_turn'] = abs(df['steering_angle']) > STEERING_ANGLE_THRESHOLD

    return df

def add_temporal_features(df):
    """Adds simulated timestamp, hour_of_day, and is_peak_hour."""
    # Create a simulated timestamp column (assuming data is recorded every second)
    start_time = datetime.now()
    df['timestamp'] = [start_time + timedelta(seconds=i) for i in range(len(df))]
    
    # Extract hour of the day
    df['hour_of_day'] = df['timestamp'].dt.hour

    # Define peak hours (rush hours: 7-9 AM & 5-7 PM)
    df['is_peak_hour'] = df['hour_of_day'].between(7, 9) | df['hour_of_day'].between(17, 19)

    return df

def add_kinematic_features(df):
    """Adds derived kinematic features: speed, jerk, and brake intensity."""
    # Simulate time index (assuming equal intervals)
    df['time'] = df.index  

    # Estimate speed using numerical integration (cumulative sum of acceleration)
    df['speed'] = df['acceleration'].cumsum()

    # Compute jerk (rate of change of acceleration)
    df['jerk'] = df['acceleration'].diff()

    # Compute braking intensity (relative to speed)
    df['brake_intensity'] = df['brake_force'] / (df['speed'].abs() + 1e-5)

    return df

def add_aggregated_statistics(df, window_size=3):
    """Adds rolling averages to smooth sudden spikes."""
    df['rolling_acceleration'] = df['acceleration'].rolling(window=window_size).mean()
    df['rolling_brake_force'] = df['brake_force'].rolling(window=window_size).mean()
    return df

def add_steering_features(df, threshold=15):
    """Adds steering rate and lane change detection."""
    df['steering_rate'] = df['steering_angle'].diff() / df['time'].diff()
    df['is_lane_change'] = (df['steering_angle'].shift(1) * df['steering_angle'] < 0) & (abs(df['steering_angle']) > threshold)
    return df

# Load the simulated data
df = pd.read_csv("data/simulated_vehicle_data.csv")

# Add temporal features
df = add_temporal_features(df)

# Add kinematic features
df = add_kinematic_features(df)

# Adds rolling averages for smoothing
df = add_aggregated_statistics(df)

# Adds steering rate & lane change detection
df = add_steering_features(df, threshold=15)

# Apply anomaly detection
df = detect_anomalies(df)

# Count detected anomalies
anomalies_count = df[['hard_braking', 'sudden_acceleration', 'sharp_turn']].sum()
print("Anomalies Detected:")
print(anomalies_count)

# Save processed data with anomalies, temporal, and kinematic features
df.to_csv("data/processed_vehicle_data.csv", index=False)
print("Processed data saved to data/processed_vehicle_data.csv")
