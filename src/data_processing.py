import pandas as pd
import numpy as np

# Load the simulated data
df = pd.read_csv("data/simulated_vehicle_data.csv")

# Define anomaly detection thresholds
ACCELERATION_THRESHOLD = 5  # m/s² (sudden acceleration)
BRAKE_FORCE_THRESHOLD = 15  # High brake force
STEERING_ANGLE_THRESHOLD = 30  # Degrees (sharp turns)

# Detect anomalies
df['hard_braking'] = df['brake_force'] > BRAKE_FORCE_THRESHOLD
df['sudden_acceleration'] = df['acceleration'] > ACCELERATION_THRESHOLD
df['sharp_turn'] = abs(df['steering_angle']) > STEERING_ANGLE_THRESHOLD

# Count detected anomalies
anomalies_count = df[['hard_braking', 'sudden_acceleration', 'sharp_turn']].sum()
print("Anomalies Detected:")
print(anomalies_count)

# Save processed data with anomalies flagged
df.to_csv("data/processed_vehicle_data.csv", index=False)
print("Processed data saved to data/processed_vehicle_data.csv")
