import numpy as np
import pandas as pd

# Define simulation parameters
time = np.arange(0, 600, 1)  # 600 seconds of data (10 minutes)
np.random.seed(42)  # For reproducibility

# Simulated acceleration values (normal driving + sudden acceleration spikes)
acceleration = np.random.normal(0, 2, len(time))  # Mean 0, std 2
sudden_acceleration_indices = np.random.choice(len(time), size=10, replace=False)
acceleration[sudden_acceleration_indices] += np.random.uniform(5, 10, 10)  # Simulate sudden bursts

# Simulated braking force (mostly 0, occasional hard braking)
brake_force = np.random.choice([0, 5, 10, 20], size=len(time), p=[0.85, 0.10, 0.03, 0.02])

# Simulated steering angle (smooth and sharp turns)
steering_angle = np.random.normal(0, 15, len(time))  # Normal driving with small variations
sharp_turn_indices = np.random.choice(len(time), size=5, replace=False)
steering_angle[sharp_turn_indices] += np.random.uniform(30, 45, 5)  # Simulate sharp turns

# Simulated ADAS events (rare anomalies)
adas_events = np.random.choice([0, 1], size=len(time), p=[0.98, 0.02])  # 2% chance of ADAS event occurring

# Create a DataFrame
df = pd.DataFrame({
    'time': time,
    'acceleration': acceleration,
    'brake_force': brake_force,
    'steering_angle': steering_angle,
    'adas_event': adas_events
})

# Save simulated data
df.to_csv("data/simulated_vehicle_data.csv", index=False)

print("Simulated vehicle data saved to data/simulated_vehicle_data.csv")
