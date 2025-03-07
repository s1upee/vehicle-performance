import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Generate more realistic vehicle data with patterns
time = np.arange(0, 600, 1)  # Simulate 600 seconds of data
np.random.seed(42)

# Acceleration: Normal driving pattern with occasional sudden accelerations
acceleration = np.random.normal(0, 2, len(time))
acceleration += np.sin(time / 30) * 3  # Simulating gradual speed changes
sudden_acceleration_indices = np.random.choice(len(time), size=10, replace=False)
acceleration[sudden_acceleration_indices] += np.random.uniform(5, 10, 10)  # Sudden spikes

# Braking: Mostly light braking with occasional emergency stops
brake_force = np.random.choice([0, 5, 10, 20], size=len(time), p=[0.85, 0.10, 0.03, 0.02])

# Steering: Simulating real turns with occasional sharp turns
steering_angle = np.random.normal(0, 15, len(time))
steering_angle += np.cos(time / 50) * 5  # Mimicking lane adjustments
sharp_turn_indices = np.random.choice(len(time), size=5, replace=False)
steering_angle[sharp_turn_indices] += np.random.uniform(30, 45, 5)

# Create a DataFrame
df = pd.DataFrame({
    'time': time,
    'acceleration': acceleration,
    'brake_force': brake_force,
    'steering_angle': steering_angle
})

# Save realistic simulated data
df.to_csv("data/realistic_vehicle_data.csv", index=False)

# Load data for ML anomaly detection
df = pd.read_csv("data/realistic_vehicle_data.csv")

# Select relevant features
features = ['acceleration', 'brake_force', 'steering_angle']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter tuning for Isolation Forest
param_grid = {'contamination': [0.01, 0.03, 0.05, 0.1], 'n_estimators': [50, 100, 200]}
best_params = {'contamination': 0.05, 'n_estimators': 100}  # Default values

best_score = -np.inf
for contamination in param_grid['contamination']:
    for n_estimators in param_grid['n_estimators']:
        model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        model.fit(X_scaled)
        scores = model.decision_function(X_scaled).mean()
        if scores > best_score:
            best_score = scores
            best_params = {'contamination': contamination, 'n_estimators': n_estimators}

# Train optimized Isolation Forest model
model = IsolationForest(**best_params, random_state=42)
model.fit(X_scaled)

# Predict anomalies (-1 = anomaly, 1 = normal)
df['ml_anomaly'] = model.predict(X_scaled)
df['ml_anomaly'] = df['ml_anomaly'].apply(lambda x: True if x == -1 else False)

# Save results
df.to_csv("data/ml_detected_realistic_anomalies.csv", index=False)
print(f"Optimized ML anomaly detection on realistic data complete with parameters: {best_params}")
print("Results saved to data/ml_detected_realistic_anomalies.csv")
