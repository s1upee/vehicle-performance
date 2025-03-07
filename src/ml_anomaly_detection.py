import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/processed_vehicle_data.csv")

# Select relevant features for anomaly detection
features = ['acceleration', 'brake_force', 'steering_angle']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)  # 5% of data expected to be anomalies
model.fit(X_scaled)

# Predict anomalies (-1 = anomaly, 1 = normal)
df['ml_anomaly'] = model.predict(X_scaled)
df['ml_anomaly'] = df['ml_anomaly'].apply(lambda x: True if x == -1 else False)

# Save results
df.to_csv("data/ml_detected_anomalies.csv", index=False)
print("Machine Learning anomaly detection complete. Results saved to data/ml_detected_anomalies.csv")
