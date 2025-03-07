import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load data
df = pd.read_csv("data/processed_vehicle_data.csv")

# Select relevant features for anomaly detection
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
df.to_csv("data/ml_detected_anomalies.csv", index=False)
print(f"Optimized ML anomaly detection complete with parameters: {best_params}")
print("Results saved to data/ml_detected_anomalies.csv")
