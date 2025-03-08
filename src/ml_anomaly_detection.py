import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Load realistic vehicle data
df = pd.read_csv("data/realistic_vehicle_data.csv")

# Select relevant features
features = ['acceleration', 'brake_force', 'steering_angle']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Try multiple anomaly detection models
models = {
    'Isolation Forest': IsolationForest(contamination=0.03, n_estimators=100, random_state=42),
    'One-Class SVM': OneClassSVM(nu=0.03, kernel="rbf"),
    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=0.03)
}

# Store results
df_results = df.copy()

for name, model in models.items():
    print(f"Training {name}...")
    if name == "Local Outlier Factor":
        df_results[f"{name}_anomaly"] = model.fit_predict(X_scaled)
    else:
        model.fit(X_scaled)
        df_results[f"{name}_anomaly"] = model.predict(X_scaled)
    
    df_results[f"{name}_anomaly"] = df_results[f"{name}_anomaly"].apply(lambda x: True if x == -1 else False)

# Majority voting: Combine results from all models
df_results['final_anomaly'] = df_results[['Isolation Forest_anomaly', 'One-Class SVM_anomaly', 'Local Outlier Factor_anomaly']].sum(axis=1) >= 2

# Save refined anomaly detection results
df_results.to_csv("data/refined_ml_detected_anomalies.csv", index=False)
print("Refined ML anomaly detection complete. Results saved to data/refined_ml_detected_anomalies.csv")
