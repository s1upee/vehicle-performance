import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load realistic vehicle data
df = pd.read_csv("data/realistic_vehicle_data.csv")

# Select relevant features
features = ['acceleration', 'brake_force', 'steering_angle']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hyperparameter tuning for Isolation Forest
isolation_forest_grid = {
    'contamination': [0.01, 0.03, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

best_isolation_forest = None
best_score = -np.inf
for contamination in isolation_forest_grid['contamination']:
    for n_estimators in isolation_forest_grid['n_estimators']:
        model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42)
        model.fit(X_scaled)
        score = model.decision_function(X_scaled).mean()
        if score > best_score:
            best_score = score
            best_isolation_forest = model

# Hyperparameter tuning for One-Class SVM
svm_grid = {'nu': [0.01, 0.03, 0.05], 'kernel': ["rbf", "poly"]}

best_svm = None
best_svm_score = -np.inf
for nu in svm_grid['nu']:
    for kernel in svm_grid['kernel']:
        model = OneClassSVM(nu=nu, kernel=kernel)
        model.fit(X_scaled)
        score = model.decision_function(X_scaled).mean()
        if score > best_svm_score:
            best_svm_score = score
            best_svm = model

# Hyperparameter tuning for Local Outlier Factor (LOF)
lof_grid = {'n_neighbors': [10, 20, 30], 'contamination': [0.01, 0.03, 0.05]}

best_lof = None
best_lof_score = -np.inf
for n_neighbors in lof_grid['n_neighbors']:
    for contamination in lof_grid['contamination']:
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        scores = model.fit_predict(X_scaled)
        avg_score = np.mean(scores)
        if avg_score > best_lof_score:
            best_lof_score = avg_score
            best_lof = model

# Train and apply the best models
df['IsolationForest_anomaly'] = best_isolation_forest.predict(X_scaled)
df['OneClassSVM_anomaly'] = best_svm.predict(X_scaled)
df['LocalOutlierFactor_anomaly'] = best_lof.fit_predict(X_scaled)

df['IsolationForest_anomaly'] = df['IsolationForest_anomaly'].apply(lambda x: True if x == -1 else False)
df['OneClassSVM_anomaly'] = df['OneClassSVM_anomaly'].apply(lambda x: True if x == -1 else False)
df['LocalOutlierFactor_anomaly'] = df['LocalOutlierFactor_anomaly'].apply(lambda x: True if x == -1 else False)

# Majority voting: Combine results from all models
df['final_anomaly'] = df[['IsolationForest_anomaly', 'OneClassSVM_anomaly', 'LocalOutlierFactor_anomaly']].sum(axis=1) >= 2

# Save refined anomaly detection results
df.to_csv("data/optimized_ml_detected_anomalies.csv", index=False)
print("Optimized ML anomaly detection complete. Results saved to data/optimized_ml_detected_anomalies.csv")
