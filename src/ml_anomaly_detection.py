import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

# Load realistic vehicle data
df = pd.read_csv("data/realistic_vehicle_data.csv")

# Select relevant features
features = ['acceleration', 'brake_force', 'steering_angle']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define parameter grids for fine-tuning
param_grid_iforest = {
    'contamination': [0.01, 0.02, 0.03, 0.05],
    'n_estimators': [50, 100, 200]
}
param_grid_ocsvm = {
    'nu': [0.01, 0.03],
    'kernel': ['rbf']
}

# Define a custom scoring function for anomaly detection
def custom_scorer(estimator, X):
    return -np.mean(estimator.fit(X).decision_function(X))  # Higher is better (minimizing negative score)

# Fine-tune Isolation Forest
iforest = GridSearchCV(IsolationForest(random_state=42), param_grid_iforest, cv=5, scoring=custom_scorer)
iforest.fit(X_scaled)
best_iforest = iforest.best_estimator_

# Fine-tune One-Class SVM
ocsvm = GridSearchCV(OneClassSVM(), param_grid_ocsvm, cv=5, scoring=custom_scorer)
ocsvm.fit(X_scaled)
best_ocsvm = ocsvm.best_estimator_

# Define multiple anomaly detection models with optimized parameters
models = {
    'Isolation Forest': best_iforest,
    'One-Class SVM': best_ocsvm,
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

# Convert final anomaly decision to integer
df_results['final_anomaly'] = df_results['final_anomaly'].astype(int)

# Generate synthetic ground truth for evaluation (for demonstration purposes)
df_results['ground_truth'] = np.random.choice([0, 1], size=len(df_results), p=[0.95, 0.05])

# Compute evaluation metrics
models_to_evaluate = ['Isolation Forest_anomaly', 'One-Class SVM_anomaly', 'Local Outlier Factor_anomaly', 'final_anomaly']
evaluation_results = {}

for model in models_to_evaluate:
    precision = precision_score(df_results['ground_truth'], df_results[model])
    recall = recall_score(df_results['ground_truth'], df_results[model])
    f1 = f1_score(df_results['ground_truth'], df_results[model])
    auc = roc_auc_score(df_results['ground_truth'], df_results[model])
    evaluation_results[model] = {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    }

# Save refined anomaly detection results
df_results.to_csv("data/optimized_ml_detected_anomalies.csv", index=False)

# Save evaluation results
evaluation_df = pd.DataFrame(evaluation_results).T
evaluation_df.to_csv("data/model_performance_metrics.csv", index=True)

print("Data Shape:", X.shape)
print("First 5 rows:\n", X.head())
print("Missing values:\n", X.isnull().sum())

print("Best Parameters for Isolation Forest:", iforest.best_params_)
print("Best Parameters for One-Class SVM:", ocsvm.best_params_)

print("Isolation Forest anomalies detected:", df_results["Isolation Forest_anomaly"].sum())
print("One-Class SVM anomalies detected:", df_results["One-Class SVM_anomaly"].sum())
print("Local Outlier Factor anomalies detected:", df_results["Local Outlier Factor_anomaly"].sum())
print("Final Anomalies (majority voting):", df_results["final_anomaly"].sum())

print(evaluation_df)

# Print evaluation results
print("Refined ML anomaly detection complete. Results saved to data/optimized_ml_detected_anomalies.csv")
print("Model evaluation complete. Results saved to data/model_performance_metrics.csv")
print(evaluation_df)
