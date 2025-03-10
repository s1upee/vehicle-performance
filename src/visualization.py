import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data with anomalies
df = pd.read_csv("data/optimized_ml_detected_anomalies.csv")

# Ensure the necessary columns exist
if "final_anomaly" not in df.columns:
    raise ValueError("Column 'final_anomaly' not found in dataset. Ensure the correct file is being used.")

# Define anomaly points
anomalies = df[df["final_anomaly"] == 1]

# Plot acceleration over time with anomalies
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df["acceleration"], label="Acceleration", color='blue')
plt.scatter(anomalies.index, anomalies["acceleration"], color='red', label="Anomaly", zorder=3)
plt.title("Vehicle Acceleration Over Time with Anomalies")
plt.xlabel("Index")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.show()

# Plot braking force over time with anomalies
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df["brake_force"], label="Brake Force", color='orange')
plt.scatter(anomalies.index, anomalies["brake_force"], color='red', label="Anomaly", zorder=3)
plt.title("Braking Force Over Time with Anomalies")
plt.xlabel("Index")
plt.ylabel("Brake Force")
plt.legend()
plt.show()

# Plot steering angle over time with anomalies
plt.figure(figsize=(10, 5))
sns.lineplot(x=df.index, y=df["steering_angle"], label="Steering Angle", color='green')
plt.scatter(anomalies.index, anomalies["steering_angle"], color='red', label="Anomaly", zorder=3)
plt.title("Steering Angle Over Time with Anomalies")
plt.xlabel("Index")
plt.ylabel("Steering Angle (Degrees)")
plt.legend()
plt.show()

print("Visualization complete.")
