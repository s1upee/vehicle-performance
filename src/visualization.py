import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv("data/processed_vehicle_data.csv")

# Plot acceleration over time
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["time"], y=df["acceleration"], label="Acceleration", color='blue')
plt.axhline(y=5, color='red', linestyle='--', label="Acceleration Threshold")
plt.title("Vehicle Acceleration Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.show()

# Plot braking force over time
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["time"], y=df["brake_force"], label="Brake Force", color='orange')
plt.axhline(y=15, color='red', linestyle='--', label="Braking Threshold")
plt.title("Braking Force Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Brake Force")
plt.legend()
plt.show()

# Plot steering angle over time
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["time"], y=df["steering_angle"], label="Steering Angle", color='green')
plt.axhline(y=30, color='red', linestyle='--', label="Steering Threshold")
plt.axhline(y=-30, color='red', linestyle='--')
plt.title("Steering Angle Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Steering Angle (Degrees)")
plt.legend()
plt.show()

print("Visualization complete.")
