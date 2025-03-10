# 🚀 Vehicle Anomaly Detection & Visualization

## 📌 Overview
This project implements a **Machine Learning-powered anomaly detection system** for vehicle performance data. It integrates advanced **data processing, anomaly detection, visualization, and a real-time dashboard** to monitor anomalies in acceleration, braking force, and steering angle.

## 🏗 Project Structure
```
📦 Project Root
├── 📂 assets                   # Images & assets for documentation
├── 📂 data                     # Raw and processed vehicle data
│   ├── realistic_vehicle_data.csv
│   ├── processed_vehicle_data.csv
│   ├── ml_detected_anomalies.csv
│   ├── optimized_ml_detected_anomalies.csv
│   └── model_performance_metrics.csv
├── 📂 notebooks                # Jupyter Notebook for visualization
│   └── visualization.ipynb
├── 📂 plots                    # Saved visualizations of anomalies
├── 📂 src                      # Core Python scripts
│   ├── data_processing.py      # Cleans & preprocesses vehicle data
│   ├── data_simulation.py      # Generates synthetic vehicle data
│   ├── ml_anomaly_detection.py # ML anomaly detection pipeline
│   ├── visualization.py        # Generates anomaly plots
│   ├── dashboard.py            # Interactive dashboard with Dash & Plotly
├── 📂 tests                    # Unit tests for data processing
│   ├── test_data_processing.py
├── README.md                   # Project documentation
```

## 🔥 Key Features
✅ **Data Processing** – Cleans, normalizes, and prepares vehicle data for analysis.
✅ **ML Anomaly Detection** – Detects anomalies using **Isolation Forest, One-Class SVM, and Local Outlier Factor**.
✅ **Visualization** – Generates **histograms, time-series plots, and scatter plots** for anomaly insights.
✅ **Interactive Dashboard** – A real-time **Dash-based UI** that updates every 5 seconds.

## 📊 Machine Learning Anomaly Detection
The anomaly detection pipeline **optimizes ML parameters** and applies multiple models:
- **Isolation Forest** 🏕 (Best contamination level: `0.05`)
- **One-Class SVM** 🎭 (Best kernel: `rbf`, nu: `0.01`)
- **Local Outlier Factor** 🏠 (k=20, contamination= `0.03`)

Final anomalies are determined through **majority voting**, ensuring more robust detection.

## 🖥 Interactive Dashboard
Run the interactive **real-time dashboard** with:
```sh
python src/dashboard.py
```
This dashboard:
- Displays **acceleration, braking force, and steering anomalies**
- Updates every **5 seconds** to reflect new anomalies detected in real-time.

## 📊 Data Visualization
Jupyter Notebook `notebooks/visualization.ipynb` contains:
- **Histograms of acceleration, braking force, and steering angle**
- **Time-series plots highlighting anomalies**
- **ML performance evaluation (Precision, Recall, F1-score, ROC AUC)**

## 🛠 Setup Instructions
1️⃣ **Clone the repository**
```sh
git clone https://github.com/your-repo.git
cd your-repo
```
2️⃣ **Install dependencies**
```sh
pip install -r requirements.txt
```
3️⃣ **Run anomaly detection**
```sh
python src/ml_anomaly_detection.py
```
4️⃣ **Launch visualization notebook**
```sh
jupyter notebook notebooks/visualization.ipynb
```

## 📌 Results
✅ ML models detected anomalies in vehicle performance with ROC AUC scores around `0.48 - 0.50`.
✅ Anomalies are **visualized in plots** & **real-time monitoring is possible via the dashboard**.
✅ **Refined & optimized ML detection pipeline** achieves better anomaly recognition.

## 💡 Future Improvements
- 🔥 **Enhance ML models** with deep learning (LSTMs for time-series)
- 📈 **Integrate GPS & sensor fusion** for spatial anomaly detection
- 🚀 **Deploy on a cloud server** for real-time vehicle monitoring

## 📜 License
This project is **open-source** under the MIT License.

---
🚀 **This repository contains one of the most powerful and refined vehicle anomaly detection tools – combining ML, data science, and real-time visualization!**

---

## 📩 Contact
If you have any questions, feel free to reach out! 😊  
🔗 **GitHub**: [s1upee](https://github.com/s1upee)  
🔗 **Email**: lisakrasiuk@gmail.com  
```

