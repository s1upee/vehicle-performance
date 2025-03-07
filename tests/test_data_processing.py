import unittest
import pandas as pd
import numpy as np
from src.data_processing import detect_anomalies  # Ensure function is defined in data_processing.py

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        # Create sample test data
        self.df = pd.DataFrame({
            'time': np.arange(0, 10),
            'acceleration': [0, 1, 2, 3, 6, 7, 2, 1, 0, -1],  # 6 and 7 exceed threshold
            'brake_force': [0, 0, 5, 10, 20, 15, 0, 0, 5, 10],  # 20 and 15 exceed threshold
            'steering_angle': [0, -5, 10, 15, 20, 35, -40, 5, 0, -30],  # 35 and -40 exceed threshold
        })

    def test_anomaly_detection(self):
        """Test if anomalies are detected correctly."""
        # Apply anomaly detection
        df_processed = detect_anomalies(self.df)

        # Debugging: Print the processed DataFrame to inspect anomaly detection
        print("\nProcessed Data for Debugging:")
        print(df_processed[['time', 'brake_force', 'hard_braking']])

        # Check if anomalies are correctly detected
        self.assertTrue(df_processed.loc[4, 'sudden_acceleration'])  # Acceleration anomaly
        self.assertTrue(df_processed.loc[5, 'hard_braking'])  # Braking anomaly
        self.assertTrue(df_processed.loc[6, 'sharp_turn'])  # Steering anomaly

    def test_no_false_positives(self):
        """Test that no false positives occur in anomaly detection."""
        df_processed = detect_anomalies(self.df)
        false_positives = df_processed[(df_processed['sudden_acceleration'] == True) & (df_processed['acceleration'] <= 5)]
        self.assertTrue(false_positives.empty, "False positive acceleration anomalies detected")

if __name__ == "__main__":
    unittest.main()
