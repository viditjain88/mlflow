from crewai.tools import BaseTool
import torch
import joblib
import pandas as pd
import numpy as np
import os
from hmdss.analytics.model import PatientVolumePredictor

class ANNPredictionTool(BaseTool):
    name: str = "ANN Prediction Tool"
    description: str = "Predicts patient volume for a given date. Input should be a date string in YYYY-MM-DD format."

    def _run(self, date_str: str) -> str:
        model_path = "hmdss/analytics/patient_volume_model.pth"
        scaler_path = "hmdss/analytics/scaler.joblib"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return "Error: ANN model not trained yet."

        try:
            # Parse date
            date = pd.to_datetime(date_str)
            day_of_week = date.dayofweek
            month = date.month
            day_of_month = date.day
            is_weekend = int(day_of_week >= 5)

            features = np.array([[day_of_week, month, day_of_month, is_weekend]])

            # Load scaler
            scaler = joblib.load(scaler_path)
            features_scaled = scaler.transform(features)

            features_tensor = torch.FloatTensor(features_scaled)

            # Load model (input_dim=4)
            model = PatientVolumePredictor(input_dim=4)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            with torch.no_grad():
                prediction = model(features_tensor).item()

            return f"Predicted Patient Volume for {date_str}: {int(prediction)}"

        except Exception as e:
            return f"Error making prediction: {str(e)}"
