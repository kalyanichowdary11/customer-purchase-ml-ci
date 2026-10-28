
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            (BASE / "data" / "customer_data_500_records.csv").exists()
        )

    def test_model_created(self):
        self.assertTrue(
            (BASE / "customer_purchase_model.pkl").exists()
        )

    def test_metrics_created(self):
        self.assertTrue((BASE / "metrics.json").exists())

    def test_accuracy_is_valid(self):
        with open(BASE / "metrics.json") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load(
            BASE / "customer_purchase_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 30,
            "Annual_Income": 50000,
            "Spending_Score": 60,
            "Experience": 5,
            "Gender": "Male",
            "City": "Guntur",
            "Education": "Graduate",
            "Marital_Status": "Single",
            "Occupation": "Engineer"
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 999)


if __name__ == "__main__":
    unittest.main()
