
import unittest
import app


class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "healthy")

    def test_prediction_endpoint(self):
        customer = {
            "Age": 30,
            "Annual_Income": 50000,
            "Spending_Score": 60,
            "Experience": 5,
            "Gender": "Female",
            "City": "New York",
            "Education": "Bachelor's",
            "Marital_Status": "Single",
            "Occupation": "Professional"
        }

        response = self.client.post("/predict", json=customer)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["prediction_code"], [0, 1])
        self.assertIn(
            response.json["prediction_label"],
            ["Purchased", "Not Purchased"]
        )

    def test_missing_fields(self):
        customer = {"Age": 30}

        response = self.client.post("/predict", json=customer)

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.json)


if __name__ == "__main__":
    unittest.main()
