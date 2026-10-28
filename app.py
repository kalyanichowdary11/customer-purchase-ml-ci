
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "customer_purchase_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Experience",
    "Gender",
    "City",
    "Education",
    "Marital_Status",
    "Occupation"
]


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Customer Purchase Prediction API is running"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send input as JSON"}), 400

    missing = [feature for feature in FEATURES if feature not in data]

    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        input_data = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        prediction = int(model.predict(input_data)[0])

        return jsonify({
            "prediction_code": prediction,
            "prediction_label": (
                "Purchased" if prediction == 1 else "Not Purchased"
            )
        }), 200

    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
