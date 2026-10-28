
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "customer_data_500_records.csv"

def train_model():
    df = pd.read_csv(DATA)

    numeric = [
        "Age", "Annual_Income",
        "Spending_Score", "Experience"
    ]
    categorical = [
        "Gender", "City", "Education",
        "Marital_Status", "Occupation"
    ]

    X = df[numeric + categorical]
    y = df["Purchased"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=42, stratify=y
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical)
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    print("Total records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(matrix)

    joblib.dump(
        model, BASE / "customer_purchase_model.pkl"
    )

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "confusion_matrix": matrix.tolist()
    }

    with open(BASE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Model and metrics saved successfully.")

if __name__ == "__main__":
    train_model()
