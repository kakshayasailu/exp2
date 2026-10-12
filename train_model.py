
import csv
import os
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def create_demo_dataset():
    random.seed(42)

    os.makedirs("data", exist_ok=True)

    with open(
        "data/customer_churn_raw.csv", "w", newline=""
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "gender", "SeniorCitizen", "Partner",
            "Dependents", "tenure", "PhoneService",
            "InternetService", "Contract",
            "MonthlyCharges", "TotalCharges", "Churn"
        ])

        for _ in range(400):
            gender = random.choice(["Female", "Male"])
            senior = random.choice([0, 1])
            partner = random.choice(["Yes", "No"])
            dependents = random.choice(["Yes", "No"])
            tenure = random.randint(0, 72)
            phone = random.choice(["Yes", "No"])
            internet = random.choice(["DSL", "Fiber optic", "No"])
            contract = random.choice([
                "Month-to-month", "One year", "Two year"
            ])
            monthly = round(random.uniform(20, 120), 2)
            total = round(monthly * tenure, 2)

            # Illustrative demo labels, not real customer outcomes
            churn_probability = 0.15

            if contract == "Month-to-month":
                churn_probability += 0.35
            if tenure < 12:
                churn_probability += 0.20
            if internet == "Fiber optic":
                churn_probability += 0.10

            churn = (
                "Yes"
                if random.random() < min(churn_probability, 0.9)
                else "No"
            )

            writer.writerow([
                gender, senior, partner, dependents,
                tenure, phone, internet, contract,
                monthly, total, churn
            ])


def main():
    dataset = "data/customer_churn_raw.csv"

    if not os.path.exists(dataset):
        create_demo_dataset()

    data = pd.read_csv(dataset)

    required = [
        "gender", "SeniorCitizen", "Partner",
        "Dependents", "tenure", "PhoneService",
        "InternetService", "Contract",
        "MonthlyCharges", "TotalCharges", "Churn"
    ]

    if not all(column in data.columns for column in required):
        raise ValueError("Dataset is missing required columns.")

    if data.empty or data["Churn"].nunique() < 2:
        raise ValueError(
            "Dataset needs rows from at least two churn classes."
        )

    # Remove rows with missing values
    data = data[required].copy()
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    data = data.dropna()

    if data.empty or data["Churn"].nunique() < 2:
        raise ValueError("Not enough valid data to train the model.")

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    # Encode categorical features into numeric columns
    X = pd.get_dummies(X, drop_first=False)

    if y.value_counts().min() < 2:
        raise ValueError(
            "Each churn class needs at least two records."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    os.makedirs("artifacts", exist_ok=True)

    # Save model and feature columns for consistent prediction
    joblib.dump(
        {
            "model": model,
            "feature_columns": X.columns.tolist()
        },
        "artifacts/customer_churn_model.joblib"
    )

    print("Customer churn model training completed.")
    print("Test accuracy:", round(accuracy, 4))
    print(
        "Model saved to "
        "artifacts/customer_churn_model.joblib"
    )


if __name__ == "__main__":
    main()
