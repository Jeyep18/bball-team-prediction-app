from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("clean_nba_data.csv")
MODEL_FILE = Path("nba_regression_model.pkl")

FEATURE_COLUMNS = [
    "FG%",
    "3P%",
    "FT%",
    "FTA",
    "TRB",
    "AST",
    "TOV",
    "STL",
    "BLK",
    "Point_Differential",
    "Opponent_Win_Rate",
    "Home",
]
TARGET_COLUMN = "Win"
RANDOM_STATE = 42


def choose_stratify_target(y: pd.Series) -> pd.Series | None:
    """Use stratified splitting only when both classes have enough rows."""
    class_counts = y.value_counts()
    if len(class_counts) == 2 and class_counts.min() >= 2:
        return y
    return None


def train_logistic_model(data: pd.DataFrame, selected_features: list[str]) -> dict:
    if len(selected_features) < 2:
        raise ValueError("Select at least 2 variables before training the model.")

    missing_columns = [column for column in selected_features + [TARGET_COLUMN] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    clean_data = data.dropna(subset=selected_features + [TARGET_COLUMN]).copy()
    X = clean_data[selected_features]
    y = clean_data[TARGET_COLUMN].astype(int)

    if y.nunique() < 2:
        raise ValueError("The target column must contain both wins and losses to train Logistic Regression.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=choose_stratify_target(y),
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logistic_regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "features": selected_features,
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def main() -> None:
    """Train and save the Logistic Regression model."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Run scrape_nba_data.py before training the model."
        )

    data = pd.read_csv(DATA_FILE)

    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing required columns: {missing_columns}")

    model_bundle = train_logistic_model(data, FEATURE_COLUMNS)
    joblib.dump(model_bundle, MODEL_FILE)

    print(f"Model saved to {MODEL_FILE}")
    print(f"Holdout accuracy: {model_bundle['accuracy']:.2%}")
    print("Confusion matrix:")
    print(model_bundle["confusion_matrix"])


if __name__ == "__main__":
    main()
