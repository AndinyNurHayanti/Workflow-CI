"""
Model Training with MLflow & XGBoost - Diabetes Dataset
Author: Andiny Nur Hayanti
"""

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
sys.stdout.reconfigure(
    encoding="utf-8"
)

import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")


def load_preprocessed_data():

    train_df = pd.read_csv("diabetes_preprocessing/diabetes_train.csv")
    test_df = pd.read_csv("diabetes_preprocessing/diabetes_test.csv")

    X_train = train_df.drop(
        "Outcome",
        axis=1
    )

    y_train = train_df["Outcome"]

    X_test = test_df.drop(
        "Outcome",
        axis=1
    )

    y_test = test_df["Outcome"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def train_xgboost():

    X_train, X_test, y_train, y_test = (
        load_preprocessed_data()
    )

    # mlflow.set_tracking_uri(
    #     "http://127.0.0.1:5000"
    # )

    mlflow.set_experiment(
        "Diabetes_XGBoost_Experiment"
    )

    mlflow.xgboost.autolog()

    with mlflow.start_run(
        run_name="XGBoost_Diabetes_V1"
    ):

        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            eval_metric="logloss"
        )

        print("Training model...")

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        acc = accuracy_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc = roc_auc_score(
            y_test,
            y_prob
        )

        mlflow.log_metric(
            "accuracy",
            acc
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc
        )

        print(f"Accuracy : {acc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"ROC-AUC : {roc:.4f}")

        print("\nSelesai dan tersimpan di MLflow")


if __name__ == "__main__":
    train_xgboost()