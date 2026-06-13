from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor, prepare_modeling_frame, stratified_split


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=60,
            max_depth=8,
            min_samples_leaf=10,
            max_samples=0.5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_predictions(y_true, y_pred, y_score) -> dict:
    return {
        "auc_pr": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_evaluate(df: pd.DataFrame, target: str, model_dir: Path, dataset_name: str) -> pd.DataFrame:
    df = prepare_modeling_frame(df, target)
    split = stratified_split(df, target)
    preprocessor = build_preprocessor(split.X_train)
    rows = []
    best_model = None
    best_score = -np.inf
    n_splits = 3 if dataset_name == "fraud" else 2

    for model_name, model in get_models().items():
        print(f"Training {dataset_name}: {model_name}")
        pipeline = ImbPipeline([
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_validate(
            pipeline,
            split.X_train,
            split.y_train,
            cv=cv,
            scoring={"auc_pr": "average_precision", "f1": "f1"},
            n_jobs=-1,
            error_score="raise",
        )
        pipeline.fit(split.X_train, split.y_train)
        y_pred = pipeline.predict(split.X_test)
        y_score = pipeline.predict_proba(split.X_test)[:, 1]
        metrics = evaluate_predictions(split.y_test, y_pred, y_score)
        row = {
            "dataset": dataset_name,
            "model": model_name,
            "cv_auc_pr_mean": cv_scores["test_auc_pr"].mean(),
            "cv_auc_pr_std": cv_scores["test_auc_pr"].std(),
            "cv_f1_mean": cv_scores["test_f1"].mean(),
            "cv_f1_std": cv_scores["test_f1"].std(),
            **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
            "confusion_matrix": metrics["confusion_matrix"],
        }
        rows.append(row)
        if row["auc_pr"] > best_score:
            best_score = row["auc_pr"]
            best_model = pipeline

    print(f"Finished {dataset_name} training")

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / f"best_{dataset_name}_model.joblib")
    return pd.DataFrame(rows)
