from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _load_pipeline(model_path: Path):
    return joblib.load(model_path)


def _as_dense_matrix(X_transformed):
    if hasattr(X_transformed, "toarray"):
        return X_transformed.toarray()
    return np.asarray(X_transformed)


def _positive_class_matrix(shap_values):
    values = np.asarray(shap_values)
    if values.ndim == 3:
        class_axes = [axis for axis, size in enumerate(values.shape) if size == 2]
        class_axis = class_axes[-1] if class_axes else values.ndim - 1
        values = np.take(values, indices=1, axis=class_axis)
    return values


def _binary_shap_values(explainer, X_transformed):
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        class_index = 1 if len(shap_values) > 1 else 0
        return _positive_class_matrix(shap_values[class_index]), class_index
    return _positive_class_matrix(shap_values), None


def feature_importance_frame(estimator, feature_names: np.ndarray) -> pd.DataFrame:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        raise ValueError("The saved model does not expose built-in feature importances.")
    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_importance_frame(shap_values: np.ndarray, feature_names: np.ndarray) -> pd.DataFrame:
    values = np.abs(_positive_class_matrix(shap_values))
    return (
        pd.DataFrame({"feature": feature_names, "importance": values.mean(axis=0)})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def select_error_cases(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]

    outcomes = pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "y_pred": y_pred,
            "y_score": y_score,
        }
    )

    selections = {}
    masks = {
        "true_positive": (outcomes["y_true"] == 1) & (outcomes["y_pred"] == 1),
        "false_positive": (outcomes["y_true"] == 0) & (outcomes["y_pred"] == 1),
        "false_negative": (outcomes["y_true"] == 1) & (outcomes["y_pred"] == 0),
    }
    score_order = {
        "true_positive": False,
        "false_positive": False,
        "false_negative": True,
    }

    for label, mask in masks.items():
        candidates = outcomes.index[mask]
        if len(candidates) == 0:
            selections[label] = None
            continue
        candidate_scores = outcomes.loc[candidates, "y_score"]
        chosen_idx = candidate_scores.idxmin() if score_order[label] else candidate_scores.idxmax()
        selections[label] = int(chosen_idx)

    return selections


def save_bar_plot(df: pd.DataFrame, out_path: Path, title: str, x_label: str = "Importance") -> None:
    top = df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_importance_comparison(builtin_df: pd.DataFrame, shap_df: pd.DataFrame, out_path: Path) -> None:
    builtin_top = builtin_df.head(10).iloc[::-1].copy()
    shap_top = shap_df.head(10).iloc[::-1].copy()
    builtin_top["rank"] = range(1, len(builtin_top) + 1)
    shap_top["rank"] = range(1, len(shap_top) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    axes[0].barh(builtin_top["feature"], builtin_top["importance"], color="#3366CC")
    axes[0].set_title("Built-in feature importance")
    axes[0].set_xlabel("Importance")

    axes[1].barh(shap_top["feature"], shap_top["importance"], color="#DC3912")
    axes[1].set_title("Mean absolute SHAP importance")
    axes[1].set_xlabel("Importance")

    fig.suptitle("Top 10 feature drivers")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_force_plot(explainer, expected_value, shap_row, row_values, feature_names, out_path: Path, title: str) -> None:
    force_plot = shap.force_plot(
        expected_value,
        shap_row,
        row_values,
        feature_names=feature_names,
        matplotlib=False,
    )
    shap.save_html(str(out_path), force_plot)


def shap_summary(model_path: Path, X_sample: pd.DataFrame, out_path: Path) -> None:
    """Generate a SHAP summary plot for a saved tree-based pipeline when supported."""
    pipeline = _load_pipeline(model_path)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    X_transformed = _as_dense_matrix(preprocess.transform(X_sample))
    feature_names = np.asarray(preprocess.get_feature_names_out())
    explainer = shap.TreeExplainer(estimator)
    shap_values, _ = _binary_shap_values(explainer, X_transformed)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()


def generate_shap_artifacts(
    model_path: Path,
    X_summary: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figures_dir: Path,
    dataset_name: str,
) -> dict:
    pipeline = _load_pipeline(model_path)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    X_summary_transformed = _as_dense_matrix(preprocess.transform(X_summary))
    feature_names = np.asarray(preprocess.get_feature_names_out())

    explainer = shap.TreeExplainer(estimator)
    shap_values, class_index = _binary_shap_values(explainer, X_summary_transformed)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1 if class_index is None else class_index]

    builtin_df = feature_importance_frame(estimator, feature_names)
    shap_df = shap_importance_frame(shap_values, feature_names)

    figures_dir.mkdir(parents=True, exist_ok=True)
    builtin_path = figures_dir / f"{dataset_name}_builtin_feature_importance.png"
    shap_summary_path = figures_dir / f"{dataset_name}_shap_summary.png"
    comparison_path = figures_dir / f"{dataset_name}_feature_importance_comparison.png"

    save_bar_plot(builtin_df, builtin_path, f"{dataset_name.capitalize()} built-in feature importance")
    shap.summary_plot(shap_values, X_summary_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=160, bbox_inches="tight")
    plt.close()
    save_importance_comparison(builtin_df, shap_df, comparison_path)

    cases = select_error_cases(pipeline, X_test, y_test)
    force_plot_paths = {}
    for label, idx in cases.items():
        if idx is None:
            force_plot_paths[label] = None
            continue
        row = X_test.iloc[[idx]]
        row_transformed = _as_dense_matrix(preprocess.transform(row))
        row_shap_values, case_class_index = _binary_shap_values(explainer, row_transformed)
        row_shap = row_shap_values[0]
        row_values = row_transformed[0]
        case_expected_value = expected_value
        if isinstance(explainer.expected_value, (list, np.ndarray)) and case_class_index is not None:
            case_expected_value = explainer.expected_value[case_class_index]
        force_path = figures_dir / f"{dataset_name}_{label}_force_plot.html"
        save_force_plot(
            explainer,
            case_expected_value,
            row_shap,
            row_values,
            feature_names,
            force_path,
            f"{dataset_name.capitalize()} {label.replace('_', ' ')}",
        )
        force_plot_paths[label] = str(force_path.name)

    summary = {
        "dataset": dataset_name,
        "builtin_feature_importance": builtin_df.head(10).to_dict(orient="records"),
        "shap_importance": shap_df.head(10).to_dict(orient="records"),
        "selected_cases": cases,
        "force_plot_paths": force_plot_paths,
        "top_drivers": shap_df.head(5)["feature"].tolist(),
    }
    (figures_dir / f"{dataset_name}_shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
