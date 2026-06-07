from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def shap_summary(model_path: Path, X_sample: pd.DataFrame, out_path: Path) -> None:
    """Generate a SHAP summary plot for a saved tree-based pipeline when supported."""
    pipeline = joblib.load(model_path)
    preprocess = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    X_transformed = preprocess.transform(X_sample)
    feature_names = preprocess.get_feature_names_out()
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_transformed)
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
