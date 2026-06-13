from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, PROCESSED_DIR


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_section(dataset: str, metrics: pd.DataFrame) -> str:
    display_name = "Credit Card" if dataset == "creditcard" else "Fraud"
    dataset_metrics = metrics.loc[metrics["dataset"] == dataset].copy()
    if dataset_metrics.empty:
        return f"## {display_name}\n\nMetrics are not available yet. Run `python scripts/train_models.py --dataset {dataset}` first.\n"

    best_row = dataset_metrics.sort_values(["auc_pr", "f1"], ascending=False).iloc[0]
    shap_summary = _load_json(FIGURES_DIR / f"{dataset}_shap_summary.json")
    top_drivers = shap_summary.get("top_drivers", [])

    top_driver_lines = "\n".join(f"- {feature}" for feature in top_drivers) if top_drivers else "- Not available yet."
    recommendations = []
    if any("time_since_signup" in feature for feature in top_drivers):
        recommendations.append("Transactions placed soon after signup should trigger step-up verification or a manual review.")
    if any("transaction_count" in feature for feature in top_drivers):
        recommendations.append("High-velocity users or devices should be rate-limited and monitored for burst behavior.")
    if any("country" in feature for feature in top_drivers):
        recommendations.append("Country mismatches between IP geolocation and customer profile should receive additional risk scoring.")
    if any("hour_of_day" in feature for feature in top_drivers):
        recommendations.append("Late-night or unusual-hour activity should be weighted more heavily in fraud scoring rules.")
    if not recommendations:
        recommendations.append("Review the strongest SHAP drivers and convert them into step-up checks and rule-based alerts.")

    recommendation_lines = "\n".join(f"- {item}" for item in recommendations[:3])

    return f"""## {display_name}

Best model: **{best_row['model']}**

- Hold-out AUC-PR: {best_row['auc_pr']:.4f}
- Hold-out F1: {best_row['f1']:.4f}
- Precision: {best_row['precision']:.4f}
- Recall: {best_row['recall']:.4f}
- Stratified CV AUC-PR: {best_row['cv_auc_pr_mean']:.4f} ± {best_row['cv_auc_pr_std']:.4f}
- Stratified CV F1: {best_row['cv_f1_mean']:.4f} ± {best_row['cv_f1_std']:.4f}

### SHAP drivers

{top_driver_lines}

### Recommendations

{recommendation_lines}

### Supporting figures

- [Built-in feature importance](figures/{dataset}_builtin_feature_importance.png)
- [SHAP summary](figures/{dataset}_shap_summary.png)
- [Built-in vs SHAP comparison](figures/{dataset}_feature_importance_comparison.png)
- [True positive force plot](figures/{dataset}_true_positive_force_plot.html)
- [False positive force plot](figures/{dataset}_false_positive_force_plot.html)
- [False negative force plot](figures/{dataset}_false_negative_force_plot.html)
"""


def build_report() -> str:
    metrics_path = PROCESSED_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
    else:
        metrics = pd.DataFrame(columns=["dataset", "model", "auc_pr", "f1", "precision", "recall", "cv_auc_pr_mean", "cv_auc_pr_std", "cv_f1_mean", "cv_f1_std"])

    parts = [
        "# Fraud Detection Final Submission Report",
        "",
        "This report consolidates the end-to-end fraud detection workflow for e-commerce and credit card transactions.",
        "",
        "## Scope",
        "",
        "- Fraud_Data.csv: e-commerce fraud classification with IP-to-country enrichment and behavioral features.",
        "- creditcard.csv: bank fraud classification using the anonymized PCA feature set.",
        "- Evaluation focuses on AUC-PR, F1, precision, recall, and confusion matrices because the target is highly imbalanced.",
        "",
        "## Task 1 Outputs",
        "",
        "- Cleaned datasets: `data/processed/fraud_features.parquet`, `data/processed/creditcard_features.parquet`.",
        "- Class balance summaries: `data/processed/fraud_class_distribution.csv`, `data/processed/creditcard_class_distribution.csv`.",
        "- Resampling log: `data/processed/resampling_summary.json`.",
        "- EDA figures: `reports/figures/fraud_class_distribution.png`, `reports/figures/creditcard_class_distribution.png`, `reports/figures/purchase_value_by_class.png`, `reports/figures/amount_by_class.png`, `reports/figures/top_country_fraud_rates.png`.",
        "",
        _dataset_section("fraud", metrics),
        _dataset_section("creditcard", metrics),
        "## Model Selection Notes",
        "",
        "The chosen model for each dataset balances performance on the minority class with interpretability. Logistic regression provides a transparent baseline, while random forest captures nonlinear fraud interactions and feeds the SHAP analysis.",
        "",
        "## Business Takeaways",
        "",
        "- Use step-up verification for high-risk behavioral patterns that emerge soon after signup.",
        "- Monitor transaction bursts by the same user or device as a velocity signal.",
        "- Combine geolocation, channel, browser, and timing signals into risk-scored review queues rather than blanket blocking.",
        "",
        "## Reproducibility",
        "",
        "Run `python scripts/run_task1.py`, `python scripts/train_models.py --dataset both`, `python scripts/run_shap.py --dataset fraud`, `python scripts/run_shap.py --dataset creditcard`, then `python scripts/generate_final_report.py`.",
    ]
    return "\n".join(parts)


def main() -> None:
    report_path = PROJECT_ROOT / "reports" / "final_report.md"
    report_path.write_text(build_report(), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()