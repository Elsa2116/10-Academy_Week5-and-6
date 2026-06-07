from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CREDITCARD, FIGURES_DIR, FRAUD_DATA, IP_COUNTRY, PROCESSED_DIR
from src.data_utils import basic_clean, class_distribution, load_creditcard, load_fraud_data, load_ip_country, require_files
from src.features import add_country_by_ip, engineer_creditcard_features, engineer_fraud_features
from src.preprocessing import apply_smote, build_preprocessor, prepare_modeling_frame, stratified_split
from src.visualization import save_class_distribution_plot, save_numeric_by_target_boxplot, save_top_country_fraud_plot


def main() -> None:
    require_files([FRAUD_DATA, IP_COUNTRY, CREDITCARD])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fraud = basic_clean(load_fraud_data(FRAUD_DATA))
    ip_country = load_ip_country(IP_COUNTRY)
    credit = basic_clean(load_creditcard(CREDITCARD))

    fraud = add_country_by_ip(fraud, ip_country)
    fraud = engineer_fraud_features(fraud)
    credit = engineer_creditcard_features(credit)

    fraud.to_parquet(PROCESSED_DIR / "fraud_features.parquet", index=False)
    credit.to_parquet(PROCESSED_DIR / "creditcard_features.parquet", index=False)

    fraud_dist = class_distribution(fraud, "class")
    credit_dist = class_distribution(credit, "Class")
    fraud_dist.to_csv(PROCESSED_DIR / "fraud_class_distribution.csv", index=False)
    credit_dist.to_csv(PROCESSED_DIR / "creditcard_class_distribution.csv", index=False)

    save_class_distribution_plot(fraud_dist, "class", "E-commerce class distribution", FIGURES_DIR / "fraud_class_distribution.png")
    save_class_distribution_plot(credit_dist, "Class", "Credit card class distribution", FIGURES_DIR / "creditcard_class_distribution.png")
    save_numeric_by_target_boxplot(fraud, "purchase_value", "class", FIGURES_DIR / "purchase_value_by_class.png")
    save_numeric_by_target_boxplot(credit, "Amount", "Class", FIGURES_DIR / "amount_by_class.png")
    save_top_country_fraud_plot(fraud, FIGURES_DIR / "top_country_fraud_rates.png")

    # Demonstrate leakage-safe resampling distribution after train-only SMOTE.
    resampling_summary = {}
    for name, df, target in [("fraud", fraud, "class"), ("creditcard", credit, "Class")]:
        modeling_df = prepare_modeling_frame(df, target)
        split = stratified_split(modeling_df, target)
        preprocessor = build_preprocessor(split.X_train)
        X_train_transformed = preprocessor.fit_transform(split.X_train)
        X_resampled, y_resampled = apply_smote(X_train_transformed, split.y_train)
        resampling_summary[name] = {
            "before": split.y_train.value_counts().sort_index().to_dict(),
            "after": pd.Series(y_resampled).value_counts().sort_index().to_dict(),
            "test_unchanged": split.y_test.value_counts().sort_index().to_dict(),
        }
    (PROCESSED_DIR / "resampling_summary.json").write_text(json.dumps(resampling_summary, indent=2), encoding="utf-8")
    print("Task 1 complete. Outputs saved to data/processed and reports/figures.")


if __name__ == "__main__":
    main()
