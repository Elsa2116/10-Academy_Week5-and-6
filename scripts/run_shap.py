from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR
from src.explainability import generate_shap_artifacts
from src.preprocessing import prepare_modeling_frame, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP summary plots.")
    parser.add_argument("--dataset", choices=["fraud", "creditcard"], required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()

    if args.dataset == "fraud":
        df = pd.read_parquet(PROCESSED_DIR / "fraud_features.parquet")
        target = "class"
    else:
        df = pd.read_parquet(PROCESSED_DIR / "creditcard_features.parquet")
        target = "Class"

    modeling_df = prepare_modeling_frame(df, target)
    split = stratified_split(modeling_df, target)
    X_sample = split.X_test.sample(min(args.sample_size, len(split.X_test)), random_state=42)
    summary = generate_shap_artifacts(
        MODELS_DIR / f"best_{args.dataset}_model.joblib",
        X_sample,
        split.X_test,
        split.y_test,
        FIGURES_DIR,
        args.dataset,
    )
    print(summary["top_drivers"])
    print(f"Saved SHAP plot for {args.dataset}.")


if __name__ == "__main__":
    main()
