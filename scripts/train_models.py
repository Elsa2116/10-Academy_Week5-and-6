from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, PROCESSED_DIR
from src.modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--dataset", choices=["fraud", "creditcard", "both"], default="both")
    args = parser.parse_args()

    outputs = []
    if args.dataset in {"fraud", "both"}:
        fraud = pd.read_parquet(PROCESSED_DIR / "fraud_features.parquet")
        outputs.append(train_and_evaluate(fraud, "class", MODELS_DIR, "fraud"))
    if args.dataset in {"creditcard", "both"}:
        credit = pd.read_parquet(PROCESSED_DIR / "creditcard_features.parquet")
        outputs.append(train_and_evaluate(credit, "Class", MODELS_DIR, "creditcard"))

    results = pd.concat(outputs, ignore_index=True)
    results.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    print(results[["dataset", "model", "auc_pr", "f1", "precision", "recall"]])


if __name__ == "__main__":
    main()
