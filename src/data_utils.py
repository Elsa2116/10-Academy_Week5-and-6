from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset(s):\n"
            + "\n".join(missing)
            + "\nPlace the raw CSV files in data/raw/ using the expected filenames."
        )


def load_fraud_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {
        "user_id", "signup_time", "purchase_time", "purchase_value", "device_id",
        "source", "browser", "sex", "age", "ip_address", "class"
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Fraud_Data.csv is missing columns: {sorted(missing)}")
    return df


def load_ip_country(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"lower_bound_ip_address", "upper_bound_ip_address", "country"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"IpAddress_to_Country.csv is missing columns: {sorted(missing)}")
    return df


def load_creditcard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"creditcard.csv is missing columns: {sorted(missing)}")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and return a copy. Missing value handling is dataset-specific."""
    return df.drop_duplicates().copy()


def class_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).rename_axis(target).reset_index(name="count")
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return counts.sort_values(target)
