from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from .config import RANDOM_STATE, TEST_SIZE


DROP_FOR_MODELING = {
    "user_id",
    "device_id",
    "signup_time",
    "purchase_time",
    "ip_address",
    "lower_bound_ip_address",
    "upper_bound_ip_address",
}


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_modeling_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Return a leakage-safe, sklearn-friendly frame for resampling/modeling.

    The processed Task 1 datasets keep raw audit columns for EDA and traceability.
    Modeling uses engineered replacements for those fields instead of timestamps,
    raw IP addresses, or range-lookup helper columns.
    """
    out = df.copy()
    datetime_columns = out.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    drop_cols = [c for c in DROP_FOR_MODELING | set(datetime_columns) if c in out.columns and c != target]
    return out.drop(columns=drop_cols)


def stratified_split(df: pd.DataFrame, target: str) -> SplitData:
    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def apply_smote(X_train, y_train):
    """Balance training data only using SMOTE.

    SMOTE is intentionally applied after splitting and preprocessing so synthetic
    examples cannot leak into validation/test data.
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    return smote.fit_resample(X_train, y_train)
