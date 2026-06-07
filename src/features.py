from __future__ import annotations

import numpy as np
import pandas as pd


def ip_to_int(ip_value) -> int | float:
    """Convert an IPv4 address stored as dotted string or numeric value to integer.

    The e-commerce dataset sometimes stores IP addresses as floats. Numeric values are
    rounded down to int. Invalid strings return NaN so they can be treated as unknown.
    """
    if pd.isna(ip_value):
        return np.nan
    if isinstance(ip_value, (int, float, np.integer, np.floating)):
        return int(ip_value)
    try:
        parts = str(ip_value).split(".")
        if len(parts) != 4:
            return np.nan
        nums = [int(part) for part in parts]
        if any(n < 0 or n > 255 for n in nums):
            return np.nan
        return (nums[0] << 24) + (nums[1] << 16) + (nums[2] << 8) + nums[3]
    except Exception:
        return np.nan


def add_country_by_ip(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    """Range-match transactions to country using efficient merge_asof.

    For each transaction IP integer, merge with the nearest lower IP range start and
    keep the match only when the IP is also <= upper_bound_ip_address.
    """
    tx = fraud_df.copy()
    ranges = ip_df.copy()

    tx["_original_order"] = np.arange(len(tx))
    tx["ip_int"] = tx["ip_address"].apply(ip_to_int)
    tx["ip_int"] = pd.to_numeric(tx["ip_int"], errors="coerce").astype("float64")
    ranges["lower_bound_ip_address"] = pd.to_numeric(ranges["lower_bound_ip_address"], errors="coerce").astype("float64")
    ranges["upper_bound_ip_address"] = pd.to_numeric(ranges["upper_bound_ip_address"], errors="coerce").astype("float64")

    tx_with_ip = tx.dropna(subset=["ip_int"]).sort_values("ip_int")
    tx_without_ip = tx[tx["ip_int"].isna()].copy()
    ranges_sorted = ranges.dropna(subset=["lower_bound_ip_address", "upper_bound_ip_address"]).sort_values("lower_bound_ip_address")

    merged = pd.merge_asof(
        tx_with_ip,
        ranges_sorted,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )
    valid = merged["ip_int"].le(merged["upper_bound_ip_address"])
    merged.loc[~valid, "country"] = "Unknown"
    merged["country"] = merged["country"].fillna("Unknown")
    tx_without_ip["country"] = "Unknown"
    out = pd.concat([merged, tx_without_ip], ignore_index=True, sort=False)
    return out.sort_values("_original_order").drop(columns=["_original_order"]).reset_index(drop=True)


def engineer_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signup_time"] = pd.to_datetime(out["signup_time"], errors="coerce")
    out["purchase_time"] = pd.to_datetime(out["purchase_time"], errors="coerce")
    out["hour_of_day"] = out["purchase_time"].dt.hour
    out["day_of_week"] = out["purchase_time"].dt.dayofweek
    out["time_since_signup_hours"] = (
        out["purchase_time"] - out["signup_time"]
    ).dt.total_seconds() / 3600
    out["time_since_signup_hours"] = out["time_since_signup_hours"].clip(lower=0)
    out["user_transaction_count"] = out.groupby("user_id")["user_id"].transform("count")
    out["device_transaction_count"] = out.groupby("device_id")["device_id"].transform("count")
    out["purchase_value_log1p"] = np.log1p(out["purchase_value"])
    return out


def engineer_creditcard_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_of_day"] = (out["Time"] // 3600) % 24
    out["amount_log1p"] = np.log1p(out["Amount"])
    return out
