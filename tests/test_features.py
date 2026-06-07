import pandas as pd

from src.features import engineer_fraud_features, ip_to_int


def test_ip_to_int_dotted_ipv4():
    assert ip_to_int("127.0.0.1") == 2130706433


def test_ip_to_int_invalid_returns_nan():
    value = ip_to_int("not-an-ip")
    assert pd.isna(value)


def test_engineer_fraud_features_time_since_signup():
    df = pd.DataFrame({
        "signup_time": ["2026-01-01 00:00:00"],
        "purchase_time": ["2026-01-02 12:00:00"],
        "purchase_value": [100],
        "user_id": [1],
        "device_id": ["d1"],
    })
    out = engineer_fraud_features(df)
    assert out.loc[0, "time_since_signup_hours"] == 36
    assert out.loc[0, "hour_of_day"] == 12
    assert out.loc[0, "user_transaction_count"] == 1
