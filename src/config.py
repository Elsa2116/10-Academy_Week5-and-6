from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

FRAUD_DATA = RAW_DIR / "Fraud_Data.csv"
IP_COUNTRY = RAW_DIR / "IpAddress_to_Country.csv"
CREDITCARD = RAW_DIR / "creditcard.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
