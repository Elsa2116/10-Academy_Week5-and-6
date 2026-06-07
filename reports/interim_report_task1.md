# Interim-1 Report: Data Analysis and Preprocessing

## Project Scope

Adey Innovations needs fraud detection workflows for two transaction streams:

- E-commerce transactions in `Fraud_Data.csv`, enriched with country from `IpAddress_to_Country.csv`.
- Bank credit card transactions in `creditcard.csv`, where most predictors are anonymized PCA features.

Task 1 prepares both datasets for modeling through cleaning, exploratory analysis, feature engineering, scaling/encoding design, and leakage-safe imbalance handling.

## Reproducible Command

Place the raw files in `data/raw/` with the required names, then run:

```bash
python scripts/run_task1.py
```

The script writes cleaned feature datasets to `data/processed/` and EDA figures to `reports/figures/`.

## Data Cleaning

The pipeline validates each dataset schema before processing:

- `Fraud_Data.csv`: user, signup, purchase, value, device, source, browser, sex, age, IP address, and `class`.
- `IpAddress_to_Country.csv`: lower IP bound, upper IP bound, and country.
- `creditcard.csv`: `Time`, `Amount`, `Class`, and `V1` through `V28`.

Duplicate rows are removed with `basic_clean`. Missing numeric values are handled later with median imputation inside the modeling preprocessor. Missing categorical values are handled with most-frequent imputation. This keeps the raw cleaned datasets traceable while ensuring model inputs are complete.

Date fields in the e-commerce data are converted with `pd.to_datetime(..., errors="coerce")`, allowing invalid timestamps to become missing values instead of crashing the workflow. Engineered time features are then imputed during preprocessing if needed.

## Exploratory Data Analysis

The Task 1 workflow records class distributions for both datasets:

- `data/processed/fraud_class_distribution.csv`
- `data/processed/creditcard_class_distribution.csv`

It also generates core plots:

- `reports/figures/fraud_class_distribution.png`
- `reports/figures/creditcard_class_distribution.png`
- `reports/figures/purchase_value_by_class.png`
- `reports/figures/amount_by_class.png`
- `reports/figures/top_country_fraud_rates.png`

These outputs are designed to answer the required EDA questions: univariate distributions, bivariate target relationships, and class imbalance severity.

## Geolocation Integration

E-commerce IP addresses are converted to integer form using `ip_to_int`. Dotted IPv4 strings are converted using bit shifts, while numeric IP values are cast to integers. Invalid or missing IP values become missing and are later labeled as `Unknown`.

Country enrichment uses a range lookup:

1. Sort transactions by integer IP.
2. Sort IP ranges by lower bound.
3. Use `pandas.merge_asof` to attach the nearest lower range.
4. Keep the country only when the transaction IP is less than or equal to the matched upper bound.
5. Assign `Unknown` when no valid range exists.

This is faster and more reliable than row-by-row range filtering.

## Feature Engineering

For `Fraud_Data.csv`, the pipeline creates:

- `hour_of_day`: purchase hour.
- `day_of_week`: purchase weekday as an integer.
- `time_since_signup_hours`: hours between signup and purchase, clipped at zero for invalid negative intervals.
- `user_transaction_count`: number of transactions observed for the user.
- `device_transaction_count`: number of transactions observed for the device.
- `purchase_value_log1p`: log-transformed purchase value.
- `country`: geolocation feature derived from IP range lookup.

For `creditcard.csv`, the pipeline creates:

- `hour_of_day`: derived from seconds elapsed since the first transaction.
- `amount_log1p`: log-transformed transaction amount.

The processed datasets retain audit columns for traceability. Modeling uses `prepare_modeling_frame` to drop raw timestamps and raw IP/range helper fields before scaling, encoding, resampling, or training.

## Data Transformation

The shared preprocessor uses:

- `SimpleImputer(strategy="median")` and `StandardScaler` for numerical features.
- `SimpleImputer(strategy="most_frequent")` and one-hot encoding for categorical features.

This gives both datasets a consistent, model-ready feature matrix while preserving separate target definitions:

- E-commerce target: `class`
- Credit card target: `Class`

## Class Imbalance Handling

Fraud is rare in both datasets, so raw accuracy is not a suitable success metric. The pipeline uses a stratified train-test split to preserve the original fraud ratio, then applies SMOTE only to the transformed training set.

This order prevents leakage:

1. Split the original processed data with stratification.
2. Fit imputers, scalers, and encoders on the training set.
3. Transform the training set.
4. Apply SMOTE to the transformed training set only.
5. Leave the test set unchanged.

The resulting before/after class counts are written to:

- `data/processed/resampling_summary.json`

SMOTE is used because it preserves all legitimate and fraud observations while creating synthetic minority examples in feature space. This is preferable to simple undersampling for the first modeling pass because undersampling can discard a large amount of legitimate transaction behavior.

## Deliverable Checklist

- Cleaned and feature-engineered datasets: `data/processed/fraud_features.parquet`, `data/processed/creditcard_features.parquet`.
- Class imbalance summaries: `data/processed/*_class_distribution.csv`, `data/processed/resampling_summary.json`.
- EDA figures: `reports/figures/*.png`.
- Reusable code: `src/data_utils.py`, `src/features.py`, `src/preprocessing.py`, `src/visualization.py`.
- Runnable Task 1 command: `python scripts/run_task1.py`.
- Supporting notebooks: `notebooks/eda-fraud-data.ipynb`, `notebooks/eda-creditcard.ipynb`, `notebooks/feature-engineering.ipynb`.
