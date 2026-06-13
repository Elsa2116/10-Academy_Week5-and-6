# Fraud Detection for E-commerce and Bank Transactions

This repository implements an end-to-end fraud detection workflow for two separate transaction streams:

1. `Fraud_Data.csv` - e-commerce transactions with user, device, timestamp, browser, and IP features.
2. `creditcard.csv` - bank card transactions with anonymized PCA features.
3. `IpAddress_to_Country.csv` - IP range lookup table used to enrich e-commerce transactions with country.

The project follows the Adey Innovations Inc. challenge requirements: clean and explore the data, engineer fraud-oriented features, handle class imbalance, train fraud classifiers, and explain the best model with SHAP.

## Repository structure

```text
fraud-detection/
├── .vscode/settings.json
├── .github/workflows/unittests.yml
├── data/                    # Source data and generated datasets
│   ├── README.md
│   ├── raw/                 # Add raw CSV files here; ignored by git
│   └── processed/           # Generated feature datasets; ignored by git
├── notebooks/                # Runnable analysis notebooks for EDA, features, modeling, and SHAP
├── src/                      # Reusable project modules
├── tests/                    # Unit tests
├── models/                   # Saved model artifacts; ignored by git
│   └── README.md
├── scripts/                  # CLI scripts for Task 1-3
├── reports/figures/          # Generated EDA/model/SHAP plots
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # PowerShell
# or: .venv\Scripts\activate.bat  # Command Prompt
pip install --upgrade pip
pip install -r requirements.txt
```

Place the three raw files in `data/raw/` with these names:

```text
data/raw/Fraud_Data.csv
data/raw/IpAddress_to_Country.csv
data/raw/creditcard.csv
```

Generated outputs are written to these paths:

- `data/processed/` for cleaned and feature-engineered datasets;
- `models/` for serialized model artifacts;
- `reports/figures/` for generated plots and visual summaries.

The repository keeps `data/` and `models/` visible with placeholder documentation so the expected project layout is clear even before files are generated.

## Run Task 1: data analysis and preprocessing

```bash
python scripts/run_task1.py
```

This script:

- validates file availability and schemas;
- removes duplicates and fixes data types;
- enriches e-commerce transactions with country using IP range lookup;
- engineers `hour_of_day`, `day_of_week`, `time_since_signup_hours`, user transaction count, and device transaction count;
- creates EDA plots in `reports/figures/`;
- saves cleaned feature datasets to `data/processed/`;
- records class imbalance before and after SMOTE on training data only.

## Run Task 2: model training

```bash
python scripts/train_models.py --dataset both
```

Models included:

- Logistic Regression baseline;
- Random Forest ensemble model;
- 5-fold stratified cross-validation;
- AUC-PR, F1-score, precision, recall, and confusion matrix outputs.

## Run Task 3: SHAP explainability

```bash
python scripts/run_shap.py --dataset fraud
python scripts/run_shap.py --dataset creditcard
python scripts/generate_final_report.py
```

The SHAP script generates:

- global SHAP summary plot;
- built-in feature-importance bar chart;
- built-in vs SHAP comparison chart;
- feature importance comparison;
- force/waterfall-style explanations for true positive, false positive, and false negative examples when available.

The report generator consolidates the metrics and SHAP outputs into `reports/final_report.md` for final submission.

## Notes on class imbalance

Accuracy is intentionally not used as the primary metric because fraud is rare. The project prioritizes AUC-PR and F1-score, with confusion matrices used to interpret false positives and false negatives.

Resampling is applied only after the stratified train-test split and only on the training set to avoid data leakage.

# 10-Academy_Week5-and-6
