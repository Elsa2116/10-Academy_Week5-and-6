# Fraud Detection Final Submission Report

This report consolidates the end-to-end fraud detection workflow for e-commerce and credit card transactions.

## Scope

- Fraud_Data.csv: e-commerce fraud classification with IP-to-country enrichment and behavioral features.
- creditcard.csv: bank fraud classification using the anonymized PCA feature set.
- Evaluation focuses on AUC-PR, F1, precision, recall, and confusion matrices because the target is highly imbalanced.

## Task 1 Outputs

- Cleaned datasets: `data/processed/fraud_features.parquet`, `data/processed/creditcard_features.parquet`.
- Class balance summaries: `data/processed/fraud_class_distribution.csv`, `data/processed/creditcard_class_distribution.csv`.
- Resampling log: `data/processed/resampling_summary.json`.
- EDA figures: `reports/figures/fraud_class_distribution.png`, `reports/figures/creditcard_class_distribution.png`, `reports/figures/purchase_value_by_class.png`, `reports/figures/amount_by_class.png`, `reports/figures/top_country_fraud_rates.png`.

## Fraud

Best model: **random_forest**

- Hold-out AUC-PR: 0.7083
- Hold-out F1: 0.6036
- Precision: 0.5238
- Recall: 0.7120
- Stratified CV AUC-PR: 0.7137 ± 0.0041
- Stratified CV F1: 0.6094 ± 0.0036

### SHAP drivers

- num__device_transaction_count
- num__time_since_signup_hours
- cat__browser_Chrome
- num__day_of_week
- cat__country_United States

### Recommendations

- Transactions placed soon after signup should trigger step-up verification or a manual review.
- High-velocity users or devices should be rate-limited and monitored for burst behavior.
- Country mismatches between IP geolocation and customer profile should receive additional risk scoring.

### Supporting figures

- [Built-in feature importance](figures/fraud_builtin_feature_importance.png)
- [SHAP summary](figures/fraud_shap_summary.png)
- [Built-in vs SHAP comparison](figures/fraud_feature_importance_comparison.png)
- [True positive force plot](figures/fraud_true_positive_force_plot.html)
- [False positive force plot](figures/fraud_false_positive_force_plot.html)
- [False negative force plot](figures/fraud_false_negative_force_plot.html)

## Credit Card

Best model: **random_forest**

- Hold-out AUC-PR: 0.7795
- Hold-out F1: 0.5536
- Precision: 0.4124
- Recall: 0.8421
- Stratified CV AUC-PR: 0.7814 ± 0.0019
- Stratified CV F1: 0.5561 ± 0.1025

### SHAP drivers

- num__V14
- num__V12
- num__V4
- num__V17
- num__V3

### Recommendations

- Review the strongest SHAP drivers and convert them into step-up checks and rule-based alerts.

### Supporting figures

- [Built-in feature importance](figures/creditcard_builtin_feature_importance.png)
- [SHAP summary](figures/creditcard_shap_summary.png)
- [Built-in vs SHAP comparison](figures/creditcard_feature_importance_comparison.png)
- [True positive force plot](figures/creditcard_true_positive_force_plot.html)
- [False positive force plot](figures/creditcard_false_positive_force_plot.html)
- [False negative force plot](figures/creditcard_false_negative_force_plot.html)

## Model Selection Notes

The chosen model for each dataset balances performance on the minority class with interpretability. Logistic regression provides a transparent baseline, while random forest captures nonlinear fraud interactions and feeds the SHAP analysis.

## Business Takeaways

- Use step-up verification for high-risk behavioral patterns that emerge soon after signup.
- Monitor transaction bursts by the same user or device as a velocity signal.
- Combine geolocation, channel, browser, and timing signals into risk-scored review queues rather than blanket blocking.

## Reproducibility

Run `python scripts/run_task1.py`, `python scripts/train_models.py --dataset both`, `python scripts/run_shap.py --dataset fraud`, `python scripts/run_shap.py --dataset creditcard`, then `python scripts/generate_final_report.py`.