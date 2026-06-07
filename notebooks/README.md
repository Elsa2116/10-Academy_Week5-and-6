# Notebooks

Recommended notebook sequence:

1. `eda-fraud-data.ipynb` - e-commerce cleaning, EDA, IP-to-country enrichment.
2. `eda-creditcard.ipynb` - credit card EDA and imbalance review.
3. `feature-engineering.ipynb` - feature creation and preprocessing checks.
4. `modeling.ipynb` - model training and comparison.
5. `shap-explainability.ipynb` - model explainability and business recommendations.

The production logic lives in `src/` and `scripts/` so notebooks stay readable and reproducible.
