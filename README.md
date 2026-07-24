# tabengine — Zero-Shot Tabular ML SDK

[![PyPI Version](https://img.shields.io/pypi/v/tabengine.svg)](https://pypi.org/project/tabengine/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`tabengine` is an open-source Python library for zero-shot predictive machine learning on tabular data. Powered by Tabular Foundation Models (TabFM) and sub-10ms distance retrieval kernels (DuckDB), it replaces traditional XGBoost model training and hyperparameter tuning pipelines with real-time in-context learning.

---

## ⚡ Quickstart

### 1. Installation

```bash
pip install tabengine
```

### 2. Zero-Shot Classification

```python
import pandas as pd
from tabengine import ZeroShotClassifier

# Historical context dataframe
df = pd.DataFrame({
    'monthly_active_days': [28, 2, 22, 5, 29],
    'support_tickets': [1, 8, 2, 6, 0],
    'account_tier': ['Enterprise', 'Basic', 'Pro', 'Basic', 'Enterprise'],
    'churn_status': ['Retained', 'Churned', 'Retained', 'Churned', 'Retained']
})

# Initialize Zero-Shot Classifier (0 seconds spent training model parameters)
model = ZeroShotClassifier(target="churn_status")
model.fit(df)

# Predict on new query row
query_row = {
    'monthly_active_days': 3,
    'support_tickets': 7,
    'account_tier': 'Basic'
}

# Get prediction with explanation & retrieved context rows
explanation = model.predict_with_explanation(query_row)
print("Prediction:", explanation["prediction"])
print("Confidence:", explanation["confidence"])
print("Top Matching Context Row:", explanation["top_contexts"][0])
```

### 3. Zero-Shot Regression

```python
from tabengine import ZeroShotRegressor

# Real estate price valuation example
model = ZeroShotRegressor(target="price")
model.fit(real_estate_df)

predicted_price = model.predict([{"sqft": 1650, "beds": 3, "baths": 2}])
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
