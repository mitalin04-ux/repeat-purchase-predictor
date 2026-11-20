import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n = 800

days_since_order = rng.integers(1, 90, n)
past_orders = rng.integers(1, 25, n)
avg_order_value = rng.integers(200, 5000, n)
discount_used = rng.integers(0, 2, n)  # 1 = used promo/discount, 0 = no

propensity = (
    -0.04 * days_since_order
    + 0.14 * past_orders
    + 0.0005 * (avg_order_value - 1500)
    + 0.65 * discount_used
    + rng.normal(0, 0.7, n)
)
repeat_buyer = (propensity > 0).astype(int)

FEATURES = ["days_since_order", "past_orders", "avg_order_value", "discount_used"]
X = pd.DataFrame({
    "days_since_order": days_since_order,
    "past_orders": past_orders,
    "avg_order_value": avg_order_value,
    "discount_used": discount_used,
})

model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, repeat_buyer)
joblib.dump(model, "model.joblib")
print("Saved model.joblib ->", FEATURES)