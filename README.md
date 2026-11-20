# repeat-purchase-predictor

Repeat Purchase Likelihood Predictor

An interactive machine learning web application built with **Streamlit** and **scikit-learn** to estimate the probability that a retail customer will make a repeat purchase.

## Overview

The app takes customer order behavior inputs and returns an estimated repeat-order probability using a `StandardScaler` + `LogisticRegression` pipeline loaded from `model.joblib`.

### Input Features
- **Days Since Last Order** (`days_since_order`): Days elapsed since the customer's most recent order (`1`–`90`).
- **Past Orders Count** (`past_orders`): Total previous orders placed by the customer (`1`–`25`).
- **Average Order Value ($)** (`avg_order_value`): Typical order spend (`200`–`5000`).
- **Used Discount / Offer?** (`discount_used`): Whether a promotional discount or offer was used (`No` / `Yes`).

