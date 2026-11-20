from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

FEATURES = ["days_since_order", "past_orders", "avg_order_value", "discount_used"]
MODEL_PATH = Path(__file__).with_name("model.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("Repeat Purchase Likelihood Predictor")
st.write("Adjust customer order metrics to estimate repeat-order probability.")

days_since_order = st.slider("Days Since Last Order", 1, 90, 15)
past_orders = st.slider("Past Orders Count", 1, 25, 5)
avg_order_value = st.slider("Average Order Value ($)", 200, 5000, 1500, step=100)
discount_used = st.radio("Used Discount / Offer?", ["No", "Yes"], horizontal=True)

if st.button("Predict"):
    if not MODEL_PATH.exists():
        st.error("Missing 'model.joblib'. Run train_model.py first so it sits next to app.py.")
        st.stop()

    row = pd.DataFrame(
        [[days_since_order, past_orders, avg_order_value, 1 if discount_used == "Yes" else 0]],
        columns=FEATURES,
    )
    prob = load_model().predict_proba(row)[0][1]

    st.metric("Estimated Repeat Order Chance", f"{prob:.0%}")
    st.caption("Educational demo trained on simulated retail order data — not business advice.")