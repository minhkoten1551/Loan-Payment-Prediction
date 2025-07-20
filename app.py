"""
app.py – Streamlit inference app (updated)
────────────────────────────────────────────────────────────────
* Uses **number_input** (free‑text typing) for all numeric fields – no sliders.
* Adds the full set of numeric attributes present in `loan_data.csv`.
* Builds a feature frame that exactly matches training order (fills any
  missing one‑hot dummies with 0).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

root = os.path.dirname(__file__)
model_path         = os.path.join(root, "C:/Users/minhk/Documents/prethesis/stacked_model_complete.pkl")
encoder_path       = os.path.join(root, "C:/Users/minhk/Documents/prethesis/purpose_le.pkl")
feature_names_path = os.path.join(root, "C:/Users/minhk/Documents/prethesis/feature_names.json")

pipeline      = joblib.load(model_path)           
purpose_le    = joblib.load(encoder_path)         
feature_names = json.load(open(feature_names_path, "r", encoding="utf‑8"))

st.set_page_config(page_title="Loan Default Predictor", page_icon="💸", layout="centered")
st.title("💸 Loan Default Predictor")

st.markdown(
    """
✅ **Fully paid (0)**  |  🚨 **Not fully paid (1)**  
Enter a borrower profile below and click **Predict**.
"""
)

def build_feature_frame(user_inputs: dict) -> pd.DataFrame:
    """Return a single‑row DataFrame whose columns match exactly the training order."""
    X = pd.DataFrame([user_inputs])

    X["purpose_label"] = purpose_le.transform(X["purpose"])

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    return X[feature_names]

raw_purposes = list(purpose_le.classes_)
with st.sidebar:
    st.header("Purpose filter")
    hidden_options = st.multiselect("Hide purpose categories", raw_purposes, default=[])

def purpose_dropdown():
    visible = [p for p in raw_purposes if p not in hidden_options]
    return st.selectbox("Loan purpose", visible)



def user_form():
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            credit_policy   = st.number_input("Credit policy (1 = meets policy, 0 = otherwise)", 0, 1, value=1)
            int_rate_pct    = st.number_input("Interest rate (%)",                value=10.00, step=0.01, format="%.2f")
            installment     = st.number_input("Monthly installment ($)",          value=300.0, step=1.0)
            log_annual_inc  = st.number_input("Log Annual income ",                value=10.0, step=0.01)
            dti             = st.number_input("Debt‑to‑income ratio",             value=18.0, step=0.1, format="%.1f")
            fico            = st.number_input("FICO score",                       value=720,   step=1)
        with col2:
            days_with_cr    = st.number_input("Days with credit line",            value=4000,  step=10)
            revol_bal       = st.number_input("Revolving balance ($)",            value=8000,  step=100)
            revol_util_pct  = st.number_input("Revolving util (%)",               value=50.0,  step=0.1)
            inq_last_6m     = st.number_input("Inquiries last 6 months",          value=0,     step=1)
            delinq_2yrs     = st.number_input("Delinquencies in past 2 years",    value=0,     step=1)
            pub_rec         = st.number_input("Public records",                   value=0,     step=1)
            purpose         = purpose_dropdown()

        submitted = st.form_submit_button("Predict")

    user_dict = {
        "credit.policy":      credit_policy,
        "int.rate":           int_rate_pct / 100.0,        
        "installment":        installment,
        "log.annual.inc":     np.log(log_annual_inc),
        "dti":                dti,
        "fico":               fico,
        "days.with.cr.line":  days_with_cr,
        "revol.bal":          revol_bal,
        "revol.util":         revol_util_pct,
        "inq.last.6mths":     inq_last_6m,
        "delinq.2yrs":        delinq_2yrs,
        "pub.rec":            pub_rec,
        "purpose":            purpose,
    }

    return submitted, user_dict


clicked, user_inputs = user_form()

if clicked:
    X = build_feature_frame(user_inputs)


    try:
        proba = float(pipeline.predict_proba(X)[0, 1])
        y_pred = int(proba > 0.50)
    except AttributeError:
        proba = None
        y_pred = int(pipeline.predict(X)[0])
    try:
        y_pred = int((proba is not None and proba > 0.50) or (proba is None and pipeline.predict(X)[0]))
    except AttributeError:
        st.error("The model doesn't support prediction. Please check the pipeline configuration.")
        st.stop()


    st.subheader("Prediction")

    if proba is not None:
        st.write(f"**Probability of NOT fully paid:** {proba:.1%}")

    if y_pred == 1:
        st.markdown("❌ **Not Fully Paid**")
    else:
        st.markdown("✅ **Fully Paid**")
