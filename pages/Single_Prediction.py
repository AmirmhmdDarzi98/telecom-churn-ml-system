import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

# --- Make src importable ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_dataframe  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Single Prediction", page_icon="🔮", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    return model, meta

model, meta = load_artifacts()
threshold = float(meta["threshold"])

st.title("🔮 Single Customer Prediction")
st.caption("Enter customer attributes and get churn probability + decision (cost-sensitive threshold).")

with st.form("input_form"):
    c1, c2 = st.columns(2)

    state = c1.text_input("State (e.g., CA)", "CA").strip().upper()
    area_code = c2.selectbox("Area code", ["408", "415", "510"], index=0)

    account_length = c1.number_input("Account length", min_value=0, value=100)
    intl_plan = c2.selectbox("International plan", ["yes", "no"], index=1)
    vmail_plan = c1.selectbox("Voice mail plan", ["yes", "no"], index=1)
    vmail_msgs = c2.number_input("Number vmail messages", min_value=0, value=0)

    day_minutes = c1.number_input("Total day minutes", min_value=0.0, value=180.0)
    day_calls = c2.number_input("Total day calls", min_value=0, value=100)

    eve_minutes = c1.number_input("Total eve minutes", min_value=0.0, value=200.0)
    eve_calls = c2.number_input("Total eve calls", min_value=0, value=100)

    night_minutes = c1.number_input("Total night minutes", min_value=0.0, value=200.0)
    night_calls = c2.number_input("Total night calls", min_value=0, value=100)

    intl_minutes = c1.number_input("Total intl minutes", min_value=0.0, value=10.0)
    intl_calls = c2.number_input("Total intl calls", min_value=0, value=3)

    cs_calls = c1.number_input("Customer service calls", min_value=0, value=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "State": state,
        "Account length": int(account_length),
        "Area code": int(area_code),
        "International plan": intl_plan,
        "Voice mail plan": vmail_plan,
        "Number vmail messages": int(vmail_msgs),
        "Total day minutes": float(day_minutes),
        "Total day calls": int(day_calls),
        "Total eve minutes": float(eve_minutes),
        "Total eve calls": int(eve_calls),
        "Total night minutes": float(night_minutes),
        "Total night calls": int(night_calls),
        "Total intl minutes": float(intl_minutes),
        "Total intl calls": int(intl_calls),
        "Customer service calls": int(cs_calls),
    }

    df_in = pd.DataFrame([row])
    df_in = preprocess_dataframe(df_in)

    proba = float(model.predict_proba(df_in)[:, 1][0])
    decision = "✅ Contact" if proba >= threshold else "➖ No Contact"

    st.subheader("Result")
    st.metric("Churn probability", f"{proba:.3f}")
    st.write(f"**Decision threshold:** {threshold:.2f}")
    st.success(decision)
