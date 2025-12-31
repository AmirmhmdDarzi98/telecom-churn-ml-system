from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_dataframe, CATEGORICAL_FEATURES  # noqa: E402
from src.drift_detection import build_drift_report, psi_severity  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" 

st.set_page_config(page_title="Monitoring & Drift", page_icon="📉", layout="wide")

st.title("📉 Monitoring & Drift Detection")
st.caption("Drift between historical training data (80%) and future data (20%) using PSI and KS-test.")

@st.cache_data
def load_report():
    df_train = preprocess_dataframe(pd.read_csv(DATA_DIR / "churn-bigml-80.csv"))
    df_test = preprocess_dataframe(pd.read_csv(DATA_DIR / "churn-bigml-20.csv"))
    report = build_drift_report(
        train_df=df_train,
        test_df=df_test,
        target_col="Churn",
        categorical_features=CATEGORICAL_FEATURES
    )
    report["psi_level"] = report["psi"].apply(psi_severity)
    return report

report = load_report()

# --- Controls ---
c1, c2, c3 = st.columns(3)
feature_type = c1.selectbox("Feature type", ["all", "numeric", "categorical"], index=0)
psi_level = c2.selectbox("PSI level", ["all", "low", "moderate", "high"], index=0)
top_n = c3.slider("Top N features", min_value=5, max_value=30, value=15, step=5)

filtered = report.copy()
if feature_type != "all":
    filtered = filtered[filtered["type"] == feature_type]
if psi_level != "all":
    filtered = filtered[filtered["psi_level"] == psi_level]

filtered = filtered.sort_values("psi", ascending=False).head(top_n)

# --- Table ---
st.subheader("Drift Report (filtered)")
st.dataframe(filtered, use_container_width=True)

# --- PSI bar chart ---
st.subheader("PSI (Top Features)")
if len(filtered) > 0:
    fig = plt.figure()
    plt.barh(filtered["feature"][::-1], filtered["psi"][::-1])
    plt.xlabel("PSI")
    plt.ylabel("Feature")
    st.pyplot(fig)
else:
    st.warning("No features match the current filters.")

st.markdown(
    """
### PSI Severity Guide
- **PSI < 0.10** → Low drift  
- **0.10 – 0.25** → Moderate drift (monitor)  
- **> 0.25** → High drift (action needed)
"""
)
