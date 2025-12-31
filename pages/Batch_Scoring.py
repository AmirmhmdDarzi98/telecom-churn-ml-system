import json
from pathlib import Path
import sys
import io

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_dataframe  # noqa: E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Batch Scoring", page_icon="📦", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    return model, meta

def make_template() -> pd.DataFrame:
    # Minimal valid template with correct columns (no Churn needed)
    return pd.DataFrame([{
        "State": "CA",
        "Account length": 100,
        "Area code": 415,
        "International plan": "no",
        "Voice mail plan": "no",
        "Number vmail messages": 0,
        "Total day minutes": 180.0,
        "Total day calls": 100,
        "Total eve minutes": 200.0,
        "Total eve calls": 100,
        "Total night minutes": 200.0,
        "Total night calls": 100,
        "Total intl minutes": 10.0,
        "Total intl calls": 3,
        "Customer service calls": 1,
    }])

model, meta = load_artifacts()
threshold = float(meta["threshold"])

st.title("📦 Batch Scoring")
st.caption("Upload a CSV, score churn probability, and download results. Includes a template for correct columns.")

# --- Template download ---
tmpl = make_template()
buf_t = io.StringIO()
tmpl.to_csv(buf_t, index=False)
st.download_button(
    "Download CSV Template",
    data=buf_t.getvalue(),
    file_name="telecom_churn_template.csv",
    mime="text/csv"
)

st.divider()

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df_up = pd.read_csv(uploaded)

    st.write("Preview (first 10 rows):")
    st.dataframe(df_up.head(10), use_container_width=True)

    # Validate columns (soft validation)
    required_cols = set(tmpl.columns)
    missing = required_cols - set(df_up.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(list(missing))}")
        st.stop()

    df_proc = preprocess_dataframe(df_up)

    # Drop target if present
    if "Churn" in df_proc.columns:
        df_proc = df_proc.drop(columns=["Churn"])

    proba = model.predict_proba(df_proc)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df_up.copy()
    out["churn_probability"] = proba
    out["decision_contact"] = (pred == 1)

    st.subheader("Scored Output (preview)")
    st.dataframe(out.head(10), use_container_width=True)

    # Download scored output
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        label="Download scored CSV",
        data=buf.getvalue(),
        file_name="churn_scored.csv",
        mime="text/csv"
    )

    st.info(f"Decision threshold (cost-sensitive): {threshold:.2f}")
