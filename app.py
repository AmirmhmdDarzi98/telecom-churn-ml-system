import streamlit as st

st.set_page_config(
    page_title="Telecom Churn ML System",
    page_icon="📞",
    layout="centered"
)

st.title("📞 Telecom Churn – Production-Style ML System")
st.write(
    """
Welcome! This portfolio project demonstrates an **end-to-end churn prediction system**
with **cost-sensitive thresholding**, **pseudo-production evaluation**, and **drift monitoring**.

Use the pages in the sidebar to:
- 🔮 Predict churn for a single customer
- 📦 Score a batch CSV file and download results
- 📉 Monitor drift between historical vs future data
"""
)

st.info("Tip: Start with **Single Prediction**, then try **Batch Scoring**.")
