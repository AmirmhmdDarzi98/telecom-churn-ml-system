import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocess import preprocess_dataframe, CATEGORICAL_FEATURES  # noqa: E402
from src.threshold_optimization import expected_cost  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" 
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

st.set_page_config(page_title="Project Report", page_icon="📑", layout="wide")

st.title("📑 Project Report")
st.caption("Summary of model performance, cost-sensitive thresholding, and threshold sensitivity on train vs future data.")

@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    meta = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    return model, meta

@st.cache_data
def load_data():
    df_train = preprocess_dataframe(pd.read_csv(DATA_DIR / "churn-bigml-80.csv"))
    df_test = preprocess_dataframe(pd.read_csv(DATA_DIR / "churn-bigml-20.csv"))
    return df_train, df_test

model, meta = load_artifacts()
df_train, df_test = load_data()

threshold_deploy = float(meta["threshold"])
cost_fn = int(meta.get("cost_fn", 100))
cost_fp = int(meta.get("cost_fp", 10))

# ---- Train/Test split
X_train = df_train.drop(columns=["Churn"])
y_train = df_train["Churn"].astype(int)

X_test = df_test.drop(columns=["Churn"])
y_test = df_test["Churn"].astype(int)

# ---- Compute probabilities
# Model already trained on full train in save_model.py, but for reporting we do:
train_proba = model.predict_proba(X_train)[:, 1]
test_proba = model.predict_proba(X_test)[:, 1]

# ---- Metrics at deployed threshold
train_pred = (train_proba >= threshold_deploy).astype(int)
test_pred = (test_proba >= threshold_deploy).astype(int)

train_auc = roc_auc_score(y_train, train_proba)
test_auc = roc_auc_score(y_test, test_proba)

st.subheader("Performance Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Train ROC-AUC", f"{train_auc:.4f}")
c2.metric("Test ROC-AUC", f"{test_auc:.4f}")
c3.metric("Deployed threshold", f"{threshold_deploy:.2f}")

# ---- Confusion matrix (test)
st.subheader("Test Confusion Matrix (Deployed Threshold)")
cm = confusion_matrix(y_test, test_pred)
st.write(cm)

st.subheader("Test Classification Report (Deployed Threshold)")
st.code(classification_report(y_test, test_pred), language="text")

# ---- Cost at deployed threshold
train_cost = expected_cost(y_train, train_pred, cost_fn=cost_fn, cost_fp=cost_fp)
test_cost = expected_cost(y_test, test_pred, cost_fn=cost_fn, cost_fp=cost_fp)

st.subheader("Business Cost (Deployed Threshold)")
c4, c5 = st.columns(2)
c4.metric("Train expected cost", f"{train_cost}")
c5.metric("Test expected cost", f"{test_cost}")

# ---- Threshold sensitivity curve (train vs test)
st.subheader("Threshold Sensitivity (Cost Curve)")
thresholds = np.linspace(0.05, 0.95, 19)

train_costs = []
test_costs = []
for t in thresholds:
    train_costs.append(expected_cost(y_train, (train_proba >= t).astype(int), cost_fn=cost_fn, cost_fp=cost_fp))
    test_costs.append(expected_cost(y_test, (test_proba >= t).astype(int), cost_fn=cost_fn, cost_fp=cost_fp))

best_train_idx = int(np.argmin(train_costs))
best_test_idx = int(np.argmin(test_costs))

best_train_t = float(thresholds[best_train_idx])
best_test_t = float(thresholds[best_test_idx])

c6, c7 = st.columns(2)
c6.metric("Best train threshold (cost)", f"{best_train_t:.2f}")
c7.metric("Best test threshold (cost)", f"{best_test_t:.2f}")

fig = plt.figure()
plt.plot(thresholds, train_costs, label="train_cost")
plt.plot(thresholds, test_costs, label="test_cost")
plt.xlabel("Threshold")
plt.ylabel("Expected cost")
plt.legend()
st.pyplot(fig)

st.info(
    "Note: A shift between train-optimal and test-optimal threshold is common in production and "
    "motivates post-deployment threshold monitoring/recalibration."
)
