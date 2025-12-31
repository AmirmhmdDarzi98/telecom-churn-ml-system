📞 Telecom Churn Prediction – ML System
A production-style machine learning project for predicting customer churn, with a focus on business-driven decision making, cost-sensitive thresholds, and post-deployment evaluation.

🚀 Live Demo
🔗 Streamlit Web App:
https://telecom-churn-ml-system-dev.streamlit.app/Single_Prediction

🎯 Problem
Predicting churn is not enough.
The real business question is:
Which customers should be contacted to prevent churn, given asymmetric costs?
Missing a churned customer is expensive
Contacting a loyal customer has a smaller cost
Decisions must be optimized based on business impact, not accuracy

🧠 Solution Overview
Leakage-safe data preprocessing
Logistic Regression baseline model
Cost-sensitive threshold optimization
Evaluation on unseen future data (pseudo-production)

Data drift monitoring (PSI & KS-test)
🖥️ Web Application Features
🔮 Single Customer Prediction – churn probability + decision
📦 Batch Scoring – upload CSV & download predictions
📉 Drift Monitoring Dashboard
📑 Project Report – metrics, confusion matrix, cost curves

🛠️ Tech Stack:
Python, Pandas, NumPy
Scikit-learn
Streamlit
Joblib

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

👤 Author
Amir Mohammad Darzi
