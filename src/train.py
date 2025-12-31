import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from preprocess import preprocess_dataframe, CATEGORICAL_FEATURES

def train_model(df: pd.DataFrame):
    df = preprocess_dataframe(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    numeric_features = [c for c in X.columns if c not in CATEGORICAL_FEATURES]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    clf.fit(X, y)
    return clf
