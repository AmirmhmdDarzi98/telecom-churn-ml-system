import json
from pathlib import Path
import joblib
import pandas as pd

from train import train_model
from preprocess import preprocess_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" 
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

OPTIMAL_THRESHOLD = 0.34  # from your training optimization

def main():
    train_path = DATA_DIR / "churn-bigml-80.csv"
    df_train = pd.read_csv(train_path)
    df_train = preprocess_dataframe(df_train)

    model = train_model(df_train)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metadata = {
        "threshold": OPTIMAL_THRESHOLD,
        "cost_fn": 100,
        "cost_fp": 10,
        "target": "Churn",
        "notes": "LogReg + leakage-safe preprocessing + cost-sensitive threshold"
    }
    (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("Saved model to artifacts/model.joblib")
    print("Saved metadata to artifacts/metadata.json")

if __name__ == "__main__":
    main()
