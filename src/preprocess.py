import pandas as pd

LEAKY_COLS = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]

CATEGORICAL_FEATURES = [
    "State",
    "Area code",
    "International plan",
    "Voice mail plan",
]

def normalize_yes_no(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .replace({"true": "yes", "false": "no"})
    )

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=LEAKY_COLS, errors="ignore")

    df["International plan"] = normalize_yes_no(df["International plan"])
    df["Voice mail plan"] = normalize_yes_no(df["Voice mail plan"])
    df["State"] = df["State"].astype(str).str.strip().str.upper()
    df["Area code"] = df["Area code"].astype(str)

    return df
