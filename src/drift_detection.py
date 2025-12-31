import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# --------------------------------------------------
# Population Stability Index (PSI) for numeric data
# --------------------------------------------------
def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    eps: float = 1e-6
) -> float:
    """
    Compute PSI between two numeric distributions.
    
    expected: training (reference) data
    actual: test / future data
    """
    expected = expected.dropna()
    actual = actual.dropna()

    # Quantile-based binning from expected (train)
    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))

    # Fallback if distribution is too discrete
    if len(breakpoints) <= 2:
        breakpoints = np.linspace(expected.min(), expected.max(), bins + 1)

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.clip(exp_perc, eps, 1)
    act_perc = np.clip(act_perc, eps, 1)

    psi_value = np.sum((exp_perc - act_perc) * np.log(exp_perc / act_perc))
    return psi_value


# --------------------------------------------------
# PSI for categorical features (based on proportions)
# --------------------------------------------------
def categorical_psi(
    expected: pd.Series,
    actual: pd.Series,
    eps: float = 1e-6
) -> float:
    """
    Compute PSI for categorical features using
    category frequency distributions.
    """
    expected_dist = expected.value_counts(normalize=True)
    actual_dist = actual.value_counts(normalize=True)

    all_categories = expected_dist.index.union(actual_dist.index)

    expected_p = expected_dist.reindex(all_categories).fillna(0).values
    actual_p = actual_dist.reindex(all_categories).fillna(0).values

    expected_p = np.clip(expected_p, eps, 1)
    actual_p = np.clip(actual_p, eps, 1)

    psi_value = np.sum((expected_p - actual_p) * np.log(expected_p / actual_p))
    return psi_value


# --------------------------------------------------
# Build full drift report
# --------------------------------------------------
def build_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    categorical_features: list
) -> pd.DataFrame:
    """
    Build a drift report comparing train vs test datasets.
    
    Returns a DataFrame with PSI and KS-test results.
    """
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    numeric_features = [
        col for col in X_train.columns if col not in categorical_features
    ]

    drift_rows = []

    # Numeric features: PSI + KS-test
    for col in numeric_features:
        psi_val = population_stability_index(
            X_train[col],
            X_test[col]
        )
        ks_stat, ks_pvalue = ks_2samp(X_train[col], X_test[col])

        drift_rows.append({
            "feature": col,
            "type": "numeric",
            "psi": psi_val,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_pvalue
        })

    # Categorical features: PSI only
    for col in categorical_features:
        psi_val = categorical_psi(
            X_train[col],
            X_test[col]
        )
        drift_rows.append({
            "feature": col,
            "type": "categorical",
            "psi": psi_val,
            "ks_stat": np.nan,
            "ks_pvalue": np.nan
        })

    drift_report = (
        pd.DataFrame(drift_rows)
        .sort_values(by="psi", ascending=False)
        .reset_index(drop=True)
    )

    return drift_report


# --------------------------------------------------
# Helper: classify PSI severity
# --------------------------------------------------
def psi_severity(psi_value: float) -> str:
    """
    Industry-standard PSI interpretation.
    """
    if psi_value < 0.10:
        return "low"
    if psi_value < 0.25:
        return "moderate"
    return "high"
