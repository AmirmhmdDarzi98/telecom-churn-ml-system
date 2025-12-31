import numpy as np

def expected_cost(y_true, y_pred, cost_fn=100, cost_fp=10):
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return fn * cost_fn + (fp + tp) * cost_fp

def optimize_threshold(y_true, y_proba, thresholds):
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        costs.append(expected_cost(y_true, y_pred))
    best_idx = int(np.argmin(costs))
    return thresholds[best_idx], costs[best_idx]
