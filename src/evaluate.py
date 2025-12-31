from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, threshold: float):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
