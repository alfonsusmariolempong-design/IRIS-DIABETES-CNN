from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

def evaluate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred)
    }