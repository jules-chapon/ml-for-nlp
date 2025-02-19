"""Evaluation functions"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Get metrics for a binary classification problem.
    Get accuracy, precision, recall, f1 score and roc auc score.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict[str, float]: Dictionary with metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Get the confusion matrix for the given true and predicted labels.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)
