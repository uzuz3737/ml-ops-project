import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from typing import Dict
from src.utils.config import load_config


def compute_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = None
) -> Dict[str, float]:
    """Computes comprehensive classification metrics."""
    if threshold is None:
        threshold = load_config()["evaluation"]["classification_threshold"]
    y_pred = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
        "pr_auc": float(average_precision_score(y_true, y_pred_prob)),
    }
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None
):
    """Generates and optionally saves confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No Default", "Default"]
    )
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_pred_prob: np.ndarray, save_path: str = None):
    """Generates and optionally saves ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_val = roc_auc_score(y_true, y_pred_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})"
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
