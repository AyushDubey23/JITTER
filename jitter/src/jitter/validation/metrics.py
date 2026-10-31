"""Classification and confidence calibration metrics for diagnosis validation."""

from __future__ import annotations

import numpy as np


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> np.ndarray:
    """Compute an N x N confusion matrix for the given labels."""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred, strict=False):
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1
    return cm


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class Precision, Recall, and F1 scores."""
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    results: dict[str, dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(np.sum(cm[i, :])),
        }

    return results


def compute_macro_f1(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> float:
    """Compute unweighted macro-averaged F1 score across all classes."""
    metrics = compute_classification_metrics(y_true, y_pred, labels)
    f1_scores = [m["f1"] for m in metrics.values() if m["support"] > 0]
    return float(np.mean(f1_scores)) if f1_scores else 0.0


def compute_brier_score(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
) -> float:
    """Compute multi-class Brier score reflecting accuracy and confidence calibration."""
    correct = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred, strict=False)]
    confs = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    return float(np.mean((confs - corr) ** 2))


def compute_expected_calibration_error(
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across equally spaced confidence bins."""
    correct = np.asarray([1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred, strict=False)])
    confs = np.asarray(confidences, dtype=float)
    n = len(correct)
    if n == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        mask = (confs >= low) & (confs < high) if i < n_bins - 1 else (confs >= low) & (confs <= high)
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confs[mask]))
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)
