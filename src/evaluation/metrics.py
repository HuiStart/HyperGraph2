"""
Evaluation metrics aligned with official DeepReview implementation.

Reference: Research/evaluate/DeepReview/evalate.py

Official metrics:
1. MSE / MAE per dimension (Rating, Soundness, Presentation, Contribution)
2. Spearman correlation per dimension
3. Decision Accuracy (Accept/Reject)
4. Decision F1 (macro average)
5. Pairwise Accuracy per dimension

Extended metrics (non-official, for comparison):
- RMSE, Pearson, QWK
"""

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_mse(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Squared Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Absolute Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root Mean Squared Error (extended metric)."""
    return float(np.sqrt(compute_mse(y_true, y_pred)))


def compute_spearman(y_true: list[float], y_pred: list[float]) -> float | str:
    """Spearman rank correlation coefficient.

    Aligns with evalate.py: spearmanr(true_ratings, pred_ratings)
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return "nan"
    try:
        corr, _ = spearmanr(y_true, y_pred)
        if corr is None or np.isnan(corr):
            return "nan"
        return float(corr)
    except Exception as e:
        logger.warning(f"Spearman computation failed: {e}")
        return "nan"


def compute_pearson(y_true: list[float], y_pred: list[float]) -> float | str:
    """Pearson correlation coefficient (extended metric)."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return "nan"
    try:
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        if np.isnan(corr):
            return "nan"
        return float(corr)
    except Exception as e:
        logger.warning(f"Pearson computation failed: {e}")
        return "nan"


def compute_qwk(y_true: list[float], y_pred: list[float], max_rating: int = 10) -> float | str:
    """Quadratic Weighted Kappa (extended metric)."""
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return "nan"

    if len(y_true) < 2:
        return "nan"

    try:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # Build confusion matrix
        min_val = int(min(min(y_true_arr), min(y_pred_arr)))
        max_val = int(max(max(y_true_arr), max(y_pred_arr)))
        num_classes = max_val - min_val + 1

        labels = list(range(min_val, max_val + 1))
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)

        # Weights
        weights = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

        # Expected matrix
        true_hist = cm.sum(axis=1)
        pred_hist = cm.sum(axis=0)
        expected = np.outer(true_hist, pred_hist) / cm.sum()

        # QWK
        numerator = np.sum(weights * cm)
        denominator = np.sum(weights * expected)
        if denominator == 0:
            return 1.0 if numerator == 0 else 0.0
        return 1.0 - numerator / denominator
    except Exception as e:
        logger.warning(f"QWK computation failed: {e}")
        return "nan"


def compute_decision_accuracy(y_true_decisions: list[str], y_pred_decisions: list[str]) -> float:
    """Decision accuracy: proportion of correct accept/reject decisions.

    Aligns with evalate.py:
        if pred_decision in true_decision: decision_acc.append(1.)
    """
    if not y_true_decisions or not y_pred_decisions:
        return 0.0

    correct = 0
    for true, pred in zip(y_true_decisions, y_pred_decisions):
        pred_norm = "accept" if "accept" in pred.lower() else "reject"
        true_norm = "accept" if "accept" in true.lower() else "reject"
        if pred_norm == true_norm:
            correct += 1

    return correct / len(y_true_decisions)


def compute_decision_f1(y_true_decisions: list[str], y_pred_decisions: list[str]) -> float:
    """Decision F1 score with macro averaging.

    Aligns with evalate.py:
        precision_recall_fscore_support(true_decisions, pred_decisions, average='macro')
    """
    if not y_true_decisions or not y_pred_decisions:
        return 0.0

    # Convert to binary (1=accept, 0=reject)
    y_true_bin = [1 if "accept" in d.lower() else 0 for d in y_true_decisions]
    y_pred_bin = [1 if "accept" in d.lower() else 0 for d in y_pred_decisions]

    try:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="macro", zero_division=0
        )
        return float(f1)
    except Exception as e:
        logger.warning(f"F1 computation failed: {e}")
        return 0.0


def compute_pairwise_accuracy(y_true: list[float], y_pred: list[float]) -> float:
    """Pairwise comparison accuracy.

    Aligns with evalate.py calculate_pairwise_accuracies():
    For all pairs of papers, check if model correctly ranks them.
    """
    if len(y_true) < 2:
        return 0.0

    n_correct = 0
    n_total = 0

    for i, j in combinations(range(len(y_true)), 2):
        true_order = y_true[i] > y_true[j]
        pred_order = y_pred[i] > y_pred[j]
        if true_order == pred_order:
            n_correct += 1
        n_total += 1

    return n_correct / n_total if n_total > 0 else 0.0


def compute_all_metrics(
    y_true: dict[str, list[float]],
    y_pred: dict[str, list[float]],
    y_true_decisions: list[str] | None = None,
    y_pred_decisions: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all official metrics across all dimensions.

    Args:
        y_true: Dict mapping dimension -> ground truth scores list.
        y_pred: Dict mapping dimension -> predicted scores list.
        y_true_decisions: Ground truth decisions list.
        y_pred_decisions: Predicted decisions list.

    Returns:
        Nested dict of metrics.
    """
    results = {}
    dimensions = ["rating", "soundness", "presentation", "contribution"]

    for dim in dimensions:
        if dim not in y_true or dim not in y_pred:
            continue

        true_vals = y_true[dim]
        pred_vals = y_pred[dim]

        if not true_vals or not pred_vals or len(true_vals) != len(pred_vals):
            continue

        dim_results = {
            "mse": compute_mse(true_vals, pred_vals),
            "mae": compute_mae(true_vals, pred_vals),
            "rmse": compute_rmse(true_vals, pred_vals),
            "spearman": compute_spearman(true_vals, pred_vals),
            "pearson": compute_pearson(true_vals, pred_vals),
            "pairwise_acc": compute_pairwise_accuracy(true_vals, pred_vals),
        }

        # QWK with appropriate max rating
        max_rating = 10 if dim == "rating" else 5
        dim_results["qwk"] = compute_qwk(true_vals, pred_vals, max_rating)

        results[dim] = dim_results

    # Decision metrics
    if y_true_decisions and y_pred_decisions:
        results["decision"] = {
            "accuracy": compute_decision_accuracy(y_true_decisions, y_pred_decisions),
            "f1_macro": compute_decision_f1(y_true_decisions, y_pred_decisions),
        }

    return results


def format_metrics_table(metrics: dict[str, Any], method_name: str = "Method") -> str:
    """Format metrics as markdown table (aligns with evalate.py create_markdown_table)."""
    lines = [f"## Results for {method_name}", ""]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")

    for dim, dim_metrics in metrics.items():
        if dim == "decision":
            lines.append(f"| Decision Accuracy | {dim_metrics['accuracy']:.4f} |")
            lines.append(f"| Decision F1 (macro) | {dim_metrics['f1_macro']:.4f} |")
        else:
            dim_name = dim.capitalize()
            for metric_name, value in dim_metrics.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                lines.append(f"| {dim_name} {metric_name.upper()} | {value_str} |")

    return "\n".join(lines)
