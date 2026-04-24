"""
Score aggregation utilities.

Supports multiple aggregation strategies for multi-reviewer scores:
- mean: arithmetic mean (aligns with evalate.py)
- median: robust to outliers
- trimmed_mean: remove outliers then mean
- weighted: weighted by confidence
"""

from typing import Any

import numpy as np


def aggregate_scores(
    scores: list[float],
    method: str = "mean",
    weights: list[float] | None = None,
) -> float | None:
    """Aggregate a list of scores.

    Args:
        scores: List of numeric scores.
        method: Aggregation method ('mean', 'median', 'trimmed_mean').
        weights: Optional weights for weighted aggregation.

    Returns:
        Aggregated score or None if no valid scores.
    """
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None

    if method == "mean":
        return round(float(np.mean(valid_scores)), 2)
    elif method == "median":
        return round(float(np.median(valid_scores)), 2)
    elif method == "trimmed_mean":
        if len(valid_scores) <= 2:
            return round(float(np.mean(valid_scores)), 2)
        # Trim 10% from each end
        trim_ratio = 0.1
        k = int(len(valid_scores) * trim_ratio)
        sorted_scores = sorted(valid_scores)
        trimmed = sorted_scores[k:len(sorted_scores) - k] if k > 0 else sorted_scores
        return round(float(np.mean(trimmed)), 2)
    elif method == "weighted" and weights:
        valid_weights = [w for s, w in zip(scores, weights) if s is not None]
        if sum(valid_weights) == 0:
            return round(float(np.mean(valid_scores)), 2)
        return round(float(np.average(valid_scores, weights=valid_weights)), 2)
    else:
        return round(float(np.mean(valid_scores)), 2)


def aggregate_reviews(
    reviews: list[dict[str, Any]],
    dimensions: list[str] | None = None,
    method: str = "mean",
) -> dict[str, float | None]:
    """Aggregate multiple reviewer scores into final scores.

    Args:
        reviews: List of review dicts with dimension scores.
        dimensions: List of dimension names to aggregate.
        method: Aggregation method.

    Returns:
        Dict mapping dimension to aggregated score.
    """
    if dimensions is None:
        dimensions = ["rating", "soundness", "presentation", "contribution"]

    result = {}
    for dim in dimensions:
        scores = [r.get(dim) for r in reviews if r.get(dim) is not None]
        result[dim] = aggregate_scores(scores, method)

    # Aggregate decision by majority vote
    decisions = [r.get("decision", "") for r in reviews]
    accepts = sum(1 for d in decisions if "accept" in str(d).lower())
    result["decision"] = "accept" if accepts >= len(decisions) / 2 else "reject"

    return result
