"""
DeepReview dataset adapter.

Converts DeepReview raw format (sample.json) to project internal unified format.
Aligns with official data structure discovered in Research/evaluate/DeepReview/sample.json.

Key mappings:
- review[].rating -> integer (overall score)
- review[].content.soundness -> "3 good" -> extract 3
- review[].content.presentation -> "3 good" -> extract 3
- review[].content.contribution -> "2 fair" -> extract 2
- Ground truth = mean across all human reviewers per dimension
"""

import json
import re
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_numeric_score(value: Any) -> float | None:
    """Extract numeric score from values like '3 good', '2 fair', '6: marginally above...'.

    Aligns with evalate.py logic: int(r['content']['soundness'][0])
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # Try to match leading number (integer or float)
    match = re.match(r'(\d+(?:\.\d+)?)', text)
    if match:
        return float(match.group(1))

    return None


def parse_human_review(review: dict[str, Any]) -> dict[str, Any]:
    """Parse a single human review from DeepReview format."""
    content = review.get("content", {})

    # Extract confidence number if present
    confidence_text = content.get("confidence", "")
    confidence = extract_numeric_score(confidence_text)

    parsed = {
        "reviewer_id": review.get("id", ""),
        "rating": review.get("rating"),
        "soundness": extract_numeric_score(content.get("soundness")),
        "presentation": extract_numeric_score(content.get("presentation")),
        "contribution": extract_numeric_score(content.get("contribution")),
        "confidence": confidence,
        "summary": content.get("summary", ""),
        "strengths": content.get("strengths", ""),
        "weaknesses": content.get("weaknesses", ""),
        "questions": content.get("questions", ""),
        "suggestions": content.get("suggestions", content.get("weakness", "")),
        "decision": "accept",  # Human reviews don't have explicit decision, use paper-level
    }

    return parsed


def compute_ground_truth(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute ground truth by averaging human reviewer scores.

    Aligns exactly with evalate.py:
        rates.mean() -> rating
        soundness.mean() -> soundness
        presentation.mean() -> presentation
        contribution.mean() -> contribution
    """
    if not reviews:
        return {
            "rating": None,
            "soundness": None,
            "presentation": None,
            "contribution": None,
            "decision": "reject",
        }

    dimensions = ["rating", "soundness", "presentation", "contribution"]
    ground_truth = {}

    for dim in dimensions:
        scores = [r[dim] for r in reviews if r.get(dim) is not None]
        if scores:
            ground_truth[dim] = sum(scores) / len(scores)
        else:
            ground_truth[dim] = None

    # Decision: use majority or accept if any reviewer accepts
    decisions = [r.get("decision", "reject") for r in reviews]
    accepts = sum(1 for d in decisions if "accept" in d.lower())
    ground_truth["decision"] = "accept" if accepts >= len(decisions) / 2 else "reject"

    return ground_truth


def adapt_sample(raw_sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a single DeepReview raw sample to unified internal format."""
    raw_reviews = raw_sample.get("review", [])
    parsed_reviews = [parse_human_review(r) for r in raw_reviews]

    # Use paper-level decision if available, else compute from reviews
    paper_decision = raw_sample.get("decision", "")
    if paper_decision:
        for r in parsed_reviews:
            r["decision"] = "accept" if "accept" in paper_decision.lower() else "reject"

    ground_truth = compute_ground_truth(parsed_reviews)
    # Prefer paper-level decision for ground truth
    if paper_decision:
        ground_truth["decision"] = "accept" if "accept" in paper_decision.lower() else "reject"

    unified = {
        "id": raw_sample.get("id", ""),
        "title": raw_sample.get("title", ""),
        "paper_context": raw_sample.get("paper_context", ""),
        "reviews": parsed_reviews,
        "ground_truth": ground_truth,
        "metadata": {
            "source": "deepreview",
            "num_reviewers": len(parsed_reviews),
            "task_type": "paper_review",
            "raw_decision": raw_sample.get("decision", ""),
        },
        # Store raw predictions for baseline comparison
        "raw_predictions": {
            "pred_fast_mode": raw_sample.get("pred_fast_mode", ""),
            "pred_standard_mode": raw_sample.get("pred_standard_mode", ""),
            "pred_best_mode": raw_sample.get("pred_best_mode", {}),
        }
    }

    return unified


def load_and_adapt(raw_path: str, output_path: str | None = None,
                   max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load DeepReview raw data, adapt to unified format, optionally save.

    Args:
        raw_path: Path to raw sample.json.
        output_path: Optional path to save processed data.
        max_samples: Optional limit on number of samples to process.

    Returns:
        List of unified samples.
    """
    raw_path = Path(raw_path)
    logger.info(f"Loading DeepReview data from {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if max_samples:
        raw_data = raw_data[:max_samples]

    logger.info(f"Adapting {len(raw_data)} samples...")
    unified_data = [adapt_sample(sample) for sample in raw_data]

    # Filter out samples with no valid ground truth
    valid_data = [
        s for s in unified_data
        if s["ground_truth"]["rating"] is not None
    ]
    logger.info(f"Valid samples with ground truth: {len(valid_data)}/{len(unified_data)}")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(valid_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed data to {out_path}")

    return valid_data


def get_ground_truth_arrays(data: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Extract ground truth arrays for evaluation.

    Returns:
        Dict mapping dimension names to lists of scores.
    """
    dimensions = ["rating", "soundness", "presentation", "contribution"]
    arrays = {dim: [] for dim in dimensions}

    for sample in data:
        gt = sample.get("ground_truth", {})
        for dim in dimensions:
            val = gt.get(dim)
            if val is not None:
                arrays[dim].append(float(val))

    return arrays


def get_decisions(data: list[dict[str, Any]]) -> list[str]:
    """Extract ground truth decisions."""
    return [s["ground_truth"]["decision"] for s in data if s["ground_truth"].get("decision")]


if __name__ == "__main__":
    # Quick test
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/deepreview_processed.json"

    data = load_and_adapt(raw_path, output_path)
    print(f"Processed {len(data)} samples")

    # Show first sample ground truth
    if data:
        print("\nFirst sample ground truth:")
        print(json.dumps(data[0]["ground_truth"], indent=2))
