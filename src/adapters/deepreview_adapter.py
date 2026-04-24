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

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger
from src.utils.parser import get_average_scores, parse_deepreviewer_output

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
        return round(float(match.group(1)), 2)

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

    Auto-detects file format (.json or .csv).

    Args:
        raw_path: Path to raw data (sample.json or test_2024.csv).
        output_path: Optional path to save processed data.
        max_samples: Optional limit on number of samples to process.

    Returns:
        List of unified samples.
    """
    raw_path = Path(raw_path)

    if raw_path.suffix.lower() == ".csv":
        return load_csv_and_adapt(str(raw_path), output_path, max_samples)

    logger.info(f"Loading DeepReview JSON from {raw_path}")

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


def adapt_csv_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Adapt a single CSV row from DeepReview dataset to unified format.

    CSV columns: inputs, outputs, year, id, mode, rating, decision, reviewer_comments
    """
    try:
        inputs = json.loads(row["inputs"])
        outputs = json.loads(row["outputs"])
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse JSON in row {row.get('id', '')}: {e}")
        return None

    # Extract paper context from user message (messages[1])
    paper_context = ""
    if len(inputs) > 1 and inputs[1].get("role") == "user":
        paper_context = inputs[1].get("content", "")

    # Extract title from paper context
    title = ""
    title_match = re.search(r'\\title\{(.*?)\}', paper_context)
    if title_match:
        title = title_match.group(1)

    # Parse human reviews from reviewer_comments
    parsed_reviews = []
    comments_raw = row.get("reviewer_comments", "")
    if comments_raw:
        try:
            comments_list = json.loads(comments_raw)
            if isinstance(comments_list, list):
                parsed_reviews = [parse_human_review(r) for r in comments_list]
        except json.JSONDecodeError:
            pass

    # Compute ground truth from human reviews (includes soundness/presentation/contribution)
    ground_truth = compute_ground_truth(parsed_reviews)

    # Override rating with explicit rating list if available
    rating_raw = row.get("rating", "")
    try:
        rating_list = ast.literal_eval(rating_raw) if rating_raw else []
        if isinstance(rating_list, list) and rating_list:
            ground_truth["rating"] = sum(rating_list) / len(rating_list)
    except Exception:
        pass

    # Override decision with paper-level decision if available
    decision_raw = row.get("decision", "")
    if decision_raw:
        ground_truth["decision"] = "accept" if "accept" in decision_raw.lower() else "reject"

    # Parse predicted scores from outputs (last assistant message) for reference only
    pred_scores = {"rating": None, "soundness": None, "presentation": None, "contribution": None, "decision": "reject"}
    if outputs and isinstance(outputs, list):
        last_msg = outputs[-1]
        if last_msg.get("role") == "assistant":
            parsed = parse_deepreviewer_output(last_msg.get("content", ""))
            sim_reviews = parsed.get("simulated_reviews", [])
            if sim_reviews:
                avg = get_average_scores(sim_reviews)
                pred_scores.update(avg)
            else:
                meta = parsed.get("meta_review", {})
                pred_scores["rating"] = meta.get("rating")
            pred_scores["decision"] = parsed.get("decision", "reject")

    unified = {
        "id": row.get("id", ""),
        "title": title,
        "paper_context": paper_context,
        "reviews": parsed_reviews,
        "ground_truth": ground_truth,
        "metadata": {
            "source": "deepreview_csv",
            "year": row.get("year", ""),
            "mode": row.get("mode", ""),
            "num_human_reviewers": len(parsed_reviews),
        },
        "raw_predictions": {
            "pred_scores": pred_scores,
        }
    }
    return unified


def load_csv_and_adapt(raw_path: str, output_path: str | None = None,
                       max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load DeepReview CSV data and adapt to unified format."""
    raw_path = Path(raw_path)
    logger.info(f"Loading DeepReview CSV from {raw_path}")

    # Increase CSV field size limit for large outputs columns
    import sys
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10

    unified_data = []
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            adapted = adapt_csv_row(row)
            if adapted:
                unified_data.append(adapted)
            if max_samples and len(unified_data) >= max_samples:
                break

    logger.info(f"Adapted {len(unified_data)} samples from CSV")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved processed data to {out_path}")

    return unified_data


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
