"""
Official evaluation pipeline aligning with Research/evaluate/DeepReview/evalate.py.

This module provides:
1. Parsing AI-generated predictions (pred_fast_mode, pred_standard_mode)
2. Extracting scores from predictions
3. Computing all official metrics
4. Side-by-side win rate evaluation
"""

import json
from pathlib import Path
from typing import Any

from src.adapters.deepreview_adapter import load_and_adapt
from src.evaluation.metrics import compute_all_metrics, format_metrics_table
from src.utils.logger import get_logger
from src.utils.parser import parse_deepreviewer_output

logger = get_logger(__name__)


def extract_pred_scores(pred_text: str) -> dict[str, Any]:
    """Extract structured scores from AI prediction text.

    Handles both fast mode (single review) and standard mode (simulated reviewers).
    For standard mode, averages across simulated reviewers.
    """
    parsed = parse_deepreviewer_output(pred_text)

    scores = {
        "rating": None,
        "soundness": None,
        "presentation": None,
        "contribution": None,
        "decision": "",
    }

    # Try simulated reviewers first (standard/best mode)
    sim_reviews = parsed.get("simulated_reviews", [])
    if sim_reviews:
        avg = {k: [] for k in ["rating", "soundness", "presentation", "contribution"]}
        for review in sim_reviews:
            for dim in avg:
                val = review.get(dim)
                if val is not None and val != "":
                    try:
                        avg[dim].append(float(val))
                    except (ValueError, TypeError):
                        pass
        for dim, vals in avg.items():
            if vals:
                scores[dim] = sum(vals) / len(vals)

    # Fall back to meta review
    meta = parsed.get("meta_review", {})
    for dim in ["rating", "soundness", "presentation", "contribution"]:
        if scores[dim] is None and meta.get(dim) is not None:
            scores[dim] = meta[dim]

    # Decision
    if parsed.get("decision"):
        scores["decision"] = parsed["decision"]
    elif meta.get("decision"):
        decision = meta["decision"].lower()
        scores["decision"] = "accept" if "accept" in decision else "reject"

    return scores


def evaluate_predictions(
    data: list[dict[str, Any]],
    pred_field: str = "pred_fast_mode",
) -> dict[str, Any]:
    """Evaluate predictions against ground truth.

    Args:
        data: List of unified samples (from deepreview_adapter).
        pred_field: Which prediction field to evaluate
                   ('pred_fast_mode', 'pred_standard_mode', or 'pred_best_mode').

    Returns:
        Metrics dictionary.
    """
    y_true = {dim: [] for dim in ["rating", "soundness", "presentation", "contribution"]}
    y_pred = {dim: [] for dim in ["rating", "soundness", "presentation", "contribution"]}
    y_true_decisions = []
    y_pred_decisions = []

    skipped = 0
    for item in data:
        raw_preds = item.get("raw_predictions", {})

        if pred_field == "pred_best_mode":
            best_mode = raw_preds.get("pred_best_mode", {})
            pred_text = best_mode.get("output", "") if isinstance(best_mode, dict) else ""
        else:
            pred_text = raw_preds.get(pred_field, "")

        if not pred_text:
            skipped += 1
            continue

        pred_scores = extract_pred_scores(pred_text)

        # Skip if no rating extracted
        if pred_scores["rating"] is None:
            skipped += 1
            continue

        gt = item["ground_truth"]

        for dim in y_true:
            gt_val = gt.get(dim)
            pred_val = pred_scores.get(dim)
            if gt_val is not None and pred_val is not None:
                y_true[dim].append(float(gt_val))
                y_pred[dim].append(float(pred_val))

        y_true_decisions.append(gt.get("decision", "reject"))
        y_pred_decisions.append(pred_scores.get("decision", "reject"))

    logger.info(f"Evaluated {len(y_true['rating'])} samples, skipped {skipped}")

    if not y_true["rating"]:
        return {"error": "No valid predictions to evaluate"}

    metrics = compute_all_metrics(y_true, y_pred, y_true_decisions, y_pred_decisions)
    metrics["n_samples"] = len(y_true["rating"])
    metrics["n_skipped"] = skipped

    return metrics


def run_official_evaluation(
    raw_data_path: str = "data/raw/sample.json",
    output_dir: str = "experiments/deepreview_baseline",
) -> dict[str, Any]:
    """Run full official evaluation on all prediction modes.

    Replicates evalate.py main() logic:
    - Evaluate fast mode
    - Evaluate standard mode
    - Evaluate best mode
    - Output markdown tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and adapt data
    data = load_and_adapt(raw_data_path)

    results = {}
    for mode in ["fast", "standard", "best"]:
        pred_field = f"pred_{mode}_mode"
        logger.info(f"Evaluating {mode} mode...")

        metrics = evaluate_predictions(data, pred_field)
        results[mode] = metrics

        # Save metrics
        mode_file = output_dir / f"metrics_{mode}.json"
        with open(mode_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        # Print markdown table
        table = format_metrics_table(metrics, f"DeepReviewer {mode} (n={metrics.get('n_samples', 0)})")
        print(f"\n{table}\n")

        md_file = output_dir / f"results_{mode}.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(table)

    # Save combined results
    combined_file = output_dir / "metrics_all.json"
    with open(combined_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"All results saved to {output_dir}")
    return results


if __name__ == "__main__":
    import sys

    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sample.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "experiments/deepreview_baseline"

    run_official_evaluation(raw_path, out_dir)
