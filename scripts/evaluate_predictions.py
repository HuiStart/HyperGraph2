"""
Evaluate generated predictions against CSV ground truth.

Usage:
    # After running baseline/ours:
    python scripts/evaluate_predictions.py \
        --ground-truth data/processed/test_2024_processed.json \
        --predictions experiments/baseline_fast_2024.json

    # Or evaluate our method:
    python scripts/evaluate_predictions.py \
        --ground-truth data/processed/test_2024_processed.json \
        --predictions experiments/ours_2024.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_all_metrics, format_metrics_table


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("-g", "--ground-truth", required=True,
                        help="Path to processed data JSON (contains ground_truth)")
    parser.add_argument("-p", "--predictions", required=True,
                        help="Path to predictions JSON (from baseline or ours)")
    parser.add_argument("-o", "--output", default=None,
                        help="Optional path to save metrics JSON")
    args = parser.parse_args()

    gt_data = load_json(args.ground_truth)
    pred_data = load_json(args.predictions)

    # Build lookup by sample_id
    gt_by_id = {s["id"]: s for s in gt_data}

    dimensions = ["rating", "soundness", "presentation", "contribution"]
    y_true = {d: [] for d in dimensions}
    y_pred = {d: [] for d in dimensions}
    decisions_true = []
    decisions_pred = []
    skipped = 0

    for pred in pred_data:
        sid = pred.get("sample_id") or pred.get("id", "")
        gt = gt_by_id.get(sid)
        if not gt:
            skipped += 1
            continue

        gt_scores = gt.get("ground_truth", {})
        pred_scores = pred.get("scores", {})

        # Check if we have at least rating
        if gt_scores.get("rating") is None or pred_scores.get("rating") is None:
            skipped += 1
            continue

        for dim in dimensions:
            gt_val = gt_scores.get(dim)
            pred_val = pred_scores.get(dim)
            if gt_val is not None and pred_val is not None:
                y_true[dim].append(float(gt_val))
                y_pred[dim].append(float(pred_val))

        decisions_true.append(gt_scores.get("decision", "reject"))
        decisions_pred.append(pred_scores.get("decision", "reject"))

    print(f"Matched samples: {len(y_true['rating'])}, Skipped: {skipped}")

    if not y_true["rating"]:
        print("No valid samples to evaluate.")
        return

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_true_decisions=decisions_true,
        y_pred_decisions=decisions_pred,
    )

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(format_metrics_table(metrics))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
