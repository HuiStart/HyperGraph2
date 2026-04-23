"""
Unified evaluation script: per-sample comparison + final metrics.

Usage:
    python scripts/evaluate_full.py \
        --ground-truth data/processed/test_2024_processed.json \
        --predictions experiments/baseline_fast_2024.json \
        --output experiments/full_eval.json

Output:
    1. Real-time per-sample comparison (true vs pred)
    2. Final metrics table (MSE/MAE/Spearman/Decision Acc/Decision F1/Pairwise Acc)
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


def fmt_val(val, default="-"):
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation: per-sample + final metrics")
    parser.add_argument("-g", "--ground-truth", required=True, help="Path to processed data JSON")
    parser.add_argument("-p", "--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("-o", "--output", default=None, help="Path to save metrics JSON")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    gt_data = load_json(args.ground_truth)
    pred_data = load_json(args.predictions)

    gt_by_id = {s["id"]: s for s in gt_data}

    dimensions = [
        ("rating", "Rating"),
        ("soundness", "Soundness"),
        ("presentation", "Presentation"),
        ("contribution", "Contribution"),
    ]

    y_true = {d[0]: [] for d in dimensions}
    y_pred = {d[0]: [] for d in dimensions}
    decisions_true = []
    decisions_pred = []
    skipped = 0
    matched = 0

    print("=" * 100)
    print(f"{'Sample':<8} {'ID':<15} {'Rating':<12} {'Soundness':<12} {'Presentation':<14} {'Contribution':<14} {'Decision':<16}")
    print("=" * 100)

    for i, pred in enumerate(pred_data):
        if args.max_samples and i >= args.max_samples:
            break

        sid = pred.get("sample_id") or pred.get("id", "")
        gt = gt_by_id.get(sid)
        if not gt:
            skipped += 1
            continue

        gt_scores = gt.get("ground_truth", {})
        pred_scores = pred.get("scores", {})

        if gt_scores.get("rating") is None or pred_scores.get("rating") is None:
            skipped += 1
            continue

        matched += 1

        # Build display row
        row_vals = [f"{i+1}/{len(pred_data)}", sid[:12]]
        for dim_key, dim_name in dimensions:
            gt_val = gt_scores.get(dim_key)
            pred_val = pred_scores.get(dim_key)
            row_vals.append(f"{fmt_val(gt_val)}/{fmt_val(pred_val)}")
            if gt_val is not None and pred_val is not None:
                y_true[dim_key].append(float(gt_val))
                y_pred[dim_key].append(float(pred_val))

        gt_dec = gt_scores.get("decision", "-")
        pred_dec = pred_scores.get("decision", "-")
        row_vals.append(f"{gt_dec.capitalize()}/{pred_dec.capitalize()}")
        decisions_true.append(gt_dec)
        decisions_pred.append(pred_dec)

        print(f"{row_vals[0]:<8} {row_vals[1]:<15} {row_vals[2]:<12} {row_vals[3]:<12} {row_vals[4]:<14} {row_vals[5]:<14} {row_vals[6]:<16}")

    print("=" * 100)
    print(f"Matched: {matched}, Skipped: {skipped}")

    if not y_true["rating"]:
        print("No valid samples to evaluate.")
        return

    # Compute metrics
    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_true_decisions=decisions_true,
        y_pred_decisions=decisions_pred,
    )

    print("\n" + "=" * 60)
    print("Final Metrics")
    print("=" * 60)
    print(format_metrics_table(metrics))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
