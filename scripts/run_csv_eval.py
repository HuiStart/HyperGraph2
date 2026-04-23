"""
Direct evaluation of existing CSV predictions (no LLM calls).

Reads data/deepreview/test_2024.csv and test_2025.csv,
extracts predicted scores from outputs column,
compares with ground truth rating/decision,
prints official metrics.

Usage:
    python scripts/run_csv_eval.py --input data/deepreview/test_2024.csv
    python scripts/run_csv_eval.py --input data/deepreview/test_2025.csv
    python scripts/run_csv_eval.py --input data/deepreview/test_2024.csv --output experiments/eval_2024.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.deepreview_adapter import load_csv_and_adapt
from src.evaluation.metrics import compute_all_metrics, format_metrics_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing CSV predictions")
    parser.add_argument("-i", "--input", required=True, help="Path to CSV file (e.g., data/deepreview/test_2024.csv)")
    parser.add_argument("-o", "--output", default=None, help="Optional path to save evaluation results JSON")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    # Load and adapt CSV
    data = load_csv_and_adapt(args.input, max_samples=args.max_samples)
    print(f"Loaded {len(data)} samples from {args.input}")

    # Build prediction arrays
    dimensions = ["rating"]  # CSV only has overall rating ground truth
    y_true = {d: [] for d in dimensions}
    y_pred = {d: [] for d in dimensions}
    decisions_true = []
    decisions_pred = []
    skipped = 0

    for sample in data:
        gt = sample.get("ground_truth", {})
        pred = sample.get("raw_predictions", {}).get("pred_scores", {})

        if gt.get("rating") is None or pred.get("rating") is None:
            skipped += 1
            continue

        y_true["rating"].append(float(gt["rating"]))
        y_pred["rating"].append(float(pred["rating"]))
        decisions_true.append(gt.get("decision", "reject"))
        decisions_pred.append(pred.get("decision", "reject"))

    print(f"Valid samples: {len(y_true['rating'])}, Skipped: {skipped}")

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
