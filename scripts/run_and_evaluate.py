"""
End-to-end: run LLM predictions and evaluate in real-time.

Usage:
    python scripts/run_and_evaluate.py \
        --input data/processed/test_2024_processed.json \
        --mode fast \
        --output-pred experiments/pred_fast.json \
        --output-metrics experiments/metrics_fast.json \
        --max-samples 5

Flow:
    1. Load processed data (ground truth + paper_context)
    2. For each sample: call LLM -> parse scores -> print true vs pred
    3. After all samples: compute and print final metrics
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import compute_all_metrics, format_metrics_table
from src.scoring.baseline import run_baseline
from src.utils.logger import setup_logger


def fmt_val(val, default="-"):
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Run LLM + evaluate in real-time")
    parser.add_argument("-i", "--input", required=True, help="Processed data JSON path")
    parser.add_argument("-m", "--mode", choices=["fast", "standard", "best"], default="fast",
                        help="Baseline mode")
    parser.add_argument("--llm-config", default="configs/llm.yaml", help="LLM config path")
    parser.add_argument("--output-pred", default=None, help="Path to save predictions JSON")
    parser.add_argument("--output-metrics", default=None, help="Path to save metrics JSON")
    parser.add_argument("-n", "--max-samples", type=int, default=None, help="Max samples")
    args = parser.parse_args()

    setup_logger("deepreview", level="INFO")

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

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

    print("=" * 110)
    print(f"{'Sample':<8} {'ID':<15} {'Rating':<12} {'Soundness':<12} {'Presentation':<14} {'Contribution':<14} {'Decision':<16}")
    print("=" * 110)

    # Run baseline and evaluate sample by sample
    results = run_baseline(
        samples=samples,
        mode=args.mode,
        output_path=args.output_pred,
        llm_config=args.llm_config,
    )

    for i, (sample, result) in enumerate(zip(samples, results)):
        sid = sample.get("id", "")
        gt_scores = sample.get("ground_truth", {})
        pred_scores = result.get("scores", {})

        # Print comparison row
        row_vals = [f"{i+1}/{len(samples)}", sid[:12]]
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

    print("=" * 110)

    if not y_true["rating"]:
        print("No valid samples to evaluate.")
        return

    # Compute final metrics
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

    if args.output_metrics:
        out_path = Path(args.output_metrics)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output_metrics}")


if __name__ == "__main__":
    main()
