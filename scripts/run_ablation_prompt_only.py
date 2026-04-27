"""
Run Prompt-Only ablation.

This is the minimal version: NO evidence, NO hypergraph, NO multi-agent, NO arbitration.
Only a single LLM call with a carefully crafted prompt.

Usage:
    python scripts/run_ablation_prompt_only.py \
        --input data/processed/test_2024_processed.json \
        --output experiments/ablation_prompt_only.json \
        -n 50
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scoring.ablation.prompt_only_scorer import PromptOnlyScorer
from src.evaluation.metrics import compute_all_metrics, format_metrics_table
from src.utils.llm_wrapper import LLMWrapper


def fmt_val(val, default="-"):
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Run Prompt-Only ablation")
    parser.add_argument("--input", "-i", default="data/processed/deepreview_processed.json",
                        help="Input processed data path")
    parser.add_argument("--output", "-o", default="experiments/ablation_prompt_only.json",
                        help="Output path for results")
    parser.add_argument("--samples", "-n", type=int, default=None,
                        help="Max number of samples to score")
    parser.add_argument("--llm-config", "-lc", default="configs/llm.yaml",
                        help="LLM config path")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.samples:
        samples = samples[:args.samples]

    # Initialize scorer
    llm = LLMWrapper(args.llm_config)
    scorer = PromptOnlyScorer(llm)

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
    results = []

    print("=" * 110)
    print(f"{'Sample':<8} {'ID':<15} {'Rating':<12} {'Soundness':<12} {'Presentation':<14} {'Contribution':<14} {'Decision':<16}")
    print("=" * 110)

    for i, sample in enumerate(samples):
        sid = sample.get("id", "")
        gt_scores = sample.get("ground_truth", {})

        result = scorer.score(
            paper_context=sample["paper_context"],
            title=sample.get("title", ""),
        )
        result["sample_id"] = sid
        result["title"] = sample.get("title", "")
        results.append(result)
        pred_scores = result.get("scores", {})

        # Debug: print raw output for first 2 samples to verify LLM response format
        if i < 2:
            print(f"\n[DEBUG] Sample {sid} raw output (first 800 chars):")
            print(result.get("raw_output", "")[:800])
            print("[DEBUG END]\n")

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

    # Save predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Final metrics
    if not y_true["rating"]:
        print("No valid samples to evaluate.")
        return

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_true_decisions=decisions_true,
        y_pred_decisions=decisions_pred,
    )

    print("\n" + "=" * 60)
    print("Final Metrics (Prompt-Only Ablation)")
    print("=" * 60)
    print(format_metrics_table(metrics))


if __name__ == "__main__":
    main()
