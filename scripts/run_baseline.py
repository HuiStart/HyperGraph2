"""
Run baseline scoring on DeepReview data.

Usage:
    python scripts/run_baseline.py --mode fast --samples 10
    python scripts/run_baseline.py --mode standard --samples 10
    python scripts/run_baseline.py --mode best --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adapters.deepreview_adapter import load_and_adapt
from src.evaluation.official_eval import evaluate_predictions
from src.evaluation.metrics import format_metrics_table
from src.scoring.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(description="Run DeepReview baseline scoring")
    # -m / --mode
    parser.add_argument("-m", "--mode", choices=["fast", "standard", "best"], default="fast",
                        help="Baseline mode to run")
    # -i / --input
    parser.add_argument("-i", "--input", default="data/processed/deepreview_processed.json",
                        help="Input processed data path")
    # -o / --output
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for predictions")
    # -s / --samples
    parser.add_argument("-s", "--samples", type=int, default=None,
                        help="Max number of samples to score")
    # -c / --llm-config (通常配置文件习惯用 -c)
    parser.add_argument("-c", "--llm-config", default="configs/llm.yaml",
                        help="LLM config path")
    # -e / --evaluate
    parser.add_argument("-e", "--evaluate", action="store_true",
                        help="Also run evaluation after scoring")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.samples:
        samples = samples[:args.samples]

    # Set default output path
    output_path = args.output or f"experiments/baseline_{args.mode}.json"

    # Run baseline
    results = run_baseline(
        samples=samples,
        mode=args.mode,
        output_path=output_path,
        llm_config=args.llm_config,
    )

    print(f"\nBaseline ({args.mode}) completed.")
    print(f"Predictions saved to: {output_path}")

    # Optionally evaluate
    if args.evaluate:
        # Add predictions back to samples for evaluation
        for i, sample in enumerate(samples):
            if i < len(results):
                sample["pred_scores"] = results[i].get("scores", {})

        # Simple evaluation (without official pred_field)
        print("\nEvaluation not available for newly generated predictions.")
        print("Use 'python src/cli/main.py evaluate --official' for built-in predictions.")


if __name__ == "__main__":
    main()
