"""
Run our enhanced multi-agent scoring method on DeepReview data.

Usage:
    python scripts/run_ours.py --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.workflow import ReviewWorkflow


def main():
    parser = argparse.ArgumentParser(description="Run our enhanced scoring method")
    parser.add_argument("--input", default="data/processed/deepreview_processed.json",
                        help="Input processed data path")
    parser.add_argument("--output", default="experiments/ours_results.json",
                        help="Output path for results")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max number of samples to score")
    parser.add_argument("--llm-config", default="configs/llm.yaml",
                        help="LLM config path")
    parser.add_argument("--use-llm-evidence", action="store_true",
                        help="Use LLM for evidence extraction")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.samples:
        samples = samples[:args.samples]

    # Initialize workflow
    from src.utils.llm_wrapper import LLMWrapper
    llm = LLMWrapper(args.llm_config)
    workflow = ReviewWorkflow(llm=llm, use_llm_evidence=args.use_llm_evidence)

    # Run workflow
    results = workflow.run_batch(samples, output_path=args.output)

    print(f"\nOur method completed.")
    print(f"Results saved to: {args.output}")

    # Print summary stats
    escalated = sum(1 for r in results if r.get("risk", {}).get("escalate", False))
    print(f"Total samples: {len(results)}")
    print(f"Escalated to human: {escalated}")


if __name__ == "__main__":
    main()
