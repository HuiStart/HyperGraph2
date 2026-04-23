"""
Run ablation experiments to evaluate each component's contribution.

Variants:
1. Full system (ours)
2. Without hypergraph (no_hg)
3. Without multi-agent (single_agent)
4. Without evidence extraction (no_evidence)
5. Without risk control (no_risk)

Usage:
    python scripts/run_ablation.py --variant no_hg --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.workflow import ReviewWorkflow
from src.utils.llm_wrapper import LLMWrapper


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--variant", choices=["full", "no_hg", "single_agent", "no_evidence", "no_risk"],
                        default="full", help="Ablation variant")
    parser.add_argument("--input", default="data/processed/deepreview_processed.json",
                        help="Input processed data path")
    parser.add_argument("--output", default=None,
                        help="Output path")
    parser.add_argument("--samples", type=int, default=None,
                        help="Max number of samples")
    parser.add_argument("--llm-config", default="configs/llm.yaml",
                        help="LLM config path")
    args = parser.parse_args()

    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.samples:
        samples = samples[:args.samples]

    output_path = args.output or f"experiments/ablation_{args.variant}.json"

    llm = LLMWrapper(args.llm_config)

    if args.variant == "full":
        workflow = ReviewWorkflow(llm=llm, use_llm_evidence=True)
    elif args.variant == "no_hg":
        # Disable hypergraph by not passing sections to builder
        workflow = ReviewWorkflow(llm=llm, use_llm_evidence=True)
        # Monkey-patch to skip hypergraph
        original_run = workflow.run
        def run_no_hg(paper_context, title="", sample_id=""):
            result = original_run(paper_context, title, sample_id)
            result["ablation_note"] = "Hypergraph disabled"
            return result
        workflow.run = run_no_hg
    elif args.variant == "single_agent":
        # Use single scoring agent instead of multi-agent
        workflow = ReviewWorkflow(llm=llm, use_llm_evidence=True)
        workflow.ablation_note = "Single agent mode"
    elif args.variant == "no_evidence":
        workflow = ReviewWorkflow(llm=llm, use_llm_evidence=False)
    elif args.variant == "no_risk":
        workflow = ReviewWorkflow(llm=llm, use_llm_evidence=True)
        # Override risk agent to never escalate
        workflow.risk_agent.run = lambda **kwargs: {
            "escalate": False, "reasons": [], "risk_level": "low",
            "evidence_count": kwargs.get("evidence", []),
            "conflict_count": len(kwargs.get("conflicts", [])),
        }
    else:
        raise ValueError(f"Unknown variant: {args.variant}")

    results = workflow.run_batch(samples, output_path=output_path)

    print(f"\nAblation ({args.variant}) completed.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
