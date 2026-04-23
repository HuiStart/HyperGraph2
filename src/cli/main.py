"""
CLI entry point for the DeepReview scoring framework.

Commands:
    preprocess    - Load and adapt DeepReview data
    baseline      - Run baseline scoring (fast/standard/best)
    evaluate      - Run official evaluation on predictions
    evidence      - Run evidence extraction
    ours          - Run our enhanced method
    ablation      - Run ablation experiments
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.adapters.deepreview_adapter import load_and_adapt
from src.evaluation.official_eval import run_official_evaluation, evaluate_predictions
from src.scoring.baseline import run_baseline
from src.utils.logger import setup_logger


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocess DeepReview data."""
    data = load_and_adapt(
        raw_path=args.input,
        output_path=args.output,
        max_samples=args.max_samples,
    )
    print(f"Processed {len(data)} samples -> {args.output}")


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run baseline scoring."""
    # Load data
    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    output_path = args.output or f"experiments/baseline_{args.mode}.json"
    results = run_baseline(
        samples=samples,
        mode=args.mode,
        output_path=output_path,
        llm_config=args.llm_config,
    )
    print(f"Baseline ({args.mode}) completed. Results: {output_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on predictions."""
    if args.official:
        # Run official evaluation on raw sample.json (uses pred_fast/standard/best)
        results = run_official_evaluation(
            raw_data_path=args.input,
            output_dir=args.output_dir,
        )
        print(f"Official evaluation completed. Results saved to {args.output_dir}")
    else:
        # Evaluate custom predictions
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        metrics = evaluate_predictions(data, pred_field=args.pred_field)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_evidence(args: argparse.Namespace) -> None:
    """Run evidence extraction."""
    from src.evidence.extractor import EvidenceExtractor

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    extractor = EvidenceExtractor()
    results = []
    for sample in samples:
        evidence = extractor.extract(sample.get("paper_context", ""), sample.get("title", ""))
        results.append({
            "sample_id": sample.get("id", ""),
            "evidence": evidence,
        })

    output_path = args.output or "experiments/evidence_results.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evidence extraction completed. Results: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeepReview Evidence-Driven Scoring Framework"
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--log-file", default=None, help="Log file path"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess DeepReview data")
    preprocess_parser.add_argument("--input", default="data/raw/sample.json", help="Raw data path")
    preprocess_parser.add_argument("--output", default="data/processed/deepreview_processed.json", help="Output path")
    preprocess_parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    preprocess_parser.set_defaults(func=cmd_preprocess)

    # baseline command
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline scoring")
    baseline_parser.add_argument("--mode", choices=["fast", "standard", "best"], default="fast", help="Baseline mode")
    baseline_parser.add_argument("--input", default="data/processed/deepreview_processed.json", help="Input data path")
    baseline_parser.add_argument("--output", default=None, help="Output path")
    baseline_parser.add_argument("--llm-config", default="configs/llm.yaml", help="LLM config path")
    baseline_parser.add_argument("--max-samples", type=int, default=None, help="Max samples to score")
    baseline_parser.set_defaults(func=cmd_baseline)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--input", default="data/raw/sample.json", help="Input data path")
    eval_parser.add_argument("--official", action="store_true", help="Run official evaluation on built-in predictions")
    eval_parser.add_argument("--pred-field", default="pred_fast_mode", help="Prediction field to evaluate")
    eval_parser.add_argument("--output-dir", default="experiments/deepreview_baseline", help="Output directory")
    eval_parser.set_defaults(func=cmd_evaluate)

    # evidence command
    evidence_parser = subparsers.add_parser("evidence", help="Run evidence extraction")
    evidence_parser.add_argument("--input", default="data/processed/deepreview_processed.json", help="Input data path")
    evidence_parser.add_argument("--output", default="experiments/evidence_results.json", help="Output path")
    evidence_parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    evidence_parser.set_defaults(func=cmd_evidence)

    args = parser.parse_args()

    # Setup logging
    setup_logger("deepreview", level=args.log_level, log_file=args.log_file)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
