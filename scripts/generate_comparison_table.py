"""
Generate a side-by-side comparison table of true vs predicted scores.

Usage:
    python scripts/generate_comparison_table.py \
        --ground-truth data/processed/test_2024_processed.json \
        --predictions experiments/baseline_fast_2024.json \
        --output results_table.md

Output format (markdown table):
| Sample | ID | Rating (true/pred) | Soundness (true/pred) | ... | Decision (true/pred) |
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_val(val, default="-"):
    if val is None:
        return default
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main():
    parser = argparse.ArgumentParser(description="Generate true vs pred comparison table")
    parser.add_argument("-g", "--ground-truth", required=True,
                        help="Path to processed data JSON (contains ground_truth)")
    parser.add_argument("-p", "--predictions", required=True,
                        help="Path to predictions JSON")
    parser.add_argument("-o", "--output", default="results_table.md",
                        help="Output markdown file path")
    parser.add_argument("-n", "--max-samples", type=int, default=None,
                        help="Max rows to output")
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

    rows = []
    for i, pred in enumerate(pred_data):
        sid = pred.get("sample_id") or pred.get("id", "")
        gt = gt_by_id.get(sid)
        if not gt:
            continue

        gt_scores = gt.get("ground_truth", {})
        pred_scores = pred.get("scores", {})

        row = {
            "sample": f"{i + 1}/{len(pred_data)}",
            "id": sid,
        }

        for dim_key, dim_name in dimensions:
            gt_val = gt_scores.get(dim_key)
            pred_val = pred_scores.get(dim_key)
            row[dim_name] = f"{format_val(gt_val)} / {format_val(pred_val)}"

        gt_dec = gt_scores.get("decision", "-")
        pred_dec = pred_scores.get("decision", "-")
        row["Decision"] = f"{gt_dec.capitalize() if gt_dec else '-'} / {pred_dec.capitalize() if pred_dec else '-'}"

        rows.append(row)

        if args.max_samples and len(rows) >= args.max_samples:
            break

    # Build markdown table
    headers = ["Sample", "ID"] + [d[1] + " (true/pred)" for d in dimensions] + ["Decision (true/pred)"]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join([" --- " for _ in headers]) + "|")

    for row in rows:
        vals = [row["sample"], row["id"]]
        for _, dim_name in dimensions:
            vals.append(row[dim_name])
        vals.append(row["Decision"])
        lines.append("| " + " | ".join(vals) + " |")

    md = "\n".join(lines)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(md)
    print(f"\nTable saved to {args.output}")


if __name__ == "__main__":
    main()
