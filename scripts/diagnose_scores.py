"""
诊断脚本：分析 ground truth 分布，验证是否存在排名颠倒。

Usage:
    python scripts/diagnose_scores.py \
        --ground-truth data/processed/test_2024_processed.json \
        --predictions experiments/ours_results.json  # optional
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--ground-truth", required=True, help="Processed JSON with ground_truth")
    parser.add_argument("-p", "--predictions", default=None, help="Optional predictions JSON")
    args = parser.parse_args()

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        samples = json.load(f)

    dimensions = ["rating", "soundness", "presentation", "contribution"]

    # 1. Ground truth distribution
    print("=" * 60)
    print("GROUND TRUTH DISTRIBUTION")
    print("=" * 60)
    for dim in dimensions:
        vals = [s["ground_truth"][dim] for s in samples if s["ground_truth"].get(dim) is not None]
        if vals:
            print(f"{dim:12s}: n={len(vals):3d}  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}  "
                  f"min={np.min(vals):.2f}  max={np.max(vals):.2f}  median={np.median(vals):.2f}")

    # 2. Paper length vs ground truth (check if length is a good predictor)
    print("\n" + "=" * 60)
    print("PAPER LENGTH vs GROUND TRUTH (Spearman)")
    print("=" * 60)
    lengths = [len(s.get("paper_context", "")) for s in samples]
    for dim in dimensions:
        vals = [s["ground_truth"][dim] for s in samples if s["ground_truth"].get(dim) is not None]
        # Align lengths with valid gt values
        valid_lengths = [len(s.get("paper_context", "")) for s in samples if s["ground_truth"].get(dim) is not None]
        if len(vals) > 1:
            corr, _ = spearmanr(valid_lengths, vals)
            print(f"{dim:12s}: length_corr = {corr:+.4f}  ({'longer=better' if corr > 0 else 'longer=worse'})")

    # 3. Top/Bottom comparison
    print("\n" + "=" * 60)
    print("TOP vs BOTTOM PAPERS (Rating)")
    print("=" * 60)
    rated_samples = [(s, s["ground_truth"]["rating"]) for s in samples if s["ground_truth"].get("rating") is not None]
    rated_samples.sort(key=lambda x: x[1])
    n = len(rated_samples)
    print(f"Bottom 3 (lowest rated by humans): {[s[0]['id'] for s in rated_samples[:3]]}")
    print(f"Scores: {[s[1] for s in rated_samples[:3]]}")
    print(f"Top 3 (highest rated by humans): {[s[0]['id'] for s in rated_samples[-3:]]}")
    print(f"Scores: {[s[1] for s in rated_samples[-3:]]}")

    # 4. Check decision distribution
    decisions = [s["ground_truth"]["decision"] for s in samples if s["ground_truth"].get("decision")]
    n_accept = sum(1 for d in decisions if "accept" in d.lower())
    print(f"\nDecision distribution: Accept={n_accept}/{len(decisions)} ({100*n_accept/len(decisions):.1f}%)  "
          f"Reject={len(decisions)-n_accept}/{len(decisions)} ({100*(len(decisions)-n_accept)/len(decisions):.1f}%)")

    # 5. If predictions provided, compute true vs pred Spearman
    if args.predictions:
        print("\n" + "=" * 60)
        print("PREDICTIONS vs GROUND TRUTH")
        print("=" * 60)
        with open(args.predictions, "r", encoding="utf-8") as f:
            preds = json.load(f)

        # Map by sample_id
        pred_map = {p.get("sample_id", ""): p.get("scores", {}) for p in preds}

        for dim in dimensions:
            y_true, y_pred = [], []
            for s in samples:
                sid = s.get("id", "")
                gt_val = s["ground_truth"].get(dim)
                pred_val = pred_map.get(sid, {}).get(dim)
                if gt_val is not None and pred_val is not None:
                    y_true.append(float(gt_val))
                    y_pred.append(float(pred_val))

            if len(y_true) > 1:
                mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
                corr, _ = spearmanr(y_true, y_pred)
                print(f"{dim:12s}: n={len(y_true):3d}  MAE={mae:.2f}  Spearman={corr:+.4f}  "
                      f"({'OK' if corr > 0.3 else 'WEAK' if corr > 0 else 'NEGATIVE - RANKING INVERTED!'})")

                # Show top 3 disagreements
                diffs = [(i, abs(y_true[i] - y_pred[i])) for i in range(len(y_true))]
                diffs.sort(key=lambda x: x[1], reverse=True)
                print(f"  Largest errors (idx): {diffs[:3]}")
            else:
                print(f"{dim:12s}: insufficient data")


if __name__ == "__main__":
    main()
