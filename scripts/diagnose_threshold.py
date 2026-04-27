"""Diagnose the relationship between rating and decision in ground truth data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/processed/test_2024_processed.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    ratings_accept = []
    ratings_reject = []

    for s in samples:
        gt = s.get("ground_truth", {})
        rating = gt.get("rating")
        decision = gt.get("decision", "").lower()
        if rating is None:
            continue
        if decision == "accept":
            ratings_accept.append(rating)
        elif decision == "reject":
            ratings_reject.append(rating)

    print(f"Total samples: {len(ratings_accept) + len(ratings_reject)}")
    print(f"Accept: {len(ratings_accept)}, Reject: {len(ratings_reject)}")
    print()

    if ratings_accept:
        print(f"Accept ratings -> min: {min(ratings_accept):.2f}, max: {max(ratings_accept):.2f}, mean: {sum(ratings_accept)/len(ratings_accept):.2f}")
    if ratings_reject:
        print(f"Reject ratings -> min: {min(ratings_reject):.2f}, max: {max(ratings_reject):.2f}, mean: {sum(ratings_reject)/len(ratings_reject):.2f}")
    print()

    # Find optimal threshold by checking all possible thresholds at 0.05 step
    all_ratings = ratings_accept + ratings_reject
    if not all_ratings:
        print("No valid data.")
        return

    best_acc = 0
    best_thresh = 0
    for thresh in [x * 0.05 for x in range(0, 201)]:
        correct = sum(1 for r in ratings_accept if r >= thresh) + sum(1 for r in ratings_reject if r < thresh)
        acc = correct / len(all_ratings)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    print(f"Optimal threshold for rating -> decision: {best_thresh:.2f} (accuracy: {best_acc:.4f})")
    print()

    # Distribution histogram
    print("Rating distribution by decision:")
    bins = [(0, 3), (3, 4), (4, 5), (5, 5.5), (5.5, 6), (6, 6.5), (6.5, 7), (7, 8), (8, 10)]
    for lo, hi in bins:
        acc = sum(1 for r in ratings_accept if lo <= r < hi)
        rej = sum(1 for r in ratings_reject if lo <= r < hi)
        total = acc + rej
        if total > 0:
            print(f"  [{lo:.1f}, {hi:.1f}): Accept={acc}, Reject={rej}, Acc%={acc/total*100:.1f}%")


if __name__ == "__main__":
    main()
