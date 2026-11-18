"""Exact retrospective bound for a fixed, per-person W8/W12 allocation.

Each person gets one mask for the entire evaluation day. Only stored counts
are needed to enumerate all masks with the prescribed number of W12 rows.
"""

import argparse
import json
from pathlib import Path


def best_static_mixed(disagreements, correct, upgraded):
    """Return the complete set of E minimizers without using true labels to break ties."""
    if upgraded not in (1, 2, 3) or len(disagreements) != 16 or len(correct) != 16:
        raise ValueError("Expected four projection rows and 1, 2 or 3 upgraded rows")
    candidates = [mask for mask in range(16) if mask.bit_count() == upgraded]
    best = min(disagreements[mask] for mask in candidates)
    winners = [mask for mask in candidates if disagreements[mask] == best]
    correct_at_best = [correct[mask] for mask in winners]
    return {
        "disagreements": best,
        "tied_masks": winners,
        "correct_min": min(correct_at_best),
        "correct_max": max(correct_at_best),
    }


def compute(rows_report, storage_report):
    if rows_report["training_freeze"] != storage_report["training_freeze"]:
        raise ValueError("Source reports come from different training freezes")
    subjects = rows_report["subjects"]
    if sorted(subject["subject"] for subject in subjects) != list(range(1, 10)):
        raise ValueError("Expected nine distinct subjects")
    configs = storage_report["per_subject"]
    by_key = {(r["subject"], r["config"]): r for r in configs}
    if len(by_key) != len(configs):
        raise ValueError("Duplicate per-person configuration")
    budgets = []
    for k in (1, 2, 3):
        people = []
        for person in subjects:
            sid = person["subject"]
            masks = person["aggregates"]["X16_W8_to_W12"]
            d = masks["reference_disagreements_by_mask"]
            correct = masks["correct_by_mask"]
            best = best_static_mixed(d, correct, k)
            train = by_key[sid, f"mixed{k}"]
            uniform = by_key[sid, f"uniform{8+k}"]
            if train["trials"] != uniform["trials"] or train["trials"] != 144:
                raise ValueError("Mismatched evaluation trial counts")
            if not any((d[m], correct[m]) == (train["input16_disagreements"], train["correct"])
                       for m in range(16) if m.bit_count() == k):
                raise ValueError("Training-day choice is absent from allowed E outcomes")
            people.append({
                "subject": sid, "trials": train["trials"],
                "train_choice_disagreements": train["input16_disagreements"],
                "uniform_disagreements": uniform["input16_disagreements"],
                "train_choice_correct": train["correct"], "uniform_correct": uniform["correct"],
                "hindsight_disagreements": best["disagreements"],
                "hindsight_masks": best["tied_masks"],
                "hindsight_correct_min": best["correct_min"],
                "hindsight_correct_max": best["correct_max"],
            })
        total_fields = ("trials", "train_choice_disagreements", "uniform_disagreements",
                        "hindsight_disagreements", "train_choice_correct", "uniform_correct",
                        "hindsight_correct_min", "hindsight_correct_max")
        totals = {key: sum(p[key] for p in people) for key in total_fields}
        gains = [p["uniform_disagreements"] - p["hindsight_disagreements"] for p in people]
        budgets.append({
            "upgraded_rows": k, "weight_payload_bytes": 11 * (8 + k),
            "uniform_bits": 8 + k, "totals": totals,
            "hindsight_mixed_vs_uniform_people": {
                "fewer": sum(g > 0 for g in gains), "equal": gains.count(0),
                "more": sum(g < 0 for g in gains),
            },
            "subjects": people,
        })
    return {
        "schema": "csp-static-budget-hindsight-v1",
        "training_freeze": rows_report["training_freeze"],
        "reference": "Fixed 16-bit input with floating-point projection weights",
        "selection_scope": "One fixed mask per person over all 144 already-observed E trials",
        "objective": "Fewest E decisions differing from the reference, independent of true labels",
        "ties": "All minimizing masks retained. Correct-count endpoints describe the full tie set.",
        "budgets": budgets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Directory containing aggregate.json and equal-storage.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path, help="Write aggregate output here")
    mode.add_argument("--check", type=Path, help="Recompute and compare with saved aggregate output")
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    result = compute(read(args.folder / "aggregate.json"), read(args.folder / "equal-storage.json"))
    if args.check is not None:
        if result != read(args.check):
            raise SystemExit("Recomputed ceiling differs from the saved report")
        print("Verified all 27 participant/budget rows and three aggregate bounds.")
    else:
        args.write.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
