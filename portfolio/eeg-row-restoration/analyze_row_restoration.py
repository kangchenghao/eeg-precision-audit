"""Counterfactual CSP row restoration from a frozen four-row power-feature log.

Input stage, subject-specific filters, epoch, log-power rule and LDA parameters
are the same on both sides of every row intervention. Each mixed row reuses
its own endpoint feature because there is no normalization across rows.
Python 3.10+ standard library only. Writes trial records into a private path.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAG = "bci-row-restoration-2026-11-18"
ARCHIVE = ROOT / "build/bci-fixedpoint-pilot-2026-11-15/private-run-02"
# Canonical digest of 20 original file hashes in the 15 November failure audit.
ORIGINAL = "117576469faa9dfabfa1482583e573dc06a574edf37747ec8baf7fa7a46d3636"
MASKS = range(16)
WIDTH_PAIRS = [(4, "fp"), (8, "fp"), (12, "fp"), (16, "fp"), (8, 12), (8, 16), (12, 16)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decision(score):
    return 2 if score > 0 else 1


def evaluate(weights, bias, features):
    return math.fsum([bias, *(w * f for w, f in zip(weights, features, strict=True))])


def group(input_type, width):
    if input_type == "X16":
        return "input16_only" if width == "fp" else f"joint16x{width}"
    return "float" if width == "fp" else f"weight{width}_only"


def key(input_type, low, high):
    return f"{input_type}_W{low}_to_W{high}"


def experiment():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ARCHIVE)
    parser.add_argument("--public", type=Path, default=ROOT / "output" / TAG)
    parser.add_argument("--private", type=Path, default=ROOT / "build" / TAG / "private")
    parser.add_argument("--independent-run", action="store_true",
                        help="Allow a fresh pilot run with different file hashes. Report its identity.")
    args = parser.parse_args()
    if args.public.resolve().is_relative_to(args.private.resolve()) or args.private.resolve().is_relative_to(args.public.resolve()):
        raise ValueError("Private and public destinations must be in separate directory trees")
    if args.private.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise ValueError("Put private outputs outside the shareable code directory")
    for folder in [args.public, args.private]:
        folder.mkdir(parents=True, exist_ok=True)
    manifest = args.run / "training_manifest.json"
    result_path = args.run / "evaluation/results.json"
    names = ["training_manifest.json", "evaluation/results.json"]
    for subject in range(1, 10):
        names.extend([f"models/s{subject:02d}.json", f"evaluation/private-predictions/s{subject:02d}.json"])
    sources = {name: digest(args.run / name) for name in names}
    historical_digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if historical_digest != ORIGINAL and not args.independent_run:
        raise ValueError("Frozen-run files do not match the original ledger")

    pilot = read(result_path)
    freeze = read(manifest)
    if not pilot["experiment_complete"] or freeze["freeze_sha256"] != pilot["training_freeze_sha256"]:
        raise ValueError("Evaluation is incomplete or not bound to the training freeze")
    archived_rows = {(int(row["subject_id"]), row["variant"]): row for row in pilot["rows"]}
    if len(archived_rows) != 90 or len(pilot["rows"]) != 90:
        raise ValueError("Expected 90 unique archived subject/variant rows")
    subset_counts = {}
    scenarios = {}
    overlap = Counter()
    exact_endpoint_labels = 0
    max_residual = 0.0
    subjects = []
    detail_path = args.private / "row-counterfactuals.jsonl"
    with detail_path.open("w", encoding="utf-8", newline="\n") as details:
        for subject in range(1, 10):
            model = read(args.run / f"models/s{subject:02d}.json")
            saved = read(args.run / f"evaluation/private-predictions/s{subject:02d}.json")
            variants = saved["predictions"]
            coef = model["lda_coefficient"][0]
            bias = model["lda_intercept"][0]
            labels = saved["labels"]
            if len(coef) != 4 or model["classes"] != [1, 2] or Counter(labels) != {1: 72, 2: 72}:
                raise ValueError("Unexpected trial balance or classifier")
            if len(set(saved["trial_ids"])) != 144:
                raise ValueError("Trial IDs are missing or duplicated")
            subject_report = {}
            fp = variants["float"]["prediction"]
            for name, row in variants.items():
                archived = archived_rows[subject, name]
                if archived["model_numeric_sha256"] != model["numeric_sha256"]:
                    raise ValueError("Subject model and predictions have different frozen hashes")
                if sum(a != b for a, b in zip(fp, row["prediction"], strict=True)) != archived["flip_count"]:
                    raise ValueError("Archived full-float flip count mismatch")
                if len(row["features"]) != 144 or len(row["score"]) != 144:
                    raise ValueError("Unexpected endpoint feature/score count")
                for features, score, pred in zip(row["features"], row["score"], row["prediction"], strict=True):
                    computed = evaluate(coef, bias, features)
                    if not math.isfinite(computed) or not math.isclose(computed, score, abs_tol=1e-12, rel_tol=1e-12) or decision(computed) != pred:
                        raise ArithmeticError("Reconstructed endpoint differs from archived decision or score")
                    exact_endpoint_labels += 1
                    max_residual = max(max_residual, abs(computed - score))
            for f, lo, hi in zip(fp, variants["joint16x8"]["prediction"], variants["joint16x12"]["prediction"], strict=True):
                overlap[f"W8_{'diff' if lo != f else 'same'}_W12_{'diff' if hi != f else 'same'}"] += 1

            for input_type in ("X16", "Xfloat"):
                fixed_ref = variants[group(input_type, "fp")]["prediction"]
                for low, high in WIDTH_PAIRS:
                    setting = key(input_type, low, high)
                    low_detail = variants[group(input_type, low)]
                    high_detail = variants[group(input_type, high)]
                    counts = subset_counts.setdefault(setting, [Counter() for _ in MASKS])
                    study = scenarios.setdefault(setting, {
                        "minimum_rows_fix_original_disagreement": Counter(),
                        "single_row_fix_options_among_original_disagreements": Counter(),
                        "count": Counter(), "cancellation_all": [], "cancellation_original_disagreements": [],
                    })
                    block = [Counter() for _ in MASKS]
                    for i, truth in enumerate(labels):
                        ref, low_pred, high_pred = fixed_ref[i], low_detail["prediction"][i], high_detail["prediction"][i]
                        low_features, high_features = low_detail["features"][i], high_detail["features"][i]
                        deltas = [w * (hi - lo) for w, lo, hi in zip(coef, low_features, high_features, strict=True)]
                        scores = [evaluate(coef, bias, [high_features[j] if mask & (1 << j) else low_features[j]
                                                        for j in range(4)]) for mask in MASKS]
                        pred = [decision(score) for score in scores]
                        if pred[0] != low_pred or pred[15] != high_pred:
                            raise ArithmeticError("Mixed-subset endpoint no longer matches archive")
                        independent = [low_detail["score"][i] + math.fsum(deltas[j] for j in range(4) if mask & (1 << j))
                                       for mask in MASKS]
                        if not all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) and decision(y) == pred[m]
                                   for m, (x, y) in enumerate(zip(scores, independent, strict=True))):
                            raise ArithmeticError("Direct and additive subset reconstruction disagree")
                        original_disagreement = low_pred != ref
                        for mask, p in enumerate(pred):
                            indicators = {
                                "trials": 1, "vs_fixed_input_Wfp": p != ref, "vs_full_float": p != fp[i],
                                "correct": p == truth, "repair": original_disagreement and p == ref,
                                "induce": not original_disagreement and p != ref,
                                "ref_correct_to_wrong": ref == truth and p != truth,
                                "ref_wrong_to_correct": ref != truth and p == truth,
                            }
                            counts[mask].update(indicators)
                            block[mask].update(indicators)
                        repaired = [m for m in MASKS if pred[m] == ref]
                        min_rows = min((m.bit_count() for m in repaired), default=None)
                        single_fixes = sum(pred[1 << j] == ref for j in range(4))
                        if original_disagreement:
                            study["minimum_rows_fix_original_disagreement"][str(min_rows) if min_rows is not None else "unreachable"] += 1
                            study["single_row_fix_options_among_original_disagreements"][str(single_fixes)] += 1
                        # A cancellation detectable at the decision boundary, distinct from merely opposite score signs.
                        hidden = low_pred == ref and high_pred == ref and any(p != ref for p in pred)
                        worsening_edges = sum(pred[m] == ref and pred[m | (1 << j)] != ref for m in MASKS
                                              for j in range(4) if not m & (1 << j))
                        study["count"].update({
                            "trials": 1, "low_disagrees": original_disagreement,
                            "high_disagrees": high_pred != ref, "both_endpoints_agree_but_intermediate_disagrees": hidden,
                            "restore_more_can_introduce_disagreement_trials": worsening_edges > 0,
                            "restore_more_introduces_disagreement_edges": worsening_edges,
                            "score_deltas_both_signs": min(deltas) < 0 < max(deltas),
                            "original_disagreements_with_both_signs": original_disagreement and min(deltas) < 0 < max(deltas),
                        })
                        l1 = math.fsum(abs(v) for v in deltas)
                        if l1 > 0:
                            cancel = 1 - abs(math.fsum(deltas)) / l1
                            study["cancellation_all"].append(cancel)
                            if original_disagreement:
                                study["cancellation_original_disagreements"].append(cancel)
                        details.write(json.dumps({
                            "subject": subject, "trial": saved["trial_ids"][i], "setting": setting,
                            "label": truth, "fixed_reference": ref, "original_float": fp[i],
                            "delta_per_restored_row": deltas, "scores_by_mask": scores,
                            "predictions_by_mask": pred, "minimum_rows_to_reference": min_rows,
                        }, separators=(",", ":"), allow_nan=False) + "\n")
                    subject_report[setting] = {
                        "reference_disagreements_by_mask": [b["vs_fixed_input_Wfp"] for b in block],
                        "correct_by_mask": [b["correct"] for b in block],
                    }
            subjects.append({"subject": subject, "aggregates": subject_report})

    flat_rows = []
    for setting, counts in subset_counts.items():
        for mask, c in enumerate(counts):
            if c["vs_fixed_input_Wfp"] != counts[0]["vs_fixed_input_Wfp"] - c["repair"] + c["induce"]:
                raise ArithmeticError("Repair/induce bookkeeping failed")
            flat_rows.append({"setting": setting, "mask": mask,
                              "restored_rows": ",".join(str(j + 1) for j in range(4) if mask & (1 << j)),
                              **c, "balanced_accuracy": c["correct"] / c["trials"]})
    for study in scenarios.values():
        for field in ("cancellation_all", "cancellation_original_disagreements"):
            values = study[field]
            study[field] = {"nonzero_L1_trials": len(values), "median": statistics.median(values) if values else None}
    public = args.public
    with (public / "subset-counts.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    dump(public / "aggregate.json", {
        "schema": "csp-row-restoration-v1", "status": "completed", "input_files_digest": historical_digest,
        "matches_2026_11_15_archived_inputs": historical_digest == ORIGINAL,
        "original_archive_frozen_models_and_predictions": sources, "training_freeze": freeze["freeze_sha256"],
        "code_sha256": digest(Path(__file__)), "decisions": "score > 0 -> class 2, otherwise class 1",
        "reference": "Floating weights on same input as both endpoint variants. full_float always uses floating input and weights.",
        "masks": "bit 0 -> first CSP row; set bits use the designated higher precision. m=0 all low, m=15 all high.",
        "scope": "Posthoc, existing E, no retraining. The 16 subsets reuse independent per-row log-mean-square power features.",
        "balanced_accuracy": "Equal numbers of trials for both classes in each subject, equal sizes across subjects.",
        "repair_induce": "Changed relative to low endpoint, agreement judged against same-input floating-weight prediction.",
        "cancellation_index": "1 - |sum(row score deltas)|/sum(|row score deltas|); undefined if denominator zero.",
        "audit": {"endpoint_predictions_checked": exact_endpoint_labels, "max_endpoint_score_residual": max_residual,
                  "subset_scores_checked_two_ways": 1296 * len(scenarios) * 16,
                  "scenario_count": len(scenarios), "subset_rows": len(flat_rows), "private_file_sha256": digest(detail_path)},
        "W8_W12_joint_overlap_vs_full_float": overlap, "scenarios": scenarios, "subjects": subjects,
    })
    print(f"{len(scenarios)} scenarios, {len(flat_rows)} subsets, {exact_endpoint_labels} archived endpoint labels verified")


if __name__ == "__main__":
    experiment()

