"""Replay equal-budget decisions from E waveforms and report aggregate overlaps."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from check_bci_precision_attribution import EpochConfig, array_hash, file_hash, load_session, read, require, row_product
from eeg_fixedpoint_audit.core import quantize_input, quantize_weights
from budget_ceiling import compute


def paired(left, right, reference):
    """Four cells comparing the two decision vectors against the same target."""
    a, b = left == reference, right == reference
    return {
        "both_match": int((a & b).sum()), "only_mixed_matches": int((a & ~b).sum()),
        "only_uniform_matches": int((~a & b).sum()), "neither_matches": int((~a & ~b).sum()),
    }


def replay(args):
    rows = read(args.results / "aggregate.json")
    storage = read(args.results / "equal-storage.json")
    bounds = compute(rows, storage)["budgets"]
    masks = read(args.selected)
    require(file_hash(args.selected) == storage["t_selection_sha256"], "Training choices do not match")
    freeze = read(args.freeze / "training_manifest.json")
    require(freeze["freeze_sha256"] == storage["training_freeze"], "Training freeze does not match")
    people = []
    for record in freeze["models"]:
        sid = record["subject"]
        model_path = args.freeze / record["model_path"]
        require(file_hash(model_path) == record["model_sha256"], "Model identity")
        model = read(model_path)
        prior = read(args.freeze / f"evaluation/private-predictions/s{sid:02d}.json")
        batch = load_session(args.raw / f"A{sid:02d}E.mat", sid, "E",
                             config=EpochConfig(**freeze["epoch_config"]),
                             fir_coefficients=freeze["fir_coefficients"],
                             allow_evaluation=True, require_official_counts=True)
        require(batch.metadata["source_sha256"] == prior["metadata"]["source_sha256"], "Raw file identity")
        require(array_hash(batch.X) == prior["test_X_sha256"] and list(batch.trial_ids) == prior["trial_ids"]
                and np.array_equal(batch.y, prior["labels"]), "Evaluation window identity")
        delta = model["quantization"]["input_delta"]
        x = quantize_input(batch.X, delta)[0] * delta
        moment = (x @ x.transpose(0, 2, 1)) / x.shape[2]
        filters = np.asarray(model["filters"])
        weights = {0: filters}
        for bits in range(8, 13):
            integer, scale, _ = quantize_weights(filters, bits)
            weights[bits] = integer * scale[:, None]
        features = {}
        for bits, w in weights.items():
            power = row_product(w, moment, w)
            features[bits] = np.log(np.maximum(power, freeze["config"]["power_floor"]))

        def prediction(values):
            scores = values @ np.asarray(model["lda_coefficient"][0]) + model["lda_intercept"][0]
            return 1 + (scores > 0)

        ref = prediction(features[0])
        for bits, name in [(0, "input16_only"), (8, "joint16x8"), (12, "joint16x12")]:
            require(np.array_equal(prediction(features[bits]), prior["predictions"][name]["prediction"]),
                    "Second-moment decision differs from the saved time-domain result")
        bank = {}
        for mask in range(1, 15):
            high = np.array([bool(mask & (1 << row)) for row in range(4)])
            bank[mask] = prediction(np.where(high, features[12], features[8]))
        for budget in bounds:
            k = budget["upgraded_rows"]
            uniform = prediction(features[8 + k])
            selected = bank[masks[str(sid)][str(k)]["mask"]]
            oracle = next(p for p in budget["subjects"] if p["subject"] == sid)
            for predicted, prefix in ((selected, "train_choice"), (uniform, "uniform")):
                require(int((predicted != ref).sum()) == oracle[prefix + "_disagreements"]
                        and int((predicted == batch.y).sum()) == oracle[prefix + "_correct"],
                        "Equal-storage time-domain reference differs")
            candidates = {m: int((p != ref).sum()) for m, p in bank.items() if m.bit_count() == k}
            winners = [m for m, errors in candidates.items() if errors == min(candidates.values())]
            require(winners == oracle["hindsight_masks"], "Hindsight ties differ from the published counts")
            distances = [int((bank[m] != uniform).sum()) for m in winners]
            people.append({
                "subject": sid, "upgraded_rows": k,
                "mixed_vs_uniform_reference": paired(selected, uniform, ref),
                "mixed_vs_uniform_class_label": paired(selected, uniform, batch.y),
                "hindsight_vs_uniform_decisions_different_min": min(distances),
                "hindsight_vs_uniform_decisions_different_max": max(distances),
            })
    summaries = []
    for k in (1, 2, 3):
        group = [p for p in people if p["upgraded_rows"] == k]
        require(sorted(p["subject"] for p in group) == list(range(1, 10)), "Participant coverage")
        summary = {"upgraded_rows": k, "uniform_bits": 8 + k}
        for field in ("mixed_vs_uniform_reference", "mixed_vs_uniform_class_label"):
            summary[field] = {cell: sum(p[field][cell] for p in group) for cell in group[0][field]}
        for field in ("hindsight_vs_uniform_decisions_different_min", "hindsight_vs_uniform_decisions_different_max"):
            summary[field] = sum(p[field] for p in group)
        summaries.append(summary)
    return {
        "schema": "csp-static-budget-pairs-v1",
        "training_freeze": freeze["freeze_sha256"],
        "paired_protocol": "Compare T-selected mixed and equal-payload uniform decisions on the identical E trials.",
        "hindsight_distance": "Across all E-reference-minimizing static masks, inclusive range of trial decisions differing from uniform.",
        "budgets": summaries, "subjects": people,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "freeze", "selected", "results"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", type=Path)
    mode.add_argument("--check", type=Path)
    args = parser.parse_args()
    report = replay(args)
    if args.check:
        require(report == read(args.check), "Saved paired report differs")
        print("Verified paired decisions in all 27 budget/participant cells.")
    else:
        args.write.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
