"""Replay frozen E sessions and independently audit mixed CSP row precision.

Uses the pilot's signal preparation, frozen models and input scaling. The
calculation after input quantization uses uncentred second moments, independent
of the pilot's channel-wise integer dot followed by time-domain mean square.
Only aggregate evidence leaves memory.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "portfolio/eeg-fixedpoint-audit/src"))
from eeg_fixedpoint_audit.data import EpochConfig, load_session


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(array).data)
    return digest.hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def row_product(left, moment, right):
    """Uncentred cross-power: one value for every trial and projection row."""
    return np.einsum("ic,tcd,id->ti", left, moment, right, optimize=True)


def quantized_weights(model, bits):
    config = model["quantization"]["weight_configs"][str(bits)]
    return np.asarray(config["integers"]) * np.asarray(config["deltas"])[:, None]


def quantiles(values):
    return dict(zip(("min", "median", "max"), map(float, np.quantile(values, [0, .5, 1])), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=ROOT / "build/bci-fixedpoint-pilot-2026-11-15/private-run-02")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/bnci2014_001/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "build/bci-attribution-independent/aggregate.json")
    parser.add_argument("--public-output", type=Path, help="Write compact aggregate checks without per-flip score values")
    parser.add_argument("--subset-counts", type=Path, help="Compare all 48 independently recomputed masks with the primary counts CSV")
    args = parser.parse_args()
    freeze = read(args.frozen_dir / "training_manifest.json")
    require(sorted(row["subject"] for row in freeze["models"]) == list(range(1, 10)), "requires nine unique frozen subjects")
    comparisons = ((8, "float"), (12, "float"), (8, 12))
    summary = {f"{low}_to_{high}": [Counter() for _ in range(16)] for low, high in comparisons}
    mechanism = {str(bits): Counter() for bits in (8, 12)}
    dist = {str(bits): defaultdict(list) for bits in (8, 12)}
    errors = defaultdict(float)
    raw_sources = []
    floor = freeze["config"]["power_floor"]
    for record in freeze["models"]:
        subject = record["subject"]
        model_path = args.frozen_dir / record["model_path"]
        require(file_hash(model_path) == record["model_sha256"], "changed model")
        model = read(model_path)
        stored = read(args.frozen_dir / f"evaluation/private-predictions/s{subject:02d}.json")
        pred = stored["predictions"]
        source = args.raw_dir / f"A{subject:02d}E.mat"
        batch = load_session(source, subject, "E", config=EpochConfig(**freeze["epoch_config"]),
                             fir_coefficients=freeze["fir_coefficients"], allow_evaluation=True,
                             require_official_counts=True)
        require(batch.metadata["source_sha256"] == stored["metadata"]["source_sha256"], "raw MAT identity")
        require(array_hash(batch.X) == stored["test_X_sha256"], "frozen filtered input identity")
        require(list(batch.trial_ids) == stored["trial_ids"] and np.array_equal(batch.y, stored["labels"]), "trial order")
        raw_sources.append({"name": source.name, "sha256": batch.metadata["source_sha256"]})
        scale = model["quantization"]["input_delta"]
        x16 = np.clip(np.rint(batch.X / scale), -32767, 32767) * scale
        moment = (x16 @ x16.transpose(0, 2, 1)) / x16.shape[2]
        rows = {"float": np.asarray(model["filters"]), 8: quantized_weights(model, 8), 12: quantized_weights(model, 12)}
        lda = np.asarray(model["lda_coefficient"][0])
        intercept = model["lda_intercept"][0]
        snap = {"float": pred["input16_only"], 8: pred["joint16x8"], 12: pred["joint16x12"]}
        ref_score = np.asarray(snap["float"]["score"])
        ref_decision = ref_score > 0
        require(np.array_equal(ref_decision, np.asarray(pred["float"]["score"]) > 0), "frozen-input and float decisions differ")
        for low, high in comparisons:
            lo_pred = np.asarray(snap[low]["score"]) > 0
            flipped = lo_pred != ref_decision
            for mask, case in enumerate(summary[f"{low}_to_{high}"]):
                restored = np.array([bool(mask & (1 << index)) for index in range(4)])[:, None]
                mixed = np.where(restored, rows[high], rows[low])
                power = row_product(mixed, moment, mixed)
                require((power > floor).all(), "unexpected power floor")
                features = np.log(power)
                stored_features = np.where(restored.T, snap[high]["features"], snap[low]["features"])
                errors["max_feature_residual"] = max(errors["max_feature_residual"], float(np.abs(features - stored_features).max()))
                require(np.allclose(features, stored_features, rtol=0, atol=1e-10), "raw moments vs stored features")
                score = features @ lda + intercept
                expected = stored_features @ lda + intercept
                errors["max_score_residual"] = max(errors["max_score_residual"], float(np.abs(score - expected).max()))
                require(np.allclose(score, expected, rtol=0, atol=1e-10), "raw moments vs logged feature assembly")
                require(np.array_equal(score > 0, expected > 0), "mixed decision disagreement")
                if mask in (0, 15):
                    endpoint = snap[low if mask == 0 else high]
                    require(np.allclose(score, endpoint["score"], rtol=0, atol=1e-10) and
                            np.array_equal(score > 0, np.asarray(endpoint["prediction"]) == 2), "frozen score endpoint")
                differs = (score > 0) != ref_decision
                case.update(trials=len(score), disagrees=int(differs.sum()), repaired=int((flipped & ~differs).sum()),
                            introduced=int((~flipped & differs).sum()))
        for bits in (8, 12):
            key = str(bits)
            w = rows["float"]
            e = rows[bits] - w
            power = row_product(w, moment, w)
            cross = 2 * row_product(w, moment, e)
            self_power = row_product(e, moment, e)
            ratio = (cross + self_power) / power
            logratio = np.log1p(ratio)
            require(np.isfinite(logratio).all() and (self_power >= 0).all(), "power decomposition domain")
            # Apply the same log secant to both power terms, preserving their sum.
            secant = np.divide(logratio, ratio, out=np.ones_like(ratio), where=ratio != 0)
            cross_score = np.sum(lda * secant * (cross / power), axis=1)
            self_score = np.sum(lda * secant * (self_power / power), axis=1)
            delta = np.asarray(snap[bits]["score"]) - ref_score
            residual = np.abs(delta - (cross_score + self_score))
            require((residual < 1e-10).all(), "cross and residual power closure")
            errors["max_lognonlinearity_decomposition_residual"] = max(errors["max_lognonlinearity_decomposition_residual"],
                                                                        float(residual.max()))
            flip = (np.asarray(snap[bits]["score"]) > 0) != ref_decision
            mixed_sign = (cross_score * self_score) < 0
            crosses_alone = ((ref_score + cross_score > 0) != ref_decision)
            m = mechanism[key]
            m.update(trials=len(delta), flipped=int(flip.sum()), row_trials=int(power.size),
                     abs_cross_power_exceeds_self_power=int((np.abs(cross) > self_power).sum()),
                     cross_score_larger_among_flips=int((flip & (np.abs(cross_score) > np.abs(self_score))).sum()),
                     opposite_score_terms_among_flips=int((flip & mixed_sign).sum()),
                     cross_score_alone_crosses_among_flips=int((flip & crosses_alone).sum()),
                     cross_score_alone_crosses_among_agreements=int((~flip & crosses_alone).sum()))
            dist[key]["abs_cross_score_given_flip"].extend(np.abs(cross_score[flip]).tolist())
            dist[key]["abs_residual_power_score_given_flip"].extend(np.abs(self_score[flip]).tolist())
            dist[key]["abs_reference_margin_given_flip"].extend(np.abs(ref_score[flip]).tolist())
    for settings, cases in summary.items():
        require(len(cases) == 16 and all(case["trials"] == 1296 for case in cases), f"incomplete E coverage: {settings}")
    mechanism_counts = {key: dict(values) for key, values in mechanism.items()}
    checked_csv_rows = 0
    if args.subset_counts:
        with args.subset_counts.open(encoding="utf-8-sig", newline="") as stream:
            lookup = {(row["setting"], int(row["mask"])): row for row in csv.DictReader(stream)}
        for low, high in comparisons:
            target = "fp" if high == "float" else high
            setting = f"X16_W{low}_to_W{target}"
            for mask, case in enumerate(summary[f"{low}_to_{high}"]):
                row = lookup[(setting, mask)]
                required_fields = {"trials": "trials", "disagrees": "vs_fixed_input_Wfp", "repaired": "repair", "introduced": "induce"}
                for ours, primary in required_fields.items():
                    require(case[ours] == int(row[primary]), f"primary CSV mismatch: {setting} mask {mask} {primary}")
                checked_csv_rows += 1
    for key, fields in dist.items():
        for label, values in fields.items():
            mechanism[key][label] = quantiles(values)
    result = {
        "status": "passed", "session": "E (previously observed, post hoc)", "frozen_model_count": len(raw_sources),
        "inputs": raw_sources, "training_freeze_sha256": freeze["freeze_sha256"],
        "independent_formula": "R = X16 X16^T / 750; score = b + sum_j a_j ln(w_j^T R w_j)",
        "reference": "input16_only shares all 1296 decisions with float, while its numerical scores can differ",
        "subset_semantics": "bit j (LSB=row 1) set means replace row j with the high endpoint",
        "evaluated_trial_mask_comparisons": sum(case["trials"] for group in summary.values() for case in group),
        "subsets": summary, "residuals": dict(errors), "mechanism": mechanism, "matching_primary_csv_rows": checked_csv_rows,
        "decomposition": "For e=wq-w: Delta P=2w^TRe+e^TRe. Multiply both terms by the same positive log secant and LDA weight, then sum over rows. Individual summands are descriptive, not alternative implemented signal paths.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.public_output:
        compact = {key: val for key, val in result.items() if key not in ("inputs", "mechanism")}
        compact.update(distinct_E_trials=1296, mechanism=mechanism_counts)
        if args.subset_counts:
            compact["checked_primary_csv_sha256"] = file_hash(args.subset_counts)
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        args.public_output.write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"count": result["evaluated_trial_mask_comparisons"], "residuals": errors, "mechanism": mechanism}))


if __name__ == "__main__":
    main()
