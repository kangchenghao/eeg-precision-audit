"""Private train/freeze/evaluate entry point; no data is downloaded implicitly.

Train can run alone so the training-prefix RTL checks can finish before any E
signal/label is read.  Evaluation verifies all nine models, executed sources,
contracts and the numerical environment against the stored training manifest.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from eeg_fixedpoint_audit.data import EpochConfig, design_fir, load_session
from eeg_fixedpoint_audit.pipeline import (
    SubjectModel, VARIANTS, aggregate_rows, canonical_hash, evaluate_subject,
    fit_subject, pipeline_config, training_kernel_fixtures,
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prior evidence is preserved; a changed implementation needs a new output.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment():
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "mne")},
    }


def source_manifest(extra_sources=()):
    extensions = {".py", ".v", ".sv", ".tcl", ".ps1", ".toml"}
    named_inputs = {"CONTRACT.md", "requirements-win-py313.lock", "data_sources.json"}
    files = []
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if path.is_file() and (path.suffix in extensions or path.name in named_inputs) and not any(
                part in {"__pycache__", ".git", ".venv", "build", "results"} for part in path.relative_to(PACKAGE_ROOT).parts):
            files.append({"path": path.relative_to(PACKAGE_ROOT).as_posix(), "sha256": digest(path)})
    extra = []
    for raw_path in extra_sources:
        path = Path(raw_path).resolve(strict=True)
        extra.append({"path": str(path), "sha256": digest(path)})
    return {"package": files, "extra_execution_sources": extra}


def contract_manifest(paths):
    if not paths:
        raise ValueError("Pass the approved contract and amendment with --contract")
    result = []
    for raw_path in paths:
        path = Path(raw_path).resolve(strict=True)
        result.append({"path": str(path), "sha256": digest(path)})
    return result


def _subjects(text):
    try:
        subjects = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Subjects must be comma-separated integers") from error
    if not subjects or len(set(subjects)) != len(subjects) or any(not 1 <= item <= 9 for item in subjects):
        raise argparse.ArgumentTypeError("Select unique subjects in 1..9")
    return subjects


def train(raw_dir, private_dir, subjects, contracts, extra_sources=()):
    private_dir = Path(private_dir).resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    if (private_dir / "training_manifest.json").exists():
        raise FileExistsError("Training manifest already exists; use a new private output directory")
    sources_before = source_manifest(extra_sources)
    contract_records = contract_manifest(contracts)
    epoch_config = EpochConfig()
    coefficients = design_fir(epoch_config)
    records = []
    started = time.monotonic()
    for subject in subjects:
        source = Path(raw_dir) / f"A{subject:02d}T.mat"
        batch = load_session(source, subject, "T", config=epoch_config,
                             fir_coefficients=coefficients, require_official_counts=True)
        model = fit_subject(batch.X, batch.y, subject, training_metadata={
            **batch.metadata, "trial_ids": list(batch.trial_ids),
            "artifact_mask": batch.artifact_mask.tolist(),
            "event_metadata": list(batch.event_metadata),
            "channel_names": list(batch.channel_names),
        })
        model_path = private_dir / "models" / f"s{subject:02d}.json"
        _json(model_path, model.to_dict())
        fixture_path = private_dir / "rtl-fixtures" / f"s{subject:02d}.npz"
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        with fixture_path.open("xb") as stream:
            np.savez_compressed(stream, **training_kernel_fixtures(model, batch.X, batch.trial_ids))
        records.append({
            "subject": subject, "source_filename": source.name,
            "source_sha256": batch.metadata["source_sha256"],
            "model_path": model_path.relative_to(private_dir).as_posix(),
            "model_sha256": digest(model_path), "model_numeric_sha256": model.numeric_sha256,
            "fixture_path": fixture_path.relative_to(private_dir).as_posix(),
            "fixture_sha256": digest(fixture_path),
            "training_trials": len(batch.X), "RTL_selected_trial_ids": list(batch.trial_ids[:2]),
            "RTL_dots_per_weight_width": 2 * 4 * 32,
        })
        print(json.dumps({"phase": "train", "subject": subject, "trials": len(batch.X),
                          "numeric_sha256": model.numeric_sha256}), flush=True)
    if source_manifest(extra_sources) != sources_before:
        raise RuntimeError("Execution source changed during training; preserve this run and start a new output")
    manifest = {
        "schema": "eeg-fixedpoint-training-freeze-v1", "state": "training_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "subjects": list(subjects), "all_nine_training_models_complete": set(subjects) == set(range(1, 10)),
        "evaluation_data_read_by_training": False, "config": pipeline_config(),
        "epoch_config": {
            "fs": epoch_config.fs, "numtaps": epoch_config.numtaps,
            "bandpass_hz": list(epoch_config.bandpass_hz), "window": epoch_config.window,
            "start_sample": epoch_config.start_sample, "stop_sample": epoch_config.stop_sample,
            "classes": list(epoch_config.classes),
        },
        "fir_coefficients": coefficients.tolist(), "environment": environment(),
        "execution_sources": sources_before, "contracts": contract_records, "models": records,
        "rtl_fixtures": [{"name": Path(record["fixture_path"]).name,
                          "path": record["fixture_path"], "sha256": record["fixture_sha256"]}
                         for record in records],
        "RTL_validation_complete": False,
        "note": "Training-prefix fixtures are ready; real RTL validation must be recorded separately before evaluation.",
    }
    manifest["freeze_sha256"] = canonical_hash(manifest)
    _json(private_dir / "training_manifest.json", manifest)
    return manifest


def verify_freeze(private_dir):
    private_dir = Path(private_dir).resolve()
    manifest = json.loads((private_dir / "training_manifest.json").read_text(encoding="utf-8"))
    saved_hash = manifest.pop("freeze_sha256")
    if canonical_hash(manifest) != saved_hash:
        raise ValueError("Training freeze manifest hash mismatch")
    manifest["freeze_sha256"] = saved_hash
    if manifest.get("schema") != "eeg-fixedpoint-training-freeze-v1" or manifest.get("config") != pipeline_config():
        raise ValueError("Frozen pipeline config differs from the executable pipeline")
    if set(manifest["subjects"]) != set(range(1, 10)) or not manifest["all_nine_training_models_complete"]:
        raise ValueError("All nine T models must be frozen before reading any E session")
    extra = [item["path"] for item in manifest["execution_sources"]["extra_execution_sources"]]
    if source_manifest(extra) != manifest["execution_sources"]:
        raise ValueError("Execution source changed after training; do not silently reuse the old freeze")
    if contract_manifest([item["path"] for item in manifest["contracts"]]) != manifest["contracts"]:
        raise ValueError("Contract evidence changed after training")
    current_environment = environment()
    if current_environment["packages"] != manifest["environment"]["packages"] or current_environment["python"] != manifest["environment"]["python"]:
        raise ValueError("Numerical package/Python versions changed after training")
    for record in manifest["models"]:
        for key, expected in (("model_path", "model_sha256"), ("fixture_path", "fixture_sha256")):
            if digest(private_dir / record[key]) != record[expected]:
                raise ValueError(f"Frozen artifact hash mismatch: {record[key]}")
        model = SubjectModel.from_dict(json.loads((private_dir / record["model_path"]).read_text(encoding="utf-8")))
        if model.numeric_sha256 != record["model_numeric_sha256"]:
            raise ValueError("Frozen numeric model digest mismatch")
    return manifest


def verify_rtl_evidence(report_path, manifest, private_dir):
    """Reject stale, incomplete or unrelated RTL reports before E is loaded."""
    if report_path is None:
        raise ValueError("A completed RTL evidence JSON is required before E evaluation")
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError("RTL evidence status is not passed")
    for key, relative in (
        ("source_sha256", "rtl/dot22.v"), ("testbench_sha256", "rtl/tb_dot22.sv"),
        ("runner_sha256", "scripts/verify_rtl.py"),
    ):
        if report.get(key) != digest(PACKAGE_ROOT / relative):
            raise ValueError(f"RTL evidence source mismatch: {relative}")
    runs = report.get("runs", [])
    if len(runs) != 4 or {run.get("weight_bits") for run in runs} != {4, 8, 12, 16}:
        raise ValueError("RTL evidence must contain each of the four widths exactly once")
    expected_fixtures = {entry["name"]: entry["sha256"] for entry in manifest["rtl_fixtures"]}
    if len(expected_fixtures) != 9:
        raise ValueError("The training freeze must bind nine distinct T fixtures")
    for entry in manifest["rtl_fixtures"]:
        if digest(Path(private_dir) / entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen training fixture has changed")
    for run in runs:
        real = run.get("real")
        if not isinstance(real, dict) or real.get("source_session") != "T" or real.get("dots") != 2304 or real.get("partial_checks") != 2304 * 22:
            raise ValueError("RTL evidence lacks the complete frozen T-prefix checks")
        fixtures = real.get("fixtures", [])
        if len(fixtures) != 9 or {item.get("name"): item.get("sha256") for item in fixtures} != expected_fixtures:
            raise ValueError("RTL evidence fixtures differ from this training freeze")
        if run.get("accumulator_bits") != 20 + run["weight_bits"]:
            raise ValueError("RTL evidence accumulator width differs from the contract")
        if run.get("synthetic_dots", 0) <= 0 or run.get("total_dots") != 2304 + run["synthetic_dots"]:
            raise ValueError("RTL evidence synthetic/real dot counts are incomplete")
        if any(run.get(stage, {}).get("exit_code") != 0 for stage in ("compile", "run")):
            raise ValueError("RTL evidence contains an unsuccessful positive run")
    required_mutants = {
        "missing_product", "21_terms", "23_terms", "zero_extension", "narrow_accumulator",
        "reset_keeps_count", "gap_advances_count", "wrong_streamed_coefficient", "wrong_weight_index",
    }
    mutants = report.get("mutants", [])
    if len(mutants) != len(required_mutants) or {item.get("name") for item in mutants} != required_mutants:
        raise ValueError("RTL evidence is missing a required deliberate-fault check")
    if any(item.get("compiled") is not True or item.get("detected") is not True for item in mutants):
        raise ValueError("A deliberate RTL fault did not compile or escape was not detected")
    return {"path": str(report_path), "sha256": digest(report_path), "status": "passed",
            "T_dots_per_width": 2304, "mutants_detected": len(mutants)}


def verify_test_evidence(report_path, manifest):
    """Bind successful synthetic tests to the exact frozen execution sources."""
    if report_path is None:
        raise ValueError("A completed synthetic-test evidence JSON is required before E evaluation")
    report_path = Path(report_path).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("tests_run", 0) < 46:
        raise ValueError("Synthetic-test evidence is not a complete passing suite")
    if report.get("failures") != 0 or report.get("errors") != 0 or report.get("skipped", 0) != 0:
        raise ValueError("Synthetic-test evidence contains failed, errored or skipped checks")
    if report.get("source_stable") is not True:
        raise ValueError("Execution sources were not stable throughout the synthetic tests")
    expected = {entry["path"]: entry["sha256"] for entry in manifest["execution_sources"]["package"]}
    if report.get("source_hashes") != expected:
        raise ValueError("Synthetic tests were not run against the frozen source/contract/lock set")
    return {"path": str(report_path), "sha256": digest(report_path), "status": "passed",
            "tests_run": report["tests_run"]}


def evaluate(raw_dir, private_dir, *, rtl_evidence=None, test_evidence=None):
    """Read E only after the complete training freeze and explicit RTL evidence."""
    private_dir = Path(private_dir).resolve()
    manifest = verify_freeze(private_dir)
    evidence = verify_rtl_evidence(rtl_evidence, manifest, private_dir)
    tests = verify_test_evidence(test_evidence, manifest)
    output = private_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=False)
    epoch_config = EpochConfig()
    rows, evaluated = [], []
    started = time.monotonic()
    for record in manifest["models"]:
        subject = record["subject"]
        # Frozen models are loaded before this explicitly authorized E reader.
        model = SubjectModel.from_dict(json.loads((private_dir / record["model_path"]).read_text(encoding="utf-8")))
        batch = load_session(Path(raw_dir) / f"A{subject:02d}E.mat", subject, "E",
                             config=epoch_config, fir_coefficients=manifest["fir_coefficients"],
                             allow_evaluation=True, require_official_counts=True)
        result = evaluate_subject(model, batch.X, batch.y, batch.trial_ids, batch.artifact_mask)
        _json(output / "private-predictions" / f"s{subject:02d}.json", {
            "schema": "private-subject-evaluation-v1", "metadata": batch.metadata,
            "event_metadata": list(batch.event_metadata), **result["details"],
        })
        rows.extend(result["rows"])
        evaluated.append({"subject": subject, "source_filename": batch.metadata["source_filename"],
                          "source_sha256": batch.metadata["source_sha256"], "trials": len(batch.X)})
        print(json.dumps({"phase": "evaluate", "subject": subject, "rows": len(result["rows"])}), flush=True)
    if len(rows) != 90 or any(sum(row["variant"] == variant for row in rows) != 9 for variant in VARIANTS):
        raise ArithmeticError("The fixed 9-subject, 10-variant matrix is incomplete")
    verify_freeze(private_dir)  # Also detect a source mutation during evaluation.
    result = {
        "schema": "eeg-fixedpoint-evaluation-v1", "experiment_complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
        "training_freeze_sha256": manifest["freeze_sha256"], "RTL_evidence": evidence,
        "synthetic_test_evidence": tests,
        "config": pipeline_config(), "evaluation_sources": evaluated,
        "row_count": len(rows), "rows": rows, "aggregate": aggregate_rows(rows),
        "limitations": [
            "Only the CSP projection is integer; filtering, power, log and LDA remain floating point.",
            "T-prefix RTL sampling is separate from complete E-session software arithmetic verification.",
            "Cued left/right task evaluation is not a clinical or pure neural-intent validation.",
            "No best bit width is selected from E-session performance.",
        ],
    }
    _json(output / "results.json", result)
    with (output / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = ["subject_id", "variant", "balanced_accuracy", "delta_balanced_accuracy", "flip_count",
                   "correct_to_wrong", "wrong_to_correct", "float_wrong_unchanged", "input_clipped_count",
                   "weight_clipped_count", "projection_rmse", "projection_max_abs", "feature_max_abs", "score_max_abs", "floor_hits"]
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in columns if key in row},
                "projection_rmse": row["projection_error"]["rmse"],
                "projection_max_abs": row["projection_error"]["max_abs"],
                "feature_max_abs": row["feature_error"]["max_abs"],
                "score_max_abs": row["score_error"]["max_abs"], "floor_hits": row["power"]["floor_hits"],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "evaluate", "full"), default="train")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Private folder containing original A01T.mat etc.")
    parser.add_argument("--output", type=Path, required=True, help="New private evidence folder, outside the public source tree")
    parser.add_argument("--subjects", type=_subjects, default=tuple(range(1, 10)))
    parser.add_argument("--contract", action="append", default=[], help="Exact approved proposal/amendment/gate file; repeat as needed")
    parser.add_argument("--extra-source", action="append", default=[], help="Executed backend source outside this package; repeat as needed")
    parser.add_argument("--rtl-evidence", type=Path, help="Completed test/RTL validation record required before E evaluation")
    parser.add_argument("--test-evidence", type=Path, help="Passing synthetic-test report bound to the frozen source set")
    args = parser.parse_args()
    output = args.output.resolve()
    # Public source and private processed EEG must remain separate.
    if output == PACKAGE_ROOT or PACKAGE_ROOT in output.parents:
        parser.error("Use a private --output outside the publishable package")
    try:
        if args.phase in ("train", "full"):
            train(args.raw_dir, output, args.subjects, args.contract, args.extra_source)
        if args.phase in ("evaluate", "full"):
            if args.rtl_evidence is None or args.test_evidence is None:
                raise ValueError("Run independent tests/RTL validation first and provide --rtl-evidence and --test-evidence")
            evaluate(args.raw_dir, output, rtl_evidence=args.rtl_evidence, test_evidence=args.test_evidence)
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        failure = output / f"failure-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.json"
        _json(failure, {"phase": args.phase, "traceback": traceback.format_exc(),
                        "note": "Prior evidence is preserved; no partial evaluation is a completed 90-row result."})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
