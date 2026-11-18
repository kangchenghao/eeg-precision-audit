"""Compare T-selected W8/W12 rows to uniform widths with equal weight payload."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mixed(features, mask):
    order = 1 << np.arange(4)
    return np.where((order & mask) != 0, features[12], features[8])


def score(model, features):
    return (features @ model.lda_coefficient.T + model.lda_intercept).ravel()


def select_masks(model, features, reference):
    choices = {}
    for count in (1, 2, 3):
        ranks = []
        for mask in range(1, 15):
            if mask.bit_count() == count:
                values = score(model, mixed(features, mask))
                flips = int(np.count_nonzero((values > 0) != (reference > 0)))
                ranks.append((flips, float(np.mean((values - reference) ** 2)), mask))
        flips, mse, mask = min(ranks)
        choices[str(count)] = dict(mask=mask, t_disagreements=flips, t_score_mse=mse)
    return choices


def run(args):
    sys.path.insert(0, str(args.pilot / "src"))
    from eeg_fixedpoint_audit import core, data, pipeline

    manifest = read(args.freeze / "training_manifest.json")
    if manifest["freeze_sha256"] != pipeline.canonical_hash({k: v for k, v in manifest.items() if k != "freeze_sha256"}):
        raise ValueError("Training manifest no longer matches its hash")
    for source in manifest["execution_sources"]["package"]:
        if digest(args.pilot / source["path"]) != source["sha256"]:
            raise ValueError(f"Frozen source changed: {source['path']}")
    sources = {entry["filename"]: entry["sha256"] for entry in read(args.pilot / "data_sources.json")["files"]}
    models = {}
    for record in manifest["models"]:
        path = args.freeze / record["model_path"]
        if digest(path) != record["model_sha256"]:
            raise ValueError("Model file hash mismatch")
        model = pipeline.SubjectModel.from_dict(read(path))
        models[record["subject"]] = model
    if set(models) != set(range(1, 10)):
        raise ValueError("Expected all nine frozen models")

    def calculate(subject, session):
        model = models[subject]
        filename = f"A{subject:02d}{session}.mat"
        batch = data.load_session(args.raw / filename, subject, session,
                                  fir_coefficients=manifest["fir_coefficients"],
                                  require_official_counts=True, allow_evaluation=(session == "E"))
        if batch.metadata["source_sha256"] != sources[filename]:
            raise ValueError("Source MAT hash mismatch")
        qx, _ = core.quantize_input(batch.X, model.quantization.input_delta)
        input16 = qx * model.quantization.input_delta
        reference = score(model, core.log_power(core.project_float(input16, model.filters))[0])
        features = {}
        for bits in ((8, 12) if session == "T" else range(8, 13)):
            weights, scales, _ = core.quantize_weights(model.filters, bits)
            dots, _ = core.dot22_integer(qx, weights, bits + 20, weight_bits=bits)
            projected = core.reconstruct_projection(dots, model.quantization.input_delta, scales)
            features[bits] = core.log_power(projected)[0]
        return batch, reference, features

    selections = {}
    for subject, model in models.items():
        batch, reference, features = calculate(subject, "T")
        if pipeline.array_hash(batch.X) != model.training["X_sha256"]:
            raise ValueError("T epoch hash mismatch")
        selections[str(subject)] = select_masks(model, features, reference)
    selection_file = args.private / "training-selections.json"
    write(selection_file, selections)
    selections = read(selection_file)

    per_subject = []
    for subject, model in models.items():
        batch, reference, features = calculate(subject, "E")
        saved = read(args.freeze / "evaluation" / "private-predictions" / f"s{subject:02d}.json")
        if (list(batch.trial_ids) != saved["trial_ids"] or batch.y.tolist() != saved["labels"]
                or pipeline.array_hash(batch.X) != saved["test_X_sha256"]):
            raise ValueError("E epoch identity mismatch")
        old = saved["predictions"]
        for name, actual in (("input16_only", reference), ("joint16x8", score(model, features[8])),
                             ("joint16x12", score(model, features[12]))):
            np.testing.assert_allclose(actual, old[name]["score"], rtol=0, atol=1e-11)
            np.testing.assert_array_equal(1 + (actual > 0), old[name]["prediction"])
        float_prediction = np.array(old["float"]["prediction"])
        configurations = {f"uniform{b}": score(model, f) for b, f in features.items()}
        for count, choice in selections[str(subject)].items():
            configurations[f"mixed{count}"] = score(model, mixed(features, choice["mask"]))
        for name, values in configurations.items():
            prediction = 1 + (values > 0)
            hit = prediction == batch.y
            baseline_hit = float_prediction == batch.y
            per_subject.append(dict(subject=subject, config=name, trials=len(hit),
                                    input16_disagreements=int(np.sum((values > 0) != (reference > 0))),
                                    float_disagreements=int(np.sum(prediction != float_prediction)),
                                    correct_to_wrong=int(np.sum(baseline_hit & ~hit)),
                                    wrong_to_correct=int(np.sum(~baseline_hit & hit)),
                                    correct=int(np.sum(hit)), score_sse=float(np.sum((values - reference) ** 2))))
    summary = {}
    for name in configurations:
        rows = [r for r in per_subject if r["config"] == name]
        total = {k: sum(r[k] for r in rows) for k in ("trials", "input16_disagreements", "float_disagreements",
                                                      "correct_to_wrong", "wrong_to_correct", "correct", "score_sse")}
        total["mean_balanced_accuracy"] = total["correct"] / total["trials"]  # 72 + 72 for each subject
        total["score_rmse_vs_input16"] = (total.pop("score_sse") / total["trials"]) ** 0.5
        summary[name] = total
    paired = {}
    for count in (1, 2, 3):
        uniform = [r["input16_disagreements"] for r in per_subject if r["config"] == f"uniform{8+count}"]
        selected = [r["input16_disagreements"] for r in per_subject if r["config"] == f"mixed{count}"]
        difference = np.subtract(selected, uniform)
        paired[str(count)] = dict(payload_bytes=11*(8+count), fewer=int(np.sum(difference < 0)),
                                  same=int(np.sum(difference == 0)), more=int(np.sum(difference > 0)))
    write(args.output, dict(protocol_sha256=digest(Path(__file__).with_name("CONTROL.md")),
                            implementation_sha256=digest(Path(__file__)), training_freeze=manifest["freeze_sha256"],
                            t_selection_sha256=digest(selection_file), summary=summary, paired=paired,
                            per_subject=per_subject))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("pilot", "raw", "freeze", "private", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    run(parser.parse_args())
