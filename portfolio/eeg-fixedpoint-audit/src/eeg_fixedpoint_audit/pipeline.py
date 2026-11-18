"""Train-only CSP/LDA fitting and frozen evaluation of ten numeric variants.

This module uses the mature estimators to train; the exported binary linear
decision rule is checked against sklearn before it is frozen.  It does not
select a bit width from evaluation-session performance.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .core import (
    CHANNELS, INPUT_BITS, POWER_FLOOR, WEIGHT_BITS, FrozenQuantization,
    dot22_integer, fit_quantization, log_power, project_float,
    quantize_input, reconstruct_projection,
    verify_dot22_bigint,
)


CSP_PARAMETERS = {
    "n_components": 4, "reg": 0.1, "log": True, "cov_est": "epoch",
    "transform_into": "average_power", "norm_trace": False, "rank": "full",
    "component_order": "alternate",
}
LDA_PARAMETERS = {"solver": "lsqr", "shrinkage": "auto"}
VARIANTS = (
    "float", "input16_only", "weight4_only", "weight8_only", "weight12_only",
    "weight16_only", "joint16x4", "joint16x8", "joint16x12", "joint16x16",
)


def canonical_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def pipeline_config():
    return {
        "schema": "eeg-csp-projection-audit-v1", "input_unit": "V",
        "channels": CHANNELS, "components": 4, "input_bits": INPUT_BITS,
        "weight_bits": list(WEIGHT_BITS), "primary_weight_bits": 12,
        "power_floor": POWER_FLOOR, "csp": deepcopy(CSP_PARAMETERS),
        "lda": deepcopy(LDA_PARAMETERS), "variants": list(VARIANTS),
        "input_integer_domain": [-32767, 32767],
        "partial_accumulator_bits": "16 + weight_bits + 4",
        "only_integer_stage": "CSP spatial projection",
    }


def _validate_X(X):
    array = np.asarray(X)
    if array.dtype.kind not in "fiu" or array.ndim != 3:
        raise ValueError("X must be a real (trials, 22, time) array")
    if array.shape[0] == 0 or array.shape[1] != CHANNELS or array.shape[2] == 0:
        raise ValueError("X must have nonempty trials/time and exactly 22 EEG channels")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinity")
    return array


def _validate_y(y, count):
    labels = np.asarray(y)
    if labels.dtype.kind not in "iu" or labels.shape != (count,):
        raise ValueError("Labels must be one integer per trial")
    labels = labels.astype(np.int64, copy=False)
    if not np.all(np.isin(labels, [1, 2])):
        raise ValueError("Only fixed left/right labels 1 and 2 are accepted")
    if set(labels.tolist()) != {1, 2}:
        raise ValueError("Both left/right classes must be present")
    return labels


def _predict_from_parameters(features, coefficient, intercept, classes):
    score = (features @ coefficient.T + intercept).ravel()
    prediction = np.asarray(classes)[(score > 0).astype(np.int64)]
    return prediction.astype(np.int64), score


@dataclass(frozen=True)
class SubjectModel:
    subject_id: str
    filters: np.ndarray
    lda_coefficient: np.ndarray
    lda_intercept: np.ndarray
    classes: np.ndarray
    quantization: FrozenQuantization
    training: dict
    checks: dict

    def numeric_snapshot(self):
        return {
            "config": pipeline_config(), "subject_id": self.subject_id,
            "filters": self.filters.tolist(),
            "lda_coefficient": self.lda_coefficient.tolist(),
            "lda_intercept": self.lda_intercept.tolist(),
            "classes": self.classes.tolist(), "quantization": self.quantization.to_dict(),
            "training_X_sha256": self.training["X_sha256"],
            "training_y_sha256": self.training["y_sha256"],
        }

    @property
    def numeric_sha256(self):
        return canonical_hash(self.numeric_snapshot())

    def to_dict(self):
        return {
            "schema": "frozen-subject-csp-projection-v1", **self.numeric_snapshot(),
            "numeric_sha256": self.numeric_sha256,
            "training": deepcopy(self.training), "checks": deepcopy(self.checks),
        }

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != "frozen-subject-csp-projection-v1":
            raise ValueError("Unsupported frozen model schema")
        if value.get("config") != pipeline_config():
            raise ValueError("Frozen model config differs from the running pipeline")
        arrays = []
        for key, shape, dtype in (
            ("filters", (4, CHANNELS), np.float64),
            ("lda_coefficient", (1, 4), np.float64),
            ("lda_intercept", (1,), np.float64), ("classes", (2,), np.int64),
        ):
            array = np.asarray(value[key], dtype=dtype)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid frozen {key}")
            array.setflags(write=False)
            arrays.append(array)
        if arrays[3].tolist() != [1, 2]:
            raise ValueError("Frozen class order must be [1, 2]")
        model = cls(str(value["subject_id"]), *arrays,
                    FrozenQuantization.from_dict(value["quantization"]),
                    deepcopy(value["training"]), deepcopy(value["checks"]))
        if model.numeric_sha256 != value["numeric_sha256"]:
            raise ValueError("Frozen numerical model hash mismatch")
        return model


def fit_subject(train_X, train_y, subject_id, training_metadata=None):
    """Fit a subject using T only; there is no evaluation argument or reader."""
    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X = _validate_X(train_X)
    y = _validate_y(train_y, len(X))
    csp = CSP(**CSP_PARAMETERS)
    csp.fit(X, y)
    weights = np.asarray(csp.filters_[:4], dtype=np.float64).copy()
    projected = project_float(X, weights)
    features, floor_stats = log_power(projected)
    library_features = csp.transform(X)
    # The guard changes only values below the declared floor.  Everything else
    # must reconstruct the mature transform at the frozen tolerance.
    powers = np.mean(projected ** 2, axis=2)
    unguarded = powers >= POWER_FLOOR
    if not np.allclose(features[unguarded], library_features[unguarded], rtol=1e-9, atol=1e-9):
        raise ArithmeticError("Exported CSP features differ from MNE")
    lda = LinearDiscriminantAnalysis(**LDA_PARAMETERS)
    lda.fit(features, y)
    coefficient, intercept = lda.coef_.copy(), lda.intercept_.copy()
    classes = lda.classes_.astype(np.int64, copy=True)
    predictions, scores = _predict_from_parameters(features, coefficient, intercept, classes)
    if not np.array_equal(predictions, lda.predict(features)):
        raise ArithmeticError("Exported LDA prediction rule differs from sklearn")
    if not np.allclose(scores, lda.decision_function(features), rtol=1e-12, atol=1e-12):
        raise ArithmeticError("Exported LDA scores differ from sklearn")

    # A separate fit verifies the unit convention, including shrinkage and
    # coefficient training.  Flipping an eigenvector sign is harmless to power;
    # component permutations are not silently matched or reordered here.
    microvolt_X = X * 1e6
    unit_csp = CSP(**CSP_PARAMETERS)
    unit_csp.fit(microvolt_X, y)
    unit_projection = project_float(microvolt_X, unit_csp.filters_[:4])
    unit_features, unit_floor = log_power(unit_projection)
    unit_lda = LinearDiscriminantAnalysis(**LDA_PARAMETERS).fit(unit_features, y)
    signs = np.where(np.sum(projected * unit_projection, axis=(0, 2)) < 0, -1.0, 1.0)
    aligned_unit_projection = unit_projection * signs[None, :, None]
    unit_checks = {
        "separate_CSP_and_LDA_fits": True, "component_order": CSP_PARAMETERS["component_order"],
        "projection_sign_alignment": signs.astype(int).tolist(),
        "projection_allclose": bool(np.allclose(projected, aligned_unit_projection, rtol=1e-9, atol=1e-9)),
        "power_allclose": bool(np.allclose(powers, np.mean(unit_projection ** 2, axis=2), rtol=1e-9, atol=1e-9)),
        "feature_allclose": bool(np.allclose(features, unit_features, rtol=1e-9, atol=1e-9)),
        "predictions_equal": bool(np.array_equal(predictions, unit_lda.predict(unit_features))),
        "max_abs_feature_error": float(np.max(np.abs(features - unit_features))),
        "unit_floor_hits": unit_floor["floor_hits"],
    }
    if not all(unit_checks[key] for key in ("projection_allclose", "power_allclose", "feature_allclose", "predictions_equal")):
        raise ArithmeticError(f"V/uV unit invariance check failed: {unit_checks}")

    quantization = fit_quantization(X, weights)
    qX, training_clip = quantize_input(X, quantization.input_delta)
    if training_clip["clipped_count"]:
        raise ArithmeticError("A scale fitted from all training epochs must not clip them")
    centered = X.transpose(1, 0, 2).reshape(CHANNELS, -1)
    empirical_rank = int(np.linalg.matrix_rank(centered - centered.mean(axis=1, keepdims=True)))
    training = {
        "trials": len(X), "samples_per_trial": X.shape[2],
        "class_counts": {str(label): int((y == label).sum()) for label in (1, 2)},
        "X_sha256": array_hash(X), "y_sha256": array_hash(y),
        "empirical_centered_EEG_rank": empirical_rank,
        "metadata": deepcopy(training_metadata or {}),
    }
    checks = {
        "MNE_feature_reconstruction": True, "MNE_atol": 1e-9, "MNE_rtol": 1e-9,
        "MNE_max_abs_feature_error_unguarded": float(np.max(np.abs(features[unguarded] - library_features[unguarded]))) if unguarded.any() else None,
        "training_floor": floor_stats, "sklearn_exported_rule_verified": True,
        "unit_invariance": unit_checks, "training_input_quantization": training_clip,
    }
    for array in (weights, coefficient, intercept, classes):
        array.setflags(write=False)
    return SubjectModel(str(subject_id), weights, coefficient, intercept, classes,
                        quantization, training, checks)


def _errors(actual, reference):
    difference = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
            "max_abs": float(np.max(np.abs(difference)))}


def _classification(y, predictions):
    matrix = [[int(np.sum((y == truth) & (predictions == prediction)))
               for prediction in (1, 2)] for truth in (1, 2)]
    totals = [sum(row) for row in matrix]
    recalls = [matrix[index][index] / total if total else None for index, total in enumerate(totals)]
    return {
        "trials": len(y), "label_order": [1, 2], "confusion": matrix,
        "balanced_accuracy": float(sum(recalls) / 2) if all(value is not None for value in recalls) else None,
        "class_recalls": recalls, "correct": int(np.sum(y == predictions)),
    }


def training_kernel_fixtures(model, train_X, trial_ids=None):
    """Private, preselected T-prefix vectors; no E data is needed by RTL."""
    X = _validate_X(train_X)
    if len(X) < 2 or X.shape[2] < 32:
        raise ValueError("Training fixture selection needs two trials and 32 samples")
    ids = tuple(str(value) for value in (trial_ids if trial_ids is not None else range(len(X))))
    if len(ids) != len(X):
        raise ValueError("Trial ID length mismatch")
    qX, stats = quantize_input(X[:2, :, :32], model.quantization.input_delta)
    payload = {"qx": qX, "trial_ids": np.asarray(ids[:2]), "subject_id": np.asarray(model.subject_id),
               "input_delta": np.asarray(model.quantization.input_delta),
               "input_clip_count": np.asarray(stats["clipped_count"])}
    for bits, config in model.quantization.weight_configs.items():
        result, _ = dot22_integer(qX, config["integers"], config["accumulator_bits"], weight_bits=bits)
        verify_dot22_bigint(qX, config["integers"], result, config["accumulator_bits"])
        payload[f"qw_{bits}"] = config["integers"]
        payload[f"expected_{bits}"] = result
        payload[f"delta_w_{bits}"] = config["deltas"]
        payload[f"accumulator_bits_{bits}"] = np.asarray(config["accumulator_bits"])
    return payload


def evaluate_subject(model, test_X, test_y, trial_ids=None, artifact_mask=None):
    """Evaluate a frozen model in all ten declared rows, without refitting."""
    X = _validate_X(test_X)
    y = _validate_y(test_y, len(X))
    ids = tuple(str(value) for value in (trial_ids if trial_ids is not None else range(len(X))))
    if len(ids) != len(X) or len(set(ids)) != len(ids):
        raise ValueError("Trial IDs must be unique and match the trial count")
    artifacts = np.zeros(len(X), dtype=bool) if artifact_mask is None else np.asarray(artifact_mask)
    if artifacts.shape != (len(X),) or artifacts.dtype.kind != "b":
        raise ValueError("artifact_mask must have one boolean per trial")
    before = model.numeric_sha256
    baseline_projection = project_float(X, model.filters)
    baseline_features, _ = log_power(baseline_projection)
    baseline_prediction, baseline_score = _predict_from_parameters(
        baseline_features, model.lda_coefficient, model.lda_intercept, model.classes)
    baseline_metrics = _classification(y, baseline_prediction)
    qX, input_stats = quantize_input(X, model.quantization.input_delta)
    restored_X = qX.astype(np.float64) * model.quantization.input_delta
    rows, predictions_detail = [], {}

    for variant in VARIANTS:
        partial_stats = None
        input_quantized = variant == "input16_only" or variant.startswith("joint")
        weight_bits = None
        if variant == "float":
            projected = baseline_projection
        elif variant == "input16_only":
            projected = project_float(restored_X, model.filters)
        elif variant.startswith("weight"):
            weight_bits = int(variant[len("weight"):].split("_")[0])
            config = model.quantization.weight_configs[weight_bits]
            restored_weights = config["integers"].astype(np.float64) * config["deltas"][:, None]
            projected = project_float(X, restored_weights)
        else:
            weight_bits = int(variant.split("x")[1])
            config = model.quantization.weight_configs[weight_bits]
            integers, partial_stats = dot22_integer(qX, config["integers"], config["accumulator_bits"], weight_bits=weight_bits)
            partial_stats["independent_bigint"] = verify_dot22_bigint(
                qX, config["integers"], integers, config["accumulator_bits"])
            projected = reconstruct_projection(integers, model.quantization.input_delta, config["deltas"])
        features, floor_stats = log_power(projected)
        prediction, score = _predict_from_parameters(features, model.lda_coefficient, model.lda_intercept, model.classes)
        classification = _classification(y, prediction)
        float_correct, correct = baseline_prediction == y, prediction == y
        row = {
            "subject_id": model.subject_id, "variant": variant,
            "input_bits": INPUT_BITS if input_quantized else None, "weight_bits": weight_bits,
            **classification, "delta_balanced_accuracy": classification["balanced_accuracy"] - baseline_metrics["balanced_accuracy"],
            "flip_count": int(np.sum(prediction != baseline_prediction)),
            "correct_to_wrong": int(np.sum(float_correct & ~correct)),
            "wrong_to_correct": int(np.sum(~float_correct & correct)),
            "float_wrong_unchanged": int(np.sum(~float_correct & (prediction == baseline_prediction))),
            "projection_error": _errors(projected, baseline_projection),
            "feature_error": _errors(features, baseline_features),
            "score_error": _errors(score, baseline_score), "power": floor_stats,
            "input_clipped_count": input_stats["clipped_count"] if input_quantized else 0,
            "weight_clipped_count": model.quantization.weight_configs[weight_bits]["stats"]["clipped_count"] if weight_bits else 0,
            "partial_accumulation": partial_stats,
            "margin_sign_preservation_count": int(np.sum(np.abs(score - baseline_score) < np.abs(baseline_score))),
            "artifact_marked": _classification(y[artifacts], prediction[artifacts]),
            "artifact_unmarked": _classification(y[~artifacts], prediction[~artifacts]),
            "model_numeric_sha256": before,
        }
        rows.append(row)
        predictions_detail[variant] = {"prediction": prediction.tolist(), "score": score.tolist(),
                                       "features": features.tolist()}
    if model.numeric_sha256 != before:
        raise ArithmeticError("Evaluation changed the frozen numerical model")
    return {
        "subject_id": model.subject_id, "rows": rows,
        "details": {"trial_ids": list(ids), "labels": y.tolist(),
                    "artifact_mask": artifacts.tolist(), "input_quantization": input_stats,
                    "test_X_sha256": array_hash(X), "test_y_sha256": array_hash(y),
                    "predictions": predictions_detail},
    }


def aggregate_rows(rows):
    """Equal-weight subject summaries; no samples are treated as subjects."""
    if any(row.get("variant") not in VARIANTS for row in rows):
        raise ValueError("Unexpected variant in the fixed matrix")
    expected_subjects = {row["subject_id"] for row in rows if row["variant"] == "float"}
    summaries = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        subjects = [row["subject_id"] for row in selected]
        if not selected or len(set(subjects)) != len(subjects) or set(subjects) != expected_subjects:
            raise ValueError(f"Missing or duplicate subject rows for {variant}")
        values = np.asarray([row["balanced_accuracy"] for row in selected])
        deltas = np.asarray([row["delta_balanced_accuracy"] for row in selected])
        summaries.append({
            "variant": variant, "subjects": len(selected),
            "balanced_accuracy_mean": float(values.mean()), "balanced_accuracy_median": float(np.median(values)),
            "balanced_accuracy_worst": float(values.min()),
            "delta_balanced_accuracy_mean": float(deltas.mean()),
            "delta_balanced_accuracy_median": float(np.median(deltas)),
            "delta_balanced_accuracy_worst": float(deltas.min()),
            "total_flips": sum(row["flip_count"] for row in selected),
            "subject_equal_weight": True,
        })
    return summaries
