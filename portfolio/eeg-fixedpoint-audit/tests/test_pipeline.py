"""Task-level checks using entirely constructed signals and known decisions."""

from copy import deepcopy
import json
import math
import unittest

import numpy as np

from eeg_fixedpoint_audit.core import fit_quantization, log_power, project_float
from eeg_fixedpoint_audit.pipeline import (
    SubjectModel, VARIANTS, aggregate_rows, evaluate_subject, fit_subject,
    training_kernel_fixtures,
)


def constructed_model():
    """A known log-power threshold; this is not a model fitted on EEG."""
    filters = np.zeros((4, 22))
    filters[np.arange(4), np.arange(4)] = 1.0
    # This declared training maximum gives an exactly known input step of 1.
    calibration = np.full((2, 22, 40), 32767.0)
    quantization = fit_quantization(calibration, filters)
    return SubjectModel(
        "synthetic-threshold", filters, np.array([[1.0, 0, 0, 0]]),
        np.array([-2 * math.log(1.25)]), np.array([1, 2]), quantization,
        {"X_sha256": "constructed-not-an-eeg-hash", "y_sha256": "constructed"}, {},
    )


def threshold_trials():
    x = np.ones((6, 22, 8))
    x[:, 0, :] = np.asarray([1.2, 1.3, 1.2, 1.3, 2.0, 0.5])[:, None]
    y = np.array([1, 2, 2, 1, 2, 2], dtype=np.int64)
    return x, y


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.model = constructed_model()
        self.x, self.y = threshold_trials()
        self.ids = tuple(f"constructed/{index}" for index in range(6))

    def test_all_variants_and_hand_computed_metrics_and_flip_directions(self):
        result = evaluate_subject(self.model, self.x, self.y, self.ids)
        rows = {row["variant"]: row for row in result["rows"]}
        self.assertEqual(tuple(rows), VARIANTS)
        self.assertEqual(rows["float"]["confusion"], [[1, 1], [2, 2]])
        self.assertEqual(rows["float"]["balanced_accuracy"], 0.5)
        changed = rows["input16_only"]
        self.assertEqual(changed["confusion"], [[2, 0], [3, 1]])
        self.assertEqual(changed["balanced_accuracy"], 0.625)
        self.assertEqual(changed["delta_balanced_accuracy"], 0.125)
        self.assertEqual(changed["flip_count"], 2)
        self.assertEqual(changed["correct_to_wrong"], 1)
        self.assertEqual(changed["wrong_to_correct"], 1)
        self.assertEqual(changed["float_wrong_unchanged"], 2)
        for name, row in rows.items():
            if name.startswith("joint"):
                self.assertEqual(row["partial_accumulation"]["overflow_count"], 0)
            else:
                self.assertIsNone(row["partial_accumulation"])

    def test_fixed_trial_label_permutation_changes_only_label_dependent_results(self):
        snapshot = self.model.to_dict()
        original = evaluate_subject(self.model, self.x, self.y, self.ids)
        # Fixed trial identities and values; only left/right labels change.
        relabeled = evaluate_subject(self.model, self.x, 3 - self.y, self.ids)
        self.assertEqual(self.model.to_dict(), snapshot)
        self.assertEqual(original["details"]["predictions"], relabeled["details"]["predictions"])
        self.assertEqual(original["details"]["trial_ids"], relabeled["details"]["trial_ids"])
        self.assertEqual(original["details"]["input_quantization"], relabeled["details"]["input_quantization"])
        # For this deliberately known example the metric does change.
        self.assertEqual(relabeled["rows"][1]["balanced_accuracy"], 0.375)

    def test_signal_sentinel_does_not_change_training_artifact(self):
        snapshot = self.model.to_dict()
        altered = self.x.copy()
        altered[:, 0, :] = 1e6
        evaluate_subject(self.model, altered, self.y, self.ids)
        self.assertEqual(self.model.to_dict(), snapshot)

    def test_constructed_margin_crossing_changes_prediction(self):
        original = evaluate_subject(self.model, self.x, self.y, self.ids)
        altered = self.x.copy()
        altered[0, 0, :] = 2.0
        changed = evaluate_subject(self.model, altered, self.y, self.ids)
        before = original["details"]["predictions"]["float"]["prediction"]
        after = changed["details"]["predictions"]["float"]["prediction"]
        self.assertEqual((before[0], after[0]), (1, 2))
        self.assertEqual(before[1:], after[1:])

    def test_quantized_zero_power_is_reported_and_not_deleted(self):
        result = evaluate_subject(self.model, self.x, self.y, self.ids)
        input_row = result["rows"][1]
        self.assertEqual(input_row["power"]["floor_hits"], 1)
        self.assertEqual(input_row["trials"], 6)
        self.assertTrue(np.isfinite(result["details"]["predictions"]["input16_only"]["features"]).all())

    def test_model_round_trip_and_tampering_detection(self):
        saved = json.loads(json.dumps(self.model.to_dict(), allow_nan=False))
        restored = SubjectModel.from_dict(saved)
        self.assertEqual(restored.numeric_sha256, self.model.numeric_sha256)
        self.assertEqual(evaluate_subject(restored, self.x, self.y, self.ids),
                         evaluate_subject(self.model, self.x, self.y, self.ids))
        corrupted = deepcopy(saved)
        corrupted["filters"][0][0] += 0.25
        with self.assertRaises(ValueError):
            SubjectModel.from_dict(corrupted)

    def test_invalid_trial_ids_labels_and_nonfinite_values_fail(self):
        for ids in (self.ids[:-1], ("duplicate",) * 6):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                evaluate_subject(self.model, self.x, self.y, ids)
        bad = self.x.copy()
        bad[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            evaluate_subject(self.model, bad, self.y)
        with self.assertRaises(ValueError):
            evaluate_subject(self.model, self.x, np.array([1, 2, 3, 1, 2, 1]))

    def test_artifact_flags_keep_all_trials_and_report_subgroups(self):
        result = evaluate_subject(self.model, self.x, self.y, artifact_mask=np.array([1, 0, 1, 0, 1, 0], dtype=bool))
        for row in result["rows"]:
            self.assertEqual(row["trials"], 6)
            self.assertEqual(row["artifact_marked"]["trials"], 3)
            self.assertEqual(row["artifact_unmarked"]["trials"], 3)

    def test_kernel_vectors_select_first_two_trials_and_first_32_samples(self):
        train_x = np.arange(3 * 22 * 40, dtype=np.float64).reshape(3, 22, 40)
        payload = training_kernel_fixtures(self.model, train_x, ["first", "second", "third"])
        np.testing.assert_array_equal(payload["trial_ids"], ["first", "second"])
        np.testing.assert_array_equal(payload["qx"], train_x[:2, :, :32].astype(np.int64))
        for bits in (4, 8, 12, 16):
            expected = np.empty((2, 4, 32), dtype=np.int64)
            for trial in range(2):
                for component in range(4):
                    for sample in range(32):
                        expected[trial, component, sample] = sum(
                            int(payload["qx"][trial, channel, sample]) *
                            int(payload[f"qw_{bits}"][component, channel])
                            for channel in range(22))
            np.testing.assert_array_equal(payload[f"expected_{bits}"], expected)

    def test_aggregation_weights_subjects_equally(self):
        first = evaluate_subject(self.model, self.x, self.y)["rows"]
        second = deepcopy(first)
        for row in second:
            row["subject_id"] = "second-constructed-subject"
            row["balanced_accuracy"] = 1.0
            row["delta_balanced_accuracy"] = 0.0
            row["trials"] = 10000
        summary = aggregate_rows(first + second)
        self.assertEqual(summary[0]["balanced_accuracy_mean"], 0.75)
        self.assertEqual(summary[0]["subjects"], 2)
        self.assertTrue(summary[0]["subject_equal_weight"])
        with self.assertRaises(ValueError):
            aggregate_rows(first + first)

    def test_aggregation_rejects_missing_subject_in_one_variant(self):
        first = evaluate_subject(self.model, self.x, self.y)["rows"]
        second = deepcopy(first)
        for row in second:
            row["subject_id"] = "second-constructed-subject"
        with self.assertRaises(ValueError):
            aggregate_rows(first + second[:-1])


class MatureEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Independent trials with a deliberately strong covariance difference.
        # This is a positive-control learning task, not a BCI accuracy claim.
        rng = np.random.default_rng(1541)
        cls.train_y = np.tile([1, 2], 12).astype(np.int64)
        cls.test_y = np.tile([1, 2], 6).astype(np.int64)
        def signals(labels):
            values = rng.normal(size=(len(labels), 22, 128)) * 1e-6
            for index, label in enumerate(labels):
                selected = slice(0, 3) if label == 1 else slice(3, 6)
                values[index, selected] *= 6
            return values
        cls.train_x, cls.test_x = signals(cls.train_y), signals(cls.test_y)
        cls.model = fit_subject(cls.train_x, cls.train_y, "constructed-covariance")

    def test_constructed_covariance_task_can_be_learned(self):
        result = evaluate_subject(self.model, self.test_x, self.test_y)
        self.assertEqual(result["rows"][0]["balanced_accuracy"], 1.0)
        self.assertEqual(len(result["rows"]), 10)
        self.assertTrue(self.model.checks["MNE_feature_reconstruction"])
        self.assertTrue(self.model.checks["sklearn_exported_rule_verified"])

    def test_exported_filters_match_independent_mean_square_log_formula(self):
        projected = project_float(self.test_x, self.model.filters)
        expected = np.empty((len(self.test_x), 4))
        for trial in range(len(self.test_x)):
            for component in range(4):
                power = math.fsum(float(value) ** 2 for value in projected[trial, component]) / 128
                expected[trial, component] = math.log(max(power, 1e-30))
        actual, _ = log_power(projected)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_unit_check_performs_separate_fits_and_preserves_features_and_predictions(self):
        checks = self.model.checks["unit_invariance"]
        for key in ("separate_CSP_and_LDA_fits", "projection_allclose", "power_allclose",
                    "feature_allclose", "predictions_equal"):
            self.assertTrue(checks[key], key)
        unit_model = fit_subject(self.train_x * 1e6, self.train_y, "constructed-covariance-uv")
        volts = evaluate_subject(self.model, self.test_x, self.test_y)["details"]["predictions"]["float"]
        microvolts = evaluate_subject(unit_model, self.test_x * 1e6, self.test_y)["details"]["predictions"]["float"]
        np.testing.assert_allclose(volts["features"], microvolts["features"], atol=1e-9, rtol=1e-9)
        self.assertEqual(volts["prediction"], microvolts["prediction"])


if __name__ == "__main__":
    unittest.main()
