"""Synthetic MAT tests: no downloaded EEG or participant-derived fixtures."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import savemat

from eeg_fixedpoint_audit.data import EpochConfig, design_fir, load_session


CLASSES = np.array(["left hand", "right hand", "feet", "tongue"], dtype=object)


def make_run(samples=16, trials=(1,), labels=(1,), artifacts=None):
    return {
        "X": np.zeros((samples, 25), dtype=np.float64),
        "trial": np.asarray(trials),
        "y": np.asarray(labels),
        "fs": 250,
        "classes": CLASSES,
        "artifacts": np.asarray([0] * len(trials) if artifacts is None else artifacts),
    }


def load_synthetic(runs, *, config=None, taps=(1.0,), official_counts=False):
    """Exercise a genuine scipy MAT round trip rather than mock the loader."""
    cells = np.empty((1, len(runs)), dtype=object)
    for index, run in enumerate(runs):
        cells[0, index] = run
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic-training.mat"
        savemat(path, {"data": cells})
        return load_session(
            path, 1, "T", config=config,
            fir_coefficients=np.asarray(taps, dtype=np.float64),
            require_official_counts=official_counts,
        )


class MatLoadingTests(unittest.TestCase):
    def setUp(self):
        self.short = EpochConfig(start_sample=0, stop_sample=4)

    def test_units_channel_order_and_eog_isolation(self):
        run = make_run()
        for channel in range(22):
            run["X"][:, channel] = channel + 1
        run["X"][:, 22:] = (1e8, -1e8, 5e8)
        batch = load_synthetic([run], config=self.short)
        expected = np.repeat(np.arange(1, 23)[:, None], 4, axis=1) * 1e-6
        np.testing.assert_array_equal(batch.X, expected[None, :, :])
        self.assertEqual(batch.X.dtype, np.float64)
        self.assertEqual(len(batch.channel_names), 22)
        self.assertEqual(len(set(batch.channel_names)), 22)

    def test_zero_eeg_stays_zero_despite_class_coded_eog(self):
        run = make_run(trials=(1, 9), labels=(1, 2))
        run["X"][:8, 22:] = 1e9
        run["X"][8:, 22:] = -1e9
        batch = load_synthetic([run], config=self.short)
        np.testing.assert_array_equal(batch.X, np.zeros((2, 22, 4)))

    def test_default_epoch_one_based_index_and_half_open_stop(self):
        run = make_run(samples=1375)
        run["X"][:, :22] = np.arange(1375)[:, None]
        batch = load_synthetic([run])
        self.assertEqual(batch.X.shape, (1, 22, 750))
        np.testing.assert_array_equal(batch.X[0, 0], np.arange(625, 1375) * 1e-6)
        self.assertEqual(batch.event_metadata[0]["trial_onset_sample_zero_based"], 0)
        self.assertEqual(batch.event_metadata[0]["epoch_start_sample_zero_based"], 625)
        self.assertEqual(batch.event_metadata[0]["epoch_stop_sample_exclusive"], 1375)

    def test_one_sample_short_is_rejected_without_padding(self):
        with self.assertRaises(ValueError):
            load_synthetic([make_run(samples=1374)])

    def test_class_selection_keeps_source_order_and_artifact_alignment(self):
        run = make_run(samples=32, trials=(1, 9, 17, 25),
                       labels=(2, 4, 1, 3), artifacts=(1, 0, 0, 1))
        run["X"][:, 0] = np.arange(32)
        batch = load_synthetic([run], config=self.short)
        np.testing.assert_array_equal(batch.y, [2, 1])
        np.testing.assert_array_equal(batch.artifact_mask, [True, False])
        np.testing.assert_array_equal(batch.X[:, 0, 0], np.array([0, 16]) * 1e-6)
        self.assertEqual(len(batch.trial_ids), 2)
        self.assertEqual(len(set(batch.trial_ids)), 2)

    def test_empty_calibration_run_is_skipped(self):
        calibration = make_run(samples=8, trials=(), labels=())
        batch = load_synthetic([calibration, make_run()], config=self.short)
        self.assertEqual(batch.X.shape, (1, 22, 4))
        self.assertEqual(len(batch.event_metadata), 1)
        self.assertEqual(batch.event_metadata[0]["source_run_index"], 1)

    def test_evaluation_requires_opt_in_before_reading_path(self):
        with self.assertRaises((ValueError, PermissionError)):
            load_session(Path("no-such-private-evaluation.mat"), 1, "E")

    def test_invalid_indices_are_not_truncated_or_wrapped(self):
        for index in (0, -1, 1.5, 17, float("nan")):
            with self.subTest(index=index), self.assertRaises(ValueError):
                load_synthetic([make_run(trials=(index,))], config=self.short)

    def test_invalid_sample_rate_channels_and_nonfinite_data_are_rejected(self):
        wrong_rate = make_run()
        wrong_rate["fs"] = 160
        wrong_channels = make_run()
        wrong_channels["X"] = np.zeros((16, 24))
        nonfinite = make_run()
        nonfinite["X"][10, 1] = np.nan
        for run in (wrong_rate, wrong_channels, nonfinite):
            with self.subTest(kind=list(run)), self.assertRaises(ValueError):
                load_synthetic([run], config=self.short)

    def test_misaligned_labels_and_artifacts_are_rejected(self):
        for field, value in (("y", np.array([1, 2])),
                             ("artifacts", np.array([0, 1]))):
            run = make_run()
            run[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                load_synthetic([run], config=self.short)

    def test_formal_count_gate_rejects_small_synthetic_session(self):
        with self.assertRaises(ValueError):
            load_synthetic([make_run()], config=self.short, official_counts=True)

    def test_formal_count_gate_rejects_47_49_runs_despite_correct_total_counts(self):
        runs = []
        for first_class_count in (11, 13, 12, 12, 12, 12):
            labels = [1] * first_class_count + [2] * 12 + [3] * 12 + [4] * 12
            runs.append(make_run(samples=60, trials=tuple(range(1, len(labels) + 1)), labels=labels))
        # Six runs, 288 trials, 72 of every class still conceal two malformed runs.
        self.assertEqual(sum(len(run["y"]) for run in runs), 288)
        for label in (1, 2, 3, 4):
            self.assertEqual(sum(int((run["y"] == label).sum()) for run in runs), 72)
        with self.assertRaises(ValueError):
            load_synthetic(runs, config=self.short, official_counts=True)

    def test_int16_minimum_has_correct_float_magnitude_metadata(self):
        run = make_run()
        run["X"] = np.zeros((16, 25), dtype=np.int16)
        run["X"][0, 0] = -32768
        batch = load_synthetic([run], config=self.short)
        self.assertEqual(batch.metadata["runs"][0]["source_eeg_max_abs_microvolts"], 32768.0)
        self.assertEqual(batch.X[0, 0, 0], -32768 * 1e-6)


class CausalFilteringTests(unittest.TestCase):
    def test_default_fir_has_63_taps_and_documented_cutoff_response(self):
        taps = design_fir()
        self.assertEqual(len(taps), 63)
        np.testing.assert_allclose(taps, taps[::-1], rtol=0, atol=1e-16)
        # Independent direct DTFT, with cutoff amplitudes rather than ideal edges.
        def magnitude(frequency):
            return abs(sum(float(tap) * np.exp(-2j * np.pi * frequency * index / 250)
                           for index, tap in enumerate(taps)))
        self.assertAlmostEqual(magnitude(19), 1.0, places=12)
        for frequency in (8, 30):
            self.assertTrue(0.47 < magnitude(frequency) < 0.53)
        self.assertLess(magnitude(80), 0.005)

    def test_filter_history_is_continuous_across_trial_start(self):
        run = make_run(samples=12, trials=(4,))
        run["X"][1, 0] = 1.0
        taps = [1.0, 2.0, 3.0, 4.0, 5.0]
        batch = load_synthetic([run], config=EpochConfig(start_sample=0, stop_sample=4), taps=taps)
        # A pre-epoch impulse contributes at the epoch start; per-trial reset is wrong.
        expected = np.convolve(run["X"][:, 0] * 1e-6, taps)[:12][3:7]
        np.testing.assert_allclose(batch.X[0, 0], expected, rtol=0, atol=1e-20)
        self.assertNotEqual(batch.X[0, 0, 0], 0)

    def test_run_boundary_resets_filter_state(self):
        first = make_run(samples=8)
        first["X"][-1, 0] = 1e6
        second = make_run(samples=8)
        batch = load_synthetic([first, second], config=EpochConfig(start_sample=0, stop_sample=4), taps=(1, 2, 3))
        np.testing.assert_array_equal(batch.X[1], np.zeros((22, 4)))

    def test_future_perturbation_does_not_change_past(self):
        run = make_run(samples=16)
        run["X"][:, 0] = np.arange(16)
        altered = {**run, "X": run["X"].copy()}
        altered["X"][4:, :22] += 1e8
        config = EpochConfig(start_sample=0, stop_sample=8)
        original = load_synthetic([run], config=config, taps=(1, -2, 3))
        perturbed = load_synthetic([altered], config=config, taps=(1, -2, 3))
        np.testing.assert_array_equal(original.X[:, :, :4], perturbed.X[:, :, :4])
        self.assertFalse(np.array_equal(original.X[:, :, 4:], perturbed.X[:, :, 4:]))


if __name__ == "__main__":
    unittest.main()
