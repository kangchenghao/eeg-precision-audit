"""Independent scalar oracles for the fixed-point spatial projection."""

import json
import math
import unittest

import numpy as np

from eeg_fixedpoint_audit.core import (
    dot22_integer, fit_quantization, log_power, power_of_two_scale,
    project_float, quantize_input, quantize_weights, reconstruct_projection,
)


def bigint_dot_oracle(x, w):
    """Use Python integers and scalar loops, not a second NumPy dot product."""
    result = np.empty((x.shape[0], w.shape[0], x.shape[2]), dtype=np.int64)
    partial_min, partial_max = 0, 0
    for trial in range(x.shape[0]):
        for component in range(w.shape[0]):
            for sample in range(x.shape[2]):
                accumulator = 0
                for channel in range(22):
                    accumulator += int(x[trial, channel, sample]) * int(w[component, channel])
                    partial_min = min(partial_min, accumulator)
                    partial_max = max(partial_max, accumulator)
                result[trial, component, sample] = accumulator
    return result, partial_min, partial_max


class QuantizationTests(unittest.TestCase):
    def test_negative_half_ties_and_real_clipping_are_distinct(self):
        values = [-32769, -32768.5, -32768, -32767.5, -2.5, -1.5, -0.5,
                  0.5, 1.5, 2.5, 32766.5, 32767, 32767.4, 32767.5, 32768,
                  1e30, -1e30]
        x = np.zeros((1, 22, len(values)))
        x[0, 0] = values
        q, stats = quantize_input(x, 1.0)
        expected = [max(-32767, min(32767, round(value))) for value in values]
        np.testing.assert_array_equal(q[0, 0], expected)
        self.assertEqual(stats["clipped_count"], 8)
        self.assertTrue(np.issubdtype(q.dtype, np.signedinteger))
        self.assertEqual(int(q.min()), -32767)
        self.assertEqual(int(q.max()), 32767)

    def test_weight_rounding_symmetric_domain_and_per_row_scale(self):
        w = np.zeros((4, 22))
        values = [-7, 7, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5]
        for row in range(4):
            w[row, :len(values)] = np.asarray(values) * (2.0 ** row)
        q, delta, stats = quantize_weights(w, 4)
        for row in range(4):
            np.testing.assert_array_equal(q[row, :len(values)], [round(value) for value in values])
        np.testing.assert_array_equal(np.asarray(delta).reshape(-1), [1, 2, 4, 8])
        self.assertEqual(stats["clipped_count"], 0)
        self.assertEqual(int(q.min()), -7)
        self.assertEqual(int(q.max()), 7)

    def test_power_of_two_scale_boundary_uses_exact_comparison(self):
        for bits in (4, 8, 12, 16):
            qmax = 2 ** (bits - 1) - 1
            boundary = qmax * 2.0 ** -20
            for maximum, expected_delta in (
                (np.nextafter(boundary, 0.0), 2.0 ** -20),
                (boundary, 2.0 ** -20),
                (np.nextafter(boundary, math.inf), 2.0 ** -19),
            ):
                w = np.full((4, 22), maximum)
                with self.subTest(bits=bits, maximum=maximum):
                    _, delta, stats = quantize_weights(w, bits)
                    np.testing.assert_array_equal(np.asarray(delta).reshape(-1), [expected_delta] * 4)
                    self.assertEqual(stats["clipped_count"], 0)

    def test_scale_is_sufficient_and_half_scale_is_insufficient(self):
        from fractions import Fraction
        for bits in (4, 8, 12, 16):
            limit = 2 ** (bits - 1) - 1
            for maximum in (1e-12, 1e-6, 1.0, 7.0, 1e20):
                delta, _ = power_of_two_scale(maximum, bits)
                exact_maximum, exact_delta = Fraction(maximum), Fraction(delta)
                self.assertLessEqual(exact_maximum / exact_delta, limit)
                self.assertGreater(exact_maximum / (exact_delta / 2), limit)

    def test_bad_scales_nonfinite_values_and_zero_weights_are_rejected(self):
        valid_x = np.ones((1, 22, 3))
        valid_w = np.ones((4, 22))
        for delta in (0, -1, math.nan, math.inf):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                quantize_input(valid_x, delta)
        for value in (math.nan, math.inf, -math.inf):
            x, w = valid_x.copy(), valid_w.copy()
            x[0, 0, 0], w[0, 0] = value, value
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantize_input(x, 1.0)
            with self.subTest(value=value), self.assertRaises(ValueError):
                quantize_weights(w, 12)
        with self.assertRaises(ValueError):
            quantize_weights(np.zeros((4, 22)), 12)

    def test_fit_configuration_snapshot_is_json_and_input_is_not_mutated(self):
        x = np.arange(1, 1 + 2 * 22 * 5, dtype=np.float64).reshape(2, 22, 5) * 1e-6
        w = np.arange(1, 89, dtype=np.float64).reshape(4, 22)
        x_before, w_before = x.copy(), w.copy()
        fitted = fit_quantization(x, w)
        self.assertIsInstance(json.loads(json.dumps(fitted.to_dict(), allow_nan=False)), dict)
        np.testing.assert_array_equal(x, x_before)
        np.testing.assert_array_equal(w, w_before)
        with self.assertRaises(ValueError):
            fit_quantization(np.zeros_like(x), w)


class IntegerProjectionTests(unittest.TestCase):
    def test_all_widths_match_bigint_oracle_and_partial_extrema(self):
        rng = np.random.default_rng(711)
        for bits in (4, 8, 12, 16):
            x = rng.integers(-32768, 32768, size=(2, 22, 7), dtype=np.int64)
            w = rng.integers(-(2 ** (bits - 1)), 2 ** (bits - 1), size=(4, 22), dtype=np.int64)
            expected, low, high = bigint_dot_oracle(x, w)
            with self.subTest(bits=bits):
                actual, stats = dot22_integer(x, w, 16 + bits + 4)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(stats["partial_min"], low)
                self.assertEqual(stats["partial_max"], high)
                self.assertEqual(stats["overflow_count"], 0)

    def test_full_signed_minima_and_negative_extremes_fit_declared_width(self):
        x = np.full((1, 22, 1), -32768, dtype=np.int64)
        for bits in (4, 8, 12, 16):
            for weight in (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1):
                w = np.full((4, 22), weight, dtype=np.int64)
                with self.subTest(bits=bits, weight=weight):
                    actual, _ = dot22_integer(x, w, 16 + bits + 4)
                    np.testing.assert_array_equal(actual, np.full((1, 4, 1), 22 * -32768 * weight))

    def test_intermediate_overflow_is_rejected_even_when_final_sum_is_zero(self):
        x = np.zeros((1, 22, 1), dtype=np.int64)
        x[0, :2, 0] = 1
        w = np.zeros((4, 22), dtype=np.int64)
        w[:, 0], w[:, 1] = 64, -64
        self.assertEqual(int((x[0, :, 0] * w[0]).sum()), 0)
        with self.assertRaises(OverflowError):
            dot22_integer(x, w, 7)

    def test_missing_channel_and_fractional_integer_operands_are_rejected(self):
        x = np.ones((1, 22, 2), dtype=np.int64)
        w = np.ones((4, 22), dtype=np.int64)
        for bad_x, bad_w in ((x[:, :21], w), (x, w[:, :21]), (x.astype(float) + 0.5, w)):
            with self.subTest(xshape=bad_x.shape, wshape=bad_w.shape), self.assertRaises(ValueError):
                dot22_integer(bad_x, bad_w, 32)

    def test_coefficient_tampering_and_missing_term_disagree_with_oracle(self):
        x = np.arange(1, 45, dtype=np.int64).reshape(1, 22, 2)
        w = np.arange(1, 89, dtype=np.int64).reshape(4, 22)
        expected, _, _ = bigint_dot_oracle(x, w)
        altered = w.copy()
        altered[0, 0] += 1
        corrupted, _ = dot22_integer(x, altered, 32)
        missing = w.copy()
        missing[:, -1] = 0
        truncated, _ = dot22_integer(x, missing, 32)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(corrupted, expected)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(truncated, expected)


class PowerFeatureTests(unittest.TestCase):
    def test_power_is_mean_square_not_demeaned_variance(self):
        projected = np.full((2, 4, 5), 3.0)
        features, stats = log_power(projected)
        np.testing.assert_allclose(features, np.full((2, 4), math.log(9)), rtol=0, atol=1e-15)
        self.assertEqual(stats["floor_hits"], 0)

    def test_floor_counts_strictly_below_threshold_and_not_equal(self):
        projected = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.5, 0.5]]])
        features, stats = log_power(projected, power_floor=1.0)
        np.testing.assert_allclose(features, [[0, 0, math.log(4), 0]], rtol=0, atol=1e-15)
        self.assertEqual(stats["floor_hits"], 2)

    def test_nonfinite_signal_and_invalid_floor_are_rejected(self):
        y = np.ones((1, 4, 2))
        for floor in (0, -1, math.nan, math.inf):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                log_power(y, floor)
        y[0, 0, 0] = math.nan
        with self.assertRaises(ValueError):
            log_power(y)


class ReconstructionTests(unittest.TestCase):
    def test_float_projection_matches_scalar_channel_oracle(self):
        rng = np.random.default_rng(421)
        x = rng.normal(size=(2, 22, 9)) * 1e-6
        w = rng.normal(size=(4, 22)) * 1e4
        expected = np.empty((2, 4, 9))
        for trial in range(2):
            for component in range(4):
                for sample in range(9):
                    expected[trial, component, sample] = math.fsum(
                        float(x[trial, channel, sample]) * float(w[component, channel])
                        for channel in range(22))
        np.testing.assert_allclose(project_float(x, w), expected, rtol=1e-13, atol=1e-15)

    def test_reconstruction_restores_input_and_each_component_scale(self):
        integers = np.arange(-12, 12, dtype=np.int64).reshape(2, 4, 3)
        deltas = np.array([0.5, 2, 8, 32])
        actual = reconstruct_projection(integers, 0.25, deltas)
        expected = np.empty((2, 4, 3))
        for trial in range(2):
            for component in range(4):
                for sample in range(3):
                    expected[trial, component, sample] = int(integers[trial, component, sample]) * 0.25 * deltas[component]
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaises(ValueError):
            reconstruct_projection(integers, 0.3, deltas)


if __name__ == "__main__":
    unittest.main()
