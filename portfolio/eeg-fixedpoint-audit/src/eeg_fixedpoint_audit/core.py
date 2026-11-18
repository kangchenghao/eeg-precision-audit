"""Exact arithmetic for the spatial-projection slice of an EEG pipeline.

Filtering, power, logarithms and classification remain floating point.  These
functions neither train a classifier nor interpret a numerical check as a
clinical guarantee.  Arrays have axes (trial, channel/component, time).
"""

from dataclasses import dataclass
from copy import deepcopy
import math
from numbers import Integral

import numpy as np


INPUT_BITS = 16
WEIGHT_BITS = (4, 8, 12, 16)
CHANNELS = 22
POWER_FLOOR = 1e-30


def _bits(value, name="bits", maximum=16):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not 2 <= value <= maximum:
        raise ValueError(f"{name} must be in 2..{maximum}")
    return value


def _finite(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.size == 0:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _readonly(values, dtype=None):
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def power_of_two_scale(maximum, bits):
    """Smallest representable power of two >= maximum / (2**(bits-1)-1).

    Integer-ratio comparisons avoid rounding log2 at a power-of-two boundary.
    The requested real scale must be representable in binary64; otherwise the
    contract is rejected instead of substituting an unrelated scale.
    """
    bits = _bits(bits)
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Scale maximum must be finite and strictly positive")
    numerator, denominator = maximum.as_integer_ratio()
    limit = (1 << (bits - 1)) - 1
    exponent = math.frexp(maximum)[1] - (bits - 1)

    def fits(candidate):
        if candidate >= 0:
            return numerator <= denominator * limit * (1 << candidate)
        return numerator * (1 << -candidate) <= denominator * limit

    while not fits(exponent):
        exponent += 1
    while fits(exponent - 1):
        exponent -= 1
    try:
        delta = math.ldexp(1.0, exponent)
    except OverflowError as error:
        raise ValueError("Scale is outside binary64 range") from error
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Scale is outside binary64 range")
    return delta, exponent


def _quantize(values, delta, bits, *, symmetric):
    array = _finite(values, "values")
    bits = _bits(bits)
    scales = _finite(delta, "delta")
    if np.any(scales <= 0):
        raise ValueError("Quantization scales must be strictly positive")
    positive = (1 << (bits - 1)) - 1
    negative = -positive if symmetric else -(1 << (bits - 1))
    # A finite out-of-domain input can overflow division; clipping that value
    # still has an unambiguous result.  It is counted and never wraps to int64.
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        scaled = array / scales
        rounded = np.rint(scaled)  # nearest, ties to even, for both signs
    clipped_mask = (rounded < negative) | (rounded > positive)
    range_mask = (scaled < negative) | (scaled > positive)
    result = np.clip(rounded, negative, positive).astype(np.int64)
    stats = {
        "bits": bits, "integer_min": negative, "integer_max": positive,
        "rounding": "nearest_ties_to_even", "symmetric_domain": bool(symmetric),
        "elements": int(array.size), "clipped_count": int(clipped_mask.sum()),
        "range_exceed_count": int(range_mask.sum()),
        "scaled_nonfinite_count": int((~np.isfinite(scaled)).sum()),
        "quantized_min": int(result.min()), "quantized_max": int(result.max()),
    }
    if array.ndim == 3:
        stats["clipped_by_trial"] = clipped_mask.sum(axis=(1, 2)).astype(int).tolist()
    return result, stats


def quantize_input(values, delta_x, bits=INPUT_BITS):
    """RNE then saturate to the symmetric input domain; never refit a scale."""
    if np.asarray(delta_x).size != 1:
        raise ValueError("All input channels must share one input scale")
    return _quantize(values, float(np.asarray(delta_x).item()), bits, symmetric=True)


def quantize_weights(weights, bits):
    """Fit one power-of-two scale per trained projection row, then RNE."""
    weights = _finite(weights, "weights")
    if weights.ndim != 2 or weights.shape[1] != CHANNELS:
        raise ValueError("Weights must have shape (components, 22)")
    maxima = np.max(np.abs(weights), axis=1)
    fitted = [power_of_two_scale(value, bits) for value in maxima]
    deltas = np.array([item[0] for item in fitted], dtype=np.float64)
    quantized, stats = _quantize(weights, deltas[:, None], bits, symmetric=True)
    stats["exponents"] = [item[1] for item in fitted]
    stats["training_abs_max_by_component"] = maxima.tolist()
    if stats["clipped_count"]:
        raise ArithmeticError("A scale fitted from these weights must not clip them")
    return quantized, deltas, stats


@dataclass(frozen=True)
class FrozenQuantization:
    input_bits: int
    input_delta: float
    input_exponent: int
    training_input_abs_max: float
    weight_configs: dict

    def to_dict(self):
        return {
            "input_bits": self.input_bits, "input_delta": self.input_delta,
            "input_exponent": self.input_exponent,
            "training_input_abs_max": self.training_input_abs_max,
            "weight_configs": {
                str(bits): {
                    "bits": bits, "accumulator_bits": config["accumulator_bits"],
                    "integers": config["integers"].tolist(),
                    "deltas": config["deltas"].tolist(),
                    "exponents": list(config["exponents"]),
                    "stats": deepcopy(config["stats"]),
                } for bits, config in self.weight_configs.items()
            },
        }

    @classmethod
    def from_dict(cls, value):
        configs = {}
        for key, config in value["weight_configs"].items():
            bits = _bits(int(key))
            configs[bits] = {
                "accumulator_bits": int(config["accumulator_bits"]),
                "integers": _readonly(config["integers"], np.int64),
                "deltas": _readonly(config["deltas"], np.float64),
                "exponents": tuple(map(int, config["exponents"])),
                "stats": deepcopy(config["stats"]),
            }
        return cls(int(value["input_bits"]), float(value["input_delta"]),
                   int(value["input_exponent"]), float(value["training_input_abs_max"]), configs)


def fit_quantization(train_X, weights, weight_bits=WEIGHT_BITS, input_bits=INPUT_BITS):
    """Fit solely from training epochs and their trained CSP filters."""
    train_X = _finite(train_X, "train_X")
    if train_X.ndim != 3 or train_X.shape[1] != CHANNELS:
        raise ValueError("Training input must have shape (trials, 22, time)")
    input_bits = _bits(input_bits)
    maximum = float(np.max(np.abs(train_X)))
    delta, exponent = power_of_two_scale(maximum, input_bits)
    choices = tuple(_bits(bits) for bits in weight_bits)
    if not choices or len(set(choices)) != len(choices):
        raise ValueError("Weight bit choices must be nonempty and unique")
    configs = {}
    for bits in choices:
        quantized, deltas, stats = quantize_weights(weights, bits)
        configs[bits] = {
            "accumulator_bits": input_bits + bits + 4,
            "integers": _readonly(quantized), "deltas": _readonly(deltas),
            "exponents": tuple(stats["exponents"]), "stats": stats,
        }
    return FrozenQuantization(input_bits, delta, exponent, maximum, configs)


def _integer_array(values, name, minimum, maximum):
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or array.size == 0:
        raise ValueError(f"{name} must contain integers")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{name} is outside its signed domain")
    return array.astype(np.int64, copy=False)


def dot22_integer(qX, qW, accumulator_bits, *, weight_bits=16):
    """Accumulate channel order 0..21, checking every signed partial sum.

    The generic arithmetic checker permits the full signed weight domain,
    including its most negative endpoint.  Fitted weights use the narrower
    symmetric domain specified by quantize_weights.
    """
    accumulator_bits = _bits(accumulator_bits, "accumulator_bits", maximum=63)
    weight_bits = _bits(weight_bits, "weight_bits")
    qX = _integer_array(qX, "qX", -32768, 32767)
    qW = _integer_array(qW, "qW", -(1 << (weight_bits - 1)), (1 << (weight_bits - 1)) - 1)
    if qX.ndim != 3 or qX.shape[1] != CHANNELS:
        raise ValueError("qX must have shape (trials, 22, time)")
    if qW.ndim != 2 or qW.shape[1] != CHANNELS:
        raise ValueError("qW must have shape (components, 22)")
    lower, upper = -(1 << (accumulator_bits - 1)), (1 << (accumulator_bits - 1)) - 1
    total = np.zeros((qX.shape[0], qW.shape[0], qX.shape[2]), dtype=np.int64)
    partial_min = partial_max = 0
    for channel in range(CHANNELS):
        total += qX[:, channel, :][:, None, :] * qW[:, channel][None, :, None]
        minimum, maximum = int(total.min()), int(total.max())
        partial_min, partial_max = min(partial_min, minimum), max(partial_max, maximum)
        if minimum < lower or maximum > upper:
            raise OverflowError(f"Channel {channel} partial sum [{minimum}, {maximum}] exceeds signed {accumulator_bits}")
    bounds = [32768 * sum(abs(int(value)) for value in row) for row in qW]
    return total, {
        "accumulator_bits": accumulator_bits, "accepted_terms_per_dot": CHANNELS,
        "dots": int(total.size), "partials_checked": int(total.size * CHANNELS),
        "partial_min": partial_min, "partial_max": partial_max,
        "max_abs_partial": max(abs(partial_min), abs(partial_max)),
        "conservative_bounds_by_component": bounds, "overflow_count": 0,
        "rounding_within_dot": "none", "overflow_policy": "reject",
    }


def verify_dot22_bigint(qX, qW, actual, accumulator_bits):
    """Independently check every dot using scalar Python integers.

    The production implementation accumulates whole arrays channel by channel.
    This checker visits one trial/time vector, then one component, and uses
    Python integers for every multiply and partial sum.  It never calls the
    production dot function, and also rejects missing/extra output positions.
    """
    inputs = _integer_array(qX, "qX", -32768, 32767)
    weights = _integer_array(qW, "qW", -32768, 32767)
    output = _integer_array(actual, "actual", -(1 << 62), (1 << 62) - 1)
    accumulator_bits = _bits(accumulator_bits, "accumulator_bits", maximum=63)
    if inputs.ndim != 3 or inputs.shape[1] != CHANNELS or weights.ndim != 2 or weights.shape[1] != CHANNELS:
        raise ValueError("Bigint oracle requires exactly 22 terms per dot")
    expected_shape = (inputs.shape[0], weights.shape[0], inputs.shape[2])
    if output.shape != expected_shape:
        raise ValueError("Bigint oracle output dimensions do not match every dot")
    lower, upper = -(1 << (accumulator_bits - 1)), (1 << (accumulator_bits - 1)) - 1
    weight_rows = weights.tolist()
    partial_min = partial_max = 0
    for trial in range(inputs.shape[0]):
        # tolist() creates genuine Python ints, including the negative endpoint.
        sample_rows = inputs[trial].T.tolist()
        for time, samples in enumerate(sample_rows):
            for component, row in enumerate(weight_rows):
                total = 0
                for sample, weight in zip(samples, row, strict=True):
                    total += sample * weight
                    partial_min, partial_max = min(partial_min, total), max(partial_max, total)
                    if not lower <= total <= upper:
                        raise OverflowError("Bigint oracle found a partial accumulator overflow")
                if total != int(output[trial, component, time]):
                    raise ArithmeticError(f"Bigint oracle mismatch at trial={trial}, component={component}, time={time}")
    return {
        "oracle": "independent_scalar_python_bigint", "bitexact": True,
        "dots_compared": int(output.size), "partials_checked": int(output.size * CHANNELS),
        "partial_min": partial_min, "partial_max": partial_max, "overflow_count": 0,
    }


def project_float(X, weights):
    X, weights = _finite(X, "X"), _finite(weights, "weights")
    if X.ndim != 3 or X.shape[1] != CHANNELS:
        raise ValueError("X must have shape (trials, 22, time)")
    if weights.ndim != 2 or weights.shape[1] != CHANNELS:
        raise ValueError("Weights must have shape (components, 22)")
    with np.errstate(over="raise", invalid="raise"):
        result = np.stack([weights @ epoch for epoch in X])
    if not np.isfinite(result).all():
        raise ArithmeticError("Projection is nonfinite")
    return result


def reconstruct_projection(integers, delta_x, delta_w):
    """Restore the original CSP units after the exact integer dot product."""
    values = _integer_array(integers, "integers", -(1 << 62), (1 << 62) - 1)
    deltas = _finite(delta_w, "delta_w")
    if values.ndim != 3 or deltas.ndim != 1 or values.shape[1] != len(deltas):
        raise ValueError("Projection/scale component dimensions differ")
    scales = [float(delta_x), *deltas.tolist()]
    exponents = []
    for scale in scales:
        if not math.isfinite(scale) or scale <= 0 or math.frexp(scale)[0] != 0.5:
            raise ValueError("Reconstruction requires exact positive power-of-two scales")
        exponents.append(math.frexp(scale)[1] - 1)
    shifts = np.asarray([exponents[0] + value for value in exponents[1:]], dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        result = np.ldexp(values.astype(np.float64), shifts[None, :, None])
    if not np.isfinite(result).all():
        raise ArithmeticError("Reconstructed projection is nonfinite")
    return result


def log_power(projected, power_floor=POWER_FLOOR):
    """Natural log of mean-square power; this does not subtract sample means."""
    projected = _finite(projected, "projected")
    if projected.ndim != 3:
        raise ValueError("Projected values must have shape (trials, components, time)")
    power_floor = float(power_floor)
    if not math.isfinite(power_floor) or power_floor <= 0:
        raise ValueError("power_floor must be finite and strictly positive")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        powers = np.mean(np.square(projected), axis=2)
        features = np.log(np.maximum(powers, power_floor))
    if not np.isfinite(features).all():
        raise ArithmeticError("Power features are nonfinite")
    return features, {
        "definition": "ln(max(mean(projected**2), power_floor))",
        "power_floor": power_floor, "floor_hits": int((powers < power_floor).sum()),
        "feature_count": int(powers.size), "power_min": float(powers.min()),
        "power_max": float(powers.max()),
    }
