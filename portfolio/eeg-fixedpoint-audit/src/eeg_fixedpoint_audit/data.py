"""Explicit, session-local preparation of official BNCI IV-2a MAT data.

Only signal preparation lives here. This module neither fits a classifier nor
searches for files or settings. The defaults are a candidate experiment setup,
not a claim that the dataset or a clinical application requires those settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.signal import firwin, lfilter


EEG_CHANNEL_NAMES = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
EOG_CHANNEL_NAMES = ("EOG1", "EOG2", "EOG3")
CLASS_NAMES = ("left hand", "right hand", "feet", "tongue")


@dataclass(frozen=True)
class EpochConfig:
    """All offsets are samples after trial start, using a half-open interval.

    The official protocol places the cue 500 samples after trial start at 250 Hz;
    MAT trial indices alone do not independently establish the cue event.
    Thus the candidate [625, 1375) interval is cue + [0.5, 3.5) seconds.
    It does not add the cue offset a second time.
    """

    fs: float = 250.0
    numtaps: int = 63
    bandpass_hz: tuple[float, float] = (8.0, 30.0)
    window: str = "hamming"
    start_sample: int = 625
    stop_sample: int = 1375
    classes: tuple[int, ...] = (1, 2)

    def __post_init__(self) -> None:
        if not np.isfinite(self.fs) or self.fs <= 0:
            raise ValueError("fs must be finite and positive")
        _positive_int(self.numtaps, "numtaps")
        _nonnegative_int(self.start_sample, "start_sample")
        _positive_int(self.stop_sample, "stop_sample")
        if self.stop_sample <= self.start_sample:
            raise ValueError("stop_sample must be greater than start_sample")
        if len(self.bandpass_hz) != 2:
            raise ValueError("bandpass_hz must contain two cutoffs")
        low, high = self.bandpass_hz
        if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high < self.fs / 2):
            raise ValueError("bandpass cutoffs must satisfy 0 < low < high < fs/2")
        if not isinstance(self.window, str) or not self.window:
            raise ValueError("window must be a nonempty SciPy window name")
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("classes must be a nonempty unique sequence")
        if any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k not in (1, 2, 3, 4) for k in self.classes):
            raise ValueError("classes must contain integer labels from 1 to 4")


@dataclass(frozen=True)
class EpochBatch:
    """Prepared data in original source run/trial order, with no trial rejection."""

    X: np.ndarray
    y: np.ndarray
    trial_ids: tuple[str, ...]
    artifact_mask: np.ndarray
    channel_names: tuple[str, ...]
    metadata: dict[str, Any]
    event_metadata: tuple[dict[str, Any], ...]


def _positive_int(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _integer_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 1 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a real integer vector")
    array = array.reshape(-1)
    if not np.all(np.isfinite(array)) or np.any(array != np.floor(array)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(array < -(2**63)) or np.any(array >= 2**63):
        raise ValueError(f"{name} exceeds signed int64")
    return array.astype(np.int64)


def design_fir(config: EpochConfig | None = None) -> np.ndarray:
    """Use the original SciPy implementation for the configured FIR."""
    config = EpochConfig() if config is None else config
    return np.asarray(firwin(
        numtaps=config.numtaps, cutoff=config.bandpass_hz, fs=config.fs,
        window=config.window, pass_zero=False, scale=True,
    ), dtype=np.float64)


def _coefficients(config: EpochConfig, supplied: Sequence[float] | np.ndarray | None) -> np.ndarray:
    coefficients = design_fir(config) if supplied is None else np.asarray(supplied)
    if coefficients.ndim != 1 or coefficients.size == 0 or not np.issubdtype(coefficients.dtype, np.number) or np.iscomplexobj(coefficients):
        raise ValueError("fir_coefficients must be a nonempty real vector")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("fir_coefficients must be finite")
    return coefficients


def load_session(
    path: str | Path,
    subject_id: int,
    session: str,
    *,
    config: EpochConfig | None = None,
    fir_coefficients: Sequence[float] | np.ndarray | None = None,
    allow_evaluation: bool = False,
    require_official_counts: bool = False,
) -> EpochBatch:
    """Load exactly one named session, filter runs continuously, and cut epochs.

    Evaluation sessions require an explicit opt-in. Callers must freeze their
    training artifacts before using that opt-in; this function cannot establish
    that scientific workflow on their behalf. It never discovers a second file.

    MAT X is interpreted as microvolts following the upstream MOABB BNCI loader.
    Only the first 22 columns enter filtering and output. Empty-trial calibration
    runs are retained in metadata and skipped as classifier input. A nonzero
    artifact flag is preserved, never used to reject a trial here.
    """
    _positive_int(subject_id, "subject_id")
    if subject_id > 9:
        raise ValueError("BNCI2014_001 subject_id must be between 1 and 9")
    if session not in ("T", "E"):
        raise ValueError("session must be exactly 'T' or 'E'")
    if session == "E" and not allow_evaluation:
        raise ValueError("Evaluation session access requires allow_evaluation=True after training is frozen")
    source = Path(path)
    named = re.fullmatch(r"A(0[1-9])([TE])\.mat", source.name)
    if named and (int(named.group(1)) != subject_id or named.group(2) != session):
        raise ValueError("Filename subject/session does not match explicit arguments")
    config = EpochConfig() if config is None else config
    if not isinstance(config, EpochConfig):
        raise ValueError("config must be an EpochConfig")
    coefficients = _coefficients(config, fir_coefficients)
    try:
        loaded = loadmat(source, struct_as_record=False, squeeze_me=True,
                         variable_names=["data"], verify_compressed_data_integrity=True)
    except (OSError, TypeError, ValueError, NotImplementedError) as error:
        raise ValueError(f"Unable to read BNCI MAT session {source.name}: {error}") from error
    if "data" not in loaded:
        raise ValueError("MAT file has no 'data' variable")
    runs = np.atleast_1d(loaded["data"]).reshape(-1)
    if len(runs) == 0:
        raise ValueError("MAT data has no runs")

    epochs: list[np.ndarray] = []
    labels: list[int] = []
    artifact_mask: list[bool] = []
    events: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []
    required = ("X", "trial", "y", "fs", "classes", "artifacts")
    for run_index, run in enumerate(runs):
        if any(not hasattr(run, name) for name in required):
            raise ValueError(f"Run {run_index} lacks required fields {required}")
        signals = np.asarray(run.X)
        if signals.ndim != 2 or signals.shape[0] == 0 or signals.shape[1] != 25 or not np.issubdtype(signals.dtype, np.number) or np.iscomplexobj(signals):
            raise ValueError(f"Run {run_index}: X must be a real samples-by-25 array")
        if not np.all(np.isfinite(signals)):
            raise ValueError(f"Run {run_index}: X contains nonfinite samples")
        rate = np.asarray(run.fs)
        if rate.size != 1 or not np.issubdtype(rate.dtype, np.number) or np.iscomplexobj(rate) or not np.isfinite(rate.item()) or rate.item() != config.fs:
            raise ValueError(f"Run {run_index}: sampling rate differs from config.fs")
        trial = _integer_vector(run.trial, f"Run {run_index} trial")
        y = _integer_vector(run.y, f"Run {run_index} y")
        flags = _integer_vector(run.artifacts, f"Run {run_index} artifacts")
        if len(trial) != len(y) or len(trial) != len(flags):
            raise ValueError(f"Run {run_index}: trial/y/artifacts lengths differ")
        if np.any(trial < 1) or np.any(trial > signals.shape[0]) or np.any(np.diff(trial) <= 0):
            raise ValueError(f"Run {run_index}: trial indices must be increasing valid one-based positions")
        if not set(np.unique(y)).issubset({1, 2, 3, 4}):
            raise ValueError(f"Run {run_index}: labels must lie in 1..4")
        if not set(np.unique(flags)).issubset({0, 1}):
            raise ValueError(f"Run {run_index}: artifact flags must be 0 or 1")
        names = tuple(str(value).strip().lower().replace("_", " ") for value in np.atleast_1d(run.classes).reshape(-1))
        if len(names) != 4 or names[:2] != CLASS_NAMES[:2] or names[2] not in ("feet", "both feet") or names[3] != "tongue":
            raise ValueError(f"Run {run_index}: unexpected class-name order")
        starts = trial - 1 + config.start_sample
        stops = trial - 1 + config.stop_sample
        if np.any(starts < 0) or np.any(stops > signals.shape[0]):
            raise ValueError(f"Run {run_index}: epoch would extend outside its source run")
        selected = np.isin(y, config.classes)
        run_summaries.append({
            "source_run_index": run_index, "samples": int(signals.shape[0]),
            "trial_count": int(len(trial)), "selected_trials": int(np.count_nonzero(selected)),
            "artifact_flagged_trials": int(np.count_nonzero(flags)),
            "selected_artifact_flagged_trials": int(np.count_nonzero(selected & (flags != 0))),
            "source_label_counts": {str(k): int(np.count_nonzero(y == k)) for k in (1, 2, 3, 4)},
            "source_eeg_min_microvolts": float(np.min(signals[:, :22])),
            "source_eeg_max_microvolts": float(np.max(signals[:, :22])),
            "source_eeg_max_abs_microvolts": float(np.max(np.abs(np.asarray(signals[:, :22], dtype=np.float64)))),
            "role": "task" if len(trial) else "calibration_no_trials",
        })
        if len(trial) == 0:
            continue
        eeg_v = np.asarray(signals[:, :22], dtype=np.float64) * 1e-6
        filtered = lfilter(coefficients, [1.0], eeg_v, axis=0)
        if not np.all(np.isfinite(filtered)):
            raise ValueError(f"Run {run_index}: filtering produced nonfinite samples")
        for trial_index in np.flatnonzero(selected):
            start, stop = int(starts[trial_index]), int(stops[trial_index])
            trial_id = f"s{subject_id:02d}/{session}/r{run_index:02d}/t{trial_index + 1:04d}"
            epochs.append(filtered[start:stop, :].T.copy())
            labels.append(int(y[trial_index]))
            artifact_mask.append(bool(flags[trial_index]))
            events.append({
                "trial_id": trial_id, "subject_id": int(subject_id), "session": session,
                "source_run_index": run_index, "source_trial_index": int(trial_index),
                "trial_onset_sample_zero_based": int(trial[trial_index] - 1),
                "epoch_start_sample_zero_based": start, "epoch_stop_sample_exclusive": stop,
                "artifact": bool(flags[trial_index]),
            })
    if not epochs:
        raise ValueError("No trials match the configured classes")
    with source.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    coefficient_sha = hashlib.sha256(np.asarray(coefficients, dtype="<f8").tobytes()).hexdigest()
    metadata: dict[str, Any] = {
        "source_filename": source.name, "source_sha256": source_sha,
        "subject_id": int(subject_id), "session": session,
        "source_units": "microvolts", "output_units": "volts", "source_to_output_scale": 1e-6,
        "config": asdict(config), "filter": "scipy.signal.lfilter; continuous within each run; zero initial state per run",
        "fir_coefficients": coefficients.tolist(), "fir_coefficients_sha256_float64_le": coefficient_sha,
        "fir_numtaps": int(len(coefficients)), "fir_order": int(len(coefficients) - 1),
        "supplied_fir_coefficients": fir_coefficients is not None,
        "channel_names": list(EEG_CHANNEL_NAMES), "excluded_channels": list(EOG_CHANNEL_NAMES),
        "artifact_policy": "preserve every selected trial; flags are metadata, not a rejection mask",
        "runs": run_summaries, "trial_order": "source run order, then original trial order; no label sorting",
        "parameter_status": "caller must freeze configuration before held-out evaluation",
    }
    batch = EpochBatch(
        X=np.ascontiguousarray(np.stack(epochs), dtype=np.float64),
        y=np.asarray(labels, dtype=np.int64),
        trial_ids=tuple(event["trial_id"] for event in events),
        artifact_mask=np.asarray(artifact_mask, dtype=np.bool_),
        channel_names=EEG_CHANNEL_NAMES, metadata=metadata, event_metadata=tuple(events),
    )
    if require_official_counts:
        validate_official_session(batch)
    metadata["official_counts_validated"] = require_official_counts
    return batch


def validate_official_session(batch: EpochBatch) -> None:
    """Check IV-2a task counts, without requiring nine total source runs.

    The explicit option on ``load_session`` keeps small synthetic MAT tests
    possible. Formal data consumers should enable it for every real session.
    """
    runs = batch.metadata["runs"]
    counts = {str(k): sum(run["source_label_counts"][str(k)] for run in runs) for k in (1, 2, 3, 4)}
    if sum(run["role"] == "task" for run in runs) != 6 or counts != {str(k): 72 for k in (1, 2, 3, 4)}:
        raise ValueError(f"Official session must have six task runs and 72 trials per class; observed {counts}")
    if any(run["trial_count"] != 48 for run in runs if run["role"] == "task"):
        raise ValueError("Each official task run must contain exactly 48 trials")
    configured_classes = batch.metadata["config"]["classes"]
    if len(batch.y) != 72 * len(configured_classes):
        raise ValueError("Selected official trial count is inconsistent with configured classes")
