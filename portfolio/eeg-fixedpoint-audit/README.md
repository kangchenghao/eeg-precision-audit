# EEG Fixed-Point Audit

A reproducible audit of how integer precision in EEG spatial projection affects
numerical errors, classifier decisions, and a small hardware arithmetic kernel.
It connects fixed-point implementation checks to a held-out decoding task.

脑电定点可靠性实验：既检查算对了没有，也检查数值误差是否改变最终判断。

```text
22 EEG channels → causal FIR → CSP spatial projection → log mean power → LDA
                   float64     only quantized stage       float64     float64
```

The algorithms are established methods. This project contributes a reproducible
paired audit, explicit numerical contracts, and independently checked arithmetic.
It does not claim a new CSP algorithm or a complete fixed-point BCI.

## Frozen experiment

The dataset is **BNCI2014_001 / BCI Competition IV-2a**. All nine participants
are included. Each participant's T session trains their own model; their E
session is evaluated after the models, configuration, and execution sources are
frozen. There is no adaptation or held-out search for a winning configuration.

- Class 1: left hand; class 2: right hand. Each session contributes 144 selected
  trials per participant. Artifact-marked trials remain included.
- All 22 EEG channels, in their original order. EOG channels are excluded.
- Causal float64 filtering uses a 63-tap Hamming FIR at 250 Hz, with 8/30 Hz
  design cutoffs. Filtering is continuous within each run.
- A fixed 750-sample window covers offsets `[625, 1375)` after trial start.
  The three-second observation window overlaps the displayed cue; this is a
  cued task evaluation. FIR group delay is 124 ms, separate from computation.
- Mature MNE CSP supplies four spatial filters. Features are
  `ln(max(mean(Y**2), 1e-30))`, **not demeaned log variance**.
- Scikit-learn LDA is trained on floating-point T features. Its coefficients
  remain identical for every precision variant.

Only the CSP projection is quantized. Input is signed 16-bit with a symmetric
`[-32767, 32767]` quantizer. Weight widths are **4, 8, 12, and 16 bits**.
**12 bits was declared the primary configuration before evaluation; 4 bits is
a stress condition.** All configurations are reported.

Input and per-filter weight scales are powers of two fitted exclusively from
T. Rounding is nearest, ties to even; saturation is counted. Combined variants
use exact integer products and partial sums, with 24/28/32/36-bit accumulator
checks. Filtering, power, logarithms, and LDA remain floating point.

Each participant has ten rows: one float, one input-only, four weight-only,
and four combined variants. **Nine participants × ten variants = 90 rows.**
The input-only and weight-only controls use dequantized float64 dot products;
integer accumulator and RTL claims apply to the combined variants.

The complete parameters and acceptance rules are in [CONTRACT.md](CONTRACT.md).

## Results from the frozen pilot

Evaluation completed on 2026-11-15. Balanced accuracy (BA) is first calculated
per participant, then averaged with equal participant weights. Flips count
changed decisions relative to float across the same 1,296 held-out trials.
Changes in BA are percentage points, not relative percentages.

| Variant | Mean BA | Change from float | Decision flips |
|---|---:|---:|---:|
| Float | 74.3056% | 0.0000 pp | 0 |
| Input16 only | 74.3056% | 0.0000 pp | 0 |
| Weight4 only | 69.5988% | −4.7068 pp | 289 |
| Weight8 only | 74.2284% | −0.0772 pp | 23 |
| Weight12 only | 74.2284% | −0.0772 pp | 1 |
| Weight16 only | 74.3056% | 0.0000 pp | 0 |
| Input16 × weight4 | 69.5988% | −4.7068 pp | 289 |
| Input16 × weight8 | 74.2284% | −0.0772 pp | 23 |
| **Input16 × weight12 — primary** | **74.2284%** | **−0.0772 pp** | **1** |
| Input16 × weight16 | 74.3056% | 0.0000 pp | 0 |

The primary configuration changed one correct float decision into an incorrect
decision. The 8-bit combined variant changed 23 decisions: 12 correct-to-wrong
and 11 wrong-to-correct. Their equal aggregate BA hides this difference.
Zero flips in the 16-bit experiment describes this finite test set; it does not
establish a universal lossless precision.

The shared input16 quantization clipped **20 of 21,384,000 evaluated input
elements** after rounding. Those trials were retained and the training scales were preserved.
No power-floor hits or integer partial overflows occurred in the formal matrix.
The float model's lowest participant BA was 50%, which is retained in the report.

Post-evaluation, all 1,296 predictions at each width were compared between the
weight-only and combined controls; those decisions matched at each width.
Their continuous numerical outputs still differ. The 4-bit result describes
this power-of-two scaling rule, not a universal limit of 4-bit implementations.
Trials from the same participant are correlated; 1,296 trials are not 1,296
independent participants.

Public evidence: [aggregate results](results/pilot-2026-11-15/results.json),
[comparison CSV](results/pilot-2026-11-15/comparison.csv),
[accuracy and decision flips](results/pilot-2026-11-15/accuracy-and-flips.png),
and [mapped resources](results/pilot-2026-11-15/resources.png).

### Arithmetic and hardware evidence

- **58 synthetic tests passed**, including rounding, saturation, independent
  arithmetic, unit invariance, leakage checks, and stale/incomplete evidence
  rejection before any E loading.
- Every combined E output was checked against scalar Python unbounded integers:
  **15,552,000 dot products** across the four widths. This is software checking.
- RTL simulation checked **2,304 selected T dot products plus 96 synthetic dot
  products per width**, including gaps and resets. All nine deliberate-fault
  checks were detected. T sampling is separate from full E software checking.

Yosys 0.70 mapped the same parameterized, dynamic-input `dot22` with the same
`synth_xilinx -family xc7 -noiopad -noclkbuf` flags:

| Weight bits | Accumulator bits | Logic LUTs | FFs | DSP48E1 | CARRY4 |
|---:|---:|---:|---:|---:|---:|
| 4 | 24 | 54 | 30 | 1 | 8 |
| 8 | 28 | 62 | 34 | 1 | 9 |
| **12** | **32** | **70** | **38** | **1** | **10** |
| 16 | 36 | 78 | 42 | 1 | 11 |

These are mapped primitive counts for the arithmetic kernel. Coefficient/data
storage, transport, filtering, power/log/LDA, and CPU logic are excluded. The
kernel accepts one pair per valid clock and produces one dot after 22 accepted
pairs. Four components require 88 accepted pairs per input time point. This
does not measure full-pipeline throughput, physical area, power, or maximum
clock frequency. No place-and-route or board experiment was performed.

The first formal mapping attempt failed because YoWASP did not accept a native
Windows absolute path. That failed record was retained. The runner was changed
to stage the identical source and use relative paths; tests and training were
rerun into new evidence directories before successful RTL/mapping checks and
the first E evaluation. No evaluation score was used to make that correction.
Source hashes detect version mismatches, not malicious rewriting of every
evidence file. RTL simulation and mapped resource counting are separate;
mapped-netlist functional equivalence has not been established.

## Reproduce locally

Run commands from this repository directory. The audited environment uses
**Windows and CPython 3.13**. The included wheel hashes are for that environment.
Icarus Verilog and Yosys are separate prerequisites for the hardware checks;
Vivado and a GPU are not required.

```powershell
py -3.13 -m venv .venv
$Py = '.venv/Scripts/python.exe'
& $Py -m pip install --require-hashes -r requirements-win-py313.lock

$Raw = '<private raw-data directory outside this repository>'
$Private = '<new private evidence directory outside this repository>'
$Iverilog = '<iverilog executable>'
$Vvp = '<vvp executable>'
$Yosys = '<yosys or yowasp-yosys executable>'
```

Obtain the official files after reviewing [DATA_NOTICE.md](DATA_NOTICE.md).
The downloader fetches and verifies bytes without parsing evaluation labels.
All 18 recordings total 779,873,919 bytes. Existing files can be checked with
`--verify-only`.

```powershell
& $Py scripts/download_data.py --output $Raw
```

Keep execution sources, `CONTRACT.md`, and the environment lock unchanged
through the following stages. A changed implementation needs a new evidence
directory; earlier failures and reports are preserved.

**1. Train all nine T models and freeze them.** This creates private models,
`training_manifest.json`, and `rtl-fixtures/s01.npz` through `s09.npz`.

```powershell
& $Py run.py --phase train --raw-dir $Raw --output $Private `
  --contract CONTRACT.md --contract requirements-win-py313.lock
```

**2. Run the independent synthetic suite against the same sources.**

```powershell
$Tests = Join-Path $Private 'synthetic-tests'
& $Py scripts/run_tests.py --output $Tests
```

**3. Verify and map the arithmetic kernel using the frozen T fixtures.**

```powershell
$Rtl = Join-Path $Private 'rtl-checks'
$Fixtures = Join-Path $Private 'rtl-fixtures'
& $Py scripts/verify_rtl.py --build $Rtl --fixtures $Fixtures `
  --iverilog $Iverilog --vvp $Vvp --yosys $Yosys
```

**4. Evaluate the frozen 90-row matrix.** The entry point first checks all nine
models, source/contract hashes, passing tests, four-width RTL evidence, and its
binding to the exact T fixtures. Missing or mismatched evidence prevents E
loading.

```powershell
& $Py run.py --phase evaluate --raw-dir $Raw --output $Private `
  --rtl-evidence (Join-Path $Rtl 'results.json') `
  --test-evidence (Join-Path $Tests 'results.json')
```

Full local metrics are written to `evaluation/results.json` and
`evaluation/comparison.csv` under the private evidence directory. Per-trial
predictions, transformed EEG, models, and real RTL vectors remain private.

## Source map and publication boundaries

| Location | Purpose |
|---|---|
| `run.py` | Separate training and frozen-evaluation entry points |
| `src/eeg_fixedpoint_audit/data.py` | Explicit MAT loading, channel selection, causal filtering and windows |
| `src/eeg_fixedpoint_audit/core.py` | Quantization, exact accumulation, independent bigint verification |
| `src/eeg_fixedpoint_audit/pipeline.py` | Mature estimator fitting, exported-rule checks and paired metrics |
| `rtl/` | Standalone arithmetic kernel and cycle-level testbench |
| `tests/` | Synthetic correctness and negative-control tests |
| `scripts/` | Verified downloads, test evidence and RTL automation |

The code is available under [MIT](LICENSE). Dataset rights remain separate;
see [DATA_NOTICE.md](DATA_NOTICE.md) for attribution, provenance and the
CC BY-ND 4.0 boundary. This repository excludes raw or transformed participant
signals and publishes aggregate statistics.

[AI_USAGE.md](AI_USAGE.md) records AI assistance and the project owner's actual
role. [LEARNING.md](LEARNING.md) lists exercises for understanding and modifying
the work. Suggested exercises are not presented as completed contributions.

This offline audit does not establish clinical utility, online control
reliability, statistically significant improvement, or scientific novelty.
