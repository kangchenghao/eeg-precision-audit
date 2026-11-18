# First pilot contract — 2026-11-15

This is an offline numerical and arithmetic-kernel audit. It is not a complete
fixed-point BCI, a CPU design, a clinical evaluation, or a claim of a new CSP or
quantization algorithm. Parameters below are fixed before evaluation scores.

## Dataset and split

- BNCI2014_001 / BCI Competition IV-2a, all nine participants. Each participant's
  T session trains their own model; E is held out. No adaptation or model search.
- Only class 1 (left hand) and class 2 (right hand), 72 trials each per session.
  Include artifact-marked trials; preserve their flags for descriptive strata.
  Do not select participants, trials, channels, or widths from E performance.
- First 22 channels are EEG; three EOG channels are not classifier inputs.
  MAT amplitudes are converted from microvolts to volts once, following the
  official loader convention. An independently fitted microvolt pipeline is a
  software unit-invariance check, not another candidate in the results table.
- Skip calibration runs by empty trial arrays, not by assuming nine runs.
  Each file must contain six task runs, 48 trials per task run, and 72 per class.
  A04T has a different number of calibration runs.
- A MAT trial index is one-based. Start at trial-1 and select offsets [625,1375),
  exactly 750 samples. The cue's +2 s offset comes from the published protocol;
  it is not an independently measured timestamp in the MAT array. This window
  overlaps the cue presentation: do not call this pure intention decoding.

## Shared floating-point processing

- Sampling 250 Hz. SciPy firwin: 63 taps (order 62), Hamming window, cutoff
  [8,30] Hz, bandpass, scale=True. These cutoffs are approximately -6 dB points,
  not ideal passband edges. Save the actual coefficients and frequency response.
- SciPy lfilter, causal float64, continuous within each run, zero initial state
  at run start. Do not reset at trials, shift windows to compensate delay, use
  filtfilt, or allow a window to cross a run. Group delay is 31/250 = 124 ms;
  the three-second observation window and compute time are separate quantities.
- MNE CSP: n_components=4, reg=0.1, log=True, cov_est='epoch',
  transform_into='average_power', norm_trace=False, rank='full',
  component_order='alternate'. No average-reference projection. Record empirical
  training rank; 22 input channels do not by themselves prove rank 22.
- Features are ln(max(mean(Y**2),1e-30)), not demeaned log variance. The same
  floor applies to every variant; report every floor hit. Exported filters must
  reproduce MNE features with atol=rtol=1e-9 in ordinary positive-power cases.
- Scikit-learn LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto'). Train
  only on floating-point T features, retain identical coefficients for all
  variants, and verify the exported score and tie rule against sklearn.

## The only quantized stage: CSP projection

X has shape (trial,22,time), W=(4,22), and Y[j,t]=sum(W[j,k]*X[k,t]).
Input width is 16; weight widths are predeclared as 4,8,12,16, with 12 the primary
engineering configuration and 4 a stress condition. There is no E-based winner.

Define Q_b=2**(b-1)-1. From all selected T windows (including artifact flags),
take M_x=max(abs(X)); from the trained W take each row's M_w. For each maximum
choose the smallest binary power Delta with M/Delta<=Q_b. Verify that half Delta
does not fit, using exact integer-ratio comparisons, not rounded float log2.
Reject nonpositive/nonfinite maxima or unrepresentable scales.

Round to nearest, ties to even, then saturate input to [-32767,32767] and weights
to [-Q_b,Q_b]. Input has one scale for all channels; weights have one scale per
projection row. Scales are frozen from T. Record E clipping; never recalibrate
or discard trials to hide it. Fitted weights must not clip.

Combined variants compute every product and partial sum exactly in int64, with
an explicit signed accumulator-width check A=16+b+4. No intermediate rounding,
wrapping, or saturation. Reconstruct Y with Delta_x*Delta_w and pass it to the
unchanged floating-point power/log/LDA. Independently check integer results with
Python unbounded integers. Hardware output accumulation is 24/28/32/36 bits.

Per participant report ten rows: float, input16_only, four weight_only variants,
and four combined variants. The input-only and weight-only controls perform
float64 dot products after dequantization: integer partial widths/overflow/RTL
claims are N/A for these controls. All nine participants give exactly 90 rows.

## Arithmetic hardware contract

dot22 has dynamic signed16 x and signed-b w inputs, clk, rst_n, and in_valid.
Every valid rising edge accepts one pair. Twenty-two accepted pairs form a dot.
Gaps preserve partial state; the first accepted pair starts/overwrites the
accumulator. On the 22nd pair, out_valid pulses for one clock and result contains
the sum including that pair. The next rising edge can start the next dot.
result is meaningful only when out_valid is high. Reset asynchronously clears
accumulator, count and valid, discarding incomplete groups; testbench release
is aligned to a nonactive clock edge.

The hardware accepts full signed endpoint values for capability tests, although
the actual quantizer uses symmetric ranges. Four projected components require
88 accepted pairs per input time point. External data/weight storage, supply,
filtering, power/log/LDA, and all CPU logic are excluded from this kernel.

Test extrema, signs, gaps, consecutive groups, all incomplete reset positions,
and fixed-seed synthetic inputs. For each width also test the first two selected
T trials per participant, four components and first 32 time points: 2304 real
dots/width. These are T samples; do not claim E RTL coverage. Deliberate missing
products, 21/23-term grouping, wrong sign extension, one-bit-narrow accumulator,
reset/gap faults and corrupted streamed coefficients must be detected.

Map the same parameterized dynamic-input top with the same Yosys xc7 flags at
each width. Report LUT, FF and DSP primitive counts separately. No constant
person-specific weights, full-system area, timing, power or thermal inference.

## Evidence and limitations

Freeze code/configuration, versions, input hashes and trained numerical models
before reading E for scores; verify those hashes on evaluation. Complete the
synthetic and RTL checks before E evaluation. Preserve failures and use a new
output directory for changed execution sources. Public results may aggregate
private evidence; original and transformed EEG remain local.

Report per-person confusion matrices, balanced accuracy (BA), paired BA changes,
decision-flip directions, unchanged floating-point mistakes, numerical errors,
clipping and overflow. Aggregate participants equally, not time samples. Do not
declare clinical usefulness, no loss, significance, or an acceptable universal
accuracy threshold. A low-scoring participant is still included.

Negative leakage tests use fixed trial IDs: label permutations cannot change
models/scales/predictions; changing evaluation signals cannot change training
artifacts. Separately constructed positive controls test response to signal and
label changes. There is no universal requirement that an arbitrary signal or
label perturbation must change predictions or BA.
