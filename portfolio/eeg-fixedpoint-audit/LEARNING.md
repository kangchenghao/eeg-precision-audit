# Reproduce, explain, change

These are suggested exercises, not a record of work already completed by the
project owner.

1. Explain why a small signal RMSE can still flip a classifier decision close
   to its boundary. Find one such case in locally regenerated private results.
2. Compute the largest 22-term signed16-by-signed12 dot product. Explain why a
   32-bit accumulator is sufficient and why one fewer bit can fail.
3. Trace a negative half-way value through nearest-even rounding and symmetric
   saturation. Run the independent rounding tests before editing the quantizer.
4. Explain why mean-square power is different from demeaned variance, and why
   the pipeline must use the same feature definition as the mature reference.
5. Draw the 22 accepted input edges with a gap and a mid-group reset. Predict
   exactly which edge can assert output-valid.
6. Reproduce the floating/input-only/weight-only/combined controls. Do not pick
   settings after inspecting held-out results. A new setting starts a clearly
   labeled exploratory experiment, not a rewrite of the frozen pilot.
7. Explain why an unchanged decision does not imply a correct decision, and why
   mapped LUT/FF/DSP counts do not establish power consumption or chip area.
