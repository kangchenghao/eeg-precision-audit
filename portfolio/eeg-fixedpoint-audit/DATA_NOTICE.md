# Data sources and publication boundaries

This repository contains original analysis code and synthetic tests. It does not
include recordings from participants, cropped EEG trials, quantized recordings,
or an alternative license for the source data.

The experiment uses **BNCI Horizon 2020 dataset 001-2014**, originally released
as **BCI Competition IV dataset 2a** by the Institute for Knowledge Discovery,
Graz University of Technology. The [official dataset index](https://bnci-horizon-2020.eu/database/data-sets)
lists **Creative Commons Attribution-NoDerivatives 4.0 International
(CC BY-ND 4.0)** for this dataset. See the [license deed](https://creativecommons.org/licenses/by-nd/4.0/)
and [legal code](https://creativecommons.org/licenses/by-nd/4.0/legalcode.en).
The repository's code license does not replace the data license. We keep raw
and transformed participant signals outside this repository and provide a
downloader, synthetic fixtures and aggregate experimental results instead.

The official MAT links resolve to the university host
`https://lampx.tugraz.at/~bci/database/001-2014/`. The 18 files total
**779,873,919 bytes**. `data_sources.json` records their URLs, byte counts and
SHA256 hashes calculated from our downloads on 2026-11-15. These are locally
established reproducibility hashes, not checksums independently published by
the original data provider. The downloader verifies content before renaming
each `.part` file and never parses labels.

Download to an explicitly chosen directory outside this repository:

```text
python scripts/download_data.py --output YOUR_PRIVATE_DATA_DIRECTORY
```

`--dry-run` only displays size; `--verify-only` checks already downloaded files
without network access. Downloads use at most three workers and three attempts
per file. On Windows a Python TLS certificate-chain failure can use the system
trust path through a hidden PowerShell process; certificate verification is
never disabled.

Protocol facts come from the [original Graz dataset description](https://bnci-horizon-2020.eu/database/data-sets/001-2014/description.pdf).
It describes nine participants, two sessions on different days, 22 EEG and
three EOG channels at 250 Hz, and 288 trials per session. Artifact flags are
expert trial-level annotations, not sample-level clean neural ground truth.
EOG channels serve artifact analysis; any EOG-only diagnostic must be reported
separately from formal EEG classification.

MAT units and the trial-index conversion are cross-checked against the
[upstream MOABB BNCI converter](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/base.py)
and [dataset loader](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py).
The converter treats `X` as microvolts, converts it to volts for MNE, and places
trial events at `trial - 1`. The dataset interval is two to six seconds after
that event. Our configurable epoch offsets are relative to trial start; do not
add the cue offset twice. The cue offset is based on the official protocol and
loader convention, not an independently measured cue event in these MAT files.
The candidate filter has 63 taps (order 62); its actual response must be reported
rather than treating the 8 and 30 Hz design cutoffs as ideal passband edges.
The current code does not claim to reproduce the
original competition's complete continuous-output or artifact-correction rules.

Please cite the dataset provider and:

Tangermann, M., et al. (2012). *Review of the BCI Competition IV.* Frontiers in
Neuroscience, 6, 55. [doi:10.3389/fnins.2012.00055](https://doi.org/10.3389/fnins.2012.00055).

Public code and aggregate results do not establish clinical utility, online
control performance, or endorsement by the dataset provider.
