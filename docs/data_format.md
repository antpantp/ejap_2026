# Input files and units

The code package contains variable metadata only. Use study inputs under the terms set by the data custodians. This document specifies structure, not permission to redistribute data.

## Raw preparation

The original preparation notebooks read participant XLSX exports and consult the semicolon-separated `data/_variable_explanations.csv` dictionary. Keep original column names and units. The notebooks separate target variables from time-series measurements. Follow-up preprocessing contains export-specific identifier handling; review its mapping when applying the code to other exports.

## Prepared baseline inputs

For each baseline participant identifier such as `CT_01`, provide:

- `data/processed/CT_01_data.csv`: comma-separated time-series table, one sample per row.
- `data/processed/CT_01_targets.csv`: comma-separated table with `VO2peak_abs` in its first data row, in L/min.

The baseline feature extractor requires `Time` in seconds and `WorkRate` in W. Signal names must match the dictionary and predefined groups in notebook 5. `VO2pulm` is required by impulse-response analysis, in L/min. Inputs for that analysis are sampled at 1 s; its implementation asserts the recorded sampling interval. Keep original time coordinates: the descriptor window is 300–1000 s, including endpoints, and it conditions on recorded work rates of 30 or 80 W.

Each available signal yields the mean and sample standard deviation at 30 W, mean and sample standard deviation at 80 W, and the difference in means (80 minus 30 W). A level requires at least 20 finite observations. Unavailable descriptors remain missing and are handled within training folds. `RpM` is pedalling cadence in rev/min.

## Prepared follow-up inputs

Use the analogous files under `data/processed_follow_up/`, with participant identifiers matched to baseline. The follow-up runner accepts `WR_cortex` as a fallback when `WorkRate` is absent. Do not convert oxygen-uptake values to mL/min or normalize them by body mass for the absolute target analyses.

## Dataset-specific assumptions

Scripts retain the original cohort and named participant checks, fixed figure examples, and minimum sensor-group sample-size rules. Data must satisfy those assumptions for exact study reproduction. New datasets require a separate review of those settings. The dictionary describes exported fields; it should not be interpreted as independent confirmation of acquisition details or instrument definitions.
