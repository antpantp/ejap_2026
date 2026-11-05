# VO₂peak prediction from PRBS physiological recordings

Research code for estimating VO₂peak from physiological recordings collected during pseudo-random binary sequence (PRBS) exercise. The revised analysis compares Ridge, Elastic Net, and partial least squares (PLS) regression using participant-level nested cross-validation, evaluates sensor groups and cadence removal, and assesses post-training prediction and impulse-response agreement.

The primary implementation is in `evaluation/revision_modeling.py` and `scripts/run_*revision.py`. The notebooks retain the original preparation and historical analyses. They are included without execution outputs; historical notebook figures predate the revised figure terminology. The revised plotting scripts produce the manuscript figures.

## Installation

Use Python 3.12.13 for the revised analysis:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_check.py
```

On Windows, activate with `.venv\Scripts\activate`. To use the notebooks, also install `requirements-notebooks.txt` and run `jupyter lab` from this directory. Core analysis dependencies are pinned; JupyterLab is an optional interface dependency.

## Reproducing the analyses

See [docs/reproduction.md](docs/reproduction.md) for the execution order, required intermediate files, and differences between the original and revised analyses. See [docs/data_format.md](docs/data_format.md) for the expected input structure and units, and [docs/software_versions.md](docs/software_versions.md) for recorded environments.

Participant data, outcome tables, fitted models, and generated results are not distributed in this code package. Reproducing the study's numerical results requires the study inputs and regenerated historical intermediates. The synthetic smoke check verifies installation and core model selection only; it does not reproduce the study results. Data access terms must be supplied by the study authors separately.

## Contents

- `evaluation/`: training-fold preprocessing and nested model-selection implementation.
- `scripts/`: revised analyses, verification, harmonized plotting, and a synthetic smoke check.
- `notebooks/`: code-only copies of original data preparation and historical analyses.
- `data/_variable_explanations.csv`: variable dictionary; no participant measurements.
- `docs/`: reproduction, software, input-format, and manual-upload instructions.

Outputs are written locally under `results/`, `images/`, and `models/`; these paths are ignored by Git. Do not include generated participant-level files in a public upload.

## Citation and licensing

The code is released under the [MIT License](LICENSE). Add the final article citation and repository URL before publishing. See [docs/publication_decisions.md](docs/publication_decisions.md).
