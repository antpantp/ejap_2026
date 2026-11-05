# Reproduction guide

Run commands from the repository root. Full study runs require the original study data; those data and all precomputed outputs are intentionally absent. The scripts retain study-specific participant identifiers and cohort sizes as analysis configuration and assertions, without distributing measurements or fitted parameters. They are research reproduction scripts rather than a general-purpose library for arbitrary cohorts.

## 1. Prepare local data

Place authorized baseline XLSX exports in `data/raw/` and follow-up exports in `data/raw_follow_up_subset/`. Open JupyterLab at the repository root and execute these notebooks in order:

1. `1_explore_raw_data.ipynb`: baseline CSV recordings and target files.
2. `1_1_explore_follow_up_data.ipynb`: follow-up CSV recordings and target files.
3. `1_2_vo2max_change_and_heterogeneity.ipynb`: paired baseline/follow-up table under `results/`.

The revised runners can also use already prepared input CSVs following `data_format.md`. In that case, the paired table and historical comparison files below must still be generated locally where required. No sample participant files are supplied.

## 2. Generate historical comparison intermediates

Execute these historical notebooks in order:

1. `2_impulse_response_baseline.ipynb`
2. `3_vo2peak_naive_baseline.ipynb`
3. `5_ablation_study_feature_models.ipynb`
4. `6_nested_cV_model_selection.ipynb`
5. `7_e1_zero_shot_followup_inference.ipynb`
6. `8_e2_pre_post_error_gap.ipynb`
7. `9_e3_delta_agreement.ipynb`
8. `10_e4_baseline_to_delta_modeling.ipynb`

Notebook 4 is an obsolete method and is not distributed. These historical analyses are provided for provenance and intermediate generation; use the scripts below for the revised results. In particular, the revised implementation fits feature availability, imputation, unit rescaling, and standardization using training subsets and bounds PLS components in inner training splits.

`run_nested_revision.py` reads `results/ablation_best_models_summary.csv` and the corresponding `ablation_<group>_<family>_loso_predictions.csv` for historical comparisons. `run_followup_revision.py` reads `results/e1_zero_shot_<group>_<family>_predictions.csv` and the paired table `results/q1_q2_vo2peak_abs_paired_table.csv`. Omitting these steps causes missing-file errors even when prepared recordings are available. These comparison outputs are generated from local inputs, not bundled with the public code.

## 3. Revised analyses

```sh
python scripts/run_nested_revision.py --jobs 4
python scripts/verify_nested_revision.py
python scripts/run_sensor_robustness.py
python scripts/verify_sensor_robustness.py
python scripts/run_followup_revision.py
python scripts/verify_followup_revision.py
python scripts/run_impulse_protocol_sensitivity.py
python scripts/run_intensity_band_revision.py
python scripts/verify_intensity_band_revision.py
```

The baseline run uses outer leave-one-participant-out evaluation and inner five-fold model-family/hyperparameter selection. It exports fixed-family comparisons and selected-family predictions. The sensor analysis depends on baseline fitted artifacts, and the follow-up analysis depends on both baseline and sensor artifacts. The intensity-band analysis depends on impulse-response trajectories and the baseline prediction summary. Bootstrap analyses and nested cross-validation can be computationally expensive; no execution-time guarantee is provided.

Original raw export time coordinates, protocol windows, random seeds, grids, and scientific calculations are retained. Internal procedure keys such as `ElasticNet` and `Nested family selection` are preserved for compatibility; displayed figure terminology is harmonized separately.

The code retains the historical label-consistency diagnostic for provenance. Following author confirmation of the target value, that sensitivity line is omitted from the revised intensity figure.

## 4. Revised figures

```sh
python scripts/render_nested_revision.py
python scripts/render_sensor_robustness.py
python scripts/render_followup_revision.py
python scripts/render_intensity_band_revision.py
python scripts/render_publication_figures.py
```

The last command renders manuscript and supplementary assets to `results/revision_round1/overleaf_package/figures/`, including vector PDF/SVG and 600-dpi PNG. The full-participant trajectory diagnostic is a multipage PDF. All rendering uses locally regenerated results. The plotting entry point omits manuscript assembly, the private manuscript archive, and reviewer correspondence.

## Validation limits

The public snapshot is checked by compilation, import checks, notebook-output checks, archive/content checks, and the included synthetic core-model smoke check. Full numerical reproduction has not been rerun on this cleaned snapshot. A successful smoke check does not establish reproduction of the study results.
