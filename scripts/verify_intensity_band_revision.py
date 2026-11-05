"""Replay band errors from raw measurements and saved kernels, without fitting."""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from run_intensity_band_revision import ROOT, SOURCE, OUT, CENTERS, load_trajectories, read


def main():
    qc, cached = load_trajectories()
    kernels = defaultdict(list)
    for row in read(SOURCE / "normalized_impulse_responses.csv"):
        kernels[row["subject_id"]].append(float(row["normalized_impulse_response"]))
    models = {r["subject_id"]: r for r in read(SOURCE / "participant_predictions.csv")}
    band_rows = read(OUT / "participant_band_metrics.csv")
    primary = {(r["subject_id"], int(r["center_percent"])): r for r in band_rows if r["scenario"] == "pre_peak_rolling"}
    measurement_max_diff, prediction_max_diff, comparisons = 0.0, 0.0, 0
    for sid, q in qc.items():
        raw = read(ROOT / "data/processed" / f"{sid}_data.csv")
        time = np.array([float(r["Time"]) for r in raw])
        vo2 = np.array([float(r["VO2pulm"]) for r in raw])
        workload = np.array([float(r["WorkRate"]) for r in raw])
        actual_windows = sliding_window_view(vo2, 30)
        valid = (np.isfinite(actual_windows) & (actual_windows > 0)).all(axis=1)
        actual30 = actual_windows.mean(axis=1)
        actual30[~valid] = np.nan
        actual30 = np.r_[np.full(29, np.nan), actual30]
        model = models[sid]
        prediction = float(model["gain"]) * np.convolve(workload, kernels[sid])[:len(workload)] + float(model["offset"])
        predicted30 = np.r_[np.full(29, np.nan), sliding_window_view(prediction, 30).mean(axis=1)]
        assert np.allclose(actual30, cached[sid][:, 1], atol=1e-12, rtol=1e-12, equal_nan=True)
        assert np.allclose(predicted30, cached[sid][:, 2], atol=1e-12, rtol=1e-12, equal_nan=True)
        measurement_max_diff = max(measurement_max_diff, float(np.nanmax(np.abs(actual30 - cached[sid][:, 1]))))
        prediction_max_diff = max(prediction_max_diff, float(np.nanmax(np.abs(predicted30 - cached[sid][:, 2]))))
        permitted = (time >= float(q["higher_load_evaluation_start"]) + 29) & (time <= float(q["time_of_full_record_30s_peak"])) & (time <= float(q["last_positive_vo2_time"]))
        permitted &= np.isfinite(actual30) & np.isfinite(predicted30)
        for center in CENTERS:
            ratio = 100 * actual30 / float(q["supplied_vo2peak"])
            selected = permitted & (ratio >= center - 5) & (ratio < center + 5)
            if not selected.any():
                assert (sid, center) not in primary
                continue
            row = primary[sid, center]
            assert int(selected.sum()) == int(row["n_windows"])
            residual = predicted30[selected] - actual30[selected]
            assert abs(float(np.mean(np.abs(residual))) - float(row["MAE"])) < 1e-10
            assert abs(float(residual.mean()) - float(row["Bias"])) < 1e-10
            assert abs(float(100 * np.mean(residual / actual30[selected])) - float(row["relative_bias_percent_measured"])) < 1e-10
            comparisons += 1
    all_summary = read(OUT / "band_summary.csv")
    for row in all_summary:
        participants = [r for r in band_rows if r["scenario"] == row["scenario"] and r["center_percent"] == row["center_percent"]]
        assert len(participants) == int(row["N_subjects"])
        assert abs(np.mean([float(r["MAE"]) for r in participants]) - float(row["mean_subject_MAE"])) < 1e-12
        assert abs(np.mean([float(r["relative_bias_percent_measured"]) for r in participants]) - float(row["mean_subject_relative_bias_percent_measured"])) < 1e-12
    for row in read(OUT / "coverage_and_overflow.csv"):
        assert int(row["total_valid_windows"]) == int(row["windows_in35_to105percent"]) + int(row["windows_below35percent"]) + int(row["windows_at_or_above105percent"])
    hashes = 0
    for path in (OUT / "input_provenance.csv", SOURCE / "input_provenance.csv"):
        for row in read(path):
            assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]
            hashes += 1
    for row in read(OUT / "scaled_target_identity_check.csv"):
        assert abs(float(row["original_nested_R2"]) - float(row["R2_after_scaling_both_targets_and_predictions"])) < 1e-12
        assert row["new_validation"] == "False"
    result = {"source_participants_checked": len(qc), "primary_participant_band_metrics_replayed": comparisons,
              "aggregate_band_summaries_checked": len(all_summary),
              "max_raw_vs_cached_30s_measurement_difference": measurement_max_diff,
              "max_reconstructed_vs_cached_30s_prediction_difference": prediction_max_diff,
              "source_and_parent_hashes_checked": hashes,
              "participant_equal_weighting_verified": True, "no_fit_or_padding_samples_scored": True,
              "target_scaling_identity_verified": True, "models_refitted_for_verification": 0}
    (OUT / "verification_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
