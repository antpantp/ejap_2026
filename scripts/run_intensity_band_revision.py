"""Retrospective intensity-band agreement of the frozen impulse-response model.

No models or previously accepted trajectories are changed. Bands are centered
at 40--90% of supplied VO2peak (width +/-5 percentage points), with a 100% band
for context. Aggregate error gives each covered participant equal weight.
"""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/revision_round1/impulse_protocol_sensitivity"
OUT = ROOT / "results/revision_round1/intensity_band_revision"
SEED = 20261104
RESAMPLES = 5000
CENTERS = tuple(range(40, 101, 10))
SCENARIOS = ("pre_peak_rolling", "full_record_rolling", "pre_peak_nonoverlap",
             "pre_peak_end_minus60", "pre_peak_end_minus120", "pre_peak_label_consistent")


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def load_trajectories():
    qc = {r["subject_id"]: r for r in read(SOURCE / "endpoint_and_target_qc.csv")}
    grouped = {sid: [] for sid in qc}
    with gzip.open(SOURCE / "trajectories.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            grouped[row["subject_id"]].append([
                float(row["time_s"]), float(row["measured_vo2_30s_Lmin"] or "nan"),
                float(row["predicted_vo2_30s_Lmin"] or "nan"), float(row["table_workload_W"]),
            ])
    return qc, {sid: np.array(rows) for sid, rows in grouped.items()}


def scenario_mask(array, q, scenario):
    time, measured, predicted = array[:, 0], array[:, 1], array[:, 2]
    first_end = float(q["higher_load_evaluation_start"]) + 29
    operational_end = float(q["last_positive_vo2_time"])
    peak_end = float(q["time_of_full_record_30s_peak"])
    end = operational_end if scenario == "full_record_rolling" else min(peak_end, operational_end)
    if scenario == "pre_peak_end_minus60":end = min(peak_end, operational_end - 60)
    if scenario == "pre_peak_end_minus120":end = min(peak_end, operational_end - 120)
    mask = (time >= first_end) & (time <= end) & np.isfinite(measured) & np.isfinite(predicted) & (measured > 0)
    if scenario == "pre_peak_nonoverlap":mask &= np.isclose(np.mod(time - first_end, 30), 0)
    if scenario == "pre_peak_label_consistent" and q["label_matches_full_record_30s_peak"] != "True":
        mask[:] = False
    return mask


def summarize_subject(sid, scenario, center, rows, peak):
    measured, predicted = rows[:, 1], rows[:, 2]
    residual = predicted - measured
    return {"subject_id": sid, "scenario": scenario, "center_percent": center,
            "n_windows": len(rows), "first_window_end_s": float(rows[:, 0].min()),
            "last_window_end_s": float(rows[:, 0].max()),
            "mean_measured_vo2": float(measured.mean()), "mean_predicted_vo2": float(predicted.mean()),
            "mean_measured_percent_peak": float(100 * measured.mean() / peak),
            "mean_workload_W": float(rows[:, 3].mean()),
            "MAE": float(np.abs(residual).mean()), "Bias": float(residual.mean()),
            "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "relative_bias_percent_measured": float(100 * np.mean(residual / measured)),
            "absolute_relative_error_percent_measured": float(100 * np.mean(np.abs(residual) / measured))}


def aggregate(rows):
    summary = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        for center in CENTERS:
            part = sorted([r for r in rows if r["scenario"] == scenario and r["center_percent"] == center], key=lambda r: r["subject_id"])
            if not part:continue
            n = len(part)
            indices = np.random.default_rng(SEED + 1000 * scenario_index + center).integers(0, n, (RESAMPLES, n))
            result = {"scenario": scenario, "center_percent": center, "lower_percent_inclusive": center - 5,
                      "upper_percent_exclusive": center + 5, "N_subjects": n,
                      "total_windows": sum(r["n_windows"] for r in part),
                      "median_windows_per_subject": float(np.median([r["n_windows"] for r in part]))}
            for key in ("MAE", "Bias", "RMSE", "relative_bias_percent_measured", "absolute_relative_error_percent_measured"):
                values = np.array([r[key] for r in part])
                result["mean_subject_" + key] = float(values.mean())
                low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
                result[key + "_ci_low"], result[key + "_ci_high"] = float(low), float(high)
            measured = np.array([r["mean_measured_vo2"] for r in part])
            predicted = np.array([r["mean_predicted_vo2"] for r in part])
            result["R2_band_means_retrospective"] = float(1 - np.sum((predicted - measured) ** 2) / np.sum((measured - measured.mean()) ** 2))
            a, p = measured[indices], predicted[indices]
            bootstrap_r2 = 1 - np.sum((p - a) ** 2, axis=1) / np.sum((a - a.mean(axis=1, keepdims=True)) ** 2, axis=1)
            low, high = np.quantile(bootstrap_r2, [0.025, 0.975])
            result["R2_band_means_ci_low"], result["R2_band_means_ci_high"] = float(low), float(high)
            summary.append(result)
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qc, arrays = load_trajectories()
    rows, coverage = [], []
    for sid, array in arrays.items():
        q = qc[sid];peak = float(q["supplied_vo2peak"])
        relative = 100 * array[:, 1] / peak
        for scenario in SCENARIOS:
            mask = scenario_mask(array, q, scenario)
            assigned = np.zeros(len(array), dtype=bool)
            for center in CENTERS:
                band = mask & (relative >= center - 5) & (relative < center + 5)
                assert not (assigned & band).any()
                assigned |= band
                if band.any():rows.append(summarize_subject(sid, scenario, center, array[band], peak))
            low = int((mask & (relative < 35)).sum())
            high = int((mask & (relative >= 105)).sum())
            assert int(mask.sum()) == int(assigned.sum()) + low + high
            coverage.append({"subject_id": sid, "scenario": scenario,
                             "total_valid_windows": int(mask.sum()), "windows_in35_to105percent": int(assigned.sum()),
                             "windows_below35percent": low, "windows_at_or_above105percent": high,
                             "supplied_peak": peak, "observed_record_peak": float(q["full_record_30s_peak"]),
                             "label_matches_record_peak": q["label_matches_full_record_30s_peak"]})
    summary = aggregate(rows)
    contrasts = []
    for scenario in SCENARIOS:
        low = {r["subject_id"]: r for r in rows if r["scenario"] == scenario and r["center_percent"] == 40}
        high = {r["subject_id"]: r for r in rows if r["scenario"] == scenario and r["center_percent"] == 90}
        common = sorted(set(low) & set(high))
        indices = np.random.default_rng(SEED).integers(0, len(common), (RESAMPLES, len(common)))
        record = {"scenario": scenario, "N_paired_subjects": len(common), "paired_subject_ids": json.dumps(common)}
        for name in ("MAE", "relative_bias_percent_measured"):
            difference = np.array([high[s][name] - low[s][name] for s in common])
            a, b = np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])
            record[name + "_90_minus40"] = float(difference.mean())
            record[name + "_difference_ci_low"], record[name + "_difference_ci_high"] = float(a), float(b)
            record[name + "_n_subjects_higher_at90"] = int((difference > 0).sum())
        contrasts.append(record)
    write("participant_band_metrics.csv", rows)
    write("band_summary.csv", summary)
    write("coverage_and_overflow.csv", coverage)
    write("paired90_vs40.csv", contrasts)
    # Algebraic check only, not new all-signals validation at scaled targets.
    primary_file = ROOT / "results/revision_round1/nested_model_selection/nested_selection_predictions.csv"
    primary = [r for r in read(primary_file) if r["group"] == "all_signals"]
    actual = np.array([float(r["actual_vo2peak"]) for r in primary])
    pred = np.array([float(r["predicted_vo2peak"]) for r in primary])
    original_r2 = 1 - np.sum((pred - actual) ** 2) / np.sum((actual - actual.mean()) ** 2)
    identity = []
    for center in range(40, 101, 10):
        fraction = center / 100
        scaled_actual, scaled_pred = fraction * actual, fraction * pred
        value = 1 - np.sum((scaled_pred - scaled_actual) ** 2) / np.sum((scaled_actual - scaled_actual.mean()) ** 2)
        assert abs(value - original_r2) < 1e-12
        identity.append({"fraction_percent": center, "original_nested_R2": float(original_r2),
                         "R2_after_scaling_both_targets_and_predictions": float(value), "new_validation": False})
    write("scaled_target_identity_check.csv", identity)
    sources = [SOURCE / "trajectories.csv.gz", SOURCE / "endpoint_and_target_qc.csv",
               SOURCE / "analysis_summary.json", SOURCE / "participant_predictions.csv",
               SOURCE / "normalized_impulse_responses.csv", SOURCE / "input_provenance.csv",
               primary_file, Path(__file__).resolve()]
    write("input_provenance.csv", [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources])
    requested = [r for r in summary if r["scenario"] == "pre_peak_rolling" and r["center_percent"] <= 90]
    metadata = {"N_source_participants": len(arrays), "band_centers_percent_requested": list(range(40, 100, 10)),
                "band_half_width_percentage_points": 5, "primary_scenario": "pre_peak_rolling",
                "primary_scope": "Fully scored 30-s windows after first workload above80W, up to observed30-s peak and within agreed operational endpoint",
                "bootstrap_unit": "participant, after averaging windows within each participant and band",
                "resamples": RESAMPLES, "seed": SEED, "model_fits_performed": 0,
                "primary_requested_bands": requested,
                "primary_paired90_vs40": next(r for r in contrasts if r["scenario"] == "pre_peak_rolling")}
    (OUT / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
