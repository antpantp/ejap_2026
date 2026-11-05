"""Recorded-workload sensitivity for the original PRBS impulse-response model.

Run: python scripts/run_impulse_protocol_sensitivity.py

The original estimation window/settings are retained. The last positive VO2
sample is the operational endpoint, following the author's exhaustion assumption.
Endpoints 60/120 s earlier are sensitivity scenarios, not confirmed exhaustion
times. Only PRBS-window VO2 is used for fitting. Requires NumPy and Matplotlib.
"""

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from audit_revision_step1 import numeric_columns, original_impulse, read_csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_round1/impulse_protocol_sensitivity"
SEED = 20261104
BOOTSTRAPS = 5000
LAG = 150
AVERAGE = 30
OFFSETS = (0, 60, 120)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rolling_mean(values, valid=None):
    """Trailing 30-sample mean, defined only for fully supported windows."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) if valid is None else (valid & np.isfinite(values))
    counts = np.convolve(valid.astype(float), np.ones(AVERAGE), "valid")
    means = np.convolve(np.where(valid, values, 0), np.ones(AVERAGE) / AVERAGE, "valid")
    means[counts < AVERAGE] = np.nan
    return np.r_[np.full(AVERAGE - 1, np.nan), means]


def fit_original(time, workload, vo2):
    valid = np.isfinite(time) & np.isfinite(workload) & np.isfinite(vo2)
    aligned = time - time[valid].min()
    mask = valid & (aligned >= 300) & (aligned <= 1000)
    x, y = workload[mask], vo2[mask]
    cc = np.correlate(y - y.mean(), x - x.mean(), mode="full")
    phi = {lag: cc[len(x) - 1 + lag] for lag in range(-LAG, LAG + 1)}
    h = np.zeros(LAG)
    for lag in range(LAG):
        if lag == 0:
            h[lag] = 2 * (phi[0] - phi[-20]) / (1 + 1 / 15)
        elif lag < 20:
            h[lag] = (phi[lag] - phi[-lag]) / ((0.5 + lag / 20 * (1 - lag / 40)) * (1 + 1 / 15))
        else:
            h[lag] = (phi[lag] - phi[-lag]) / (1 + 1 / 15)
    h /= h.sum()
    calibration_x = np.convolve(x, h)[:len(x)]
    gain, offset = np.linalg.lstsq(np.column_stack([calibration_x, np.ones(len(x))]), y, rcond=None)[0]
    return h, float(gain), float(offset), mask


def predict(workload, h, gain, offset):
    return gain * np.convolve(workload, h)[:len(workload)] + offset


def pair_metrics(actual, pred):
    residual = pred - actual
    return {"MAE": float(np.abs(residual).mean()), "RMSE": float(np.sqrt(np.mean(residual ** 2))),
            "R2": float(1 - np.sum(residual ** 2) / np.sum((actual - actual.mean()) ** 2)),
            "Bias": float(residual.mean())}


def bootstrap_metric_rows(label, actual, pred, indices):
    point = pair_metrics(actual, pred)
    a, p = actual[indices], pred[indices]
    e = p - a
    boot = {"MAE": np.abs(e).mean(axis=1), "RMSE": np.sqrt((e ** 2).mean(axis=1)),
            "R2": 1 - np.sum(e ** 2, axis=1) / np.sum((a - a.mean(axis=1, keepdims=True)) ** 2, axis=1),
            "Bias": e.mean(axis=1)}
    result = {"scenario": label, "N": len(actual), "n_underestimates": int((pred < actual).sum())}
    for key, value in point.items():
        result[key] = value
        result[key + "_ci_low"], result[key + "_ci_high"] = map(float, np.quantile(boot[key], [0.025, 0.975]))
    return result


def draw_subject(ax, sid, data, show_labels=True):
    time, workload, vo2, measured30, predicted30, end, target = data
    ax.plot(time, np.where(vo2 > 0, vo2, np.nan), color="0.75", lw=0.5, label="Measured VO₂ (1 s)")
    ax.plot(time, measured30, color="#17679c", lw=1.4, label="Measured VO₂ (30-s mean)")
    ax.plot(time, predicted30, color=COLORS["Predicted"], lw=1.5, label="Predicted VO₂ (30-s mean)")
    ax.axhline(target, color="#17679c", ls=":", lw=1, label="VO₂peak")
    ax.axvspan(120, 820, color="#d8e8d2", alpha=0.45, label="Estimation window")
    ax.axvspan(end, time[-1], color="#f1d4d1", alpha=0.5, label="After end of exercise")
    ax.axvline(end, color="#8b3333", lw=1)
    ax.set_xlim(-180, time[-1])
    ax.set_ylim(0, max(target, float(np.nanmax(measured30)), float(np.max(predicted30[149:]))) * 1.12)
    ax.set_title(sid)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VO₂ (L/min)")
    ax.grid(alpha=0.2)
    if show_labels:
        ax.legend(fontsize=7, loc="upper left")
    other = ax.twinx()
    other.plot(time, workload, color="0.3", ls="--", alpha=0.5, lw=0.6)
    other.set_ylabel("Work rate (W)", color="0.4")
    other.set_ylim(0, 460)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data/processed").glob("CT_*_data.csv"))
    records, trajectories, qc, kernels, provenance = [], {}, [], [], []
    trace_rows = []
    verification = {"original_prediction_max_abs_difference": 0.0,
                    "post_fit_vo2_mutation_max_parameter_difference": 0.0,
                    "future_workload_mutation_max_past_prediction_difference": 0.0,
                    "n_uniform_1hz_recordings": 0}
    shared_workload = None
    for path in files:
        sid = path.stem.removesuffix("_data")
        target_path = path.parent / f"{sid}_targets.csv"
        headers, rows = read_csv(path)
        cols = numeric_columns(headers, rows)
        _, targets = read_csv(target_path)
        target = float(targets[0]["VO2peak_abs"])
        time, workload, vo2 = (cols[c] for c in ("Time", "WorkRate", "VO2pulm"))
        assert np.allclose(np.diff(time), 1) and np.isfinite(workload).all() and np.isfinite(vo2).all()
        verification["n_uniform_1hz_recordings"] += 1
        if shared_workload is None:
            shared_workload = workload.copy()
        else:
            assert np.array_equal(shared_workload, workload), "Workload templates differ"
        h, gain, offset, fit_mask = fit_original(time, workload, vo2)
        assert np.isfinite(h).all() and abs(h.sum() - 1) < 1e-10
        legacy = predict(np.linspace(80, 300, 301), h, gain, offset)
        old_pred = original_impulse(cols)
        diff = abs(float(legacy.max()) - old_pred)
        verification["original_prediction_max_abs_difference"] = max(verification["original_prediction_max_abs_difference"], diff)
        assert diff < 1e-10
        changed_vo2 = vo2.copy()
        changed_vo2[time > time[fit_mask].max()] += 100
        other_h, other_gain, other_offset, _ = fit_original(time, workload, changed_vo2)
        parameter_difference = max(np.max(np.abs(other_h - h)), abs(other_gain - gain), abs(other_offset - offset))
        verification["post_fit_vo2_mutation_max_parameter_difference"] = max(verification["post_fit_vo2_mutation_max_parameter_difference"], float(parameter_difference))
        assert parameter_difference < 1e-10

        full_pred = predict(workload, h, gain, offset)
        measured30 = rolling_mean(vo2, valid=vo2 > 0)
        predicted30 = rolling_mean(full_pred)
        end = float(time[vo2 > 0][-1])
        assert np.all(vo2[time > end] == 0), "Unexpected values in trailing VO2 padding"
        changed_workload = workload.copy()
        changed_workload[time > end] += 100
        changed_prediction = predict(changed_workload, h, gain, offset)
        causal_diff = float(np.max(np.abs(changed_prediction[time <= end] - full_pred[time <= end])))
        verification["future_workload_mutation_max_past_prediction_difference"] = max(verification["future_workload_mutation_max_past_prediction_difference"], causal_diff)
        assert causal_diff < 1e-10
        start = float(time[workload > 80][0])
        peak_index = int(np.nanargmax(measured30))
        record = {"subject_id": sid, "actual_vo2peak": target, "gain": gain, "offset": offset,
                  "legacy_ramp300_instantaneous_max": float(legacy.max()),
                  "legacy_ramp300_30s_max": float(np.nanmax(rolling_mean(legacy)))}
        q = {"subject_id": sid, "provided_peak_power": targets[0].get("PeakPower", ""),
             "fit_time_start": float(time[fit_mask].min()), "fit_time_end": float(time[fit_mask].max()),
             "higher_load_evaluation_start": start, "last_positive_vo2_time": end,
             "last_positive_vent_time": float(time[cols["Vent"] > 0][-1]),
             "last_file_time": float(time[-1]), "n_trailing_vo2_zeros": int((time > end).sum()),
             "workload_at_operational_end": float(workload[time == end][0]),
             "time_of_full_record_30s_peak": float(time[peak_index]),
             "full_record_30s_peak": float(measured30[peak_index]),
             "supplied_vo2peak": target,
             "label_matches_full_record_30s_peak": bool(np.isclose(target, measured30[peak_index], atol=1e-10, rtol=1e-10)),
             "seconds_from_peak_to_last_positive_vo2": end - float(time[peak_index])}
        for seconds in OFFSETS:
            scenario_end = end - seconds
            mask = (time >= start + AVERAGE - 1) & (time <= scenario_end) & np.isfinite(measured30)
            assert mask.any()
            pred_peak = float(predicted30[mask].max())
            residual = predicted30[mask] - measured30[mask]
            key = f"recorded_end_minus{seconds}s_30s_max"
            record[key] = pred_peak
            record[f"trajectory_mae_minus{seconds}s"] = float(np.abs(residual).mean())
            record[f"trajectory_bias_minus{seconds}s"] = float(residual.mean())
            record[f"observed_segment_peak_minus{seconds}s"] = float(measured30[mask].max())
            q[f"n_evaluation_windows_minus{seconds}s"] = int(mask.sum())
            q[f"truncation_omits_full_record_peak_minus{seconds}s"] = bool(time[peak_index] > scenario_end)

        # An independent input-profile control: the same terminal workload, 290 W.
        # 290 W is the last scheduled step below the original 300 W endpoint.
        fast_times = np.arange(int(np.ceil((290 - 80) / (44 / 60))) + 1)
        fast290 = np.minimum(80 + (44 / 60) * fast_times, 290)
        fast290_history = np.r_[np.full(LAG - 1, 80.0), fast290]
        fast290_pred = predict(fast290_history, h, gain, offset)[LAG - 1:]
        common_step_end = float(time[workload == 290][-1])
        common_step_mask = (time >= start + AVERAGE - 1) & (time <= common_step_end)
        record["fast_ramp290_30s_max"] = float(np.nanmax(rolling_mean(fast290_pred)))
        record["step_template290_30s_max"] = float(predicted30[common_step_mask].max())
        record["common_endpoint290_difference"] = record["step_template290_30s_max"] - record["fast_ramp290_30s_max"]
        record["fast290_duration_s"] = int(fast_times[-1])
        record["step290_final_time"] = common_step_end
        record["common290_simulation_exceeds_operational_end"] = bool(common_step_end > end)
        records.append(record)
        qc.append(q)
        trajectories[sid] = (time, workload, vo2, measured30, predicted30, end, target)
        for lag, value in enumerate(h):
            kernels.append({"subject_id": sid, "lag_s": lag, "normalized_impulse_response": float(value)})
        for i in range(len(time)):
            trace_rows.append({"subject_id": sid, "time_s": float(time[i]), "table_workload_W": float(workload[i]),
                               "measured_vo2_Lmin": float(vo2[i]), "measured_vo2_30s_Lmin": float(measured30[i]) if np.isfinite(measured30[i]) else "",
                               "predicted_vo2_Lmin": float(full_pred[i]), "predicted_vo2_30s_Lmin": float(predicted30[i]) if np.isfinite(predicted30[i]) else "",
                               "after_operational_end": bool(time[i] > end)})
        for input_path in (path, target_path):
            provenance.append({"path": str(input_path.relative_to(ROOT)), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()})

    actual = np.array([r["actual_vo2peak"] for r in records])
    bootstrap_indices = np.random.default_rng(SEED).integers(0, len(records), (BOOTSTRAPS, len(records)))
    scenarios = ["legacy_ramp300_instantaneous_max", "legacy_ramp300_30s_max"] + [f"recorded_end_minus{s}s_30s_max" for s in OFFSETS]
    aggregate, comparisons, trajectory_metrics = [], [], []
    legacy_error = np.abs(np.array([r[scenarios[0]] for r in records]) - actual)
    consistent_error = np.abs(np.array([r[scenarios[1]] for r in records]) - actual)
    for scenario in scenarios:
        pred = np.array([r[scenario] for r in records])
        aggregate.append(bootstrap_metric_rows(scenario, actual, pred, bootstrap_indices))
        if scenario.startswith("recorded"):
            error = np.abs(pred - actual)
            for name, reference_error in [("legacy_instantaneous", legacy_error), ("legacy_30s", consistent_error)]:
                delta = error - reference_error
                lo, hi = np.quantile(delta[bootstrap_indices].mean(axis=1), [0.025, 0.975])
                comparisons.append({"scenario": scenario, "reference": name, "N": len(records),
                                    "delta_MAE_recorded_minus_reference": float(delta.mean()),
                                    "delta_MAE_ci_low": float(lo), "delta_MAE_ci_high": float(hi),
                                    "n_smaller_absolute_error": int((delta < 0).sum())})
    for seconds in OFFSETS:
        row = {"endpoint_seconds_earlier": seconds, "N": len(records)}
        for name in ("mae", "bias"):
            values = np.array([r[f"trajectory_{name}_minus{seconds}s"] for r in records])
            low, high = np.quantile(values[bootstrap_indices].mean(axis=1), [0.025, 0.975])
            row[f"mean_subject_{name}"] = float(values.mean())
            row[f"mean_subject_{name}_ci_low"] = float(low)
            row[f"mean_subject_{name}_ci_high"] = float(high)
        row["n_truncations_omitting_observed_peak"] = sum(q[f"truncation_omits_full_record_peak_minus{seconds}s"] for q in qc)
        trajectory_metrics.append(row)
    common_diff = np.array([r["common_endpoint290_difference"] for r in records])
    common_low, common_high = np.quantile(common_diff[bootstrap_indices].mean(axis=1), [0.025, 0.975])
    write_csv("participant_predictions.csv", records)
    write_csv("endpoint_and_target_qc.csv", qc)
    write_csv("peak_agreement_metrics.csv", aggregate)
    write_csv("paired_error_changes.csv", comparisons)
    write_csv("trajectory_metrics.csv", trajectory_metrics)
    label_consistent = np.array([q["label_matches_full_record_30s_peak"] for q in qc])
    label_sensitivity = []
    for scenario in scenarios:
        pred = np.array([r[scenario] for r in records])
        label_sensitivity.append({"scenario": scenario,
                                  "subset": "provided label agrees with available full-record 30s maximum",
                                  "N": int(label_consistent.sum()),
                                  **pair_metrics(actual[label_consistent], pred[label_consistent]),
                                  "n_underestimates": int((pred[label_consistent] < actual[label_consistent]).sum())})
    write_csv("target_label_sensitivity.csv", label_sensitivity)
    write_csv("normalized_impulse_responses.csv", kernels)
    with gzip.GzipFile(filename=str(OUT / "trajectories.csv.gz"), mode="wb", mtime=0) as compressed:
        import io
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)
    for source in (Path(__file__).resolve(), ROOT / "scripts/audit_revision_step1.py", ROOT / "notebooks/2_impulse_response_baseline.ipynb"):
        provenance.append({"path": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write_csv("input_provenance.csv", provenance)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False})
    with PdfPages(OUT / "all_participant_trajectories.pdf") as pdf:
        for sid, data in trajectories.items():
            fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
            draw_subject(ax, sid, data)
            fig.suptitle("Recorded-workload diagnostic; operational endpoint assumes last positive VO₂ = exhaustion", fontsize=10)
            style_figure(fig)
            pdf.savefig(fig)
            plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for ax, sid in zip(axes.flat, ["CT_01", "CT_02", "CT_03", "CT_07", "CT_19", "CT_30"]):
        draw_subject(ax, sid, trajectories[sid], show_labels=False)
    fig.suptitle("Endpoint and trajectory examples — all 32 participants are in the accompanying PDF", fontsize=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.045), ncol=3, fontsize=9)
    save_figure(fig, OUT / "trajectory_examples.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    colors = ["0.5", "#c77b36", "#17679c", "#41977e", "#9276af"]
    labels = ["Synthetic ramp, instantaneous", "Synthetic ramp, 30 s", "Recorded protocol",
              "Recorded protocol, end −60 s", "Recorded protocol, end −120 s"]
    values = [r["MAE"] for r in aggregate]
    axes[0].barh(range(5), values, color=colors)
    axes[0].set_yticks(range(5), labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (L/min)")
    for label, scenario, color in [("Synthetic ramp", scenarios[0], colors[1]),
                                    ("Recorded protocol", scenarios[2], colors[2])]:
        axes[1].scatter(actual, [r[scenario] for r in records], label=label, color=color, s=23)
    bounds = [1.5, 4.8]
    axes[1].plot(bounds, bounds, "k--", lw=1)
    axes[1].set(xlabel="VO₂peak (L/min)", ylabel="Predicted VO₂peak (L/min)")
    axes[1].legend(fontsize=7)
    axes[2].scatter([r["fast_ramp290_30s_max"] for r in records], [r["step_template290_30s_max"] for r in records], color="#17679c", s=23)
    axes[2].plot([1.5, 3.5], [1.5, 3.5], "k--", lw=1)
    axes[2].set(xlabel="Predicted VO₂peak, fast ramp to 290 W (L/min)", ylabel="Predicted VO₂peak, step profile to 290 W (L/min)")
    for ax in axes:
        ax.grid(alpha=0.15)
    save_figure(fig, OUT / "sensitivity_summary.png", dpi=180)
    plt.close(fig)
    summary = {"N": len(records), "bootstrap_replicates": BOOTSTRAPS, "bootstrap_seed": SEED,
               "endpoint_rule": "last strictly positive VO2pulm sample, assumed exhaustion per author; trailing zeros excluded",
               "evaluation_start_rule": "first table workload >80 W; 30-s windows must lie fully after this start",
               "fit_recorded_time_interval": [120, 820],
               "peak_comparison_status": "retrospective, conditional on shared workload template and operational endpoints; not a PRBS-only capacity predictor",
               "common_endpoint290_mean_step_minus_fast_Lmin": float(common_diff.mean()),
               "common_endpoint290_ci_low": float(common_low), "common_endpoint290_ci_high": float(common_high),
               "common_endpoint290_n_positive_differences": int((common_diff > 0).sum()),
               "n_supplied_labels_disagreeing_with_full_record_30s_max": sum(not q["label_matches_full_record_30s_peak"] for q in qc),
               "n_participants_operational_end_before_common290_simulation_end": sum(r["common290_simulation_exceeds_operational_end"] for r in records),
               "verification": verification}
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
