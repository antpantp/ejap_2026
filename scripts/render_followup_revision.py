"""Render follow-up diagnostics from saved revised predictions and summaries."""

import csv
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_round1/followup_revision"


def read(name):
    with (OUT / name).open() as stream:return list(csv.DictReader(stream))


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    frozen = read("frozen_model_predictions.csv")
    primary = sorted([r for r in frozen if r["group"] == "all_signals" and r["selected_by_baseline_inner_cv"] == "True"], key=lambda r: r["subject_id"])
    ridge = sorted([r for r in frozen if r["group"] == "all_signals" and r["family"] == "Ridge"], key=lambda r: r["subject_id"])
    direct = sorted([r for r in read("e4_direct_change_predictions.csv") if r["group"] == "all_signals" and r["selected_by_inner_cv"] == "True"], key=lambda r: r["subject_id"])
    benchmark = sorted(read("benchmark_subject_predictions.csv"), key=lambda r: r["subject_id"])
    y_post = np.array([float(r["actual_post"]) for r in primary]);y_pre = np.array([float(r["actual_pre"]) for r in primary])
    p_post = np.array([float(r["pred_post"]) for r in primary]);p_pre = np.array([float(r["pred_pre"]) for r in primary])
    observed_delta = y_post - y_pre
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(y_post, p_post, color="#17679c", label="Nested selection")
    ax.scatter(y_post, [float(r["pred_post"]) for r in ridge], color=COLORS["Fixed Ridge"], marker="x", label="Fixed Ridge")
    ax.plot([2.7, 4.9], [2.7, 4.9], "k--", lw=0.8)
    ax.set(xlabel="Post-training VO₂peak (L/min)", ylabel="Predicted post-training VO₂peak (L/min)", title="E1: post-training estimation, N=15")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for a, b in zip(abs(p_pre - y_pre), abs(p_post - y_post)):
        ax.plot([0, 1], [a, b], color="0.7", lw=0.8)
    ax.scatter(np.zeros(15), abs(p_pre - y_pre), color="#41977e", label="Baseline")
    ax.scatter(np.ones(15), abs(p_post - y_post), color="#17679c", label="Post-training")
    ax.set_xticks([0, 1], ["Baseline", "Post-training"])
    ax.set(ylabel="Absolute error (L/min)", title="E2: same participants and fitted models")
    ax = axes[1, 0]
    ax.scatter(observed_delta, p_post - p_pre, color="#17679c")
    ax.plot([-.5, .8], [-.5, .8], "k--", lw=.8)
    ax.axhline(0, color="0.6", lw=.8);ax.axvline(0, color="0.6", lw=.8)
    ax.set(xlabel="Observed change (L/min)", ylabel="Predicted change (L/min)", title="E3: change tracking requires post-training features")
    ax = axes[1, 1]
    ax.scatter(observed_delta, [float(r["predicted_delta"]) for r in direct], color="#17679c", label="Model prediction")
    ax.scatter(observed_delta, [float(r["other14_mean_change"]) for r in benchmark], color="#c77b36", marker="x", label="Mean change of other participants")
    ax.plot([-.5, .8], [-.5, .8], "k--", lw=.8)
    ax.axhline(0, color="0.6", lw=.8);ax.axvline(0, color="0.6", lw=.8)
    ax.set(xlabel="Observed change (L/min)", ylabel="Predicted change (L/min)", title="E4: prospective change prediction ≈ training mean")
    ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    save_figure(fig, OUT / "followup_summary.png", dpi=170);save_figure(fig, OUT / "followup_summary.pdf");plt.close(fig)

    groups = ["all_signals", "no_gas_exchange", "hemo_plus_nirs", "nirs", "all_signals_without_cadence"]
    labels = ["All signals", "No gas exchange", "Hemodynamics + NIRS", "NIRS", "All signals without cadence"]
    rows = read("e1_post_training_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for p, offset, color in [("Nested family selection", -.1, "#17679c"), ("Ridge", .1, COLORS["Fixed Ridge"])]:
        for i, group in enumerate(groups):
            row = next(r for r in rows if r["group"] == group and r["procedure"] == p)
            ax.hlines(i + offset, float(row["MAE_ci_low"]), float(row["MAE_ci_high"]), color=color)
            ax.scatter(float(row["MAE"]), i + offset, color=color, label=procedure_label(p) if i == 0 else None)
    ax.set_yticks(range(len(groups)), labels);ax.invert_yaxis()
    ax.set(xlabel="Post-training MAE (L/min; 95% CI)", title="Frozen baseline models applied at follow-up, N=15")
    ax.grid(axis="x", alpha=.2);ax.legend(loc="upper left", bbox_to_anchor=(0, -.14), ncol=2)
    save_figure(fig, OUT / "followup_sensor_groups.png", dpi=170);save_figure(fig, OUT / "followup_sensor_groups.pdf");plt.close(fig)

    qc = {r["subject_id"]: r for r in read("frozen_model_qc.csv") if r["group"] == "all_signals" and r["selected_by_baseline_inner_cv"] == "True"}
    ids = [r["subject_id"] for r in primary]
    categories = {"Cadence": [], "Smoothed VO₂": [], "HR": [], "NIRS": [], "Hemodynamics / other": []}
    for sid in ids:
        missing = json.loads(qc[sid]["post_missing_features"])
        counts = {key: 0 for key in categories}
        for n in missing:
            if n.startswith("RpM_"):key = "Cadence"
            elif n.startswith("VO2_movMean_"):key = "Smoothed VO₂"
            elif n.startswith("HR_"):key = "HR"
            elif n.startswith(("absO2Hb", "absHHb", "TSI")):key = "NIRS"
            else:key = "Hemodynamics / other"
            counts[key] += 1
        for key in categories:categories[key].append(counts[key])
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    bottom = np.zeros(len(ids))
    for (label, counts), color in zip(categories.items(), ["#c77b36", "#17679c", "#41977e", "#9276af", "0.6"]):
        ax.bar(ids, counts, bottom=bottom, label=label, color=color)
        bottom += counts
    ax.set(ylabel="Imputed descriptors (of 155)", title="Post-training availability differs from baseline")
    ax.tick_params(axis="x", rotation=45);ax.legend(fontsize=8)
    save_figure(fig, OUT / "followup_missing_features.png", dpi=170);save_figure(fig, OUT / "followup_missing_features.pdf");plt.close(fig)


if __name__ == "__main__":
    main()
