"""Render cadence, sensor-comparison, and coefficient-stability review figures."""

import csv
import re
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_round1/sensor_robustness"


def read(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def short_name(name):
    return (re.sub(r"_(mean|std)_wr(30|80)", lambda m: f" {dict(mean='mean', std='SD')[m[1]]} ({m[2]} W)",
                   name.replace("_mean_delta_wr80_minus_wr30", " Δ (80–30 W)"))
                .replace("absO2Hb_Rx1", "O₂Hb").replace("absHHb_Rx1", "HHb")
                .replace("VO2_movMean", "Smoothed VO₂").replace("VO2pulm", "Pulmonary VO₂"))


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    cadence = read("cadence_ablation_comparisons.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    y = np.arange(len(cadence))
    axes[0].scatter([float(r["reference_MAE"]) for r in cadence], y - 0.08, label="With cadence", color="#17679c")
    axes[0].scatter([float(r["comparator_MAE"]) for r in cadence], y + 0.08, label="Without cadence", color="#c77b36")
    axes[0].set_yticks(y, [procedure_label(r["procedure"]) for r in cadence]);axes[0].invert_yaxis()
    axes[0].set(xlabel="MAE (L/min)", title="Paired cadence removal, N=32")
    axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.15))
    for i, row in enumerate(cadence):
        axes[1].hlines(i, float(row["ci95_low"]), float(row["ci95_high"]), color="#17679c", lw=2)
        axes[1].scatter(float(row["delta_MAE_reference_minus_comparator"]), i, color="#17679c")
    axes[1].set_yticks(y, [procedure_label(r["procedure"]) for r in cadence]);axes[1].invert_yaxis()
    axes[1].axvline(0, ls="--", color="black", lw=0.8)
    axes[1].set(xlabel="ΔMAE, with − without cadence (L/min)", title="Negative values favor retaining cadence")
    for ax in axes:ax.grid(axis="x", alpha=0.2)
    save_figure(fig, OUT / "cadence_ablation.png", dpi=170);save_figure(fig, OUT / "cadence_ablation.pdf");plt.close(fig)

    contrasts = read("paired_sensor_comparisons.csv")
    groups = ["gas_exchange_ventilation", "no_gas_exchange", "hemo_plus_nirs", "hemodynamics", "hr_plus_nirs", "hr_plus_bf", "nirs"]
    labels = ["Gas exchange + ventilation", "No gas exchange", "Hemodynamics + NIRS", "Hemodynamics (n = 30)", "HR + NIRS", "HR + breathing frequency", "NIRS"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
    for ax, procedure in zip(axes, ("Nested family selection", "Ridge")):
        rows = {r["reduced_group"]: r for r in contrasts if r["procedure"] == procedure and r["reference_includes_cadence"] == "False"}
        for i, group in enumerate(groups):
            row = rows[group]
            ax.hlines(i, float(row["seven_contrast_adjusted_low"]), float(row["seven_contrast_adjusted_high"]), color="0.65", lw=2, label="Adjusted 95% CI" if i == 0 else None)
            ax.hlines(i, float(row["ci95_low"]), float(row["ci95_high"]), color=COLORS[procedure_label(procedure)], lw=4, label="95% CI" if i == 0 else None)
            ax.scatter(float(row["delta_MAE_reference_minus_comparator"]), i, color=COLORS[procedure_label(procedure)], s=23)
        ax.set_yticks(range(len(groups)), labels);ax.invert_yaxis()
        ax.axvline(0, color="black", ls="--", lw=0.8);ax.grid(axis="x", alpha=0.2)
        ax.set(xlabel="ΔMAE, full − reduced model (L/min)", title=procedure)
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.16))
    fig.suptitle("Cadence-controlled sensor contrasts; negative values favor the full panel", fontsize=12)
    save_figure(fig, OUT / "paired_sensor_comparisons.png", dpi=170);save_figure(fig, OUT / "paired_sensor_comparisons.pdf");plt.close(fig)

    rows = read("ridge_bootstrap_coefficient_stability.csv")[:15]
    loso = {r["feature"]: r for r in read("ridge_loso_coefficient_stability.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        axes[0].hlines(i, float(row["coef_2_5pct_when_retained"]), float(row["coef_97_5pct_when_retained"]), color=COLORS["Fixed Ridge"], lw=2)
        axes[0].scatter(float(row["reference_loso_mean_signed_coef"]), i, color=COLORS["Fixed Ridge"], s=23)
    labels = [short_name(r["feature"]) for r in rows]
    axes[0].set_yticks(y, labels);axes[0].invert_yaxis();axes[0].axvline(0, color="black", ls="--", lw=0.8)
    axes[0].set(xlabel="Standardized coefficient (L/min per SD)", title="Mean LOSO weights and bootstrap percentile ranges")
    axes[1].scatter([100 * float(loso[r["feature"]]["top15_fraction_all_resamples"]) for r in rows], y - 0.08, color="0.6", label="LOSO folds")
    axes[1].scatter([100 * float(r["top15_fraction_all_resamples"]) for r in rows], y + 0.08, color=COLORS["Fixed Ridge"], label="Bootstrap refits (n = 500)")
    axes[1].set_yticks(y, labels);axes[1].invert_yaxis();axes[1].set_xlim(-2, 102)
    axes[1].set(xlabel="Top-15 membership (%)", title="Ranking depends on the sampled participants")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1))
    for ax in axes:ax.grid(axis="x", alpha=0.2)
    save_figure(fig, OUT / "ridge_coefficient_stability.png", dpi=170);save_figure(fig, OUT / "ridge_coefficient_stability.pdf");plt.close(fig)


if __name__ == "__main__":
    main()
