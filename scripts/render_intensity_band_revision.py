"""Standalone intensity-band figures and a manuscript-ready descriptive table."""

import csv
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/revision_round1/intensity_band_revision"


def read(name):
    with (OUT / name).open() as stream:return list(csv.DictReader(stream))


def main():
    summary = read("band_summary.csv")
    primary = [r for r in summary if r["scenario"] == "pre_peak_rolling" and int(r["center_percent"]) <= 90]
    centers = np.array([int(r["center_percent"]) for r in primary])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, value, low, high, title, ylabel in [
        (axes[0], "mean_subject_MAE", "MAE_ci_low", "MAE_ci_high", "Error increases across the measured range", "MAE (L/min)"),
        (axes[1], "mean_subject_relative_bias_percent_measured", "relative_bias_percent_measured_ci_low", "relative_bias_percent_measured_ci_high", "Underestimation is present in the lowest band", "Relative bias (%)"),
    ]:
        values = np.array([float(r[value]) for r in primary])
        lower = np.array([float(r[low]) for r in primary]);upper = np.array([float(r[high]) for r in primary])
        ax.errorbar(centers, values, yerr=np.vstack([values - lower, upper - values]), fmt="o-", color="#17679c", capsize=3, label="Primary analysis (95% CI)")
        for scenario, color, marker, label in [
            ("pre_peak_nonoverlap", "#41977e", "x", "Non-overlapping windows"),
        ]:
            selected = [r for r in summary if r["scenario"] == scenario and int(r["center_percent"]) <= 90]
            ax.plot(centers, [float(r[value]) for r in selected], marker=marker, ls="--", lw=.7, color=color, label=label)
        ax.set(title=title, xlabel="Intensity band (% VO₂peak)", ylabel=ylabel)
        ax.set_xticks(centers);ax.grid(alpha=.15)
    axes[1].axhline(0, color="black", ls="--", lw=.7)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -.17))
    n = [int(r["N_subjects"]) for r in primary]
    axes[2].bar(centers, n, width=6, color="#17679c")
    axes[2].set(ylim=(0, 35), xlabel="Intensity band (% VO₂peak)", ylabel="Participants (n)", title="No missing intensity levels are filled in")
    axes[2].set_xticks(centers)
    save_figure(fig, OUT / "intensity_band_summary.png", dpi=180);save_figure(fig, OUT / "intensity_band_summary.pdf");plt.close(fig)

    band_rows = read("participant_band_metrics.csv")
    ids = sorted(set(r["subject_id"] for r in band_rows))
    matrix = np.full((len(ids), len(centers)), np.nan)
    for row in band_rows:
        if row["scenario"] == "pre_peak_rolling" and int(row["center_percent"]) in centers:
            matrix[ids.index(row["subject_id"]), list(centers).index(int(row["center_percent"]))] = float(row["relative_bias_percent_measured"])
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    cmap = plt.colormaps["RdBu_r"].copy();cmap.set_bad("0.85")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-45, vmax=45)
    ax.set_xticks(range(len(centers)), [f"{c}%" for c in centers]);ax.set_yticks(range(len(ids)), ids)
    ax.set(xlabel="Intensity band (% VO₂peak)", title="Participant-specific signed error; grey = no covered window")
    fig.colorbar(im, ax=ax, label="Relative bias (%)")
    save_figure(fig, OUT / "participant_band_bias.png", dpi=170);save_figure(fig, OUT / "participant_band_bias.pdf");plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for scenario, label, color in [("pre_peak_rolling", "Up to VO₂peak", "#17679c"),
                                    ("full_record_rolling", "Complete recording", "#c77b36")]:
        rows = [r for r in summary if r["scenario"] == scenario and int(r["center_percent"]) <= 90]
        axes[0].plot(centers, [float(r["mean_subject_relative_bias_percent_measured"]) for r in rows], "o-", label=label, color=color)
    axes[0].set(xlabel="Intensity band (% VO₂peak)", ylabel="Relative bias (%)", title="Late falling responses can alter band errors")
    axes[0].legend(fontsize=8);axes[0].axhline(0, color="black", ls="--", lw=.7)
    for row in primary:
        x = int(row["center_percent"]);point = float(row["R2_band_means_retrospective"])
        axes[1].vlines(x, float(row["R2_band_means_ci_low"]), float(row["R2_band_means_ci_high"]), color="#17679c")
        axes[1].scatter(x, point, color="#17679c")
    axes[1].axhline(0, color="black", ls="--", lw=.7)
    axes[1].set(xlabel="Intensity band (% VO₂peak)", ylabel="R² (band means)", title="Retrospective agreement, not all-signals capacity CV")
    for ax in axes:ax.set_xticks(centers);ax.grid(alpha=.15)
    save_figure(fig, OUT / "band_scope_and_agreement.png", dpi=180);save_figure(fig, OUT / "band_scope_and_agreement.pdf");plt.close(fig)

    lines = [r"\begin{tabular}{lrrrr}", r"\hline",
             r"Band (\% VO$_2$peak) & $N$ & MAE (L/min) & Bias (L/min) & Relative bias (\%) \\", r"\hline"]
    for row in primary:
        center = int(row["center_percent"])
        lines.append(f"{center - 5}--{center + 5} & {row['N_subjects']} & {float(row['mean_subject_MAE']):.3f} & "
                     f"{float(row['mean_subject_Bias']):.3f} & {float(row['mean_subject_relative_bias_percent_measured']):.1f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}",
                  "% Bands are half-open, defined by observed 30-s VO2 / supplied peak.",
                  "% Metrics average within each participant first, then give participants equal weight.",
                  "% Primary diagnostic ends at observed peak within the agreed operational endpoint.",
                  "% These are retrospective impulse-response errors, not all-signals VO2peak CV results."])
    (OUT / "intensity_band_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
