"""Render review figures and LaTeX tables from the completed nested-CV results."""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/revision_round1/nested_model_selection"
GROUPS = ["all_signals", "no_gas_exchange", "hemo_plus_nirs", "gas_exchange_ventilation",
          "hr_plus_nirs", "hemodynamics", "hr_plus_bf", "nirs"]
LABELS = ["All signals", "No gas exchange", "Hemodynamics + NIRS", "Gas exchange + ventilation",
          "HR + NIRS", "Hemodynamics", "HR + breathing frequency", "NIRS"]


def read(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    rows = read("summary_metrics.csv")
    nested = {r["group"]: r for r in rows if r["procedure"] == "Nested family selection"}
    counts = {(r["group"], r["family"]): int(r["n_outer_folds_selected"]) for r in read("family_selection_counts.csv")}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.45, 1, 1]})
    y = np.arange(len(GROUPS))
    estimate = np.array([float(nested[g]["R2"]) for g in GROUPS])
    lower = np.array([float(nested[g]["R2_ci_low"]) for g in GROUPS])
    upper = np.array([float(nested[g]["R2_ci_high"]) for g in GROUPS])
    axes[0].errorbar(estimate, y, xerr=np.vstack([estimate - lower, upper - estimate]), fmt="o", color="#17679c", capsize=3, label="Nested selection")
    axes[0].axvline(0, color="black", ls="--", lw=0.8)
    axes[0].set_yticks(y, LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Cross-validated R² (95% CI)")
    axes[0].set_title("Performance of the family-selection procedure")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14))
    axes[0].grid(axis="x", alpha=0.2)

    procedures = ["Nested family selection", "Ridge", "PLS", "ElasticNet"]
    all_rows = {r["procedure"]: r for r in rows if r["group"] == "all_signals"}
    for i, procedure in enumerate(procedures):
        row = all_rows[procedure]
        point, lo, hi = (float(row[k]) for k in ("R2", "R2_ci_low", "R2_ci_high"))
        axes[1].errorbar(point, i, xerr=[[point - lo], [hi - point]], fmt="o",
                         color=COLORS[procedure_label(procedure)], capsize=3)
    axes[1].set_yticks(range(4), ["Nested selection", "Fixed Ridge", "Fixed PLS", "Fixed Elastic Net"])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", ls="--", lw=0.8)
    axes[1].set_title("All signals: different procedures")
    axes[1].set_xlabel("Cross-validated R² (95% CI)")
    axes[1].grid(axis="x", alpha=0.2)

    left = np.zeros(len(GROUPS))
    for family, color in [("Ridge", COLORS["Fixed Ridge"]), ("ElasticNet", COLORS["Fixed Elastic Net"]), ("PLS", COLORS["Fixed PLS"])]:
        n = np.array([counts[g, family] for g in GROUPS])
        axes[2].barh(y, n, left=left, color=color, label=family.replace("ElasticNet", "Elastic Net"))
        left += n
    axes[2].set_yticks(y, LABELS)
    axes[2].invert_yaxis()
    axes[2].set_title("Family chosen inside training folds")
    axes[2].set_xlabel("Outer folds (n)")
    axes[2].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3)
    save_figure(fig, OUT / "nested_selection_summary.png", dpi=180)
    save_figure(fig, OUT / "nested_selection_summary.pdf")
    plt.close(fig)

    lines = [r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Sensor group & $N$ & $N_F$ & MAE & RMSE & $R^2$ & $R^2$ 95\% CI \\", r"\hline"]
    for group, label in zip(GROUPS, LABELS):
        row = nested[group]
        lines.append(f"{label} & {row['N']} & {row['n_candidate_features']} & "
                     f"{float(row['MAE']):.3f} & {float(row['RMSE']):.3f} & {float(row['R2']):.3f} & "
                     f"[{float(row['R2_ci_low']):.3f}, {float(row['R2_ci_high']):.3f}] " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}",
                  "% MAE/RMSE: L/min. Family and hyperparameters selected within each outer training fold.",
                  "% N_F is the input schema size; availability filtering is fitted inside every training split.",
                  "% RpM is confirmed as cycle-ergometer pedalling cadence (rev/min) and retained."])
    (OUT / "nested_selection_table.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{llrrrr}", r"\hline",
             r"Sensor group & Fixed family & $N$ & MAE & RMSE & $R^2$ \\", r"\hline"]
    for group, label in zip(GROUPS, LABELS):
        for family in ("Ridge", "ElasticNet", "PLS"):
            row = next(r for r in rows if r["group"] == group and r["procedure"] == family)
            lines.append(f"{label} & {family} & {row['N']} & {float(row['MAE']):.3f} & "
                         f"{float(row['RMSE']):.3f} & {float(row['R2']):.3f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", "% Exploratory family-specific evaluations; no winner is selected from these test results."])
    (OUT / "fixed_family_supplement_table.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
