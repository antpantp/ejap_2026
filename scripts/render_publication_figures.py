"""Render manuscript figures from locally regenerated analysis outputs."""
import csv
import gzip
import importlib.util
import os
import shutil
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figure_style import COLORS, procedure_label, save_figure, style_figure
import numpy as np
from matplotlib.figure import Figure
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/revision_round1'
OUT = BASE / 'overleaf_package'
FIG = OUT / 'figures'

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def save(fig, name):
    for extension in ('pdf', 'svg', 'png'):
        save_figure(fig, FIG / f'{name}.{extension}', dpi=600, bbox_inches='tight')
    plt.close(fig)

def export_diagnostics():
    """Re-render existing figures from copied aggregates, preserving originals."""
    real_savefig = Figure.savefig

    def publication_savefig(fig, filename, *args, **kwargs):
        path = Path(filename)
        if path.suffix == '.png':
            kwargs['dpi'] = 600
        result = real_savefig(fig, filename, *args, **kwargs)
        if path.suffix == '.pdf':
            real_savefig(fig, path.with_suffix('.svg'), bbox_inches='tight')
        return result
    Figure.savefig = publication_savefig
    try:
        for stem, folder in [('nested', 'nested_model_selection'), ('sensor_robustness', 'sensor_robustness'), ('followup', 'followup_revision'), ('intensity_band', 'intensity_band_revision')]:
            script = {'nested': 'render_nested_revision.py', 'sensor_robustness': 'render_sensor_robustness.py', 'followup': 'render_followup_revision.py', 'intensity_band': 'render_intensity_band_revision.py'}[stem]
            temp = ROOT / '.cache/publication_render' / folder
            temp.mkdir(parents=True, exist_ok=True)
            for path in (BASE / folder).glob('*.csv'):
                shutil.copy2(path, temp / path.name)
            spec = importlib.util.spec_from_file_location(f'publication_{stem}', ROOT / 'scripts' / script)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.OUT = temp
            module.main()
            for path in temp.iterdir():
                if path.suffix in ('.pdf', '.svg', '.png'):
                    shutil.copy2(path, FIG / 'diagnostics' / path.name)
    finally:
        Figure.savefig = real_savefig

def publication_figures():
    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    for start, end, color, label in [(0, 3, '#d6e8f8', 'Rest / recovery'), (3, 6, '#dcefdc', 'Warm-up'), (6, 16, '#f4dddd', 'PRBS: 30/80 W'), (16, 40, '#fff0d9', 'Incremental step test: +30 W every 3 min'), (40, 50, '#d6e8f8', None)]:
        ax.axvspan(start, end, color=color, label=label)
    pattern = [80, 30, 80, 30, 30, 80, 80, 30, 80, 80, 80, 30, 30, 30, 30]
    t = [0, 3, 6]
    w = [0, 30, 80]
    for k in range(30):
        t.append(6 + k / 3)
        w.append(pattern[k % 15])
    for k in range(8):
        t.append(16 + 3 * k)
        w.append(80 + 30 * k)
    t.extend([40, 50])
    w.extend([0, 0])
    ax.step(t, w, where='post', lw=1.5, color='#a83026')
    ax.set(xlim=(0, 50), ylim=(-5, 330), xlabel='Time (min; schematic)', ylabel='Work rate (W)')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(alpha=0.15)
    save(fig, 'Fig1_protocol')
    rows = [r for r in read(BASE / 'nested_model_selection/summary_metrics.csv') if r['procedure'] == 'Nested family selection']
    groups = ['all_signals', 'no_gas_exchange', 'hemo_plus_nirs', 'gas_exchange_ventilation', 'hr_plus_nirs', 'hemodynamics', 'hr_plus_bf', 'nirs']
    labels = ['All signals (32)', 'No gas exchange (32)', 'Hemodynamics + NIRS (32)', 'Gas exchange + ventilation (32)', 'HR + NIRS (32)', 'Hemodynamics (30)', 'HR + breathing frequency (32)', 'NIRS (32)']
    lookup = {r['group']: r for r in rows}
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for i, g in enumerate(groups):
        r = lookup[g]
        point, lo, hi = [float(r[k]) for k in ('R2', 'R2_ci_low', 'R2_ci_high')]
        ax.errorbar(point, i, xerr=[[point - lo], [hi - point]], fmt='o', capsize=3, color='#17679c')
    ax.set_yticks(range(8), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='black', ls='--', lw=0.8)
    ax.set(xlabel='Cross-validated R² (95% CI)')
    ax.grid(axis='x', alpha=0.15)
    save(fig, 'Fig3_nested_sensor_groups')
    rows = [r for r in read(BASE / 'intensity_band_revision/band_summary.csv') if r['scenario'] == 'pre_peak_rolling' and int(r['center_percent']) <= 90]
    x = np.array([int(r['center_percent']) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    for ax, value, low, high, label in [(axes[0], 'mean_subject_MAE', 'MAE_ci_low', 'MAE_ci_high', 'MAE (L/min)'), (axes[1], 'mean_subject_relative_bias_percent_measured', 'relative_bias_percent_measured_ci_low', 'relative_bias_percent_measured_ci_high', 'Relative bias (%)')]:
        v = np.array([float(r[value]) for r in rows])
        lo = np.array([float(r[low]) for r in rows])
        hi = np.array([float(r[high]) for r in rows])
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt='o-', capsize=3, color='#17679c')
        ax.set_xticks(x)
        ax.set(xlabel='Intensity band (% VO₂peak)', ylabel=label)
        ax.grid(alpha=0.15)
    save(fig, 'Fig2_intensity_agreement')
    rows = read(BASE / 'sensor_robustness/ridge_bootstrap_coefficient_stability.csv')[:15]
    from render_sensor_robustness import short_name
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.barh(range(15), [100 * float(r['top15_fraction_all_resamples']) for r in rows], color=COLORS['Fixed Ridge'])
    ax.set_yticks(range(15), [short_name(r['feature']) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set(xlabel='Top-15 membership across bootstrap refits (%)')
    ax.grid(axis='x', alpha=0.15)
    save(fig, 'Fig4_ridge_rank_stability')
    from run_impulse_protocol_sensitivity import draw_subject
    q = {r['subject_id']: r for r in read(BASE / 'impulse_protocol_sensitivity/endpoint_and_target_qc.csv')}
    grouped = {sid: [] for sid in q}
    with gzip.open(BASE / 'impulse_protocol_sensitivity/trajectories.csv.gz', 'rt') as f:
        for r in csv.DictReader(f):
            grouped[r['subject_id']].append([float(r[k] or 'nan') for k in ('time_s', 'table_workload_W', 'measured_vo2_Lmin', 'measured_vo2_30s_Lmin', 'predicted_vo2_30s_Lmin')])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for ax, sid in zip(axes.flat, ['CT_01', 'CT_02', 'CT_03', 'CT_07', 'CT_19', 'CT_30']):
        a = np.array(grouped[sid])
        draw_subject(ax, sid, (*a.T, float(q[sid]['last_positive_vo2_time']), float(q[sid]['supplied_vo2peak'])), show_labels=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.045), ncol=3, fontsize=9)
    save(fig, 'trajectory_examples')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(FIG / 'diagnostics/all_participant_trajectories.pdf') as pdf:
        for sid, values in grouped.items():
            a = np.array(values)
            fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
            draw_subject(ax, sid, (*a.T, float(q[sid]['last_positive_vo2_time']), float(q[sid]['supplied_vo2peak'])))
            fig.suptitle('Recorded-workload diagnostic; last positive VO₂ is the operational endpoint', fontsize=10)
            style_figure(fig)
            pdf.savefig(fig)
            plt.close(fig)
    aggregate = read(BASE / 'impulse_protocol_sensitivity/peak_agreement_metrics.csv')
    participants = read(BASE / 'impulse_protocol_sensitivity/participant_predictions.csv')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    axes[0].barh(range(5), [float(r['MAE']) for r in aggregate], color=['0.5', '#c77b36', '#17679c', '#41977e', '#9276af'])
    axes[0].set_yticks(range(5), ['Synthetic ramp, instantaneous', 'Synthetic ramp, 30 s', 'Recorded protocol', 'Recorded protocol, end −60 s', 'Recorded protocol, end −120 s'], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set(xlabel='MAE (L/min)')
    actual = [float(q[r['subject_id']]['supplied_vo2peak']) for r in participants]
    for scenario, label, color in [('legacy_ramp300_30s_max', 'Synthetic ramp', '#c77b36'), ('recorded_end_minus0s_30s_max', 'Recorded protocol', '#17679c')]:
        axes[1].scatter(actual, [float(r[scenario]) for r in participants], label=label, color=color, s=23)
    axes[1].plot([1.5, 4.8], [1.5, 4.8], 'k--', lw=0.8)
    axes[1].set(xlabel='VO₂peak (L/min)', ylabel='Predicted VO₂peak (L/min)', title='Matched averaging; different workload conditions')
    axes[1].legend(fontsize=7)
    axes[2].scatter([float(r['fast_ramp290_30s_max']) for r in participants], [float(r['step_template290_30s_max']) for r in participants], color='#17679c')
    axes[2].plot([1.5, 3.5], [1.5, 3.5], 'k--', lw=0.8)
    axes[2].set(xlabel='Predicted VO₂peak, fast ramp to 290 W (L/min)', ylabel='Predicted VO₂peak, step profile to 290 W (L/min)', title='Common-endpoint input-profile control')
    for ax in axes:
        ax.grid(alpha=0.15)
    save(fig, 'sensitivity_summary')

def vector_heatmap():
    rows = [r for r in read(BASE / 'intensity_band_revision/participant_band_metrics.csv') if r['scenario'] == 'pre_peak_rolling' and int(r['center_percent']) <= 90]
    ids = sorted({r['subject_id'] for r in rows})
    centers = list(range(40, 100, 10))
    values = np.full((len(ids), 6), np.nan)
    for r in rows:
        values[ids.index(r['subject_id']), centers.index(int(r['center_percent']))] = float(r['relative_bias_percent_measured'])
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    cmap = plt.colormaps['RdBu_r'].copy()
    cmap.set_bad('.85')
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=-45, vmax=45, rasterized=False)
    ax.set_xticks(np.arange(6) + 0.5, [f'{x}%' for x in centers])
    ax.set_yticks(np.arange(len(ids)) + 0.5, ids)
    ax.invert_yaxis()
    ax.set(xlabel='Intensity band (% VO₂peak)', title='Participant-specific signed error; grey = no covered window')
    cb = fig.colorbar(mesh, ax=ax, label='Relative bias (%)')
    cb.solids.set_rasterized(False)
    save(fig, 'diagnostics/participant_band_bias')

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / "diagnostics").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    export_diagnostics()
    publication_figures()
    vector_heatmap()

if __name__ == "__main__":
    main()

