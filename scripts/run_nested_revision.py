"""Fully nested family selection with original windows, sensor groups, and grids.

Run with the isolated environment:
    python scripts/run_nested_revision.py

Only new files below results/revision_round1/nested_model_selection are written.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from audit_revision_step1 import features, numeric_columns, read_csv, sensor_groups
from evaluation.revision_modeling import (FAMILIES, candidate_grid, estimator,
                                         evaluate_candidates, inner_design, pipeline)


SEED = 20261104
BOOTSTRAPS = 5000


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_datasets():
    groups = sensor_groups()
    groups["all_signals"] = sorted(set(groups["all_signals"]) | {"RpM"})
    recordings, targets, source_paths = {}, {}, []
    for path in sorted((ROOT / "data/processed").glob("CT_*_data.csv")):
        sid = path.stem.removesuffix("_data")
        header, rows = read_csv(path)
        recordings[sid] = numeric_columns(header, rows)
        target_path = path.parent / f"{sid}_targets.csv"
        _, target = read_csv(target_path)
        targets[sid] = float(target[0]["VO2peak_abs"])
        source_paths.extend([path, target_path])
    datasets, availability = {}, []
    for group, signals in groups.items():
        ids = [sid for sid, cols in recordings.items() if any(s in cols for s in signals)]
        availability.append({"group": group, "N": len(ids), "included": len(ids) >= 25,
                             "excluded_subject_ids": json.dumps(sorted(set(recordings) - set(ids)))})
        if len(ids) < 25:
            continue
        feature_rows = []
        for sid in ids:
            # Preserve the original feature order, including coordinate-descent
            # order for ElasticNet. The full model historically used raw-column
            # order; reduced groups used their predefined signal-list order.
            selected = [name for name in recordings[sid] if name in signals] if group == "all_signals" else signals
            feature_rows.append(features(recordings[sid], selected))
        names = list(dict.fromkeys(name for row in feature_rows for name in row))
        X = np.array([[row.get(name, np.nan) for name in names] for row in feature_rows])
        y = np.array([targets[sid] for sid in ids])
        assert np.isfinite(y).all()
        # No cohort-wide availability threshold is applied here.
        datasets[group] = (ids, names, X, y)
    return datasets, availability, source_paths


def process_outer(group, heldout, ids, names, X_train, y_train, X_test, out):
    start = time.monotonic()
    assert heldout not in ids
    with threadpool_limits(limits=1):
        candidates, best, family_best, cv_rows, inner_rows, notices = evaluate_candidates(X_train, y_train, ids, names)
        selected_family = candidates[best][0]
        predictions, manifest = [], []
        for family in FAMILIES:
            index = family_best[family]
            params = candidates[index][1]
            fitted = pipeline(family, params)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted.fit(X_train, y_train)
                pred = float(np.asarray(fitted.predict(X_test)).ravel()[0])
            for notice in caught:
                notices.append({"stage": "outer_refit", "inner_fold": "", "candidate": index,
                                "family": family, "params": json.dumps(params, sort_keys=True),
                                "category": notice.category.__name__, "message": str(notice.message)})
            support = fitted.named_steps["preprocess"].named_steps["missingness"].support_
            retained = [n for n, keep in zip(names, support) if keep]
            relative = Path("models") / group / f"{heldout}_{family}.joblib"
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {"estimator": fitted, "model_name": family, "left_out_subject": heldout,
                       "train_subject_ids": ids, "feature_columns": names, "retained_features": retained,
                       "best_params": params, "selected_by_inner_cv": family == selected_family,
                       "target_column": "VO2peak_abs", "recorded_time_window": [300, 1000],
                       "work_rate_levels": [30, 80], "min_samples_per_level": 20,
                       "min_training_feature_availability": 0.6, "inner_mean_mse": cv_rows[index]["mean_inner_mse"]}
            joblib.dump(payload, destination, compress=3)
            predictions.append({"family": family, "predicted_vo2peak": pred,
                                "params": json.dumps(params, sort_keys=True), "selected_by_inner_cv": family == selected_family,
                                "inner_mean_mse": cv_rows[index]["mean_inner_mse"],
                                "n_retained_features": len(retained)})
            manifest.append({"family": family, "artifact_path": str(relative),
                             "selected_by_inner_cv": family == selected_family,
                             "train_subject_ids": json.dumps(ids), "n_train": len(ids),
                             "retained_features": json.dumps(retained)})
    return {"group": group, "subject_id": heldout, "predictions": predictions,
            "manifest": manifest, "candidates": cv_rows, "inner_folds": inner_rows,
            "warnings": notices, "seconds": time.monotonic() - start}


def validate_against_grid_search(datasets):
    """Check optimized CV against full Pipeline/GridSearchCV on two real folds."""
    checks = []
    for group, heldout in [("all_signals", "CT_01"), ("hr_plus_bf", "CT_08")]:
        ids, names, X, y = datasets[group]
        train = np.array([i for i, sid in enumerate(ids) if sid != heldout])
        X_train, y_train = X[train], y[train]
        train_ids = [ids[i] for i in train]
        candidates, best, _, rows, _, _ = evaluate_candidates(X_train, y_train, train_ids, names)
        splits, max_components = inner_design(X_train)
        grid = [{"model": [estimator(family, params)]} for family, params in candidate_grid(max_components)]
        reference = GridSearchCV(pipeline("Ridge", {"alpha": 1}), grid, cv=splits,
                                 scoring="neg_mean_squared_error", refit=False, n_jobs=1, error_score="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference.fit(X_train, y_train)
        ours = np.array([row["mean_inner_mse"] for row in rows])
        theirs = -reference.cv_results_["mean_test_score"]
        difference = float(np.max(np.abs(ours - theirs)))
        assert difference < 1e-10, (group, difference)
        assert best == int(reference.best_index_)
        checks.append({"group": group, "heldout_subject": heldout, "n_candidates": len(candidates),
                       "max_inner_mse_difference": difference, "same_selected_candidate": True,
                       "max_pls_components": max_components})
    return checks


def metrics(actual, pred, indices):
    residual = pred - actual
    point = {"MAE": np.abs(residual).mean(), "RMSE": np.sqrt(np.mean(residual ** 2)),
             "R2": 1 - np.sum(residual ** 2) / np.sum((actual - actual.mean()) ** 2),
             "Bias": residual.mean()}
    a, p = actual[indices], pred[indices]
    e = p - a
    boot = {"MAE": np.abs(e).mean(axis=1), "RMSE": np.sqrt(np.mean(e ** 2, axis=1)),
            "R2": 1 - np.sum(e ** 2, axis=1) / np.sum((a - a.mean(axis=1, keepdims=True)) ** 2, axis=1),
            "Bias": e.mean(axis=1)}
    row = {}
    for name in point:
        row[name] = float(point[name])
        row[name + "_ci_low"], row[name + "_ci_high"] = map(float, np.quantile(boot[name], [0.025, 0.975]))
    return row


def summarize(out, datasets, results):
    prediction_rows, manifests, candidate_rows, inner_rows, warning_rows = [], [], [], [], []
    for result in sorted(results, key=lambda r: (r["group"], r["subject_id"])):
        group, sid = result["group"], result["subject_id"]
        ids, _, _, y = datasets[group]
        actual = float(y[ids.index(sid)])
        for row in result["predictions"]:
            prediction_rows.append({"group": group, "subject_id": sid, "actual_vo2peak": actual,
                                    **row, "residual": row["predicted_vo2peak"] - actual})
        for source, destination in [("manifest", manifests), ("candidates", candidate_rows),
                                    ("inner_folds", inner_rows), ("warnings", warning_rows)]:
            destination.extend({"group": group, "subject_id": sid, **row} for row in result[source])
    write_csv(out / "fixed_family_predictions.csv", prediction_rows)
    nested_rows = [row for row in prediction_rows if row["selected_by_inner_cv"]]
    write_csv(out / "nested_selection_predictions.csv", nested_rows)
    write_csv(out / "artifact_manifest.csv", manifests)
    write_csv(out / "inner_candidate_scores.csv", candidate_rows)
    write_csv(out / "inner_fold_membership.csv", inner_rows)
    if warning_rows:
        write_csv(out / "fit_warnings.csv", warning_rows)
    summary, frequencies, comparisons = [], [], []
    _, old_summary = read_csv(ROOT / "results/ablation_best_models_summary.csv")
    old_best = {r["Group"]: r for r in old_summary}
    for group, (ids, names, _, _) in datasets.items():
        group_nested = sorted([r for r in nested_rows if r["group"] == group], key=lambda r: r["subject_id"])
        actual = np.array([r["actual_vo2peak"] for r in group_nested])
        selected = np.array([r["predicted_vo2peak"] for r in group_nested])
        indices = np.random.default_rng(SEED).integers(0, len(ids), (BOOTSTRAPS, len(ids)))
        summary.append({"group": group, "procedure": "Nested family selection", "N": len(ids),
                        "n_candidate_features": len(names), **metrics(actual, selected, indices)})
        for family in FAMILIES:
            family_rows = sorted([r for r in prediction_rows if r["group"] == group and r["family"] == family], key=lambda r: r["subject_id"])
            pred = np.array([r["predicted_vo2peak"] for r in family_rows])
            summary.append({"group": group, "procedure": family, "N": len(ids),
                            "n_candidate_features": len(names), **metrics(actual, pred, indices)})
            frequencies.append({"group": group, "family": family,
                                "n_outer_folds_selected": sum(r["family"] == family for r in group_nested), "N": len(ids)})
        old_family = old_best[group]["Model"]
        _, old_rows = read_csv(ROOT / "results" / f"ablation_{group}_{old_family}_loso_predictions.csv")
        old_map = {r["subject_id"]: float(r["predicted_vo2peak"]) for r in old_rows}
        old = np.array([old_map[sid] for sid in ids])
        change = np.abs(selected - actual) - np.abs(old - actual)
        lo, hi = np.quantile(change[indices].mean(axis=1), [0.025, 0.975])
        comparisons.append({"group": group, "submitted_best_family": old_family, "N": len(ids),
                            "submitted_MAE": float(old_best[group]["MAE"]),
                            "submitted_RMSE": float(old_best[group]["RMSE"]),
                            "submitted_R2": float(old_best[group]["R2"]),
                            "revised_nested_MAE": float(np.abs(selected - actual).mean()),
                            "delta_MAE_nested_minus_submitted": float(change.mean()),
                            "delta_MAE_ci_low": float(lo), "delta_MAE_ci_high": float(hi)})
    write_csv(out / "summary_metrics.csv", summary)
    write_csv(out / "family_selection_counts.csv", frequencies)
    write_csv(out / "comparison_with_submitted.csv", comparisons)
    return summary, prediction_rows, nested_rows, warning_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results/revision_round1/nested_model_selection"
    out.mkdir(parents=True, exist_ok=True)
    datasets, availability, source_paths = prepare_datasets()
    write_csv(out / "sensor_group_eligibility.csv", availability)
    print("Validating optimized inner CV against Pipeline/GridSearchCV...", flush=True)
    with threadpool_limits(limits=1):
        checks = validate_against_grid_search(datasets)
    write_csv(out / "grid_search_equivalence_checks.csv", checks)
    print(json.dumps(checks), flush=True)
    if args.validate_only:
        return
    tasks = []
    for group, (ids, names, X, y) in datasets.items():
        for heldout_index, sid in enumerate(ids):
            train = np.array([i for i in range(len(ids)) if i != heldout_index])
            train_ids = [ids[i] for i in train]
            tasks.append(delayed(process_outer)(group, sid, train_ids, names,
                                                X[train], y[train], X[[heldout_index]], out))
    results = []
    completed = {group: 0 for group in datasets}
    start = time.monotonic()
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        generator = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(tasks)
        for result in generator:
            results.append(result)
            group = result["group"]
            completed[group] += 1
            # Save each result immediately; it is independently reviewable.
            fold_path = out / "fold_results" / group / f"{result['subject_id']}.json"
            fold_path.parent.mkdir(parents=True, exist_ok=True)
            fold_path.write_text(json.dumps(result, indent=2) + "\n")
            if completed[group] % 8 == 0 or completed[group] == len(datasets[group][0]):
                print(f"{group}: {completed[group]}/{len(datasets[group][0])} outer folds; elapsed {time.monotonic() - start:.1f}s", flush=True)
    assert len(results) == sum(len(d[0]) for d in datasets.values())
    summary, predictions, nested, notices = summarize(out, datasets, results)
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    (out / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    (out / "requirements.txt").write_text("\n".join(f"{name}=={value}" for name, value in versions.items() if name != "python") + "\nthreadpoolctl==3.6.0\n")
    source_paths.extend([Path(__file__).resolve(), ROOT / "evaluation/revision_modeling.py", ROOT / "scripts/audit_revision_step1.py",
                         ROOT / "notebooks/5_ablation_study_feature_models.ipynb", ROOT / "results/ablation_best_models_summary.csv"])
    provenance = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_paths]
    write_csv(out / "input_provenance.csv", provenance)
    selected_convergence = [w for r in results for w in r["warnings"]
                            if w["category"] == "ConvergenceWarning" and any(p["selected_by_inner_cv"] and p["params"] == w["params"] and p["family"] == w["family"] for p in r["predictions"])]
    run_summary = {"outer_folds": len(results), "sensor_groups": len(datasets),
                   "inner_candidate_evaluations": sum(len(r["candidates"]) * 5 for r in results),
                   "fixed_family_predictions": len(predictions), "nested_predictions": len(nested),
                   "fit_warning_count": len(notices), "selected_candidate_convergence_warning_count": len(selected_convergence),
                   "bootstrap_seed": SEED, "bootstrap_replicates": BOOTSTRAPS,
                   "elapsed_seconds": time.monotonic() - start, "equivalence_checks": checks}
    (out / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2), flush=True)
    print(json.dumps([r for r in summary if r["procedure"] == "Nested family selection"], indent=2), flush=True)


if __name__ == "__main__":
    main()
