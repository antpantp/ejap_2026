"""Paired sensor contrasts, cadence sensitivity, and fixed-Ridge stability.

Run with python scripts/run_sensor_robustness.py.
Accepted primary/impulse results are read-only. New outputs are written below
results/revision_round1/sensor_robustness. Cadence is retained in the primary model.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from audit_revision_step1 import numeric_columns
from run_nested_revision import (FAMILIES, SEED, metrics, prepare_datasets,
                                 process_outer, read_csv, write_csv)
from evaluation.revision_modeling import estimator, pipeline, preprocessing


BASE = ROOT / "results/revision_round1/nested_model_selection"
OUT = ROOT / "results/revision_round1/sensor_robustness"
N_BOOTSTRAPS = 500
N_PAIR_RESAMPLES = 10000


def variant_datasets(datasets):
    ids, names, X, y = datasets["all_signals"]
    noncadence = np.array([not n.startswith("RpM_") for n in names])
    assert noncadence.sum() == 150 and (~noncadence).sum() == 5
    common = np.array([sid in datasets["hemodynamics"][0] for sid in ids])
    assert common.sum() == 30
    return {
        "all_signals_without_cadence": (ids, [n for n, keep in zip(names, noncadence) if keep], X[:, noncadence], y),
        "all_signals_matched_hemodynamics": ([sid for sid, keep in zip(ids, common) if keep], names, X[common], y[common]),
        "all_signals_without_cadence_matched_hemodynamics":
            ([sid for sid, keep in zip(ids, common) if keep], [n for n, keep in zip(names, noncadence) if keep], X[common][:, noncadence], y[common]),
    }


def paired_error_summary(reference, comparator, family_size=7):
    assert set(reference) == set(comparator)
    ids = sorted(reference)
    actual = np.array([reference[s][0] for s in ids])
    assert np.allclose(actual, [comparator[s][0] for s in ids], rtol=0, atol=1e-12)
    ref = np.array([reference[s][1] for s in ids])
    comp = np.array([comparator[s][1] for s in ids])
    delta = np.abs(ref - actual) - np.abs(comp - actual)
    indices = np.random.default_rng(SEED).integers(0, len(ids), (N_PAIR_RESAMPLES, len(ids)))
    samples = delta[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    # Bonferroni-style marginal percentile intervals within seven contrasts.
    adjusted_lo, adjusted_hi = np.quantile(samples, [0.025 / family_size, 1 - 0.025 / family_size])
    return {"N": len(ids), "reference_MAE": float(np.abs(ref - actual).mean()),
            "comparator_MAE": float(np.abs(comp - actual).mean()),
            "delta_MAE_reference_minus_comparator": float(delta.mean()),
            "ci95_low": float(lo), "ci95_high": float(hi),
            "seven_contrast_adjusted_low": float(adjusted_lo), "seven_contrast_adjusted_high": float(adjusted_hi),
            "n_reference_smaller_absolute_error": int((delta < 0).sum()),
            "matched_subject_ids": json.dumps(ids)}


def coefficient_ranks(coef, support):
    ranks = np.full(len(coef), len(coef) + 1.0)
    ranks[support] = rankdata(-np.abs(coef[support]), method="average")
    return ranks


def ridge_bootstrap(rep, sample, ids, names, X, y):
    with threadpool_limits(limits=1):
        Xb, yb = X[sample], y[sample]
        groups = np.array(ids)[sample]
        splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=42).split(Xb, yb, groups))
        scores = np.empty((12, 5))
        alphas = np.logspace(-3, 2, 12)
        folds, notices = [], []
        for fold, (train, validation) in enumerate(splits):
            train_ids = sorted(set(groups[train]))
            val_ids = sorted(set(groups[validation]))
            assert set(train_ids).isdisjoint(val_ids)
            prep = preprocessing().fit(Xb[train])
            Xt, Xv = prep.transform(Xb[train]), prep.transform(Xb[validation])
            folds.append({"replicate": rep, "inner_fold": fold + 1,
                          "train_unique_subject_ids": json.dumps(train_ids), "validation_unique_subject_ids": json.dumps(val_ids),
                          "train_row_indices": json.dumps(train.tolist()), "validation_row_indices": json.dumps(validation.tolist()),
                          "n_retained_features": int(prep.named_steps["missingness"].support_.sum())})
            for a, alpha in enumerate(alphas):
                model = estimator("Ridge", {"alpha": float(alpha)})
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(Xt, yb[train])
                scores[a, fold] = np.mean((model.predict(Xv).ravel() - yb[validation]) ** 2)
                notices.extend({"replicate": rep, "stage": "inner", "category": n.category.__name__, "message": str(n.message)} for n in caught)
        best = int(np.argmin(scores.mean(axis=1)))
        fitted = pipeline("Ridge", {"alpha": float(alphas[best])})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted.fit(Xb, yb)
        notices.extend({"replicate": rep, "stage": "refit", "category": n.category.__name__, "message": str(n.message)} for n in caught)
        support = fitted.named_steps["preprocess"].named_steps["missingness"].support_
        coef = np.zeros(len(names))
        coef[support] = fitted.named_steps["model"].coef_.ravel()
        return {"replicate": rep, "sample_indices": sample.tolist(), "n_unique_subjects": len(set(groups)),
                "selected_alpha": float(alphas[best]), "candidate_mean_mse": scores.mean(axis=1).tolist(),
                "coef": coef.tolist(), "support": support.tolist(), "ranks": coefficient_ranks(coef, support).tolist(),
                "inner_folds": folds, "warnings": notices}


def summarize_coefficients(coefs, supports, ranks, names):
    rows = []
    for j, name in enumerate(names):
        mask = supports[:, j]
        values = coefs[mask, j]
        if not len(values):
            raise ValueError(f"Feature never retained: {name}")
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({"feature": name, "retained_fraction": float(mask.mean()),
                     "mean_coef_when_retained": float(values.mean()), "mean_abs_coef_when_retained": float(np.abs(values).mean()),
                     "coef_2_5pct_when_retained": float(low), "coef_97_5pct_when_retained": float(high),
                     "sign_consistency_when_retained": float(max((values > 0).sum(), (values < 0).sum()) / len(values)),
                     "top15_fraction_all_resamples": float((ranks[:, j] <= 15).mean()),
                     "median_abs_rank_including_omitted": float(np.median(ranks[:, j])),
                     "rank_25pct": float(np.quantile(ranks[:, j], 0.25)), "rank_75pct": float(np.quantile(ranks[:, j], 0.75))})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets, _, source_paths = prepare_datasets()
    variants = variant_datasets(datasets)
    jobs = []
    for group, (ids, names, X, y) in variants.items():
        for heldout, sid in enumerate(ids):
            train = np.array([i for i in range(len(ids)) if i != heldout])
            jobs.append(delayed(process_outer)(group, sid, [ids[i] for i in train], names,
                                               X[train], y[train], X[[heldout]], OUT))
    results = []
    started = time.monotonic()
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=4, return_as="generator_unordered")(jobs):
            results.append(result)
            path = OUT / "fold_results" / result["group"] / f"{result['subject_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
            if len(results) % 16 == 0 or len(results) == len(jobs):
                print(f"Additional nested folds: {len(results)}/{len(jobs)}; {time.monotonic() - started:.1f}s", flush=True)
    additional, manifests = [], []
    for result in sorted(results, key=lambda r: (r["group"], r["subject_id"])):
        group, sid = result["group"], result["subject_id"]
        ids, _, _, y = variants[group]
        actual = float(y[ids.index(sid)])
        additional.extend({"group": group, "subject_id": sid, "actual_vo2peak": actual, **r} for r in result["predictions"])
        manifests.extend({"group": group, "subject_id": sid, **r} for r in result["manifest"])
    write_csv(OUT / "additional_predictions.csv", additional)
    write_csv(OUT / "additional_artifact_manifest.csv", manifests)
    _, primary = read_csv(BASE / "fixed_family_predictions.csv")
    all_rows = primary + additional
    def prediction_map(group, procedure):
        rows = [r for r in all_rows if r["group"] == group and
                ((procedure == "Nested family selection" and str(r["selected_by_inner_cv"]) == "True") or r["family"] == procedure)]
        return {r["subject_id"]: (float(r["actual_vo2peak"]), float(r["predicted_vo2peak"])) for r in rows}

    variant_metrics, frequencies = [], []
    for group, (ids, names, _, _) in variants.items():
        for procedure in ("Nested family selection",) + FAMILIES:
            p = prediction_map(group, procedure)
            actual = np.array([p[s][0] for s in sorted(p)])
            pred = np.array([p[s][1] for s in sorted(p)])
            idx = np.random.default_rng(SEED).integers(0, len(ids), (N_PAIR_RESAMPLES, len(ids)))
            variant_metrics.append({"group": group, "procedure": procedure, "N": len(ids), "n_features": len(names), **metrics(actual, pred, idx)})
        for family in FAMILIES:
            frequencies.append({"group": group, "family": family,
                                "n_selected": sum(r["group"] == group and r["family"] == family and r["selected_by_inner_cv"] for r in additional)})
    write_csv(OUT / "additional_summary_metrics.csv", variant_metrics)
    write_csv(OUT / "additional_family_selection_counts.csv", frequencies)
    contrasts = []
    for procedure in ("Nested family selection", "Ridge"):
        for cadence in (True, False):
            for reduced in (g for g in datasets if g != "all_signals"):
                if reduced == "hemodynamics":
                    reference = "all_signals_matched_hemodynamics" if cadence else "all_signals_without_cadence_matched_hemodynamics"
                else:
                    reference = "all_signals" if cadence else "all_signals_without_cadence"
                contrasts.append({"procedure": procedure, "reference": reference, "reduced_group": reduced,
                                  "reference_includes_cadence": cadence,
                                  **paired_error_summary(prediction_map(reference, procedure), prediction_map(reduced, procedure))})
    write_csv(OUT / "paired_sensor_comparisons.csv", contrasts)
    cadence_contrasts = [{"procedure": p, "reference": "all_signals_with_cadence", "comparator": "all_signals_without_cadence",
                          **paired_error_summary(prediction_map("all_signals", p), prediction_map("all_signals_without_cadence", p), family_size=1)}
                         for p in ("Nested family selection",) + FAMILIES]
    write_csv(OUT / "cadence_ablation_comparisons.csv", cadence_contrasts)

    ids, names, X, y = datasets["all_signals"]
    coef_rows, support_rows = [], []
    _, full_manifest = read_csv(BASE / "artifact_manifest.csv")
    for row in full_manifest:
        if row["group"] != "all_signals" or row["family"] != "Ridge":
            continue
        path = BASE / row["artifact_path"]
        source_paths.append(path)
        payload = joblib.load(path)
        assert payload["feature_columns"] == names
        fitted = payload["estimator"]
        support = fitted.named_steps["preprocess"].named_steps["missingness"].support_
        coef = np.zeros(len(names));coef[support] = fitted.named_steps["model"].coef_.ravel()
        coef_rows.append(coef);support_rows.append(support)
    loso = np.asarray(coef_rows);loso_support = np.asarray(support_rows)
    loso_ranks = np.array([coefficient_ranks(c, s) for c, s in zip(loso, loso_support)])
    loso_stats = summarize_coefficients(loso, loso_support, loso_ranks, names)
    reference_order = np.argsort(-np.abs(loso.mean(axis=0)), kind="stable")
    for index, feature_index in enumerate(reference_order):
        loso_stats[feature_index]["rank_abs_mean_signed_coef"] = index + 1
    write_csv(OUT / "ridge_loso_coefficient_stability.csv", sorted(loso_stats, key=lambda r: r["rank_abs_mean_signed_coef"]))

    samples = np.random.default_rng(SEED).integers(0, len(ids), (N_BOOTSTRAPS, len(ids)))
    boot_results = []
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        jobs = [delayed(ridge_bootstrap)(rep, sample, ids, names, X, y) for rep, sample in enumerate(samples)]
        for result in Parallel(n_jobs=4, return_as="generator_unordered")(jobs):
            boot_results.append(result)
            if len(boot_results) % 100 == 0:
                print(f"Ridge participant bootstraps: {len(boot_results)}/{N_BOOTSTRAPS}", flush=True)
    boot_results.sort(key=lambda r: r["replicate"])
    (OUT / "ridge_bootstrap_records.json").write_text(json.dumps(boot_results) + "\n")
    coef = np.array([r["coef"] for r in boot_results]);support = np.array([r["support"] for r in boot_results], dtype=bool)
    ranks = np.array([r["ranks"] for r in boot_results])
    stats = summarize_coefficients(coef, support, ranks, names)
    name_index = {n: i for i, n in enumerate(names)}
    for row in stats:
        j = name_index[row["feature"]]
        row["reference_loso_rank"] = loso_stats[j]["rank_abs_mean_signed_coef"]
        row["reference_loso_mean_signed_coef"] = float(loso[:, j].mean())
    write_csv(OUT / "ridge_bootstrap_coefficient_stability.csv", sorted(stats, key=lambda r: r["reference_loso_rank"]))
    write_csv(OUT / "ridge_bootstrap_inner_folds.csv", [f for r in boot_results for f in r["inner_folds"]])
    write_csv(OUT / "ridge_bootstrap_tuning.csv", [{"replicate": r["replicate"], "n_unique_subjects": r["n_unique_subjects"],
                                                   "sample_indices": json.dumps(r["sample_indices"]), "selected_alpha": r["selected_alpha"],
                                                   "candidate_mean_mse": json.dumps(r["candidate_mean_mse"])} for r in boot_results])
    # Descriptive correlation only: never fed into selection or tuning.
    imputed = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    centered = imputed - imputed.mean(axis=0)
    cross = centered.T @ centered
    denom = np.sqrt(np.diag(cross)[:, None] * np.diag(cross)[None, :])
    correlation = np.divide(cross, denom, out=np.zeros_like(cross), where=denom != 0)
    np.save(OUT / "descriptive_feature_correlations.npy", correlation)
    correlation_pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            if abs(correlation[a, b]) >= 0.9:
                correlation_pairs.append({"feature_a": names[a], "feature_b": names[b], "pearson_r_descriptive": float(correlation[a, b])})
    write_csv(OUT / "highly_correlated_feature_pairs.csv", correlation_pairs)
    (OUT / "correlation_feature_order.json").write_text(json.dumps(names, indent=2) + "\n")

    cadence_records = []
    for path in sorted((ROOT / "data/processed").glob("CT_*_data.csv")):
        sid = path.stem.removesuffix("_data");h, rows = read_csv(path);cols = numeric_columns(h, rows)
        record = {"subject_id": sid, "cadence_available": "RpM" in cols, "n_window_samples": 0,
                  "window_mean_rpm": "", "window_min_rpm": "", "window_max_rpm": "", "fraction_below60_rpm": ""}
        if "RpM" in cols:
            mask = (cols["Time"] >= 300) & (cols["Time"] <= 1000) & np.isin(cols["WorkRate"], [30, 80])
            v = cols["RpM"][mask];v = v[np.isfinite(v)]
            record.update(n_window_samples=len(v), window_mean_rpm=float(v.mean()), window_min_rpm=float(v.min()),
                          window_max_rpm=float(v.max()), fraction_below60_rpm=float((v < 60).mean()))
        cadence_records.append(record)
    write_csv(OUT / "cadence_recording_summary.csv", cadence_records)
    source_paths.extend([BASE / "fixed_family_predictions.csv", BASE / "artifact_manifest.csv",
                         ROOT / "data/_variable_explanations.csv", ROOT / "docs/cadence.txt",
                         Path(__file__).resolve(), ROOT / "scripts/run_nested_revision.py", ROOT / "evaluation/revision_modeling.py"])
    unique_paths = list(dict.fromkeys(source_paths))
    write_csv(OUT / "input_provenance.csv", [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in unique_paths])
    top = sorted(stats, key=lambda r: r["reference_loso_rank"])[:15]
    warnings_all = [w for r in results for w in r["warnings"]] + [w for r in boot_results for w in r["warnings"]]
    summary = {"additional_outer_folds": len(results), "additional_models": len(additional),
               "paired_sensor_contrasts": len(contrasts), "cadence_contrasts": len(cadence_contrasts),
               "coefficient_bootstrap_replicates": N_BOOTSTRAPS, "bootstrap_inner_group_splits": 5 * N_BOOTSTRAPS,
               "fit_warnings": len(warnings_all), "pair_resamples": N_PAIR_RESAMPLES, "seed": SEED,
               "original_top15_bootstrap_top15_frequency_min": min(r["top15_fraction_all_resamples"] for r in top),
               "original_top15_bootstrap_top15_frequency_median": float(np.median([r["top15_fraction_all_resamples"] for r in top])),
               "original_top15_bootstrap_top15_frequency_max": max(r["top15_fraction_all_resamples"] for r in top),
               "original_top15_conditional_coef_intervals_excluding_zero": sum(r["coef_2_5pct_when_retained"] > 0 or r["coef_97_5pct_when_retained"] < 0 for r in top),
               "cadence_comparisons": cadence_contrasts, "elapsed_seconds": time.monotonic() - started}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
