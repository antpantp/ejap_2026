"""Consistent follow-up transport, change agreement, and direct-delta validation.

Baseline-selected artifacts are frozen for E1--E3. E4 is a separate prospective
baseline-to-change task, preserving its original grids/MAE tuning criterion while
nesting family selection and feature filtering. Original outputs are read-only.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import warnings

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
from joblib import Parallel, delayed
import numpy as np
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from audit_revision_step1 import features, numeric_columns
from run_nested_revision import FAMILIES, SEED, metrics, prepare_datasets, read_csv, write_csv
from evaluation.revision_modeling import TrainingMissingnessFilter


OUT = ROOT / "results/revision_round1/followup_revision"
BASE = ROOT / "results/revision_round1/nested_model_selection"
ROBUST = ROOT / "results/revision_round1/sensor_robustness"
GROUPS = ("all_signals", "no_gas_exchange", "hemo_plus_nirs", "nirs")
RESAMPLES = 5000


def extra_metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    if np.ptp(pred) > 1e-12 and np.ptp(actual) > 1e-12:
        result = stats.pearsonr(actual, pred)
        correlation, pvalue = float(result.statistic), float(result.pvalue)
        calibration = stats.linregress(pred, actual)
        slope, intercept = float(calibration.slope), float(calibration.intercept)
    else:
        correlation, pvalue, slope, intercept = None, None, None, None
    return {"Pearson_r": correlation, "Pearson_p_descriptive": pvalue,
            "calibration_slope": slope, "calibration_intercept": intercept}


def delta_metrics(actual, pred, indices):
    return {**metrics(actual, pred, indices), **extra_metrics(actual, pred),
            "n_sign_matches": int((np.sign(actual) == np.sign(pred)).sum()),
            "sign_agreement_pct": float(100 * np.mean(np.sign(actual) == np.sign(pred)))}


def delta_estimator(family, params):
    if family == "Ridge":
        return Ridge(random_state=42, **params)
    if family == "ElasticNet":
        return ElasticNet(random_state=42, max_iter=10000, **params)
    if family == "PLS":
        return PLSRegression(**params)
    raise ValueError(family)


def delta_preprocessing():
    # E4 originally used imputation + scaling, without the baseline unit rescaler.
    return Pipeline([("missingness", TrainingMissingnessFilter()),
                     ("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def fit_with_solver_retry(model, X, y):
    """Preserve original tolerances; retry a nonconvergent solver at more iterations."""
    with warnings.catch_warnings(record=True) as initial:
        warnings.simplefilter("always")
        model.fit(X, y)
    retries = any(issubclass(n.category, ConvergenceWarning) for n in initial)
    final_warnings = []
    if retries:
        key = "model__max_iter" if isinstance(model, Pipeline) else "max_iter"
        model = clone(model).set_params(**{key: 100000})
        with warnings.catch_warnings(record=True) as final:
            warnings.simplefilter("always")
            model.fit(X, y)
        if any(issubclass(n.category, ConvergenceWarning) for n in final):
            raise RuntimeError("Direct-change candidate did not converge after solver retry")
        final_warnings = list(final)
    notices = [{"category": n.category.__name__, "message": str(n.message),
                "resolved_by_solver_retry": retries and issubclass(n.category, ConvergenceWarning)} for n in initial]
    notices.extend({"category": n.category.__name__, "message": str(n.message), "resolved_by_solver_retry": False} for n in final_warnings)
    return model, notices


def direct_delta_fold(group, sid, train_ids, feature_names, X_train, y_train, X_test):
    """No held-out change label is passed into this fitting function."""
    assert sid not in train_ids
    with threadpool_limits(limits=1):
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
        max_components = min(8, min(len(t) - 1 for t, _ in splits),
                             min(int((np.isfinite(X_train[t]).mean(axis=0) >= 0.6).sum()) for t, _ in splits))
        candidates = [("Ridge", {"alpha": a}) for a in (0.01, 0.1, 1.0, 10.0, 100.0, 300.0)]
        candidates.extend(("ElasticNet", {"alpha": a, "l1_ratio": ratio})
                          for a in (0.001, 0.01, 0.1, 1.0, 10.0) for ratio in (0.1, 0.3, 0.5, 0.7, 0.9))
        candidates.extend(("PLS", {"n_components": n}) for n in range(1, max_components + 1))
        errors = np.empty((len(candidates), 5))
        folds, notices = [], []
        for fold, (train, val) in enumerate(splits):
            prep = delta_preprocessing().fit(X_train[train])
            Xt, Xv = prep.transform(X_train[train]), prep.transform(X_train[val])
            support = prep.named_steps["missingness"].support_
            folds.append({"inner_fold": fold + 1, "train_ids": [train_ids[i] for i in train],
                          "validation_ids": [train_ids[i] for i in val],
                          "retained_features": [n for n, keep in zip(feature_names, support) if keep]})
            for i, (family, params) in enumerate(candidates):
                model = delta_estimator(family, params)
                model, caught = fit_with_solver_retry(model, Xt, y_train[train])
                pred = np.asarray(model.predict(Xv)).ravel()
                errors[i, fold] = np.abs(pred - y_train[val]).mean()
                notices.extend({"stage": "inner", "inner_fold": fold + 1, "family": family,
                                "params": params, **n} for n in caught)
        means = errors.mean(axis=1)
        best = int(np.argmin(means))
        family_best = {family: min((i for i, (f, _) in enumerate(candidates) if f == family), key=lambda i: means[i]) for family in FAMILIES}
        rows, manifest = [], []
        for family in FAMILIES:
            index = family_best[family];params = candidates[index][1]
            fitted = Pipeline([("preprocess", delta_preprocessing()), ("model", delta_estimator(family, params))])
            fitted, caught = fit_with_solver_retry(fitted, X_train, y_train)
            pred = float(np.asarray(fitted.predict(X_test)).ravel()[0])
            notices.extend({"stage": "refit", "inner_fold": "", "family": family, "params": params, **n} for n in caught)
            relative = Path("direct_delta_models") / group / f"{sid}_{family}.joblib"
            path = OUT / relative;path.parent.mkdir(parents=True, exist_ok=True)
            support = fitted.named_steps["preprocess"].named_steps["missingness"].support_
            payload = {"estimator": fitted, "model_name": family, "train_subject_ids": train_ids,
                       "left_out_subject": sid, "feature_columns": feature_names,
                       "target": "post_minus_pre_VO2peak_abs", "best_params": params,
                       "selected_by_inner_cv": index == best,
                       "retained_features": [n for n, keep in zip(feature_names, support) if keep]}
            joblib.dump(payload, path, compress=3)
            rows.append({"family": family, "predicted_delta": pred, "selected_by_inner_cv": index == best,
                         "params": json.dumps(params, sort_keys=True), "inner_mean_mae": float(means[index]),
                         "n_retained_features": int(support.sum())})
            manifest.append({"family": family, "path": str(relative), "train_subject_ids": json.dumps(train_ids),
                             "selected_by_inner_cv": index == best})
        candidate_rows = [{"candidate": i, "family": f, "params": json.dumps(p, sort_keys=True),
                           "mean_mae": float(means[i]), "fold_mae": errors[i].tolist(), "selected_overall": i == best}
                          for i, (f, p) in enumerate(candidates)]
    return {"group": group, "subject_id": sid, "predictions": rows, "manifest": manifest,
            "inner_folds": folds, "candidate_scores": candidate_rows, "warnings": notices}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    datasets, _, source_paths = prepare_datasets()
    followup, follow_targets = {}, {}
    for path in sorted((ROOT / "data/processed_follow_up").glob("CT_*_data.csv")):
        sid = path.stem.removesuffix("_data")
        header, rows = read_csv(path);cols = numeric_columns(header, rows)
        if "WorkRate" not in cols:
            cols["WorkRate"] = cols["WR_cortex"]
        followup[sid] = cols
        target_path = path.parent / f"{sid}_targets.csv"
        _, target = read_csv(target_path);follow_targets[sid] = float(target[0]["VO2peak_abs"])
        source_paths.extend([path, target_path])
    ids = sorted(followup)
    assert len(ids) == 15
    pre_ids, _, _, pre_y = datasets["all_signals"]
    pre_targets = dict(zip(pre_ids, pre_y))
    paired = [{"subject_id": sid, "actual_pre": float(pre_targets[sid]), "actual_post": follow_targets[sid],
               "actual_delta": float(follow_targets[sid] - pre_targets[sid])} for sid in ids]
    write_csv(OUT / "paired_targets.csv", paired)
    indices = np.random.default_rng(SEED).integers(0, len(ids), (RESAMPLES, len(ids)))
    # Artifact assignments come solely from frozen baseline-validation records;
    # no post-training outcome is used to select or fit transport models.
    _, manifest = read_csv(BASE / "artifact_manifest.csv")
    entries = [(BASE, r) for r in manifest if r["group"] in GROUPS and r["subject_id"] in ids]
    _, extra_manifest = read_csv(ROBUST / "additional_artifact_manifest.csv")
    entries.extend((ROBUST, r) for r in extra_manifest if r["group"] == "all_signals_without_cadence" and r["subject_id"] in ids)
    feature_cache, post_rows, model_qc = {}, [], []
    for folder, row in entries:
        group, sid, family = row["group"], row["subject_id"], row["family"]
        artifact = folder / row["artifact_path"]
        source_paths.append(artifact)
        payload = joblib.load(artifact)
        assert payload["left_out_subject"] == sid and sid not in payload["train_subject_ids"]
        names = payload["feature_columns"]
        if (group, sid) not in feature_cache:
            channels = list(dict.fromkeys(n.split("_mean_", 1)[0].split("_std_", 1)[0] for n in names))
            extracted = features(followup[sid], channels)
            feature_cache[group, sid] = {n: extracted.get(n, np.nan) for n in names}
        values = feature_cache[group, sid]
        X_post = np.array([[values[n] for n in names]])
        pred_post = float(np.asarray(payload["estimator"].predict(X_post)).ravel()[0])
        # Re-evaluate baseline using the very same frozen fitted object.
        baseline_group = group if group in datasets else "all_signals"
        b_ids, b_names, b_X, _ = datasets[baseline_group]
        b_cols = [b_names.index(n) for n in names]
        X_pre = b_X[[b_ids.index(sid)]][:, b_cols]
        pred_pre = float(np.asarray(payload["estimator"].predict(X_pre)).ravel()[0])
        support = payload["estimator"].named_steps["preprocess"].named_steps["missingness"].support_
        post_rows.append({"group": group, "subject_id": sid, "family": family,
                          "selected_by_baseline_inner_cv": str(row["selected_by_inner_cv"]) == "True",
                          "actual_pre": float(pre_targets[sid]), "actual_post": follow_targets[sid],
                          "pred_pre": pred_pre, "pred_post": pred_post,
                          "actual_delta": float(follow_targets[sid] - pre_targets[sid]), "pred_delta": pred_post - pred_pre,
                          "n_input_features": len(names), "n_retained_features": int(support.sum()),
                          "n_missing_post_retained": int(np.isnan(X_post[:, support]).sum()),
                          "n_missing_pre_retained": int(np.isnan(X_pre[:, support]).sum())})
        model_qc.append({"group": group, "subject_id": sid, "family": family,
                         "artifact": str(artifact.relative_to(ROOT)),
                         "train_subject_ids": json.dumps(payload["train_subject_ids"]),
                         "retained_features": json.dumps([n for n, keep in zip(names, support) if keep]),
                         "post_missing_features": json.dumps([n for n, keep, v in zip(names, support, X_post.ravel()) if keep and np.isnan(v)]),
                         "selected_by_baseline_inner_cv": str(row["selected_by_inner_cv"]) == "True"})
    write_csv(OUT / "frozen_model_predictions.csv", post_rows)
    write_csv(OUT / "frozen_model_qc.csv", model_qc)
    e1, e2, e3, error_rows = [], [], [], []
    for group in GROUPS + ("all_signals_without_cadence",):
        for procedure in ("Nested family selection",) + FAMILIES:
            rows = sorted([r for r in post_rows if r["group"] == group and
                           ((procedure == "Nested family selection" and r["selected_by_baseline_inner_cv"]) or r["family"] == procedure)], key=lambda r: r["subject_id"])
            assert [r["subject_id"] for r in rows] == ids
            actual_pre = np.array([r["actual_pre"] for r in rows]);actual_post = np.array([r["actual_post"] for r in rows])
            pred_pre = np.array([r["pred_pre"] for r in rows]);pred_post = np.array([r["pred_post"] for r in rows])
            e1.append({"group": group, "procedure": procedure, "N": len(ids),
                       **metrics(actual_post, pred_post, indices), **extra_metrics(actual_post, pred_post)})
            gap = np.abs(pred_post - actual_post) - np.abs(pred_pre - actual_pre)
            lo, hi = np.quantile(gap[indices].mean(axis=1), [0.025, 0.975])
            paired_t = stats.ttest_rel(np.abs(pred_post - actual_post), np.abs(pred_pre - actual_pre))
            wilcoxon = stats.wilcoxon(gap) if not np.allclose(gap, 0) else None
            e2.append({"group": group, "procedure": procedure, "N": len(ids),
                       "MAE_pre_same15": float(np.abs(pred_pre - actual_pre).mean()), "MAE_post": float(np.abs(pred_post - actual_post).mean()),
                       "delta_MAE_post_minus_pre": float(gap.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
                       "n_errors_smaller_post": int((gap < 0).sum()),
                       "paired_t_p_descriptive": float(paired_t.pvalue), "wilcoxon_p_descriptive": float(wilcoxon.pvalue) if wilcoxon else None})
            e3.append({"group": group, "procedure": procedure, "N": len(ids),
                       **delta_metrics(actual_post - actual_pre, pred_post - pred_pre, indices)})
            error_rows.extend({"group": group, "procedure": procedure, "subject_id": r["subject_id"],
                               "absolute_error_pre": float(abs(r["pred_pre"] - r["actual_pre"])),
                               "absolute_error_post": float(abs(r["pred_post"] - r["actual_post"])),
                               "actual_delta": r["actual_delta"], "pred_delta": r["pred_delta"]} for r in rows)
    write_csv(OUT / "e1_post_training_summary.csv", e1)
    write_csv(OUT / "e2_pre_post_error_gap.csv", e2)
    write_csv(OUT / "e3_change_agreement_summary.csv", e3)
    write_csv(OUT / "e2_e3_subject_records.csv", error_rows)

    actual_delta = np.array([r["actual_delta"] for r in paired])
    actual_post = np.array([r["actual_post"] for r in paired])
    actual_pre = np.array([r["actual_pre"] for r in paired])
    baseline_mean = np.array([np.mean([v for s, v in pre_targets.items() if s != sid]) for sid in ids])
    e1_benchmarks = [{"benchmark": name, "N": len(ids), **metrics(actual_post, p, indices), **extra_metrics(actual_post, p)}
                     for name, p in [("Baseline-training mean (other 31 subjects)", baseline_mean),
                                     ("Persistence of measured baseline VO2peak", actual_pre)]]
    write_csv(OUT / "e1_context_benchmarks.csv", e1_benchmarks)
    mean_delta = (actual_delta.sum() - actual_delta) / (len(ids) - 1)
    benchmarks = [{"benchmark": name, "N": len(ids), **delta_metrics(actual_delta, p, indices)}
                  for name, p in [("Zero change", np.zeros(len(ids))), ("Other14 mean observed change", mean_delta)]]
    write_csv(OUT / "change_benchmarks.csv", benchmarks)
    write_csv(OUT / "benchmark_subject_predictions.csv", [{"subject_id": sid, "actual_delta": float(actual_delta[i]),
                                                           "baseline_training_mean": float(baseline_mean[i]),
                                                           "other14_mean_change": float(mean_delta[i]), "zero_change": 0.0} for i, sid in enumerate(ids)])

    # Direct prospective delta prediction is trained separately on the paired 15.
    tasks = []
    for group in GROUPS:
        b_ids, names, b_X, _ = datasets[group]
        X = b_X[[b_ids.index(sid) for sid in ids]]
        for test, sid in enumerate(ids):
            train = np.array([i for i in range(len(ids)) if i != test])
            tasks.append(delayed(direct_delta_fold)(group, sid, [ids[i] for i in train], names, X[train], actual_delta[train], X[[test]]))
    direct = []
    with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
        for result in Parallel(n_jobs=4, return_as="generator_unordered")(tasks):
            direct.append(result)
            p = OUT / "direct_delta_folds" / result["group"] / f"{result['subject_id']}.json"
            p.parent.mkdir(parents=True, exist_ok=True);p.write_text(json.dumps(result, indent=2) + "\n")
            if len(direct) % 15 == 0:
                print(f"Direct-change outer folds: {len(direct)}/60", flush=True)
    e4_predictions, e4_manifest = [], []
    for result in sorted(direct, key=lambda r: (r["group"], r["subject_id"])):
        group, sid = result["group"], result["subject_id"]
        actual = float(actual_delta[ids.index(sid)])
        e4_predictions.extend({"group": group, "subject_id": sid, "actual_delta": actual, **r} for r in result["predictions"])
        e4_manifest.extend({"group": group, "subject_id": sid, **r} for r in result["manifest"])
    write_csv(OUT / "e4_direct_change_predictions.csv", e4_predictions)
    write_csv(OUT / "e4_artifact_manifest.csv", e4_manifest)
    e4, e4_comparisons = [], []
    for group in GROUPS:
        for procedure in ("Nested family selection",) + FAMILIES:
            rows = sorted([r for r in e4_predictions if r["group"] == group and
                           ((procedure == "Nested family selection" and r["selected_by_inner_cv"]) or r["family"] == procedure)], key=lambda r: r["subject_id"])
            assert [r["subject_id"] for r in rows] == ids
            p = np.array([r["predicted_delta"] for r in rows])
            e4.append({"group": group, "procedure": procedure, "N": len(ids), **delta_metrics(actual_delta, p, indices)})
            error_gap = np.abs(p - actual_delta) - np.abs(mean_delta - actual_delta)
            low, high = np.quantile(error_gap[indices].mean(axis=1), [0.025, 0.975])
            e4_comparisons.append({"group": group, "procedure": procedure, "reference": "Other14 mean observed change",
                                   "delta_MAE_model_minus_mean_change": float(error_gap.mean()), "ci95_low": float(low), "ci95_high": float(high),
                                   "max_prediction_difference_from_mean_change": float(np.max(np.abs(p - mean_delta)))})
    write_csv(OUT / "e4_direct_change_summary.csv", e4)
    write_csv(OUT / "e4_vs_training_mean.csv", e4_comparisons)

    changes = []
    for group in GROUPS:
        for family in FAMILIES:
            _, old = read_csv(ROOT / "results" / f"e1_zero_shot_{group}_{family}_predictions.csv")
            old_map = {r["subject_id"]: float(r["predicted_vo2peak"]) for r in old}
            current = [r for r in post_rows if r["group"] == group and r["family"] == family]
            diff = max(abs(r["pred_post"] - old_map[r["subject_id"]]) for r in current)
            changes.append({"group": group, "family": family, "N": len(current), "max_abs_difference_from_original_post_predictions": diff})
            source_paths.append(ROOT / "results" / f"e1_zero_shot_{group}_{family}_predictions.csv")
    write_csv(OUT / "fixed_family_reproduction_checks.csv", changes)
    mean_change = float(actual_delta.mean());se = stats.sem(actual_delta)
    low, high = stats.t.interval(0.95, len(ids) - 1, loc=mean_change, scale=se)
    cohort = {"N": len(ids), "pre_mean": float(actual_pre.mean()), "post_mean": float(actual_post.mean()),
              "mean_change": mean_change, "mean_change_ci95_low": float(low), "mean_change_ci95_high": float(high),
              "paired_t_p": float(stats.ttest_rel(actual_post, actual_pre).pvalue),
              "wilcoxon_p": float(stats.wilcoxon(actual_delta).pvalue),
              "n_improved": int((actual_delta > 0).sum()), "n_declined": int((actual_delta < 0).sum())}
    (OUT / "cohort_change_statistics.json").write_text(json.dumps(cohort, indent=2) + "\n")
    source_paths.extend([BASE / "artifact_manifest.csv", BASE / "nested_selection_predictions.csv", ROBUST / "additional_artifact_manifest.csv",
                         ROOT / "results/q1_q2_vo2peak_abs_paired_table.csv", ROOT / "notebooks/10_e4_baseline_to_delta_modeling.ipynb",
                         Path(__file__).resolve(), ROOT / "scripts/run_nested_revision.py", ROOT / "evaluation/revision_modeling.py"])
    write_csv(OUT / "input_provenance.csv", [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in dict.fromkeys(source_paths)])
    summary = {"followup_participants": len(ids), "transport_groups": len(GROUPS) + 1, "frozen_model_inferences": len(post_rows),
               "direct_delta_outer_folds": len(direct), "direct_delta_saved_models": len(e4_manifest),
               "direct_delta_initial_fit_warnings": sum(len(r["warnings"]) for r in direct),
               "direct_delta_solver_warnings_resolved": sum(n["resolved_by_solver_retry"] for r in direct for n in r["warnings"]),
               "direct_delta_unresolved_fit_warnings": sum(not n["resolved_by_solver_retry"] for r in direct for n in r["warnings"]),
               "bootstrap_seed": SEED, "bootstrap_resamples": RESAMPLES,
               "primary_post_training": next(r for r in e1 if r["group"] == "all_signals" and r["procedure"] == "Nested family selection"),
               "primary_error_gap": next(r for r in e2 if r["group"] == "all_signals" and r["procedure"] == "Nested family selection"),
               "primary_change_tracking": next(r for r in e3 if r["group"] == "all_signals" and r["procedure"] == "Nested family selection"),
               "primary_direct_change": next(r for r in e4 if r["group"] == "all_signals" and r["procedure"] == "Nested family selection"),
               "max_original_fixed_family_post_difference": max(r["max_abs_difference_from_original_post_predictions"] for r in changes),
               "cohort": cohort}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
