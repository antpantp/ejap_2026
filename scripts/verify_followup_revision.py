"""Verify frozen-model inference, direct-change exclusion, and nested score replay."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from audit_revision_step1 import features, numeric_columns
from run_nested_revision import prepare_datasets, read_csv
from run_followup_revision import OUT, BASE, ROBUST, delta_estimator, delta_preprocessing


def main():
    datasets, _, _ = prepare_datasets()
    _, targets = read_csv(OUT / "paired_targets.csv")
    ids = [r["subject_id"] for r in targets]
    actual_delta = np.array([float(r["actual_delta"]) for r in targets])
    _, old_targets = read_csv(ROOT / "results/q1_q2_vo2peak_abs_paired_table.csv")
    old_targets = {r["subject_id"]: r for r in old_targets}
    for row in targets:
        old = old_targets[row["subject_id"]]
        assert np.isclose(float(row["actual_pre"]), float(old["VO2peak_abs_pre"]), atol=1e-12, rtol=0)
        assert np.isclose(float(row["actual_post"]), float(old["VO2peak_abs_post"]), atol=1e-12, rtol=0)
    _, frozen = read_csv(OUT / "frozen_model_predictions.csv")
    _, qc = read_csv(OUT / "frozen_model_qc.csv")
    _, base_family = read_csv(BASE / "fixed_family_predictions.csv")
    _, extra_family = read_csv(ROBUST / "additional_predictions.csv")
    accepted_pre = {(r["group"], r["subject_id"], r["family"]): r for r in base_family + extra_family}
    lookup = {(r["group"], r["subject_id"], r["family"]): r for r in frozen}
    max_post_difference = 0.0
    for row in qc:
        group, sid, family = row["group"], row["subject_id"], row["family"]
        payload = joblib.load(ROOT / row["artifact"])
        assert sid not in payload["train_subject_ids"] and payload["left_out_subject"] == sid
        assert payload["train_subject_ids"] == json.loads(row["train_subject_ids"])
        h, raw = read_csv(ROOT / "data/processed_follow_up" / f"{sid}_data.csv")
        cols = numeric_columns(h, raw)
        if "WorkRate" not in cols:cols["WorkRate"] = cols["WR_cortex"]
        names = payload["feature_columns"]
        channels = list(dict.fromkeys(n.split("_mean_", 1)[0].split("_std_", 1)[0] for n in names))
        derived = features(cols, channels)
        X = np.array([[derived.get(n, np.nan) for n in names]])
        pred = float(np.asarray(payload["estimator"].predict(X)).ravel()[0])
        stored = lookup[group, sid, family]
        assert abs(float(stored["pred_pre"]) - float(accepted_pre[group, sid, family]["predicted_vo2peak"])) < 1e-10
        diff = abs(pred - float(stored["pred_post"]))
        max_post_difference = max(max_post_difference, diff)
        assert diff < 1e-10
        assert np.isclose(float(stored["pred_delta"]), float(stored["pred_post"]) - float(stored["pred_pre"]), atol=1e-12)
        assert bool(payload["selected_by_inner_cv"]) == (stored["selected_by_baseline_inner_cv"] == "True")
        support = payload["estimator"].named_steps["preprocess"].named_steps["missingness"].support_
        assert int(stored["n_missing_post_retained"]) == int(np.isnan(X[:, support]).sum())
    _, baseline_selected = read_csv(BASE / "nested_selection_predictions.csv")
    baseline_map = {(r["group"], r["subject_id"]): r for r in baseline_selected}
    for row in frozen:
        if row["group"] in datasets and row["selected_by_baseline_inner_cv"] == "True":
            old = baseline_map[row["group"], row["subject_id"]]
            assert old["family"] == row["family"]
            assert abs(float(old["predicted_vo2peak"]) - float(row["pred_pre"])) < 1e-10

    _, manifest = read_csv(OUT / "e4_artifact_manifest.csv")
    _, predictions = read_csv(OUT / "e4_direct_change_predictions.csv")
    direct_map = {(r["group"], r["subject_id"], r["family"]): r for r in predictions}
    max_direct_difference = 0.0
    for row in manifest:
        group, sid, family = row["group"], row["subject_id"], row["family"]
        b_ids, names, b_X, _ = datasets[group]
        X = b_X[[b_ids.index(i) for i in ids]]
        test = ids.index(sid);train = [i for i in range(len(ids)) if i != test]
        payload = joblib.load(OUT / row["path"])
        assert payload["train_subject_ids"] == [ids[i] for i in train]
        assert payload["left_out_subject"] == sid and sid not in payload["train_subject_ids"]
        assert payload["feature_columns"] == names
        prep = payload["estimator"].named_steps["preprocess"]
        support = np.isfinite(X[train]).mean(axis=0) >= 0.6
        assert np.array_equal(prep.named_steps["missingness"].support_, support)
        assert np.allclose(prep.named_steps["imputer"].statistics_, np.nanmedian(X[train][:, support], axis=0), atol=1e-12, rtol=1e-12)
        pred = float(np.asarray(payload["estimator"].predict(X[[test]])).ravel()[0])
        diff = abs(pred - float(direct_map[group, sid, family]["predicted_delta"]))
        max_direct_difference = max(max_direct_difference, diff)
        assert diff < 1e-10
    split_count = 0
    records = {}
    for path in (OUT / "direct_delta_folds").rglob("*.json"):
        record = json.loads(path.read_text());sid = record["subject_id"];group = record["group"]
        records[group, sid] = record
        b_ids, names, b_X, _ = datasets[group]
        validation_ids = []
        for fold in record["inner_folds"]:
            train, val = fold["train_ids"], fold["validation_ids"]
            assert sid not in train + val and set(train).isdisjoint(val)
            assert set(train) | set(val) == set(ids) - {sid}
            selected = np.isfinite(b_X[[b_ids.index(i) for i in train]]).mean(axis=0) >= 0.6
            assert fold["retained_features"] == [n for n, keep in zip(names, selected) if keep]
            validation_ids.extend(val);split_count += 1
        assert sorted(validation_ids) == sorted(set(ids) - {sid})
        best = min(record["candidate_scores"], key=lambda r: r["mean_mae"])
        chosen = [r for r in record["predictions"] if r["selected_by_inner_cv"]]
        assert len(chosen) == 1 and chosen[0]["family"] == best["family"] and chosen[0]["params"] == best["params"]
    replays = []
    with threadpool_limits(limits=1):
        for group, sid in [("all_signals", "CT_01"), ("nirs", "CT_01")]:
            record = records[group, sid]
            b_ids, names, b_X, _ = datasets[group]
            train_ids = [i for i in ids if i != sid]
            X = b_X[[b_ids.index(i) for i in train_ids]]
            y = actual_delta[[ids.index(i) for i in train_ids]]
            grid = []
            for candidate in record["candidate_scores"]:
                model = delta_estimator(candidate["family"], json.loads(candidate["params"]))
                if candidate["family"] == "ElasticNet":model.set_params(max_iter=100000)
                grid.append({"model": [model]})
            pipe = Pipeline([("preprocess", delta_preprocessing()), ("model", delta_estimator("Ridge", {"alpha": 1}))])
            cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
            search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", refit=False, n_jobs=1, error_score="raise")
            search.fit(X, y)
            diff = float(np.max(np.abs(-search.cv_results_["mean_test_score"] - [r["mean_mae"] for r in record["candidate_scores"]])))
            assert diff < 1e-10
            assert int(search.best_index_) == int(np.argmin([r["mean_mae"] for r in record["candidate_scores"]]))
            replays.append({"group": group, "subject_id": sid, "max_inner_mae_difference": diff, "same_selected_candidate": True})
    _, hashes = read_csv(OUT / "input_provenance.csv")
    for row in hashes:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    _, benchmarks = read_csv(OUT / "benchmark_subject_predictions.csv")
    for row in benchmarks:
        test = ids.index(row["subject_id"])
        assert abs(float(row["other14_mean_change"]) - float(np.delete(actual_delta, test).mean())) < 1e-12
    summary = {"frozen_model_inferences_checked": len(qc), "max_frozen_prediction_difference": max_post_difference,
               "direct_change_models_checked": len(manifest), "max_direct_prediction_difference": max_direct_difference,
               "direct_change_inner_splits_checked": split_count, "direct_change_outer_selections_checked": len(records),
               "no_subject_present_in_own_model_training": True, "baseline_artifact_assignments_preserved": True,
               "paired_labels_preserved": True, "all_baseline_predictions_match_accepted_artifacts": True,
               "other14_mean_benchmark_exclusion_verified": True,
               "source_hashes_checked": len(hashes), "independent_grid_search_replays": replays}
    (OUT / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
