"""Check additional models, paired cohorts, bootstrap grouping and refit replay."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from run_nested_revision import prepare_datasets, read_csv
from run_sensor_robustness import BASE, OUT, variant_datasets
from evaluation.revision_modeling import pipeline


def main():
    datasets, _, _ = prepare_datasets()
    variants = variant_datasets(datasets)
    _, predictions = read_csv(OUT / "additional_predictions.csv")
    lookup = {(r["group"], r["subject_id"], r["family"]): r for r in predictions}
    _, manifest = read_csv(OUT / "additional_artifact_manifest.csv")
    max_pred_error = 0.0
    for row in manifest:
        ids, names, X, _ = variants[row["group"]]
        sid = row["subject_id"]
        test = ids.index(sid)
        train = [i for i, other in enumerate(ids) if other != sid]
        payload = joblib.load(OUT / row["artifact_path"])
        assert payload["left_out_subject"] == sid
        assert payload["train_subject_ids"] == [ids[i] for i in train]
        assert payload["feature_columns"] == names
        support = np.isfinite(X[train]).mean(axis=0) >= 0.6
        assert np.array_equal(payload["estimator"].named_steps["preprocess"].named_steps["missingness"].support_, support)
        pred = float(np.asarray(payload["estimator"].predict(X[[test]])).ravel()[0])
        diff = abs(pred - float(lookup[row["group"], sid, row["family"]]["predicted_vo2peak"]))
        max_pred_error = max(max_pred_error, diff)
        assert diff < 1e-10
    # The full and reduced hemodynamics evaluations must have identical train IDs.
    _, base_manifest = read_csv(BASE / "artifact_manifest.csv")
    hemo = {r["subject_id"]: json.loads(r["train_subject_ids"]) for r in base_manifest if r["group"] == "hemodynamics" and r["family"] == "Ridge"}
    for row in manifest:
        if row["group"].endswith("matched_hemodynamics"):
            assert json.loads(row["train_subject_ids"]) == hemo[row["subject_id"]]
    records = json.loads((OUT / "ridge_bootstrap_records.json").read_text())
    ids, names, X, y = datasets["all_signals"]
    split_count = 0
    for record in records:
        sample = np.array(record["sample_indices"])
        groups = np.array(ids)[sample]
        support = np.isfinite(X[sample]).mean(axis=0) >= 0.6
        assert np.array_equal(record["support"], support)
        assert int(np.argmin(record["candidate_mean_mse"])) == int(np.argmin(abs(np.logspace(-3, 2, 12) - record["selected_alpha"])))
        validation_counts = np.zeros(len(sample), dtype=int)
        for fold in record["inner_folds"]:
            train = np.array(json.loads(fold["train_row_indices"]))
            val = np.array(json.loads(fold["validation_row_indices"]))
            assert set(groups[train]).isdisjoint(groups[val])
            assert set(train).isdisjoint(val) and set(train) | set(val) == set(range(32))
            assert set(groups[train]) == set(json.loads(fold["train_unique_subject_ids"]))
            assert set(groups[val]) == set(json.loads(fold["validation_unique_subject_ids"]))
            validation_counts[val] += 1
            split_count += 1
        assert np.all(validation_counts == 1)
    replay = []
    with threadpool_limits(limits=1):
        for rep in (0, 1, 2, 42, 499):
            record = records[rep]
            sample = np.array(record["sample_indices"])
            splits = [(np.array(json.loads(f["train_row_indices"])), np.array(json.loads(f["validation_row_indices"]))) for f in record["inner_folds"]]
            search = GridSearchCV(pipeline("Ridge", {"alpha": 1}), {"model__alpha": np.logspace(-3, 2, 12)},
                                  cv=splits, scoring="neg_mean_squared_error", refit=True, error_score="raise")
            search.fit(X[sample], y[sample])
            assert search.best_params_["model__alpha"] == record["selected_alpha"]
            score_diff = float(np.max(np.abs(-search.cv_results_["mean_test_score"] - record["candidate_mean_mse"])))
            support = np.array(record["support"], dtype=bool)
            coef_diff = float(np.max(np.abs(search.best_estimator_.named_steps["model"].coef_.ravel() - np.array(record["coef"])[support])))
            assert score_diff < 1e-10 and coef_diff < 1e-10
            replay.append({"replicate": rep, "max_cv_mse_difference": score_diff, "max_refit_coef_difference": coef_diff})
    _, hashes = read_csv(OUT / "input_provenance.csv")
    for row in hashes:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    summary = {"serialized_additional_models_checked": len(manifest), "max_serialized_prediction_difference": max_pred_error,
               "bootstrap_records_checked": len(records), "duplicate_safe_inner_splits_checked": split_count,
               "matched_hemodynamics_training_cohorts_verified": True, "input_hashes_checked": len(hashes),
               "independent_grid_search_replays": replay}
    (OUT / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
