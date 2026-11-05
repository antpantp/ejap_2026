"""Verify fold separation, preprocessing, serialized predictions, and provenance.

Run with python scripts/verify_nested_revision.py.
No models are refitted and original files are not modified.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from run_nested_revision import ROOT, prepare_datasets, read_csv, write_csv


def main():
    out = ROOT / "results/revision_round1/nested_model_selection"
    datasets, _, _ = prepare_datasets()
    _, pred_rows = read_csv(out / "fixed_family_predictions.csv")
    _, selected = read_csv(out / "nested_selection_predictions.csv")
    _, manifest = read_csv(out / "artifact_manifest.csv")
    _, inner = read_csv(out / "inner_fold_membership.csv")
    _, candidates = read_csv(out / "inner_candidate_scores.csv")
    predictions = {(r["group"], r["subject_id"], r["family"]): r for r in pred_rows}
    by_outer = defaultdict(list)
    for row in inner:
        by_outer[row["group"], row["subject_id"]].append(row)
    candidates_by_outer = defaultdict(list)
    for row in candidates:
        candidates_by_outer[row["group"], row["subject_id"]].append(row)
    max_prediction_error, outer_checks, inner_checks = 0.0, 0, 0
    for row in manifest:
        group, sid, family = row["group"], row["subject_id"], row["family"]
        ids, names, X, y = datasets[group]
        test_index = ids.index(sid)
        train = np.array([i for i, other in enumerate(ids) if other != sid])
        train_ids = [ids[i] for i in train]
        payload = joblib.load(out / row["artifact_path"])
        assert payload["left_out_subject"] == sid and payload["train_subject_ids"] == train_ids
        assert sid not in payload["train_subject_ids"]
        assert payload["feature_columns"] == names
        assert json.loads(row["train_subject_ids"]) == train_ids
        fitted = payload["estimator"]
        prep = fitted.named_steps["preprocess"]
        support = np.isfinite(X[train]).mean(axis=0) >= 0.6
        assert np.array_equal(prep.named_steps["missingness"].support_, support)
        assert payload["retained_features"] == [n for n, keep in zip(names, support) if keep]
        values = X[train][:, support]
        scales = np.nanstd(values, axis=0, ddof=0)
        scales[~np.isfinite(scales) | (scales == 0)] = 1
        assert np.allclose(prep.named_steps["unit_rescaler"].scales_, scales, atol=1e-12, rtol=1e-12)
        median = np.nanmedian(values / scales, axis=0)
        assert np.allclose(prep.named_steps["imputer"].statistics_, median, atol=1e-12, rtol=1e-12)
        imputed = np.where(np.isnan(values / scales), median, values / scales)
        assert np.allclose(prep.named_steps["scaler"].mean_, imputed.mean(axis=0), atol=1e-12, rtol=1e-12)
        predicted = float(np.asarray(fitted.predict(X[[test_index]])).ravel()[0])
        stored = predictions[group, sid, family]
        difference = abs(predicted - float(stored["predicted_vo2peak"]))
        max_prediction_error = max(max_prediction_error, difference)
        assert difference < 1e-10
        assert np.isclose(float(stored["actual_vo2peak"]), y[test_index])
        assert payload["best_params"] == json.loads(stored["params"])
        outer_checks += 1
    for (group, sid), folds in by_outer.items():
        ids, names, X, _ = datasets[group]
        expected = set(ids) - {sid}
        validation_counts = Counter()
        assert len(folds) == 5
        for row in folds:
            train_ids = json.loads(row["train_subject_ids"])
            val_ids = json.loads(row["validation_subject_ids"])
            assert set(train_ids).isdisjoint(val_ids)
            assert sid not in train_ids + val_ids
            assert set(train_ids) | set(val_ids) == expected
            validation_counts.update(val_ids)
            mask = np.isfinite(X[[ids.index(i) for i in train_ids]]).mean(axis=0) >= 0.6
            assert json.loads(row["retained_features"]) == [n for n, keep in zip(names, mask) if keep]
            inner_checks += 1
        assert set(validation_counts) == expected and set(validation_counts.values()) == {1}
        scores = candidates_by_outer[group, sid]
        best = min(scores, key=lambda r: float(r["mean_inner_mse"]))
        chosen = [r for r in selected if r["group"] == group and r["subject_id"] == sid]
        assert len(chosen) == 1
        assert chosen[0]["family"] == best["family"] and chosen[0]["params"] == best["params"]
    _, source_hashes = read_csv(out / "input_provenance.csv")
    for row in source_hashes:
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]

    # Identify differences attributable to the training-only filter relative to
    # saved family-specific evaluations; do not confuse these with family choice.
    old_cache, differences = {}, []
    for row in pred_rows:
        group, family, sid = row["group"], row["family"], row["subject_id"]
        key = group, family
        if key not in old_cache:
            _, old = read_csv(ROOT / "results" / f"ablation_{group}_{family}_loso_predictions.csv")
            old_cache[key] = {r["subject_id"]: float(r["predicted_vo2peak"]) for r in old}
        delta = float(row["predicted_vo2peak"]) - old_cache[key][sid]
        changed_inner = any(int(r["n_retained_features"]) != int(row["n_retained_features"]) for r in by_outer[group, sid])
        if abs(delta) > 1e-7:
            assert changed_inner, (group, family, sid, delta)
        differences.append({"group": group, "family": family, "subject_id": sid,
                            "revised_minus_saved_prediction": delta,
                            "any_inner_fold_changes_feature_schema": changed_inner})
    write_csv(out / "fixed_family_changes_from_saved.csv", differences)
    summary = {"serialized_models_checked": outer_checks, "inner_splits_checked": inner_checks,
               "outer_selections_checked": len(by_outer), "input_hashes_checked": len(source_hashes),
               "max_serialized_prediction_difference": max_prediction_error,
               "fixed_family_predictions_changed_above_1e_7": sum(abs(r["revised_minus_saved_prediction"]) > 1e-7 for r in differences),
               "all_larger_fixed_family_changes_localized_to_changed_inner_schemas": True,
               "heldout_subjects_excluded_from_all_inner_and_outer_training": True,
               "training_only_feature_filters_imputers_and_scalers_verified": True}
    (out / "verification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
