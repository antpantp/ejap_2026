"""Training-fold preprocessing and model selection for the round-1 revision.

Classes live in an importable module so saved pipelines do not depend on a
notebook's or command-line script's __main__ namespace.
"""

import json
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


FAMILIES = ("Ridge", "ElasticNet", "PLS")


class TrainingMissingnessFilter(TransformerMixin, BaseEstimator):
    def __init__(self, min_valid_fraction=0.6):
        self.min_valid_fraction = min_valid_fraction

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.valid_fraction_ = np.isfinite(values).mean(axis=0)
        self.support_ = self.valid_fraction_ >= self.min_valid_fraction
        if not self.support_.any():
            raise ValueError("No features meet the training-fold availability threshold")
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        values = np.asarray(X, dtype=float)
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Input schema differs from fitted feature schema")
        return values[:, self.support_]


class FoldwiseUnitRescaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.scales_ = np.nanstd(values, axis=0, ddof=0)
        self.scales_[~np.isfinite(self.scales_) | (self.scales_ == 0)] = 1.0
        return self

    def transform(self, X):
        check_is_fitted(self, "scales_")
        return np.asarray(X, dtype=float) / self.scales_


def preprocessing():
    return Pipeline([
        ("missingness", TrainingMissingnessFilter()),
        ("unit_rescaler", FoldwiseUnitRescaler()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def estimator(family, params):
    if family == "Ridge":
        return Ridge(**params)
    if family == "ElasticNet":
        return ElasticNet(max_iter=50000, tol=1e-4, random_state=42, **params)
    if family == "PLS":
        return PLSRegression(**params)
    raise ValueError(f"Unknown model family {family}")


def pipeline(family, params):
    return Pipeline([("preprocess", preprocessing()), ("model", estimator(family, params))])


def inner_design(X_train):
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
    retained_counts = [int((np.isfinite(X_train[t]).mean(axis=0) >= 0.6).sum()) for t, _ in splits]
    max_components = min(10, min(retained_counts), min(len(t) for t, _ in splits))
    return splits, max_components


def candidate_grid(max_components):
    candidates = [("Ridge", {"alpha": float(a)}) for a in np.logspace(-3, 2, 12)]
    candidates.extend(("ElasticNet", {"alpha": float(a), "l1_ratio": ratio})
                      for a in np.logspace(-3, 1, 12) for ratio in (0.2, 0.4, 0.6, 0.8))
    candidates.extend(("PLS", {"n_components": n}) for n in range(1, max_components + 1))
    return candidates


def evaluate_candidates(X_train, y_train, train_ids, feature_names):
    """Return inner scores without accessing outer-test features or outcomes.

    Training-only preprocessing is shared across candidates within a split.
    This is equivalent to fitting the same preprocessing pipeline independently
    for every candidate, and is checked against GridSearchCV by the runner.
    """
    splits, max_components = inner_design(X_train)
    candidates = candidate_grid(max_components)
    scores = np.zeros((len(candidates), len(splits)))
    notices, fold_records = [], []
    for fold, (train, validation) in enumerate(splits):
        prep = preprocessing().fit(X_train[train])
        X_inner = prep.transform(X_train[train])
        X_validation = prep.transform(X_train[validation])
        retained = prep.named_steps["missingness"].support_
        fold_records.append({"inner_fold": fold + 1,
                             "train_subject_ids": json.dumps([train_ids[i] for i in train]),
                             "validation_subject_ids": json.dumps([train_ids[i] for i in validation]),
                             "n_retained_features": int(retained.sum()),
                             "retained_features": json.dumps([n for n, keep in zip(feature_names, retained) if keep])})
        for candidate, (family, params) in enumerate(candidates):
            model = estimator(family, params)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(X_inner, y_train[train])
                pred = np.asarray(model.predict(X_validation)).ravel()
            scores[candidate, fold] = np.mean((pred - y_train[validation]) ** 2)
            for notice in caught:
                notices.append({"stage": "inner", "inner_fold": fold + 1, "candidate": candidate,
                                "family": family, "params": json.dumps(params, sort_keys=True),
                                "category": notice.category.__name__, "message": str(notice.message)})
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite inner validation scores")
    means = scores.mean(axis=1)
    overall_best = int(np.argmin(means))
    family_best = {family: min((i for i, (f, _) in enumerate(candidates) if f == family), key=lambda i: means[i])
                   for family in FAMILIES}
    cv_records = []
    for i, (family, params) in enumerate(candidates):
        cv_records.append({"candidate": i, "family": family, "params": json.dumps(params, sort_keys=True),
                           "mean_inner_mse": float(means[i]),
                           **{f"inner_mse_fold{k + 1}": float(scores[i, k]) for k in range(len(splits))},
                           "best_within_family": i == family_best[family], "selected_overall": i == overall_best})
    return candidates, overall_best, family_best, cv_records, fold_records, notices
