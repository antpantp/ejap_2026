"""Exercise nested selection on generated values, without reading study data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from evaluation.revision_modeling import evaluate_candidates, pipeline
from audit_revision_step1 import features


def main():
    rng = np.random.default_rng(20261104)
    X = rng.normal(size=(32, 6))
    y = 3 + .4 * X[:, 0] - .2 * X[:, 1] + rng.normal(scale=.08, size=32)
    X[:4, 2] = np.nan
    X[:25, 5] = np.nan
    ids = [f'synthetic_{i:02d}' for i in range(26)]
    names = [f'x{i}' for i in range(6)]
    candidates, best, _, scores, folds, _ = evaluate_candidates(X[:26], y[:26], ids, names)
    assert scores[best]['mean_inner_mse'] == min(row['mean_inner_mse'] for row in scores)
    for fold in folds:
        import json
        train = set(json.loads(fold['train_subject_ids']))
        validation = set(json.loads(fold['validation_subject_ids']))
        assert train.isdisjoint(validation)
    family, params = candidates[best]
    model = pipeline(family, params).fit(X[:26], y[:26])
    prediction = np.asarray(model.predict(X[26:])).ravel()
    assert prediction.shape == (6,) and np.isfinite(prediction).all()
    assert not model.named_steps['preprocess'].named_steps['missingness'].support_[5]
    assert np.mean((prediction - y[26:]) ** 2) < .2
    time = np.arange(1200.)
    work = np.where((time.astype(int) // 20) % 2, 80, 30)
    descriptor = features({'Time': time, 'WorkRate': work, 'synthetic': work * .01}, ['synthetic'])
    assert np.isclose(descriptor['synthetic_mean_delta_wr80_minus_wr30'], .5)
    print(f'Synthetic smoke check passed: {len(candidates)} candidates, five inner folds, {family} selected.')
    print('No study data were read; study numerical results were not reproduced.')


if __name__ == '__main__':
    main()
