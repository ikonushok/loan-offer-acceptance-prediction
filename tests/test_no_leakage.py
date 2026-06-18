"""Leakage protected contracts (AGENTS.md):

- target_value is never a model feature.
- Engineered features are label-free (do not change when the target is permuted).
- Time folds never leak the future: train period strictly precedes validation period.
- front_id / decision_day are excluded from the feature matrix.
"""

import numpy as np
import pandas as pd
import pytest

from baseline_catboost_time import (
    TARGET, ID_COL, DAY_COL, prepare_features, make_time_folds,
)

FE = {"enabled": True, "add_pairwise_features": False,
      "add_missing_flags": False, "add_context_offer_features": True}
CUTOFFS = ["2025-01-01", "2025-03-01", "2025-04-01"]


@pytest.fixture(scope="module")
def small(train_df, test_df):
    return train_df.sample(4000, random_state=0).reset_index(drop=True), test_df.sample(2000, random_state=0).reset_index(drop=True)


def test_target_and_ids_not_in_features(small):
    tr, te = small
    tx, ex, y, cats, cols, _ = prepare_features(tr, te, FE, {"enabled": False})
    assert TARGET not in cols, "target_value must not be a feature"
    assert ID_COL not in cols, "front_id must not be a feature"
    assert DAY_COL not in cols, "decision_day must not be a raw feature when time features disabled"


def test_engineered_features_are_label_free(small):
    """Permuting the target must not change the feature matrix (no target-derived features)."""
    tr, te = small
    tx1, _, _, _, cols1, _ = prepare_features(tr, te, FE, {"enabled": False})
    tr2 = tr.copy()
    tr2[TARGET] = np.random.RandomState(1).permutation(tr2[TARGET].to_numpy())
    tx2, _, _, _, cols2, _ = prepare_features(tr2, te, FE, {"enabled": False})
    assert cols1 == cols2
    pd.testing.assert_frame_equal(tx1.reset_index(drop=True), tx2.reset_index(drop=True))


def test_time_folds_have_no_future_leak(train_df):
    folds = make_time_folds(train_df, CUTOFFS)
    assert len(folds) == 3
    days = pd.to_datetime(train_df[DAY_COL])
    for f in folds:
        tr_days = days.iloc[f["train_idx"]]
        va_days = days.iloc[f["valid_idx"]]
        assert tr_days.max() < va_days.min(), f"fold {f['fold']}: train must strictly precede valid"
        assert len(set(f["train_idx"]) & set(f["valid_idx"])) == 0, "train/valid indices must not overlap"


def test_train_test_feature_columns_aligned(small):
    tr, te = small
    tx, ex, *_ = prepare_features(tr, te, FE, {"enabled": False})
    assert list(tx.columns) == list(ex.columns), "train/test feature columns must align"
