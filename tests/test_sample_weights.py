"""Sample-weight invariants (AGENTS.md: weights must be label-free, positive, finite).

Covers recency strategies and the adversarial_file strategy used for drift adaptation.
"""

import numpy as np
import pandas as pd
import pytest

from baseline_catboost_time import make_sample_weights


def test_disabled_returns_none(train_df):
    assert make_sample_weights(train_df, None) is None
    assert make_sample_weights(train_df, {"enabled": False}) is None


def test_exponential_recency_positive_finite(train_df):
    w = make_sample_weights(train_df, {
        "enabled": True, "strategy": "exponential_recency",
        "half_life_days": 120.0, "min_weight": 0.25, "normalize_mean": True,
    })
    assert len(w) == len(train_df)
    assert np.isfinite(w).all()
    assert (w > 0).all()
    assert w.mean() == pytest.approx(1.0, rel=1e-6)


def test_adversarial_file_maps_all_ids(train_df, tmp_path):
    # synthetic weight file covering every train front_id
    wf = tmp_path / "w.csv"
    pd.DataFrame({"front_id": train_df["front_id"], "sample_weight": 1.0}).to_csv(wf, index=False)
    w = make_sample_weights(train_df, {
        "enabled": True, "strategy": "adversarial_file",
        "weights_path": str(wf), "normalize_mean": False,
    })
    assert len(w) == len(train_df)
    assert (w > 0).all() and np.isfinite(w).all()


def test_adversarial_file_missing_id_raises(train_df, tmp_path):
    wf = tmp_path / "w_partial.csv"
    partial = train_df["front_id"].iloc[:-10]  # drop some ids
    pd.DataFrame({"front_id": partial, "sample_weight": 1.0}).to_csv(wf, index=False)
    with pytest.raises(ValueError, match="missing"):
        make_sample_weights(train_df, {
            "enabled": True, "strategy": "adversarial_file",
            "weights_path": str(wf), "normalize_mean": False,
        })
