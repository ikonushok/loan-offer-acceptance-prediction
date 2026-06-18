#!/usr/bin/env python
"""Probe whether self-training pseudo-labeling helps on a future holdout WITH known labels.

Design (scientifically valid offline test):
  - Treat H3 (2025-05-01 .. 2025-06-05) as a stand-in "test" period where we DO have labels.
  - Base model: trial0070 params, trained on train rows < 2025-05-01.
  - Predict on H3, take confident predictions as pseudo-labels.
  - Retrain on (train<cutoff) + (confident H3 pseudo rows, reduced weight).
  - Compare AUC on H3 TRUE labels: base vs pseudo-augmented.

If pseudo-labeling improves AUC on H3 true labels, it is worth applying to the real test.
Otherwise it is confirmed unhelpful and we save an upload.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_catboost_time import DAY_COL, TARGET, prepare_features

REPO = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2025-05-01")
H3_END = pd.Timestamp("2025-06-05")

TRIAL0070 = dict(
    iterations=1347, learning_rate=0.0207102246213281, depth=6,
    l2_leaf_reg=4.71255063500703, auto_class_weights="SqrtBalanced",
    bagging_temperature=0.2762006775607772, border_count=128,
    grow_policy="Depthwise", min_data_in_leaf=79, random_strength=1.269328964812104,
    random_seed=42, verbose=0,
)
FE = {"enabled": True, "add_pairwise_features": False, "add_missing_flags": False, "add_context_offer_features": True}


def fit_predict(train_df, eval_df, weights=None):
    eval_df = eval_df.drop(columns=[TARGET]) if TARGET in eval_df.columns else eval_df
    tx, ex, y, cats, cols, _ = prepare_features(train_df, eval_df, FE, {"enabled": False})
    m = CatBoostClassifier(**TRIAL0070)
    m.fit(Pool(tx, y, cat_features=cats, weight=weights), verbose=0)
    return m.predict_proba(Pool(ex, cat_features=cats))[:, 1]


def main():
    train = pd.read_csv(REPO / "data/raw/train_apps.csv")
    days = pd.to_datetime(train[DAY_COL])

    base_tr = train[days < CUTOFF].reset_index(drop=True)
    h3 = train[(days >= CUTOFF) & (days <= H3_END)].reset_index(drop=True)
    y_h3 = h3[TARGET].to_numpy()
    print(f"base train rows: {len(base_tr)} | H3 rows: {len(h3)} | H3 pos rate: {y_h3.mean():.4f}")

    # ── Base model ──
    base_pred = fit_predict(base_tr, h3)
    base_auc = roc_auc_score(y_h3, base_pred)
    print(f"\nBASE H3 AUC: {base_auc:.6f}")

    # ── Pseudo-labeling: confident H3 predictions become pseudo train rows ──
    for thr_hi, thr_lo, pw in [(0.9, 0.05, 0.3), (0.85, 0.1, 0.3), (0.9, 0.05, 0.5)]:
        pseudo_mask = (base_pred > thr_hi) | (base_pred < thr_lo)
        h3_pseudo = h3[pseudo_mask].copy()
        h3_pseudo[TARGET] = (base_pred[pseudo_mask] > 0.5).astype(int)
        n_pos = int(h3_pseudo[TARGET].sum())
        aug = pd.concat([base_tr, h3_pseudo], ignore_index=True)
        w = np.r_[np.ones(len(base_tr)), np.full(len(h3_pseudo), pw)]
        aug_pred = fit_predict(aug, h3, weights=w)
        aug_auc = roc_auc_score(y_h3, aug_pred)
        print(f"thr {thr_hi}/{thr_lo} w={pw}: +{len(h3_pseudo)} pseudo ({n_pos} pos) "
              f"-> H3 AUC {aug_auc:.6f}  delta {aug_auc-base_auc:+.6f}")


if __name__ == "__main__":
    main()
