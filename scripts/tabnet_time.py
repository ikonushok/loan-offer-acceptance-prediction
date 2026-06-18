#!/usr/bin/env python
"""TabNet on the same time folds + context-offer features as the GBM champion.

Purpose: a structurally different (attention-based) tabular model whose errors may
decorrelate from the GBM family (CatBoost/XGB/LGBM all corr >0.87). Standalone AUC
need not beat champion; value is blend diversity. Produces fold metrics, late
holdouts (H1/H2/H3), test predictions, and correlation vs champion OOF/test.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_catboost_time import DAY_COL, TARGET, ID_COL, prepare_features, make_time_folds

REPO = Path(__file__).resolve().parents[1]
CUTOFFS = ["2025-01-01", "2025-03-01", "2025-04-01"]
HOLDOUTS = [("H3_2025_05_01_to_end", "2025-05-01")]
H_END = pd.Timestamp("2025-06-05")
FE = {"enabled": True, "add_pairwise_features": False, "add_missing_flags": True, "add_context_offer_features": True}
SEED = 42


def encode(tx, ex, cats):
    """Ordinal-encode categoricals on train+eval union; fill NaN; standardize numerics."""
    tx, ex = tx.copy(), ex.copy()
    for c in cats:
        u = pd.concat([tx[c], ex[c]], axis=0).astype("category")
        mapping = {v: i for i, v in enumerate(u.cat.categories)}
        tx[c] = tx[c].map(mapping).fillna(-1).astype(int)
        ex[c] = ex[c].map(mapping).fillna(-1).astype(int)
    num = [c for c in tx.columns if c not in cats]
    for c in num:
        med = tx[c].median()
        tx[c] = tx[c].fillna(med); ex[c] = ex[c].fillna(med)
        mu, sd = tx[c].mean(), tx[c].std()
        sd = sd if sd and np.isfinite(sd) and sd > 0 else 1.0
        tx[c] = (tx[c] - mu) / sd; ex[c] = (ex[c] - mu) / sd
    return tx.to_numpy(np.float32), ex.to_numpy(np.float32)


def fit_predict(train_df, eval_df, seed=SEED):
    ev = eval_df.drop(columns=[TARGET]) if TARGET in eval_df.columns else eval_df
    tx, ex, y, cats, cols, _ = prepare_features(train_df, ev, FE, {"enabled": False})
    Xtr, Xev = encode(tx, ex, cats)
    torch.manual_seed(seed); np.random.seed(seed)
    clf = TabNetClassifier(
        n_d=16, n_a=16, n_steps=4, gamma=1.5, lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-2), seed=seed, verbose=0,
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )
    yv = y.to_numpy()
    clf.fit(Xtr, yv, max_epochs=120, patience=25, batch_size=4096, virtual_batch_size=512,
            eval_set=[(Xtr, yv)], eval_metric=["auc"], weights=1)
    return clf.predict_proba(Xev)[:, 1]


def main():
    train = pd.read_csv(REPO / "data/raw/train_apps.csv")
    days = pd.to_datetime(train[DAY_COL])
    test = pd.read_csv(REPO / "data/raw/test_apps.csv")

    folds = make_time_folds(train, CUTOFFS)
    oof = np.full(len(train), np.nan)
    rows = []
    for f in folds:
        tr = train.iloc[f["train_idx"]].reset_index(drop=True)
        va = train.iloc[f["valid_idx"]].reset_index(drop=True)
        p = fit_predict(tr, va)
        auc = roc_auc_score(va[TARGET], p)
        oof[f["valid_idx"]] = p
        rows.append({"fold": f["fold"], "valid_rows": len(va), "roc_auc": auc})
        print(f"Fold {f['fold']}: rows={len(va)} AUC={auc:.6f}")
    mask = ~np.isnan(oof)
    oof_auc = roc_auc_score(train[TARGET].to_numpy()[mask], oof[mask])
    print(f"OOF AUC: {oof_auc:.6f}")

    # Late holdouts
    print("\nLate holdouts:")
    lh = {}
    for name, cut in HOLDOUTS:
        cutoff = pd.Timestamp(cut)
        tr = train[days < cutoff].reset_index(drop=True)
        ho = train[(days >= cutoff) & (days <= H_END)].reset_index(drop=True)
        p = fit_predict(tr, ho)
        auc = roc_auc_score(ho[TARGET], p)
        lh[name] = auc
        print(f"  {name}: AUC={auc:.6f}")
    lh_mean = float(np.mean(list(lh.values())))
    lh_min = float(np.min(list(lh.values())))
    print(f"  LATE_HOLDOUT_MEAN: {lh_mean:.6f} | MIN: {lh_min:.6f}")

    # Full model -> test predictions
    print("\nTraining full model for test predictions...")
    test_pred = fit_predict(train, test)

    # Correlation vs champion
    champ_oof = pd.read_csv(REPO / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z/valid_predictions_time.csv")
    champ_test = pd.read_csv(REPO / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z/test_predictions.csv")
    co = champ_oof.set_index("row_index")["prediction"]
    common = [i for i in np.where(mask)[0] if i in co.index]
    corr_oof = float(np.corrcoef(oof[common], co.loc[common])[0, 1])
    corr_test = float(np.corrcoef(test_pred, champ_test["prediction"])[0, 1])
    print(f"\nCorrelation vs champion: OOF={corr_oof:.4f} | test={corr_test:.4f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "experiments/runs" / f"tabnet_context_offer_v1_{stamp}_seed{SEED}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(out / "test_predictions.csv", index=False)
    oof_df = train[[ID_COL]].copy(); oof_df["row_index"] = np.arange(len(train)); oof_df["prediction"] = oof
    oof_df[mask].to_csv(out / "valid_predictions_time.csv", index=False)
    summary = {"fold_metrics": rows, "oof_auc": oof_auc, "late_holdouts": lh,
               "lh_mean": lh_mean, "lh_min": lh_min,
               "corr_vs_champion_oof": corr_oof, "corr_vs_champion_test": corr_test,
               "model": "TabNet", "seed": SEED}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {out}")


if __name__ == "__main__":
    main()
