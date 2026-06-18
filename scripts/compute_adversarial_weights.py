#!/usr/bin/env python
"""Compute label-free adversarial (train-vs-test) importance weights for train rows.

Density-ratio weighting: fit a domain classifier (train=0, test=1) on features only
(NO target, NO decision_day), take out-of-fold P(test) for each train row, and set
weight = p/(1-p) (clipped, mean-normalized). Rows that look more like the test
distribution get larger weight. Used to adapt training toward the future test regime.

Outputs front_id,sample_weight CSV plus a summary JSON for traceability.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_COL = "front_id"
TARGET = "target_value"
DAY_COL = "decision_day"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train_apps.csv")
    parser.add_argument("--test", default="data/raw/test_apps.csv")
    parser.add_argument("--out-dir", default="reports/validation")
    parser.add_argument("--clip", type=float, default=1e-3)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(REPO_ROOT / args.train)
    test = pd.read_csv(REPO_ROOT / args.test)

    drop = [ID_COL, TARGET, DAY_COL]
    feats = [c for c in test.columns if c not in drop]

    X = pd.concat([train[feats], test[feats]], axis=0, ignore_index=True)
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category").cat.codes
    y_dom = np.r_[np.zeros(len(train)), np.ones(len(test))]

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, random_state=args.seed
    )
    p = cross_val_predict(clf, X, y_dom, cv=3, method="predict_proba")[:, 1]
    adv_auc = float(roc_auc_score(y_dom, p))

    p_train = np.clip(p[: len(train)], args.clip, 1 - args.clip)
    w = p_train / (1 - p_train)
    w = w / w.mean()
    w = np.minimum(w, args.max_weight)
    w = w / w.mean()  # renormalize after capping

    ess = float((w.sum() ** 2) / (w ** 2).sum())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / args.out_dir
    weights_path = out_dir / f"adversarial_weights_{stamp}.csv"
    summary_path = out_dir / f"adversarial_weights_{stamp}_summary.json"

    pd.DataFrame({ID_COL: train[ID_COL], "sample_weight": w}).to_csv(weights_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adversarial_auc_oof": adv_auc,
        "features_used": feats,
        "feature_count": len(feats),
        "excluded": drop,
        "clip": args.clip,
        "max_weight": args.max_weight,
        "seed": args.seed,
        "train_rows": int(len(train)),
        "effective_sample_size": ess,
        "ess_fraction": ess / len(train),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_p99": float(np.percentile(w, 99)),
        "rows_weight_below_0p01": int((w < 0.01).sum()),
        "weights_path": str(weights_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Adversarial AUC (OOF, no decision_day): {adv_auc:.4f}")
    print(f"ESS: {ess:.0f} ({ess/len(train)*100:.1f}% of {len(train)})")
    print(f"Weights: {weights_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
