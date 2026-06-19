#!/usr/bin/env python
"""Second representation: XGBoost learning-to-rank (rank:pairwise) on decision-day groups.

Same data, same context-offer feature set, same time folds as the champion — but a
fundamentally different OBJECTIVE: instead of pointwise P(accept) per row, the model
learns to RANK offers competing on the same decision_day. Goal is a low rank-correlation
with the pointwise champion so the rank-blend gains genuine diversity.

Grouping: decision_day (not context group — context groups are 94% singletons, which
would zero out almost all pairs). Day groups use all data and define a real ranking task.

Outputs OOF + test predictions; corr-vs-champion gate is run separately.

Usage:
    PY=/opt/homebrew/Caskroom/miniforge/base/bin/python3
    $PY scripts/xgb_rank_objective_time.py --config configs/diversity_experiments/xgb_rank_objective_day_v1.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBRanker

from baseline_catboost_time import (
    DAY_COL,
    ID_COL,
    TARGET,
    load_config,
    log,
    make_time_folds,
    prepare_features,
    resolve_path,
    sha256_file,
)
from baseline_xgb_time import prepare_xgb_categories


def make_run_id(prefix: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_seed{seed}"


def get_ranker(model_cfg: dict[str, Any], seed: int, use_early_stopping: bool) -> XGBRanker:
    params = dict(model_cfg)
    early_stopping_rounds = params.pop("early_stopping_rounds", None)
    params.setdefault("objective", "rank:pairwise")
    params.setdefault("eval_metric", "auc")
    params.setdefault("random_state", seed)
    params.setdefault("tree_method", "hist")
    params.setdefault("enable_categorical", True)
    params.setdefault("n_jobs", -1)
    if use_early_stopping and early_stopping_rounds is not None:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    return XGBRanker(**params)


def group_sizes(qid_sorted: np.ndarray) -> np.ndarray:
    """Contiguous group sizes for a sorted qid array."""
    _, counts = np.unique(qid_sorted, return_counts=True)
    # np.unique returns sorted unique; since qid_sorted is sorted, counts align contiguously
    return counts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = resolve_path(args.config, repo_root)
    cfg = load_config(config_path)
    train_path = resolve_path(cfg["train"], repo_root)
    test_path = resolve_path(cfg["test"], repo_root)
    out_base = resolve_path(cfg["out_dir"], repo_root)
    seed = int(cfg["seed"])
    model_cfg = cfg["model"]

    run_id = make_run_id(prefix=cfg["run_prefix"], seed=seed)
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log(f"Run id: {run_id}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    log(f"Loaded train shape={train.shape}, test shape={test.shape}")

    train_x, test_x, y, categorical_cols, feature_cols, feature_manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": True}),
        excluded_features=cfg.get("excluded_features", []),
    )
    train_x, test_x = prepare_xgb_categories(train_x, test_x, categorical_cols)
    log(f"Prepared features: count={len(feature_cols)}, categorical={categorical_cols}")

    # decision_day -> integer qid, aligned positionally to train_x rows
    day_code = pd.factorize(train[DAY_COL].to_numpy(), sort=True)[0].astype(np.int64)

    folds = make_time_folds(train, cfg["cutoffs"])
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    log(f"Starting XGBRanker time-holdout: folds={len(folds)}, objective={model_cfg.get('objective','rank:pairwise')}")

    for fold in folds:
        tr = np.asarray(fold["train_idx"])
        va = np.asarray(fold["valid_idx"])
        if y.iloc[va].nunique() < 2:
            raise ValueError(f"Fold {fold['fold']} validation target has only one class")

        # sort train by qid (groups must be contiguous)
        otr = np.argsort(day_code[tr], kind="stable")
        tr_s = tr[otr]
        qtr = day_code[tr_s]
        Xtr = train_x.iloc[tr_s]
        ytr = y.iloc[tr_s]

        ova = np.argsort(day_code[va], kind="stable")
        va_s = va[ova]
        qva = day_code[va_s]
        Xva = train_x.iloc[va_s]
        yva = y.iloc[va_s]

        log(f"Fold {fold['fold']}: train_rows={len(tr_s)} groups={len(np.unique(qtr))}; "
            f"valid_rows={len(va_s)} groups={len(np.unique(qva))} pos_rate={float(yva.mean()):.6f}")

        model = get_ranker(model_cfg=model_cfg, seed=seed, use_early_stopping=True)
        model.fit(
            Xtr, ytr,
            qid=qtr,
            eval_set=[(Xva, yva)],
            eval_qid=[qva],
            verbose=100,
        )
        pred = model.predict(Xva)
        auc = float(roc_auc_score(yva.to_numpy(), pred))
        best_iteration = int(getattr(model, "best_iteration", 0) or 0)
        log(f"Fold {fold['fold']} done: roc_auc={auc:.6f} best_iter={best_iteration}")

        fold_rows.append({
            "fold": fold["fold"], "cutoff": fold["cutoff"],
            "valid_rows": len(va_s), "valid_positive_rate": float(yva.mean()),
            "best_iteration": best_iteration, "roc_auc": auc,
        })
        pred_parts.append(pd.DataFrame({
            "row_index": va_s, TARGET: yva.to_numpy(),
            "prediction": pred, "fold": fold["fold"],
        }))

    fold_metrics = pd.DataFrame(fold_rows)
    valid_predictions = pd.concat(pred_parts, ignore_index=True).sort_values("row_index")
    oof_auc = float(roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"]))
    log(f"OOF time-holdout: rows={len(valid_predictions)} roc_auc={oof_auc:.6f}")
    fold_metrics.loc[len(fold_metrics)] = {
        "fold": "OOF_TIME_HOLDOUT", "cutoff": "",
        "valid_rows": len(valid_predictions),
        "valid_positive_rate": float(valid_predictions[TARGET].mean()),
        "best_iteration": np.nan, "roc_auc": oof_auc,
    }

    # final model on full train, predict test
    ofull = np.argsort(day_code, kind="stable")
    qfull = day_code[ofull]
    log(f"Training final XGBRanker on full train: rows={len(train_x)} groups={len(np.unique(qfull))}")
    final = get_ranker(model_cfg=model_cfg, seed=seed, use_early_stopping=False)
    final.fit(train_x.iloc[ofull], y.iloc[ofull], qid=qfull, verbose=False)
    test_score = final.predict(test_x)
    log(f"Test score range: min={float(test_score.min()):.4f} mean={float(test_score.mean()):.4f} max={float(test_score.max()):.4f}")

    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_score}).to_csv(out_dir / "test_predictions.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_partial_time_holdout",
        "model": "XGBRanker",
        "objective": model_cfg.get("objective", "rank:pairwise"),
        "grouping": "decision_day",
        "seed": seed,
        "params": model_cfg,
        "feature_engineering": cfg.get("feature_engineering", {}),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "feature_count": len(feature_cols),
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "note": "Second representation (ranking objective) for diversity blend. Scores are unbounded ranks, not probabilities — rank-normalize before blending.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Fold metrics ==")
    print(fold_metrics.to_string(index=False))
    print(f"\nRun dir: {out_dir}")


if __name__ == "__main__":
    main()
