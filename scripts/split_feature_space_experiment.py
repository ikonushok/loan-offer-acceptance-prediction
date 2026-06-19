#!/usr/bin/env python
"""Два независимых представления: absolute vs context-relative feature sets.

Обе модели — CatBoost pointwise (те же гиперпараметры и фолды, что у champion),
но на НЕПЕРЕСЕКАЮЩИХСЯ наборах фичей. Цель: декорреляция без жертвы силой.

Набор A (absolute): raw числовые + категории — профиль клиента и оффера.
Набор B (context-relative): context-offer фичи + offer_cols (нужны для ранга).

Выводит: per-fold AUC каждой модели, OOF AUC, spearman/pearson корреляцию,
и скан бленда (rank-normalized).

Usage:
    PY=/opt/homebrew/Caskroom/miniforge/base/bin/python3
    PYTHONPATH=scripts $PY scripts/split_feature_space_experiment.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from baseline_catboost_time import (
    DAY_COL,
    ID_COL,
    TARGET,
    log,
    make_time_folds,
    prepare_features,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# champion trial0070 гиперпараметры
CATBOOST_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "min_data_in_leaf": 100,
    "random_seed": 42,
    "task_type": "CPU",
    "eval_metric": "AUC",
    "verbose": 200,
    "early_stopping_rounds": 100,
    "use_best_model": True,
}

CONTEXT_FEATURE_NAMES = [
    "context_offer_count",
    "offered_rate_context_spread",
    "offered_rate_minus_context_min",
    "offered_rate_minus_context_mean",
    "offered_rate_rank_pct_in_context",
    "overdraft_limit_min_context_spread",
    "overdraft_limit_min_minus_context_min",
    "overdraft_limit_min_minus_context_mean",
    "overdraft_limit_min_rank_pct_in_context",
    "overdraft_limit_max_context_spread",
    "overdraft_limit_max_minus_context_min",
    "overdraft_limit_max_minus_context_mean",
    "overdraft_limit_max_rank_pct_in_context",
]

OFFER_COLS = ["offered_rate", "overdraft_limit_min", "overdraft_limit_max"]


def rank_normalize(a: np.ndarray) -> np.ndarray:
    o = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[o] = np.arange(1, len(a) + 1)
    return r / len(a)


def train_evaluate(
    name: str,
    train_x: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    cat_cols: list[str],
    folds: list,
) -> pd.DataFrame:
    """Обучение CatBoost на выбранных фичах, возвращает OOF predictions."""
    preds = []
    for fold in folds:
        tr, va = fold["train_idx"], fold["valid_idx"]
        tr_pool = Pool(train_x[feature_cols].iloc[tr], y.iloc[tr], cat_features=cat_cols)
        va_pool = Pool(train_x[feature_cols].iloc[va], y.iloc[va], cat_features=cat_cols)
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(tr_pool, eval_set=va_pool)
        pred = model.predict_proba(va_pool)[:, 1]
        auc = roc_auc_score(y.iloc[va], pred)
        log(f"  {name} Fold {fold['fold']}: AUC={auc:.6f} features={len(feature_cols)}")
        preds.append(pd.DataFrame({
            "row_index": va,
            TARGET: y.iloc[va].to_numpy(),
            "prediction": pred,
            "fold": fold["fold"],
        }))
    oof = pd.concat(preds, ignore_index=True).sort_values("row_index")
    oof_auc = roc_auc_score(oof[TARGET], oof["prediction"])
    log(f"  {name} OOF AUC={oof_auc:.6f}")
    return oof


def main() -> None:
    train = pd.read_csv(REPO_ROOT / "data/raw/train_apps.csv")
    test = pd.read_csv(REPO_ROOT / "data/raw/test_apps.csv")
    log(f"Загружены данные: train={train.shape}, test={test.shape}")

    fe_cfg = {
        "enabled": True,
        "add_pairwise_features": False,
        "add_missing_flags": False,
        "add_context_offer_features": True,
    }
    train_x, test_x, y, cat_cols, all_features, _ = prepare_features(
        train=train, test=test,
        feature_engineering_cfg=fe_cfg,
        time_features_cfg={"enabled": False},
        excluded_features=[],
    )

    # Разделяем фичи
    context_set = set(CONTEXT_FEATURE_NAMES)
    offer_set = set(OFFER_COLS)

    features_B = [f for f in all_features if f in context_set or f in offer_set]
    features_A = [f for f in all_features if f not in context_set]
    cat_A = [c for c in cat_cols if c in features_A]
    cat_B = [c for c in cat_cols if c in features_B]

    log(f"Набор A (absolute): {len(features_A)} фичей, cat={cat_A}")
    log(f"Набор B (context-relative + offer): {len(features_B)} фичей, cat={cat_B}")
    log(f"Пересечение: {set(features_A) & set(features_B)}")

    cutoffs = ["2025-01-01", "2025-03-01", "2025-04-01"]
    folds = make_time_folds(train, cutoffs)

    log("=== Модель A (absolute) ===")
    oof_A = train_evaluate("A", train_x, y, features_A, cat_A, folds)

    log("=== Модель B (context-relative) ===")
    oof_B = train_evaluate("B", train_x, y, features_B, cat_B, folds)

    # Загружаем champion OOF
    champ_oof = pd.read_csv(
        REPO_ROOT / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z/valid_predictions_time.csv"
    )

    m = oof_A.merge(oof_B, on="row_index", suffixes=("_A", "_B"))
    m = m.merge(champ_oof[["row_index", "prediction"]].rename(columns={"prediction": "pred_champ"}), on="row_index")
    assert (m["target_value_A"] == m["target_value_B"]).all()
    yt = m["target_value_A"].to_numpy()

    sp_AB = spearmanr(m["prediction_A"], m["prediction_B"]).correlation
    sp_A_ch = spearmanr(m["prediction_A"], m["pred_champ"]).correlation
    sp_B_ch = spearmanr(m["prediction_B"], m["pred_champ"]).correlation

    log(f"\nКорреляция (Spearman):")
    log(f"  A vs B: {sp_AB:.4f}")
    log(f"  A vs champion: {sp_A_ch:.4f}")
    log(f"  B vs champion: {sp_B_ch:.4f}")

    rA = rank_normalize(m["prediction_A"].to_numpy())
    rB = rank_normalize(m["prediction_B"].to_numpy())
    rC = rank_normalize(m["pred_champ"].to_numpy())

    log("\nСкан бленда A+B (rank OOF):")
    for wB in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        bl = (1 - wB) * rA + wB * rB
        log(f"  wB={wB:.1f}  AUC={roc_auc_score(yt, bl):.6f}")

    log("\nСкан бленда champion+A (rank OOF):")
    for wA in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        bl = (1 - wA) * rC + wA * rA
        log(f"  wA={wA:.1f}  AUC={roc_auc_score(yt, bl):.6f}")

    log("\nСкан бленда champion+B (rank OOF):")
    for wB in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        bl = (1 - wB) * rC + wB * rB
        log(f"  wB={wB:.1f}  AUC={roc_auc_score(yt, bl):.6f}")

    log("\nСкан тройного бленда champion+A+B (rank OOF, wA=wB):")
    for w in [0.0, 0.05, 0.10, 0.15, 0.20]:
        bl = (1 - 2 * w) * rC + w * rA + w * rB
        log(f"  wA=wB={w:.2f}  AUC={roc_auc_score(yt, bl):.6f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "experiment": "split_feature_space_v1",
        "features_A": features_A,
        "features_B": features_B,
        "oof_auc_A": float(roc_auc_score(oof_A[TARGET], oof_A["prediction"])),
        "oof_auc_B": float(roc_auc_score(oof_B[TARGET], oof_B["prediction"])),
        "spearman_A_vs_B": float(sp_AB),
        "spearman_A_vs_champion": float(sp_A_ch),
        "spearman_B_vs_champion": float(sp_B_ch),
        "verdict": "TBD",
    }
    out = REPO_ROOT / "reports" / "validation" / f"split_feature_space_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log(f"\nАртефакт: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
