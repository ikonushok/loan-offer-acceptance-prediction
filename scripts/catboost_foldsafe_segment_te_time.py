#!/usr/bin/env python3
"""CatBoost experiment with fold-safe segment statistics and target encoding.

The derived features in this script are fit inside each rolling fold:
- segment statistics are computed on the fold training part only;
- target encoding uses inner OOF estimates for fold training rows and maps
  validation rows from the fold training part only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from baseline_catboost_time import (
    DAY_COL,
    ID_COL,
    TARGET,
    get_model,
    load_config,
    log,
    make_sample_weights,
    make_time_folds,
    prepare_features,
    resolve_path,
    sha256_file,
)


def make_run_id(prefix: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_seed{seed}"


def safe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-12) -> pd.Series:
    num_values = num.astype(float)
    den_values = den.astype(float)
    out = np.where(np.abs(den_values) > eps, num_values / den_values, np.nan)
    return pd.Series(out, index=num.index)


def resolve_feature_columns(columns: pd.Index, requested: list[str]) -> list[str]:
    return [col for col in requested if col in columns]


def fit_segment_stats(
    fit_x: pd.DataFrame,
    transform_x: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    if not cfg.get("enabled", False):
        return transform_x.copy(), []

    transformed = transform_x.copy()
    cat_cols = resolve_feature_columns(fit_x.columns, cfg.get("categorical_cols", []))
    num_cols = resolve_feature_columns(fit_x.columns, cfg.get("numeric_cols", []))
    min_count = int(cfg.get("min_count", 20))
    add_diff = bool(cfg.get("add_diff", True))
    add_ratio = bool(cfg.get("add_ratio", True))
    add_zscore = bool(cfg.get("add_zscore", True))

    new_cols: list[str] = []
    for cat in cat_cols:
        for num in num_cols:
            stats = fit_x.groupby(cat, observed=True)[num].agg(["mean", "std", "count"])
            stats = stats.rename(
                columns={
                    "mean": f"{cat}__{num}__seg_mean",
                    "std": f"{cat}__{num}__seg_std",
                    "count": f"{cat}__{num}__seg_count",
                }
            )
            stats.loc[stats[f"{cat}__{num}__seg_count"] < min_count, :] = np.nan

            mapped = transform_x[[cat]].merge(stats, left_on=cat, right_index=True, how="left")
            mean_col = f"{cat}__{num}__seg_mean"
            std_col = f"{cat}__{num}__seg_std"
            count_col = f"{cat}__{num}__seg_count"
            for col in [mean_col, std_col, count_col]:
                transformed[col] = mapped[col].to_numpy()
                new_cols.append(col)

            if add_diff:
                col = f"{cat}__{num}__minus_seg_mean"
                transformed[col] = transform_x[num] - transformed[mean_col]
                new_cols.append(col)
            if add_ratio:
                col = f"{cat}__{num}__to_seg_mean"
                transformed[col] = safe_divide(transform_x[num], transformed[mean_col])
                new_cols.append(col)
            if add_zscore:
                col = f"{cat}__{num}__seg_zscore"
                transformed[col] = safe_divide(transform_x[num] - transformed[mean_col], transformed[std_col])
                new_cols.append(col)

    transformed[new_cols] = transformed[new_cols].replace([np.inf, -np.inf], np.nan)
    return transformed, new_cols


def smoothed_target_map(
    categories: pd.Series,
    target: pd.Series,
    smoothing: float,
    prior: float,
) -> dict[Any, float]:
    agg = pd.DataFrame({"category": categories, "target": target}).groupby("category", observed=True)["target"].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + prior * smoothing) / (agg["count"] + smoothing)
    return smoothed.to_dict()


def add_target_encoding(
    fit_x: pd.DataFrame,
    fit_y: pd.Series,
    transform_x: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    is_fit_frame: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if not cfg.get("enabled", False):
        return transform_x.copy(), []

    transformed = transform_x.copy()
    cat_cols = resolve_feature_columns(fit_x.columns, cfg.get("categorical_cols", []))
    smoothing = float(cfg.get("smoothing", 20.0))
    n_splits = int(cfg.get("inner_splits", 5))
    prior = float(fit_y.mean())
    new_cols: list[str] = []

    for cat in cat_cols:
        out_col = f"{cat}__foldsafe_te"
        if is_fit_frame:
            encoded = pd.Series(prior, index=fit_x.index, dtype=float)
            min_class = int(fit_y.value_counts().min())
            actual_splits = max(2, min(n_splits, min_class))
            splitter = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
            for inner_train_idx, inner_valid_idx in splitter.split(fit_x, fit_y):
                inner_train = fit_x.iloc[inner_train_idx]
                inner_y = fit_y.iloc[inner_train_idx]
                fmap = smoothed_target_map(inner_train[cat], inner_y, smoothing=smoothing, prior=prior)
                encoded.iloc[inner_valid_idx] = fit_x.iloc[inner_valid_idx][cat].map(fmap).fillna(prior).to_numpy()
            transformed[out_col] = encoded.reindex(transformed.index).to_numpy()
        else:
            fmap = smoothed_target_map(fit_x[cat], fit_y, smoothing=smoothing, prior=prior)
            transformed[out_col] = transform_x[cat].map(fmap).fillna(prior).to_numpy()
        new_cols.append(out_col)

    return transformed, new_cols


def add_foldsafe_features(
    fit_x: pd.DataFrame,
    fit_y: pd.Series,
    transform_x: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    is_fit_frame: bool,
) -> tuple[pd.DataFrame, list[str]]:
    transformed, segment_cols = fit_segment_stats(
        fit_x=fit_x,
        transform_x=transform_x,
        cfg=cfg.get("segment_stats", {}),
    )
    transformed, te_cols = add_target_encoding(
        fit_x=fit_x,
        fit_y=fit_y,
        transform_x=transformed,
        cfg=cfg.get("target_encoding", {}),
        seed=seed,
        is_fit_frame=is_fit_frame,
    )
    return transformed, segment_cols + te_cols


def evaluate_time_folds(
    train_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    foldsafe_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    all_foldsafe_cols: set[str] = set()

    log(f"Starting fold-safe CatBoost time-holdout evaluation: folds={len(folds)}")

    for fold in folds:
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        if y_valid.nunique() < 2:
            raise ValueError(f"Fold {fold['fold']} validation target has only one class")

        fold_seed = seed + int(fold["fold"])
        fold_train_x, train_added = add_foldsafe_features(
            fit_x=train_x.iloc[train_idx],
            fit_y=y_train,
            transform_x=train_x.iloc[train_idx],
            cfg=foldsafe_cfg,
            seed=fold_seed,
            is_fit_frame=True,
        )
        fold_valid_x, valid_added = add_foldsafe_features(
            fit_x=train_x.iloc[train_idx],
            fit_y=y_train,
            transform_x=train_x.iloc[valid_idx],
            cfg=foldsafe_cfg,
            seed=fold_seed,
            is_fit_frame=False,
        )
        all_foldsafe_cols.update(train_added)
        all_foldsafe_cols.update(valid_added)
        if list(fold_train_x.columns) != list(fold_valid_x.columns):
            raise ValueError(f"Fold {fold['fold']} train/valid columns are not aligned after fold-safe features")

        cat_features = [fold_train_x.columns.get_loc(c) for c in categorical_cols if c in fold_train_x.columns]

        log(
            "Fold {fold}: train {train_start}..{train_end} rows={train_rows}; "
            "valid {valid_start}..{valid_end} rows={valid_rows}, positive_rate={positive_rate:.6f}, features={features}".format(
                fold=fold["fold"],
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                train_rows=len(train_idx),
                valid_start=fold["valid_start"],
                valid_end=fold["valid_end"],
                valid_rows=len(valid_idx),
                positive_rate=float(y_valid.mean()),
                features=fold_train_x.shape[1],
            )
        )

        train_pool = Pool(
            fold_train_x,
            label=y_train,
            weight=None if sample_weights is None else sample_weights.iloc[train_idx],
            cat_features=cat_features,
        )
        valid_pool = Pool(fold_valid_x, label=y_valid, cat_features=cat_features)
        model = get_model(model_cfg=model_cfg, seed=seed)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        pred = model.predict_proba(valid_pool)[:, 1]
        auc = float(roc_auc_score(y_valid, pred))
        best_iteration = int(model.get_best_iteration() or model_cfg["iterations"])
        log(f"Fold {fold['fold']} done: roc_auc={auc:.6f}, best_iteration={best_iteration}")

        fold_rows.append(
            {
                "fold": fold["fold"],
                "cutoff": fold["cutoff"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "valid_positive_rate": float(y_valid.mean()),
                "best_iteration": best_iteration,
                "roc_auc": auc,
                "feature_count": fold_train_x.shape[1],
                "foldsafe_feature_count": len(train_added),
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_index": valid_idx,
                    TARGET: y_valid.to_numpy(),
                    "prediction": pred,
                    "fold": fold["fold"],
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    valid_predictions = pd.concat(pred_parts, ignore_index=True).sort_values("row_index")
    oof_auc = float(roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"]))
    log(f"OOF time-holdout done: rows={len(valid_predictions)}, roc_auc={oof_auc:.6f}")
    fold_metrics.loc[len(fold_metrics)] = {
        "fold": "OOF_TIME_HOLDOUT",
        "cutoff": "",
        "train_start": "",
        "train_end": "",
        "valid_start": "",
        "valid_end": "",
        "train_rows": np.nan,
        "valid_rows": len(valid_predictions),
        "valid_positive_rate": float(valid_predictions[TARGET].mean()),
        "best_iteration": np.nan,
        "roc_auc": oof_auc,
        "feature_count": np.nan,
        "foldsafe_feature_count": len(all_foldsafe_cols),
    }
    return fold_metrics, valid_predictions, sorted(all_foldsafe_cols)


def train_full_and_predict_test(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    model_cfg: dict[str, Any],
    foldsafe_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
) -> tuple[pd.Series, list[str]]:
    full_train_x, train_added = add_foldsafe_features(
        fit_x=train_x,
        fit_y=y,
        transform_x=train_x,
        cfg=foldsafe_cfg,
        seed=seed,
        is_fit_frame=True,
    )
    full_test_x, test_added = add_foldsafe_features(
        fit_x=train_x,
        fit_y=y,
        transform_x=test_x,
        cfg=foldsafe_cfg,
        seed=seed,
        is_fit_frame=False,
    )
    if list(full_train_x.columns) != list(full_test_x.columns):
        raise ValueError("Full train/test columns are not aligned after fold-safe features")

    cat_features = [full_train_x.columns.get_loc(c) for c in categorical_cols if c in full_train_x.columns]
    train_pool = Pool(full_train_x, label=y, weight=sample_weights, cat_features=cat_features)
    test_pool = Pool(full_test_x, cat_features=cat_features)

    log(f"Training final fold-safe model on full train: rows={len(full_train_x)}, features={full_train_x.shape[1]}")
    model = get_model(model_cfg=model_cfg, seed=seed)
    model.fit(train_pool)

    log(f"Predicting test: rows={len(full_test_x)}")
    pred = np.clip(model.predict_proba(test_pool)[:, 1], 0.0, 1.0)
    if not np.isfinite(pred).all():
        raise ValueError("Test predictions contain NaN or inf")
    log(
        "Test prediction range: min={:.6f}, mean={:.6f}, max={:.6f}".format(
            float(np.min(pred)),
            float(np.mean(pred)),
            float(np.max(pred)),
        )
    )
    return pd.Series(pred, name="prediction"), sorted(set(train_added) | set(test_added))


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
    foldsafe_cfg = cfg.get("fold_safe_features", {})

    log(f"Config: {config_path}")
    log(f"Train: {train_path}")
    log(f"Test: {test_path}")

    run_id = make_run_id(prefix=cfg["run_prefix"], seed=seed)
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log(f"Run id: {run_id}")
    log(f"Output dir: {out_dir}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    log(f"Loaded train shape={train.shape}, test shape={test.shape}")

    feature_engineering_cfg = cfg.get("feature_engineering", {"enabled": False})
    time_features_cfg = cfg.get("time_features", {"enabled": True})
    excluded_features = cfg.get("excluded_features", [])
    train_x, test_x, y, categorical_cols, feature_cols, feature_manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=feature_engineering_cfg,
        time_features_cfg=time_features_cfg,
        excluded_features=excluded_features,
    )
    log(f"Prepared base features: count={len(feature_cols)}, categorical={categorical_cols}")

    sample_weight_cfg = cfg.get("sample_weight", {"enabled": False})
    sample_weights = make_sample_weights(train, sample_weight_cfg)

    folds = make_time_folds(train, cfg["cutoffs"])
    if not folds:
        raise ValueError("No valid time folds were created")
    fold_metrics, valid_predictions, cv_foldsafe_cols = evaluate_time_folds(
        train_x=train_x,
        y=y,
        categorical_cols=categorical_cols,
        folds=folds,
        model_cfg=model_cfg,
        foldsafe_cfg=foldsafe_cfg,
        seed=seed,
        sample_weights=sample_weights,
    )
    test_pred, full_foldsafe_cols = train_full_and_predict_test(
        train_x=train_x,
        test_x=test_x,
        y=y,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
        foldsafe_cfg=foldsafe_cfg,
        seed=seed,
        sample_weights=sample_weights,
    )

    foldsafe_cols = sorted(set(cv_foldsafe_cols) | set(full_foldsafe_cols))
    foldsafe_manifest = pd.DataFrame(
        {
            "column": foldsafe_cols,
            "role": "fold_safe_derived_feature",
            "reason": "fit on rolling-fold train part only; validation/test transformed from training statistics",
        }
    )
    feature_manifest = pd.concat([feature_manifest, foldsafe_manifest], ignore_index=True)

    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_partial_time_holdout",
        "model": "CatBoostClassifier",
        "seed": seed,
        "config_path": str(config_path),
        "params": model_cfg,
        "feature_engineering": feature_engineering_cfg,
        "fold_safe_features": foldsafe_cfg,
        "time_features": time_features_cfg,
        "sample_weight": sample_weight_cfg,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "target": TARGET,
        "positive_class": 1,
        "excluded_features": [ID_COL, DAY_COL] + excluded_features,
        "categorical_features": categorical_cols,
        "base_feature_count": len(feature_cols),
        "foldsafe_feature_count": len(foldsafe_cols),
        "foldsafe_feature_columns": foldsafe_cols,
        "validation_policy": {
            "type": "rolling_time_holdout",
            "cutoffs": cfg["cutoffs"],
            "reason": "test period is future-dated relative to train; segment stats and target encodings are fit inside folds",
        },
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "artifacts": {
            "config_used": str(out_dir / "config_used.json"),
            "fold_metrics": str(out_dir / "fold_metrics.csv"),
            "valid_predictions": str(out_dir / "valid_predictions_time.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "feature_manifest": str(out_dir / "feature_manifest.csv"),
            "summary": str(out_dir / "run_summary.json"),
        },
        "risks": [
            "OOF target encoding uses inner random StratifiedKFold inside each rolling train part; validation remains time-safe.",
            "Segment statistics are learned from fold train only and may be sparse for rare categories.",
            "No late-holdout battery is run by this script.",
        ],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Fold metrics ==")
    print(fold_metrics.to_string(index=False))
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
