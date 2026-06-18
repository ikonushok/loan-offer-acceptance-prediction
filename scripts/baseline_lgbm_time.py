#!/usr/bin/env python
"""Time-aware LightGBM diversity baseline for the Alfa credit-offer task."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baseline_catboost_time import (
    DAY_COL,
    ID_COL,
    TARGET,
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


def prepare_lgbm_categories(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = train_x.copy()
    test_x = test_x.copy()
    for col in categorical_cols:
        categories = pd.Index(pd.concat([train_x[col], test_x[col]], ignore_index=True).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        train_x[col] = train_x[col].astype(str).astype(dtype)
        test_x[col] = test_x[col].astype(str).astype(dtype)
    return train_x, test_x


def get_model(model_cfg: dict[str, Any], seed: int) -> lgb.LGBMClassifier:
    params = dict(model_cfg)
    params.setdefault("objective", "binary")
    params.setdefault("metric", "auc")
    params.setdefault("random_state", seed)
    params.setdefault("verbosity", -1)
    params.setdefault("n_jobs", -1)
    return lgb.LGBMClassifier(**params)


def evaluate_time_folds(
    train_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
    early_stopping_rounds: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    log(f"Starting LightGBM time-holdout evaluation: folds={len(folds)}")

    for fold in folds:
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        if y_valid.nunique() < 2:
            raise ValueError(f"Fold {fold['fold']} validation target has only one class")

        log(
            "Fold {fold}: train {train_start}..{train_end} rows={train_rows}; "
            "valid {valid_start}..{valid_end} rows={valid_rows}, positive_rate={positive_rate:.6f}".format(
                fold=fold["fold"],
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                train_rows=len(train_idx),
                valid_start=fold["valid_start"],
                valid_end=fold["valid_end"],
                valid_rows=len(valid_idx),
                positive_rate=float(y_valid.mean()),
            )
        )

        model = get_model(model_cfg=model_cfg, seed=seed)
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        model.fit(
            train_x.iloc[train_idx],
            y_train,
            sample_weight=None if sample_weights is None else sample_weights.iloc[train_idx],
            eval_set=[(train_x.iloc[valid_idx], y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=callbacks,
        )

        pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y_valid, pred))
        best_iteration = int(model.best_iteration_ or model_cfg.get("n_estimators", 0))

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
    }
    return fold_metrics, valid_predictions


def train_full_and_predict_test(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    model_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
) -> pd.Series:
    log(f"Training final LightGBM on full train: rows={len(train_x)}, features={train_x.shape[1]}")
    model = get_model(model_cfg=model_cfg, seed=seed)
    model.fit(
        train_x,
        y,
        sample_weight=sample_weights,
        categorical_feature=categorical_cols,
    )

    log(f"Predicting test: rows={len(test_x)}")
    pred = np.clip(model.predict_proba(test_x)[:, 1], 0.0, 1.0)
    if not np.isfinite(pred).all():
        raise ValueError("Test predictions contain NaN or inf")
    log(
        "Test prediction range: min={:.6f}, mean={:.6f}, max={:.6f}".format(
            float(np.min(pred)),
            float(np.mean(pred)),
            float(np.max(pred)),
        )
    )
    return pd.Series(pred, name="prediction")


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
    model_cfg = cfg["model"]
    seed = int(cfg["seed"])

    log(f"Config: {config_path}")
    log(f"Train: {train_path}")
    log(f"Test: {test_path}")

    run_id = make_run_id(prefix=cfg["run_prefix"], seed=seed)
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    log(f"Run id: {run_id}")
    log(f"Output dir: {out_dir}")

    log("Reading CSV files")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    log(f"Loaded train shape={train.shape}, test shape={test.shape}")

    log("Preparing features")
    train_x, test_x, y, categorical_cols, feature_cols, feature_manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": True}),
        excluded_features=cfg.get("excluded_features", []),
    )
    train_x, test_x = prepare_lgbm_categories(train_x, test_x, categorical_cols)
    log(f"Prepared features: count={len(feature_cols)}, categorical={categorical_cols}")

    sample_weight_cfg = cfg.get("sample_weight", {"enabled": False})
    sample_weights = make_sample_weights(train, sample_weight_cfg)
    if sample_weights is not None:
        log(
            "Sample weights: strategy={strategy}, min={min_weight:.6f}, mean={mean_weight:.6f}, max={max_weight:.6f}".format(
                strategy=sample_weight_cfg.get("strategy"),
                min_weight=float(sample_weights.min()),
                mean_weight=float(sample_weights.mean()),
                max_weight=float(sample_weights.max()),
            )
        )

    folds = make_time_folds(train, cfg["cutoffs"])
    if not folds:
        raise ValueError("No valid time folds were created")

    fold_metrics, valid_predictions = evaluate_time_folds(
        train_x=train_x,
        y=y,
        categorical_cols=categorical_cols,
        folds=folds,
        model_cfg=model_cfg,
        seed=seed,
        sample_weights=sample_weights,
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 100)),
    )

    test_pred = train_full_and_predict_test(
        train_x=train_x,
        test_x=test_x,
        y=y,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
        seed=seed,
        sample_weights=sample_weights,
    )

    log("Writing artifacts")
    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_partial_time_holdout",
        "model": "LGBMClassifier",
        "seed": seed,
        "config_path": str(config_path),
        "params": model_cfg,
        "feature_engineering": cfg.get("feature_engineering", {"enabled": False}),
        "time_features": cfg.get("time_features", {"enabled": True}),
        "sample_weight": sample_weight_cfg,
        "sample_weight_stats": None
        if sample_weights is None
        else {
            "min": float(sample_weights.min()),
            "mean": float(sample_weights.mean()),
            "max": float(sample_weights.max()),
        },
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "target": TARGET,
        "positive_class": 1,
        "excluded_features": [ID_COL, DAY_COL] + cfg.get("excluded_features", []),
        "categorical_features": categorical_cols,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "validation_policy": {
            "type": "rolling_time_holdout",
            "cutoffs": cfg["cutoffs"],
            "reason": "same folds as CatBoost champion for diversity comparison",
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
            "LightGBM categorical handling differs from CatBoost; use primarily as diversity candidate.",
            "No platform upload authorized by this script.",
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
