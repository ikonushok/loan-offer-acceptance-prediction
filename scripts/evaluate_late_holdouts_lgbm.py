#!/usr/bin/env python
"""Evaluate LightGBM configs on overlapping late-period holdouts.

Mirrors evaluate_late_holdouts_xgb.py but for LightGBM.
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

from baseline_catboost_time import (
    DAY_COL,
    TARGET,
    load_config,
    log,
    make_sample_weights,
    prepare_features,
    resolve_path,
    sha256_file,
)
from baseline_lgbm_time import get_model, prepare_lgbm_categories


DEFAULT_HOLDOUTS: list[dict[str, str]] = [
    {"name": "H1_2025_03_01_to_end", "cutoff": "2025-03-01"},
    {"name": "H2_2025_04_01_to_end", "cutoff": "2025-04-01"},
    {"name": "H3_2025_05_01_to_end", "cutoff": "2025-05-01"},
]


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        if np.isnan(item) or np.isinf(item):
            return None
        return item
    return value


def make_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def evaluate_late_holdouts(
    train: pd.DataFrame,
    train_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    model_cfg: dict[str, Any],
    seed: int,
    holdouts: list[dict[str, str]],
    sample_weights: pd.Series | None = None,
    early_stopping_rounds: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    metric_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    for holdout in holdouts:
        name = holdout["name"]
        cutoff = pd.Timestamp(holdout["cutoff"])
        valid_end = pd.Timestamp(holdout["valid_end"]) if holdout.get("valid_end") else days.max()
        train_idx = np.where(days < cutoff)[0]
        valid_mask = (days >= cutoff) & (days <= valid_end)
        valid_idx = np.where(valid_mask)[0]

        if len(train_idx) == 0:
            raise ValueError(f"{name}: empty train split")
        if len(valid_idx) == 0:
            raise ValueError(f"{name}: empty validation split")

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        if y_valid.nunique() < 2:
            raise ValueError(f"{name}: validation target has only one class")

        log(
            "{name}: train {train_start}..{train_end} rows={train_rows}; "
            "valid {valid_start}..{valid_end} rows={valid_rows}, positive_rate={positive_rate:.6f}".format(
                name=name,
                train_start=days.iloc[train_idx].min().date(),
                train_end=days.iloc[train_idx].max().date(),
                train_rows=len(train_idx),
                valid_start=days.iloc[valid_idx].min().date(),
                valid_end=days.iloc[valid_idx].max().date(),
                valid_rows=len(valid_idx),
                positive_rate=float(y_valid.mean()),
            )
        )

        import lightgbm as lgb

        model = get_model(model_cfg=model_cfg, seed=seed)
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=200),
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

        log(f"{name} done: roc_auc={auc:.6f}, best_iteration={best_iteration}")

        metric_rows.append(
            {
                "holdout": name,
                "cutoff": str(cutoff.date()),
                "valid_end_requested": holdout.get("valid_end", ""),
                "train_start": str(days.iloc[train_idx].min().date()),
                "train_end": str(days.iloc[train_idx].max().date()),
                "valid_start": str(days.iloc[valid_idx].min().date()),
                "valid_end": str(days.iloc[valid_idx].max().date()),
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                "valid_positive_rate": float(y_valid.mean()),
                "best_iteration": best_iteration,
                "roc_auc": auc,
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "holdout": name,
                    "row_index": valid_idx,
                    TARGET: y_valid.to_numpy(),
                    "prediction": pred,
                }
            )
        )

    metrics = pd.DataFrame(metric_rows)
    preds = pd.concat(pred_parts, ignore_index=True)
    auc_values = metrics["roc_auc"].astype(float)
    summary_rows = [
        {"holdout": "LATE_HOLDOUT_MEAN", "roc_auc": float(auc_values.mean())},
        {"holdout": "LATE_HOLDOUT_STD", "roc_auc": float(auc_values.std(ddof=0))},
        {"holdout": "LATE_HOLDOUT_MIN", "roc_auc": float(auc_values.min())},
    ]
    metrics = pd.concat([metrics, pd.DataFrame(summary_rows)], ignore_index=True)
    return metrics, preds


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--holdouts-json", default="")
    parser.add_argument("--out-dir", default="reports/validation/late_holdouts")
    args = parser.parse_args()

    config_path = resolve_path(args.config, repo_root)
    cfg = load_config(config_path)
    train_path = resolve_path(cfg["train"], repo_root)
    test_path = resolve_path(cfg["test"], repo_root)
    out_base = resolve_path(args.out_dir, repo_root)
    out_base.mkdir(parents=True, exist_ok=True)

    holdouts = json.loads(args.holdouts_json) if args.holdouts_json else DEFAULT_HOLDOUTS
    if not isinstance(holdouts, list) or not holdouts:
        raise ValueError("holdouts must be a non-empty JSON list")

    seed = int(cfg["seed"])
    model_cfg = cfg["model"]
    run_id = make_run_id(f"late_holdouts_{cfg['run_prefix']}")
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    log(f"Config: {config_path}")
    log(f"Train: {train_path}")
    log(f"Run id: {run_id}")
    log("Reading CSV files")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    log("Preparing features")
    train_x, test_x, y, categorical_cols, feature_cols, feature_manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": True}),
        excluded_features=cfg.get("excluded_features", []),
    )
    train_x, _ = prepare_lgbm_categories(train_x, test_x, categorical_cols)
    log(f"Prepared features: count={len(feature_cols)}, categorical={categorical_cols}")

    sample_weight_cfg = cfg.get("sample_weight", {"enabled": False})
    sample_weights = make_sample_weights(train, sample_weight_cfg)

    metrics, predictions = evaluate_late_holdouts(
        train=train,
        train_x=train_x,
        y=y,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
        seed=seed,
        holdouts=holdouts,
        sample_weights=sample_weights,
        early_stopping_rounds=int(cfg.get("early_stopping_rounds", 150)),
    )

    metrics_path = out_dir / "late_holdout_metrics.csv"
    preds_path = out_dir / "late_holdout_predictions.csv"
    summary_path = out_dir / "late_holdout_summary.json"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(preds_path, index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L4_late_holdout_battery",
        "config_path": str(config_path),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "seed": seed,
        "model": "LGBMClassifier",
        "params": model_cfg,
        "feature_engineering": cfg.get("feature_engineering", {"enabled": False}),
        "time_features": cfg.get("time_features", {"enabled": True}),
        "sample_weight": sample_weight_cfg,
        "feature_count": len(feature_cols),
        "categorical_features": categorical_cols,
        "holdouts": holdouts,
        "metrics": sanitize_for_json(metrics.to_dict(orient="records")),
        "artifacts": {
            "metrics": str(metrics_path),
            "predictions": str(preds_path),
            "summary": str(summary_path),
        },
        "notes": [
            "Holdouts may overlap, so no combined OOF ROC-AUC is reported.",
            "This script does not predict test_apps.csv and does not build a submission.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("\n== Late holdout metrics ==")
    print(metrics.to_string(index=False))
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
