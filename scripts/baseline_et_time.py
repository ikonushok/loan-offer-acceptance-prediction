#!/usr/bin/env python
"""Time-aware ExtraTrees / RandomForest diversity baseline for the Alfa credit-offer task.

ExtraTrees typically has lower correlation with gradient boosting models and can
provide useful blend diversity even if standalone AUC is lower.

Usage:
    python scripts/baseline_et_time.py --config configs/diversity_experiments/et_context_offer_v1.json
    python scripts/baseline_et_time.py --config configs/diversity_experiments/rf_context_offer_v1.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

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


def prepare_et_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode categoricals with OrdinalEncoder (handles unseen with 'use_encoded_value')."""
    train_x = train_x.copy()
    test_x = test_x.copy()
    if not categorical_cols:
        return train_x, test_x

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
        dtype=np.float32,
    )
    combined = pd.concat([train_x[categorical_cols], test_x[categorical_cols]], ignore_index=True)
    enc.fit(combined.astype(str))
    train_x[categorical_cols] = enc.transform(train_x[categorical_cols].astype(str))
    test_x[categorical_cols] = enc.transform(test_x[categorical_cols].astype(str))
    return train_x, test_x


def get_model(model_cfg: dict[str, Any], seed: int) -> ExtraTreesClassifier | RandomForestClassifier:
    params = dict(model_cfg)
    model_type = params.pop("model_type", "ExtraTreesClassifier")
    params["random_state"] = seed
    params.setdefault("n_jobs", -1)
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    return ExtraTreesClassifier(**params)


def evaluate_time_folds(
    train_x: pd.DataFrame,
    y: pd.Series,
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    log(f"Starting time-holdout evaluation: folds={len(folds)}")

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
        sw = None if sample_weights is None else sample_weights.iloc[train_idx].to_numpy()
        model.fit(train_x.iloc[train_idx].to_numpy(dtype=np.float32), y_train.to_numpy(), sample_weight=sw)

        pred = model.predict_proba(train_x.iloc[valid_idx].to_numpy(dtype=np.float32))[:, 1]
        auc = float(roc_auc_score(y_valid, pred))

        log(f"Fold {fold['fold']} done: roc_auc={auc:.6f}")

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
                "best_iteration": np.nan,
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
    log(f"OOF done: rows={len(valid_predictions)}, roc_auc={oof_auc:.6f}")

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
    model_cfg: dict[str, Any],
    seed: int,
    sample_weights: pd.Series | None = None,
) -> pd.Series:
    log(f"Training full model: rows={len(train_x)}, features={train_x.shape[1]}")
    model = get_model(model_cfg=model_cfg, seed=seed)
    sw = None if sample_weights is None else sample_weights.to_numpy()
    model.fit(train_x.to_numpy(dtype=np.float32), y.to_numpy(), sample_weight=sw)
    pred = np.clip(model.predict_proba(test_x.to_numpy(dtype=np.float32))[:, 1], 0.0, 1.0)
    log(
        "Test prediction range: min={:.6f}, mean={:.6f}, max={:.6f}".format(
            float(np.min(pred)), float(np.mean(pred)), float(np.max(pred))
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
    seed = int(cfg["seed"])
    model_cfg = cfg["model"]

    run_id = make_run_id(cfg["run_prefix"], seed)
    out_dir = out_base / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    log(f"Config: {config_path}")
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
    train_x, test_x = prepare_et_features(train_x, test_x, categorical_cols)
    log(f"Features: {len(feature_cols)}, categorical (ordinal-encoded): {categorical_cols}")

    sample_weight_cfg = cfg.get("sample_weight", {"enabled": False})
    sample_weights = make_sample_weights(train, sample_weight_cfg)

    folds = make_time_folds(train, cfg["cutoffs"])
    if not folds:
        raise ValueError("No valid time folds created")

    fold_metrics, valid_predictions = evaluate_time_folds(
        train_x=train_x,
        y=y,
        folds=folds,
        model_cfg=model_cfg,
        seed=seed,
        sample_weights=sample_weights,
    )

    test_pred = train_full_and_predict_test(
        train_x=train_x,
        test_x=test_x,
        y=y,
        model_cfg=model_cfg,
        seed=seed,
        sample_weights=sample_weights,
    )

    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    model_type = model_cfg.get("model_type", "ExtraTreesClassifier")
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_partial_time_holdout",
        "model": model_type,
        "seed": seed,
        "config_path": str(config_path),
        "params": model_cfg,
        "feature_engineering": cfg.get("feature_engineering", {"enabled": False}),
        "time_features": cfg.get("time_features", {"enabled": True}),
        "sample_weight": sample_weight_cfg,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "target": TARGET,
        "excluded_features": [ID_COL, DAY_COL] + cfg.get("excluded_features", []),
        "categorical_features_ordinal_encoded": categorical_cols,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
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
            f"{model_type} is slow on large datasets; use n_jobs=-1.",
            "No early stopping; n_estimators is fixed.",
            "Standalone AUC may be lower than GBM but low correlation makes it useful for blending.",
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
