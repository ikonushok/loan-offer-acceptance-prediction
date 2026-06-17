#!/usr/bin/env python
"""CatBoost SHAP diagnostics for Fold 3 time holdout.

The script is PyCharm-friendly:
- run this file directly;
- default config is configs/baseline_catboost_time.json;
- paths are resolved from repository root.

It trains only the Fold 3 model, computes CatBoost native SHAP values on a
validation sample, and saves feature-importance diagnostics under
reports/validation/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


TARGET = "target_value"
ID_COL = "front_id"
DAY_COL = "decision_day"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def load_baseline_module(repo_root: Path) -> Any:
    script_path = repo_root / "scripts" / "baseline_catboost_time.py"
    spec = importlib.util.spec_from_file_location("baseline_catboost_time", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_model(model_cfg: dict[str, Any], seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=int(model_cfg["iterations"]),
        learning_rate=float(model_cfg["learning_rate"]),
        depth=int(model_cfg["depth"]),
        l2_leaf_reg=float(model_cfg["l2_leaf_reg"]),
        random_seed=seed,
        auto_class_weights=model_cfg.get("auto_class_weights"),
        allow_writing_files=False,
        verbose=model_cfg.get("verbose", False),
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_catboost_time.json")
    parser.add_argument("--out-dir", default="reports/validation")
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--fold-number", type=int, default=3)
    args = parser.parse_args()

    config_path = resolve_path(args.config, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    seed = int(cfg["seed"])
    feature_engineering_cfg = cfg.get("feature_engineering", {"enabled": False})
    time_features_cfg = cfg.get("time_features", {"enabled": True})

    baseline = load_baseline_module(repo_root)

    train_path = resolve_path(cfg["train"], repo_root)
    test_path = resolve_path(cfg["test"], repo_root)

    log(f"Config: {config_path}")
    log(f"Train: {train_path}")
    log(f"Test: {test_path}")
    log("Reading CSV files")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    log("Preparing features through baseline module")
    train_x, _test_x, y, categorical_cols, feature_cols, feature_manifest = baseline.prepare_features(
        train,
        test,
        feature_engineering_cfg=feature_engineering_cfg,
        time_features_cfg=time_features_cfg,
    )

    folds = baseline.make_time_folds(train, cfg["cutoffs"])
    selected = [f for f in folds if int(f["fold"]) == int(args.fold_number)]
    if len(selected) != 1:
        raise ValueError(f"Cannot find fold {args.fold_number}; available folds: {[f['fold'] for f in folds]}")
    fold = selected[0]

    train_idx = fold["train_idx"]
    valid_idx = fold["valid_idx"]

    cat_features = [train_x.columns.get_loc(c) for c in categorical_cols]

    train_pool = Pool(
        train_x.iloc[train_idx],
        label=y.iloc[train_idx],
        cat_features=cat_features,
    )
    valid_pool = Pool(
        train_x.iloc[valid_idx],
        label=y.iloc[valid_idx],
        cat_features=cat_features,
    )

    log(
        "Training fold {fold}: train {train_start}..{train_end} rows={train_rows}; "
        "valid {valid_start}..{valid_end} rows={valid_rows}".format(
            fold=fold["fold"],
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            train_rows=len(train_idx),
            valid_start=fold["valid_start"],
            valid_end=fold["valid_end"],
            valid_rows=len(valid_idx),
        )
    )

    model = get_model(model_cfg=model_cfg, seed=seed)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    valid_pred = model.predict_proba(valid_pool)[:, 1]
    auc = roc_auc_score(y.iloc[valid_idx], valid_pred)
    log(f"Fold {fold['fold']} ROC-AUC: {auc:.6f}; best_iteration={model.get_best_iteration()}")

    rng = np.random.default_rng(seed)
    sample_n = min(int(args.sample_size), len(valid_idx))
    sampled_valid_positions = rng.choice(np.arange(len(valid_idx)), size=sample_n, replace=False)
    sampled_global_idx = valid_idx[sampled_valid_positions]

    sample_pool = Pool(
        train_x.iloc[sampled_global_idx],
        label=y.iloc[sampled_global_idx],
        cat_features=cat_features,
    )

    log(f"Computing CatBoost SHAP values: sample_size={sample_n}, features={len(feature_cols)}")
    shap_values = model.get_feature_importance(sample_pool, type="ShapValues")
    shap_feature_values = shap_values[:, :-1]

    mean_abs_shap = np.abs(shap_feature_values).mean(axis=0)
    mean_shap = shap_feature_values.mean(axis=0)
    max_abs_shap = np.abs(shap_feature_values).max(axis=0)

    shap_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "max_abs_shap": max_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    sampled_rows = pd.DataFrame(
        {
            "row_index": sampled_global_idx,
            ID_COL: train.iloc[sampled_global_idx][ID_COL].to_numpy(),
            DAY_COL: train.iloc[sampled_global_idx][DAY_COL].to_numpy(),
            TARGET: y.iloc[sampled_global_idx].to_numpy(),
            "prediction": model.predict_proba(sample_pool)[:, 1],
        }
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_prefix = cfg["run_prefix"]
    report_prefix = f"shap_fold{fold['fold']}_{safe_prefix}_{stamp}_seed{seed}"

    importance_path = out_dir / f"{report_prefix}_importance.csv"
    rows_path = out_dir / f"{report_prefix}_sample_rows.csv"
    manifest_path = out_dir / f"{report_prefix}_feature_manifest.csv"
    summary_path = out_dir / f"{report_prefix}_summary.json"

    shap_importance.to_csv(importance_path, index=False)
    sampled_rows.to_csv(rows_path, index=False)
    feature_manifest.to_csv(manifest_path, index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_interpretability_fold_holdout",
        "config_path": str(config_path),
        "run_prefix": cfg["run_prefix"],
        "fold": {
            "fold": fold["fold"],
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "sample_rows": sample_n,
            "roc_auc": float(auc),
            "best_iteration": int(model.get_best_iteration() or model_cfg["iterations"]),
        },
        "feature_engineering": feature_engineering_cfg,
        "time_features": time_features_cfg,
        "model_params": model_cfg,
        "artifacts": {
            "importance": str(importance_path),
            "sample_rows": str(rows_path),
            "feature_manifest": str(manifest_path),
            "summary": str(summary_path),
        },
        "notes": [
            "CatBoost native SHAP values were computed on validation fold sample only.",
            "SHAP is diagnostic and does not replace time-holdout ROC-AUC.",
            "No target encoding, sample labels, or test labels are used.",
        ],
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== TOP 30 FEATURES BY MEAN ABS SHAP ==")
    print(shap_importance.head(30).to_string(index=False))
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
