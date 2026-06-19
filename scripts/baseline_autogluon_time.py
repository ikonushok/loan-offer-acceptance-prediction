#!/usr/bin/env python3
"""Time-aware AutoGluon Tabular diversity baseline.

This script intentionally follows the local project validation contract:
models are selected by rolling time folds and optional late-holdout follow-up,
not by AutoGluon's internal leaderboard.
"""

from __future__ import annotations

import argparse
import inspect
import json
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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


def import_autogluon() -> tuple[Any, Any]:
    try:
        from autogluon.tabular import TabularDataset, TabularPredictor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "AutoGluon is not installed in this Python environment. "
            "Use a Python version supported by AutoGluon and install "
            "`autogluon.tabular>=1.1,<2.0` before running this script."
        ) from exc
    return TabularDataset, TabularPredictor


def make_run_id(prefix: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_seed{seed}"


def package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not_installed"


def prepare_autogluon_frame(features: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    frame = features.copy()
    for col in categorical_cols:
        frame[col] = frame[col].astype("category")
    return frame


def positive_class_scores(proba: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            scores = proba[1].to_numpy(dtype=float)
        elif "1" in proba.columns:
            scores = proba["1"].to_numpy(dtype=float)
        else:
            scores = proba.iloc[:, -1].to_numpy(dtype=float)
    elif isinstance(proba, pd.Series):
        scores = proba.to_numpy(dtype=float)
    else:
        arr = np.asarray(proba, dtype=float)
        scores = arr[:, -1] if arr.ndim == 2 else arr
    scores = np.clip(scores, 0.0, 1.0)
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain NaN or inf")
    return scores


def get_fit_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    fit_kwargs = dict(model_cfg.get("fit_kwargs", {}))
    fit_kwargs.setdefault("presets", model_cfg.get("presets", "good_quality"))
    fit_kwargs.setdefault("time_limit", model_cfg.get("time_limit", 1800))
    fit_kwargs.setdefault("num_cpus", model_cfg.get("num_cpus", "auto"))
    fit_kwargs.setdefault("ag_args_fit", {"num_gpus": 0})
    if "verbosity" not in fit_kwargs:
        fit_kwargs["verbosity"] = int(model_cfg.get("fit_verbosity", 2))
    return fit_kwargs


def filter_fit_kwargs(predictor: Any, fit_kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(predictor.fit)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fit_kwargs

    supported = set(signature.parameters)
    filtered = {key: value for key, value in fit_kwargs.items() if key in supported}
    ignored = sorted(set(fit_kwargs) - set(filtered))
    if ignored:
        log(f"AutoGluon fit kwargs ignored because this version does not support them: {ignored}")
    return filtered


def build_predictor(model_dir: Path, verbosity: int) -> Any:
    _, TabularPredictor = import_autogluon()
    return TabularPredictor(
        label=TARGET,
        eval_metric="roc_auc",
        problem_type="binary",
        path=str(model_dir),
        verbosity=verbosity,
    )


def fit_predictor(
    train_frame: pd.DataFrame,
    y: pd.Series,
    model_dir: Path,
    model_cfg: dict[str, Any],
    seed: int,
) -> Any:
    TabularDataset, _ = import_autogluon()
    np.random.seed(seed)
    fit_data = train_frame.copy()
    fit_data[TARGET] = y.to_numpy()
    predictor = build_predictor(model_dir=model_dir, verbosity=int(model_cfg.get("predictor_verbosity", 2)))
    predictor.fit(
        train_data=TabularDataset(fit_data),
        **filter_fit_kwargs(predictor, get_fit_kwargs(model_cfg=model_cfg)),
    )
    return predictor


def evaluate_time_folds(
    train_x: pd.DataFrame,
    y: pd.Series,
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    log(f"Starting AutoGluon time-holdout evaluation: folds={len(folds)}")

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

        model_dir = out_dir / "ag_models" / f"fold_{fold['fold']}"
        predictor = fit_predictor(
            train_frame=train_x.iloc[train_idx],
            y=y_train,
            model_dir=model_dir,
            model_cfg=model_cfg,
            seed=seed + int(fold["fold"]),
        )
        pred = positive_class_scores(predictor.predict_proba(train_x.iloc[valid_idx]))
        auc = float(roc_auc_score(y_valid, pred))

        leaderboard_path = out_dir / f"leaderboard_fold_{fold['fold']}.csv"
        predictor.leaderboard(silent=True).to_csv(leaderboard_path, index=False)

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
                "roc_auc": auc,
                "leaderboard": str(leaderboard_path),
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
        "roc_auc": oof_auc,
        "leaderboard": "",
    }
    return fold_metrics, valid_predictions


def train_full_and_predict_test(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    model_cfg: dict[str, Any],
    seed: int,
    out_dir: Path,
) -> pd.Series:
    log(f"Training final AutoGluon model on full train: rows={len(train_x)}, features={train_x.shape[1]}")
    predictor = fit_predictor(
        train_frame=train_x,
        y=y,
        model_dir=out_dir / "ag_models" / "full_train",
        model_cfg=model_cfg,
        seed=seed,
    )
    predictor.leaderboard(silent=True).to_csv(out_dir / "leaderboard_full_train.csv", index=False)
    log(f"Predicting test: rows={len(test_x)}")
    pred = positive_class_scores(predictor.predict_proba(test_x))
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
    seed = int(cfg["seed"])
    model_cfg = cfg["model"]

    if cfg.get("sample_weight", {"enabled": False}).get("enabled", False):
        raise ValueError("AutoGluon baseline does not support sample weights yet; keep sample_weight.enabled=false")

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
    train_x = prepare_autogluon_frame(train_x, categorical_cols)
    test_x = prepare_autogluon_frame(test_x, categorical_cols)
    log(f"Prepared features: count={len(feature_cols)}, categorical={categorical_cols}")

    folds = make_time_folds(train, cfg["cutoffs"])
    if not folds:
        raise ValueError("No valid time folds were created")

    fold_metrics, valid_predictions = evaluate_time_folds(
        train_x=train_x,
        y=y,
        folds=folds,
        model_cfg=model_cfg,
        seed=seed,
        out_dir=out_dir,
    )

    test_pred = train_full_and_predict_test(
        train_x=train_x,
        test_x=test_x,
        y=y,
        model_cfg=model_cfg,
        seed=seed,
        out_dir=out_dir,
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
        "model": "AutoGluon TabularPredictor",
        "package_versions": {
            "autogluon.tabular": package_version("autogluon.tabular"),
            "autogluon.core": package_version("autogluon.core"),
        },
        "seed": seed,
        "config_path": str(config_path),
        "params": model_cfg,
        "feature_engineering": feature_engineering_cfg,
        "time_features": time_features_cfg,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "target": TARGET,
        "positive_class": 1,
        "excluded_features": [ID_COL, DAY_COL] + excluded_features,
        "categorical_features": categorical_cols,
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "validation_policy": {
            "type": "rolling_time_holdout",
            "cutoffs": cfg["cutoffs"],
            "reason": "test period is future-dated relative to train; AutoGluon internal leaderboard is diagnostic only",
        },
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "artifacts": {
            "config_used": str(out_dir / "config_used.json"),
            "fold_metrics": str(out_dir / "fold_metrics.csv"),
            "valid_predictions": str(out_dir / "valid_predictions_time.csv"),
            "test_predictions": str(out_dir / "test_predictions.csv"),
            "feature_manifest": str(out_dir / "feature_manifest.csv"),
            "summary": str(out_dir / "run_summary.json"),
            "leaderboard_full_train": str(out_dir / "leaderboard_full_train.csv"),
        },
        "risks": [
            "AutoGluon internal leaderboard is not the selection metric.",
            "No sample weights are used in this baseline.",
            "No late-holdout battery is run by this script; run it only if Fold3/OOF are competitive.",
            "This script predicts only test_apps.csv rows.",
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
