#!/usr/bin/env python
"""Scan hard recent-window time CV for drift adaptation.

This diagnostic script intentionally does not train final test models and does
not create submissions. It reuses the existing fold-safe feature preparation and
model factories, then restricts each fold's training rows to the last N days
before that fold's validation cutoff.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_catboost_time import (  # noqa: E402
    DAY_COL,
    TARGET,
    evaluate_time_folds as evaluate_catboost_time_folds,
    load_config,
    make_sample_weights,
    prepare_features,
    resolve_path,
    sha256_file,
)
from baseline_xgb_time import (  # noqa: E402
    evaluate_time_folds as evaluate_xgb_time_folds,
    prepare_xgb_categories,
)


def make_recent_time_folds(
    train: pd.DataFrame,
    cutoffs: list[str],
    window_days: int | None,
) -> list[dict[str, Any]]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    folds: list[dict[str, Any]] = []
    parsed_cutoffs = [pd.Timestamp(c) for c in cutoffs]

    for fold_number, cutoff in enumerate(parsed_cutoffs, start=1):
        next_cutoff = (
            parsed_cutoffs[fold_number]
            if fold_number < len(parsed_cutoffs)
            else days.max() + pd.Timedelta(days=1)
        )
        train_mask = days < cutoff
        if window_days is not None:
            train_mask &= days >= cutoff - pd.Timedelta(days=window_days)
        valid_mask = (days >= cutoff) & (days < next_cutoff)

        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        folds.append(
            {
                "fold": fold_number,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "train_start": str(days.iloc[train_idx].min().date()),
                "train_end": str(days.iloc[train_idx].max().date()),
                "valid_start": str(days.iloc[valid_idx].min().date()),
                "valid_end": str(days.iloc[valid_idx].max().date()),
                "cutoff": str(cutoff.date()),
                "next_cutoff": str(next_cutoff.date()),
                "window_days": window_days,
            }
        )

    return folds


def fold3_auc(fold_metrics: pd.DataFrame) -> float:
    value = fold_metrics.loc[fold_metrics["fold"].astype(str) == "3", "roc_auc"]
    if value.empty:
        return float("nan")
    return float(value.iloc[0])


def oof_auc(fold_metrics: pd.DataFrame) -> float:
    value = fold_metrics.loc[fold_metrics["fold"].astype(str) == "OOF_TIME_HOLDOUT", "roc_auc"]
    if value.empty:
        return float("nan")
    return float(value.iloc[0])


def late_holdout_auc(valid_predictions: pd.DataFrame, valid_start_row: int) -> float:
    subset = valid_predictions[valid_predictions["row_index"] >= valid_start_row]
    if subset.empty or subset[TARGET].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(subset[TARGET], subset["prediction"]))


def run_scan(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, repo_root)
    cfg = load_config(config_path)
    train_path = resolve_path(cfg["train"], repo_root)
    test_path = resolve_path(cfg["test"], repo_root)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_x, test_x, y, categorical_cols, feature_cols, feature_manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": True}),
        excluded_features=cfg.get("excluded_features", []),
    )

    if args.model == "xgb":
        train_x, test_x = prepare_xgb_categories(train_x, test_x, categorical_cols)

    sample_weights = make_sample_weights(train, cfg.get("sample_weight", {"enabled": False}))
    model_cfg = dict(cfg["model"])
    if args.quiet:
        model_cfg["verbose"] = False

    windows: list[int | None] = [None if w == "expanding" else int(w) for w in args.windows.split(",")]
    rows: list[dict[str, Any]] = []
    prediction_files: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_fold3_start = int(np.where(pd.to_datetime(train[DAY_COL]) >= pd.Timestamp("2025-04-01"))[0].min())

    for window_days in windows:
        folds = make_recent_time_folds(train, cfg["cutoffs"], window_days)
        if not folds:
            raise ValueError(f"No folds for window_days={window_days}")

        if args.model == "catboost":
            fold_metrics, valid_predictions = evaluate_catboost_time_folds(
                train_x=train_x,
                y=y,
                categorical_cols=categorical_cols,
                folds=folds,
                model_cfg=model_cfg,
                seed=int(cfg["seed"]),
                sample_weights=sample_weights,
            )
        elif args.model == "xgb":
            fold_metrics, valid_predictions = evaluate_xgb_time_folds(
                train_x=train_x,
                y=y,
                folds=folds,
                model_cfg=model_cfg,
                seed=int(cfg["seed"]),
                sample_weights=sample_weights,
            )
        else:
            raise ValueError(f"Unsupported model: {args.model}")

        window_label = "expanding" if window_days is None else f"{window_days}d"
        pred_path = output_dir / f"recent_window_{args.model}_{window_label}_{stamp}_valid_predictions.csv"
        valid_predictions.to_csv(pred_path, index=False)
        prediction_files.append(str(pred_path))

        for _, row in fold_metrics.iterrows():
            metric_row = row.to_dict()
            metric_row.update(
                {
                    "model": args.model,
                    "config": str(config_path),
                    "window_days": window_days,
                    "window_label": window_label,
                    "feature_count": len(feature_cols),
                }
            )
            rows.append(metric_row)

        rows.append(
            {
                "model": args.model,
                "config": str(config_path),
                "fold": "FOLD3_PLUS",
                "window_days": window_days,
                "window_label": window_label,
                "train_start": "",
                "train_end": "",
                "valid_start": "2025-04-01",
                "valid_end": str(pd.to_datetime(train[DAY_COL]).max().date()),
                "train_rows": np.nan,
                "valid_rows": int((valid_predictions["row_index"] >= valid_fold3_start).sum()),
                "valid_positive_rate": float(
                    valid_predictions.loc[valid_predictions["row_index"] >= valid_fold3_start, TARGET].mean()
                ),
                "best_iteration": np.nan,
                "roc_auc": late_holdout_auc(valid_predictions, valid_fold3_start),
                "feature_count": len(feature_cols),
            }
        )

    results = pd.DataFrame(rows)
    summary_path = output_dir / f"recent_window_{args.model}_{stamp}_summary.csv"
    results.to_csv(summary_path, index=False)

    compact = (
        results[results["fold"].astype(str).isin(["1", "2", "3", "OOF_TIME_HOLDOUT", "FOLD3_PLUS"])]
        .pivot_table(index=["model", "window_label"], columns="fold", values="roc_auc", aggfunc="first")
        .reset_index()
    )
    compact_path = output_dir / f"recent_window_{args.model}_{stamp}_compact.csv"
    compact.to_csv(compact_path, index=False)

    best_fold3 = (
        results[results["fold"].astype(str) == "3"]
        .sort_values("roc_auc", ascending=False)
        .head(1)
        .to_dict(orient="records")
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_diagnostic_recent_window_cv",
        "model": args.model,
        "config_path": str(config_path),
        "windows": ["expanding" if w is None else w for w in windows],
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "feature_count": len(feature_cols),
        "categorical_cols": categorical_cols,
        "best_by_fold3": best_fold3,
        "artifacts": {
            "summary_csv": str(summary_path),
            "compact_csv": str(compact_path),
            "valid_predictions": prediction_files,
        },
        "risk": "Diagnostic only; no submission generated. Window selected on validation would need late-holdout/red-team before upload.",
    }
    report_path = output_dir / f"recent_window_{args.model}_{stamp}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(compact.to_string(index=False))
    print(f"\nsummary_csv: {summary_path}")
    print(f"compact_csv: {compact_path}")
    print(f"report_json: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["catboost", "xgb"], required=True)
    parser.add_argument("--windows", default="expanding,90,120,180,240,365")
    parser.add_argument("--output-dir", default="reports/validation/recent_windows")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
