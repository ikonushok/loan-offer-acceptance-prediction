#!/usr/bin/env python
"""Evaluate whether the Alfa credit-scoring dataset helps offer acceptance.

Assumption for this diagnostic: using the external "Кредитный скоринг" parquet
features is allowed. The script still avoids using the scoring target label as a
feature. It aggregates scoring rows by `id`, joins them to offer `front_id`, and
compares XGBoost time-CV with and without those external aggregates.
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
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_catboost_time import DAY_COL, ID_COL, TARGET, load_config, prepare_features, resolve_path, sha256_file
from baseline_xgb_time import prepare_xgb_categories


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def locate_scoring_dir(path: str) -> Path:
    scoring_dir = Path(path)
    required = ["train_data.parquet", "test_data.parquet", "train_target.csv"]
    missing = [name for name in required if not (scoring_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing scoring files in {scoring_dir}: {missing}")
    return scoring_dir


def filter_parquet_for_ids(path: Path, ids: set[int], batch_size: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    columns = parquet_file.schema_arrow.names
    parts: list[pd.DataFrame] = []
    log(f"Scanning {path}: rows={parquet_file.metadata.num_rows}, columns={len(columns)}")
    for batch_number, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=columns), start=1):
        chunk = batch.to_pandas()
        chunk = chunk[chunk["id"].isin(ids)]
        if not chunk.empty:
            parts.append(chunk)
        if batch_number % 10 == 0:
            log(f"  batches={batch_number}, matched_parts={len(parts)}")
    if not parts:
        return pd.DataFrame(columns=columns)
    matched = pd.concat(parts, ignore_index=True)
    log(f"Matched {len(matched)} rows from {path.name}")
    return matched


def aggregate_scoring_rows(scoring_rows: pd.DataFrame) -> pd.DataFrame:
    if scoring_rows.empty:
        return pd.DataFrame(columns=["front_id"])

    value_cols = [c for c in scoring_rows.columns if c not in {"id"}]
    aggregations: dict[str, list[str]] = {"rn": ["count", "max"]}
    for col in value_cols:
        if col == "rn":
            continue
        aggregations[col] = ["mean", "max", "min"]

    grouped = scoring_rows.groupby("id", sort=False).agg(aggregations)
    grouped.columns = [f"scoring_{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index().rename(columns={"id": ID_COL})

    overdue_cols = [c for c in scoring_rows.columns if c.startswith("pre_loans") and c not in {"pre_loans_credit_limit"}]
    if overdue_cols:
        overdue_flag = scoring_rows[overdue_cols].gt(0).any(axis=1).astype("int8")
        tmp = pd.DataFrame({"id": scoring_rows["id"], "scoring_any_overdue_row": overdue_flag})
        extra = tmp.groupby("id", sort=False)["scoring_any_overdue_row"].agg(["mean", "max"]).reset_index()
        extra = extra.rename(
            columns={
                "id": ID_COL,
                "mean": "scoring_any_overdue_row_mean",
                "max": "scoring_any_overdue_row_max",
            }
        )
        grouped = grouped.merge(extra, on=ID_COL, how="left")

    return grouped


def make_time_folds(train: pd.DataFrame, cutoffs: list[str]) -> list[dict[str, Any]]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    parsed_cutoffs = [pd.Timestamp(c) for c in cutoffs]
    folds: list[dict[str, Any]] = []
    for index, cutoff in enumerate(parsed_cutoffs):
        next_cutoff = parsed_cutoffs[index + 1] if index + 1 < len(parsed_cutoffs) else days.max() + pd.Timedelta(days=1)
        train_idx = np.where(days < cutoff)[0]
        valid_idx = np.where((days >= cutoff) & (days < next_cutoff))[0]
        folds.append(
            {
                "fold": index + 1,
                "cutoff": str(cutoff.date()),
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "train_start": str(days.iloc[train_idx].min().date()),
                "train_end": str(days.iloc[train_idx].max().date()),
                "valid_start": str(days.iloc[valid_idx].min().date()),
                "valid_end": str(days.iloc[valid_idx].max().date()),
            }
        )
    return folds


def fit_eval_xgb(
    train_x: pd.DataFrame,
    y: pd.Series,
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = dict(model_cfg)
    early_stopping_rounds = int(params.pop("early_stopping_rounds", 100))
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("random_state", seed)
    params.setdefault("tree_method", "hist")
    params.setdefault("enable_categorical", True)
    params.setdefault("n_jobs", -1)

    fold_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []
    for fold in folds:
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]
        model = XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        log(
            "Fold {fold}: train {train_start}..{train_end} rows={train_rows}; "
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
        model.fit(
            train_x.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False,
        )
        pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], pred))
        best_iteration = int(getattr(model, "best_iteration", params.get("n_estimators", 0)) or 0)
        log(f"Fold {fold['fold']} auc={auc:.6f}, best_iteration={best_iteration}")
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
                "valid_positive_rate": float(y.iloc[valid_idx].mean()),
                "best_iteration": best_iteration,
                "roc_auc": auc,
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "row_index": valid_idx,
                    TARGET: y.iloc[valid_idx].to_numpy(),
                    "prediction": pred,
                    "fold": fold["fold"],
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    valid_predictions = pd.concat(pred_parts, ignore_index=True).sort_values("row_index")
    oof_auc = float(roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"]))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offer-dir", required=True)
    parser.add_argument("--scoring-dir", required=True)
    parser.add_argument("--config", default="configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1.json")
    parser.add_argument("--output-dir", default="reports/validation/external_scoring")
    parser.add_argument("--batch-size", type=int, default=500_000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    offer_dir = Path(args.offer_dir)
    scoring_dir = locate_scoring_dir(args.scoring_dir)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config_path = resolve_path(args.config, repo_root)
    cfg = load_config(config_path)
    train = pd.read_csv(offer_dir / "train_apps.csv")
    test = pd.read_csv(offer_dir / "test_apps.csv")
    required_ids = set(pd.concat([train[ID_COL], test[ID_COL]], ignore_index=True).astype(int).unique())

    scoring_train_rows = filter_parquet_for_ids(scoring_dir / "train_data.parquet", required_ids, args.batch_size)
    scoring_test_rows = filter_parquet_for_ids(scoring_dir / "test_data.parquet", required_ids, args.batch_size)
    scoring_rows = pd.concat([scoring_train_rows, scoring_test_rows], ignore_index=True)
    scoring_rows_path = output_dir / f"external_scoring_matched_rows_{stamp}.parquet"
    scoring_rows.to_parquet(scoring_rows_path, index=False)

    scoring_agg = aggregate_scoring_rows(scoring_rows)
    scoring_agg_path = output_dir / f"external_scoring_aggregates_{stamp}.csv"
    scoring_agg.to_csv(scoring_agg_path, index=False)

    scoring_cols = [c for c in scoring_agg.columns if c != ID_COL]
    train_ext = train.merge(scoring_agg, on=ID_COL, how="left")
    test_ext = test.merge(scoring_agg, on=ID_COL, how="left")
    for col in scoring_cols:
        train_ext[col] = train_ext[col].astype("float32")
        test_ext[col] = test_ext[col].astype("float32")
    train_ext["scoring_has_history"] = train_ext[scoring_cols].notna().any(axis=1).astype("int8")
    test_ext["scoring_has_history"] = test_ext[scoring_cols].notna().any(axis=1).astype("int8")

    coverage = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "scoring_matched_rows": int(len(scoring_rows)),
        "aggregate_ids": int(len(scoring_agg)),
        "train_covered_rows": int(train_ext["scoring_has_history"].sum()),
        "test_covered_rows": int(test_ext["scoring_has_history"].sum()),
        "train_coverage": float(train_ext["scoring_has_history"].mean()),
        "test_coverage": float(test_ext["scoring_has_history"].mean()),
        "train_covered_positive_rate": float(train_ext.loc[train_ext["scoring_has_history"] == 1, TARGET].mean()),
        "train_uncovered_positive_rate": float(train_ext.loc[train_ext["scoring_has_history"] == 0, TARGET].mean()),
    }
    log(json.dumps(coverage, ensure_ascii=False))

    train_x_base, test_x_base, y, categorical_cols, base_feature_cols, _ = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": True}),
        excluded_features=cfg.get("excluded_features", []),
    )
    train_x_base, test_x_base = prepare_xgb_categories(train_x_base, test_x_base, categorical_cols)

    train_x_ext = pd.concat(
        [train_x_base.reset_index(drop=True), train_ext[scoring_cols + ["scoring_has_history"]].reset_index(drop=True)],
        axis=1,
    )
    test_x_ext = pd.concat(
        [test_x_base.reset_index(drop=True), test_ext[scoring_cols + ["scoring_has_history"]].reset_index(drop=True)],
        axis=1,
    )
    folds = make_time_folds(train, cfg["cutoffs"])

    log("Evaluating base XGB")
    base_metrics, base_predictions = fit_eval_xgb(
        train_x=train_x_base,
        y=y,
        folds=folds,
        model_cfg=cfg["model"],
        seed=int(cfg["seed"]),
    )
    log("Evaluating external-scoring XGB")
    ext_metrics, ext_predictions = fit_eval_xgb(
        train_x=train_x_ext,
        y=y,
        folds=folds,
        model_cfg=cfg["model"],
        seed=int(cfg["seed"]),
    )

    base_metrics["feature_set"] = "base"
    ext_metrics["feature_set"] = "base_plus_external_scoring"
    metrics = pd.concat([base_metrics, ext_metrics], ignore_index=True)
    metrics_path = output_dir / f"external_scoring_xgb_cv_metrics_{stamp}.csv"
    metrics.to_csv(metrics_path, index=False)
    base_predictions.to_csv(output_dir / f"external_scoring_base_valid_predictions_{stamp}.csv", index=False)
    ext_predictions.to_csv(output_dir / f"external_scoring_ext_valid_predictions_{stamp}.csv", index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_diagnostic_external_data_cv",
        "assumption": "External scoring parquet features are allowed; scoring target label is not used.",
        "offer_dir": str(offer_dir),
        "scoring_dir": str(scoring_dir),
        "config_path": str(config_path),
        "offer_train_sha256": sha256_file(offer_dir / "train_apps.csv"),
        "offer_test_sha256": sha256_file(offer_dir / "test_apps.csv"),
        "coverage": coverage,
        "base_feature_count": len(base_feature_cols),
        "external_feature_count": len(scoring_cols) + 1,
        "total_ext_feature_count": train_x_ext.shape[1],
        "fold_metrics": metrics.to_dict(orient="records"),
        "artifacts": {
            "matched_scoring_rows": str(scoring_rows_path),
            "aggregates": str(scoring_agg_path),
            "metrics": str(metrics_path),
        },
        "risk": "Diagnostic only. If positive, external-data permission and leakage review are required before submission.",
    }
    report_path = output_dir / f"external_scoring_xgb_cv_report_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics.pivot_table(index="feature_set", columns="fold", values="roc_auc", aggfunc="first").to_string())
    print(f"report_json: {report_path}")


if __name__ == "__main__":
    main()
