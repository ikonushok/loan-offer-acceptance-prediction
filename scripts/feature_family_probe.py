#!/usr/bin/env python
"""Fast XGBoost probes for untested label-free feature families.

This is a diagnostic script: no final model, no test predictions, no submission.
It compares feature families on the same rolling time folds as the champion.
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
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_catboost_time import DAY_COL, TARGET, load_config, make_time_folds, prepare_features, resolve_path, sha256_file
from baseline_xgb_time import prepare_xgb_categories


REPO = Path(__file__).resolve().parents[1]
MONEY_COLS = [
    "loan_amount_last",
    "overdraft_limit_min",
    "overdraft_limit_max",
    "sum_deb_ul_90",
    "sum_deb_ul_30",
    "balance_rur_amt_30_min",
    "loan_rev_max_start_non_fin",
    "loan_rev_min_start_fin",
    "sum_deb_investment_90",
]
NUMERIC_RAW = [
    "loan_amount_last",
    "overdraft_limit_min",
    "overdraft_limit_max",
    "offered_rate",
    "cb_rate",
    "corp_credit_products",
    "sum_deb_ul_90",
    "sum_deb_ul_30",
    "cnt_deb_loan_90",
    "cnt_deb_ul_ip_90",
    "cnt_deb_ul_ip_30",
    "balance_rur_amt_30_min",
    "cnt_cred_loan_90",
    "loan_rev_max_start_non_fin",
    "loan_rev_min_start_fin",
    "app_term_mean_360",
    "overdraft_app_term_max_360",
    "days_from_authperson_registration",
    "fl_hdb_bki_total_active_products",
    "corp_list",
    "count_all_corp_dashboard_events",
    "p75_time_spent_minutes",
    "sum_deb_investment_90",
]
CATEGORICAL_RAW = ["db_group_last", "fl_adminarea"]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def signed_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def add_log_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> list[str]:
    added: list[str] = []
    for col in MONEY_COLS:
        if col in train_x.columns:
            name = f"{col}_signed_log1p"
            train_x[name] = signed_log1p(train_x[col])
            test_x[name] = signed_log1p(test_x[col])
            added.append(name)
    return added


def add_zero_flag_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> list[str]:
    added: list[str] = []
    for col in NUMERIC_RAW:
        if col in train_x.columns:
            name = f"{col}_is_zero"
            train_x[name] = (pd.to_numeric(train_x[col], errors="coerce") == 0).astype("int8")
            test_x[name] = (pd.to_numeric(test_x[col], errors="coerce") == 0).astype("int8")
            added.append(name)
    train_x["row_missing_count"] = train_x[[c for c in NUMERIC_RAW if c in train_x.columns]].isna().sum(axis=1).astype("int16")
    test_x["row_missing_count"] = test_x[[c for c in NUMERIC_RAW if c in test_x.columns]].isna().sum(axis=1).astype("int16")
    added.append("row_missing_count")
    return added


def add_frequency_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> list[str]:
    added: list[str] = []
    for col in CATEGORICAL_RAW + ["corp_list", "fl_hdb_bki_total_active_products"]:
        if col not in train_x.columns:
            continue
        combined = pd.concat([train_x[col], test_x[col]], ignore_index=True).astype("string").fillna("__MISSING__")
        freq = combined.value_counts(dropna=False)
        name = f"{col}_combined_freq"
        train_x[name] = train_x[col].astype("string").fillna("__MISSING__").map(freq).astype("float32")
        test_x[name] = test_x[col].astype("string").fillna("__MISSING__").map(freq).astype("float32")
        added.append(name)
    return added


def add_quantile_bin_features(train_x: pd.DataFrame, test_x: pd.DataFrame, categorical_cols: list[str]) -> list[str]:
    added: list[str] = []
    for col in NUMERIC_RAW:
        if col not in train_x.columns:
            continue
        combined = pd.concat([train_x[col], test_x[col]], ignore_index=True)
        ranks = combined.rank(method="first")
        try:
            bins = pd.qcut(ranks, q=16, labels=False, duplicates="drop").astype("int16")
        except ValueError:
            continue
        name = f"{col}_qbin16"
        train_x[name] = bins.iloc[: len(train_x)].to_numpy()
        test_x[name] = bins.iloc[len(train_x) :].to_numpy()
        added.append(name)
    return added


def add_cat_interactions(train_x: pd.DataFrame, test_x: pd.DataFrame, categorical_cols: list[str]) -> list[str]:
    added: list[str] = []
    pairs = [
        ("db_group_last", "fl_adminarea"),
        ("db_group_last", "corp_list"),
        ("fl_adminarea", "corp_list"),
    ]
    for left, right in pairs:
        if left not in train_x.columns or right not in train_x.columns:
            continue
        name = f"{left}__x__{right}"
        train_x[name] = train_x[left].astype("string").fillna("__MISSING__") + "__" + train_x[right].astype("string").fillna("__MISSING__")
        test_x[name] = test_x[left].astype("string").fillna("__MISSING__") + "__" + test_x[right].astype("string").fillna("__MISSING__")
        categorical_cols.append(name)
        added.append(name)
    return added


def add_time_regime_features(train: pd.DataFrame, test: pd.DataFrame, train_x: pd.DataFrame, test_x: pd.DataFrame, categorical_cols: list[str]) -> list[str]:
    train_days = pd.to_datetime(train[DAY_COL], errors="raise")
    test_days = pd.to_datetime(test[DAY_COL], errors="raise")
    origin = train_days.min()
    train_x["decision_day_num_probe"] = (train_days - origin).dt.days.astype("int32")
    test_x["decision_day_num_probe"] = (test_days - origin).dt.days.astype("int32")
    train_x["decision_month_probe"] = train_days.dt.to_period("M").astype(str)
    test_x["decision_month_probe"] = test_days.dt.to_period("M").astype(str)
    train_x["decision_week_probe"] = train_days.dt.isocalendar().week.astype("int16")
    test_x["decision_week_probe"] = test_days.dt.isocalendar().week.astype("int16")
    categorical_cols.extend(["decision_month_probe"])
    return ["decision_day_num_probe", "decision_month_probe", "decision_week_probe"]


def apply_variant(
    variant: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical_cols: list[str],
) -> list[str]:
    added: list[str] = []
    if variant in {"log", "all_label_free"}:
        added.extend(add_log_features(train_x, test_x))
    if variant in {"zero_flags", "all_label_free"}:
        added.extend(add_zero_flag_features(train_x, test_x))
    if variant in {"frequency", "all_label_free"}:
        added.extend(add_frequency_features(train_x, test_x))
    if variant in {"quantile_bins", "all_label_free"}:
        added.extend(add_quantile_bin_features(train_x, test_x, categorical_cols))
    if variant in {"cat_interactions", "all_label_free"}:
        added.extend(add_cat_interactions(train_x, test_x, categorical_cols))
    if variant == "time_regime":
        added.extend(add_time_regime_features(train, test, train_x, test_x, categorical_cols))
    return added


def evaluate_xgb(
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
    rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for fold in folds:
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]
        model = XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        model.fit(
            train_x.iloc[train_idx],
            y.iloc[train_idx],
            eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])],
            verbose=False,
        )
        pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], pred))
        rows.append(
            {
                "fold": fold["fold"],
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "best_iteration": int(getattr(model, "best_iteration", model_cfg.get("n_estimators", 0)) or 0),
                "roc_auc": auc,
            }
        )
        parts.append(pd.DataFrame({"row_index": valid_idx, TARGET: y.iloc[valid_idx].to_numpy(), "prediction": pred, "fold": fold["fold"]}))
    valid_predictions = pd.concat(parts, ignore_index=True).sort_values("row_index")
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.loc[len(fold_metrics)] = {
        "fold": "OOF_TIME_HOLDOUT",
        "train_rows": np.nan,
        "valid_rows": len(valid_predictions),
        "best_iteration": np.nan,
        "roc_auc": float(roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"])),
    }
    return fold_metrics, valid_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1.json")
    parser.add_argument("--variants", default="base,log,zero_flags,frequency,quantile_bins,cat_interactions,time_regime,all_label_free")
    parser.add_argument("--output-dir", default="reports/validation/feature_family_probes")
    args = parser.parse_args()

    config_path = resolve_path(args.config, REPO)
    cfg = load_config(config_path)
    train_path = resolve_path(cfg["train"], REPO)
    test_path = resolve_path(cfg["test"], REPO)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    folds = make_time_folds(train, cfg["cutoffs"])
    output_dir = resolve_path(args.output_dir, REPO)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    metric_parts: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        log(f"Preparing variant={variant}")
        train_x, test_x, y, categorical_cols, feature_cols, _ = prepare_features(
            train=train,
            test=test,
            feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
            time_features_cfg=cfg.get("time_features", {"enabled": True}),
            excluded_features=cfg.get("excluded_features", []),
        )
        added_features: list[str] = []
        if variant != "base":
            added_features = apply_variant(variant, train, test, train_x, test_x, categorical_cols)
        train_x, test_x = prepare_xgb_categories(train_x, test_x, categorical_cols)
        log(f"Evaluating variant={variant}, features={train_x.shape[1]}, added={len(added_features)}")
        fold_metrics, valid_predictions = evaluate_xgb(train_x, y, folds, cfg["model"], int(cfg["seed"]))
        fold_metrics["variant"] = variant
        fold_metrics["feature_count"] = train_x.shape[1]
        fold_metrics["added_feature_count"] = len(added_features)
        metric_parts.append(fold_metrics)
        pred_path = output_dir / f"feature_probe_{variant}_{stamp}_valid_predictions.csv"
        valid_predictions.to_csv(pred_path, index=False)
        reports.append(
            {
                "variant": variant,
                "feature_count": train_x.shape[1],
                "added_features": added_features,
                "predictions": str(pred_path),
                "metrics": fold_metrics.to_dict(orient="records"),
            }
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics_path = output_dir / f"feature_family_probe_{stamp}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    compact = metrics.pivot_table(index="variant", columns="fold", values="roc_auc", aggfunc="first").reset_index()
    compact_path = output_dir / f"feature_family_probe_{stamp}_compact.csv"
    compact.to_csv(compact_path, index=False)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_diagnostic_feature_family_probe",
        "config_path": str(config_path),
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "reports": reports,
        "artifacts": {
            "metrics": str(metrics_path),
            "compact": str(compact_path),
        },
        "risk": "Diagnostic only; variants selected on validation require late-holdout/red-team before submission.",
    }
    report_path = output_dir / f"feature_family_probe_{stamp}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(compact.to_string(index=False))
    print(f"report_json: {report_path}")


if __name__ == "__main__":
    main()
