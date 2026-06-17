#!/usr/bin/env python
"""Time-aware CatBoost baseline for the Alfa credit-offer task.

Default run mode is PyCharm-friendly:
- run this file directly;
- settings are loaded from configs/baseline_catboost_time.json;
- relative paths are resolved from repository root.

The script does not build a platform submission and does not use
sample_submission.csv.
"""

from __future__ import annotations

import argparse
import hashlib
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

RAW_CATEGORICAL_COLS = [
    "db_group_last",
    "fl_adminarea",
]


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file is missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_run_id(prefix: str, seed: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_seed{seed}"


def safe_divide(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Element-wise safe division. Returns NaN when denominator is near zero."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return pd.Series(
        np.where(den.abs() > eps, num / den, np.nan),
        index=numerator.index,
    )


def add_label_free_features(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical_cols: list[str],
    feature_engineering_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add label-free derived features using train/test raw columns only.

    No target values, supervised statistics, target encoding, or sample labels are used.
    """
    if not feature_engineering_cfg.get("enabled", False):
        return train_x, test_x, []

    derived_features: list[str] = []

    def add_feature(name: str, train_values: pd.Series, test_values: pd.Series) -> None:
        train_x[name] = train_values.replace([np.inf, -np.inf], np.nan)
        test_x[name] = test_values.replace([np.inf, -np.inf], np.nan)
        derived_features.append(name)

    if feature_engineering_cfg.get("add_pairwise_features", True):
        paired_specs = [
            ("rate_spread", "offered_rate", "cb_rate", "diff"),
            ("rate_ratio", "offered_rate", "cb_rate", "ratio"),
            ("overdraft_limit_spread", "overdraft_limit_max", "overdraft_limit_min", "diff"),
            ("loan_amount_to_limit_min", "loan_amount_last", "overdraft_limit_min", "ratio"),
            ("loan_amount_to_limit_max", "loan_amount_last", "overdraft_limit_max", "ratio"),
            ("sum_deb_ul_30_to_90", "sum_deb_ul_30", "sum_deb_ul_90", "ratio"),
            ("cnt_deb_ul_ip_30_to_90", "cnt_deb_ul_ip_30", "cnt_deb_ul_ip_90", "ratio"),
            ("cnt_cred_to_deb_loan_90", "cnt_cred_loan_90", "cnt_deb_loan_90", "ratio"),
            ("overdraft_term_to_app_term_360", "overdraft_app_term_max_360", "app_term_mean_360", "ratio"),
            ("balance_to_loan_amount", "balance_rur_amt_30_min", "loan_amount_last", "ratio"),
            ("time_spent_per_dashboard_event", "p75_time_spent_minutes", "count_all_corp_dashboard_events", "ratio"),
        ]

        for name, left, right, op in paired_specs:
            if left not in train_x.columns or right not in train_x.columns:
                continue
            if op == "diff":
                add_feature(name, train_x[left] - train_x[right], test_x[left] - test_x[right])
            elif op == "ratio":
                add_feature(name, safe_divide(train_x[left], train_x[right]), safe_divide(test_x[left], test_x[right]))
            else:
                raise ValueError(f"Unknown derived feature op: {op}")

        if {"overdraft_limit_min", "overdraft_limit_max"}.issubset(train_x.columns):
            add_feature(
                "overdraft_limit_mid",
                (train_x["overdraft_limit_min"] + train_x["overdraft_limit_max"]) / 2.0,
                (test_x["overdraft_limit_min"] + test_x["overdraft_limit_max"]) / 2.0,
            )
            add_feature(
                "loan_amount_to_limit_mid",
                safe_divide(train_x["loan_amount_last"], train_x["overdraft_limit_mid"]),
                safe_divide(test_x["loan_amount_last"], test_x["overdraft_limit_mid"]),
            )

    if feature_engineering_cfg.get("add_context_offer_features", False):
        offer_cols = ["offered_rate", "overdraft_limit_min", "overdraft_limit_max"]
        pre_context_derived_features = set(derived_features)
        context_cols = [
            c
            for c in train_x.columns
            if c not in offer_cols and c not in pre_context_derived_features and not c.endswith("_is_missing")
        ]

        if all(c in train_x.columns for c in offer_cols) and context_cols:
            train_context_sig = pd.util.hash_pandas_object(train_x[context_cols], index=False)
            test_context_sig = pd.util.hash_pandas_object(test_x[context_cols], index=False)

            context_feature_names = ["context_offer_count"]
            for col in offer_cols:
                context_feature_names.extend(
                    [
                        f"{col}_context_spread",
                        f"{col}_minus_context_min",
                        f"{col}_minus_context_mean",
                        f"{col}_rank_pct_in_context",
                    ]
                )

            def add_context_features(df: pd.DataFrame, sig: pd.Series) -> None:
                group_size = sig.map(sig.value_counts()).astype("int32")
                df["context_offer_count"] = group_size

                for col in offer_cols:
                    grouped = df.groupby(sig, dropna=False)[col]
                    group_min = grouped.transform("min")
                    group_max = grouped.transform("max")
                    group_mean = grouped.transform("mean")
                    rank_pct = grouped.rank(method="average", pct=True)

                    new_cols = {
                        f"{col}_context_spread": group_max - group_min,
                        f"{col}_minus_context_min": df[col] - group_min,
                        f"{col}_minus_context_mean": df[col] - group_mean,
                        f"{col}_rank_pct_in_context": rank_pct,
                    }

                    for name, values in new_cols.items():
                        df[name] = values.replace([np.inf, -np.inf], np.nan)

            add_context_features(train_x, train_context_sig)
            add_context_features(test_x, test_context_sig)
            derived_features.extend(context_feature_names)

    if feature_engineering_cfg.get("add_missing_flags", True):
        base_cols = [
            c
            for c in train_x.columns
            if c not in categorical_cols and not c.endswith("_is_missing")
        ]
        for col in base_cols:
            if train_x[col].isna().any() or test_x[col].isna().any():
                flag = f"{col}_is_missing"
                train_x[flag] = train_x[col].isna().astype("int8")
                test_x[flag] = test_x[col].isna().astype("int8")
                derived_features.append(flag)

    return train_x, test_x, derived_features


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_engineering_cfg: dict[str, Any],
    time_features_cfg: dict[str, Any] | None = None,
    excluded_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str], pd.DataFrame]:
    y = train[TARGET].copy()

    train_x = train.drop(columns=[TARGET]).copy()
    test_x = test.copy()

    if ID_COL not in train_x.columns or ID_COL not in test_x.columns:
        raise ValueError(f"{ID_COL} must exist in train and test")
    if DAY_COL not in train_x.columns or DAY_COL not in test_x.columns:
        raise ValueError(f"{DAY_COL} must exist in train and test")

    train_days = pd.to_datetime(train_x[DAY_COL], errors="raise")
    test_days = pd.to_datetime(test_x[DAY_COL], errors="raise")
    min_day = train_days.min()
    time_features_cfg = time_features_cfg or {"enabled": True}
    time_features_enabled = bool(time_features_cfg.get("enabled", True))
    derived_time_features: list[str] = []

    if time_features_enabled:
        for df, days in [(train_x, train_days), (test_x, test_days)]:
            df["decision_day_num"] = (days - min_day).dt.days.astype("int32")
            df["decision_month"] = days.dt.to_period("M").astype(str)
            df["decision_dayofweek"] = days.dt.dayofweek.astype("int8")
        derived_time_features = ["decision_day_num", "decision_month", "decision_dayofweek"]

    excluded_features = excluded_features or []
    forbidden_exclusions = {ID_COL, TARGET}
    bad_exclusions = sorted(set(excluded_features) & forbidden_exclusions)
    if bad_exclusions:
        raise ValueError(f"Do not pass ID/target columns via excluded_features: {bad_exclusions}")

    excluded_cols = [ID_COL, DAY_COL]
    for col in excluded_features:
        if col in train_x.columns and col not in excluded_cols:
            excluded_cols.append(col)

    train_x = train_x.drop(columns=excluded_cols)
    test_x = test_x.drop(columns=[c for c in excluded_cols if c in test_x.columns])

    if list(train_x.columns) != list(test_x.columns):
        raise ValueError("Train/test feature columns are not aligned after preprocessing")

    categorical_cols = [
        c
        for c in RAW_CATEGORICAL_COLS + ["decision_month"]
        if c in train_x.columns
    ]

    for col in categorical_cols:
        train_x[col] = train_x[col].astype("string").fillna("__MISSING__").astype(str)
        test_x[col] = test_x[col].astype("string").fillna("__MISSING__").astype(str)

    train_x, test_x, engineered_features = add_label_free_features(
        train_x=train_x,
        test_x=test_x,
        categorical_cols=categorical_cols,
        feature_engineering_cfg=feature_engineering_cfg,
    )

    if list(train_x.columns) != list(test_x.columns):
        raise ValueError("Train/test feature columns are not aligned after feature engineering")

    feature_manifest_rows: list[dict[str, Any]] = []
    for col in train.columns:
        if col == TARGET:
            role = "target"
            used = False
            transformed_to = ""
            reason = "target label"
        elif col == ID_COL:
            role = "id_excluded"
            used = False
            transformed_to = ""
            reason = "identifier excluded"
        elif col == DAY_COL:
            role = "date_transformed"
            used = False
            transformed_to = ", ".join(derived_time_features)
            reason = "date converted to label-free temporal features" if time_features_enabled else "date excluded by config"
        elif col in excluded_features:
            role = "excluded_feature"
            used = False
            transformed_to = ""
            reason = "excluded by config"
        elif col in categorical_cols:
            role = "categorical_feature"
            used = True
            transformed_to = col
            reason = "CatBoost categorical feature"
        else:
            role = "numeric_feature"
            used = col in train_x.columns
            transformed_to = col if used else ""
            reason = "model feature" if used else "not used"

        feature_manifest_rows.append(
            {
                "source_column": col,
                "role": role,
                "used_directly": used,
                "transformed_to": transformed_to,
                "reason": reason,
            }
        )

    for derived in derived_time_features:
        feature_manifest_rows.append(
            {
                "source_column": DAY_COL,
                "role": "derived_feature",
                "used_directly": True,
                "transformed_to": derived,
                "reason": "safe label-free temporal transform",
            }
        )

    for derived in engineered_features:
        feature_manifest_rows.append(
            {
                "source_column": "",
                "role": "label_free_engineered_feature",
                "used_directly": True,
                "transformed_to": derived,
                "reason": "configured label-free feature engineering",
            }
        )

    feature_manifest = pd.DataFrame(feature_manifest_rows)

    return train_x, test_x, y, categorical_cols, list(train_x.columns), feature_manifest



def make_time_folds(train: pd.DataFrame, cutoffs: list[str]) -> list[dict[str, Any]]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    folds: list[dict[str, Any]] = []

    parsed_cutoffs = [pd.Timestamp(c) for c in cutoffs]
    for i, cutoff in enumerate(parsed_cutoffs):
        next_cutoff = (
            parsed_cutoffs[i + 1]
            if i + 1 < len(parsed_cutoffs)
            else days.max() + pd.Timedelta(days=1)
        )

        train_idx = np.where(days < cutoff)[0]
        valid_idx = np.where((days >= cutoff) & (days < next_cutoff))[0]

        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        folds.append(
            {
                "fold": i + 1,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "train_start": str(days.iloc[train_idx].min().date()),
                "train_end": str(days.iloc[train_idx].max().date()),
                "valid_start": str(days.iloc[valid_idx].min().date()),
                "valid_end": str(days.iloc[valid_idx].max().date()),
                "cutoff": str(cutoff.date()),
                "next_cutoff": str(next_cutoff.date()),
            }
        )

    return folds


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


def evaluate_time_folds(
    train_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    folds: list[dict[str, Any]],
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_features = [train_x.columns.get_loc(c) for c in categorical_cols]

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

        train_pool = Pool(
            train_x.iloc[train_idx],
            label=y_train,
            cat_features=cat_features,
        )
        valid_pool = Pool(
            train_x.iloc[valid_idx],
            label=y_valid,
            cat_features=cat_features,
        )

        model = get_model(model_cfg=model_cfg, seed=seed)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        pred = model.predict_proba(valid_pool)[:, 1]
        auc = roc_auc_score(y_valid, pred)
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
                "roc_auc": float(auc),
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

    oof_auc = roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"])
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
        "roc_auc": float(oof_auc),
    }

    return fold_metrics, valid_predictions


def train_full_and_predict_test(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    model_cfg: dict[str, Any],
    seed: int,
) -> pd.Series:
    cat_features = [train_x.columns.get_loc(c) for c in categorical_cols]

    train_pool = Pool(train_x, label=y, cat_features=cat_features)
    test_pool = Pool(test_x, cat_features=cat_features)

    log(f"Training final model on full train: rows={len(train_x)}, features={train_x.shape[1]}")
    model = get_model(model_cfg=model_cfg, seed=seed)
    model.fit(train_pool)

    log(f"Predicting test: rows={len(test_x)}")
    pred = model.predict_proba(test_pool)[:, 1]
    pred = np.clip(pred, 0.0, 1.0)

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
    parser.add_argument(
        "--config",
        default="configs/baseline_catboost_time.json",
        help="Path to JSON config. Relative paths are resolved from repo root.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config, repo_root)
    cfg = load_config(config_path)

    train_path = resolve_path(cfg["train"], repo_root)
    test_path = resolve_path(cfg["test"], repo_root)
    out_base = resolve_path(cfg["out_dir"], repo_root)
    model_cfg = cfg["model"]
    seed = int(cfg["seed"])

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

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
        train,
        test,
        feature_engineering_cfg=feature_engineering_cfg,
        time_features_cfg=time_features_cfg,
        excluded_features=excluded_features,
    )
    log(f"Prepared features: count={len(feature_cols)}, categorical={categorical_cols}")

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
    )

    test_pred = train_full_and_predict_test(
        train_x=train_x,
        test_x=test_x,
        y=y,
        categorical_cols=categorical_cols,
        model_cfg=model_cfg,
        seed=seed,
    )

    log("Writing artifacts")
    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: test[ID_COL],
            "prediction": test_pred,
        }
    ).to_csv(out_dir / "test_predictions.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    (out_dir / "config_used.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_partial_time_holdout",
        "model": "CatBoostClassifier",
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
            "reason": "test period is future-dated relative to train; random CV is not primary validation",
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
            "Submission row contract unresolved; this script predicts only test_apps.csv rows.",
            "Repeated context/sibling-offer groups exist; time split is primary mitigation.",
            "No hyperparameter tuning performed.",
            "CatBoost auto_class_weights makes scores useful for ranking but not calibrated probabilities.",
        ],
    }

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n== Fold metrics ==")
    print(fold_metrics.to_string(index=False))
    print("\nArtifacts:")
    for name, path in summary["artifacts"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
