#!/usr/bin/env python3
"""Optuna HPO runner for LightGBM time-validation experiments.

Mirrors optuna_catboost_time.py but uses LightGBM.

Usage:
    python scripts/optuna_lgbm_time.py --config configs/optuna_lgbm_context_offer_fold3_hpo100.json

After HPO completes, export top configs for revalidation:
    python scripts/optuna_lgbm_time.py --config configs/optuna_lgbm_context_offer_fold3_hpo100.json \\
        --export-configs-dir configs/lgbm_optuna_revalidated --top-k 10

Then revalidate each config on all folds using baseline_lgbm_time.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_catboost_time import (  # noqa: E402
    DAY_COL,
    TARGET,
    load_config,
    log,
    make_sample_weights,
    make_time_folds,
    prepare_features,
    resolve_path,
    sha256_file,
)
from scripts.baseline_lgbm_time import get_model, prepare_lgbm_categories  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def build_model_params(trial: optuna.Trial, search_space: dict, fixed_params: dict) -> dict[str, Any]:
    params: dict[str, Any] = dict(fixed_params)
    for name, spec in search_space.items():
        stype = spec["type"]
        if stype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif stype == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif stype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search space type: {stype}")
    # subsample_freq must be 1 when subsample < 1.0 for LightGBM
    if "subsample" in params and params.get("subsample", 1.0) < 1.0:
        params["subsample_freq"] = 1
    return params


def make_objective(
    *,
    train_x: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    folds: list[dict[str, Any]],
    objective_fold_idx: int,
    search_space: dict,
    fixed_params: dict,
    seed: int,
    early_stopping_rounds: int,
) -> Any:
    from sklearn.metrics import roc_auc_score

    target_fold = folds[objective_fold_idx]

    def objective(trial: optuna.Trial) -> float:
        model_params = build_model_params(trial, search_space, fixed_params)
        train_idx = target_fold["train_idx"]
        valid_idx = target_fold["valid_idx"]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = get_model(model_cfg=model_params, seed=seed)
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model.fit(
            train_x.iloc[train_idx],
            y_train,
            eval_set=[(train_x.iloc[valid_idx], y_valid)],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=callbacks,
        )
        pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y_valid, pred))
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or model_params.get("n_estimators", 0)))
        trial.set_user_attr("valid_rows", int(len(valid_idx)))
        trial.set_user_attr("valid_positive_rate", float(y_valid.mean()))
        return auc

    return objective


def run_hpo(args: argparse.Namespace) -> None:
    hpo_cfg = _read_json(_resolve(args.config))
    base_cfg = load_config(_resolve(hpo_cfg["base_config"]))

    train_path = _resolve(base_cfg["train"])
    test_path = _resolve(base_cfg["test"])
    out_dir = _resolve(hpo_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(hpo_cfg.get("seed", 42))
    study_name = hpo_cfg["study_name"]
    n_trials = int(hpo_cfg.get("n_trials", 100))
    n_startup = int(hpo_cfg.get("n_startup_trials", 30))
    objective_fold_num = int(hpo_cfg.get("objective_fold", 3))
    early_stopping_rounds = int(hpo_cfg.get("early_stopping_rounds", 150))
    search_space = hpo_cfg["search_space"]
    fixed_params = hpo_cfg.get("fixed_model_params", {})
    storage = hpo_cfg.get("storage")
    timeout = hpo_cfg.get("timeout_seconds")

    log(f"Study: {study_name}")
    log(f"Config: {args.config}, Base: {hpo_cfg['base_config']}")
    log(f"Trials: {n_trials}, Startup: {n_startup}, Objective fold: {objective_fold_num}")
    log("Reading data")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    log("Preparing features")
    train_x, test_x, y, categorical_cols, feature_cols, _ = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=base_cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=base_cfg.get("time_features", {"enabled": True}),
        excluded_features=base_cfg.get("excluded_features", []),
    )
    train_x, _ = prepare_lgbm_categories(train_x, test_x, categorical_cols)
    log(f"Features: {len(feature_cols)}, categorical: {categorical_cols}")

    folds = make_time_folds(train, base_cfg["cutoffs"])
    if not folds:
        raise ValueError("No valid time folds created")
    fold_nums = [f["fold"] for f in folds]
    if objective_fold_num not in fold_nums:
        raise ValueError(f"objective_fold={objective_fold_num} not in folds: {fold_nums}")
    objective_fold_idx = fold_nums.index(objective_fold_num)
    log(f"Objective fold: {objective_fold_num} (idx={objective_fold_idx}), valid_rows={len(folds[objective_fold_idx]['valid_idx'])}")

    # Optuna sampler
    sampler_cfg = hpo_cfg.get("sampler", {})
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=sampler_cfg.get("multivariate", True),
        group=sampler_cfg.get("group", True),
        n_startup_trials=n_startup,
    )

    load_if_exists = hpo_cfg.get("load_if_exists", False)
    if storage and hpo_cfg.get("reset_existing_artifacts", False) and not load_if_exists:
        db_path = Path(storage.replace("sqlite:///", "").replace("sqlite://", ""))
        if not db_path.is_absolute():
            db_path = ROOT / db_path
        if db_path.exists():
            db_path.unlink()
            log(f"Deleted existing DB: {db_path}")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=load_if_exists,
    )

    objective = make_objective(
        train_x=train_x,
        y=y,
        categorical_cols=categorical_cols,
        folds=folds,
        objective_fold_idx=objective_fold_idx,
        search_space=search_space,
        fixed_params=fixed_params,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )

    show_progress = hpo_cfg.get("show_progress_bar", True)
    log("Starting Optuna HPO...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress,
        n_jobs=1,
    )

    best = study.best_trial
    log(f"Best trial #{best.number}: AUC={best.value:.6f}, params={best.params}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trials_csv = out_dir / f"{study_name}_{stamp}_trials.csv"
    summary_json = out_dir / f"{study_name}_{stamp}_summary.json"

    trials_df = study.trials_dataframe()
    trials_df.to_csv(trials_csv, index=False)
    log(f"Trials CSV: {trials_csv}")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(_resolve(args.config)),
        "base_config": hpo_cfg["base_config"],
        "study_name": study_name,
        "validation_level": "L3_partial_fold_only_hpo",
        "objective_fold": objective_fold_num,
        "n_trials_requested": n_trials,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_startup_trials": n_startup,
        "sampler": sampler_cfg,
        "best_trial_number": int(best.number),
        "best_objective_auc": float(best.value),
        "best_iteration": best.user_attrs.get("best_iteration"),
        "best_params": best.params,
        "fixed_model_params": fixed_params,
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "trials_csv": str(trials_csv),
        "risks": [
            "This is fold-only HPO. Top trials must be revalidated on all rolling folds before model selection.",
            "No public leaderboard was used.",
        ],
    }
    _write_json(summary_json, summary)
    log(f"Summary JSON: {summary_json}")

    print(f"\nBest trial #{best.number}: Fold{objective_fold_num} AUC = {best.value:.6f}")
    print(f"Best params: {json.dumps(best.params, indent=2)}")
    print(f"\nNext step: revalidate top trials on all folds.")
    print(f"  trials_csv: {trials_csv}")
    print(f"  Run: python scripts/optuna_lgbm_time.py --config {args.config} --export-configs-dir configs/lgbm_optuna_revalidated --trials-csv {trials_csv} --top-k 8")


def export_configs(args: argparse.Namespace) -> None:
    """Export top-k trial params as standalone config JSONs for revalidation."""
    hpo_cfg = _read_json(_resolve(args.config))
    base_cfg_path = _resolve(hpo_cfg["base_config"])
    base_cfg = load_config(base_cfg_path)
    study_name = hpo_cfg["study_name"]
    out_dir = _resolve(hpo_cfg["out_dir"])

    if args.trials_csv:
        trials_csv = _resolve(args.trials_csv)
    else:
        candidates = sorted(out_dir.glob(f"{study_name}_*_trials.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No trials CSV found for {study_name} in {out_dir}")
        trials_csv = candidates[0]
    log(f"Using trials CSV: {trials_csv}")

    trials = pd.read_csv(trials_csv)
    completed = trials[trials["state"] == "COMPLETE"].sort_values("value", ascending=False)
    top_k = min(int(args.top_k), len(completed))
    top_trials = completed.head(top_k)

    export_dir = _resolve(args.export_configs_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    fixed_params = hpo_cfg.get("fixed_model_params", {})

    for _, row in top_trials.iterrows():
        trial_num = int(row["number"])
        auc = float(row["value"])
        model_params: dict[str, Any] = dict(fixed_params)
        for col in row.index:
            if not col.startswith("params_"):
                continue
            name = col.removeprefix("params_")
            val = row[col]
            if pd.isna(val) or val == "None":
                continue
            if val == "True":
                val = True
            elif val == "False":
                val = False
            # Type casting
            if name in ("n_estimators", "num_leaves", "min_child_samples"):
                val = int(float(val))
            elif name in ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"):
                val = float(val)
            model_params[name] = val

        new_cfg = dict(base_cfg)
        new_cfg["run_prefix"] = f"lgbm_optuna_trial{trial_num:04d}"
        new_cfg["model"] = model_params
        new_cfg["_hpo_source"] = {
            "study_name": study_name,
            "trial_number": trial_num,
            "fold_auc": round(auc, 6),
        }
        fname = f"lgbm_optuna_trial{trial_num:04d}.json"
        out_path = export_dir / fname
        out_path.write_text(json.dumps(new_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log(f"  Exported trial #{trial_num:4d}: AUC={auc:.6f} → {fname}")

    print(f"\nExported {top_k} configs to {export_dir}")
    print("Revalidate with:")
    for _, row in top_trials.iterrows():
        trial_num = int(row["number"])
        print(f"  python scripts/baseline_lgbm_time.py --config configs/lgbm_optuna_revalidated/lgbm_optuna_trial{trial_num:04d}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for LightGBM time-holdout experiments")
    parser.add_argument("--config", required=True, help="HPO config JSON path")
    parser.add_argument("--export-configs-dir", default="", help="If set, export top-k configs and exit")
    parser.add_argument("--trials-csv", default="", help="Explicit trials CSV for export (optional)")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top trials to export")
    args = parser.parse_args()

    if args.export_configs_dir:
        export_configs(args)
    else:
        run_hpo(args)


if __name__ == "__main__":
    main()
