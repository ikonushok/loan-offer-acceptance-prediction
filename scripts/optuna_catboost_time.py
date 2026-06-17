#!/usr/bin/env python3
"""Optuna HPO runner for CatBoost time-validation experiments."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from optuna.exceptions import ExperimentalWarning
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=ExperimentalWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_catboost_time import make_time_folds, prepare_features  # noqa: E402


TARGET = "target_value"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _resolve_from_root(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _find_latest_trials_csv(study_name: str, out_dir: Path) -> Path:
    candidates = sorted(
        out_dir.glob(f"{study_name}_*_trials.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No trials CSV found for study={study_name!r} in {out_dir}. "
            "Pass --trials-csv explicitly after Optuna finishes."
        )
    return candidates[0]


def _normalize_trial_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        if value == "None":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
    return value


def _extract_trial_params(row: pd.Series) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for col, value in row.items():
        if not col.startswith("params_"):
            continue
        name = col.removeprefix("params_")
        parsed = _normalize_trial_value(value)
        if parsed is not None:
            params[name] = parsed
    return params


def _cast_model_params(params: dict[str, Any]) -> dict[str, Any]:
    int_params = {"iterations", "depth", "border_count", "min_data_in_leaf", "max_leaves"}
    float_params = {"learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature"}

    casted: dict[str, Any] = {}
    for name, value in params.items():
        if name in int_params:
            casted[name] = int(value)
        elif name in float_params:
            casted[name] = float(value)
        else:
            casted[name] = value
    return casted


def _export_top_configs(
    *,
    args: argparse.Namespace,
    hpo_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
    base_cfg_path: Path,
    out_dir: Path,
) -> None:
    trials_csv = _resolve_from_root(args.trials_csv) if args.trials_csv else _find_latest_trials_csv(
        hpo_cfg["study_name"],
        out_dir,
    )
    export_dir = _resolve_from_root(args.export_configs_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    trials = pd.read_csv(trials_csv)
    if "value" not in trials.columns:
        raise ValueError(f"Missing required column 'value' in {trials_csv}")
    if "number" not in trials.columns:
        raise ValueError(f"Missing required column 'number' in {trials_csv}")

    if "state" in trials.columns:
        trials = trials[trials["state"].eq("COMPLETE")].copy()

    trials = (
        trials.dropna(subset=["value"])
        .sort_values(["value", "number"], ascending=[False, True])
        .head(int(args.top_n))
        .reset_index(drop=True)
    )
    if trials.empty:
        raise ValueError(f"No completed trials with non-null value found in {trials_csv}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    created_configs: list[dict[str, Any]] = []

    for rank, row in trials.iterrows():
        trial_number = int(row["number"])
        objective_auc = float(row["value"])
        optuna_params = _extract_trial_params(row)

        cfg = json.loads(json.dumps(base_cfg))
        model_cfg = dict(cfg["model"])
        model_cfg.update(_cast_model_params(optuna_params))
        model_cfg["verbose"] = int(args.export_verbose)

        cfg["seed"] = int(args.export_seed)
        cfg["run_prefix"] = f"{args.export_run_prefix}_trial{trial_number:04d}"
        cfg["model"] = model_cfg
        cfg["optuna_source"] = {
            "study_name": hpo_cfg["study_name"],
            "trials_csv": str(trials_csv),
            "trial_number": trial_number,
            "rank_by_objective": int(rank + 1),
            "objective_fold": int(hpo_cfg["objective_fold"]),
            "objective_metric": "Fold ROC-AUC",
            "objective_auc": objective_auc,
            "params": optuna_params,
            "selection_note": "Fold-only HPO screen; this config must be revalidated on all rolling folds before model selection.",
        }

        config_path = export_dir / f"{args.export_run_prefix}_trial{trial_number:04d}_{stamp}.json"
        _write_json(config_path, cfg)

        created_configs.append(
            {
                "rank": int(rank + 1),
                "trial_number": trial_number,
                "objective_auc": objective_auc,
                "config_path": str(config_path.relative_to(ROOT)),
                "run_prefix": cfg["run_prefix"],
                "params": optuna_params,
            }
        )

        print(
            f"created rank={rank + 1:02d} trial={trial_number:04d} "
            f"objective_auc={objective_auc:.6f} config={config_path.relative_to(ROOT)}"
        )

    manifest_path = export_dir / f"{args.export_run_prefix}_{stamp}_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L0_config_generation",
        "source": "scripts/optuna_catboost_time.py --export-top-configs",
        "study_name": hpo_cfg["study_name"],
        "trials_csv": str(trials_csv),
        "base_config": str(base_cfg_path),
        "top_n": int(args.top_n),
        "export_seed": int(args.export_seed),
        "export_run_prefix": args.export_run_prefix,
        "created_configs": created_configs,
        "next_step": "Run scripts/baseline_catboost_time.py --config <config_path> for each created config to perform full 3-fold revalidation.",
    }
    _write_json(manifest_path, manifest)

    print("\nmanifest:", manifest_path.relative_to(ROOT))
    print("\nNext command template:")
    print("python scripts/baseline_catboost_time.py --config <one_created_config.json>")


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1)))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported search-space type for {name}: {kind}")


def _make_sampler(cfg: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_cfg = cfg["sampler"]
    sampler_type = sampler_cfg.get("type", "TPESampler")
    if sampler_type != "TPESampler":
        raise ValueError(f"Unsupported sampler type: {sampler_type}")

    return optuna.samplers.TPESampler(
        seed=int(cfg["seed"]),
        n_startup_trials=int(cfg["n_startup_trials"]),
        multivariate=bool(sampler_cfg.get("multivariate", True)),
        group=bool(sampler_cfg.get("group", True)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/optuna_catboost_context_offer_fold3_hpo200.json", help="Path to Optuna HPO config JSON.")
    parser.add_argument("--export-top-configs", action="store_true", help="Export top Optuna trials as baseline CatBoost configs and exit.")
    parser.add_argument("--trials-csv", default=None, help="Optuna trials CSV. Defaults to latest trials CSV for the configured study.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top completed trials to export.")
    parser.add_argument("--export-configs-dir", default="configs/optuna_revalidated", help="Directory for exported baseline configs.")
    parser.add_argument("--export-run-prefix", default="catboost_optuna_context_offer_revalidated", help="Run prefix for exported baseline configs.")
    parser.add_argument("--export-seed", type=int, default=42, help="Seed for exported baseline configs.")
    parser.add_argument("--export-verbose", type=int, default=100, help="CatBoost verbose value for exported baseline configs.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    hpo_cfg = _read_json(cfg_path)

    base_cfg_path = Path(hpo_cfg["base_config"])
    if not base_cfg_path.is_absolute():
        base_cfg_path = ROOT / base_cfg_path
    base_cfg = _read_json(base_cfg_path)

    out_dir = Path(hpo_cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(base_cfg["train"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    test_path = Path(base_cfg["test"])
    if not test_path.is_absolute():
        test_path = ROOT / test_path

    if args.export_top_configs:
        print(f"Config: {cfg_path.resolve()}")
        print(f"Base config: {base_cfg_path.resolve()}")
        print(f"Output dir: {out_dir.resolve()}")
        print(f"Study: {hpo_cfg['study_name']}")
        _export_top_configs(
            args=args,
            hpo_cfg=hpo_cfg,
            base_cfg=base_cfg,
            base_cfg_path=base_cfg_path,
            out_dir=out_dir,
        )
        return

    if bool(hpo_cfg.get("reset_existing_artifacts", False)):
        study_name = hpo_cfg["study_name"]
        removed_paths: list[str] = []

        for pattern in (
            f"{study_name}.db",
            f"{study_name}.db-journal",
            f"{study_name}.db-shm",
            f"{study_name}.db-wal",
            f"{study_name}_*_trials.csv",
            f"{study_name}_*_summary.json",
            f"{study_name}_*_optimization_history.html",
            f"{study_name}_*_param_importances.html",
            f"{study_name}_*_parallel_coordinate.html",
            f"{study_name}_*_slice.html",
        ):
            for artifact_path in out_dir.glob(pattern):
                if artifact_path.is_file():
                    artifact_path.unlink()
                    removed_paths.append(str(artifact_path.relative_to(ROOT)))

        if removed_paths:
            print("Removed previous Optuna artifacts:")
            for removed_path in removed_paths:
                print(f"- {removed_path}")
        else:
            print("No previous Optuna artifacts to remove.")

    print(f"Config: {cfg_path.resolve()}")
    print(f"Base config: {base_cfg_path.resolve()}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Study: {hpo_cfg['study_name']}")
    print(f"Trials: {hpo_cfg['n_trials']}; startup_trials: {hpo_cfg['n_startup_trials']}")
    print(f"Objective fold: {hpo_cfg['objective_fold']}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_x, test_x, y, categorical_cols, feature_cols, manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=base_cfg["feature_engineering"],
        time_features_cfg=base_cfg["time_features"],
        excluded_features=base_cfg.get("excluded_features", []),
    )

    folds = make_time_folds(train, base_cfg["cutoffs"])
    objective_fold = int(hpo_cfg["objective_fold"])
    fold = [f for f in folds if int(f["fold"]) == objective_fold][0]

    tr_idx = fold["train_idx"]
    va_idx = fold["valid_idx"]

    cat_features = [train_x.columns.get_loc(c) for c in categorical_cols]
    train_pool = Pool(train_x.iloc[tr_idx], label=y.iloc[tr_idx], cat_features=cat_features)
    valid_pool = Pool(train_x.iloc[va_idx], label=y.iloc[va_idx], cat_features=cat_features)

    fixed_model_params = dict(hpo_cfg["fixed_model_params"])
    search_space = dict(hpo_cfg["search_space"])
    conditional_search_space = dict(hpo_cfg.get("conditional_search_space", {}))

    def objective(trial: optuna.Trial) -> float:
        params = dict(fixed_model_params)

        for name, spec in search_space.items():
            params[name] = _suggest(trial, name, spec)

        grow_policy = params.get("grow_policy")
        for condition, conditional_specs in conditional_search_space.items():
            key, expected = condition.split("==", 1)
            key = key.strip()
            expected = expected.strip()
            if str(params.get(key)) == expected:
                for name, spec in conditional_specs.items():
                    params[name] = _suggest(trial, name, spec)

        params["random_seed"] = int(hpo_cfg["seed"])
        params["allow_writing_files"] = False
        params["verbose"] = False

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=int(hpo_cfg["early_stopping_rounds"]),
        )

        pred = model.predict_proba(valid_pool)[:, 1]
        auc = roc_auc_score(y.iloc[va_idx], pred)

        best_iter = model.get_best_iteration()
        trial.set_user_attr("best_iteration", int(best_iter if best_iter is not None else params["iterations"]))
        trial.set_user_attr("feature_count", int(len(feature_cols)))
        trial.set_user_attr("categorical_cols", list(categorical_cols))
        trial.set_user_attr("valid_rows", int(len(va_idx)))
        trial.set_user_attr("valid_positive_rate", float(y.iloc[va_idx].mean()))

        return float(auc)

    sampler = _make_sampler(hpo_cfg)

    storage = hpo_cfg.get("storage")
    if storage and storage.startswith("sqlite:///"):
        sqlite_path = Path(storage.replace("sqlite:///", "", 1))
        if not sqlite_path.is_absolute():
            sqlite_path = ROOT / sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        storage = "sqlite:///" + str(sqlite_path)

    study = optuna.create_study(
        study_name=hpo_cfg["study_name"],
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=bool(hpo_cfg.get("load_if_exists", True)),
    )

    study.optimize(
        objective,
        n_trials=int(hpo_cfg["n_trials"]),
        timeout=hpo_cfg.get("timeout_seconds"),
        show_progress_bar=bool(hpo_cfg.get("show_progress_bar", True)),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trials_df = study.trials_dataframe()
    trials_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_trials.csv"
    summary_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_summary.json"

    optimization_history_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_optimization_history.html"
    param_importances_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_param_importances.html"
    parallel_coordinate_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_parallel_coordinate.html"
    slice_path = out_dir / f"{hpo_cfg['study_name']}_{stamp}_slice.html"

    trials_df.to_csv(trials_path, index=False)

    visualization_paths: dict[str, str] = {}
    visualization_errors: dict[str, str] = {}

    visualization_specs = [
        ("optimization_history", optimization_history_path, optuna.visualization.plot_optimization_history),
        ("param_importances", param_importances_path, optuna.visualization.plot_param_importances),
        ("parallel_coordinate", parallel_coordinate_path, optuna.visualization.plot_parallel_coordinate),
        ("slice", slice_path, optuna.visualization.plot_slice),
    ]

    for viz_name, viz_path, viz_fn in visualization_specs:
        try:
            fig = viz_fn(study)
            fig.write_html(str(viz_path))
            visualization_paths[viz_name] = str(viz_path)
        except Exception as exc:
            visualization_errors[viz_name] = repr(exc)

    best = study.best_trial
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "base_config": hpo_cfg["base_config"],
        "study_name": hpo_cfg["study_name"],
        "validation_level": "L3_partial_fold_only_hpo",
        "objective_fold": objective_fold,
        "n_trials_requested": int(hpo_cfg["n_trials"]),
        "n_startup_trials": int(hpo_cfg["n_startup_trials"]),
        "sampler": hpo_cfg["sampler"],
        "best_trial_number": int(best.number),
        "best_objective_auc": float(best.value),
        "best_iteration": best.user_attrs.get("best_iteration"),
        "best_params": best.params,
        "fixed_model_params": fixed_model_params,
        "trials_csv": str(trials_path),
        "visualizations": visualization_paths,
        "visualization_errors": visualization_errors,
        "risks": [
            "This is fold-only HPO. Top trials must be revalidated on all rolling folds before model selection.",
            "No public leaderboard was used.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    print("\n===== Optuna HPO done =====")
    print(f"trials_csv: {trials_path}")
    print(f"summary: {summary_path}")
    if visualization_paths:
        print("visualizations:")
        for viz_name, viz_path in visualization_paths.items():
            print(f"- {viz_name}: {viz_path}")
    if visualization_errors:
        print("visualization_errors:")
        for viz_name, err in visualization_errors.items():
            print(f"- {viz_name}: {err}")
    print(f"best_trial: {best.number}")
    print(f"best_objective_auc: {best.value:.6f}")
    print(f"best_iteration: {best.user_attrs.get('best_iteration')}")
    print("best_params:")
    print(json.dumps(best.params, ensure_ascii=False, indent=2))

    cols = ["number", "value", "user_attrs_best_iteration"] + [
        c for c in trials_df.columns if c.startswith("params_")
    ]
    print("\nTop 20:")
    print(trials_df.sort_values("value", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
