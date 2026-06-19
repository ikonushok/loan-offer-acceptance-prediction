#!/usr/bin/env python3
"""Multi-seed and stability checks for the XGB time-regime branch.

This script is self-contained and does not use leaderboard feedback. It reuses
the project's local feature builders, trains additional seeds, scans
rank-blends against the champion, and runs a compact time-feature ablation on
Fold3 plus late holdouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_catboost_time import DAY_COL, ID_COL, TARGET, load_config, make_time_folds, prepare_features, resolve_path, sha256_file
from baseline_xgb_time import get_model, prepare_xgb_categories


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1_seed43.json"
REPORT_DIR = REPO / "reports/validation/time_regime_multiseed"
RUN_DIR = REPO / "experiments/runs"
UPLOAD_DIR = REPO / "submissions/upload_20260620"
CARDS_DIR = REPO / "submissions/cards"

CHAMPION_RUN = REPO / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z"
SEED42_VALID = REPO / "reports/validation/feature_family_probes/feature_probe_time_regime_20260619T041432Z_valid_predictions.csv"
SEED42_LATE = REPO / "reports/validation/late_holdouts/late_holdouts_xgb_hpo_time_regime_probe_20260619T041911Z/late_holdout_predictions.csv"
SEED42_TEST = REPO / "experiments/runs/xgb_hpo_time_regime_probe_full_20260619T0420Z_seed42/test_predictions.csv"
TEST_PATH = REPO / "data/raw/test_apps.csv"
HOLDOUTS = [
    {"name": "H1_2025_03_01_to_end", "cutoff": "2025-03-01"},
    {"name": "H2_2025_04_01_to_end", "cutoff": "2025-04-01"},
    {"name": "H3_2025_05_01_to_end", "cutoff": "2025-05-01"},
]
HOLDOUT_NAMES = [h["name"] for h in HOLDOUTS]
CHAMP_COMPONENTS = {
    "t070": (REPO / "reports/validation/late_holdouts/late_holdouts_catboost_optuna_context_offer_revalidated_trial0070_20260617T130617Z/late_holdout_predictions.csv", 0.70),
    "t041": (REPO / "reports/validation/late_holdouts/late_holdouts_catboost_optuna_context_offer_revalidated_trial0041_20260617T131136Z/late_holdout_predictions.csv", 0.04),
    "t164": (REPO / "reports/validation/late_holdouts/late_holdouts_catboost_optuna_context_offer_revalidated_trial0164_20260617T131519Z/late_holdout_predictions.csv", 0.26),
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(method="average") / len(series)


def auc_by_fold(df: pd.DataFrame, col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        out[f"fold{fold}_auc"] = float(roc_auc_score(sub[TARGET], sub[col]))
    out["oof_auc"] = float(roc_auc_score(df[TARGET], df[col]))
    return out


def auc_by_holdout(df: pd.DataFrame, col: str) -> dict[str, float]:
    out: dict[str, float] = {}
    values: list[float] = []
    for holdout in HOLDOUT_NAMES:
        sub = df[df["holdout"] == holdout]
        auc = float(roc_auc_score(sub[TARGET], sub[col]))
        out[holdout] = auc
        values.append(auc)
    out["lh_mean"] = float(np.mean(values))
    out["lh_min"] = float(np.min(values))
    return out


def add_time_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    categorical_cols: list[str],
    variant: str,
) -> list[str]:
    train_days = pd.to_datetime(train[DAY_COL], errors="raise")
    test_days = pd.to_datetime(test[DAY_COL], errors="raise")
    origin = train_days.min()
    added: list[str] = []

    def add_day_num() -> None:
        name = "decision_day_num_probe"
        train_x[name] = (train_days - origin).dt.days.astype("int32")
        test_x[name] = (test_days - origin).dt.days.astype("int32")
        added.append(name)

    def add_month() -> None:
        name = "decision_month_probe"
        train_x[name] = train_days.dt.to_period("M").astype(str)
        test_x[name] = test_days.dt.to_period("M").astype(str)
        categorical_cols.append(name)
        added.append(name)

    def add_week() -> None:
        name = "decision_week_probe"
        train_x[name] = train_days.dt.isocalendar().week.astype("int16")
        test_x[name] = test_days.dt.isocalendar().week.astype("int16")
        added.append(name)

    variants = {
        "time_all": (add_day_num, add_month, add_week),
        "day_num_only": (add_day_num,),
        "month_only": (add_month,),
        "week_only": (add_week,),
        "no_day_num": (add_month, add_week),
        "no_month": (add_day_num, add_week),
        "no_week": (add_day_num, add_month),
    }
    if variant not in variants:
        raise ValueError(f"Unknown time variant: {variant}")
    for func in variants[variant]:
        func()
    return added


def prepare_xy(cfg: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], list[str]]:
    train_x, test_x, y, categorical_cols, feature_cols, _manifest = prepare_features(
        train=train,
        test=test,
        feature_engineering_cfg=cfg.get("feature_engineering", {"enabled": False}),
        time_features_cfg=cfg.get("time_features", {"enabled": False}),
        excluded_features=cfg.get("excluded_features", []),
    )
    added = add_time_features(train, test, train_x, test_x, categorical_cols, variant)
    train_x, test_x = prepare_xgb_categories(train_x, test_x, categorical_cols)
    return train_x, test_x, y, categorical_cols, feature_cols + added


def fit_predict(
    train_x: pd.DataFrame,
    y: pd.Series,
    valid_idx: np.ndarray,
    train_idx: np.ndarray,
    model_cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, int, float]:
    model = get_model(model_cfg=model_cfg, seed=seed, use_early_stopping=True)
    model.fit(train_x.iloc[train_idx], y.iloc[train_idx], eval_set=[(train_x.iloc[valid_idx], y.iloc[valid_idx])], verbose=False)
    pred = model.predict_proba(train_x.iloc[valid_idx])[:, 1]
    auc = float(roc_auc_score(y.iloc[valid_idx], pred))
    best_iteration = int(getattr(model, "best_iteration", model_cfg.get("n_estimators", 0)) or 0)
    return pred, best_iteration, auc


def evaluate_folds(train: pd.DataFrame, train_x: pd.DataFrame, y: pd.Series, folds: list[dict[str, Any]], model_cfg: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for fold in folds:
        log(f"seed={seed} fold={fold['fold']}")
        train_idx = fold["train_idx"]
        valid_idx = fold["valid_idx"]
        pred, best_iteration, auc = fit_predict(train_x, y, valid_idx, train_idx, model_cfg, seed)
        rows.append({"fold": fold["fold"], "best_iteration": best_iteration, "roc_auc": auc, "valid_rows": int(len(valid_idx))})
        parts.append(pd.DataFrame({"row_index": valid_idx, TARGET: y.iloc[valid_idx].to_numpy(), "prediction": pred, "fold": fold["fold"]}))
    preds = pd.concat(parts, ignore_index=True).sort_values("row_index")
    rows.append({"fold": "OOF_TIME_HOLDOUT", "best_iteration": np.nan, "roc_auc": float(roc_auc_score(preds[TARGET], preds["prediction"])), "valid_rows": int(len(preds))})
    return pd.DataFrame(rows), preds


def evaluate_fold3(train_x: pd.DataFrame, y: pd.Series, folds: list[dict[str, Any]], model_cfg: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold = [item for item in folds if int(item["fold"]) == 3][0]
    valid_idx = fold["valid_idx"]
    pred, best_iteration, auc = fit_predict(train_x, y, valid_idx, fold["train_idx"], model_cfg, seed)
    metrics = pd.DataFrame([{"fold": 3, "best_iteration": best_iteration, "roc_auc": auc, "valid_rows": int(len(valid_idx))}])
    preds = pd.DataFrame({"row_index": valid_idx, TARGET: y.iloc[valid_idx].to_numpy(), "prediction": pred, "fold": 3})
    return metrics, preds


def evaluate_late(train: pd.DataFrame, train_x: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    rows: list[dict[str, Any]] = []
    parts: list[pd.DataFrame] = []
    for holdout in HOLDOUTS:
        cutoff = pd.Timestamp(holdout["cutoff"])
        train_idx = np.where(days < cutoff)[0]
        valid_idx = np.where(days >= cutoff)[0]
        log(f"seed={seed} late={holdout['name']}")
        pred, best_iteration, auc = fit_predict(train_x, y, valid_idx, train_idx, model_cfg, seed)
        rows.append({"holdout": holdout["name"], "best_iteration": best_iteration, "roc_auc": auc, "valid_rows": int(len(valid_idx))})
        parts.append(pd.DataFrame({"holdout": holdout["name"], "row_index": valid_idx, TARGET: y.iloc[valid_idx].to_numpy(), "prediction": pred}))
    metrics = pd.DataFrame(rows)
    metrics = pd.concat(
        [
            metrics,
            pd.DataFrame(
                [
                    {"holdout": "LATE_HOLDOUT_MEAN", "roc_auc": float(metrics["roc_auc"].mean())},
                    {"holdout": "LATE_HOLDOUT_MIN", "roc_auc": float(metrics["roc_auc"].min())},
                ]
            ),
        ],
        ignore_index=True,
    )
    return metrics, pd.concat(parts, ignore_index=True)


def train_full(train_x: pd.DataFrame, test_x: pd.DataFrame, y: pd.Series, model_cfg: dict[str, Any], seed: int) -> pd.Series:
    log(f"seed={seed} full train")
    model = get_model(model_cfg=model_cfg, seed=seed, use_early_stopping=False)
    model.fit(train_x, y, verbose=False)
    return pd.Series(np.clip(model.predict_proba(test_x)[:, 1], 0, 1), name="prediction")


def train_seed(cfg: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, seed: int, stamp: str, skip_if_exists: bool) -> dict[str, str]:
    run_id = f"xgb_hpo_time_regime_multiseed_v1_{stamp}_seed{seed}"
    out_dir = RUN_DIR / run_id
    if out_dir.exists() and skip_if_exists:
        return {
            "run_dir": str(out_dir),
            "valid": str(out_dir / "valid_predictions_time.csv"),
            "late": str(out_dir / "late_holdout_predictions.csv"),
            "test": str(out_dir / "test_predictions.csv"),
        }
    out_dir.mkdir(parents=True, exist_ok=False)
    train_x, test_x, y, categorical_cols, feature_cols = prepare_xy(cfg, train, test, "time_all")
    model_cfg = dict(cfg["model"])
    folds = make_time_folds(train, cfg["cutoffs"])
    fold_metrics, valid = evaluate_folds(train, train_x, y, folds, model_cfg, seed)
    late_metrics, late = evaluate_late(train, train_x, y, model_cfg, seed)
    test_pred = train_full(train_x, test_x, y, model_cfg, seed)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    valid.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    late_metrics.to_csv(out_dir / "late_holdout_metrics.csv", index=False)
    late.to_csv(out_dir / "late_holdout_predictions.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "prediction": test_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    summary = {
        "run_id": run_id,
        "seed": seed,
        "variant": "time_all",
        "feature_count": len(feature_cols),
        "categorical_cols": categorical_cols,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "late_metrics": late_metrics.to_dict(orient="records"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_dir": str(out_dir),
        "valid": str(out_dir / "valid_predictions_time.csv"),
        "late": str(out_dir / "late_holdout_predictions.csv"),
        "test": str(out_dir / "test_predictions.csv"),
    }


def ablate_variant(cfg: dict[str, Any], train: pd.DataFrame, test: pd.DataFrame, variant: str, seed: int, stamp: str) -> dict[str, Any]:
    out_dir = REPORT_DIR / f"ablation_{variant}_{stamp}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=False)
    train_x, _test_x, y, categorical_cols, feature_cols = prepare_xy(cfg, train, test, variant)
    model_cfg = dict(cfg["model"])
    folds = make_time_folds(train, cfg["cutoffs"])
    fold3_metrics, fold3 = evaluate_fold3(train_x, y, folds, model_cfg, seed)
    late_metrics, late = evaluate_late(train, train_x, y, model_cfg, seed)
    fold3.to_csv(out_dir / "fold3_predictions.csv", index=False)
    late.to_csv(out_dir / "late_holdout_predictions.csv", index=False)
    fold3_metrics.to_csv(out_dir / "fold3_metrics.csv", index=False)
    late_metrics.to_csv(out_dir / "late_holdout_metrics.csv", index=False)
    late_values = late_metrics[late_metrics["holdout"].isin(HOLDOUT_NAMES)]["roc_auc"].astype(float)
    return {
        "variant": variant,
        "seed": seed,
        "feature_count": len(feature_cols),
        "categorical_cols": categorical_cols,
        "fold3_auc": float(fold3_metrics.loc[0, "roc_auc"]),
        "lh_mean": float(late_values.mean()),
        "lh_min": float(late_values.min()),
        "artifacts": str(out_dir),
    }


def load_valid(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"prediction": label})
    return df[["row_index", TARGET, "fold", label]]


def load_late(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"prediction": label})
    return df[["holdout", "row_index", TARGET, label]]


def average_seed_oof(seed_artifacts: dict[int, dict[str, str]]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    seed_cols: list[str] = []
    for seed, artifacts in sorted(seed_artifacts.items()):
        col = f"time_seed{seed}"
        df = load_valid(Path(artifacts["valid"]), col)
        if base is None:
            base = df[["row_index", TARGET, "fold"]].copy()
        base = base.merge(df, on=["row_index", TARGET, "fold"], how="inner", validate="one_to_one")
        base[f"{col}_rank"] = base.groupby("fold")[col].transform(rank_normalize)
        seed_cols.append(f"{col}_rank")
    if base is None:
        raise ValueError("No seed artifacts")
    base["time_multiseed"] = base[seed_cols].mean(axis=1)
    return base


def average_seed_late(seed_artifacts: dict[int, dict[str, str]]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    seed_cols: list[str] = []
    for seed, artifacts in sorted(seed_artifacts.items()):
        col = f"time_seed{seed}"
        df = load_late(Path(artifacts["late"]), col)
        if base is None:
            base = df[["holdout", "row_index", TARGET]].copy()
        base = base.merge(df, on=["holdout", "row_index", TARGET], how="inner", validate="one_to_one")
        base[f"{col}_rank"] = base.groupby("holdout")[col].transform(rank_normalize)
        seed_cols.append(f"{col}_rank")
    if base is None:
        raise ValueError("No seed artifacts")
    base["time_multiseed"] = base[seed_cols].mean(axis=1)
    return base


def load_champion_oof() -> pd.DataFrame:
    df = load_valid(CHAMPION_RUN / "valid_predictions_time.csv", "champion")
    df["champion_rank"] = df.groupby("fold")["champion"].transform(rank_normalize)
    return df


def load_champion_late() -> pd.DataFrame:
    base: pd.DataFrame | None = None
    for label, (path, weight) in CHAMP_COMPONENTS.items():
        comp = load_late(path, label)
        if base is None:
            base = comp[["holdout", "row_index", TARGET]].copy()
        base[label] = comp[label].to_numpy(dtype=float)
    if base is None:
        raise ValueError("No champion late components")
    base["champion"] = sum(base[label] * weight for label, (_path, weight) in CHAMP_COMPONENTS.items())
    base["champion_rank"] = base.groupby("holdout")["champion"].transform(rank_normalize)
    return base


def scan_champion_time(seed_artifacts: dict[int, dict[str, str]], step: float) -> pd.DataFrame:
    oof = load_champion_oof().merge(average_seed_oof(seed_artifacts)[["row_index", TARGET, "fold", "time_multiseed"]], on=["row_index", TARGET, "fold"], how="inner")
    late = load_champion_late().merge(average_seed_late(seed_artifacts)[["holdout", "row_index", TARGET, "time_multiseed"]], on=["holdout", "row_index", TARGET], how="inner")
    rows: list[dict[str, Any]] = []
    for w_time in np.round(np.arange(0, 1 + 1e-9, step), 10):
        w_champ = 1.0 - float(w_time)
        oof["blend"] = w_champ * oof["champion_rank"] + float(w_time) * oof["time_multiseed"]
        late["blend"] = w_champ * late["champion_rank"] + float(w_time) * late["time_multiseed"]
        row = {"w_champion": w_champ, "w_time_multiseed": float(w_time), **auc_by_fold(oof, "blend"), **auc_by_holdout(late, "blend")}
        row["selection_score"] = row["lh_mean"] + 0.25 * row["fold3_auc"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["lh_mean", "lh_min", "fold3_auc", "oof_auc"], ascending=False)


def build_submission(seed_artifacts: dict[int, dict[str, str]], best: dict[str, Any], label: str, scan_path: Path) -> tuple[str, str]:
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID_COL])[ID_COL]
    champion = pd.read_csv(CHAMPION_RUN / "test_predictions.csv").set_index(ID_COL)["prediction"].reindex(test_ids.to_numpy())
    if champion.isna().any():
        raise ValueError("Missing champion test predictions")
    seed_ranks: list[np.ndarray] = []
    for seed, artifacts in sorted(seed_artifacts.items()):
        pred = pd.read_csv(artifacts["test"]).set_index(ID_COL)["prediction"].reindex(test_ids.to_numpy())
        if pred.isna().any():
            raise ValueError(f"Missing test predictions for seed={seed}")
        seed_ranks.append(rank_normalize(pred).to_numpy(dtype=float))
    time_avg = np.mean(seed_ranks, axis=0)
    blend = float(best["w_champion"]) * rank_normalize(champion).to_numpy(dtype=float) + float(best["w_time_multiseed"]) * time_avg
    out = pd.DataFrame({ID_COL: test_ids.to_numpy(), TARGET: blend})
    if len(out) != len(test_ids) or list(out[ID_COL]) != list(test_ids):
        raise ValueError("Submission alignment failed")
    if not out[TARGET].between(0, 1).all():
        raise ValueError("Submission values outside [0,1]")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = UPLOAD_DIR / f"{label}.csv"
    out.to_csv(out_path, index=False, float_format="%.10f")
    digest = sha(out_path)
    card = {
        "file_path": str(out_path),
        "sha256": digest,
        "format": "test-only, 36311 rows, front_id,target_value, order as test_apps.csv, target_value %.10f",
        "generation": {
            "script": "scripts/scan_xgb_time_regime_multiseed.py",
            "blend": f"champion_rank*{best['w_champion']:.2f} + xgb_time_regime_multiseed_rank*{best['w_time_multiseed']:.2f}",
            "seed_artifacts": seed_artifacts,
        },
        "selection": {"method": "offline Fold3 + late-holdout scan; no leaderboard", "scan_csv": str(scan_path), "best_row": best},
        "submission_checks": {
            "rows": int(len(out)),
            "columns": [ID_COL, TARGET],
            "front_id_order_matches_test_apps": True,
            "front_id_unique": bool(out[ID_COL].is_unique),
            "probability_range_ok": True,
            "nan_count": int(out[TARGET].isna().sum()),
            "unique_predictions": int(out[TARGET].nunique()),
        },
        "known_risks": [
            "Time-regime features may extrapolate poorly to unseen test months.",
            "Weights selected offline on late-holdout diagnostics, not leaderboard.",
            "Rank averaging optimizes ordering, not calibration.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    card_path = CARDS_DIR / f"{label}_card.json"
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path), digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--include-existing-seed42", action="store_true")
    parser.add_argument("--ablation-seed", type=int, default=42)
    parser.add_argument("--ablation-variants", default="day_num_only,month_only,week_only,no_day_num,no_month,no_week")
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--skip-if-exists", action="store_true")
    args = parser.parse_args()

    cfg = load_config(CONFIG)
    train_path = resolve_path(cfg["train"], REPO)
    test_path = resolve_path(cfg["test"], REPO)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    seed_artifacts: dict[int, dict[str, str]] = {}
    if args.include_existing_seed42:
        seed_artifacts[42] = {"valid": str(SEED42_VALID), "late": str(SEED42_LATE), "test": str(SEED42_TEST), "run_dir": str(SEED42_TEST.parent)}

    for seed in [int(item) for item in args.seeds.split(",") if item.strip()]:
        cfg["seed"] = seed
        seed_artifacts[seed] = train_seed(cfg, train, test, seed, stamp, args.skip_if_exists)

    scan = scan_champion_time(seed_artifacts, args.step)
    scan_path = REPORT_DIR / f"multiseed_champion_time_scan_{stamp}.csv"
    scan.to_csv(scan_path, index=False)

    ablation_rows = []
    for variant in [item.strip() for item in args.ablation_variants.split(",") if item.strip()]:
        log(f"ablation variant={variant}")
        ablation_rows.append(ablate_variant(cfg, train, test, variant, args.ablation_seed, stamp))
    ablation = pd.DataFrame(ablation_rows).sort_values(["lh_mean", "lh_min", "fold3_auc"], ascending=False)
    ablation_path = REPORT_DIR / f"time_regime_ablation_{stamp}.csv"
    ablation.to_csv(ablation_path, index=False)

    best = scan.iloc[0].to_dict()
    built = None
    if not args.no_build:
        label = args.label or f"candidate_20260620_upload5_RETEST_xgb_time_regime_multiseed_c{int(round(best['w_champion'] * 100)):02d}_t{int(round(best['w_time_multiseed'] * 100)):02d}"
        out_path, digest = build_submission(seed_artifacts, best, label, scan_path)
        built = {"submission": out_path, "sha256": digest}

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L4_multiseed_and_ablation_diagnostic",
        "train_sha256": sha256_file(train_path),
        "test_sha256": sha256_file(test_path),
        "seed_artifacts": seed_artifacts,
        "scan_csv": str(scan_path),
        "ablation_csv": str(ablation_path),
        "best": best,
        "built": built,
        "baseline_upload3": {"fold3": 0.7571, "lh_mean": 0.7665, "lh_min": 0.7579},
        "risk": "Offline diagnostics only; no leaderboard probing.",
    }
    report_path = REPORT_DIR / f"time_regime_multiseed_report_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== multiseed scan top ==")
    print(scan.head(12).to_string(index=False))
    print("\n== ablation ==")
    print(ablation.to_string(index=False))
    print(f"\nreport: {report_path.relative_to(REPO)}")
    if built:
        print(f"submission: {Path(built['submission']).relative_to(REPO)}")
        print(f"sha256: {built['sha256']}")


if __name__ == "__main__":
    main()
