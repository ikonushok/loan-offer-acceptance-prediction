#!/usr/bin/env python3
"""Scan 3-way rank blends: champion + XGB HPO + XGB time-regime.

No model training and no leaderboard signal. Selection uses aligned rolling-fold
OOF predictions plus late-holdout predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_COL = "front_id"
TARGET_COL = "target_value"

CHAMPION_RUN = REPO_ROOT / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z"
XGB_HPO_RUN = REPO_ROOT / "experiments/runs/xgb_hpo_depth3_child80_reg30_v1_20260617T235100Z_seed42"
TIME_REGIME_VALID = REPO_ROOT / "reports/validation/feature_family_probes/feature_probe_time_regime_20260619T041432Z_valid_predictions.csv"
TIME_REGIME_RUN = REPO_ROOT / "experiments/runs/xgb_hpo_time_regime_probe_full_20260619T0420Z_seed42"
TEST_PATH = REPO_ROOT / "data/raw/test_apps.csv"

LH = REPO_ROOT / "reports/validation/late_holdouts"
HOLDOUT_NAMES = [
    "H1_2025_03_01_to_end",
    "H2_2025_04_01_to_end",
    "H3_2025_05_01_to_end",
]
CHAMP_COMPONENTS = {
    "t070": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0070_20260617T130617Z" / "late_holdout_predictions.csv", 0.70),
    "t041": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0041_20260617T131136Z" / "late_holdout_predictions.csv", 0.04),
    "t164": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0164_20260617T131519Z" / "late_holdout_predictions.csv", 0.26),
}
XGB_HPO_LATE = LH / "late_holdouts_xgb_hpo_depth3_child80_reg30_v1_20260617T235308Z" / "late_holdout_predictions.csv"
TIME_REGIME_LATE = LH / "late_holdouts_xgb_hpo_time_regime_probe_20260619T041911Z" / "late_holdout_predictions.csv"

REPORT_DIR = REPO_ROOT / "reports/validation/time_regime_three_way"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
UPLOAD_DIR = SUBMISSIONS_DIR / "upload_20260620"
CARDS_DIR = SUBMISSIONS_DIR / "cards"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_normalize(series: pd.Series) -> pd.Series:
    return series.rank(method="average") / len(series)


def load_valid(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"prediction": label})
    return df[["row_index", TARGET_COL, "fold", label]]


def load_late(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"prediction": label})
    return df[["holdout", "row_index", TARGET_COL, label]]


def auc_by_fold(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    out = {}
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        out[f"fold{fold}_auc"] = float(roc_auc_score(sub[TARGET_COL], sub[score_col]))
    out["oof_auc"] = float(roc_auc_score(df[TARGET_COL], df[score_col]))
    return out


def auc_by_holdout(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    out = {}
    values = []
    for holdout in HOLDOUT_NAMES:
        sub = df[df["holdout"] == holdout]
        auc = float(roc_auc_score(sub[TARGET_COL], sub[score_col]))
        out[holdout] = auc
        values.append(auc)
    out["lh_mean"] = float(np.mean(values))
    out["lh_min"] = float(np.min(values))
    return out


def build_oof_frame() -> pd.DataFrame:
    df = load_valid(CHAMPION_RUN / "valid_predictions_time.csv", "champion")
    df = df.merge(load_valid(XGB_HPO_RUN / "valid_predictions_time.csv", "xgb_hpo"), on=["row_index", TARGET_COL, "fold"], how="inner", validate="one_to_one")
    df = df.merge(load_valid(TIME_REGIME_VALID, "xgb_time_regime"), on=["row_index", TARGET_COL, "fold"], how="inner", validate="one_to_one")
    for col in ["champion", "xgb_hpo", "xgb_time_regime"]:
        df[f"{col}_rank"] = df.groupby("fold")[col].transform(rank_normalize)
    return df


def build_late_frame() -> pd.DataFrame:
    base = None
    for label, (path, weight) in CHAMP_COMPONENTS.items():
        comp = load_late(path, label)
        if base is None:
            base = comp[["holdout", "row_index", TARGET_COL]].copy()
        base[label] = comp[label].to_numpy(dtype=float)
    if base is None:
        raise ValueError("No champion late-holdout components configured")
    base["champion"] = sum(base[label] * weight for label, (_path, weight) in CHAMP_COMPONENTS.items())
    base = base.merge(load_late(XGB_HPO_LATE, "xgb_hpo")[["holdout", "row_index", "xgb_hpo"]], on=["holdout", "row_index"], how="inner", validate="one_to_one")
    base = base.merge(load_late(TIME_REGIME_LATE, "xgb_time_regime")[["holdout", "row_index", "xgb_time_regime"]], on=["holdout", "row_index"], how="inner", validate="one_to_one")
    for col in ["champion", "xgb_hpo", "xgb_time_regime"]:
        base[f"{col}_rank"] = base.groupby("holdout")[col].transform(rank_normalize)
    return base


def score_weights(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    w_champion, w_xgb, w_time = weights
    return (
        w_champion * df["champion_rank"]
        + w_xgb * df["xgb_hpo_rank"]
        + w_time * df["xgb_time_regime_rank"]
    )


def scan(oof: pd.DataFrame, late: pd.DataFrame, step: float) -> pd.DataFrame:
    rows = []
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    for w_champion in grid:
        for w_xgb in grid:
            w_time = 1.0 - float(w_champion) - float(w_xgb)
            if w_time < -1e-9:
                continue
            w_time = max(0.0, w_time)
            weights = (float(w_champion), float(w_xgb), float(w_time))
            oof_score = score_weights(oof, weights)
            late_score = score_weights(late, weights)
            oof_tmp = oof.assign(blend=oof_score)
            late_tmp = late.assign(blend=late_score)
            row = {
                "w_champion": weights[0],
                "w_xgb_hpo": weights[1],
                "w_xgb_time_regime": weights[2],
                **auc_by_fold(oof_tmp, "blend"),
                **auc_by_holdout(late_tmp, "blend"),
            }
            row["fold_mean"] = float(np.mean([row["fold1_auc"], row["fold2_auc"], row["fold3_auc"]]))
            row["selection_score"] = row["lh_mean"] + 0.25 * row["fold3_auc"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_submission(weights: tuple[float, float, float], label: str, best_row: dict, scan_path: Path) -> tuple[Path, Path, str]:
    test_ids = pd.read_csv(TEST_PATH, usecols=[ID_COL])[ID_COL]
    parts = []
    for path, col in [
        (CHAMPION_RUN / "test_predictions.csv", "champion"),
        (XGB_HPO_RUN / "test_predictions.csv", "xgb_hpo"),
        (TIME_REGIME_RUN / "test_predictions.csv", "xgb_time_regime"),
    ]:
        pred = pd.read_csv(path).set_index(ID_COL)["prediction"].reindex(test_ids.to_numpy())
        if pred.isna().any():
            raise ValueError(f"Missing predictions after test alignment: {path}")
        parts.append(rank_normalize(pred).to_numpy(dtype=float))
    blend = weights[0] * parts[0] + weights[1] * parts[1] + weights[2] * parts[2]
    sub = pd.DataFrame({ID_COL: test_ids.to_numpy(), TARGET_COL: blend})
    if len(sub) != len(test_ids):
        raise ValueError("Submission row count mismatch")
    if list(sub[ID_COL]) != list(test_ids):
        raise ValueError("Submission order mismatch")
    if not sub[TARGET_COL].between(0, 1).all():
        raise ValueError("Submission predictions out of [0,1]")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = UPLOAD_DIR / f"{label}.csv"
    sub.to_csv(out_csv, index=False, float_format="%.10f")
    digest = sha256_file(out_csv)
    card = {
        "file_path": str(out_csv),
        "sha256": digest,
        "format": "test-only, 36311 rows, front_id,target_value, order as test_apps.csv, target_value %.10f",
        "generation": {
            "script": "scripts/scan_time_regime_three_way_blend.py",
            "blend": (
                f"champion_rank*{weights[0]:.2f} + "
                f"xgb_hpo_rank*{weights[1]:.2f} + "
                f"xgb_time_regime_rank*{weights[2]:.2f}"
            ),
            "source_runs": {
                "champion": str(CHAMPION_RUN),
                "xgb_hpo": str(XGB_HPO_RUN),
                "xgb_time_regime": str(TIME_REGIME_RUN),
            },
            "weights": {
                "champion_rank": round(weights[0], 4),
                "xgb_hpo_rank": round(weights[1], 4),
                "xgb_time_regime_rank": round(weights[2], 4),
            },
        },
        "selection": {
            "method": "offline 3-way rank-blend scan on Fold3 + late-holdout; no leaderboard",
            "scan_csv": str(scan_path),
            "best_row": best_row,
        },
        "submission_checks": {
            "rows": int(len(sub)),
            "columns": [ID_COL, TARGET_COL],
            "front_id_order_matches_test_apps": True,
            "front_id_unique": bool(sub[ID_COL].is_unique),
            "probability_range_ok": True,
            "nan_count": int(sub[TARGET_COL].isna().sum()),
            "unique_predictions": int(sub[TARGET_COL].nunique()),
        },
        "known_risks": [
            "Time-regime features may overfit validation periods and extrapolate poorly to unseen test months.",
            "Blend weights selected offline by late-holdout scan, not by leaderboard.",
            "Rank transform optimizes ordering and discards calibration scale; acceptable for ROC-AUC.",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    card_path = CARDS_DIR / f"{label}_card.json"
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_csv, card_path, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    oof = build_oof_frame()
    late = build_late_frame()
    results = scan(oof, late, step=args.step)
    results = results.sort_values(["lh_mean", "lh_min", "fold3_auc", "oof_auc"], ascending=False)
    scan_path = REPORT_DIR / f"time_regime_three_way_scan_{stamp}.csv"
    results.to_csv(scan_path, index=False)

    correlations = []
    for scope, df, group_col in [("OOF", oof, "fold"), ("LATE", late, "holdout")]:
        for group, sub in df.groupby(group_col):
            for left, right in [
                ("champion", "xgb_hpo"),
                ("champion", "xgb_time_regime"),
                ("xgb_hpo", "xgb_time_regime"),
            ]:
                correlations.append(
                    {
                        "scope": scope,
                        "group": group,
                        "left": left,
                        "right": right,
                        "pearson": sub[left].corr(sub[right], method="pearson"),
                        "spearman": sub[left].corr(sub[right], method="spearman"),
                    }
                )
    corr_path = REPORT_DIR / f"time_regime_three_way_correlations_{stamp}.csv"
    pd.DataFrame(correlations).to_csv(corr_path, index=False)

    best = results.iloc[0].to_dict()
    weights = (best["w_champion"], best["w_xgb_hpo"], best["w_xgb_time_regime"])
    label = args.label or (
        "candidate_20260620_upload4_RETEST_three_way_"
        f"c{int(round(weights[0] * 100)):02d}_x{int(round(weights[1] * 100)):02d}_t{int(round(weights[2] * 100)):02d}"
    )
    built = None
    if not args.no_build:
        out_csv, card_path, digest = build_submission(weights, label, best, scan_path)
        built = {"submission": str(out_csv), "card": str(card_path), "sha256": digest}

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scan_step": args.step,
        "scan_csv": str(scan_path),
        "correlations_csv": str(corr_path),
        "best_by_lh_mean": best,
        "built": built,
        "note": "Offline Fold3 + late-holdout scan only; no leaderboard signal used.",
    }
    report_path = REPORT_DIR / f"time_regime_three_way_report_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== Top 15 by lh_mean ==")
    print(results.head(15).to_string(index=False))
    print(f"\nscan: {scan_path.relative_to(REPO_ROOT)}")
    print(f"corr: {corr_path.relative_to(REPO_ROOT)}")
    print(f"report: {report_path.relative_to(REPO_ROOT)}")
    if built:
        print(f"submission: {Path(built['submission']).relative_to(REPO_ROOT)}")
        print(f"card: {Path(built['card']).relative_to(REPO_ROOT)}")
        print(f"sha256: {built['sha256']}")


if __name__ == "__main__":
    main()
