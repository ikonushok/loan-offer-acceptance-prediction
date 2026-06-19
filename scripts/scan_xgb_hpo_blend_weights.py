#!/usr/bin/env python
"""Scan HPO-XGB rank-blend weights against the champion on the late-holdout battery.

Confirmed lever (public 2026-06-19): champion_rank×(1-w) + xgb_hpo_rank×w.
  c62_x38 (w=0.38, untuned XGB) -> 76.362
  c51_x49 (w=0.49, HPO XGB)     -> 76.388 (best)
Offline order predicts public order, so we scan w upward and gate on Fold3+lh_mean
WITHOUT using leaderboard scores.

Does NOT retrain. Reads late-holdout predictions for champion components and the
HPO-XGB run. numpy + pandas only.

Usage:
    python3.13 scripts/scan_xgb_hpo_blend_weights.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LH = REPO_ROOT / "reports" / "validation" / "late_holdouts"
REPORTS_DIR = REPO_ROOT / "reports" / "validation"

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
XGB_HPO_PATH = LH / "late_holdouts_xgb_hpo_depth3_child80_reg30_v1_20260617T235308Z" / "late_holdout_predictions.csv"

WEIGHTS = [0.49, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="stable")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    ties = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(score_sorted):
        j = i
        while j < len(score_sorted) and score_sorted[j] == score_sorted[i]:
            j += 1
        ties[i:j] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = ties[y_sorted.astype(bool)].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rank_normalize(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=np.float64)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return pd.Series(ranks / len(arr), index=series.index)


def load_component(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"prediction": label})
    return df[["holdout", "row_index", "target_value", label]]


def auc_by_holdout(df: pd.DataFrame, col: str) -> dict:
    aucs = []
    out = {}
    for h in HOLDOUT_NAMES:
        sub = df[df["holdout"] == h]
        a = roc_auc(sub["target_value"].to_numpy(), sub[col].to_numpy())
        out[h] = a
        aucs.append(a)
    out["LATE_HOLDOUT_MEAN"] = float(np.mean(aucs))
    out["LATE_HOLDOUT_MIN"] = float(np.min(aucs))
    return out


def main() -> None:
    base = None
    for label, (path, _w) in CHAMP_COMPONENTS.items():
        comp = load_component(path, label)
        if base is None:
            base = comp[["holdout", "row_index", "target_value"]].copy()
        base[label] = comp[label].values
    base["champion"] = base["t070"] * 0.70 + base["t041"] * 0.04 + base["t164"] * 0.26

    xgb = load_component(XGB_HPO_PATH, "xgb_hpo")
    base = base.merge(xgb[["holdout", "row_index", "xgb_hpo"]], on=["holdout", "row_index"], how="inner")

    champ_auc = auc_by_holdout(base, "champion")

    # precompute per-holdout ranks once
    parts = []
    for h in HOLDOUT_NAMES:
        sub = base[base["holdout"] == h].copy()
        sub["c_rank"] = rank_normalize(sub["champion"])
        sub["x_rank"] = rank_normalize(sub["xgb_hpo"])
        parts.append(sub)
    ranked = pd.concat(parts, ignore_index=True)

    print("=" * 74)
    print("HPO-XGB rank-blend weight scan  (champion_rank*(1-w) + xgb_hpo_rank*w)")
    print("=" * 74)
    print(f"champion baseline: lh_mean={champ_auc['LATE_HOLDOUT_MEAN']:.6f} lh_min={champ_auc['LATE_HOLDOUT_MIN']:.6f}")
    print(f"{'w_xgb':>6} {'H1':>9} {'H2':>9} {'H3':>9} {'lh_mean':>9} {'lh_min':>9} {'Δmean':>9}")

    results = []
    for w in WEIGHTS:
        ranked["blend"] = (1 - w) * ranked["c_rank"] + w * ranked["x_rank"]
        a = auc_by_holdout(ranked, "blend")
        dmean = a["LATE_HOLDOUT_MEAN"] - champ_auc["LATE_HOLDOUT_MEAN"]
        results.append({"w_xgb": w, **a, "delta_mean_vs_champion": dmean})
        print(f"{w:>6.2f} {a[HOLDOUT_NAMES[0]]:>9.6f} {a[HOLDOUT_NAMES[1]]:>9.6f} "
              f"{a[HOLDOUT_NAMES[2]]:>9.6f} {a['LATE_HOLDOUT_MEAN']:>9.6f} "
              f"{a['LATE_HOLDOUT_MIN']:>9.6f} {dmean:>+9.5f}")

    best = max(results, key=lambda r: r["LATE_HOLDOUT_MEAN"])
    print("-" * 74)
    print(f"BEST by lh_mean: w_xgb={best['w_xgb']:.2f}  lh_mean={best['LATE_HOLDOUT_MEAN']:.6f}  lh_min={best['LATE_HOLDOUT_MIN']:.6f}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = REPORTS_DIR / f"xgb_hpo_blend_weight_scan_{stamp}.json"
    out_path.write_text(json.dumps({
        "champion_late_holdout": champ_auc,
        "scan": results,
        "best": best,
        "note": "Offline late-holdout scan; public order matches offline order. No LB used.",
    }, indent=2))
    print(f"saved: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
