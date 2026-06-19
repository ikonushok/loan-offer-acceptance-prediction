#!/usr/bin/env python
"""Точный скан весов трёхстороннего бленда: champion / catboost_no_month / xgb_no_month.

Все три rank-нормализуются на каждом late-holdout, веса неотрицательны, сумма = 1.
Отбор по late-holdout (lh_mean основной, lh_min гейт). Без LB-probing.

Усиление offline→public ~27× (подтверждено no_month 76.505).

Usage:
    PY=/opt/homebrew/Caskroom/miniforge/base/bin/python3
    $PY scripts/scan_no_month_three_way_blend.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
LH = REPO_ROOT / "reports/validation/late_holdouts"
HOLD = ["H1_2025_03_01_to_end", "H2_2025_04_01_to_end", "H3_2025_05_01_to_end"]

CHAMP = {
    "t070": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0070_20260617T130617Z" / "late_holdout_predictions.csv", 0.70),
    "t041": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0041_20260617T131136Z" / "late_holdout_predictions.csv", 0.04),
    "t164": (LH / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0164_20260617T131519Z" / "late_holdout_predictions.csv", 0.26),
}
XGB_NM = REPO_ROOT / "experiments/runs/xgb_hpo_time_regime_no_month_v1_20260619T055715Z_seed42/late_holdout_predictions.csv"
CAT_NM = LH / "late_holdouts_catboost_trial0070_no_month_v1_20260619T212856Z/late_holdout_predictions.csv"


def load(p, name):
    df = pd.read_csv(p).rename(columns={"prediction": name})
    return df[["holdout", "row_index", "target_value", name]]


def rk(s):
    a = s.to_numpy(dtype=np.float64)
    o = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[o] = np.arange(1, len(a) + 1)
    return r / len(a)


def main():
    base = None
    for name, (p, _w) in CHAMP.items():
        c = load(p, name)
        base = c if base is None else base.merge(c, on=["holdout", "row_index", "target_value"])
    base = base.merge(load(XGB_NM, "xgb_nm"), on=["holdout", "row_index", "target_value"])
    base = base.merge(load(CAT_NM, "cat_nm"), on=["holdout", "row_index", "target_value"])
    base["champion"] = base["t070"] * 0.70 + base["t041"] * 0.04 + base["t164"] * 0.26

    # precompute per-holdout ranks
    parts = []
    for h in HOLD:
        sub = base[base.holdout == h].copy()
        sub["r_champ"] = rk(sub["champion"])
        sub["r_xgb"] = rk(sub["xgb_nm"])
        sub["r_cat"] = rk(sub["cat_nm"])
        parts.append(sub)
    R = pd.concat(parts, ignore_index=True)

    def blend_lh(wc, wcat, wx):
        aucs = []
        for h in HOLD:
            sub = R[R.holdout == h]
            bl = wc * sub["r_champ"] + wcat * sub["r_cat"] + wx * sub["r_xgb"]
            aucs.append(roc_auc_score(sub["target_value"], bl))
        return float(np.mean(aucs)), float(np.min(aucs)), aucs

    # grid: champion + cat + xgb, шаг 0.05
    results = []
    grid = np.arange(0.0, 1.0001, 0.05)
    for wx in grid:
        for wcat in grid:
            wc = 1.0 - wx - wcat
            if wc < -1e-9 or wc > 1.0 + 1e-9:
                continue
            wc = max(0.0, wc)
            mn, mi, a = blend_lh(wc, wcat, wx)
            results.append({"w_champ": round(wc, 2), "w_cat": round(wcat, 2), "w_xgb": round(wx, 2),
                            "lh_mean": mn, "lh_min": mi, "H1": a[0], "H2": a[1], "H3": a[2]})

    df = pd.DataFrame(results)
    # текущий best A для референса
    a_mn, a_mi, _ = blend_lh(0.20, 0.0, 0.80)
    print(f"Референс A (champion*0.20 + xgb_nm*0.80, public 76.505): lh_mean={a_mn:.6f} lh_min={a_mi:.6f}\n")

    best_mean = df.sort_values("lh_mean", ascending=False).head(8)
    print("ТОП-8 по lh_mean:")
    print(best_mean[["w_champ", "w_cat", "w_xgb", "lh_mean", "lh_min", "H1", "H2", "H3"]].to_string(index=False))

    # гейт: lh_min не ниже текущего best A
    gated = df[df["lh_min"] >= a_mi - 1e-6].sort_values("lh_mean", ascending=False)
    print(f"\nТОП-8 по lh_mean ПРИ lh_min >= {a_mi:.6f} (не хуже A):")
    print(gated[["w_champ", "w_cat", "w_xgb", "lh_mean", "lh_min", "H1", "H2", "H3"]].head(8).to_string(index=False))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "reports/validation" / f"no_month_three_way_scan_{stamp}.csv"
    df.to_csv(out, index=False)

    best_row = gated.iloc[0] if len(gated) else best_mean.iloc[0]
    rep = {
        "reference_A": {"w_champ": 0.20, "w_cat": 0.0, "w_xgb": 0.80, "lh_mean": a_mn, "lh_min": a_mi, "public": 76.505},
        "best_gated": best_row[["w_champ", "w_cat", "w_xgb", "lh_mean", "lh_min"]].to_dict(),
        "scan_csv": str(out.relative_to(REPO_ROOT)),
        "amplification_note": "offline lh_mean -> public ~27x (no_month подтвердил).",
    }
    (REPO_ROOT / "reports/validation" / f"no_month_three_way_scan_{stamp}.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nАртефакт: {out.relative_to(REPO_ROOT)}")
    print(f"Рекомендован (гейт lh_min): w_champ={best_row['w_champ']} w_cat={best_row['w_cat']} w_xgb={best_row['w_xgb']} lh_mean={best_row['lh_mean']:.6f} lh_min={best_row['lh_min']:.6f}")


if __name__ == "__main__":
    main()
