#!/usr/bin/env python
"""Constrained blend builder with Fold3 + late-holdout gate.

Scans blend weights for available models, selects the best combination
based on Fold3 AUC (primary) and late-holdout mean (secondary gate).
Does NOT use public leaderboard scores for weight selection.

Usage:
    python scripts/build_constrained_blend.py

Reads OOF predictions from experiments/runs/ and late-holdout predictions from
reports/validation/late_holdouts/. Saves blend report and submission CSV.

Gate rules (configurable):
  - Fold3 AUC must not degrade vs current champion
  - Late-holdout mean must not degrade by more than LATE_HOLDOUT_DEGRADATION_TOLERANCE
  - Late-holdout min must not degrade by more than LATE_HOLDOUT_MIN_TOLERANCE
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
LATE_HOLDOUTS_BASE = REPO_ROOT / "reports" / "validation" / "late_holdouts"
REPORTS_DIR = REPO_ROOT / "reports" / "validation"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
CARDS_DIR = SUBMISSIONS_DIR / "cards"

# ── Gate config ───────────────────────────────────────────────────────────────
FOLD3_DEGRADATION_TOLERANCE = -0.0002   # allow tiny rounding noise
LATE_HOLDOUT_MEAN_TOLERANCE = -0.001    # late-holdout mean must not drop by more
LATE_HOLDOUT_MIN_TOLERANCE = -0.002     # late-holdout min must not drop by more

# ── Current champion ──────────────────────────────────────────────────────────
CHAMPION_RUN = "blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z"
CHAMPION_PUBLIC_AUC = 0.76054

# ── Component definitions ──────────────────────────────────────────────────────
# Add new candidates here as they become available.
# Each entry: key -> (run_dir_name, late_holdout_dir_prefix_or_None, blend_mode)
# blend_mode: "raw" | "rank"
COMPONENTS: dict[str, dict] = {
    "champion": {
        "run": CHAMPION_RUN,
        "late_holdout_prefix": None,  # computed from component parts
        "blend_mode": "raw",
        "note": "Current champion: t070×0.70 + t041×0.04 + t164×0.26",
    },
    "xgb_unweighted": {
        "run": "xgb_context_offer_unweighted_v1_20260617T232530Z_seed42",
        "late_holdout_prefix": "late_holdouts_xgb_context_offer_unweighted_v1_20260617T235004Z",
        "blend_mode": "raw",
        "note": "XGBoost unweighted raw blend, corr=0.880 vs champion",
    },
    "xgb_unweighted_rank": {
        "run": "xgb_context_offer_unweighted_v1_20260617T232530Z_seed42",
        "late_holdout_prefix": "late_holdouts_xgb_context_offer_unweighted_v1_20260617T235004Z",
        "blend_mode": "rank",
        "note": "XGBoost unweighted rank blend; ablation best was c62_x38",
    },
    # These will be available after running lgbm late holdouts + et/rf:
    # "lgbm_unbalanced": { ... },
    # "et_v1": { ... },
    # "rf_v1": { ... },
    # "lgbm_optuna_best": { ... },  # after LGBM HPO
}

# ── Late holdout champion components ─────────────────────────────────────────
CHAMP_LATE_HOLDOUT_COMPONENTS = {
    "t070": (
        LATE_HOLDOUTS_BASE
        / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0070_20260617T130617Z"
        / "late_holdout_predictions.csv",
        0.70,
    ),
    "t041": (
        LATE_HOLDOUTS_BASE
        / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0041_20260617T131136Z"
        / "late_holdout_predictions.csv",
        0.04,
    ),
    "t164": (
        LATE_HOLDOUTS_BASE
        / "late_holdouts_catboost_optuna_context_offer_revalidated_trial0164_20260617T131519Z"
        / "late_holdout_predictions.csv",
        0.26,
    ),
}

HOLDOUT_NAMES = [
    "H1_2025_03_01_to_end",
    "H2_2025_04_01_to_end",
    "H3_2025_05_01_to_end",
]


# ── AUC (no sklearn) ─────────────────────────────────────────────────────────

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


def rank_normalize_per_fold(df: pd.DataFrame, pred_col: str, fold_col: str = "fold") -> pd.Series:
    result = pd.Series(index=df.index, dtype=np.float64)
    for fold_val in df[fold_col].unique():
        mask = df[fold_col] == fold_val
        arr = df.loc[mask, pred_col].to_numpy(dtype=np.float64)
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
        result.loc[df.index[mask]] = ranks / len(arr)
    return result


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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_oof(run_dir_name: str) -> pd.DataFrame:
    path = RUNS_DIR / run_dir_name / "valid_predictions_time.csv"
    if not path.exists():
        raise FileNotFoundError(f"OOF not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"prediction": run_dir_name})
    # Normalise column names
    if "row_index" not in df.columns:
        df["row_index"] = np.arange(len(df))
    return df[["row_index", "fold", "target_value", run_dir_name]]


def load_test_preds(run_dir_name: str) -> pd.Series:
    path = RUNS_DIR / run_dir_name / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test predictions not found: {path}")
    df = pd.read_csv(path)
    pred_col = "prediction" if "prediction" in df.columns else df.columns[-1]
    return df[pred_col]


def load_late_holdout_preds(prefix: str) -> pd.DataFrame:
    candidates = sorted(LATE_HOLDOUTS_BASE.glob(f"{prefix}*"))
    if not candidates:
        raise FileNotFoundError(f"Late holdout dir not found with prefix: {prefix}")
    path = candidates[-1] / "late_holdout_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Late holdout predictions not found: {path}")
    return pd.read_csv(path)


def build_champion_late_holdout(base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build champion late-holdout predictions from components."""
    champ_df = None
    for label, (path, weight) in CHAMP_LATE_HOLDOUT_COMPONENTS.items():
        comp = pd.read_csv(path).rename(columns={"prediction": label})
        if champ_df is None:
            champ_df = comp[["holdout", "row_index", "target_value"]].copy()
        champ_df[label] = comp[label].values
    assert champ_df is not None
    champ_df["champion"] = champ_df["t070"] * 0.70 + champ_df["t041"] * 0.04 + champ_df["t164"] * 0.26
    return champ_df[["holdout", "row_index", "target_value", "champion"]]


def compute_oof_fold_auc(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    result = {}
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        if sub["target_value"].nunique() < 2:
            continue
        result[str(fold)] = roc_auc(sub["target_value"].to_numpy(), sub[pred_col].to_numpy())
    all_preds = df.sort_values("row_index")
    result["OOF"] = roc_auc(all_preds["target_value"].to_numpy(), all_preds[pred_col].to_numpy())
    return result


def compute_late_holdout_auc(lh_df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    result = {}
    aucs = []
    for h in HOLDOUT_NAMES:
        sub = lh_df[lh_df["holdout"] == h]
        if len(sub) == 0 or sub["target_value"].nunique() < 2:
            continue
        auc = roc_auc(sub["target_value"].to_numpy(), sub[pred_col].to_numpy())
        result[h] = auc
        aucs.append(auc)
    if aucs:
        result["MEAN"] = float(np.mean(aucs))
        result["MIN"] = float(np.min(aucs))
    return result


# ── Blend scan ────────────────────────────────────────────────────────────────

def scan_blend(
    merged: pd.DataFrame,
    test_champion: pd.Series,
    test_candidate: pd.Series,
    lh_merged: pd.DataFrame | None,
    candidate_name: str,
    blend_mode: str,
    champ_fold3_auc: float,
    champ_lh_mean: float,
    champ_lh_min: float,
    weight_step: float = 0.05,
    max_candidate_weight: float = 0.50,
) -> list[dict[str, Any]]:
    """Scan candidate weights and return rows passing the gate.

    merged: DataFrame with columns [row_index, fold, target_value, 'champion', candidate_name]
    lh_merged: DataFrame with columns [holdout, row_index, target_value, 'champion', candidate_name] or None
    """

    weights = np.round(np.arange(0.0, max_candidate_weight + weight_step / 2, weight_step), 4)
    rows = []

    for wc in weights:
        wchamp = round(1.0 - wc, 4)
        if wchamp < 0:
            break

        # OOF blend
        if blend_mode == "rank":
            champ_rank = rank_normalize_per_fold(merged, "champion")
            cand_rank = rank_normalize_per_fold(merged, candidate_name)
            blended_oof = wchamp * champ_rank + wc * cand_rank
        else:
            blended_oof = wchamp * merged["champion"] + wc * merged[candidate_name]

        merged["blend"] = blended_oof.values
        fold_aucs = compute_oof_fold_auc(merged, "blend")
        fold3_auc = fold_aucs.get("3", float("nan"))

        # Test blend
        if blend_mode == "rank":
            tc_rank = rank_normalize(test_champion.reset_index(drop=True))
            tk_rank = rank_normalize(test_candidate.reset_index(drop=True))
            blended_test = wchamp * tc_rank + wc * tk_rank
        else:
            blended_test = wchamp * test_champion.reset_index(drop=True) + wc * test_candidate.reset_index(drop=True)

        # Late holdout
        lh_mean = lh_min = float("nan")
        if lh_merged is not None:
            if blend_mode == "rank":
                lh_parts = []
                for h in HOLDOUT_NAMES:
                    sub = lh_merged[lh_merged["holdout"] == h].copy()
                    sub["champ_r"] = rank_normalize(sub["champion"])
                    sub["cand_r"] = rank_normalize(sub[candidate_name])
                    sub["blend_lh"] = wchamp * sub["champ_r"] + wc * sub["cand_r"]
                    lh_parts.append(sub)
                lh_full = pd.concat(lh_parts, ignore_index=True)
            else:
                lh_full = lh_merged.copy()
                lh_full["blend_lh"] = wchamp * lh_full["champion"] + wc * lh_full[candidate_name]
            lh_aucs = compute_late_holdout_auc(lh_full, "blend_lh")
            lh_mean = lh_aucs.get("MEAN", float("nan"))
            lh_min = lh_aucs.get("MIN", float("nan"))

        # Gate check
        fold3_ok = fold3_auc >= champ_fold3_auc + FOLD3_DEGRADATION_TOLERANCE
        lh_ok = (
            (np.isnan(lh_mean) or lh_mean >= champ_lh_mean + LATE_HOLDOUT_MEAN_TOLERANCE)
            and (np.isnan(lh_min) or lh_min >= champ_lh_min + LATE_HOLDOUT_MIN_TOLERANCE)
        )

        rows.append({
            "candidate": candidate_name,
            "blend_mode": blend_mode,
            "w_champion": wchamp,
            "w_candidate": wc,
            "fold1_auc": fold_aucs.get("1", float("nan")),
            "fold2_auc": fold_aucs.get("2", float("nan")),
            "fold3_auc": fold3_auc,
            "oof_auc": fold_aucs.get("OOF", float("nan")),
            "lh_mean": lh_mean,
            "lh_min": lh_min,
            "fold3_delta": fold3_auc - champ_fold3_auc,
            "lh_mean_delta": lh_mean - champ_lh_mean if not np.isnan(lh_mean) else float("nan"),
            "lh_min_delta": lh_min - champ_lh_min if not np.isnan(lh_min) else float("nan"),
            "fold3_gate": fold3_ok,
            "lh_gate": lh_ok,
            "pass_all_gates": fold3_ok and lh_ok,
            "test_pred_mean": float(blended_test.mean()),
            "test_pred_min": float(blended_test.min()),
            "test_pred_max": float(blended_test.max()),
        })

    return rows


def main() -> None:
    print("=" * 70)
    print("Constrained blend builder")
    print("=" * 70)

    # ── Load champion OOF ─────────────────────────────────────────────────────
    print("\nLoading champion OOF predictions...")
    champ_oof = load_oof(CHAMPION_RUN).rename(columns={CHAMPION_RUN: "champion"})
    champ_test = load_test_preds(CHAMPION_RUN).rename("champion")

    champ_fold_aucs = compute_oof_fold_auc(champ_oof, "champion")
    champ_fold3_auc = champ_fold_aucs.get("3", 0.0)
    champ_oof_auc = champ_fold_aucs.get("OOF", 0.0)
    print(f"  Champion Fold3={champ_fold3_auc:.6f}  OOF={champ_oof_auc:.6f}")

    # ── Load champion late holdout ─────────────────────────────────────────────
    print("Loading champion late-holdout predictions from components...")
    champ_lh = build_champion_late_holdout()
    champ_lh_aucs = compute_late_holdout_auc(champ_lh, "champion")
    champ_lh_mean = champ_lh_aucs.get("MEAN", float("nan"))
    champ_lh_min = champ_lh_aucs.get("MIN", float("nan"))
    print(f"  Champion LH_mean={champ_lh_mean:.6f}  LH_min={champ_lh_min:.6f}")

    # ── Scan available candidates ──────────────────────────────────────────────
    all_scan_rows: list[dict] = []
    best_candidates: list[dict] = []

    for key, meta in COMPONENTS.items():
        if key == "champion":
            continue
        run_name = meta["run"]
        lh_prefix = meta.get("late_holdout_prefix")
        blend_mode = meta.get("blend_mode", "raw")

        print(f"\nScanning candidate: {key}  blend_mode={blend_mode}")
        try:
            cand_oof = load_oof(run_name).rename(columns={run_name: key})
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        try:
            cand_test = load_test_preds(run_name).rename(key)
        except FileNotFoundError as e:
            print(f"  SKIP test preds — {e}")
            continue

        lh_cand: pd.DataFrame | None = None
        if lh_prefix:
            try:
                lh_raw = load_late_holdout_preds(lh_prefix)
                lh_cand = lh_raw.rename(columns={"prediction": key})
            except FileNotFoundError as e:
                print(f"  Late holdout UNAVAILABLE — {e}. Gate will use OOF-only for candidate weight scan.")

        # Merge OOF with champion
        cand_oof_merged = champ_oof.merge(
            cand_oof[["row_index", "fold", key]],
            on=["row_index", "fold"],
            how="inner",
        )
        # Merge late holdout with champion
        lh_merged: pd.DataFrame | None = None
        if lh_cand is not None:
            lh_merged = champ_lh.merge(
                lh_cand[["holdout", "row_index", key]],
                on=["holdout", "row_index"],
                how="inner",
            )

        # Compute standalone candidate metrics
        cand_fold_aucs = compute_oof_fold_auc(cand_oof_merged, key)
        corr = cand_oof_merged["champion"].corr(cand_oof_merged[key])
        print(f"  Standalone Fold3={cand_fold_aucs.get('3',float('nan')):.6f}  OOF={cand_fold_aucs.get('OOF',float('nan')):.6f}  corr={corr:.4f}")

        rows = scan_blend(
            merged=cand_oof_merged,
            test_champion=champ_test,
            test_candidate=cand_test,
            lh_merged=lh_merged,
            candidate_name=key,
            blend_mode=blend_mode,
            champ_fold3_auc=champ_fold3_auc,
            champ_lh_mean=champ_lh_mean,
            champ_lh_min=champ_lh_min,
        )
        all_scan_rows.extend(rows)

        passing = [r for r in rows if r["pass_all_gates"]]
        if passing:
            best = max(passing, key=lambda r: (r["fold3_auc"], r.get("lh_mean", 0)))
            best_candidates.append(best)
            print(f"  PASS: best w_candidate={best['w_candidate']}  Fold3={best['fold3_auc']:.6f}  LH_mean={best['lh_mean']:.6f}  LH_delta={best['lh_mean_delta']:+.6f}")
        else:
            # Find best Fold3 even if gate fails
            best_fold3 = max(rows, key=lambda r: r["fold3_auc"])
            print(f"  NO PASS: best Fold3={best_fold3['fold3_auc']:.6f} at w={best_fold3['w_candidate']} (gate fail: fold3_ok={best_fold3['fold3_gate']} lh_ok={best_fold3['lh_gate']})")

    # ── Save scan results ──────────────────────────────────────────────────────
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_str = stamp[:8]

    if all_scan_rows:
        scan_df = pd.DataFrame(all_scan_rows).sort_values(["candidate", "w_candidate"])
        scan_path = REPORTS_DIR / f"constrained_blend_scan_{date_str}.csv"
        scan_df.to_csv(scan_path, index=False)
        print(f"\nScan CSV: {scan_path}")

    # ── Report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Champion: Fold3={champ_fold3_auc:.6f}  OOF={champ_oof_auc:.6f}  LH_mean={champ_lh_mean:.6f}  LH_min={champ_lh_min:.6f}")

    if best_candidates:
        print("\nPassing blend candidates:")
        for b in best_candidates:
            print(f"  {b['candidate']} ({b['blend_mode']}) w={b['w_candidate']:.2f}: "
                  f"Fold3={b['fold3_auc']:.6f} (Δ{b['fold3_delta']:+.5f})  "
                  f"LH_mean={b['lh_mean']:.6f} (Δ{b['lh_mean_delta']:+.5f})  "
                  f"OOF={b['oof_auc']:.6f}")
    else:
        print("\nNo candidates passed all gates. No submission built.")
        print("Recommendations:")
        print("  - Run LGBM late holdouts (scripts/evaluate_late_holdouts_lgbm.py)")
        print("  - Run ExtraTrees/RF (scripts/baseline_et_time.py)")
        print("  - Run LGBM HPO (scripts/optuna_lgbm_time.py)")
        return

    # ── Build best submission ──────────────────────────────────────────────────
    best_overall = max(best_candidates, key=lambda r: (r["fold3_auc"], r.get("lh_mean", 0)))
    print(f"\nSelected for submission: {best_overall['candidate']} w={best_overall['w_candidate']:.2f}")

    cand_run = COMPONENTS[best_overall["candidate"]]["run"]
    cand_test = load_test_preds(cand_run)
    blend_mode = best_overall["blend_mode"]
    wchamp = best_overall["w_champion"]
    wcand = best_overall["w_candidate"]

    if blend_mode == "rank":
        tc_rank = rank_normalize(champ_test.reset_index(drop=True))
        tk_rank = rank_normalize(cand_test.reset_index(drop=True))
        final_pred = wchamp * tc_rank + wcand * tk_rank
    else:
        final_pred = wchamp * champ_test.reset_index(drop=True) + wcand * cand_test.reset_index(drop=True)

    # Load front_id order from test
    test_df = pd.read_csv(REPO_ROOT / "data" / "raw" / "test_apps.csv", usecols=["front_id"])
    assert len(test_df) == len(final_pred), f"Row count mismatch: {len(test_df)} vs {len(final_pred)}"

    sub_name = f"constrained_blend_{best_overall['candidate']}_{blend_mode}_c{int(wchamp*100):02d}_x{int(wcand*100):02d}_{date_str}.csv"
    sub_path = SUBMISSIONS_DIR / sub_name
    sub_df = pd.DataFrame({"front_id": test_df["front_id"].values, "target_value": final_pred.values})
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission: {sub_path}")
    print(f"  rows={len(sub_df)}, min={final_pred.min():.4f}, mean={final_pred.mean():.4f}, max={final_pred.max():.4f}, nan={final_pred.isna().sum()}")

    # Card
    import hashlib
    sha256 = hashlib.sha256(sub_path.read_bytes()).hexdigest()
    card = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L4_constrained_blend_gate",
        "verdict": "PASS_WITH_RISKS",
        "submission": str(sub_path),
        "sha256": sha256,
        "blend": {
            "champion_run": CHAMPION_RUN,
            "candidate_run": cand_run,
            "candidate_key": best_overall["candidate"],
            "blend_mode": blend_mode,
            "w_champion": wchamp,
            "w_candidate": wcand,
        },
        "fold_metrics": {
            "fold1": best_overall["fold1_auc"],
            "fold2": best_overall["fold2_auc"],
            "fold3": best_overall["fold3_auc"],
            "oof": best_overall["oof_auc"],
            "fold3_delta_vs_champion": best_overall["fold3_delta"],
        },
        "late_holdout": {
            "lh_mean": best_overall["lh_mean"],
            "lh_min": best_overall["lh_min"],
            "lh_mean_delta": best_overall["lh_mean_delta"],
            "lh_min_delta": best_overall["lh_min_delta"],
        },
        "gates_passed": {
            "fold3_gate": best_overall["fold3_gate"],
            "lh_gate": best_overall["lh_gate"],
        },
        "champion_public_auc": CHAMPION_PUBLIC_AUC,
        "data_contract": {
            "rows": len(sub_df),
            "columns": ["front_id", "target_value"],
            "front_id_unique": bool(sub_df["front_id"].nunique() == len(sub_df)),
            "target_nan": int(sub_df["target_value"].isna().sum()),
        },
    }
    card_path = CARDS_DIR / sub_name.replace(".csv", "_card.json")
    CARDS_DIR.mkdir(exist_ok=True)
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Card: {card_path}")
    print(f"SHA256: {sha256}")


if __name__ == "__main__":
    main()
