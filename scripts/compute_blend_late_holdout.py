#!/usr/bin/env python
"""Compute late-holdout AUC for blend candidates from component late-holdout prediction files.

Supports:
  - raw probability blend (champion raw probabilities)
  - rank-percentile blend (XGB rank blend)
  - raw diversity blend (LGBM diversity blend)

Does NOT retrain models. Reads late_holdout_predictions.csv from each component.
Requires only numpy and pandas (no sklearn/scipy).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LATE_HOLDOUTS_BASE = REPO_ROOT / "reports" / "validation" / "late_holdouts"
REPORTS_DIR = REPO_ROOT / "reports" / "validation"

# ── Component directories ──────────────────────────────────────────────────────

# Champion blend: trial0070 (w=0.70) + trial0041 (w=0.04) + trial0164 (w=0.26)
CHAMP_COMPONENTS = {
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

XGB_PRED_PATH = (
    LATE_HOLDOUTS_BASE
    / "late_holdouts_xgb_context_offer_unweighted_v1_20260617T235004Z"
    / "late_holdout_predictions.csv"
)

LGBM_DIRS = {
    "lgbm_unbalanced": "late_holdouts_lgbm_context_offer_unbalanced_v1",
    "lgbm_sqrtpos": "late_holdouts_lgbm_context_offer_sqrtpos_v1",
}

HOLDOUT_NAMES = [
    "H1_2025_03_01_to_end",
    "H2_2025_04_01_to_end",
    "H3_2025_05_01_to_end",
]


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn using Mann-Whitney U statistic.

    Formula (Wilcoxon rank-sum):
        AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    where R_pos = sum of ranks of positive samples when ALL samples are ranked
    in ASCENDING score order (rank 1 = lowest score).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative samples")
    # Sort ascending (rank 1 = lowest score, rank N = highest)
    order = np.argsort(y_score, kind="stable")
    y_sorted = y_true[order]
    score_sorted = y_score[order]
    # Assign average ranks to ties
    ties = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(score_sorted):
        j = i
        while j < len(score_sorted) and score_sorted[j] == score_sorted[i]:
            j += 1
        ties[i:j] = (i + 1 + j) / 2.0
        i = j
    pos_rank_sum = ties[y_sorted.astype(bool)].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def rank_normalize(series: pd.Series) -> pd.Series:
    """Convert to rank percentile [0,1] with average ties."""
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
    if not path.exists():
        raise FileNotFoundError(f"Missing late-holdout predictions for {label}: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"prediction": label})
    return df[["holdout", "row_index", "target_value", label]]


def compute_auc_by_holdout(df: pd.DataFrame, pred_col: str, target_col: str = "target_value") -> dict:
    rows = []
    for h in HOLDOUT_NAMES:
        sub = df[df["holdout"] == h]
        if len(sub) == 0 or sub[target_col].nunique() < 2:
            continue
        auc = roc_auc(sub[target_col].to_numpy(), sub[pred_col].to_numpy())
        rows.append({"holdout": h, "roc_auc": auc})
    aucs = [r["roc_auc"] for r in rows]
    rows.append({"holdout": "LATE_HOLDOUT_MEAN", "roc_auc": float(np.mean(aucs))})
    rows.append({"holdout": "LATE_HOLDOUT_STD", "roc_auc": float(np.std(aucs, ddof=0))})
    rows.append({"holdout": "LATE_HOLDOUT_MIN", "roc_auc": float(np.min(aucs))})
    return {r["holdout"]: r["roc_auc"] for r in rows}


def find_lgbm_dir(key: str) -> Path | None:
    prefix = LGBM_DIRS[key]
    candidates = sorted(LATE_HOLDOUTS_BASE.glob(f"{prefix}*"))
    return candidates[-1] if candidates else None


def main() -> None:
    print("=" * 70)
    print("Blend late-holdout evaluation")
    print("=" * 70)

    # ── Step 1: champion blend ─────────────────────────────────────────────────
    print("\n[1/3] Champion blend: t070×0.70 + t041×0.04 + t164×0.26")
    base_df = None
    for label, (path, weight) in CHAMP_COMPONENTS.items():
        comp = load_component(path, label)
        if base_df is None:
            base_df = comp[["holdout", "row_index", "target_value"]].copy()
        base_df[label] = comp[label].values

    assert base_df is not None
    base_df["champion"] = (
        base_df["t070"] * 0.70
        + base_df["t041"] * 0.04
        + base_df["t164"] * 0.26
    )
    champ_auc = compute_auc_by_holdout(base_df, "champion")
    for h, v in champ_auc.items():
        print(f"  {h}: {v:.6f}")

    # ── Step 2: XGB rank blend ────────────────────────────────────────────────
    print("\n[2/3] XGB rank blend: champion_rank×0.62 + xgb_rank×0.38 (rank per holdout)")
    xgb_preds = load_component(XGB_PRED_PATH, "xgb_unweighted")
    base_df = base_df.merge(
        xgb_preds[["holdout", "row_index", "xgb_unweighted"]],
        on=["holdout", "row_index"],
        how="inner",
    )

    blend_parts = []
    for h in HOLDOUT_NAMES:
        sub = base_df[base_df["holdout"] == h].copy()
        sub["champion_rank"] = rank_normalize(sub["champion"])
        sub["xgb_rank"] = rank_normalize(sub["xgb_unweighted"])
        sub["xgb_rank_blend"] = 0.62 * sub["champion_rank"] + 0.38 * sub["xgb_rank"]
        blend_parts.append(sub)
    xgb_blend_df = pd.concat(blend_parts, ignore_index=True)
    xgb_blend_auc = compute_auc_by_holdout(xgb_blend_df, "xgb_rank_blend")
    for h, v in xgb_blend_auc.items():
        print(f"  {h}: {v:.6f}")

    # ── Step 3: LGBM diversity blend ──────────────────────────────────────────
    print("\n[3/3] LGBM diversity blend: champion×0.88 + lgbm_unb×0.11 + lgbm_sq×0.01")
    lgbm_unb_dir = find_lgbm_dir("lgbm_unbalanced")
    lgbm_sqr_dir = find_lgbm_dir("lgbm_sqrtpos")

    lgbm_blend_auc: dict | None = None
    if lgbm_unb_dir is None or lgbm_sqr_dir is None:
        missing = []
        if lgbm_unb_dir is None:
            missing.append("lgbm_unbalanced")
        if lgbm_sqr_dir is None:
            missing.append("lgbm_sqrtpos")
        print(f"  SKIPPED — missing late-holdout dirs: {missing}")
        print("  Run first:")
        print("    python scripts/evaluate_late_holdouts_lgbm.py --config configs/diversity_experiments/lgbm_context_offer_unbalanced_v1.json")
        print("    python scripts/evaluate_late_holdouts_lgbm.py --config configs/diversity_experiments/lgbm_context_offer_sqrtpos_v1.json")
    else:
        lgbm_unb = load_component(lgbm_unb_dir / "late_holdout_predictions.csv", "lgbm_unbalanced")
        lgbm_sqr = load_component(lgbm_sqr_dir / "late_holdout_predictions.csv", "lgbm_sqrtpos")
        lgbm_df = (
            base_df
            .merge(lgbm_unb[["holdout", "row_index", "lgbm_unbalanced"]], on=["holdout", "row_index"], how="inner")
            .merge(lgbm_sqr[["holdout", "row_index", "lgbm_sqrtpos"]], on=["holdout", "row_index"], how="inner")
        )
        lgbm_df["lgbm_diversity_blend"] = (
            lgbm_df["champion"] * 0.88
            + lgbm_df["lgbm_unbalanced"] * 0.11
            + lgbm_df["lgbm_sqrtpos"] * 0.01
        )
        lgbm_blend_auc = compute_auc_by_holdout(lgbm_df, "lgbm_diversity_blend")
        for h, v in lgbm_blend_auc.items():
            print(f"  {h}: {v:.6f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    display_holdouts = HOLDOUT_NAMES + ["LATE_HOLDOUT_MEAN", "LATE_HOLDOUT_MIN"]
    header = f"{'holdout':<32} {'champion':>10} {'xgb_rank':>10}"
    if lgbm_blend_auc:
        header += f" {'lgbm_div':>10}"
    print(header)
    for h in display_holdouts:
        cv = champ_auc.get(h, float("nan"))
        xv = xgb_blend_auc.get(h, float("nan"))
        row = f"{h:<32} {cv:>10.6f} {xv:>10.6f} (Δ{xv-cv:+.5f})"
        if lgbm_blend_auc:
            lv = lgbm_blend_auc.get(h, float("nan"))
            row += f" {lv:>10.6f} (Δ{lv-cv:+.5f})"
        print(row)

    # ── Save report ────────────────────────────────────────────────────────────
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    date_str = stamp[:8]
    report_path = REPORTS_DIR / f"blend_late_holdout_comparison_{date_str}.md"

    lines = [
        f"# Blend late-holdout comparison — {date_str}",
        "",
        "## Champion blend",
        "- Components: trial0070 × 0.70 + trial0041 × 0.04 + trial0164 × 0.26",
        "",
        "| holdout | champion | xgb_rank_blend | Δ_xgb |"
        + (" lgbm_diversity | Δ_lgbm |" if lgbm_blend_auc else ""),
        "|---|---|---|---|" + ("---|---|" if lgbm_blend_auc else ""),
    ]
    for h in display_holdouts:
        cv = champ_auc.get(h)
        xv = xgb_blend_auc.get(h)
        if cv is None or xv is None:
            continue
        row = f"| {h} | {cv:.6f} | {xv:.6f} | {xv-cv:+.6f} |"
        if lgbm_blend_auc:
            lv = lgbm_blend_auc.get(h)
            row += f" {lv:.6f} | {lv-cv:+.6f} |" if lv is not None else " — | — |"
        lines.append(row)

    lines += [
        "",
        "## XGB rank blend",
        "- Blend: `0.62 × champion_rank + 0.38 × xgb_unweighted_rank` (rank-percentile per holdout)",
        "",
    ]
    if lgbm_blend_auc:
        lines += [
            "## LGBM diversity blend",
            "- Blend: `0.88 × champion + 0.11 × lgbm_unbalanced + 0.01 × lgbm_sqrtpos`",
            "",
        ]
    else:
        lines += ["## LGBM diversity blend", "SKIPPED — run lgbm late holdouts first.", ""]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path}")

    # ── Update submission cards ────────────────────────────────────────────────
    _update_card(
        REPO_ROOT / "submissions" / "cards" / "candidate_20260619_upload2_RETEST_xgb_rank_c62_x38_card.json",
        xgb_blend_auc,
        champ_auc,
        "xgb_rank_blend_late_holdout",
    )
    if lgbm_blend_auc is not None:
        _update_card(
            REPO_ROOT / "submissions" / "cards" / "candidate_20260619_upload2_RETEST_lgbm_diversity_c88_u11_s01_card.json",
            lgbm_blend_auc,
            champ_auc,
            "lgbm_diversity_blend_late_holdout",
        )


def _update_card(card_path: Path, blend_auc: dict, champ_auc: dict, blend_key: str) -> None:
    if not card_path.exists():
        print(f"Card not found, skipping: {card_path.name}")
        return
    card = json.loads(card_path.read_text(encoding="utf-8"))
    delta_mean = blend_auc.get("LATE_HOLDOUT_MEAN", 0) - champ_auc.get("LATE_HOLDOUT_MEAN", 0)
    delta_min = blend_auc.get("LATE_HOLDOUT_MIN", 0) - champ_auc.get("LATE_HOLDOUT_MIN", 0)
    verdict = "PASS_WITH_RISKS" if (delta_min >= -0.001 and delta_mean >= 0.0) else "RETEST"
    late = {
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_late_holdout": champ_auc,
        blend_key: blend_auc,
        "delta_mean": round(delta_mean, 6),
        "delta_min": round(delta_min, 6),
        "verdict": verdict,
        "note": "Computed by compute_blend_late_holdout.py from component late-holdout prediction files.",
    }
    card["late_holdout_battery"] = late
    # Upgrade verdict from RETEST to PASS_WITH_RISKS only if late holdout passes
    if verdict == "PASS_WITH_RISKS" and card.get("verdict") == "RETEST":
        card["verdict"] = "PASS_WITH_RISKS"
    card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated card: {card_path.name}  late_holdout verdict={verdict}  Δmean={delta_mean:+.6f}  Δmin={delta_min:+.6f}")


if __name__ == "__main__":
    main()
