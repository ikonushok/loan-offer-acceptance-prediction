# Blend late-holdout comparison — 20260618

## Champion blend
- Components: trial0070 × 0.70 + trial0041 × 0.04 + trial0164 × 0.26

| holdout | champion | xgb_rank_blend | Δ_xgb | lgbm_diversity | Δ_lgbm |
|---|---|---|---|---|---|
| H1_2025_03_01_to_end | 0.762230 | 0.765189 | +0.002959 | 0.762667 | +0.000437 |
| H2_2025_04_01_to_end | 0.753249 | 0.755244 | +0.001995 | 0.753488 | +0.000239 |
| H3_2025_05_01_to_end | 0.765072 | 0.768880 | +0.003808 | 0.765496 | +0.000424 |
| LATE_HOLDOUT_MEAN | 0.760184 | 0.763104 | +0.002921 | 0.760550 | +0.000366 |
| LATE_HOLDOUT_MIN | 0.753249 | 0.755244 | +0.001995 | 0.753488 | +0.000239 |

## XGB rank blend
- Blend: `0.62 × champion_rank + 0.38 × xgb_unweighted_rank` (rank-percentile per holdout)

## LGBM diversity blend
- Blend: `0.88 × champion + 0.11 × lgbm_unbalanced + 0.01 × lgbm_sqrtpos`

