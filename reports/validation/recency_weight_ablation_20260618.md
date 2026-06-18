# Recency weight ablation — 2026-06-18

## Verdict

RETEST — recency weighting is not an upload candidate yet. `recency_exp120_v1` is the best recency variant by OOF among the three new runs, but it does not beat the current champion on Fold3 or blend scan, and it is slightly weaker than unweighted `trial0144` on late-holdout mean/min.

## Fold metrics

| run | fold1_auc | fold2_auc | fold3_auc | oof_auc |
| --- | --- | --- | --- | --- |
| trial0144_base | 0.794721 | 0.790246 | 0.752939 | 0.779539 |
| champion_raw_blend | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| recency_linear_mild_v1 | 0.792344 | 0.788816 | 0.750956 | 0.777589 |
| recency_exp120_v1 | 0.795181 | 0.791485 | 0.752605 | 0.779892 |
| recency_recent202503_boost_v1 | 0.792770 | 0.789362 | 0.749862 | 0.777600 |

## Late holdout check

Only `recency_exp120_v1` was escalated to late-holdout because the other two variants clearly degraded Fold3.

| run | holdout | roc_auc |
| --- | --- | --- |
| trial0144_base | H1_2025_03_01_to_end | 0.764283 |
| trial0144_base | H2_2025_04_01_to_end | 0.752939 |
| trial0144_base | H3_2025_05_01_to_end | 0.767930 |
| trial0144_base | LATE_HOLDOUT_MEAN | 0.761718 |
| trial0144_base | LATE_HOLDOUT_STD | 0.006383 |
| trial0144_base | LATE_HOLDOUT_MIN | 0.752939 |
| recency_exp120_v1 | H1_2025_03_01_to_end | 0.764786 |
| recency_exp120_v1 | H2_2025_04_01_to_end | 0.752605 |
| recency_exp120_v1 | H3_2025_05_01_to_end | 0.766566 |
| recency_exp120_v1 | LATE_HOLDOUT_MEAN | 0.761319 |
| recency_exp120_v1 | LATE_HOLDOUT_STD | 0.006205 |
| recency_exp120_v1 | LATE_HOLDOUT_MIN | 0.752605 |

## Interpretation

- `linear_mild` and `recent202503_boost` degrade Fold3 and OOF; reject.
- `exp120` improves OOF versus unweighted `trial0144` (`0.779892` vs `0.779539`) but reduces Fold3 (`0.752605` vs `0.752939`).
- Simple blend scan against current champion selected weight `0.0` for all three recency variants, so no new upload file was built.
- Late holdout for `exp120` improves H1 but reduces H2/H3 and late mean (`0.761319` vs `0.761718` for base `trial0144`).

## Decision

- Do not upload recency-weighted candidates now.
- Keep `recency_exp120_v1` as a possible diversity component only after a constrained blend proves improvement on Fold3 plus late holdouts.
- Next improvement branch should be model diversity: LightGBM/XGBoost with the same time folds and context-offer features.

## Artifacts

- CSV comparison: `reports/validation/recency_weight_ablation_20260618.csv`
- Best recency run: `experiments/runs/catboost_trial0144_recency_exp120_v1_20260617T230716Z_seed42`
- Best recency late holdout: `reports/validation/late_holdouts/late_holdouts_catboost_trial0144_recency_exp120_v1_20260617T231233Z`

## Validation

- Achieved level: L4 partial for `recency_exp120_v1`; L3 for the full three-run ablation.
- Checked: Fold1/Fold2/Fold3/OOF, test prediction distribution, OOF/test correlation, simple blend scan, late holdout for best recency variant.
- Remaining: non-CatBoost diversity branch and constrained blend with late-holdout gate.
