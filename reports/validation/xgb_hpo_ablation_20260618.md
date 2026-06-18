# XGB HPO + rank-blend ablation — 2026-06-18

## Verdict
PASS_WITH_RISKS — подготовлен кандидат #2 на завтра: rank-blend accepted champion + XGB-HPO, локально лучше принятого 3dp-кандидата на Fold3 и OOF.

## Goal
Улучшить текущий лучший публичный результат `0.76054` без повторного округления accepted-кандидата и без нарушения submission-контракта.

## Inputs
- Accepted public best: `submissions/accepted_public_76054_3dp_fold3best070.csv`, public AUC `0.76054`.
- Accepted champion run: `experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z`.
- New model family: XGBoost native categorical on the same time-aware feature pipeline.
- Primary validation: time Fold3 `2025-04-01..2025-06-05`, because test period is later and train/test drift is severe.
- Secondary validation: OOF over all time folds and overlapping late holdouts H1/H2/H3.

## XGB HPO results

| run | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| accepted champion blend | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| xgb_context_offer_unweighted_v1 | 0.793841 | 0.792726 | 0.753735 | 0.779273 |
| xgb_hpo_depth3_child50_reg20_v1 | 0.794646 | 0.793452 | 0.752577 | 0.779603 |
| xgb_hpo_depth3_child80_reg30_v1 | 0.794574 | 0.791883 | 0.755030 | 0.779816 |
| xgb_hpo_depth4_child100_sub75_v1 | 0.794411 | 0.793113 | 0.754352 | 0.779978 |
| xgb_hpo_depth4_child50_reg30_v1 | 0.794833 | 0.791700 | 0.752606 | 0.779099 |
| xgb_hpo_depth5_child120_reg40_v1 | 0.795191 | 0.793141 | 0.752603 | 0.779626 |

Best standalone HPO model by primary Fold3: `xgb_hpo_depth3_child80_reg30_v1`.

## Late holdout for selected XGB-HPO

| holdout | period | AUC |
|---|---|---:|
| H1 | 2025-03-01..2025-06-05 | 0.765375 |
| H2 | 2025-04-01..2025-06-05 | 0.755030 |
| H3 | 2025-05-01..2025-06-05 | 0.769460 |
| mean | overlapping late holdouts | 0.763289 |
| min | overlapping late holdouts | 0.755030 |

XGB-HPO improves the prior unweighted XGB late mean/min (`0.762932` / `0.753735`) and is competitive with the accepted champion on the most important late fold.

## Selected blend

Rank average:

```text
0.51 * accepted_champion_rank + 0.49 * xgb_hpo_depth3_child80_reg30_rank
```

Reasoning:
- rank-blend is metric-aligned for ROC-AUC and avoids probability-scale mismatch between CatBoost/blends and XGBoost;
- `0.49` XGB weight is a compromise: almost max Fold3 among scanned weights while preserving strong OOF;
- XGB component is diverse enough to improve ranking but not strong enough to replace accepted champion entirely.

| model | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| accepted champion blend | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| selected XGB-HPO standalone | 0.794574 | 0.791883 | 0.755030 | 0.779816 |
| selected rank-blend c51/x49 | 0.796663 | 0.794185 | 0.756746 | 0.782234 |

Local lift vs accepted champion:
- Fold3: `+0.001946` AUC;
- OOF: `+0.001094` AUC.

## Submission candidate

- File: `submissions/candidate_20260619_upload2_RETEST_xgb_hpo_rank_c51_x49.csv`
- SHA256: `509ca6811e9515b355d0a07978a3f3b000acaae9d795a9466fb3a3b2c43da4fa`
- Rows: `36311`
- Columns: `front_id,target_value`
- Format: test-only, order as `test_apps.csv`, unrounded probabilities with `%.10f`
- Prediction range: `[0.0001118118, 0.9999022335]`
- Unique predictions: `35415`
- Card: `submissions/cards/candidate_20260619_upload2_RETEST_xgb_hpo_rank_c51_x49_card.json`
- Run artifacts: `experiments/runs/blend_xgb_hpo_rank_c51_x49_20260618T2354Z`

## Upload recommendation

Recommended upload order for 2026-06-19:
1. Upload #1: `submissions/candidate_20260619_upload1_raw_unrounded_fold3best070.csv` — lowest-risk test of whether removing 3dp rounding improves public LB.
2. Upload #2: `submissions/candidate_20260619_upload2_RETEST_xgb_hpo_rank_c51_x49.csv` — model-improvement candidate with better local Fold3/OOF than accepted champion.
3. Keep upload #3 unused unless the first two scores give a clear signal for targeted adjustment.

## Risks
- Public LB is affected by severe temporal/test drift; local Fold3/OOF lift may not transfer exactly.
- Blend weight was selected by local validation, not by hidden labels.
- Rank transform discards calibration, but ROC-AUC only needs ordering.
- No conclusion about private LB is possible from public score alone.

## Validation
- Achieved level: L4 for local validation, L5 for CSV format readiness.
- Checked: time CV metrics, late holdout for selected XGB component, row count, columns, `front_id` order, probability range, SHA256/card.
- Unchecked: platform public score for the new candidate and private leaderboard behavior.
