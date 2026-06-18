# XGBoost diversity ablation — 2026-06-18

## Verdict

PASS_WITH_RISKS for offline candidate generation; RETEST for upload. XGBoost is the first diversity branch that gives a meaningful constrained blend lift over the current champion on all three rolling folds.

## Standalone comparison

| run | corr | fold1 | fold2 | fold3 | oof |
| --- | --- | --- | --- | --- | --- |
| champion | 1.000000 | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| lgbm_unb | 0.878266 | 0.792706 | 0.788555 | 0.745580 | 0.774954 |
| lgbm_sqrt | 0.952168 | 0.791533 | 0.787670 | 0.746302 | 0.774741 |
| xgb_unw | 0.880017 | 0.793841 | 0.792726 | 0.753735 | 0.779273 |
| xgb_sqrt | 0.970534 | 0.794895 | 0.792732 | 0.752541 | 0.779400 |

## Selected rank blend

Selected blend: `0.62 * champion_raw_blend_rank + 0.38 * xgb_unweighted_rank`.

| fold | champion | selected_blend | delta |
| --- | --- | --- | --- |
| Fold1 | 0.795951 | 0.796492 | +0.000541 |
| Fold2 | 0.792685 | 0.794671 | +0.001986 |
| Fold3 | 0.754800 | 0.755986 | +0.001185 |
| OOF | 0.781140 | 0.782076 | +0.000936 |

Top individual scan rows:

| mode | candidate | w | fold1 | fold2 | fold3 | oof |
| --- | --- | --- | --- | --- | --- | --- |
| rank | xgb_unw | 0.380000 | 0.796492 | 0.794671 | 0.755986 | 0.782076 |
| rank | xgb_unw | 0.360000 | 0.796502 | 0.794617 | 0.755985 | 0.782075 |
| rank | xgb_unw | 0.390000 | 0.796483 | 0.794696 | 0.755983 | 0.782072 |
| rank | xgb_unw | 0.370000 | 0.796503 | 0.794640 | 0.755982 | 0.782076 |
| rank | xgb_unw | 0.400000 | 0.796467 | 0.794702 | 0.755974 | 0.782064 |
| rank | xgb_unw | 0.350000 | 0.796509 | 0.794550 | 0.755957 | 0.782066 |
| rank | xgb_unw | 0.340000 | 0.796529 | 0.794517 | 0.755936 | 0.782069 |
| rank | xgb_unw | 0.330000 | 0.796531 | 0.794493 | 0.755924 | 0.782067 |
| rank | xgb_unw | 0.320000 | 0.796537 | 0.794447 | 0.755901 | 0.782062 |
| rank | xgb_unw | 0.310000 | 0.796530 | 0.794422 | 0.755899 | 0.782053 |

## Candidate artifact

- Submission: `submissions/candidate_20260619_upload2_RETEST_xgb_rank_c62_x38.csv`
- SHA256: `901015ff18f184a7bbcce5a6505871b62d90bf2c5d0dc72ca4eaee6f179df026`
- Source run: `experiments/runs/blend_xgb_rank_c62_x38_20260618T2327Z`
- Card: `submissions/cards/candidate_20260619_upload2_RETEST_xgb_rank_c62_x38_card.json`

## Decision

- Candidate #1 for tomorrow remains raw unrounded CatBoost blend: `submissions/candidate_20260619_upload1_raw_unrounded_fold3best070.csv`.
- This XGB rank blend is the best current candidate for upload #2, but only after seeing candidate #1 public result.
- If upload #1 improves over `0.76054`, do not rush upload #2; keep it as reserve.
- If upload #1 is flat or worse, upload #2 can be justified by the offline Fold3/OOF lift, with explicit risk acceptance.

## Validation

- Achieved level: L3 for XGB models and selected blend; L5 partial format check for generated submission.
- Checked: same rolling time folds, OOF/Fold1/Fold2/Fold3, correlation vs champion, rank/raw blend scan, test row order, probability range, SHA256 card.
- Remaining: late-holdout battery for XGB/rank blend and red-team review before upload #2.
