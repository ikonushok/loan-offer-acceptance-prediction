# LightGBM diversity ablation — 2026-06-18

## Verdict

RETEST — LightGBM is weaker as a standalone model, but useful as a low-weight diversity component. A constrained raw blend improves all three rolling folds and OOF slightly, but the gain is small and should not be uploaded before candidate #1 result is known.

## Standalone comparison

| run | corr_vs_champion | fold1_auc | fold2_auc | fold3_auc | oof_auc |
| --- | --- | --- | --- | --- | --- |
| champion_raw_blend | 1.000000 | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| lgbm_balanced | 0.921908 | 0.789256 | 0.786423 | 0.744686 | 0.773350 |
| lgbm_unbalanced | 0.878266 | 0.792706 | 0.788555 | 0.745580 | 0.774954 |
| lgbm_sqrtpos | 0.952168 | 0.791533 | 0.787670 | 0.746302 | 0.774741 |
| lgbm_pairwise | 0.865583 | 0.793109 | 0.785078 | 0.742493 | 0.771080 |

## Best constrained blend

Selected blend: `0.88 * champion_raw_blend + 0.11 * lgbm_unbalanced + 0.01 * lgbm_sqrtpos`.

| fold | champion | selected_blend | delta |
| --- | --- | --- | --- |
| Fold1 | 0.795951 | 0.796176 | +0.000224 |
| Fold2 | 0.792685 | 0.793148 | +0.000463 |
| Fold3 | 0.754800 | 0.754871 | +0.000070 |
| OOF | 0.781140 | 0.781359 | +0.000219 |

Top combo scan rows:

| wc | wu | ws | fold1_auc | fold2_auc | fold3_auc | oof_auc |
| --- | --- | --- | --- | --- | --- | --- |
| 0.880000 | 0.110000 | 0.010000 | 0.796176 | 0.793148 | 0.754871 | 0.781359 |
| 0.890000 | 0.030000 | 0.080000 | 0.796202 | 0.793272 | 0.754870 | 0.781375 |
| 0.900000 | 0.080000 | 0.020000 | 0.796160 | 0.793115 | 0.754869 | 0.781342 |
| 0.890000 | 0.100000 | 0.010000 | 0.796160 | 0.793118 | 0.754866 | 0.781345 |
| 0.890000 | 0.080000 | 0.030000 | 0.796178 | 0.793159 | 0.754865 | 0.781357 |
| 0.880000 | 0.120000 | 0.000000 | 0.796149 | 0.793152 | 0.754865 | 0.781350 |
| 0.870000 | 0.120000 | 0.010000 | 0.796192 | 0.793217 | 0.754863 | 0.781374 |
| 0.870000 | 0.130000 | 0.000000 | 0.796174 | 0.793195 | 0.754863 | 0.781365 |
| 0.860000 | 0.130000 | 0.010000 | 0.796221 | 0.793248 | 0.754863 | 0.781392 |
| 0.860000 | 0.140000 | 0.000000 | 0.796195 | 0.793241 | 0.754863 | 0.781380 |

## Candidate artifact

- Submission: `submissions/candidate_20260619_upload2_RETEST_lgbm_diversity_c88_u11_s01.csv`
- SHA256: `f3b5e0e0974a4ee73fba5e70a8826c2d79ae1b829920de3340ee2b27f8522f06`
- Source run: `experiments/runs/blend_lgbm_diversity_c88_u11_s01_20260618T2321Z_v2`
- Card: `submissions/cards/candidate_20260619_upload2_RETEST_lgbm_diversity_c88_u11_s01_card.json`

## Decision

- Do not upload this before candidate #1 raw-unrounded score is known.
- If candidate #1 improves over 0.76054, hold this candidate unless we need a controlled upload #2.
- If candidate #1 does not improve, this candidate is a plausible #2, but expected gain is uncertain because offline delta is only about `+0.00007` on Fold3 and `+0.00022` OOF.
- Next work should be XGBoost or a stronger LGBM tuning pass; current LGBM is mainly a diversity feature, not a standalone challenger.

## Validation

- Achieved level: L3 for LGBM runs and selected blend; L5 partial format check for the generated submission.
- Checked: same time folds, OOF/Fold1/Fold2/Fold3, prediction correlations, constrained blend scan, test row order, probability range.
- Remaining: late-holdout battery for non-CatBoost/blend candidates and red-team review before any upload #2.
