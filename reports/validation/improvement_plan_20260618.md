# Improvement plan — Alfa credit offer acceptance

## Verdict

PASS_WITH_RISKS — first next upload should be the raw unrounded version of the already accepted best candidate; larger improvement requires new offline evidence, not public-score probing.

## Current best known state

- Best accepted public score: `0.76054`.
- Best accepted file: `submissions/accepted_public_76054_3dp_fold3best070.csv`.
- Prepared next candidate: `submissions/candidate_20260619_upload1_raw_unrounded_fold3best070.csv`.
- Candidate SHA256: `ba3b32669f7afb8ca5cc186de46292e3084e69776d9a2592314c6ea777fbb627`.
- Candidate policy: test-only, `36311` rows, columns `front_id,target_value`, order matches `test_apps.csv`.
- Candidate rationale: same ranking source as accepted `3dp`, but raw probabilities reduce ROC-AUC ties.

## Upload plan

1. Upload `submissions/candidate_20260619_upload1_raw_unrounded_fold3best070.csv` as the first attempt after the daily platform reset.
2. Stop after upload #1 unless the public score improves materially or a stronger offline candidate is created before upload #2.
3. Do not upload additional rounded variants; prior `1dp -> 2dp -> 3dp` results already show that reducing ties improves ROC-AUC.
4. Do not use public leaderboard as a hyperparameter search loop. Any upload #2/#3 must have a submission card and Fold3/late-holdout evidence.

## Main diagnosis

- The task is dominated by temporal drift: train ends at `2025-06-05`, while almost all test rows are future-dated.
- Random CV is not a reliable model-selection criterion.
- Fold3 and late holdouts should drive model selection; OOF is secondary.
- Current strongest modeling path is CatBoost with no raw time feature, context-offer features, Fold3-oriented HPO, and conservative blending.
- The remaining gap to `0.775078` likely needs better future-period generalization and model diversity, not more rounding or format changes.

## Priority 1 — robust validation gate

Acceptance rule for any new candidate:

- Fold3 ROC-AUC must improve over current robust blend or be clearly complementary.
- Late holdout mean/min must not degrade materially.
- Prediction correlation with current champion should be measured; new models with high correlation and no Fold3 gain should not be uploaded.
- Blend weights must be selected on offline validation only.
- Candidate must pass test row order, probability range, NaN/inf, and SHA256 card checks.

Recommended validation table:

| check | required |
|---|---|
| Fold1/Fold2/Fold3 ROC-AUC | yes |
| OOF ROC-AUC | yes, secondary |
| Late holdout H1/H2/H3 ROC-AUC | yes |
| Test prediction distribution | yes |
| Correlation vs champion | yes |
| Submission format card | yes |

## Priority 2 — recent-weighted CatBoost

Goal: improve future generalization without leaking test labels.

Experiments:

- Train with higher weights for recent train months, especially `2025-03..2025-06`.
- Try weight schedules: linear recency, exponential recency, and late-only boost.
- Keep `decision_day` excluded as a raw feature; use only safe transformations if ablation proves useful.
- Compare `Balanced`, `SqrtBalanced`, and explicit `scale_pos_weight`/sample weights.

Selection:

- Prefer candidates that improve Fold3 and `2025-05-01..2025-06-05` holdout.
- Reject candidates that only improve early folds.

## Priority 3 — diversify models

Goal: create predictors with different error structure for blending.

Experiments:

- LightGBM with categorical handling, monotone-free baseline, and tuned regularization.
- XGBoost histogram model with one-hot/ordinal encoding inside fold-safe preprocessing.
- ExtraTrees/RandomForest as low-correlation ranker, even if standalone AUC is lower.
- Logistic/linear model on robust transformed numeric features as calibration/rank sanity baseline.

Selection:

- Standalone model does not need to beat CatBoost if it improves blend Fold3/late-holdout.
- Reject weak models with high correlation to champion and no blend gain.

## Priority 4 — feature ablations

Candidate feature groups:

- Rate economics: `offered_rate - cb_rate`, `offered_rate / cb_rate`, bins by rate spread.
- Limit economics: loan-to-min/max/mid limit ratios, within-limit flags, spread ratios.
- Activity ratios: 30/90 ratios and normalized activity intensity.
- Robust transforms: `log1p` for monetary/count-like heavy-tailed fields.
- Missingness blocks: test whether missing indicators help on late holdouts, not only OOF.
- Context-offer variants: old context, same-day context, and conservative context rank features.

Rule:

- Add one feature group at a time and keep only groups that survive Fold3 + late-holdout checks.

## Priority 5 — conservative blending

Blend candidates:

- Current champion raw probabilities.
- Best recent-weighted CatBoost.
- Best tuned CatBoost from Fold3/late-holdout.
- One or two diverse non-CatBoost models.
- Optional rank blend only if probability scales differ strongly.

Weighting:

- Optimize weights on Fold3 + late holdouts with constraints, not public score.
- Use small weights for high-risk candidates.
- Prefer stable improvement over maximum OOF.

## Stop rules

- Stop upload attempts for the day after a clear non-improvement unless a new offline candidate has stronger evidence.
- Do not upload any file without a card, hash, and exact row-order check.
- Do not trust a candidate selected only by OOF if Fold3 or late holdout degrades.
- Do not spend uploads on variants whose only difference is rounding or file naming.

## Next owner actions

1. Upload prepared raw candidate after daily reset.
2. Run recent-weighted CatBoost experiments.
3. Run one properly tuned LightGBM/XGBoost branch for blend diversity.
4. Build a constrained blend from candidates that pass Fold3 and late-holdout gates.
5. Produce a new submission card before any second upload.

