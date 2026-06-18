# agents/

Short specialized prompts for the Alfa Bank credit-offer acceptance workflow. This directory is part of a 25-file Plus-ready pack and excludes migration/change-history documents.

Do not load all agents at once. Use:

```text
AGENTS.md + context_router.md + one primary agent + zero or one reviewer
```

## Routing and control

- `context_router.md` — choose task mode, context, primary agent, reviewer, validation level.
- `architect.md` — design/scope control for non-trivial changes.
- `task_spec_short.md` — compact task spec.
- `red_team.md` — adversarial review before important submissions/decisions.
- `test_validation.md` — validation level and evidence sufficiency.
- `readme_consistency_reviewer.md` — README/spec/experiment-log drift control.
- `decision_log_handoff.md` — reproducibility and handoff records.

## Data, leakage, and features

- `data_quality.md` — CSV schema, missingness, duplicates, drift, target sanity.
- `eda_analyst.md` — feature-target relationships and business interpretation.
- `leakage_guard.md` — target, time, ID, encoder, and test leakage.
- `feature_engineer.md` — safe derived features and preprocessing pipeline review.

## Validation and modeling

- `cv_validator.md` — split strategy, out-of-fold predictions, seed control.
- `baseline_builder.md` — minimal reproducible baseline.
- `model_trainer.md` — model training, tuning, and artifact saving.
- `model_ensembler.md` — blending/stacking and diversity checks.
- `metric_validator.md` — ROC-AUC and probability-output checks.

## Submission and operations

- `submission_builder.md` — CSV creation and final format checks.
- `experiment_manager.md` — run registry, configs, seeds, artifact comparison.
- `interpretability_reviewer.md` — feature importance, SHAP/PDP, business sanity.
- `reproducibility_reviewer.md` — environment, deterministic rerun, dependency and artifact checks.

## Recommended flow

1. `task_spec_short.md` for non-trivial tasks.
2. `context_router.md` to select scope.
3. `data_quality.md` before any modeling; include full schema discovery and repeated-offer/request structure checks.
4. `leakage_guard.md` before trusting CV; explicitly review customer/request grouping and `decision_day` temporal leakage.
5. `baseline_builder.md` for first reproducible score.
6. `model_trainer.md` or `model_ensembler.md` only after baseline is stable.
7. `metric_validator.md` and `submission_builder.md` before exporting CSV.
8. `red_team.md` before a platform upload.
9. `decision_log_handoff.md` after a selected run or submission.
