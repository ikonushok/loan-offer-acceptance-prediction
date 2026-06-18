# context_router

Use first when a task can touch several parts of the Alfa credit-offer workflow.

Goal: choose the smallest useful context, task mode, primary agent, reviewer, and validation level before analysis or patching.

## Classify task category

Choose one or more:

- `data_quality_review` — full CSV schema discovery, missingness, duplicates, repeated request/client/offer groups, target distribution, train/test drift.
- `eda_review` — feature distributions, feature-target relationships, business interpretation.
- `leakage_review` — target leakage, test leakage, temporal leakage, ID leakage, encoder leakage.
- `feature_engineering` — derived features, transformations, pipeline changes.
- `cv_design` — split strategy, group/time policy, fold reproducibility, OOF predictions.
- `baseline_build` — simple model and first reliable validation score.
- `model_training` — advanced model training/tuning.
- `ensemble_review` — blending, stacking, rank averaging, model diversity.
- `metric_validation` — ROC-AUC calculation and probability-output checks.
- `drift_adaptation` — train/test distribution-shift adaptation and data-ceiling analysis.
- `submission_build` — final CSV generation and sample format checks.
- `experiment_tracking` — experiment log, configs, seeds, artifact registry.
- `interpretability_review` — feature importance and business sanity.
- `reproducibility_review` — environment, lockfile, deterministic rerun.
- `docs_sync` — README/spec/decision log consistency.
- `red_team` — adversarial review before platform upload or model selection.

## Choose task mode

Pick one:

- `inspect_only` — read and assess, no edits;
- `plan_only` — design plan, no edits;
- `patch_small` — one minimal local change;
- `review_only` — review evidence/results without changing files;
- `docs_sync` — update docs only after verified behavior/contract change;
- `red_team` — attack assumptions and conclusions.

## Minimal context rule

Default:

```text
AGENTS.md
agents/context_router.md
one primary agent
zero or one reviewer
```

Do not load all agents. Do request actual CSV schemas, script outputs, fold metrics, group/time diagnostics, and generated submission previews when conclusions depend on evidence.

## Routing

| Task | Primary agent | Reviewer if needed |
|---|---|---|
| unclear scope / architecture | `architect.md` | `red_team.md` |
| repository/data intake | `data_quality.md` | `test_validation.md` |
| EDA and target relationships | `eda_analyst.md` | `leakage_guard.md` |
| leakage suspicion or target encoding | `leakage_guard.md` | `cv_validator.md` |
| feature generation | `feature_engineer.md` | `leakage_guard.md` |
| validation split / OOF / group or time policy | `cv_validator.md` | `leakage_guard.md` |
| first model | `baseline_builder.md` | `metric_validator.md` |
| tuning / model comparison | `model_trainer.md` | `metric_validator.md` |
| blend / ensemble | `model_ensembler.md` | `red_team.md` |
| train/test drift / stalled improvement / ceiling | `drift_adaptation.md` | `red_team.md` |
| ROC-AUC dispute | `metric_validator.md` | `cv_validator.md` |
| submission CSV | `submission_builder.md` | `metric_validator.md` |
| experiment registry | `experiment_manager.md` | `reproducibility_reviewer.md` |
| feature importance | `interpretability_reviewer.md` | `leakage_guard.md` |
| README / log update | `readme_consistency_reviewer.md` | `decision_log_handoff.md` |
| pre-upload review | `red_team.md` | `submission_builder.md` |

## Forbidden broad changes

- Do not turn a quick schema check into model tuning.
- Do not add target encodings without fold-safe implementation.
- Do not update README as if planned behavior is implemented.
- Do not generate a platform-ready submission without format and probability checks.
- Do not recommend a platform upload from train-only metrics.
- Do not trust random CV until repeated-offer/request grouping and `decision_day` temporal risk are checked.
- Do not select multiple reviewers unless risk requires it.

## Output format

```markdown
## Routing decision
Category: ...
Mode: ...
Primary agent: ...
Reviewer: ... / none
Context to inspect: ...
Forbidden changes: ...
Minimum validation level: L0/L1/L2/L3/L4/L5
Expected output: ...
Schema/grouping checks required: yes/no
```
