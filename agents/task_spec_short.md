# task_spec_short

Use this before any non-trivial Codex task.

## Fill this in

```markdown
## Task
<one sentence>

## Mode
inspect_only / plan_only / patch_small / data_quality_review / eda_review / leakage_review / feature_engineering / cv_design / baseline_build / model_training / ensemble_review / metric_validation / submission_build / experiment_tracking / docs_sync / red_team

## Goal
<what should be true after the task>

## Inputs
- files/data/artifacts to inspect:
- assumptions:

## Non-goals
- what must not be changed or decided:

## Protected contracts
- no target/test leakage
- preserve sample submission format
- full schema discovery before feature exclusion
- repeated-offer/request and decision_day risks checked before trusting random CV
- reproducible split/seed/config
- probability outputs in [0, 1]
- open-source Python 3.10+ only

## Expected edits
- none / specific files only:

## Validation target
L0 / L1 / L2 / L3 / L4 / L5

## Stop conditions
- conditions that should make Codex stop and report instead of patching:
```

## Compact Codex prompt template

```text
Use AGENTS.md + agents/context_router.md + agents/<primary>.md + agents/<reviewer_if_needed>.md.
Mode: <mode>.
Task: <task>.
Inputs: <paths>.
Constraints: do not use target/test leakage; preserve sample_submission.csv contract; inspect full schema; check repeated-offer/request and decision_day risks; report achieved validation level.
Stop if <stop conditions>.
```
