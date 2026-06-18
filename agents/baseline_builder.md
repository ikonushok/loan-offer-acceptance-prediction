# baseline_builder

Agent for building the first minimal reproducible model.

## Mission

Create a simple, correct baseline that establishes data loading, preprocessing, CV, ROC-AUC computation, and submission-generation plumbing. Prioritize trustworthiness over score.

## Inputs to inspect

- Data-quality report with full schema and grouping/time diagnostics, plus paths to train/test/sample submission.
- Existing scripts/notebooks.
- Package availability.
- `AGENTS.md` protected contracts.

## Baseline options

Choose the simplest reliable option for the repository:

- Logistic regression with robust imputation, scaling, and one-hot encoding.
- Random forest or extra trees as a non-linear sanity baseline.
- HistGradientBoosting/LightGBM/CatBoost if installed and used with a clean pipeline.
- CatBoost can handle categoricals natively, but categorical column handling must be explicit.

## Required artifacts

- A single runnable script or notebook section.
- Fixed seed and config.
- Full train/test schema check and explicit column-use/exclusion manifest.
- Fold assignments or deterministic split with documented group/time policy.
- Per-fold ROC-AUC and OOF ROC-AUC.
- Optional baseline submission only after `submission_builder.md` checks.

## Critical blocks

- Baseline script computes train ROC-AUC only or ignores detected group/time leakage risk.
- Preprocessing does not handle missing values or categoricals.
- Submission output is written without sample format validation.
- The baseline code has hidden state from notebook execution order.

## Output

```markdown
## Baseline verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Implemented baseline
- model: ...
- preprocessing: ...
- feature coverage: all columns used/transformed/excluded with reason
- split: ...

## Scores
- per-fold ROC-AUC: ...
- OOF ROC-AUC: ...

## Artifacts
- script/notebook: ...
- predictions/submission: ...

## Limitations
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
```
