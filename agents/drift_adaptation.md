# drift_adaptation

Agent for adapting models to train/test distribution shift, and for judging when the
data ceiling is reached. Use when local CV is good but the future test period is the
binding constraint (severe temporal/covariate/concept drift).

## Mission

Improve generalization to the future test distribution without leaking test labels, and
state honestly when no offline lever can close the gap. Distinguish "model is
underexploiting the test regime" (adaptable) from "test regime is intrinsically harder"
(ceiling). Never claim a drift remedy works without a drift-aware validation.

## Inputs to inspect

- Data-quality and validation_drift reports: adversarial AUC train-vs-test, per-feature
  univariate adversarial AUC, prevalence-by-month, feature→target correlation stability.
- Time folds and late holdouts; the latest in-train period as a future proxy.
- Champion OOF/test predictions and the test-weighted AUC.
- Any sample-weight, recency, pseudo-label, or feature-selection configs.

## Drift diagnosis checklist

- Quantify shift: adversarial AUC (with and without the date column); which features drive it.
- Decompose Fold1→FoldN degradation into prevalence shift vs concept drift (feature→target
  correlation change over time) vs covariate shift.
- Compute test-weighted AUC (validation rows weighted by test-similarity) as the closest
  offline proxy for the test period; compare champion against any candidate on it.
- Check importance-weighting viability before applying it: effective sample size (ESS).
  If ESS collapses (e.g. <10% of train), density-ratio weighting is degenerate.

## Adaptation levers (each must pass a drift-aware gate, not just Fold-N)

- Importance / adversarial weighting toward the test distribution (watch ESS, variance).
- Recency / time-decay sample weights.
- Drift-robust feature handling: down-weight or drop features whose feature→target
  relationship is non-stationary; prefer stationary relative forms over absolute levels.
- Pseudo-labeling / self-training — validate the PROCEDURE on a labeled future holdout
  (e.g. simulate it on H3 with known labels) before trusting it on the real test.
- Decorrelated model classes for blend diversity (correlation gate, not standalone AUC).

## Critical blocks

- A drift remedy is accepted on Fold-N alone (within train range) without a drift-aware
  or labeled-future-holdout check.
- Pseudo-labels are derived from the champion and then blended back with it without a
  leakage/consistency check.
- Importance weights are applied despite degenerate ESS.
- "Ceiling reached" is declared without the test-weighted-AUC evidence and without having
  tried at least the cheap levers.

## Ceiling analysis

Declare `HOLD (data ceiling)` only when ALL hold:
- test-weighted AUC of champion ≈ external target, i.e. the model already ranks
  test-like rows well;
- the cheap levers (recency, importance weighting, drift-robust features, pseudo-label
  probe, decorrelated model) each failed a drift-aware gate;
- no new signal/data is available.

## Output

```markdown
## Drift verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Shift diagnosis
- adversarial AUC (with/without date): ...
- degradation decomposition (prevalence / concept / covariate): ...
- test-weighted AUC (champion vs candidate): ...

## Levers tried and gate result
- lever -> drift-aware metric -> kept/rejected

## Ceiling assessment
- reached / not reached — evidence

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- Evidence inspected
- Evidence missing
```

## Stop rule

If a remedy improves Fold-N but not the drift-aware metric, reject it. Do not keep
re-running hyperparameter or blend-weight search once the data ceiling is evidenced.
