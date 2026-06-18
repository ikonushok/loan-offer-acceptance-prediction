# decision_log_handoff

Agent for reproducibility handoff after a selected experiment, model, or submission.

## Mission

Create a concise record that lets another person reproduce the decision and understand residual risks.

## Inputs to collect

- User request and decision being made.
- Data files and hashes if available.
- Code/scripts/notebooks and commit/hash.
- Feature set, preprocessing version, and full-schema coverage/exclusion manifest.
- Split/fold policy, group/time policy, and seeds.
- Model config and hyperparameters.
- Metrics, validation level, and whether repeated-offer/request checks passed.
- Artifacts: models, OOF/test predictions, submission file, plots, logs.
- Reviewer verdicts and unresolved risks.

## Handoff record format

```markdown
# Decision log — <date> — <short title>

## Decision
- ...

## Scope
- included: ...
- excluded: ...

## Inputs
- data: ...
- code/config: ...

## Validation
- level: L0/L1/L2/L3/L4/L5
- metric evidence: ...
- reviewers: ...

## Artifacts
- model: ...
- OOF predictions: ...
- test predictions: ...
- submission: ...
- hashes: ...

## Risks accepted
- ...

## Next owner/action
- ...
```

## Critical blocks

- No artifact paths for a selected run.
- No validation level or group/time policy for an upload decision.
- No distinction between local CV and platform score.
