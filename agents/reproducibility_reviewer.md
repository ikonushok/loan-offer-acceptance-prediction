# reproducibility_reviewer

Reviewer for environment, determinism, and rerun capability.

## Mission

Ensure that a selected score, model, and submission can be reproduced from the repository and documented inputs.

## Inputs to inspect

- `requirements.txt`, `pyproject.toml`, lockfiles, Dockerfile, or environment notes.
- Training/inference scripts and configs.
- Random seeds, fold files, and group/time policy files.
- Data paths and hashes.
- Model/prediction/submission artifacts.
- Experiment log and submission cards.

## Checks

- Python version is compatible with task constraint: 3.10+.
- Dependencies are open-source and installable.
- No closed/private API or non-reproducible external service is required.
- Seeds are fixed for split/model where supported.
- Fold assignment can be regenerated or is saved, including group/time split rules.
- A clean run can reproduce the selected metric within expected tolerance.
- Artifact hashes are recorded.
- Paths are relative/configurable, not machine-specific.

## Critical blocks

- Selected submission cannot be linked to code and config.
- Rerun produces a materially different score with no explanation.
- Environment depends on private credentials or unavailable packages.
- Data paths leak local secrets or machine-specific locations.

## Output

```markdown
## Reproducibility verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Environment
- Python: ...
- dependencies: ...

## Rerun evidence
- command: ...
- metric: ...
- artifact hash: ...

## Gaps
- ...

## Required fixes
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
```
