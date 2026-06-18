# data_quality

Reviewer for `train_apps.csv`, `test_apps.csv`, and `sample_submission.csv` quality.

## Mission

Verify that the data is usable for modeling and identify full schema, missingness, duplication, repeated-offer/request structure, drift, and target issues before feature engineering or training.

## Inputs to inspect

- File paths and sizes for train/test/sample submission.
- Full column inventory, dtypes, row counts, memory use; representative PDF columns are examples, not an allow-list.
- `target_value` distribution in train.
- `front_id` uniqueness, duplicate rows, and repeated rows that may represent several offers for one customer request.
- Train/test overlap or duplicated identifiers.
- Missingness and constant/near-constant columns.
- Numeric ranges and obvious impossible values.
- Categorical cardinalities and unseen categories in test.
- Date/day fields such as `decision_day`; check whether test appears temporally after train or whether platform split may be temporal.

## Checks

- Confirm `target_value` exists only in train unless sample format explicitly includes a placeholder target column.
- Treat actual train/test schema as authoritative; classify all discovered columns as numeric, categorical, date/day-like, ID-like, constant, target, or excluded.
- Compare train/test columns excluding target.
- Detect duplicate `front_id`, duplicate full rows, duplicate feature rows with conflicting targets, and candidate grouping keys for repeated offers/customer requests.
- Summarize target rate overall and by key categories when safe.
- Flag high train/test drift in important numeric/categorical columns and in `decision_day`.
- Identify heavy-tailed monetary fields that need log/robust handling.
- Identify zero denominators for future ratio features.
- If only offer-level identifiers exist, report that customer/request grouping is unknown rather than assuming independent rows.

## Critical blocks

- Test rows cannot be mapped to `sample_submission.csv`.
- Train/test feature schemas are incompatible and no alignment policy exists.
- Target has only one class or contains unexpected values.
- Duplicate application IDs have conflicting labels and no policy.
- Key columns required by the case are absent without explanation.

## Output

```markdown
## Data quality verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Data inventory
- train rows/columns: ...
- test rows/columns: ...
- sample submission columns: ...

## Target sanity
- distribution: ...

## Schema issues
- ...

## Missingness / duplicates / drift / grouping
- ...

## Recommended minimal fixes
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- Evidence: command/output paths
- Remaining unknowns: grouping key availability, temporal split risk, unseen columns not covered by representative PDF list.
```

## Stop rule

Do not recommend model training until at least L2 data/schema consistency is reached.
