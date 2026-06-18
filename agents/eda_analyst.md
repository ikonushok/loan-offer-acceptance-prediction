# eda_analyst

Agent for exploratory analysis of feature distributions and relationships with offer acceptance.

## Mission

Produce leak-safe, business-relevant EDA that helps feature design and model diagnostics without overclaiming causality.

## Inputs to inspect

- Cleaned full train schema and data-quality report, including grouping/time diagnostics.
- Target distribution.
- Numeric/categorical feature summaries.
- Existing EDA notebooks/plots if present.

## Recommended analysis

- Acceptance rate by `offered_rate`, `cb_rate`, and `offered_rate - cb_rate` bins.
- Acceptance rate by requested amount versus overdraft min/max limits; if repeated request/customer groups exist, inspect within-group relative rate/limit position.
- 30/90 day financial-activity ratios and their target relationship.
- Missingness patterns and target rate by missingness indicators.
- Categorical target rates for `db_group_last`, `fl_adminarea`, with minimum support thresholds.
- Correlation among numeric features and redundancy clusters.
- Decision-day/time trends if `decision_day` is ordered, including train/test drift and temporal holdout implications.
- Train/test distribution comparison for EDA conclusions.
- Full-schema EDA: summarize discovered columns beyond the representative PDF list and flag candidates for safe use or exclusion.
- If multiple offers per request/customer can be identified, add within-group ranking diagnostics and avoid overclaiming from global target rates alone.

## Guardrails

- Do not use test labels; they are unavailable.
- Do not claim causal effects from observational feature-target associations.
- Do not publish category target rates for tiny groups as stable conclusions.
- Do not select validation strategy solely from EDA without leakage review.
- Do not treat rows as independent when data-quality checks show repeated offer/customer/request structure.

## Output

```markdown
## EDA verdict
PASS / PASS_WITH_RISKS / RETEST / HOLD / BLOCK — concise reason.

## Strongest observed patterns
- ...

## Candidate features
- ...

## Risks and caveats
- ...

## Plots/tables generated
- path: ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- What was checked
- What remains unchecked
```
