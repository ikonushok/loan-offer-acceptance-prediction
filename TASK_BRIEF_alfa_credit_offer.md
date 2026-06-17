# Task brief — Alfa Bank x MFTI “Отклик на кредитный оффер”

## Goal

Predict the probability that a corporate client accepts a concrete credit offer. The final model should rank offers by acceptance probability. The PDF states that one customer request may receive several possible offers, so validation should check repeated request/client/offer structure rather than assuming independent rows.

## Data

Expected platform files:

- `train_apps.csv` — train data with `target_value`;
- `test_apps.csv` — test data without `target_value`;
- `sample_submission.csv` — required submit schema.

The train data includes application parameters, offer terms, client characteristics, and financial-activity features.

## Target and metric

- `target_value = 1` — accepted offer;
- `target_value = 0` — refused offer;
- metric: ROC-AUC.

## Representative predictors

The columns below are examples from the PDF, not a complete allow-list. Codex must inspect the actual train/test schema and either use, transform, or explicitly exclude every discovered column.

- offer terms: `loan_amount_last`, `overdraft_limit_min`, `overdraft_limit_max`, `offered_rate`, `cb_rate`;
- timing: `decision_day`;
- banking activity: debit/credit sums and counts over 30/90/360 day windows;
- liquidity/balance: `balance_rur_amt_30_min`, investment sums;
- credit history: active products, loan-related counters, months since credit deals;
- digital behavior: dashboard events, time spent;
- categoricals: `db_group_last`, `fl_adminarea`.

## Constraints

Use Python 3.10+ and open-source libraries. Use only provided train/test data unless additional files are explicitly allowed. Build a valid CSV submission and avoid unnecessary platform uploads. Use no more than 3 platform uploads per day. For ChatGPT Plus project context, keep the agent pack to 25 files and omit migration/changelog/history-diff documents.
