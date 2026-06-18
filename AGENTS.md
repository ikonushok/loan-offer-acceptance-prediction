# AGENTS.md — Alfa Bank / MFTI credit offer acceptance project

This project uses an agent-native workflow for tabular ML development, validation, and submission building for the Alfa Bank x MFTI case “Отклик на кредитный оффер”. This Plus-ready pack intentionally contains exactly 26 project files and excludes migration/change-history documents.

Primary objective: build a reproducible model that predicts `P(target_value = 1)` for each row in `test_apps.csv`, where `1` means a corporate client accepted the proposed credit-product conditions. The main quality criterion is ROC-AUC. Do not optimize for leaderboard score at the expense of leakage, irreproducibility, or invalid submission format.

## Operating model: Codex + ChatGPT

Use this split by default:

- **Codex**: repository inspection, data/schema checks, local patches, notebooks/scripts, experiments, tests, artifact generation, submission CSV checks.
- **ChatGPT**: task framing, architecture reasoning, experiment design, leakage review, red-team critique, prompt preparation, result interpretation.
- **README / experiment logs / submission cards**: project memory and living specification, not proof of model quality.
- **Scripts, notebooks, validation output, metrics, and generated submissions**: evidence layer.

Default context:

```text
AGENTS.md + agents/context_router.md + one primary agent + zero or one reviewer
```

Do not load all agents for a single task.

## Case facts to preserve

Expected files:

- `train_apps.csv` — training tabular data;
- `test_apps.csv` — test sample;
- `sample_submission.csv` — required submission schema example.

Core fields from the case statement, used as examples rather than a closed feature list:

- `front_id` — application identifier;
- `decision_day` — decision day;
- `loan_amount_last` — requested limit in the current application;
- `overdraft_limit_min`, `overdraft_limit_max` — minimum and maximum available overdraft limits;
- `offered_rate` — offered interest rate;
- `cb_rate` — Central Bank key rate at application time;
- customer activity fields over 30/90/360 day windows;
- categorical fields such as `db_group_last` and `fl_adminarea`;
- `target_value` in train only: `1` accepted, `0` refused.

Modeling objective: rank offers by probability of acceptance; metric: ROC-AUC; final artifact: valid CSV submission with probabilities. The task context says one customer request can produce several possible offers, so validation must check whether rows form repeated request/client/offer groups before trusting random CV.

Allowed tools: Python 3.10+ and open-source ML/preprocessing libraries. Do not use closed libraries, private APIs, paid external datasets, or borrowed code without a compatible open license. Use only training and test data provided by the task unless the user explicitly gives additional allowed files from the platform.

Important platform constraint: do not burn daily submissions. The case says no more than 3 platform uploads per day. Treat each exported submission as a scarce artifact and label it clearly.

## Task modes

Declare one mode before non-trivial work:

- `inspect_only` — read/analyze, no edits.
- `plan_only` — propose plan, no edits.
- `patch_small` — one local code/docs/config change.
- `data_quality_review` — schema, missingness, duplicates, train/test drift, target sanity.
- `eda_review` — relationships between features and acceptance, distribution checks.
- `leakage_review` — target leakage, test leakage, split leakage, ID/time leakage.
- `feature_engineering` — create/validate derived predictors.
- `cv_design` — validation strategy and split design.
- `baseline_build` — minimal reproducible baseline model.
- `model_training` — train/tune models and save artifacts.
- `ensemble_review` — blending/stacking and diversity checks.
- `metric_validation` — ROC-AUC and probability-output validation.
- `drift_adaptation` — adapt to train/test distribution shift; judge data ceiling.
- `submission_build` — generate and validate CSV submission.
- `experiment_tracking` — update experiment log and artifact registry.
- `docs_sync` — README/decision-log update after real behavior/contract change.
- `red_team` — adversarial review before important submission/use.

## Global working rules

1. Spec first for non-trivial work: goal, inputs, affected files, protected contracts, validation.
2. Inspect before patching. Do not change files by memory.
3. Prefer minimal localized changes. No broad refactor unless explicitly requested.
4. Separate EDA, training, validation, inference, and submission generation.
5. Never use `target_value` outside the training rows. Never infer test labels from sample submission.
6. Run full schema discovery. Do not restrict modeling to the representative columns named in the PDF; every train/test feature must be classified, used safely, or explicitly excluded with a reason.
7. Detect repeated applications, customers, or request-like groups that may correspond to multiple offers for one request. Use group-aware validation or stress tests when such structure exists.
8. Validate schema and row order before writing a submission.
9. Report what was checked, what was not checked, and residual risk.
10. Explanations to the project owner should be in Russian unless code/comments/file content require English.

## Protected contracts

These are mandatory across all agents:

- Every accepted result must be traceable to input files, script/notebook version, config, random seed, split definition, model version, feature list, metric output, and artifact paths.
- `target_value` must be used only as the supervised label in training/validation, never as a feature.
- Test-set features may be used only for schema/drift-aware preprocessing that does not inspect labels and does not create target-derived encodings from test outcomes.
- Submission must match `sample_submission.csv` columns, row count, row order or identifier mapping, and probability range `[0, 1]`.
- ROC-AUC must be computed on held-out validation folds with probability scores, not hard labels.
- Cross-validation must avoid leakage from duplicate IDs, repeated applications, repeated customer/request offer groups, temporal ordering, or target encodings.
- Categorical encodings, imputers, scalers, target encoders, and feature selectors must be fit inside each training fold/pipeline where applicable.
- Do not compare experiments if splits, features, preprocessing, target definition, group policy, or metric implementation differ without explicit labels.
- Do not select a model only by one lucky fold or one public submission. Inspect variance, calibration shape, and robustness.
- Any leaderboard upload recommendation must include a submission card with source commit/script, validation score, generated file hash, and known risks.
- README/decision-log updates must distinguish what is implemented, what is planned, and what is confirmed by checks.

## Feature engineering principles

Prioritize interpretable, leak-safe features after full schema discovery:

- rate economics: `offered_rate - cb_rate`, `offered_rate / cb_rate`, rate bins;
- limit economics: requested amount relative to `overdraft_limit_min/max`, within-limit flags, max/min spread;
- activity recency: 30/90 ratios, activity intensity, transaction-count ratios;
- stability/liquidity: log balances, min balance flags, investment/debit relationships;
- credit history: active products, loan payment counts, months from deal starts;
- behavioral features: dashboard events, time spent percentiles;
- offer-context diagnostics: if multiple offers per customer/request can be identified, validate within-group ranking and never leak accepted offer identity across folds;
- categorical handling for `db_group_last`, `fl_adminarea`, and any discovered categorical columns;
- missingness indicators where missingness is plausibly informative.

Use log/robust transforms for heavy-tailed monetary fields. Handle division by zero explicitly. Never create a feature that directly encodes the target or future outcomes.

## Validation levels

| Level | Meaning | Example |
|---|---|---|
| L0 | Static/document check | file tree, Markdown consistency, obvious contradictions |
| L1 | Local syntax/smoke | script imports, one small run, CLI `--help`, notebook executes first cells |
| L2 | Data/schema consistency | train/test columns, dtypes, missingness, duplicates, target distribution, submission shape |
| L3 | Reproducible CV validation | deterministic folds, per-fold ROC-AUC, out-of-fold predictions, saved config/artifacts |
| L4 | Robustness/regression | alternative splits, seed variance, feature ablation, drift checks, leakage checks, ensemble sanity |
| L5 | Submission readiness | red-team + submission card + hash + exact sample format + residual risk accepted |

Always state the achieved validation level. Do not use `PASS` when only README/agent files were inspected.

## Active agents

Use these first:

- `agents/context_router.md` — selects task mode, minimal context, primary agent, reviewer, validation level.
- `agents/architect.md` — task decomposition, scope control, protected contracts.
- `agents/task_spec_short.md` — compact task spec for non-trivial work.
- `agents/test_validation.md` — validation plan and evidence sufficiency.
- `agents/red_team.md` — adversarial review before important decisions/submission.
- `agents/readme_consistency_reviewer.md` — README/spec/decision-log drift control.
- `agents/decision_log_handoff.md` — reproducibility and handoff records.

This pack does not ship separate migration, changelog, or historical-diff files. Current instructions are authoritative.

Domain agents:

- `agents/data_quality.md` — CSV schema, missingness, duplicates, drift, target sanity.
- `agents/eda_analyst.md` — feature/target relationships and business interpretation.
- `agents/leakage_guard.md` — target/test/time/ID leakage review.
- `agents/feature_engineer.md` — safe feature design and pipeline implementation review.
- `agents/cv_validator.md` — split design and out-of-fold validation.
- `agents/baseline_builder.md` — minimal reproducible baseline.
- `agents/model_trainer.md` — model training, tuning, and artifact saving.
- `agents/model_ensembler.md` — blending/stacking and ensemble validation.
- `agents/metric_validator.md` — ROC-AUC/probability-output checks.
- `agents/submission_builder.md` — submission CSV generation and final checks.
- `agents/experiment_manager.md` — experiment registry, seeds, configs, run comparison.
- `agents/interpretability_reviewer.md` — feature importance/SHAP/PDP and business sanity.
- `agents/reproducibility_reviewer.md` — environment, deterministic rerun, dependency and artifact checks.
- `agents/drift_adaptation.md` — train/test distribution-shift adaptation and data-ceiling analysis.

## Routing examples

- New repository/data intake -> `context_router.md` -> `data_quality.md` + `test_validation.md`; include full schema discovery and repeated-offer/request checks.
- Baseline script request -> `baseline_builder.md` + `metric_validator.md`.
- Feature engineering patch -> `feature_engineer.md` + `leakage_guard.md`.
- Validation split design -> `cv_validator.md` + `leakage_guard.md`.
- CatBoost/LightGBM/XGBoost tuning -> `model_trainer.md` + `metric_validator.md`.
- Blending several models -> `model_ensembler.md` + `red_team.md`.
- Severe train/test drift or stalled improvement -> `drift_adaptation.md` + `red_team.md`.
- Generate final CSV -> `submission_builder.md` + `metric_validator.md`.
- Before platform upload -> `red_team.md` + `submission_builder.md` + `decision_log_handoff.md`.
- README/experiment log update -> `readme_consistency_reviewer.md` + `decision_log_handoff.md`.

## Decision states

Use these consistently:

- `BLOCK` — must not proceed; critical evidence, leakage, data, metric, or submission-format gap.
- `HOLD` — insufficient evidence; get more checks/data/output first.
- `RETEST` — plausible but requires rerun or stronger validation.
- `PASS_WITH_RISKS` — acceptable for the stated scope with explicit caveats.
- `PASS` — acceptable only for the reviewed scope; never means leaderboard success is guaranteed.

## Default review output

```markdown
## Verdict
<state> — one sentence.

## Critical
- ...

## Medium
- ...

## Minimal patch / action
- ...

## Validation
- Achieved level: L0/L1/L2/L3/L4/L5
- What was checked
- What remains unchecked

## Decision log
- Inputs, assumptions, artifacts, and next owner
```

## Embedded record templates

Keep operational records in repository files created by Codex rather than relying on separate template files in this pack:

- Experiment log columns: `run_id`, `timestamp`, `data_hash`, `split`, `seed`, `feature_set`, `model`, `params_ref`, `fold_auc`, `oof_auc`, `test_pred`, `submission`, `notes`.
- Submission card fields: file path, SHA256, generation script/config, source run id, fold/OOF ROC-AUC, leakage/metric/submission/red-team verdicts, sample-format checks, upload recommendation, daily upload count.
- Validation matrix fields: check, validation level, owner agent, evidence path, status, notes.

## Strict final rule

If required evidence is missing, say exactly what cannot be concluded. Do not fill missing data, metric, leakage, or submission facts by assumption.
