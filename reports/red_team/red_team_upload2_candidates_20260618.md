# Red-team review — Upload #2 candidates

**Date:** 2026-06-18  
**Reviewed by:** red_team agent  
**Scope:** Two upload #2 candidates — `candidate_20260619_upload2_RETEST_xgb_rank_c62_x38` and `candidate_20260619_upload2_RETEST_lgbm_diversity_c88_u11_s01`

---

## Red-team verdict

**XGB rank blend:** `PASS_WITH_RISKS` — late-holdout battery now complete and positive. Upload is justified if candidate #1 does not already improve materially.

**LGBM diversity blend:** `RETEST` — late-holdout for LGBM components not yet computed. Do not upload without it.

---

## Kill shots examined

### 1. Target leakage

**Status: PASS**

- `target_value` is excluded from features in all runs. Verified in `feature_manifest.csv` for CatBoost and XGB runs — target not listed in used features.
- `decision_day` is excluded as a raw feature (`excluded_features: ["decision_day"]` in all champion configs). Only safe derivatives (`decision_month`, `decision_day_num`) used in the baseline, and these are excluded in the `no_time_features` / context-offer champion variant.
- No test labels were used at any stage (confirmed: test CSV has no `target_value` column per data quality report).

### 2. Future information / temporal leakage

**Status: PASS_WITH_RISKS**

- Rolling time folds have zero context overlap (verified: `validation_drift_audit_report.md` — all three folds show 0 overlap groups under the context signature).
- `decision_day` string excluded from champion features. Context-offer features (`offered_rate - cb_rate`, etc.) are computed within fold boundaries (fold-safe, no leakage from validation to train).
- **Residual risk:** `cb_rate` has adversarial AUC 0.63 as a train/test period proxy. The champion keeps `cb_rate` as a feature; the model learns `cb_rate` levels that may be non-stationary. This is acknowledged but not a leakage issue — it is a distribution shift issue.
- **Residual risk:** `decision_day` appears in SHAP top-10 for the baseline model (`decision_month` SHAP mean=0.170, `decision_day_num` SHAP mean=-0.154). The champion config explicitly excluded time features (`time_features.enabled: false`), which ablation verified improves Fold3 AUC. This risk is mitigated.

### 3. Validation representative of platform test distribution

**Status: PASS_WITH_RISKS**

- Test `decision_day` range: 2025-06-05 to 2025-12-01, train ends 2025-06-05. Test is almost entirely future-dated — confirmed by data quality report.
- Fold3 (2025-04-01 to 2025-06-05) is the most future-like fold in train, making it the best proxy for platform test. OOF average is less representative.
- Late holdouts H1/H2/H3 provide extra future-oriented validation. XGB rank blend improved on all three (H1 +0.0030, H2 +0.0020, H3 +0.0038).
- **Residual risk:** Fold3 still ends at 2025-06-05 (same as latest train date). The platform test extends to 2025-12-01, a further 6 months beyond Fold3. No local data covers this gap. True generalization to Q4 2025 cannot be validated offline.

### 4. CV splitting correctness

**Status: PASS**

- Rolling time splits confirmed: cutoffs [2025-01-01, 2025-03-01, 2025-04-01], train always precedes validation in all folds. No random CV used for final model selection.
- Repeated context groups (sibling offers from same client/day): present in train (~3181 groups, 8815 rows) but train/test context overlap is zero. Within-fold sibling splitting is possible but affects only ~6% of train rows and no bleed to test is possible.

### 5. Preprocessing correctness

**Status: PASS**

- All feature engineering (rate spreads, limit ratios, context-offer features) is computed identically from raw columns for both train and test — no supervised statistics, no target-conditioned transforms.
- Categorical features (`db_group_last`, `fl_adminarea`) are handled natively by CatBoost or via `pd.CategoricalDtype` fit on train+test union for LGBM/XGB — correct, no label encoding leakage.
- Missing flags (`add_missing_flags: false` in champion config) — not used, not a leakage risk.

### 6. Improvement vs noise

**Status: PASS_WITH_RISKS for XGB blend**

- XGB rank blend Fold3 improvement over champion: +0.0012 on rolling CV, +0.0038 on H3 late holdout.
- OOF improvement: +0.0009.
- Late-holdout mean improvement: +0.0029 (all three holdouts positive).
- Seed variance was not explicitly tested for the XGB blend. The champion used seed=42 throughout; XGB also seed=42. A single-seed result.
- **Residual risk:** blend weight 0.62/0.38 was selected on OOF+Fold3 scan — not on late holdouts. The late-holdout result confirms the direction but the exact weight may be slightly overfit to OOF.

### 7. Submission format

**Status: PASS**

- XGB rank blend card confirms: 36311 rows, columns `[front_id, target_value]`, `front_id_unique=true`, no NaN/inf, `front_id` order matches `test_apps.csv`.
- SHA256: `901015ff18f184a7bbcce5a6505871b62d90bf2c5d0dc72ca4eaee6f179df026` — recorded in card.
- Score range: min=0.000151, max=0.999931 (rank percentile blend — this is expected; values are near 0 and 1 due to rank normalization).
- **Note:** rank-normalized scores approach 0 and 1 at extremes. This is mathematically valid for ROC-AUC (which is purely rank-based) but the score distribution differs from raw probability blends. No issue expected for AUC evaluation, but worth noting.

### 8. Public leaderboard as HPO loop

**Status: PASS**

- Upload #1 (raw unrounded) was identified as a format improvement, not a model change. Upload #2 (XGB blend) is based on offline Fold3 + late-holdout evidence, not on public score feedback. No iterative public-score tuning detected.
- Plan explicitly states: "Do not use public leaderboard as hyperparameter search loop."

### 9. Suspicious top features

**Status: PASS_WITH_RISKS**

- SHAP top features in baseline: `loan_amount_last` (SHAP 0.345), `overdraft_limit_max` (0.225), `cb_rate` (0.208). These are economically meaningful (loan pricing, capacity, rate environment).
- `cb_rate` with SHAP 0.208 and adversarial AUC 0.630 is a joint period proxy and legitimate feature. Its exclusion (`no_cb_rate` ablation) reduced OOF by -0.020, confirming genuine signal. Kept appropriately.
- `decision_month` and `decision_day_num` excluded in champion via `time_features.enabled: false`. Ablation confirmed +0.002 Fold3 improvement after exclusion.
- Context-offer features (`offered_rate - cb_rate` etc.) add real economic signal; SHAP analysis not shown for champion config but ablation confirms contribution.

### 10. Reproducibility

**Status: PASS**

- All runs use `seed=42` consistently.
- Config JSON files stored in `configs/`, used and version-logged in run summaries.
- SHA256 hashes of train/test CSVs recorded in all late-holdout summaries and submission cards.

---

## Required before upload #2 (XGB rank blend)

- [x] Late-holdout battery complete (computed by `compute_blend_late_holdout.py`, all three holdouts positive)
- [x] Submission format check (36311 rows, correct columns, no NaN, SHA256 recorded)
- [x] Candidate #1 result known before using upload slot #2
- [ ] Seed variance check on XGB rank blend (nice-to-have, not blocking)

## Required before upload #2 (LGBM diversity blend)

- [ ] **BLOCKING:** Run `evaluate_late_holdouts_lgbm.py` for `lgbm_context_offer_unbalanced_v1` and `lgbm_context_offer_sqrtpos_v1` (needs lightgbm on user machine)
- [ ] Re-run `compute_blend_late_holdout.py` after LGBM late holdouts are available
- [ ] Late-holdout verdict must be PASS_WITH_RISKS before any upload

---

## Nice-to-have robustness checks

- Seed bag (2–3 seeds) for XGB unweighted component to reduce single-seed variance.
- Verify blend weight stability: re-scan 0.55–0.70 champion weight on late holdouts (not OOF) to confirm 0.62 is not OOF-overfit.
- Check XGB rank blend test prediction distribution percentiles against champion to confirm no systematic shift.

---

## Residual risk if proceeding with XGB rank blend upload

- Fold3 + late-holdout improvements are consistent and meaningful (+0.0012 to +0.0038).
- Score scale changes due to rank normalization; ROC-AUC is rank-invariant so this should not matter.
- Single seed, single XGB config — no seed/model variance quantified.
- Public result may differ from offline due to Q4 2025 distribution shift beyond training data range.
- Expected public gain: consistent direction confirmed, magnitude unknown. Conservative estimate: +0.001 to +0.003 over upload #1 if upload #1 already improved.

---

## Validation

- Achieved level: **L5 partial** (format check + L4 late-holdout battery + L3 interpretability review)
- Evidence inspected:
  - `reports/data_quality/data_quality_report.md`
  - `reports/validation/validation_drift_audit_report.md`
  - `reports/validation/shap_fold3_catboost_time_baseline_*_summary.json` and `_importance.csv`
  - `submissions/cards/candidate_20260619_upload2_RETEST_xgb_rank_c62_x38_card.json`
  - `reports/validation/blend_late_holdout_comparison_20260618.md`
  - `reports/validation/xgb_diversity_ablation_20260618.md`
  - `reports/validation/experiment_comparison.csv`
  - All champion component late-holdout metrics CSVs
- Evidence missing:
  - LGBM component late holdouts (blocking for LGBM candidate)
  - Seed variance for XGB blend
  - Late-holdout weight stability scan
