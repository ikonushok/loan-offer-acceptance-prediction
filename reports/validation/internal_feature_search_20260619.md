# Internal feature search — 2026-06-19

## Verdict

RETEST — найден новый внутренний сигнал: XGB с `time_regime` признаками (`decision_day_num`, `decision_month`, `decision_week`) улучшает Fold3 и late-holdout относительно базового XGB. Лучший late rank-blend с champion — `champion_rank * 0.20 + xgb_time_regime_rank * 0.80`.

## Scope

Mode: `feature_engineering`.

Цель: искать улучшения внутри offer-задачи без внешних данных, без target leakage и без public-LB probing.

Проверенные семьи признаков:

- signed log transforms для денежных колонок;
- zero flags + row missing count;
- unsupervised frequency encodings;
- quantile bins;
- categorical interactions;
- time/regime features;
- combined label-free feature set.

## Feature-family probe

Base config: `configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1.json`.

Artifacts:

- `reports/validation/feature_family_probes/feature_family_probe_20260619T041432Z_compact.csv`
- `reports/validation/feature_family_probes/feature_family_probe_20260619T041432Z_report.json`

| variant | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| base | 0.794574 | 0.791883 | 0.755030 | 0.779816 |
| log | 0.793732 | 0.793122 | 0.754032 | 0.779407 |
| zero_flags | 0.794900 | 0.793431 | 0.755401 | 0.780477 |
| frequency | 0.793336 | 0.793431 | 0.754577 | 0.779461 |
| quantile_bins | 0.795977 | 0.792244 | 0.754368 | 0.780497 |
| cat_interactions | 0.793475 | 0.791528 | 0.752779 | 0.778729 |
| all_label_free | 0.796376 | 0.792156 | 0.756214 | 0.781446 |
| time_regime | 0.792971 | **0.796512** | **0.757085** | 0.780618 |

Interpretation:

- `time_regime` is the best primary-Fold3 variant.
- `all_label_free` improves OOF/Fold3 but is less robust than isolated `time_regime`.
- log/frequency/categorical interactions alone are not useful.

## Blend scan vs champion

Artifact:

- `reports/validation/feature_family_probes/feature_probe_blend_scan_20260619T041758Z.csv`

Best OOF/Fold3 rank-blend by variant:

| variant | w_candidate | corr vs champion | OOF | Fold1 | Fold2 | Fold3 |
|---|---:|---:|---:|---:|---:|---:|
| time_regime | 0.68 | 0.871842 | 0.782711 | 0.795337 | 0.796892 | **0.758215** |
| all_label_free | 0.64 | 0.875896 | 0.783109 | 0.797671 | 0.794339 | 0.757669 |
| zero_flags | 0.56 | 0.877852 | 0.782475 | 0.796740 | 0.794907 | 0.757011 |
| base | 0.54 | 0.876195 | 0.782154 | 0.796575 | 0.794075 | 0.756748 |

Interpretation:

- `time_regime` is the strongest primary-Fold3 blend.
- Correlation remains high, so expected gain is incremental, not a leaderboard-gap solution.

## Late-holdout check

Standalone `xgb_time_regime` late holdout:

| model | H1 | H2 | H3 | lh_mean | lh_min |
|---|---:|---:|---:|---:|---:|
| xgb_base | 0.765375 | 0.755030 | 0.769460 | 0.763289 | 0.755030 |
| xgb_time_regime | **0.770766** | **0.757085** | **0.770680** | **0.766177** | **0.757085** |

Artifact:

- `reports/validation/late_holdouts/late_holdouts_xgb_hpo_time_regime_probe_20260619T041911Z`

Late rank-blend scan:

| candidate | w_candidate | H1 | H2 | H3 | lh_mean | lh_min |
|---|---:|---:|---:|---:|---:|---:|
| xgb_time_regime | 0.80 | 0.770481 | 0.757915 | 0.771134 | **0.766510** | **0.757915** |
| xgb_base | 0.69 | 0.766194 | 0.756296 | 0.769920 | 0.764137 | 0.756296 |

Artifact:

- `reports/validation/feature_family_probes/time_regime_late_blend_scan_20260619T041949Z.csv`

Interpretation:

- `time_regime` dominates base XGB on all late holdouts.
- Best late weight is higher (`0.80`) than OOF/Fold3 weight (`0.68`), consistent with public results favoring stronger XGB weight.

## Candidate artifact

Generated RETEST candidate:

- `submissions/candidate_20260620_upload3_RETEST_xgb_time_regime_rank_c20_x80.csv`
- card: `submissions/cards/candidate_20260620_upload3_RETEST_xgb_time_regime_rank_c20_x80_card.json`
- SHA256: `46d7f44306626b6a2b6b65bb2aaa4defca61165b58988c081de16a770fc8668b`

Submission checks:

- rows: 36,311;
- columns: `front_id,target_value`;
- `front_id` order matches `test_apps.csv`;
- probabilities in `[0, 1]`;
- NaN count: 0;
- unique predictions: 33,852.

## Risks

- `time_regime` uses calendar features and test extends to unseen future months; this can overfit the local validation period.
- The best blend weight is selected by late-holdout scan; red-team review is required before upload.
- This is an incremental candidate. It does not explain the leader gap to `0.789188`.

## Minimal action

- Treat `candidate_20260620_upload3_RETEST_xgb_time_regime_rank_c20_x80.csv` as the next best internal-task candidate.
- Do not upload before red-team review.
- If upload budget is scarce, compare against already planned `c35_x65`/`c30_x70`: time-regime candidate has stronger offline late metrics than base-XGB blend.

## Validation

- Achieved level: L4 diagnostic + submission contract check.
- Checked: rolling-fold CV, OOF, rank-blend scan, late-holdout battery, late rank-blend scan, CSV contract.
- Not checked: red-team review and public/private behavior.
