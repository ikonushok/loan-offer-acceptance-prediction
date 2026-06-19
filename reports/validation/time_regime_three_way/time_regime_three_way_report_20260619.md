# 3-way rank-blend scan: champion + XGB HPO + XGB time-regime — 2026-06-19

## Verdict

HOLD — 3-way blend не дал нового кандидата лучше текущего `upload3`: оптимальный вес для `xgb_hpo` равен `0.00`, а лучший результат совпадает с уже существующим `candidate_20260620_upload3_RETEST_xgb_time_regime_rank_c20_x80.csv`.

## Setup

- Script: `scripts/scan_time_regime_three_way_blend.py`
- Components:
  - `champion`: `experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z`
  - `xgb_hpo`: `experiments/runs/xgb_hpo_depth3_child80_reg30_v1_20260617T235100Z_seed42`
  - `xgb_time_regime`: `experiments/runs/xgb_hpo_time_regime_probe_full_20260619T0420Z_seed42`
- Validation: aligned Fold1/Fold2/Fold3 OOF + late-holdout H1/H2/H3.
- No leaderboard signal used.

## Best weights

| scan | w_champion | w_xgb_hpo | w_xgb_time_regime | Fold3 | OOF | lh_mean | lh_min |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 grid | 0.20 | 0.00 | 0.80 | 0.758034 | 0.781782 | 0.766511 | 0.757916 |
| 0.01 grid | 0.20 | 0.00 | 0.80 | 0.758034 | 0.781782 | 0.766511 | 0.757916 |

Interpretation: base `xgb_hpo` does not add incremental value once `xgb_time_regime` is included. The best 3-way scan collapses to the existing 2-way time-regime blend.

## Current upload comparison

| file | weights | SHA256 | status |
|---|---|---|---|
| `candidate_20260620_upload3_RETEST_xgb_time_regime_rank_c20_x80.csv` | champion 0.20 + time-regime 0.80 | `46d7f44306626b6a2b6b65bb2aaa4defca61165b58988c081de16a770fc8668b` | existing candidate |
| `candidate_20260620_upload4_RETEST_three_way_c20_x00_t80.csv` | champion 0.20 + XGB HPO 0.00 + time-regime 0.80 | `46d7f44306626b6a2b6b65bb2aaa4defca61165b58988c081de16a770fc8668b` | duplicate, do not count as new candidate |

## Artifacts

- Coarse scan: `reports/validation/time_regime_three_way/time_regime_three_way_scan_20260619T052410Z.csv`
- Fine scan: `reports/validation/time_regime_three_way/time_regime_three_way_scan_20260619T052435Z.csv`
- Correlations: `reports/validation/time_regime_three_way/time_regime_three_way_correlations_20260619T052435Z.csv`
- JSON reports: `reports/validation/time_regime_three_way/time_regime_three_way_report_20260619T052410Z.json`, `reports/validation/time_regime_three_way/time_regime_three_way_report_20260619T052435Z.json`

## Minimal action

- Keep `upload3` as the strongest offline candidate.
- Do not upload `upload4`; it is byte-identical to `upload3`.
- Do not add base `xgb_hpo` into the time-regime blend unless a new time-regime seed/HPO changes the correlation structure.

## Validation

- Achieved level: L4 diagnostic for blend scan.
- Checked: OOF folds, late-holdout H1/H2/H3, test row/order/probability checks for generated duplicate.
- Not checked: red-team review of `upload3` extrapolation risk remains required before upload.
