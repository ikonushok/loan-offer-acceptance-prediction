# Fold-safe segment stats + target encoding probe — 2026-06-19

## Verdict

HOLD — идеи из внешнего решения проверены в fold-safe постановке, но текущая реализация не проходит gate: Fold3 ниже champion/XGB, корреляция с champion слишком высокая, rank-blend не выбирает эту ветку как полезный компонент.

## Setup

- Script: `scripts/catboost_foldsafe_segment_te_time.py`
- Config: `configs/feature_experiments/catboost_trial0070_foldsafe_segment_te_v1.json`
- Run: `experiments/runs/catboost_trial0070_foldsafe_segment_te_v1_20260619T051058Z_seed42`
- Base feature policy: context-offer features, no time features, no missing flags, no pairwise rate/limit features.
- Added fold-safe features: 86 columns.
- Segment stats: fit on rolling-fold train part only, map validation/test from train statistics.
- Target encoding: OOF inside each rolling train part for train rows; validation/test map from rolling train part only.

## Fold metrics

| model | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| champion | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| xgb_hpo | 0.794574 | 0.791883 | 0.755030 | 0.779816 |
| foldsafe_segment_te | 0.796248 | 0.788172 | 0.747927 | 0.777698 |

Interpretation: Fold1 gets a small local gain, but Fold2/Fold3 deteriorate. This is consistent with segment/TE features being temporally unstable rather than robust signal.

## Diversity check

| pair | OOF pearson | OOF spearman | Fold3 spearman |
|---|---:|---:|---:|
| champion vs xgb_hpo | 0.876195 | 0.962131 | 0.958875 |
| champion vs foldsafe_segment_te | 0.975547 | 0.975104 | 0.972597 |
| xgb_hpo vs foldsafe_segment_te | 0.897417 | 0.949666 | 0.944942 |

The new branch is almost a champion clone in rank space while being weaker on late Fold3, so it does not add useful diversity.

## Rank-blend scan

Named rank blends, rank-normalized within each fold:

| blend | w_champion | w_xgb_hpo | w_foldsafe_segment_te | Fold3 | OOF |
|---|---:|---:|---:|---:|---:|
| champion_only | 1.00 | 0.00 | 0.00 | 0.754800 | 0.780124 |
| xgb_only | 0.00 | 1.00 | 0.00 | 0.755030 | 0.779453 |
| foldsafe_segment_te_only | 0.00 | 0.00 | 1.00 | 0.747927 | 0.776753 |
| c35_x65 | 0.35 | 0.65 | 0.00 | 0.756678 | 0.781303 |
| c50_x50 | 0.50 | 0.50 | 0.00 | 0.756791 | 0.781540 |

Best simplex by Fold3 on 0.05 grid: `w_champion=0.50`, `w_xgb_hpo=0.50`, `w_foldsafe_segment_te=0.00`, Fold3 `0.756791`, OOF `0.781540`.

## Artifacts

- Metrics: `reports/validation/foldsafe_segment_te/foldsafe_segment_te_metrics_20260619T051058Z.csv`
- Correlations: `reports/validation/foldsafe_segment_te/foldsafe_segment_te_correlations_20260619T051058Z.csv`
- Simplex scan: `reports/validation/foldsafe_segment_te/foldsafe_segment_te_simplex_scan_20260619T051058Z.csv`
- Named blends: `reports/validation/foldsafe_segment_te/foldsafe_segment_te_named_blends_20260619T051058Z.csv`
- JSON summary: `reports/validation/foldsafe_segment_te/foldsafe_segment_te_report_20260619T051058Z.json`

## Minimal action

- Do not promote this feature branch to late-holdout/submission.
- Do not spend upload slots on this run.
- If revisiting, isolate only label-free segment stats or only target encoding; current combined variant is too drift-sensitive.

## Validation

- Achieved level: L3 diagnostic.
- Checked: three rolling folds, OOF AUC, pairwise correlations, rank-blend simplex scan.
- Not checked: late-holdout H1/H2/H3, because Fold3 and blend gate failed.
