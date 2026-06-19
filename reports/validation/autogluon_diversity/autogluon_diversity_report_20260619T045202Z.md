# AutoGluon diversity probe — 2026-06-19

## Verdict

HOLD — AutoGluon Tabular `good_quality` на нашей context-offer feature policy не проходит diversity gate: solo Fold3 ниже champion/XGB, корреляция высокая, simplex rank-blend не выбирает AutoGluon как полезный компонент.

## Setup

- Script: `scripts/baseline_autogluon_time.py`
- Config: `configs/diversity_experiments/autogluon_context_offer_good_v1.json`
- Run: `experiments/runs/autogluon_context_offer_good_v1_20260619T045202Z_seed42`
- Features: same context-offer policy as current XGB diversity branch; no time features, no missing flags, no pairwise rate/limit features.
- AutoGluon: `autogluon.tabular==1.5.0`, `good_quality`, `num_bag_folds=0`, `num_stack_levels=0`, excluded `FASTAI`/`NN_TORCH`; PyTorch unavailable in current venv.

## Fold metrics

| model | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| champion | 0.795951 | 0.792685 | 0.754800 | 0.781140 |
| xgb_hpo | 0.794574 | 0.791883 | 0.755030 | 0.779816 |
| autogluon | 0.789588 | 0.790543 | 0.743211 | 0.773520 |

## Diversity check

| pair | OOF pearson | OOF spearman | Fold3 spearman |
|---|---:|---:|---:|
| champion vs xgb_hpo | 0.876195 | 0.962131 | 0.958875 |
| champion vs autogluon | 0.869219 | 0.942354 | 0.921271 |
| xgb_hpo vs autogluon | 0.949477 | 0.932747 | 0.906581 |

AutoGluon is slightly less rank-correlated with champion on Fold3 than XGB is, but its Fold3 quality loss is too large to pay for the diversification.

## Rank-blend scan

Named rank blends, rank-normalized within each fold:

| blend | w_champion | w_xgb_hpo | w_autogluon | Fold3 | OOF |
|---|---:|---:|---:|---:|---:|
| champion_only | 1.00 | 0.00 | 0.00 | 0.754800 | 0.780124 |
| xgb_only | 0.00 | 1.00 | 0.00 | 0.755030 | 0.779453 |
| autogluon_only | 0.00 | 0.00 | 1.00 | 0.743211 | 0.772941 |
| c35_x65 | 0.35 | 0.65 | 0.00 | 0.756678 | 0.781303 |
| c30_x70 | 0.30 | 0.70 | 0.00 | 0.756588 | 0.781147 |

Best simplex by Fold3 on 0.05 grid: `w_champion=0.50`, `w_xgb_hpo=0.50`, `w_autogluon=0.00`, Fold3 `0.756791`, OOF `0.781540`.

## Artifacts

- Metrics: `reports/validation/autogluon_diversity/autogluon_diversity_metrics_20260619T045202Z.csv`
- Correlations: `reports/validation/autogluon_diversity/autogluon_diversity_correlations_20260619T045202Z.csv`
- Simplex scan: `reports/validation/autogluon_diversity/autogluon_diversity_simplex_scan_20260619T045202Z.csv`
- Named blends: `reports/validation/autogluon_diversity/autogluon_diversity_named_blends_20260619T045202Z.csv`
- JSON summary: `reports/validation/autogluon_diversity/autogluon_diversity_report_20260619T045202Z.json`

## Minimal action

- Do not promote this AutoGluon run to late-holdout/submission candidate.
- If revisiting AutoGluon, change the hypothesis first: higher-resource bagging/stacking or different feature policy. Do not spend uploads on the current run.

## Validation

- Achieved level: L3 diagnostic.
- Checked: three rolling folds, OOF AUC, pairwise correlations, rank-blend simplex scan.
- Not checked: late-holdout H1/H2/H3, because Fold3 and blend gate failed.
