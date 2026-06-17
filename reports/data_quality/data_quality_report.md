# Data quality report

## Verdict

PASS_WITH_RISKS for train/test modeling data; HOLD for final submission policy until the platform row contract is confirmed.

## Files

- train: `/Users/bobrsubr/PycharmProjects/_researches/loan-offer-acceptance-prediction/data/raw/train_apps.csv`, size_mb=43.81, sha256=`988e874ae9a877642537822ae5399435839799c32fded503feff7ccdf88506d7`
- test: `/Users/bobrsubr/PycharmProjects/_researches/loan-offer-acceptance-prediction/data/raw/test_apps.csv`, size_mb=11.78, sha256=`74f788c6d38a4ce2784804446d8a257715138aaa43580035bcbe59765c890904`
- sample: `/Users/bobrsubr/PycharmProjects/_researches/loan-offer-acceptance-prediction/data/raw/sample_submission.csv`, size_mb=0.52, sha256=`ea2e7721fd1a68386145c838fd847009ad1564c77ca49a416bd5a6c638068164`

## Inventory

- train shape: (145241, 28)
- test shape: (36311, 27)
- sample_submission shape: (45032, 2)
- train/test schema compatible excluding target: True
- target in train: True
- target in test: False

## Target sanity

- target counts: {'0': 136395, '1': 8846}
- target unique values: [0, 1]
- positive rate: 0.060906

## Decision day

- train: {'dataset': 'train', 'exists': True, 'missing': 0, 'parse_failures': 0, 'nunique': 485, 'min': '2024-02-01', 'max': '2025-06-05'}
- test: {'dataset': 'test', 'exists': True, 'missing': 0, 'parse_failures': 0, 'nunique': 177, 'min': '2025-06-05', 'max': '2025-12-01'}
- train/test day intersection: 1
- test-only days: 176

## Duplicate and repeated-context checks

- train_full_duplicate_rows: 0
- test_full_duplicate_rows: 0
- train_duplicate_feature_rows_excluding_id_target: 2416
- test_duplicate_feature_rows_excluding_id: 428
- train_duplicate_feature_groups_with_conflicting_target: 47
- train_duplicate_feature_rows_with_conflicting_target: 96
- feature_signature_overlap_train_test_groups: 0
- train_rows_with_feature_signature_in_test: 0
- test_rows_with_feature_signature_in_train: 0

### Repeated context

- train: {'context_groups': 139607, 'repeated_context_groups': 3181, 'rows_in_repeated_context_groups': 8815, 'max_context_group_size': 19, 'groups_with_varying_offer_terms': 2130, 'rows_in_groups_with_varying_offer_terms': 6640, 'groups_with_conflicting_target': 147, 'rows_in_conflicting_target_groups': 449}
- test: {'context_groups': 35659, 'repeated_context_groups': 504, 'rows_in_repeated_context_groups': 1156, 'max_context_group_size': 8, 'groups_with_varying_offer_terms': 324, 'rows_in_groups_with_varying_offer_terms': 784}
- train/test context overlap groups: 0

## Sample submission compatibility

- sample_columns: ['front_id', 'target_value']
- test_rows: 36311
- sample_rows: 45032
- sample_has_front_id: True
- sample_has_target_value: True
- status: HOLD_SAMPLE_CONTAINS_EXTRA_IDS_BUT_TEST_BLOCK_IS_ALIGNED
- train_test_overlap_ids: 0
- test_sample_overlap_ids: 36311
- train_sample_overlap_ids: 8721
- sample_minus_test_ids: 8721
- test_minus_sample_ids: 0
- sample_front_id_unique: True
- sample_target_nunique: 1
- sample_target_counts: {'0.5': 45032}
- sample_target_min: 0.5
- sample_target_max: 0.5
- sample_rows_in_test: 36311
- sample_rows_not_in_test: 8721
- sample_test_subset_same_order_as_test: True

## Recommended validation policy

- Primary validation should be time-aware because test is future-dated relative to train.
- Random StratifiedKFold may be used only as a sanity check, not as the main model-selection score.
- Repeated context groups exist; avoid splitting same-day sibling offers across folds when using non-temporal CV.
- Exclude `front_id` from features.
- Treat `decision_day` as a temporal feature/risk; raw string usage is not recommended.

## Generated CSV artifacts

- `schema_train_test.csv`
- `missingness.csv`
- `nunique.csv`
- `numeric_ranges.csv`
- `categorical_overlap.csv`
- `target_by_month.csv`
- `test_rows_by_month.csv`

## Validation

- Achieved level: L2 partial
- Checked: file presence, schema, target sanity, missingness, dtypes, constants, duplicates, temporal split, categorical overlap, repeated context, sample/test mapping.
- Remaining: official submission row contract, baseline CV implementation, leakage review of future feature code.
