# External scoring diagnostic — 2026-06-19

## Verdict

HOLD / negative — даже при предположении, что датасет `Кредитный скоринг` разрешён, его агрегаты по `id -> front_id` не улучшают offer acceptance CV. Полный набор агрегатов ухудшает Fold3 с `0.755030` до `0.748100`; top-K отбор не даёт прироста.

## Inputs

- Scoring dataset: `/Users/bobrsubr/PycharmProjects/_researches/credit-default-prediction/data/Кредитный скорринг`
- Offer dataset: `/Users/bobrsubr/PycharmProjects/_researches/loan-offer-acceptance-prediction/data/Кредитный оффер`
- Offer model config: `configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1.json`
- Assumption: external scoring parquet features are allowed.
- Safety constraint used in this diagnostic: `train_target.flag` from scoring is not used as a model feature.

## Data bridge

Scoring parquet schemas:

- `train_data.parquet`: 18,317,016 rows, 61 columns.
- `test_data.parquet`: 7,845,701 rows, 61 columns.
- Key candidate: scoring `id` matched to offer `front_id`.

Filtered matched rows:

- scoring matched rows: 1,451,923;
- aggregate IDs: 181,552;
- offer train coverage: 145,241 / 145,241 = 100%;
- offer test coverage: 36,311 / 36,311 = 100%.

Coverage by itself is not proof of semantic identity: scoring IDs span `0..2,999,999`, while offer front IDs are a dense subrange around `100k..281k`.

## Feature construction

Aggregated by scoring `id`:

- `rn`: count, max;
- all other scoring columns: mean, max, min;
- additional overdue row flags.

Generated external aggregate features: 181 scoring features + `scoring_has_history`.

Artifacts:

- `reports/validation/external_scoring/external_scoring_aggregates_20260619T033923Z.csv`
- `reports/validation/external_scoring/external_scoring_xgb_cv_report_20260619T033923Z.json`
- `reports/validation/external_scoring/external_scoring_topk_xgb_report_20260619T034123Z.json`

## CV result — full aggregate set

| feature_set | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| base | 0.794574 | 0.791883 | **0.755030** | **0.779816** |
| base + external scoring | 0.792335 | 0.788748 | 0.748100 | 0.776176 |

Result: full external scoring feature set is consistently worse.

## CV result — top-K external scoring features

Top-K features were selected diagnostically by univariate absolute AUC on folds 1–2, then added to the base XGB feature set.

| K | Fold1 | Fold2 | Fold3 | OOF |
|---:|---:|---:|---:|---:|
| 1 | 0.794054 | 0.793544 | 0.755018 | **0.779982** |
| 3 | 0.793018 | 0.791482 | 0.753628 | 0.778598 |
| 5 | 0.794500 | 0.792853 | 0.753452 | 0.779572 |
| 10 | 0.793193 | 0.790830 | 0.752755 | 0.778310 |
| 20 | 0.793716 | 0.789178 | 0.752839 | 0.778119 |
| 50 | 0.793032 | 0.791006 | 0.749241 | 0.777303 |
| 100 | 0.792574 | 0.789245 | 0.748548 | 0.776382 |

Best top-K (`K=1`) is effectively flat on Fold3 (`0.755018` vs base `0.755030`) and not a real improvement.

## Univariate signal check

The strongest external features are weak:

- best folds 1–2 selector AUC: about `0.532`;
- same feature Fold3 absolute AUC: about `0.512`;
- most top features decay to near-random on Fold3.

This pattern suggests the `id -> front_id` bridge is not a useful semantic customer/application link for this offer target, or the scoring history encodings are irrelevant/noisy for acceptance.

## Interpretation

The external scoring dataset does not explain the leaderboard gap under the tested bridge:

1. Direct scoring target join was already near random (`AUC ≈ 0.4987`).
2. Full raw-history aggregates hurt all folds.
3. Top-K selected aggregates do not improve Fold3.

If there is a way to use the scoring dataset productively, it likely requires a different, currently unknown key or mapping. Simple `scoring.id == offer.front_id` is not enough.

## Minimal action

- Do not use these external scoring aggregates in the current submission candidate.
- Do not spend upload slots on scoring-augmented models from this bridge.
- If continuing this line, first find a documented shared key or ask organizers whether `id` and `front_id` are intended to refer to the same entity.

## Validation

- Achieved level: L3 diagnostic.
- Checked: parquet schema, matched row extraction, ID coverage, aggregate generation, XGB time CV full features, XGB time CV top-K features.
- Not checked: CatBoost with external aggregates; not prioritized because XGB full/top-K already shows no useful Fold3 signal and univariate signal is weak.
