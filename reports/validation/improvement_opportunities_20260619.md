# Проверка оставшихся возможностей улучшения — 2026-06-19

## Verdict

HOLD — явного локального рычага уровня `+0.02..0.03 AUC` не найдено; гипотеза hard recent training закрыта отрицательно, простые time/regime postprocess дают только шумовой прирост.

## Контекст

- Public best проекта: `76.388`.
- Известный лидер: `0.789188` за 2 попытки.
- Full-sample submission exploit закрыт ответом организаторов: отправлять нужно только строки из `test_apps.csv`.
- Цель проверки: найти оставшиеся не-HPO рычаги, которые могли бы объяснить разрыв.

## Проверка 1 — hard recent-window training

Проверены окна `90/120/180/240/365` дней перед каждым validation fold против expanding baseline.

### XGBoost HPO

Артефакты:

- `reports/validation/recent_windows/recent_window_xgb_20260619T031843Z_compact.csv`
- `reports/validation/recent_windows/recent_window_xgb_20260619T031843Z_report.json`

| window | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| expanding | 0.794574 | 0.791883 | **0.755030** | **0.779816** |
| 90d | 0.782625 | 0.769653 | 0.732890 | 0.763168 |
| 120d | 0.788371 | 0.772625 | 0.737342 | 0.766922 |
| 180d | 0.792278 | 0.779486 | 0.736908 | 0.769799 |
| 240d | 0.793652 | 0.781844 | 0.743178 | 0.773223 |
| 365d | 0.794574 | 0.793374 | 0.753531 | 0.779680 |

Вывод: hard-window ухудшает primary Fold3. Лучший вариант остаётся expanding.

### CatBoost trial0070

Артефакты:

- `reports/validation/recent_windows/recent_window_catboost_20260619T032004Z_compact.csv`
- `reports/validation/recent_windows/recent_window_catboost_20260619T032004Z_report.json`

| window | Fold1 | Fold2 | Fold3 | OOF |
|---|---:|---:|---:|---:|
| expanding | **0.796080** | **0.791001** | **0.752437** | **0.779990** |
| 90d | 0.786020 | 0.772827 | 0.741321 | 0.767666 |
| 120d | 0.790443 | 0.778609 | 0.740264 | 0.767846 |
| 180d | 0.792925 | 0.783051 | 0.738731 | 0.772074 |
| 240d | 0.794079 | 0.785412 | 0.745761 | 0.775597 |
| 365d | 0.796080 | 0.789843 | 0.749678 | 0.778961 |

Вывод: hard-window также ухудшает CatBoost. Это не объясняет лидерский скачок.

## Проверка 2 — time/regime postprocessing

Проверены:

- rank-blend веса CatBoost champion + XGB HPO;
- additive offsets по `row_idx`, `decision_day`, `cb_rate`, `offered_rate`;
- segment-rank mix внутри `month`, `cb_rate_bucket`, `offered_rate_bucket`, `loan_amount_last_bucket`.

Артефакты:

- `reports/validation/postprocess_scans/postprocess_regime_scan_20260619T032900Z.csv`
- `reports/validation/postprocess_scans/postprocess_regime_scan_20260619T032900Z_best.csv`
- `reports/validation/postprocess_scans/postprocess_regime_scan_20260619T032900Z_report.json`

Baseline `c51_x49`:

| OOF | Fold1 | Fold2 | Fold3 |
|---:|---:|---:|---:|
| 0.782234 | 0.796663 | 0.794185 | 0.756746 |

Best postprocess:

| method | params | OOF | Fold1 | Fold2 | Fold3 |
|---|---|---:|---:|---:|---:|
| segment-rank mix | `offered_rate_bucket`, lambda `0.20` | 0.782256 | 0.797390 | 0.793079 | **0.756866** |

Вывод: лучший прирост Fold3 `+0.000120`, при ухудшении Fold2. Это шумовой эффект, не submission-кандидат без late-holdout подтверждения.

## Проверка 3 — внешний датасет `Кредитный скоринг`

Фактология:

- `train_data.parquet`: 18 317 016 строк, 61 колонка;
- `test_data.parquet`: 7 845 701 строк, 61 колонка;
- `train_target.csv`: 2 100 000 ID;
- `sample_submission.csv`: 900 000 ID.

ID-пересечение с `front_id` есть, но ожидаемо случайное из-за широкого диапазона ID. Быстрый join `train_target.flag` к `train_apps.front_id`:

| check | value |
|---|---:|
| covered offer rows | 101806 / 145241 |
| covered positive rate | 0.061293 |
| uncovered positive rate | 0.059998 |
| `scoring_flag` AUC vs offer target | **0.498708** |

Вывод: `id/front_id` не является устойчивым мостом между задачами; прямой external-label join не даёт сигнала. Использование этого датасета также не разрешено текущими правилами проекта.

## Остаточные гипотезы

1. **Неиспользованный признак из исходного offer CSV** — маловероятно: схема полностью короткая, признаки уже перечислены, context/rate/limit/activity/missing/time варианты проверялись.
2. **Другая трактовка задачи как pairwise/group ranking** — слабый кандидат: same-day varying-offer groups покрывают малую долю позитивов, а `rank:pairwise` уже был слабым.
3. **Public/private split сильно отличается от local Fold3** — вероятно, но это не даёт локального способа выбрать модель без LB-probing.
4. **Разрешённый внешний источник/скрытая связка данных** — единственное объяснение, способное дать большой скачок, но текущий найденный скоринг-датасет прямого сигнала не показал.

## Minimal action

- Не тратить upload на hard-window или найденный postprocess.
- Сначала загрузить уже подготовленный `c35_x65` только если дневной слот всё ещё актуален.
- Если искать дальше: проверять не HPO, а постановку данных и возможные официально разрешённые дополнительные источники/признаки.

## Validation

- Achieved level: L3 diagnostic.
- Checked: recent-window CV для XGB/CatBoost, postprocess scan на OOF/Fold3, внешний скоринг label-join sanity.
- Not checked: full aggregation of scoring parquet features, because current project rules forbid external data and direct target join already has AUC ≈ 0.5.
