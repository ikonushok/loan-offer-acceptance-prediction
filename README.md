# Loan Offer Acceptance Prediction

Проект для задачи Альфа-Банка x МФТИ «Отклик на кредитный оффер».

## Цель проекта

Построить воспроизводимую ML-модель, которая прогнозирует вероятность согласия корпоративного клиента на конкретный кредитный оффер.

Целевая переменная:

- `target_value = 1` — клиент принял оффер;
- `target_value = 0` — клиент отказался от оффера.

Основная метрика качества:

- ROC-AUC.

Итоговый артефакт:

- CSV-файл с вероятностями для `test_apps.csv`, совместимый с форматом `sample_submission.csv`.

## Данные

Ожидаемые локальные файлы:

    data/raw/train_apps.csv
    data/raw/test_apps.csv
    data/raw/sample_submission.csv

Файлы с данными не коммитятся в Git.

## Ограничения

- Python 3.10+.
- Только open-source библиотеки.
- Используются только данные, предоставленные в рамках задания.
- Нельзя использовать `target_value` как признак.
- Нельзя использовать test labels или информацию из public leaderboard для подгонки модели.
- Сабмиты на платформу считаются ограниченным ресурсом: не более 3 загрузок в день.

## Структура проекта

    data/raw/              # исходные CSV-файлы, не коммитятся

    src/alfa_credit/       # основной Python-пакет проекта
      data.py              # загрузка данных и schema checks
      features.py          # feature engineering
      validation.py        # CV, group/time split checks
      metrics.py           # ROC-AUC и проверки предсказаний
      models.py            # фабрики моделей
      train.py             # обучение и OOF-предсказания
      predict.py           # инференс
      submit.py            # сборка и проверка submission
      utils.py             # вспомогательные функции

    scripts/               # CLI-скрипты: обучение, HPO, бленды, диагностика
    configs/               # JSON-конфиги экспериментов и HPO
    experiments/runs/      # артефакты запусков: OOF/test predictions, метрики
    submissions/           # финальные CSV-сабмиты, submission cards, папки upload
    reports/validation/    # late-holdout батарея, daily progress, weight scans
    tests/                 # contract-тесты (leakage, submission format, weights)
    agents/                # 21 специализированный агент (см. agents/README.md)

## Рекомендуемый порядок работы

### 1. Data quality review

Перед моделированием необходимо проверить:

- размеры train/test/sample;
- наличие `target_value` только в train;
- совместимость train/test schema;
- типы колонок;
- пропуски;
- константные и почти константные признаки;
- дубликаты строк и `front_id`;
- возможные повторяющиеся заявки, клиенты или офферы;
- распределение `decision_day`;
- train/test drift;
- совместимость `test_apps.csv` и `sample_submission.csv`.

Ожидаемые артефакты:

    reports/data_quality/data_quality_report.md
    reports/data_quality/schema_train.csv
    reports/data_quality/schema_test.csv
    reports/data_quality/missingness.csv
    reports/data_quality/drift_summary.csv

### 2. EDA и leakage review

Нужно изучить:

- связь `offered_rate`, `cb_rate`, `offered_rate - cb_rate` с target;
- отношение запрошенной суммы к лимитам;
- активность клиента за 30/90/360 дней;
- missingness patterns;
- категориальные признаки;
- временные паттерны по `decision_day`;
- признаки, похожие на ID или потенциальную утечку.

### 3. Validation design

Используется **rolling time folds** с расширяющимся обучающим окном:

- Fold1: train до 2025-01-01, val 2025-01-01..03-01;
- Fold2: train до 2025-03-01, val 2025-03-01..04-01;
- Fold3: train до 2025-04-01, val 2025-04-01..06-05 (**primary selection criterion**).

Дополнительно — **late holdouts** (H1/H2/H3) на расширяющихся будущих периодах для оценки drift-устойчивости (`lh_mean`, `lh_min`).

Fold assignments детерминированы по `decision_day` (не по random seed).

### 4. Baseline model

Первый baseline должен быть простым и проверяемым:

- обработка numeric/categorical признаков;
- отсутствие target leakage;
- per-fold ROC-AUC;
- OOF ROC-AUC;
- сохраненные OOF/test predictions;
- feature manifest с описанием использованных и исключенных колонок.

### 5. Feature engineering

Рабочие признаки (используются в champion):

- **context-offer features** — относительный ранг оффера внутри контекстной группы (стабильны при drift);
- 30/90/360 activity ratios;
- безопасная обработка категориальных признаков (`db_group_last`, `fl_adminarea`);
- log-transform для денежных признаков.

Проверены и **закрыты** (экспериментально отрицательные):

- `offered_rate - cb_rate`, `offered_rate / cb_rate` — temporal drift из-за нестационарности `cb_rate`;
- rate/limit ratios (`loan_amount_to_limit_*`, `overdraft_limit_spread`) — тот же drift;
- missingness flags — GBM ест NaN нативно, явные флаги переобучают (Fold3 −0.0017);
- дроп дрейфующих фичей — Fold3 плоско, не помогает.

Все supervised-преобразования выполняются внутри CV-folds.

### 6. Model training

Проверенные модели и итоги:

| модель | лучший Fold3 | lh_mean | роль в финальном решении |
|---|---|---|---|
| **CatBoost** blend (t070×0.70 + t041×0.04 + t164×0.26) | 0.7548 | 0.7602 | champion-компонента rank-бленда |
| **XGBoost HPO** (depth=3, child=80, reg=30) | 0.7550 | 0.7633 | диверсификатор; rank-blend w=0.49 дал public 76.388 |
| XGBoost untuned | 0.7520 | 0.7631 | проверочный; rank-blend w=0.38 дал public 76.362 |
| CatBoost HPO ext1 (150 trials, seed=137) | 0.7536 | — | закрыт (ниже champion, corr ~0.99) |
| LightGBM HPO (100 trials) | 0.7520 | 0.7607 | закрыт (слабее XGB, corr 0.976) |
| TabNet (attention NN) | OOF 0.6955 | 0.672 | закрыт (слишком слаб, бленд ухудшает) |
| ExtraTrees | 0.7065 | — | закрыт |

Финальное решение — **не одна модель, а rank-blend** двух GBM-семейств (CatBoost champion + XGBoost HPO). Standalone-метрики моделей ниже, чем у бленда, потому что прирост идёт за счёт ранговой диверсификации.

Сравнивать можно только эксперименты с одинаковой CV-схемой, одинаковым target definition и понятной feature policy.

### 7. Ensemble

Ансамбль допустим только при наличии aligned OOF/test predictions и подтвержденного прироста на OOF ROC-AUC.

Рабочий метод — **rank-percentile blend**:

    blend = champion_rank × (1 - w) + xgb_hpo_rank × w

Rank-нормализация выполняется отдельно на каждом holdout (и на test) для корректности. Это ключевой рычаг: дал весь прирост 76.054 → 76.388. Оптимальный вес XGB по late-holdout скану: w = 0.65–0.70.

Raw-probability blend и stacking проверены — не дают диверсификации (corr > 0.87 между GBM).

### 8. Submission build

Перед отправкой на платформу нужно проверить:

- колонки совпадают с `sample_submission.csv`;
- число строк совпадает с `test_apps.csv`;
- порядок строк или ID mapping проверен;
- вероятности находятся в диапазоне `[0, 1]`;
- нет NaN/inf;
- файл имеет уникальное имя;
- SHA256 записан в submission card;
- проведен red-team review.

## Валидационные уровни

- L0 — статическая проверка файлов и документации.
- L1 — smoke/syntax checks.
- L2 — проверка данных и схем.
- L3 — воспроизводимая CV-валидация с OOF ROC-AUC.
- L4 — robustness checks, alternative splits, leakage review.
- L5 — submission readiness: sample-format check, hash, submission card, red-team review.

## Текущий статус

**Public best: 76.388** (ROC-AUC × 100). Стадия: оптимизация rank-blend весов.

Champion — CatBoost blend (trial0070 × 0.70 + trial0041 × 0.04 + trial0164 × 0.26). Ключевой рычаг — rank-blend с HPO-XGBoost (weight scan 0.65–0.70 по late-holdout).

### Загрузки и реальные результаты

| дата | файл | offline Fold3 | offline lh_mean | Public AUC |
|---|---|---|---|---|
| ≤06-18 | champion, 1dp-округление | 0.7548 | 0.7602 | 75.448 |
| 06-18 | `accepted_public_76054_3dp_fold3best070` (champion, 3dp) | 0.7548 | 0.7602 | **76.054** |
| 06-19 #1 | `candidate…upload1_raw_unrounded_fold3best070` | 0.7548 | 0.7602 | 76.057 |
| 06-19 #2 | `candidate…upload2_RETEST_xgb_rank_c62_x38` (untuned XGB, w=0.38) | 0.7566 | 0.7631 | 76.362 |
| 06-19 #3 | `candidate…upload2_RETEST_xgb_hpo_rank_c51_x49` (HPO-XGB, w=0.49) | 0.7567 | 0.7638 | **76.388** |

### Кандидаты на 06-20 (готовы, ещё не грузились)

| слот | файл | w_xgb | offline lh_mean | offline lh_min | прогноз |
|---|---|---|---|---|---|
| #1 | `candidate_20260620_upload1_xgb_hpo_rank_c35_x65` | 0.65 | 0.764113 | **0.756306** | > 76.388 |
| #2 | `candidate_20260620_upload2_xgb_hpo_rank_c30_x70` | 0.70 | **0.764126** | 0.756288 | > 76.388 |
| #3 | резерв (мульти-seed XGB) | — | — | — | решить после #1/#2 |

Закрытые направления: HPO (потолок Fold3 ≈ 0.755), feature engineering (temporal drift), adversarial weighting, pseudo-labeling, TabNet. Подробнее — `reports/validation/daily_progress_20260618.md`.
