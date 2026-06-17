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
    data/interim/          # промежуточные артефакты: folds, schema reports
    data/processed/        # подготовленные признаки, если понадобятся

    notebooks/             # EDA и диагностические ноутбуки

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

    scripts/               # CLI-скрипты для запусков
    experiments/           # логи экспериментов, OOF/test predictions, configs
    submissions/           # финальные CSV-сабмиты и submission cards
    reports/               # отчеты по данным, EDA, leakage, validation
    tests/                 # smoke/regression тесты

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

До доверия к ROC-AUC нужно определить безопасную CV-схему:

- `StratifiedKFold`, если нет групповой или временной структуры;
- `GroupKFold` / `StratifiedGroupKFold`, если найдены повторяющиеся заявки, клиенты или офферы;
- time holdout, если `decision_day` показывает временной сдвиг;
- сохранение fold assignments для воспроизводимости.

### 4. Baseline model

Первый baseline должен быть простым и проверяемым:

- обработка numeric/categorical признаков;
- отсутствие target leakage;
- per-fold ROC-AUC;
- OOF ROC-AUC;
- сохраненные OOF/test predictions;
- feature manifest с описанием использованных и исключенных колонок.

### 5. Feature engineering

Приоритетные группы признаков:

- `offered_rate - cb_rate`;
- `offered_rate / cb_rate`;
- отношение суммы заявки к `overdraft_limit_min` и `overdraft_limit_max`;
- флаги попадания суммы в лимиты;
- 30/90/360 activity ratios;
- log-transform для денежных признаков;
- missingness flags;
- безопасная обработка категориальных признаков.

Все supervised-преобразования должны выполняться внутри CV-folds.

### 6. Model training

Кандидаты моделей:

- CatBoost;
- LightGBM;
- XGBoost;
- sklearn HistGradientBoosting;
- ExtraTrees / RandomForest;
- Logistic Regression как sanity baseline.

Сравнивать можно только эксперименты с одинаковой CV-схемой, одинаковым target definition и понятной feature policy.

### 7. Ensemble

Ансамбль допустим только при наличии aligned OOF/test predictions и подтвержденного прироста на OOF ROC-AUC.

Возможные методы:

- simple mean;
- weighted mean;
- rank averaging;
- stacking только через корректные OOF meta-features.

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

## Стартовая команда для Codex

    Use AGENTS.md + agents/context_router.md + agents/data_quality.md + agents/test_validation.md.

    Mode: data_quality_review.

    Task: Inspect train_apps.csv, test_apps.csv, and sample_submission.csv for the Alfa Bank credit-offer acceptance task.

    Inputs:
    - data/raw/train_apps.csv
    - data/raw/test_apps.csv
    - data/raw/sample_submission.csv

    Constraints:
    - Do not train a model yet.
    - Do not use target_value outside train labels.
    - Inspect full schema; do not rely only on representative PDF columns.
    - Check target distribution, train/test schema compatibility, missingness, duplicates, front_id uniqueness, candidate repeated request/client/offer structure, decision_day temporal risk, and sample_submission compatibility.
    - Save reports under reports/data_quality/.
    - Report achieved validation level.

    Stop if data files are missing, train/test schema cannot be aligned, target has unexpected values, or test rows cannot be mapped to sample_submission.

## Текущий статус

Проект инициализирован. Следующий обязательный шаг — data quality review до построения baseline-модели.
