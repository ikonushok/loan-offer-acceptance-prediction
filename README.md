# Loan Offer Acceptance Prediction

Проект для задачи Альфа-Банка × МФТИ «Отклик на кредитный оффер».

**Финальный результат: ROC-AUC = 0.76744** (Public Leaderboard).

Подробное описание решения — в [SOLUTION.md](SOLUTION.md).

## Quick Start — воспроизведение результата

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Положить данные
#    data/raw/train_apps.csv
#    data/raw/test_apps.csv
#    data/raw/sample_submission.csv

# 3. Запустить воспроизведение (~10-15 мин на CPU)
bash reproduce.sh

# 4. Результат
#    submissions/final/submission.csv
```

Скрипт `reproduce.sh` последовательно обучает 3 CatBoost-модели и 1 XGBoost-модель
с календарными признаками (`day_num` + `week`, без `month`), затем собирает
rank-percentile blend (champion_no_month × 0.30 + xgb_no_month × 0.70).

## Цель проекта

Построить воспроизводимую ML-модель, которая прогнозирует вероятность согласия корпоративного клиента на конкретный кредитный оффер.

- Целевая переменная: `target_value` (1 — принял, 0 — отказался).
- Метрика: ROC-AUC.
- Итоговый артефакт: CSV с вероятностями для `test_apps.csv`.

## Данные

    data/raw/train_apps.csv     # обучение (145k строк, 28 колонок)
    data/raw/test_apps.csv      # тест (36k строк, 27 колонок)
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

## Текущая реализация

Текущий рабочий контур реализован не через один универсальный entrypoint, а через набор воспроизводимых скриптов и артефактов:

- `scripts/baseline_catboost_time.py` — базовый CatBoost по rolling time folds;
- `scripts/optuna_catboost_time.py` — HPO CatBoost;
- `scripts/baseline_xgb_time.py` — базовый XGBoost;
- `scripts/evaluate_late_holdouts.py` и `scripts/evaluate_late_holdouts_xgb.py` — late-holdout батарея `H1/H2/H3`;
- `scripts/scan_xgb_hpo_blend_weights.py` — подбор весов rank-blend;
- `scripts/build_xgb_hpo_rank_blend_submission.py` — сборка test-only submission;
- `scripts/feature_family_probe.py` — быстрый перебор семейств признаков;
- `scripts/recent_window_scan.py`, `scripts/baseline_torch_tabular.py`, `scripts/evaluate_external_scoring_features.py` — диагностические ветки, которые проверены и на текущий момент закрыты.

Ключевые подтверждённые артефакты:

- champion CatBoost blend: `experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z`;
- XGB HPO run: `experiments/runs/xgb_hpo_depth3_child80_reg30_v1_20260617T235100Z_seed42`;
- отчёты по late-holdout и сравнительным сканам: `reports/validation`;
- карточки сабмитов: `submissions/cards`;
- папка актуальных кандидатов на загрузку: `submissions/upload_20260620`.

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

CV и late-holdout используются для выбора конфигураций и весов; финальные `test_predictions.csv` для upload-кандидатов строятся retrain'ом выбранных конфигураций на всём `train_apps.csv`.

Перед отправкой на платформу нужно проверить:

- колонки совпадают с `sample_submission.csv`;
- число строк совпадает с `test_apps.csv` (а не с полным `sample_submission.csv`);
- порядок строк или ID mapping проверен;
- вероятности находятся в диапазоне `[0, 1]`;
- нет NaN/inf;
- файл имеет уникальное имя;
- SHA256 записан в submission card;
- проведен red-team review.

Важно: организаторы явно подтвердили, что на платформу нужно отправлять **только предсказания для строк из `test_apps.csv`**. `sample_submission.csv` используется как ориентир по колонкам, а не как обязательный row-count шаблон.

## Валидационные уровни

- L0 — статическая проверка файлов и документации.
- L1 — smoke/syntax checks.
- L2 — проверка данных и схем.
- L3 — воспроизводимая CV-валидация с OOF ROC-AUC.
- L4 — robustness checks, alternative splits, leakage review.
- L5 — submission readiness: sample-format check, hash, submission card, red-team review.

## Текущий статус

**Финальный Public ROC-AUC: 0.76744 (76.744).** Прогресс за проект: +1.296.

Финальное решение: rank-blend `full_champion_no_month × 0.30 + xgb_no_month × 0.70`.
Главный рычаг — календарный `day_num` (линейный счётчик дней) во всех компонентах.

### Загрузки и реальные результаты

| дата | модель | offline lh_mean | Public AUC |
|---|---|---|---|
| ≤06-18 | champion 1dp-округление | 0.7602 | 75.448 |
| 06-18 | champion 3dp | 0.7602 | 76.054 |
| 06-19 | raw_unrounded | 0.7602 | 76.057 |
| 06-19 | XGB rank-blend (w=0.38) | 0.7631 | 76.362 |
| 06-19 | HPO-XGB rank-blend (w=0.49) | 0.7638 | 76.388 |
| 06-20 | XGB rank-blend (w=0.65) | 0.7641 | 76.383 |
| 06-20 | XGB no_month c20_n80 | 0.7681 | 76.505 |
| **06-21** | **full_champion_no_month c30_x70** | **0.7692** | **76.744** |
