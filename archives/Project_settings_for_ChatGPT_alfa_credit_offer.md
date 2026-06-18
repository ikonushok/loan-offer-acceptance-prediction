# Project settings — Alfa credit offer

Отвечай на русском. Код, CLI, файлы, конфиги и commits — на английском, если в repo нет другого стиля.

## Жесткий CLI-режим

Работаем **одна команда за раз**. После команды жди мой вывод. Анализируй только его и потом давай следующий шаг. Если я пишу `+`, команда прошла без ошибок; переходи дальше.

Правила:
1. Не давай несколько команд сразу, если я явно не попросил.
2. Для CLI: один командный блок + ожидаемый результат.
3. Перед изменением файла запроси фрагмент через `sed`, `grep`, `find`, `ls`, `git status` или короткий `python -c`.
4. Не меняй файл по памяти.
5. После просмотра кратко объясни, что видно.
6. Затем дай одну точечную команду изменения.
7. После изменения дай одну команду проверки.
8. Не используй canvas/canmore/skills для правок проекта; код меняется только CLI-командой.
9. Не обещай сделать позже. Выполняй текущий шаг сейчас или фиксируй, чего нельзя заключить без вывода/файлов.
10. Большую задачу дели на безопасные шаги и останавливайся после ближайшей команды.

Формат CLI-ответа:

Выполни одну команду:

```bash
...
```

Ожидаемый результат: ...

## Проект

Локальный корень: `/Users/bobrsubr/PycharmProjects/_researches/loan-offer-acceptance-prediction`

Задача: Alfa Bank x MFTI «Отклик на кредитный оффер». Нужно построить воспроизводимую модель вероятности согласия клиента на кредитный оффер.

Target: `target_value`, где `1` — принятие, `0` — отказ. Метрика: ROC-AUC. Итог: CSV для `test_apps.csv`, совместимый с `sample_submission.csv`.

Данные: `data/raw/train_apps.csv`, `data/raw/test_apps.csv`, `data/raw/sample_submission.csv`.

CSV из `data/raw` не коммитить.

## Ограничения

- Python 3.10+, предпочтительно 3.11.
- Только open-source библиотеки.
- Не использовать закрытые API, приватные/paid данные или нелицензированный код.
- Не использовать `target_value` как feature.
- Не использовать test labels и target-информацию из `sample_submission.csv`.
- Не оптимизироваться по public leaderboard как по validation set.
- Не больше 3 platform uploads в день.

## Нельзя без явного согласования

- `pip install`, изменение окружения, lock-файлов, `pyproject.toml` или `requirements.txt`.
- Удаление файлов, массовое форматирование, широкий рефакторинг.
- Изменение ML-контрактов, схем данных, API, форматов отчетов/артефактов.
- Запуск тяжелого обучения, полного pipeline или большого hyperparameter search.
- Изменение метрики, target definition, thresholds, CV, group/time policy или логики сабмита.
- Platform submission без sample-format check, submission card и red-team review.

Если ошибка похожа на проблему окружения, сначала диагностируй его.

## AGENTS

Перед нетривиальной задачей учитывай `AGENTS.md`, `agents/context_router.md` и один профильный файл из `agents/`. Не загружай всех агентов сразу.

Минимальный контекст: `AGENTS.md + agents/context_router.md + one primary agent + zero or one reviewer`.

Для нетривиальной задачи сначала кратко определи task, mode, goal, inputs, non-goals, validation target и stop conditions.

Обычная маршрутизация: data -> `data_quality.md`; EDA -> `eda_analyst.md`; leakage -> `leakage_guard.md`; features -> `feature_engineer.md`; CV -> `cv_validator.md`; baseline -> `baseline_builder.md`; training -> `model_trainer.md`; ensemble -> `model_ensembler.md`; metrics -> `metric_validator.md`; submission -> `submission_builder.md`; experiments -> `experiment_manager.md`; docs -> `readme_consistency_reviewer.md`; pre-upload -> `red_team.md`.

Architecture review нужен только при изменении границ модулей, схем, API, pipeline stages, ML/submission contracts.

## Проверки до моделирования

Перед baseline/training проверить:
- наличие train/test/sample файлов;
- размеры train/test/sample;
- `target_value` есть только в train;
- допустимые значения target;
- совместимость train/test schema без target;
- полный список колонок и dtypes;
- пропуски, константные признаки, дубликаты;
- `front_id` uniqueness;
- repeated request/client/offer groups;
- `decision_day` temporal risk и train/test drift;
- совместимость `test_apps.csv` и `sample_submission.csv`.

Не доверять random CV без проверки групп/повторов и `decision_day` temporal risk.

## Leakage и validation

Блокируй:
- `target_value` в features;
- supervised transforms до CV split;
- target encoding/feature selection на full train перед CV;
- train-only ROC-AUC как validation score;
- hard labels вместо continuous scores для ROC-AUC;
- duplicate/sibling offers across folds без group policy;
- test/sample usage для подгонки labels, thresholds или весов;
- submission без row-order/ID/probability checks.

ROC-AUC считать только на held-out/OOF predictions. Positive class: `target_value = 1`. Predictions: float в `[0, 1]`, без NaN/inf.

## Features, submissions, experiments

Сначала full schema discovery. Каждую колонку классифицировать как raw/derived/excluded/grouping/target с причиной.

Приоритетные признаки: rate spread/ratio, amount к limits, within-limit flags, limit spread, 30/90/360 ratios, log monetary transforms, missingness flags, safe categorical handling. Деление на ноль обрабатывать явно. Group-relative features только label-free и fold-safe.

Каждый run фиксирует: run_id, data hash/version, code/config, seed, folds, group/time policy, feature set, params, fold/OOF ROC-AUC, prediction paths, hashes.

Submission требует: columns как sample, row count, ID/order mapping, probability range, no NaN/inf, unique filename, SHA256, source run, validation score, leakage/metric/submission verdicts, red-team review.

## Validation levels

- L0: static/doc check.
- L1: syntax/smoke.
- L2: data/schema consistency.
- L3: reproducible CV with OOF ROC-AUC.
- L4: robustness/leakage/alternative split/seed checks.
- L5: submission readiness: sample-format check, hash, submission card, red-team review.

Всегда пиши validation level и что осталось непроверенным.
