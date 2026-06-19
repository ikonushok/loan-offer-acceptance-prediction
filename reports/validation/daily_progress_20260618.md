# Дневной прогресс — 2026-06-18

## Статус к концу дня

### Загрузки сегодня (3 slot использовано)
Результаты задокументированы в карточках.

### Кандидаты на завтра (готовы)
| слот | файл | Fold3 | lh_mean | статус |
|---|---|---|---|---|
| #1 | `candidate_20260619_upload1_raw_unrounded_fold3best070.csv` | 0.7548 | 0.7602 | PASS_WITH_RISKS |
| #2 | `candidate_20260619_upload2_RETEST_xgb_rank_c62_x38.csv` | 0.7566 | 0.7631 | PASS_WITH_RISKS |

## Выполнено сегодня

### 1. LGBM late holdouts (шаг 1 плана — ✅)
- `lgbm_context_offer_unbalanced_v1`: lh_mean=0.7553, lh_min=0.7456
- `lgbm_context_offer_sqrtpos_v1`: lh_mean=0.7545, lh_min=0.7463
- LGBM diversity blend (c88/u11/s01): lh_mean=0.7606 (+0.00037 vs champion) — **PASS_WITH_RISKS**, но слабее XGB rank blend

### 2. Feature engineering (шаг 3 плана — ✅ все пути проверены)
Все варианты отрицательные для Fold3:

| эксперимент | Fold3 | Δ vs champion | вывод |
|---|---|---|---|
| CatBoost + all pairwise | 0.7509 | -0.0039 | drift: activity ratios |
| CatBoost + rate/limit only | 0.7503 | -0.0045 | drift: cb_rate нестабилен |
| XGB + rate/limit only | 0.7520 | -0.0028 | то же |
| ExtraTrees | 0.7065 | -0.0483 | слишком слаб |

**Причина:** `rate_spread = offered_rate - cb_rate` зависит от уровня cb_rate в период обучения. Fold3 (апр-июн 2025) имеет другой cb_rate, чем Fold1/2. Context-offer features (относительный ранг в группе) — стабильны, уже оптимальны.

Добавлен флаг `add_rate_limit_features` в `scripts/baseline_catboost_time.py` для изолированного тестирования rate/limit экономики.

### 3. LGBM Optuna HPO 100 trials (шаг 4 плана — ✅)
- Best trial #91: Fold3=**0.7520** (untuned было 0.7456, +0.0064)
- Late holdout: lh_mean=0.7607, lh_min=0.7520
- Rank blend champion+lgbm91: Fold3=0.7558, lh_mean < XGB rank blend
- XGB и LGBM91 корреляция = 0.976 — почти идентичны по структуре ошибок
- **Вывод:** tuned LGBM слабее XGB на lh_mean (0.7607 vs 0.7633), не даёт новой диверсификации

### 4. CatBoost HPO ext1 — завершён, отрицательный
- 150 trials, seed=137, расширенный search space (глубина 4-8, l2_leaf_reg до 80, min_data_in_leaf 20-200)
- Config: `configs/optuna_catboost_context_offer_fold3_hpo_ext1.json`
- Артефакты: `reports/validation/optuna_catboost_context_offer_fold3_hpo_ext1_20260618T023937Z_{trials.csv,summary.json}`
- **Best trial #66: Fold3 = 0.75357** — НИЖЕ champion (0.7548)
- Objective = Fold3 напрямую, так что 0.7536 и есть честный Fold3 лучшего trial
- **Вывод: нового кандидата нет.** Сработало стоп-правило (best Fold3 ≤ ~0.753). Регион champion близок к оптимуму для текущего feature set — другой seed и более глубокая регуляризация ничего выше не нашли. Ревалидация top-trials не нужна: та же CatBoost+context-offer архитектура (corr с champion ~0.99), Fold3 уже ниже → апсайд бленда около нуля.

### 5. Late-holdout всего бленда c51_x49 (champion×0.51 + xgb_hpo×0.49)
- Посчитан полный late-holdout (не только XGB-компонента): lh_mean=0.763783, lh_min=0.756184
- Доминирует c62_x38 (lh_mean 0.763104, lh_min 0.755244) по ВСЕМ метрикам: Fold3 0.7567 vs 0.7560, OOF 0.7822 vs 0.7821, H1/H2/H3 все выше
- **Решение: НЕ менять upload #2.** Прирост +0.0007 lh_mean в пределах непроверенной seed-вариативности; c62_x38 уже полностью оформлен (карточка + SHA256 + red-team), c51_x49 потребовал бы нового red-team. Цена замены > пользы. c51_x49 держим как знание/резерв.

### 6. Диагностика разрыва до 0.775 (inspect) — все дешёвые гипотезы негативны
- **Групповая структура** (один запрос → N офферов): multi-offer группы = 6.1% строк, acceptance НЕ «выбор 1 из N» (в multi-группах даже ниже, 0.017). train/test пересечение групп = 0. Не рычаг.
- **Schema audit**: все 26 сырых колонок используются, ничего не выкинуто молча.
- **Формат сабмишена**: 8721 лишних ID в sample не имеют признаков (это train-ID как padding). Организаторы подтвердили — сабмит только test-only 36311. Баллов не теряем.

### 7. Adversarial validation / importance weighting — тупик
- Adversarial AUC train-vs-test (OOF, без decision_day) = 0.7365 → умеренно-сильный дрейф.
- Density-ratio веса, ESS=27983 (19.3% с capping max_weight=20). Артефакты: `scripts/compute_adversarial_weights.py`, `reports/validation/adversarial_weights_20260618T030206Z*`.
- Переобученный champion (trial0070 + adv-веса) ХУЖЕ на всех метриках: OOF 0.7637 vs 0.7811, test-weighted AUC 0.7588 vs 0.7755.
- **Вывод: importance weighting не восстанавливает тест-качество, только теряет данные.**

### 8. Pseudo-labeling — тупик
- Научно-корректная проверка на H3 (2025-05-01..06-05, истинные метки известны). Артефакт: `scripts/pseudo_label_h3_probe.py`.
- base H3 AUC=0.7564; лучший псевдо-вариант +0.0003 (шум), остальные −0.006/−0.0014.
- **Вывод: уверенные псевдо-метки = лёгкие строки, ранжировку не двигают; больший вес → confirmation bias.**

### 9. Чистый missing-flags тест (GBM) — отрицательный, корпус валиден
- Прошлый missing-flags ablation был загрязнён (time_features=true). Чистый тест (trial0070, no-time): Fold3 0.7507 vs champion 0.7524 (−0.0017), OOF −0.0005. Конфиг: `configs/feature_experiments/catboost_trial0070_missingflags_v1.json`.
- Причина: CatBoost/XGB/LGBM едят NaN **нативно** (числовые NaN не импутируются, см. baseline_catboost_time.py) → явные флаги лишь добавляют переобучение.
- **Следствие: все GBM-запуски (champion/HPO/XGB/LGBM/бленды) НЕ требуют пересмотра** — они уже используют missingness, на едином сравнимом препроцессинге.

### 10. Концепт-дрейф Fold1→Fold3 — виден в фичах, но не управляем
- Деградация Fold1 0.796 → Fold2 0.791 → Fold3 0.752 — это реальный концепт-дрейф.
- Prevalence нестационарна 5× (0.031 фев'24 → 0.152 дек'24 → 0.07-0.09 апр-июн'25).
- Связь фича→таргет распадается: `loan_rev_max_start_non_fin` corr 0.34→0.00, cnt_*_loan/limits затухают вдвое.
- Дроп самой дрейфующей фичи (`catboost_trial0070_dropdrift_v1`): Fold3 плоско (−0.0001), Fold1 −0.006, OOF −0.0026 — **не помогает** (дерево само управляет падающей ценностью фичи).
- **Больше фолдов** улучшит измерение/отбор, но не научит модель ловить будущий сдвиг (тест на 6 мес дальше train).

### 11. Tabular NN (TabNet) — слишком слаб, закрыт
- TabNet на тех же фолдах/фичах (torch 2.12 + pytorch-tabnet). Артефакт: `scripts/tabnet_time.py`, `experiments/runs/tabnet_context_offer_v1_*`.
- Baseline (median-fill): Fold1/2/3 = 0.748/0.651/0.689, **OOF 0.6955** (vs champion 0.7811), late-holdout mean 0.672.
- **Декорреляция есть** (corr с champion OOF 0.62 / test 0.72 vs >0.87 у GBM), НО weight-scan: любой вес TabNet монотонно **ухудшает** OOF и Fold3 → разрыв 0.08 не окупается.
- Версия с missing-flags (2 фолда, прерывалась): 0.724/0.671 — не лучше baseline.
- **Вывод: TabNet слишком слаб для пользы в бленде; NN-направление закрыто.**

## КЛЮЧЕВОЙ ИНСАЙТ
**test-weighted AUC champion = 0.7755 ≈ ориентир 0.775.** На тест-подобных данных ранжировка champion уже хороша. Разрыв до публичного 0.76054 — остаточная сложность периода Q4 2025, НЕ закрываемая пересборкой существующих данных. Доп. данных не будет (подтверждено).

> ⚠️ ЧАСТИЧНО ПЕРЕСМОТРЕНО публичными результатами 2026-06-19 (см. блок в конце файла): вывод «все offline-рычаги исчерпаны» верен для HPO/feature-перебора, но НЕ для **блендовой XGB-диверсификации** — она дала 76.054→76.388 на паблике. Offline-метрики были консервативны, но направленно верны.

## Ключевые выводы

1. **Feature space исчерпан** для текущего набора признаков: любые новые ratios имеют temporal drift, context-offer features уже захватывают устойчивый относительный сигнал.

2. **Диверсификации недостаточно**: XGB и все LGBM-варианты коррелируют >0.87 с champion, и XGB/LGBM между собой 0.976. Blend gain ограничен +0.002 Fold3.

3. **CatBoost HPO ext1 подтвердил оптимальность champion**: другой seed и более глубокая регуляризация не нашли Fold3 выше 0.7536 < champion 0.7548. Регион champion близок к оптимуму для текущего feature set.

4. **Все offline-пути на сегодня исчерпаны**: feature engineering (×4 отрицательных), LGBM HPO (слабее XGB), CatBoost HPO ext1 (ниже champion). Реальный сигнал к улучшению теперь даст только публичный результат #1/#2 завтра либо принципиально новый источник признаков/данных.

## Список загрузок на завтра (зафиксирован)

| слот | модель | Fold3 / OOF / lh_mean | статус |
|---|---|---|---|
| #1 | upload1_raw_unrounded_fold3best070 | 0.7548 / 0.7811 / 0.7602 | готов (калибровка raw vs 3dp) |
| #2 | xgb_rank_c62_x38 | 0.7560 / 0.7821 / 0.7631 | готов, red-team пройден |
| #3 | резерв | — | решить после результата #1/#2 |

Оба готовых файла — в `submissions/upload_20260619/`, SHA256 сверены с карточками.

## Следующие действия (на завтра)

1. Загрузить #1 (raw) после сброса дневного лимита.
2. По результату #1 решить судьбу #2 (c62_x38) и слота #3.
3. Если потребуется ещё попытка — c51_x49 как резерв (offline чуть сильнее c62_x38, но нужен red-team перед загрузкой).
4. Новый offline-прирост искать только через новые признаки/данные, не через перебор гиперпараметров текущей архитектуры.

## Validation level
- L4: champion blend и XGB-блендовые кандидаты (Fold3 + полная late-holdout батарея + red-team для c62_x38)
- L3: LGBM HPO trials (Fold3 + late holdout, без red-team)
- HPO ext1 завершён, отрицательный — кандидата не дал

---

## ОБНОВЛЕНИЕ 2026-06-19 — публичные результаты загрузок

Прогресс паблика (3/3 загрузок за день):

| # | Файл | Public AUC | Δ |
|---|---|---|---|
| 1 | candidate_20260619_upload1_raw_unrounded_fold3best070 | 76.057 | калибровка raw≈3dp |
| 2 | candidate_20260619_upload2_RETEST_xgb_rank_c62_x38 (untuned XGB, вес 0.38) | 76.362 | +0.305 |
| 3 | candidate_20260619_upload2_RETEST_xgb_hpo_rank_c51_x49 (HPO-XGB, вес 0.49) | **76.388** | +0.026 (NEW BEST) |

**Выводы:**
- XGB-диверсификация — продуктивный рычаг (вопреки раннему выводу «потолок», который верен лишь для HPO/feature-перебора).
- Паблик усиливает XGB-сигнал (~100× к offline lh_mean), но offline-ПОРЯДОК предсказывает публичный (c51>c62 в обоих).
- Закономерность: больше/сильнее XGB-компонента → выше паблик.

**План на завтра (offline-обоснованный, с картами):**
1. Увеличить вес XGB (0.49 → скан 0.55-0.7) с проверкой Fold3+lh_mean.
2. Сильнее/разнообразнее XGB: мульти-seed rank-бленд, доп. HPO.
3. Отбор кандидатов ОФФЛАЙН (порядок предсказывает паблик), не LB-probing.

### Скан веса XGB выполнен (2026-06-19, offline) — кандидаты готовы

`scripts/scan_xgb_hpo_blend_weights.py` — бленд `champion_rank×(1-w) + xgb_hpo_rank×w`,
late-holdout батарея. Чёткий внутренний оптимум (champion baseline lh_mean=0.760184):

| w_xgb | lh_mean | lh_min | Δmean vs champ |
|---|---|---|---|
| 0.49 (=c51_x49, паблик 76.388) | 0.763783 | 0.756184 | +0.00360 |
| 0.60 | 0.764045 | 0.756288 | +0.00386 |
| **0.65** | **0.764113** | **0.756306** ← пик min | +0.00393 |
| **0.70** | **0.764126** ← пик mean | 0.756288 | +0.00394 |
| 0.75 | 0.764085 | 0.756177 | +0.00390 |
| 0.80–1.00 | спад | спад | — |

Артефакт скана: `reports/validation/xgb_hpo_blend_weight_scan_*.json`.

**Кандидаты на загрузку (offline доминируют c51_x49 по обеим метрикам):**
| слот | файл | w_xgb | lh_mean | lh_min | примечание |
|---|---|---|---|---|---|
| #1 | `candidate_20260620_upload1_xgb_hpo_rank_c35_x65.csv` | 0.65 | 0.764113 | **0.756306** | лучший lh_min, робастный |
| #2 | `candidate_20260620_upload2_xgb_hpo_rank_c30_x70.csv` | 0.70 | **0.764126** | 0.756288 | лучший lh_mean |

Оба в `submissions/upload_20260620/`, карты + SHA256 сверены
(`scripts/build_xgb_hpo_rank_blend_submission.py`). Слот #3 — резерв (мульти-seed XGB
rank-бленд для доп. диверсификации, если #1/#2 подтвердят прирост на паблике).
