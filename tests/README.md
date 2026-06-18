# Tests

Защищают protected contracts из `AGENTS.md`. Не требуют обучения моделей (быстро, ~2.5 с).

## Запуск

Нужен интерпретатор с зависимостями проекта (catboost, pandas, sklearn). Системный
`python3` (Framework 3.11) их не содержит — используйте окружение проекта:

```bash
python3.13 -m pytest tests/ -q          # miniforge env с catboost+pytest
```

## Покрытие

- `test_submission_contract.py` — сабмит test-only (36311 строк, колонки, порядок,
  вероятности в [0,1], без NaN/inf), и SHA256 staged-файлов совпадает с карточкой.
- `test_no_leakage.py` — target/front_id/decision_day не в фичах; инженерные фичи
  label-free (не меняются при перестановке таргета); time folds без утечки будущего.
- `test_sample_weights.py` — веса положительны/конечны; adversarial_file покрывает все id.
