#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# reproduce.sh — воспроизведение финального решения от data/raw/ до submission
#
# Результат: submissions/final/submission.csv (ROC-AUC 0.76744 на public LB)
#
# Требования:
#   - Python 3.11+ с пакетами из requirements.txt
#   - data/raw/train_apps.csv, data/raw/test_apps.csv, data/raw/sample_submission.csv
#
# Время: ~10–15 минут (CPU, Apple M3 Pro)
# ============================================================================

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"

# Определяем Python (miniforge > system)
if command -v python3.13 &>/dev/null; then
    PY=python3.13
elif [ -x "/opt/homebrew/Caskroom/miniforge/base/bin/python3" ]; then
    PY="/opt/homebrew/Caskroom/miniforge/base/bin/python3"
else
    PY=python3
fi
echo "Python: $($PY --version)"

export PYTHONPATH="$REPO/scripts"

# --- проверка данных ---
for f in data/raw/train_apps.csv data/raw/test_apps.csv data/raw/sample_submission.csv; do
    [ -f "$f" ] || { echo "ОШИБКА: не найден $f"; exit 1; }
done
echo "Данные на месте."

# ============================================================================
# Шаг 1. Обучение CatBoost trial0070 + no_month (OOF + test)
# ============================================================================
echo ""
echo "=== Шаг 1/5: CatBoost trial0070 no_month ==="
$PY scripts/baseline_catboost_time.py \
    --config configs/feature_experiments/catboost_trial0070_no_month_v1.json

T070_DIR=$(ls -dt experiments/runs/catboost_trial0070_no_month_v1_*_seed42 | head -1)
echo "  → $T070_DIR"

# ============================================================================
# Шаг 2. Обучение CatBoost trial0041 + no_month
# ============================================================================
echo ""
echo "=== Шаг 2/5: CatBoost trial0041 no_month ==="
$PY scripts/baseline_catboost_time.py \
    --config configs/feature_experiments/catboost_trial0041_no_month_v1.json

T041_DIR=$(ls -dt experiments/runs/catboost_trial0041_no_month_v1_*_seed42 | head -1)
echo "  → $T041_DIR"

# ============================================================================
# Шаг 3. Обучение CatBoost trial0164 + no_month
# ============================================================================
echo ""
echo "=== Шаг 3/5: CatBoost trial0164 no_month ==="
$PY scripts/baseline_catboost_time.py \
    --config configs/feature_experiments/catboost_trial0164_no_month_v1.json

T164_DIR=$(ls -dt experiments/runs/catboost_trial0164_no_month_v1_*_seed42 | head -1)
echo "  → $T164_DIR"

# ============================================================================
# Шаг 4. Обучение XGBoost HPO + no_month
# ============================================================================
echo ""
echo "=== Шаг 4/5: XGBoost HPO no_month ==="

# Нужен конфиг XGB с no_month time_features
XGB_NM_CFG="configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_no_month_v1.json"
if [ ! -f "$XGB_NM_CFG" ]; then
    $PY -c "
import json
base = json.load(open('configs/xgb_hpo_experiments/xgb_hpo_depth3_child80_reg30_v1.json'))
base['time_features'] = {
    'enabled': True,
    'include_day_num': True,
    'include_month': False,
    'include_dayofweek': False,
    'include_week': True
}
base['run_prefix'] = 'xgb_hpo_depth3_child80_reg30_no_month_v1'
json.dump(base, open('$XGB_NM_CFG', 'w'), indent=2)
"
fi

$PY scripts/baseline_xgb_time.py --config "$XGB_NM_CFG"

XGB_DIR=$(ls -dt experiments/runs/xgb_hpo_depth3_child80_reg30_no_month_v1_*_seed42 | head -1)
echo "  → $XGB_DIR"

# ============================================================================
# Шаг 5. Rank-blend и сборка финального submission
# ============================================================================
echo ""
echo "=== Шаг 5/5: Rank-blend → submission ==="

mkdir -p submissions/final

$PY - <<PYEOF
import pandas as pd, numpy as np, hashlib

ID = "front_id"
TGT = "target_value"

test_ids = pd.read_csv("data/raw/test_apps.csv", usecols=[ID])[ID]

def load_test(path):
    return pd.read_csv(path).set_index(ID)["prediction"].reindex(test_ids.values)

def rank_normalize(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    return ranks / len(a)

# CatBoost champion no_month blend: t070*0.70 + t041*0.04 + t164*0.26
champion = (
    load_test("$T070_DIR/test_predictions.csv") * 0.70
    + load_test("$T041_DIR/test_predictions.csv") * 0.04
    + load_test("$T164_DIR/test_predictions.csv") * 0.26
)

# XGBoost no_month
xgb = load_test("$XGB_DIR/test_predictions.csv")

assert not champion.isna().any(), "missing champion predictions"
assert not xgb.isna().any(), "missing xgb predictions"

# Rank-blend: champion_rank * 0.30 + xgb_rank * 0.70
blend = 0.30 * rank_normalize(champion.to_numpy()) + 0.70 * rank_normalize(xgb.to_numpy())

submission = pd.DataFrame({ID: test_ids.values, TGT: blend})
assert len(submission) == 36311, f"expected 36311 rows, got {len(submission)}"
assert submission[TGT].between(0, 1).all(), "probabilities out of range"

out = "submissions/final/submission.csv"
submission.to_csv(out, index=False, float_format="%.10f")
sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
print(f"  rows={len(submission)}, unique={submission[TGT].nunique()}")
print(f"  SHA256: {sha}")
print(f"  → {out}")
PYEOF

echo ""
echo "============================================================================"
echo "ГОТОВО. Финальный submission: submissions/final/submission.csv"
echo "Ожидаемый ROC-AUC на public LB: 0.76744"
echo "============================================================================"
