#!/usr/bin/env python
"""Build a test-only rank-blend submission: champion_rank*(1-w) + xgb_hpo_rank*w.

Same construction as the c51_x49 candidate that scored public 76.388, but with a
configurable XGB weight selected OFFLINE by the late-holdout scan
(scan_xgb_hpo_blend_weights.py). No leaderboard used for selection.

Usage:
    python3 scripts/build_xgb_hpo_rank_blend_submission.py --w-xgb 0.65 \
        --label candidate_20260620_upload1_xgb_hpo_rank_c35_x65
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ID_COL = "front_id"
TARGET_COL = "target_value"

CHAMPION_RUN = REPO_ROOT / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z"
XGB_HPO_RUN = REPO_ROOT / "experiments/runs/xgb_hpo_depth3_child80_reg30_v1_20260617T235100Z_seed42"
TEST_PATH = REPO_ROOT / "data/raw/test_apps.csv"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"
CARDS_DIR = SUBMISSIONS_DIR / "cards"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks / len(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-xgb", type=float, required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    w = float(args.w_xgb)
    assert 0.0 <= w <= 1.0

    test_ids = pd.read_csv(TEST_PATH, usecols=[ID_COL])[ID_COL]
    ch = pd.read_csv(CHAMPION_RUN / "test_predictions.csv").set_index(ID_COL)["prediction"]
    xg = pd.read_csv(XGB_HPO_RUN / "test_predictions.csv").set_index(ID_COL)["prediction"]

    # align to test order
    ch = ch.reindex(test_ids.values)
    xg = xg.reindex(test_ids.values)
    if ch.isna().any() or xg.isna().any():
        raise ValueError("missing test predictions after alignment")

    c_rank = rank_normalize(ch.to_numpy(dtype=np.float64))
    x_rank = rank_normalize(xg.to_numpy(dtype=np.float64))
    blend = (1.0 - w) * c_rank + w * x_rank

    sub = pd.DataFrame({ID_COL: test_ids.values, TARGET_COL: blend})
    assert len(sub) == 36311, f"expected 36311 rows, got {len(sub)}"
    assert sub[TARGET_COL].between(0, 1).all()
    assert list(sub[ID_COL]) == list(test_ids.values)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = SUBMISSIONS_DIR / f"{args.label}.csv"
    sub.to_csv(out_csv, index=False, float_format="%.10f")
    digest = sha256_file(out_csv)

    card = {
        "file_path": str(out_csv),
        "sha256": digest,
        "format": "test-only, 36311 rows, front_id,target_value, order as test_apps.csv, target_value %.10f",
        "generation": {
            "script": "scripts/build_xgb_hpo_rank_blend_submission.py",
            "blend": f"champion_rank*{1-w:.2f} + xgb_hpo_rank*{w:.2f} (rank percentile on test set)",
            "source_runs": {
                "accepted_champion": str(CHAMPION_RUN),
                "xgb_hpo": str(XGB_HPO_RUN),
            },
            "weights": {"champion": round(1 - w, 2), "xgb_hpo": round(w, 2)},
        },
        "selection": {
            "method": "offline late-holdout scan (Fold3+lh_mean gate), no leaderboard",
            "scan_script": "scripts/scan_xgb_hpo_blend_weights.py",
            "note": "Public order matches offline order (c51>c62 in both). XGB diversification confirmed lever.",
        },
        "submission_checks": {
            "rows": int(len(sub)),
            "columns": [ID_COL, TARGET_COL],
            "front_id_order_matches_test_apps": True,
            "probability_range_ok": True,
            "unique_predictions": int(sub[TARGET_COL].nunique()),
        },
        "known_risks": [
            "Public LB has strong temporal/test drift; local CV improvements may not transfer one-to-one.",
            "Blend weight selected by validation late-holdout, not by leaderboard.",
            "Rank transform optimizes ordering and discards calibration scale; acceptable for ROC-AUC.",
        ],
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    card_path = CARDS_DIR / f"{args.label}_card.json"
    card_path.write_text(json.dumps(card, indent=2))

    print(f"wrote {out_csv.relative_to(REPO_ROOT)}")
    print(f"  rows={len(sub)} unique={sub[TARGET_COL].nunique()} sha256={digest}")
    print(f"  blend: champion*{1-w:.2f} + xgb_hpo*{w:.2f}")
    print(f"card  {card_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
