#!/usr/bin/env python
"""Build a test-only submission candidate from saved test predictions.

This script intentionally supports only the current working policy:
- output rows correspond to data/raw/test_apps.csv only;
- columns match sample_submission.csv: front_id,target_value;
- sample_submission.csv has extra rows in this dataset, so full-sample mode is blocked.

No platform upload is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COL = "front_id"
TARGET_COL = "target_value"


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_for_json(value: Any) -> Any:
    """Convert NaN/inf values into JSON-safe null recursively."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="experiments/runs/catboost_time_no_time_features_20260617T040855Z_seed42",
        help="Champion run directory containing test_predictions.csv and run_summary.json.",
    )
    parser.add_argument("--test", default="data/raw/test_apps.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--cards-dir", default="submissions/cards")
    parser.add_argument(
        "--policy",
        default="test_only",
        choices=["test_only"],
        help="Only test_only is supported because sample_submission has extra non-test rows.",
    )
    args = parser.parse_args()

    run_dir = resolve_path(args.run_dir, repo_root)
    test_path = resolve_path(args.test, repo_root)
    sample_path = resolve_path(args.sample, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    cards_dir = resolve_path(args.cards_dir, repo_root)

    prediction_path = run_dir / "test_predictions.csv"
    summary_path = run_dir / "run_summary.json"

    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(test_path, usecols=[ID_COL])
    sample = pd.read_csv(sample_path)
    pred = pd.read_csv(prediction_path)
    summary = load_json(summary_path)

    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if list(pred.columns) != [ID_COL, "prediction"]:
        raise ValueError(f"Unexpected prediction columns: {list(pred.columns)}")
    if len(pred) != len(test):
        raise ValueError(f"Prediction row count {len(pred)} != test row count {len(test)}")
    if not pred[ID_COL].equals(test[ID_COL]):
        raise ValueError("Prediction front_id order does not match test_apps.csv")
    if pred[ID_COL].duplicated().any():
        raise ValueError("Prediction front_id has duplicates")
    if pred["prediction"].isna().any():
        raise ValueError("Predictions contain NaN")
    if np.isinf(pred["prediction"]).any():
        raise ValueError("Predictions contain inf")
    if not pred["prediction"].between(0, 1).all():
        raise ValueError("Predictions are outside [0, 1]")

    test_in_sample = int(test[ID_COL].isin(sample[ID_COL]).sum())
    sample_extra = int((~sample[ID_COL].isin(test[ID_COL])).sum())
    test_missing_from_sample = int((~test[ID_COL].isin(sample[ID_COL])).sum())

    if test_missing_from_sample != 0:
        raise ValueError("Some test IDs are missing from sample_submission.csv")
    if sample_extra != 0:
        print(
            "WARNING: sample_submission.csv has extra non-test rows. "
            f"sample_extra={sample_extra}; building test-only submission candidate."
        )

    submission = pred.rename(columns={"prediction": TARGET_COL})[[ID_COL, TARGET_COL]].copy()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = summary.get("run_id", run_dir.name)
    out_name = f"submission_test_only_{run_id}_{stamp}.csv"
    out_path = out_dir / out_name
    submission.to_csv(out_path, index=False)

    submission_hash = sha256_file(out_path)

    card = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L5_partial_test_only_format_check",
        "policy": args.policy,
        "verdict": "PASS_WITH_RISKS",
        "risks": [
            "sample_submission.csv contains 8721 extra non-test IDs; this file follows test-only working policy.",
            "Platform row contract is not fully resolved.",
            "No red-team review has been completed yet.",
            "No platform upload is authorized by this script.",
        ],
        "source_run": {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "run_summary": str(summary_path),
            "test_predictions": str(prediction_path),
            "config_path": summary.get("config_path"),
            "validation_level": summary.get("validation_level"),
            "fold_metrics": summary.get("fold_metrics"),
        },
        "data_contract": {
            "test_path": str(test_path),
            "sample_path": str(sample_path),
            "test_rows": int(len(test)),
            "sample_rows": int(len(sample)),
            "prediction_rows": int(len(pred)),
            "submission_rows": int(len(submission)),
            "test_in_sample": test_in_sample,
            "test_missing_from_sample": test_missing_from_sample,
            "sample_extra_not_in_test": sample_extra,
            "columns": list(submission.columns),
            "front_id_order_matches_test": bool(submission[ID_COL].equals(test[ID_COL])),
            "front_id_unique": bool(not submission[ID_COL].duplicated().any()),
            "target_col": TARGET_COL,
            "target_min": float(submission[TARGET_COL].min()),
            "target_max": float(submission[TARGET_COL].max()),
            "target_mean": float(submission[TARGET_COL].mean()),
            "target_nan": int(submission[TARGET_COL].isna().sum()),
            "target_inf": int(np.isinf(submission[TARGET_COL]).sum()),
        },
        "hashes": {
            "train_sha256": summary.get("train_sha256"),
            "test_sha256": summary.get("test_sha256"),
            "submission_sha256": submission_hash,
        },
        "artifacts": {
            "submission": str(out_path),
            "card": "",
        },
    }

    card_path = cards_dir / f"{out_path.stem}_card.json"
    card["artifacts"]["card"] = str(card_path)
    card = sanitize_for_json(card)
    card_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("Built test-only submission candidate")
    print(f"submission: {out_path}")
    print(f"card: {card_path}")
    print(f"submission_sha256: {submission_hash}")
    print("\n== CONTRACT CHECK ==")
    for key, value in card["data_contract"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
