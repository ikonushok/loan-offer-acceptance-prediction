"""Submission-format protected contract (AGENTS.md):

A submission must contain only test_apps.csv rows (36311), columns front_id,target_value,
front_id order matching test_apps.csv, unique ids, probabilities in [0,1], no NaN/inf.
Files staged for upload must additionally match the SHA256 recorded in their card.
"""

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
UPLOAD_DIR = REPO / "submissions/upload_20260619"
CARDS_DIR = REPO / "submissions/cards"

UPLOAD_FILES = sorted(glob.glob(str(UPLOAD_DIR / "*.csv")))


def _card_hash_for(csv_path: Path) -> str | None:
    """Find the card whose recorded submission hash should match this file."""
    stem = csv_path.name
    for c in glob.glob(str(CARDS_DIR / "*.json")):
        d = json.load(open(c))
        h = d.get("hashes", {}).get("submission_sha256") or d.get("sha256")
        if h and stem.replace(".csv", "") in c.replace("_card.json", ""):
            return h
    return None


@pytest.mark.parametrize("csv_path", UPLOAD_FILES, ids=[Path(p).name for p in UPLOAD_FILES])
def test_upload_matches_test_contract(csv_path, test_df):
    sub = pd.read_csv(csv_path)
    assert list(sub.columns) == ["front_id", "target_value"], "columns must be front_id,target_value"
    assert len(sub) == len(test_df) == 36311, "must be exactly the 36311 test rows"
    assert sub["front_id"].is_unique, "front_id must be unique"
    # order matches test_apps.csv exactly (test-only policy, confirmed by organizers)
    assert sub["front_id"].tolist() == test_df["front_id"].tolist(), "front_id order must match test_apps.csv"
    t = sub["target_value"].to_numpy()
    assert np.isfinite(t).all(), "no NaN/inf in target_value"
    assert t.min() >= 0.0 and t.max() <= 1.0, "probabilities must be in [0,1]"


@pytest.mark.parametrize("csv_path", UPLOAD_FILES, ids=[Path(p).name for p in UPLOAD_FILES])
def test_upload_hash_matches_card(csv_path):
    expected = _card_hash_for(Path(csv_path))
    if expected is None:
        pytest.skip(f"no card hash found for {Path(csv_path).name}")
    actual = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    assert actual == expected, "staged file content must match the SHA256 recorded in its card"


def test_no_test_rows_missing_from_sample(sample_df, test_df):
    # Every test id must be representable in the sample schema.
    assert set(test_df["front_id"]).issubset(set(sample_df["front_id"]))
