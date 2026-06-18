"""Shared fixtures and path setup for the test suite."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def train_df() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/raw/train_apps.csv")


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/raw/test_apps.csv")


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return pd.read_csv(REPO / "data/raw/sample_submission.csv")
