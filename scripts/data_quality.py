#!/usr/bin/env python
"""Generate L2 data-quality reports for the Alfa credit-offer task.

This script performs schema, target, missingness, duplicate, temporal,
categorical-overlap, repeated-offer/context, and sample-submission checks.

It does not train models and does not use sample_submission values as labels.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET = "target_value"
ID_COL = "front_id"
DAY_COL = "decision_day"

OFFER_COLS = [
    "offered_rate",
    "overdraft_limit_min",
    "overdraft_limit_max",
]

CATEGORICAL_COLS = [
    "db_group_last",
    "fl_adminarea",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} file is missing: {path}")


def value_counts_dict(s: pd.Series) -> dict[str, int]:
    counts = s.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def safe_date_summary(df: pd.DataFrame, name: str) -> dict[str, Any]:
    if DAY_COL not in df.columns:
        return {
            "dataset": name,
            "exists": False,
            "missing": None,
            "nunique": None,
            "min": None,
            "max": None,
        }

    parsed = pd.to_datetime(df[DAY_COL], errors="coerce")
    return {
        "dataset": name,
        "exists": True,
        "missing": int(df[DAY_COL].isna().sum()),
        "parse_failures": int(parsed.isna().sum() - df[DAY_COL].isna().sum()),
        "nunique": int(df[DAY_COL].nunique(dropna=False)),
        "min": str(parsed.min().date()) if parsed.notna().any() else None,
        "max": str(parsed.max().date()) if parsed.notna().any() else None,
    }


def schema_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_cols = list(train.columns)
    for col in test.columns:
        if col not in all_cols:
            all_cols.append(col)

    for col in all_cols:
        role = "feature"
        if col == TARGET:
            role = "target"
        elif col == ID_COL:
            role = "id_excluded"
        elif col == DAY_COL:
            role = "date_grouping_feature_candidate"
        elif col in CATEGORICAL_COLS:
            role = "categorical_feature"
        elif col in test.columns:
            role = "numeric_feature_candidate"

        rows.append(
            {
                "column": col,
                "role_initial": role,
                "in_train": col in train.columns,
                "in_test": col in test.columns,
                "train_dtype": str(train[col].dtype) if col in train.columns else "",
                "test_dtype": str(test[col].dtype) if col in test.columns else "",
                "train_missing_rate": train[col].isna().mean() if col in train.columns else np.nan,
                "test_missing_rate": test[col].isna().mean() if col in test.columns else np.nan,
                "train_nunique": train[col].nunique(dropna=False) if col in train.columns else np.nan,
                "test_nunique": test[col].nunique(dropna=False) if col in test.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missingness_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in train.columns:
        if col == TARGET:
            continue
        if col not in test.columns:
            continue
        tr = train[col].isna().mean()
        te = test[col].isna().mean()
        rows.append(
            {
                "column": col,
                "train_missing_rate": tr,
                "test_missing_rate": te,
                "test_minus_train": te - tr,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "test_minus_train",
        key=lambda s: s.abs(),
        ascending=False,
    )


def nunique_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in train.columns:
        rows.append(
            {
                "column": col,
                "train_nunique": train[col].nunique(dropna=False),
                "test_nunique": test[col].nunique(dropna=False) if col in test.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def numeric_ranges_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_cols = [
        c
        for c in test.columns
        if pd.api.types.is_numeric_dtype(test[c])
    ]

    for col in numeric_cols:
        rows.append(
            {
                "column": col,
                "train_min": train[col].min(skipna=True),
                "train_max": train[col].max(skipna=True),
                "train_mean": train[col].mean(skipna=True),
                "train_std": train[col].std(skipna=True),
                "test_min": test[col].min(skipna=True),
                "test_max": test[col].max(skipna=True),
                "test_mean": test[col].mean(skipna=True),
                "test_std": test[col].std(skipna=True),
            }
        )
    return pd.DataFrame(rows)


def categorical_overlap_frame(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in CATEGORICAL_COLS:
        if col not in train.columns or col not in test.columns:
            continue

        train_values = set(train[col].dropna().astype(str))
        test_values = set(test[col].dropna().astype(str))
        rows.append(
            {
                "column": col,
                "train_nunique_non_null": len(train_values),
                "test_nunique_non_null": len(test_values),
                "unseen_in_test_count": len(test_values - train_values),
                "unseen_in_test_values": "|".join(sorted(test_values - train_values)),
                "missing_train": int(train[col].isna().sum()),
                "missing_test": int(test[col].isna().sum()),
                "missing_rate_train": train[col].isna().mean(),
                "missing_rate_test": test[col].isna().mean(),
            }
        )
    return pd.DataFrame(rows)


def monthly_frames(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_tmp = train.copy()
    test_tmp = test.copy()
    train_tmp[DAY_COL] = pd.to_datetime(train_tmp[DAY_COL], errors="coerce")
    test_tmp[DAY_COL] = pd.to_datetime(test_tmp[DAY_COL], errors="coerce")

    train_tmp["month"] = train_tmp[DAY_COL].dt.to_period("M").astype(str)
    test_tmp["month"] = test_tmp[DAY_COL].dt.to_period("M").astype(str)

    train_monthly = (
        train_tmp.groupby("month", dropna=False)[TARGET]
        .agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "rows", "mean": "target_rate"})
    )
    test_monthly = (
        test_tmp.groupby("month", dropna=False)[ID_COL]
        .size()
        .reset_index(name="rows")
    )
    return train_monthly, test_monthly


def duplicate_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

    train_feature_sig = pd.util.hash_pandas_object(train[feature_cols], index=False)
    test_feature_sig = pd.util.hash_pandas_object(test[feature_cols], index=False)

    train_dup_mask = train_feature_sig.duplicated(keep=False)
    test_dup_mask = test_feature_sig.duplicated(keep=False)

    train_dup = train.loc[train_dup_mask].copy()
    train_dup["_feature_sig"] = train_feature_sig[train_dup_mask].values

    if train_dup.empty:
        conflicting_groups = 0
        conflicting_rows = 0
    else:
        grouped = train_dup.groupby("_feature_sig")[TARGET].agg(["size", "nunique"])
        conflicting_groups = int((grouped["nunique"] > 1).sum())
        conflicting_sigs = set(grouped.index[grouped["nunique"] > 1])
        conflicting_rows = int(train_dup["_feature_sig"].isin(conflicting_sigs).sum())

    overlap = set(train_feature_sig) & set(test_feature_sig)

    return {
        "train_full_duplicate_rows": int(train.duplicated().sum()),
        "test_full_duplicate_rows": int(test.duplicated().sum()),
        "train_duplicate_feature_rows_excluding_id_target": int(train_dup_mask.sum()),
        "test_duplicate_feature_rows_excluding_id": int(test_dup_mask.sum()),
        "train_duplicate_feature_groups_with_conflicting_target": conflicting_groups,
        "train_duplicate_feature_rows_with_conflicting_target": conflicting_rows,
        "feature_signature_overlap_train_test_groups": len(overlap),
        "train_rows_with_feature_signature_in_test": int(train_feature_sig.isin(overlap).sum()),
        "test_rows_with_feature_signature_in_train": int(test_feature_sig.isin(overlap).sum()),
    }


def context_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    excluded = set(OFFER_COLS + [ID_COL, TARGET])
    context_cols = [c for c in train.columns if c not in excluded]

    def one(df: pd.DataFrame, has_target: bool) -> dict[str, Any]:
        sig = pd.util.hash_pandas_object(df[context_cols], index=False)
        sizes = sig.value_counts()
        repeated_sigs = set(sizes[sizes > 1].index)
        repeated_mask = sig.isin(repeated_sigs)

        out: dict[str, Any] = {
            "context_groups": int(sig.nunique()),
            "repeated_context_groups": len(repeated_sigs),
            "rows_in_repeated_context_groups": int(repeated_mask.sum()),
            "max_context_group_size": int(sizes.max()),
        }

        if len(repeated_sigs) == 0:
            out.update(
                {
                    "groups_with_varying_offer_terms": 0,
                    "rows_in_groups_with_varying_offer_terms": 0,
                }
            )
            if has_target:
                out.update(
                    {
                        "groups_with_conflicting_target": 0,
                        "rows_in_conflicting_target_groups": 0,
                    }
                )
            return out

        tmp = df.copy()
        tmp["_context_sig"] = sig
        rep = tmp[tmp["_context_sig"].isin(repeated_sigs)].copy()

        agg: dict[str, tuple[str, str]] = {
            "rows": (ID_COL, "size"),
            "unique_offered_rate": ("offered_rate", "nunique"),
            "unique_limit_min": ("overdraft_limit_min", "nunique"),
            "unique_limit_max": ("overdraft_limit_max", "nunique"),
        }
        if has_target:
            agg.update(
                {
                    "target_nunique": (TARGET, "nunique"),
                    "positives": (TARGET, "sum"),
                }
            )

        grp = rep.groupby("_context_sig", dropna=False).agg(**agg).reset_index()
        varying = grp[
            (grp["unique_offered_rate"] > 1)
            | (grp["unique_limit_min"] > 1)
            | (grp["unique_limit_max"] > 1)
        ]

        out.update(
            {
                "groups_with_varying_offer_terms": int(len(varying)),
                "rows_in_groups_with_varying_offer_terms": int(varying["rows"].sum()),
            }
        )

        if has_target:
            conflict = grp[grp["target_nunique"] > 1]
            out.update(
                {
                    "groups_with_conflicting_target": int(len(conflict)),
                    "rows_in_conflicting_target_groups": int(conflict["rows"].sum()),
                }
            )

        return out

    train_sig = pd.util.hash_pandas_object(train[context_cols], index=False)
    test_sig = pd.util.hash_pandas_object(test[context_cols], index=False)
    overlap = set(train_sig) & set(test_sig)

    return {
        "context_columns": context_cols,
        "train": one(train, True),
        "test": one(test, False),
        "train_test_context_overlap_groups": len(overlap),
        "train_rows_with_context_in_test": int(train_sig.isin(overlap).sum()),
        "test_rows_with_context_in_train": int(test_sig.isin(overlap).sum()),
    }


def sample_summary(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sample_columns": list(sample.columns),
        "test_rows": len(test),
        "sample_rows": len(sample),
        "sample_has_front_id": ID_COL in sample.columns,
        "sample_has_target_value": TARGET in sample.columns,
        "status": "UNKNOWN",
    }

    if ID_COL not in sample.columns:
        out["status"] = "BLOCK_NO_FRONT_ID"
        return out

    train_ids = set(train[ID_COL])
    test_ids = set(test[ID_COL])
    sample_ids = set(sample[ID_COL])

    out.update(
        {
            "train_test_overlap_ids": len(train_ids & test_ids),
            "test_sample_overlap_ids": len(test_ids & sample_ids),
            "train_sample_overlap_ids": len(train_ids & sample_ids),
            "sample_minus_test_ids": len(sample_ids - test_ids),
            "test_minus_sample_ids": len(test_ids - sample_ids),
            "sample_front_id_unique": int(sample[ID_COL].nunique(dropna=False)) == len(sample),
        }
    )

    if TARGET in sample.columns:
        out.update(
            {
                "sample_target_nunique": int(sample[TARGET].nunique(dropna=False)),
                "sample_target_counts": value_counts_dict(sample[TARGET]),
                "sample_target_min": float(sample[TARGET].min()),
                "sample_target_max": float(sample[TARGET].max()),
            }
        )

    in_test = sample[ID_COL].isin(test_ids)
    out["sample_rows_in_test"] = int(in_test.sum())
    out["sample_rows_not_in_test"] = int((~in_test).sum())

    sample_test_ids = sample.loc[in_test, ID_COL].reset_index(drop=True)
    test_ids_ordered = test[ID_COL].reset_index(drop=True)
    out["sample_test_subset_same_order_as_test"] = bool(sample_test_ids.equals(test_ids_ordered))

    if (
        out["test_minus_sample_ids"] == 0
        and out["sample_test_subset_same_order_as_test"]
        and out["sample_minus_test_ids"] == 0
    ):
        out["status"] = "PASS_SAMPLE_MATCHES_TEST"
    elif out["test_minus_sample_ids"] == 0 and out["sample_test_subset_same_order_as_test"]:
        out["status"] = "HOLD_SAMPLE_CONTAINS_EXTRA_IDS_BUT_TEST_BLOCK_IS_ALIGNED"
    else:
        out["status"] = "BLOCK_SAMPLE_TEST_MAPPING_UNCLEAR"

    return out


def write_markdown_report(
    out_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    summaries: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Data quality report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "PASS_WITH_RISKS for train/test modeling data; HOLD for final submission policy "
        "until the platform row contract is confirmed."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name, item in summaries["files"].items():
        lines.append(f"- {name}: `{item['path']}`, size_mb={item['size_mb']:.2f}, sha256=`{item['sha256']}`")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"- train shape: {train.shape}")
    lines.append(f"- test shape: {test.shape}")
    lines.append(f"- sample_submission shape: {sample.shape}")
    lines.append(f"- train/test schema compatible excluding target: {summaries['schema']['compatible_excluding_target']}")
    lines.append(f"- target in train: {TARGET in train.columns}")
    lines.append(f"- target in test: {TARGET in test.columns}")
    lines.append("")
    lines.append("## Target sanity")
    lines.append("")
    lines.append(f"- target counts: {summaries['target']['counts']}")
    lines.append(f"- target unique values: {summaries['target']['unique_values']}")
    lines.append(f"- positive rate: {summaries['target']['positive_rate']:.6f}")
    lines.append("")
    lines.append("## Decision day")
    lines.append("")
    lines.append(f"- train: {summaries['decision_day']['train']}")
    lines.append(f"- test: {summaries['decision_day']['test']}")
    lines.append(f"- train/test day intersection: {summaries['decision_day']['intersection_days']}")
    lines.append(f"- test-only days: {summaries['decision_day']['test_only_days']}")
    lines.append("")
    lines.append("## Duplicate and repeated-context checks")
    lines.append("")
    for key, value in summaries["duplicates"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("### Repeated context")
    lines.append("")
    lines.append(f"- train: {summaries['context']['train']}")
    lines.append(f"- test: {summaries['context']['test']}")
    lines.append(f"- train/test context overlap groups: {summaries['context']['train_test_context_overlap_groups']}")
    lines.append("")
    lines.append("## Sample submission compatibility")
    lines.append("")
    for key, value in summaries["sample"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Recommended validation policy")
    lines.append("")
    lines.append("- Primary validation should be time-aware because test is future-dated relative to train.")
    lines.append("- Random StratifiedKFold may be used only as a sanity check, not as the main model-selection score.")
    lines.append("- Repeated context groups exist; avoid splitting same-day sibling offers across folds when using non-temporal CV.")
    lines.append("- Exclude `front_id` from features.")
    lines.append("- Treat `decision_day` as a temporal feature/risk; raw string usage is not recommended.")
    lines.append("")
    lines.append("## Generated CSV artifacts")
    lines.append("")
    lines.append("- `schema_train_test.csv`")
    lines.append("- `missingness.csv`")
    lines.append("- `nunique.csv`")
    lines.append("- `numeric_ranges.csv`")
    lines.append("- `categorical_overlap.csv`")
    lines.append("- `target_by_month.csv`")
    lines.append("- `test_rows_by_month.csv`")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append("- Achieved level: L2 partial")
    lines.append("- Checked: file presence, schema, target sanity, missingness, dtypes, constants, duplicates, temporal split, categorical overlap, repeated context, sample/test mapping.")
    lines.append("- Remaining: official submission row contract, baseline CV implementation, leakage review of future feature code.")

    (out_dir / "data_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_path(path_value: str, repo_root: Path) -> Path:
    """Resolve CLI paths relative to repo root unless they are absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train_apps.csv")
    parser.add_argument("--test", default="data/raw/test_apps.csv")
    parser.add_argument("--sample", default="data/raw/sample_submission.csv")
    parser.add_argument("--out-dir", default="reports/data_quality")
    args = parser.parse_args()

    train_path = resolve_path(args.train, repo_root)
    test_path = resolve_path(args.test, repo_root)
    sample_path = resolve_path(args.sample, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)

    ensure_exists(train_path, "train")
    ensure_exists(test_path, "test")
    ensure_exists(sample_path, "sample_submission")

    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if TARGET not in train.columns:
        raise ValueError(f"{TARGET} is missing from train")
    if TARGET in test.columns:
        raise ValueError(f"{TARGET} unexpectedly exists in test")

    train_feature_cols = set(train.columns) - {TARGET}
    test_cols = set(test.columns)

    target_values = sorted(train[TARGET].dropna().unique().tolist())
    target_counts = value_counts_dict(train[TARGET])
    if target_values != [0, 1]:
        raise ValueError(f"Unexpected target values: {target_values}")

    schema = schema_frame(train, test)
    missingness = missingness_frame(train, test)
    nunique = nunique_frame(train, test)
    numeric_ranges = numeric_ranges_frame(train, test)
    categorical_overlap = categorical_overlap_frame(train, test)
    target_by_month, test_rows_by_month = monthly_frames(train, test)

    schema.to_csv(out_dir / "schema_train_test.csv", index=False)
    missingness.to_csv(out_dir / "missingness.csv", index=False)
    nunique.to_csv(out_dir / "nunique.csv", index=False)
    numeric_ranges.to_csv(out_dir / "numeric_ranges.csv", index=False)
    categorical_overlap.to_csv(out_dir / "categorical_overlap.csv", index=False)
    target_by_month.to_csv(out_dir / "target_by_month.csv", index=False)
    test_rows_by_month.to_csv(out_dir / "test_rows_by_month.csv", index=False)

    train_days = set(train[DAY_COL].dropna()) if DAY_COL in train.columns else set()
    test_days = set(test[DAY_COL].dropna()) if DAY_COL in test.columns else set()

    summaries: dict[str, Any] = {
        "files": {
            "train": {
                "path": str(train_path),
                "size_mb": train_path.stat().st_size / 1024 / 1024,
                "sha256": sha256_file(train_path),
            },
            "test": {
                "path": str(test_path),
                "size_mb": test_path.stat().st_size / 1024 / 1024,
                "sha256": sha256_file(test_path),
            },
            "sample": {
                "path": str(sample_path),
                "size_mb": sample_path.stat().st_size / 1024 / 1024,
                "sha256": sha256_file(sample_path),
            },
        },
        "schema": {
            "compatible_excluding_target": train_feature_cols == test_cols,
            "missing_in_test": sorted(train_feature_cols - test_cols),
            "extra_in_test": sorted(test_cols - train_feature_cols),
        },
        "target": {
            "counts": target_counts,
            "unique_values": target_values,
            "positive_rate": float(train[TARGET].mean()),
        },
        "decision_day": {
            "train": safe_date_summary(train, "train"),
            "test": safe_date_summary(test, "test"),
            "intersection_days": len(train_days & test_days),
            "test_only_days": len(test_days - train_days),
            "train_only_days": len(train_days - test_days),
        },
        "duplicates": duplicate_summary(train, test),
        "context": context_summary(train, test),
        "sample": sample_summary(train, test, sample),
    }

    write_markdown_report(out_dir, train, test, sample, summaries)

    print("Wrote data-quality report artifacts:")
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            print(f"- {path}")


if __name__ == "__main__":
    main()
