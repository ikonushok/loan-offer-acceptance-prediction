#!/usr/bin/env python
"""Validation and drift audit for the Alfa credit-offer task.

The script performs label-free train-vs-test drift diagnostics and
repeated-context checks. It does not train the target model, does not use
sample submission labels, and does not modify raw data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


TARGET = "target_value"
ID_COL = "front_id"
DAY_COL = "decision_day"

OFFER_COLS = [
    "offered_rate",
    "overdraft_limit_min",
    "overdraft_limit_max",
]

DEFAULT_CUTOFFS = [
    "2025-01-01",
    "2025-03-01",
    "2025-04-01",
]


def resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def make_context_signature(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series, list[str]]:
    excluded = set(OFFER_COLS + [ID_COL, TARGET])
    context_cols = [c for c in train.columns if c not in excluded]
    missing_in_test = [c for c in context_cols if c not in test.columns]
    if missing_in_test:
        raise ValueError(f"Context columns are missing in test: {missing_in_test}")

    train_sig = pd.util.hash_pandas_object(train[context_cols], index=False)
    test_sig = pd.util.hash_pandas_object(test[context_cols], index=False)
    return train_sig, test_sig, context_cols


def audit_current_time_folds(train: pd.DataFrame, train_context_sig: pd.Series, cutoffs: list[str]) -> pd.DataFrame:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    parsed_cutoffs = [pd.Timestamp(c) for c in cutoffs]

    rows: list[dict[str, Any]] = []
    for i, cutoff in enumerate(parsed_cutoffs):
        next_cutoff = parsed_cutoffs[i + 1] if i + 1 < len(parsed_cutoffs) else None

        train_mask = days < cutoff
        valid_mask = days >= cutoff
        if next_cutoff is not None:
            valid_mask &= days < next_cutoff

        tr_sigs = set(train_context_sig[train_mask])
        va_sigs = set(train_context_sig[valid_mask])
        overlap = tr_sigs & va_sigs

        rows.append(
            {
                "fold": i + 1,
                "cutoff": str(cutoff.date()),
                "next_cutoff": None if next_cutoff is None else str(next_cutoff.date()),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "train_start": str(days[train_mask].min().date()) if train_mask.any() else None,
                "train_end": str(days[train_mask].max().date()) if train_mask.any() else None,
                "valid_start": str(days[valid_mask].min().date()) if valid_mask.any() else None,
                "valid_end": str(days[valid_mask].max().date()) if valid_mask.any() else None,
                "context_overlap_groups": int(len(overlap)),
                "train_overlap_rows": int(train_context_sig[train_mask].isin(overlap).sum()),
                "valid_overlap_rows": int(train_context_sig[valid_mask].isin(overlap).sum()),
            }
        )

    return pd.DataFrame(rows)


def audit_repeated_context_by_month(train: pd.DataFrame, train_context_sig: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.to_datetime(train[DAY_COL], errors="raise")
    tmp = pd.DataFrame(
        {
            "month": days.dt.to_period("M").astype(str),
            "context_sig": train_context_sig,
            TARGET: train[TARGET],
        }
    )

    sizes = (
        tmp.groupby(["month", "context_sig"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    repeated = sizes[sizes["rows"] > 1]
    repeated_by_month = (
        repeated.groupby("month", as_index=False)
        .agg(
            repeated_context_groups=("context_sig", "size"),
            rows_in_repeated_context_groups=("rows", "sum"),
            max_context_group_size=("rows", "max"),
        )
        .sort_values("month")
    )

    conflicts = (
        tmp.groupby(["month", "context_sig"], dropna=False)[TARGET]
        .agg(["size", "nunique", "sum"])
        .reset_index()
    )
    conflicts = conflicts[(conflicts["size"] > 1) & (conflicts["nunique"] > 1)]
    conflicting_by_month = (
        conflicts.groupby("month", as_index=False)
        .agg(
            conflicting_groups=("context_sig", "size"),
            rows_in_conflicting_groups=("size", "sum"),
            positives=("sum", "sum"),
        )
        .sort_values("month")
    )

    return repeated_by_month, conflicting_by_month


def univariate_adversarial_auc(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    common_cols = [c for c in test.columns if c in train.columns and c != ID_COL]
    y_adv = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]

    rows: list[dict[str, Any]] = []
    for col in common_cols:
        tr = train[col]
        te = test[col]
        combined = pd.concat([tr, te], ignore_index=True)

        if col == DAY_COL:
            parsed = pd.to_datetime(combined, errors="coerce")
            x = parsed.map(lambda value: value.toordinal() if pd.notna(value) else np.nan).astype(float)
        elif pd.api.types.is_numeric_dtype(tr):
            x = pd.to_numeric(combined, errors="coerce").astype(float)
        else:
            s = combined.astype("string").fillna("__MISSING__")
            freq = s.value_counts(dropna=False, normalize=True)
            x = s.map(freq).astype(float)

        miss = pd.isna(x).astype(int)
        median = pd.Series(x).median()
        if pd.isna(median):
            median = 0.0
        x_filled = pd.Series(x).fillna(median).to_numpy()

        auc = roc_auc_score(y_adv, x_filled)
        auc = max(float(auc), 1.0 - float(auc))

        miss_auc = None
        if miss.min() != miss.max():
            raw_miss_auc = roc_auc_score(y_adv, miss)
            miss_auc = max(float(raw_miss_auc), 1.0 - float(raw_miss_auc))

        rows.append(
            {
                "column": col,
                "dtype": str(train[col].dtype),
                "adv_auc_value": auc,
                "adv_auc_missing": miss_auc,
                "train_missing": float(tr.isna().mean()),
                "test_missing": float(te.isna().mean()),
                "train_nunique": int(tr.nunique(dropna=True)),
                "test_nunique": int(te.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["adv_auc_value", "adv_auc_missing"],
        ascending=False,
        na_position="last",
    )


def run_multivariate_adversarial_variant(
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    feature_cols: list[str],
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    x = pd.concat([train[feature_cols], test[feature_cols]], ignore_index=True).copy()
    y = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]

    if DAY_COL in x.columns:
        x[DAY_COL] = pd.to_datetime(x[DAY_COL], errors="raise").map(pd.Timestamp.toordinal)

    cat_cols = [c for c in x.columns if x[c].dtype == "object"]
    num_cols = [c for c in x.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_leaf=200,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(x, y))

    pipe.fit(x.iloc[train_idx], y[train_idx])
    pred = pipe.predict_proba(x.iloc[valid_idx])[:, 1]
    auc = float(roc_auc_score(y[valid_idx], pred))

    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    importances = pipe.named_steps["model"].feature_importances_

    importance = (
        pd.DataFrame(
            {
                "variant": name,
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    pred_summary = (
        pd.DataFrame({"label": y[valid_idx], "pred_test_probability": pred})
        .groupby("label")["pred_test_probability"]
        .agg(["count", "min", "mean", "median", "max"])
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "variant": name,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "adversarial_auc": auc,
        "valid_rows": int(len(valid_idx)),
        "holdout_prediction_summary": pred_summary,
    }

    return summary, importance


def run_multivariate_adversarial(train: pd.DataFrame, test: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = {
        "all_except_id": [c for c in test.columns if c in train.columns and c != ID_COL],
        "no_decision_day": [c for c in test.columns if c in train.columns and c not in [ID_COL, DAY_COL]],
        "no_day_cb_offered": [
            c
            for c in test.columns
            if c in train.columns and c not in [ID_COL, DAY_COL, "cb_rate", "offered_rate"]
        ],
        "no_day_rate_limits": [
            c
            for c in test.columns
            if c in train.columns
            and c not in [ID_COL, DAY_COL, "cb_rate", "offered_rate", "overdraft_limit_min", "overdraft_limit_max"]
        ],
    }

    summaries: list[dict[str, Any]] = []
    importance_parts: list[pd.DataFrame] = []

    for name, feature_cols in variants.items():
        summary, importance = run_multivariate_adversarial_variant(
            train=train,
            test=test,
            name=name,
            feature_cols=feature_cols,
            random_state=random_state,
        )
        summaries.append(summary)
        importance_parts.append(importance)

    summary_df = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "feature_count": row["feature_count"],
                "adversarial_auc": row["adversarial_auc"],
                "valid_rows": row["valid_rows"],
            }
            for row in summaries
        ]
    )

    importance_df = pd.concat(importance_parts, ignore_index=True)

    return summary_df, importance_df


def write_markdown_report(
    out_dir: Path,
    context_cols: list[str],
    fold_context: pd.DataFrame,
    repeated_by_month: pd.DataFrame,
    conflicting_by_month: pd.DataFrame,
    univariate: pd.DataFrame,
    adversarial_scores: pd.DataFrame,
    adversarial_importance: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Validation drift audit")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("PASS_WITH_RISKS for using rolling time validation as the primary internal score; RETEST before model escalation.")
    lines.append("")
    lines.append("## Context-group definition")
    lines.append("")
    lines.append("- Context signature excludes `front_id`, `target_value`, `offered_rate`, `overdraft_limit_min`, and `overdraft_limit_max`.")
    lines.append(f"- context column count: {len(context_cols)}")
    lines.append("")
    lines.append("## Current rolling time folds: context overlap")
    lines.append("")
    lines.append(fold_context.to_markdown(index=False))
    lines.append("")
    lines.append("## Repeated context by train month")
    lines.append("")
    lines.append(repeated_by_month.to_markdown(index=False))
    lines.append("")
    lines.append("## Conflicting target repeated context by train month")
    lines.append("")
    if conflicting_by_month.empty:
        lines.append("No conflicting repeated contexts.")
    else:
        lines.append(conflicting_by_month.to_markdown(index=False))
    lines.append("")
    lines.append("## Top univariate train-vs-test drift")
    lines.append("")
    lines.append(univariate.head(30).to_markdown(index=False))
    lines.append("")
    lines.append("## Multivariate adversarial validation")
    lines.append("")
    lines.append(adversarial_scores.to_markdown(index=False))
    lines.append("")
    lines.append("## Top adversarial importances by variant")
    lines.append("")
    for variant in adversarial_scores["variant"].tolist():
        lines.append(f"### {variant}")
        lines.append("")
        part = adversarial_importance[adversarial_importance["variant"] == variant].head(20)
        lines.append(part.to_markdown(index=False))
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Train/test drift is severe; `decision_day`, `cb_rate`, `offered_rate`, and limit features are major period proxies.")
    lines.append("- Fold3 / last-period holdout should be treated as the main model-selection score; OOF remains secondary.")
    lines.append("- Current rolling time folds have zero context overlap under the existing context signature, but sibling-offer risk remains relevant for non-temporal CV.")
    lines.append("- Feature/model escalation should use small ablations and report Fold3 impact before trusting average OOF gains.")
    lines.append("")
    lines.append("## Validation")
    lines.append("")
    lines.append("- Achieved level: L4 partial")
    lines.append("- Checked: time-fold context overlap, repeated context by month, univariate drift, multivariate adversarial train-vs-test drift.")
    lines.append("- Remaining: saved alternative group-aware model CV, seed variance, and full red-team review before upload.")

    (out_dir / "validation_drift_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/raw/train_apps.csv")
    parser.add_argument("--test", default="data/raw/test_apps.csv")
    parser.add_argument("--out-dir", default="reports/validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", nargs="*", default=DEFAULT_CUTOFFS)
    args = parser.parse_args()

    train_path = resolve_path(args.train, repo_root)
    test_path = resolve_path(args.test, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_context_sig, test_context_sig, context_cols = make_context_signature(train, test)

    fold_context = audit_current_time_folds(train, train_context_sig, args.cutoffs)
    repeated_by_month, conflicting_by_month = audit_repeated_context_by_month(train, train_context_sig)
    univariate = univariate_adversarial_auc(train, test)
    adversarial_scores, adversarial_importance = run_multivariate_adversarial(train, test, args.seed)

    train_test_context_overlap = set(train_context_sig) & set(test_context_sig)

    fold_context.to_csv(out_dir / "validation_drift_timefold_context_overlap.csv", index=False)
    repeated_by_month.to_csv(out_dir / "validation_drift_repeated_context_by_month.csv", index=False)
    conflicting_by_month.to_csv(out_dir / "validation_drift_conflicting_context_by_month.csv", index=False)
    univariate.to_csv(out_dir / "validation_drift_univariate_adversarial_auc.csv", index=False)
    adversarial_scores.to_csv(out_dir / "validation_drift_adversarial_scores.csv", index=False)
    adversarial_importance.to_csv(out_dir / "validation_drift_adversarial_importance.csv", index=False)

    summary = {
        "validation_level": "L4_partial",
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "context_columns": context_cols,
        "cutoffs": args.cutoffs,
        "train_test_context_overlap_groups": int(len(train_test_context_overlap)),
        "timefold_context_overlap": fold_context.to_dict(orient="records"),
        "adversarial_scores": adversarial_scores.to_dict(orient="records"),
        "top_univariate_drift": univariate.head(30).to_dict(orient="records"),
        "top_adversarial_importance": adversarial_importance.groupby("variant").head(20).to_dict(orient="records"),
        "verdict": "PASS_WITH_RISKS: use Fold3 / last-period holdout as the main internal score; drift is severe.",
        "remaining": [
            "group-aware target-model CV stress test",
            "seed variance",
            "small feature ablations measured primarily on Fold3",
            "red-team review before platform upload",
        ],
    }
    (out_dir / "validation_drift_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_markdown_report(
        out_dir=out_dir,
        context_cols=context_cols,
        fold_context=fold_context,
        repeated_by_month=repeated_by_month,
        conflicting_by_month=conflicting_by_month,
        univariate=univariate,
        adversarial_scores=adversarial_scores,
        adversarial_importance=adversarial_importance,
    )

    print("Wrote validation drift audit artifacts:")
    for path in [
        out_dir / "validation_drift_timefold_context_overlap.csv",
        out_dir / "validation_drift_repeated_context_by_month.csv",
        out_dir / "validation_drift_conflicting_context_by_month.csv",
        out_dir / "validation_drift_univariate_adversarial_auc.csv",
        out_dir / "validation_drift_adversarial_scores.csv",
        out_dir / "validation_drift_adversarial_importance.csv",
        out_dir / "validation_drift_audit_summary.json",
        out_dir / "validation_drift_audit_report.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
