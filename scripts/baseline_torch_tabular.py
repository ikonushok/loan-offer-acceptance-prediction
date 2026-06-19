#!/usr/bin/env python
"""Torch tabular diversity baselines for the Alfa credit-offer task.

Models:
- `mlp`: numeric features + categorical embeddings.
- `ft_transformer`: compact FT-Transformer-style model over numeric/cat tokens.

The script is diagnostic: it trains on the same rolling time folds, writes OOF
predictions/metrics, and checks correlation/blend potential vs the current GBM
champion. It does not build submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline_catboost_time import DAY_COL, ID_COL, TARGET, make_time_folds, prepare_features  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CUTOFFS = ["2025-01-01", "2025-03-01", "2025-04-01"]
DEFAULT_FE = {
    "enabled": True,
    "add_pairwise_features": False,
    "add_missing_flags": False,
    "add_context_offer_features": True,
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))


@dataclass
class EncodedData:
    x_num_train: np.ndarray
    x_cat_train: np.ndarray
    x_num_eval: np.ndarray
    x_cat_eval: np.ndarray
    cat_cardinalities: list[int]
    numeric_cols: list[str]
    categorical_cols: list[str]


def encode_fold_features(
    train_x: pd.DataFrame,
    eval_x: pd.DataFrame,
    categorical_cols: list[str],
) -> EncodedData:
    train_x = train_x.copy()
    eval_x = eval_x.copy()

    cat_train_parts: list[np.ndarray] = []
    cat_eval_parts: list[np.ndarray] = []
    cat_cardinalities: list[int] = []

    for col in categorical_cols:
        train_values = train_x[col].astype("string").fillna("__MISSING__")
        eval_values = eval_x[col].astype("string").fillna("__MISSING__")
        categories = pd.Index(train_values.unique())
        mapping = {value: idx + 1 for idx, value in enumerate(categories)}
        train_codes = train_values.map(mapping).fillna(0).astype("int64").to_numpy()
        eval_codes = eval_values.map(mapping).fillna(0).astype("int64").to_numpy()
        cat_train_parts.append(train_codes)
        cat_eval_parts.append(eval_codes)
        cat_cardinalities.append(len(mapping) + 1)

    numeric_cols = [col for col in train_x.columns if col not in categorical_cols]
    if numeric_cols:
        num_train = train_x[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num_eval = eval_x[numeric_cols].apply(pd.to_numeric, errors="coerce")
        medians = num_train.median(axis=0)
        num_train = num_train.fillna(medians).fillna(0.0)
        num_eval = num_eval.fillna(medians).fillna(0.0)
        means = num_train.mean(axis=0)
        stds = num_train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        num_train = ((num_train - means) / stds).clip(-10.0, 10.0)
        num_eval = ((num_eval - means) / stds).clip(-10.0, 10.0)
        x_num_train = num_train.to_numpy(dtype=np.float32)
        x_num_eval = num_eval.to_numpy(dtype=np.float32)
    else:
        x_num_train = np.zeros((len(train_x), 0), dtype=np.float32)
        x_num_eval = np.zeros((len(eval_x), 0), dtype=np.float32)

    if cat_train_parts:
        x_cat_train = np.column_stack(cat_train_parts).astype(np.int64)
        x_cat_eval = np.column_stack(cat_eval_parts).astype(np.int64)
    else:
        x_cat_train = np.zeros((len(train_x), 0), dtype=np.int64)
        x_cat_eval = np.zeros((len(eval_x), 0), dtype=np.int64)

    return EncodedData(
        x_num_train=x_num_train,
        x_cat_train=x_cat_train,
        x_num_eval=x_num_eval,
        x_cat_eval=x_cat_eval,
        cat_cardinalities=cat_cardinalities,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )


class MLPEmbeddingModel(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dim_sum = 0
        for cardinality in cat_cardinalities:
            dim = min(32, max(4, int(round(1.6 * math.sqrt(cardinality)))))
            self.embeddings.append(nn.Embedding(cardinality, dim))
            embedding_dim_sum += dim

        input_dim = n_num + embedding_dim_sum
        layers: list[nn.Module] = [nn.BatchNorm1d(input_dim)]
        current = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current = hidden_dim
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for index, embedding in enumerate(self.embeddings):
            parts.append(embedding(x_cat[:, index]))
        x = torch.cat(parts, dim=1)
        return self.net(x).squeeze(1)


class FTTransformerModel(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_num = n_num
        self.num_weight = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_token))
        self.cat_embeddings = nn.ModuleList([nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []
        if self.n_num:
            tokens.append(x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0))
        if self.cat_embeddings:
            cat_tokens = [embedding(x_cat[:, index]).unsqueeze(1) for index, embedding in enumerate(self.cat_embeddings)]
            tokens.append(torch.cat(cat_tokens, dim=1))
        x = torch.cat(tokens, dim=1)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(1)


def make_model(args: argparse.Namespace, encoded: EncodedData) -> nn.Module:
    if args.model == "mlp":
        hidden_dims = [int(value) for value in args.hidden_dims.split(",") if value]
        return MLPEmbeddingModel(
            n_num=encoded.x_num_train.shape[1],
            cat_cardinalities=encoded.cat_cardinalities,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
    if args.model == "ft_transformer":
        return FTTransformerModel(
            n_num=encoded.x_num_train.shape[1],
            cat_cardinalities=encoded.cat_cardinalities,
            d_token=args.d_token,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def predict_in_batches(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    dataset = TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for x_num_batch, x_cat_batch in loader:
            logits = model(x_num_batch.to(device), x_cat_batch.to(device))
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def fit_one_fold(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eval_features = eval_df.drop(columns=[TARGET]) if TARGET in eval_df.columns else eval_df
    train_x, eval_x, y, categorical_cols, feature_cols, _ = prepare_features(
        train=train_df,
        test=eval_features,
        feature_engineering_cfg=DEFAULT_FE,
        time_features_cfg={"enabled": False},
        excluded_features=[],
    )
    encoded = encode_fold_features(train_x, eval_x, categorical_cols)
    set_seed(seed)

    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    model = make_model(args, encoded).to(device)
    y_train = y.to_numpy(dtype=np.float32)
    y_eval = eval_df[TARGET].to_numpy(dtype=np.float32) if TARGET in eval_df.columns else None

    n_pos = max(float(y_train.sum()), 1.0)
    n_neg = max(float(len(y_train) - y_train.sum()), 1.0)
    if args.pos_weight == "sqrt":
        pos_weight_value = math.sqrt(n_neg / n_pos)
    elif args.pos_weight == "balanced":
        pos_weight_value = n_neg / n_pos
    else:
        pos_weight_value = 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset = TensorDataset(
        torch.from_numpy(encoded.x_num_train),
        torch.from_numpy(encoded.x_cat_train),
        torch.from_numpy(y_train),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, drop_last=False)

    best_auc = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = args.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses: list[float] = []
        for x_num_batch, x_cat_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num_batch.to(device), x_cat_batch.to(device))
            loss = criterion(logits, y_batch.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if y_eval is not None:
            pred = predict_in_batches(model, encoded.x_num_eval, encoded.x_cat_eval, args.eval_batch_size, device)
            auc = float(roc_auc_score(y_eval, pred))
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses)), "valid_auc": auc})
            if auc > best_auc + args.min_delta:
                best_auc = auc
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_left = args.patience
            else:
                patience_left -= 1
            if args.verbose_epochs and (epoch == 1 or epoch % args.verbose_epochs == 0):
                log(f"epoch={epoch} loss={np.mean(losses):.5f} valid_auc={auc:.6f} best={best_auc:.6f}")
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_eval = predict_in_batches(model, encoded.x_num_eval, encoded.x_cat_eval, args.eval_batch_size, device)
    info = {
        "best_auc": None if y_eval is None else float(best_auc),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(history) if history else args.max_epochs),
        "pos_weight": float(pos_weight_value),
        "feature_count": len(feature_cols),
        "numeric_count": len(encoded.numeric_cols),
        "categorical_cols": categorical_cols,
        "cat_cardinalities": encoded.cat_cardinalities,
        "history": history,
    }
    return pred_eval, info


def evaluate_folds(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    train = pd.read_csv(REPO / "data/raw/train_apps.csv")
    folds = make_time_folds(train, DEFAULT_CUTOFFS)
    oof = np.full(len(train), np.nan, dtype=np.float32)
    oof_fold = np.full(len(train), np.nan, dtype=np.float32)
    fold_rows: list[dict[str, Any]] = []
    fold_infos: list[dict[str, Any]] = []

    for fold in folds:
        train_df = train.iloc[fold["train_idx"]].reset_index(drop=True)
        valid_df = train.iloc[fold["valid_idx"]].reset_index(drop=True)
        log(
            "Fold {fold}: train {train_start}..{train_end} rows={train_rows}; "
            "valid {valid_start}..{valid_end} rows={valid_rows}".format(
                fold=fold["fold"],
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                train_rows=len(train_df),
                valid_start=fold["valid_start"],
                valid_end=fold["valid_end"],
                valid_rows=len(valid_df),
            )
        )
        pred, info = fit_one_fold(train_df, valid_df, args, args.seed + int(fold["fold"]))
        auc = float(roc_auc_score(valid_df[TARGET], pred))
        oof[fold["valid_idx"]] = pred
        oof_fold[fold["valid_idx"]] = float(fold["fold"])
        fold_rows.append(
            {
                "fold": fold["fold"],
                "cutoff": fold["cutoff"],
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "valid_start": fold["valid_start"],
                "valid_end": fold["valid_end"],
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "valid_positive_rate": float(valid_df[TARGET].mean()),
                "best_epoch": info["best_epoch"],
                "epochs_run": info["epochs_run"],
                "roc_auc": auc,
            }
        )
        info["fold"] = fold["fold"]
        fold_infos.append(info)
        log(f"Fold {fold['fold']} done: auc={auc:.6f}, best_epoch={info['best_epoch']}")

    valid_predictions = pd.DataFrame(
        {
            "row_index": np.where(~np.isnan(oof))[0],
            TARGET: train.loc[~np.isnan(oof), TARGET].to_numpy(),
            "prediction": oof[~np.isnan(oof)],
            "fold": oof_fold[~np.isnan(oof)].astype("int32"),
        }
    )
    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.loc[len(fold_metrics)] = {
        "fold": "OOF_TIME_HOLDOUT",
        "cutoff": "",
        "train_start": "",
        "train_end": "",
        "valid_start": "",
        "valid_end": "",
        "train_rows": np.nan,
        "valid_rows": len(valid_predictions),
        "valid_positive_rate": float(valid_predictions[TARGET].mean()),
        "best_epoch": np.nan,
        "epochs_run": np.nan,
        "roc_auc": float(roc_auc_score(valid_predictions[TARGET], valid_predictions["prediction"])),
    }
    return fold_metrics, valid_predictions, fold_infos


def scan_blend(valid_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    champion_path = REPO / "experiments/runs/blend_ext_raw_fold3best070_t041004_t164026_20260617T123116Z/valid_predictions_time.csv"
    if not champion_path.exists():
        return pd.DataFrame(), {"available": False}
    champion = pd.read_csv(champion_path).rename(columns={"prediction": "champion_prediction"})
    merged = valid_predictions.merge(champion[["row_index", "champion_prediction"]], on="row_index", how="inner")
    merged["nn_rank"] = merged["prediction"].rank(method="average", pct=True)
    merged["champion_rank"] = merged["champion_prediction"].rank(method="average", pct=True)
    rows: list[dict[str, Any]] = []
    for weight_nn in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        score = (1.0 - weight_nn) * merged["champion_rank"] + weight_nn * merged["nn_rank"]
        row = {
            "weight_nn": float(weight_nn),
            "oof_auc": float(roc_auc_score(merged[TARGET], score)),
        }
        for fold in sorted(merged.get("fold", pd.Series(dtype=int)).dropna().astype(int).unique()):
            fold_mask = merged["fold"] == fold
            row[f"fold{fold}_auc"] = float(roc_auc_score(merged.loc[fold_mask, TARGET], score.loc[fold_mask]))
        rows.append(row)
    corr = float(np.corrcoef(merged["prediction"], merged["champion_prediction"])[0, 1])
    scan = pd.DataFrame(rows)
    best = scan.sort_values(["fold3_auc" if "fold3_auc" in scan.columns else "oof_auc", "oof_auc"], ascending=False).head(1)
    return scan, {
        "available": True,
        "corr_vs_champion_oof": corr,
        "best_blend": best.to_dict(orient="records")[0] if not best.empty else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "ft_transformer"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden-dims", default="256,128")
    parser.add_argument("--d-token", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--pos-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--verbose-epochs", type=int, default=0)
    parser.add_argument("--run-prefix", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    prefix = args.run_prefix or f"torch_{args.model}_context_offer_v1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{prefix}_{stamp}_seed{args.seed}"
    out_dir = REPO / "experiments/runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    fold_metrics, valid_predictions, fold_infos = evaluate_folds(args)
    train = pd.read_csv(REPO / "data/raw/train_apps.csv", usecols=[ID_COL])
    valid_predictions.insert(1, ID_COL, train.loc[valid_predictions["row_index"], ID_COL].to_numpy())

    blend_scan, blend_summary = scan_blend(valid_predictions)

    valid_predictions.to_csv(out_dir / "valid_predictions_time.csv", index=False)
    fold_metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    if not blend_scan.empty:
        blend_scan.to_csv(out_dir / "blend_scan_vs_champion.csv", index=False)

    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_level": "L3_diagnostic_torch_tabular",
        "model": args.model,
        "seed": args.seed,
        "args": vars(args),
        "feature_engineering": DEFAULT_FE,
        "time_features": {"enabled": False},
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "fold_infos": fold_infos,
        "blend_summary": blend_summary,
        "artifacts": {
            "fold_metrics": str(out_dir / "fold_metrics.csv"),
            "valid_predictions": str(out_dir / "valid_predictions_time.csv"),
            "blend_scan": str(out_dir / "blend_scan_vs_champion.csv") if not blend_scan.empty else None,
            "summary": str(out_dir / "run_summary.json"),
        },
        "gates": {
            "standalone_fold3_min": 0.745,
            "standalone_oof_min": 0.770,
            "corr_vs_champion_max": 0.85,
            "blend_must_improve_fold3": True,
        },
        "risk": "Diagnostic only; no test predictions or submission generated.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n== Fold metrics ==")
    print(fold_metrics.to_string(index=False))
    print("\n== Blend summary ==")
    print(json.dumps(blend_summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
