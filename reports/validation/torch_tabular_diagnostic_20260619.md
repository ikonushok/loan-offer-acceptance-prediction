# Torch tabular diagnostic — 2026-06-19

## Verdict

RETEST / negative for current NN direction — MLP embeddings and compact FT-Transformer do not pass the minimal diversity gate. Standalone Fold3 is too weak, and rank-blending with the GBM champion does not improve primary Fold3.

## Scope

Mode: `model_training`.

Goal: check whether neural tabular models can provide a decorrelated signal inside the offer task, without using external data and without creating a submission.

Models checked:

- `mlp`: numeric features + categorical embeddings;
- `ft_transformer`: compact FT-Transformer-style token model.

Common setup:

- data: `data/raw/train_apps.csv`, `data/raw/test_apps.csv`;
- folds: rolling time folds `[2025-01-01, 2025-03-01, 2025-04-01]`;
- features: context-offer features, no raw time features, no target encodings;
- loss: BCEWithLogits with sqrt positive class weight;
- output: OOF predictions + blend scan vs current GBM champion;
- no test prediction / submission generated.

## Results

| model | Fold1 | Fold2 | Fold3 primary | OOF | corr vs champion | best blend Fold3 |
|---|---:|---:|---:|---:|---:|---:|
| MLP embeddings | 0.767745 | 0.745022 | 0.709661 | 0.727907 | 0.7215 | 0.754609 |
| tiny FT-Transformer | 0.772770 | 0.763985 | 0.724791 | 0.754203 | 0.8633 | 0.754800 |
| GBM champion reference | 0.795951 | 0.792685 | 0.754800 | 0.781140 | — | 0.754800 |

## Gate check

Required before further NN investment:

- standalone Fold3 `>= 0.745`;
- standalone OOF `>= 0.770`;
- corr vs champion `<= 0.85`;
- blend with champion improves Fold3.

| model | Fold3 gate | OOF gate | corr gate | blend gate | verdict |
|---|---|---|---|---|---|
| MLP embeddings | FAIL | FAIL | PASS | FAIL | reject |
| tiny FT-Transformer | FAIL | FAIL | FAIL | FAIL | reject |

## Interpretation

MLP is decorrelated enough (`corr=0.72`) but too weak: Fold3 `0.7097` makes its errors too noisy for useful blending.

Tiny FT-Transformer is stronger than MLP but still far below the GBM champion and too correlated (`corr=0.8633`). The best blend weight is `0.0`, meaning the scan keeps only the champion.

This is consistent with the earlier TabNet result:

- TabNet OOF `0.6955`;
- TabNet Fold3 `0.6889`;
- TabNet had decorrelation but insufficient standalone strength.

## Artifacts

- script: `scripts/baseline_torch_tabular.py`
- MLP run: `experiments/runs/torch_mlp_context_offer_v1_20260619T035014Z_seed42`
- FT run: `experiments/runs/torch_ft_tiny_context_offer_v1_20260619T035535Z_seed42`

## Minimal action

- Do not build a submission from these NN models.
- Do not spend upload slots on NN blends.
- Do not tune TabNet further.
- If NN work continues, only one path remains plausible: stronger pretrained/self-supervised representation on train+test followed by a supervised head. It must first beat Fold3 `0.745` before any blend work.

## Validation

- Achieved level: L3 diagnostic.
- Checked: rolling-fold ROC-AUC, OOF ROC-AUC, OOF correlation vs champion, rank-blend scan vs champion.
- Not checked: late-holdout battery for NN models; not justified because primary gates failed.
