# TADF photoluminescence prediction

Code for predicting thermally activated delayed fluorescence (TADF) photoluminescence properties from molecular structure using graph neural networks (GNNs), plus utilities for dataset construction and simpler fingerprint baselines.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/gnn/` | GraphSAGE-based model (homo/hetero), training loop, experiment entry points |
| `src/data_processing/` | `TadfPL` dataset (PyG `InMemoryDataset`), raw CSV/SDF splits |
| `src/fingerprints/` | Ridge regression on molecular fingerprints (Jupyter notebook) |
| `src/utils/` | Shared visualization helpers |
| `mining-codes/` | Literature extraction pipeline (ChemDataExtractor extensions; optional) |

## Requirements

Python **3.10+** is recommended. Dependencies are split by use case:

- **`requirements-model-training.txt`** — GNN training: PyTorch, PyTorch Geometric, RDKit, Lightning, Weights & Biases, etc.  
  Many entries are pinned to **CUDA 12.4** wheels (`+cu124`). On CPU-only or other CUDA versions, install a matching [PyTorch](https://pytorch.org/get-started/locally/) stack first, then install the remaining packages (you may need to adjust or omit `pyg-lib` / `torch_*` extension wheels to match your PyTorch build).

- **`requirements-data-mining.txt`** — Rebuilding or extending the literature-mining pipeline. This file reflects a **cluster/conda** environment (local `file://` wheels, private Git SSH URLs, and optional packages). Treat it as a reference; expect to curate a smaller subset for your machine.

## Dataset

The bundled **TadfPL** dataset lives under `src/data_processing/tadf_pl/`:

- Structures and targets are built by `src/data_processing/tadf_dataset.py` (`TadfPL`).
- Train / validation / test molecule IDs are listed in `src/data_processing/tadf_pl/raw/train.csv`, `val.csv`, and `test.csv`.
- `TadfPL_rdkit_one_conf.sdf` and `TadfPL.csv` provide raw inputs for processing.

RDKit is used when preprocessing; a preprocessed cache can be used when RDKit is unavailable (see the class docstring in `tadf_dataset.py`).

## Train the GNN

From the **`src/gnn`** directory (so relative paths resolve to `src/data_processing`):

```bash
cd src/gnn
pip install -r ../../requirements-model-training.txt  # or install PyTorch/GNN stack manually first
python run_experiment.py --seed 2000
```

Behavior is controlled by **`gnn_config.yaml`**:

- Architecture: heterogeneous **GraphSAGE** on typed bonds (`hetero: true`), MLP head, optional solvent/molecular-size features.
- Training: batch size, epochs, learning rate, normalization, and toggles for train/eval/test, checkpointing, and plots.
- Set `disable_wandb: true` in the YAML to run without logging in to [Weights & Biases](https://wandb.ai/).

Optional: `save_model: true` and `save_path` control where checkpoints are written.

### Hyperparameter sweep (W&B)

Still from **`src/gnn`**:

```bash
python sweep.py
```

This reads **`sweep_config.yaml`** and runs a Weights & Biases sweep (`wandb.sweep` / `wandb.agent`). You need a valid W&B login and project access.

## Other entry points

- **`src/fingerprints/train_fp_ridge_regression.ipynb`** — Ridge regression baseline using fingerprints on the TadfPL table data.
- **`mining-codes/`** — Scripts for extracting TADF-related quantities from publications (`tadf_model_extractor.py`, Wiley loader, shell helpers). These depend on the custom ChemDataExtractor stack referenced in `requirements-data-mining.txt`.

## Citation

Please use the following the citation if you use any part of this codebase in your work.

```bibtex
```

## License

This project is licensed under the **MIT License**; see [`LICENSE`](LICENSE).

Copyright (c) 2025 Dingyun Huang.
