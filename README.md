![](https://raw.githubusercontent.com/dingyun-huang/tadf-photoluminescence-prediction/feature/inference-script/assets/demo.mp4)

# TADF photoluminescence prediction

Code for predicting thermally activated delayed fluorescence (TADF) photoluminescence properties from molecular structure using graph neural networks (GNNs), plus utilities for dataset construction and simpler fingerprint baselines.

If you encounter problems running the code, please [open an issue](https://github.com/Dingyun-Huang/tadf-photoluminescence-prediction/issues).

## Repository layout

| Path | Purpose |
|------|---------|
| `src/gnn/` | Heterogeneous GraphSAGE (HGNN) model, training loop, experiment and inference entry points |
| `src/fingerprints/` | Ridge regression on molecular fingerprints (Jupyter notebook) |
| `mining-codes/` | Literature extraction pipeline (ChemDataExtractor extensions; optional) |

## Requirements

Python **3.10–3.12** is recommended.

- **`requirements-model-training.txt`** — Direct dependencies for HGNN training (PyTorch Geometric, RDKit, Weights & Biases, etc.). PyTorch and PyG extension wheels must be installed **first** using the staged commands below; they are not resolved from PyPI alone.
- **`requirements-data-mining.txt`** — Reference snapshot for the literature-mining pipeline (cluster/conda environment with local wheels and private Git URLs). Treat as a reference only. See also [chemdataextractorTADF](https://github.com/Dingyun-Huang/chemdataextractorTADF) for data-mining setup.

## Quick start (HGNN training)

The model is a heterogeneous **GraphSAGE** network (`hetero: true` in `gnn_config.yaml`), with bond-type-specific message passing.

### 1. Create an environment

```bash
conda create -n tadf-predict python=3.11
conda activate tadf-predict
```

### 2. Install PyTorch

Pick the line that matches your hardware. For other CUDA versions, see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

**CUDA 12.4 (GPU):**

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**CPU only:**

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### 3. Install PyG extension wheels

PyG publishes separate wheels per PyTorch/CUDA build. Use `torch-2.4.0` in the URL for the PyTorch 2.4.* line ([PyG installation docs](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).

**CUDA 12.4:**

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

**CPU:**

```bash
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

### 4. Install remaining training dependencies

From the repository root:

```bash
pip install -r requirements-model-training.txt
```

### 5. Inference (HGNN)

Run predictions from a SMILES string using a saved model archive. You **must** run from `src/gnn` (same as training).

```bash
cd src/gnn
python run_inference.py --smiles "c1ccc2c(c1)Oc1ccccc1N2c1ccc2c3c(cccc13)-c1nc3cc(-c4ccncc4)c(-c4ccncc4)cc3nc1-2"
```

Optional arguments:

- `--model PATH` — path to a saved archive (default: `model/graphsage_hetero_model/model_8_layer_28_h.pt`)
- `--config PATH` — model configuration YAML (default: `gnn_config.yaml`)
- `--device {auto,cpu,cuda}` — compute device (default: `auto`)

The script prints predicted photoluminescence in **eV** and **nm**. If the archive does not include normalization statistics (older state-dict-only checkpoints), they are recomputed from the training split.

### 6. Run training

You **must** run from `src/gnn` so relative paths resolve to `src/data_processing`:

```bash
cd src/gnn
python run_experiment.py --seed 2000
```

**First run:** RDKit preprocesses `TadfPL_rdkit_one_conf.sdf` into `src/data_processing/tadf_pl/processed/data_one_conf.pt`. This takes a few minutes and requires RDKit (included in the requirements file).

**Default training:** 400 epochs. For a quick smoke test, temporarily lower `epochs` in `gnn_config.yaml` (e.g. to `2`).

### Weights & Biases

By default, `disable_wandb: true` in `gnn_config.yaml`, so no W&B account is needed for a local run. To enable online logging, set `disable_wandb: false` and run `wandb login`.

## Dataset

The bundled **TadfPL** dataset lives under `src/data_processing/tadf_pl/`:

- Features and targets are built by `src/data_processing/tadf_dataset.py` (`TadfPL`).
- Train / validation / test molecule IDs are in `src/data_processing/tadf_pl/raw/train.csv`, `val.csv`, and `test.csv`.
- `TadfPL_rdkit_one_conf.sdf` and `TadfPL.csv` are the raw inputs for preprocessing.

**RDKit is required** with the bundled data. The repository does not ship a preprocessed cache; the dataset is built on first run.

## Configuration

Training behavior is controlled by **`src/gnn/gnn_config.yaml`**:

- Architecture: heterogeneous GraphSAGE on typed bonds (`hetero: true`), MLP head, optional solvent/molecular-size features.
- Training: batch size, epochs, learning rate, normalization, and toggles for train/eval/test, checkpointing, and plots.
- Checkpoints: set `save_model: true` and `save_path` to write a model archive (weights, config, and normalization stats).

### Hyperparameter sweep (W&B)

From **`src/gnn`**, with W&B enabled and logged in:

```bash
python sweep.py
```

This reads **`sweep_config.yaml`** and runs a Weights & Biases sweep (`wandb.sweep` / `wandb.agent`).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No matching distribution found for torch==2.4.1+cu124` | Install PyTorch separately with `--index-url` (step 2 above); do not rely on `pip install -r` alone for CUDA wheels. |
| PyG import error or `torch-scatter` mismatch | Reinstall PyG extensions using the wheel URL that matches your installed PyTorch and CUDA (step 3 above). |
| `wandb.errors.AuthenticationError` | Set `disable_wandb: true` in `gnn_config.yaml`, or run `wandb login`. |
| `FileNotFoundError` for split CSVs or SDF | Ensure you are in `src/gnn` when running `run_experiment.py`. |
| `NotImplementedError` about RDKit | Install RDKit: `pip install rdkit` (included in `requirements-model-training.txt`). |
| Training is very slow | Use a CUDA PyTorch build if a GPU is available; reduce `epochs` in `gnn_config.yaml` for testing. |


## Citation

Please use the following citation if you use any part of this codebase in your work.

```bibtex
@article{doi:10.1021/acs.jcim.6c00425,
  author = {Huang, Dingyun and Cole, Jacqueline M.},
  title = {Machine-Learning Predictions of Photoluminescence in Molecules Exhibiting Thermally Activated Delayed Fluorescence with Implicit Experimental Validation},
  journal = {Journal of Chemical Information and Modeling},
  volume = {0},
  number = {0},
  pages = {null},
  year = {0},
  doi = {10.1021/acs.jcim.6c00425},
  URL = {https://doi.org/10.1021/acs.jcim.6c00425},
  eprint = {https://doi.org/10.1021/acs.jcim.6c00425}
}
```

## License

This project is licensed under the **MIT License**; see [`LICENSE`](LICENSE).

Copyright (c) 2025 Dingyun Huang.
