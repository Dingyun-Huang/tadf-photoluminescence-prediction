![](./assets/demo_2x.gif)

# TADF photoluminescence prediction

Code repository for predicting thermally activated delayed fluorescence (TADF) photoluminescence properties from molecular structure using graph neural networks (GNNs), plus utilities for dataset construction and simpler fingerprint baselines.

Paper Here: [Machine-Learning Predictions of Photoluminescence in Molecules Exhibiting Thermally Activated Delayed Fluorescence with Implicit Experimental Validation](https://doi.org/10.1021/acs.jcim.6c00425).

If you encounter problems running the code, please [open an issue](https://github.com/Dingyun-Huang/tadf-photoluminescence-prediction/issues).

## Repository layout

| Path | Purpose |
|------|---------|
| `src/gnn/` | Heterogeneous GraphSAGE (HGNN) model, training loop, experiment and inference entry points |
| `src/fingerprints/` | Ridge regression on molecular fingerprints (Jupyter notebook) |
| `mining-codes/` | Literature extraction pipeline (ChemDataExtractor extensions; optional) |

## Requirements

Python **3.10–3.12** is recommended.

- **`requirements-model-training.txt`** — All dependencies for HGNN training (PyTorch, PyG extension wheels, RDKit, Weights & Biases, etc.), installable in one command. Defaults to a CPU build; see the [Quick start](#2-install-dependencies) below.
- **`requirements-data-mining.txt`** — Reference snapshot for the literature-mining pipeline (cluster/conda environment with local wheels and private Git URLs). Treat as a reference only. See also [chemdataextractorTADF](https://github.com/Dingyun-Huang/chemdataextractorTADF) for data-mining setup.

## Quick start (HGNN training)

The model is a heterogeneous **GraphSAGE** network (`hetero: true` in `gnn_config.yaml`), with bond-type-specific message passing.

### 1. Create an environment

```bash
conda create -n tadf-predict python=3.11
conda activate tadf-predict
```

### 2. Install dependencies

From the repository root:

```bash
pip install -r requirements-model-training.txt
```

This installs a CPU build that works everywhere. For GPU acceleration, edit the PyTorch/PyG URLs at the top of `requirements-model-training.txt` to match your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).

### 3. Inference (HGNN)

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

### 4. Run training

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
| PyTorch/PyG wheel not found, or `torch-scatter` mismatch | Make sure the `--extra-index-url` and `-f` URLs at the top of `requirements-model-training.txt` match your platform and CUDA build (CPU by default). |
| `wandb.errors.AuthenticationError` | Set `disable_wandb: true` in `gnn_config.yaml`, or run `wandb login`. |
| `FileNotFoundError` for split CSVs or SDF | Ensure you are in `src/gnn` when running `run_experiment.py`. |
| `NotImplementedError` about RDKit | Install RDKit: `pip install rdkit` (included in `requirements-model-training.txt`). |
| Training is very slow | Use a CUDA PyTorch build if a GPU is available; reduce `epochs` in `gnn_config.yaml` for testing. |


## Citation

Please use the following citation if you use any part of this codebase and/or the TadfPL dataset in your work.

```bibtex
@article{doi:10.1021/acs.jcim.6c00425,
  author = {Huang, Dingyun and Cole, Jacqueline M.},
  title = {Machine-Learning Predictions of Photoluminescence in Molecules Exhibiting Thermally Activated Delayed Fluorescence with Implicit Experimental Validation},
  journal = {Journal of Chemical Information and Modeling},
  volume = {66},
  number = {10},
  pages = {5757-5763},
  year = {2026},
  doi = {10.1021/acs.jcim.6c00425},
  URL = {https://doi.org/10.1021/acs.jcim.6c00425},
  eprint = {https://doi.org/10.1021/acs.jcim.6c00425}
}
```

## Acknowledgements

D.H. is thankful to the Cambridge Commonwealth, European and International Trust and the China Scholarship Council, for a Ph.D. scholarship. J.M.C. is grateful for funding from the EPSRC AI Hub, AIChemy (grant references EP/Y028775/1 and EP/Y028759/1). The authors thank the Argonne Leadership Computing Facility, which is a DOE Office of Science Facility, for use of its research resources, under contract no. DE-AC02-06CH11357.

## License

This project is licensed under the **MIT License**; see [`LICENSE`](LICENSE).

Copyright (c) 2025 Dingyun Huang.
