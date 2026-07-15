![](./assets/demo_2x.gif)

## TADF photoluminescence prediction

Code repository for curating a dataset of experimental PL wavelengths and predicting them for thermally activated delayed fluorescence (TADF) molecules from molecular structure using graph neural networks (GNNs) and ChemDataExtractorTADF, plus simpler fingerprint baselines.

Paper Here: [Machine-Learning Predictions of Photoluminescence in Molecules Exhibiting Thermally Activated Delayed Fluorescence with Implicit Experimental Validation](https://doi.org/10.1021/acs.jcim.6c00425).

If you encounter problems running the code, please [open an issue](https://github.com/Dingyun-Huang/tadf-photoluminescence-prediction/issues).

**Note:** Please refer to and install [ChemDataExtractorTADF](https://github.com/Dingyun-Huang/chemdataextractorTADF) if you want to rerun the dataset extraction for your own task.

### Requirements

Python **3.10–3.12** is recommended.

**Create an environment**

```bash
conda create -n tadf-predict python=3.11
conda activate tadf-predict
```

**Install dependencies**

```bash
pip install -r requirements-model-training.txt
```

This installs a CPU build that works everywhere. For GPU acceleration, edit the PyTorch/PyG URLs at the top of `requirements-model-training.txt` to match your CUDA version (see [pytorch.org](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)).

### Inference (HGNN)

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

### Run training

You **must** run from `src/gnn` so relative paths resolve to `src/data_processing`:

```bash
cd src/gnn
python run_experiment.py --seed 2000
```

**Configuration**

Training behavior is controlled by **`src/gnn/gnn_config.yaml`**:

- Architecture: heterogeneous GraphSAGE on typed bonds (`hetero: true`), MLP head, optional solvent/molecular-size features.
- Training: batch size, epochs, learning rate, normalization, and toggles for train/eval/test, checkpointing, and plots.
- Checkpoints: set `save_model: true` and `save_path` to write a model archive (weights, config, and normalization stats).


### Citation

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

### Acknowledgements

D.H. is thankful to the Cambridge Commonwealth, European and International Trust and the China Scholarship Council, for a Ph.D. scholarship. J.M.C. is grateful for funding from the EPSRC AI Hub, AIChemy (grant references EP/Y028775/1 and EP/Y028759/1). The authors thank the Argonne Leadership Computing Facility, which is a DOE Office of Science Facility, for use of its research resources, under contract no. DE-AC02-06CH11357.

### License

This project is licensed under the **MIT License**; see [`LICENSE`](LICENSE).

Copyright (c) 2025 Dingyun Huang.
