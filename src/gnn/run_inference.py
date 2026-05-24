"""
Run HGNN inference on a SMILES string using a saved model archive.
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

DATA_DIR = "../data_processing"
module_path = os.path.abspath(DATA_DIR)
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.insert(0, ".")

from archs import GNNNet
from data import TadfDataManager
from data.featurizer import smiles_to_hetero_data


DEFAULT_MODEL_PATH = os.path.join(
    "../../model_archive", "model_8_layer_28_h.pt" # change the path to the checkpoint that you want to use
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model_archive(model_path: str, config: dict, device: str):
    """
    Load a saved model archive.

    Supports either:
    - a state-dict checkpoint (.pt), or
    - a full archive dict with keys: state_dict, config, pl_mean, pl_std
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        archive_config = checkpoint.get("config", config)
        pl_mean = checkpoint.get("pl_mean")
        pl_std = checkpoint.get("pl_std")
    else:
        state_dict = checkpoint
        archive_config = config
        pl_mean = None
        pl_std = None

    model = GNNNet(**archive_config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, archive_config, pl_mean, pl_std


def get_normalization_stats(config: dict, data_manager: TadfDataManager):
    """Compute PL normalization stats from the training split."""
    tadf_train, _, tadf_val = data_manager.load_tadf_data(
        force_reload=False,
        use_H=config.get("use_H", False),
    )
    if config.get("use_validation_split", True):
        tadf_train = tadf_train + tadf_val

    pl_values = [data.pl for data in tadf_train]
    return float(np.mean(pl_values)), float(np.std(pl_values))


def predict_pl(
    smiles: str,
    model_path: str = DEFAULT_MODEL_PATH,
    config_path: str = "gnn_config.yaml",
    device: str = "auto",
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = load_config(config_path)
    model, archive_config, pl_mean, pl_std = load_model_archive(
        model_path, config, device
    )

    if pl_mean is None or pl_std is None:
        data_manager = TadfDataManager()
        pl_mean, pl_std = get_normalization_stats(archive_config, data_manager)

    graph = smiles_to_hetero_data(smiles, use_H=archive_config.get("use_H", False))
    graph.pl = torch.tensor(
        [(0.0 - pl_mean) / pl_std], dtype=torch.float
    )

    loader = DataLoader([graph], batch_size=1, shuffle=False)
    data = next(iter(loader)).to(device)

    with torch.no_grad():
        edge_index_dict = {key: value.edge_index for key, value in data.edge_items()}
        normalized_prediction = model(data["atom"].x, edge_index_dict, data)

    pl_ev = float(normalized_prediction.item() * pl_std + pl_mean)
    pl_nm = 1240.0 / pl_ev if pl_ev > 0 else float("nan")

    return {
        "smiles": smiles,
        "pl_ev": pl_ev,
        "pl_nm": pl_nm,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict TADF photoluminescence from a SMILES string."
    )
    parser.add_argument(
        "--smiles",
        required=True,
        help="SMILES string for the molecule to predict.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the saved model archive (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--config",
        default="gnn_config.yaml",
        help="Path to the model configuration YAML file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(
            f"Model archive not found: {args.model}\n"
            "Train a model with save_model: true in gnn_config.yaml, "
            "or pass --model to an existing checkpoint."
        )

    results = predict_pl(
        smiles=args.smiles,
        model_path=args.model,
        config_path=args.config,
        device=args.device,
    )

    print(f"SMILES: {results['smiles']}")
    print(f"Predicted PL energy: {results['pl_ev']:.4f} eV")
    print(f"Predicted PL wavelength: {results['pl_nm']:.1f} nm")


if __name__ == "__main__":
    main()
