"""
Main script to run hyperparameter sweep using wandb.
"""

import os
import sys
import yaml
import wandb
from torch_geometric.seed import seed_everything
import torch

# Add data processing path
DATA_DIR = "../../data_processing"
module_path = os.path.abspath(DATA_DIR)
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.insert(0, ".")

# Import modular components
from archs import GNNNet
from data import TadfDataManager
from training import GNNTrainer

def train(config=None):
    """
    Main training function for wandb sweep.
    """
    with open("gnn_config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    with wandb.init(config=config):
        config = wandb.config
        
        # Update base_config with sweep parameters
        base_config['batch_size'] = config.batch_size
        base_config['epochs'] = config.epochs
        base_config['learning_rate'] = config.learning_rate
        base_config['gnn']['hidden_channels'] = config.hidden_channels
        base_config['gnn']['out_channels'] = config.get("out_channels", config.hidden_channels)
        base_config['gnn']['num_layers'] = config.num_layers
        base_config['gnn']['aggr'] = config.aggr
        base_config['gnn']['jk'] = config.jk if config.jk != 'null' else None
        base_config['gnn']['act'] = config.act
        base_config['dropout_rate'] = config.dropout_rate
        base_config['input_noise_std'] = config.input_noise_std
        base_config['mlp_hidden_dim'] = config.mlp_hidden_dim
        base_config['use_solvent'] = config.use_solvent
        base_config['use_mol_size'] = config.use_mol_size
        base_config['hetero'] = config.hetero
        base_config['augmentation'] = config.augmentation

        # Set seed for reproducibility
        seed_everything(42)
        
        # Load data
        data_manager = TadfDataManager()
        tadf_train, tadf_test, tadf_val = data_manager.load_tadf_data()
        
        # For final model, combine train and val
        if not base_config.get("use_validation_split", True):
            tadf_train += tadf_val
        
        # Create model
        model = GNNNet(**base_config)
        
        # Setup data loaders
        train_loader, pl_mean, pl_std = data_manager.get_train_dataloader(
            tadf_train, **base_config
        )
        
        val_loader = data_manager.get_eval_dataloader(
            tadf_val, pl_mean, pl_std, batch_size=base_config["batch_size"]
        )
        
        # Setup trainer
        trainer = GNNTrainer(model, y_mean=pl_mean, y_std=pl_std)
        trainer.setup_optimization(
            learning_rate=base_config["learning_rate"],
            epochs=base_config["epochs"],
            scheduler_type="cosine"
        )
        
        wandb.watch(model, log="all")
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=base_config["epochs"],
            eval_interval=10
        )
        
        # Evaluate on test set
        test_loader = data_manager.get_eval_dataloader(
            tadf_test, pl_mean, pl_std, batch_size=base_config["batch_size"]
        )
        test_results = trainer.evaluate_test_set(test_loader, tadf_test)
        
        wandb.log({
            "test_mae": test_results['mae'],
            "test_r2": test_results['r2_score'],
            "test_rmse": test_results['rmse']
        })
        
        wandb.finish()

def main():
    """Main entry point for sweep."""
    # Load sweep configuration from YAML file
    with open("sweep_config.yaml", 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="tadf_gnn_sweep")
    wandb.agent(sweep_id, train, count=50)

if __name__ == "__main__":
    main()
