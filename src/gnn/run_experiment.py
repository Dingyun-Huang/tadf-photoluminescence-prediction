"""
Refactored main training script using modular components
"""

import os
import sys
import torch
import yaml
import wandb
import argparse
from torch_geometric.seed import seed_everything

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
from matplotlib import pyplot as plt

def plot_predictions(train_data, val_data, test_data, train_pred_y, val_pred_y, test_pred_y):
    
    fig = plt.figure(figsize=(5, 6))
    train_pred_y = [y for y in train_pred_y]
    plt.plot([1, 4.5], [1, 4.5], linestyle='--', c='k')
    plt.scatter([d.pl for d in train_data], train_pred_y, alpha=0.3, label ="tadf train")
    if val_pred_y is not None:
        val_pred_y = [y for y in val_pred_y]
        plt.scatter([d.pl for d in val_data], val_pred_y, alpha=0.3, label='tadf val')
    if test_pred_y is not None:
        test_pred_y = [y for y in test_pred_y]
        plt.scatter([d.pl for d in test_data], test_pred_y, alpha=0.3, label='tadf test')
    plt.xlim([1.5,4.5])
    plt.ylim([1.5,4.5])
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('TADF GAT Predictions')
    plt.grid()
    plt.show()


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, config_path="gnn_config.yaml", seed=2000):
        """Initialize experiment with configuration."""
        self.config = self.load_config(config_path)
        self.data_manager = TadfDataManager()
        self.seed = seed
        self.setup_experiment()

    def load_config(self, config_path):
        """Load experiment configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_experiment(self):
        """Setup experiment components."""
        # Set seed for reproducibility
        seed_everything(self.seed)
        
        # Load data
        print(f"Using hydrogen atoms: {self.config.get('use_H', False)}")
        self.tadf_train, self.tadf_test, self.tadf_val = self.data_manager.load_tadf_data(use_H=self.config.get("use_H", False))
        
        # For final model, combine train and val
        if self.config.get("use_validation_split", True):
            self.tadf_train += self.tadf_val
    
    def run_training(self):
        """Run the training experiment."""
        config = self.config
        
        # Create model
        model = GNNNet(**config)
        
        # Setup data loaders
        train_loader, pl_mean, pl_std = self.data_manager.get_train_dataloader(
            self.tadf_train, 
            batch_size=config["batch_size"],
            augmentation=config.get("augmentation", None)
        )
        
        val_loader = None
        if config.get("do_eval", True):
            val_loader = self.data_manager.get_eval_dataloader(
                self.tadf_val, pl_mean, pl_std
            )
        
        # Setup trainer
        if config.get("criterion", None):
            criterion = getattr(torch.nn, config["criterion"])()
        else:
            criterion = None
        trainer = GNNTrainer(model, y_mean=pl_mean, y_std=pl_std, criterion=criterion)
        trainer.setup_optimization(
            learning_rate=config["learning_rate"],
            epochs=config["epochs"]
        )
        
        # Initialize wandb
        if config.get("do_train", True):
            wandb.init(
                project="tadf_gnn_training",
                group=config.get("group"),
                tags=config.get("tags", []),
                config=config,
                mode="disabled" if config.get("disable_wandb", False) else "online",
            )
            wandb.watch(model, log="all")
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"]
        )
        
        # Save model if requested
        if config.get("save_model", False):
            save_path = config.get("save_path", "default_save_dir")
            os.makedirs(save_path, exist_ok=True)
            trainer.save_model(os.path.join(save_path, "model_8_layer_28_h.pt"))
        
        # Evaluate on test set
        if config.get("do_predict", True):
            test_loader = self.data_manager.get_eval_dataloader(
                self.tadf_test, pl_mean, pl_std
            )
            test_results = trainer.evaluate_test_set(test_loader, self.tadf_test)
            
            print("Test Set Results:")
            for key, value in test_results.items():
                if key not in ['predictions', 'true_values']:
                    print(f"{key}: {value:.4f}")
        
        # Plot predictions if requested
        if config.get("plot_predictions", False):
            self._plot_results(trainer, pl_mean, pl_std)
        
        wandb.finish()
        return trainer
    
    def _plot_results(self, trainer, pl_mean, pl_std):
        """Generate prediction plots."""
        # Get predictions for all splits
        train_loader = self.data_manager.get_eval_dataloader(
            self.tadf_train, pl_mean, pl_std
        )
        val_loader = self.data_manager.get_eval_dataloader(
            self.tadf_val, pl_mean, pl_std
        )
        test_loader = self.data_manager.get_eval_dataloader(
            self.tadf_test, pl_mean, pl_std
        )
        
        train_results = trainer.evaluate_test_set(train_loader, self.tadf_train)
        val_results = trainer.evaluate_test_set(val_loader, self.tadf_val)
        test_results = trainer.evaluate_test_set(test_loader, self.tadf_test)
        
        plot_predictions(
            train_data=self.tadf_train,
            val_data=self.tadf_val,
            test_data=self.tadf_test,
            train_pred_y=train_results['predictions'],
            val_pred_y=val_results['predictions'],
            test_pred_y=test_results['predictions']
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GNN training experiment.")
    parser.add_argument('--seed', type=int, default=2000, help='Random seed for the experiment.')
    args = parser.parse_args()
    
    runner = ExperimentRunner(seed=args.seed)
    runner.run_training()


if __name__ == "__main__":
    main()