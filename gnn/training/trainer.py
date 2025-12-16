"""
Enhanced Trainer class for GNN model training
"""

import torch
import wandb
from tqdm import tqdm
from sklearn import metrics
import numpy as np
from torchinfo import summary


class GNNTrainer:
    """
    Enhanced trainer class for GNN models with better structure and functionality.
    """
    
    def __init__(self, model, device="auto", criterion=None, **kwargs):
        """
        Initialize trainer.
        
        Args:
            model: The GNN model to train
            device: Device to use ('auto', 'cuda', or 'cpu')
            criterion: Loss function (defaults to L1Loss)
        """
        self.model = model
        self.device = self._setup_device(device)
        self.criterion = criterion or torch.nn.L1Loss()
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.step = 0
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }
        
        # Set additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _setup_device(self, device):
        """Setup compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def setup_optimization(self, learning_rate=1e-3,
                          scheduler_type="cosine", epochs=100):
        """
        Setup optimizer and scheduler.
        
        Args:
            learning_rate: Learning rate
            scheduler_type: Type of scheduler ('cosine', 'plateau', etc.)
            epochs: Total number of epochs (needed for cosine scheduler)
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )
        
        # Setup scheduler
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=0
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=15
            )
        else:
            self.scheduler = None
    
    def train_epoch(self, train_loader, log_interval=10):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            log_interval: Interval for logging training loss
        """
        self.model.train()
        epoch_loss = 0.0
        
        for i, data in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            edge_index_dict = {key: value.edge_index for key, value in data.edge_items()}
            predictions = self.model(data['atom'].x, edge_index_dict, data)
            loss = self.criterion(predictions, data.pl)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update step counter and log
            self.step += 1
            epoch_loss += loss.item()
            
            if self.step % log_interval == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }, step=self.step)
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader, return_predictions=False):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            return_predictions: Whether to return predictions
            
        Returns:
            tuple: (mae, loss, predictions if requested)
        """
        self.model.eval()
        total_mae = 0.0
        total_loss = 0.0
        predictions = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                edge_index_dict = {key: value.edge_index for key, value in data.edge_items()}
                preds = self.model(data['atom'].x, edge_index_dict, data)

                mae = torch.nn.L1Loss()(preds, data.pl)
                loss = self.criterion(preds, data.pl)
                
                total_mae += mae.item()
                total_loss += loss.item()
                
                if return_predictions:
                    predictions.extend(preds.cpu().numpy())
        
        avg_mae = total_mae / len(val_loader)
        avg_loss = total_loss / len(val_loader)
        
        # Denormalize predictions if needed
        if return_predictions and hasattr(self, 'y_mean') and hasattr(self, 'y_std'):
            predictions = [p * self.y_std + self.y_mean for p in predictions]
        
        if return_predictions:
            return avg_mae * self.y_std, avg_loss, predictions
        return avg_mae * self.y_std, avg_loss
    
    def train(self, train_loader, val_loader, epochs, eval_interval=10):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            eval_interval: How often to evaluate on validation set
        """
        print(f"Training on {self.device}")
        summary(self.model)
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            if (epoch + 1) % eval_interval == 0 and val_loader is not None:
                train_mae, train_loss_eval = self.validate(train_loader)
                val_mae, val_loss = self.validate(val_loader)
                
                # Log metrics
                wandb.log({
                    "train_mae": train_mae,
                    "val_mae": val_mae,
                    "train_loss": train_loss_eval,
                    "val_loss": val_loss,
                    "epoch": epoch + 1,
                }, step=self.step)
                
                # Store metrics
                self.metrics_history["train_mae"].append(train_mae)
                self.metrics_history["val_mae"].append(val_mae)
                self.metrics_history["train_loss"].append(train_loss_eval)
                self.metrics_history["val_loss"].append(val_loss)
                
                # Print progress
                tqdm.write(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train MAE: {train_mae:.4f} - "
                    f"Val MAE: {val_mae:.4f} - "
                    f"Train Loss: {train_loss_eval:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in locals():
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
    
    def evaluate_test_set(self, test_loader, test_data_raw):
        """
        Evaluate on test set and return detailed metrics.
        
        Args:
            test_loader: Test data loader
            test_data_raw: Raw test data for true values
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        mae, loss, predictions = self.validate(test_loader, return_predictions=True)
        
        # Get true values (denormalized)
        true_values = [data.pl for data in test_data_raw]
        
        # Calculate additional metrics
        r2_score = metrics.r2_score(true_values, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(true_values, predictions))
        
        return {
            'mae': mae,
            'loss': loss,
            'r2_score': r2_score,
            'rmse': rmse,
            'predictions': predictions,
            'true_values': true_values
        }
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))