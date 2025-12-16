"""
GNN Model Definition for TADF Property Prediction
"""

import torch
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import to_hetero


class GNNNet(torch.nn.Module):
    """
    Graph Neural Network for TADF property prediction.
    
    Args:
        gnn: Dictionary containing GNN configuration parameters
        use_solvent: Whether to include solvent features
        use_mol_size: Whether to include molecular size features
        dropout_rate: Dropout rate for regularization
        input_noise_std: Standard deviation for input noise augmentation
        mlp_hidden_dim: Hidden dimension for the MLP layers
    """
    
    def __init__(self, **kwargs):
        super(GNNNet, self).__init__()
        
        # Extract configuration
        gnn_config = kwargs.get("gnn", {})
        self.use_solvent = kwargs.get("use_solvent", False)
        self.use_mol_size = kwargs.get("use_mol_size", False)
        self.dropout_rate = kwargs.get("dropout_rate", 0.1)
        self.input_noise_std = kwargs.get("input_noise_std", 0.1)
        
        # Input features (12 for atomic features)
        self.input_features = 12
        self.lin_input = torch.nn.Linear(self.input_features, self.input_features)
        
        # GNN backbone
        gnn_block = GraphSAGE(
            in_channels=self.input_features,
            **gnn_config
        )
        if not kwargs.get("hetero", False):
            print("Using homogeneous GNN architecture")
            self.gnn_block = gnn_block
        else:
            print("Using heterogeneous GNN architecture")
            self.gnn_block = to_hetero(gnn_block, metadata=(['atom'],
                                                            [('atom', 'single', 'atom'),
                                                             ('atom', 'double', 'atom'),
                                                             ('atom', 'triple', 'atom'),
                                                             ('atom', 'aromatic', 'atom')]))
        
        # Set output channels if not specified
        if "out_channels" not in gnn_config.keys():
            gnn_config["out_channels"] = gnn_config["hidden_channels"]
        
        # Batch normalization layers
        self.batch_norm0 = BatchNorm(self.input_features)
        self.batch_norm1 = BatchNorm(gnn_config["out_channels"])
        
        # Calculate MLP input dimension
        mlp_hidden_dim = kwargs.get("mlp_hidden_dim", 64)
        mlp_input_dim = gnn_config["out_channels"]
        if self.use_solvent:
            mlp_input_dim += 12  # Solvent features
        
        # MLP layers
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.lin1 = torch.nn.Linear(mlp_input_dim, mlp_hidden_dim)
        self.batch_norm2 = BatchNorm(mlp_hidden_dim)
        
        if self.use_mol_size:
            mlp_hidden_dim += 1   # Molecular size
        self.lin2 = torch.nn.Linear(mlp_hidden_dim, 1)

    def forward(self, x, edge_index_dict, data):
        """
        Forward pass of the GNN model.
        
        Args:
            x: Node features
            bond_index: Edge indices
            bond_attr: Edge attributes
            data: Batch data object containing additional features
            
        Returns:
            Predicted values (flattened tensor)
        """
        # Sort edge indices for consistency
        # bond_index = sort_edge_index(bond_index, sort_by_row=False)
        
        # Input normalization and noise augmentation
        x = self.lin_input(x)
        x = self.batch_norm0(x)
        if self.training:
            x += torch.randn_like(x) * self.input_noise_std
        x = torch.nn.functional.silu(x)
        
        # GNN forward pass
        x = self.gnn_block({'atom': x}, edge_index_dict)
        
        # Graph-level pooling
        x = global_max_pool(x['atom'], data['atom'].batch)
        x = self.batch_norm1(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        
        # Add additional features if specified
        if self.use_solvent:
            x = torch.cat([x, data.solvent], dim=1)
        # if self.use_mol_size:
        #     x = torch.cat([x, data.mol_size.unsqueeze(1)], dim=1)
        
        # MLP layers
        x = self.lin1(x)
        x = self.batch_norm2(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        if self.use_mol_size:
            x = torch.cat([x, data.mol_size.unsqueeze(1)], dim=1)
        x = self.lin2(x)
        
        return x.flatten()
    
    def get_config(self):
        """Return model configuration for reproducibility."""
        return {
            'use_solvent': self.use_solvent,
            'use_mol_size': self.use_mol_size,
            'dropout_rate': self.dropout_rate,
            'input_noise_std': self.input_noise_std,
            'input_features': self.input_features
        }