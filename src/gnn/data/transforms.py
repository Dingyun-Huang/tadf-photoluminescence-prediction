"""
Data transformation utilities
"""

import torch


def augment_pl_with_target_noise(data, noise_std=0.008):
    """
    Augment the data by adding noise to the pl values.
    
    Args:
        data: Data object to augment
        noise_std: Standard deviation of the noise to add
        
    Returns:
        Augmented data object
    """
    noisy_data = data.clone()
    noise = torch.randn_like(data.pl) * noise_std
    noisy_data.pl = data.pl + noise
    return noisy_data


def denormalize_tensor(pl_tensor, pl_mean, pl_std):
    """Denormalize a tensor of PL values."""
    return pl_tensor * pl_std + pl_mean