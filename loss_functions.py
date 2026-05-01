"""
Loss function implementations for CoOp training.

This module provides various loss functions for training.
Add custom loss implementations here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(loss_fn_name):
    """
    Get the loss function by name.
    
    Args:
        loss_fn_name (str): Name of the loss function ('cross_entropy', 'custom', etc.)
    
    Returns:
        callable: Loss function that takes (output, label) and returns loss
    """
    if loss_fn_name == "cross_entropy":
        return cross_entropy_loss
    elif loss_fn_name == "custom":
        return custom_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


def cross_entropy_loss(output, label):
    """
    Standard cross-entropy loss.
    
    Args:
        output (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        label (torch.Tensor): Ground truth labels of shape (batch_size,)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    return F.cross_entropy(output, label)


def custom_loss(output, label):
    """
    Placeholder for custom loss function.
    
    TODO: Implement your custom loss function here.
    
    Args:
        output (torch.Tensor): Model output logits of shape (batch_size, num_classes)
        label (torch.Tensor): Ground truth labels of shape (batch_size,)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Placeholder: falls back to cross-entropy for now
    # Replace this implementation with your custom loss function
    raise NotImplementedError(
        "Custom loss function has not been implemented yet. "
        "Please implement the custom_loss function in loss_functions.py"
    )
