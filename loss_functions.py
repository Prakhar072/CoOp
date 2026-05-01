"""
Loss function implementations for CoOp training.

This module provides various loss functions for training.
Add custom loss implementations here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gloss import gloss as gloss_loss


def get_loss_function(loss_fn_name, cfg=None):
    """
    Get the loss function by name.
    
    Args:
        loss_fn_name (str): Name of the loss function ('cross_entropy', 'gloss', etc.)
        cfg (Config, optional): Configuration object. Required for 'gloss' loss function.
    
    Returns:
        callable: Loss function that takes (output, label) and returns loss
    """
    if loss_fn_name == "cross_entropy":
        return cross_entropy_loss
    elif loss_fn_name == "gloss" or loss_fn_name == "custom":
        if cfg is None:
            raise ValueError("cfg is required for gloss loss function")
        return _create_gloss_wrapper(cfg)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


def _create_gloss_wrapper(cfg):
    """
    Create a gloss loss wrapper function that captures config parameters.
    
    Args:
        cfg (Config): Configuration object with TRAINER.GLOSS parameters
    
    Returns:
        callable: Loss function with signature (output, label) -> loss
    """
    def gloss_wrapper(output, label):
        """
        Wrapper that adapts gloss to standard loss function interface.
        
        Args:
            output (torch.Tensor): Model output logits of shape (batch_size, num_classes)
            label (torch.Tensor): Ground truth class indices of shape (batch_size,)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Extract parameters from config
        sigma = cfg.TRAINER.GLOSS.SIGMA
        gamma = cfg.TRAINER.GLOSS.GAMMA
        class_weights = cfg.TRAINER.GLOSS.CLASS_WEIGHTS
        
        # If class_weights is None, use uniform weights
        if class_weights is None:
            num_labels = output.shape[1]
            class_weights = [1.0] * num_labels
        
        # Infer num_labels from output shape
        num_labels = output.shape[1]
        
        # Get device from output tensor
        device = output.device
        
        # Call gloss with extracted parameters
        return gloss_loss(output, label, sigma, num_labels, gamma, device, class_weights)
    
    return gloss_wrapper


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


# def custom_loss(output, label):
#     """
#     Placeholder for custom loss function.
    
#     TODO: Implement your custom loss function here.
    
#     Args:
#         output (torch.Tensor): Model output logits of shape (batch_size, num_classes)
#         label (torch.Tensor): Ground truth labels of shape (batch_size,)
    
#     Returns:
#         torch.Tensor: Scalar loss value
#     """
#     # Placeholder: falls back to cross-entropy for now
#     # Replace this implementation with your custom loss function
#     raise NotImplementedError(
#         "Custom loss function has not been implemented yet. "
#         "Please implement the custom_loss function in loss_functions.py"
#     )
