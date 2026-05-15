"""
Loss function implementations for CoOp training.

This module provides various loss functions for training.
Add custom loss implementations here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
from datetime import datetime


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
    elif loss_fn_name == "custom":
        return gloss
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_name}")


def cross_entropy_loss(output, label, n_cls):
    return F.cross_entropy(output, label)

def gaussian_similarity(emb, sigma):
        """Compute Gaussian kernel weights."""
        sq_dists = torch.cdist(emb, emb, p=2) ** 2 
        weight = torch.exp(-sq_dists / (2*(sigma**2)))
        weight = weight - torch.diag(torch.diag(weight))
        return weight

def normalize_adj(adj: torch.Tensor):
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degrees = adj.sum(dim=1)
    degrees = degrees.clamp(min=1e-6)
    deg_inv_sqrt = torch.pow(degrees, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def plot_graph(adj):
    base_dir = "output/graphs"
    os.makedirs(base_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(adj.detach().cpu().numpy(), cmap="coolwarm", vmin=0.0, vmax=1.0)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(f"Adjacency matrix")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.tight_layout()
    plt.savefig(osp.join(base_dir, "adj_mat.png"), dpi=300)
    plt.close()

# def plot_graph(adj):
#     """Save adjacency matrix visualization to output/graphs/ with timestamp."""
#     base_dir = "output/graphs"
#     os.makedirs(base_dir, exist_ok=True)
    
#     # Use timestamp to avoid overwriting
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
#     filename = f"adj_mat_{timestamp}.png"
#     filepath = osp.join(base_dir, filename)
    
#     plt.figure(figsize=(6, 6))
#     plt.imshow(adj.detach().cpu().numpy(), cmap="coolwarm", vmin=0.0, vmax=1.0)
#     plt.colorbar(fraction=0.046, pad=0.04)
#     plt.title("Adjacency matrix")
#     plt.xlabel("Node")
#     plt.ylabel("Node")
#     plt.tight_layout()
#     plt.savefig(filepath, dpi=300)
#     plt.close()
#     print(f"Adjacency matrix saved to {filepath}")

def gloss_lpa(train_emb, test_emb, Ytrain, sigma, num_labels):
    device = train_emb.device
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0]

    labels = torch.cat(
        [Ytrain, torch.zeros(test_emb.shape[0], dtype=Ytrain.dtype, device=device)],
        dim=0,
    )

    Y = torch.zeros((num_nodes, num_labels), dtype=torch.float32, device=device)
    for k in range(num_labels):
        Y[labels == k, k] = 1.0

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[: Ytrain.shape[0]] = True

    # emb = emb / emb.norm(dim=1, keepdim=True)
    # if torch.isnan(emb).any() or torch.isinf(emb).any():
    #     raise ValueError("NaN or inf in embeddings")

    adj = gaussian_similarity(emb, sigma).to(torch.float32) 
    plot_graph(adj)
    adj = adj + adj.t()
    adj_norm = normalize_adj(adj).to_dense()
    #todo:might not work error 
    #print(f"Adjacency matrix stats: min={adj_norm.min().item():.4f}, max={adj_norm.max().item():.4f}, mean={adj_norm.mean().item():.4f}, std={adj_norm.std().item():.4f}")

    # Tran = adj_norm / adj_norm.sum(dim=0, keepdim=True)
    # row_sum = Tran.sum(dim=1, keepdim=True)
    # T = Tran / row_sum

    T = adj_norm / adj_norm.sum(dim=1, keepdim=True).clamp(min=1e-6)
    N_l = train_emb.shape[0]
    T_ul = T[N_l:, :N_l]
    T_uu = T[N_l:, N_l:]

    I = torch.eye(T_uu.shape[0], dtype=torch.float32, device=device)
    F_UU = torch.linalg.solve(I - T_uu, T_ul.mm(Y[train_mask]))
    # F_UU = torch.linalg.solve((I - T_uu) + 1e-8 * I, T_ul.mm(Y[train_mask]))

    if torch.isnan(F_UU).any() or torch.isinf(F_UU).any():
        raise ValueError("NaN or inf in F_UU before normalization")

    return F_UU

def gloss(output, labels, n_cls, sigma=2.0, gamma=0.9):
    # gloss_temp = nn.Parameter(torch.tensor(1.0))
    # print("Computing GLoss ...")
    # print(f"GLoss parameters: sigma={sigma}, gamma={gamma}")
    # print(f"output shape: {output.shape}")
    mask1 = torch.randperm(output.size(0)) < output.size(0) * gamma
    mask2 = ~mask1
    emb_lab_set = output[mask1]
    emb_eval_set = output[mask2]
    labels_lab_set = labels[mask1]
    labels_eval_set = labels[mask2]
    pred = gloss_lpa(
        emb_lab_set,
        emb_eval_set,
        labels_lab_set,
        sigma,
        n_cls)
        #self.model.prompt_learner.n_cls
        ## get_loss_function(...)
        #def get_loss_function(name, n_cls=None):
        #   if name == "custom":
        #   return lambda out, lab: gloss(out, lab, n_cls)
        #in Trainer.forward_backward
        #loss_fn = get_loss_function(self.cfg.TRAINER.LOSS_FUNCTION, n_cls=self.model.prompt_learner.n_cls)
    loss = F.cross_entropy(pred, labels_eval_set)
    return loss
