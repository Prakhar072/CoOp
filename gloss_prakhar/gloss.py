import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

EPS = 1e-10


def normalize_adj(adj):
    """Normalize adjacency matrix."""
    adj = adj.to_dense() if hasattr(adj, 'to_dense') else adj
    rowsum = torch.sum(adj, dim=1)       
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    ret = adj.mm(d_mat_inv_sqrt).transpose(0, 1).mm(d_mat_inv_sqrt)
    ret = torch.where(torch.isnan(ret), torch.tensor(EPS, device=ret.device, dtype=ret.dtype), ret)
    ret = torch.where(torch.isinf(ret), torch.tensor(EPS, device=ret.device, dtype=ret.dtype), ret)
    return ret


def guassian(emb, sigma):
    """Compute Gaussian kernel weights."""
    sq_dists = torch.cdist(emb, emb, p=2) ** 2 
    weight = torch.exp(-sq_dists / (2*(sigma**2)))
    weight = weight - torch.diag(torch.diag(weight))
    return weight


def modified_lpa(train_emb, test_emb, Ytrain, sigma, num_labels, device, labels_orig=None):
    """Modified Label Propagation Algorithm for unsupervised setting."""
    # Get dtype from input embeddings to handle mixed precision training
    dtype = train_emb.dtype
    
    emb = torch.cat((train_emb, test_emb), dim=0)
    num_nodes = emb.shape[0]
    labels = torch.cat((Ytrain, torch.zeros(test_emb.shape[0], device=device, dtype=Ytrain.dtype)), dim=0)

    Y = torch.zeros((num_nodes, num_labels), device=device, dtype=dtype)
    for k in range(num_labels):
        Y[labels == k, k] = 1

    train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    train_mask[:Ytrain.shape[0]] = 1
    test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    test_mask[Ytrain.shape[0]:Ytrain.shape[0]+test_emb.shape[0]] = 1

    emb = emb / emb.norm(dim=1, keepdim=True)
    
    adj = guassian(emb, sigma).to(device)
    adj = adj + adj.t()
    
    if (torch.sum(torch.isnan(adj))) or (torch.sum(torch.isinf(adj))):
        raise ValueError("NaN in adj after symmetrization")
    
    adj_norm = normalize_adj(adj)
    adj_norm = adj_norm.to_dense()
    
    if (torch.sum(torch.isnan(adj_norm))) or (torch.sum(torch.isinf(adj_norm))):
        raise ValueError("NaN in adj after processing")

    # Transition matrix
    Tran = adj_norm / adj_norm.sum(dim=0, keepdim=True)
    row_sum = Tran.sum(dim=1, keepdim=True)
    T = Tran / row_sum

    N_l = train_emb.shape[0]
    T_ul = T[N_l:, :N_l]
    T_uu = T[N_l:, N_l:]

    I = torch.eye(T_uu.shape[0], device=device, dtype=dtype)
    
    # torch.linalg.solve requires float32 or float64, so convert if needed
    if dtype in [torch.float16, torch.bfloat16]:
        I_solve = I.float()
        T_uu_solve = T_uu.float()
        T_ul_solve = T_ul.float()
        Y_train_solve = Y[train_mask].float()
        F_UU = torch.linalg.solve(I_solve - T_uu_solve, T_ul_solve.mm(Y_train_solve))
        F_UU = F_UU.to(dtype)
    else:
        F_UU = torch.linalg.solve(I - T_uu, T_ul.mm(Y[train_mask]))
    
    if torch.any(torch.isnan(F_UU)) or torch.any(torch.isinf(F_UU)):
        raise ValueError("NaN in F_UU before normalization")

    return F_UU


def gloss(X, labels, sigma, num_labels, gamma, device, class_weights): 
    """Compute G-Loss using Label Propagation.
    
    Args:
        X (torch.Tensor): Model output logits of shape (batch_size, num_labels)
        labels (torch.Tensor): Ground truth class indices of shape (batch_size,)
        sigma (float): Gaussian kernel bandwidth parameter
        num_labels (int): Number of classes
        gamma (float): Fraction of samples to use as labeled set (0 < gamma < 1)
        device (torch.device): Device to perform computations on
        class_weights (list or torch.Tensor): Class weight for loss computation
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    embedding = X    
    
    mask1 = torch.rand(embedding.size(0), device=device) < gamma
    mask2 = ~mask1
    emb_lab_set = embedding[mask1]
    emb_eval_set = embedding[mask2]

    labels_lab_set = labels[mask1]
    labels_eval_set = labels[mask2]
    predicted_labels = modified_lpa(emb_lab_set, emb_eval_set, labels_lab_set, sigma, num_labels, device=device, labels_orig=labels)
    loss = F.cross_entropy(predicted_labels, labels_eval_set, weight=torch.tensor(class_weights, dtype=embedding.dtype, device=device))
    return loss