# GLOSS Loss Function Architecture

## Overview
The GLOSS (Graph-based Label-propagation Loss) system is a semi-supervised learning loss function that uses label propagation on graph embeddings. This document explains the flow and interactions between different components.

## File-by-File Breakdown

### 1. **train.py** - Entry Point & Configuration
**Role**: Main training script that sets up configuration and initiates training

**Key Functions**:
- `extend_cfg(cfg)`: Adds GLOSS-specific configuration parameters to the config object
  ```python
  cfg.TRAINER.GLOSS.SIGMA = 0.1          # Gaussian kernel bandwidth
  cfg.TRAINER.GLOSS.GAMMA = 0.5          # Fraction of samples used as labeled set
  cfg.TRAINER.GLOSS.CLASS_WEIGHTS = None # Optional class weights
  ```

**Config File Flow**:
1. Default config created by dassl
2. Extended with custom GLOSS parameters via `extend_cfg()`
3. Merged with dataset-specific config (e.g., `configs/datasets/oxford_pets.yaml`)
4. Merged with trainer-specific config (e.g., `configs/trainers/CoOp/end.yaml`)
5. Command-line arguments override config values
6. Config frozen (immutable) and passed to trainers

**How you invoke it**:
```bash
python train.py \
  --config-file configs/trainers/CoOp/end.yaml \
  --dataset-config-file configs/datasets/oxford_pets.yaml \
  --loss-function gloss \
  --output-dir output/oxford_pets/CoOp/end_16shots/loss_gloss/seed1 \
  TRAINER.GLOSS.SIGMA 0.1 \
  TRAINER.GLOSS.GAMMA 0.5
```

---

### 2. **loss_functions.py** - Loss Wrapper & Factory
**Role**: Provides a factory pattern to create loss functions with configuration baked in

**Key Functions**:

- `get_loss_function(loss_fn_name, cfg)`: Factory function that returns a callable loss function
  - Takes a string name ("cross_entropy", "gloss", "custom")
  - Returns a function that takes (output, label) → scalar loss
  - For "gloss", creates a wrapper with config parameters bound in

- `_create_gloss_wrapper(cfg)`: Creates a closure that captures config parameters
  ```python
  def gloss_wrapper(output, label):
      # Extract GLOSS hyperparameters from cfg
      sigma = cfg.TRAINER.GLOSS.SIGMA
      gamma = cfg.TRAINER.GLOSS.GAMMA
      class_weights = cfg.TRAINER.GLOSS.CLASS_WEIGHTS
      
      # Call the actual gloss implementation with hydrated parameters
      return gloss_loss(output, label, sigma, num_labels, gamma, device, class_weights)
  ```
  
  This is necessary because PyTorch needs a standard loss function signature `(output, label) → loss`, but GLOSS requires additional parameters like sigma, gamma, etc.

**Why this abstraction exists**:
- The trainer expects loss functions with signature `loss_fn(output, label)`
- GLOSS needs hyperparameters (sigma, gamma, class_weights)
- The wrapper "bakes in" these parameters, making GLOSS compatible with the trainer interface

**Usage in trainers/coop.py**:
```python
def forward_backward(self, batch):
    loss_fn = get_loss_function(self.cfg.TRAINER.LOSS_FUNCTION, self.cfg)
    # Now loss_fn is a callable: loss_fn(output, label)
    loss = loss_fn(output, label)
```

---

### 3. **gloss.py** - Core Label Propagation Algorithm
**Role**: Implements the graph-based label propagation loss computation

**Key Functions**:

#### `guassian(emb, sigma)`
Computes a Gaussian kernel similarity matrix between embeddings
- Input: embeddings (batch_size, embedding_dim)
- Computes pairwise distances and exponential similarity
- Returns: weight matrix (batch_size, batch_size)

#### `normalize_adj(adj)`
Normalizes an adjacency matrix using symmetric normalization
- Handles numerical issues (NaN, Inf)
- Returns: normalized adjacency matrix

#### `modified_lpa(train_emb, test_emb, Ytrain, sigma, num_labels, device, labels_orig)`
**The core label propagation algorithm**

**What it does**:
1. Concatenates labeled and unlabeled embeddings
2. Builds a Gaussian kernel similarity matrix based on embeddings
3. Creates transition matrix from normalized adjacency
4. Solves a linear system to propagate labels from labeled to unlabeled samples
5. Returns predicted label distributions (probabilities) for unlabeled samples

**Key steps**:
```
1. Embedding construction: cat([train_emb, test_emb]) → embeddings of all samples
2. Create one-hot label matrix Y: maps sample index to one-hot class vector
3. Build Gaussian similarity matrix: W[i,j] = exp(-||emb[i]-emb[j]||²/(2σ²))
4. Normalize adjacency: symmetric graph laplacian normalization
5. Build transition matrix T: each row sums to 1 (row-normalized)
6. Split transition matrix: T_uu (unlabeled-to-unlabeled), T_ul (unlabeled-to-labeled)
7. Solve linear system: F_UU = (I - T_uu)⁻¹ * T_ul * Y_labeled
   - This gives predicted label distributions for unlabeled samples
```

**Why torch.linalg.solve needs special handling**:
- The function requires float32 or float64, but with mixed-precision training (AMP), tensors are float16
- Solution: temporarily convert to float32, solve, then convert back

#### `gloss(X, labels, sigma, num_labels, gamma, device, class_weights)`
**The main loss function called during training**

**What it does**:
1. Splits batch into labeled (gamma fraction) and unlabeled (1-gamma fraction) samples
2. Calls `modified_lpa()` to get predicted labels for unlabeled samples
3. Computes cross-entropy loss between predicted and actual labels
4. Returns scalar loss that backpropagates to model parameters

**Key insight**:
- The model outputs (logits) serve as embeddings
- The first `gamma * batch_size` samples are treated as having "ground truth" labels
- The remaining samples are treated as unlabeled
- Label propagation predicts labels for "unlabeled" samples
- Loss is computed only on the predicted vs actual labels for the unlabeled set

---

## Data Flow During Training

```
Model Input (images)
    ↓
[Model forward pass - in CoOp: PromptLearner + CLIP]
    ↓
Model Output: Logits (batch_size, num_classes)  ← Has requires_grad=True
    ↓
loss_fn = get_loss_function("gloss", cfg)  ← Factory creates wrapper
    ↓
loss_fn(logits, labels)  ← Wrapper calls gloss()
    ↓
gloss(logits, labels, sigma=0.1, gamma=0.5, ...):
  - Split into labeled (first 50% of batch) and unlabeled (rest)
  - Call modified_lpa(labeled_emb, unlabeled_emb, labeled_labels, ...)
    ├─ Build Gaussian similarity matrix from embeddings
    ├─ Normalize adjacency matrix
    ├─ Create transition matrix
    ├─ Solve: F_UU = (I - T_uu)⁻¹ * T_ul * Y
    └─ Return predicted label probabilities for unlabeled set
  - Compute cross_entropy(predicted_labels, actual_labels)
    ↓
loss scalar (differentiable through all operations)
    ↓
loss.backward()
    ↓
Model parameters updated (gradient descent)
```

---

## Configuration Flow

```
Command line / shell script
    ↓
--loss-function gloss
--output-dir output/.../loss_gloss/...
TRAINER.GLOSS.SIGMA 0.1
TRAINER.GLOSS.GAMMA 0.5
    ↓
train.py parse_args() & setup_cfg()
    ↓
cfg.TRAINER.LOSS_FUNCTION = "gloss"
cfg.TRAINER.GLOSS.SIGMA = 0.1
cfg.TRAINER.GLOSS.GAMMA = 0.5
    ↓
trainer = build_trainer(cfg)
    ↓
trainer.forward_backward():
  loss_fn = get_loss_function(cfg.TRAINER.LOSS_FUNCTION, cfg)
    ├─ Reads cfg.TRAINER.GLOSS.SIGMA, GAMMA, CLASS_WEIGHTS
    ├─ Creates gloss_wrapper() with these values captured
    └─ Returns gloss_wrapper
  
  loss = loss_fn(output, label)
    ├─ Calls gloss_wrapper(output, label)
    ├─ Which calls gloss(output, label, sigma=0.1, gamma=0.5, ...)
    └─ Returns loss
```

---

## Why Loss Wasn't Changing (Issue #3)

**Root Causes**:

1. **Random Masking Issue** (Original Code):
   - Used `torch.rand()` to create different labeled/unlabeled splits each batch
   - This randomness broke consistency and could cause gradient issues
   - **Fix**: Use deterministic masking (first `gamma*batch_size` samples as labeled)

2. **Small Batch Size Issue**:
   - If batch size is small and gamma splits the batch, unlabeled set becomes too small
   - Label propagation might fail or provide weak gradients
   - **Fix**: Added fallback to standard cross-entropy for very small batches

3. **Missing Gradient Check**:
   - If model outputs don't require gradients, loss won't propagate back
   - **Fix**: Added warning to detect this condition

4. **Numerical Instability**:
   - `torch.linalg.solve()` is numerically sensitive
   - Mixed precision (float16) can exacerbate this
   - **Fix**: Convert to float32 for solving, then back to original dtype

---

## Practical Tips

### Tuning Hyperparameters

- **SIGMA** (default 0.1): Gaussian kernel bandwidth
  - Lower values: sharper similarity matrix, more local structure
  - Higher values: smoother similarity matrix, more global structure
  - Start with 0.1-1.0

- **GAMMA** (default 0.5): Fraction of batch treated as labeled
  - Higher values: more labeled data, less label propagation
  - Lower values: more unlabeled data, more aggressive label propagation
  - Try 0.3-0.7

- **CLASS_WEIGHTS**: Optional weighting for class imbalance
  - Set to None for uniform weights
  - Set to list of floats for weighted classes

### Debugging

Check your output directory structure:
```bash
# Before fix:
output/oxford_pets/CoOp/end_16shots/.../seed1/
output/oxford_pets/CoOp/end_16shots/.../seed2/

# After fix (with loss function in path):
output/oxford_pets/CoOp/end_16shots/loss_gloss/seed1/
output/oxford_pets/CoOp/end_16shots/loss_cross_entropy/seed2/
```

Monitor loss curves:
- Loss should decrease over epochs when properly trained
- If loss is constant, check for gradient issues or numerical problems
- If loss diverges, try lowering SIGMA or increasing GAMMA
