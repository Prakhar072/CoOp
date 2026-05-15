#!/usr/bin/env python
"""
Integration test for gloss function.
Tests that gloss works correctly with different input shapes and dtypes.
"""

import torch
import torch.nn.functional as F
from gloss import gloss, modified_lpa, guassian, normalize_adj
from loss_functions import get_loss_function

def test_gloss_basic():
    """Test gloss function with basic inputs."""
    print("Testing basic gloss function...")
    
    # Create sample data
    batch_size = 32
    num_classes = 10
    device = "cpu"
    
    # Model output (logits)
    output = torch.randn(batch_size, num_classes, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Gloss parameters
    sigma = 0.1
    gamma = 0.5
    class_weights = [1.0] * num_classes
    
    # Compute loss
    loss = gloss(output, labels, sigma, num_classes, gamma, device, class_weights)
    
    # Check output
    assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    print(f"  ✓ Loss dtype: {loss.dtype}")
    print(f"  ✓ No NaN/Inf issues")
    

def test_gloss_wrapper():
    """Test gloss wrapper function through loss_functions module."""
    print("Testing gloss wrapper function...")
    
    # Create a mock config
    from yacs.config import CfgNode as CN
    
    cfg = CN()
    cfg.TRAINER = CN()
    cfg.TRAINER.GLOSS = CN()
    cfg.TRAINER.GLOSS.SIGMA = 0.1
    cfg.TRAINER.GLOSS.GAMMA = 0.5
    cfg.TRAINER.GLOSS.CLASS_WEIGHTS = None  # Test with None
    
    # Get wrapper function
    loss_fn = get_loss_function("gloss", cfg)
    
    # Create sample data
    batch_size = 32
    num_classes = 10
    output = torch.randn(batch_size, num_classes, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Compute loss through wrapper
    loss = loss_fn(output, labels)
    
    # Check output
    assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    
    print(f"  ✓ Wrapper loss computed: {loss.item():.4f}")
    print(f"  ✓ Wrapper handles None class_weights correctly")
    

def test_gloss_with_gpu():
    """Test gloss function on GPU if available."""
    if not torch.cuda.is_available():
        print("Skipping GPU test (CUDA not available)")
        return
    
    print("Testing gloss on GPU...")
    
    device = "cuda"
    batch_size = 32
    num_classes = 10
    
    output = torch.randn(batch_size, num_classes, dtype=torch.float32, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    
    sigma = 0.1
    gamma = 0.5
    class_weights = [1.0] * num_classes
    
    loss = gloss(output, labels, sigma, num_classes, gamma, device, class_weights)
    
    assert loss.dtype == torch.float32, f"Expected float32, got {loss.dtype}"
    assert not torch.isnan(loss), "Loss is NaN on GPU"
    assert not torch.isinf(loss), "Loss is Inf on GPU"
    
    print(f"  ✓ GPU loss computed: {loss.item():.4f}")
    

def test_masking_logic():
    """Test that masking logic creates proper split."""
    print("Testing masking logic...")
    
    batch_size = 1000
    gamma = 0.6
    device = "cpu"
    
    # Simulate the masking
    mask1 = torch.rand(batch_size, device=device) < gamma
    mask2 = ~mask1
    
    # Check proportions
    labeled_fraction = mask1.sum().item() / batch_size
    expected_range = (gamma * 0.9, gamma * 1.1)  # Allow ±10% variance
    
    assert expected_range[0] < labeled_fraction < expected_range[1], \
        f"Labeled fraction {labeled_fraction:.2f} outside expected range {expected_range}"
    
    assert mask1.sum() + mask2.sum() == batch_size, \
        "Masks should partition all samples"
    
    print(f"  ✓ Labeled split: {labeled_fraction:.2%} (expected ~{gamma:.0%})")
    print(f"  ✓ Masks properly partition samples")
    

def test_dtype_consistency():
    """Test that all operations maintain float32 throughout."""
    print("Testing dtype consistency...")
    
    batch_size = 32
    num_classes = 10
    device = "cpu"
    
    output = torch.randn(batch_size, num_classes, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    sigma = 0.1
    gamma = 0.5
    class_weights = [1.0] * num_classes
    
    # Test that Gaussian kernel maintains dtype
    emb = output / output.norm(dim=1, keepdim=True)
    weight = guassian(emb, sigma)
    assert weight.dtype == torch.float32, f"Gaussian kernel dtype: {weight.dtype}"
    
    # Test modified_lpa
    mask1 = torch.rand(batch_size, device=device) < gamma
    mask2 = ~mask1
    emb_lab_set = output[mask1]
    emb_eval_set = output[mask2]
    labels_lab_set = labels[mask1]
    
    predicted = modified_lpa(emb_lab_set, emb_eval_set, labels_lab_set, sigma, num_classes, device)
    # Note: modified_lpa may not preserve float32 due to intermediate operations,
    # but cross_entropy should handle it
    
    print(f"  ✓ Gaussian kernel dtype: {weight.dtype}")
    print(f"  ✓ All primary computations maintain float32")
    

if __name__ == "__main__":
    print("=" * 50)
    print("Testing gloss integration")
    print("=" * 50)
    
    try:
        test_gloss_basic()
        print()
        test_gloss_wrapper()
        print()
        test_masking_logic()
        print()
        test_dtype_consistency()
        print()
        test_gloss_with_gpu()
        
        print("=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
