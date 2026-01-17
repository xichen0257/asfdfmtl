"""
Quick test to verify all imports and basic functionality work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

# Test task similarity imports
try:
    from decentralized.task_similarity import (
        compute_gradient_similarity,
        compute_pairwise_similarity,
        select_top_k_neighbors,
        compute_model_gradient
    )
    print("[OK] task_similarity module imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import task_similarity: {e}")
    sys.exit(1)

# Test decentralized client import
try:
    from decentralized.decentralized_client import DecentralizedClient
    print("[OK] DecentralizedClient imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import DecentralizedClient: {e}")
    sys.exit(1)

# Test other dependencies
try:
    import torch
    import numpy as np
    import yaml
    print("[OK] Core dependencies (torch, numpy, yaml) available")
except Exception as e:
    print(f"[FAIL] Failed to import core dependencies: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting basic functionality...")

# Test gradient similarity computation
try:
    grad_a = torch.randn(100)
    grad_b = torch.randn(100)
    similarity = compute_gradient_similarity(grad_a, grad_b)
    assert -1.0 <= similarity <= 1.0, "Similarity should be in [-1, 1]"
    print(f"[OK] Gradient similarity computation works (similarity={similarity:.4f})")
except Exception as e:
    print(f"[FAIL] Gradient similarity test failed: {e}")
    sys.exit(1)

# Test pairwise similarity matrix
try:
    gradients = [torch.randn(50) for _ in range(5)]
    sim_matrix = compute_pairwise_similarity(gradients)
    assert sim_matrix.shape == (5, 5), "Similarity matrix should be 5x5"
    assert np.allclose(sim_matrix, sim_matrix.T), "Similarity matrix should be symmetric"
    print(f"[OK] Pairwise similarity matrix computation works (shape={sim_matrix.shape})")
except Exception as e:
    print(f"[FAIL] Pairwise similarity test failed: {e}")
    sys.exit(1)

# Test neighbor selection
try:
    sim_matrix = np.random.rand(6, 6)
    np.fill_diagonal(sim_matrix, 1.0)
    neighbors = select_top_k_neighbors(client_idx=0, similarity_matrix=sim_matrix, k=3)
    assert len(neighbors) == 3, "Should select 3 neighbors"
    assert 0 not in neighbors, "Should not include self"
    print(f"[OK] Neighbor selection works (selected: {neighbors})")
except Exception as e:
    print(f"[FAIL] Neighbor selection test failed: {e}")
    sys.exit(1)

# Test config loading
try:
    config_path = 'configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Config file loaded successfully (title: {config['general']['title']})")
    else:
        print(f"[WARN] Config file not found: {config_path}")
except Exception as e:
    print(f"[FAIL] Config loading test failed: {e}")

print("\n" + "="*60)
print("All basic tests passed!")
print("="*60)
print("\nYou can now run the full experiment with:")
print("  python src/decentralized/run_decentralized.py \\")
print("    --config configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml")
