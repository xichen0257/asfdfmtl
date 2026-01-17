"""
Decentralized Federated Multi-Task Learning with Dynamic Soft Clustering

This package implements a fully decentralized FMTL system with:
- Peer-to-peer communication (simulated)
- Dynamic soft clustering based on task similarity
- Gradient-based task similarity computation
- Local HCA aggregation

Key modules:
- task_similarity: Compute task similarity and select neighbors
- decentralized_client: Decentralized client with local aggregation
- run_decentralized: Main training script
"""

from .task_similarity import (
    compute_gradient_similarity,
    compute_pairwise_similarity,
    select_top_k_neighbors,
    compute_model_gradient
)

from .decentralized_client import DecentralizedClient

__all__ = [
    'compute_gradient_similarity',
    'compute_pairwise_similarity',
    'select_top_k_neighbors',
    'compute_model_gradient',
    'DecentralizedClient'
]
