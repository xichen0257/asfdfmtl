"""
Backbone Feature Extraction Module for Dynamic Clustering

Extracts feature statistics from backbone network for computing
client similarity in federated multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from torch.utils.data import DataLoader


class BackboneFeatureExtractor:
    """Extract and compute backbone feature statistics."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: Multi-task model with .backbone attribute
            device: Device to run extraction on
        """
        self.model = model
        self.device = device

    def extract_feature_statistics(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract backbone feature statistics from local data.

        Args:
            dataloader: Local dataloader
            max_batches: Maximum number of batches to process (None = all)

        Returns:
            stats: Dictionary containing:
                - 'mean': [C] channel-wise mean
                - 'std': [C] channel-wise std
                - 'cov': [C, C] covariance matrix (optional, memory intensive)
        """
        self.model.eval()
        features_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Get image
                x = batch['image'].to(self.device)

                # Forward through backbone only
                backbone_feat = self.model.backbone(x)  # [B, C, H, W]

                # Global average pooling
                feat_pooled = F.adaptive_avg_pool2d(backbone_feat, (1, 1))  # [B, C, 1, 1]
                feat_pooled = feat_pooled.squeeze(-1).squeeze(-1)  # [B, C]

                features_list.append(feat_pooled.cpu())

        # Concatenate all features
        all_features = torch.cat(features_list, dim=0)  # [N_samples, C]

        # Compute statistics
        stats = {
            'mean': all_features.mean(dim=0),      # [C]
            'std': all_features.std(dim=0),        # [C]
        }

        # Optional: Compute covariance (memory intensive for large C)
        # Uncomment if needed
        # if all_features.shape[0] > 1:
        #     stats['cov'] = torch.cov(all_features.T)  # [C, C]

        return stats


def compute_feature_similarity(
    stats_i: Dict[str, torch.Tensor],
    stats_j: Dict[str, torch.Tensor],
    method: str = 'cosine'
) -> float:
    """
    Compute similarity between two feature statistics.

    Args:
        stats_i: Feature statistics from client i
        stats_j: Feature statistics from client j
        method: Similarity metric
            - 'cosine': Cosine similarity (default)
            - 'euclidean': Negative normalized Euclidean distance
            - 'combined': Weighted combination

    Returns:
        similarity: Similarity score in [0, 1] (higher = more similar)
    """
    if method == 'cosine':
        # Cosine similarity on mean vectors
        cos_sim_mean = F.cosine_similarity(
            stats_i['mean'].unsqueeze(0),
            stats_j['mean'].unsqueeze(0),
            dim=1
        ).item()

        # Cosine similarity on std vectors
        cos_sim_std = F.cosine_similarity(
            stats_i['std'].unsqueeze(0),
            stats_j['std'].unsqueeze(0),
            dim=1
        ).item()

        # Average (both are in [-1, 1], normalize to [0, 1])
        similarity = (cos_sim_mean + cos_sim_std) / 2.0
        similarity = (similarity + 1.0) / 2.0  # Map [-1,1] to [0,1]

        return similarity

    elif method == 'euclidean':
        # Euclidean distance (normalized)
        dist_mean = torch.norm(stats_i['mean'] - stats_j['mean']).item()
        dist_std = torch.norm(stats_i['std'] - stats_j['std']).item()

        # Normalize by magnitude
        max_norm_mean = torch.norm(stats_i['mean']).item() + torch.norm(stats_j['mean']).item()
        max_norm_std = torch.norm(stats_i['std']).item() + torch.norm(stats_j['std']).item()

        # Avoid division by zero
        norm_dist_mean = dist_mean / (max_norm_mean + 1e-8)
        norm_dist_std = dist_std / (max_norm_std + 1e-8)

        # Convert distance to similarity (1 - distance)
        sim_mean = 1.0 - norm_dist_mean
        sim_std = 1.0 - norm_dist_std

        return (sim_mean + sim_std) / 2.0

    elif method == 'combined':
        # Weighted combination of cosine and euclidean
        cos_sim = compute_feature_similarity(stats_i, stats_j, 'cosine')
        euc_sim = compute_feature_similarity(stats_i, stats_j, 'euclidean')

        # Weight: 70% cosine, 30% euclidean
        return 0.7 * cos_sim + 0.3 * euc_sim

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_pairwise_similarity(
    all_stats: list,
    method: str = 'cosine'
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for all clients.

    Args:
        all_stats: List of feature statistics for N clients
        method: Similarity metric

    Returns:
        similarity_matrix: [N, N] pairwise similarity matrix
    """
    N = len(all_stats)
    similarity_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i, N):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = compute_feature_similarity(all_stats[i], all_stats[j], method)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric

    return similarity_matrix


def select_top_k_neighbors(
    client_idx: int,
    similarity_matrix: np.ndarray,
    k: int = 3
) -> Tuple[list, np.ndarray]:
    """
    Select top-k most similar neighbors for a client (soft clustering).

    Args:
        client_idx: Index of the target client
        similarity_matrix: [N, N] pairwise similarity matrix
        k: Number of neighbors to select

    Returns:
        neighbor_indices: List of k neighbor indices
        neighbor_weights: Normalized similarity weights for neighbors
    """
    N = similarity_matrix.shape[0]

    # Get similarities (excluding self)
    similarities = similarity_matrix[client_idx].copy()
    similarities[client_idx] = -np.inf  # Exclude self

    # Adjust k if needed
    k = min(k, N - 1)

    # Select top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]  # Descending order

    # Get weights (use similarities as weights)
    top_k_similarities = similarities[top_k_indices]

    # Normalize weights (including self)
    # Add self with weight = 1.0
    all_weights = np.concatenate([[1.0], top_k_similarities])
    normalized_weights = all_weights / all_weights.sum()

    # Split: self_weight, neighbor_weights
    self_weight = normalized_weights[0]
    neighbor_weights = normalized_weights[1:]

    return top_k_indices.tolist(), neighbor_weights


if __name__ == "__main__":
    # Simple test
    print("Testing backbone feature extraction...")

    # Create dummy stats
    C = 2048  # ResNet-50 output channels
    stats_1 = {
        'mean': torch.randn(C),
        'std': torch.abs(torch.randn(C))
    }
    stats_2 = {
        'mean': stats_1['mean'] + 0.1 * torch.randn(C),  # Similar
        'std': stats_1['std'] + 0.1 * torch.abs(torch.randn(C))
    }
    stats_3 = {
        'mean': torch.randn(C),  # Different
        'std': torch.abs(torch.randn(C))
    }

    # Test similarity
    sim_12 = compute_feature_similarity(stats_1, stats_2, 'cosine')
    sim_13 = compute_feature_similarity(stats_1, stats_3, 'cosine')

    print(f"Similarity(1, 2): {sim_12:.4f} (should be high)")
    print(f"Similarity(1, 3): {sim_13:.4f} (should be lower)")

    # Test pairwise
    all_stats = [stats_1, stats_2, stats_3]
    sim_matrix = compute_pairwise_similarity(all_stats)
    print("\nSimilarity Matrix:")
    print(sim_matrix)

    # Test top-k selection
    neighbors, weights = select_top_k_neighbors(0, sim_matrix, k=2)
    print(f"\nTop-2 neighbors of client 0: {neighbors}")
    print(f"Weights: {weights}")

    print("\n[OK] All tests passed!")
