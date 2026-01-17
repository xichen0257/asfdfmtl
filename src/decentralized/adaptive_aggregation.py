"""
Adaptive Aggregation Methods for Decentralized FMTL

This module implements various adaptive aggregation strategies:
1. Weighted Aggregation: Similarity-based soft weights
2. Adaptive Filter: Dynamic neighbor filtering based on threshold
3. Progressive Neighbors: Gradually increasing neighbor count
4. Hybrid Adaptive: Combination of multiple strategies
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_softmax_weights(similarities: List[float], temperature: float = 1.0) -> List[float]:
    """
    Compute softmax weights from similarity scores.

    Args:
        similarities: List of similarity scores
        temperature: Temperature parameter (higher = more uniform, lower = more peaked)

    Returns:
        List of softmax weights (sum to 1.0)
    """
    if len(similarities) == 0:
        return []

    # Apply temperature scaling
    scaled_sims = [s / temperature for s in similarities]

    # Compute softmax
    exp_sims = [np.exp(s) for s in scaled_sims]
    sum_exp = sum(exp_sims)

    weights = [e / sum_exp for e in exp_sims]

    return weights


def get_self_weight(round_num: int, max_rounds: int,
                    initial_weight: float = 0.7,
                    final_weight: float = 0.3,
                    decay_type: str = 'linear') -> float:
    """
    Compute self-preservation weight for current round.

    Early rounds: Higher self weight (preserve personalization)
    Late rounds: Lower self weight (leverage collaboration)

    Args:
        round_num: Current round (1-indexed)
        max_rounds: Total number of rounds
        initial_weight: Self weight at round 1
        final_weight: Self weight at final round
        decay_type: 'linear', 'exponential', or 'step'

    Returns:
        Self weight in [final_weight, initial_weight]
    """
    progress = (round_num - 1) / max(1, max_rounds - 1)  # 0.0 to 1.0

    if decay_type == 'linear':
        weight = initial_weight - progress * (initial_weight - final_weight)
    elif decay_type == 'exponential':
        weight = initial_weight * ((final_weight / initial_weight) ** progress)
    elif decay_type == 'step':
        # Step decay: high for first half, low for second half
        weight = initial_weight if progress < 0.5 else final_weight
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")

    return weight


def filter_neighbors_by_similarity(
    neighbor_indices: List[int],
    similarities: List[float],
    threshold: float = 0.3,
    min_neighbors: int = 1,
    max_neighbors: int = 3
) -> Tuple[List[int], List[float]]:
    """
    Filter neighbors based on similarity threshold.

    Args:
        neighbor_indices: List of neighbor indices
        similarities: Corresponding similarity scores
        threshold: Minimum similarity threshold
        min_neighbors: Keep at least this many neighbors (best ones)
        max_neighbors: Keep at most this many neighbors

    Returns:
        Tuple of (filtered_indices, filtered_similarities)
    """
    if len(neighbor_indices) == 0:
        return [], []

    # Combine indices and similarities
    pairs = list(zip(neighbor_indices, similarities))

    # Sort by similarity (descending)
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold
    filtered_pairs = [(idx, sim) for idx, sim in pairs if sim >= threshold]

    # Ensure min_neighbors
    if len(filtered_pairs) < min_neighbors:
        # Keep top min_neighbors regardless of threshold
        filtered_pairs = pairs[:min_neighbors]

    # Ensure max_neighbors
    filtered_pairs = filtered_pairs[:max_neighbors]

    filtered_indices = [idx for idx, _ in filtered_pairs]
    filtered_similarities = [sim for _, sim in filtered_pairs]

    return filtered_indices, filtered_similarities


def get_progressive_neighbor_count(round_num: int, schedule: List[Dict]) -> int:
    """
    Get neighbor count based on progressive schedule.

    Args:
        round_num: Current round (1-indexed)
        schedule: List of dicts with 'rounds' and 'n_neighbors'
                  e.g., [{'rounds': [1,2,3], 'n_neighbors': 1}, ...]

    Returns:
        Number of neighbors for this round
    """
    for entry in schedule:
        if round_num in entry['rounds']:
            return entry['n_neighbors']

    # Default: use last entry's neighbor count
    return schedule[-1]['n_neighbors']


def adaptive_weighted_aggregate(
    backbones: List[Dict[str, torch.Tensor]],
    similarities: List[float],
    round_num: int,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted aggregation with similarity-based weights and self-preservation.

    Aggregation formula:
        aggregated = self_weight * self_model + (1 - self_weight) * weighted_neighbor_avg

    where weighted_neighbor_avg = sum(w_i * neighbor_i) for softmax weights w_i

    Args:
        backbones: [self_backbone, neighbor1_backbone, neighbor2_backbone, ...]
        similarities: [sim_with_neighbor1, sim_with_neighbor2, ...] (no self similarity)
        round_num: Current round
        config: Configuration dict with:
            - similarity_threshold: Filter neighbors below this
            - self_weight_initial: Initial self weight
            - self_weight_final: Final self weight
            - self_weight_decay: Decay type
            - use_softmax_weights: Whether to use softmax
            - temperature: Softmax temperature

    Returns:
        Aggregated backbone
    """
    # Extract config
    threshold = config.get('similarity_threshold', 0.0)
    self_weight_initial = config.get('self_weight_initial', 0.7)
    self_weight_final = config.get('self_weight_final', 0.3)
    decay_type = config.get('self_weight_decay', 'linear')
    use_softmax = config.get('use_softmax_weights', True)
    temperature = config.get('temperature', 1.0)
    max_rounds = config.get('max_rounds', 10)

    # Self backbone
    self_backbone = backbones[0]
    neighbor_backbones = backbones[1:]

    # Filter neighbors by similarity
    valid_neighbors = []
    valid_similarities = []
    for i, (backbone, sim) in enumerate(zip(neighbor_backbones, similarities)):
        if sim >= threshold:
            valid_neighbors.append(backbone)
            valid_similarities.append(sim)

    # If no valid neighbors, return self
    if len(valid_neighbors) == 0:
        print(f"  [WARNING] No neighbors above threshold {threshold:.3f}, returning self model")
        return self_backbone

    # Compute neighbor weights
    if use_softmax:
        neighbor_weights = compute_softmax_weights(valid_similarities, temperature)
    else:
        # Equal weights
        neighbor_weights = [1.0 / len(valid_neighbors)] * len(valid_neighbors)

    # Compute self weight
    self_weight = get_self_weight(
        round_num, max_rounds,
        self_weight_initial, self_weight_final,
        decay_type
    )

    print(f"  Adaptive Weighted: {len(valid_neighbors)} neighbors (filtered from {len(neighbor_backbones)}), "
          f"self_weight={self_weight:.3f}")

    # Aggregate neighbors
    neighbor_aggregate = {}
    for key in self_backbone.keys():
        original_dtype = self_backbone[key].dtype
        neighbor_aggregate[key] = torch.zeros_like(self_backbone[key], dtype=torch.float32)

        for backbone, weight in zip(valid_neighbors, neighbor_weights):
            neighbor_aggregate[key] += weight * backbone[key].float()

        neighbor_aggregate[key] = neighbor_aggregate[key].to(original_dtype)

    # Final aggregation: self + neighbors
    aggregated = {}
    for key in self_backbone.keys():
        original_dtype = self_backbone[key].dtype
        aggregated[key] = (
            self_weight * self_backbone[key].float() +
            (1 - self_weight) * neighbor_aggregate[key].float()
        ).to(original_dtype)

    return aggregated


def adaptive_filter_aggregate(
    backbones: List[Dict[str, torch.Tensor]],
    similarities: List[float],
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Perform aggregation with adaptive neighbor filtering.

    Only aggregates with neighbors above similarity threshold.
    Uses equal weights for filtered neighbors.

    Args:
        backbones: [self_backbone, neighbor1_backbone, ...]
        similarities: [sim_with_neighbor1, ...]
        config: Configuration dict

    Returns:
        Aggregated backbone
    """
    threshold = config.get('similarity_threshold', 0.35)
    min_neighbors = config.get('min_neighbors', 1)
    max_neighbors = config.get('max_neighbors', 3)
    self_weight = config.get('include_self_weight', 0.5)

    self_backbone = backbones[0]
    neighbor_backbones = backbones[1:]

    # Filter neighbors
    valid_neighbors = []
    valid_sims = []
    for backbone, sim in zip(neighbor_backbones, similarities):
        if sim >= threshold:
            valid_neighbors.append(backbone)
            valid_sims.append(sim)

    # Ensure min/max neighbors
    if len(valid_neighbors) < min_neighbors:
        # Keep top min_neighbors
        pairs = list(zip(neighbor_backbones, similarities))
        pairs.sort(key=lambda x: x[1], reverse=True)
        valid_neighbors = [b for b, _ in pairs[:min_neighbors]]

    valid_neighbors = valid_neighbors[:max_neighbors]

    if len(valid_neighbors) == 0:
        print(f"  [WARNING] No valid neighbors, returning self model")
        return self_backbone

    print(f"  Adaptive Filter: {len(valid_neighbors)} neighbors (threshold={threshold:.3f})")

    # Equal weights for neighbors
    neighbor_weight_total = 1.0 - self_weight
    neighbor_weight_each = neighbor_weight_total / len(valid_neighbors)

    # Aggregate
    aggregated = {}
    for key in self_backbone.keys():
        original_dtype = self_backbone[key].dtype
        aggregated[key] = self_weight * self_backbone[key].float()

        for backbone in valid_neighbors:
            aggregated[key] += neighbor_weight_each * backbone[key].float()

        aggregated[key] = aggregated[key].to(original_dtype)

    return aggregated


def progressive_neighbor_aggregate(
    backbones: List[Dict[str, torch.Tensor]],
    round_num: int,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Perform aggregation with progressive neighbor count.

    Early rounds: Few neighbors (e.g., N=1)
    Late rounds: More neighbors (e.g., N=3)

    Args:
        backbones: [self_backbone, neighbor1_backbone, ...] (pre-sorted by similarity)
        round_num: Current round
        config: Configuration dict with neighbor_schedule

    Returns:
        Aggregated backbone
    """
    schedule = config.get('neighbor_schedule', [])
    n_neighbors = get_progressive_neighbor_count(round_num, schedule)

    # Only use top N neighbors
    effective_backbones = backbones[:n_neighbors+1]  # +1 for self

    print(f"  Progressive: Using {n_neighbors} neighbors (round {round_num})")

    # Equal weights
    weights = [1.0 / len(effective_backbones)] * len(effective_backbones)

    aggregated = {}
    for key in backbones[0].keys():
        original_dtype = backbones[0][key].dtype
        aggregated[key] = torch.zeros_like(backbones[0][key], dtype=torch.float32)

        for backbone, weight in zip(effective_backbones, weights):
            aggregated[key] += weight * backbone[key].float()

        aggregated[key] = aggregated[key].to(original_dtype)

    return aggregated


def hybrid_adaptive_aggregate(
    backbones: List[Dict[str, torch.Tensor]],
    similarities: List[float],
    round_num: int,
    config: Dict
) -> Dict[str, torch.Tensor]:
    """
    Hybrid: Combine progressive neighbors + weighted aggregation.

    Args:
        backbones: [self_backbone, neighbor1_backbone, ...]
        similarities: [sim_with_neighbor1, ...]
        round_num: Current round
        config: Combined configuration

    Returns:
        Aggregated backbone
    """
    # First, determine number of neighbors (progressive)
    schedule = config.get('neighbor_schedule', [])
    n_neighbors = get_progressive_neighbor_count(round_num, schedule)

    # Filter to top N neighbors
    if len(backbones) > n_neighbors + 1:
        # Sort neighbors by similarity
        pairs = list(zip(backbones[1:], similarities))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top_neighbors = [b for b, _ in pairs[:n_neighbors]]
        top_similarities = [s for _, s in pairs[:n_neighbors]]

        filtered_backbones = [backbones[0]] + top_neighbors
        filtered_similarities = top_similarities
    else:
        filtered_backbones = backbones
        filtered_similarities = similarities

    print(f"  Hybrid Adaptive: {n_neighbors} neighbors (progressive), using weighted aggregation")

    # Then, use weighted aggregation
    return adaptive_weighted_aggregate(
        filtered_backbones,
        filtered_similarities,
        round_num,
        config
    )
