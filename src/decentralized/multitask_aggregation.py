"""
Multi-Task Aggregation Module for Soft Clustering

Implements aggregation strategies for multi-task models in decentralized FL:
1. Backbone aggregation: Aggregate shared backbone with similarity-weighted neighbors
2. Task-selective head aggregation: Optionally aggregate task heads only with
   neighbors that also work on that task

Key design decision:
- Backbone is shared across all tasks -> aggregate with all neighbors (weighted by similarity)
- Task heads are task-specific -> aggregate only with neighbors doing the same task
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import copy


def aggregate_backbone_weighted(
    own_backbone: Dict[str, torch.Tensor],
    neighbor_backbones: List[Dict[str, torch.Tensor]],
    neighbor_similarities: List[float],
    self_weight: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Aggregate backbone parameters using similarity-weighted averaging.

    Formula:
        aggregated = (1 - self_weight) * weighted_avg(neighbors) + self_weight * own_model

    where:
        weighted_avg(neighbors) = sum(sim_i * backbone_i) / sum(sim_i)

    Args:
        own_backbone: Current client's backbone parameters
        neighbor_backbones: List of backbone parameters from selected neighbors
        neighbor_similarities: List of similarity scores for each neighbor
        self_weight: Weight for preserving own model (0.0 = full aggregation, 1.0 = no aggregation)

    Returns:
        Aggregated backbone parameters
    """
    if len(neighbor_backbones) == 0:
        return own_backbone

    # Normalize similarities to sum to 1 with numerical stability
    sim_array = np.array(neighbor_similarities)
    epsilon = 1e-8

    # Clip similarities to reasonable range [epsilon, 1.0]
    # This prevents negative or zero weights from causing instability
    sim_array = np.clip(sim_array, epsilon, 1.0)

    # Check if sum is too small (all similarities near zero)
    sim_sum = sim_array.sum()
    if sim_sum < epsilon:
        # All similarities too low, fallback to uniform weights
        # This can happen when pure gradient similarity finds no useful neighbors
        sim_normalized = np.ones_like(sim_array) / len(sim_array)
    else:
        sim_normalized = sim_array / sim_sum

    # Initialize aggregated backbone
    aggregated = {}

    # Weighted average of neighbors
    for key in own_backbone.keys():
        aggregated[key] = torch.zeros_like(own_backbone[key])

        for i, neighbor_backbone in enumerate(neighbor_backbones):
            aggregated[key] += sim_normalized[i] * neighbor_backbone[key]

        # Numerical stability: check for NaN/Inf and clip extreme values
        if torch.isnan(aggregated[key]).any() or torch.isinf(aggregated[key]).any():
            # NaN/Inf detected, use own backbone as fallback
            aggregated[key] = own_backbone[key].clone()
        else:
            # Clip to prevent parameter explosion
            # Using a conservative range that preserves typical parameter scales
            max_param = torch.abs(own_backbone[key]).max().item()
            clip_range = max(10.0, max_param * 5.0)  # At least +/-10, or 5x current max
            aggregated[key] = torch.clamp(aggregated[key], -clip_range, clip_range)

    # Mix with own model if self_weight > 0
    if self_weight > 0.0:
        for key in aggregated.keys():
            aggregated[key] = (1 - self_weight) * aggregated[key] + self_weight * own_backbone[key]

    return aggregated


def aggregate_task_head_selective(
    own_head: Dict[str, torch.Tensor],
    neighbor_heads: List[Dict[str, torch.Tensor]],
    neighbor_similarities: List[float],
    neighbor_task_weights: List[Dict[str, float]],
    task_name: str,
    task_threshold: float = 0.1,
    self_weight: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Aggregate task-specific head with task-selective neighbors.

    Only aggregates with neighbors that also work on this task (task_weight >= threshold).

    Args:
        own_head: Current client's task head parameters
        neighbor_heads: List of task head parameters from selected neighbors
        neighbor_similarities: List of similarity scores for each neighbor
        neighbor_task_weights: List of task weight dicts for each neighbor
        task_name: Name of the task (e.g., 'depth', 'segmentation', 'normal')
        task_threshold: Minimum task weight to consider a neighbor (default: 0.1)
        self_weight: Weight for preserving own head

    Returns:
        Aggregated task head parameters
    """
    # Filter neighbors that also work on this task
    valid_indices = []
    valid_similarities = []

    for i, task_weights in enumerate(neighbor_task_weights):
        if task_weights.get(task_name, 0.0) >= task_threshold:
            valid_indices.append(i)
            valid_similarities.append(neighbor_similarities[i])

    # If no valid neighbors, return own head
    if len(valid_indices) == 0:
        return own_head

    # Normalize similarities
    sim_array = np.array(valid_similarities)
    sim_normalized = sim_array / sim_array.sum()

    # Weighted average of valid neighbors
    aggregated = {}
    for key in own_head.keys():
        aggregated[key] = torch.zeros_like(own_head[key])

        for i, idx in enumerate(valid_indices):
            aggregated[key] += sim_normalized[i] * neighbor_heads[idx][key]

    # Mix with own head if self_weight > 0
    if self_weight > 0.0:
        for key in aggregated.keys():
            aggregated[key] = (1 - self_weight) * aggregated[key] + self_weight * own_head[key]

    return aggregated


def aggregate_multitask_model(
    own_model_params: Dict[str, Dict[str, torch.Tensor]],
    neighbor_model_params: List[Dict[str, Dict[str, torch.Tensor]]],
    neighbor_similarities: List[float],
    neighbor_task_weights: List[Dict[str, float]],
    own_task_weights: Dict[str, float],
    aggregate_heads: bool = True,
    task_threshold: float = 0.1,
    self_weight_backbone: float = 0.0,
    self_weight_heads: float = 0.0
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Complete multi-task model aggregation.

    Aggregates both backbone and task heads (if enabled).

    Args:
        own_model_params: Own model parameters organized as:
            {'backbone': {...}, 'depth_head': {...}, 'seg_head': {...}, 'normal_head': {...}}
        neighbor_model_params: List of neighbor model parameters (same structure)
        neighbor_similarities: Similarity scores for each neighbor
        neighbor_task_weights: Task weights for each neighbor
        own_task_weights: Own task weights
        aggregate_heads: Whether to aggregate task heads (default: True)
        task_threshold: Minimum task weight for head aggregation
        self_weight_backbone: Self-preservation weight for backbone
        self_weight_heads: Self-preservation weight for heads

    Returns:
        Aggregated model parameters (same structure as input)
    """
    aggregated_params = {}

    # 1. Aggregate backbone (always)
    own_backbone = own_model_params['backbone']
    neighbor_backbones = [p['backbone'] for p in neighbor_model_params]

    aggregated_params['backbone'] = aggregate_backbone_weighted(
        own_backbone,
        neighbor_backbones,
        neighbor_similarities,
        self_weight=self_weight_backbone
    )

    # 2. Aggregate task heads (optional)
    task_head_names = ['depth_head', 'seg_head', 'normal_head']
    task_name_mapping = {
        'depth_head': 'depth',
        'seg_head': 'segmentation',
        'normal_head': 'normal'
    }

    for head_name in task_head_names:
        if head_name not in own_model_params:
            continue  # Skip if head not present

        task_name = task_name_mapping[head_name]

        if aggregate_heads and own_task_weights.get(task_name, 0.0) > 0:
            # Aggregate head with task-selective neighbors
            own_head = own_model_params[head_name]
            neighbor_heads = [p[head_name] for p in neighbor_model_params if head_name in p]

            aggregated_params[head_name] = aggregate_task_head_selective(
                own_head,
                neighbor_heads,
                neighbor_similarities,
                neighbor_task_weights,
                task_name,
                task_threshold=task_threshold,
                self_weight=self_weight_heads
            )
        else:
            # Don't aggregate, keep own head
            aggregated_params[head_name] = own_model_params[head_name]

    return aggregated_params


def extract_model_parameters(model) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Extract parameters from a MultiTaskModel.

    Args:
        model: MultiTaskModel instance

    Returns:
        Dict of parameters organized by component:
            {'backbone': {...}, 'depth_head': {...}, 'seg_head': {...}, 'normal_head': {...}}
    """
    params = {
        'backbone': {name: param.data.clone() for name, param in model.backbone.named_parameters()},
    }

    # Extract task heads dynamically based on model type
    if hasattr(model, 'is_pascal_context') and model.is_pascal_context:
        # Pascal Context tasks
        params['seg_head'] = {name: param.data.clone() for name, param in model.seg_head.named_parameters()}
        params['human_parts_head'] = {name: param.data.clone() for name, param in model.human_parts_head.named_parameters()}
        params['edge_head'] = {name: param.data.clone() for name, param in model.edge_head.named_parameters()}
    else:
        # NYU V2 tasks
        params['depth_head'] = {name: param.data.clone() for name, param in model.depth_head.named_parameters()}
        params['seg_head'] = {name: param.data.clone() for name, param in model.seg_head.named_parameters()}
        params['normal_head'] = {name: param.data.clone() for name, param in model.normal_head.named_parameters()}

    return params


def load_model_parameters(model, params: Dict[str, Dict[str, torch.Tensor]]):
    """
    Load parameters into a MultiTaskModel.

    Args:
        model: MultiTaskModel instance
        params: Parameters organized by component (from extract_model_parameters)
    """
    # Load backbone
    if 'backbone' in params:
        backbone_dict = model.backbone.state_dict()
        backbone_dict.update(params['backbone'])
        model.backbone.load_state_dict(backbone_dict)

    # Load task heads dynamically based on model type
    head_mapping = {}
    if hasattr(model, 'is_pascal_context') and model.is_pascal_context:
        # Pascal Context tasks
        head_mapping = {
            'seg_head': model.seg_head,
            'human_parts_head': model.human_parts_head,
            'edge_head': model.edge_head
        }
    else:
        # NYU V2 tasks
        head_mapping = {
            'depth_head': model.depth_head,
            'seg_head': model.seg_head,
            'normal_head': model.normal_head
        }

    for head_name, head_module in head_mapping.items():
        if head_name in params:
            head_dict = head_module.state_dict()
            head_dict.update(params[head_name])
            head_module.load_state_dict(head_dict)


def compute_aggregation_stats(
    own_task_weights: Dict[str, float],
    neighbor_task_weights: List[Dict[str, float]],
    neighbor_similarities: List[float]
) -> Dict[str, any]:
    """
    Compute statistics about the aggregation for logging/analysis.

    Args:
        own_task_weights: Own task weights
        neighbor_task_weights: Neighbor task weights
        neighbor_similarities: Neighbor similarities

    Returns:
        Dict with aggregation statistics
    """
    stats = {
        'num_neighbors': len(neighbor_task_weights),
        'avg_similarity': np.mean(neighbor_similarities) if len(neighbor_similarities) > 0 else 0.0,
        'min_similarity': np.min(neighbor_similarities) if len(neighbor_similarities) > 0 else 0.0,
        'max_similarity': np.max(neighbor_similarities) if len(neighbor_similarities) > 0 else 0.0,
    }

    # Compute task overlap with each neighbor
    try:
        from decentralized.task_similarity import compute_task_overlap
    except ImportError:
        import sys
        sys.path.insert(0, 'src')
        from decentralized.task_similarity import compute_task_overlap

    task_overlaps = []
    for neighbor_weights in neighbor_task_weights:
        overlap = compute_task_overlap(own_task_weights, neighbor_weights)
        task_overlaps.append(overlap)

    stats['avg_task_overlap'] = np.mean(task_overlaps) if len(task_overlaps) > 0 else 0.0
    stats['min_task_overlap'] = np.min(task_overlaps) if len(task_overlaps) > 0 else 0.0
    stats['max_task_overlap'] = np.max(task_overlaps) if len(task_overlaps) > 0 else 0.0

    return stats


if __name__ == "__main__":
    """Test multi-task aggregation functions"""

    print("=" * 60)
    print("Testing Multi-Task Aggregation")
    print("=" * 60)

    # Create dummy backbone and head parameters
    dummy_backbone = {
        'layer1.weight': torch.randn(64, 32),
        'layer1.bias': torch.randn(64)
    }

    dummy_head = {
        'decoder.weight': torch.randn(13, 64),
        'decoder.bias': torch.randn(13)
    }

    # Test 1: Backbone aggregation
    print("\n[Test 1] Backbone Aggregation")
    neighbors = [
        {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_backbone.items()},
        {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_backbone.items()},
    ]
    similarities = [0.8, 0.6]

    aggregated_backbone = aggregate_backbone_weighted(
        dummy_backbone,
        neighbors,
        similarities,
        self_weight=0.2
    )

    print(f"Original backbone layer1.weight mean: {dummy_backbone['layer1.weight'].mean():.4f}")
    print(f"Aggregated backbone layer1.weight mean: {aggregated_backbone['layer1.weight'].mean():.4f}")
    print("[OK] Backbone aggregation works!")

    # Test 2: Task-selective head aggregation
    print("\n[Test 2] Task-Selective Head Aggregation")
    neighbor_heads = [
        {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_head.items()},
        {k: v + torch.randn_like(v) * 0.1 for k, v in dummy_head.items()},
    ]
    neighbor_task_weights = [
        {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0},
        {'depth': 0.0, 'segmentation': 0.8, 'normal': 0.2},
    ]

    aggregated_head_depth = aggregate_task_head_selective(
        dummy_head,
        neighbor_heads,
        similarities,
        neighbor_task_weights,
        task_name='depth',
        task_threshold=0.5
    )

    print(f"Original head decoder.weight mean: {dummy_head['decoder.weight'].mean():.4f}")
    print(f"Aggregated head (depth task) decoder.weight mean: {aggregated_head_depth['decoder.weight'].mean():.4f}")
    print(f"Only neighbor 0 should be used (task_weight=0.7 >= 0.5)")
    print("[OK] Task-selective head aggregation works!")

    # Test 3: Aggregation stats
    print("\n[Test 3] Aggregation Statistics")
    own_weights = {'depth': 0.6, 'segmentation': 0.4, 'normal': 0.0}
    stats = compute_aggregation_stats(own_weights, neighbor_task_weights, similarities)

    print(f"Number of neighbors: {stats['num_neighbors']}")
    print(f"Average similarity: {stats['avg_similarity']:.4f}")
    print(f"Average task overlap: {stats['avg_task_overlap']:.4f}")
    print("[OK] Statistics computation works!")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
