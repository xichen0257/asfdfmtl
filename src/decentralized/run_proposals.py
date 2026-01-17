"""
Run Proposal 1 (Backbone-Only) and Proposal 2 (Hierarchical) experiments

Usage:
    python run_proposals.py --config configs/decentralized/nyuv2_proposal1_backbone_only.yml
    python run_proposals.py --config configs/decentralized/nyuv2_proposal2_hierarchical.yml
"""

import argparse
import yaml
import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decentralized.multitask_client import MultiTaskClient, create_multitask_clients
from decentralized.backbone_feature_extractor import (
    BackboneFeatureExtractor,
    compute_pairwise_similarity,
    select_top_k_neighbors
)
from decentralized.dynamic_clustering import (
    topk_soft_clustering,
    spectral_clustering_auto,
    hierarchical_clustering_threshold,
    compute_clustering_stability,
    compute_cluster_purity
)
from decentralized.task_similarity import compute_pairwise_multitask_similarity
from data_handling.nyuv2 import DMNYUDepthV2
from client_handling.seed import set_seed


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_task_output_similarity(
    client_i: MultiTaskClient,
    client_j: MultiTaskClient,
    shared_data_loader,
    device: str = 'cuda'
) -> float:
    """
    Compute task similarity based on model outputs on shared data.

    Args:
        client_i: First client
        client_j: Second client
        shared_data_loader: Shared validation data
        device: Device to use

    Returns:
        similarity: Similarity score in [0, 1]
    """
    client_i.model.eval()
    client_j.model.eval()

    similarities = []

    with torch.no_grad():
        for batch in shared_data_loader:
            x = batch['image'].to(device)

            # Get predictions from both clients
            pred_i = client_i.model(x)
            pred_j = client_j.model(x)

            # Compute similarity for each task
            for task in ['depth', 'segmentation', 'normal']:
                if task in pred_i and task in pred_j:
                    # Flatten predictions
                    pred_i_flat = pred_i[task].flatten()
                    pred_j_flat = pred_j[task].flatten()

                    # Cosine similarity
                    sim = torch.nn.functional.cosine_similarity(
                        pred_i_flat.unsqueeze(0),
                        pred_j_flat.unsqueeze(0),
                        dim=1
                    ).item()

                    # Map to [0, 1]
                    sim = (sim + 1.0) / 2.0
                    similarities.append(sim)

    return np.mean(similarities) if similarities else 0.0


def run_proposal1_round(
    clients: List[MultiTaskClient],
    round_num: int,
    config: Dict,
    verbose: bool = True
) -> Dict:
    """
    Run one round of Proposal 1: Backbone-Only Dynamic Clustering.

    Steps:
        1. Local training
        2. Extract backbone features
        3. Compute feature similarity
        4. Dynamic top-K clustering
        5. Aggregate backbones only (heads unchanged)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"ROUND {round_num} - Proposal 1: Backbone-Only Dynamic Clustering")
        print(f"{'='*80}")

    # Step 1: Local training
    if verbose:
        print("\n[1/5] Local training...")
    for client in clients:
        client.local_training(config['training']['local_epochs'])

    # Step 2: Extract backbone features
    if verbose:
        print("[2/5] Extracting backbone features...")

    all_feature_stats = []
    for client in clients:
        extractor = BackboneFeatureExtractor(client.model, device='cuda')
        stats = extractor.extract_feature_statistics(
            client.train_loader,
            max_batches=config['setup'].get('max_feature_batches', 50)
        )
        all_feature_stats.append(stats)

    # Step 3: Compute feature similarity
    if verbose:
        print("[3/5] Computing pairwise feature similarity...")

    similarity_matrix = compute_pairwise_similarity(
        all_feature_stats,
        method=config['setup'].get('feature_similarity_method', 'cosine')
    )

    if verbose:
        print(f"  Similarity matrix (first 3x3):")
        print(f"  {similarity_matrix[:3, :3].round(3)}")

    # Step 4: Dynamic clustering (Top-K)
    if verbose:
        print("[4/5] Dynamic top-K clustering...")

    k_neighbors = config['setup'].get('k_neighbors', 3)
    clusters = topk_soft_clustering(similarity_matrix, k=k_neighbors)

    if verbose:
        for client_idx, neighbors in clusters.items():
            print(f"  Client {client_idx} -> Neighbors {neighbors}")

    # Step 5: Aggregate backbones only
    if verbose:
        print("[5/5] Aggregating backbones (heads remain unchanged)...")

    for client_idx, client in enumerate(clients):
        neighbors = clusters[client_idx]

        # Get neighbor weights
        _, neighbor_weights = select_top_k_neighbors(
            client_idx, similarity_matrix, k=k_neighbors
        )

        # Aggregate backbones
        aggregated_backbone = {}

        for key in client.model.backbone.state_dict().keys():
            param = client.model.backbone.state_dict()[key]

            # Skip aggregation for integer types (e.g., num_batches_tracked in BatchNorm)
            if param.dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                aggregated_backbone[key] = param.clone()  # Just use own value
                continue

            # Weighted average: self + neighbors
            params = [param]
            weights_list = [1.0]  # Self weight

            for neighbor_idx, weight in zip(neighbors, neighbor_weights):
                params.append(clients[neighbor_idx].model.backbone.state_dict()[key])
                weights_list.append(weight)

            # Normalize weights
            weights_tensor = torch.tensor(weights_list)
            weights_tensor = weights_tensor / weights_tensor.sum()

            # Weighted sum
            aggregated_backbone[key] = sum(
                w * p for w, p in zip(weights_tensor, params)
            )

        # Update client backbone
        client.model.backbone.load_state_dict(aggregated_backbone)

    # Evaluation
    if verbose:
        print("\n[Evaluation] Running validation...")

    metrics = {}
    for client_idx, client in enumerate(clients):
        val_metrics = client.evaluate()
        metrics[f'client_{client_idx}'] = val_metrics

        if verbose:
            print(f"  Client {client_idx}: Loss = {val_metrics['total_loss']:.4f}")

    # Log clustering info
    metrics['clustering'] = {
        'method': 'backbone_only_topk',
        'k_neighbors': k_neighbors,
        'similarity_matrix': similarity_matrix.tolist(),
        'clusters': clusters
    }

    return metrics


def run_proposal2_round(
    clients: List[MultiTaskClient],
    round_num: int,
    config: Dict,
    shared_val_loader,
    verbose: bool = True
) -> Dict:
    """
    Run one round of Proposal 2: Hierarchical Dynamic Clustering.

    Steps:
        1. Local training
        2. Coarse clustering (backbone features) -> Aggregate backbones
        3. Fine clustering (task outputs) -> Aggregate heads
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"ROUND {round_num} - Proposal 2: Hierarchical Dynamic Clustering")
        print(f"{'='*80}")

    # Step 1: Local training
    if verbose:
        print("\n[1/6] Local training...")
    for client in clients:
        client.local_training(config['training']['local_epochs'])

    # Step 2: Coarse-grained clustering (Backbone features)
    if verbose:
        print("[2/6] Coarse-grained clustering (backbone features)...")

    all_feature_stats = []
    for client in clients:
        extractor = BackboneFeatureExtractor(client.model, device='cuda')
        stats = extractor.extract_feature_statistics(
            client.train_loader,
            max_batches=config['setup'].get('max_feature_batches', 50)
        )
        all_feature_stats.append(stats)

    feature_sim_matrix = compute_pairwise_similarity(
        all_feature_stats,
        method=config['setup']['coarse_clustering'].get('similarity_method', 'cosine')
    )

    # Spectral clustering
    coarse_clusters, n_coarse = spectral_clustering_auto(
        feature_sim_matrix,
        min_clusters=config['setup']['coarse_clustering'].get('min_clusters', 2),
        max_clusters=config['setup']['coarse_clustering'].get('max_clusters', 4)
    )

    if verbose:
        print(f"  Found {n_coarse} coarse clusters:")
        for cluster_id, members in coarse_clusters.items():
            print(f"    Cluster {cluster_id}: clients {members}")

    # Step 3: Aggregate backbones within coarse clusters
    if verbose:
        print("[3/6] Aggregating backbones within coarse clusters...")

    for cluster_id, member_indices in coarse_clusters.items():
        cluster_clients = [clients[i] for i in member_indices]

        # Average backbones
        avg_backbone = {}
        for key in cluster_clients[0].model.backbone.state_dict().keys():
            params = [c.model.backbone.state_dict()[key] for c in cluster_clients]
            # Skip averaging for integer types (e.g., num_batches_tracked in BatchNorm)
            if params[0].dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                avg_backbone[key] = params[0].clone()  # Just use first client's value
            else:
                avg_backbone[key] = torch.stack(params).mean(dim=0)

        # Update all members
        for client in cluster_clients:
            client.model.backbone.load_state_dict(avg_backbone)

    # Step 4: Fine-grained clustering (Task similarity)
    if verbose:
        print("[4/6] Fine-grained clustering (task outputs)...")

    all_fine_clusters = []

    for coarse_cluster_id, coarse_members in coarse_clusters.items():
        if len(coarse_members) == 1:
            # Single client, no need for fine clustering
            all_fine_clusters.append(coarse_members)
            continue

        # Compute task similarity within coarse cluster
        N_coarse = len(coarse_members)
        task_sim_matrix = np.zeros((N_coarse, N_coarse))

        for i in range(N_coarse):
            for j in range(i, N_coarse):
                if i == j:
                    task_sim_matrix[i, j] = 1.0
                else:
                    sim = compute_task_output_similarity(
                        clients[coarse_members[i]],
                        clients[coarse_members[j]],
                        shared_val_loader
                    )
                    task_sim_matrix[i, j] = sim
                    task_sim_matrix[j, i] = sim

        # Hierarchical clustering
        fine_clusters_dict = hierarchical_clustering_threshold(
            task_sim_matrix,
            threshold=config['setup']['fine_clustering'].get('distance_threshold', 0.4)
        )

        # Map back to global client indices
        for fine_cluster_members in fine_clusters_dict.values():
            global_indices = [coarse_members[i] for i in fine_cluster_members]
            all_fine_clusters.append(global_indices)

    if verbose:
        print(f"  Found {len(all_fine_clusters)} fine clusters:")
        for idx, members in enumerate(all_fine_clusters):
            print(f"    Fine cluster {idx}: clients {members}")

    # Step 5: Aggregate heads within fine clusters
    if verbose:
        print("[5/6] Aggregating heads within fine clusters...")

    for fine_cluster in all_fine_clusters:
        if len(fine_cluster) == 1:
            continue  # Single client, no aggregation

        cluster_clients = [clients[i] for i in fine_cluster]

        # Get common tasks
        common_tasks = set(cluster_clients[0].task_weights.keys())
        for client in cluster_clients[1:]:
            common_tasks &= set(client.task_weights.keys())

        # Aggregate each common task's head
        # Map task names to head attribute names
        task_to_head = {
            'depth': 'depth_head',
            'segmentation': 'seg_head',
            'normal': 'normal_head'
        }

        for task in common_tasks:
            head_name = task_to_head.get(task, f'{task}_head')

            # Check if head exists on first client
            if not hasattr(cluster_clients[0].model, head_name):
                print(f"Warning: Head '{head_name}' not found for task '{task}', skipping...")
                continue

            avg_head = {}
            for key in getattr(cluster_clients[0].model, head_name).state_dict().keys():
                params = [getattr(c.model, head_name).state_dict()[key] for c in cluster_clients]
                # Skip averaging for integer types (e.g., num_batches_tracked in BatchNorm)
                if params[0].dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
                    avg_head[key] = params[0].clone()  # Just use first client's value
                else:
                    avg_head[key] = torch.stack(params).mean(dim=0)

            # Update all members
            for client in cluster_clients:
                getattr(client.model, head_name).load_state_dict(avg_head)

    # Step 6: Evaluation
    if verbose:
        print("[6/6] Running validation...")

    metrics = {}
    for client_idx, client in enumerate(clients):
        val_metrics = client.evaluate()
        metrics[f'client_{client_idx}'] = val_metrics

        if verbose:
            print(f"  Client {client_idx}: Loss = {val_metrics['total_loss']:.4f}")

    # Log clustering info
    metrics['clustering'] = {
        'method': 'hierarchical',
        'n_coarse_clusters': n_coarse,
        'n_fine_clusters': len(all_fine_clusters),
        'coarse_clusters': {k: v for k, v in coarse_clusters.items()},
        'fine_clusters': {i: cluster for i, cluster in enumerate(all_fine_clusters)},
        'feature_similarity': feature_sim_matrix.tolist()
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run Proposal 1 or 2 experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    args = parser.parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load config
    config = load_config(args.config)
    set_seed(config['general']['seed'])

    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f)

    # Create data manager
    print("\n[Setup] Creating NYU Depth V2 data manager...")
    data_manager = DMNYUDepthV2(
        seed=config['general']['seed'],
        num_clients=config['setup']['num_clients'],
        task_weights_per_client=config['setup']['task_weights_per_client'],
        dataset_fraction=config['data'].get('dataset_fraction', 1.0),
        batch_size=config['training']['batch_size'],
        root_dir=config['data']['root_dir'],
        download=True
    )
    print("[OK] Data manager created")

    # Create clients
    print("\n[Setup] Creating multi-task clients...")
    clients = create_multitask_clients(
        data_manager=data_manager,
        task_weights_per_client=config['setup']['task_weights_per_client'],
        num_seg_classes=config['model']['num_seg_classes'],
        n_neighbors=config['setup'].get('k_neighbors', 3),  # Use k_neighbors from config
        alpha=config['setup'].get('alpha', 0.5),
        aggregate_heads=config['setup'].get('aggregate_heads', True),
        device=device
    )
    print(f"[OK] Created {len(clients)} clients")

    # Prepare shared validation data (for Proposal 2)
    shared_val_loader = None
    if config['setup']['clustering_method'] == 'hierarchical':
        print("[Setup] Preparing shared validation data...")
        from torch.utils.data import Subset, DataLoader

        # Use first client's dataset
        val_dataset = clients[0].val_loader.dataset
        n_shared = config['setup'].get('shared_val_samples', 100)
        indices = np.random.choice(len(val_dataset), min(n_shared, len(val_dataset)), replace=False)
        shared_dataset = Subset(val_dataset, indices)
        shared_val_loader = DataLoader(shared_dataset, batch_size=8, shuffle=False)

    # Training loop
    print(f"\n[Training] Starting {config['training']['num_rounds']} rounds...")

    all_metrics = []

    for round_num in range(1, config['training']['num_rounds'] + 1):
        if config['setup']['clustering_method'] == 'backbone_only':
            metrics = run_proposal1_round(clients, round_num, config, verbose=True)
        elif config['setup']['clustering_method'] == 'hierarchical':
            metrics = run_proposal2_round(clients, round_num, config, shared_val_loader, verbose=True)
        else:
            raise ValueError(f"Unknown clustering method: {config['setup']['clustering_method']}")

        all_metrics.append(metrics)

        # Save metrics every round (for peak analysis)
        if config['output'].get('save_metrics_every_round', False):
            metrics_path = output_dir / f'metrics_round_{round_num}.json'
            with open(metrics_path, 'w') as f:
                json.dump(convert_numpy_types(metrics), f, indent=2)

        # Save checkpoint
        if round_num % config['output']['checkpoint_frequency'] == 0:
            checkpoint_path = output_dir / f'checkpoint_round_{round_num}.pt'
            torch.save({
                'round': round_num,
                'clients': [c.model.state_dict() for c in clients],
                'metrics': all_metrics
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

        # Also save cumulative metrics after each round
        cumulative_metrics_path = output_dir / 'metrics_all_rounds.json'
        with open(cumulative_metrics_path, 'w') as f:
            json.dump(convert_numpy_types(all_metrics), f, indent=2)

    # Save final results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_numpy_types(all_metrics), f, indent=2)

    print(f"\n[OK] Training complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
