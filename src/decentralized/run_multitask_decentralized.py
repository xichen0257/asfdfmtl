"""
Multi-Task Decentralized Federated Learning with Soft Clustering

This script implements decentralized FMTL for multi-task dense prediction:
1. No central server - all communication is P2P
2. Clients work on multiple tasks with different weights (soft clustering)
3. Task-aware similarity: combines task overlap and gradient similarity
4. Task-selective aggregation: backbone with all neighbors, heads with task-specific neighbors

Usage:
    python run_multitask_decentralized.py --config configs/decentralized/nyuv2_pairwise.yml
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decentralized.multitask_client import MultiTaskClient, create_multitask_clients
from decentralized.task_similarity import (
    compute_pairwise_multitask_similarity,
    log_multitask_clustering_info,
    compute_cross_loss_similarity
)
from decentralized.training_utils import (
    LearningRateScheduler,
    EarlyStopping,
    create_lr_scheduler,
    create_early_stopping,
    TrainingMonitor
)
from data_handling.nyuv2 import DMNYUDepthV2
from data_handling.pascal_context import DMPascalContext
from client_handling.seed import set_seed


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_multitask_federated_round(
    clients: List[MultiTaskClient],
    round_num: int,
    config: Dict,
    verbose: bool = True
) -> Dict:
    """
    Execute one round of multi-task federated learning.

    Phases:
    1. All clients perform local training
    2. All clients compute similarity (after training)
    3. Each client selects neighbors and aggregates
    4. Evaluate all clients

    Args:
        clients: List of MultiTaskClient instances
        round_num: Current round number
        config: Experiment configuration
        verbose: Whether to print detailed info

    Returns:
        Dict with round results
    """
    num_local_epochs = config['training']['local_epochs']
    n_clients = len(clients)

    if verbose:
        print("\n" + "="*80)
        print(f"ROUND {round_num}")
        print("="*80)
        sys.stdout.flush()

    # Phase 1: Local Training (all clients)
    if verbose:
        print(f"\n[Phase 1] Local Training ({num_local_epochs} epochs per client)")
        sys.stdout.flush()

    for client in clients:
        if verbose:
            print(f"\nClient {client.client_id} | Task weights: {client.task_weights}")
            sys.stdout.flush()

        # Store prev params before training
        from decentralized.multitask_aggregation import extract_model_parameters
        client.prev_params = extract_model_parameters(client.model)

        # Store prev backbone for HCA (if using HCA)
        if client.aggregation_method == 'hca':
            client.prev_params_backbone = client.prev_params['backbone']

        # Train
        client.local_training(num_epochs=num_local_epochs, verbose=True)

    # Phase 2: Compute similarity (all clients)
    if verbose:
        print(f"\n[Phase 2] Computing Task-Aware Similarity")
        sys.stdout.flush()

    # Get similarity method from config
    similarity_method = config['setup'].get('similarity_method', 'gradient')

    if similarity_method == 'cross_loss':
        # Use cross-loss similarity (evaluates models on each other's validation data)
        if verbose:
            print(f"  Using cross-loss similarity method")
            sys.stdout.flush()

        num_eval_batches = config['setup'].get('similarity_eval_batches', 10)
        similarity_matrix = compute_cross_loss_similarity(clients, num_eval_batches)

    else:
        # Use gradient-based similarity (default)
        if verbose:
            print(f"  Using gradient-based similarity (alpha={config['setup']['alpha']})")
            sys.stdout.flush()

        # Collect task weights and gradients
        client_task_weights = [c.task_weights for c in clients]
        client_gradients = []

        for client in clients:
            from decentralized.task_similarity import compute_model_gradient
            from decentralized.multitask_aggregation import extract_model_parameters

            if client.prev_params is not None:
                curr_params = extract_model_parameters(client.model)
                gradient = compute_model_gradient(
                    curr_params['backbone'],
                    client.prev_params['backbone']
                )
            else:
                # First round: dummy gradient
                gradient = torch.zeros(100)

            client_gradients.append(gradient)

        # Compute pairwise similarity
        similarity_matrix = compute_pairwise_multitask_similarity(
            client_task_weights,
            client_gradients,
            alpha=config['setup']['alpha']
        )

    if verbose:
        if similarity_method == 'cross_loss':
            print(f"\nSimilarity Matrix (method=cross_loss):")
        else:
            print(f"\nSimilarity Matrix (method=gradient, alpha={config['setup']['alpha']}):")
        print("     ", end="")
        for i in range(n_clients):
            print(f"C{i:2d}  ", end="")
        print()
        for i in range(n_clients):
            print(f"C{i:2d} ", end="")
            for j in range(n_clients):
                if i == j:
                    print(f" --- ", end="")
                else:
                    print(f"{similarity_matrix[i,j]:.3f}", end=" ")
            print()

    # Phase 3: Neighbor Selection & Aggregation (each client)
    if verbose:
        print(f"\n[Phase 3] Neighbor Selection & Aggregation")
        sys.stdout.flush()

    for client in clients:
        # Select neighbors
        neighbor_indices = client.select_neighbors(similarity_matrix)
        neighbor_similarities = [
            similarity_matrix[client.client_id, idx] for idx in neighbor_indices
        ]

        if verbose:
            neighbor_task_weights = [clients[idx].task_weights for idx in neighbor_indices]
            print(f"\nClient {client.client_id}:")
            print(f"  Task weights: {client.task_weights}")
            print(f"  Selected neighbors: {neighbor_indices}")
            print(f"  Neighbor task weights:")
            for idx, weights in zip(neighbor_indices, neighbor_task_weights):
                print(f"    C{idx}: {weights}")
            print(f"  Similarities: {[f'{s:.4f}' for s in neighbor_similarities]}")

        # Aggregate
        client.aggregate_with_neighbors(
            neighbor_indices,
            neighbor_similarities,
            verbose=False
        )

    # Phase 4: Evaluation
    if verbose:
        print(f"\n[Phase 4] Evaluation")

    round_results = {
        'round': round_num,
        'clients': []
    }

    total_loss = 0.0
    # Dynamically get task names from first client
    task_names = list(clients[0].task_weights.keys())
    task_totals = {task: 0.0 for task in task_names}

    for client in clients:
        eval_results = client.evaluate()

        client_result = {
            'client_id': client.client_id,
            'task_weights': client.task_weights,
            'eval_results': eval_results
        }
        round_results['clients'].append(client_result)

        total_loss += eval_results['total_loss']
        # Dynamically accumulate task losses
        for task in task_names:
            if task in eval_results:
                task_totals[task] += eval_results[task]

        if verbose:
            print(f"\nClient {client.client_id}:")
            print(f"  Total Loss: {eval_results['total_loss']:.4f}")
            for task in task_names:
                if task in eval_results:
                    print(f"  {task.replace('_', ' ').title()} Loss: {eval_results[task]:.4f}")

    # Average metrics
    round_results['avg_total_loss'] = total_loss / n_clients
    for task in task_names:
        round_results[f'avg_{task}_loss'] = task_totals[task] / n_clients

    if verbose:
        print(f"\n{'='*80}")
        print(f"Round {round_num} Summary:")
        print(f"  Avg Total Loss: {round_results['avg_total_loss']:.4f}")
        for task in task_names:
            print(f"  Avg {task.replace('_', ' ').title()} Loss: {round_results[f'avg_{task}_loss']:.4f}")
        print("="*80)

    return round_results


def main():
    parser = argparse.ArgumentParser(description='Multi-Task Decentralized FL')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    set_seed(config['general']['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Print experiment info
    print("\n" + "="*80)
    print("MULTI-TASK DECENTRALIZED FEDERATED LEARNING")
    print("="*80)
    # Determine dataset type
    dataset_type = config['data'].get('dataset', 'nyuv2').lower()

    # Dataset-specific configuration
    if dataset_type == 'pascal_context':
        dataset_name = "Pascal Context"
        tasks_str = "segmentation, human_parts, edge"
        num_seg_classes = config['model']['num_seg_classes']  # 59
        num_human_parts_classes = config['model']['num_human_parts_classes']  # 15
        # Edge detection is binary (1 channel)
    else:  # Default to nyuv2
        dataset_name = "NYU Depth V2"
        tasks_str = "depth, segmentation, normal"
        num_seg_classes = 13
        num_human_parts_classes = None

    print(f"Experiment: {config['general']['title']}")
    print(f"Dataset: {dataset_name}")
    print(f"Tasks: {tasks_str}")
    print(f"Clients: {config['setup']['num_clients']}")
    print(f"Rounds: {config['training']['num_rounds']}")
    print(f"Local epochs: {config['training']['local_epochs']}")
    print(f"Neighbors: {config['setup']['n_neighbors']}")
    print(f"Alpha (task overlap vs gradient): {config['setup']['alpha']}")
    print(f"Aggregate heads: {config['setup'].get('aggregate_heads', True)}")
    print("="*80)

    # Create data manager based on dataset type
    print(f"\nCreating {dataset_name} data manager...")
    if dataset_type == 'pascal_context':
        data_manager = DMPascalContext(
            seed=config['general']['seed'],
            num_clients=config['setup']['num_clients'],
            task_weights_per_client=config['setup']['task_weights_per_client'],
            dataset_fraction=config['data'].get('dataset_fraction', 1.0),
            batch_size=config['training']['batch_size'],
            root_dir=config['data']['root_dir'],
            download=True
        )
    else:  # nyuv2
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
    print("\nCreating multi-task clients...")
    client_params = {
        'data_manager': data_manager,
        'task_weights_per_client': config['setup']['task_weights_per_client'],
        'num_seg_classes': num_seg_classes,
        'n_neighbors': config['setup']['n_neighbors'],
        'alpha': config['setup']['alpha'],
        'aggregate_heads': config['setup'].get('aggregate_heads', True),
        'aggregation_method': config['setup'].get('aggregation_method', 'weighted'),
        'hca_alpha': config['setup'].get('hca_alpha', 0.7),
        'device': device,
        'gradient_clip_norm': config['training'].get('gradient_clip_norm', None),
        'lr': config['training'].get('learning_rate', 0.001)
    }

    # Add human_parts_classes for Pascal Context
    if dataset_type == 'pascal_context':
        client_params['num_human_parts_classes'] = num_human_parts_classes

    clients = create_multitask_clients(**client_params)
    print(f"[OK] Created {len(clients)} clients")

    # Print aggregation method
    agg_method = config['setup'].get('aggregation_method', 'weighted')
    if agg_method == 'hca':
        print(f"  Aggregation method: HCA (alpha={config['setup'].get('hca_alpha', 0.7)})")
    else:
        print(f"  Aggregation method: Weighted")

    # Print client configuration
    print("\nClient Task Weights:")
    for client in clients:
        print(f"  Client {client.client_id}: {client.task_weights}")

    # Run federated learning
    print("\n" + "="*80)
    print("STARTING FEDERATED TRAINING")
    print("="*80)

    all_results = []

    # Initialize learning rate scheduler
    lr_scheduler = None
    if 'lr_scheduler' in config['training'] and config['training']['lr_scheduler'].get('enabled', False):
        # Create dummy optimizer for LR computation
        dummy_params = [torch.nn.Parameter(torch.zeros(1))]
        dummy_optimizer = torch.optim.Adam(dummy_params)
        lr_scheduler = LearningRateScheduler(
            optimizer=dummy_optimizer,
            schedule_type=config['training']['lr_scheduler'].get('type', 'cosine'),
            initial_lr=config['training']['learning_rate'],
            min_lr=config['training']['lr_scheduler'].get('min_lr', 0.0001),
            total_rounds=config['training']['num_rounds'],
            warmup_rounds=config['training']['lr_scheduler'].get('warmup_rounds', 0),
            decay_rate=config['training']['lr_scheduler'].get('decay_rate', 0.5),
            decay_steps=config['training']['lr_scheduler'].get('decay_steps', 10)
        )
        print(f"\n[OK] Learning rate scheduler enabled: {config['training']['lr_scheduler'].get('type', 'cosine')}")
        print(f"  Initial LR: {config['training']['learning_rate']}, Min LR: {config['training']['lr_scheduler'].get('min_lr', 0.0001)}")

    # Initialize early stopping
    early_stopping = None
    if 'early_stopping' in config['training'] and config['training']['early_stopping'].get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping'].get('patience', 5),
            min_delta=config['training']['early_stopping'].get('min_delta', 0.005),
            mode=config['training']['early_stopping'].get('mode', 'min'),
            restore_best_weights=config['training']['early_stopping'].get('restore_best_weights', True),
            verbose=config['training']['early_stopping'].get('verbose', True)
        )
        print(f"\n[OK] Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    # Store best client models for early stopping
    best_client_states = None

    for round_num in range(1, config['training']['num_rounds'] + 1):
        # Update learning rate
        if lr_scheduler is not None:
            current_lr = lr_scheduler.step(round_num - 1)
            # Update all client optimizers
            for client in clients:
                for param_group in client.model.optimizer.param_groups:
                    param_group['lr'] = current_lr
            print(f"\n[LR Update] Round {round_num} learning rate: {current_lr:.6f}")

        round_results = run_multitask_federated_round(
            clients=clients,
            round_num=round_num,
            config=config,
            verbose=True
        )
        all_results.append(round_results)

        # Early stopping check
        if early_stopping is not None:
            avg_loss = round_results['avg_total_loss']

            # Save current client states before early stopping check
            # (in case this is the best round)
            current_client_states = [
                {k: v.cpu().clone() for k, v in client.model.state_dict().items()}
                for client in clients
            ]

            # Check if this is the best round
            if early_stopping.best_score is None or avg_loss < early_stopping.best_score - early_stopping.min_delta:
                best_client_states = current_client_states

            # Call early stopping
            should_stop = early_stopping(avg_loss, round_num=round_num)

            if should_stop:
                print(f"\n{'='*80}")
                print("EARLY STOPPING TRIGGERED")
                print(f"{'='*80}")
                print(f"Training stopped at round {round_num}")
                print(f"Best score: {early_stopping.best_score:.6f} at round {early_stopping.best_round}")

                # Restore best models
                if early_stopping.restore_best_weights and best_client_states is not None:
                    print("\nRestoring best model weights from all clients...")
                    for i, client in enumerate(clients):
                        client.model.load_state_dict(best_client_states[i])
                    print("[OK] Best weights restored")

                print(f"{'='*80}\n")
                break

    # Save results
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{config['general']['title']}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': all_results
        }, f, indent=2)

    print(f"\n[OK] Results saved to {results_file}")

    # Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"Final metrics (Round {config['training']['num_rounds']}):")
    final_results = all_results[-1]
    print(f"  Avg Total Loss: {final_results['avg_total_loss']:.4f}")
    # Dynamically print task losses
    task_names = list(config['setup']['task_weights_per_client'][0].keys())
    for task in task_names:
        task_key = f'avg_{task}_loss'
        if task_key in final_results:
            print(f"  Avg {task.replace('_', ' ').title()} Loss: {final_results[task_key]:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
