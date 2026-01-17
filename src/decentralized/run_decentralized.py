"""
Decentralized Federated Multi-Task Learning with Dynamic Soft Clustering

This script implements a fully decentralized FMTL system where:
1. No central server exists - all communication is P2P
2. Clients dynamically form soft clusters (N=3) based on task similarity
3. Task similarity is computed using gradient cosine similarity
4. HCA aggregation is performed locally by each client with selected neighbors

Usage:
    python run_decentralized.py --config configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml
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

from decentralized.decentralized_client import DecentralizedClient
from decentralized.task_similarity import (
    compute_pairwise_similarity,
    compute_pairwise_parameter_similarity,
    compute_task_type_similarity,
    compute_cross_loss_similarity,
    log_clustering_info
)
from data_handling.data_manager import get_data_manager
from client_handling.seed import set_seed
import torchvision
from models.single_label_classification_model import SingleLabelClassificationModel


def load_config(config_path: str) -> Dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_decentralized_clients(
    config: Dict,
    data_manager,
    device: torch.device
) -> List[DecentralizedClient]:
    """
    Create all decentralized clients for the experiment.

    Args:
        config: Experiment configuration
        data_manager: Data manager object for loading data
        device: Computing device (cuda/cpu)

    Returns:
        List of DecentralizedClient instances
    """
    clients = []
    task_configs = config['setup']['task_configs']
    title = config['general']['title']
    backbone_layers = config['general']['backbone_layers']
    n_neighbors = config['setup']['n_neighbors']
    hca_alpha = config['setup']['hca_alpha']
    aggregation_method = config['setup'].get('aggregation_method', 'hca')  # Default to HCA
    learning_rate = config['setup'].get('learning_rate', 0.01)  # Default for CIFAR-10

    # Hybrid strategy parameters
    aggregation_start_round = config['setup'].get('aggregation_start_round', 1)  # Default: start from round 1
    aggregation_switch_round = config['setup'].get('aggregation_switch_round', None)  # Default: no switching

    # Adaptive aggregation parameters
    adaptive_aggregation = config['setup'].get('adaptive_aggregation', False)
    adaptive_method = config['setup'].get('adaptive_method', 'weighted')
    adaptive_config = config['setup'].get('adaptive_config', {})

    print("\n" + "="*60)
    print("Creating Decentralized Clients")
    print("="*60)
    if adaptive_aggregation:
        print(f"Using adaptive aggregation: {adaptive_method}")
        print(f"Adaptive config: {adaptive_config}")
    print(f"Aggregation method: {aggregation_method.upper()}")
    if aggregation_method == 'hybrid':
        print(f"  Hybrid strategy: FedAvg (rounds 1-{aggregation_switch_round-1}) -> HCA (rounds {aggregation_switch_round}+)")
    if aggregation_start_round > 1:
        print(f"  Delayed aggregation: Starting from round {aggregation_start_round}")

    # Create Animal task clients
    animals_config = task_configs['Animals']
    for idx, client_id in enumerate(animals_config['client_ids']):
        seed = 300 + idx
        set_seed(seed)

        model = SingleLabelClassificationModel(
            num_classes=animals_config['num_classes'],
            model=torchvision.models.resnet18(pretrained=False),
            train_loader=data_manager.train_animals_loaders[idx],
            val_loader=data_manager.val_animals_loaders[idx],
            test_loader=data_manager.test_animals_loader,
            backbone_layers=backbone_layers,
            lr=learning_rate,
        ).to(device)

        client = DecentralizedClient(
            c_id=client_id,
            title=title,
            tasktype='Animals',
            model=model,
            seed=seed,
            n_neighbors=n_neighbors,
            hca_alpha=hca_alpha,
            aggregation_method=aggregation_method,
            aggregation_start_round=aggregation_start_round,
            aggregation_switch_round=aggregation_switch_round,
            adaptive_aggregation=adaptive_aggregation,
            adaptive_method=adaptive_method,
            adaptive_config=adaptive_config
        )

        clients.append(client)
        print(f"  Created client: {client_id} (Animals, seed={seed})")

    # Create Object task clients
    objects_config = task_configs['Objects']
    for idx, client_id in enumerate(objects_config['client_ids']):
        seed = 400 + idx
        set_seed(seed)

        model = SingleLabelClassificationModel(
            num_classes=objects_config['num_classes'],
            model=torchvision.models.resnet18(pretrained=False),
            train_loader=data_manager.train_objects_loaders[idx],
            val_loader=data_manager.val_objects_loaders[idx],
            test_loader=data_manager.test_objects_loader,
            backbone_layers=backbone_layers,
            lr=learning_rate,
        ).to(device)

        client = DecentralizedClient(
            c_id=client_id,
            title=title,
            tasktype='Objects',
            model=model,
            seed=seed,
            n_neighbors=n_neighbors,
            hca_alpha=hca_alpha,
            aggregation_method=aggregation_method,
            aggregation_start_round=aggregation_start_round,
            aggregation_switch_round=aggregation_switch_round,
            adaptive_aggregation=adaptive_aggregation,
            adaptive_method=adaptive_method,
            adaptive_config=adaptive_config
        )

        clients.append(client)
        print(f"  Created client: {client_id} (Objects, seed={seed})")

    # Set all_clients reference for P2P communication
    for client in clients:
        client.set_all_clients(clients)

    print(f"\nTotal clients created: {len(clients)}")
    return clients


def compute_similarity_matrix(clients: List[DecentralizedClient], round_num: int, config: Dict) -> np.ndarray:
    """
    Compute pairwise task similarity matrix for all clients.

    Supports multiple similarity computation methods via config:
    - 'gradient': Gradient-based similarity (parameter changes)
    - 'parameter': Parameter-based similarity (actual parameter values)
    - 'cross_loss': Cross-validation loss based similarity
    - 'task_prior': Task-type prior knowledge (same-task vs cross-task)

    Args:
        clients: List of all clients
        round_num: Current round number
        config: Experiment configuration dictionary

    Returns:
        N x N similarity matrix
    """
    # Read similarity method from config (default to gradient if not specified)
    similarity_method = config['setup'].get('similarity_method', 'gradient')

    print(f"\n[Round {round_num}] Computing task similarity using method: {similarity_method}")

    # Compute similarity based on selected method
    if similarity_method == 'gradient':
        # Gradient-based similarity (original method)
        print("  Method: Gradient-based (parameter changes)")
        gradients = []
        for client in clients:
            grad = client.compute_gradient()
            if grad is not None:
                grad_np = grad.detach().cpu().numpy()
                grad_norm = np.linalg.norm(grad_np)
                print(f"  Client {client.c_id} ({client.task_type}) gradient norm: {grad_norm:.4f}")
                gradients.append(grad)
            else:
                print(f"  Client {client.c_id} gradient is None.")
                gradients.append(torch.zeros(1))

        similarity_matrix = compute_pairwise_similarity(gradients)

    elif similarity_method == 'parameter':
        # Parameter-based similarity (Solution D1)
        print("  Method: Parameter-based (actual parameter values)")
        parameters = []
        for client in clients:
            params = client.compute_parameters()
            if params is not None:
                param_np = params.detach().cpu().numpy()
                param_norm = np.linalg.norm(param_np)
                print(f"  Client {client.c_id} ({client.task_type}) parameter norm: {param_norm:.4f}")
                parameters.append(params)
            else:
                print(f"  Client {client.c_id} parameters are None.")
                parameters.append(torch.zeros(1))

        similarity_matrix = compute_pairwise_parameter_similarity(parameters)

    elif similarity_method == 'cross_loss':
        # Cross-validation loss based similarity (Solution E)
        print("  Method: Cross-loss based (cross-validation performance)")
        num_eval_batches = config['setup'].get('similarity_eval_batches', 10)
        print(f"  Using {num_eval_batches} batches per cross-evaluation")

        similarity_matrix = compute_cross_loss_similarity(clients, num_eval_batches)

    elif similarity_method == 'task_prior':
        # Task-type prior knowledge (Solution B)
        print("  Method: Task-type prior (domain knowledge)")
        same_task_sim = config['setup'].get('same_task_similarity', 0.9)
        cross_task_sim = config['setup'].get('cross_task_similarity', 0.1)
        print(f"  Same-task similarity: {same_task_sim}, Cross-task similarity: {cross_task_sim}")

        similarity_matrix = compute_task_type_similarity(clients, same_task_sim, cross_task_sim)

    else:
        raise ValueError(f"Unknown similarity_method: {similarity_method}. "
                        f"Must be one of: gradient, parameter, cross_loss, task_prior")

    # Print similarity matrix statistics
    print(f"\n  Similarity matrix shape: {similarity_matrix.shape}")
    print(f"  Similarity matrix:\n{similarity_matrix}")
    print(f"  Average similarity: {np.mean(similarity_matrix[similarity_matrix < 1.0]):.4f}")
    print(f"  Min similarity: {np.min(similarity_matrix[similarity_matrix < 1.0]):.4f}")
    print(f"  Max similarity: {np.max(similarity_matrix[similarity_matrix < 1.0]):.4f}")

    # Show task type distribution for reference
    animals = [c.c_id for c in clients if c.task_type == 'Animals']
    objects = [c.c_id for c in clients if c.task_type == 'Objects']
    print(f"  Task distribution - Animals: {animals}, Objects: {objects}")

    return similarity_matrix


def evaluate_clients(clients: List[DecentralizedClient], round_num: int) -> Dict:
    """
    Evaluate all clients and aggregate metrics.

    Args:
        clients: List of all clients
        round_num: Current round number

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n[Round {round_num}] Evaluating all clients...")

    results = {
        'round': round_num,
        'clients': {},
        'animals_avg': {},
        'objects_avg': {},
        'overall_avg': {}
    }

    animals_metrics = {'test_loss': [], 'test_acc': []}
    objects_metrics = {'test_loss': [], 'test_acc': []}

    for client in clients:
        # Get test metrics (returns dict with keys: loss, precision, recall, f1)
        test_metrics = client.model.test_model()
        test_loss = test_metrics['loss']
        test_f1 = test_metrics['f1'].item() if hasattr(test_metrics['f1'], 'item') else float(test_metrics['f1'])  # Convert to Python float

        # Convert all metrics to Python floats
        precision = test_metrics['precision'].item() if hasattr(test_metrics['precision'], 'item') else float(test_metrics['precision'])
        recall = test_metrics['recall'].item() if hasattr(test_metrics['recall'], 'item') else float(test_metrics['recall'])

        results['clients'][client.c_id] = {
            'test_loss': test_loss,
            'test_f1': test_f1,
            'test_precision': precision,
            'test_recall': recall,
            'tasktype': client.task_type
        }

        # Aggregate by task type
        if client.task_type == 'Animals':
            animals_metrics['test_loss'].append(test_loss)
            animals_metrics['test_acc'].append(test_f1)  # Use F1 as proxy for accuracy
        else:
            objects_metrics['test_loss'].append(test_loss)
            objects_metrics['test_acc'].append(test_f1)

        print(f"  {client.c_id} ({client.task_type}): Loss={test_loss:.4f}, F1={test_f1:.4f}")

    # Compute averages
    results['animals_avg'] = {
        'test_loss': np.mean(animals_metrics['test_loss']),
        'test_acc': np.mean(animals_metrics['test_acc'])
    }
    results['objects_avg'] = {
        'test_loss': np.mean(objects_metrics['test_loss']),
        'test_acc': np.mean(objects_metrics['test_acc'])
    }
    results['overall_avg'] = {
        'test_loss': np.mean(animals_metrics['test_loss'] + objects_metrics['test_loss']),
        'test_acc': np.mean(animals_metrics['test_acc'] + objects_metrics['test_acc'])
    }

    print(f"\n  Animals Average: Loss={results['animals_avg']['test_loss']:.4f}, "
          f"Acc={results['animals_avg']['test_acc']:.4f}")
    print(f"  Objects Average: Loss={results['objects_avg']['test_loss']:.4f}, "
          f"Acc={results['objects_avg']['test_acc']:.4f}")
    print(f"  Overall Average: Loss={results['overall_avg']['test_loss']:.4f}, "
          f"Acc={results['overall_avg']['test_acc']:.4f}")

    return results


def save_results(results_history: List[Dict], config: Dict, output_dir: str):
    """
    Save experiment results to JSON file.

    Args:
        results_history: List of results from each round
        config: Experiment configuration
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'results.json')

    output_data = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'results_by_round': results_history
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def analyze_clustering_patterns(clients: List[DecentralizedClient], output_dir: str):
    """
    Analyze and save clustering patterns across rounds.

    Args:
        clients: List of all clients
        output_dir: Output directory path
    """
    print("\n" + "="*60)
    print("Analyzing Clustering Patterns")
    print("="*60)

    clustering_analysis = {}

    for client in clients:
        neighbor_history = client.get_neighbor_history()

        # Convert neighbor indices to client IDs
        neighbor_ids_history = []
        for neighbors in neighbor_history:
            neighbor_ids = [clients[idx].c_id for idx in neighbors]
            neighbor_ids_history.append(neighbor_ids)

        clustering_analysis[client.c_id] = {
            'tasktype': client.task_type,
            'neighbor_history': neighbor_ids_history
        }

        print(f"\n{client.c_id} ({client.task_type}):")
        for round_idx, neighbor_ids in enumerate(neighbor_ids_history):
            print(f"  Round {round_idx}: {neighbor_ids}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save clustering analysis
    output_file = os.path.join(output_dir, 'clustering_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(clustering_analysis, f, indent=2)

    print(f"\nClustering analysis saved to: {output_file}")


def run_experiment(config_path: str):
    """
    Run the complete decentralized federated learning experiment.

    Args:
        config_path: Path to experiment configuration file
    """
    print("\n" + "="*80)
    print("DECENTRALIZED FEDERATED MULTI-TASK LEARNING WITH DYNAMIC SOFT CLUSTERING")
    print("="*80)

    # Load configuration
    config = load_config(config_path)
    print(f"\nExperiment: {config['general']['title']}")
    print(f"Config: {config_path}")

    # Set random seed
    set_seed(config['general']['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create data manager
    print("\nLoading dataset...")
    task_configs = config['setup']['task_configs']
    task_1_clients_ids = task_configs['Animals']['client_ids']
    task_2_clients_ids = task_configs['Objects']['client_ids']

    data_manager = get_data_manager(
        dataset=config['general']['dataset'],
        task_1_clients_ids=task_1_clients_ids,
        task_2_clients_ids=task_2_clients_ids,
        dataset_fraction=config['setup']['dataset_fraction']
    )

    # Create clients
    clients = create_decentralized_clients(config, data_manager, device)

    # Initialize previous backbones for all clients
    print("\nInitializing client backbones...")
    for client in clients:
        client.initialize_prev_backbone()

    # Get training parameters
    num_rounds = config['general']['rounds']
    local_epochs = config['setup']['local_epochs']

    # Storage for results
    results_history = []

    # Initial evaluation (Round 0)
    print("\n" + "="*80)
    print("INITIAL EVALUATION (Round 0)")
    print("="*80)
    initial_results = evaluate_clients(clients, round_num=0)
    results_history.append(initial_results)

    # Training loop
    print("\n" + "="*80)
    print("STARTING DECENTRALIZED TRAINING")
    print("="*80)
    print(f"Total rounds: {num_rounds}")
    print(f"Local epochs per round: {local_epochs}")
    print(f"Soft clustering: N={config['setup']['n_neighbors']} neighbors")
    print(f"HCA alpha: {config['setup']['hca_alpha']}")

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*80}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*80}")

        # Phase 1: All clients train locally
        print(f"\n[Round {round_num}] Phase 1: Local training...")
        for client in clients:
            client.local_training_phase(
                round_num=round_num,
                epochs=local_epochs,
                verbose=False
            )

        # Phase 2: All clients save snapshots of trained backbones
        print(f"[Round {round_num}] Phase 2: Saving trained backbone snapshots...")
        for client in clients:
            client.snapshot_trained_backbone()

        # Phase 3: Compute task similarity matrix (AFTER training!)
        # This ensures gradients are based on actual training, not zero
        print(f"[Round {round_num}] Phase 3: Computing task similarity...")
        similarity_matrix = compute_similarity_matrix(clients, round_num, config)

        # Phase 4: All clients aggregate (read neighbors' snapshots and prev)
        print(f"[Round {round_num}] Phase 4: Aggregation with snapshots...")
        for client in clients:
            client.aggregation_phase(
                round_num=round_num,
                similarity_matrix=similarity_matrix,
                verbose=False
            )

        # Phase 5: All clients update prev_backbone (after all aggregated)
        print(f"[Round {round_num}] Phase 5: Update previous backbones...")
        for client in clients:
            client.update_prev_backbone_phase()

        # Evaluate all clients
        round_results = evaluate_clients(clients, round_num)
        results_history.append(round_results)

    # Analyze clustering patterns
    output_dir = os.path.join(
        config['general']['result_folder'],
        config['general']['result_sub_folder'],
        config['general']['title']
    )
    analyze_clustering_patterns(clients, output_dir)

    # Save results
    save_results(results_history, config, output_dir)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED")
    print("="*80)
    print(f"\nFinal Results:")
    final_results = results_history[-1]
    print(f"  Animals Average: Loss={final_results['animals_avg']['test_loss']:.4f}, "
          f"Acc={final_results['animals_avg']['test_acc']:.4f}")
    print(f"  Objects Average: Loss={final_results['objects_avg']['test_loss']:.4f}, "
          f"Acc={final_results['objects_avg']['test_acc']:.4f}")
    print(f"  Overall Average: Loss={final_results['overall_avg']['test_loss']:.4f}, "
          f"Acc={final_results['overall_avg']['test_acc']:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Decentralized Federated Multi-Task Learning with Dynamic Soft Clustering'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml',
        help='Path to experiment configuration file'
    )

    args = parser.parse_args()

    # Run experiment
    run_experiment(args.config)


if __name__ == '__main__':
    main()
