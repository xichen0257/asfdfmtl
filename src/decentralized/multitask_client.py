"""
Multi-Task Soft Clustering Client for Decentralized Federated Learning

Extends decentralized FL to support multi-task learning with soft clustering:
- Each client works on multiple tasks with different weights
- Task-aware similarity computation (task overlap + gradient similarity)
- Task-selective aggregation (backbone with all, heads with task-specific neighbors)
"""

import torch
import numpy as np
from typing import List, Dict, Optional
import copy

from decentralized.task_similarity import (
    compute_model_gradient,
    compute_pairwise_multitask_similarity,
    select_top_k_neighbors,
    log_multitask_clustering_info
)
from decentralized.multitask_aggregation import (
    extract_model_parameters,
    load_model_parameters,
    aggregate_multitask_model,
    compute_aggregation_stats
)
from client_handling.hca import conflict_averse


class MultiTaskClient:
    """
    Multi-task client for decentralized federated learning with soft clustering.

    Key features:
    - Works on multiple tasks simultaneously with task weights
    - Computes task-aware similarity with neighbors
    - Aggregates backbone with all neighbors (shared features)
    - Aggregates task heads only with task-specific neighbors
    """

    def __init__(
        self,
        client_id: int,
        model,  # MultiTaskModel instance
        train_loader,
        val_loader,
        task_weights: Dict[str, float],
        all_clients: Optional[List['MultiTaskClient']] = None,
        n_neighbors: int = 3,
        alpha: float = 0.5,  # Balance between task overlap and gradient similarity
        aggregate_heads: bool = True,
        task_threshold: float = 0.1,
        self_weight_backbone: float = 0.0,
        self_weight_heads: float = 0.0,
        aggregation_method: str = 'weighted',  # 'weighted' or 'hca'
        hca_alpha: float = 0.7  # HCA hyperparameter (conflict-averse strength)
    ):
        """
        Initialize multi-task client.

        Args:
            client_id: Unique client identifier
            model: MultiTaskModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            task_weights: Dict of task weights, e.g., {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0}
            all_clients: List of all clients in the network
            n_neighbors: Number of neighbors to select
            alpha: Balance between task overlap (alpha) and gradient similarity (1-alpha)
            aggregate_heads: Whether to aggregate task heads
            task_threshold: Minimum task weight for head aggregation
            self_weight_backbone: Self-preservation weight for backbone
            self_weight_heads: Self-preservation weight for heads
            aggregation_method: Aggregation method ('weighted' or 'hca')
            hca_alpha: HCA hyperparameter for conflict-averse aggregation
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_weights = task_weights
        self.all_clients = all_clients if all_clients is not None else []
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.aggregate_heads = aggregate_heads
        self.task_threshold = task_threshold
        self.self_weight_backbone = self_weight_backbone
        self.self_weight_heads = self_weight_heads
        self.aggregation_method = aggregation_method
        self.hca_alpha = hca_alpha

        # Training state
        self.prev_params = None  # For gradient computation
        self.prev_params_backbone = None  # For HCA aggregation
        self.current_round = 0

        # Clustering history
        self.neighbor_history = []
        self.similarity_history = []

    def local_training(self, num_epochs: int = 1, verbose: bool = False):
        """
        Perform local training on multi-task model.

        Args:
            num_epochs: Number of local training epochs
            verbose: Whether to print training progress

        Returns:
            Training results (list of dicts with metrics per epoch)
        """
        # Store parameters before training for gradient computation
        self.prev_params = extract_model_parameters(self.model)

        # Train model
        results = self.model.train_model(
            train_loader=self.train_loader,
            num_epochs=num_epochs,
            verbose=verbose
        )

        return results

    def compute_similarity_with_neighbors(
        self,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute task-aware similarity with all other clients.

        Args:
            similarity_matrix: Optional pre-computed similarity matrix
                             If None, computes from scratch

        Returns:
            N x N similarity matrix
        """
        if similarity_matrix is not None:
            return similarity_matrix

        n_clients = len(self.all_clients)

        # Collect task weights and gradients from all clients
        client_task_weights = []
        client_gradients = []

        for client in self.all_clients:
            client_task_weights.append(client.task_weights)

            # Compute gradient (parameter change)
            if client.prev_params is not None:
                curr_params = extract_model_parameters(client.model)
                gradient = compute_model_gradient(
                    curr_params['backbone'],
                    client.prev_params['backbone']
                )
            else:
                # First round: use zero gradient
                gradient = torch.zeros(100)  # Dummy gradient

            client_gradients.append(gradient)

        # Compute pairwise multi-task similarity
        similarity_matrix = compute_pairwise_multitask_similarity(
            client_task_weights,
            client_gradients,
            alpha=self.alpha
        )

        return similarity_matrix

    def select_neighbors(
        self,
        similarity_matrix: np.ndarray
    ) -> List[int]:
        """
        Select top-K most similar neighbors.

        Args:
            similarity_matrix: N x N pairwise similarity matrix

        Returns:
            List of K neighbor indices
        """
        neighbor_indices = select_top_k_neighbors(
            self.client_id,
            similarity_matrix,
            k=self.n_neighbors
        )

        # Store in history
        self.neighbor_history.append(neighbor_indices)
        neighbor_similarities = [
            similarity_matrix[self.client_id, idx] for idx in neighbor_indices
        ]
        self.similarity_history.append(neighbor_similarities)

        return neighbor_indices

    def aggregate_with_neighbors(
        self,
        neighbor_indices: List[int],
        neighbor_similarities: List[float],
        verbose: bool = False
    ):
        """
        Aggregate model with selected neighbors.

        Args:
            neighbor_indices: Indices of selected neighbors
            neighbor_similarities: Similarity scores for each neighbor
            verbose: Whether to print aggregation info
        """
        if len(neighbor_indices) == 0:
            if verbose:
                print(f"[Client {self.client_id}] No neighbors to aggregate with")
            return

        # Extract own model parameters
        own_params = extract_model_parameters(self.model)

        # Extract neighbor parameters
        neighbor_params = []
        neighbor_task_weights = []

        for idx in neighbor_indices:
            neighbor = self.all_clients[idx]
            neighbor_params.append(extract_model_parameters(neighbor.model))
            neighbor_task_weights.append(neighbor.task_weights)

        # Compute aggregation stats (for logging)
        if verbose:
            stats = compute_aggregation_stats(
                self.task_weights,
                neighbor_task_weights,
                neighbor_similarities
            )
            print(f"\n[Client {self.client_id}] Aggregation Stats:")
            print(f"  Neighbors: {neighbor_indices}")
            print(f"  Avg similarity: {stats['avg_similarity']:.4f}")
            print(f"  Avg task overlap: {stats['avg_task_overlap']:.4f}")

        # Choose aggregation method
        if self.aggregation_method == 'hca':
            # HCA (Hyper Conflict-Averse) aggregation for backbone only
            if verbose:
                print(f"[Client {self.client_id}] Using HCA aggregation (alpha={self.hca_alpha})")

            # Prepare current backbones (self + neighbors)
            curr_backbones = [own_params['backbone']] + [p['backbone'] for p in neighbor_params]

            # Prepare previous backbones
            if self.prev_params_backbone is not None:
                # Use saved previous backbones
                prev_backbones = [self.prev_params_backbone]
                for idx in neighbor_indices:
                    neighbor = self.all_clients[idx]
                    if neighbor.prev_params_backbone is not None:
                        prev_backbones.append(neighbor.prev_params_backbone)
                    else:
                        # Fallback: use current as previous (first round)
                        prev_backbones.append(neighbor_params[neighbor_indices.index(idx)]['backbone'])

                # Perform HCA aggregation
                try:
                    aggregated_backbones = conflict_averse(
                        curr_backbones,
                        prev_backbones,
                        self.hca_alpha
                    )

                    # Update own backbone with HCA result
                    aggregated_params = copy.deepcopy(own_params)
                    aggregated_params['backbone'] = aggregated_backbones[0]

                    if verbose:
                        print(f"[Client {self.client_id}] HCA aggregation successful")

                except Exception as e:
                    print(f"[Client {self.client_id}] HCA aggregation failed: {e}")
                    print(f"  Falling back to weighted aggregation")
                    # Fallback to weighted aggregation
                    aggregated_params = aggregate_multitask_model(
                        own_params,
                        neighbor_params,
                        neighbor_similarities,
                        neighbor_task_weights,
                        self.task_weights,
                        aggregate_heads=self.aggregate_heads,
                        task_threshold=self.task_threshold,
                        self_weight_backbone=self.self_weight_backbone,
                        self_weight_heads=self.self_weight_heads
                    )
            else:
                # First round: no previous params, use weighted aggregation
                if verbose:
                    print(f"[Client {self.client_id}] First round, using weighted aggregation")
                aggregated_params = aggregate_multitask_model(
                    own_params,
                    neighbor_params,
                    neighbor_similarities,
                    neighbor_task_weights,
                    self.task_weights,
                    aggregate_heads=self.aggregate_heads,
                    task_threshold=self.task_threshold,
                    self_weight_backbone=self.self_weight_backbone,
                    self_weight_heads=self.self_weight_heads
                )

        else:
            # Default weighted aggregation
            aggregated_params = aggregate_multitask_model(
                own_params,
                neighbor_params,
                neighbor_similarities,
                neighbor_task_weights,
                self.task_weights,
                aggregate_heads=self.aggregate_heads,
                task_threshold=self.task_threshold,
                self_weight_backbone=self.self_weight_backbone,
                self_weight_heads=self.self_weight_heads
            )

        # Load aggregated parameters back into model
        load_model_parameters(self.model, aggregated_params)

    def evaluate(self, data_loader=None):
        """
        Evaluate model on validation/test data.

        Args:
            data_loader: DataLoader to evaluate on (defaults to val_loader)

        Returns:
            Dict of evaluation metrics
        """
        if data_loader is None:
            data_loader = self.val_loader

        self.model.eval()

        total_loss = 0.0
        # Initialize task losses based on dataset type
        if hasattr(self.model, 'is_pascal_context') and self.model.is_pascal_context:
            task_losses = {'segmentation': 0.0, 'human_parts': 0.0, 'edge': 0.0}
        else:
            task_losses = {'depth': 0.0, 'segmentation': 0.0, 'normal': 0.0}
        num_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.model.device)

                # Build targets dict based on dataset type
                if hasattr(self.model, 'is_pascal_context') and self.model.is_pascal_context:
                    targets = {
                        'segmentation': batch['segmentation'].to(self.model.device),
                        'human_parts': batch['human_parts'].to(self.model.device),
                        'edge': batch['edge'].to(self.model.device)
                    }
                    task_list = ['segmentation', 'human_parts', 'edge']
                else:
                    targets = {
                        'depth': batch['depth'].to(self.model.device),
                        'segmentation': batch['segmentation'].to(self.model.device),
                        'normal': batch['normal'].to(self.model.device)
                    }
                    task_list = ['depth', 'segmentation', 'normal']

                # Forward pass
                predictions = self.model(images, return_all=True)

                # Compute loss
                loss, loss_dict = self.model.compute_loss(predictions, targets)

                total_loss += loss.item()
                for task in task_list:
                    if task in loss_dict:
                        task_losses[task] += loss_dict[task]

                num_batches += 1

        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_task_losses = {
            task: loss / num_batches if num_batches > 0 else 0.0
            for task, loss in task_losses.items()
        }

        return {
            'total_loss': avg_loss,
            **avg_task_losses
        }

    def run_round(
        self,
        round_num: int,
        similarity_matrix: Optional[np.ndarray] = None,
        num_local_epochs: int = 1,
        verbose: bool = False
    ) -> Dict:
        """
        Execute a complete federated learning round.

        Steps:
        1. Local training
        2. Compute similarity with neighbors
        3. Select top-K neighbors
        4. Aggregate with neighbors
        5. Evaluate

        Args:
            round_num: Current round number
            similarity_matrix: Optional pre-computed similarity matrix
            num_local_epochs: Number of local training epochs
            verbose: Whether to print detailed info

        Returns:
            Dict with round results
        """
        self.current_round = round_num

        if verbose:
            print(f"\n{'='*60}")
            print(f"[Client {self.client_id}] Round {round_num}")
            print(f"Task weights: {self.task_weights}")
            print(f"{'='*60}")

        # Step 1: Local training
        if verbose:
            print(f"\n[Step 1] Local Training ({num_local_epochs} epochs)")

        train_results = self.local_training(num_local_epochs, verbose=verbose)

        # Step 2: Compute similarity
        if verbose:
            print(f"\n[Step 2] Computing Similarity")

        similarity_matrix = self.compute_similarity_with_neighbors(similarity_matrix)

        # Step 3: Select neighbors
        if verbose:
            print(f"\n[Step 3] Selecting Neighbors")

        neighbor_indices = self.select_neighbors(similarity_matrix)
        neighbor_similarities = [
            similarity_matrix[self.client_id, idx] for idx in neighbor_indices
        ]

        if verbose:
            neighbor_task_weights = [self.all_clients[idx].task_weights for idx in neighbor_indices]
            log_multitask_clustering_info(
                round_num,
                self.client_id,
                self.task_weights,
                neighbor_indices,
                neighbor_task_weights,
                similarity_matrix
            )

        # Step 4: Aggregate
        if verbose:
            print(f"\n[Step 4] Aggregating with Neighbors")

        self.aggregate_with_neighbors(
            neighbor_indices,
            neighbor_similarities,
            verbose=verbose
        )

        # Step 5: Evaluate
        if verbose:
            print(f"\n[Step 5] Evaluation")

        eval_results = self.evaluate()

        if verbose:
            print(f"  Total loss: {eval_results['total_loss']:.4f}")
            print(f"  Depth loss: {eval_results['depth']:.4f}")
            print(f"  Segmentation loss: {eval_results['segmentation']:.4f}")
            print(f"  Normal loss: {eval_results['normal']:.4f}")

        # Compile round results
        round_results = {
            'round': round_num,
            'client_id': self.client_id,
            'neighbors': neighbor_indices,
            'similarities': neighbor_similarities,
            'train_results': train_results,
            'eval_results': eval_results
        }

        return round_results


def create_multitask_clients(
    data_manager,
    task_weights_per_client: List[Dict[str, float]],
    num_seg_classes: int = 13,
    n_neighbors: int = 3,
    alpha: float = 0.5,
    aggregate_heads: bool = True,
    aggregation_method: str = 'weighted',
    hca_alpha: float = 0.7,
    device: str = 'cuda',
    num_human_parts_classes: int = 6,  # For Pascal Context
    gradient_clip_norm: float = None,  # Optional gradient clipping
    lr: float = 0.001  # Learning rate
) -> List[MultiTaskClient]:
    """
    Create a network of multi-task clients.

    Args:
        data_manager: DMNYUDepthV2 instance with data loaders
        task_weights_per_client: List of task weight dicts
        num_seg_classes: Number of segmentation classes
        n_neighbors: Number of neighbors per client
        alpha: Balance parameter for similarity
        aggregate_heads: Whether to aggregate task heads
        aggregation_method: Aggregation method ('weighted' or 'hca')
        hca_alpha: HCA hyperparameter
        device: Device to use ('cuda' or 'cpu')
        gradient_clip_norm: Max norm for gradient clipping (None = no clipping)
        lr: Learning rate for optimizer. Default: 0.001

    Returns:
        List of MultiTaskClient instances
    """
    from models.multitask_dense_prediction_model import MultiTaskModel

    num_clients = data_manager.num_clients
    clients = []

    # Create clients
    for i in range(num_clients):
        # Create model for this client
        model = MultiTaskModel(
            task_weights=task_weights_per_client[i],
            num_seg_classes=num_seg_classes,
            out_size=(288, 384),
            pretrained=True,
            num_human_parts_classes=num_human_parts_classes,
            gradient_clip_norm=gradient_clip_norm,
            lr=lr
        )
        model.to(device)

        # Get data loaders
        train_loader = data_manager.get_train_loader(i)
        val_loader = data_manager.get_val_loader(i)

        # Create client
        client = MultiTaskClient(
            client_id=i,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            task_weights=task_weights_per_client[i],
            all_clients=None,  # Will set later
            n_neighbors=n_neighbors,
            alpha=alpha,
            aggregate_heads=aggregate_heads,
            aggregation_method=aggregation_method,
            hca_alpha=hca_alpha
        )

        clients.append(client)

    # Set all_clients reference for each client
    for client in clients:
        client.all_clients = clients

    return clients


if __name__ == "__main__":
    """Test multi-task client"""
    print("=" * 60)
    print("Multi-Task Client Test")
    print("=" * 60)
    print("\nNote: Full test requires NYU Depth V2 dataset.")
    print("This is a basic structure test.")
    print("\n[OK] MultiTaskClient class defined successfully!")
    print("[OK] Helper functions available:")
    print("  - create_multitask_clients()")
    print("  - client.run_round()")
    print("  - client.local_training()")
    print("  - client.aggregate_with_neighbors()")
