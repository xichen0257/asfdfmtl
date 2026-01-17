"""
Task Similarity Computation Module for Dynamic Soft Clustering

This module computes task similarity between clients based on their model gradients.
The similarity metric is used to dynamically form soft clusters (N=3) in a decentralized manner.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_gradient_similarity(gradient_a: torch.Tensor, gradient_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two gradient tensors.

    Args:
        gradient_a: Flattened gradient tensor from client A
        gradient_b: Flattened gradient tensor from client B

    Returns:
        Cosine similarity score in range [-1, 1]
        Higher values indicate more similar tasks
    """
    # Ensure gradients are flattened
    grad_a_flat = gradient_a.flatten()
    grad_b_flat = gradient_b.flatten()

    # Compute cosine similarity
    dot_product = torch.dot(grad_a_flat, grad_b_flat)
    norm_a = torch.norm(grad_a_flat)
    norm_b = torch.norm(grad_b_flat)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = (dot_product / (norm_a * norm_b)).item()
    return similarity


def compute_pairwise_similarity(client_gradients: List[torch.Tensor]) -> np.ndarray:
    """
    Compute pairwise task similarity matrix for all clients.

    Args:
        client_gradients: List of flattened gradient tensors, one per client

    Returns:
        N x N similarity matrix where entry (i,j) is similarity between client i and j
    """
    n_clients = len(client_gradients)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i, n_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
            else:
                sim = compute_gradient_similarity(client_gradients[i], client_gradients[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix

    return similarity_matrix


def select_top_k_neighbors(
    client_idx: int,
    similarity_matrix: np.ndarray,
    k: int = 3
) -> List[int]:
    """
    Select top-k most similar neighbors for a given client (soft clustering).

    Args:
        client_idx: Index of the target client
        similarity_matrix: N x N pairwise similarity matrix
        k: Number of neighbors to select (default: 3 for soft clustering)

    Returns:
        List of k client indices representing the most similar neighbors
        (excluding the client itself)
    """
    n_clients = similarity_matrix.shape[0]

    # Get similarity scores for this client (excluding self)
    similarities = similarity_matrix[client_idx].copy()
    similarities[client_idx] = -np.inf  # Exclude self

    # Select top-k most similar clients
    if k >= n_clients:
        k = n_clients - 1  # Can't select more than N-1 neighbors

    top_k_indices = np.argsort(similarities)[-k:][::-1]  # Descending order

    return top_k_indices.tolist()


def compute_model_gradient(
    curr_params: Dict[str, torch.Tensor],
    prev_params: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Compute model gradient (parameter change) between current and previous state.

    Args:
        curr_params: Current model parameters as state_dict (already filtered to backbone)
        prev_params: Previous model parameters as state_dict (already filtered to backbone)

    Returns:
        Flattened gradient tensor
    """
    gradients = []

    for key in curr_params.keys():
        delta = curr_params[key] - prev_params[key]
        gradients.append(delta.flatten())

    # Concatenate all gradients into single tensor
    if len(gradients) == 0:
        return torch.tensor([0.0])

    return torch.cat(gradients)


def adaptive_cluster_selection(
    client_idx: int,
    similarity_matrix: np.ndarray,
    similarity_threshold: float = 0.3,
    min_neighbors: int = 1,
    max_neighbors: int = 3
) -> List[int]:
    """
    Adaptive soft clustering: select neighbors based on similarity threshold.

    Unlike fixed N=3, this method selects neighbors that exceed a similarity threshold,
    allowing cluster size to vary between min_neighbors and max_neighbors.

    Args:
        client_idx: Index of the target client
        similarity_matrix: N x N pairwise similarity matrix
        similarity_threshold: Minimum similarity score to include a neighbor
        min_neighbors: Minimum number of neighbors (even if below threshold)
        max_neighbors: Maximum number of neighbors

    Returns:
        List of selected neighbor indices
    """
    n_clients = similarity_matrix.shape[0]
    similarities = similarity_matrix[client_idx].copy()
    similarities[client_idx] = -np.inf  # Exclude self

    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]  # Descending

    # Select neighbors above threshold
    selected = []
    for idx in sorted_indices:
        if similarities[idx] >= similarity_threshold and len(selected) < max_neighbors:
            selected.append(idx)

    # Ensure minimum number of neighbors
    if len(selected) < min_neighbors:
        selected = sorted_indices[:min_neighbors].tolist()

    return selected


def compute_parameter_similarity(params_a: torch.Tensor, params_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two parameter tensors.

    This computes similarity based on actual parameter values (not changes/gradients),
    which may better capture task similarity when clients have diverged.

    Args:
        params_a: Flattened parameter tensor from client A
        params_b: Flattened parameter tensor from client B

    Returns:
        Cosine similarity score in range [-1, 1]
        Higher values indicate more similar model states
    """
    # Ensure parameters are flattened
    params_a_flat = params_a.flatten()
    params_b_flat = params_b.flatten()

    # Compute cosine similarity
    dot_product = torch.dot(params_a_flat, params_b_flat)
    norm_a = torch.norm(params_a_flat)
    norm_b = torch.norm(params_b_flat)

    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = (dot_product / (norm_a * norm_b)).item()
    return similarity


def compute_pairwise_parameter_similarity(client_params: List[torch.Tensor]) -> np.ndarray:
    """
    Compute pairwise parameter similarity matrix for all clients.

    Unlike gradient-based similarity (which uses parameter changes), this uses
    the actual parameter values, which may provide better task differentiation
    after several rounds of training.

    Args:
        client_params: List of flattened parameter tensors, one per client

    Returns:
        N x N similarity matrix where entry (i,j) is similarity between client i and j
    """
    n_clients = len(client_params)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i, n_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
            else:
                sim = compute_parameter_similarity(client_params[i], client_params[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Symmetric matrix

    return similarity_matrix


def compute_cross_loss_similarity(clients: List, num_eval_batches: int = 10) -> np.ndarray:
    """
    Compute pairwise similarity based on cross-validation loss.

    For each pair of clients (i, j):
    - Evaluate client i's model on client j's validation data
    - Lower loss indicates higher task similarity
    - Converts losses to similarity scores in [0, 1]

    This method directly measures task compatibility through actual performance,
    avoiding issues with initialization-based methods (gradient/parameter similarity).

    Args:
        clients: List of DecentralizedClient instances
        num_eval_batches: Number of validation batches to use for evaluation

    Returns:
        N x N similarity matrix where entry (i,j) represents how well
        client i's model performs on client j's data
    """
    n_clients = len(clients)
    loss_matrix = np.zeros((n_clients, n_clients))

    print(f"\n[Cross-Loss Similarity] Evaluating {n_clients} clients...")

    # Compute cross-validation losses
    for i, client_i in enumerate(clients):
        for j, client_j in enumerate(clients):
            if i == j:
                # Self-evaluation (will be set to 1.0 in similarity matrix)
                loss_matrix[i, j] = 0.0
            else:
                # Evaluate client i's model on client j's validation data
                # Support both single-task clients (val_loader in model) and multi-task clients (val_loader in client)
                if hasattr(client_j.model, 'val_loader'):
                    val_loader = client_j.model.val_loader  # Single-task client
                elif hasattr(client_j, 'val_loader'):
                    val_loader = client_j.val_loader  # Multi-task client
                else:
                    raise AttributeError(f"Client {j} has no val_loader attribute")

                model = client_i.model

                # For multi-task models, use the model's compute_loss method directly
                # For single-task models, check output dimensions
                if hasattr(model, 'compute_loss'):
                    # Multi-task model - use its loss computation
                    avg_loss = _compute_multitask_cross_loss(model, val_loader, num_eval_batches)
                else:
                    # Single-task model - check output dimensions
                    num_classes_i = model.head[-1].out_features
                    num_classes_j = client_j.model.head[-1].out_features

                    if num_classes_i != num_classes_j:
                        # Incompatible output dimensions - use filtered evaluation
                        avg_loss = _compute_filtered_cross_loss(
                            model, val_loader, num_classes_i, num_eval_batches
                        )
                    else:
                        # Compatible dimensions - standard evaluation
                        avg_loss = _compute_cross_loss(model, val_loader, num_eval_batches)

                loss_matrix[i, j] = avg_loss

    print(f"  Loss matrix computed. Average cross-loss: {np.mean(loss_matrix[loss_matrix > 0]):.4f}")

    # Convert losses to similarities
    similarity_matrix = _loss_to_similarity(loss_matrix)

    return similarity_matrix


def _compute_multitask_cross_loss(model, val_loader, num_eval_batches: int) -> float:
    """
    Compute average loss for multi-task models on validation data.

    Args:
        model: Multi-task model with compute_loss method
        val_loader: Validation data loader
        num_eval_batches: Number of batches to evaluate

    Returns:
        Average total loss across batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_eval_batches:
                break

            # Unpack batch data
            images = batch['image'].to(model.device)
            targets = {}
            for task in ['segmentation', 'human_parts', 'edge']:
                if task in batch:
                    targets[task] = batch[task].to(model.device)

            # Forward pass
            outputs = model(images)

            # Compute loss using model's compute_loss method
            # Returns (total_loss, loss_dict) tuple
            loss, loss_dict = model.compute_loss(outputs, targets)

            # Handle case where loss might be a float (no valid tasks)
            if isinstance(loss, torch.Tensor):
                total_loss += loss.item()
            else:
                total_loss += loss

            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 1.0  # Return high loss if no batches


def _compute_cross_loss(model, val_loader, num_eval_batches: int) -> float:
    """
    Compute average loss on validation data (for single-task models).

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        num_eval_batches: Number of batches to evaluate

    Returns:
        Average loss across batches
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx >= num_eval_batches:
                break

            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            outputs = model(inputs)
            loss = model.criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def _compute_filtered_cross_loss(model, val_loader, max_classes: int, num_eval_batches: int) -> float:
    """
    Compute cross-loss with label filtering for heterogeneous tasks.

    When evaluating a model with fewer output classes on data with more classes,
    we filter out samples whose labels exceed the model's output dimension.

    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        max_classes: Maximum number of classes the model supports
        num_eval_batches: Number of batches to evaluate

    Returns:
        Average loss on filtered samples
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx >= num_eval_batches:
                break

            inputs = inputs.to(model.device)
            targets = targets.to(model.device)

            # Filter samples with valid labels
            valid_mask = targets < max_classes
            if valid_mask.sum() == 0:
                continue  # Skip batch if no valid samples

            inputs_filtered = inputs[valid_mask]
            targets_filtered = targets[valid_mask]

            outputs = model(inputs_filtered)
            loss = model.criterion(outputs, targets_filtered)

            total_loss += loss.item() * len(targets_filtered)
            num_samples += len(targets_filtered)

    return total_loss / num_samples if num_samples > 0 else 1.5  # Return default high loss if no samples


def _loss_to_similarity(loss_matrix: np.ndarray) -> np.ndarray:
    """
    Convert loss matrix to similarity matrix.

    Lower loss = higher similarity
    Uses inverse transformation: similarity = 1 / (loss + epsilon)
    Then normalizes to [0, 1] range with diagonal = 1.0

    Args:
        loss_matrix: N x N matrix of cross-validation losses

    Returns:
        N x N similarity matrix in [0, 1]
    """
    n_clients = loss_matrix.shape[0]
    epsilon = 0.01

    # Inverse transformation
    similarity_matrix = 1.0 / (loss_matrix + epsilon)

    # Set diagonal to 1.0 (self-similarity)
    for i in range(n_clients):
        similarity_matrix[i, i] = 1.0

    # Normalize off-diagonal elements to [0, 0.99]
    off_diagonal_mask = ~np.eye(n_clients, dtype=bool)
    off_diagonal_values = similarity_matrix[off_diagonal_mask]

    if len(off_diagonal_values) > 0:
        off_diagonal_max = np.max(off_diagonal_values)
        off_diagonal_min = np.min(off_diagonal_values)

        for i in range(n_clients):
            for j in range(n_clients):
                if i != j:
                    normalized = (similarity_matrix[i, j] - off_diagonal_min) / (off_diagonal_max - off_diagonal_min + epsilon)
                    similarity_matrix[i, j] = 0.99 * normalized

    return similarity_matrix


def compute_task_type_similarity(clients: List, same_task_sim: float = 0.9, cross_task_sim: float = 0.1) -> np.ndarray:
    """
    Compute similarity matrix based on task types (prior knowledge).

    This is useful for Round 1 when gradient-based similarity is not informative
    due to all clients starting from the same random initialization.

    Args:
        clients: List of DecentralizedClient instances
        same_task_sim: Similarity score for clients with same task type (default: 0.9)
        cross_task_sim: Similarity score for clients with different task types (default: 0.1)

    Returns:
        N x N similarity matrix based on task types
    """
    n_clients = len(clients)
    similarity_matrix = np.eye(n_clients)  # Diagonal is 1.0 (self-similarity)

    for i in range(n_clients):
        for j in range(i+1, n_clients):
            if clients[i].task_type == clients[j].task_type:
                sim = same_task_sim  # Same task: high similarity
            else:
                sim = cross_task_sim  # Different task: low similarity

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Symmetric

    return similarity_matrix


def log_clustering_info(
    round_num: int,
    client_idx: int,
    selected_neighbors: List[int],
    similarity_matrix: np.ndarray
):
    """
    Log information about the clustering for debugging/analysis.

    Args:
        round_num: Current training round
        client_idx: Index of the client
        selected_neighbors: List of selected neighbor indices
        similarity_matrix: Similarity matrix
    """
    print(f"\n[Round {round_num}] Client {client_idx} Clustering:")
    print(f"  Selected neighbors: {selected_neighbors}")

    for neighbor in selected_neighbors:
        sim_score = similarity_matrix[client_idx, neighbor]
        print(f"    Neighbor {neighbor}: similarity = {sim_score:.4f}")


# ============================================================================
# Multi-Task Soft Clustering Functions
# ============================================================================

def compute_task_overlap(
    task_weights_i: Dict[str, float],
    task_weights_j: Dict[str, float]
) -> float:
    """
    Compute task overlap between two clients based on their task weight vectors.

    Task overlap measures how much the task compositions of two clients overlap.
    Formula: overlap(i, j) = sum_k min(w_i,k, w_j,k)

    Args:
        task_weights_i: Task weights for client i, e.g., {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0}
        task_weights_j: Task weights for client j

    Returns:
        Task overlap score in [0, 1]
        - 1.0 if both clients have identical task compositions
        - 0.0 if clients have completely disjoint tasks

    Examples:
        >>> compute_task_overlap({'depth': 1.0, 'seg': 0.0}, {'depth': 1.0, 'seg': 0.0})
        1.0
        >>> compute_task_overlap({'depth': 0.7, 'seg': 0.3}, {'depth': 0.3, 'seg': 0.7})
        0.6
        >>> compute_task_overlap({'depth': 1.0, 'seg': 0.0}, {'depth': 0.0, 'seg': 1.0})
        0.0
    """
    # Ensure both have the same task keys
    assert set(task_weights_i.keys()) == set(task_weights_j.keys()), \
        "Task weights must have the same task keys"

    overlap = 0.0
    for task in task_weights_i.keys():
        overlap += min(task_weights_i[task], task_weights_j[task])

    return overlap


def compute_multitask_similarity(
    task_weights_i: Dict[str, float],
    task_weights_j: Dict[str, float],
    gradient_sim: float,
    alpha: float = 0.5
) -> float:
    """
    Compute combined similarity for multi-task soft clustering.

    Combines task overlap (structural similarity) with gradient similarity (learned similarity):
        sim(i, j) = alpha * overlap(i, j) + (1 - alpha) * sim_grad(i, j)

    Args:
        task_weights_i: Task weights for client i
        task_weights_j: Task weights for client j
        gradient_sim: Gradient-based similarity (from compute_gradient_similarity)
        alpha: Balance between task overlap and gradient similarity (default: 0.5)
               - alpha = 1.0: only use task overlap (structural)
               - alpha = 0.0: only use gradient similarity (learned)
               - alpha = 0.5: equal balance

    Returns:
        Combined similarity score in [0, 1] (assuming gradient_sim is normalized to [0, 1])
    """
    task_overlap = compute_task_overlap(task_weights_i, task_weights_j)

    # Normalize gradient similarity from [-1, 1] to [0, 1]
    gradient_sim_normalized = (gradient_sim + 1.0) / 2.0

    # Combined similarity
    combined_sim = alpha * task_overlap + (1 - alpha) * gradient_sim_normalized

    return combined_sim


def compute_pairwise_multitask_similarity(
    client_task_weights: List[Dict[str, float]],
    client_gradients: List[torch.Tensor],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Compute pairwise multi-task similarity matrix for all clients.

    For each pair of clients (i, j), combines:
    1. Task overlap: How much their task compositions overlap
    2. Gradient similarity: How similar their learning directions are

    Args:
        client_task_weights: List of task weight dicts, one per client
        client_gradients: List of flattened gradient tensors, one per client
        alpha: Balance parameter for task overlap vs gradient similarity

    Returns:
        N x N similarity matrix where entry (i,j) is the combined similarity
    """
    n_clients = len(client_task_weights)
    assert len(client_gradients) == n_clients, "Mismatch between task weights and gradients"

    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i, n_clients):
            if i == j:
                similarity_matrix[i, j] = 1.0  # Self-similarity is 1
            else:
                # Compute gradient similarity
                grad_sim = compute_gradient_similarity(client_gradients[i], client_gradients[j])

                # Compute combined multi-task similarity
                combined_sim = compute_multitask_similarity(
                    client_task_weights[i],
                    client_task_weights[j],
                    grad_sim,
                    alpha=alpha
                )

                similarity_matrix[i, j] = combined_sim
                similarity_matrix[j, i] = combined_sim  # Symmetric

    return similarity_matrix


def log_multitask_clustering_info(
    round_num: int,
    client_idx: int,
    client_task_weights: Dict[str, float],
    selected_neighbors: List[int],
    neighbor_task_weights: List[Dict[str, float]],
    similarity_matrix: np.ndarray
):
    """
    Log multi-task clustering information for debugging/analysis.

    Args:
        round_num: Current training round
        client_idx: Index of the client
        client_task_weights: Task weights of the client
        selected_neighbors: List of selected neighbor indices
        neighbor_task_weights: Task weights of each selected neighbor
        similarity_matrix: Similarity matrix
    """
    print(f"\n[Round {round_num}] Client {client_idx} Multi-Task Clustering:")
    print(f"  Client task weights: {client_task_weights}")
    print(f"  Selected neighbors: {selected_neighbors}")

    for i, neighbor_idx in enumerate(selected_neighbors):
        sim_score = similarity_matrix[client_idx, neighbor_idx]
        task_overlap = compute_task_overlap(client_task_weights, neighbor_task_weights[i])

        print(f"    Neighbor {neighbor_idx}:")
        print(f"      Task weights: {neighbor_task_weights[i]}")
        print(f"      Task overlap: {task_overlap:.4f}")
        print(f"      Combined similarity: {sim_score:.4f}")
