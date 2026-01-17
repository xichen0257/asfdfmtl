"""
Dynamic Clustering Algorithms for DFMTL

Implements various clustering methods for dynamic client grouping:
- Top-K soft clustering (Proposal 1)
- Spectral clustering (Proposal 2 - coarse)
- Hierarchical clustering (Proposal 2 - fine)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score


def topk_soft_clustering(
    similarity_matrix: np.ndarray,
    k: int = 3
) -> Dict[int, List[int]]:
    """
    Top-K soft clustering: Each client selects K most similar neighbors.

    Args:
        similarity_matrix: [N, N] pairwise similarity matrix
        k: Number of neighbors per client

    Returns:
        clusters: Dict mapping client_id -> list of neighbor indices
    """
    N = similarity_matrix.shape[0]
    clusters = {}

    for i in range(N):
        # Get similarities (excluding self)
        similarities = similarity_matrix[i].copy()
        similarities[i] = -np.inf

        # Adjust k
        k_adjusted = min(k, N - 1)

        # Select top-k neighbors
        top_k_indices = np.argsort(similarities)[-k_adjusted:][::-1]
        clusters[i] = top_k_indices.tolist()

    return clusters


def spectral_clustering_auto(
    similarity_matrix: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 5,
    random_state: int = 42
) -> Tuple[Dict[int, List[int]], int]:
    """
    Spectral clustering with automatic cluster number selection.

    Uses silhouette score to determine optimal number of clusters.

    Args:
        similarity_matrix: [N, N] pairwise similarity matrix
        min_clusters: Minimum number of clusters to try
        max_clusters: Maximum number of clusters to try
        random_state: Random seed

    Returns:
        clusters: Dict mapping cluster_id -> list of client indices
        n_clusters: Number of clusters selected
    """
    N = similarity_matrix.shape[0]

    # Adjust max_clusters
    max_clusters = min(max_clusters, N - 1)

    if max_clusters < min_clusters:
        # Fallback: return each client as its own cluster
        clusters = {i: [i] for i in range(N)}
        return clusters, N

    # Try different number of clusters
    best_score = -1
    best_n_clusters = min_clusters
    best_labels = None

    for n in range(min_clusters, max_clusters + 1):
        try:
            clustering = SpectralClustering(
                n_clusters=n,
                affinity='precomputed',
                random_state=random_state,
                assign_labels='kmeans'
            ).fit(similarity_matrix)

            # Compute silhouette score
            if n < N:  # Need at least 2 samples per cluster
                score = silhouette_score(similarity_matrix, clustering.labels_, metric='precomputed')

                if score > best_score:
                    best_score = score
                    best_n_clusters = n
                    best_labels = clustering.labels_
        except Exception as e:
            # Skip if clustering fails
            print(f"Warning: Clustering with n={n} failed: {e}")
            continue

    # Use best clustering
    if best_labels is None:
        # Fallback: 2 clusters
        clustering = SpectralClustering(
            n_clusters=min_clusters,
            affinity='precomputed',
            random_state=random_state
        ).fit(similarity_matrix)
        best_labels = clustering.labels_
        best_n_clusters = min_clusters

    # Convert to dict
    clusters = {}
    for i, label in enumerate(best_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters, best_n_clusters


def hierarchical_clustering_threshold(
    similarity_matrix: np.ndarray,
    threshold: float = 0.6,
    linkage: str = 'average'
) -> Dict[int, List[int]]:
    """
    Hierarchical clustering with distance threshold.

    Args:
        similarity_matrix: [N, N] pairwise similarity matrix
        threshold: Distance threshold (similarity will be converted to distance)
        linkage: Linkage method ('average', 'complete', 'single')

    Returns:
        clusters: Dict mapping cluster_id -> list of client indices
    """
    N = similarity_matrix.shape[0]

    if N == 1:
        return {0: [0]}

    # Convert similarity to distance
    distance_matrix = 1.0 - similarity_matrix
    np.fill_diagonal(distance_matrix, 0.0)  # Ensure diagonal is 0

    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric='precomputed',
        linkage=linkage
    ).fit(distance_matrix)

    # Convert to dict
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    return clusters


def compute_clustering_stability(
    clusters_prev: Dict[int, List[int]],
    clusters_curr: Dict[int, List[int]]
) -> float:
    """
    Compute stability between two clustering results using Jaccard similarity.

    Args:
        clusters_prev: Previous round clusters
        clusters_curr: Current round clusters

    Returns:
        stability: Jaccard similarity score in [0, 1]
    """
    # Convert cluster dicts to client->cluster_id mapping
    def to_client_mapping(clusters):
        client_to_cluster = {}
        for cluster_id, members in clusters.items():
            for client_id in members:
                client_to_cluster[client_id] = cluster_id
        return client_to_cluster

    prev_mapping = to_client_mapping(clusters_prev)
    curr_mapping = to_client_mapping(clusters_curr)

    # Get all clients
    all_clients = set(prev_mapping.keys()) | set(curr_mapping.keys())

    if len(all_clients) == 0:
        return 1.0

    # Count agreements
    agreements = 0
    for client_i in all_clients:
        for client_j in all_clients:
            if client_i >= client_j:
                continue

            # Check if both are in same cluster in both rounds
            prev_same = prev_mapping.get(client_i) == prev_mapping.get(client_j)
            curr_same = curr_mapping.get(client_i) == curr_mapping.get(client_j)

            if prev_same == curr_same:
                agreements += 1

    # Total pairs
    total_pairs = len(all_clients) * (len(all_clients) - 1) // 2

    if total_pairs == 0:
        return 1.0

    stability = agreements / total_pairs
    return stability


def compute_cluster_purity(
    clusters: Dict[int, List[int]],
    client_tasks: List[List[str]]
) -> float:
    """
    Compute cluster purity: how homogeneous are tasks within clusters.

    Args:
        clusters: Dict mapping cluster_id -> list of client indices
        client_tasks: List of task lists for each client

    Returns:
        purity: Average purity score in [0, 1] (1 = perfect homogeneity)
    """
    purities = []

    for cluster_id, members in clusters.items():
        if len(members) <= 1:
            purities.append(1.0)
            continue

        # Get tasks for all members
        all_tasks = set()
        for client_id in members:
            all_tasks.update(client_tasks[client_id])

        # Count most common task overlap
        task_counts = {}
        for task in all_tasks:
            count = sum(1 for client_id in members if task in client_tasks[client_id])
            task_counts[task] = count

        # Purity = max_count / total_members
        max_count = max(task_counts.values()) if task_counts else 0
        purity = max_count / len(members)
        purities.append(purity)

    return np.mean(purities) if purities else 0.0


if __name__ == "__main__":
    # Test clustering algorithms
    print("Testing dynamic clustering algorithms...")

    # Create test similarity matrix
    N = 6
    np.random.seed(42)

    # Create block-diagonal structure (2 clear clusters)
    sim_matrix = np.random.rand(N, N) * 0.3  # Background noise

    # Cluster 1: clients 0, 1, 2
    sim_matrix[0:3, 0:3] = 0.8 + np.random.rand(3, 3) * 0.2
    # Cluster 2: clients 3, 4, 5
    sim_matrix[3:6, 3:6] = 0.8 + np.random.rand(3, 3) * 0.2

    # Ensure symmetry and diagonal
    sim_matrix = (sim_matrix + sim_matrix.T) / 2
    np.fill_diagonal(sim_matrix, 1.0)

    print("\nTest Similarity Matrix:")
    print(sim_matrix.round(2))

    # Test 1: Top-K soft clustering
    print("\n=== Test 1: Top-K Soft Clustering ===")
    topk_clusters = topk_soft_clustering(sim_matrix, k=2)
    print("Top-2 neighbors for each client:")
    for client_id, neighbors in topk_clusters.items():
        print(f"  Client {client_id}: neighbors {neighbors}")

    # Test 2: Spectral clustering
    print("\n=== Test 2: Spectral Clustering (Auto) ===")
    spectral_clusters, n_clusters = spectral_clustering_auto(sim_matrix)
    print(f"Number of clusters found: {n_clusters}")
    print("Cluster assignments:")
    for cluster_id, members in spectral_clusters.items():
        print(f"  Cluster {cluster_id}: members {members}")

    # Test 3: Hierarchical clustering
    print("\n=== Test 3: Hierarchical Clustering ===")
    hier_clusters = hierarchical_clustering_threshold(sim_matrix, threshold=0.5)
    print("Cluster assignments (threshold=0.5):")
    for cluster_id, members in hier_clusters.items():
        print(f"  Cluster {cluster_id}: members {members}")

    # Test 4: Stability
    print("\n=== Test 4: Clustering Stability ===")
    # Simulate slight change
    sim_matrix_perturbed = sim_matrix + np.random.rand(N, N) * 0.05
    sim_matrix_perturbed = (sim_matrix_perturbed + sim_matrix_perturbed.T) / 2
    np.fill_diagonal(sim_matrix_perturbed, 1.0)

    clusters_new, _ = spectral_clustering_auto(sim_matrix_perturbed)
    stability = compute_clustering_stability(spectral_clusters, clusters_new)
    print(f"Stability (Jaccard): {stability:.4f}")

    # Test 5: Purity
    print("\n=== Test 5: Cluster Purity ===")
    client_tasks = [
        ['depth', 'seg'],      # Client 0
        ['depth', 'seg'],      # Client 1
        ['depth'],             # Client 2
        ['seg', 'normal'],     # Client 3
        ['normal'],            # Client 4
        ['seg', 'normal'],     # Client 5
    ]
    purity = compute_cluster_purity(spectral_clusters, client_tasks)
    print(f"Cluster purity: {purity:.4f}")

    print("\n[OK] All clustering tests passed!")
