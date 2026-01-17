"""
Decentralized Client for Dynamic Soft Clustering Federated Multi-Task Learning

This module extends the base Client class to support:
1. Peer-to-peer communication (simulated)
2. Dynamic soft clustering (N=3 neighbors per round)
3. Local HCA aggregation
4. Task similarity-based neighbor selection
"""

import torch
import copy
from typing import List, Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client_handling.client import Client
from client_handling.hca import conflict_averse
from decentralized.task_similarity import (
    compute_model_gradient,
    select_top_k_neighbors,
    log_clustering_info
)
from decentralized.adaptive_aggregation import (
    adaptive_weighted_aggregate,
    adaptive_filter_aggregate,
    progressive_neighbor_aggregate,
    hybrid_adaptive_aggregate
)


class DecentralizedClient(Client):
    """
    Decentralized client that performs dynamic soft clustering and local aggregation.

    Key differences from centralized Client:
    - Maintains list of neighboring clients for P2P communication
    - Computes task similarity with neighbors each round
    - Selects top-K most similar neighbors (K=3 for soft clustering)
    - Performs local HCA aggregation with selected neighbors
    - No communication with central server
    """

    def __init__(
        self,
        c_id: str,
        title: str,
        tasktype: str,
        model,
        seed: int,
        all_clients: Optional[List['DecentralizedClient']] = None,
        n_neighbors: int = 3,
        hca_alpha: float = 0.4,
        aggregation_method: str = 'hca',
        aggregation_start_round: int = 1,
        aggregation_switch_round: Optional[int] = None,
        adaptive_aggregation: bool = False,
        adaptive_method: str = 'weighted',
        adaptive_config: Optional[Dict] = None
    ):
        """
        Initialize decentralized client.

        Args:
            c_id: Client ID
            title: Experiment title
            tasktype: Task type (e.g., "Animals", "Objects")
            model: PyTorch model
            seed: Random seed
            all_clients: List of all clients in the network (for P2P simulation)
            n_neighbors: Number of neighbors for soft clustering (default: 3)
            hca_alpha: HCA conflict-averse hyperparameter (default: 0.4)
            aggregation_method: Aggregation method - 'hca', 'fedavg', or 'hybrid' (default: 'hca')
            aggregation_start_round: Start aggregation from this round (default: 1, set >1 for delayed aggregation)
            aggregation_switch_round: Switch from FedAvg to HCA at this round (for hybrid mode)
        """
        super().__init__(c_id, title, tasktype, model, seed)

        self.all_clients = all_clients if all_clients is not None else []
        self.n_neighbors = n_neighbors
        self.hca_alpha = hca_alpha
        self.aggregation_method = aggregation_method
        self.aggregation_start_round = aggregation_start_round
        self.aggregation_switch_round = aggregation_switch_round

        # Adaptive aggregation settings
        self.adaptive_aggregation = adaptive_aggregation
        self.adaptive_method = adaptive_method
        self.adaptive_config = adaptive_config if adaptive_config is not None else {}

        # Track selected neighbors for each round
        self.neighbor_history = []

        # Store previous backbone for gradient computation
        self.prev_backbone = None

        # Snapshot of trained backbone (before aggregation)
        # This ensures all clients read consistent post-training, pre-aggregation states
        self.trained_backbone_snapshot = None

    def set_all_clients(self, all_clients: List['DecentralizedClient']):
        """
        Set the list of all clients in the network.

        This is needed for P2P communication simulation.

        Args:
            all_clients: List of all DecentralizedClient instances
        """
        self.all_clients = all_clients

    def compute_gradient(self) -> torch.Tensor:
        """
        Compute gradient (parameter change) of this client's backbone.

        Returns:
            Flattened gradient tensor representing parameter changes
        """
        curr_backbone = self.extract_curr_backbone()

        if self.prev_backbone is None:
            # First round: no gradient available, return zero tensor
            dummy_grad = torch.zeros(1)
            return dummy_grad

        # Compute gradient as difference between current and previous
        gradient = compute_model_gradient(curr_backbone, self.prev_backbone)

        return gradient

    def compute_parameters(self) -> torch.Tensor:
        """
        Extract current backbone parameters as a flattened tensor.

        This is used for parameter-based similarity computation (Solution D1),
        which computes similarity based on actual parameter values rather than
        parameter changes (gradients).

        Returns:
            Flattened parameter tensor representing current backbone state
        """
        curr_backbone = self.extract_curr_backbone()

        # Flatten all backbone parameters into a single tensor
        params = []
        for key in sorted(curr_backbone.keys()):  # Sort for consistency
            params.append(curr_backbone[key].flatten())

        if len(params) == 0:
            return torch.tensor([0.0])

        return torch.cat(params)

    def select_neighbors(self, similarity_matrix) -> List[int]:
        """
        Select top-K most similar neighbors based on task similarity.

        Args:
            similarity_matrix: N x N pairwise similarity matrix

        Returns:
            List of K client indices representing selected neighbors
        """
        # Find this client's index in the global client list
        client_idx = None
        for idx, client in enumerate(self.all_clients):
            if client.c_id == self.c_id:
                client_idx = idx
                break

        if client_idx is None:
            raise ValueError(f"Client {self.c_id} not found in all_clients list")

        # Select top-K neighbors
        neighbors = select_top_k_neighbors(
            client_idx=client_idx,
            similarity_matrix=similarity_matrix,
            k=self.n_neighbors
        )

        return neighbors

    def aggregate_with_neighbors(
        self,
        neighbor_indices: List[int],
        neighbor_similarities: List[float],
        round_num: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform local aggregation with selected neighbors.

        Supports multiple aggregation strategies:
        - HCA: Hyper Conflict-Averse aggregation (gradient-based optimization)
        - FedAvg: Simple weighted average (numerically stable)
        - Hybrid: Start with FedAvg, switch to HCA later (or delayed aggregation)
        - Adaptive: Various adaptive strategies (weighted, filter, progressive, hybrid_adaptive)

        IMPORTANT: This method uses trained_backbone_snapshot instead of
        extract_curr_backbone() to ensure all clients read consistent
        post-training, pre-aggregation states.

        Args:
            neighbor_indices: List of neighbor client indices
            neighbor_similarities: Similarity scores for each neighbor
            round_num: Current round number (for logging)

        Returns:
            Aggregated backbone parameters (or self's backbone if no aggregation)
        """
        # Check if aggregation should be skipped (delayed aggregation)
        if round_num < self.aggregation_start_round:
            print(f"[Round {round_num}] Client {self.c_id}: Skipping aggregation (delayed start, will start at round {self.aggregation_start_round})")
            return self.get_trained_backbone_snapshot()

        # Determine actual aggregation method for this round
        current_method = self._get_current_aggregation_method(round_num)

        # Collect SNAPSHOT backbones (post-training, pre-aggregation) from self and neighbors
        # This prevents reading already-aggregated backbones from earlier clients
        curr_backbones = [self.get_trained_backbone_snapshot()]
        prev_backbones = [self.extract_prev_backbone()]

        for neighbor_idx in neighbor_indices:
            neighbor = self.all_clients[neighbor_idx]
            # Use snapshot instead of extract_curr_backbone()
            curr_backbones.append(neighbor.get_trained_backbone_snapshot())
            prev_backbones.append(neighbor.extract_prev_backbone())

        print(f"[Round {round_num}] Client {self.c_id}: Aggregating with {len(neighbor_indices)} neighbors using {current_method.upper()}")

        # Check if using adaptive aggregation
        if self.adaptive_aggregation:
            print(f"  Using adaptive aggregation: {self.adaptive_method}")

            # Prepare config with max_rounds
            config = self.adaptive_config.copy()
            config['max_rounds'] = 10  # Default, can be overridden in config

            # Call appropriate adaptive aggregation method
            if self.adaptive_method == 'weighted':
                aggregated_backbone = adaptive_weighted_aggregate(
                    backbones=curr_backbones,
                    similarities=neighbor_similarities,
                    round_num=round_num,
                    config=config
                )
            elif self.adaptive_method == 'filter':
                aggregated_backbone = adaptive_filter_aggregate(
                    backbones=curr_backbones,
                    similarities=neighbor_similarities,
                    config=config
                )
            elif self.adaptive_method == 'progressive':
                aggregated_backbone = progressive_neighbor_aggregate(
                    backbones=curr_backbones,
                    round_num=round_num,
                    config=config
                )
            elif self.adaptive_method == 'hybrid_adaptive':
                aggregated_backbone = hybrid_adaptive_aggregate(
                    backbones=curr_backbones,
                    similarities=neighbor_similarities,
                    round_num=round_num,
                    config=config
                )
            else:
                raise ValueError(f"Unknown adaptive_method: {self.adaptive_method}")

        # Standard aggregation (HCA or FedAvg)
        elif current_method == 'hca':
            # HCA aggregation (existing method)
            aggregated_backbones = conflict_averse(
                curr_backbones_dicts=curr_backbones,
                prev_backbones_dicts=prev_backbones,
                ca_c=self.hca_alpha
            )
            aggregated_backbone = aggregated_backbones[0]

        elif current_method == 'fedavg':
            # FedAvg: Simple weighted average (equal weights for simplicity)
            aggregated_backbone = self._fedavg_aggregate(curr_backbones)

        else:
            raise ValueError(f"Unknown aggregation_method: {current_method}. "
                           f"Must be 'hca' or 'fedavg'")

        # Return aggregated backbone
        return aggregated_backbone

    def _get_current_aggregation_method(self, round_num: int) -> str:
        """
        Determine which aggregation method to use for current round.

        Supports hybrid mode where method switches from FedAvg to HCA.

        Args:
            round_num: Current round number

        Returns:
            'hca' or 'fedavg'
        """
        if self.aggregation_method == 'hybrid':
            # Hybrid mode: FedAvg -> HCA
            if self.aggregation_switch_round is None:
                raise ValueError("aggregation_switch_round must be specified for hybrid mode")

            if round_num < self.aggregation_switch_round:
                return 'fedavg'
            else:
                return 'hca'
        else:
            return self.aggregation_method

    def _fedavg_aggregate(self, backbones: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Perform FedAvg aggregation (simple weighted average).

        Args:
            backbones: List of backbone state_dicts (self + neighbors)

        Returns:
            Averaged backbone state_dict
        """
        # Equal weights for all participants (including self)
        n = len(backbones)
        weights = [1.0 / n] * n

        # Initialize aggregated backbone
        aggregated = {}

        # Average each parameter
        for key in backbones[0].keys():
            # Convert to float for aggregation, then convert back to original dtype
            original_dtype = backbones[0][key].dtype
            aggregated[key] = torch.zeros_like(backbones[0][key], dtype=torch.float32)

            for i, backbone in enumerate(backbones):
                aggregated[key] += weights[i] * backbone[key].float()

            # Convert back to original dtype
            aggregated[key] = aggregated[key].to(original_dtype)

        return aggregated

    def local_training_phase(
        self,
        round_num: int,
        epochs: int,
        verbose: bool = False
    ):
        """
        Phase 1: Local training only (no aggregation yet).

        This ensures all clients train before any aggregation happens,
        preventing zero-gradient issues in HCA aggregation.

        Args:
            round_num: Current training round number
            epochs: Number of local training epochs
            verbose: Whether to print detailed logs
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Client {self.c_id} - Round {round_num} - Training Phase")
            print(f"{'='*60}")

        self.conduct_training(num_epochs=epochs, current_round=round_num)

    def aggregation_phase(
        self,
        round_num: int,
        similarity_matrix,
        verbose: bool = False
    ):
        """
        Phase 2: Neighbor selection and aggregation (without updating prev).

        By this point, all clients have completed training, so all gradients
        are valid (non-zero) when performing HCA aggregation.

        IMPORTANT: This method does NOT update prev_backbone. That must be done
        separately after ALL clients have aggregated, to prevent sequential
        prev updates from causing zero gradients.

        Args:
            round_num: Current training round number
            similarity_matrix: N x N task similarity matrix
            verbose: Whether to print detailed logs
        """
        # Step 1: Select neighbors based on task similarity
        neighbor_indices = self.select_neighbors(similarity_matrix)
        self.neighbor_history.append(neighbor_indices)

        # Extract similarity scores for selected neighbors (needed for adaptive aggregation)
        client_idx = self._get_client_index()
        neighbor_similarities = [similarity_matrix[client_idx][n_idx] for n_idx in neighbor_indices]

        if verbose:
            log_clustering_info(
                round_num=round_num,
                client_idx=client_idx,
                selected_neighbors=neighbor_indices,
                similarity_matrix=similarity_matrix
            )

        # Step 2: Aggregate with selected neighbors
        aggregated_backbone = self.aggregate_with_neighbors(
            neighbor_indices=neighbor_indices,
            neighbor_similarities=neighbor_similarities,
            round_num=round_num
        )

        # Step 3: Update model with aggregated backbone
        self.replace_backbone(aggregated_backbone)

        # NOTE: prev_backbone is NOT updated here!
        # It will be updated in update_prev_backbone_phase()

    def update_prev_backbone_phase(self):
        """
        Phase 4: Update previous backbone for next round.

        This must be called AFTER all clients have completed aggregation_phase,
        to ensure no client reads an updated prev_backbone during aggregation.
        """
        self.prev_backbone = copy.deepcopy(self.extract_curr_backbone())

    def snapshot_trained_backbone(self):
        """
        Phase 2: Save snapshot of trained backbone (before aggregation).

        This creates a frozen snapshot of the post-training, pre-aggregation state.
        During aggregation, clients will read these snapshots instead of current
        backbones, preventing the sequential aggregation issue where later clients
        read already-aggregated backbones from earlier clients.
        """
        self.trained_backbone_snapshot = copy.deepcopy(self.extract_curr_backbone())

    def get_trained_backbone_snapshot(self) -> Dict[str, torch.Tensor]:
        """
        Get the snapshot of trained backbone (before aggregation).

        Returns:
            Dictionary of backbone parameters from snapshot
        """
        if self.trained_backbone_snapshot is not None:
            return self.trained_backbone_snapshot

        # Fallback: use current backbone if no snapshot exists
        return self.extract_curr_backbone()

    def decentralized_training_round(
        self,
        round_num: int,
        epochs: int,
        similarity_matrix,
        verbose: bool = True
    ):
        """
        Execute one round of decentralized training (legacy method).

        DEPRECATED: This method combines training and aggregation in one call,
        which causes zero-gradient issues. Use local_training_phase() and
        aggregation_phase() separately instead.

        Steps:
        1. Local training for E epochs
        2. Select top-K similar neighbors based on similarity matrix
        3. Aggregate backbone with selected neighbors using HCA
        4. Update model with aggregated backbone

        Args:
            round_num: Current training round number
            epochs: Number of local training epochs
            similarity_matrix: N x N task similarity matrix
            verbose: Whether to print detailed logs
        """
        # Step 1: Local training
        if verbose:
            print(f"\n{'='*60}")
            print(f"Client {self.c_id} - Round {round_num}")
            print(f"{'='*60}")

        self.conduct_training(num_epochs=epochs, current_round=round_num)

        # Step 2: Select neighbors based on task similarity
        neighbor_indices = self.select_neighbors(similarity_matrix)
        self.neighbor_history.append(neighbor_indices)

        if verbose:
            log_clustering_info(
                round_num=round_num,
                client_idx=self._get_client_index(),
                selected_neighbors=neighbor_indices,
                similarity_matrix=similarity_matrix
            )

        # Step 3: Aggregate with selected neighbors
        aggregated_backbone = self.aggregate_with_neighbors(
            neighbor_indices=neighbor_indices,
            round_num=round_num
        )

        # Step 4: Update model with aggregated backbone
        self.replace_backbone(aggregated_backbone)

        # Step 5: Update previous backbone for next round
        self.prev_backbone = copy.deepcopy(self.extract_curr_backbone())

    def _get_client_index(self) -> int:
        """Get this client's index in the global client list."""
        for idx, client in enumerate(self.all_clients):
            if client.c_id == self.c_id:
                return idx
        return -1

    def get_neighbor_history(self) -> List[List[int]]:
        """
        Get history of selected neighbors across all rounds.

        Returns:
            List where each element is the list of neighbor indices for that round
        """
        return self.neighbor_history

    def extract_curr_backbone(self) -> Dict[str, torch.Tensor]:
        """
        Extract current backbone parameters.

        Returns:
            Dictionary of backbone parameters
        """
        # Update checkpoint to get latest model state
        self.update_checkpoint(epoch=0)
        curr_backbone = {
            k: v.clone() for k, v in self.current_checkpoint["model_state_dict"].items()
            if "backbone" in k
        }
        return curr_backbone

    def extract_prev_backbone(self) -> Dict[str, torch.Tensor]:
        """
        Extract previous backbone parameters.

        Returns:
            Dictionary of backbone parameters
        """
        if self.prev_backbone is not None:
            return self.prev_backbone

        # If no previous backbone, use current as previous (first round case)
        return self.extract_curr_backbone()

    def initialize_prev_backbone(self):
        """
        Initialize previous backbone for gradient computation.
        Should be called before first training round.
        """
        self.prev_backbone = copy.deepcopy(self.extract_curr_backbone())
