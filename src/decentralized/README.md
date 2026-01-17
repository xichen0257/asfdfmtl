# Decentralized Federated Multi-Task Learning with Dynamic Soft Clustering

## Overview

This implementation extends Nicolas Kohler's centralized FMTL framework to support:

1. **Decentralized Architecture**: Peer-to-peer communication without a central server
2. **Dynamic Task Clustering**: Automatic discovery of task similarity each round
3. **Soft Clustering (N=3)**: Each client aggregates with 3 most similar neighbors
4. **Gradient-based Similarity**: Task similarity computed using cosine similarity of model gradients
5. **Local HCA Aggregation**: Each client performs conflict-averse aggregation locally

## Architecture

### Key Differences from Centralized Approach

| Aspect | Centralized (Nicolas) | Decentralized (This Work) |
|--------|----------------------|---------------------------|
| **Communication** | Client <-> Server | Client <-> Client (P2P) |
| **Task Groups** | Static (predefined) | Dynamic (similarity-based) |
| **Clustering** | Hard (Animals OR Objects) | Soft (N=3 overlapping neighbors) |
| **Aggregation** | Central server | Local (each client) |
| **Similarity Metric** | N/A (static groups) | Gradient cosine similarity |

### System Flow

```
Round t:
  For each client i:
    1. Local Training (E epochs)
       -> Update model parameters

    2. Gradient Computation
       -> delta_w_i = w_i^t - w_i^{t-1}

    3. Similarity Matrix Computation (P2P)
       -> S[i,j] = cosine_sim(delta_w_i, delta_w_j)

    4. Dynamic Neighbor Selection
       -> N_i = top-3 most similar clients

    5. Local HCA Aggregation
       -> w_i^{t+1} = HCA(w_i^t, {w_j^t | j in N_i})
```

## Module Structure

```
src/decentralized/
|-- __init__.py                     # Package initialization
|-- task_similarity.py              # Task similarity computation
|-- decentralized_client.py         # Decentralized client class
|-- run_decentralized.py            # Main training script
`-- README.md                       # This file

configs/decentralized/
`-- cifar10_dynamic_soft_clustering_quick.yml  # Quick experiment config
```

## Quick Start

### 1. Environment Setup

Ensure you have the base environment set up:

```bash
conda env create -f environment.yml
conda activate asfdfmtl
```

### 2. Run Quick Validation Experiment

```bash
cd /home/ubuntu/asfdfmtl
python src/decentralized/run_decentralized.py \
  --config configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml
```

This runs a lightweight experiment:
- 2 rounds
- 1 epoch per round
- 50% of dataset
- 6 clients (3 Animals + 3 Objects)
- N=3 soft clustering

### 3. View Results

Results are saved to:
```
results/decentralized_experiments/CIFAR10_Decentralized_DynamicSoftClustering_Quick/
|-- results.json              # Performance metrics by round
`-- clustering_analysis.json  # Neighbor selection history
```

## Configuration

### Experiment Config File

```yaml
general:
  title: CIFAR10_Decentralized_DynamicSoftClustering_Quick
  dataset: cifar10
  rounds: 2                    # Number of training rounds
  seed: 42
  backbone_layers: minus1       # ResNet18 backbone (all except last layer)

setup:
  dataset_fraction: 0.5         # Use 50% of data for quick experiments
  hca_alpha: 0.4                # HCA conflict-averse parameter
  n_neighbors: 3                # Soft clustering: aggregate with 3 neighbors
  local_epochs: 1               # Epochs per round

  # Task configurations (for data loading)
  task_configs:
    Animals:
      num_classes: 6
      client_ids: [AN_Cifar10_C0, AN_Cifar10_C1, AN_Cifar10_C2]
    Objects:
      num_classes: 4
      client_ids: [OB_Cifar10_C0, OB_Cifar10_C1, OB_Cifar10_C2]
```

### Key Parameters

- **rounds**: Number of communication rounds
- **local_epochs**: Local training epochs per round
- **n_neighbors**: Number of neighbors for soft clustering (default: 3)
- **hca_alpha**: HCA conflict-averse parameter (controls gradient conflict resolution)
- **dataset_fraction**: Fraction of dataset to use (0.5 = 50% for quick experiments)

## Implementation Details

### Task Similarity Computation

Task similarity is computed using **gradient cosine similarity**:

```python
def compute_gradient_similarity(gradient_a, gradient_b):
    """
    Cosine similarity between gradients:

    sim(i,j) = (delta_w_i * delta_w_j) / (||delta_w_i|| x ||delta_w_j||)

    Range: [-1, 1]
    - sim = 1: Identical task objectives
    - sim = 0: Orthogonal tasks
    - sim = -1: Conflicting tasks
    """
    return cosine_similarity(gradient_a, gradient_b)
```

**Why gradient similarity?**
- Captures task relatedness through optimization direction
- Higher similarity -> clients learn similar features
- Suitable for both task heterogeneity and label heterogeneity

### Soft Clustering (N=3)

Each client selects **top-3 most similar neighbors** each round:

```python
def select_top_k_neighbors(client_idx, similarity_matrix, k=3):
    """
    For client i, select k neighbors j_1, j_2, ..., j_k where:
    - sim(i, j_1) >= sim(i, j_2) >= ... >= sim(i, j_k)
    - j != i (exclude self)

    This forms overlapping clusters:
    - Client A -> [B, C, D]
    - Client B -> [A, C, E]
    - Overlap: {A, B, C} form a soft cluster
    """
    similarities = similarity_matrix[client_idx].copy()
    similarities[client_idx] = -inf  # Exclude self
    top_k_indices = argsort(similarities)[-k:][::-1]
    return top_k_indices
```

**Why N=3?**
- FLSC paper shows N=3 is optimal balance
- N=1: Too rigid (similar to hard clustering)
- N>3: Includes dissimilar tasks, hurts performance

### Local HCA Aggregation

Each client performs HCA aggregation with selected neighbors:

```python
def aggregate_with_neighbors(self, neighbor_indices):
    """
    HCA aggregation formula (from FedHCA^2 paper):

    w_i^{t+1} = w_i^t + delta_w_i + lambda*U_w

    where:
    - delta_w_i = local gradient
    - U_w = conflict-averse update (minimizes gradient conflicts)
    - lambda = scaling factor (controlled by hca_alpha)
    """
    # Collect backbones from self and neighbors
    curr_backbones = [self.backbone] + [neighbor.backbone for neighbor in neighbors]
    prev_backbones = [self.prev_backbone] + [neighbor.prev_backbone for neighbor in neighbors]

    # Perform HCA aggregation
    aggregated = conflict_averse(curr_backbones, prev_backbones, hca_alpha)

    return aggregated[0]  # Return this client's aggregated backbone
```

## Expected Behavior

### Clustering Patterns

With Animals={bird, cat, deer, dog, frog, horse} and Objects={airplane, automobile, ship, truck}:

**Expected clustering**:
- Animal clients should select other animal clients (high similarity)
- Object clients should select other object clients (high similarity)
- Cross-task similarity should be lower

**Example output**:
```
Round 1 Clustering:
  AN_Cifar10_C0: [AN_Cifar10_C1, AN_Cifar10_C2, OB_Cifar10_C0]
  AN_Cifar10_C1: [AN_Cifar10_C0, AN_Cifar10_C2, OB_Cifar10_C1]
  OB_Cifar10_C0: [OB_Cifar10_C1, OB_Cifar10_C2, AN_Cifar10_C0]
  ...
```

Note: Some cross-task neighbors may appear, especially in early rounds.

### Performance Metrics

Track these metrics in `results.json`:

- **test_loss**: Test loss per client
- **test_acc**: Test accuracy per client
- **animals_avg**: Average metrics for animal clients
- **objects_avg**: Average metrics for object clients
- **overall_avg**: Overall average across all clients

## Experiments

### Quick Validation (Provided)

```bash
python src/decentralized/run_decentralized.py \
  --config configs/decentralized/cifar10_dynamic_soft_clustering_quick.yml
```

**Expected runtime**: ~5-10 minutes (CPU) / ~2-3 minutes (GPU)

### Full Experiment (To be created)

For your thesis, create a full experiment config:

```yaml
general:
  rounds: 10                   # More rounds for convergence

setup:
  dataset_fraction: 1.0        # Full dataset
  local_epochs: 3              # More epochs per round
```

**Expected runtime**: ~2-3 hours (GPU)

## Comparison with Baselines

To validate your approach, compare against:

1. **Centralized FedPer + HCA** (Nicolas's baseline)
   ```bash
   python src/run.py --config configs/cifar-10_vepr/cifar-10_fedper_hca_2epr.yml
   ```

2. **Decentralized Dynamic Soft Clustering** (This work)
   ```bash
   python src/decentralized/run_decentralized.py --config configs/decentralized/...yml
   ```

Expected improvements:
- **Dynamic clustering** adapts to task similarity (vs static groups)
- **Soft clustering** shares knowledge across overlapping groups
- **Decentralized** removes single point of failure

## Future Extensions

### 1. PASCAL Context Dataset
Replace CIFAR-10 with PASCAL Context for realistic soft clustering:

```python
# Instead of Animals vs Objects (sharp boundary)
# Use overlapping semantic groups:
task_configs:
  Indoor:
    classes: [floor, wall, ceiling, table, chair, window]
  Furniture:
    classes: [chair, table, sofa, bed, cabinet, shelf]
  # Overlap: chair, table, window
```

### 2. Adaptive Neighbor Selection

Instead of fixed N=3, select neighbors based on similarity threshold:

```python
# In task_similarity.py
def adaptive_cluster_selection(
    client_idx,
    similarity_matrix,
    threshold=0.3,  # Only include neighbors with sim > 0.3
    min_neighbors=1,
    max_neighbors=3
):
    # Select neighbors above threshold (between 1 and 3)
    pass
```

### 3. True P2P Communication

Current implementation simulates P2P. For real deployment:

```python
# Use network protocols (gRPC, PyTorch RPC, etc.)
class RealP2PClient(DecentralizedClient):
    def send_model(self, neighbor_address):
        # Send model to neighbor via network
        pass

    def receive_models(self, neighbor_addresses):
        # Receive models from neighbors
        pass
```

## Troubleshooting

### GPU Memory Issues

If you encounter OOM errors:
1. Reduce `dataset_fraction` in config (e.g., 0.25)
2. Reduce number of clients
3. Reduce `n_neighbors` (e.g., 2 instead of 3)

### Slow Training

For faster experiments:
1. Use smaller `dataset_fraction`
2. Reduce `local_epochs`
3. Use GPU: `export CUDA_VISIBLE_DEVICES=0`

### Import Errors

Ensure you're running from project root:
```bash
cd /home/ubuntu/asfdfmtl
python src/decentralized/run_decentralized.py --config ...
```

## References

1. **ColNet** (Feng et al., 2025): Decentralized FMTL with static task groups
2. **FLSC** (Ghosh et al., 2020): Soft clustering with N=3 optimal
3. **FedHCA^2** (Lu et al., 2024): Hyper conflict-averse aggregation
4. **FedPer** (Arivazhagan et al., 2019): Personalized FL with shared backbone

## Contact

For questions about this implementation, refer to:
- Nicolas Kohler's original codebase: `/home/ubuntu/asfdfmtl/README.md`
- ColNet paper: `papers/ColNet.pdf`
- Your thesis advisor: Chao Feng
