# Dynamic Task Clustering and Aggregation for Decentralized Federated Multitask Learning

**Xi Chen, 2026, University of Zurich, Department of Informatics (IFI), Communication Systems Group (CSG)**

This repository contains the source code for implementing and evaluating a decentralized federated multi-task learning (FMTL) framework. The framework addresses client heterogeneity through dynamic task clustering and adaptive aggregation strategies, operating in a fully decentralized peer-to-peer architecture without requiring a central server.

## Key Features

- **Decentralized Architecture**: Peer-to-peer communication without central server coordination
- **Dynamic Task Clustering**: Adaptive soft clustering based on task similarity metrics
- **Multi-Task Support**: Handles both single-task (CIFAR-10) and multi-task scenarios (NYU Depth V2, Pascal Context)
- **Flexible Aggregation**: Multiple aggregation strategies including FedAvg, HCA, gradient-based, and cross-loss similarity
- **Configurable Learning**: All hyperparameters (learning rate, epochs, clustering parameters) managed through YAML configs

## Acknowledgement

This work builds upon and extends previous research in federated multi-task learning:

### Theoretical Foundation
- **ColNet**: Feng, C., Kohler, N. F., Wang, Z., Niu, W., Huertas Celdran, A., Bovet, G., & Stiller, B. (2025). ColNet: Collaborative Optimization in Decentralized Federated Multi-task Learning Systems. arXiv preprint arXiv:2501.10347. [https://arxiv.org/abs/2501.10347](https://arxiv.org/abs/2501.10347)
  - Provides the theoretical foundation for task similarity-based aggregation in decentralized FMTL

### Code Base
- **Nicolas Kohler's FMTL Framework**: [https://github.com/nicolas1a2b/asfdfmtl](https://github.com/nicolas1a2b/asfdfmtl)
  - This project is forked from and extends Nicolas Kohler's centralized federated multi-task learning implementation
  - Extended from centralized server-client architecture to fully decentralized peer-to-peer architecture

## Installation

### Prerequisites
- CUDA-capable GPU (strongly recommended, code not tested on CPU)
- CUDA Toolkit 12.1 or higher
- Conda package manager

### Environment Setup

1. **Install Conda** (if not already installed):
   - Download from [https://anaconda.org/anaconda/conda](https://anaconda.org/anaconda/conda)

2. **Create the conda environment**:
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**:
   ```bash
   conda activate asfdfmtl
   ```

4. **Verify CUDA installation**:
   ```bash
   nvidia-smi
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Datasets

The framework supports three datasets, which are automatically downloaded on first run:

### 1. CIFAR-10 (Single-Label Classification)
- **Tasks**: Animal classification (6 classes) and Object classification (4 classes)
- **Heterogeneity Type**: Class label distribution heterogeneity
- **Clients**: 6 clients (3 per task group)
- **Download**: Automatic via torchvision

### 2. NYU Depth V2 (Multi-Task Dense Prediction)
- **Tasks**: Depth estimation, semantic segmentation (13 classes), surface normal prediction
- **Heterogeneity Type**: Task heterogeneity
- **Clients**: 6 clients with varying task weights
- **Download**: Automatic on first run
- **Resolution**: 288x384
- **Dataset Link**: [https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

### 3. Pascal Context (Multi-Task Dense Prediction)
- **Tasks**: Semantic segmentation (59 classes), human parts segmentation (6 classes), edge detection
- **Heterogeneity Type**: Task heterogeneity
- **Clients**: 6 clients with varying task weights
- **Download**: Automatic on first run
- **Resolution**: 288x384
- **Dataset Link**: [https://cs.stanford.edu/~roozbeh/pascal-context/](https://cs.stanford.edu/~roozbeh/pascal-context/)

## Running Experiments

### CIFAR-10 Experiments

All CIFAR-10 experiments use single-label classification with ResNet-18 backbone.

#### 1. FedAvg with Gradient Similarity
```bash
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_fedavg_gradient.yml
```

#### 2. FedAvg with Cross-Loss Similarity
```bash
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_fedavg_crossloss.yml
```

#### 3. Dynamic Clustering
```bash
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_dynamic_clustering.yml
```

#### 4. Hybrid Strategy (Switch at Round 4/6/8)
```bash
# Switch from intra-task to cross-task at round 4
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_hybrid_switch_round4.yml

# Switch at round 6
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_hybrid_switch_round6.yml

# Switch at round 8
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_hybrid_switch_round8.yml
```

#### 5. Hybrid with Delayed Start
```bash
python src/decentralized/run_decentralized.py \
    --config configs/decentralized/cifar10/cifar10_hybrid_delayed_start.yml
```

### NYU Depth V2 Experiments

All NYU-V2 experiments use ResNet-50 backbone with multi-task heads.

#### A1: Single-Task Baseline with Weighted Aggregation
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_weighted_a1.yml
```

#### A2: Pairwise Task Combination
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_pairwise_weighted_a2.yml
```

#### B1: Gradient Similarity (Backbone Only)
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_gradient_backbone_b1.yml
```

#### B2: HCA Aggregation (Backbone Only)
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_hca_backbone_b2.yml
```

#### B3: HCA Aggregation (Full Model)
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_hca_full_b3.yml
```

#### B4: Multi-Task with Gradient Similarity
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_multitask_gradient_b4.yml
```

#### C1: Dynamic Clustering
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_dynamic_c1.yml
```

#### C2: Hierarchical Clustering
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/nyuv2/nyuv2_singletask_hierarchical_c2.yml
```

### Pascal Context Experiments

All Pascal Context experiments use ResNet-50 backbone with multi-task heads.

#### A1: Single-Task Baseline
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/pascal/pascal_singletask_weighted_a1.yml
```

#### A2: Pairwise Task Combination
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/pascal/pascal_pairwise_weighted_a2.yml
```

#### B1: Cross-Loss Similarity
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/pascal/pascal_singletask_crossloss_b1.yml
```

#### B1: Gradient Similarity (Backbone Only)
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/pascal/pascal_singletask_gradient_backbone_b1.yml
```

#### B4: Multi-Task with Gradient Similarity
```bash
python src/decentralized/run_multitask_decentralized.py \
    --config configs/decentralized/pascal/pascal_multitask_gradient_b4.yml
```

## Configuration Files

All experiments are configured through YAML files located in `configs/decentralized/`. Each configuration specifies:

- **General settings**: Title, dataset, number of rounds, seed
- **Setup parameters**: Number of neighbors, similarity method, aggregation method
- **Training parameters**: Local epochs, learning rate, batch size
- **Task configuration**: Task weights per client, number of classes

### Example Configuration Structure

```yaml
general:
  title: "Experiment_Name"
  dataset: cifar10  # or nyuv2, pascal_context
  rounds: 30
  seed: 42

setup:
  n_neighbors: 3  # Number of neighbors for soft clustering
  similarity_method: gradient  # gradient, crossloss, parameter
  aggregation_method: fedavg  # fedavg, hca, weighted
  learning_rate: 0.01  # CIFAR-10: 0.01, NYU/Pascal: 0.001

training:
  local_epochs: 3
  batch_size: 4
  learning_rate: 0.001
```

## Project Structure

```
asfdfmtl/
├── configs/
│   └── decentralized/
│       ├── cifar10/          # CIFAR-10 experiment configs (7 files)
│       ├── nyuv2/            # NYU-V2 experiment configs (8 files)
│       └── pascal/           # Pascal Context configs (5 files)
├── src/
│   ├── decentralized/        # Decentralized FL implementation
│   │   ├── run_decentralized.py           # CIFAR-10 runner
│   │   ├── run_multitask_decentralized.py # NYU-V2/Pascal runner
│   │   ├── decentralized_client.py        # Single-task client
│   │   ├── multitask_client.py            # Multi-task client
│   │   ├── task_similarity.py             # Similarity metrics
│   │   └── ...
│   ├── models/              # Neural network models
│   │   ├── single_label_classification_model.py  # ResNet-18 for CIFAR-10
│   │   └── multitask_dense_prediction_model.py   # ResNet-50 multi-task
│   ├── data_handling/       # Dataset loaders
│   │   ├── data_manager.py  # CIFAR-10 data manager
│   │   ├── nyuv2.py         # NYU Depth V2 loader
│   │   └── pascal_context.py # Pascal Context loader
│   ├── client_handling/     # Client utilities
│   └── visualization/       # Plotting utilities
├── results/                 # Experiment results (auto-generated)
├── figures/                 # Visualization outputs
├── environment.yml          # Conda environment specification
└── README.md
```

## Results and Outputs

Running an experiment will create the following structure in `results/`:

```
results/
└── [dataset]_[experiment]/
    ├── [experiment_name]/
    │   ├── metrics/
    │   │   ├── client_0_metrics.json
    │   │   ├── client_1_metrics.json
    │   │   └── ...
    │   ├── similarity_matrices/
    │   │   ├── round_1_similarity.npy
    │   │   └── ...
    │   └── config.yml  # Copy of experiment config
```

### Metrics Tracked

For each client and round, the following metrics are logged:

**CIFAR-10**:
- Training/validation loss
- Precision, recall, F1-score
- Per-class accuracy

**NYU Depth V2**:
- Depth: Mean Absolute Error (MAE)
- Segmentation: Cross-entropy loss, mIoU
- Normal: Cosine similarity loss

**Pascal Context**:
- Segmentation: Cross-entropy loss, mIoU
- Human Parts: Cross-entropy loss
- Edge: Binary cross-entropy

## Reproducibility

To ensure reproducibility:

1. **Fixed seeds**: All experiments use `seed: 42` in configs
2. **Deterministic operations**: PyTorch deterministic mode enabled
3. **Fixed learning rates**: Configurable via YAML files
4. **Version control**: All dependencies pinned in `environment.yml`

Note: Perfect reproducibility across different hardware is not guaranteed due to GPU non-determinism. See [PyTorch Randomness Documentation](https://pytorch.org/docs/stable/notes/randomness.html) for details.

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3080/3090)
- **VRAM**: Minimum 8GB (16GB+ recommended for multi-task experiments)
- **RAM**: 16GB+ system memory
- **Storage**: ~50GB for datasets and checkpoints

### Typical Training Times (RTX 3080)

- CIFAR-10 (30 rounds, 3 epochs/round): ~2-3 hours
- NYU Depth V2 (50 rounds, 5 epochs/round): ~8-12 hours
- Pascal Context (50 rounds, 5 epochs/round): ~10-15 hours

## Advanced Features

### Learning Rate Scheduling

All multi-task experiments support learning rate scheduling:

```yaml
training:
  lr_scheduler:
    enabled: true
    type: cosine  # cosine, step, exponential
    min_lr: 0.0001
    warmup_rounds: 3
```

### Early Stopping

Prevent overfitting with early stopping:

```yaml
training:
  early_stopping:
    enabled: true
    patience: 8
    min_delta: 0.003
    mode: min  # min for loss, max for accuracy
```

### Gradient Clipping

Stabilize training with gradient clipping:

```yaml
training:
  gradient_clip_norm: 1.0  # Max gradient norm
```
