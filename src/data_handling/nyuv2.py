"""
NYU Depth V2 Multi-Task Data Manager

Supports three tasks:
- Depth estimation (regression)
- Semantic segmentation (13-class classification)
- Surface normal prediction (3D vector prediction)

Each client can have different task preferences (soft clustering).
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import h5py
from typing import Dict, List, Tuple, Optional
import warnings


class NYUDepthV2Dataset(Dataset):
    """
    NYU Depth V2 dataset for multi-task learning.

    Returns:
        dict: {
            'image': RGB image tensor (3, H, W)
            'depth': depth map tensor (1, H, W)
            'segmentation': segmentation mask (H, W) - long tensor
            'normal': surface normal map (3, H, W)
        }
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform=None,
        target_size: Tuple[int, int] = (288, 384),
        download: bool = True
    ):
        """
        Args:
            root_dir: Root directory for NYU Depth V2 data
            split: 'train' or 'test'
            transform: Optional transform to apply to images
            target_size: (height, width) to resize images
            download: Whether to download dataset if not found
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_size = target_size

        # Create data directory
        os.makedirs(root_dir, exist_ok=True)

        # Dataset paths
        self.data_file = os.path.join(root_dir, f'nyu_depth_v2_labeled.mat')

        if download and not os.path.exists(self.data_file):
            self._download_dataset()

        # Load data
        self._load_data()

    def _download_dataset(self):
        """
        Download NYU Depth V2 dataset.

        Note: The full dataset is ~2.8GB. This will download the labeled subset
        with 1449 images from the official source.
        """
        print("=" * 60)
        print("NYU Depth V2 Dataset Download")
        print("=" * 60)
        print(f"Downloading to: {self.root_dir}")
        print("Dataset size: ~2.8 GB (labeled subset)")
        print("This may take several minutes...")
        print()

        import urllib.request

        url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

        try:
            urllib.request.urlretrieve(
                url,
                self.data_file,
                reporthook=self._download_progress_hook
            )
            print("\nDownload completed successfully!")
        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nAlternative: Download manually from:")
            print("https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html")
            raise

    def _download_progress_hook(self, count, block_size, total_size):
        """Progress bar for download"""
        percent = min(int(count * block_size * 100 / total_size), 100)
        if count % 50 == 0:  # Update every 50 blocks
            print(f"\rProgress: {percent}% ({count * block_size / 1e6:.1f} MB / "
                  f"{total_size / 1e6:.1f} MB)", end='')

    def _load_data(self):
        """
        Load NYU Depth V2 data from .mat file.

        The .mat file contains:
        - images: (3, 640, 480, 1449) - RGB images
        - depths: (640, 480, 1449) - depth maps
        - labels: (640, 480, 1449) - 13-class segmentation
        - instances: (640, 480, 1449) - instance masks (not used)
        """
        print(f"Loading NYU Depth V2 {self.split} split...")

        try:
            with h5py.File(self.data_file, 'r') as f:
                # Load images - actual format: (N, C, H, W)
                images = np.array(f['images'])  # (1449, 3, 640, 480)
                depths = np.array(f['depths'])  # (1449, 640, 480)
                labels = np.array(f['labels'])  # (1449, 640, 480)

                # Transpose images from (N, C, H, W) to (N, H, W, C)
                self.images = np.transpose(images, (0, 2, 3, 1))  # (1449, 640, 480, 3)
                self.depths = depths  # Already (N, H, W)
                self.labels = labels  # Already (N, H, W)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load NYU Depth V2 data: {e}\n"
                "Please ensure the dataset is downloaded correctly."
            )

        # Split into train/test (795 train, 654 test)
        n_total = self.images.shape[0]
        n_train = 795

        if self.split == 'train':
            self.images = self.images[:n_train]
            self.depths = self.depths[:n_train]
            self.labels = self.labels[:n_train]
        else:  # test
            self.images = self.images[n_train:]
            self.depths = self.depths[n_train:]
            self.labels = self.labels[n_train:]

        # Compute surface normals from depth maps
        print("Computing surface normals from depth maps...")
        self.normals = self._compute_normals_batch(self.depths)

        print(f"Loaded {len(self)} images for {self.split} split")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Depth shape: {self.depths.shape}")
        print(f"  Labels shape: {self.labels.shape}")
        print(f"  Normals shape: {self.normals.shape}")

    def _compute_normals_batch(self, depth_maps: np.ndarray) -> np.ndarray:
        """
        Compute surface normals from depth maps using gradient-based method.

        Args:
            depth_maps: (N, H, W) depth maps

        Returns:
            normals: (N, H, W, 3) surface normal maps (unit vectors)
        """
        N, H, W = depth_maps.shape
        normals = np.zeros((N, H, W, 3), dtype=np.float32)

        for i in range(N):
            depth = depth_maps[i]

            # Compute gradients
            dz_dy, dz_dx = np.gradient(depth)

            # Normal vector = (-dz/dx, -dz/dy, 1)
            normal = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))

            # Normalize to unit length
            norm = np.linalg.norm(normal, axis=2, keepdims=True)
            norm = np.where(norm < 1e-6, 1.0, norm)  # Avoid division by zero
            normal = normal / norm

            normals[i] = normal

        return normals

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict: {
                'image': (3, H, W) float32
                'depth': (1, H, W) float32
                'segmentation': (H, W) long
                'normal': (3, H, W) float32
            }
        """
        # Load data
        image = self.images[idx]  # (H, W, 3) uint8
        depth = self.depths[idx]  # (H, W) float
        label = self.labels[idx]  # (H, W) int
        normal = self.normals[idx]  # (H, W, 3) float

        # Map labels >= 13 to ignore_index (255)
        # NYU Depth V2 has 895 classes, but we use only 13-class subset
        label = np.where(label >= 13, 255, label)

        # Convert to PIL Images for resizing
        image = Image.fromarray(image.astype(np.uint8))
        depth_pil = Image.fromarray(depth.astype(np.float32), mode='F')
        label_pil = Image.fromarray(label.astype(np.uint8), mode='L')

        # Resize
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        depth_pil = depth_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        label_pil = label_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        # Resize normals (H, W, 3) -> (new_H, new_W, 3)
        normal_resized = np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        for c in range(3):
            normal_c = Image.fromarray(normal[:, :, c], mode='F')
            normal_c_resized = normal_c.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            normal_resized[:, :, c] = np.array(normal_c_resized)

        # Convert to numpy
        image = np.array(image)  # (H, W, 3)
        depth = np.array(depth_pil)  # (H, W)
        label = np.array(label_pil)  # (H, W)
        normal = normal_resized  # (H, W, 3)

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert to tensors
        depth = torch.from_numpy(depth).unsqueeze(0).float()  # (1, H, W)
        label = torch.from_numpy(label).long()  # (H, W)
        normal = torch.from_numpy(normal).permute(2, 0, 1).float()  # (3, H, W)

        return {
            'image': image,
            'depth': depth,
            'segmentation': label,
            'normal': normal
        }


class MultiTaskSubset(Dataset):
    """
    Subset of NYU Depth V2 for a specific client with task weights.

    This wrapper allows different clients to focus on different tasks
    by providing task-specific sampling and weighting.
    """

    def __init__(
        self,
        dataset: NYUDepthV2Dataset,
        indices: List[int],
        task_weights: Dict[str, float]
    ):
        """
        Args:
            dataset: The full NYU Depth V2 dataset
            indices: Subset of indices for this client
            task_weights: Dict of task weights, e.g.,
                         {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0}
        """
        self.dataset = dataset
        self.indices = indices
        self.task_weights = task_weights

        # Validate task weights
        expected_tasks = {'depth', 'segmentation', 'normal'}
        assert set(task_weights.keys()) == expected_tasks, \
            f"Task weights must include all tasks: {expected_tasks}"
        assert abs(sum(task_weights.values()) - 1.0) < 1e-6, \
            f"Task weights must sum to 1.0, got {sum(task_weights.values())}"

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item and attach task weights"""
        actual_idx = self.indices[idx]
        sample = self.dataset[actual_idx]

        # Add task weights to the sample
        sample['task_weights'] = self.task_weights

        return sample


class DMNYUDepthV2:
    """
    Data Manager for NYU Depth V2 multi-task learning.

    Manages data distribution across clients with soft clustering support.
    Each client gets a subset of data and has task preferences.
    """

    def __init__(
        self,
        seed: int,
        num_clients: int,
        task_weights_per_client: List[Dict[str, float]],
        dataset_fraction: float = 1.0,
        root_dir: str = './data/nyuv2',
        batch_size: int = 8,
        download: bool = True
    ):
        """
        Args:
            seed: Random seed
            num_clients: Number of clients
            task_weights_per_client: List of task weight dicts for each client
                Example: [
                    {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0},
                    {'depth': 0.0, 'segmentation': 0.7, 'normal': 0.3},
                    ...
                ]
            dataset_fraction: Fraction of dataset to use (for quick testing)
            root_dir: Root directory for dataset
            batch_size: Batch size for data loaders
            download: Whether to download dataset if not found
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_clients = num_clients
        self.task_weights_per_client = task_weights_per_client
        self.dataset_fraction = dataset_fraction
        self.batch_size = batch_size

        assert len(task_weights_per_client) == num_clients, \
            f"Must provide task weights for all {num_clients} clients"

        # Load datasets
        print("Loading NYU Depth V2 dataset...")
        self.train_dataset = NYUDepthV2Dataset(
            root_dir=root_dir,
            split='train',
            transform=self._get_train_transforms(),
            download=download
        )

        self.test_dataset = NYUDepthV2Dataset(
            root_dir=root_dir,
            split='test',
            transform=self._get_test_transforms(),
            download=False  # Already downloaded in train
        )

        # Prepare client splits
        self._prepare_client_splits()

        # Create data loaders
        self._prepare_loaders()

    def _get_train_transforms(self):
        """Data augmentation for training"""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # ImageNet normalization (commonly used)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _get_test_transforms(self):
        """Normalization for testing (no augmentation)"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _prepare_client_splits(self):
        """Split training data among clients"""
        n_train = len(self.train_dataset)
        n_use = int(n_train * self.dataset_fraction)

        # Use subset if dataset_fraction < 1.0
        all_indices = list(range(n_train))
        np.random.shuffle(all_indices)
        all_indices = all_indices[:n_use]

        # Split indices among clients (IID split)
        indices_per_client = np.array_split(all_indices, self.num_clients)

        # Further split each client's data into train/val (90/10)
        self.train_datasets = []
        self.val_datasets = []

        for client_idx in range(self.num_clients):
            client_indices = indices_per_client[client_idx].tolist()
            n_client = len(client_indices)
            n_train_client = int(n_client * 0.9)

            train_indices = client_indices[:n_train_client]
            val_indices = client_indices[n_train_client:]

            # Create multi-task subsets with task weights
            task_weights = self.task_weights_per_client[client_idx]

            train_subset = MultiTaskSubset(
                self.train_dataset,
                train_indices,
                task_weights
            )

            val_subset = MultiTaskSubset(
                self.train_dataset,
                val_indices,
                task_weights
            )

            self.train_datasets.append(train_subset)
            self.val_datasets.append(val_subset)

            print(f"Client {client_idx}: {len(train_indices)} train, "
                  f"{len(val_indices)} val samples | "
                  f"Task weights: {task_weights}")

        # Test dataset is shared (same for all clients)
        # Each client can evaluate on all tasks
        print(f"Shared test dataset: {len(self.test_dataset)} samples")

    def _prepare_loaders(self):
        """Create data loaders for all clients"""
        self.train_loaders = []
        self.val_loaders = []

        for client_idx in range(self.num_clients):
            train_loader = DataLoader(
                self.train_datasets[client_idx],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            val_loader = DataLoader(
                self.val_datasets[client_idx],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )

            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)

        # Shared test loader
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def get_train_loader(self, client_idx: int) -> DataLoader:
        """Get train loader for a specific client"""
        return self.train_loaders[client_idx]

    def get_val_loader(self, client_idx: int) -> DataLoader:
        """Get validation loader for a specific client"""
        return self.val_loaders[client_idx]

    def get_test_loader(self) -> DataLoader:
        """Get shared test loader"""
        return self.test_loader

    def get_task_weights(self, client_idx: int) -> Dict[str, float]:
        """Get task weights for a specific client"""
        return self.task_weights_per_client[client_idx]


if __name__ == "__main__":
    """Test the data manager"""

    print("=" * 60)
    print("Testing NYU Depth V2 Data Manager")
    print("=" * 60)

    # Test with 6 clients (soft clustering setup)
    task_weights = [
        {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0},  # Client 0
        {'depth': 0.7, 'segmentation': 0.0, 'normal': 0.3},  # Client 1
        {'depth': 0.3, 'segmentation': 0.7, 'normal': 0.0},  # Client 2
        {'depth': 0.0, 'segmentation': 0.7, 'normal': 0.3},  # Client 3
        {'depth': 0.3, 'segmentation': 0.0, 'normal': 0.7},  # Client 4
        {'depth': 0.0, 'segmentation': 0.3, 'normal': 0.7},  # Client 5
    ]

    dm = DMNYUDepthV2(
        seed=42,
        num_clients=6,
        task_weights_per_client=task_weights,
        dataset_fraction=0.1,  # Use 10% for quick testing
        batch_size=4,
        download=True
    )

    print("\n" + "=" * 60)
    print("Testing data loading...")
    print("=" * 60)

    # Test loading from client 0
    train_loader = dm.get_train_loader(0)
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Depth shape: {batch['depth'].shape}")
    print(f"Segmentation shape: {batch['segmentation'].shape}")
    print(f"Normal shape: {batch['normal'].shape}")
    print(f"Task weights: {batch['task_weights']}")

    print("\nData manager test passed!")
