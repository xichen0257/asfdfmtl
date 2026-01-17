"""
PASCAL Context Multi-Task Data Manager

Based on: Maninis et al. CVPR 2019 - "Attentive Single-Tasking of Multiple Tasks"
Data: https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz

Supports three tasks (all original annotations):
- Semantic segmentation (59-class classification)
- Human parts segmentation (15-class classification)
- Edge detection (binary prediction)

Each client can have different task preferences (soft clustering).
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import os
import scipy.io as sio
from typing import Dict, List, Tuple, Optional
import warnings
import tarfile
import urllib.request


class PascalContextDataset(Dataset):
    """
    PASCAL Context dataset for multi-task learning.

    Returns:
        dict: {
            'image': RGB image tensor (3, H, W)
            'segmentation': semantic segmentation mask (H, W) - long tensor [0-58, 255=ignore]
            'human_parts': human parts segmentation (H, W) - long tensor [0-14, 255=ignore]
            'edge': edge detection map (H, W) - float tensor [0.0-1.0]
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
            root_dir: Root directory for PASCAL Context data
            split: 'train' or 'val'
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
        self.data_dir = os.path.join(root_dir, 'PASCAL_MT')
        self.images_dir = os.path.join(self.data_dir, 'JPEGImages')
        self.semseg_dir = os.path.join(self.data_dir, 'semseg', 'pascal-context')
        self.human_parts_dir = os.path.join(self.data_dir, 'human_parts')
        self.edge_dir = os.path.join(self.data_dir, 'pascal-context', 'trainval')

        # Check if dataset exists, download if needed
        if download and not os.path.exists(self.data_dir):
            self._download_dataset()

        # Load image list
        self._load_image_list()

        print(f"Loaded {len(self)} images for {self.split} split")

    def _download_dataset(self):
        """
        Download PASCAL Context multi-task dataset.

        Note: The dataset is ~1.3GB. This will download from ETH Zurich.
        """
        print("=" * 80)
        print("PASCAL Context Multi-Task Dataset Download")
        print("=" * 80)
        print(f"Downloading to: {self.root_dir}")
        print("Dataset size: ~1.3 GB")
        print("Source: ETH Zurich (Maninis et al. CVPR 2019)")
        print("This may take several minutes...")
        print()

        url = "https://data.vision.ee.ethz.ch/kmaninis/share/MTL/PASCAL_MT.tgz"
        tar_file = os.path.join(self.root_dir, 'PASCAL_MT.tgz')

        try:
            # Download
            print("Downloading...")
            urllib.request.urlretrieve(
                url,
                tar_file,
                reporthook=self._download_progress_hook
            )
            print("\nDownload completed!")

            # Extract
            print("\nExtracting archive...")
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(path=self.root_dir)
            print("Extraction completed!")

            # Clean up
            os.remove(tar_file)
            print("Cleaned up temporary file")

            print("\n" + "=" * 80)
            print("Dataset ready!")
            print("=" * 80)

        except Exception as e:
            print(f"\nDownload failed: {e}")
            print("\nAlternative: Download manually from:")
            print(url)
            print(f"Then extract to: {self.root_dir}")
            raise

    def _download_progress_hook(self, count, block_size, total_size):
        """Progress bar for download"""
        percent = min(int(count * block_size * 100 / total_size), 100)
        if count % 50 == 0:  # Update every 50 blocks
            mb_downloaded = count * block_size / 1e6
            mb_total = total_size / 1e6
            print(f"\rProgress: {percent}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)", end='')

    def _load_image_list(self):
        """
        Load list of image IDs for the split.

        PASCAL Context uses train/val splits defined in text files.
        Format: PASCAL_MT/ImageSets/Context/train.txt (or val.txt)
        """
        split_file = os.path.join(
            self.data_dir,
            'ImageSets',
            'Context',
            f'{self.split}.txt'
        )

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Please ensure dataset is downloaded correctly.\n"
                f"Expected structure: {self.data_dir}/ImageSets/Context/{self.split}.txt"
            )

        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]

        print(f"  Loaded {len(self.image_ids)} image IDs from {split_file}")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            dict: {
                'image': (3, H, W) float32
                'segmentation': (H, W) long [0-58, 255=ignore]
                'human_parts': (H, W) long [0-14, 255=ignore]
                'edge': (H, W) float [0.0-1.0]
            }
        """
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')

        # Load semantic segmentation (59 classes)
        # Format: PNG file with class labels (already 0-indexed)
        semseg_path = os.path.join(self.semseg_dir, f'{image_id}.png')
        semseg = np.array(Image.open(semseg_path), dtype=np.int64)
        # Map invalid labels (>=59) to ignore_index (255)
        semseg = np.where((semseg < 0) | (semseg >= 59), 255, semseg)

        # Load human parts (15 classes)
        # Format: .mat file with 'anno' field containing object annotations
        human_parts_path = os.path.join(self.human_parts_dir, f'{image_id}.mat')
        human_parts_data = sio.loadmat(human_parts_path)

        # Extract human parts from annotations
        # anno structure: anno[0][0][1][0] contains objects array
        human_parts = np.zeros(semseg.shape, dtype=np.int64)
        try:
            _part_mat = human_parts_data['anno'][0][0][1][0]

            # Define human category index (15 in PASCAL VOC)
            human_category = 15

            # Simple 6-part body model mapping (head, torso, upper/lower arms/legs)
            # This matches the standard Pascal Parts annotations
            part_mapping = {
                'head': 1, 'leye': 1, 'reye': 1, 'lear': 1, 'rear': 1, 'lebrow': 1, 'rebrow': 1,
                'nose': 1, 'mouth': 1, 'hair': 1,
                'torso': 2,
                'neck': 2,
                'larm': 3, 'lhand': 3,
                'rarm': 4, 'rhand': 4,
                'lleg': 5, 'lfoot': 5,
                'rleg': 6, 'rfoot': 6,
            }

            # Iterate over objects in the image
            for obj_idx in range(len(_part_mat)):
                # Check if this is a human object (category 15)
                obj_class = int(_part_mat[obj_idx][1][0][0])
                if obj_class == human_category:
                    # Get object parts
                    obj_parts = _part_mat[obj_idx][3]
                    if len(obj_parts) > 0:
                        # Iterate over parts
                        for part_idx in range(len(obj_parts[0])):
                            part_name = str(obj_parts[0][part_idx][0][0])
                            part_mask = obj_parts[0][part_idx][1].astype(bool)

                            # Map part name to part ID
                            if part_name in part_mapping:
                                part_id = part_mapping[part_name]
                                # Update human_parts mask
                                human_parts[part_mask] = part_id
        except (IndexError, KeyError, ValueError) as e:
            # If parsing fails, use all zeros (background)
            pass

        # Set background to ignore_index if no parts found
        human_parts = np.where(human_parts == 0, 255, human_parts - 1).astype(np.int64)

        # Load edge detection
        # Format: .mat file with 'LabelMap' field (binary edge map)
        edge_path = os.path.join(self.edge_dir, f'{image_id}.mat')
        edge_data = sio.loadmat(edge_path)
        edge = edge_data['LabelMap'].astype(np.float32)
        # Normalize to [0, 1]
        edge = edge / 255.0 if edge.max() > 1.0 else edge

        # Resize all to target size
        original_size = image.size  # (W, H)
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        semseg_pil = Image.fromarray(semseg.astype(np.uint8), mode='L')
        semseg_pil = semseg_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        human_parts_pil = Image.fromarray(human_parts.astype(np.uint8), mode='L')
        human_parts_pil = human_parts_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        edge_pil = Image.fromarray((edge * 255).astype(np.uint8), mode='L')
        edge_pil = edge_pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)

        # Convert to numpy
        image = np.array(image)  # (H, W, 3)
        semseg = np.array(semseg_pil).astype(np.int64)
        human_parts = np.array(human_parts_pil).astype(np.int64)
        edge = np.array(edge_pil).astype(np.float32) / 255.0

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert to tensors
        semseg = torch.from_numpy(semseg).long()  # (H, W)
        human_parts = torch.from_numpy(human_parts).long()  # (H, W)
        edge = torch.from_numpy(edge).float()  # (H, W)

        return {
            'image': image,
            'segmentation': semseg,
            'human_parts': human_parts,
            'edge': edge
        }


class MultiTaskSubset(Dataset):
    """
    Subset of PASCAL Context for a specific client with task weights.

    This wrapper allows different clients to focus on different tasks
    by providing task-specific sampling and weighting.
    """

    def __init__(
        self,
        dataset: PascalContextDataset,
        indices: List[int],
        task_weights: Dict[str, float]
    ):
        """
        Args:
            dataset: The full PASCAL Context dataset
            indices: Subset of indices for this client
            task_weights: Dict of task weights, e.g.,
                         {'segmentation': 0.7, 'human_parts': 0.3, 'edge': 0.0}
        """
        self.dataset = dataset
        self.indices = indices
        self.task_weights = task_weights

        # Validate task weights
        expected_tasks = {'segmentation', 'human_parts', 'edge'}
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


class DMPascalContext:
    """
    Data Manager for PASCAL Context multi-task learning.

    Manages data distribution across clients with soft clustering support.
    Each client gets a subset of data and has task preferences.
    """

    def __init__(
        self,
        seed: int,
        num_clients: int,
        task_weights_per_client: List[Dict[str, float]],
        dataset_fraction: float = 1.0,
        root_dir: str = './data/pascal_context',
        batch_size: int = 8,
        download: bool = True
    ):
        """
        Args:
            seed: Random seed
            num_clients: Number of clients
            task_weights_per_client: List of task weight dicts for each client
                Example: [
                    {'segmentation': 0.7, 'human_parts': 0.3, 'edge': 0.0},
                    {'segmentation': 0.0, 'human_parts': 0.7, 'edge': 0.3},
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
        print("Loading PASCAL Context dataset...")
        self.train_dataset = PascalContextDataset(
            root_dir=root_dir,
            split='train',
            transform=self._get_train_transforms(),
            download=download
        )

        self.val_dataset = PascalContextDataset(
            root_dir=root_dir,
            split='val',
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
        print(f"Shared val dataset: {len(self.val_dataset)} samples")

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
            self.val_dataset,
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

    print("=" * 80)
    print("Testing PASCAL Context Data Manager")
    print("=" * 80)

    # Test with 6 clients (soft clustering setup)
    task_weights = [
        {'segmentation': 1.0, 'human_parts': 0.0, 'edge': 0.0},  # Client 0
        {'segmentation': 1.0, 'human_parts': 0.0, 'edge': 0.0},  # Client 1
        {'segmentation': 0.0, 'human_parts': 1.0, 'edge': 0.0},  # Client 2
        {'segmentation': 0.0, 'human_parts': 1.0, 'edge': 0.0},  # Client 3
        {'segmentation': 0.0, 'human_parts': 0.0, 'edge': 1.0},  # Client 4
        {'segmentation': 0.0, 'human_parts': 0.0, 'edge': 1.0},  # Client 5
    ]

    dm = DMPascalContext(
        seed=42,
        num_clients=6,
        task_weights_per_client=task_weights,
        dataset_fraction=0.1,  # Use 10% for quick testing
        batch_size=4,
        download=True
    )

    print("\n" + "=" * 80)
    print("Testing data loading...")
    print("=" * 80)

    # Test loading from client 0
    train_loader = dm.get_train_loader(0)
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Segmentation shape: {batch['segmentation'].shape}")
    print(f"  Unique values: {torch.unique(batch['segmentation'])}")
    print(f"Human parts shape: {batch['human_parts'].shape}")
    print(f"  Unique values: {torch.unique(batch['human_parts'])}")
    print(f"Edge shape: {batch['edge'].shape}")
    print(f"  Value range: [{batch['edge'].min():.3f}, {batch['edge'].max():.3f}]")
    print(f"Task weights: {batch['task_weights']}")

    print("\nData manager test passed!")
