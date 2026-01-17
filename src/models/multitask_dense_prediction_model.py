"""
Multi-Task Dense Prediction Model for NYU Depth V2

Supports three tasks:
- Depth estimation (regression)
- Semantic segmentation (13-class classification)
- Surface normal prediction (3D vector prediction)

Architecture:
- Shared backbone (ResNet-50)
- Task-specific decoders/heads for each task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class DepthHead(nn.Module):
    """
    Decoder for depth estimation (regression task).

    Upsamples backbone features to full resolution and predicts depth map.
    """

    def __init__(self, in_channels: int = 2048, out_size: Tuple[int, int] = (288, 384)):
        """
        Args:
            in_channels: Number of input channels from backbone
            out_size: (height, width) of output depth map
        """
        super(DepthHead, self).__init__()
        self.out_size = out_size

        # Decoder layers (progressive upsampling)
        self.decoder = nn.Sequential(
            # 2048 -> 1024
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 1024 -> 512
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Final prediction layer (64 -> 1 depth channel)
        self.pred = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)  # Depth is always positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) backbone features

        Returns:
            depth: (B, 1, out_H, out_W) predicted depth map
        """
        x = self.decoder(x)
        depth = self.pred(x)

        # Resize to target size if needed
        if depth.shape[2:] != self.out_size:
            depth = F.interpolate(depth, size=self.out_size, mode='bilinear', align_corners=False)

        return depth


class SegmentationHead(nn.Module):
    """
    Decoder for semantic segmentation (13-class classification).

    Similar to U-Net decoder but without skip connections for simplicity.
    """

    def __init__(self, in_channels: int = 2048, num_classes: int = 13,
                 out_size: Tuple[int, int] = (288, 384)):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_classes: Number of segmentation classes
            out_size: (height, width) of output segmentation map
        """
        super(SegmentationHead, self).__init__()
        self.num_classes = num_classes
        self.out_size = out_size

        # Decoder layers
        self.decoder = nn.Sequential(
            # 2048 -> 1024
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 1024 -> 512
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Segmentation prediction (64 -> num_classes)
        self.pred = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) backbone features

        Returns:
            logits: (B, num_classes, out_H, out_W) class logits
        """
        x = self.decoder(x)
        logits = self.pred(x)

        # Resize to target size if needed
        if logits.shape[2:] != self.out_size:
            logits = F.interpolate(logits, size=self.out_size, mode='bilinear', align_corners=False)

        return logits


class SurfaceNormalHead(nn.Module):
    """
    Decoder for surface normal prediction (3D unit vector).

    Predicts (x, y, z) components of surface normals.
    """

    def __init__(self, in_channels: int = 2048, out_size: Tuple[int, int] = (288, 384)):
        """
        Args:
            in_channels: Number of input channels from backbone
            out_size: (height, width) of output normal map
        """
        super(SurfaceNormalHead, self).__init__()
        self.out_size = out_size

        # Decoder layers (same as depth head)
        self.decoder = nn.Sequential(
            # 2048 -> 1024
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 1024 -> 512
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 512 -> 256
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 256 -> 128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # 128 -> 64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        # Normal prediction (64 -> 3 channels for x, y, z)
        self.pred = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) backbone features

        Returns:
            normals: (B, 3, out_H, out_W) unit normal vectors
        """
        x = self.decoder(x)
        normals = self.pred(x)

        # Resize to target size if needed
        if normals.shape[2:] != self.out_size:
            normals = F.interpolate(normals, size=self.out_size, mode='bilinear', align_corners=False)

        # Normalize to unit length
        normals = F.normalize(normals, p=2, dim=1)

        return normals


class MultiTaskModel(nn.Module):
    """
    Complete multi-task model with shared backbone and task-specific heads.
    """

    def __init__(
        self,
        task_weights: Dict[str, float],
        num_seg_classes: int = 13,
        out_size: Tuple[int, int] = (288, 384),
        pretrained: bool = True,
        num_human_parts_classes: int = 6,  # For Pascal Context
        gradient_clip_norm: float = None,  # Optional gradient clipping
        lr: float = 0.001  # Learning rate
    ):
        """
        Args:
            task_weights: Dict of task weights
                NYU V2: {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0}
                Pascal: {'segmentation': 0.7, 'human_parts': 0.3, 'edge': 0.0}
            num_seg_classes: Number of segmentation classes (13 for NYU, 59 for Pascal)
            out_size: (height, width) of output maps
            pretrained: Whether to use ImageNet pretrained backbone
            num_human_parts_classes: Number of human parts classes (for Pascal Context)
            gradient_clip_norm: Max norm for gradient clipping (None = no clipping)
            lr: Learning rate for optimizer. Default: 0.001
        """
        super(MultiTaskModel, self).__init__()

        self.task_weights = task_weights
        self.out_size = out_size
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Detect dataset type from task keys
        task_keys = set(task_weights.keys())
        self.is_pascal_context = 'human_parts' in task_keys or 'edge' in task_keys

        # Shared backbone: ResNet-50 (without final FC layer)
        resnet50 = models.resnet50(pretrained=pretrained)

        # Remove final pooling and FC layers
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

        # Backbone output: (B, 2048, H/32, W/32) for input (B, 3, H, W)
        backbone_out_channels = 2048

        # Task-specific heads based on dataset type
        if self.is_pascal_context:
            # Pascal Context: segmentation (59), human_parts (6), edge (1)
            self.seg_head = SegmentationHead(backbone_out_channels, num_seg_classes, out_size)
            self.human_parts_head = SegmentationHead(backbone_out_channels, num_human_parts_classes, out_size)
            self.edge_head = DepthHead(backbone_out_channels, out_size)  # Reuse depth head for edge (single channel)

            # Loss functions for Pascal Context
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.human_parts_criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.edge_criterion = nn.BCEWithLogitsLoss()  # Binary cross entropy for edge
        else:
            # NYU V2: depth, segmentation (13), normal
            self.depth_head = DepthHead(backbone_out_channels, out_size)
            self.seg_head = SegmentationHead(backbone_out_channels, num_seg_classes, out_size)
            self.normal_head = SurfaceNormalHead(backbone_out_channels, out_size)

            # Loss functions for NYU V2
            self.depth_criterion = nn.L1Loss()  # MAE for depth
            self.seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
            self.normal_criterion = self._cosine_similarity_loss  # Cosine for normals

        # Optimizer (will train all parameters)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)

        self.to(self.device)

    def _cosine_similarity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss for surface normals.

        Loss = 1 - cos(pred, target) averaged over all pixels
        """
        # Compute cosine similarity per pixel
        cos_sim = F.cosine_similarity(pred, target, dim=1)  # (B, H, W)

        # Loss is 1 - similarity (want to maximize similarity)
        loss = 1.0 - cos_sim.mean()

        return loss

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through shared backbone and all task heads.

        Args:
            x: (B, 3, H, W) input images
            return_all: If True, return all task predictions. If False, only active tasks.

        Returns:
            dict: Task predictions
                NYU V2: {'depth': (B, 1, H, W), 'segmentation': (B, num_classes, H, W), 'normal': (B, 3, H, W)}
                Pascal: {'segmentation': (B, 59, H, W), 'human_parts': (B, 6, H, W), 'edge': (B, 1, H, W)}
        """
        # Shared backbone
        features = self.backbone(x)

        # Check for NaN in backbone
        assert not torch.isnan(features).any(), "NaN detected in backbone features!"

        # Task-specific heads based on dataset type
        outputs = {}

        if self.is_pascal_context:
            # Pascal Context tasks
            if return_all or self.task_weights.get('segmentation', 0) > 0:
                outputs['segmentation'] = self.seg_head(features)

            if return_all or self.task_weights.get('human_parts', 0) > 0:
                outputs['human_parts'] = self.human_parts_head(features)

            if return_all or self.task_weights.get('edge', 0) > 0:
                outputs['edge'] = self.edge_head(features)
        else:
            # NYU V2 tasks
            if return_all or self.task_weights.get('depth', 0) > 0:
                outputs['depth'] = self.depth_head(features)

            if return_all or self.task_weights.get('segmentation', 0) > 0:
                outputs['segmentation'] = self.seg_head(features)

            if return_all or self.task_weights.get('normal', 0) > 0:
                outputs['normal'] = self.normal_head(features)

        return outputs

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Dict of task predictions
            targets: Dict of ground truth targets

        Returns:
            total_loss: Weighted sum of task losses
            loss_dict: Dict of individual task losses (for logging)
        """
        loss_dict = {}
        total_loss = 0.0

        if self.is_pascal_context:
            # Pascal Context tasks
            # Segmentation loss
            if self.task_weights.get('segmentation', 0) > 0 and 'segmentation' in predictions:
                seg_loss = self.seg_criterion(predictions['segmentation'], targets['segmentation'])
                loss_dict['segmentation'] = seg_loss.item()
                total_loss += self.task_weights['segmentation'] * seg_loss

            # Human parts loss
            if self.task_weights.get('human_parts', 0) > 0 and 'human_parts' in predictions:
                human_parts_loss = self.human_parts_criterion(predictions['human_parts'], targets['human_parts'])
                # Handle NaN (occurs when all pixels in batch are ignore_index=255)
                if not torch.isnan(human_parts_loss):
                    loss_dict['human_parts'] = human_parts_loss.item()
                    total_loss += self.task_weights['human_parts'] * human_parts_loss
                else:
                    # Skip this batch - no valid pixels to learn from
                    loss_dict['human_parts'] = 0.0

            # Edge loss
            if self.task_weights.get('edge', 0) > 0 and 'edge' in predictions:
                # Squeeze edge predictions to match target shape: (B, 1, H, W) -> (B, H, W)
                edge_pred = predictions['edge'].squeeze(1)
                edge_loss = self.edge_criterion(edge_pred, targets['edge'])
                loss_dict['edge'] = edge_loss.item()
                total_loss += self.task_weights['edge'] * edge_loss
        else:
            # NYU V2 tasks
            # Depth loss
            if self.task_weights.get('depth', 0) > 0 and 'depth' in predictions:
                depth_loss = self.depth_criterion(predictions['depth'], targets['depth'])
                loss_dict['depth'] = depth_loss.item()
                total_loss += self.task_weights['depth'] * depth_loss

            # Segmentation loss
            if self.task_weights.get('segmentation', 0) > 0 and 'segmentation' in predictions:
                seg_loss = self.seg_criterion(predictions['segmentation'], targets['segmentation'])
                loss_dict['segmentation'] = seg_loss.item()
                total_loss += self.task_weights['segmentation'] * seg_loss

            # Normal loss
            if self.task_weights.get('normal', 0) > 0 and 'normal' in predictions:
                normal_loss = self.normal_criterion(predictions['normal'], targets['normal'])
                loss_dict['normal'] = normal_loss.item()
                total_loss += self.task_weights['normal'] * normal_loss

        # Handle case where total_loss is still a float (no valid losses)
        if isinstance(total_loss, torch.Tensor):
            loss_dict['total'] = total_loss.item()
        else:
            # No valid losses - create a zero tensor for backward compatibility
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss_dict['total'] = 0.0

        return total_loss, loss_dict

    def train_model(self, train_loader, num_epochs: int = 1, verbose: bool = True):
        """
        Train the multi-task model.

        Args:
            train_loader: DataLoader with multi-task samples
            num_epochs: Number of training epochs
            verbose: Whether to show progress bar

        Returns:
            List of dicts with training metrics per epoch
        """
        self.train()
        results = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # Initialize task losses based on dataset type
            if self.is_pascal_context:
                epoch_task_losses = {'segmentation': 0.0, 'human_parts': 0.0, 'edge': 0.0}
                task_list = ['segmentation', 'human_parts', 'edge']
            else:
                epoch_task_losses = {'depth': 0.0, 'segmentation': 0.0, 'normal': 0.0}
                task_list = ['depth', 'segmentation', 'normal']
            num_batches = 0

            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") if verbose else train_loader

            for batch in iterator:
                # Move to device
                images = batch['image'].to(self.device)

                # Build targets dict based on dataset type
                if self.is_pascal_context:
                    targets = {
                        'segmentation': batch['segmentation'].to(self.device),
                        'human_parts': batch['human_parts'].to(self.device),
                        'edge': batch['edge'].to(self.device)
                    }
                else:
                    targets = {
                        'depth': batch['depth'].to(self.device),
                        'segmentation': batch['segmentation'].to(self.device),
                        'normal': batch['normal'].to(self.device)
                    }

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.forward(images, return_all=False)

                # Compute loss
                loss, loss_dict = self.compute_loss(predictions, targets)

                # Backward pass (only if loss has valid gradients)
                if isinstance(loss, torch.Tensor) and loss.requires_grad:
                    loss.backward()

                    # Apply gradient clipping if configured
                    if self.gradient_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(),
                            self.gradient_clip_norm
                        )

                    self.optimizer.step()

                    # Accumulate losses
                    epoch_loss += loss.item()
                else:
                    # Skip this batch - no valid loss to learn from
                    if isinstance(loss, torch.Tensor):
                        epoch_loss += loss.item()
                    else:
                        epoch_loss += 0.0
                for task in task_list:
                    if task in loss_dict:
                        epoch_task_losses[task] += loss_dict[task]

                num_batches += 1

                if verbose:
                    iterator.set_postfix({'loss': loss.item()})

            # Average losses
            avg_loss = epoch_loss / num_batches
            avg_task_losses = {k: v / num_batches for k, v in epoch_task_losses.items()}

            results.append({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                **avg_task_losses
            })

            if verbose:
                if self.is_pascal_context:
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                          f"Edge={avg_task_losses.get('edge', 0):.4f}, "
                          f"HumanParts={avg_task_losses.get('human_parts', 0):.4f}, "
                          f"Seg={avg_task_losses.get('segmentation', 0):.4f}")
                else:
                    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                          f"Depth={avg_task_losses.get('depth', 0):.4f}, "
                          f"Seg={avg_task_losses.get('segmentation', 0):.4f}, "
                          f"Normal={avg_task_losses.get('normal', 0):.4f}")

        return results

    def get_backbone_params(self) -> Dict[str, torch.Tensor]:
        """Get backbone parameters (for aggregation)"""
        return {name: param.data.clone() for name, param in self.backbone.named_parameters()}

    def set_backbone_params(self, params: Dict[str, torch.Tensor]):
        """Set backbone parameters (after aggregation)"""
        backbone_dict = self.backbone.state_dict()
        backbone_dict.update(params)
        self.backbone.load_state_dict(backbone_dict)

    def get_head_params(self, task: str) -> Dict[str, torch.Tensor]:
        """Get task-specific head parameters"""
        if task == 'depth':
            head = self.depth_head
        elif task == 'segmentation':
            head = self.seg_head
        elif task == 'normal':
            head = self.normal_head
        else:
            raise ValueError(f"Unknown task: {task}")

        return {name: param.data.clone() for name, param in head.named_parameters()}

    def set_head_params(self, task: str, params: Dict[str, torch.Tensor]):
        """Set task-specific head parameters"""
        if task == 'depth':
            head = self.depth_head
        elif task == 'segmentation':
            head = self.seg_head
        elif task == 'normal':
            head = self.normal_head
        else:
            raise ValueError(f"Unknown task: {task}")

        head_dict = head.state_dict()
        head_dict.update(params)
        head.load_state_dict(head_dict)


if __name__ == "__main__":
    """Test the multi-task model"""

    print("Testing Multi-Task Model")
    print("=" * 60)

    # Create model with pairwise task weights
    task_weights = {'depth': 0.7, 'segmentation': 0.3, 'normal': 0.0}

    model = MultiTaskModel(
        task_weights=task_weights,
        num_seg_classes=13,
        out_size=(288, 384),
        pretrained=False  # For quick testing
    )

    # Test forward pass
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 288, 384).to(model.device)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Device: {model.device}")

    with torch.no_grad():
        outputs = model(dummy_input)

    print("\nOutput shapes:")
    for task, output in outputs.items():
        print(f"  {task}: {output.shape}")

    # Test loss computation
    dummy_targets = {
        'depth': torch.rand(batch_size, 1, 288, 384).to(model.device),
        'segmentation': torch.randint(0, 13, (batch_size, 288, 384)).to(model.device),
        'normal': F.normalize(torch.randn(batch_size, 3, 288, 384), dim=1).to(model.device)
    }

    loss, loss_dict = model.compute_loss(outputs, dummy_targets)

    print(f"\nLoss computation:")
    for task, value in loss_dict.items():
        print(f"  {task}: {value:.4f}")

    # Test parameter extraction
    backbone_params = model.get_backbone_params()
    print(f"\nBackbone parameters: {len(backbone_params)} tensors")

    depth_head_params = model.get_head_params('depth')
    print(f"Depth head parameters: {len(depth_head_params)} tensors")

    print("\nMulti-task model test passed!")
