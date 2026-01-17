"""
Training utilities for improved convergence in federated learning.

Includes:
- Learning rate schedulers (cosine, step, exponential)
- Early stopping mechanism
- Training state tracking
"""

import torch
import numpy as np
from typing import Optional, Dict, List
import math


class LearningRateScheduler:
    """
    Learning rate scheduler for federated learning rounds.

    Supports cosine annealing, step decay, and exponential decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedule_type: str = 'cosine',
        initial_lr: float = 0.001,
        min_lr: float = 0.0001,
        total_rounds: int = 30,
        warmup_rounds: int = 0,
        decay_rate: float = 0.1,
        decay_steps: int = 10
    ):
        """
        Initialize LR scheduler.

        Args:
            optimizer: PyTorch optimizer
            schedule_type: 'cosine', 'step', or 'exponential'
            initial_lr: Starting learning rate
            min_lr: Minimum learning rate
            total_rounds: Total number of training rounds
            warmup_rounds: Number of warmup rounds
            decay_rate: Decay rate for step/exponential
            decay_steps: Steps between decay for step scheduler
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.current_round = 0
        self.lr_history = []

    def step(self, round_num: int) -> float:
        """
        Update learning rate for the current round.

        Args:
            round_num: Current training round (0-indexed)

        Returns:
            New learning rate
        """
        self.current_round = round_num

        # Warmup phase
        if round_num < self.warmup_rounds:
            lr = self.initial_lr * (round_num + 1) / self.warmup_rounds
        else:
            adjusted_round = round_num - self.warmup_rounds
            adjusted_total = self.total_rounds - self.warmup_rounds

            if self.schedule_type == 'cosine':
                lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * \
                     (1 + math.cos(math.pi * adjusted_round / adjusted_total))

            elif self.schedule_type == 'step':
                lr = self.initial_lr * (self.decay_rate ** (adjusted_round // self.decay_steps))
                lr = max(lr, self.min_lr)

            elif self.schedule_type == 'exponential':
                lr = self.initial_lr * (self.decay_rate ** (adjusted_round / adjusted_total))
                lr = max(lr, self.min_lr)

            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.lr_history.append(lr)
        return lr

    def get_last_lr(self) -> float:
        """Get the last learning rate."""
        if self.lr_history:
            return self.lr_history[-1]
        return self.initial_lr

    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            'current_round': self.current_round,
            'lr_history': self.lr_history,
            'initial_lr': self.initial_lr,
            'min_lr': self.min_lr
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.current_round = state_dict['current_round']
        self.lr_history = state_dict['lr_history']
        self.initial_lr = state_dict['initial_lr']
        self.min_lr = state_dict['min_lr']


class EarlyStopping:
    """
    Early stopping to prevent overfitting and wasted computation.

    Monitors validation loss and stops training when no improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.005,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of rounds to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
            restore_best_weights: Whether to restore best model weights
            verbose: Whether to print status messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_round = 0
        self.early_stop = False
        self.best_weights = None

        self.score_history = []

    def __call__(self, score: float, model=None, round_num: int = 0) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (loss or accuracy)
            model: PyTorch model (needed if restore_best_weights=True)
            round_num: Current training round

        Returns:
            True if training should stop, False otherwise
        """
        self.score_history.append(score)

        # Determine if current score is better
        if self.mode == 'min':
            score_improved = (self.best_score is None or
                            score < self.best_score - self.min_delta)
        else:
            score_improved = (self.best_score is None or
                            score > self.best_score + self.min_delta)

        if score_improved:
            # New best score
            self.best_score = score
            self.best_round = round_num
            self.counter = 0

            # Save best weights
            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }

            if self.verbose:
                print(f"  Early stopping: New best score {score:.6f} at round {round_num}")

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                print(f"  Early stopping: No improvement for {self.counter}/{self.patience} rounds")

            if self.counter >= self.patience:
                self.early_stop = True

                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"EARLY STOPPING TRIGGERED")
                    print(f"{'='*60}")
                    print(f"Best score: {self.best_score:.6f} at round {self.best_round}")
                    print(f"Current score: {score:.6f} at round {round_num}")
                    print(f"{'='*60}\n")

                # Restore best weights if requested
                if self.restore_best_weights and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best model weights")

        return self.early_stop

    def state_dict(self) -> Dict:
        """Get state dictionary for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_round': self.best_round,
            'early_stop': self.early_stop,
            'score_history': self.score_history,
            'best_weights': self.best_weights
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_round = state_dict['best_round']
        self.early_stop = state_dict['early_stop']
        self.score_history = state_dict['score_history']
        self.best_weights = state_dict['best_weights']


class TrainingMonitor:
    """
    Monitor training progress and detect issues.

    Tracks loss, learning rate, and other metrics across rounds.
    """

    def __init__(self, window_size: int = 5):
        """
        Initialize training monitor.

        Args:
            window_size: Window for moving average computation
        """
        self.window_size = window_size
        self.metrics_history = []
        self.round_times = []

    def log_round(
        self,
        round_num: int,
        avg_loss: float,
        lr: float,
        client_losses: Optional[List[float]] = None
    ):
        """
        Log metrics for a training round.

        Args:
            round_num: Current round number
            avg_loss: Average loss across clients
            lr: Current learning rate
            client_losses: Individual client losses
        """
        metrics = {
            'round': round_num,
            'avg_loss': avg_loss,
            'lr': lr,
            'client_losses': client_losses if client_losses else []
        }

        if client_losses:
            metrics['std_loss'] = np.std(client_losses)
            metrics['min_loss'] = np.min(client_losses)
            metrics['max_loss'] = np.max(client_losses)

        self.metrics_history.append(metrics)

    def detect_oscillation(self) -> bool:
        """
        Detect if training is oscillating.

        Returns:
            True if oscillation detected
        """
        if len(self.metrics_history) < self.window_size:
            return False

        recent_losses = [m['avg_loss'] for m in self.metrics_history[-self.window_size:]]

        # Count direction changes
        direction_changes = 0
        for i in range(1, len(recent_losses) - 1):
            if (recent_losses[i] - recent_losses[i-1]) * (recent_losses[i+1] - recent_losses[i]) < 0:
                direction_changes += 1

        # Oscillation if more than half the windows show direction change
        return direction_changes >= self.window_size // 2

    def get_moving_average(self, key: str = 'avg_loss', window: int = None) -> float:
        """
        Get moving average of a metric.

        Args:
            key: Metric key
            window: Window size (default: self.window_size)

        Returns:
            Moving average
        """
        if window is None:
            window = self.window_size

        if len(self.metrics_history) < window:
            values = [m[key] for m in self.metrics_history if key in m]
        else:
            values = [m[key] for m in self.metrics_history[-window:] if key in m]

        return np.mean(values) if values else 0.0

    def print_summary(self):
        """Print training summary."""
        if not self.metrics_history:
            print("No training metrics recorded")
            return

        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total rounds: {len(self.metrics_history)}")
        print(f"Initial loss: {self.metrics_history[0]['avg_loss']:.6f}")
        print(f"Final loss: {self.metrics_history[-1]['avg_loss']:.6f}")

        best_round = min(self.metrics_history, key=lambda x: x['avg_loss'])
        print(f"Best loss: {best_round['avg_loss']:.6f} at round {best_round['round']}")

        improvement = (self.metrics_history[0]['avg_loss'] -
                      self.metrics_history[-1]['avg_loss']) / self.metrics_history[0]['avg_loss'] * 100
        print(f"Overall improvement: {improvement:.2f}%")

        if self.detect_oscillation():
            print("\n[WARN]  WARNING: Oscillation detected in recent rounds")

        print(f"{'='*60}\n")


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict
) -> Optional[LearningRateScheduler]:
    """
    Create LR scheduler from config.

    Args:
        optimizer: PyTorch optimizer
        config: Training config dictionary

    Returns:
        LR scheduler or None if not configured
    """
    if 'lr_scheduler' not in config:
        return None

    lr_config = config.get('lr_scheduler', {})

    if not lr_config.get('enabled', True):
        return None

    return LearningRateScheduler(
        optimizer=optimizer,
        schedule_type=lr_config.get('type', 'cosine'),
        initial_lr=config.get('learning_rate', 0.001),
        min_lr=lr_config.get('min_lr', 0.0001),
        total_rounds=config.get('num_rounds', 30),
        warmup_rounds=lr_config.get('warmup_rounds', 0),
        decay_rate=lr_config.get('decay_rate', 0.1),
        decay_steps=lr_config.get('decay_steps', 10)
    )


def create_early_stopping(config: Dict) -> Optional[EarlyStopping]:
    """
    Create early stopping from config.

    Args:
        config: Training config dictionary

    Returns:
        Early stopping or None if not configured
    """
    if 'early_stopping' not in config:
        return None

    es_config = config.get('early_stopping', {})

    if not es_config.get('enabled', True):
        return None

    return EarlyStopping(
        patience=es_config.get('patience', 5),
        min_delta=es_config.get('min_delta', 0.005),
        mode=es_config.get('mode', 'min'),
        restore_best_weights=es_config.get('restore_best_weights', True),
        verbose=es_config.get('verbose', True)
    )
