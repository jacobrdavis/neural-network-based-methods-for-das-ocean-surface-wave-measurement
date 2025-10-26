"""
Functions for training PyTorch models.
"""

from typing import Callable, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader


def train_regression_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    eval_metric: Optional[Callable] = None,
    device: Optional[torch.device] = None,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Train a PyTorch regression model in epochs.

    Returns a tuple containing lists of training losses, validation
    losses, training evaluation metrics, and validation evaluation
    metrics for each epoch.
    """
    train_losses = []
    val_losses = []
    train_eval_metrics = []
    val_eval_metrics = []

    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_eval = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating (1 epoch).
        for (features, targets) in train_loader:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions = model(features)

            batch_loss = loss(predictions, targets)
            train_loss = train_loss + batch_loss.item()

            if eval_metric is not None:
                with torch.no_grad():
                    batch_eval = eval_metric(predictions, targets)
                    train_eval = train_eval + batch_eval.item()

            batch_loss.backward()
            optimizer.step()

        # Calculate mean training loss for the epoch.
        train_loss_mean = train_loss / len(train_loader)
        train_losses.append(train_loss_mean)

        if eval_metric is not None:
            with torch.no_grad():
                train_eval_mean = train_eval / len(train_loader)
                train_eval_metrics.append(train_eval_mean)

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_eval = 0.0

        with torch.no_grad():
            for (features, targets) in val_loader:
                features, targets = features.to(device), targets.to(device)
                predictions = model(features)
                val_batch_loss = loss(predictions, targets)
                val_loss = val_loss + val_batch_loss.item()

                if eval_metric is not None:
                    batch_eval = eval_metric(predictions, targets)
                    val_eval = val_eval + batch_eval.item()

            # Calculate mean validation loss for the epoch.
            val_loss_mean = val_loss / len(val_loader)
            val_losses.append(val_loss_mean)

            if eval_metric is not None:
                val_eval_mean = val_eval / len(val_loader)
                val_eval_metrics.append(val_eval_mean)

    # (Any checkpointing would go here.)

    return train_losses, val_losses, train_eval_metrics, val_eval_metrics


def apply_regression_model(
    model: nn.Module,
    features: torch.Tensor
) -> torch.Tensor:
    """Set a model to evaluation mode and apply it to features."""
    model.eval()
    # Disable gradient calculations during inference.
    with torch.no_grad():
        output = model(features)
    return output
