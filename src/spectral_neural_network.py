"""
Spectral Neural Network model, loss, norms, and functions.
"""

from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import xarray as xr
import torch.nn.functional as F

from src import transforms


class SpectralNeuralNetwork(nn.Module):
    """3-layer (2 hidden, 1 output) fully connected neural network."""
    def __init__(
        self,
        dim_in: int,
        n_hidden_layers_1: int,
        n_hidden_layers_2: int,
        p_dropout: float,
        dim_out: int,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, n_hidden_layers_1),  # [in, out]
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(n_hidden_layers_1, n_hidden_layers_2),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(n_hidden_layers_2, dim_out),
         )

    def forward(self, x):
        return self.model(x)


class SpectralLoss(nn.Module):
    """Spectral neural network loss function.

    Balances linear and log MSE losses using weighting coefficient
    `alpha`.
    """
    def __init__(
        self,
        target_norm: transforms.Normalization,
        target_transform: transforms.Transform,
        alpha=0.5
    ):
        super(SpectralLoss, self).__init__()
        self.target_norm = target_norm
        self.target_transform = target_transform
        self.alpha = alpha

    def forward(self, prediction, target):
        # Calculate MSE loss for spectral comparison by frequency
        # using the log-transformed data.
        per_freq_loss = F.mse_loss(prediction, target)

        # Calculate a second per frequency loss using non-transformed
        # prediction and target data.
        input_no_norm = self.target_norm.inverse_transform(prediction)
        target_no_norm = self.target_norm.inverse_transform(target)
        input_no_log = self.target_transform.inverse_transform(input_no_norm)
        target_no_log = self.target_transform.inverse_transform(target_no_norm)

        # Create a new min-max normalizer for the non-log data. Min and
        # max are set from inverse transform of the log-normalized norm.
        target_norm_no_log = transforms.MinMaxNormalization()
        target_norm_no_log.min_ = self.target_transform.inverse_transform(self.target_norm.min_)
        target_norm_no_log.max_ = self.target_transform.inverse_transform(self.target_norm.max_)

        # Apply normalization and compute the second per-frequency loss.
        input_no_log = target_norm_no_log.transform(input_no_log)
        target_no_log = target_norm_no_log.transform(target_no_log)
        per_freq_loss_no_log = F.mse_loss(input_no_log, target_no_log)

        # Alpha is hyperparameter.
        return self.alpha * per_freq_loss + (1 - self.alpha) * per_freq_loss_no_log


class SpectralEvalMetric(nn.Module):
    """Seafloor pressure spectral density evaluation metric.

    Used as a metric for model selection that is independent from
    the loss, which may vary depending on the model and hyperparameters.

    Both prediction and target should be detached from the computational
    graph (disable gradient calculation prior to evaluating).
    """
    def __init__(
        self,
        target_norm: transforms.Normalization,
        target_transform: transforms.Transform,
    ):
        super(SpectralEvalMetric, self).__init__()
        self.target_norm = target_norm
        self.target_transform = target_transform

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Calculate mean squared error between prediction and target.

        Note: Metric is applied after inverting normalization and
        transforms.
        """
        input_no_transform = self._inverse_transform(prediction)
        target_no_transform = self._inverse_transform(target)
        return F.mse_loss(input_no_transform, target_no_transform)

    def _inverse_transform(self, pressure_torch: torch.Tensor) -> torch.Tensor:
        # Inverse the target normalization.
        pressure_torch_no_norm = self.target_norm.inverse_transform(pressure_torch)

        # Inverse the pressure spectral density transform. Tranformation
        # must happen AFTER the normalization is removed.
        if self.target_transform is not None:
            return self.target_transform.inverse_transform(pressure_torch_no_norm)

        return pressure_torch_no_norm


class SpectralNeuralNetworkDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the spectral neural network model.

    Calling the dataset with an indexer, e.g., `dataset[0:10]`, returns
    a tuple of (features, targets) for the given indices.

    Note: Assumes features are concatenated such that the first
    `start_spec` features are scalar (e.g., depth, direction), and the
    remaining features are spectral (e.g., strain spectra).

    Args:
        features (torch.Tensor): Model features.
        targets (torch.Tensor): Model targets.
        scalar_feature_norm (_type_, optional): Normalization for scalar
            features. Defaults to None.
        spectral_feature_norm (_type_, optional): Normalization for
            spectral features. Defaults to None.
        target_norm (_type_, optional): Normalization for target
            features. Defaults to None.
        start_spec (int, optional): Index where spectral features start.
            Defaults to 2.

    """
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        scalar_feature_norm: Optional[transforms.Normalization] = None,
        spectral_feature_norm: Optional[transforms.Normalization] = None,
        target_norm: Optional[transforms.Normalization] = None,
        start_spec: int = 2,
    ):
        self.features = features
        self.targets = targets
        self.scalar_feature_norm = scalar_feature_norm
        self.spectral_feature_norm = spectral_feature_norm
        self.target_norm = target_norm
        self.start_spec = start_spec

    @property
    def norms(self):
        """Return the scalar, spectral and target norms as a tuple."""
        return (self.scalar_feature_norm,
                self.spectral_feature_norm,
                self.target_norm)

    @property
    def any_norms_fitted(self) -> bool:
        """Check if any norms have been fitted.

        Note: Returns False if all norms are None.
        """
        return any(norm.fitted for norm in self.norms if norm is not None)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        # Inputs are cloned to avoid in-place modification when
        # normalizing. Detach ensures no gradients are tracked.
        feature_sample = self.features[idx].detach().clone()
        target_sample = self.targets[idx].detach().clone()
        return self._normalize(feature_sample, target_sample)

    def _normalize(
        self,
        feature_sample: torch.Tensor,
        target_sample: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scalar_feature_norm is not None:
            feature_sample[..., 0:self.start_spec] = self.scalar_feature_norm.transform(feature_sample[..., 0:self.start_spec])
        if self.spectral_feature_norm is not None:
            feature_sample[..., self.start_spec:] = self.spectral_feature_norm.transform(feature_sample[..., self.start_spec:])
        if self.target_norm is not None:
            target_sample = self.target_norm.transform(target_sample)
        return feature_sample, target_sample


def feature_tensor_from_dataset(
    das_ds: xr.Dataset,
    scalar_feature_transform: Optional[transforms.Transform] = None,
    spectral_feature_transform: Optional[transforms.Transform] = transforms.LogTransform(),
) -> torch.Tensor:
    """Build Pytorch feature tensor from a DAS Xarray dataset."""
    # Get feature DataArrays.
    strain_da = das_ds['strain_spectral_density']
    depth_da = das_ds['depth']
    direction_da = das_ds['cosine_squared_wave_direction']

    # Cast DataArrays to PyTorch tensors.
    strain_tensor = torch.from_numpy(strain_da.to_numpy())
    depth_tensor = torch.from_numpy(depth_da.to_numpy())
    direction_tensor = torch.from_numpy(direction_da.to_numpy())

    # Apply transforms.
    if scalar_feature_transform is not None:
        depth_tensor = scalar_feature_transform.transform(depth_tensor)
        direction_tensor = scalar_feature_transform.transform(direction_tensor)
    if spectral_feature_transform is not None:
        strain_tensor = spectral_feature_transform.transform(strain_tensor)

    # Concatenate features into a single tensor along the frequency dim.
    # Non-spectral features are broadcast in this dimension. Here,
    # strain spectral densities MUST be the final entry!
    feature_tensor = torch.concat([
            depth_tensor[..., None],
            direction_tensor[..., None],
            strain_tensor,
        ],
        dim=-1,
    )
    return feature_tensor.float()


def target_tensor_from_dataset(
    das_ds: xr.Dataset,
    target_transform: Optional[transforms.Transform] = transforms.LogTransform(),
) -> torch.Tensor:
    """Build Pytorch target tensor from a DAS Xarray dataset."""

    # Get target DataArrays.
    target_da = das_ds['target_seafloor_pressure_spectral_density']

    # Cast DataArrays to PyTorch tensors.
    target_tensor = torch.from_numpy(target_da.to_numpy())

    # Apply transforms.
    if target_transform is not None:
        target_tensor = target_transform.transform(target_tensor)

    return target_tensor.float()
