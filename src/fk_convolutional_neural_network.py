"""
Frequency-wavenumber Convolutional Neural Network model, loss, norms,
and functions.
"""

from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import xarray as xr
import torch.nn.functional as F

from src import transforms


class FkConvolutionalNeuralNetwork(nn.Module):
    """ F-k convolutional neural network encoder-decoder model.

    The model's architecture is adapted from im2spec, a convolutional
    encoder-decoder network developed in the materials science domain
    (Kalinin et al., 2021; Roccapriore et al., 2021). See
    https://github.com/ziatdinovmax/im2spec and
    https://github.com/pycroscopy/atomai for original (and more general)
    implementations of the im2spec model.

    Args:
        dim_in: (Tuple[int, int]): Input f-k spectrum dimensions (k, f)
        dim_out: (int): Predicted seafloor pressure spectrum prediction
            dimension (f,)
        latent_dim: (int): Size of the latent space vector.
        nb_filters_enc: (int): Number of filters in the encoder.
        nb_filters_dec: (int): Number of filters in the decoder.

    References:
    Kalinin, S. V., Kelley, K., Vasudevan, R. K., & Ziatdinov, M.
        (2021). Toward Decoding the Relationship between Domain
        Structure and Functionality in Ferroelectrics via Hidden Latent
        Variables. ACS Applied Materials & Interfaces, 13(1), 1693–1703.
        https://doi.org/10.1021/acsami.0c15085

    Roccapriore, K. M., Ziatdinov, M., Cho, S. H., Hachtel, J. A., &
        Kalinin, S. V. (2021). Predictability of Localized Plasmonic
        Responses in Nanoparticle Assemblies. Small, 17(21), 2100181.
        https://doi.org/10.1002/smll.202100181

    """
    def __init__(
        self,
        dim_in: Tuple[int, int],
        dim_out: int,
        latent_dim: int,
        nb_filters_enc: int,
        nb_filters_dec: int,
    ):
        super().__init__()
        self.k, self.f = dim_in
        self.dim_out = dim_out
        self.latent_dim = latent_dim
        self.nb_filters_enc = nb_filters_enc
        self.nb_filters_dec = nb_filters_dec

        # Define encoder layers. The convolutional block has multiple
        # convolutional layers with activation and batch normalization.
        # The final linear layer maps the encoder output to the n-dim
        # latent space vector (`latent_dim`).

        # The convolutional layer kwargs are the same across all layers.
        # These are tunable hyperparameters, but the defaults work well.
        nb_block_layers = 3
        enc_conv_kwargs = dict(
            kernel_size=3,
            stride=1,
            padding=1,
            negative_slope=0.1,
        )
        self.enc_conv_layers = EncConvBlock(
            in_channels=1,
            out_channels=nb_filters_enc,
            nb_block_layers=nb_block_layers,
            **enc_conv_kwargs
        )

        # Linear layer maps encoder output to latent space
        self.enc_linear_in_size = nb_filters_enc * self.k * self.f
        self.enc_linear_layer = nn.Linear(
            self.enc_linear_in_size,
            latent_dim,
        )

        # Define decoder layers. The first linear layer maps the latent
        # space vector back to a higher-dimensional representation. The
        # decoder has both a dilated convolutional block and a standard
        # convolutional block. The final convolutional layer maps the
        # decoder output to seafloor pressure spectra.

        # Linear layer maps latent space to first decoder block
        self.dec_linear_layer = nn.Linear(
            latent_dim,
            nb_filters_dec * self.dim_out,
        )

        # Decoder dilated convolutional block
        nb_block_layers = 4
        dec_dilated_conv_kwargs = dict(
            kernel_size=3,
            stride=1,
            negative_slope=0.1,
        )
        dec_paddings = [1, 2, 3, 4]
        dec_dilations = [1, 2, 3, 4]
        self.dec_dilated_conv_layers = DecDilatedConvBlock(
            in_channels=nb_filters_dec,
            out_channels=nb_filters_dec,
            nb_block_layers=nb_block_layers,
            dilations=dec_dilations,
            paddings=dec_paddings,
            **dec_dilated_conv_kwargs
        )

        # Decoder convolutional block
        nb_block_layers = 1
        dec_conv_kwargs = dict(
            kernel_size=3,
            stride=1,
            padding=1,
            negative_slope=0.1,
        )
        self.dec_conv_layers = DecConvBlock(
            in_channels=nb_filters_dec,
            out_channels=1,
            nb_block_layers=1,
            **dec_conv_kwargs
        )

        # Final output convolutional layer
        self.output_conv_layer = nn.Conv1d(1, 1, 1)

    def encoder(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Encodes input f-k spectrum to latent space."""
        x = self.enc_conv_layers(feature_tensor)
        x = x.reshape(-1, self.enc_linear_in_size)
        return self.enc_linear_layer(x)

    def decoder(self, encoded_tensor: torch.Tensor) -> torch.Tensor:
        """Decodes latent space tensor to seafloor pressure spectra."""
        x = self.dec_linear_layer(encoded_tensor)
        x = x.reshape(-1, self.nb_filters_dec, self.dim_out)
        x = self.dec_dilated_conv_layers(x)
        x = self.dec_conv_layers(x)
        return self.output_conv_layer(x)

    def forward(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder-decoder model."""
        encoded_tensor = self.encoder(feature_tensor)
        decoded_tensor = self.decoder(encoded_tensor)
        return decoded_tensor.squeeze(dim=1)


class EncConvBlock(nn.Module):
    """Encoder convolutional block.

    Multi-layer encoder convolutional block (`nb_block_layers`) with
    LeakyReLU activation and batch normalization.  Each layer has the
    same hyperparameters (`kernel_size`, `stride`, `padding`,
    `negative_slope`).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        nb_block_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        block_layers = []
        for layer in range(nb_block_layers):
            # Any subsequent layers need to have input==output
            if layer > 0:
                in_channels = out_channels

            block_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            block_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            block_layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*block_layers)

    def forward(self, x):
        """Forward pass through the encoder convolutional block."""
        return self.block(x)


class DecDilatedConvBlock(nn.Module):
    """Decoder dilated convolutional block.

    Multi-layer dilated convolutional block (`nb_block_layers`) with
    `dilations` and `paddings` specified per layer (e.g.,
    `len(dilations)` == `nb_block_layers`). Each layer has the same
    hyperparameters (`kernel_size`, `stride`, `negative_slope`).

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        nb_block_layers: int,
        dilations: List[int],
        paddings: List[int],
        kernel_size: int = 3,
        stride: int = 1,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        block_layers = []
        for layer in range(nb_block_layers):
            # Any subsequent layers need to have input==output
            if layer > 0:
                in_channels = out_channels

            block_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=paddings[layer],
                    dilation=dilations[layer],
                )
            )
            block_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            block_layers.append(nn.BatchNorm1d(out_channels))

        self.block = nn.Sequential(*block_layers)

    def forward(self, x):
        """Forward pass through the dilated convolutional block."""
        # Stack outputs from each dilated layer and sum element-wise.
        # (Output shape is equivalent to the input shape)
        dilated_layers = []
        for conv_layer in self.block:
            x = conv_layer(x)
            dilated_layers.append(x.unsqueeze(-1))
        return torch.sum(torch.cat(dilated_layers, dim=-1), dim=-1)


class DecConvBlock(nn.Module):
    """Decoder convolutional block.

    Multi-layer decoder convolutional block (`nb_block_layers`) with
    LeakyReLU activation and batch normalization.  Each layer has the
    same hyperparameters (`kernel_size`, `stride`, `padding`,
    `negative_slope`).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        nb_block_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        negative_slope: float = 0.1,
    ):
        super().__init__()

        block_layers = []
        for layer in range(nb_block_layers):
            # Any subsequent layers need to have input==output
            if layer > 0:
                in_channels = out_channels

            block_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            block_layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            block_layers.append(nn.BatchNorm1d(out_channels))

        self.block = nn.Sequential(*block_layers)

    def forward(self, x):
        """Forward pass through the decoder convolutional block."""
        return self.block(x)


class SpectralLoss(nn.Module):
    """f-k convolutional neural network loss function.

    This is just a per-frequency mean squared error loss, but could be
    extended in the future.
    """
    def __init__(self):
        super(SpectralLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, input, target):
        return self.mse_loss(input, target)


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
        target_transform: Optional[transforms.Transform],
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


class FkConvolutionalNeuralNetworkDataset(torch.utils.data.Dataset):
    """PyTorch dataset for the f-k convolutional neural network model.

    Calling the dataset with an indexer, e.g., `dataset[0:10]`, returns
    a tuple of (features, targets) for the given indices.

    Model features are 2-D frequency-wavenumber spectral densities and
    targets are 1-D seafloor pressure spectral densities.

    Args:
        features (torch.Tensor): Model features.
        targets (torch.Tensor): Model targets.
        spectral_feature_norm (_type_, optional): Normalization for
            spectral features. Defaults to None.
        target_norm (_type_, optional): Normalization for target
            features. Defaults to None.

    """
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        spectral_feature_norm: Optional[transforms.Normalization] = None,
        target_norm: Optional[transforms.Normalization] = None,
    ):
        self.features = features
        self.targets = targets
        self.spectral_feature_norm = spectral_feature_norm
        self.target_norm = target_norm

    @property
    def norms(self):
        """Return the spectral and target norms as a tuple."""
        return (self.spectral_feature_norm, self.target_norm)

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
        if self.spectral_feature_norm is not None:
            feature_sample = self.spectral_feature_norm.transform(feature_sample)
        if self.target_norm is not None:
            target_sample = self.target_norm.transform(target_sample)
        return feature_sample, target_sample


def feature_tensor_from_dataset(
    das_ds: xr.Dataset,
    spectral_feature_transform: Optional[transforms.Transform] = None,
) -> torch.Tensor:
    """Build Pytorch feature tensor from a DAS Xarray dataset."""
    # Get feature and target DataArrays.
    strain_da = das_ds['strain_fk_spectral_density']

    # Cast DataArrays to PyTorch tensors.
    strain_tensor = torch.from_numpy(strain_da.to_numpy())

    # Squeeze singular time dimensions to accomodate PyTorch DataLoader.
    if strain_tensor.shape[0] == 1:
        strain_tensor = torch.squeeze(strain_tensor, 0)

    # Expand shape to add a single color channel.
    strain_tensor = torch.unsqueeze(strain_tensor, -3)

    # Apply transforms.
    if spectral_feature_transform is not None:
        strain_tensor = spectral_feature_transform.transform(strain_tensor)

    # Note: This has no effect, but could be used if scalar features
    # (e.g., depth, direction) are added in the future:
    # Concatenate features into a single tensor along the frequency dim.
    # Non-spectral features are broadcast in this dimension.
    feature_tensor = torch.concat([
            strain_tensor,
        ],
        dim=-1,
    )

    return feature_tensor.float()


def target_tensor_from_dataset(
    das_ds: xr.Dataset,
    target_transform: Optional[transforms.Transform] = None,
) -> torch.Tensor:
    """Build Pytorch target tensor from a DAS Xarray dataset."""

    # Get target DataArrays.
    target_da = das_ds['target_seafloor_pressure_spectral_density']

    # Cast DataArrays to PyTorch tensors.
    target_tensor = torch.from_numpy(target_da.to_numpy())

    # Squeeze singular time dimensions to accomodate PyTorch DataLoader.
    if target_tensor.shape[0] == 1:
        target_tensor = torch.squeeze(target_tensor, 0)

    # Apply transforms.
    if target_transform is not None:
        target_tensor = target_transform.transform(target_tensor)

    return target_tensor.float()
