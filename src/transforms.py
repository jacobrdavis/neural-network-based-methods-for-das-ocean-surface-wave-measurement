"""
Data transformation and normalization classes.
"""
import os
from typing import Any, Dict, Optional, Protocol, Tuple, Union

import torch
from torch import nn


# Define the Protocol
class Transform(Protocol):
    """ Transform (and inverse transform) an input tensor. """
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the input tensor."""
        ...

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse transforms the input tensor."""
        ...


class LogTransform():
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return torch.log(tensor)

    @staticmethod
    def inverse_transform(tensor: torch.Tensor) -> torch.Tensor:
        return torch.exp(tensor)


class SqrtTransform():
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(tensor)

    @staticmethod
    def inverse_transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** 2


class PowerLawTransform():
    def __init__(self, exponent: float):
        self.exponent = exponent

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** self.exponent

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** (1 / self.exponent)


class Normalization(Protocol):
    """Normalize (and unnormalize) an input tensor."""
    @property
    def fitted(self) -> bool:
        """Checks if the normalizer has been fitted."""
        ...

    def fit(self, tensor: torch.Tensor) -> 'Normalization':
        """Fits the normalizer to the data."""
        ...

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the input tensor."""
        ...

    def inverse_transform(self, tensor_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transforms the input tensor."""
        ...


class MinMaxNormalization(nn.Module):
    """ PyTorch min-max normalizer.

    Scales input tensor to [0, 1] based on min and max values along
    specified dimension(s). Once the fit method has been called, the min
    and max are stored as buffers, allowing the normalization to be used
    with nn.Sequential and moved between devices. The normalization can
    be saved and loaded as part of a model's state_dict.

    Args:
        dim (int or tuple of int): Dimension(s) along which to
            compute the min and max. Default is 0.
        keepdim (bool): Whether to squeeze dimensions with size
            one. Default is False.
        eps (float): Small value to avoid division by zero when the
            range is zero. Default is 1e-12.

    Attributes:
        min_ (torch.Tensor): Minimum value(s) along specified dimension(s).
        max_ (torch.Tensor): Maximum value(s) along specified dimension(s).
        fitted (bool): Whether the normalizer has been fitted.

    Example:
       Create and fit a norm, then save and load the state_dict.
            >>> import torch
            >>> torch.manual_seed(0)
            >>> x = torch.randn(4, 3)
            >>> norm = MinMaxNormalization(dim=0).fit(x)
            >>> y = norm.transform(x)
            >>> torch.save(norm.state_dict(), "norm.pt")
            >>> torch.load("norm.pt")

    """
    def __init__(
        self,
        dim: Union[int, Tuple[int, ...]] = 0,
        keepdim: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.eps = eps

        # Add min and max to state_dict. Moves with .to()
        self.register_buffer("min_", torch.empty(0), persistent=True)
        self.register_buffer("max_", torch.empty(0), persistent=True)

        # TODO:
        # # Register a pre-hook to allow variable min_ and max_ shapes
        # def _pre_hook(state_dict, prefix, *_):
        #     for name in ("min_", "max_"):
        #         key = prefix + name
        #         if key in state_dict:
        #             # Adopt tensor of any shape, but keep key.
        #             # setattr(self, name, state_dict.pop(key))
        #             setattr(self, name, state_dict[key])
        #             state_dict[key] = getattr(self, name)

        # self._load_hook = self.register_load_state_dict_pre_hook(_pre_hook)

    @property
    def fitted(self) -> bool:
        """Check if the normalization has been fitted."""
        return (self.min_ is not None) and (self.max_ is not None)

    def fit(self, tensor: torch.Tensor) -> 'MinMaxNormalization':
        """Fit the normalizer to the data.

        Args:
            tensor (torch.Tensor): Input tensor to fit the normalizer.

        Returns:
            MinMaxNormalization: Fitted normalizer.
        """
        # amax/amin supports reducing on multiple dimensions
        self.min_ = tensor.amin(dim=self.dim, keepdim=self.keepdim).detach()
        self.max_ = tensor.amax(dim=self.dim, keepdim=self.keepdim).detach()
        return self

    def _range(self) -> torch.Tensor:
        rng_ = self.max_ - self.min_
        return torch.where(rng_.abs() < self.eps, torch.ones_like(rng_), rng_)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor to normalize.

        Raises:
            ValueError: If normalization has not been fitted.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if not self.fitted:
            raise ValueError('Normalization has not been fitted.')
        # Match device and dtype of input tensor.
        min_ = self.min_.to(tensor)
        rng_ = self._range().to(tensor)
        return (tensor - min_) / rng_

    def inverse_transform(self, tensor_norm: torch.Tensor) -> torch.Tensor:
        """Inverse transform the normalized tensor.

        Args:
            tensor_norm (torch.Tensor): _description_

        Raises:
            ValueError: If normalization has not been fitted.

        Returns:
            torch.Tensor: Inverse transformed tensor.
        """
        if not self.fitted:
            raise ValueError('Normalization has not been fitted.')
        # Match device and dtype of input tensor.
        min_ = self.min_.to(tensor_norm)
        rng_ = self._range().to(tensor_norm)
        return tensor_norm * rng_ + min_

    # For nn.Sequential compatibility.
    forward = transform

    def extra_repr(self) -> str:
        return (f"min_={self.min_},\n"
                f"max_={self.max_},\n"
                f"dim={self.dim},\n"
                f"keepdim={self.keepdim},\n"
                f"eps={self.eps}")

    # TODO: will want to update pytorch to > and use pre hook in init...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        # Adopt stats (any shape) and remove from dict so superclass won't size-check them
        for name in ("min_", "max_"):
            k = prefix + name

            if k in state_dict:
                setattr(self, name, state_dict[k])  # adopt any shape
                # do NOT pop; keep key so super() sees it and no "missing key"

        # Let Module load anything else
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                    missing_keys, unexpected_keys, error_msgs)
