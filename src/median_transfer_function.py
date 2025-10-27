"""
Median/Mean Transfer Function implementation.
"""

import os
import pickle
from typing import Literal, Optional, Union

import xarray as xr


class RatioTransferFunction():
    """Transfer function as the ratio of pressure and strain.

    Spectral transfer function calculated from a statistic (mean or
    median) of the ratio of pressure and strain spectra. Estimate the
    transfer function using the `calculate` method. Pressure spectra
    predictions are calculated by calling the RatioTransferFunction
    object with new strain spectra as an argument.

    Args:
        transfer_function (Optional[xr.DataArray]): A previously
            calculated transfer function (used if restoring).
    """
    def __init__(self, transfer_function: Optional[xr.DataArray] = None):
        # If not restoring from the `load` method, this is assigned by
        # the `calculate` method, which assigns a DataArray to
        # self.transfer_function with statistic stored in its attr dict.
        self.transfer_function = transfer_function

    def calculate(
        self,
        pressure_spectra: xr.DataArray,
        strain_spectra: xr.DataArray,
        statistic: Literal['mean', 'median'] = 'median',
        frequency_dim: str = 'frequency',
    ) -> None:
        """Calculate the transfer function.

        Inputs `pressure_spectra` and `strain_spectra` must have a
        `frequency_dim` dimension.  Remaining dimensions will be reduced by
        `statistic`. New pressure spectra can be calculated by calling the
        RatioTransferFunction object with new strain spectra as an argument.

        Args:
            pressure_spectra (xr.DataArray): Ground truth pressure
                spectral density with minimal dimension `frequency_dim`.
            strain_spectra (xr.DataArray): DAS-measured strain spectral
                density with minimal dimension `frequency_dim`.
            statistic (Literal['mean', 'median'], optional): Xarray
                function used to reduce remaining dimensions. Defaults
                to 'median'.
            frequency_dim (str, optional): Name of the frequency
                dimension.
        """
        # Frequency should be the last dimension.
        pressure_spectra = pressure_spectra.transpose(..., frequency_dim)
        strain_spectra = strain_spectra.transpose(..., frequency_dim)

        # Calculate all transfer functions. All dimensions, except frequency,
        # need to be reduced.
        all_tf = (pressure_spectra / strain_spectra)
        all_tf = all_tf.transpose(..., frequency_dim)
        dim_to_reduce = all_tf.dims[:-1]

        # Reduce along `dim_to_reduce` using the specified statistic.
        if statistic == 'median':
            self.transfer_function = all_tf.median(dim=dim_to_reduce)
        elif statistic == 'mean':
            self.transfer_function = all_tf.mean(dim=dim_to_reduce)
        else:
            raise ValueError(f"{statistic} not supported.")

        # Assign the statistic to the DataArray's attributes
        self.transfer_function.attrs['statistic'] = statistic

    def __call__(self, new_strain_spectra: xr.DataArray) -> xr.DataArray:
        """Return a pressure spectral density estimate from newly observed
        strain spectra.

        Args:
            new_strain_spectra (xr.DataArray): New DAS-measured strain spectral
            density.

        Returns:
            xr.DataArray: Pressure spectral density estimate.
        """
        if self.transfer_function is not None:
            return self.transfer_function * new_strain_spectra
        else:
            return xr.DataArray()

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return (f'{class_name}(\n'
                f'transfer_function={self.transfer_function}\n)')

    def state_dict(self) -> dict:
        """Return a transfer function state dictionary."""
        if self.transfer_function is not None:
            return self.transfer_function.to_dict()
        else:
            return dict()

    def save(self, path: os.PathLike) -> None:
        """Serialize the transfer function to `path` as a pickle file. """
        with open(path, 'wb') as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: os.PathLike):
        """Load a serialzed transfer function from `path`. """
        with open(path, 'rb') as handle:
            state_dict = pickle.load(handle)
        transfer_function = xr.DataArray.from_dict(state_dict)
        return RatioTransferFunction(transfer_function=transfer_function)
