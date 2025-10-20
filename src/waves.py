"""
Ocean surface wave functions for DAS Xarray Datasets.

Functions with leading underscores are numpy-based and usually have
accompanying Xarray wrappers.
"""
from typing import Any, Callable, Tuple, Optional, Union, List, Literal

import numpy as np
import pandas as pd

import xarray as xr
from scipy.optimize import newton, minimize
from scipy.interpolate import RegularGridInterpolator

from src import das_xarray_helpers

GRAVITY = 9.81


def spectral_moment(
    da: xr.DataArray,
    n: int,
    frequency_min: Optional[float] = None,
    frequency_max: Optional[float] = None,
    frequency_coord: str = 'frequency',
) -> xr.DataArray:
    """ Calculate the `n`th spectral moment and return as a DataArray. """
    valid_frequencies = das_xarray_helpers.between(
        da[frequency_coord],
        lower=frequency_min,
        upper=frequency_max,
        as_bool=True
    )

    moment_da = (
        (da * da[frequency_coord]**n)
        .where(valid_frequencies, drop=True)
        .integrate(coord=frequency_coord)
    )
    moment_da.attrs['description'] = f'Spectral moment n={n}'
    moment_da.attrs['frequency_min'] = frequency_min
    moment_da.attrs['frequency_max'] = frequency_max
    return moment_da


def spectral_var(
    da: xr.DataArray,
    frequency_min: Optional[float] = 0.05,
    frequency_max: Optional[float] = 0.5,
    frequency_coord: str = 'frequency',
) -> xr.DataArray:
    """ Return spectral variance as a DataArray. """
    var_da = spectral_moment(
        da,
        n=0,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_coord=frequency_coord,
    )
    var_da.attrs['description'] = 'Spectral variance'
    var_da.attrs['frequency_min'] = frequency_min
    var_da.attrs['frequency_max'] = frequency_max
    return var_da


def energy_period(
    da,
    frequency_min: Optional[float] = 0.05,
    frequency_max: Optional[float] = 0.5,
    frequency_coord: str = 'frequency',
    return_as_frequency=False,
) -> xr.DataArray:
    """
    Calculate energy-weighted frequency as the ratio of the first and
    zeroth moments of the one-dimensional frequency spectrum.
    """
    moment_0 = spectral_moment(
        da,
        n=0,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_coord=frequency_coord,
    )
    moment_1 = spectral_moment(
        da,
        n=1,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_coord=frequency_coord,
    )
    energy_frequency_da = moment_1 / moment_0

    if return_as_frequency:
        energy_frequency_da.attrs['description'] = 'Energy-weighted frequency'
        energy_frequency_da.attrs['frequency_min'] = frequency_min
        energy_frequency_da.attrs['frequency_max'] = frequency_max
        return energy_frequency_da

    energy_period_da = energy_frequency_da**(-1)
    energy_period_da.attrs['description'] = 'Energy-weighted period'
    energy_period_da.attrs['frequency_min'] = frequency_min
    energy_period_da.attrs['frequency_max'] = frequency_max
    return energy_period_da


def significant_wave_height(
    da: xr.DataArray,
    frequency_min: Optional[float] = 0.05,
    frequency_max: Optional[float] = 0.5,
    frequency_coord: str = 'frequency',
) -> xr.DataArray:
    """ Return significant wave height as a DataArray.

    Calculate significant wave height as four times the square root
    of spectral variance.

    Note: NaNs must be filled prior to using this method (e.g. using
    `fillna` or `where`).

    """
    variance_da = spectral_var(
        da,
        frequency_min=frequency_min,
        frequency_max=frequency_max,
        frequency_coord=frequency_coord,
    )
    standard_deviation = np.sqrt(variance_da)
    sig_height_da = 4 * standard_deviation

    sig_height_da.attrs['description'] = 'Significant wave height'
    sig_height_da.attrs['frequency_min'] = frequency_min
    sig_height_da.attrs['frequency_max'] = frequency_max
    return sig_height_da


def pressure_spectra_to_surface_spectra(
    depth: xr.DataArray,
    frequency: xr.DataArray,
    depth_convention: Literal['pos_up', 'pos_down'] = 'pos_up',
    apply_ufunc_kwargs: Optional[dict] = None,
    **kwargs,
) -> xr.DataArray:
    """ Return a pressure to surface elevation variance transfer function.

    The transfer function maps pressure spectral density to surface
    elevation spectral density using linear wave theory. Depths are
    converted to positive down.

    Args:
        depth (xr.DataArray): Water depths.
        frequency (xr.DataArray): Spectral frequency bins.
        depth_convention (Literal['pos_up', 'pos_down'], optional):
            Water depth sign convention. Defaults to 'pos_up'.
        apply_ufunc_kwargs (Optional[dict], optional): Keyword
            arguments passed to xr.apply_ufunc. Defaults to None.

    Returns:
        xr.DataArray: Transfer function.
    """
    if apply_ufunc_kwargs is None:
        apply_ufunc_kwargs = {}

    # Positive up depths are converted to positive down (* -1).
    if depth_convention == 'pos_up':
        depth_conversion = -1
    else:
        depth_conversion = 1

    # Function expects depth and frequency to have matching dimensions.
    frequency = frequency.expand_dims(time=depth['time'])
    depth = depth.expand_dims(frequency=frequency['frequency'])

    transfer_function = xr.apply_ufunc(
        _pressure_to_surface_transfer_function,
        frequency,
        depth * depth_conversion,
        kwargs=kwargs,
        input_core_dims=[["time", "frequency"], ["time", "frequency"]],
        output_core_dims=[["time", "frequency"]],
        vectorize=True,
        **apply_ufunc_kwargs,
    )

    # Transfer function is unitless.
    transfer_function.attrs['description'] = (
        'Pressure to surface elevation variance transfer function'
    )
    transfer_function.attrs['units'] = '(m^2/Hz)/(m^2/Hz)'

    return transfer_function


def direction_from_fk_spectrum(
    fk_spectral_density: xr.DataArray,
    wavenumber: xr.DataArray,
    frequency: xr.DataArray,
    depth: xr.DataArray,
    depth_convention: Literal['pos_up', 'pos_down'] = 'pos_up',
    apply_ufunc_kwargs: Optional[dict] = None,
    **kwargs,
) -> xr.DataArray:
    """Return wave direction from a frequency-wavenumber spectrum.

    Wave direction relative to the along-cable axis is estimated from
    frequency-wavenumber spectral density using nonlinear least squares
    optimization.

    Note: Input wavenumbers and frequencies are NOT angular (i.e., not
    multiplied by 2*pi). This method currently assumes no currents and
    does not leverage beamforming information.

    Args:
        fk_spectral_density (xr.DataArray): Frequency-wavenumber spectra.
        wavenumber (xr.DataArray): Wavenumber bins.
        frequency (xr.DataArray): Frequency bins.
        depth (xr.DataArray): Water depths.
        depth_convention (Literal['pos_up', 'pos_down'], optional):
            water depth sign convention. Defaults to 'pos_up'.
        apply_ufunc_kwargs (Optional[dict], optional): keyword
            arguments passed to xr.apply_ufunc. Defaults to None.

    Returns:
        xr.DataArray: Wave directions in degrees.
    """
    if apply_ufunc_kwargs is None:
        apply_ufunc_kwargs = {}

    if depth_convention == 'pos_up':
        depth_conversion = -1
    else:
        depth_conversion = 1

    wave_direction = xr.apply_ufunc(
        _direction_from_fk_spectrum,
        fk_spectral_density,
        wavenumber,
        frequency,
        depth * depth_conversion,
        kwargs=kwargs,
        input_core_dims=[
            [frequency, wavenumber],
            [wavenumber],
            [frequency],
            []
        ],
        output_core_dims=[[]],
        **apply_ufunc_kwargs,
        vectorize=True,  # Loop over time
    )

    wave_direction.attrs['description'] = (
        'Incident wave direction relative to along-cable axis'
    )
    wave_direction.attrs['units'] = 'degrees'
    return wave_direction


def _pressure_to_surface_transfer_function(
    frequency: np.ndarray,
    depth: np.ndarray,
    wavenumber: Optional[np.ndarray] = None,
    sensor_depth: Optional[float] = None,
):
    """ Return a pressure to surface elevation variance transfer function.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the
        number of frequencies and d is the number of depths. The input
        `frequency` is the frequency in Hz and NOT the angular
        frequency, omega or w, and depth is positive-down.

    Args:
        frequency (np.ndarray): Frequencies with shape (d, f).
        depth (np.ndarray): positive-down water depths with shape (d, f).
        wavenumber (Optional[np.ndarray], optional): 1-D wavenumbers
            with shape (d, f). Defaults to None.
        sensor_depth (Optional[float], optional): sensor depth below the
            surface, if different than depth. Defaults to None.

    Returns:
        np.ndarray: Transfer function with shape (d, f).
    """
    # Dispersion solver expects shape (d, f).
    if wavenumber is None:
        wavenumber = _inverse_dispersion_solver(frequency, depth)

    if sensor_depth is not None:
        kh_rel = wavenumber * (depth - sensor_depth)
    else:
        kh_rel = 0.0

    kh = wavenumber * depth
    attenuation = np.cosh(kh) / np.cosh(kh_rel)

    # Square for energy/variance.
    return attenuation**2


def _inverse_dispersion_solver(
    frequency: np.ndarray,
    depth: Union[float, np.ndarray],
) -> np.ndarray:
    r"""Solve the linear dispersion relationship.

    Solves the linear dispersion relationship w^2 = gk tanh(kh) using a
    Scipy Newton-Raphson root-finding implementation.

    Note:
        Expects input as numpy.ndarrays of shape (d,f) where f is the number
        of frequencies and d is the number of depths. The input `frequency` is
        the frequency in Hz and NOT the angular frequency, omega or w.

    Args:
        frequency (np.ndarray): frequencies in [Hz] with shape (d,f).
        depth (np.ndarray): positive-down water depths with shape (d,f).

    Returns:
        np.ndarray: wavenumbers with shape (d,f).
    """
    angular_frequency = frequency_to_angular_frequency(frequency)

    wavenumber_deep = _deep_water_dispersion(frequency)

    wavenumber = newton(func=_dispersion_root,
                        x0=wavenumber_deep,
                        args=(angular_frequency, depth),
                        fprime=_dispersion_derivative)
    return np.asarray(wavenumber)


def _dispersion_root(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return gk * np.tanh(kh) - angular_frequency**2


def _dispersion_derivative(wavenumber, angular_frequency, depth):
    gk = GRAVITY * wavenumber
    kh = wavenumber * depth
    return GRAVITY * np.tanh(kh) + gk * depth * (1 - np.tanh(kh)**2)


def frequency_to_angular_frequency(frequency):
    """Helper function to convert frequency (f) to angular frequency (omega)"""
    return 2 * np.pi * frequency


def _deep_water_dispersion(frequency):
    """Computes wavenumber from the deep water linear dispersion relationship.

    Given frequencies (in Hz) solve the linear dispersion relationship in the
    deep water limit for the corresponding wavenumbers, k. The linear
    dispersion relationship in the deep water limit, tanh(kh) -> 1, has the
    closed form solution k = omega^2 / g and is (approximately) valid for
    kh > np.pi (h > L/2).

    Args:
        frequency (np.ndarray): of any shape containing frequencies
            in [Hz]. NOT the angular frequency, omega or w.

    Returns:
        np.ndarray: (of shape equal to the input shape) containing wavenumbers.
    """
    angular_frequency = frequency_to_angular_frequency(frequency)
    return angular_frequency**2 / GRAVITY


def _direction_from_fk_spectrum(
    fk_spectral_density: np.ndarray,
    wavenumber_x: np.ndarray,
    frequency: np.ndarray,
    depth: float,
    theta_0: float = 10.0,
    **kwargs,
) -> float:
    """Estimate wave direction from a frequency-wavenumber spectrum.

    Estimate incident wave direction relative to the along-cable axis
    from frequency-wavenumber spectral density using nonlinear least
    squares optimization.

    Note: Input wavenumbers and frequencies are not angular (i.e., not
    multiplied by 2*pi). This method currently assumes no currents.

    Args:
        fk_spectral_density (np.ndarray): frequency-wavenumber spectral
            density with shape (f, k).
        wavenumber_x (np.ndarray): Along-cable wavenumbers in units of
            1/distance with shape (k,).
        frequency (np.ndarray): Frequencies in Hz with shape (f,).
        depth (float): Mean water depth.
        theta_0 (float, optional): Initial wave direction guess in
            degrees. Defaults to 10.0.

    Returns:
        float: Incident wave direction, relative to the along-cable
            axis, in degrees.
    """
    # Initial guess
    theta_0_rad = np.deg2rad(theta_0)

    # Run optimization
    # TODO: provide jacobian?
    res = minimize(
        _dispersion_energy,
        [theta_0_rad],
        args=(
            np.abs(fk_spectral_density),
            wavenumber_x,
            frequency,
            depth
        ),
        bounds=[(-np.pi/2, np.pi/2)],
        **kwargs,
    )

    theta_fit = np.rad2deg(res.x[0])

    return theta_fit


def _dispersion_energy(
    theta: np.ndarray[float],
    fk_spectral_density: np.ndarray,
    wavenumber_x: np.ndarray,
    frequency: np.ndarray,
    depth: float,
) -> float:
    """
    Compute energy along the dispersion relationship. The squared sum of
    (negative) energy is returned for estimating wave direction from an
    f-k spectrum via minimization (see `direction_from_fk_spectrum`).
    Note `theta` is in radians.
    """
    TWO_PI = 2 * np.pi

    wavenumber = _inverse_dispersion_solver(frequency, depth)
    wavenumber_estimate = wavenumber * np.cos(theta)

    # Interpolate spectral density at the model wavenumber and frequency.
    interpolator = RegularGridInterpolator(
        (frequency, TWO_PI * wavenumber_x),
        fk_spectral_density,
        bounds_error=False,
        fill_value=0
    )
    sample_points = np.column_stack((frequency, wavenumber_estimate))
    fk_spectral_density_estimate = interpolator(sample_points)

    # Minimize the negative of the spectral density estimate.
    return -np.sum(fk_spectral_density_estimate**2)
