"""
Spectral analysis functions used to estimate scalar and f-k spectra from
DAS strain/strain rate. Provided for reference only.
"""

from typing import Any, Optional, Union, Tuple

import numpy as np
import scipy
import skimage
import xarray as xr


def welch_fk(
    arr: np.ndarray,
    dt: float,
    dx: float,
    segment_length: int,
    stride: int,
    window: str = 'hann',
    n_merge_f: int = 0,
    n_merge_k: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate f-k spectral density using Welch's method.

    Estimate frequency-wavenumber spectral density of `arr` using
    Welch's method in two dimensions.  Overlapping segments are created
    and averaged only in the time dimension.

    Input `arr` must have shape `(x, t)` where `x` is the spatial
    dimension and `t` is the time dimension (i.e., time dimension along
    last axis). Returned wavenumber, frequency, and spectral density
    arrays have shapes `(k,)`, `(f,)`, and `(k, f)`, respectively.


    Args:
        arr (np.ndarray): 2D array with space and time dimensions.
        dt (float): Constant step size in the time dimension.
        dx (float): Constant step size in the spatial dimension.
        segment_length (int): Number of points in each segment.
        stride (int): Stride between segments
            (n_overlap = segment_length - stride)
        window (str, optional): Any window type supported by
            `scipy.signal.get_window`. Defaults to 'hann'.
        n_merge_f (int): Number of adjacent frequencies to merge.
        n_merge_k (int): Number of adjacent wavenumbers to merge.

    Returns:
        Tuple: Wavenumber, frequency, and spectral density arrays.

    See Also:
        scipy.signal.welch
    """
    CYLC = 1  # cycle size (could be 2pi)
    RESCALE = 2
    INITIAL_TIME_AXIS = 1
    SEGMENT_AXIS = 0

    # Demean
    arr = arr - arr.mean()

    # Segment along the time axis. Segments are inserted at position of
    # the initial time axis.
    arr_segmented = create_overlapping_segments(arr=arr,
                                                segment_length=segment_length,
                                                stride=stride,
                                                axis=INITIAL_TIME_AXIS)

    # Move dimension corresponding to number of segments to the first axis.
    arr_segmented = np.swapaxes(arr_segmented, SEGMENT_AXIS, INITIAL_TIME_AXIS)

    # Get number of segments and and new shape in space and time.
    n_windows, n_x, n_t = arr_segmented.shape

    # Window the segments.
    window_function_2d = skimage.filters.window(window, (n_x, n_t))
    arr_windowed = arr_segmented * window_function_2d

    # Compute 2D FFT of segments.  Axes are (segment, x, t).
    # TODO: use 's' input for zero-padding (but adjust n_t and n_x)
    # TODO: could play with fft normalization method for different results
    segment_transforms = scipy.fft.fft2(arr_windowed, axes=(1, 2))
    segment_transforms = scipy.fft.fftshift(segment_transforms, axes=(1, 2)) / (n_x * n_t)

    # Define frequency and wavenumber resolution and arrays.
    df = CYLC/(n_t * dt)
    dk = CYLC/(n_x * dx)
    frequency = scipy.fft.fftshift(scipy.fft.fftfreq(n_t, d=dt)) * CYLC
    wavenumber = scipy.fft.fftshift(scipy.fft.fftfreq(n_x, d=dx)) * CYLC

    # Calculate spectral densities and retain real values.
    segment_spectra = segment_transforms * np.conj(segment_transforms) / (df * dk)  # (m/s)ˆ2/cpd/cpm
    segment_spectra = segment_spectra.real

    # Retain positive frequencies. Positive and negative wavenumbers
    # represent forward and backward propagation.
    pos_frequency = frequency > 0
    frequency = frequency[pos_frequency]

    # Retain (+f, +k) and (+f, -k) plane and rescale.
    segment_spectra = segment_spectra[:, :, pos_frequency] * RESCALE

    # Average spectra across segments.
    spectrum = segment_spectra.mean(axis=SEGMENT_AXIS)

    # Merge frequencies along frequency axis.
    if n_merge_f > 1:
        frequency, spectrum = merge_frequencies(frequency,
                                                spectrum,
                                                n_merge=n_merge_f,
                                                axis=1)
    if n_merge_k > 1:
        wavenumber, spectrum = merge_frequencies(wavenumber,
                                                 spectrum,
                                                 n_merge=n_merge_k,
                                                 axis=0)

    # Rescale spectra to recover variance lost during windowing.
    time_var = arr.var()
    spectral_var = np.trapz(
        np.trapz(
            spectrum,
            x=wavenumber,
            axis=0,
        ),
        x=frequency,
        axis=-1,
    )
    var_ratio = spectral_var / time_var
    spectrum = spectrum * var_ratio**(-1)

    return wavenumber, frequency, spectrum


def interpolate_fk_spectra(
    fk_spectrum: np.ndarray,
    wavenumber: np.ndarray,
    frequency: np.ndarray,
    n_points_f: int = 2**7,
    n_points_k: int = 2**7,
    method: str = 'linear',
):
    """Interpolate f-k spectra onto a new regular grid.

    The new grid is linearly spaced with `n_points_k` and `n_points_f`
    between the min and max of the original wavenumber and frequency
    arrays, respectively.

    Returns arrays with shapes `(k', f', t)` where k' and f' are the
    number of wavenumber and frequency points in the interpolated grid.

    Args:
        fk_spectrum (np.ndarray): f-k spectra with shape (k, f, t)
        wavenumber (np.ndarray): Wavenumbers with shape (k,)
        frequency (np.ndarray): Frequencies with shape (f,)
        n_points_f (int, optional): Number of frequency points in
            interpolated spectra. Defaults to 2**7.
        n_points_k (int, optional): Number of wavenumber points in
            interpolated spectra. Defaults to 2**7.
        method (str, optional): Interpolation method.  See
            `scipy.interpolate.RegularGridInterpolator`. Defaults to
            'linear'.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Interpolated
            wavenumber, frequency, and f-k spectra arrays with shapes
            (k', f', t).
    """
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (wavenumber, frequency),
        fk_spectrum,
        method=method,
    )

    # Define the new points to interpolate onto.
    frequency_interp = np.linspace(frequency.min(), frequency.max(), n_points_f)
    wavenumber_interp = np.linspace(wavenumber.min(), wavenumber.max(), n_points_k)
    wavenumber_interp_grid, frequency_interp_grid = np.meshgrid(
        frequency_interp,
        wavenumber_interp,
        indexing='xy',
    )
    points_new = np.stack(
        (frequency_interp_grid.flatten(), wavenumber_interp_grid.flatten()),
        axis=-1
    )

    # Perform the interpolation.
    new_shape = (wavenumber_interp.size, frequency_interp.size, fk_spectrum.shape[-1])
    spectra_interp = interpolator(points_new).reshape(new_shape)

    return wavenumber_interp, frequency_interp, spectra_interp


def merge_frequencies(
    frequency: np.ndarray,
    spectral_density: np.ndarray,
    n_merge: int,
    axis: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge neighboring frequencies in a spectrum.

    Note:
        Assumes the frequency dimension is along the last axis of
        `spectral_density` unless otherwise specified using `axis`.

    Args:
        frequency (np.ndarray): Frequencies with shape (f,)
        spectral_density (np.ndarray): Frequency spectra with shape (f,)
            or shape (..., f).
        n_merge (int): Number of adjacent frequencies to merge.
        axis (int, optional): Axis along which to merge frequencies.
            Defaults to -1.

    Returns:
        (np.ndarray, np.ndarray): Merged frequency and spectra with
            shape (f//n_merge) along `axis`.

    Example:
    ```
    >>> f = np.arange(0, 9, 1)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    >>> e = f * 3
    array([ 0,  3,  6,  9, 12, 15, 18, 21, 24])

    >>> merge_frequencies(f, e, n_merge=3)
    (array([1., 4., 7.]), array([ 3., 12., 21.]))
    ```
    """
    n_groups = len(frequency) // n_merge
    frequency_merged = _average_groups(frequency, n_groups)
    spectral_density_merged = np.apply_along_axis(_average_groups,
                                                  axis=axis,
                                                  arr=spectral_density,
                                                  n_groups=n_groups)
    return frequency_merged, spectral_density_merged


def _average_groups(arr, n_groups):
    """
    Adapted from Divakar via https://stackoverflow.com/questions/53178018/
    average-of-elements-in-a-subarray.

    Functionally equivalent to (but faster than):
    arr_split = np.array_split(arr, n_groups)
    arr_merged = np.array([group.mean() for group in arr_split])
    """
    n = len(arr)
    m = n // n_groups
    w = np.full(n_groups, m)
    w[:n - m*n_groups] += 1
    sums = np.add.reduceat(arr, np.r_[0, w.cumsum()[:-1]])
    return sums / w


def create_every_n_slice(n: int, ndim: int, axis: int) -> Tuple:
    """
    Return a tuple of slices to take every nth element along an axis.

    For example, a slice to return every 2nd element (`n=2`) from
    two-dimensional array (`ndim=2`) along the second axis (`axis=1`) is
    equivalent to `[:, ::2]` or `(slice(None), slice(None, None, 2)`.

    Args:
        n (int): Take every nth element.
        ndim (int): Number of dimensions.
        axis (int): Axis to take every nth element along.

    Returns:
        Tuple[slice]: Tuple of slices
    """
    # Create a list of slice objects for each dimension
    slices = [slice(None)] * ndim

    # Set the `axis` slice to take every nth element
    slices[axis] = slice(None, None, n)
    return tuple(slices)


def create_overlapping_segments(
    arr: np.ndarray,
    segment_length: int,
    stride: int,
    axis: int = 0
) -> np.ndarray:
    """
    Create overlapping segments of length `segment_length` with `stride`
    along `axis`.

    The array is return with a new dimension where the original
    dim along `axis` is replaced with `(num_segments`, segment_length)`
    where: `num_segments = (arr.shape[axis] - segment_length) / stride + 1`

    For segments with 50% overlap, set `stride = segment_length / 2`.

    Args:
        arr (np.ndarray): Array to segment.
        segment_length (int): Length of segments.
        stride (int): Stride between segments.
        axis (int, optional): Axis to segment along. Defaults to 0.

    Raises:
        ValueError: If the number of segments is not an integer.

    Returns:
        np.ndarray: Segmented `arr`.

    Examples:
    ```python
    >>> arr = np.tile(np.arange(3), 4)
    >>> arr
    array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    >>> create_overlapping_segments(arr, segment_length=3, stride=3, axis=0)
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]])
    ```
    """
    # Calculate number of segments. Also note the overlap for each segment
    # is `overlap_size = segment_length - stride`.
    num_segments = (arr.shape[axis] - segment_length) / stride + 1
    if not num_segments.is_integer():
        raise ValueError(
            f"The number of segments ({num_segments}) must be an integer"
        )
    # Create a sliding window view of the array.  By default, this
    # function returns segments with a stride = 1.
    arr_segmented = np.lib.stride_tricks.sliding_window_view(arr,
                                                             segment_length,
                                                             axis=axis)
    # Create a slice object to take every nth segment based on `stride`.
    every_nth_segment = create_every_n_slice(stride, arr.ndim, axis)
    return arr_segmented[every_nth_segment]
