"""
Plotting functions and configurations for Oliktok DAS analysis.
"""
from typing import Optional, Callable, Union, List, Literal

import colorcet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr
from brokenaxes import brokenaxes, BrokenAxes
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.lines import Line2D


# Configure matplotlib
figure_full_width = 5.5
normal_font_size = 10
small_font_size = 8
rc_params = {
    'font.size': normal_font_size,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'mathtext.default': 'regular',
    'axes.titlesize': normal_font_size,
    'axes.linewidth': 0.5,
    'axes.labelsize': normal_font_size,
    'lines.markersize': 3,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 2,
    'ytick.minor.size': 2,
    'legend.fontsize': small_font_size,
    'xtick.labelsize': small_font_size,
    'ytick.labelsize': small_font_size,
    'figure.dpi': 300,
    'figure.figsize': (figure_full_width, 4.125),
}
plt.rcParams.update(rc_params)

# Define colors
median_tf_color = '#f768a1'
spectral_nn_color = '#c51b8a'
fk_cnn_color = '#7a0177'

land_color = '#888072'  # arctic permafrost muck color
ocean_color = '#87acd4'
cable_color = '#fcef97'

mooring_site_colors = {
    1: '#a6a6a6',
    2: '#636363',
    3: '#252525',
}
das_site_colors = {
    1: '#67a9cf',
    2: '#1c9099',
    3: '#016c59',
}

mooring_plot_kwargs = {
    'linewidth': 0.75,
    'linestyle': '-',
}

das_plot_kwargs = {
    'linewidth': 0.75,
    'linestyle': '-',
}

train_scatter_kwargs = dict(
    marker='o',
    facecolor='none',
    linewidth=0.5,
    s=1.5,
)

test_scatter_kwargs = dict(
    marker='o',
    linewidth=0.5,
    s=1.5,
)

# Broken time axis parameters
START_BREAK = pd.Timestamp('2023-08-30 00:00:00')
END_BREAK = pd.Timestamp('2023-09-17 16:00:00')


def create_time_series_axes(start_time, end_time, spec):
    """Create time series axes broken at missing data period."""
    baxes = create_broken_time_axes(
        left_lim=start_time,
        right_lim=end_time,
        start_break=START_BREAK,
        end_break=END_BREAK,
        subplot_spec=spec,
    )
    return baxes


def create_broken_time_axes(
    start_break: pd.Timestamp,
    end_break: pd.Timestamp,
    time: Optional[np.ndarray] = None,
    left_lim: Optional[pd.Timestamp] = None,
    right_lim: Optional[pd.Timestamp] = None,
    **kwargs,
) -> BrokenAxes:
    """Create broken time axes using the brokenaxes package."""
    if left_lim is None:
        left_lim = time.min()
    if right_lim is None:
        right_lim = time.max()

    bax = brokenaxes(
        xlims=(
            (left_lim, start_break),
            (end_break, right_lim)
        ),
        **kwargs,
    )

    return bax


def configure_time_series_xaxis(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    baxes: BrokenAxes,
    freq: str = '2D',
    labels: bool = True,
    date_format: str = '%d %b',
) -> None:
    """Set date ticks and labels on broken time axes."""
    date_format = mpl.dates.DateFormatter(date_format)

    for ax in baxes.axs:
        ax.xaxis.set_major_formatter(date_format)

    baxes.tick_params(axis='x', labelrotation=45)

    baxes.axs[0].set_xticks(
        pd.date_range(start_time, START_BREAK.round('D'), freq=freq)
    )
    baxes.axs[1].set_xticks(
        pd.date_range(END_BREAK.round('D'), end_time, freq=freq)
    )

    if not labels:
        baxes.axs[0].set_xticklabels([])
        baxes.axs[1].set_xticklabels([])


def plot_spectra(
    spectral_density_da: xr.DataArray,
    frequency_dim: str = 'frequency',
    c_da: Optional[xr.DataArray] = None,
    ax: Optional[Axes] = None,
    configure_axes: bool = True,
    **kwargs: dict,
) -> Union[List[Line2D], LineCollection]:
    """Plot scalar energy density spectra from an xarray DataArray.

    Expects `spectral_density_da` with dimensions (..., `frequency_dim`)
    where `frequency_dim` is the name of the frequency dimension.
    Non-frequency dimensions are stacked and plotted as individual
    spectra. Assumes the frequency dimension and coordinate share the
    same name.

    Optional `c_da` can be provided to color the spectra. If provided,
    it must have the same dimensions as `spectral_density_da`, except
    for the frequency dimension. This is ignored if the only dimension
    of `spectral_density_da` is frequency.
    """
    if ax is None:
        ax = plt.gca()

    # All dimensions, except frequency, need to be stacked (only 2D arrays
    # can be plotted). Frequency should be the last dimension.
    spectral_density_da = spectral_density_da.transpose(..., frequency_dim)
    dim_to_stack = spectral_density_da.dims[:-1]

    # If `c_da` is provided, its dimensions should be a subset of (or equal
    # to) `dim_to_stack`. Otherwise, raise an exception. If `dim_to_stack`
    # is empty, `c_da` is ignored.
    if c_da is not None and dim_to_stack:
        if set(dim_to_stack) == set(c_da.dims):
            pass
        elif set(dim_to_stack).issuperset(set(c_da.dims)):
            c_da = c_da.broadcast_like(spectral_density_da,
                                       exclude=[frequency_dim])
        else:
            raise ValueError(
                f'Either none of `c_da` dims {c_da.dims} are in this '
                f"array's` dims {dim_to_stack}\n(excluding frequency), or"
                f'`c_da` has extra dims not in this array '
                f'(including frequency).'
            )
        c = c_da.stack(z=dim_to_stack).values
    else:
        c = None

    # Stack spectral density along `dim_to_stack` and transpose so that
    # frequency is the last dimension.
    if dim_to_stack:
        spectral_density_da = (spectral_density_da
                               .stack(z=dim_to_stack)
                               .transpose())

    # Broadcast frequencies to the same shape as spectral density.
    frequency_da = (spectral_density_da[frequency_dim]
                    .broadcast_like(spectral_density_da))

    spectrum_plt = scalar_spectrum(
        spectral_density=spectral_density_da.values,
        frequency=frequency_da.values,
        c=c,
        ax=ax,
        **kwargs,
    )

    # Get and configure current axes (ax may still be None).
    if configure_axes:
        ax = get_ax(ax)
        ax.set_yscale('log')
        ax.set_xscale('log')

        if 'units' in frequency_da.attrs:
            ax.set_xlabel(f"{frequency_da.name} ({frequency_da.attrs['units']})")
        if 'units' in spectral_density_da.attrs:
            ax.set_ylabel(f"{spectral_density_da.name} ({spectral_density_da.attrs['units']})")

    return spectrum_plt


def plot_spectra_by_time(
    spectral_density_da: xr.DataArray,
    frequency_dim: str = 'frequency',
    time_dim: str = 'time',
    ax: Optional[Axes] = None,
    plot_colorbar: bool = True,
    colorbar_kwargs: Optional[dict] = None,
    **kwargs: dict,
) -> Union[List[Line2D], LineCollection]:
    """Plot scalar energy density spectra colored by time.

    Expects `spectral_density_da` with dimensions (..., `time_dim`,
    `frequency_dim`) where `time_dim` and `frequency_dim` are the names
    of the time and frequency dimensions, respectively.

    If `plot_colorbar` is True, a colorbar is added to the current
    figure with ticks and labels corresponding to datetime values.

    """
    # Convert datetimes to numeric values.
    c_da = xr.apply_ufunc(pd.to_numeric, spectral_density_da[time_dim])

    spectrum_plt = plot_spectra(
        spectral_density_da=spectral_density_da,
        frequency_dim=frequency_dim,
        c_da=c_da,
        ax=ax,
        **kwargs
    )

    # Set the colorbar using date range created from the norm.
    if plot_colorbar:
        if colorbar_kwargs is None:
            colorbar_kwargs = {}
        cbar = plt.gcf().colorbar(spectrum_plt, **colorbar_kwargs)
        cbar_dates = pd.date_range(start=c_da.min().item(),
                                   end=c_da.max().item(),
                                   periods=5)
        cbar_ticks = pd.to_numeric(cbar_dates)
        cbar_ticklabels = cbar_dates.strftime('%Y-%m-%d')
        cbar.set_ticks(ticks=cbar_ticks, labels=cbar_ticklabels)

    return spectrum_plt


def plot_fk_spectra(
    spectral_density_da: xr.DataArray,
    wavenumber_dim: str = 'wavenumber',
    frequency_dim: str = 'frequency',
    ax: Optional[Axes] = None,
    statistic: Optional[Literal['mean', 'median']] = None,
    configure_axes: bool = True,
    plot_colorbar: bool = True,
    **kwargs: dict,
) -> QuadMesh:
    """Plot frequency-wavenumber spectral density from an xarray DataArray.

    Expects `spectral_density_da` with dimensions (...,
    `wavenumber_dim`, `frequency_dim`) where `wavenumber_dim` and
    `frequency_dim` are the names of the wavenumber and frequency
    dimensions, respectively. If `spectral_density_da.ndim` == 2 (e.g.,
    wavenumber and frequency), a single f-k spectrum is plotted. If
    `spectral_density_da.ndim` > 2, then `statistic` must be specified
    and `spectral_density_da` is reduced along all dimensions except
    wavenumber and frequency (the last two dimensions).

    Optional `c_da` can be provided to color the spectra. If provided,
    it must have the same dimensions as `spectral_density_da`, except
    for the frequency dimension. This is ignored if the only dimension
    of `spectral_density_da` is frequency.
    """

    if ax is None:
        ax = plt.gca()

    # Wavenumber and frequency should be second to last and last
    # dimensions in spectral density.
    spectral_density_da = spectral_density_da.transpose(..., wavenumber_dim, frequency_dim)

    # Wavenumber and frequency arrays have shapes (k,) and (f,).
    wavenumber_da = spectral_density_da[wavenumber_dim]
    frequency_da = spectral_density_da[frequency_dim]

    if spectral_density_da.ndim > 2 and statistic is None:
        raise ValueError(
            'If the provided DataArray has more than 2 dimensions, '
            'a `statistic` must be provided.'
        )

    spectrum_plt = fk_spectrum(
        spectral_density=spectral_density_da.to_numpy(),
        wavenumber=wavenumber_da.to_numpy(),
        frequency=frequency_da.to_numpy(),
        ax=ax,
        statistic=statistic,
        **kwargs,
    )

    # Get and configure current axes (ax may still be None).
    if configure_axes:
        ax = get_ax(ax)

        if 'units' in wavenumber_da.attrs:
            ax.set_xlabel(f"{wavenumber_da.name} ({wavenumber_da.attrs['units']})")
        if 'units' in frequency_da.attrs:
            ax.set_ylabel(f"{frequency_da.name} ({frequency_da.attrs['units']})")

    if plot_colorbar:
        cbar = plt.gcf().colorbar(spectrum_plt)
        if 'units' in spectral_density_da.attrs:
            cbar.set_label(f"{spectral_density_da.name} ({spectral_density_da.attrs['units']})")

    return spectrum_plt


def scalar_spectrum(
    spectral_density: np.ndarray,
    frequency: np.ndarray,
    c: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    statistic: Optional[Literal['mean', 'median']] = None,
    **kwargs: dict,
) -> Union[List[Line2D], LineCollection]:
    """Plot scalar energy density spectra.

    Expects inputs of shape `(...,f)` where f is the number of
    frequencies. If `spectral_density.ndim` == 2 (e.g., time and
    frequency) and `statistic` is None, all spectra are plotted as a
    multiline plot. Otherwise, `spectral_density` is reduced along the
    first dimension.

    If `spectral_density.ndim` > 2, a `statistic` must
    be specified and `spectral_density` is reduced along all dimensions
    except frequency (the last dimension).

    Args:
        spectral_density (np.ndarray): Spectral densities with shape
            (..., f).
        frequency (np.ndarray): Frequencies with shape (..., f).
        c (Optional[np.ndarray], optional): Array used to color spectra.
            If provided, must have shape (n,) and spectral_density and
            frequency must both have shape (n,f) Defaults to None.
        ax (Optional[Axes], optional): Axis to plot on. Defaults to None.
        statistic (Optional[Literal['mean', 'median']], optional):
            Statistic used to reduce non-frequency dimensions.
            Defaults to None.

    Returns:
        Union[List[Line2D], LineCollection]: Spectral plot as Line2D if
            `c` is None, otherwise a LineCollection.
    """
    ax = get_ax(ax)

    # Reduce spectral_density along non-frequency dimensions.
    if statistic == 'mean':
        axes_to_reduce = tuple(range(spectral_density.ndim - 1))
        spectral_density = spectral_density.mean(axis=axes_to_reduce)
        frequency = frequency.mean(axis=axes_to_reduce)
    elif statistic == 'median':
        axes_to_reduce = tuple(range(spectral_density.ndim - 1))
        spectral_density = np.median(spectral_density, axis=axes_to_reduce)
        frequency = np.median(frequency, axis=axes_to_reduce)
    else:
        pass

    if c is not None:
        kwargs.setdefault('norm', plt.Normalize(vmin=c.min(), vmax=c.max()))

        line_plot = _multiline(
            frequency,
            spectral_density,
            c=c,
            ax=ax,
            **kwargs,
        )
    else:
        line_plot = ax.plot(
            frequency.T,
            spectral_density.T,
            **kwargs,
        )

    return line_plot


def fk_spectrum(
    spectral_density: np.ndarray,
    wavenumber: np.ndarray,
    frequency: np.ndarray,
    ax: Optional[Axes] = None,
    statistic: Optional[Literal['mean', 'median']] = None,
    **kwargs: dict,
) -> QuadMesh:
    """Plot frequency-wavenumber spectral density.

    Expects `spectral_density` input of shape `(..., k, f)` where f is
    the number of frequencies and k is the number of wavenumbers. Input
    `wavenumber` and `frequency` have shapes `(..., k)` and `(..., f)`,
    respectively, and are automatically meshed via np.meshgrid.

    If `spectral_density.ndim` == 2 (e.g., wavenumber and frequency) and
    `statistic` is None, the singular f-k spectrum is plotted.

    If `spectral_density.ndim` > 2, then `statistic` must be specified
    and `spectral_density` is reduced along all dimensions except
    wavenumber and frequency (the last two dimensions).

    Args:
        spectral_density (np.ndarray): Spectral densities with shape
            (..., k, f).
        wavenumbers (np.ndarray): Wavenumbers with shape (..., k).
        frequency (np.ndarray): Frequencies with shape (..., f).
        ax (Optional[Axes], optional): Axis to plot on. Defaults to None.
        statistic (Optional[Literal['mean', 'median']], optional):
            Statistic used to reduce non-frequency dimensions. Defaults
            to None.

    Returns:
        Union[QuadMesh]: Image-like plot of a single
        frequency-wavenumber spectrum.
    """
    # Set default kwargs for pcolormesh.
    default_norm = mpl.colors.LogNorm(vmin=spectral_density.min(),
                                      vmax=spectral_density.max())
    kwargs.setdefault('norm', default_norm)
    kwargs.setdefault('cmap', colorcet.cm.CET_L17_r)

    ax = get_ax(ax)

    # Reduce spectral_density along all dimensions except k and f.
    if statistic == 'mean':
        axes_to_reduce = tuple(range(spectral_density.ndim - 2))
        spectral_density = spectral_density.mean(axis=axes_to_reduce)
    elif statistic == 'median':
        axes_to_reduce = tuple(range(spectral_density.ndim - 2))
        spectral_density = np.median(spectral_density, axis=axes_to_reduce)
    else:
        pass

    # Create a frequency-wavenumber mesh and plot.
    f_grid, wn_grid = np.meshgrid(frequency, wavenumber, indexing='xy')
    pcm = ax.pcolormesh(
        wn_grid,
        f_grid,
        spectral_density,
        **kwargs,
    )

    return pcm


def _multiline(
    xs: np.ndarray,
    ys: np.ndarray,
    c: np.ndarray,
    ax: Optional[Axes] = None,
    **kwargs: dict,
) -> LineCollection:
    """Line plot with colormap.

    Adapted from digbyterrell at https://stackoverflow.com/questions/38208700/
    matplotlib-plot-lines-with-colors-through-colormap.
    """
    ax = get_ax(ax)

    # Collect segments.
    segments = np.stack((xs, ys), axis=-1)
    lc = LineCollection(segments, array=c, **kwargs)

    # Add and rescale. Adding a collection doesn't autoscale xlim/ylim.
    ax.add_collection(lc)
    ax.autoscale()

    return lc


def get_ax(ax):
    "Helper function to get current axes."
    if ax is None:
        ax = plt.gca()
    return ax


def remove_top_and_right_spines(ax):
    """Remove the top and right spines from an axis."""
    ax.spines[['right', 'top']].set_visible(False)


def set_square_aspect(ax):
    """Set the aspect ratio of the axes to be square."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if ax.get_xscale() == 'log':
        xlim = np.log10(xlim)
    if ax.get_yscale() == 'log':
        ylim = np.log10(ylim)
    aspect = (xlim[1]-xlim[0]) / (ylim[1]-ylim[0])
    ax.set_aspect(aspect, adjustable='box')


def create_inset_colorbar(plot_handle, ax, bounds=None, **kwargs):
    """Create an inset colorbar.

    bounds = [x0, y0, width, height]
    """
    if bounds is None:
        bounds = [0.93, 0.5, 0.02, 0.45]
    cax = ax.inset_axes(bounds, axes_class=Axes)
    cbar = plt.colorbar(plot_handle, cax=cax, **kwargs)
    return cbar, cax


def get_empty_legend_placeholder() -> Line2D:
    """Return an empty legend placeholder."""
    return mpl.lines.Line2D([], [], color="none")


# Subplot functions and classes
class SubplotLabeler:
    """Subplot labeler that tracks label iteration."""
    def __init__(
        self,
        label_type: Literal['letter', 'number'] = 'letter',
        label_formatter: Optional[Callable] = None,
    ):
        self.count = 0
        self.label_type = label_type
        if label_formatter is None:
            self.label_formatter = lambda label: f'({label})'
        else:
            self.label_formatter = label_formatter

    def increment_counter(self):
        self.count += 1

    def add_label(self, ax, **kwargs):
        if self.label_type == 'letter':
            label = self._letter_label()
        elif self.label_type == 'number':
            label = str(self.count + 1)
        else:
            raise ValueError(f'Unknown label type: {self.label_type}')

        text = self.label_formatter(label)
        label_subplot(ax, text, **kwargs)
        self.increment_counter()

    def _letter_label(self):
        """Return a letter label."""
        label_letter = chr(ord('@') + (self.count % 26 + 1))
        return label_letter.lower()


def label_subplot(
    ax,
    text,
    fontsize=normal_font_size,
    loc='upper left',
    nudge_x=0,
    nudge_y=0,
    **kwargs,
):
    """Add text to subplot in the specified location."""
    if loc == 'upper left':
        xy = (0.05 + nudge_x, 0.95 + nudge_y)
        ha = 'left'
        va = 'top'
    elif loc == 'lower left':
        xy = (0.05 + nudge_x, 0.05 + nudge_y)
        ha = 'left'
        va = 'bottom'
    elif loc == 'upper right':
        xy = (0.95 + nudge_x, 0.95 + nudge_y)
        ha = 'right'
        va = 'top'
    elif loc == 'lower right':
        xy = (0.95 + nudge_x, 0.05 + nudge_y)
        ha = 'right'
        va = 'bottom'

    ax.annotate(
        text=text,
        xy=xy,
        xycoords='axes fraction',
        ha=ha,
        va=va,
        fontsize=fontsize,
        **kwargs,
    )
