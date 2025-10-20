"""
Xarray-based DAS helper functions.
"""
from typing import Any, Optional, Union

import numpy as np
import xarray as xr


def between(
    da: xr.DataArray,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    as_bool: bool = False,
) -> xr.DataArray:
    """Return values between `lower` and `upper`(bounds inclusive)."""
    if lower is not None and upper is None:
        bool_da = da >= lower
    elif lower is None and upper is not None:
        bool_da = da <= upper
    elif lower is not None and upper is not None:
        bool_da = np.logical_and(da >= lower, da <= upper)
    else:
        bool_da = da.where(False, other=True).astype(bool)
    if as_bool:
        return bool_da
    else:
        return da.where(bool_da)


def stack_ch_and_time(
    das_xr: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """Stack channel and time dimensions into a single dimension.

    For time index 't' and channel index 'c', the equivalent stacked
    observation time is das_xr.isel(time_ch=(t*n_ch + c)), where
    `n_ch` is the number of channels.
    """
    # Note that stack automatically places new index at the end, hence
    # the need for transpose. TODO: The transposition could be more
    # general (e.g., by looking at original order of dims) but it works.
    new_das_xr = (das_xr
                  .stack(time_ch=('time', 'ch'))
                  .transpose('time_ch', ...))
    return new_das_xr


def unstack_ch_and_time(
    das_xr: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """Inverse operation to `stack_ch_and_time`. """
    new_das_xr = (das_xr
                  .unstack('time_ch')
                  .transpose('time', 'ch', ...))
    return new_das_xr


def index_by_ch(das_ds: xr.Dataset) -> xr.Dataset:
    """Return a new DAS Dataset indexed by the channel coordinate.

    If the `'ch'` (channel) dimension is currently indexed by the `'site'`
    coordinate, drop this index and set the `'ch'` coordinate as the
    index along the `'ch'` dimension.  If `'ch'` is already the current
    index, return a copy of the DataArray. Both `'ch'` and `'site'` must be
    valid coordinates along the `'ch'` dim.
    """
    if 'site' in das_ds.indexes:
        new_ds = das_ds.drop_indexes('site').set_xindex('ch')
    elif 'ch' in das_ds.indexes:
        new_ds = das_ds.copy()
    else:
        raise KeyError("Neither 'site' nor 'ch' are in indexes.")
    return new_ds


def index_by_site(das_ds: xr.Dataset) -> xr.Dataset:
    """Return a new DAS Dataset indexed by the site coordinate.

    If the `'ch'` (channel) dimension is currently indexed by the `'ch'`
    coordinate, drop this index and set the `'site'` coordinate as the
    index along the `'ch'` dimension.  If `'site'` is already the current
    index, return a copy of the DataArray. Both `'ch'` and `'site'` must be
    valid coordinates along the `'ch'` dim.
    """
    if 'site' in das_ds.indexes:
        new_ds = das_ds.copy()
    elif 'ch' in das_ds.indexes:
        new_ds = das_ds.drop_indexes('ch').set_xindex('site')
    else:
        raise KeyError("Neither 'site' nor 'ch' are in indexes.")
    return new_ds


def get_train(das_ds: xr.Dataset, drop=True) -> xr.Dataset:
    """Return this Dataset's train subset as a new Dataset.

    Note: Dataset must contain an 'is_train' DataArray.

    Args:
        das_ds (xr.Dataset): DAS training and test data.
        drop (bool, optional):  If True, coordinate labels that only
        correspond to False values of the condition are dropped from the
        result. Defaults to True.

    Returns:
        Dataset: Values in Dataset where Dataset['is_train'] is True.
    """
    return das_ds.where(das_ds['is_train'], drop=drop)


def get_test(das_ds: xr.Dataset, drop=True) -> xr.Dataset:
    """Return this Dataset's test subset as a new Dataset.

    Note: Dataset must contain an 'is_test' DataArray.

    Args:
        das_ds (xr.Dataset): DAS training and test data.
        drop (bool, optional):  If True, coordinate labels that only
        correspond to False values of the condition are dropped from the
        result. Defaults to True.

    Returns:
        Dataset: Values in Dataset where Dataset['is_test'] is True.
    """
    return das_ds.where(das_ds['is_test'], drop=drop)
