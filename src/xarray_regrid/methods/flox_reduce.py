"""Implementation of flox reduction based regridding methods."""

from typing import overload

import flox.xarray
import numpy as np
import pandas as pd
import xarray as xr

from xarray_regrid import utils
from xarray_regrid.methods._shared import (
    construct_intervals,
    reduce_data_to_new_domain,
    restore_properties,
)


@overload
def statistic_reduce(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
) -> xr.DataArray: ...


@overload
def statistic_reduce(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
) -> xr.Dataset: ...


def statistic_reduce(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
) -> xr.DataArray | xr.Dataset:
    """Upsampling of data using statistical methods (e.g. the mean or variance).

    We use flox Aggregations to perform a "groupby" over multiple dimensions, which we
    reduce using the specified method.
    https://flox.readthedocs.io/en/latest/aggregations.html

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        time_dim: Name of the time dimension. Defaults to "time". Use `None` to force
            regridding over the time dimension.
        method: One of the following reduction methods: "sum", "mean", "var", "std",
            or "median.
        skipna: If NaN values should be ignored.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    valid_methods = ["sum", "mean", "var", "std", "median"]
    if method not in valid_methods:
        msg = f"Invalid method. Please choose from '{valid_methods}'."
        raise ValueError(msg)

    if skipna:
        method = "nan" + method

    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    target_ds_sorted = target_ds.sortby(list(coords))

    bounds = tuple(
        construct_intervals(target_ds_sorted[coord].to_numpy()) for coord in coords
    )

    data = reduce_data_to_new_domain(data, target_ds_sorted, coords)

    result: xr.Dataset = flox.xarray.xarray_reduce(
        data.compute(),
        *coords,
        func=method,
        expected_groups=bounds,
        skipna=skipna,
    )

    result = restore_properties(result, data, target_ds_sorted, coords)
    result = result.reindex_like(target_ds, copy=False)
    return result


def find_matching_int_dtype(
    a: np.ndarray,
) -> type[np.signedinteger] | type[np.unsignedinteger]:
    """Find the smallest integer datatype that can cover the given array."""
    # Integer types in increasing memory use
    int_types: list[type[np.signedinteger] | type[np.unsignedinteger]] = [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
    ]
    for dtype in int_types:
        if (a.max() <= np.iinfo(dtype).max) and (a.min() >= np.iinfo(dtype).min):
            return dtype
    return np.int64


def get_most_common_value(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    expected_groups: np.ndarray,
    time_dim: str | None,
    inverse: bool = False,
) -> xr.DataArray:
    """Upsample the input data using a "most common label" (mode) approach.

    Args:
        data: Input DataArray, with an integer data type. If your data does not consist
            of integer type values, you will have to encode them to integer types.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        expected_groups: Numpy array containing all labels expected to be in the input
            data. For example, `np.array([0, 2, 4])`, if the data only contains the
            values 0, 2 and 4.
        time_dim: Name of the time dimension. Defaults to "time". Use `None` to force
            regridding over the time dimension.
        inverse: Find the least-common-value (anti-mode).

    Raises:
        ValueError: if the input data is not of an integer dtype.

    Returns:
        xarray.DataArray with regridded categorical data.
    """
    array_name = data.name if data.name is not None else "DATA_NAME"

    # Must be categorical data (integers)
    if not np.issubdtype(data.dtype, np.integer):
        msg = (
            "Your input data has to be of an integer datatype for this method.\n"
            f"    instead, your data is of type '{data.dtype}'."
            "You can convert the data with:\n        `dataset.astype(int)`."
        )
        raise ValueError(msg)

    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    target_ds_sorted = target_ds.sortby(list(coords))

    bounds = tuple(
        construct_intervals(target_ds_sorted[coord].to_numpy()) for coord in coords
    )

    data = reduce_data_to_new_domain(data, target_ds_sorted, coords)

    # Reduce memory usage by picking the most minimal integer type
    dtype = find_matching_int_dtype(expected_groups)

    result: xr.DataArray = flox.xarray.xarray_reduce(
        xr.ones_like(data, dtype=bool),
        data.astype(dtype),  # important, needs to be int
        *coords,
        dim=coords,
        func="count",
        expected_groups=(pd.Index(expected_groups.astype(dtype)), *bounds),
        fill_value=-1,
    )
    result = result.idxmax(array_name) if not inverse else result.idxmin(array_name)

    result = restore_properties(result, data, target_ds_sorted, coords)
    result = result.reindex_like(target_ds, copy=False)
    return result
