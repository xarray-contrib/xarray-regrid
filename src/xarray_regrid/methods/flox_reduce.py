"""Implementation of flox reduction based regridding methods."""

from typing import Any, overload

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
    fill_value: None | Any = None,
) -> xr.DataArray: ...


@overload
def statistic_reduce(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
    fill_value: None | Any = None,
) -> xr.Dataset: ...


def statistic_reduce(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str | None,
    method: str,
    skipna: bool = False,
    fill_value: None | Any = None,
) -> xr.DataArray | xr.Dataset:
    """Upsampling of data using statistical methods (e.g. the mean or variance).

    We use flox Aggregations to perform a "groupby" over multiple dimensions, which we
    reduce using the specified method.
    https://flox.readthedocs.io/en/latest/aggregations.html

    Args:
        data: Input dataset.
            It is assumed that the coordinates of this data are sorted.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        time_dim: Name of the time dimension. Defaults to "time". Use `None` to force
            regridding over the time dimension.
        method: One of the following reduction methods: "sum", "mean", "var", "std",
            or "median.
        skipna: If NaN values should be ignored.
        fill_value: What value to fill uncovered parts of the target grid. By default
            this will be NaN, and integer type data will be cast to float to accomodate
            this.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    valid_methods = ["sum", "mean", "var", "std", "median", "max", "min"]
    if method not in valid_methods:
        msg = f"Invalid method. Please choose from '{valid_methods}'."
        raise ValueError(msg)

    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    target_coords = xr.Dataset(target_ds.coords)  # coords target coords for reindexing
    sorted_target_coords = target_coords.sortby(coords)

    bounds = tuple(
        construct_intervals(sorted_target_coords[coord].to_numpy()) for coord in coords
    )

    data = reduce_data_to_new_domain(data, sorted_target_coords, coords)

    result: xr.Dataset = flox.xarray.xarray_reduce(
        data,
        *coords,
        func=method,
        expected_groups=bounds,
        skipna=skipna,
        fill_value=fill_value,
    )

    result = restore_properties(result, data, target_ds, coords, fill_value)
    result = result.reindex_like(target_coords, copy=False)
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


def compute_mode(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    values: np.ndarray,
    time_dim: str | None,
    fill_value: None | Any = None,
    anti_mode: bool = False,
) -> xr.DataArray:
    """Upsample the input data using a "most common label" (mode) approach.

    Args:
        data: Input DataArray, with an integer data type. If your data does not consist
            of integer type values, you will have to encode them to integer types.
            It is assumed that the coordinates of this data are sorted.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        values: Numpy array containing all labels expected to be in the input
            data. For example, `np.array([0, 2, 4])`, if the data only contains the
            values 0, 2 and 4.
        time_dim: Name of the time dimension. Defaults to "time". Use `None` to force
            regridding over the time dimension.
        fill_value: What value to fill uncovered parts of the target grid. By default
            this will be NaN, and integer type data will be cast to float to accomodate
            this.
        anti_mode: Find the least-common-value (anti-mode).

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
    target_coords = xr.Dataset(target_ds.coords)  # stores coords for reindexing later
    sorted_target_coords = target_coords.sortby(coords)

    bounds = tuple(
        construct_intervals(sorted_target_coords[coord].to_numpy()) for coord in coords
    )

    data = reduce_data_to_new_domain(data, sorted_target_coords, coords)

    result: xr.DataArray = flox.xarray.xarray_reduce(
        xr.ones_like(data, dtype=bool),
        data,  # important, needs to be int
        *coords,
        dim=coords,
        func="count",
        expected_groups=(pd.Index(values.astype(data)), *bounds),
        fill_value=-1,
    )
    result = result.idxmax(array_name) if not anti_mode else result.idxmin(array_name)

    result = restore_properties(result, data, target_ds, coords, fill_value)
    result = result.reindex_like(target_coords, copy=False)
    return result
