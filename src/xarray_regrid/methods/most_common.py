"""Implementation of the "most common value" regridding method."""

from itertools import product
from typing import Any, overload

import flox.xarray
import numpy as np
import numpy_groupies as npg  # type: ignore
import pandas as pd
import xarray as xr
from flox import Aggregation

from xarray_regrid import utils


@overload
def most_common_wrapper(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    time_dim: str = "",
    max_mem: int | None = None,
) -> xr.DataArray: ...


@overload
def most_common_wrapper(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str = "",
    max_mem: int | None = None,
) -> xr.Dataset: ...


def most_common_wrapper(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    time_dim: str = "",
    max_mem: int | None = None,
) -> xr.DataArray | xr.Dataset:
    """Wrapper for the most common regridder, allowing for analyzing larger datasets.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        time_dim: Name of the time dimension, as the regridders do not regrid over time.
            Defaults to "time".
        max_mem: (Approximate) maximum memory in bytes that the regridding routines can
            use. Note that this is not the total memory consumption and does not include
            the size of the final dataset.
            If this kwargs is used, the regridding will be split up into more manageable
            chunks, and combined for the final dataset.

    Returns:
        xarray.dataset with regridded categorical data.
    """
    da_name = None
    if isinstance(data, xr.DataArray):
        da_name = "da" if data.name is None else data.name
        data = data.to_dataset(name=da_name)

    coords = utils.common_coords(data, target_ds)
    target_ds_sorted = target_ds.sortby(list(coords))
    coord_size = [data[coord].size for coord in coords]
    mem_usage = np.prod(coord_size) * np.zeros((1,), dtype=np.int64).itemsize

    if max_mem is not None and mem_usage > max_mem:
        result = split_combine_most_common(
            data=data, target_ds=target_ds_sorted, time_dim=time_dim, max_mem=max_mem
        )
    else:
        result = most_common(data=data, target_ds=target_ds_sorted, time_dim=time_dim)

    result = result.reindex_like(target_ds, copy=False)

    if da_name is not None:
        return result[da_name]
    else:
        return result


def split_combine_most_common(
    data: xr.Dataset, target_ds: xr.Dataset, time_dim: str, max_mem: int = int(1e9)
) -> xr.Dataset:
    """Use a split-combine strategy to reduce the memory use of the most_common regrid.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        time_dim: Name of the time dimension, as the regridders do not regrid over time.
            Defaults to "time".
        max_mem: (Approximate) maximum memory in bytes that the regridding routines can
            use. Note that this is not the total memory consumption and does not include
            the size of the final dataset. Defaults to 1e9 (1 GB).

    Returns:
        xarray.dataset with regridded categorical data.
    """
    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    max_datapoints = max_mem // 8  # ~8 bytes per item.
    max_source_coord_size = max_datapoints ** (1 / len(coords))
    size_ratios = {
        coord: (
            np.median(np.diff(data[coord].to_numpy(), 1))
            / np.median(np.diff(target_ds[coord].to_numpy(), 1))
        )
        for coord in coords
    }
    max_coord_size = {
        coord: int(size_ratios[coord] * max_source_coord_size) for coord in coords
    }

    blocks = {
        coord: np.arange(0, target_ds[coord].size, max_coord_size[coord])
        for coord in coords
    }

    subsets = []
    for vals in product(*blocks.values()):
        isel = {}
        for coord, val in zip(blocks.keys(), vals, strict=True):
            isel[coord] = slice(val, val + max_coord_size[coord])
        subsets.append(most_common(data, target_ds.isel(isel), time_dim=time_dim))

    return xr.merge(subsets)


def most_common(data: xr.Dataset, target_ds: xr.Dataset, time_dim: str) -> xr.Dataset:
    """Upsampling of data with a "most common label" approach.

    The implementation includes two steps:
    - "groupby" coordinates
    - select most common label

    We use flox to perform "groupby" multiple dimensions. Here is an example:
    https://flox.readthedocs.io/en/latest/intro.html#histogramming-binning-by-multiple-variables

    To embed our customized function for most common label selection, we need to
    create our `flox.Aggregation`, for instance:
    https://flox.readthedocs.io/en/latest/aggregations.html

    `flox.Aggregation` function works with `numpy_groupies.aggregate_numpy.aggregate
    API. Therefore this function also depends on `numpy_groupies`. For more information,
    check the following example:
    https://flox.readthedocs.io/en/latest/user-stories/custom-aggregations.html

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.

    Returns:
        xarray.dataset with regridded land cover categorical data.
    """
    dim_order = data.dims
    coords = utils.common_coords(data, target_ds, remove_coord=time_dim)
    coord_attrs = {coord: data[coord].attrs for coord in target_ds.coords}

    bounds = tuple(
        _construct_intervals(target_ds[coord].to_numpy()) for coord in coords
    )

    # Slice the input data to the bounds of the target dataset
    data = data.sortby(list(coords))
    for coord in coords:
        coord_res = np.median(np.diff(target_ds[coord].to_numpy(), 1))
        data = data.sel(
            {
                coord: slice(
                    target_ds[coord].min().to_numpy() - coord_res,
                    target_ds[coord].max().to_numpy() + coord_res,
                )
            }
        )

    most_common = Aggregation(
        name="most_common",
        numpy=_custom_grouped_reduction,  # type: ignore
        chunk=None,
        combine=None,
    )

    ds_regrid: xr.Dataset = flox.xarray.xarray_reduce(
        data.compute(),
        *coords,
        func=most_common,
        expected_groups=bounds,
    )

    ds_regrid = ds_regrid.rename({f"{coord}_bins": coord for coord in coords})
    for coord in coords:
        ds_regrid[coord] = target_ds[coord]

        # Replace zeros outside of original data grid with NaNs
        uncovered_target_grid = (target_ds[coord] <= data[coord].max()) & (
            target_ds[coord] >= data[coord].min()
        )
        ds_regrid = ds_regrid.where(uncovered_target_grid)

        ds_regrid[coord].attrs = coord_attrs[coord]

    return ds_regrid.transpose(*dim_order)


def _construct_intervals(coord: np.ndarray) -> pd.IntervalIndex:
    """Create pandas.intervals with given coordinates."""
    step_size = np.median(np.diff(coord, n=1))
    breaks = np.append(coord, coord[-1] + step_size) - step_size / 2

    # Note: closed="both" triggers an `NotImplementedError`
    return pd.IntervalIndex.from_breaks(breaks, closed="left")


def _most_common_label(neighbors: np.ndarray) -> np.ndarray:
    """Find the most common label in a neighborhood.

    Note that if more than one labels have the same frequency which is the highest,
    then the first label in the list will be picked.
    """
    unique_labels, counts = np.unique(neighbors, return_counts=True)
    return unique_labels[np.argmax(counts)]  # type: ignore


def _custom_grouped_reduction(
    group_idx: np.ndarray,
    array: np.ndarray,
    *,
    axis: int = -1,
    size: int | None = None,
    fill_value: Any = None,
    dtype: Any = None,
) -> np.ndarray:
    """Custom grouped reduction for flox.Aggregation to get most common label.

    Args:
        group_idx : integer codes for group labels (1D)
        array : values to reduce (nD)
        axis : axis of array along which to reduce.
            Requires array.shape[axis] == len(group_idx)
        size : expected number of groups. If none,
            output.shape[-1] == number of uniques in group_idx
        fill_value : fill_value for when number groups in group_idx is less than size
        dtype : dtype of output

    Returns:
        np.ndarray with array.shape[-1] == size, containing a single value per group
    """
    agg: np.ndarray = npg.aggregate_numpy.aggregate(
        group_idx,
        array,
        func=_most_common_label,
        axis=axis,
        size=size,
        fill_value=fill_value,
        dtype=dtype,
    )
    return agg
