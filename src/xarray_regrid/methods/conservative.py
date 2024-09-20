"""Conservative regridding implementation."""

from collections.abc import Hashable
from typing import overload

import numpy as np
import xarray as xr
from sparse import COO  # type: ignore

from xarray_regrid import utils

EMPTY_DA_NAME = "FRAC_EMPTY"


@overload
def conservative_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
) -> xr.DataArray: ...


@overload
def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
) -> xr.Dataset: ...


def conservative_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | Hashable | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using conservative regridding.

    The method implementation is based on a post by Stephan Hoyer; "For the case of
    interpolation between rectilinear grids (even on the sphere), you can factorize
    regridding along each axis. This is less general but makes the entire calculation
    much simpler, because its feasible to store interpolation weights as dense matrices
    and to use dense matrix multiplication."
    https://discourse.pangeo.io/t/conservative-region-aggregation-with-xarray-geopandas-and-sparse/2715/3

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        latitude_coord: Name of the latitude coordinate. If not provided, attempt to
            infer it from the first coordinate equaling the string 'lat' or 'latitude'.
        skipna: If True, enable handling for NaN values. This adds some overhead,
            so should be disabled for optimal performance on data without NaNs.
        nan_threshold: Threshold value that will retain any output points containing
            at least this many non-null input points. The default value is 1.0,
            which will keep output points containing any non-null inputs. The threshold
            is applied sequentially to each dimension, and may produce different results
            than a threshold applied concurrently to all regridding dimensions.

    Returns:
        Regridded input dataset
    """
    # Attempt to infer the latitude coordinate
    if latitude_coord is None:
        for coord in data.coords:
            if str(coord).lower() in ["lat", "latitude"]:
                latitude_coord = coord
                break

    # Make sure the regridding coordinates are sorted
    coord_names = [coord for coord in target_ds.coords if coord in data.coords]
    target_ds_sorted = xr.Dataset(coords=target_ds.coords)
    for coord_name in coord_names:
        target_ds_sorted = utils.ensure_monotonic(target_ds_sorted, coord_name)
        data = utils.ensure_monotonic(data, coord_name)
    coords = {name: target_ds_sorted[name] for name in coord_names}

    regridded_data = utils.call_on_dataset(
        conservative_regrid_dataset,
        data,
        coords,
        latitude_coord,
        skipna,
        nan_threshold,
    )

    regridded_data = regridded_data.reindex_like(target_ds, copy=False)

    return regridded_data


def conservative_regrid_dataset(
    data: xr.Dataset,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: Hashable,
    skipna: bool,
    nan_threshold: float,
) -> xr.Dataset:
    """Dataset implementation of the conservative regridding method."""
    data_vars = dict(data.data_vars)
    data_coords = dict(data.coords)
    valid_fracs = {v: xr.DataArray(name=EMPTY_DA_NAME) for v in data_vars}
    data_attrs = {v: data_vars[v].attrs for v in data_vars}
    coord_attrs = {c: data_coords[c].attrs for c in data_coords}
    ds_attrs = data.attrs

    for coord in coords:
        covered_grid = (coords[coord] <= data[coord].max()) & (
            coords[coord] >= data[coord].min()
        )

        target_coords = coords[coord].to_numpy()
        source_coords = data[coord].to_numpy()
        nd_weights = get_weights(source_coords, target_coords)

        # Modify weights to correct for latitude distortion
        weights = utils.create_dot_dataarray(
            nd_weights, str(coord), target_coords, source_coords
        )
        if coord == latitude_coord:
            weights = apply_spherical_correction(weights, latitude_coord)

        for array in data_vars.keys():
            if coord in data_vars[array].dims:
                var_weights = sparsify_weights(weights, data_vars[array])
                data_vars[array], valid_fracs[array] = apply_weights(
                    da=data_vars[array],
                    weights=var_weights,
                    coord=coord,
                    valid_frac=valid_fracs[array],
                    skipna=skipna,
                )
                # Mask out any regridded points outside the original domain
                data_vars[array] = data_vars[array].where(covered_grid)

    if skipna:
        # Mask out any points that don't meet the nan threshold
        valid_threshold = get_valid_threshold(nan_threshold)
        for array, da in data_vars.items():
            data_vars[array] = da.where(valid_fracs[array] >= valid_threshold)

    for array, attrs in data_attrs.items():
        data_vars[array].attrs = attrs

    ds_regridded = xr.Dataset(data_vars=data_vars, attrs=ds_attrs)

    for coord, attrs in coord_attrs.items():
        if coord not in ds_regridded.coords:
            # Add back any additional coordinates from the original dataset
            ds_regridded[coord] = data_coords[coord]
        ds_regridded[coord].attrs = attrs

    return ds_regridded


def apply_weights(
    da: xr.DataArray,
    weights: xr.DataArray,
    coord: Hashable,
    valid_frac: xr.DataArray,
    skipna: bool,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply the weights to convert data to the new coordinates."""
    coord_map = {f"target_{coord}": coord}
    weights_norm = weights.copy()

    if skipna:
        notnull = da.notnull()
        # Renormalize the weights along this dim by the accumulated valid_frac
        # along previous dimensions
        if valid_frac.name != EMPTY_DA_NAME:
            weights_norm = weights * (valid_frac / valid_frac.mean(dim=[coord])).fillna(
                0
            )

    da_reduced: xr.DataArray = xr.dot(
        da.fillna(0), weights_norm, dim=[coord], optimize=True
    )
    da_reduced = da_reduced.rename(coord_map).transpose(*da.dims)

    if skipna:
        weights_valid_sum: xr.DataArray = xr.dot(
            notnull, weights_norm, dim=[coord], optimize=True
        )
        weights_valid_sum = weights_valid_sum.rename(coord_map)
        da_reduced /= weights_valid_sum.clip(1e-6, None)

        if valid_frac.name == EMPTY_DA_NAME:
            # Begin tracking the valid fraction
            valid_frac = weights_valid_sum

        else:
            # Update the valid points on this dimension
            valid_frac = xr.dot(valid_frac, weights, dim=[coord], optimize=True)
            valid_frac = valid_frac.rename(coord_map)
            valid_frac = valid_frac.clip(0, 1)

    # In some cases, dot product of dask data and sparse weights fails
    # to densify, which prevents future conversion to numpy
    if da_reduced.chunks and isinstance(da_reduced.data._meta, COO):
        da_reduced.data = da_reduced.data.map_blocks(
            lambda x: x.todense(), dtype=da_reduced.dtype
        )

    return da_reduced, valid_frac


def get_valid_threshold(nan_threshold: float) -> float:
    """Invert the nan_threshold and coerce it to just above zero and below
    one to handle numerical precision limitations in the weight sum."""
    # This matches xesmf where na_thresh=0 keeps points with any valid data
    valid_threshold: float = 1 - np.clip(nan_threshold, 1e-6, 1.0 - 1e-6)
    return valid_threshold


def get_weights(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Determine the weights to map from the old coordinates to the new coordinates.

    Args:
        source_coords: Source coordinates (center points)
        target_coords Target coordinates (center points)

    Returns:
        Weights, which can be used with a dot product to apply the conservative regrid.
    """
    target_intervals = utils.to_intervalindex(target_coords)
    source_intervals = utils.to_intervalindex(source_coords)

    overlap = utils.overlap(source_intervals, target_intervals)
    return utils.normalize_overlap(overlap)


def apply_spherical_correction(
    dot_array: xr.DataArray, latitude_coord: Hashable
) -> xr.DataArray:
    """Apply a sperical earth correction on the prepared dot product weights."""
    da = dot_array.copy()
    latitude_res = np.median(np.diff(dot_array[latitude_coord].to_numpy(), 1))
    lat_weights = lat_weight(dot_array[latitude_coord].to_numpy(), latitude_res)
    da.values = utils.normalize_overlap(dot_array.values * lat_weights[:, np.newaxis])
    return da


def lat_weight(latitude: np.ndarray, latitude_res: float) -> np.ndarray:
    """Return the weight of gridcells based on their latitude.

    Args:
        latitude: (Center) latitude values of the gridcells, in degrees.
        latitude_res: Resolution/width of the grid cells, in degrees.

    Returns:
        Weights, same shape as latitude input.
    """
    dlat: float = np.radians(latitude_res)
    lat = np.radians(latitude)
    h = np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)
    return h * dlat / (np.pi * 4)  # type: ignore


def sparsify_weights(weights: xr.DataArray, da: xr.DataArray) -> xr.DataArray:
    """Create a sparse version of the weights that matches the dtype and chunks
    of the array to be regridded. Even though the weights can be constructed as
    dense arrays, contraction is more efficient with sparse operations."""
    new_weights = weights.copy().astype(da.dtype)
    if da.chunks:
        chunks = {k: v for k, v in da.chunksizes.items() if k in weights.dims}
        new_weights.data = new_weights.chunk(chunks).data.map_blocks(COO)
    else:
        new_weights.data = COO(weights.data)

    return new_weights
