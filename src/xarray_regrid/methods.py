from typing import Literal

import xarray as xr

from xarray_regrid import utils


def interp_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.Dataset:
    """Refine a dataset using xarray's interp method.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        method: Which interpolation method to use (e.g. 'linear', 'nearest').

    Returns:
        Regridded input dataset
    """
    coord_names = set(target_ds.coords).intersection(set(data.coords))
    coords = {name: target_ds[name] for name in coord_names}

    return data.interp(
        coords=coords,
        method=method,
    )


def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
) -> xr.Dataset:
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

    Returns:
        Regridded input dataset
    """
    coord_names = set(target_ds.coords).intersection(set(data.coords))
    coords = {name: target_ds[name] for name in coord_names}
    data = data.sortby(list(coord_names))

    # TODO: filter out data vars lacking the target coordinates
    data_vars: list[str] = list(data.data_vars)
    dataarrays = [data[var] for var in data_vars]

    for coord in coords:
        target_coords = coords[coord].to_numpy()
        # TODO: better resolution/IntervalIndex inference
        target_intervals = utils.to_intervalindex(
            target_coords, resolution=target_coords[1] - target_coords[0]
        )
        source_coords = data[coord].to_numpy()
        source_intervals = utils.to_intervalindex(
            source_coords, resolution=source_coords[1] - source_coords[0]
        )
        overlap = utils.overlap(source_intervals, target_intervals)
        weights = utils.normalize_overlap(overlap)

        # TODO: Use `sparse.COO(weights)`. xr.dot does not support this. Much faster!
        dot_array = utils.create_dot_dataarray(
            weights, str(coord), target_coords, source_coords
        )
        # TODO: modify weights to correct for latitude.
        dataarrays = [
            xr.dot(da, dot_array).rename({f"target_{coord}": coord}).rename(da.name)
            for da in dataarrays
        ]
    return xr.merge(dataarrays)  # TODO: add other coordinates/data variables back in.
