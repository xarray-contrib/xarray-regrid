from typing import Literal

import xarray as xr


def _interp_regrid(
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
    coord_names = list(target_ds.coords)
    coords = {name: target_ds[name] for name in coord_names}

    return data.interp(
        coords=coords,
        method=method,
    )


def linear_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
) -> xr.Dataset:
    """Refine a dataset using linear interpolation.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.

    Returns:
        Regridded input dataset
    """
    return _interp_regrid(data=data, target_ds=target_ds, method="linear")


def nearest_neigbour_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
) -> xr.Dataset:
    """Refine a dataset using 2d nearest neighbor interpolation.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.

    Returns:
        Regridded input dataset
    """
    return _interp_regrid(data=data, target_ds=target_ds, method="nearest")


def cubic_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
) -> xr.Dataset:
    """Refine a dataset using cubic interpolation.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.

    Returns:
        Regridded input dataset
    """
    return _interp_regrid(data=data, target_ds=target_ds, method="cubic")
