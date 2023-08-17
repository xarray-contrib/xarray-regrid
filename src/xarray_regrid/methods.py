from typing import Literal

import xarray as xr


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
