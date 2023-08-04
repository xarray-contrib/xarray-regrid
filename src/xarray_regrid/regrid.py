from typing import Literal

import xarray as xr

from xarray_regrid import methods


@xr.register_dataset_accessor("regrid")
class Regridder:
    """Regridding xarray datasets.

    Available methods:
        to_dataset: returns a dataset regridded to the coordinates of a target dataset.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def to_dataset(
        self,
        ds_target_grid: xr.Dataset,
        method: Literal["linear", "nearest", "cubic"],
    ) -> xr.Dataset:
        """Return a dataset regridded to the coordinates of the target dataset.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            method: Which regridding method should be used. Available methods:
                - "linear" for linear, bilinear, or higher dimensional linear
                    interpolation.
                - "nearest" for nearest-neighbor regridding.
                - "cubic" for cubic spline regridding.
        Returns:
            Dataset regridded to the target dataset coordinates.
        """
        if method not in ["linear", "nearest", "cubic"]:
            msg = f"Unknown method '{method}'"
            raise ValueError(msg)

        # Remove time dim/coord from target grid.
        if "time" in ds_target_grid.coords:
            ds_target_grid = ds_target_grid.isel(time=0)

        if not set(ds_target_grid.dims).issubset(set(self._obj.dims)):
            msg = (
                "Not all dims in the target dataset are in the \n"
                "dataset: regridding is not possible.\n"
                f"Target dims: {list(ds_target_grid.dims)}"
                f"Dataset dims: {list(self._obj.dims)}"
            )
            raise ValueError(msg)

        if not set(ds_target_grid.coords).issubset(set(self._obj.coords)):
            msg = (
                "Not all coords in the target dataset are in the \n"
                "dataset: regridding is not possible.\n"
                f"Target coords: {ds_target_grid.coords}"
                f"Dataset coords: {self._obj.coords}"
            )
            raise ValueError(msg)

        if method == "linear":
            return methods.linear_regrid(self._obj, ds_target_grid)
        elif method == "nearest":
            return methods.nearest_neigbour_regrid(self._obj, ds_target_grid)
        if method == "cubic":
            return methods.cubic_regrid(self._obj, ds_target_grid)
