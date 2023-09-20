import xarray as xr

from xarray_regrid import methods


@xr.register_dataarray_accessor("regrid")
class DataArrayRegridder:
    """Regridding xarray dataarrays.

    Available methods:
        linear: linear, bilinear, or higher dimensional linear interpolation.
        nearest: nearest-neighbor regridding.
        cubic: cubic spline regridding.
        conservative: conservative regridding.
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def linear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray:
        """Return a dataset regridded linearily to the coords of the target dataset.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Dataset regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "linear")

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray:
        """Return a dataset regridded by taking the values of the nearest target coords.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Dataset regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "nearest")

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "cubic")

    def conservative(
        self,
        ds_target_grid: xr.Dataset,
        latitude_coord: str | None,
        time_dim: str = "time",
    ) -> xr.DataArray:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.conservative_regrid(self._obj, ds_target_grid, latitude_coord)


@xr.register_dataset_accessor("regrid")
class DatasetRegridder:
    """Regridding xarray datasets.

    Available methods:
        linear: linear, bilinear, or higher dimensional linear interpolation.
        nearest: nearest-neighbor regridding.
        cubic: cubic spline regridding.
        conservative: conservative regridding.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def linear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.Dataset:
        """Return a dataset regridded linearily to the coords of the target dataset.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Dataset regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "linear")

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.Dataset:
        """Return a dataset regridded by taking the values of the nearest target coords.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Dataset regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "nearest")

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.Dataset:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.interp_regrid(self._obj, ds_target_grid, "cubic")

    def conservative(
        self,
        ds_target_grid: xr.Dataset,
        latitude_coord: str | None,
        time_dim: str = "time",
    ) -> xr.Dataset:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return methods.conservative_regrid(self._obj, ds_target_grid, latitude_coord)


def validate_input(
    data: xr.DataArray | xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str,
) -> xr.Dataset:
    if time_dim in ds_target_grid.coords:
        ds_target_grid = ds_target_grid.isel(time=0).reset_coords()

    if len(set(data.dims).intersection(set(ds_target_grid.dims))) == 0:
        msg = (
            "None of the target dims are in the data:\n"
            " regridding is not possible.\n"
            f"Target dims: {list(ds_target_grid.dims)}\n"
            f"Source dims: {list(data.dims)}"
        )
        raise ValueError(msg)

    if len(set(data.coords).intersection(set(ds_target_grid.coords))) == 0:
        msg = (
            "None of the target coords are in the data:\n"
            " regridding is not possible.\n"
            f"Target coords: {ds_target_grid.coords}\n"
            f"Dataset coords: {data.coords}"
        )
        raise ValueError(msg)

    return ds_target_grid
