from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr


class InvalidBoundsError(Exception): ...


@dataclass
class Grid:
    """Object storing grid information."""

    north: float
    east: float
    south: float
    west: float
    resolution_lat: float
    resolution_lon: float

    def __post_init__(self) -> None:
        """Validate the initialized SpatialBounds class."""
        msg = None
        if self.south > self.north:
            msg = (
                "Value of north bound is greater than south bound."
                "\nPlease check the bounds input."
            )
            pass
        if self.west > self.east:
            msg = (
                "Value of west bound is greater than east bound."
                "\nPlease check the bounds input."
            )
        if msg is not None:
            raise InvalidBoundsError(msg)

    def create_regridding_dataset(
        self, lat_name: str = "latitude", lon_name: str = "longitude"
    ) -> xr.Dataset:
        """Create a dataset to use for regridding.

        Args:
            grid: Grid object containing the bounds and resolution of the
                cartesian grid.
            lat_name: Name for the latitudinal coordinate and dimension.
                Defaults to "latitude".
            lon_name: Name for the longitudinal coordinate and dimension.
                Defaults to "longitude".

        Returns:
            A dataset with the latitude and longitude coordinates corresponding to the
                specified grid. Contains no data variables.
        """
        return create_regridding_dataset(self, lat_name, lon_name)


def create_lat_lon_coords(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Create latitude and longitude coordinates based on the provided grid parameters.

    Args:
        grid: Grid object.

    Returns:
        Latititude coordinates, longitude coordinates.
    """

    if np.remainder((grid.north - grid.south), grid.resolution_lat) > 0:
        lat_coords = np.arange(grid.south, grid.north, grid.resolution_lat)
    else:
        lat_coords = np.arange(
            grid.south, grid.north + grid.resolution_lat, grid.resolution_lat
        )

    if np.remainder((grid.east - grid.west), grid.resolution_lat) > 0:
        lon_coords = np.arange(grid.west, grid.east, grid.resolution_lon)
    else:
        lon_coords = np.arange(
            grid.west, grid.east + grid.resolution_lon, grid.resolution_lon
        )
    return lat_coords, lon_coords


def create_regridding_dataset(
    grid: Grid, lat_name: str = "latitude", lon_name: str = "longitude"
) -> xr.Dataset:
    """Create a dataset to use for regridding.

    Args:
        grid: Grid object containing the bounds and resolution of the cartesian grid.
        lat_name: Name for the latitudinal coordinate and dimension.
            Defaults to "latitude".
        lon_name: Name for the longitudinal coordinate and dimension.
            Defaults to "longitude".

    Returns:
        A dataset with the latitude and longitude coordinates corresponding to the
            specified grid. Contains no data variables.
    """
    lat_coords, lon_coords = create_lat_lon_coords(grid)
    return xr.Dataset(
        {
            lat_name: ([lat_name], lat_coords, {"units": "degrees_north"}),
            lon_name: ([lon_name], lon_coords, {"units": "degrees_east"}),
        }
    )


def to_intervalindex(coords: np.ndarray) -> pd.IntervalIndex:
    """Convert a 1-d coordinate array to a pandas IntervalIndex. Take
    the midpoints between the coordinates as the interval boundaries.

    Args:
        coords: 1-d array containing the coordinate values.

    Returns:
        A pandas IntervalIndex containing the intervals corresponding to the input
            coordinates.
    """
    if len(coords) > 1:
        midpoints = (coords[:-1] + coords[1:]) / 2

        # Extrapolate outer bounds beyond the first and last coordinates
        left_bound = 2 * coords[0] - midpoints[0]
        right_bound = 2 * coords[-1] - midpoints[-1]

        breaks = np.concatenate([[left_bound], midpoints, [right_bound]])
        intervals = pd.IntervalIndex.from_breaks(breaks)

    else:
        # If the target grid has a single point, set search interval to span all space
        intervals = pd.IntervalIndex.from_breaks([-np.inf, np.inf])

    return intervals


def overlap(a: pd.IntervalIndex, b: pd.IntervalIndex) -> np.ndarray:
    """Calculate the overlap between two sets of intervals.

    Args:
        a: Pandas IntervalIndex containing the first set of intervals.
        b: Pandas IntervalIndex containing the second set of intervals.

    Returns:
        2D numpy array containing overlap (as a fraction) between the intervals of a
            and b. If there is no overlap, the value will be 0.
    """
    # TODO: newaxis on B and transpose is MUCH faster on benchmark.
    #  likely due to it being the bigger dimension.
    #  size(a) > size(b) leads to better perf than size(b) > size(a)
    mins = np.minimum(a.right.to_numpy(), b.right.to_numpy()[:, np.newaxis])
    maxs = np.maximum(a.left.to_numpy(), b.left.to_numpy()[:, np.newaxis])
    overlap: np.ndarray = np.maximum(mins - maxs, 0).T
    return overlap


def normalize_overlap(overlap: np.ndarray) -> np.ndarray:
    """Normalize overlap values so they sum up to 1.0 along the first axis."""
    overlap_sum: np.ndarray = overlap.sum(axis=0)
    overlap_sum[overlap_sum == 0] = 1e-12  # Avoid dividing by 0.
    return overlap / overlap_sum  # type: ignore


def create_dot_dataarray(
    weights: np.ndarray,
    coord: str,
    target_coords: np.ndarray,
    source_coords: np.ndarray,
) -> xr.DataArray:
    """Create a DataArray to be used at dot product compatible with xr.dot."""
    return xr.DataArray(
        data=weights,
        dims=[coord, f"target_{coord}"],
        coords={
            coord: source_coords,
            f"target_{coord}": target_coords,
        },
    )


def common_coords(
    data1: xr.DataArray | xr.Dataset,
    data2: xr.DataArray | xr.Dataset,
    remove_coord: str | None = None,
) -> list[str]:
    """Return a set of coords which two dataset/arrays have in common."""
    coords = set(data1.coords).intersection(set(data2.coords))
    if remove_coord in coords:
        coords.remove(remove_coord)
    return sorted([str(coord) for coord in coords])


@overload
def call_on_dataset(
    func: Callable[..., xr.Dataset],
    obj: xr.DataArray,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray: ...


@overload
def call_on_dataset(
    func: Callable[..., xr.Dataset],
    obj: xr.Dataset,
    *args: Any,
    **kwargs: Any,
) -> xr.Dataset: ...


def call_on_dataset(
    func: Callable[..., xr.Dataset],
    obj: xr.DataArray | xr.Dataset,
    *args: Any,
    **kwargs: Any,
) -> xr.DataArray | xr.Dataset:
    """Use to call a function that expects a Dataset on either a Dataset or
    DataArray, round-tripping to a temporary dataset."""
    placeholder_name = "_UNNAMED_ARRAY"
    if isinstance(obj, xr.DataArray):
        tmp_name = obj.name if obj.name is not None else placeholder_name
        ds = obj.to_dataset(name=tmp_name)
    else:
        ds = obj

    result = func(ds, *args, **kwargs)

    if isinstance(obj, xr.DataArray) and isinstance(result, xr.Dataset):
        msg = "Trying to convert Dataset with more than one data variable to DataArray"
        if len(result.data_vars) > 1:
            raise TypeError(msg)
        return next(iter(result.data_vars.values())).rename(obj.name)

    return result


def format_for_regrid(
    obj: xr.DataArray | xr.Dataset, target: xr.Dataset
) -> xr.DataArray | xr.Dataset:
    """Apply any pre-formatting to the input dataset to prepare for regridding.
    Currently handles padding of spherical geometry if appropriate coordinate names
    can be inferred containing 'lat' and 'lon'.
    """
    lat_coord = None
    lon_coord = None

    for coord in obj.coords.keys():
        if str(coord).lower().startswith("lat"):
            lat_coord = coord
        elif str(coord).lower().startswith("lon"):
            lon_coord = coord

    if lon_coord is not None or lat_coord is not None:
        obj = format_spherical(obj, target, lat_coord, lon_coord)

    return obj


def format_spherical(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    lat_coord: Hashable,
    lon_coord: Hashable,
) -> xr.DataArray | xr.Dataset:
    """Infer whether a lat/lon source grid represents a global domain and
    automatically apply spherical padding to improve edge effects.

    For longitude, shift the coordinate to line up with the target values, then
    add a single wraparound padding column if the domain is global and the east
    or west edges of the target lie outside the source grid centers.

    For latitude, add a single value at each pole computed as the mean of the last
    row for global source grids where the first or last point lie equatorward of 90.
    """

    orig_chunksizes = obj.chunksizes

    # If the source coord fully covers the target, don't modify them
    if lat_coord and not coord_is_covered(obj, target, lat_coord):
        obj = obj.sortby(lat_coord)
        target = target.sortby(lat_coord)

        # Only pad if global but don't have edge values directly at poles
        polar_lat = 90
        dy = obj[lat_coord].diff(lat_coord).max().values

        # South pole
        if dy - polar_lat >= obj[lat_coord][0] > -polar_lat:
            south_pole = obj.isel({lat_coord: 0})
            # This should match the Pole="all" option of ESMF
            if lon_coord is not None:
                south_pole = south_pole.mean(lon_coord)
            obj = xr.concat([south_pole, obj], dim=lat_coord)
            obj[lat_coord].values[0] = -polar_lat

        # North pole
        if polar_lat - dy <= obj[lat_coord][-1] < polar_lat:
            north_pole = obj.isel({lat_coord: -1})
            if lon_coord is not None:
                north_pole = north_pole.mean(lon_coord)
            obj = xr.concat([obj, north_pole], dim=lat_coord)
            obj[lat_coord].values[-1] = polar_lat

        # Coerce back to a single chunk if that's what was passed
        if len(orig_chunksizes.get(lat_coord, [])) == 1:
            obj = obj.chunk({lat_coord: -1})

    if lon_coord and not coord_is_covered(obj, target, lon_coord):
        obj = obj.sortby(lon_coord)
        target = target.sortby(lon_coord)

        target_lon = target[lon_coord].values
        # Find a wrap point outside of the left and right bounds of the target
        # This ensures we have coverage on the target and handles global > regional
        wrap_point = (target_lon[-1] + target_lon[0] + 360) / 2
        lon = obj[lon_coord].values
        lon = np.where(lon < wrap_point - 360, lon + 360, lon)
        lon = np.where(lon > wrap_point, lon - 360, lon)
        obj[lon_coord].values[:] = lon

        # Shift operations can produce duplicates
        # Simplest solution is to drop them and add back when padding
        obj = obj.sortby(lon_coord).drop_duplicates(lon_coord)

        # Only pad if domain is global in lon
        dx_s = obj[lon_coord].diff(lon_coord).max().values
        dx_t = target[lon_coord].diff(lon_coord).max().values
        is_global_lon = obj[lon_coord].max() - obj[lon_coord].min() >= 360 - dx_s

        if is_global_lon:
            left_pad = (obj[lon_coord][0] - target[lon_coord][0] + dx_t / 2) / dx_s
            right_pad = (target[lon_coord][-1] - obj[lon_coord][-1] + dx_t / 2) / dx_s
            left_pad = int(np.ceil(np.max([left_pad, 0])))
            right_pad = int(np.ceil(np.max([right_pad, 0])))
            lon = obj[lon_coord].values
            obj = obj.pad(
                {lon_coord: (left_pad, right_pad)}, mode="wrap", keep_attrs=True
            )
            if left_pad:
                obj[lon_coord].values[:left_pad] = lon[-left_pad:] - 360
            if right_pad:
                obj[lon_coord].values[-right_pad:] = lon[:right_pad] + 360

            # Coerce back to a single chunk if that's what was passed
            if len(orig_chunksizes.get(lon_coord, [])) == 1:
                obj = obj.chunk({lon_coord: -1})

    return obj


def coord_is_covered(
    obj: xr.DataArray | xr.Dataset, target: xr.Dataset, coord: Hashable
) -> bool:
    """Check if the source coord fully covers the target coord."""
    pad = target[coord].diff(coord).max().values
    left_covered = obj[coord].min() <= target[coord].min() - pad
    right_covered = obj[coord].max() >= target[coord].max() + pad
    return bool(left_covered.item() and right_covered.item())
