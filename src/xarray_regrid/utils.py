from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, TypedDict, overload

import numpy as np
import pandas as pd
import xarray as xr


class InvalidBoundsError(Exception): ...


class CoordHandler(TypedDict):
    names: list[str]
    func: Callable


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
    Currently handles padding of spherical geometry if lat/lon coordinates can
    be inferred and the domain size requires boundary padding.
    """
    orig_chunksizes = obj.chunksizes

    # Special-cased coordinates with accepted names and formatting function
    coord_handlers: dict[str, CoordHandler] = {
        "lat": {"names": ["lat", "latitude"], "func": format_lat},
        "lon": {"names": ["lon", "longitude"], "func": format_lon},
    }
    # Identify coordinates that need to be formatted
    formatted_coords = {}
    for coord_type, handler in coord_handlers.items():
        for coord in obj.coords.keys():
            if str(coord).lower() in handler["names"]:
                formatted_coords[coord_type] = str(coord)

    # Apply formatting
    for coord_type, coord in formatted_coords.items():
        # Make sure formatted coords are sorted
        obj = obj.sortby(coord)
        target = target.sortby(coord)
        obj = coord_handlers[coord_type]["func"](obj, target, formatted_coords)
        # Coerce back to a single chunk if that's what was passed
        if len(orig_chunksizes.get(coord, [])) == 1:
            obj = obj.chunk({coord: -1})

    return obj


def format_lat(
    obj: xr.DataArray | xr.Dataset,
    target: xr.Dataset,  # noqa ARG001
    formatted_coords: dict[str, str],
) -> xr.DataArray | xr.Dataset:
    """For latitude, add a single value at each pole computed as the mean of the last
    row for global source grids where the first or last point lie equatorward of 90.
    """
    lat_coord = formatted_coords["lat"]
    lon_coord = formatted_coords.get("lon")

    # Concat a padded value representing the mean of the first/last lat bands
    # This should match the Pole="all" option of ESMF
    # TODO: with cos(90) = 0 weighting, these weights might be 0?

    polar_lat = 90
    dy = obj.coords[lat_coord].diff(lat_coord).max().values.item()

    # Only pad if global but don't have edge values directly at poles
    # South pole
    if dy - polar_lat >= obj.coords[lat_coord].values[0] > -polar_lat:
        south_pole = obj.isel({lat_coord: 0})
        if lon_coord is not None:
            south_pole = south_pole.mean(lon_coord)
        obj = xr.concat([south_pole, obj], dim=lat_coord)  # type: ignore
        obj.coords[lat_coord].values[0] = -polar_lat

    # North pole
    if polar_lat - dy <= obj.coords[lat_coord].values[-1] < polar_lat:
        north_pole = obj.isel({lat_coord: -1})
        if lon_coord is not None:
            north_pole = north_pole.mean(lon_coord)
        obj = xr.concat([obj, north_pole], dim=lat_coord)  # type: ignore
        obj.coords[lat_coord].values[-1] = polar_lat

    return obj


def format_lon(
    obj: xr.DataArray | xr.Dataset, target: xr.Dataset, formatted_coords: dict[str, str]
) -> xr.DataArray | xr.Dataset:
    """For longitude, shift the coordinate to line up with the target values, then
    add a single wraparound padding column if the domain is global and the east
    or west edges of the target lie outside the source grid centers.
    """
    lon_coord = formatted_coords["lon"]

    # Find a wrap point outside of the left and right bounds of the target
    # This ensures we have coverage on the target and handles global > regional
    source_vals = obj.coords[lon_coord].values
    target_vals = target.coords[lon_coord].values
    wrap_point = (target_vals[-1] + target_vals[0] + 360) / 2
    source_vals = np.where(
        source_vals < wrap_point - 360, source_vals + 360, source_vals
    )
    source_vals = np.where(source_vals > wrap_point, source_vals - 360, source_vals)
    obj.coords[lon_coord].values[:] = source_vals

    # Shift operations can produce duplicates
    # Simplest solution is to drop them and add back when padding
    obj = obj.sortby(lon_coord).drop_duplicates(lon_coord)

    # Only pad if domain is global in lon
    source_lon = obj.coords[lon_coord]
    target_lon = target.coords[lon_coord]
    dx_s = source_lon.diff(lon_coord).max().values.item()
    dx_t = target_lon.diff(lon_coord).max().values.item()
    is_global_lon = source_lon.max().values - source_lon.min().values >= 360 - dx_s

    if is_global_lon:
        left_pad = (source_lon.values[0] - target_lon.values[0] + dx_t / 2) / dx_s
        right_pad = (target_lon.values[-1] - source_lon.values[-1] + dx_t / 2) / dx_s
        left_pad = int(np.ceil(np.max([left_pad, 0])))
        right_pad = int(np.ceil(np.max([right_pad, 0])))
        obj = obj.pad({lon_coord: (left_pad, right_pad)}, mode="wrap", keep_attrs=True)
        if left_pad:
            obj.coords[lon_coord].values[:left_pad] = (
                source_lon.values[-left_pad:] - 360
            )
        if right_pad:
            obj.coords[lon_coord].values[-right_pad:] = (
                source_lon.values[:right_pad] + 360
            )

    return obj


def coord_is_covered(
    obj: xr.DataArray | xr.Dataset, target: xr.Dataset, coord: Hashable
) -> bool:
    """Check if the source coord fully covers the target coord."""
    pad = target[coord].diff(coord).max().values
    left_covered = obj[coord].min() <= target[coord].min() - pad
    right_covered = obj[coord].max() >= target[coord].max() + pad
    return bool(left_covered.item() and right_covered.item())
