from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr


class InvalidBoundsError(Exception):
    ...


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

    if np.remainder((grid.north - grid.south), grid.resolution_lat) > 0:
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


def to_intervalindex(coords: np.ndarray, resolution: float) -> pd.IntervalIndex:
    """Convert a list of (regularly spaced) 1-d coordinates to pandas IntervalIndex.

    Args:
        coords: 1-d array containing the coordinate values.
        resolution: spatial resolution of the coordinates.

    Returns:
        A pandas IntervalIndex containing the intervals corresponding to the input
            coordinates.
    """
    return pd.IntervalIndex(
        [
            pd.Interval(left=coord - resolution / 2, right=coord + resolution / 2)
            for coord in coords
        ]
    )


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
