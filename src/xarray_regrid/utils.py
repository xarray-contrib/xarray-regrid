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
            pd.Interval(left=coord - resolution/2, right=coord + resolution/2)
            for coord in coords
        ]
    )


def overlaps(a: pd.Interval, b: pd.Interval):
    """Return the overlap (fraction) between two Pandas intervals."""
    return max(
        min(a.right, b.right) - max(a.left, b.left),
        0
    )


def normalize_overlap(overlap: np.ndarray) -> np.ndarray:
    """Normalize overlap values so they sum up to 1.0 along the first axis."""
    overlap_sum = overlap.sum(axis=0)
    overlap_sum[overlap_sum==0] = 1e-6  # Avoid dividing by 0
    return (overlap / overlap_sum)


def create_dot_dataarray(weights, coord, target_coords, source_coords):
    """Create a DataArray to be used at dot product compatible with xr.dot."""
    return xr.DataArray(
        data=weights,
        dims=[coord, f"target_{coord}"],
        coords={
            coord: source_coords,
            f"target_{coord}": target_coords,
        },
    )
