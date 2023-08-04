from dataclasses import dataclass

import numpy as np
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
