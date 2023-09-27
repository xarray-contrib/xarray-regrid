import numpy as np
import pytest
import xarray as xr

from xarray_regrid import Grid, create_regridding_dataset


@pytest.fixture
def dummy_lc_data():
    np.random.seed(0)
    data = np.random.randint(0, 3, size=(11, 11))
    lat_coords = np.linspace(0, 40, num=11)
    lon_coords = np.linspace(0, 40, num=11)

    return xr.Dataset(
        data_vars={
            "lc": (["latitude", "longitude"], data),
        },
        coords={
            "longitude": (["longitude"], lon_coords),
            "latitude": (["latitude"], lat_coords),
        },
    )


@pytest.fixture
def dummy_target_grid():
    new_grid = Grid(
        north=40,
        east=40,
        south=0,
        west=0,
        resolution_lat=8,
        resolution_lon=8,
    )
    return create_regridding_dataset(new_grid)


def test_most_common(dummy_lc_data, dummy_target_grid):
    expected = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 1, 2, 0, 0, 2],
            [1, 0, 1, 2, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 2, 2, 0, 2, 2],
        ]
    )

    np.testing.assert_array_equal(
        dummy_lc_data.regrid.most_common(
            dummy_target_grid,
        )["lc"].values,
        expected,
    )
