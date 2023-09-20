from pathlib import Path

import pytest
import xarray as xr

import xarray_regrid

DATA_PATH = Path(__file__).parent.parent / "benchmarks" / "data"

CDO_DATA = {
    "linear": DATA_PATH / "cdo_bilinear_64b.nc",
    "nearest": DATA_PATH / "cdo_nearest_64b.nc",
    "conservative": DATA_PATH / "cdo_conservative_64b.nc",
}


@pytest.fixture
def sample_input_data() -> xr.Dataset:
    return xr.open_dataset(DATA_PATH / "era5_2m_dewpoint_temperature_2000_monthly.nc")


@pytest.fixture
def sample_grid_ds():
    grid = xarray_regrid.Grid(
        north=90,
        east=180,
        south=45,
        west=90,
        resolution_lat=0.17,
        resolution_lon=0.17,
    )

    return xarray_regrid.create_regridding_dataset(grid)


@pytest.mark.parametrize(
    "method, cdo_file",
    [
        ("linear", CDO_DATA["linear"]),
        ("nearest", CDO_DATA["nearest"]),
    ],
)
def test_basic_regridders_ds(sample_input_data, sample_grid_ds, method, cdo_file):
    """Test the dataset regridders (except conservative)."""
    regridder = getattr(sample_input_data.regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    ds_cdo = xr.open_dataset(cdo_file)
    xr.testing.assert_allclose(ds_regrid.compute(), ds_cdo.compute())


@pytest.mark.parametrize(
    "method, cdo_file",
    [
        ("linear", CDO_DATA["linear"]),
        ("nearest", CDO_DATA["nearest"]),
    ],
)
def test_basic_regridders_da(sample_input_data, sample_grid_ds, method, cdo_file):
    """Test the dataarray regridders (except conservative)."""
    regridder = getattr(sample_input_data["d2m"].regrid, method)
    da_regrid = regridder(sample_grid_ds)
    ds_cdo = xr.open_dataset(cdo_file)
    xr.testing.assert_allclose(da_regrid.compute(), ds_cdo["d2m"].compute())


@pytest.fixture
def conservative_input_data() -> xr.Dataset:
    return xr.open_dataset(DATA_PATH / "era5_total_precipitation_2020_monthly.nc")


@pytest.fixture
def conservative_sample_grid():
    grid = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=2.2,
        resolution_lon=2.2,
    )

    return xarray_regrid.create_regridding_dataset(grid)


def test_conservative_regridder(conservative_input_data, conservative_sample_grid):
    ds_regrid = conservative_input_data.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = xr.open_dataset(CDO_DATA["conservative"])

    # Cut of the edges: edge performance to be improved later (hopefully)
    no_edges = {"latitude": slice(-85, 85), "longitude": slice(5, 355)}

    xr.testing.assert_allclose(
        ds_regrid["tp"]
        .sel(no_edges)
        .compute()
        .transpose("time", "latitude", "longitude"),
        ds_cdo["tp"].sel(no_edges).compute(),
        rtol=0.002,
        atol=2e-6,
    )
