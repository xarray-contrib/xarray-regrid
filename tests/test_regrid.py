from pathlib import Path

import pytest
import xarray as xr

import xarray_regrid

DATA_PATH = Path(__file__).parent.parent / "benchmarks" / "data"

CDO_DATA = {
    "linear": DATA_PATH / "cdo_bilinear_64b.nc",
    "nearest": DATA_PATH / "cdo_nearest_64b.nc",
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
    "method, cdo_file", [
        ("linear", CDO_DATA["linear"]),
        ("nearest", CDO_DATA["nearest"]),
    ])
def test_regridder(sample_input_data, sample_grid_ds, method, cdo_file):
    ds_regrid = sample_input_data.regrid.regrid(sample_grid_ds, method=method)
    ds_cdo = xr.open_dataset(cdo_file)
    xr.testing.assert_allclose(ds_regrid.compute(), ds_cdo.compute())
