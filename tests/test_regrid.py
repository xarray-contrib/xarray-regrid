from copy import deepcopy
from pathlib import Path

import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import xarray_regrid

DATA_PATH = Path(__file__).parent.parent / "benchmarks" / "data"

CDO_DATA = {
    "linear": DATA_PATH / "cdo_bilinear_64b.nc",
    "nearest": DATA_PATH / "cdo_nearest_64b.nc",
    "conservative": DATA_PATH / "cdo_conservative_64b.nc",
}


@pytest.fixture(scope="session")
def load_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_2m_dewpoint_temperature_2000_monthly.nc")
    return ds.compute()


@pytest.fixture
def sample_input_data(load_input_data) -> xr.Dataset:
    return deepcopy(load_input_data)


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


@pytest.fixture(scope="session")
def load_conservative_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_total_precipitation_2020_monthly.nc")
    return ds.compute()


@pytest.fixture
def conservative_input_data(load_conservative_input_data) -> xr.Dataset:
    return deepcopy(load_conservative_input_data)


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


def test_conservative_nans(conservative_input_data, conservative_sample_grid):
    ds = conservative_input_data
    ds["tp"] = ds["tp"].where(ds.latitude >= 0).where(ds.longitude < 180)
    ds_regrid = ds.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = xr.open_dataset(CDO_DATA["conservative"])

    # Cut of the edges: edge performance to be improved later (hopefully)
    no_edges = {"latitude": slice(-85, 85), "longitude": slice(5, 355)}
    no_nans = {"latitude": slice(1, 90), "longitude": slice(None, 179)}
    xr.testing.assert_allclose(
        ds_regrid["tp"]
        .sel(no_edges)
        .sel(no_nans)
        .compute()
        .transpose("time", "latitude", "longitude"),
        ds_cdo["tp"].sel(no_edges).sel(no_nans).compute(),
        rtol=0.002,
        atol=2e-6,
    )


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataarray(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data["d2m"].regrid, method)
    da_regrid = regridder(sample_grid_ds)
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataarray_conservative(sample_input_data, sample_grid_ds):
    da_regrid = sample_input_data["d2m"].regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataset(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data.regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataset_conservative(sample_input_data, sample_grid_ds):
    ds_regrid = sample_input_data.regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["d2m"].attrs == sample_input_data["d2m"].attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


class TestCoordOrder:
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_original(self, sample_input_data, sample_grid_ds, method, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_reversed(
        self, sample_input_data, sample_grid_ds, method, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_original(self, sample_input_data, sample_grid_ds, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_reversed(
        self, sample_input_data, sample_grid_ds, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])
