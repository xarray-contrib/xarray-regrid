import numpy as np
import xarray as xr

import xarray_regrid
from xarray_regrid.utils import format_for_regrid


def test_covered():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=80,
        east=350,
        south=-80,
        west=10,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Formatting utils shouldn't modify this one at all
    xr.testing.assert_equal(source, formatted)


def test_no_edges():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90 - dx_source / 2,
        east=360 - dx_source / 2,
        south=-90 + dx_source / 2,
        west=0 + dx_source / 2,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Should add wraparound and polar padding rows/columns
    assert formatted.latitude[0] == -90
    assert formatted.latitude[-1] == 90
    assert formatted.longitude[0] == -1
    assert formatted.longitude[-1] == 361
    assert (formatted.longitude.diff("longitude") == 2).all()


def test_360_to_180():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=90,
        east=180,
        south=-90,
        west=-180,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Should produce a shift to target plus wraparound padding
    assert formatted.longitude[0] == -182
    assert formatted.longitude[-1] == 182
    assert (formatted.longitude.diff("longitude") == 2).all()


def test_180_to_360():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90,
        east=180,
        south=-90,
        west=-180,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Should produce a shift to target plus wraparound padding
    assert formatted.longitude[0] == -2
    assert formatted.longitude[-1] == 362
    assert (formatted.longitude.diff("longitude") == 2).all()


def test_0_to_360():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90,
        east=0,
        south=-90,
        west=-360,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Should produce a shift to target plus wraparound padding
    assert formatted.longitude[0] == -2
    assert formatted.longitude[-1] == 362
    assert (formatted.longitude.diff("longitude") == 2).all()


def test_global_to_local_shift():
    dx_source = 2
    source = xarray_regrid.Grid(
        north=90,
        east=180,
        south=-90,
        west=-180,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()

    dx_target = 1
    target = xarray_regrid.Grid(
        north=90,
        east=300,
        south=-90,
        west=270,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target)

    # Should produce a shift to cover the target range
    assert formatted.longitude.min() <= 270
    assert formatted.longitude.max() >= 300
    assert (formatted.longitude.diff("longitude") == 2).all()


def test_stats():
    """Special handling for statistical aggregations."""
    dx_source = 1
    source = xarray_regrid.Grid(
        north=90 - dx_source / 2,
        east=360 - dx_source / 2,
        south=-90 + dx_source / 2,
        west=0 + dx_source / 2,
        resolution_lat=dx_source,
        resolution_lon=dx_source,
    ).create_regridding_dataset()
    source["data"] = xr.DataArray(
        np.random.randint(0, 10, (source.latitude.size, source.longitude.size)),
        dims=["latitude", "longitude"],
        coords={"latitude": source.latitude, "longitude": source.longitude},
    )

    dx_target = 2
    target = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=dx_target,
        resolution_lon=dx_target,
    ).create_regridding_dataset()

    formatted = format_for_regrid(source, target, stats=True)

    # Statistical aggregations should skip Polar padding
    assert formatted.latitude.equals(source.latitude)
    # But should apply wraparound longitude padding
    assert formatted.longitude[0] == -1.5
    assert formatted.longitude[-1] == 361.5
    # And preserve integer dtypes
    assert formatted.data.dtype == source.data.dtype
    assert (formatted.longitude.diff("longitude") == 1).all()
