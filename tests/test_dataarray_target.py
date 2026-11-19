import dask.array as da
import numpy as np
import pytest
import xarray as xr
from dask import delayed

import xarray_regrid  # noqa: F401


@pytest.mark.parametrize(
    ("method", "as_dataset"),
    [
        (method, as_dataset)
        for method in ["linear", "nearest", "cubic", "conservative", "stat"]
        for as_dataset in [False, True]
    ]
    + [("most_common", False), ("least_common", False)],
)
@pytest.mark.parametrize("target_name", [None, "target"])
@pytest.mark.parametrize("target_has_time", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_dataarray_target_matches_dataset(
    method, as_dataset, target_name, target_has_time, chunked
):
    source = xr.DataArray(
        np.tile(np.arange(36).reshape(6, 6) % 3, (2, 1, 1)),
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": np.arange(6), "x": np.arange(6)},
        name="source",
    )
    if as_dataset:
        source = source.to_dataset()
    # The same grid, built once as a Dataset and once as a DataArray.
    grid = xr.Dataset(coords={"y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]})
    target = xr.DataArray(
        np.full((3, 3), np.nan),
        dims=("y", "x"),
        coords={"y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]},
        name=target_name,
    )
    if target_has_time:
        grid = grid.assign_coords(time=[10, 20])
        target = target.expand_dims(time=[10, 20])
    if chunked:
        source = source.chunk({"time": 1})
        target = target.chunk()
    original = target.copy(deep=True)
    kwargs = {}
    if method in ["most_common", "least_common"]:
        kwargs["values"] = np.array([0, 1, 2])
    elif method == "stat":
        kwargs["method"] = "mean"
    regrid = getattr(source.regrid, method)
    expected = regrid(grid, **kwargs)
    actual = regrid(target, **kwargs)
    xr.testing.assert_identical(actual.compute(), expected.compute())
    xr.testing.assert_identical(target, original)


@pytest.mark.parametrize("method", ["linear", "conservative"])
def test_unnamed_dataarray_target_with_time_values(method):
    # The source only varies along x (0, 1, 2, 0, 1, 2), so the values on the
    # target grid can be worked out by hand.
    source = xr.DataArray(
        np.tile(np.arange(6.0) % 3, (2, 6, 1)),
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": np.arange(6), "x": np.arange(6)},
    )
    target = xr.DataArray(
        np.full((3, 3), np.nan),
        dims=("y", "x"),
        coords={"y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]},
    ).expand_dims(time=[10, 20])
    expected = xr.DataArray(
        np.tile([0.5, 1.0, 1.5], (2, 3, 1)),
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]},
    )
    xr.testing.assert_allclose(getattr(source.regrid, method)(target), expected)


def test_dataarray_target_without_coords_reports_missing_coords():
    source = xr.DataArray(
        np.ones((6, 6)),
        dims=("y", "x"),
        coords={"y": np.arange(6), "x": np.arange(6)},
    )
    target = xr.DataArray(np.ones((3, 3)), dims=("y", "x"))
    with pytest.raises(ValueError, match="None of the target coords are in the data"):
        source.regrid.linear(target)


def test_dataarray_target_dim_without_coord_matches_dataset():
    source = xr.DataArray(
        np.arange(36.0).reshape(6, 6),
        dims=("y", "x"),
        coords={"y": np.arange(6), "x": np.arange(6)},
    )
    target = xr.DataArray(
        np.ones((3, 3, 2)),
        dims=("y", "x", "bnds"),
        coords={"y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]},
    )
    expected = source.regrid.linear(target.to_dataset(name="target"))
    xr.testing.assert_identical(source.regrid.linear(target), expected)


def test_dataarray_target_values_are_not_computed():
    @delayed
    def unavailable_values():
        msg = "Target data values must not be computed"
        raise AssertionError(msg)

    target = xr.DataArray(
        da.from_delayed(unavailable_values(), shape=(3,), dtype=float),
        dims="x",
        coords={"x": [0.5, 2.5, 4.5]},
    ).expand_dims(time=[10, 20])
    source = xr.DataArray(np.ones(6), dims="x", coords={"x": np.arange(6)})
    actual = source.regrid.linear(target).compute()
    xr.testing.assert_identical(
        actual, xr.DataArray(np.ones(3), dims="x", coords={"x": [0.5, 2.5, 4.5]})
    )
