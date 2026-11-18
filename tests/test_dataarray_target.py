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
    target = xr.DataArray(
        np.full((3, 3), np.nan),
        dims=("y", "x"),
        coords={"y": [0.5, 2.5, 4.5], "x": [0.5, 2.5, 4.5]},
        name=target_name,
    )
    if target_has_time:
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
    expected = regrid(target.coords.to_dataset(), **kwargs)
    actual = regrid(target, **kwargs)
    xr.testing.assert_identical(actual.compute(), expected.compute())
    xr.testing.assert_identical(target, original)


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
