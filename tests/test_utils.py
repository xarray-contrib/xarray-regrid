import numpy as np
import xarray as xr

from xarray_regrid.utils import format_lat


def test_format_lat():
    lat_vals = np.arange(-89.5, 89.5 + 1, 1)
    lon_vals = np.arange(-179.5, 179.5 + 1, 1)
    x_vals = np.broadcast_to(lat_vals, (len(lon_vals), len(lat_vals)))
    ds = xr.Dataset(
        data_vars={"x": (("lon", "lat"), x_vals)},
        coords={"lat": lat_vals, "lon": lon_vals},
        attrs={"foo": "bar"},
    )
    ds.lat.attrs["is"] = "coord"
    ds.x.attrs["is"] = "data"

    formatted = format_lat(ds, ds, {"lat": "lat", "lon": "lon"})
    # Check that lat has been extended to include poles
    assert formatted.lat.values[0] == -90
    assert formatted.lat.values[-1] == 90
    # Check that data has been extrapolated to include poles
    assert (formatted.x.isel(lat=0) == -89.5).all()
    assert (formatted.x.isel(lat=-1) == 89.5).all()
    # Check that attrs have been preserved
    assert formatted.attrs["foo"] == "bar"
    assert formatted.lat.attrs["is"] == "coord"
    assert formatted.x.attrs["is"] == "data"
