Quickstart
==========

``xarray-regrid`` allows you to regrid xarray Datasets or DataArrays to a new resolution.
You can install with pip:

.. code:: shell

   pip install xarray-regrid


To use the package, import ``xarray_regrid``. This will register the ``.regrid`` 'accessor' so it can be used.
Next load in the data you want to regrid, and the data with a grid you want to regrid to:

.. code:: python

    import xarray_regrid
    import xarray

    ds = xr.open_dataset("input_data.nc")
    ds_grid = xr.open_dataset("target_grid.nc")

    ds.regrid.linear(ds_grid)

Multiple regridding methods are available:

* linear interpolation (``.regrid.linear``)
* nearest-neighbor (``.regrid.nearest``)
* cubic interpolation (``.regrid.cubic``)
* conservative regridding (``.regrid.conservative``)

Additionally, a zonal statistics method to compute the most common value is available
(``.regrid.most_common``).
This can be used to upscale very fine categorical data to a more course resolution.
