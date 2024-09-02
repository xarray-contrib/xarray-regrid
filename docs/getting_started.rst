Quickstart
==========

``xarray-regrid`` allows you to regrid xarray Datasets or DataArrays to a new resolution.
You can install xarray-regrid with pip:

.. code:: shell

   pip install xarray-regrid


To use the package, import ``xarray_regrid``. This will register the ``.regrid`` 'accessor' so it can be used.
Next load in the data you want to regrid, and the data with a grid you want to regrid to:

.. code:: python

    import xarray_regrid
    import xarray

    ds = xr.open_dataset("input_data.nc")
    ds_grid = xr.open_dataset("target_grid.nc")

    ds = ds.regrid.linear(ds_grid)

    # or, for example:
    ds = ds.regrid.conservative(ds_grid, latitude_coord="lat")


Multiple regridding methods are available:

* `linear interpolation <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.linear>`_ (``.regrid.linear``)
* `nearest-neighbor <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative>`_ (``.regrid.nearest``)
* `cubic interpolation  <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.cubic>`_ (``.regrid.cubic``)
* `conservative regridding <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative>`_ (``.regrid.conservative``)

Additionally, a zonal statistics `method to compute the most common value <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.most_common>`_
is available (``.regrid.most_common``).
This can be used to upscale very fine categorical data to a more course resolution.
