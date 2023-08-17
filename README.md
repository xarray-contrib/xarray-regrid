# xarray-regrid
Regridding utilities for xarray.

Note: currently only rectilinear grids are supported.

For now xarray-regrid is mostly a wrapper around `ds.interp`, however, conservative regridding is not possible with `interp`, and will need a custom solution.

## Installation

```console
pip install xarray-regrid
```

## Usage
The xarray-regrid routines are accessed using the "regrid" accessor on an xarray Dataset:
```py
import xarray_regrid

ds = xr.open_dataset("input_data.nc")
ds_grid = xr.open_dataset("target_grid.nc")

ds.regrid.regrid(ds_grid, method="linear")
```
Currently implemented are the methods linear, nearest and cubic.

For examples, see the benchmark notebooks.

## Benchmarks
The benchmark notebooks contain comparisons to more standard methods (CDO, xESMF).

To be able to run the notebooks, a conda environment is required (due to ESMF and CDO).
You can install this environment using the `environment.yml` file in this repository.
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a lightweight version of the much faster "mamba" conda alternative.

## Planned features
- Support conservative regridding
