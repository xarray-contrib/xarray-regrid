# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

# 0.3.0 (2024-09-05)

New contributors:
 - [@slevang](https://github.com/slevang)

Fixed:
 - conservative regridding now can be constructed fully lazily [#39](https://github.com/EXCITED-CO2/xarray-regrid/pull/39).

Added:
 - documentation for the package, including readthedocs integration ([#40](https://github.com/EXCITED-CO2/xarray-regrid/pull/40)).
 - better handling of NaNs to the conservative regridding routine, with a `nan_threshold` keyword argument. For more information see the notebooks on the documentation [#39](https://github.com/EXCITED-CO2/xarray-regrid/pull/39) \& [#41](https://github.com/EXCITED-CO2/xarray-regrid/pull/41).
 - `create_regridding_dataset` as a method of the `xarray_regrid.Grid` dataclass [#41](https://github.com/EXCITED-CO2/xarray-regrid/pull/41).

## v0.2.3 (2024-02-29)

New contributors:
 - [@kjdoore](https://github.com/kjdoore)

Fixed:
 - Ensure all attributes are kept upon regridding (dataset, variable and coordinate attrs) ([#27](https://github.com/EXCITED-CO2/xarray-regrid/pull/27)).
 - The target grid can now have coordinates sorted in decending order, instead of the regridding failing ([#29](https://github.com/EXCITED-CO2/xarray-regrid/pull/29)).
 - Regridding to larger grid now result in NaNs at locations outside of starting data grid ([#33](https://github.com/EXCITED-CO2/xarray-regrid/pull/33)).

Changed:
 - Moved to the Ruff formatter, instead of black ([#27](https://github.com/EXCITED-CO2/xarray-regrid/pull/27)).

## v0.2.2 (2023-11-24)

Added:
 - CITATION.cff file for Zenodo integration.

## v0.2.1 (2023-09-05)

Fixed:
 - Datasets containing NaN values can now be regridded using the conservative method. This previously produced only NaN values.

## v0.2.0 (2023-09-02)

Added:
 - xarray DataArrays are now supported
 - Conservative regridding method, along with a benchmark notebook.
 - A "most common value" regridding method, along with a demo notebook.

Changed:
 - The API has changed. Regridding is now done with `xr.Dataset.regrid.method()`. 
   - E.g. `xr.Dataset.regrid.linear()`


## v0.1.0 (2023-08-17)
First release of `xarray-regrid`, containing:
- the implementation of linear, nearest-neighbor and cubic regridding.
- benchmarks against CDO and xESMF for linear and nearest-neighbor.
