# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

Added:
 - xarray DataArrays are now supported
 - A conservative regridding method has been added, along with a benchmark notebook.

Changed:
 - The API has changed. Regridding is now done with `xr.Dataset.regrid.method()`. 
   - E.g. `xr.Dataset.regrid.linear()`


## v0.1.0 (2023-08-17)
First release of `xarray-regrid`, containing:
- the implementation of linear, nearest-neighbor and cubic regridding.
- benchmarks against CDO and xESMF for linear and nearest-neighbor.
