
.. toctree::
  :maxdepth: 3
  :caption: User Guide
  :hidden:
  
  Quickstart <getting_started>
  Example Notebooks <notebooks/index>


.. toctree::
  :maxdepth: 3
  :caption: Technical information
  :hidden:

  Changelog <changelog_link>

xarray-regrid: Regridding utilities for xarray
**********************************************

|PyPI| |DOI|

Overview
========

``xarray-regrid`` extends xarray with regridding methods, making it possibly to easily and effiently regrid between two rectilinear grids. 

The following methods are supported:

* Linear
* Nearest-neighbor
* Conservative
* Cubic
* "Most common value" (zonal statistics)

Note that "Most common value" is designed to regrid categorical data to a coarse resolution. For regridding categorical data to a finer resolution, please use "nearest-neighbor" regridder.

Installing
==========

.. code:: shell

   pip install xarray-regrid


Acknowledgements
================

This package was developed under Netherlands eScience Center grant `NLESC.OEC.2022.017 <https://research-software-directory.org/projects/excited>`_.

Some methods were inspired by discussions in the `Pangeo <https://pangeo.io>`_ community.

.. |PyPI| image:: https://img.shields.io/pypi/v/xarray-regrid.svg?style=flat
   :target: https://pypi.python.org/pypi/xarray-regrid/

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10203304.svg
   :target: https://doi.org/10.5281/zenodo.10203304
