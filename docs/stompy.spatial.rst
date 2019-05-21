stompy\.spatial — Spatial analysis, data management and manipulation.
=======================

For GIS-related operations, and spatial tools to support model
development and data analysis.

algorithms
----------

General geometric algorithms that do not fit cleanly into
more special-purpose modules.

.. automodule:: stompy.spatial.algorithms
    :members:
    :undoc-members:
    :show-inheritance:

field
-----

Various classes representing a 2D scalar-valued function.  Useful
for bathymetry processing, creating parameter maps for model input.
Includes both point-based and raster-based classes.

.. automodule:: stompy.spatial.field
    :members:
    :undoc-members:
    :show-inheritance:

gen\_spatial\_index
-------------------

A generic interface to several implementations of spatial indexes.

.. automodule:: stompy.spatial.gen_spatial_index
    :members:
    :undoc-members:
    :show-inheritance:

interp\_4d module
-----------------

A heat-diffusion method for interpolating sparse data in space and time onto
a grid.  This can be used to take point samples collected over time and create
a spatially smooth interpolation which respects shorelines and other features
in a computational grid.

.. automodule:: stompy.spatial.interp_4d
    :members:
    :undoc-members:
    :show-inheritance:

interp\_coverage
----------------------------------------

Special-purpose constrained linear interpolation.  Used by some methods in
field.

.. automodule:: stompy.spatial.interp_coverage
    :members:
    :undoc-members:
    :show-inheritance:

join\_features
--------------

Join many line segments into topologically consistent rings or
polygons (potentially with holes). Useful if you have a shoreline
made of many line segments, but want a single shoreline polyline
or polygon.

.. automodule:: stompy.spatial.join_features
    :members:
    :undoc-members:
    :show-inheritance:

kdtree\_spatialindex
--------------------

Wrapper for using scipy's kdtree as a spatial index.

.. automodule:: stompy.spatial.kdtree_spatialindex
    :members:
    :undoc-members:
    :show-inheritance:

linestring\_utils
-----------------

Simple methods for resampling a polyline to higher or lower resolution.

.. automodule:: stompy.spatial.linestring_utils
    :members:
    :undoc-members:
    :show-inheritance:

proj\_utils
-----------------------------------

A few utility functions related to geographic projections, mainly
the mapper() function.

.. automodule:: stompy.spatial.proj_utils
    :members:
    :undoc-members:
    :show-inheritance:

qgis\_spatialindex module
------------------------------------------

A wrapper for using the QGIS spatial index class in stompy.

.. automodule:: stompy.spatial.qgis_spatialindex
    :members:
    :undoc-members:
    :show-inheritance:

robust\_predicates module
------------------------------------------

A pure python implementation of Jonathan Shewchuk's robust
geometric predicates.  Exact evaluation of tests for collinearity
(does point A fall left/right/on a line joining points B,C), and
in-circle (does point D fall inside/outside/on a circle defined
by points A,B,C).

.. automodule:: stompy.spatial.robust_predicates
    :members:
    :undoc-members:
    :show-inheritance:

wkb2shp module
-------------------------------

Read and write data to/from shapefiles.

.. automodule:: stompy.spatial.wkb2shp
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: stompy.spatial
    :members:
    :undoc-members:
    :show-inheritance:
