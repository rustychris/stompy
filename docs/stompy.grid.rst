stompy\.grid — Grid reading, writing and manipulating
====================

Modules for generating, converting, plotting, and editing
unstructured grids as used in hydrodynamic models (e.g.
UnTRIM, SUNTANS, DFlow-FM).

For generating grids, the only stable and robust code for
this is in `tom.py` (triangular orthogonal mesher) and `paver.py`.
`tom.py` is a command line interface to `paver.py`.  A simple
example of calling tom is in `stompy/tests/test_tom.sh`, and
invoking `tom.py -h` will show the other options available.

For most other grid-related tasks, the best module to use
is `unstructured_grid.py`, as it supports non-triangular meshes
(such as mixed triangles/quads), and is actively developed.
Grid generation methods built on unstructured_grid.py are
in front.py, but these are not stable and generally should not
be used.

Note that for any significant amount of grid modification, the
CGAL python bindings are essential. A backup pure python
implementation is included, but will be orders of magnitude
slower and likely less robust numerically.  Watch for error
messages near the start of using tom.py to see whether there
are issues loading CGAL.

Submodules
----------

stompy\.grid\.depth\_connectivity module
----------------------------------------

.. automodule:: stompy.grid.depth_connectivity
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.exact\_delaunay module
------------------------------------

.. automodule:: stompy.grid.exact_delaunay
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.front module
--------------------------

.. automodule:: stompy.grid.front
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.merge\_ugrid\_subgrids module
-------------------------------------------

.. automodule:: stompy.grid.merge_ugrid_subgrids
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.orthogonalize module
----------------------------------

.. automodule:: stompy.grid.orthogonalize
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.orthomaker module
-------------------------------

.. automodule:: stompy.grid.orthomaker
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.paver module
--------------------------

.. automodule:: stompy.grid.paver
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.tom module
------------------------

.. automodule:: stompy.grid.tom
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.trigrid module
----------------------------

.. automodule:: stompy.grid.trigrid
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.ugrid module
--------------------------

.. automodule:: stompy.grid.ugrid
    :members:
    :undoc-members:
    :show-inheritance:

stompy\.grid\.unstructured\_grid module
---------------------------------------

.. automodule:: stompy.grid.unstructured_grid
    :members:
    :undoc-members:
    :show-inheritance:


Module contents
---------------

.. automodule:: stompy.grid
    :members:
    :undoc-members:
    :show-inheritance:
