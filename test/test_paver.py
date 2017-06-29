from stompy.grid import paver

##

# wrap/adapt exact_delaunay to the point that it can stand in for
# CGAL in paver.py.

# who uses CGAL? 

##

# will have to return to CGAL, as the conda package is broken and I can't
# update it right now.

import CGAL
from CGAL import CGAL_Triangulation_2

# this was failing because of..

# ImportError: dlopen(/Users/rusty/anaconda/lib/python2.7/site-packages/CGAL/_CGAL_Triangulation_2.so, 2): Library not loaded: @rpath/libgmp.10.dylib
#   Referenced from: /Users/rusty/anaconda/lib/python2.7/site-packages/CGAL/_CGAL_Triangulation_2.so
#   Reason: Incompatible library version: _CGAL_Triangulation_2.so requires version 14.0.0 or later, but libgmp.10.dylib provides version 12.0.0

# and this got past conda because the build of mpfr isn't specific about the version of gmp, just says
# it depends on gmp.

# conda update gmp to install 6.1.2
# that allows the import to succeed.

##

from stompy.grid import paver

# Import a bunch of libraries.
import numpy as np
import matplotlib.pyplot as plt
from stompy.spatial import field
from stompy.spatial import constrained_delaunay

##

# Define a polygon
boundary=np.array([[0,0],[1000,0],[1000,1000],[0,1000]])
island  =np.array([[200,200],[600,200],[200,600]])

rings=[boundary,island]

##

# And the scale:
scale=field.ConstantField(50)

##
import pdb
# It fails here, in initialize_rings, in Paving.__init__
# initialize_boundaries survives, presumably it's the refresh_metadata
# call which fails
# pdb.run("paver.Paving(rings=rings,density=scale)")
p=paver.Paving(rings=rings,density=scale)


##

p.pave_all() # writes cryptic progress messages - takes about 30s
