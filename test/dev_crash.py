import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb

from scipy import optimize as opt

from stompy.spatial import field
from stompy import utils

from stompy.grid import (unstructured_grid, exact_delaunay, front)

import logging
logging.basicConfig(level=logging.INFO)

from stompy.spatial.linestring_utils import upsample_linearring,resample_linearring
from stompy.spatial import field,constrained_delaunay,wkb2shp

## Curve -

def hex_curve():
    hexagon = np.array( [[0,11],
                         [10,0],
                         [30,0],
                         [40,9],
                         [30,20],
                         [10,20]] )
    return front.Curve(hexagon)


def test_basic_setup():
    boundary=hex_curve()
    af=front.AdvancingTriangles()
    scale=field.ConstantField(3)

    af.add_curve(boundary)
    af.set_edge_scale(scale)

    # create boundary edges based on scale and curves:
    af.initialize_boundaries()

    return af

# when resample nodes on a sliding boundary, want to calculate the available
# span, and if it's small, start distributing the nodes evenly.
# where small is defined by local_scale * max_span_factor

def test_resample():
    af=test_basic_setup()
    a=0
    b=af.grid.node_to_nodes(a)[0]
    he=af.grid.nodes_to_halfedge(a,b)
    anchor=he.node_rev()
    n=he.node_fwd()
    n2=he.rev().node_rev()
    # Fails here, in grid.modify_node
    af.resample(n=n,anchor=anchor,scale=25,direction=1)
    af.resample(n=n2,anchor=anchor,scale=25,direction=-1)
    
test_resample()


# during modify_node(n=9)
# 9 comes in as node b in call to line_is_free
#   vertex handle gives it as 22.5,0.0, which is the new location
# lw from line_walk is bad.
# after_add_node() just inserts the new point into the DT.
#   - could be related to premature garbage collection of points?
#     nope.
#   - related to init_face? has to be there for proper functioning
#   - or failure to remove the original vertex before creating the new one?
#     no, that seems to be taken care of.

# does a line free call work before modifying the node?
#  nope. So maybe something else in the early part of before_modify_node
#  invalidates the state?
# it's the second time through the loop that fails?
# 10--9 crashes, even when it's the first in the loop
# even if we could drop init_face, it segfaults without it.
# segfaults when performing the line walk on a deep copy of DT.
# the test it is attempting is along an existing finite edge.
# happens whether the edge is constrained or not.

# Possible next steps:
#  1. could remove the node, insert in the new spot, maybe do a locate first?
#     and for any nodes which are now DT neighbors clearly we can skip the
#     line_is_free.
#  2. hand-write the line_is_free stuff, ala live_dt.
#  3. Abstract out the line_is_free stuff in live_dt, and both that and this
#     can use it.
