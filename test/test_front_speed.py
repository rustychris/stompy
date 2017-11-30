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

    if 0:
        plt.clf()
        g=af.grid
        g.plot_edges()
        g.plot_nodes()

        # 
        coll=g.plot_halfedges(values=g.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')

    return af


def test_no_lookahead():
    af=test_basic_setup()
    af.log.setLevel(logging.INFO)
    af.cdt.post_check=False

    af.current=af.root=front.DTChooseSite(af)

    def cb():
        af.plot_summary(label_nodes=False)
        try:
            af.current.site.plot()
        except: # AttributeError:
            pass

    while 1:
        if not af.current.children:
            break # we're done?

        for child_i in range(len(af.current.children)):
            if af.current.try_child(child_i):
                # Accept the first child which returns true
                break
        else:
            assert False # none of the children worked out
    return af
##
# 172 cells, with CGAL, takes 8.7s ? so 20 cells/second.

test_no_lookahead()

# 10s in prun
# removed some log.debug statements
# 8.3s
# 5.8s in optimize, 1.1s in line_walk

## 
