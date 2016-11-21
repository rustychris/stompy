#import matplotlib.pyplot as plt
plt=None
import numpy as np
import field
from scipy import optimize as opt
import utils

import unstructured_grid
import exact_delaunay
import front

#-# Curve -

def hex_curve():
    hexagon = np.array( [[0,1],
                         [1,0],
                         [3,0],
                         [4,1],
                         [3,2],
                         [1,2]] )
    return front.Curve(10*hexagon)

def test_basic_setup():
    boundary=hex_curve()
    af=front.AdvancingFront()
    scale=field.ConstantField(3)

    af.add_curve(boundary)
    af.set_edge_scale(scale)

    # create boundary edges based on scale and curves:
    af.initialize_boundaries()

    if plt:
        plt.clf()
        g=af.grid
        g.plot_edges()
        g.plot_nodes()

        # 
        coll=g.plot_halfedges(values=g.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')
        

    return af

af=test_basic_setup()

af.loop()

# where is the bulk of the time?

# slow...
# 37s for ~200 cells.
# 11.6s in optimization (one_point_cost: 9s)
# 23s in exact_delaunay:modify_node
# so it's roughly 2/3 delaunay maintenance, 1/3 node optimization (not
# counting the resulting edit)

# node optimization is mostly evaluating cost
# delaunay maintenance is over 40% double-checking the structure,
# 30% propagating flip

# line profile of one-point cost turns up a max of 12% one the line
# calculating all_angles.
# maybe there is a better way to calculate that, but the bottom line is
# that no single change is going to improve time by more than about 10%.
# got 84484 hits on the function

# currently using opt.fmin, which is nelder-mead, same as what paver used.
# any chance of evaluating the hessian directly?  we're only in 2D, so if
# it's more expensive than two calls to cost, it's probably not
# worth it.

# what about propagating_flip?
# 32% in he.fwd()
# 53% in flip_edge()
# another 10% in he.rev().opposite()

# and nbr() in HalfEdge? 6.6s total (so about 20-25%)
#  86% is angle_sort_adjacent_nodes
#  9% HalfEdge.from_nodes
