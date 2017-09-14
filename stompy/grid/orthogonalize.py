"""
Prototyping some approaches for local orthogonalization
"""
from __future__ import print_function

import numpy as np

from stompy.grid import unstructured_grid

from stompy.utils import (mag, circumcenter, circular_pairs,signed_area, poly_circumcenter,
                          orient_intersection,array_append,within_2d, to_unit,
                          recarray_add_fields,recarray_del_fields)


# approach: adjust a single node relative to all of its
# surrounding cells, at first worrying only about orthogonality
# then start from a cell, and adjust each of its nodes w.r.t 
# to the nodes' neighbors.

class Tweaker(object):
    """
    Bundle optimization methods for unstructured grids.

    Separated from the grid representation itself, this class contains methods
    which act on the given grid.  
    """
    def __init__(self,g):
        self.g=g

    def nudge_node_orthogonal(self,n):
        g=self.g
        n_cells=g.node_to_cells(n)

        centers = g.cells_center(refresh=n_cells,mode='sequential')

        targets=[] # list of (x,y) which fit the individual cell circumcenters
        for n_cell in n_cells:
            cell_nodes=g.cell_to_nodes(n_cell)
            # could potentially skip n_cell==n, since we can move that one.
            if len(cell_nodes)<=3:
                continue # no orthogonality constraints from triangles at this point.

            offsets = g.nodes['x'][cell_nodes] - centers[n_cell,:]
            dists = mag(offsets)
            radius=np.mean(dists)

            # so for this cell, we would like n to be a distance of radius away
            # from centers[n_cell]
            n_unit=to_unit(g.nodes['x'][n]-centers[n_cell])

            good_xy=centers[n_cell] + n_unit*radius
            targets.append(good_xy)
        if len(targets):
            target=np.mean(targets,axis=0)
            g.modify_node(n,x=target)
            return True
        else:
            return False

    def nudge_cell_orthogonal(self,c):
        for n in self.g.cell_to_nodes(c):
            self.nudge_node_orthogonal(n)


if 0: # dev code.
    from stompy.model.delft import dfm_grid
    import matplotlib.pyplot as plt
    from stompy.plot import plot_utils

    g=dfm_grid.DFMGrid('/home/rusty/models/grids/lsb_combined/lsb_combined_v03_net.nc')
    errs=g.circumcenter_errors(radius_normalized=True)

    if 1:
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        coll=g.plot_cells(values=errs,ax=ax,lw=0)
        plot_utils.cbar(coll)

    ## 

    c=45085 # starts with orthogonality error of 0.042
    n=39775

    ## 
    # nudge_cell_orthogonal(g,c)

    while 1:

        try:
            zoom=ax.axis()
        except NameError:
            zoom=(587768.37013030041, 589520.37679034972, 4144031.4469170086, 4145625.8027230976)

        c=g.select_cells_nearest( plt.ginput()[0] )
        err_start=g.circumcenter_errors(radius_normalized=True,cells=[c])[0]

        nudge_cell_orthogonal(g,c)

        errs=g.circumcenter_errors(radius_normalized=True)
        print("Error for cell %d %.4f => %.4f"%(c,err_start,errs[c]))

        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        coll=g.plot_cells(values=errs,ax=ax,lw=0,clip=zoom)
        coll.set_clim([0,0.025])
        plot_utils.cbar(coll)
        ax.axis(zoom)

    ## 
