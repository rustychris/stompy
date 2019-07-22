from .. import utils
import numpy as np
from ..spatial import field
from . import unstructured_grid, front

def triangulate_hole(grid,seed_point,max_nodes=5000):
    # manually tell it where the region to be filled is.
    # 5000 ought to be plenty of nodes to get around this loop
    nodes=grid.enclosing_nodestring(seed_point,max_nodes)
    xy_shore=grid.nodes['x'][nodes]

    # Construct a scale based on existing spacing
    # But only do this for edges that are part of one of the original grids
    grid.edge_to_cells() # update edges['cells']
    sample_xy=[]
    sample_scale=[]
    ec=grid.edges_center()
    el=grid.edges_length()

    for na,nb in utils.circular_pairs(nodes):
        j=grid.nodes_to_edge([na,nb])
        if np.any( grid.edges['cells'][j] >= 0 ):
            sample_xy.append(ec[j])
            sample_scale.append(el[j])

    sample_xy=np.array(sample_xy)
    sample_scale=np.array(sample_scale)

    apollo=field.PyApolloniusField(X=sample_xy,F=sample_scale)

    # Prepare that shoreline for grid generation.

    grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=6)

    AT=front.AdvancingTriangles(grid=grid_to_pave)

    AT.add_curve(xy_shore)
    # This should be safe about not resampling existing edges
    AT.scale=field.ConstantField(50000)

    AT.initialize_boundaries()

    AT.grid.nodes['fixed'][:]=AT.RIGID
    AT.grid.edges['fixed'][:]=AT.RIGID

    # Old code compared nodes to original grids to figure out RIGID
    # more general, if it works, to see if a node participates in any cells.
    # At the same time, record original nodes which end up HINT, so they can
    # be removed later on.
    src_hints=[]
    for n in AT.grid.valid_node_iter():
        n_src=grid.select_nodes_nearest(AT.grid.nodes['x'][n])
        delta=utils.dist( grid.nodes['x'][n_src], AT.grid.nodes['x'][n] )
        assert delta<0.1 # should be 0.0

        if len(grid.node_to_cells(n_src))==0:
            # It should be a HINT
            AT.grid.nodes['fixed'][n]=AT.HINT
            src_hints.append(n_src)
            # And any edges it participates in should not be RIGID either.
            for j in AT.grid.node_to_edges(n):
                AT.grid.edges['fixed'][j]=AT.UNSET

    AT.scale=apollo
    
    if AT.loop():
        AT.grid.renumber()
    else:
        print("Grid generation failed")
        return AT # for debugging -- need to keep a handle on this to see what's up.

    for n in src_hints:
        grid.delete_node_cascade(n)
        
    grid.add_grid(AT.grid)

    # Surprisingly, this works!
    grid.merge_duplicate_nodes()

    grid.renumber()

    return grid
