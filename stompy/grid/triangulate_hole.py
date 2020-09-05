from .. import utils
import numpy as np
from ..spatial import field
from . import unstructured_grid, front, rebay

def triangulate_hole(grid,seed_point=None,nodes=None,max_nodes=5000,hole_rigidity='cells',
                     splice=True,return_value='grid',dry_run=False,
                     method='front'):
    """
    Specify one of
      seed_point: find node string surrounding this point
      nodes: specify CCW ordered nodes making up the hole.

    method:
     'front': use front.py, which produces high quality grids very 
        slowly.
     'rebay': use rebay.py, much faster, but without any guarantees on cell quality,
        and without support for sliding boundaries (hole_rigidity must be 'all')

    hole_rigidity: 
       'cells' nodes and edges which are part of a cell are considered rigid.
       'all': all nodes and edges of the hole are considered rigid.
       'all-nodes': edges can be subdivided, but nodes cannot be moved.
    This affects both the calculation of local scale and the state of nodes
     and edges during triangulation.

    splice: if true, the new grid is spliced into grid. if false, just returns
     the grid covering the hole.

    return_value: grid: return the resulting grid
    front: advancing front instance
    dry_run: if True, get everything set up but don't triangulate
    """
    if method=='rebay' and hole_rigidity!='all':
        raise Exception("Currently rebay can only handle fully rigid hole (hole_rigidity='all')")
    
    # manually tell it where the region to be filled is.
    # 5000 ought to be plenty of nodes to get around this loop
    if nodes is None:
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
        if hole_rigidity=='cells':
            if np.all( grid.edges['cells'][j] < 0):
                continue
        elif hole_rigidity in ['all','all-nodes']:
            pass 
        sample_xy.append(ec[j])
        sample_scale.append(el[j])

    assert len(sample_xy)
    sample_xy=np.array(sample_xy)
    sample_scale=np.array(sample_scale)

    apollo=field.PyApolloniusField(X=sample_xy,F=sample_scale)

    # For hole_rigidity=='all', there are no hints, and this stays
    # empty.  Only for method=='front' can there be other values of
    # hole_rigidity.
    src_hints=[]

    # Prepare that shoreline for grid generation.
    if method=='front':
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
        for n in AT.grid.valid_node_iter():
            n_src=grid.select_nodes_nearest(AT.grid.nodes['x'][n])
            delta=utils.dist( grid.nodes['x'][n_src], AT.grid.nodes['x'][n] )
            assert delta<0.1 # should be 0.0

            if len(grid.node_to_cells(n_src))==0:
                if hole_rigidity=='cells':
                    # It should be a HINT
                    AT.grid.nodes['fixed'][n]=AT.HINT
                    src_hints.append(n_src)
                if hole_rigidity in ['cells','all-nodes']:
                    # And any edges it participates in should not be RIGID either.
                    for j in AT.grid.node_to_edges(n):
                        AT.grid.edges['fixed'][j]=AT.UNSET

        AT.scale=apollo

        if dry_run:
            if return_value=='grid':
                return AT.grid
            else:
                return AT

        if AT.loop():
            AT.grid.renumber()
            g_result=AT.grid
        else:
            print("Grid generation failed")
            return AT # for debugging -- need to keep a handle on this to see what's up.
    elif method=='rebay':
        grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=len(xy_shore))
        nodes=[grid_to_pave.add_node(x=xy) for xy in xy_shore]
        c=grid_to_pave.add_cell_and_edges(nodes)
        rad=rebay.RebayAdvancingDelaunay(grid=grid_to_pave,scale=apollo,
                                         heap_sign=1)
        rad.execute()
        # make it look kind of like the advancing triangles output
        g_result=rad.extract_result()
        AT=rad

    if not splice:
        if return_value=='grid':
            return AT.grid
        else:
            return AT
    else:    
        for n in src_hints:
            grid.delete_node_cascade(n)

        grid.add_grid(g_result)

        # Surprisingly, this works!
        grid.merge_duplicate_nodes()

        grid.renumber(reorient_edges=False)

        if return_value=='grid':
            return grid
        else:
            return AT
