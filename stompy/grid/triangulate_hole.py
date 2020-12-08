from .. import utils
import numpy as np
from textwrap import dedent
from ..spatial import field
from . import unstructured_grid, front, rebay
import pickle

import logging
logger=logging.getLogger(__name__)

class Gmsher(object):
    scale_file='scale.pkl'
    geo_file='tmp.geo'
    msh_file='tmp.msh'
    gmsh='gmsh'
    output='capture'
    tol=1e-3
    def __init__(self,g_in,scale):
        self.g_in=g_in
        self.scale=scale
    def execute(self):
        import subprocess
        with open(self.scale_file,'wb') as fp:
            pickle.dump(self.scale,fp)

        self.g_in.write_gmsh_geo(self.geo_file)

        with open(self.geo_file,'at') as fp:
            # Turns out exec() in field.py is not python2 compatible.
            # hack fix force python3
            fp.write(dedent("""
            Field[1] = ExternalProcess;
            Field[1].CommandLine = "python3 -m stompy.grid.gmsh_scale_helper %s";
            Field[2] = MathEval;
            // helps avoid subdividing boundaries
            Field[2].F = "F1*1.5";

            Background Field = 2;
            Mesh.CharacteristicLengthExtendFromBoundary = 0;
            Mesh.CharacteristicLengthFromPoints = 0;
            Mesh.CharacteristicLengthFromCurvature = 0;
            """%(self.scale_file)) )

        # When this is invoked inside QGIS, have to be careful
        # not to send output to stdout.  QGIS is frozen while this
        # runs, and stdout will (I think) clog up and lead to a
        # deadlock.
        sub_args={}
        if self.output=='capture':
            sub_args['stdout']=subprocess.PIPE
            sub_args['stderr']=subprocess.PIPE

        self.process=subprocess.run([self.gmsh,self.geo_file,'-2'],**sub_args)
        self.grid=unstructured_grid.UnstructuredGrid.read_gmsh(self.msh_file)
        self.grid.orient_cells()

        for n in self.g_in.valid_node_iter():
            x_in=self.g_in.nodes['x'][n]
            n_new=self.grid.select_nodes_nearest(x_in, max_dist=self.tol)
            if n_new is not None:
                self.grid.nodes['x'][n_new] = x_in

def triangulate_hole(grid,seed_point=None,nodes=None,max_nodes=-1,hole_rigidity='cells',
                     splice=True,return_value='grid',dry_run=False,apollo_rate=1.1,
                     density=None,method='front',method_kwargs={}):
    """
    Specify one of
      seed_point: find node string surrounding this point
      nodes: specify CCW ordered nodes making up the hole.

    method:
     'front': use front.py, which produces high quality grids very 
        slowly.
     'rebay': use rebay.py, much faster, but without any guarantees on cell quality,
        and without support for sliding boundaries (hole_rigidity must be 'all')
     'gmsh': invoke gmsh as a subprocess.

    hole_rigidity: 
       'cells' nodes and edges which are part of a cell are considered rigid.
       'all': all nodes and edges of the hole are considered rigid.
       'all-nodes': edges can be subdivided, but nodes cannot be moved.
    This affects both the calculation of local scale and the state of nodes
     and edges during triangulation.

    splice: if true, the new grid is spliced into grid. if false, just returns
     the grid covering the hole.

    density: By default an Apollonius scale field is generated from the rigid
     edges of the hole and apollo_rate. Use this argument to provide a non-default
     density field.

    return_value: grid: return the resulting grid
    front: advancing front instance
    dry_run: if True, get everything set up but don't triangulate
    """
    if method=='rebay' and hole_rigidity!='all':
        raise Exception("Currently rebay can only handle fully rigid hole (hole_rigidity='all')")
    
    # manually tell it where the region to be filled is.
    # max_nodes defaults to -1 which implies Nnodes.
    if nodes is None:
        nodes=grid.enclosing_nodestring(seed_point,max_nodes)

    if len(nodes)<3:
        raise Exception('Triangulate hole: failed to find the hole')

    xy_shore=grid.nodes['x'][nodes]

    # Construct a scale based on existing spacing
    # But only do this for edges that are part of one of the original grids
    grid.edge_to_cells() # update edges['cells']
    sample_xy=[]
    sample_scale=[]
    ec=grid.edges_center()
    el=grid.edges_length()

    src_hint_edges=[]

    # Scan the node string to both (a) find hint edges that
    # can be removed before merging, and (b) select length scales
    # for the density field.
    for na,nb in utils.circular_pairs(nodes):
        j=grid.nodes_to_edge([na,nb])
        assert j is not None
        has_no_cells=np.all( grid.edges['cells'][j] < 0)
        if has_no_cells:
            # if hole_rigidity is 'all', the edge will exactly exist
            # in both the new and original grids. Not a problem to delete
            # it, though might lose some edge-data.
            src_hint_edges.append(j)

        if hole_rigidity=='cells' and has_no_cells:
            # This edge doesn't impart any scale constraints
            continue
        # elif hole_rigidity in ['all','all-nodes']: pass
        sample_xy.append(ec[j])
        sample_scale.append(el[j])

    if density is not None:
        scale=density
    else:
        assert len(sample_xy)
        sample_xy=np.array(sample_xy)
        sample_scale=np.array(sample_scale)
        scale=field.PyApolloniusField(X=sample_xy,F=sample_scale,r=apollo_rate)

    # For hole_rigidity=='all', there are no hints, and this stays
    # empty.  Only for method=='front' or 'gmsh' can there be other values of
    # hole_rigidity.
    src_hints=[]

    # Prepare that shoreline for grid generation.
    if method=='front':
        grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=6)

        AT=front.AdvancingTriangles(grid=grid_to_pave,**method_kwargs)

        AT.add_curve(xy_shore)
        
        # This should be safe about not resampling existing edges
        # HERE: that's a problem. it defeats non-local strategy.
        AT.scale=scale # field.ConstantField(50000)

        AT.initialize_boundaries(upsample=False)

        AT.grid.nodes['fixed'][:]=AT.RIGID
        AT.grid.edges['fixed'][:]=AT.RIGID

        # Old code compared nodes to original grids to figure out RIGID
        # more general, if it works, to see if a node participates in any cells.
        # At the same time, record original nodes which end up HINT, so they can
        # be removed later on.
        node_to_src_node={}
        for n in AT.grid.valid_node_iter():
            n_src=grid.select_nodes_nearest(AT.grid.nodes['x'][n],max_dist=0.0)
            assert n_src is not None,"How are we not finding the original src node?"
            node_to_src_node[n]=n_src
            
            if len(grid.node_to_cells(n_src))==0:
                if hole_rigidity=='cells':
                    # It should be a HINT
                    AT.grid.nodes['fixed'][n]=AT.HINT
                    src_hints.append(n_src)
                # This misses edges that have no cell, but connect nodes that
                # do have cells
                # if hole_rigidity in ['cells','all-nodes']:
                #     # And any edges it participates in should not be RIGID either.
                #     for j in AT.grid.node_to_edges(n):
                #         AT.grid.edges['fixed'][j]=AT.UNSET
        # Not entirely clear on why I moved away from using the original grid
        # more directly.  This below may be a regression.
        if hole_rigidity in ['cells','all-nodes']:
            for j in AT.grid.valid_edge_iter():
                n1,n2=AT.grid.edges['nodes'][j]
                n1s=node_to_src_node[n1]
                n2s=node_to_src_node[n2]
                jsrc=grid.nodes_to_edge(n1s,n2s)
                assert jsrc is not None
                cells=grid.edge_to_cells(jsrc)
                if (cells[0]<0) and (cells[1]<0):
                    AT.grid.edges['fixed'][j]=AT.UNSET

        # resample edges that are not fixed
        AT.resample_cycles()
        
        if dry_run:
            if return_value=='grid':
                return AT.grid
            else:
                return AT

        if AT.loop():
            AT.grid.renumber()
        else:
            print("Grid generation failed")
            return AT # for debugging -- need to keep a handle on this to see what's up.
    elif method=='rebay':
        grid_to_pave=unstructured_grid.UnstructuredGrid(max_sides=len(xy_shore))
        pave_nodes=[grid_to_pave.add_node(x=xy) for xy in xy_shore]
        c=grid_to_pave.add_cell_and_edges(pave_nodes)
        rad=rebay.RebayAdvancingDelaunay(grid=grid_to_pave,scale=scale,
                                         heap_sign=1,**method_kwargs)
        if dry_run:
            if return_value=='grid':
                return grid_to_pave
            else:
                return rad
        rad.execute()
        # make it look kind of like the advancing triangles output
        AT=rad
        AT.grid=rad.extract_result()
    elif method=='gmsh':
        # make it look kind of like the advancing triangles output
        g_in=unstructured_grid.UnstructuredGrid()
        g_in.add_linestring(xy_shore,closed=True)

        # refactor!  a little tricky, as the three methods all have slightly
        # different capacities to deal with hint edges/nodes in the input.
        for n in g_in.valid_node_iter():
            n_src=grid.select_nodes_nearest(g_in.nodes['x'][n],max_dist=0.0)
            assert n_src is not None,"How are we not finding the original src node?"

            if len(grid.node_to_cells(n_src))==0:
                if hole_rigidity=='cells':
                    src_hints.append(n_src)
        
        AT=Gmsher(g_in=g_in,scale=scale)
        AT.gmsh=method_kwargs.pop('gmsh','gmsh')
        AT.output=method_kwargs.pop('output','capture')
        AT.execute()
        
    else:
        raise Exception("Bad method '%s'"%method)
        
    if not splice:
        if return_value=='grid':
            return AT.grid
        else:
            return AT
    else:
        # Scan src_hints once to find edges that need to be replaced after the merge.
        # These are edges that aren't involved in cells, but also not part of the
        # boundary linestring.
        edges_to_replace=[] # node in AT.grid and node in grid that should get an edge after merge
        for n in src_hints:
            n_nbrs=grid.node_to_nodes(n)
            if len(n_nbrs)<=2: continue # can't have any extra edges

            n_idx=nodes.index(n)
            
            # Not quite good enough to just check neighbors against
            # n_nbrs, since the fixed nodes will be neighbors of n
            # but not in src_hints.
            for nbr in n_nbrs:
                # This is an older check:
                #if nbr in src_hints: continue
                #if len(grid.node_to_cells(nbr))>0: continue
                if nbr not in nodes:
                    n_new=AT.grid.select_nodes_nearest( grid.nodes['x'][n] )
                    # so there is an edge in the original grid n--nbr and I want
                    # that to become n_new--nbr in the merge. 
                    # tempting to pass these as merge_nodes to add_grid, but then I
                    # have to clean up the edges, and I would get the old location of
                    # the node.
                    # Have to annotate the nodes with which grid they refer to, since
                    # new nodes are going to get remapped
                    edges_to_replace.append( [('new',n_new),('old',nbr)] )
                else:
                    nbr_idx=nodes.index(nbr)
                    N=len(nodes)
                    if nbr_idx in [ (n_idx+1)%N, (n_idx-1)%N]:
                        # Just a hint edge 
                        continue
                    else:
                        # A special case, untested at this point.
                        logger.warning("Untested territory -- a non-hint edge joining two hint nodes")
                        n_new=AT.grid.select_nodes_nearest( grid.nodes['x'][n] )
                        nbr_new=AT.grid.select_nodes_nearest( grid.nodes['x'][nbr] )
                        edges_to_replace.append( [('new',n_new), ('new',nbr_new)] )

        if hole_rigidity!='all':
            # if hole_rigidity were 'all', then prefer to keep these
            # edges since they may have some data on them.
            for j in src_hint_edges:
                grid.delete_edge(j)
        # Scan nodes again, this time deleting
        for n in src_hints:
            grid.delete_node_cascade(n)

        # Merge, then add those edges back in
        node_map,edge_map,cell_map = grid.add_grid(AT.grid)
        for (new_src,n_new),(nbr_src,nbr) in edges_to_replace:
            if new_src=='new':
                n_new=node_map[n_new]
            if nbr_src=='new':
                nbr=node_map[nbr]
                
            grid.add_edge( nodes=[n_new,nbr] )

        # Constrained nodes match exactly and get merged here
        # Surprisingly, this works!  Usually.
        grid.merge_duplicate_nodes()

        grid.renumber(reorient_edges=False)

        if return_value=='grid':
            return grid
        else:
            return AT
