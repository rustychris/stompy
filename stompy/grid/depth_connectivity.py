"""
depth_connectivity.py
---

Assign bathymetry to a grid based on cell-to-cell connectivity
derived from a high-resolution DEM.

Adapted from Holleman and Stacey, JPO, 2014.

Primary entry point:
edge_depths=edge_connection_depth(g,dem,edge_mask=None,centers='lowest')

see end of file

"""

# Copied from .../research/spatialdata/us/ca/lidar/direct_biased/direct_biased.py
from __future__ import print_function

import numpy as np
import pdb
from scipy.ndimage import label

from .. import utils

if 1:
    debug=0
else:
    debug=1
    import matplotlib.pyplot as plt

try:
    # gone away as of mpl 1.3
    from matplotlib.nxutils import points_inside_poly
except ImportError:
    from matplotlib import path
    def points_inside_poly(points,ijs):
        # closed path likes the first/last nodes to coincide
        ijs=np.concatenate( (ijs,ijs[:1]) )
        p=path.Path(ijs,closed=True)
        return p.contains_points(points)


def greedy_edgemin_to_node(g,orig_node_depth,edge_min_depth):
    """
    A simple approach to moving edge depths to nodes, when the
    hydro model (i.e. DFM) will use the minimum of the nodes to
    set the edge.
    It sounds roundabout because it is, but there is not a
    supported way to assign edge depth directly.

    For each edge, want to enforce a minimum depth in two sense:
    1. one of its nodes is at the minimum depth
    2. neither of the nodes are below the minimum depth
    and..
    3. the average of the two nodes is close to the average DEM depth
       of the edge
    Not yet sure of how to get all of those.  This method focuses on
    the first point, but in some situations that is still problematic.
    The 3rd point is not attempted at all, but in DFM would only be
    relevant for nonlinear edge depths which are possibly not even supported
    for 3D runs.
    """

    conn_depth=np.nan*orig_node_depth

    # N.B. nans sort to the end
    edge_min_ordering=np.argsort(edge_min_depth)

    # The greedy aspect is that we start with edges at the
    # lowest target depth, ensuring their elevations before
    # setting higher edges
    for j in edge_min_ordering:
        if np.isnan(edge_min_depth[j]):
            break # done with all of the target depths

        nodes=g.edges['nodes'][j]

        # is this edge is already low enough, based on minimum of
        # node elevations set so far?
        if ( np.any( np.isfinite(conn_depth[nodes]) ) and 
             (np.nanmin(conn_depth[nodes])<=edge_min_depth[j] ) ):
            continue # yes, move on.

        orig_z=orig_node_depth[nodes]

        # original code skipped edges where either of the nodes were nan
        # in the original grid -- seems unnecessary

        # instead, now choose the node to update based orig_node_depth is possible,
        # considering nan to be above finite.  failing that, choose nodes[0] arbitrarily.
        if np.isnan(orig_z[0]):
            if np.isnan(orig_z[1]):
                target_node=nodes[0] # arbitrary
            else:
                target_node=nodes[1]
        elif np.isnan(orig_z[1]):
            target_node=nodes[0]
        elif orig_z[0]<orig_z[1]:
            target_node=nodes[0]
        else:
            target_node=nodes[1]
            
        conn_depth[target_node]=edge_min_depth[j]

    missing=np.isnan(conn_depth)
    conn_depth[missing]=orig_node_depth[missing]
    return conn_depth


def greedy_edge_mean_to_node(g,orig_node_depth=None,edge_depth=None,n_iter=100):
    """
    Return node depths such that the mean of the node depths on each
    edge approximate the provided edge_mean_depth.
    The approach is iterative, starting with the largest errors, visiting
    each edge a max of once.

    Still in development, has not been tested.
    """
    from scipy.optimize import fmin

    if edge_depth is None:
        if 'depth' in g.edges.dtype.names:
            edge_depth=g.edges['depth']

    assert edge_depth is not None

    if orig_node_depth is None:
        if 'depth' in g.nodes.dtype.names:
            orig_node_depth=g.nodes['depth']
        else:
            # Rough starting guess:
            orig_node_depth=np.zeros( g.Nnodes(), 'f8')
            for n in range(g.Nnodes()):
                orig_node_depth[n] = edge_depth[g.node_to_edges(n)].mean()

    # The one we'll be updating:
    conn_depth=orig_node_depth.copy()

    node_mean=conn_depth[g.edges['nodes']].mean(axis=1)
    errors=node_mean - edge_depth
    errors[ np.isnan(errors) ] = 0.0
    
    potential=np.ones(g.Nedges())

    for loop in range(n_iter):
        verbose= (loop%100==0)

        # Find an offender
        j_bad=np.argmax(potential*errors)
        if potential[j_bad]==0:
            print("DONE")
            break

        potential[j_bad]=0 # only visit each edge once.

        # Get the neighborhood of nodes:
        # nodes=
        jj_nbrs=np.concatenate( [ g.node_to_edges(n)
                                  for n in g.edges['nodes'][j_bad] ] )
        jj_nbrs=np.unique(jj_nbrs)
        jj_nbrs = jj_nbrs[ np.isfinite(edge_depth[jj_nbrs]) ]

        n_bad=g.edges['nodes'][j_bad]

        def cost(ds):
            # Cost function over the two depths of the ends of j_bad:
            conn_depth[n_bad]=ds
            new_errors=conn_depth[g.edges['nodes'][jj_nbrs]].mean(axis=1) - edge_depth[jj_nbrs]
            # weight high edges 10x more than low edges:
            cost=new_errors.clip(0,np.inf).sum() - 0.5 * new_errors.clip(-np.inf,0).sum()
            return cost
        ds0=conn_depth[n_bad]
        cost0=cost(ds0)

        ds=fmin(cost,ds0,disp=False)
        costn=cost(ds)
        conn_depth[n_bad]=ds

        if verbose:
            print("Loop %d: %d/%d edges  starting error: j=%d => %.4f"%(loop,potential.sum(),len(potential),
                                                                        j_bad,errors[j_bad]))

        node_mean=conn_depth[g.edges['nodes']].mean(axis=1)
        errors=node_mean - edge_depth
        errors[ np.isnan(errors) ] = 0.0

        if verbose:
            print("    ending error: j=%d => %.4f"%(j_bad,errors[j_bad]))
    return conn_depth

    
def points_to_mask(hull_ijs,nx,ny):
    # This seems inefficient, but actually timed out at 0.3ms
    # very reasonable.
    # Create vertex coordinates for each grid cell...
    # (<0,0> is at the top left of the grid in this system)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T
    mask = points_inside_poly(points, hull_ijs)
    return mask.reshape((ny,nx))


def min_connection_elevation(ijs,min_depth,max_depth,F):
    while max_depth - min_depth > 0.01: # cm accuracy
        # use a binary search, and the numpy labelling routines
        mid_depth = 0.5*(max_depth+min_depth)

        Fthresh = F <= mid_depth
        labels,nlabels = label(Fthresh)

        l1,l2 = labels[ ijs[:,1],ijs[:,0]]
        if l1 != l2:
            # too shallow -
            min_depth = mid_depth
        else:
            # deep enough
            max_depth = mid_depth

    return 0.5*(min_depth+max_depth)

def min_graph_elevation_for_edge(g,dem,j,starts='lowest'):
    """
    g: unstructured_grid
    j: edge index
    dem: a Field subclass which supports extract_tile().

    starts:
     'circumcenter' connections are between voronoi centers
     'centroid' connections are between cell centroids
     'lowest'  connections are between lowest point in cell

    returns: the minimum edge elevation at which the cells adjacent
      to j are hydraulically connected.  nan if j is not adjacent to 
      two cells (i.e. boundary).
    """
    # get the bounding box for the neighboring cells.
    nc = g.edges['cells'][j]
    if nc[0]<0 or nc[1]<0:
        return np.nan

    nc0_nodes=list(g.cell_to_nodes(nc[0]))
    nc1_nodes=list(g.cell_to_nodes(nc[1]))

    all_nodes=( nc0_nodes + nc1_nodes )

    pnts = g.nodes['x'][all_nodes]

    # asserts/assumes that the extents are multiples of dx,dy.
    dx=dem.dx ; dy=dem.dy
    dxy=np.array([dx,dy])
    #assert dem.extents[0] % dem.dx == 0
    #assert dem.extents[2] % dem.dy == 0

    # protects from roundoff cases
    pad=1
    ll = np.floor(pnts.min(axis=0) / dxy - pad) * dxy
    ur = np.ceil(pnts.max(axis=0) / dxy + pad) * dxy
    xxyy = [ll[0],ur[0],ll[1],ur[1]]

    # for a raster field, crop is much much faster than extract_tile
    # tile = dem.extract_tile(xxyy)
    tile=dem.crop(xxyy)

    # Some of the above is for precise usage of SimpleGrid.
    # but in some cases we're dealing with a MultiRasterField, and the
    # local resolution is coarser:
    dx=tile.dx ; dy=tile.dy
    
    if tile is None:
        return np.nan

    # if the tile is not fully populated, also give up
    if ( (tile.extents[0]>xxyy[0]) or
         (tile.extents[1]<xxyy[1]) or
         (tile.extents[2]>xxyy[2]) or
         (tile.extents[3]<xxyy[3]) ):
        print("Tile clipped by edge of DEM")
        return np.nan

    if debug:
        fig=plt.figure(101)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        tile.plot(interpolation='nearest',ax=ax)
        ax.set_title('Extracted tile')

    # old code manually constructed the convex hull, but with quads
    # and so forth, it gets complicated - punt to shapely for the moment.
    # hull_poly=g.cell_polygon(nc[0]).union(g.cell_polygon(nc[1]))
    # hull_points=np.array(hull_poly.exterior)

    nA,nB=g.edges['nodes'][j]
    # nA_idx0=nc0_nodes.index(nA)
    nB_idx0=nc0_nodes.index(nB)
    nA_idx1=nc1_nodes.index(nA)
    # rearrange so that nc0_nodes starts with B, ends with A
    nc0_nodes=nc0_nodes[nB_idx0:] + nc0_nodes[:nB_idx0]
    assert nc0_nodes[-1] == nA
    # rearrange so that nc1_nodes starts with A, ends with B
    nc1_nodes=nc1_nodes[nA_idx1:] + nc1_nodes[:nA_idx1]
    assert nc1_nodes[-1] == nB

    # A,B appear consecutively in nc0, reversed in nc1
    hull_nodes=nc0_nodes + nc1_nodes[1:-1]
    hull_points=g.nodes['x'][hull_nodes]

    tile_origin = np.array( [ tile.extents[0], tile.extents[2]] )
    tile_dxy = np.array( [tile.dx,tile.dy] )

    def xy_to_ij(xy):
        return (( xy - tile_origin ) / tile_dxy).astype(np.int32)

    hull_ijs = xy_to_ij(hull_points)

    # blank out the dem outside the two cells
    ny, nx = tile.F.shape
    valid = points_to_mask(hull_ijs,nx,ny)
    F = tile.F.copy()
    F[~valid] = 1e6

    if starts in ['circumcenter','centroid']:
        # map the two cell centers ij indices into the tile:
        if starts=='circumcenter':
            centers = g.cells_center(nc)[nc]
        else:
            centers = g.cells_centroid(nc)
        lcenters = centers - tile_origin
        
        # note that this is i -> x coordinate, j -> y coordinate
        ijs = (lcenters / tile_dxy).astype(np.int32)
    elif starts=='lowest':
        nc0_ijs=xy_to_ij( g.nodes['x'][nc0_nodes] )
        nc1_ijs=xy_to_ij( g.nodes['x'][nc1_nodes] )
        valid0 = points_to_mask(nc0_ijs,nx,ny)
        valid1 = points_to_mask(nc1_ijs,nx,ny)
        j0,i0 = np.nonzero(valid0)
        j1,i1 = np.nonzero(valid1)

        linear0_min = F[valid0].argmin()
        linear1_min = F[valid1].argmin()

        ijs = np.array([ [i0[linear0_min],j0[linear0_min]],
                         [i1[linear1_min],j1[linear1_min]] ] )
    else:
        raise ValueError("'%s' not understood"%starts)

    if debug:
        fig=plt.figure(102)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        ax.imshow(F,origin='lower',interpolation='nearest',vmin=-2,vmax=4)
        ax.plot( ijs[:,0],ijs[:,1],'go')
        ax.set_title('Centers')

    if not valid[ijs[0,1],ijs[0,0]] or not valid[ijs[1,1],ijs[1,0]]:
        print("Cell circumcenter(s) not in cell!")
        return np.nan

    # will probably end up grabbing the real cell depths here, rather
    # than estimating by a point measurement on the DEM.
    lcenter_depths = F[ijs[:,1],ijs[:,0]]

    # clearly path cannot have max. elevation lower than either of the
    # endpoints:
    min_depth = lcenter_depths.max() 
    max_depth = F[valid].max()

    # and this part takes 3.5ms - tolerable.
    return min_connection_elevation(ijs,min_depth,max_depth,F)


def edge_connection_depth(g,dem,edge_mask=None,centers='circumcenter'):
    """
    Return an array g.Nedges() where the selected edges have
    a depth value corresponding to the minimum elevation at which
    adjacent cells are hydraulically connected, evaluated on the
    dem.  
    g: instance of UnstructuredGrid
    dem: field.SimpleGrid instance, usually GdalGrid
    edge_mask: bitmask for which edges to calculate, defaults to bounds of dem.

    centers controls the reference point for each cell.
      'circumcenter': use cell circumcenter
      'centroid': use cell centroid
      'lowest': use lowest point within the cell.
    """
    if edge_mask is None:
        # use to default to all edges
        # edge_mask=np.ones(g.Nedges(),'b1')
        # this makes more sense, though
        edge_mask=g.edge_clip_mask(dem.bounds())

    sel_edges=np.nonzero(edge_mask)[0]
    count=np.sum(edge_mask)

    edge_elevations=np.nan*np.ones(g.Nedges())
    g.edge_to_cells()

    for ji,j in enumerate(sel_edges): 
        if ji%100==0:
            print("%d/%d"%(ji,count))

        elev = min_graph_elevation_for_edge(g,dem,j,starts=centers)

        edge_elevations[j] = elev

    return edge_elevations

    
def poly_mean_elevation(dem,pnts):
    # asserts/assumes that the extents are multiples of dx,dy.
    dx=dem.dx ; dy=dem.dy
    dxy=np.array([dx,dy])
    
    # protects from roundoff cases
    pad=1
    ll = np.floor(pnts.min(axis=0) / dxy - pad) * dxy
    ur = np.ceil(pnts.max(axis=0) / dxy + pad) * dxy
    xxyy = [ll[0],ur[0],ll[1],ur[1]]

    # crop first - much faster
    tile=dem.crop(xxyy)

    # Some of the above is for precise usage of SimpleGrid.
    # but in some cases we're dealing with a MultiRasterField, and the
    # local resolution is coarser:
    dx=tile.dx ; dy=tile.dy
    
    if tile is None:
        return np.nan

    # if the tile is not fully populated, also give up
    if ( (tile.extents[0]>xxyy[0]) or
         (tile.extents[1]<xxyy[1]) or
         (tile.extents[2]>xxyy[2]) or
         (tile.extents[3]<xxyy[3]) ):
        print("Tile clipped by edge of DEM")
        return np.nan

    tile_origin = np.array( [ tile.extents[0], tile.extents[2]] )
    tile_dxy = np.array( [tile.dx,tile.dy] )

    def xy_to_ij(xy):
        return (( xy - tile_origin ) / tile_dxy).astype(np.int32)

    hull_ijs = xy_to_ij(pnts)

    # blank out the dem outside the two cells
    ny, nx = tile.F.shape
    valid = points_to_mask(hull_ijs,nx,ny)
    return tile.F[valid].mean()
    

def cell_mean_depth(g,dem):
    """
    Calculate "true" mean depth for each cell, at the resolution of
    the DEM.  This does not split pixels, though.
    """
    cell_z_bed=np.nan*np.ones(g.Ncells())
    
    for c in utils.progress(range(g.Ncells())):
        cell_z_bed[c]=poly_mean_elevation(dem, g.nodes['x'][ g.cell_to_nodes(c) ])
        
    return cell_z_bed
