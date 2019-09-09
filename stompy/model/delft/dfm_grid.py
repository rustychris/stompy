"""
Read and write DFM formatted netcdf grids (old-style, not quite ugrid)

Also includes methods for modifying depths on a grid to allow inflows
in poorly resolved areas which would otherwise be dry.
"""

from __future__ import print_function
from stompy.grid import unstructured_grid
import numpy as np
import logging
log=logging.getLogger(__name__)

from shapely import geometry
import xarray as xr


# TODO: migrate to xarray
from ...io import qnc
from ... import utils

# for now, only supports 2D/3D grid - no mix with 1D

# First try - RGFGRID says it can't read it.
# okay - make sure we're outputting the same netcdf 
# version...

# r17b_net.nc: first line just says netcdf r17b_net {s
# file says:
# r17b_net.nc: NetCDF Data Format data
# for default qnc output, file says its HDF5.

# okay - now it gets the nodes, but doesn't have any 
# edges.

# even reading/writing the existing DFM grid does not 
# get the edges.

# does including the projection definition help? nope.

def write_dfm(ug,nc_fn,overwrite=False):
    ug.write_dfm(nc_fn,overwrite=overwrite)

class DFMGrid(unstructured_grid.UnstructuredGrid):
    def __init__(self,nc=None,fn=None,
                 cells_from_edges='auto',max_sides=6,cleanup=False):
        """
        nc: An xarray dataset or path to netcdf file holding the grid
        fn: path to netcdf file holding the grid (redundant with nc)
        cells_from_edges: 'auto' create cells based on edges if cells do not exist in the dataset
          specify True or False to force or disable this.
        max_sides: maximum number of sides per cell, used both for initializing datastructures, and
          for determining cells from edge connectivity.
        cleanup: for grids created from multiple subdomains, there are sometime duplicate edges and nodes.
          this will remove those duplicates, though there are no guarantees of indices.
        """
        self.filename=None

        if nc is None:
            assert fn
            self.filename=fn
            nc=xr.open_dataset(fn)

        if isinstance(nc,str):
            self.filename=nc
            nc=xr.open_dataset(nc)

        # Default names for fields
        var_points_x='NetNode_x'
        var_points_y='NetNode_y'

        var_edges='NetLink'
        var_cells='NetElemNode' # often have to infer the cells

        meshes=[v for v in nc.data_vars if getattr(nc[v],'cf_role','none')=='mesh_topology']
        if meshes:
            mesh=nc[meshes[0]]
            var_points_x,var_points_y = mesh.node_coordinates.split(' ')
            var_edges=mesh.edge_node_connectivity
            try:
                var_cells=mesh.face_node_connectivity
            except AttributeError:
                var_cells='not specified'
                cells_from_edges=True

        # probably this ought to attempt to find a mesh variable
        # with attributes that tell the correct names, and lacking
        # that go with these as defaults
        # seems we always get nodes and edges
        edge_start_index=nc[var_edges].attrs.get('start_index',1)

        kwargs=dict(points=np.array([nc[var_points_x].values,
                                     nc[var_points_y].values]).T,
                    edges=nc[var_edges].values-edge_start_index)

        # some nc files also have elements...
        if var_cells in nc.variables:
            cells=nc[var_cells].values.copy()

            # missing values come back in different ways -
            # might come in masked, might also have some huge negative values,
            # and regardless it will be one-based.
            if isinstance(cells,np.ma.MaskedArray):
                cells=cells.filled(0)

            if np.issubdtype(cells.dtype,np.floating):
                bad=np.isnan(cells)
                cells=cells.astype(np.int32)
                cells[bad]=0

            # just to be safe, do this even if it came from Masked.
            cell_start_index=nc[var_cells].attrs.get('start_index',1)
            cells-=cell_start_index # force to 0-based
            cells[ cells<0 ] = -1

            kwargs['cells']=cells
            if cells_from_edges=='auto':
                cells_from_edges=False

        var_depth='NetNode_z'

        if var_depth in nc.variables: # have depth at nodes
            kwargs['extra_node_fields']=[ ('depth','f4') ]

        if cells_from_edges: # True or 'auto'
            self.max_sides=max_sides


        # Partition handling - at least the output of map_merge
        # does *not* remap indices in edges and cells
        if 'partitions_node_start' in nc.variables:
            nodes_are_contiguous = np.all( np.diff(nc.partitions_node_start.values) == nc.partitions_node_count.values[:-1] )
            assert nodes_are_contiguous, "Merged grids can only be handled when node indices are contiguous"
        else:
            nodes_are_contiguous=True

        if 'partitions_edge_start' in nc.variables:
            edges_are_contiguous = np.all( np.diff(nc.partitions_edge_start.values) == nc.partitions_edge_count.values[:-1] )
            assert edges_are_contiguous, "Merged grids can only be handled when edge indices are contiguous"
        else:
            edges_are_contiguous=True

        if 'partitions_face_start' in nc.variables:
            faces_are_contiguous = np.all( np.diff(nc.partitions_face_start.values) == nc.partitions_face_count.values[:-1] )
            assert faces_are_contiguous, "Merged grids can only be handled when face indices are contiguous"
            if cleanup:
                log.warning("Some MPI grids have duplicate cells, which cannot be cleaned, but cleanup=True")
        else:
            face_are_contiguous=True

        if 0: # This is for hints to possibly handling non-contiguous indices in the future. caveat emptor.
            node_offsets=nc.partitions_node_start.values-1

            cell_missing=kwargs['cells']<0

            if 'FlowElemDomain' in nc:
                cell_domains=nc.FlowElemDomain.values # hope that's 0-based?
            else:
                HERE

            cell_node_offsets=node_offsets[cell_domains]

            kwargs['cells']+=cell_node_offsets[:,None]

            # for part_i in range(nc.NumPartitionsInFile):
            #     edge_start=nc.partitions_edge_start.values[part_i]
            #     edge_count=nc.partitions_edge_count.values[part_i]
            #     node_start=nc.partitions_node_start.values[part_i]
            #     node_count=nc.partitions_node_count.values[part_i]
            #     cell_start=nc.partitions_face_start.values[part_i]
            #     cell_count=nc.partitions_face_count.values[part_i]
            #
            #     kwargs['edges'][edge_start-1:edge_start-1+edge_count] += node_start-1
            #     kwargs['cells'][cell_start-1:cell_start-1+cell_count] += node_start-1

            # Reset the missing nodes
            kwargs['cells'][cell_missing]=-1
            # And force valid values for over-the-top cells:
            bad=kwargs['cells']>=len(kwargs['points'])
            kwargs['cells'][bad]=0

        super(DFMGrid,self).__init__(**kwargs)

        if cells_from_edges:
            print("Making cells from edges")
            self.make_cells_from_edges()

        if var_depth in nc.variables: # have depth at nodes
            self.nodes['depth']=nc[var_depth].values.copy()

        if cleanup:
            cleanup_multidomains(self)


def cleanup_multidomains(grid):
    """
    Given an unstructured grid which was the product of DFlow-FM
    multiple domains stitched together, fix some of the extraneous
    geometries left behind.
    Grid doesn't have to have been read as a DFMGrid.
    """
    log.info("Regenerating edges")
    grid.make_edges_from_cells()
    log.info("Removing orphaned nodes")
    grid.delete_orphan_nodes()
    log.info("Removing duplicate nodes")
    grid.merge_duplicate_nodes()
    log.info("Renumbering nodes")
    grid.renumber_nodes()
    log.info("Extracting grid boundary")
    return grid

def polyline_to_boundary_edges(g,linestring,rrtol=3.0):
    """
    This has moved to grid.select_edge_by_polyline.
    Mimic FlowFM boundary edge selection from polyline to edges.
    Currently does not get into any of the interpolation, just
    identifies boundary edges which would be selected as part of the
    boundary group.

    g: UnstructuredGrid instance
    linestring: [N,2] polyline data
    rrtol: controls search distance away from boundary. Defaults to
    roughly 3 cell length scales out from the boundary.

    returns ndarray of edge indices into g.
    """
    return g.select_edges_by_polyline(linestring,rrtol=rrtol)
    # linestring=np.asanyarray(linestring)
    # 
    # g.edge_to_cells()
    # boundary_edges=np.nonzero( np.any(g.edges['cells']<0,axis=1) )[0]
    # adj_cells=g.edges['cells'][boundary_edges].max(axis=1)
    # 
    # adj_centers=g.cells_center()[adj_cells]
    # edge_centers=g.edges_center()[boundary_edges]
    # cell_to_edge=edge_centers-adj_centers
    # cell_to_edge_dist=utils.dist(cell_to_edge)
    # if 0: # older code that assumes grid is orthogonal, with centers
    #     # properly inside cells.
    #     outward=cell_to_edge / cell_to_edge_dist[:,None]
    # else:
    #     # use newer grid code which only needs edge geometry and
    #     # edge_to_cells().
    #     outward=-g.edges_normals(edges=boundary_edges,force_inward=True)
    # 
    # dis=np.maximum( 0.5*np.sqrt(g.cells_area()[adj_cells]),
    #                 cell_to_edge_dist )
    # probes=edge_centers+(2*rrtol*dis)[:,None]*outward
    # segs=np.array([adj_centers,probes]).transpose(1,0,2)
    # if 0: # plotting for verification
    #     lcoll=collections.LineCollection(segs)
    #     ax.add_collection(lcoll)
    # 
    # linestring_geom= geometry.LineString(linestring)
    # 
    # probe_geoms=[geometry.LineString(seg) for seg in segs]
    # 
    # hits=[idx
    #       for idx,probe_geom in enumerate(probe_geoms)
    #       if linestring_geom.intersects(probe_geom)]
    # edge_hits=boundary_edges[hits]
    # return edge_hits

def dredge_boundary(g,linestring,dredge_depth):
    """
    Lower bathymetry in the vicinity of external boundary, defined
    by a linestring.

    g: instance of unstructured_grid, with a node field 'depth'
    linestring: [N,2] array of coordinates
    dredge_depth: positive-up bed-level for dredged areas

    Modifies depth information in-place.
    """
    # Carve out bathymetry near sources:
    cells_to_dredge=[]

    linestring=np.asarray(linestring)
    assert linestring.ndim==2,"dredge_boundary requires [N,2] array of points"

    feat_edges=polyline_to_boundary_edges(g,linestring)
    if len(feat_edges)==0:
        raise Exception("No boundary edges matched by %s"%(str(linestring)))
    cells_to_dredge=g.edges['cells'][feat_edges].max(axis=1)

    nodes_to_dredge=np.concatenate( [g.cell_to_nodes(c)
                                     for c in cells_to_dredge] )
    nodes_to_dredge=np.unique(nodes_to_dredge)

    g.nodes['depth'][nodes_to_dredge] = np.minimum(g.nodes['depth'][nodes_to_dredge],
                                                   dredge_depth)


def dredge_discharge(g,linestring,dredge_depth):
    linestring=np.asarray(linestring)
    pnt=linestring[-1,:]
    cell=g.select_cells_nearest(pnt,inside=True)
    assert cell is not None
    nodes_to_dredge=g.cell_to_nodes(cell)

    g.nodes['depth'][nodes_to_dredge] = np.minimum(g.nodes['depth'][nodes_to_dredge],
                                                   dredge_depth)


