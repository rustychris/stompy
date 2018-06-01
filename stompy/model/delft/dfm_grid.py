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
    nc=qnc.empty(fn=nc_fn,overwrite=overwrite,format='NETCDF3_CLASSIC')

    # schema copied from r17b_net.nc as written by rgfgrid
    nc.createDimension('nNetNode',ug.Nnodes())
    nc.createDimension('nNetLink',ug.Nedges())
    nc.createDimension('nNetLinkPts',2)

    node_x=nc.createVariable('NetNode_x','f8',('nNetNode'))
    node_x[:] = ug.nodes['x'][:,0]
    node_x.units='m'
    node_x.standard_name = "projection_x_coordinate"
    node_x.long_name="x-coordinate of net nodes"
    node_x.grid_mapping = "projected_coordinate_system" 

    node_y=nc.createVariable('NetNode_y','f8',('nNetNode'))
    node_y[:] = ug.nodes['x'][:,1]
    node_y.units = "m" 
    node_y.standard_name = "projection_y_coordinate" 
    node_y.long_name = "y-coordinate of net nodes"
    node_y.grid_mapping = "projected_coordinate_system" 

    if 1:
        # apparently this doesn't have to be correct -
        proj=nc.createVariable('projected_coordinate_system',
                          'i4',())
        proj.setncattr('name',"Unknown projected")
        proj.epsg = 28992 
        proj.grid_mapping_name = "Unknown projected" 
        proj.longitude_of_prime_meridian = 0. 
        proj.semi_major_axis = 6378137. 
        proj.semi_minor_axis = 6356752.314245 
        proj.inverse_flattening = 298.257223563 
        proj.proj4_params = "" 
        proj.EPSG_code = "EPGS:28992" 
        proj.projection_name = "" 
        proj.wkt = "" 
        proj.comment = "" 
        proj.value = "value is equal to EPSG code" 
        proj[...]=28992

    if ('lon' in ug.nodes.dtype.names) and ('lat' in ug.nodes.dtype.names):
        print("Will include longitude & latitude")
        node_lon=nc.createVariable('NetNode_lon','f8',('nNetNode'))
        node_lon[:]=ug.nodes['lon'][:]
        node_lon.units = "degrees_east" 
        node_lon.standard_name = "longitude" 
        node_lon.long_name = "longitude" 
        node_lon.grid_mapping = "wgs84" 

        node_lat=nc.createVariable('NetNode_lat','f8',('nNetNode'))
        node_lat.units = "degrees_north" 
        node_lat.standard_name = "latitude" 
        node_lat.long_name = "latitude" 
        node_lat.grid_mapping = "wgs84" 

    if 1:
        wgs=nc.createVariable('wgs84','i4',())
        wgs.setncattr('name',"WGS84")
        wgs.epsg = 4326 
        wgs.grid_mapping_name = "latitude_longitude" 
        wgs.longitude_of_prime_meridian = 0. 
        wgs.semi_major_axis = 6378137. 
        wgs.semi_minor_axis = 6356752.314245 
        wgs.inverse_flattening = 298.257223563 
        wgs.proj4_params = "" 
        wgs.EPSG_code = "EPGS:4326" 
        wgs.projection_name = "" 
        wgs.wkt = "" 
        wgs.comment = "" 
        wgs.value = "value is equal to EPSG code" 

    if 'depth' in ug.nodes.dtype.names:
        node_z = nc.createVariable('NetNode_z','f8',('nNetNode'))
        node_z[:] = ug.nodes['depth'][:]
        node_z.units = "m" 
        node_z.positive = "up" 
        node_z.standard_name = "sea_floor_depth" 
        node_z.long_name = "Bottom level at net nodes (flow element\'s corners)" 
        node_z.coordinates = "NetNode_x NetNode_y" 
        node_z.grid_mapping = "projected_coordinate_system" 

    links = nc.createVariable('NetLink','i4',('nNetLink','nNetLinkPts'))
    links[:,:]=ug.edges['nodes'] + 1 # to 1-based!
    links.standard_name = "netlink" 
    links.long_name = "link between two netnodes" 

    link_types=nc.createVariable('NetLinkType','i4',('nNetLink'))
    link_types[:] = 2 # always seems to be 2 for these grids
    link_types.long_name = "type of netlink" 
    link_types.valid_range = [0, 2] 
    link_types.flag_values = [0, 1, 2]
    link_types.flag_meanings = "closed_link_between_2D_nodes link_between_1D_nodes link_between_2D_nodes" 

    # global attributes - probably ought to allow passing in values for these...
    nc.institution = "SFEI et al" 
    nc.references = "http://github.com/rustychris/stompy" 
    nc.history = "stompy unstructured_grid" 

    nc.source = "Deltares, D-Flow FM Version 1.1.135.38878MS, Feb 26 2015, 17:00:33, model" 
    nc.Conventions = "CF-1.5:Deltares-0.1" 

    if 1: 
        # add the complines to encode islands
        lines=ug.boundary_linestrings()
        nc.createDimension('nNetCompLines',len(lines))

        # And add the cells:
        nc.createDimension('nNetElemMaxNode',ug.max_sides)
        nc.createDimension('nNetElem',ug.Ncells())
        missing=-2147483647 # DFM's preferred missing value

        cell_var=nc.createVariable('NetElemNode','i4',('nNetElem','nNetElemMaxNode'),
                                   fill_value=missing)
        # what to do about missing nodes?
        cell_nodes=ug.cells['nodes'] + 1 #make it 1-based
        cell_nodes[ cell_nodes<1 ] = missing
        cell_var[:,:] =cell_nodes

        # Write the complines
        for i,line in enumerate(lines):
            dimname='nNetCompLineNode_%d'%(i+1)
            nc.createDimension(dimname,len(line))

            compline_x=nc.createVariable('NetCompLine_x_%d'%i,'f8',(dimname,))
            compline_y=nc.createVariable('NetCompLine_y_%d'%i,'f8',(dimname,))
            
            compline_x[:] = line[:,0]
            compline_y[:] = line[:,1]

    nc.close()

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
        if nc is None:
            assert fn
            #nc=qnc.QDataset(fn)
            # Trying out xarray instead
            nc=xr.open_dataset(fn)

        if isinstance(nc,str):
            #nc=qnc.QDataset(nc)
            nc=xr.open_dataset(nc)

        #if isinstance(nc,xr.Dataset):
        #    raise Exception("Pass the filename or a qnc.QDataset.  Not ready for xarray")

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
    Mimic FlowFM boundary edge selection from polyline to edges.
    Currently does not get into any of the interpolation, just
    identifies boundary edges which would be selected as part of the
    boundary group.

    g: UnstructuredGrid instance
    linestring: [N,2] polyline data
    rrtol: controls search distance away from boundary. Defaults to
    roughly 3 cell length scales out from the boundary.
    """
    linestring=np.asanyarray(linestring)
    
    g.edge_to_cells()
    boundary_edges=np.nonzero( np.any(g.edges['cells']<0,axis=1) )[0]
    adj_cells=g.edges['cells'][boundary_edges].max(axis=1)

    # some of this assumes that the grid is orthogonal, so we're not worrying
    # about overridden cell centers
    adj_centers=g.cells_center()[adj_cells]
    edge_centers=g.edges_center()[boundary_edges]
    cell_to_edge=edge_centers-adj_centers
    cell_to_edge_dist=utils.dist(cell_to_edge)
    outward=cell_to_edge / cell_to_edge_dist[:,None]
    dis=np.maximum( 0.5*np.sqrt(g.cells_area()[adj_cells]),
                    cell_to_edge_dist )
    probes=edge_centers+(2*rrtol*dis)[:,None]*outward
    segs=np.array([adj_centers,probes]).transpose(1,0,2)
    if 0: # plotting for verification
        lcoll=collections.LineCollection(segs)
        ax.add_collection(lcoll)

    linestring_geom= geometry.LineString(linestring)

    probe_geoms=[geometry.LineString(seg) for seg in segs]

    hits=[idx
          for idx,probe_geom in enumerate(probe_geoms)
          if linestring_geom.intersects(probe_geom)]
    edge_hits=boundary_edges[hits]
    return edge_hits


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
        
    
