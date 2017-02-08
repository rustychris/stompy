from __future__ import print_function
from stompy.grid import unstructured_grid
import numpy as np

# TODO: migrate to xarray
from stompy.io import qnc


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
                 cells_from_edges='auto',max_sides=6):
        if nc is None:
            assert fn
            nc=qnc.QDataset(fn)

        if isinstance(nc,str):
            nc=qnc.QDataset(nc)

        # probably this ought to attempt to find a mesh variable
        # with attributes that tell the correct names, and lacking
        # that go with these as defaults
        # seems we always get nodes and edges
        kwargs=dict(points=np.array([nc.NetNode_x[:],nc.NetNode_y[:]]).T,
                    edges=nc.NetLink[:,:]-1)

        # some nc files also have elements...
        if 'NetElemNode' in nc.variables:
            cells=nc.NetElemNode[:,:] 
            cells[ cells<0 ] = 0 
            cells-=1
            kwargs['cells']=cells
            if cells_from_edges=='auto':
                cells_from_edges=False

        if 'NetNode_z' in nc.variables: # have depth at nodes
            kwargs['extra_node_fields']=[ ('depth','f4') ]

        if cells_from_edges: # True or 'auto'
            self.max_sides=max_sides

        # account for 1-based => 0-based indices
        super(DFMGrid,self).__init__(**kwargs)

        if cells_from_edges:
            print("Making cells from edges")
            self.make_cells_from_edges()

        if 'NetNode_z' in nc.variables: # have depth at nodes
            self.nodes['depth']=nc.NetNode_z[:]
