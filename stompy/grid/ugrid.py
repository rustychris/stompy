"""
wrapper around unstructured netcdf output with various
helper functions.  mostly just a place to collect relevant
bits of code.
"""
from __future__ import print_function

import numpy as np
import netCDF4
import trigrid
import unstructured_grid
from matplotlib.dates import date2num
import datetime
import pytz
import qnc

def ncslice(ncvar,**kwargs):
    """
    slicing a netcdf var by dimension names, passed
    as keyword arguments.

    for now, all slices are done by netcdf
    often this is slower, and it's better to fetch all the
    data and slice in numpy - not sure how to test for which
    would be faster
    """
    slices=[slice(None)]*len(ncvar.dimensions)
    for dim,dim_slice in kwargs.items():
        sidx=ncvar.dimensions.index(dim)
        slices[sidx]=dim_slice
    return ncvar[tuple(slices)]

def cf_to_datenums(nc_t_var):
    """ parse the 'units since epoch' style of time axis
    to python datenums
    """ 
    units,origin = nc_t_var.units.split(" since ")
    try:
        origin_date = datetime.datetime.strptime(origin,'%Y-%m-%d %H:%M:%S %Z')
    except ValueError:
        origin_date = datetime.datetime.strptime(origin,'%Y-%m-%d %H:%M:%S')
        
    if origin_date.tzinfo is not None:
        origin_date = origin_date.astimezone(pytz.utc)
    else:
        # Not sure why this happens...
        # print "Failed to get timezone - assuming netcdf time is UTC"
        pass
    tzero = date2num( origin_date )

    div = dict(seconds=86400.,
               minutes=60*24,
               hours=24,
               days=1)[units]
    return tzero + nc_t_var[:] / div

class Ugrid(object):
    surface_dzmin = 2*0.001 # common value, but no guarantee that this matches suntans code.
    
    def __init__(self,nc):
        """
        nc: path to netcdf dataset, or open dataset
        """

        if isinstance(nc,str):
            self.nc_filename = nc
            # self.nc = netCDF4.Dataset(self.nc_filename,'r')
            self.nc = qnc.QDataset(self.nc_filename,'r')
        else:
            self.nc_filename = None
            self.nc = nc

        # for operations which require a mesh to be specified, this is the default:
        self.mesh_name = self.mesh_names()[0]

    def find_var(self,**kwargs):
        """ find a variable name based on attributes (and other details, as
        added)
        """
        def is_match(attr,pattern):
            # eventually allow wildcards, negation, etc.
            if isinstance(pattern,list):
                return attr in pattern
            else:
                return attr==pattern
        
        for vname in self.nc.variables.keys():
            var=self.nc.variables[vname]
            for k,v in kwargs.iteritems():
                if k in var.ncattrs() and is_match(getattr(var,k),v):
                    pass
                else:
                    break # skips else clause below
            else:
                # completed all iterations - we found it!
                return vname
        
    def data_variable_names(self,mesh_name):
        """ return list of variables which appear to have real data (i.e. not just mesh
        geometry / topology)
        """
        mesh_name = mesh_name or self.mesh_name
        
        data_names = []

        prefix = mesh_name+'_'
        for vname in self.nc.variables.keys():
            if vname.startswith(prefix):
                if self.nc.dimensions.has_key(vname):
                    continue
                if hasattr(self.nc.variables[vname],'cf_role'):
                    continue
                data_names.append( vname[len(prefix):] )
        return data_names
        
    def mesh_names(self):
        """
        Find the meshes in the file, based on cf_role == 'mesh_topology'
        """
        meshes = []
        for vname in self.nc.variables.keys():
            try:
                if self.nc.variables[vname].cf_role == 'mesh_topology':
                    meshes.append(vname)
            except AttributeError:
                pass
        return meshes

    _node_cache = None
    def get_node_array(self,node_coordinates):
        """ given a 'node_x_coordinates node_y_coordinates'
        attribute from a mesh_topology variable, return an [Nnodes,3]
        array for the node locations.  The z coordinate is set to 0.
        Note that this array is cached, so z should not be modified
        in the return value.
        """
        if self._node_cache is None:
            self._node_cache = {}

        if not self._node_cache.has_key(node_coordinates):
            node_x_name,node_y_name = node_coordinates.split()
            node_x = self.nc.variables[node_x_name][:]
            node_y = self.nc.variables[node_y_name][:]
            node_z = 0.0 * node_x

            self._node_cache[node_coordinates] = np.vstack( (node_x,node_y,node_z) ).T
        return self._node_cache[node_coordinates]
    
    def Ncells(self,mesh_name=None):
        mesh_name = mesh_name or self.mesh_name
        mesh = self.nc.variables[mesh_name]
        # FIX: this makes a big assumption on the order of dimensions!
        return len( self.nc.dimensions[self.nc.variables[mesh.face_node_connectivity].dimensions[0]] )

    def Nkmax(self,mesh_name=None):
        mesh_name = mesh_name or self.mesh_name

        for dim_name in 'n%s_layers'%mesh_name, 'nMeshGlobal_layers':
            try:
                return len( self.nc.dimensions[dim_name] )
            except KeyError:
                pass
        raise Exception("Failed to find vertical dimension")

    def get_cell_velocity(self,time_slice,mesh_name=None,face_slice=slice(None)):
        """ Return 2-vector valued velocity.
        ordering of dimensions is same as in the netcdf variables, with
        velocity component at the end
        (which at least with the local suntans nc code is (face,layer,time) )
        """ 
        # time_slice used to be time_step
        mesh_name = mesh_name or self.mesh_name

        slices={'time':time_slice,
                'n'+mesh_name+'_face':face_slice}
        u_comp=ncslice(self.nc.variables[mesh_name+'_cell_east_velocity'],**slices)
        v_comp=ncslice(self.nc.variables[mesh_name+'_cell_north_velocity'],**slices)

        U=np.concatenate( (u_comp[...,None],
                           v_comp[...,None]),
                          axis=-1 )
            
        # U = np.zeros( (self.Ncells(mesh_name),self.Nkmax(),2), np.float64)
        # U[:,:,0] = self.nc.variables[mesh_name + '_cell_east_velocity'][:,:,time_slice]
        # U[:,:,1] = self.nc.variables[mesh_name + '_cell_north_velocity'][:,:,time_slice]
        return U

    def get_cell_scalar(self,label,time_step,mesh_name=None):
        mesh_name = mesh_name or self.mesh_name
        
        # totally untested!
        return self.nc.variables[mesh_name + '_' + label][:,:,time_step]
    
    def to_trigrid(self,mesh_name=None,skip_edges=False):
        nc = self.nc
        mesh_name = mesh_name or self.mesh_name
        mesh = nc.variables[mesh_name]

        node_x_name,node_y_name = mesh.node_coordinates.split()
        
        node_xy = np.array( [nc.variables[node_x_name][...],
                             nc.variables[node_y_name][...]]).T
        faces = nc.variables[mesh.face_node_connectivity][...]
        edges = nc.variables[mesh.edge_node_connectivity][...] # [N,2]
        g = trigrid.TriGrid(points=node_xy,cells=faces,edges=edges)
        
        if not skip_edges:
            # g.make_edges_from_cells() # this completely recreates the edges
            # instead, we need to discern the edge-cell connectivity from the
            # supplied edges
            pass
        return g
        
    def grid(self,mesh_name=None,skip_edges=False):
        """ return an UnstructuredGrid object
        """
        nc = self.nc
        mesh_name = mesh_name or self.mesh_name
        mesh = nc.variables[mesh_name]

        node_x_name,node_y_name = mesh.node_coordinates.split()
        
        node_xy = np.array( [nc.variables[node_x_name][...],
                             nc.variables[node_y_name][...]]).T
        faces = nc.variables[mesh.face_node_connectivity][...]
        edges = nc.variables[mesh.edge_node_connectivity][...] # [N,2]
        ug = unstructured_grid.UnstructuredGrid(points=node_xy,cells=faces,edges=edges)
        
        return ug

    def layer_var_name(self):
        # this had been searching in dimensions, but that doesn't seem quite 
        # right
        for name in self.nc.variables.keys(): # but also try looking for it.
            try:
                if self.nc.variables[name].standard_name == 'ocean_zlevel_coordinate':
                    return name
                # should check for more standard names...
            except KeyError:
                pass
            except AttributeError:
                pass
        else:
            return 'nMeshGlobal_layers'
    
    def mesh_face_dim(self,mesh):
        face_node=self.nc.variables[mesh].face_node_connectivity
        return self.nc.variables[face_node].dimensions[0]
    def mesh_edge_dim(self,mesh):
        edge_node=self.nc.variables[mesh].edge_node_connectivity
        return self.nc.variables[edge_node].dimensions[0]

    def vertical_averaging_weights(self,time_slice=slice(None),
                                   ztop=None,zbottom=None,dz=None,
                                   mesh_name=None,
                                   face_slice=slice(None)):
        """
        reimplementation of sunreader.Sunreader::averaging_weights
        
        Returns weights as array [Nk] to average over a cell-centered quantity
        for the range specified by ztop,zbottom, and dz.

        range is specified by 2 of the 3 of ztop, zbottom, dz, all non-negative.
        ztop: distance from freesurface
        zbottom: distance from bed
        dz: thickness

        if the result would be an empty region, return nans.

        cell_select: an object which can be used to index into the cell dimension
          defaults to all cells.
          
        this thing is slow! - lots of time in adjusting all_dz

        order of dimensions has been altered to match local suntans netcdf code,
         i.e. face,level,time
        """
        mesh_name = mesh_name or self.mesh_name
        
        mesh = self.nc.variables[mesh_name]

        face_dim=self.mesh_face_dim(mesh_name)

        surface=self.find_var(standard_name='sea_surface_height_above_geoid',
                              mesh=mesh_name)
        
        face_select={face_dim:face_slice}
        h = ncslice(self.nc.variables[surface],time=time_slice,
                    **face_select)

        depth=self.find_var(standard_name=["sea_floor_depth_below_geoid",
                                           "sea_floor_depth"],
                            mesh=mesh_name,
                            location='face') # ala 'Mesh_depth'
        
        bed = ncslice(self.nc.variables[depth],**face_select)
        if getattr(self.nc.variables[depth],'positive','up') == 'down':
            bed=-bed

        # expand bed to include a time dimension so that in the code below
        # h and bed have the same form, even though bed is not changing in time
        # but h is.
        if h.ndim>bed.ndim:
            bed=bed[...,None] * np.ones_like(h)
        
        # for now, can only handle an array of cells - i.e. if you want
        # a single face, it's still going to process an array, just with
        # length 1.
        Ncells = len(bed)

        # 
        layers = self.nc.variables[self.layer_var_name()]
        layer_bounds = self.nc.variables[ layers.bounds ][...]

        # hmm - some discrepancies over the dimensionality of layer_interfaces
        # assumption is probably that the dimensions are [layer,{top,bottom}]
        if layer_bounds.ndim==2 and layer_bounds.shape[1]==2:
            # same layer interfaces for all cells, all time.
            layer_interfaces = np.concatenate( (layer_bounds[:,0],layer_bounds[-1:,1]) )
            # this is a bit trickier, because there could be lumping.  for now, it should work okay
            # with 2-d, but won't be good for 3-d
            Nk = np.searchsorted(-layer_interfaces,-bed)
        else:
            raise Exception("Not smart enough about layer_bounds to do this")
        
        one_dz = -np.diff(layer_interfaces)
        # all_dz = one_dz[None,:].repeat(Ncells,axis=0)
        all_dz=np.ones(h.shape+one_dz.shape)*one_dz
        all_k = np.ones(h.shape+one_dz.shape)*np.arange(len(one_dz))


        # adjust bed and 
        # 3 choices here..
        # try to clip to reasonable values at the same time:
        if ztop is not None:
            if ztop != 0:
                h = h - ztop # don't modify h
                # don't allow h to go below the bed
                h[ h<bed ] = bed[ h<bed ]
            if dz is not None:
                # don't allow bed to be below the real bed.
                bed = np.maximum( h - dz, bed)
        if zbottom is not None:
            # no clipping checks for zbottom yet.
            if zbottom != 0:
                bed = bed + zbottom # don't modify bed!
            if dz is not None:
                h = bed + dz

        # so now h and bed are elevations bounding the integration region
        z = layer_bounds.min(axis=1) # bottom of each cell
        ctops = np.searchsorted(-z - self.surface_dzmin, -h)

        # default h_to_ctop will use the dzmin appropriate for the surface,
        # but at the bed, it goes the other way - safest just to say dzmin=0,
        # and also clamp to known Nk
        cbeds = np.searchsorted(-z,-bed) + 1 # it's an exclusive index

        # dimension problems here - Nk has dimensions like face_slice or face_slice,time_slice
        # cbeds has dimensions like face_slice,time_slice
        # how to conditionally add dimensions to Nk?
        # for now, ASSUME that time is after face, and use shape of h to
        # figure out how to pad it
        while h.ndim > Nk.ndim:
            Nk=Nk[...,None]
        # also have to expand Nk so that the boolean indexing works
        cbeds[ cbeds>Nk ] = (Nk*np.ones_like(cbeds))[ cbeds>Nk ]

        # seems that there is a problem with how dry cells are handled -
        # for the exploratorium display this ending up with a number of cells with
        # salinity close to 1e6.
        # in the case of a dry cell, ctop==cbed==Nk[i]
        drymask = (all_k < ctops[...,None]) | (all_k>=cbeds[...,None])
        all_dz[drymask] = 0.0

        ii = tuple(np.indices( h.shape ) )
        all_dz[ii+(ctops[ii],)] = h-z[ctops]
        all_dz[ii+(cbeds[ii]-1,)] -= bed - z[cbeds-1]
        
        # ctops has indices into the z dimension, and it has
        #   cell,time shape
        #  h also has cell,time shape
        # old code:
        # all_dz[ii,ctops] = h - z[ctops]
        # all_dz[ii,cbeds-1] -= bed - z[cbeds-1]

        # make those weighted averages
        # have to add extra axis to get broadcasting correct
        all_dz = all_dz / np.sum(all_dz,axis=-1)[...,None]

        if all_dz.ndim==3:
            # we have both time and level
            # transpose to match the shape of velocity data -
            all_dz = all_dz.transpose([0,2,1])
        return all_dz

    def datenums(self):
        """ return datenums, referenced to UTC
        """
        return cf_to_datenums(self.nc.variables['time'])

