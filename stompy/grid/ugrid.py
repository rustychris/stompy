"""
wrapper around unstructured netcdf output with various
helper functions.  mostly just a place to collect relevant
bits of code.
"""
from __future__ import print_function

import six
import numpy as np
import netCDF4
from . import trigrid
from . import unstructured_grid
import logging
log=logging.getLogger('ugrid')
from matplotlib.dates import date2num
import datetime
import pytz
from ..io import qnc
import xarray as xr
from .. import utils
from . import multi_ugrid as mu
import time

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
            for k,v in six.iteritems(kwargs):
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
        if 'bounds' in layers.attrs:
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
        else:
            import pdb
            pdb.set_trace()


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

class UgridXr(object):
    """
    Transition to xarray instead of qnc.
    """
    surface_dzmin = 2*0.001 # common value, but no guarantee that this matches suntans code.

    # These are set at instantiation
    time_dim='time'
    time_vname='time'
    face_dim=None
    edge_dim=None
    layer_dim=None

    # These are figured out on the fly if not specified
    face_u_vname=None
    face_v_vname=None
    face_eta_vname=None
    face_depth_vname=None
    edge_normal_vnames=None # (x-normal,y-normal)
    layer_vname=None

    def __init__(self,nc,mesh_name=None,**kw):
        """
        nc: path to netcdf dataset, or open dataset
        mesh_name: which mesh to use if the file contains multiple.
        """
        self.__dict__.update(kw)
        if isinstance(nc,str):
            self.nc_filename = nc
            self.nc = xr.open_dataset(self.nc_filename)
        else:
            self.nc_filename = None
            self.nc = nc

        # for operations which require a mesh to be specified, this is the default:
        if mesh_name is not None:
            self.mesh_name = mesh_name
        else:
            assert len(self.mesh_names())==1
            self.mesh_name=self.mesh_names()[0]
        self.find_dims()

    def find_dims(self):
        if self.face_dim is None:
            self.face_dim=self.find_face_dim()
        if self.edge_dim is None:
            self.edge_dim=self.find_edge_dim()
        if self.layer_dim is None:
            # bit of a punt, makes assumptions
            self.layer_dim=self.nc[self.layer_var_name()].dims[0]

        # Time is slightly different. defaults to 'time', but that may not exist.
        if self.time_dim is not None:
            if self.time_dim not in self.nc.dims:
                self.time_dim=None
                self.time_vname=None

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
        
        for vname in self.nc.variables:
            var=self.nc[vname]
            for k,v in six.iteritems(kwargs):
                if k in var.attrs and is_match(var.attrs[k],v):
                    pass
                else:
                    break # skips else clause below
            else:
                # completed all iterations - we found it!
                return vname
    def data_variable_names(self):
        """ return list of variables which appear to have real data (i.e. not just mesh
        geometry / topology)
        """
        data_names = []

        # this is a bit more general/lenient than the Qnc version
        for vname in self.nc.data_vars:
            if vname in self.nc.dims:
                continue
            if 'cf_role' in self.nc[vname].attrs:
                continue
            data_names.append( vname )
        return data_names

    def mesh_names(self):
        """
        Find the meshes in the file, based on cf_role == 'mesh_topology'
        """
        meshes = []
        for vname in self.nc.variables:
            if self.nc[vname].attrs.get('cf_role') == 'mesh_topology':
                meshes.append(vname)
        return meshes

    def get_node_array(self,*a,**k):
        assert False,"Use grid.nodes instead"

    def Nkmax(self):
        # This will need to be generalized
        for dim_name in 'n%s_layers'%self.mesh_name, 'nMeshGlobal_layers':
            try:
                return self.nc.dims[dim_name]
            except KeyError:
                pass
        raise Exception("Failed to find vertical dimension")

    def get_cell_velocity(self,time_slice,face_slice=None):
        """ Return 2-vector valued velocity.
        OLD: ordering of dimensions is same as in the netcdf variables, with
             velocity component at the end
        NEW: force ordering to be time,face,layer,component

        (which at least with the local suntans nc code is (face,layer,time) )
        """ 
        # time_slice used to be time_step
        mesh_name = self.mesh_name

        if self.time_dim is not None:
            slices={self.time_dim:time_slice}
        else:
            slices={}
            
        if face_slice is not None:
            slices[self.face_dim]=face_slice

        u_slc=self.nc[self.face_u_vname].isel(**slices)
        v_slc=self.nc[self.face_v_vname].isel(**slices)
        
        # this is the area where it bogs down big time.
        # depending on the slices, some dimensions may have disappeared,
        # but force an ordering on whatever is left.
        new_dims=[d for d in [self.time_dim,self.face_dim,self.layer_dim]
                  if d in u_slc.dims]
        if 0: 
            u_comp=u_slc.transpose(*new_dims).values
            v_comp=v_slc.transpose(*new_dims).values
        else:
            # Try a numpy approach.
            tran=[u_slc.dims.index(nd) for nd in new_dims]
            u_comp=u_slc.values.transpose(tran)
            v_comp=v_slc.values.transpose(tran)
            
        U=np.concatenate( (u_comp[...,None],
                           v_comp[...,None]),
                          axis=-1 )
        return U

    def get_cell_scalar(self,label,time_step):
        # totally untested!  and not much of a savings
        return self.nc[label].isel({self.time_dim:time_step}).values

    _grid=None
    @property
    def grid(self):
        """ return an UnstructuredGrid object
        """
        if self._grid is None:
            self._grid=unstructured_grid.UnstructuredGrid.from_ugrid(self.nc,mesh_name=self.mesh_name)
        return self._grid

    def layer_var_name(self):
        # this had been searching in dimensions, but that doesn't seem quite 
        # right
        if self.layer_vname is not None:
            return self.layer_vname
        
        for name in self.nc.variables: # but also try looking for it.
            if self.nc[name].attrs.get('standard_name') in ['ocean_zlevel_coordinate',
                                                            'ocean_z_coordinate',
                                                            'ocean_sigma_coordinate']:
                return name
        else:
            return 'nMeshGlobal_layers' # total punt
    
    def find_face_dim(self):
        mesh_var=self.nc[self.mesh_name]
        if 'face_dimension' in mesh_var.attrs:
            return mesh_var.attrs['face_dimension']
        else:
            face_node=mesh_var.attrs['face_node_connectivity']
            return self.nc[face_node].dims[0]
    def find_edge_dim(self):
        mesh_var=self.nc[self.mesh_name]
        if 'edge_dimension' in mesh_var.attrs:
            return mesh_var.attrs['edge_dimension']
        else:
            edge_node=mesh_var.attrs['edge_node_connectivity']
            return self.nc[edge_node].dims[0]

    def vertical_averaging_weights(self,time_slice=slice(None),
                                   ztop=None,zbottom=None,dz=None,
                                   face_slice=slice(None),
                                   query='weight'):
        """
        reimplementation of sunreader.Sunreader::averaging_weights
        
        returns: weights as array [faces,Nk] to average over a cell-centered quantity
        for the range specified by ztop,zbottom, and dz.

        query: by default returns averaging weights, but can also specify
          'dz': thickness of each 3D cell
          'z_center': elevation of the middle of each 3D cell
          'z_bottom': elevation of the bottom of each 3D cell
          'z_top': elevation of the top of each 3D cell

        can also be a list of the same

        range is specified by 2 of the 3 of ztop, zbottom, dz, all non-negative.
        ztop: dimensional distance from freesurface, 
        zbottom: dimensional distance from bed
        dz: thickness

        if the result would be an empty region, return nans.

        cell_select: an object which can be used to index into the cell dimension
          defaults to all cells.
          
        this thing is slow! - lots of time in adjusting all_dz

        order of dimensions has been altered to match local suntans netcdf code,
         i.e. face,level,time
        """
        mesh = self.nc[self.mesh_name]

        face_dim=self.face_dim

        if self.time_dim is not None:
            time_kw={self.time_dim:time_slice}
        else:
            time_kw={}
            
        if self.face_eta_vname is None:
            self.face_eta_vname=self.find_var(standard_name='sea_surface_height_above_geoid')
            assert self.face_eta_vname is not None,"Failed to discern eta variable"
        surface=self.face_eta_vname

        face_select={}
        hsel={}
        
        if face_slice!=slice(None):
            face_select[face_dim]=face_slice
            hsel[face_dim]=face_slice
            
        hsel.update(time_kw)
        h=self.nc[self.face_eta_vname]
        if len(hsel):
            h = h.isel(**hsel)

        if self.face_depth_vname is None:
            self.face_depth_vname=self.find_var(standard_name=["sea_floor_depth_below_geoid",
                                                               "sea_floor_depth"],
                                                location='face') # ala 'Mesh_depth'
        if self.face_depth_vname is None:
            # c'mon people -- should be fixed in source now,
            self.face_depth_vname=self.find_var(stanford_name=["sea_floor_depth_below_geoid",
                                                               "sea_floor_depth"],
                                                location='face') # ala 'Mesh_depth'
        if self.face_depth_vname is None:
            self.face_depth_vname=self.find_var(standard_name=['altitude'],
                                                location='face')
            
        depth=self.face_depth_vname
        assert depth is not None,"Failed to find depth variable"
        
        bed = self.nc[depth]
        if len(face_select):
            bed=bed.isel(**face_select)
            
        if self.nc[depth].attrs.get('positive')=='down':
            log.debug("Cell depth is positive-down")
            bed=-bed
        else:
            log.debug("Cell depth is positive-up, or at least that is the assumption")

        # special handling for multi-ugrid
        if isinstance(h,mu.MultiVar):
            h=h.to_dataarray()
        if isinstance(bed,mu.MultiVar):
            bed=bed.to_dataarray()
            
        h,bed=xr.broadcast(h,bed)
        
        # for now, can only handle an array of cells - i.e. if you want
        # a single face, it's still going to process an array, just with
        # length 1.
        Ncells = len(bed)

        layers = self.nc[self.layer_var_name()]
        layer_vals=layers.values
        if layers.attrs.get('positive')=='down':
            layer_vals=-layer_vals
            
        if 'bounds' in layers.attrs:
            layer_bounds = self.nc[ layers.attrs['bounds'] ].values

            # hmm - some discrepancies over the dimensionality of layer_interfaces
            # assumption is probably that the dimensions are [layer,{top,bottom}]
            if layer_bounds.ndim==2 and layer_bounds.shape[1]==2:
                # same layer interfaces for all cells, all time.
                layer_interfaces = np.concatenate( (layer_bounds[:,0],layer_bounds[-1:,1]) )
                if layers.attrs.get('positive')=='down':
                    layer_interfaces=-layer_interfaces
            else:
                raise Exception("Not smart enough about layer_bounds to do this")
        else:
            dz_single=0-bed.values.min() # assumes typ eta of 0.  only matters for 2D
            layer_interfaces=utils.center_to_edge(layer_vals,dx_single=dz_single)
            layer_bounds=np.concatenate( (layer_interfaces[:-1, None],
                                          layer_interfaces[1:, None]),
                                         axis=1)
        # used to retain layer_interfaces for the top of the top and the
        # bottom of the bottom.  But that just makes for more cleanup
        # so now clip this to be interfaces between two layers.
        layer_interfaces=layer_interfaces[1:-1]
        
        # Calls to searchsorted below may need to negate both arguments
        # if increasing k maps to decreasing elevation.
        if np.all( np.diff(layer_interfaces) < 0 ):
            k_sign=-1
        elif np.all( np.diff(layer_interfaces) > 0):
            k_sign=1
        else:
            raise Exception("Confused about the ordering of k")
            
        # this is a bit trickier, because there could be lumping.  for now, it should work okay
        # with 2-d, but won't be good for 3-d HERE if k is increasing up, this is WRONG
        # this used to be called Nk, but that's misleading.  it's the k index
        # of the bed layer, not the number of layers per water column.
        kbed = np.searchsorted(k_sign*layer_interfaces,k_sign*bed)
        # print("kbed: ",kbed)

        one_dz = k_sign*(layer_bounds[:,1]-layer_bounds[:,0])
        all_dz=np.ones(h.shape+one_dz.shape)*one_dz
        all_k = np.ones(h.shape+one_dz.shape,np.int32)*np.arange(len(one_dz))

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
        # with this min call it's only correct for k_sign==-1
        ctops = np.searchsorted(k_sign*(layer_interfaces + self.surface_dzmin), 
                                k_sign*h)
        # print("k_sign:",k_sign)
        #print("k_sign*h", k_sign*h)
        #print("k_sign*(layer_interfaces+self.surface_dzmin)\n",
        #      k_sign*(layer_interfaces + self.surface_dzmin))
        #print("surface_dzmin: ", self.surface_dzmin)
        
        # default h_to_ctop will use the dzmin appropriate for the surface,
        # but at the bed, it goes the other way - safest just to say dzmin=0,
        # and also clamp to known Nk
        cbeds = np.searchsorted(k_sign*layer_interfaces,k_sign*bed)

        # 2022-08-01 RH: 
        # pretty sure that ctops should never be below cbed. even for a dry
        # water column?
        if k_sign==1:
            ctops=np.maximum(ctops,cbeds)
        else:
            ctops=np.minimum(ctops,cbeds)

        # print("bed:",bed,"  h: ",h)

        # dimension problems here - Nk has dimensions like face_slice or face_slice,time_slice
        # cbeds has dimensions like face_slice,time_slice
        # how to conditionally add dimensions to Nk?
        # for now, ASSUME that time is after face, and use shape of h to
        # figure out how to pad it
        while h.ndim > kbed.ndim:
            kbed=kbed[...,None]
        # also have to expand Nk so that the boolean indexing works
        
        # use to make cbeds exclusive indexing, but its cleaner to leave
        # ctops and cbeds both as inclusive, since it changes based on
        # k_sign
        if k_sign==-1:
            # keep cbed valid w.r.t. to deepest layer kbed,            
            cbeds=np.minimum(cbeds,kbed)
            drymask = (all_k < ctops[...,None]) | (all_k>cbeds[...,None])
        else:
            cbeds=np.maximum(cbeds,kbed) # maybe redundant now
            drymask = (all_k < cbeds[...,None]) | (all_k>ctops[...,None])

        # print("cbeds:",cbeds)
            
        all_dz[drymask] = 0.0

        ii = tuple(np.indices( h.shape ) )
        z = layer_bounds.min(axis=1) # bottom of each cell

        # print("h",h)
        # print("ii",ii)
        # print("ctops[ii]",ctops[ii])
        # print("z[ctop]",z[ctops])
        
        all_dz[ii+(ctops[ii],)] = h-z[ctops]
        # isub doesn't play nicely with dask array, so write out the isub
        all_dz[ii+(cbeds[ii],)] = all_dz[ii+(cbeds[ii],)] - (bed - z[cbeds])

        # DBG
        # print("all_dz",all_dz)
        

        # handle the various query options
        if isinstance(query,str):
            queries=[query]
            singleton=True
        else:
            queries=query
            singleton=False

        results=[]
        for query in queries:
            if query in ['weight','dz']:
                if query=='weight':
                    # make those weighted averages
                    # have to add extra axis to get broadcasting correct
                    # avoid warnings.
                    denom= np.sum(all_dz,axis=-1)[...,None]
                    denom[denom==0.0]=np.nan
                    result = all_dz/denom
                else:
                    result = all_dz
            elif query=='z_bottom':
                # Might need to be smarter if all_dz include time.
                result=bed.values[:,None]+np.cumsum(all_dz,axis=-1) - all_dz
            elif query=='z_top':
                result=bed.values[:,None]+np.cumsum(all_dz,axis=-1)
            elif query=='z_center':
                result=bed.values[:,None]+np.cumsum(all_dz,axis=-1) - 0.5*all_dz
            else:
                raise Exception("Unknown query %s"%q)

            if result.ndim==3:
                # we have both time and level
                # transpose to match the shape of velocity data -
                result = result.transpose([0,2,1])
                
            results.append(result)
        if singleton:
            return results[0]
        else:
            return results

    def datenums(self):
        """ return datenums, referenced to UTC
        """
        return utils.to_dnum(self.times())
    def times(self):
        """ return time steps as datetime64, referenced to UTC
        """
        return self.nc['time'].values

    def interp_perot(self,edge_values):
        """
        Interpolate an edge-centered, normal vector component 
        value to a cell centered vector value.
        returns float64[Ncells,2]
        """
        # Originally borrowed model.stream_tracer.U_perot, but
        # (a) that's a weird place for the code to live, and
        # (b) that code seems to have a different sign convention, or
        # at least it was giving bad results in some spot tests.
        # And now it has moved again to unstructured grid...
        return self.grid.interp_perot(edge_values,edge_normals=self.edge_normals)

    def edge_normals(self):
        if self.edge_normal_vnames is not None:
            ex,ey=self.edge_normal_vnames
            return np.c_[self.nc[ex].values, self.nc[ey].values]
        else:
            return self.grid.edges_normals()
