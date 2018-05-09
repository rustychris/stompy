#!/usr/bin/env python
from __future__ import print_function

# unstructured_grid.py
#  common methods for manipulating unstructured grids, specifically mixed quad/tri
#  grids from FISH-PTM and UnTRIM

import sys,os,types
import logging
 
try:
    from osgeo import ogr,osr
except ImportError:
    logging.info("OGR, GDAL unavailable")
import copy
try:
    import cPickle as pickle # python2
except ImportError: 
    import pickle # python3
import six

# trying to be responsible and not pollute the namespace.
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

from shapely import wkt,geometry,wkb,ops
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.path import Path

from ..spatial import gen_spatial_index, proj_utils
from ..utils import (mag, circumcenter, circular_pairs,signed_area, poly_circumcenter,
                     orient_intersection,array_append,within_2d, to_unit,
                     recarray_add_fields,recarray_del_fields)

try:
    import netCDF4    
    from ..io import qnc
except ImportError:
    logging.info("netcdf unavailable")
    netCDF4=None
    
try:
    # both depend on ogr
    from ..spatial import (wkb2shp, join_features)
except ImportError:
    logging.info( "No wkb2shp, join_features" )

try:
    import xarray as xr
except ImportError:
    logging.warning("No xarray - some functions may not work")

    
from .. import undoer

try:
    from .. import priority_queue as pq
except ImportError:
    pq = 'NO priority queue found!'


class GridException(Exception):
    pass

class Missing(GridException):
    pass

def request_square(ax):
    """
    Attempt to set a square aspect ratio on matplotlib axes ax
    """
    # in older matplotlib, this was sufficient:
    # ax.axis('equal')
    # But in newer matplotlib, if the axes are shared,
    # that fails.
    # Maybe this is better?
    plt.setp(ax,aspect=1.0,adjustable='box-forced')
    

class HalfEdge(object):
    def __init__(self,grid,edge,orient):
        """
        orient: 0 means the usual from node 0 to node 1, i.e. "adjacent"
        to edges['cells'][j,0]
        """
        self.grid=grid
        self.j=edge
        # orient:
        self.orient=orient

    def __str__(self):
        return "<HalfEdge %d -> %d>"%( self.node_rev(),self.node_fwd() )

    def __repr__(self):
        return self.__str__()

    def nbr(self,direc):
        """ direc: 0 means fwd, 1 is reverse
        This returns an adjacent half-edge
        """
        n1=self.grid.edges['nodes'][self.j,0]
        n2=self.grid.edges['nodes'][self.j,1]

        if (direc+self.orient) % 2 == 0:
            # combination of the half-edge orientation and
            # the requested direction mean we follow the natural
            # direction of the edge
            rev_node = n1
            fwd_node = n2
        else:
            # they combine to go the opposite direction
            rev_node = n2
            fwd_node = n1
        # tmp: 
        nbrs=self.grid.angle_sort_adjacent_nodes(fwd_node,ref_nbr=rev_node)
        if direc==0:
            fwdfwd_node=nbrs[-1] # the most ccw, or first cw neighbor
            return HalfEdge.from_nodes(self.grid,fwd_node,fwdfwd_node)
        else:
            # not sure about this...
            # the weird modulo is because nbrs may have only 1 item.
            fwdfwd_node=nbrs[1%len(nbrs)] # next ccw neighbor
            return HalfEdge.from_nodes(self.grid,fwdfwd_node,fwd_node)
    def fwd(self):
        return self.nbr(direc=0)
    def rev(self):
        return self.nbr(direc=1)

    def cell(self):
        # the cell (or a <0 flag) which this half-edge faces
        return self.grid.edges['cells'][self.j,self.orient]
    def cell_opp(self):
        return self.grid.edges['cells'][self.j,1-self.orient]

    def opposite(self):
        return HalfEdge(grid=self.grid,edge=self.j,orient=1-self.orient)
    
    def node_rev(self):
        """ index of the node in the reverse direction of the halfedge """
        return self.grid.edges['nodes'][self.j, self.orient]
    def node_fwd(self):
        """ index of the node in the forward direction of the halfedge """
        return self.grid.edges['nodes'][self.j, 1-self.orient]
    def nodes(self):
        """ 
        equivalent to [node_rev(),node_fwd()]
        """
        return self.grid.edges['nodes'][self.j, [self.orient, 1-self.orient]]

    @staticmethod
    def from_nodes(grid,rev,fwd):
        j=grid.nodes_to_edge(rev,fwd)
        if j is None:
            return None
        j1,j2 = grid.edges['nodes'][j,:]
        
        if (j1,j2) == (rev,fwd):
            orient=0
        elif (j2,j1) == (rev,fwd):
            orient=1
        else:
            assert False 
        return HalfEdge(grid,j,orient)
    def __eq__(self,other):
        return ( (other.grid   == self.grid) and
                 (other.j      == self.j )   and
                 (other.orient == self.orient) )

    def __ne__(self,other):
        # per python data model, equals does not imply
        # opposite of not equals, and must be explicitly handled.
        return not self.__eq__(other)


def rec_to_dict(r):
    d={}
    for name in r.dtype.names:
        d[name]=r[name]
    return d

# two parts - a baseclass which handles the real work
# of registering listeners for a particular method,
# and a decorator to streamline setting which methods
# can be monitored.

class Listenable(object):
    def __init__(self,*a,**k):
        super(Listenable,self).__init__(*a,**k)
        self.__post_listeners=defaultdict(list) # func_name => list of functions
        self.__pre_listeners =defaultdict(list) # ditto

    def subscribe_after(self,func_name,callback):
        if callback not in self.__post_listeners[func_name]:
            self.__post_listeners[func_name].append(callback)
    def subscribe_before(self,func_name,callback):
        if callback not in self.__pre_listeners[func_name]:
            self.__pre_listeners[func_name].append(callback)
    def unsubscribe_after(self,func_name,callback):
        if callback in self.__post_listeners[func_name]:
            self.__post_listeners[func_name].remove(callback)
    def unsubscribe_before(self,func_name,callback):
        if callback in self.__pre_listeners[func_name]:
            self.__pre_listeners[func_name].remove(callback)
        
    def fire_after(self,func_name,*a,**k):
        for func in self.__post_listeners[func_name]:
            func(self,func_name,*a,**k)
    def fire_before(self,func_name,*a,**k):
        for func in self.__pre_listeners[func_name]:
            func(self,func_name,*a,**k)

    def __getstate__(self):
        # awkward, verbose code to get around the 'hiding' when attributes
        # start with __
        save_pre=self.__pre_listeners
        save_post=self.__post_listeners
        self.__pre_listeners=defaultdict(list)
        self.__post_listeners=defaultdict(list)
        
        try:
            d=super(Listenable,self).__getstate__()
        except AttributeError:
            d = dict(self.__dict__)

        self.__pre_listeners=save_pre
        self.__post_listeners=save_post

        return d

    #def __setstate__(self,state):
    #    self.__dict__.update(state)
        
from functools import wraps
def listenable(f): 
    @wraps(f)
    def wrapper(self,*args,**kwargs):
        func_name=f.__name__ # used to be f.func_name, but that disappeared in py3k
        self.fire_before(func_name,*args,**kwargs)
        val=f(self,*args,**kwargs)
        self.fire_after(func_name,*args,return_value=val,**kwargs)
        return val

    return wrapper

class UnstructuredGrid(Listenable,undoer.OpHistory):
    #-# Basic definition of interface:
    max_sides = 4 # N.B. - must be set before or during __init__, or taken from cells.shape

    # useful constants - part of the class to make it easier when subclassing
    # in other modules
    UNKNOWN = -99 # need to recalculate
    UNMESHED = -2 # edges['cells'] for edge with an unmeshed boundary
    # for nodes or edges beyond the number of sides of a given cell
    # also for a edges['cells'] when at a boundary
    UNDEFINED= -1
    
    INTERNAL=0 # edge mark - regular internal computational edges
    LAND=1  # edge mark - may mean land, or a grid boundary which hasn't been marked as flow
    FLOW=2  # edge mark
    BOUNDARY=3 # cell mark

    GridException=GridException
    
    xxyy = np.array([0,0,1,1])
    xyxy = np.array([0,1,0,1])

    # Define the data stored for each point
    # some of these are dependent on other aspects of the geometry, and should be 
    # set to nan if they become stale.
    # dependent values are prefixed with an underscore, and [eventually] there are
    # getter methods which will take care of updating/calculating the values if needed.
    # so don't use self.cells['_area'], use self.cells_area()

    # this is mostly convention, but in some places, like the refine code, fields starting
    # with an underscore are not copied since it's safe to recalculate afterwards.

    # the exceptions are cells['edges'] and edges['cells'] - even though these are
    # just consequences of the rest of the topology, the assumed invariant is that
    # (outside of initialization or during modification), they are kept consistent.
    
    node_dtype = [ ('x',(np.float64,2)),('deleted',np.bool8) ]
    cell_dtype  = [ # edges/nodes are set dynamically in __init__ since max_sides can change
                    ('_center',(np.float64,2)),  # typ. voronoi center
                    ('mark',np.int32),
                    ('_area',np.float64),
                    ('deleted',np.bool8)]
    edge_dtype = [ ('nodes',(np.int32,2)),
                   ('mark',np.int32),
                   ('cells',(np.int32,2)),
                   ('deleted',np.bool8)]

    ##
    def __init__(self,
                 grid = None,
                 edges=[],points=[],cells=[],
                 extra_node_fields=[],
                 extra_cell_fields=[],
                 extra_edge_fields=[],
                 max_sides=None):
        """
        grid: another UnstructuredGrid to copy from
        or...
        points: [N,2] node coordinates
        edges: [N,2] indices into points, 0-based
        cells: [N,maxsides] indices into points, 0-based, -1 for missing nodes.
        """
        cells=np.asanyarray(cells)
        super(UnstructuredGrid,self).__init__()

        self.init_log()
        # Do this early, so any early allocations use the right set of fields
        # but be careful not to modify the class-wide defs - just the instance's.
        self.node_dtype = self.node_dtype + extra_node_fields

        if max_sides is not None:
            self.max_sides=max_sides
        elif (cells is not None) and len(cells):
            self.max_sides=cells.shape[1]
        elif grid is not None:
            self.max_sides=grid.max_sides
        #otherwise default of 4.

        self.cell_dtype = [('edges',(np.int32,self.max_sides)),('nodes',(np.int32,self.max_sides))] +\
                          self.cell_dtype + extra_cell_fields
        self.edge_dtype = self.edge_dtype + extra_edge_fields
        
        if grid is not None:
            self.copy_from_grid(grid)
        else:
            self.from_simple_data(points=points,edges=edges,cells=cells)

    def copy(self):
        # maybe subclasses shouldn't be used here - for example,
        # this requires that every subclass include 'grid' in its
        # __init__.  Maybe more subclasses should just be readers?
        # return self.__class__(grid=self)
        return UnstructuredGrid(grid=self)

    def copy_from_grid(self,grid):
        # this takes care of allocation, and setting the most basic topology
        self.from_simple_data(points=grid.nodes['x'],
                              edges=grid.edges['nodes'],
                              cells=grid.cells['nodes'])
        for field in ['cells','mark','deleted']:
            self.edges[field] = grid.edges[field]
        for field in ['mark','_center','_area','deleted']:
            self.cells[field] = grid.cells[field]
        # special handling for edges, so it's not required for max_sides to
        # match up
        if self.max_sides < grid.max_sides:
            assert np.all(grid.cells['edges'][:,self.max_sides:]<0)
            self.cells['edges']=grid.cells['edges'][:,:self.max_sides]
        else:
            self.cells['edges'][:,grid.max_sides:]=self.UNDEFINED
            self.cells['edges'][:,:grid.max_sides]=grid.cells['edges']

    def reproject(self,src_srs,dest_srs):
        xform=proj_utils.mapper(src_srs,dest_srs)
        new_g=self.copy()
        new_g.nodes['x'] = xform(self.nodes['x'])
        new_g.cells['_center']=np.nan
        new_g._node_index=None
        new_g._cell_center_index=None

        return new_g

    def modify_max_sides(self,max_sides):
        """ 
        In-place modification of maximum number of sides for cells.
        Can be larger or smaller than current max_sides, but if smaller
        all existing cells must fit in the new max_sides.
        """
        if max_sides<self.max_sides:
            if not np.all( self.cells['nodes'][:,max_sides:] == self.UNDEFINED ):
                raise GridException("Some cells cannot fit in requested max_sides")

        old_max_sides=self.max_sides
        self.max_sides=max_sides

        # update dtypes for cells
        # for now, assume that only nodes and edges are affected.
        old_dtype=np.dtype( self.cell_dtype ).descr

        new_cell_dtype=[]
        for typeinfo in old_dtype:
            name=typeinfo[0]
            vtype=typeinfo[1]
            if len(typeinfo)>2:
                shape=typeinfo[2]
            else:
                shape=None

            if name in ['edges','nodes']:
                new_cell_dtype.append( (name,vtype,self.max_sides) )
            else:
                new_cell_dtype.append( typeinfo ) # just copy

        self.cell_dtype=new_cell_dtype
        new_cells=np.zeros(self.Ncells(),new_cell_dtype)

        for typeinfo in old_dtype:
            name=typeinfo[0]
            if name in ['edges','nodes']:
                if old_max_sides > self.max_sides:
                    new_cells[name][:,:] = self.cells[name][:,:self.max_sides]
                else:
                    new_cells[name][:,:old_max_sides] = self.cells[name][:,:]
                    new_cells[name][:,old_max_sides:]=self.UNDEFINED
            else:
                new_cells[name]=self.cells[name]

        self.cells=new_cells
        
    @staticmethod
    def from_trigrid(g):
        return UnstructuredGrid(edges=g.edges,points=g.points,cells=g.cells)

    @staticmethod
    def from_ugrid(nc,mesh_name=None,skip_edges=False,fields='auto'):
        """ extract 2D grid from netcdf/ugrid
        nc: either a filename or an xarray dataset.
         THIS IS NEW -- old code used a QDataset, but that is being phased out.
        fields: 'auto' [new] populate additional node,edge and cell fields
        based on matching dimensions.
        """
        if isinstance(nc,str):
            # nc=qnc.QDataset(nc)
            nc=xr.open_dataset(nc)

        if mesh_name is None:
            meshes=[]
            for vname in nc.variables.keys():
                if nc[vname].attrs.get('cf_role',None) == 'mesh_topology':
                    meshes.append(vname)
            assert len(meshes)==1
            mesh_name=meshes[0]

        mesh = nc[mesh_name]

        node_x_name,node_y_name = mesh.node_coordinates.split()

        node_x=nc[node_x_name]
        node_y=nc[node_y_name]
        try:
            # xarray access is slow - pull complete arrays beforehand
            node_x=node_x.values
            node_y=node_y.values
        except AttributeError:
            # unless it's a regular netcdf dataset:
            node_x=node_x[:]
            node_y=node_y[:]
        node_xy = np.array([node_x,node_y]).T
        def process_as_index(varname):
            ncvar=nc[varname]
            try:
                start_index=ncvar.start_index
            except AttributeError:
                start_index=0
            idxs=ncvar[...] - start_index
            # force the move to numpy land
            idxs=np.asanyarray(idxs)
            
            # but might still be masked -- change to unmasked,
            # but our own choice of the invalid value:
            try:
                if np.any( idxs.mask ):
                    idxs=idxs.filled(UnstructuredGrid.UNDEFINED)
            except AttributeError:
                pass
            # ideally this wouldn't be needed, but as an index, really
            # any negative value is bad, and more likely to signal that
            # masks or MissingValue attributes were lost, so better to
            # fix those now
            # so be proactive about various ways of undefined nodes coming in:
            idxs[np.isnan(idxs)]=UnstructuredGrid.UNDEFINED
            idxs[idxs<0]=UnstructuredGrid.UNDEFINED
            return idxs

        faces = process_as_index(mesh.face_node_connectivity)
        edges = process_as_index(mesh.edge_node_connectivity) # [N,2]

        ug = UnstructuredGrid(points=node_xy,cells=faces,edges=edges)

        if fields=='auto':
            # doing this after the fact is inefficient, but a useful
            # simplification during development
            for dim_attr,struct,adder in [('node_dimension',ug.nodes,ug.add_node_field),
                                          ('edge_dimension',ug.edges,ug.add_edge_field),
                                          ('face_dimension',ug.cells,ug.add_cell_field)]:
                dim_name=mesh.attrs.get(dim_attr,None)
                if dim_name:
                    for vname in nc.data_vars:
                        # At this point, only scalar values
                        if nc[vname].dims==(dim_name,):
                            if vname in struct.dtype.names:
                                # already exists, just copy
                                struct[vname]=nc[vname].values
                            else:
                                adder( vname, nc[vname].values )
                
        return ug

    def write_to_xarray(self,ds=None,mesh_name='mesh'):
        """ write grid definition, ugrid-ish, to a new xarray dataset
        """
        import xarray as xr
        if ds is None:
            ds=xr.Dataset()

        ds[mesh_name]=1
        ds[mesh_name].attrs['cf_role']='mesh_topology'
        ds[mesh_name].attrs['node_coordinates']='node_x node_y'
        ds[mesh_name].attrs['face_node_connectivity']='face_node'
        ds[mesh_name].attrs['edge_node_connectivity']='edge_node'
        ds[mesh_name].attrs['face_dimension']='face'
        ds[mesh_name].attrs['edge_dimension']='edge'

        ds['node_x']= ( ('node',),self.nodes['x'][:,0])
        ds['node_y']= ( ('node',),self.nodes['x'][:,1])

        ds['face_node']= ( ('face','maxnode_per_face'), self.cells['nodes'] )

        ds['edge_node']= ( ('edge','node_per_edge'), self.edges['nodes'] )

        return ds

    def write_ugrid(self,
                    fn,
                    mesh_name='mesh',
                    fields='auto',
                    overwrite=False):
        """ 
        rough ugrid writing - doesn't set the full complement of
        attributes (missing_value, edge-face connectivity, others...)
        really just a starting point.
        """
        if os.path.exists(fn):
            if overwrite:
                os.unlink(fn)
            else:
                raise GridException("File %s exists"%(fn))

        if 1: # xarray-based code
            ds=xr.Dataset()
            ds[mesh_name]=1

            mesh_var=ds[mesh_name]
            mesh_var.attrs['cf_role']='mesh_topology'
            mesh_var.attrs['node_coordinates']='node_x node_y'
            mesh_var.attrs['face_node_connectivity']='face_node'
            mesh_var.attrs['edge_node_connectivity']='edge_node'
            mesh_var.attrs['node_dimension']='node'
            mesh_var.attrs['edge_dimension']='edge'
            mesh_var.attrs['face_dimension']='face'
            
            ds['node_x'] = ('node',),self.nodes['x'][:,0]
            ds['node_y'] = ('node',),self.nodes['x'][:,1]

            ds['face_node'] = ('face','maxnode_per_face'),self.cells['nodes']

            ds['edge_node']=('edge','node_per_edge'),self.edges['nodes']

            if fields=='auto':
                for src_data,dim_name in [ (self.cells,'face'),
                                           (self.edges,'edge'),
                                           (self.nodes,'node') ]:
                    for field in src_data.dtype.names:
                        if field.startswith('_'):
                            continue
                        if field in ['cells','nodes','edges','deleted']:
                            continue # already included
                        if src_data[field].ndim != 1:
                            continue # not smart enough for that yet
                        if field in ds:
                            out_field = dim_name + "_" + field
                        else:
                            out_field=field
                            
                        ds[out_field] = (dim_name,),src_data[field]
            ds.to_netcdf(fn)
            
        if 0: # old qnc-based code
            nc=qnc.empty(fn)

            nc[mesh_name]=1
            mesh_var=nc.variables[mesh_name]
            mesh_var.cf_role='mesh_topology'

            mesh_var.node_coordinates='node_x node_y'
            nc['node_x']['node']=self.nodes['x'][:,0]
            nc['node_y']['node']=self.nodes['x'][:,1]

            mesh_var.face_node_connectivity='face_node'
            nc['face_node']['face','maxnode_per_face']=self.cells['nodes']

            mesh_var.edge_node_connectivity='edge_node'
            nc['edge_node']['edge','node_per_edge']=self.edges['nodes']

            nc.close()

    @staticmethod
    def from_shp(shp_fn):
        # bit of extra work to find the number of nodes required
        feats=wkb2shp.shp2geom(shp_fn)['geom']
        if feats[0].type=='Polygon':
            nsides=[len(geom.exterior.coords)
                    for geom in wkb2shp.shp2geom(shp_fn)['geom']]
            nsides=np.max(nsides)
        else:
            nsides=10 # total punt

        g=UnstructuredGrid(max_sides=nsides)
        g.add_from_shp(shp_fn)
        return g
    def add_from_shp(self,shp_fn):
        """ Add features in the given shapefile to this grid.
        Limited support: only polygons, must conform to self.max_sides,
        and caller is responsible for adding the edges. (i.e. make_edges_from_cells)
        updated: now uses add_cell_and_edges(), so edges should exist from the start.
        """
        geoms=wkb2shp.shp2geom(shp_fn)
        for geo in geoms['geom']:
            if geo.type =='Polygon':
                coords=np.array(geo.exterior)
                if np.all(coords[-1] ==coords[0] ):
                    coords=coords[:-1]

                # also check for ordering - force CCW.
                if signed_area(coords)<0:
                    coords=coords[::-1]
                    
                # used to always return a new node - bad!
                nodes=[self.add_or_find_node(x=x)
                       for x in coords]
                # this used to be just add_cell(), but new logic in add_cell()
                # really needs edges to exist first.
                self.add_cell_and_edges(nodes=nodes)
            elif geo.type=='LineString':
                coords=np.array(geo)
                nodes=[self.add_or_find_node(x=x)
                       for x in coords]
                for a,b in zip(nodes[:-1],nodes[1:]):
                    self.add_edge(nodes=[a,b])
            else:
                raise GridException("Not ready for geometry type %s"%geo.type)
        # still need to collapse duplicate nodes
    def from_simple_data(self,points=[],edges=[],cells=[]):
        """
        Convenience method for taking basic descriptions of geometry and
        putting into proper structure arrays.
        
        points: either None, or an [N,2] array of xy pairs.
        edges: None, or [N,2] array of 0-based indexes into points
        cells: None, or [N,max_sides] array of 0-based indexes into nodes

        Note: this does not handle any inference of additional topology - it won't
         figure out which edges make up a cell, for instance (call update_cell_edges()
         afterwards for that)
        """

        if isinstance(points,int):
            self.nodes = np.zeros( points, self.node_dtype)
        else:
            self.nodes = np.zeros( len(points), self.node_dtype)
            if len(points):
                points = np.asarray(points)
                self.nodes['x'] = points[:,:2]  # discard any z's that come in

        if isinstance(cells,int):
            self.cells = np.zeros(cells,self.cell_dtype)
        else:
            self.cells = np.zeros( len(cells), self.cell_dtype)
            if len(cells):
                cells = np.asarray(cells)
                self.cells['nodes'][...] = -1
                # allow for initializing the grid with a cell
                # array with fewer than maxsides
                Nnode_in = cells.shape[1]
                self.cells['nodes'][:,:Nnode_in] = cells
        self.cells['_center'] = np.nan # signal stale
        self.cells['_area'] = np.nan   # signal stale
        self.cells['edges'] = self.UNKNOWN # and no data here

        if isinstance(edges,int):
            self.edges = np.zeros(edges,self.edge_dtype)
        else:
            self.edges = np.zeros(len(edges),self.edge_dtype)
            if len(edges):
                edges = np.asarray(edges)
                # incoming edges may just have connectivity
                if edges.shape[1] == 2:
                    self.edges['nodes'] = edges

                    self.edges['mark'] = self.UNKNOWN
                    self.edges['cells'][:,0] = self.UNKNOWN
                    self.edges['cells'][:,1] = self.UNKNOWN
                elif edges.shape[1] == 5:
                    self.edges['nodes'] = edges[:,:2]
                    self.edges['mark'] = edges[:,2]
                    self.edges['cells'] = edges[:,3:5]
                else:
                    raise GridException("Edges should have 2 or 5 entries per edge")
                
        self.refresh_metadata()

    def update_cell_edges(self):
        """ from edges['nodes'] and cells['nodes'], set cells['edges']
        """
        self.cells['edges'] = -1
        for c in range(self.Ncells()):
            for i,(a,b) in enumerate(circular_pairs(self.cell_to_nodes(c))):
                self.cells['edges'][c,i] = self.nodes_to_edge(a,b)

    def update_cell_nodes(self):
        """ from edges['nodes'] and cells['edges'], set cells['nodes']
        """
        self.cells['nodes'] = -1

        for c in range(self.Ncells()):
            # consider two edges at a time, and find the common node
            for i,(ja,jb) in enumerate(circular_pairs(self.cell_to_edges(c))):
                for n in self.edges['nodes'][ja,:]: 
                    if n in self.edges['nodes'][jb]:
                        self.cells['nodes'][c,i] = n
                        break

    def add_edge_field(self,name,data,on_exists='fail'):
        """
        modifies edge_dtype to include a new field given by name,
        initialize with data.  NB this requires copying the edges
        array - not fast!
        """
        if name in np.dtype(self.edge_dtype).names:
            if on_exists == 'fail':
                raise GridException("Edge field %s already exists"%name)
            elif on_exists == 'pass':
                return
            elif on_exists == 'overwrite':
                self.edges[name] = data
        else:
            self.edges=recarray_add_fields(self.edges,
                                           [(name,data)])
            self.edge_dtype=self.edges.dtype
    def delete_edge_field(self,*names):
        self.edges=recarray_del_fields(self.edges,names)
        self.edge_dtype=self.edges.dtype

    def add_node_field(self,name,data,on_exists='fail'):
        """ add a new field to nodes, amend node_dtype """
        if name in np.dtype(self.node_dtype).names:
            if on_exists == 'fail':
                raise GridException("Node field %s already exists"%name)
            elif on_exists == 'pass':
                return
            elif on_exists == 'overwrite':
                self.nodes[name] = data
        else:
            self.nodes=recarray_add_fields(self.nodes,
                                           [(name,data)])
            self.node_dtype=self.nodes.dtype
    def delete_node_field(self,*names):
        self.nodes=recarray_del_fields(self.nodes,names)
        self.node_dtype=self.nodes.dtype
        
    def add_cell_field(self,name,data,on_exists='fail'):
        """
        modifies cell_dtype to include a new field given by name,
        initialize with data.  NB this requires copying the cells
        array - not fast!
        """
        # will need to get fancier to discern vector dtypes
        # assert data.ndim==1  - maybe no need to be smart?
        if name in np.dtype(self.cell_dtype).names:
            if on_exists == 'fail':
                raise GridException("Node field %s already exists"%name)
            elif on_exists == 'pass':
                return
            elif on_exists == 'overwrite':
                self.cells[name] = data
        else:
            self.cells=recarray_add_fields(self.cells,
                                           [(name,data)])
            self.cell_dtype=self.cells.dtype
    def delete_cell_field(self,*names):
        self.cells=recarray_del_fields(self.cells,names)
        self.cell_dtype=self.cells.dtype
        
    def renumber(self):
        self.renumber_nodes()
        self.renumber_edges()
        self.renumber_cells()

    def renumber_nodes_ordering(self):
        return np.argsort(self.nodes['deleted'],kind='mergesort')
    
    def renumber_nodes(self,order=None):
        if order is None:
            nsort=self.renumber_nodes_ordering()
        else:
            nsort=order
        Nactive = np.sum(~self.nodes['deleted'])

        node_map = np.zeros(self.Nnodes()+1) # do this before truncating nodes
        self.nodes = self.nodes[nsort][:Nactive]
        
        node_map[:]=-999
        # and this after truncating nodes:
        # This is causing a problem, where Nnodes is smaller than the size of nsort
        # Should be fixed by taking only active portion of nsort
        node_map[:-1][nsort[:Nactive]] = np.arange(self.Nnodes())
        node_map[-1] = -1 # missing nodes map -1 to -1

        self.edges['nodes'] = node_map[self.edges['nodes']]
        self.cells['nodes'] = node_map[self.cells['nodes']]

        self._node_to_edges = None
        self._node_to_cells = None
        self._node_index = None

    def delete_orphan_nodes(self):
        """ Scan for nodes not part of an edge, and delete them.
        """
        used=np.zeros( self.Nnodes(),'b1')
        valid_cells=~self.cells['deleted']
        valid_nodes=self.cells['nodes'][valid_cells,:].ravel()
        valid_nodes=valid_nodes[ valid_nodes>=0 ]
        used[ valid_nodes ]=True

        valid_edges=~self.edges['deleted']
        valid_nodes=self.edges['nodes'][valid_edges,:].ravel()
        used[ valid_nodes ]=True
        
        self.log.info("%d nodes found to be orphans"%np.sum(~used))

        for n in np.nonzero(~used)[0]:
            self.delete_node(n)

    def merge_duplicate_nodes(self):
        """ Match up nodes based on *exact* coordinates.
        When two nodes share the same coordinates, attempt
        to merge them.  This can get dicey!

        Returns a dict mapping the deleted nodes with the node they
        were merged with.
        """
        merges={}
        xys={}
        for n in self.valid_node_iter():
            k=tuple(self.nodes['x'][n])
            if k in xys:
                merges[n]=xys[k]
                self.merge_nodes(xys[k],n)
            else:
                xys[k]=n
        return merges

    def renumber_cells_ordering(self): 
        """ return cell indices in the order they should appear, and
        with deleted cells not appearing at all.
        """
        Nactive = sum(~self.cells['deleted'])
        return np.argsort( self.cells['deleted'],kind='mergesort')[:Nactive]
        
    def renumber_cells(self,order=None):
        if order is None:
            csort = self.renumber_cells_ordering()
        else:
            csort= order
        Nneg=-min(-1,self.edges['cells'].min())
        cell_map = np.zeros(self.Ncells()+Nneg) # do this before truncating cells
        self.cells = self.cells[csort]
        
        # and remap ids:
        # if csort[a] = b, then the a'th cell of the new grid is the b'th cell of the old grid
        # what I want:
        # cell_map[b] = a
        # or cell_map[csort[a]] = a
        # for all a, so
        # cell_map[csort[arange(Ncells)]] = arange(Ncells)
        cell_map[:] = -999 # these should only remain for deleted cells, and never show up in the output
        cell_map[:-Nneg][csort] = np.arange(self.Ncells())
        # cell_map[-1] = -1 # land edges map cell -1 to -1
        # allow broader range of negatives:
        # map cell -1 to -1, -2 to -2, etc.
        cell_map[-Nneg:] = np.arange(-Nneg,0) 

        self.edges['cells'] = cell_map[self.edges['cells']]
        self._cell_center_index=None

    def renumber_edges_ordering(self):
        Nactive = sum(~self.edges['deleted'])
        return np.argsort( self.edges['deleted'],kind='mergesort')[:Nactive]
        
    def renumber_edges(self,order=None):
        if order is None:
            esort=self.renumber_edges_ordering()
        else:
            esort=order
            
        # edges take a little extra work, for handling -1 missing edges
        # Follows same logic as for cells
        Nneg=-min(-1,self.cells['edges'].min())
        edge_map = np.zeros(self.Nedges()+Nneg) # do this before truncating
        self.edges = self.edges[esort]

        edge_map[:] = -999 # these should only remain for deleted edges, which won't show up in the output
        edge_map[:-Nneg][esort] = np.arange(self.Nedges()) # and this after truncating
        #edge_map[-1] = -1 # triangles have a -1 -> -1 edge mapping
        edge_map[-Nneg:] = np.arange(-Nneg,0)

        self.cells['edges'] = edge_map[self.cells['edges']]

    def add_grid(self,ugB,merge_nodes=None):
        """
        Add the nodes, edges, and cells from another grid to this grid.
        Copies fields with common names, any other fields are dropped from ugB.
        Assumes (for the moment) that max_sides is compatible. 

        merge_nodes: [ (self_node,ugB_node), ... ]
          Nodes which overlap and will be mapped instead of added.
        """
        node_map=np.zeros( ugB.Nnodes(), 'i4')-1
        edge_map=np.zeros( ugB.Nedges(), 'i4')-1
        cell_map=np.zeros( ugB.Ncells(), 'i4')-1

        if merge_nodes is not None:
            for my_node,B_node in merge_nodes:
                node_map[B_node]=my_node

        def bad_fields(Adata,Bdata): # field froms B which get dropped
            A_fields =Adata.dtype.names
            B_fields =Bdata.dtype.names

            B_bad=[f for f in B_fields
                   if (f=='deleted') or (f not in A_fields)]
            return B_bad

        B_bad=bad_fields(self.nodes,ugB.nodes)

        for n in ugB.valid_node_iter():
            if node_map[n]>=0:
                continue # must be part of merge_nodes
            kwargs=rec_to_dict(ugB.nodes[n])
            for f in B_bad:
                del kwargs[f]

            node_map[n]=self.add_node(**kwargs)

        B_bad=bad_fields(self.edges,ugB.edges)
        # Easier to let add_cell fix this up
        B_bad.append('cells')

        for n in ugB.valid_edge_iter():
            kwargs=rec_to_dict(ugB.edges[n])
            for f in B_bad:
                del kwargs[f]

            kwargs['nodes']=node_map[kwargs['nodes']]

            # when merge_nodes is specified, have to also check
            # for preexisting edges
            if merge_nodes is not None:
                j=self.nodes_to_edge(kwargs['nodes'])
                if j is not None:
                    edge_map[n]=j
                    continue
            edge_map[n]=self.add_edge(**kwargs)

        B_bad=bad_fields(self.cells,ugB.cells)

        for n in ugB.valid_cell_iter():
            kwargs=rec_to_dict(ugB.cells[n])
            for f in B_bad:
                del kwargs[f]

            # avoid mutating ugB.
            orig_nodes=kwargs['nodes']
            kwargs['nodes'] = orig_nodes.copy()
            kwargs['edges'] = kwargs['edges'].copy()

            for i,node in enumerate(kwargs['nodes']):
                if node>=0:
                    kwargs['nodes'][i]=node_map[node]

            # less common, but still need to check for duplicated cells
            # when merge_nodes is used.
            if merge_nodes is not None:
                c=self.nodes_to_cell( kwargs['nodes'], fail_hard=False)
                if c is not None:
                    cell_map[n]=c
                    print("Skipping existing cell: %d: %s => %d: %s"%( n,str(orig_nodes),
                                                                       c,str(kwargs['nodes'])))
                    continue

            for i,edge in enumerate(kwargs['edges']):
                if edge>=0:
                    kwargs['edges'][i]=edge_map[edge]

            cell_map[n]=self.add_cell(**kwargs)

        return node_map,edge_map,cell_map

    def boundary_cycle(self):
        # find a point outside the domain:
        x0=self.nodes['x'].min(axis=0)-1.0
        # grab nearby edges
        edges_near=self.select_edges_nearest(x0,count=6)
        max_nodes=self.Nnodes()
        potential_cells=self.find_cycles(max_cycle_len=max_nodes,
                                         starting_edges=edges_near,
                                         check_area=False)
        best=(0.0,None) # Area, nodes
        for pc in potential_cells:
            segs=self.nodes['x'][pc]
            A=signed_area(segs)
            
            # Looking for the most negative area
            if A<best[0]:
                best=(A,pc)
        return np.array(best[1][::-1]) # reverse, to return a CCW, positive area string.
        
    def find_cycles(self,max_cycle_len=4,starting_edges=None,check_area=True):
        """ traverse edges, returning a list of lists, each list giving the
        CCW-ordered node indices which make up a 'facet' or cycle in the graph
        (i.e. potentially a cell).
        starting_edges: iterable of edge indices from which to start the cycle
          traversal.  
        check_area: if True, make sure that any returned cycles form a polygon
         with positive area.   This can be an issue if the outer ring of the grid is
         short enough to be a cycle itself.
        """
        def traverse(a,b):
            cs=self.angle_sort_adjacent_nodes(b,ref_nbr=a)
            return b,cs[-1]

        visited=set() # directed tuple of nodes

        cycles=[]

        if starting_edges is None:
            starting_edges=self.valid_edge_iter()

        for j in starting_edges:
            if j % 10000==0:
                print("Edge %d/%d, %d cycles"%(j,self.Nedges(),len(cycles)))
            # iterate over the two half-edges
            for A,B in (self.edges['nodes'][j], self.edges['nodes'][j,::-1]):
                cycle=[A]

                while (A,B) not in visited and len(cycle)<max_cycle_len:
                    visited.add( (A,B) )
                    cycle.append(B)
                    A,B = traverse(A,B)
                    if B==cycle[0]:
                        if check_area:
                            A=signed_area( self.nodes['x'][cycle] )
                            if A>0:
                                cycles.append(cycle)
                        else:
                            cycles.append(cycle)
                        break
        return cycles
    def make_cells_from_edges(self,max_sides=None):
        max_sides=max_sides or self.max_sides
        assert max_sides<=self.max_sides
        cycles=self.find_cycles(max_cycle_len=max_sides)
        ncells=len(cycles)
        if ncells:
            self.cells = np.zeros( ncells, self.cell_dtype)
            self.cells['nodes'][...] = -1
            self.cells['_center'] = np.nan # signal stale
            self.cells['_area'] = np.nan   # signal stale
            self.cells['edges'] = self.UNKNOWN # and no data here
            for i,cycle in enumerate(cycles):
                self.cells['nodes'][i,:len(cycle)]=cycle

    def interp_cell_to_node(self,cval):
        result=np.zeros(self.Nnodes(),cval.dtype)
        for n in range(self.Nnodes()):
            result[n]=cval[self.node_to_cells(n)].mean()
        return result
    def interp_node_to_cell(self,nval):
        """
        nval: scalar value for each node.
        returns an array with scalar value for each cell, taken from
              average of nodes.
        """
        nodes=self.cells['nodes']
        weights=(nodes>=0).astype(np.int32)
        # a little extra work to avoid nan contamination from undefined
        # nodes.  nans for defined nodes still contaminate
        vertex_vals=(nval[nodes]*weights)
        vertex_vals[weights==0]=0
        cvals=vertex_vals.sum(axis=1)/weights.sum(axis=1)
        return cvals

    def cells_to_edge(self,a,b):
        j1=self.cell_to_edges(a)
        j2=self.cell_to_edges(b)
        for j in j1:
            if j in j2:
                return j
        return None
    
    def edge_to_cells(self,e=slice(None),recalc=False):
        """
        recalc: if True, forces recalculation of all edges.

        e: limit output to a single edge, a slice or a bitmask
          may also limit updates to the requested edges
        """
        # try to be a little clever about recalculating - 
        # it can be very slow to check all edges for UNKNOWN
        
        if recalc:
            self.edges['cells'][:,:]=self.UNMESHED
            self.log.info("Recalculating edge to cells" )
            all_c=range(self.Ncells())
        else:
            if e is None:
                e=slice(None)
            # on-demand approach
            if isinstance(e,slice):
                js=np.arange(e.start or 0,
                             e.stop or self.Nedges(),
                             e.step or 1)
            else:
                try:
                    L=len(e)
                    if L==self.Nedges() and np.issubdtype(np.bool,e.dtype):
                        js=np.nonzero(e)
                    else:
                        js=e
                except TypeError:
                    js=[e]

            all_c=set()
            for j in js:
                ec=self.edges['cells'][j]
                if (ec[0]!=self.UNKNOWN) and (ec[1]!=self.UNKNOWN):
                    continue

                for n in self.edges['nodes'][j]:
                    # don't assume that cells['edges'] is set, either.
                    for c in self.node_to_cells(n):
                        all_c.add(c)

        # Do the actual work
        # don't assume that cells['edges'] is set, either.
        for c in all_c:
            nodes=self.cell_to_nodes(c)
            for i in range(len(nodes)):
                a=nodes[i]
                b=nodes[(i+1)%len(nodes)]
                j=self.nodes_to_edge(a,b)
                if j is None:
                    print( "Failed to find an edge" )
                    continue
                # will have to trial-and-error to get the right
                # left/right sense here.
                if self.edges['nodes'][j,0]==a:
                    self.edges['cells'][j,0]=c
                else:
                    self.edges['cells'][j,1]=c

        return self.edges['cells'][e]

    def make_cell_nodes_from_edge_nodes(self):
        """ some formats (old UnTRIM...) list edges that make up cells, but not
        the nodes.  This method uses cells['edges'] and edges['nodes'] to populate
        cells['nodes']
        """
        for c in self.valid_cell_iter():
            # be sure to ask for unordered, since ordered requires nodes to be 
            # present.
            nodes=[]
            js=self.cell_to_edges(c,ordered=False)
            nodes=list( self.edges['nodes'][js[0],:] )

            if nodes[0] in self.edges['nodes'][js[1],:]:
                nodes=nodes[::-1]

            for j in js[1:]:
                ns=self.edges['nodes'][j]
                if ns[0]==nodes[-1]:
                    nodes.append(ns[1])
                else:
                    nodes.append(ns[0])
            assert nodes[-1]==nodes[0]
            nodes=nodes[:-1]
            self.cells['nodes'][c,:len(nodes)]=nodes
            self.cells['nodes'][c,len(nodes):]=self.UNDEFINED

    def make_edges_from_cells(self):
        edge_map = {} # keys are tuples of node indices, mapping to index into new_edges
        new_edges = [] # each entry is [n1,n2,c1,c2].  assume for now that c1 is on the left of the edge n1->n2

        for c in range(self.Ncells()):
            for i,(a,b) in enumerate(circular_pairs(self.cell_to_nodes(c))):
                if a<b:
                    k = (a,b)
                else:
                    k = (b,a)
                if k not in edge_map:
                    j = len(new_edges)
                    edge_map[k] = j
                    # we know everything but the opposite cell
                    new_edges.append([a,b,c,-1])
                else:
                    # if it's already in there, then we must be the right-hand side cell
                    j = edge_map[k]
                    new_edges[j][3] = c
                # not 100% sure that this will put the edges in with the expected order
                # this should have edge i immediately CCW from node i.
                self.cells['edges'][c,i] = j

        new_edges = np.array(new_edges)
        self.edges = np.zeros( len(new_edges),self.edge_dtype )
        self.edges['nodes'] = new_edges[:,:2]
        self.edges['cells'] = new_edges[:,2:4]
        self._node_to_edges=None

    def refresh_metadata(self):
        """ Call this when the cells, edges and nodes may be out of sync with indices
        and the like.  doesn't force a rebuild, just clears out potentially stale information.
        """
        #self.node_index = None
        #self.edge_index = None
        #self._calc_edge_centers = False
        #self._calc_cell_centers = False
        #self._calc_vcenters = False
        self._node_to_edges = None
        self._node_to_cells = None

    def Nnodes(self):
        """
        total number of allocated nodes -- may include deleted nodes
        """
        return len(self.nodes)
    def Ncells(self):
        """
        total number of allocated cells -- may include deleted cells
        """
        return len(self.cells)
    def Nedges(self):
        """
        total number of allocated edges -- may include deleted edges
        """
        return len(self.edges)

    def Ncells_valid(self):
        return np.sum(~self.cells['deleted'])
    def Nedges_valid(self):
        return np.sum(~self.edges['deleted'])
    def Nnodes_valid(self):
        return np.sum(~self.nodes['deleted'])

    def valid_edge_iter(self):
        for j in range(self.Nedges()):
            if ~self.edges['deleted'][j]:
                yield j
    def valid_node_iter(self):
        for n in range(self.Nnodes()):
            if ~self.nodes['deleted'][n]:
                yield n
    def valid_cell_iter(self):
        """ generator for cell indexes which are not deleted.
        it is safe to delete cells during this call, but not
        to create cells.
        """
        for c in range(self.Ncells()):
            if ~self.cells['deleted'][c]:
                yield c

    def cell_Nsides(self,c):
        return np.sum(self.cells['nodes'][c]>=0)

    def edges_center(self):
        centers=np.zeros( (self.Nedges(), 2), 'f8')
        valid=~self.edges['deleted']
        centers[valid,:] = self.nodes['x'][self.edges['nodes'][valid,:]].mean(axis=1)
        centers[~valid,:]=np.nan

        # unsafe for deleted edges referencing deleted/truncated nodes
        # return self.nodes['x'][self.edges['nodes']].mean(axis=1)
        return centers

    def cells_centroid(self,ids=None):
        if ids is None:
            ids=np.arange(self.Ncells())
            
        centroids=np.zeros( (len(ids),2),'f8')*np.nan
        
        for ci,c in enumerate(ids):
            if not self.cells['deleted'][c]:
                centroids[ci]= np.array(self.cell_polygon(c).centroid)
        return centroids

    def cells_centroid_py(self):
        """
        This is not currently any faster than using the above shapely 
        code, but is pasted in here since it may become faster with 
        some tweaking, or be more amenable to numba or cython acceleration
        in the future.
        """
        A=self.cells_area()
        cxy=np.zeros( (self.Ncells(),2), np.float64)

        refs=self.nodes['x'][self.cells['nodes'][:,0]]

        all_pnts=self.nodes['x'][self.cells['nodes']] - refs[:,None,:]

        for c in np.nonzero(~self.cells['deleted'])[0]:
            nodes=self.cell_to_nodes(c)

            i=np.arange(len(nodes))
            ip1=(i+1)%len(nodes)
            nA=all_pnts[c,i]
            nB=all_pnts[c,ip1]

            tmp=(nA[:,0]*nB[:,1] - nB[:,0]*nA[:,1])
            cxy[c,0] = ( (nA[:,0]+nB[:,0])*tmp).sum()
            cxy[c,1] = ( (nA[:,1]+nB[:,1])*tmp).sum()
        cxy /= 6*A[:,None]    
        cxy += refs
        return cxy
    
    def cells_center(self,refresh=False,mode='first3'):
        """ calling this method is preferable to direct access to the
        array, since cell centers can possibly be stale if the grid has been
        modified (though no such checking exists yet).

        For now, circumcenter is calculated only from 1st 3 points, even if it's
        a quad.

        refresh: must be True, False, or something slice-like (slice, bitmap, integer array)
        mode: first3 - estimate circumcenter from the first 3 nodes
        """
        if refresh is True:
            to_update=slice(None)
        elif refresh is not False:
            to_update=refresh
        else:
            to_update = np.isnan(self.cells['_center'][:,0])

        if np.sum(to_update) > 0:
            if mode=='first3':
                p1,p2,p3 = [self.nodes['x'][self.cells['nodes'][to_update,i]] for i in [0,1,2]]
                self.cells['_center'][to_update] = circumcenter(p1,p2,p3)
            elif mode=='sequential':
                for c in np.arange(self.Ncells())[to_update]:
                    points=self.nodes['x'][self.cell_to_nodes(c)]
                    self.cells['_center'][c] = poly_circumcenter(points)
        
        return self.cells['_center']

    def bounds(self,order='xxyy'):
        b=[self.nodes['x'][:,0].min(),
           self.nodes['x'][:,0].max(),
           self.nodes['x'][:,1].min(),
           self.nodes['x'][:,1].max() ]

        if order=='xxyy':
            return b
        else:
            return [b[0],b[2],b[1],b[3]]

    def edges_normals(self,edges=slice(None)):
        # does not assume the grid is orthogonal - normals are found by rotating
        # the tangent vector
        # I think this is positive towards c2, i.e. from left to right looking
        # from n1 to n2

        # starts as the vector from node1 to node2
        # say c1 was on the left, c2 on the right.  so n1 -> n2 is (0,1)
        # then that is changed to (1,0), then (1,-0)
        # so this pointing left to right, and is in fact pointing towards c2.
        # had been axis=1, and [:,0,::-1]
        # but with edges possibly a single index, make it more general
        normals = np.diff(self.nodes['x'][self.edges['nodes'][edges]],axis=-2)[...,0,::-1]
        normals[...,1] *= -1
        normals /= mag(normals)[...,None]
        return normals

    # Variations on access to topology
    _node_to_cells = None
    def node_to_cells(self,n):
        if self._node_to_cells is None:
            self.build_node_to_cells()
        return self._node_to_cells[n]

    def node_to_nodes(self,n):
        """ Return an ndarray of the node indices which share edges with 
        node n.
        """
        js = self.node_to_edges(n)
        all_nodes = self.edges['nodes'][js].ravel()
        # return np.setdiff1d(all_nodes,[n]) # significantly slower than lists
        return np.array( [nbr for nbr in all_nodes if nbr!=n] )

    def angle_sort_adjacent_nodes(self,n,ref_nbr=None):
        """
        return array of node indices for nodes connected to n by an
        edge, sorted CCW by angle.
        ref_nbr: if given, roll the indices so that ref_nbr appears first.
        """
        nbrs=self.node_to_nodes(n)
        if len(nbrs)==0:
            return []
        diffs=self.nodes['x'][nbrs] - self.nodes['x'][n]
        angles=np.arctan2(diffs[:,1],diffs[:,0])
        nbrs=nbrs[np.argsort(angles)]
        if ref_nbr is not None: 
            i=list(nbrs).index(ref_nbr)
            nbrs=np.roll(nbrs,-i)
        return nbrs

    def build_node_to_cells(self):
        n2c = defaultdict(list)
        for c in range(self.Ncells()):
            if self.cells['deleted'][c]:
                continue
            for cn in self.cell_to_nodes(c):
                n2c[cn].append(c)
        self._node_to_cells = n2c
        
    _node_to_edges = None
    def node_to_edges(self,n):
        if self._node_to_edges is None:
            self.build_node_to_edges()
        return self._node_to_edges[n]
    def node_degree(self,n):
        return len(self.node_to_edges(n))

    def edge_to_edges(self,e):
        e_adj=[]
        for n in self.edges['nodes'][e]:
            e_adj += list(self.node_to_edges(n))
        return np.unique(e_adj)

    def build_node_to_edges(self):
        n2e = defaultdict(list)
        for e in self.valid_edge_iter(): 
            for i in [0,1]:
                n2e[self.edges['nodes'][e,i]].append(e)
        self._node_to_edges = n2e
        
    def nodes_to_edge(self,n1,n2=None):
        if n2 is None:
            n1,n2=n1

        candidates1 = self.node_to_edges(n1)
        candidates2 = self.node_to_edges(n2)

        # about twice as fast to loop this way
        for e in candidates1:
            if e in candidates2:
                return e
        return None

        # # this way has nodes_to_edge taking 3x longer than just the node_to_edges call
        # for e in candidates1:
        #     if n2 in self.edges['nodes'][e]:
        #         return e
        # return None

    def nodes_to_cell(self,ns,fail_hard=True):
        cells=self.node_to_cells(ns[0])
        for n in ns[1:]:
            if len(cells)==0:
                break
            cells2=self.node_to_cells(n)
            cells=[c
                   for c in cells
                   if c in cells2]
        if len(cells)==0:
            if fail_hard:
                raise self.GridException("Cell not found")
            else:
                return None
        # shouldn't be possible to get more than one hit.
        assert len(cells)==1

        return cells[0]

    def cell_to_edges(self,c,ordered=False):
        """ returns the indices of edges making up this cell -
        including trimming to the number of sides for this cell.

        ordered: return edges ordered by the node ordering,
            with the first edge connecting the first two nodes
        """
        if ordered:
            e=[]
            # okay to be UNDEFINED, but not UNKNOWN.
            assert np.all(self.cells['nodes'][c]!=self.UNKNOWN)
            nodes=self.cell_to_nodes(c)
            N=len(nodes)

            for na in range(N):
                nb=(na+1)%N
                e.append( self.nodes_to_edge( nodes[na],nodes[nb] ) )
            return np.array(e)
        else:
            e = self.cells['edges'][c]
            if np.any( e==self.UNKNOWN ):
                e=self.cell_to_edges(c,ordered=True)
                self.cells['edges'][c][:len(e)]=e
                self.cells['edges'][c][len(e):]=-1 
            return e[e>=0]

    def cell_to_nodes(self,c):
        """ returns nodes making up this cell, including trimming
        to number of nodes in this cell.
        """
        n = self.cells['nodes'][c]
        return n[n>=0]

    def cell_to_cells(self,c,ordered=False):
        """ Return adjacent cells for c. if ordered, follow suntans convention for
        order of neighbors.  cells[0] has nodes 
        self.cells['nodes'][c,0]
        and 
        self.cells['nodes'][c,1]
        """
        js=self.cell_to_edges(c,ordered=ordered)

        e2c=self.edge_to_cells(js) # make sure it's fresh

        nbrs=[]
        # self.cell_to_edges(c,ordered=ordered):
        for j,(c1,c2) in zip(js,e2c):
            # c1,c2=e2c[j]
            if c1==c:
                nbrs.append(c2)
            else:
                nbrs.append(c1)
        return nbrs

    def is_boundary_cell(self,c):
        """ True if any of this cells edges lie on the boundary
        (i.e. have only one adjacent cell)
        default (and currently only) behavior is that it doesn't 
        pay attention to differences between UNMESHED, UNKNOWN,
        LAND, etc.
        """
        edges=self.cell_to_edges(c)
        return np.any( self.edge_to_cells()[edges] < 0 )
    def is_boundary_edge(self,e):
        return np.any(self.edge_to_cells(e) < 0)

    def cell_to_adjacent_boundary_cells(self,c):
        """
        returns list of cells which are on the boundary, and have an 
        edge adjacent to an edge of c
        """
        j_boundary=[j for j in self.cell_to_edges(c)
                    if self.is_boundary_edge(j)]

        adj_edges=[]
        for j in j_boundary:
            adj_edges+=list(self.edge_to_edges(j))
        adj_edges=filter(lambda jj: self.is_boundary_edge(jj) and jj!=j,adj_edges)
        cells=[c
               for j in adj_edges
               for c in self.edge_to_cells(j) if c>=0 ]
        return cells


    ### Changing topology:
    # probably need to give more thought to the exact semantics here
    # and when operations cascade (like deleting cells neighboring an
    # edge), and when they only perform a single operation.

    # initial design:
    #  delete_<elt>: ignores any dependence.  Marks that element deleted, clears
    #     appropriate fields.
    #  delete_<elt>_cascade: remove dependent entities, too.
    #  add_<elt>: just create that element, nothing sneaky.
    #     add methods take only keyword arguments, corresponding to fields in the dtype.

    def add_or_find_node(self,x,tolerance=0.0,**kwargs):
        """ if a node already exists with a location within tolerance distance 
        of x, return its index, otherwise create a new node.
        """
        hit=self.select_nodes_nearest(x)
        if hit is not None:
            dist = mag( x - self.nodes['x'][hit] )
            if dist<=tolerance:
                return hit
        return self.add_node(x=x,**kwargs)
    
    @listenable
    def add_node(self,**kwargs):
        i=None
        if '_index' in kwargs:
            i=kwargs.pop('_index')
            if i==len(self.nodes):
                i=None # the index we'd get anyway
            else:
                assert i<len(self.nodes)
                assert self.nodes[i]['deleted']
                self.nodes[i]['deleted']=False

        if i is None: # have to extend the array
            n=np.zeros( (), dtype=self.node_dtype)
            self.nodes=array_append(self.nodes,n)
            i=len(self.nodes)-1

        for k,v in six.iteritems(kwargs):
            # oddly, the ordering of the indexing matters
            self.nodes[k][i]=v

        if self._node_index is not None:
            self._node_index.insert(i, self.nodes['x'][i,self.xxyy] )            
        self.push_op(self.unadd_node,i)

        return i

    def unadd_node(self,idx):
        self.delete_node(idx)

    class InvalidEdge(GridException):
        pass

    @listenable
    def add_edge(self,_check_existing=True,**kwargs):
        """
        Does *not* check topology / planarity
        """
        j=None
        if '_index' in kwargs:
            j=kwargs.pop('_index')
            if j==len(self.edges):
                # this is the index we'd get anyway.
                j=None
            else:
                assert len(self.edges)>j
                assert self.edges[j]['deleted']

        if _check_existing:
            j_exists=self.nodes_to_edge(*kwargs['nodes'])
            if j_exists is not None:
                raise GridException("Edge already exists")
                
        if j is None:
            e=np.zeros( (),dtype=self.edge_dtype)
            self.edges=array_append(self.edges,e)
            j=len(self.edges)-1

        # default values
        self.edges[j]['cells'][:]=-1
        self.edges[j]['deleted']=False

        for k,v in six.iteritems(kwargs):
            self.edges[k][j]=v

        # most basic checks on edge validity:
        if self.edges[j]['nodes'][0]==self.edges[j]['nodes'][1]:
            raise self.InvalidEdge('duplicate nodes')

        if self._node_to_edges is not None:
            n1,n2=self.edges['nodes'][j]
            self._node_to_edges[n1].append(j)
            self._node_to_edges[n2].append(j)

        self.push_op(self.unadd_edge,j)
        return j

    def unadd_edge(self,j):
        self.delete_edge(j)

    # the delete_* operations require that there are no dependent
    # entities, while the delete_*_cascade operations will check
    # and remove dependent entitites
    @listenable
    def delete_edge(self,j):
        if np.any(self.edges['cells'][j]>=0):
            raise GridException("Edge %d has cell neighbors"%j)
        self.edges['deleted'][j] = True
        if self._node_to_edges is not None:
            for n in self.edges['nodes'][j]:
                self._node_to_edges[n].remove(j)

        self.push_op(self.undelete_edge,j,self.edges[j].copy())

        # special case for undo:
        if j+1==len(self.edges):
            self.edges=self.edges[:-1]

    def undelete_edge(self,j,edge_data):
        d=rec_to_dict(edge_data)
        d['_index']=j
        d['deleted']=False
        self.add_edge(**d)

    def delete_edge_cascade(self,j):
        # for ci in [0,1]:
        #     c=self.edges['cells'][j,ci]
        #     if c>=0:
        #         self.delete_cell(c)
        #     self.edges['cells'][j,ci]=self.UNKNOWN

        for c in self.edge_to_cells(j):
            if c>=0:
                self.delete_cell(c)
            # this should be handled by the delete_cell() code.
            # self.edges['cells'][j,ci]=self.UNKNOWN
        self.delete_edge(j)


    def merge_edges(self,edges=None,node=None):
        """ Given a pair of edges sharing a node,
        with no adjacent cells or additional edges,
        remove/delete the nodes, combined the edges
        to a single edge, and return the index of the
        resulting edge.
        """
        if edges is None:
            edges=self.node_to_edges(node)
            assert len(edges)==2
        if node is None:
            Na=self.edge_to_nodes(edges[0])
            Nb=self.edge_to_nodes(edges[1])
            for node in Na:
                if node in Nb:
                    break
            else:
                raise self.GridException("Edges %s do not share a node"%(edges))
        A,C=edges
        B=node
        # which side is which?
        if self.edges['nodes'][A,0] == B:
            Ab=0
        else:
            Ab=1
        if self.edges['nodes'][C,0] == B:
            Cb=0
        else:
            Cb=1

        # safety checks - respective sides of the edges should be compatible.
        # left side cells, in the sense of looking from A to C
        assert self.edges['cells'][A,1-Ab] == self.edges['cells'][C,Cb]
        assert self.edges['cells'][A,Ab] == self.edges['cells'][C,1-Cb]

        # cell/edge invariants do not hold for a brief moment
        # this could be a problem if modify_cell tries to update a lookup
        # for edges.  May have to revisit.
        for c in self.edges['cells'][A]:
            if c>=0: # it's a real cell
                c_nodes=[n
                         for n in self.cell_to_nodes(c)
                         if n!=B ]
                self.modify_cell(c,nodes=c_nodes)

        # Edge A will be the one to keep
        # modify_edge knows about changes to nodes
        new_nodes=[ self.edges['nodes'][A,1-Ab],
                    self.edges['nodes'][C,1-Cb] ]
        if Ab==0: # take care to preserve orientation
            new_nodes=new_nodes[::-1]

        self.delete_edge(C)
        # expanding modify_edge into a delete/add allows
        # a ShadowCDT to maintain valid state
        # self.modify_edge(A,nodes=new_nodes)
        # be careful to copy A's entries, as they will get overwritten
        # during the delete/add process.
        edge_data=rec_to_dict(self.edges[A].copy())

        self.delete_edge(A)
        self.delete_node(B)
        edge_data['nodes']=new_nodes
        self.add_edge(_index=A,**edge_data)
        return A
    def split_edge(self,j,**node_args):
        """
        The opposite of merge_edges, take an existing edge and insert
        a node into the middle of it.
        Does not allow for any cells to be involved.
        Defaults to midpoint
        """
        nA,nC=self.edges['nodes'][j]
        assert np.all( self.edge_to_cells(j) < 0 )

        edge_data=rec_to_dict(self.edges[j].copy())
        
        self.delete_edge(j)

        # choose midpoint as default
        loc_args=dict(x= 0.5*(self.nodes['x'][nA] + self.nodes['x'][nC]))
        loc_args.update(node_args) # but args will overwrite

        nB=self.add_node(**loc_args)
        edge_data['nodes'][1]=nB
        self.add_edge(_index=j,**edge_data)
        # this way we get the same cell marks, too.
        # this helps in tracking marks like UNDEFINED vs. UNPAVED
        edge_data['nodes']=[nB,nC]
        jnew=self.add_edge(**edge_data)
        return jnew,nB

    def merge_nodes(self,n0,n1):
        """ 
        Merge or fuse two nodes.  Attempts to merge associated
        topology from n1 to n0, i.e. edges and cells.

        The fields of n1 are lost, and n1 is deleted.

        Functionality here is in progress - not all cases for merging nodes
        are handled.  In particular, it will fail if there is an edge between
        n0 and n1, or if a single cell contains both n0 and n1.
        """
        # -- Sanity checks - does not yet allow for collapsing edges.

        # if they share any cells, would update the cells, but for now
        # just signal failure.
        n0_cells=list(self.node_to_cells(n0))
        n1_cells=list(self.node_to_cells(n1))
        cell_to_edge_cache={}

        for c in n1_cells:
            if c in n0_cells:
                print("cell %d common to both nodes"%c)
                raise GridException("Not ready for merging nodes in the same cell")
                # otherwise record and fix up below

            # while we're looping, cache the edges as they will
            # be mutated along the way.
            cell_to_edge_cache[c]=self.cell_to_edges(c).copy()

        # do they share an edge, but not already fixed in the above stanza?
        j=self.nodes_to_edge(n0,n1)
        if j is not None:
            raise GridException("Not ready for merging endpoints of an edge")

        edge_map={} # index of superceded edge => superceding edge

        # Update edges of n1 to point to n0
        # if that would cause a duplicate edge, then the n1 version is deleted
        n1_edges=list(self.node_to_edges(n1)) # make copy since we'll mutate it
        for j in n1_edges:
            if self.edges['nodes'][j,0]==n1:
                nj=0
            elif self.edges['nodes'][j,1]==n1:
                nj=1
            else:
                assert False # sanity check
            newnodes=self.edges[j]['nodes'].copy()
            newnodes[nj]=n0
            # it's possible that this is an edge which already exists
            jother=self.nodes_to_edge(*newnodes)
            if jother is not None:
                # want to keep jother, delete j.  but is there info on
                # cells which should be brought over?
                edge_map[j]=jother
                # wait to delete j until after cells have been moved to jother.
            else:
                self.log.debug("Modifying edge j=%d"%j)
                self.modify_edge(j,nodes=newnodes)

        # -- Transition any cells.  
        for c in n1_cells:
            # update the node list:
            cnodes=self.cell_to_nodes(c).copy()
            nc=list(cnodes).index(n1)
            cnodes[nc]=n0

            # Dangerous to use cell_to_edges, since it may
            # have to consult the edge topology, which is disrupted
            # in the above code. 
            # cell_to_edges: first checks cells['edges'], may 
            # go to cell_to_nodes(c): that's safe.
            # and   nodes_to_edge
            #     -> node_to_edges, which in turn may consult self.edges['nodes']

            #cedges=self.cell_to_edges(c).copy()
            cedges=cell_to_edge_cache[c]

            for ji,j in enumerate(cedges):
                if j in edge_map:
                    # is there were edges['cells'] should be updated?

                    # sever the edge=>cell pointer, to p
                    # could just set to [-1,-1], but this keeps things very explicit
                    # for debugging
                    j_cells=list(self.edges['cells'][j])
                    j_cells_side=j_cells.index(c)
                    j_cells[ j_cells_side ] = -1
                    self.modify_edge(j,cells=j_cells)

                    # and modify the receiving edge, too
                    jo=edge_map[j]
                    jo_cells=list(self.edges['cells'][jo])
                    # which side of jo?  a bit tedious...
                    if list(self.edges['nodes'][j]).index(n1) == list(self.edges['nodes'][jo]).index(n0):
                        # same orientation
                        jo_cells_side=j_cells_side
                    elif list( self.edges['nodes'][j]).index(n1) == 1-list(self.edges['nodes'][jo]).index(n0):
                        jo_cells_side=1-j_cells_side
                    else:
                        raise Exception("Failed in some tedium")
                    assert jo_cells[jo_cells_side]<0
                    jo_cells[jo_cells_side]=c
                    self.modify_edge(edge_map[j],cells=jo_cells)
                    # yikes.  any chance that worked?

                    cedges[ji]=edge_map[j]

            # maybe this is where we'd update cells['edges'] too?
            self.modify_cell(c,nodes=cnodes,edges=cedges)

        for dead_edge in edge_map:
            self.delete_edge(dead_edge)

        self.delete_node(n1)
        
    @listenable
    def delete_node(self,n):
        """ opportunistic error checking - if the _node_to_* hashes are
        around make sure they reflect that no dependent entities are around,
        but don't build them just to check. The "contract" here does *not*
        promise to check!
        """
        if self._node_to_edges is not None:
            if len(self._node_to_edges[n])>0:
                print( "Node %d has edges: %s"%(n,self._node_to_edges[n]) )
                raise GridException("Node still has edges referring to it")
            del self._node_to_edges[n]
        if self._node_to_cells is not None:
            if len(self._node_to_cells[n])>0:
                raise GridException("Node still has cells referring to it")
            del self._node_to_cells[n]
        if self._node_index is not None:
            self._node_index.delete(n, self.nodes['x'][n,self.xxyy] )

        self.push_op(self.undelete_node,n,self.nodes[n].copy())

        self.nodes['deleted'][n] = True
        
        # special case, used for undo, reverts to previous state
        # more completely.
        if len(self.nodes)==n+1:
            self.nodes=self.nodes[:-1]

    def undelete_node(self,n,node_data):
        d=rec_to_dict(node_data)
        d['_index']=n
        self.add_node(**d)
        

    def delete_node_cascade(self,n):
        """ delete any edges related to this node, then delete this node.
        """
        # list will get mutated - copy preemptively
        for j in list(self.node_to_edges(n)):
            self.delete_edge_cascade(j)
        self.delete_node(n)

    @listenable
    def delete_cell(self,i,check_active=True):
        if check_active and self.cells['deleted'][i]!=False:
            raise Exception("delete_cell(%d) - appears already deleted"%(i))

        # better to go ahead and use the dynamic updates
        # must come before too many modifications, in case we end
        # up recalculating cell center or truncating self.cells
        if self._cell_center_index is not None:
            if self.cell_center_index_point=='circumcenter':
                pnt=self.cells_center()[i]
            else: # centroid
                pnt=self.cells_centroid([i])[0]
            self._cell_center_index.delete(i,pnt[self.xxyy])                

        # remove links from edges:
        for j in self.cell_to_edges(i): # self.cells['edges'][i]:
            if j>=0:
                # hmm - how would j be negative?
                for lr in [0,1]:
                    if self.edges['cells'][j,lr]==i:
                        self.edges['cells'][j,lr]=self.UNMESHED
                        break
                    
        if self._node_to_cells is not None:
            for n in self.cell_to_nodes(i):
                self._node_to_cells[n].remove(i)
            
        self.push_op(self.undelete_cell,i,self.cells[i].copy())

        self.cells['deleted'][i]=True

        
        # special case for undo:
        if i+1==len(self.cells):
            self.cells=self.cells[:-1]

    
    def undelete_cell(self,i,cell_data):
        d=rec_to_dict(cell_data)
        d['_index']=i
        self.add_cell(**d)

    def toggle_cell_at_point(self,x,**kw):
        """ if x is inside a cell, delete it.
        if x is inside a potential cell, create it.
        """
        c=self.delete_cell_at_point(x)
        if c is None:
            c=self.add_cell_at_point(x,**kw)
        return c
    def delete_cell_at_point(self,x):
        c=self.select_cells_nearest(x,inside=True)
        if c is not None:
            self.delete_cell(c)
        return c

    def enclosing_nodestring(self,x,max_nodes=None):
        """
        Given a coordinate pair x, look for a string of nodes and edges
        which form a closed polygon around it.
        max_nodes defaults to self.max_sides.
        Return None if nothing is found, otherwise a list of node indexes.
        """
        if max_nodes<0:
            max_nodes=self.Nnodes()
        elif max_nodes is None:
            max_nodes=self.max_sides

        # lame stand-in for a true bounding polygon test
        edges_near=self.select_edges_nearest(x,count=6)
        potential_cells=self.find_cycles(max_cycle_len=max_nodes,
                                         starting_edges=edges_near)
        pnt=geometry.Point(x)
        for pc in potential_cells:
            poly=geometry.Polygon( self.nodes['x'][pc] )
            if poly.contains(pnt):
                return pc

    def add_cell_at_point(self,x,**kw):
        pc=self.enclosing_nodestring(x,max_nodes=self.max_sides)
        if pc is not None:
            return self.add_cell(nodes=pc,**kw)
        return None

    @listenable
    def add_cell(self,**kwargs):
        """
        Does *not* check topology / planarity.  Assumes that edges already exist
        """
        i=None
        if '_index' in kwargs:
            i=kwargs.pop('_index')
            if i==len(self.cells): # had been self.edges, seems wrong
                # this is the index we'd get anyway.
                i=None
            else:
                assert len(self.cells)>i
                assert self.cells[i]['deleted']

        if i is None:
            c=np.zeros( (),dtype=self.cell_dtype)
            self.cells=array_append(self.cells,c)
            i=len(self.cells)-1
        else:
            pass

        # default values for native fields
        self.cells['_center'][i]=np.nan
        self.cells['_area'][i]=np.nan
        self.cells['edges'][i]=self.UNKNOWN 

        for k,v in six.iteritems(kwargs):
            if k in ['edges','nodes']: # may have to make this more generic..
                self.cells[k][i][:len(v)] = v
                self.cells[k][i][len(v):] = self.UNDEFINED # -1
            else:
                self.cells[k][i]=v

        # Avoids issue with bogus value of 'deleted' coming in with kwargs
        self.cells['deleted'][i]=False

        if self._node_to_cells is not None:
            for n in self.cell_to_nodes(i):
                self._node_to_cells[n].append(i)

        if self._cell_center_index is not None:
            if self.cell_center_index_point=='circumcenter':
                cc=self.cells_center()[i]
            else: # centroid
                cc=self.cells_centroid([i])[0]
            self._cell_center_index.insert(i,cc[self.xxyy])

        # updated 2016-08-25 - not positive here.
        # This whole chunk needs testing.
        # maybe some confusion over when edges has to be set
        edges=self.cell_to_edges(i)
        
        if 'edges' not in kwargs:
            # wait - is this circular??
            self.cells['edges'][i,:len(edges)]=edges
            self.cells['edges'][i,len(edges):]=self.UNDEFINED

        nodes=self.cell_to_nodes(i)
        
        for side in range(len(edges)):
            j=edges[side]
            n1=nodes[side]
            n2=nodes[ (side+1)%len(nodes) ]
            
            if ( (n1==self.edges['nodes'][j,0]) and
                 (n2==self.edges['nodes'][j,1]) ):
                # this cell is on the 'left' side of the edge
                assert self.edges['cells'][j,0]<0
                # TODO: probably this ought to be using modify_edge
                self.edges['cells'][j,0]=i
            elif ( (n1==self.edges['nodes'][j,1]) and
                   (n2==self.edges['nodes'][j,0]) ):
                # this cell is on the 'right' side of the edge
                assert self.edges['cells'][j,1]<0
                # TODO: probably this ought to be using modify_edge
                self.edges['cells'][j,1]=i
            else:
                assert False
            
        self.push_op(self.unadd_cell,i)

        return i

    def unadd_cell(self,i):
        self.delete_cell(i)

    def add_cell_and_edges(self,nodes,**kws):
        """ convenience wrapper for add_cell which makes sure all 
        the edges exist first.
        """ 
        for a,b in circular_pairs(nodes):
            j=self.nodes_to_edge(a,b)
            if j is None:
                self.add_edge(nodes=[a,b])
        return self.add_cell(nodes=nodes,**kws)
        

    @listenable
    def modify_cell(self,c,**kws):
        """ largely incomplete.  This will need to 
        update any geometry and topology details
        """
        if 'nodes' in kws and self._node_to_cells is not None:
            for n in self.cell_to_nodes(c):
                self._node_to_cells[n].remove(c)

        for k,v in six.iteritems(kws):
            if k in ('nodes','edges'):
                self.cells[k][c,:len(v)]=v
                self.cells[k][c,len(v):]=self.UNDEFINED
            else:                
                self.cells[k][c]=v

        if 'nodes' in kws and self._node_to_cells is not None:
            for n in self.cell_to_nodes(c):
                self._node_to_cells[n].append(c)

    @listenable
    def modify_edge(self,j,**kws):
        # likewise, this will have to get smarter about patching up derived
        # geometry and topology

        if 'nodes' in kws and self._node_to_edges is not None:
            for n in self.edges['nodes'][j]:
                self._node_to_edges[n].remove(j)

        for k,v in six.iteritems(kws):
            self.edges[k][j]=v

        if 'nodes' in kws and self._node_to_edges is not None:
            for n in self.edges['nodes'][j]:
                self._node_to_edges[n].append(j)
            
    @listenable
    def modify_node(self,n,**kws):
        if self._cell_center_index:
            my_cells=self.node_to_cells(n)

            if self.cell_center_index_point=='circumcenter':
                ccs=self.cells_center()[my_cells]
            else:
                ccs=self.cells_centroid(my_cells)
                
            for c,cc in zip(my_cells,ccs):
                self._cell_center_index.delete(c,cc[self.xxyy])
        else:
            cc=None

        if 'x' in kws and self._node_index is not None:
            self._node_index.delete(n,self.nodes['x'][n][self.xxyy])

        undo={}
        for k,v in six.iteritems(kws):
            # this copy is probably going to lead to heartache.
            undo[k]=copy.copy( self.nodes[k][n] )
            self.nodes[k][n]=v

        # in this case, the reverse operation can be handled by the same method
        # as the forward
        self.push_op(self.modify_node,n,**undo)

        if 'x' in kws and self._node_index is not None:
            self._node_index.insert(n,self.nodes['x'][n][self.xxyy])

        if self._cell_center_index:
            if self.cell_center_index_point=='circumcenter':
                ccs=self.cells_center(refresh=my_cells)[my_cells]
            else: # centroid
                ccs=self.cells_centroid(my_cells)
            for c,cc in zip(my_cells,ccs):
                self._cell_center_index.insert(c,cc[self.xxyy])

    def elide_node(self,n):
        """ 
        Delete a node, patching up a pair of edges and possibly
        adjacent cells.
        Has not been tested against undo, and many of these operations
        have not been tested against geometry/topology updates
        """
        
        js=self.node_to_edges(n)
        assert len(js)==2
        # have to copy this, as the original gets modified by delete
        cs=list(self.node_to_cells(n))
        assert len(cs)<=2
        # second edge is totally removed:
        cell_nodes=[self.cell_to_nodes(c) for c in cs]
        for c in cs:
            self.delete_cell(c)
        new_edge_nodes=[nn
                        for nn in self.edges['nodes'][js].ravel()
                        if nn!=n]
        self.delete_edge(js[1])
        self.modify_edge(js[0],nodes=new_edge_nodes)
        for c,nodes in zip(cs,cell_nodes):
            nodes=[nn for nn in nodes if nn!=n]
            self.add_cell(_index=c,nodes=nodes)
        self.delete_node(n)
            
    def cell_replace_node(self,c,n_old,n_new):
        """ if n_old is part of the given cell, change it to n_new,
        and update any cached mapping.  Doesn't try to change edges or
        do any futher checking
        """
        for ni in range(self.max_sides):
            if self.cells['nodes'][c,ni] == n_old:
                self.cells['nodes'][c,ni] = n_new
                if self._node_to_cells is not None:
                    self._node_to_cells[n_old].remove(c)
                    self._node_to_cells[n_new].append(c)
    def edge_replace_node(self,j,n_old,n_new):
        """ see cell_replace_node
        """
        for ni in [0,1]:
            if self.edges['nodes'][j,ni] == n_old:
                self.edges['nodes'][j,ni] = n_new
                if self._node_to_edges is not None:
                    self._node_to_edges[n_old].remove(j)
                    self._node_to_edges[n_new].append(j)

    #-# higher level topology modifications
    def collapse_short_edges(self,l_thresh=1.0):
        """ edges shorter than the given length are collapsed.
        """
        l = self.edges_length()
        to_collapse = np.nonzero(l<l_thresh)[0]

        for j in to_collapse:
            print( "Collapsing edge",j)
            self.collapse_edge(j)

    def collapse_edge(self,j_del):
        n1,n2 = self.edges['nodes'][j_del]
        # keep the smaller one - shouldn't really matter, but for refined
        # grids, this should keep the original node, and discard the bad
        # center node.
        n_keep = min(n1,n2)
        n_del = max(n1,n2)

        c1,c2 = self.edges['cells'][j_del]
        for eci,c in enumerate([c1,c2]):
            if c<0:
                continue

            if self.cell_Nsides(c)<=3:
                raise GridException("not implemented")
            else:
                c_n = list(self.cells['nodes'][c])
                c_n.remove(n_del)
                c_n.append(-1)
                self.cells['nodes'][c] = c_n
                if self._node_to_cells is not None:
                    self._node_to_cells[n_del].remove(c)

                c_e = list(self.cells['edges'][c])
                c_e.remove(j_del)
                c_e.append(-1)
                self.cells['edges'][c] = c_e
            # even though it's about to be deleted, keep this consistent by removing
            # any mention of this cell:
            self.edges['cells'][j_del,eci] = self.UNKNOWN

        self.delete_edge(j_del)

        # take care of the other ones:
        # note that these have to be copied since the original lists will be modified
        # during the loops
        for c in list(self.node_to_cells(n_del)):
            self.cell_replace_node(c,n_del,n_keep)
        for j in list(self.node_to_edges(n_del)):
            self.edge_replace_node(j,n_del,n_keep)

        self.delete_node(n_del) 

    #-# Plotting
    def plot_boundary(self,ax=None,**kwargs):
        return self.plot_edges(mask=self.edges['mark']>0,**kwargs)

    def node_clip_mask(self,clip):
        return within_2d(self.nodes['x'],clip)
    
    def plot_nodes(self,ax=None,mask=None,values=None,sizes=20,labeler=None,clip=None,
                   **kwargs):
        """ plot nodes as scatter
        labeler: callable taking (node index, node record), return string
        """
        ax=ax or plt.gca()
            
        if mask is None:
            mask=~self.nodes['deleted']

        if clip is not None: # convert clip to mask
            mask=mask & self.node_clip_mask(clip)

        if values is not None:
            values=values[mask]
            kwargs['c']=values

        if labeler is not None:
            if labeler=='id':
                labeler=lambda n,rec: str(n)
                
            # weirdness to account for mask being indices vs. bitmask
            for n in np.arange(self.Nnodes())[mask]: # np.nonzero(mask)[0]:
                ax.text(self.nodes['x'][n,0],
                        self.nodes['x'][n,1],
                        labeler(n,self.nodes[n]))

        coll=ax.scatter(self.nodes['x'][mask][:,0],
                        self.nodes['x'][mask][:,1],
                        sizes,
                        **kwargs)
        request_square(ax)
        return coll
    
    def plot_edges(self,ax=None,mask=None,values=None,clip=None,labeler=None,
                   **kwargs):
        """
        plot edges as a LineCollection.
        optionally select a subset of edges with boolean array mask.
        Note that mask is over all edges, even deleted ones, and overrides
          internal masking of deleted edges.
        and set scalar values on edges with values
         - values can have size either Nedges, or sum(mask)
        """
        ax = ax or plt.gca()

        edge_nodes = self.edges['nodes']
        if mask is None:
            mask = ~self.edges['deleted']

        if values is not None:
            values = np.asarray(values)

        if clip is not None:
            mask=mask & self.edge_clip_mask(clip)
            
        #if mask is not None:
        edge_nodes = edge_nodes[mask]
        # try to be smart about when to slice the edge values and when
        # they come pre-sliced
        if values is not None and len(values)==self.Nedges():
            values = values[mask]
                
        segs = self.nodes['x'][edge_nodes]
        if values is not None:
            kwargs['array'] = values
            
        lcoll = LineCollection(segs,**kwargs)

        if labeler is not None:
            ec=self.edges_center()
            # weirdness to account for mask being indices vs. bitmask
            for n in np.arange(self.Nedges())[mask]:
                ax.text(ec[n,0], ec[n,1],
                        labeler(n,self.edges[n]))

        ax.add_collection(lcoll)
        request_square(ax)
        return lcoll

    def plot_halfedges(self,ax=None,mask=None,values=None,clip=None,
                       labeler=None,
                       offset=0.2,**kwargs):
        """
        plot a scatter and/or labels, two per edge, corresponding to
        two half-edges (i.e. edges['cells']).

        mask selects a subset of edges with boolean array mask.
        mask is over all edges, even deleted ones, and overrides
          internal masking of deleted edges.
        values: scalar values for scatter.  size  (Nedges,2) or (sum(mask),2)
        labeler: show text label at each location. callable function f(edge_idx,{0,1})
        offset: fraction of edge length to offset the half-edge location
        """
        ax = ax or plt.gca()

        edge_nodes = self.edges['nodes']
        if mask is None:
            mask = ~self.edges['deleted']
        if values is not None:
            values = np.asarray(values)
        if clip is not None:
            mask=mask & self.edge_clip_mask(clip)
            
        edge_nodes = edge_nodes[mask]

        segs = self.nodes['x'][edge_nodes]
        midpoints = segs.mean(axis=1)
        deltas = segs[:,:,:] - midpoints[:,None,:]
        # rotate by 90
        deltas=offset*deltas[:,:,::-1]
        deltas[:,:,1]*=-1 
        offset_points = midpoints[:,None,:] + deltas

        # try to be smart about when to slice the edge values and when
        # they come pre-sliced
        if values is not None and len(values)==self.Nedges():
            values = values[mask]
                
        if values is not None:
            coll = plt.scatter(offset_points[:,:,0].ravel(),
                               offset_points[:,:,1].ravel(),
                               30,
                               values.ravel())
        else:
            coll = ax.plot(offset_points[:,:,0].ravel(),
                           offset_points[:,:,1].ravel(),
                           '.')

        if labeler is not None:
            # offset_points has already been masked, so use the 
            # enumerated ji there, but labeler expects original
            # edge indices, pre-mask, so use j.
            for ji,j in enumerate( np.nonzero(mask)[0] ):
                for side in [0,1]:
                    ax.text( offset_points[ji,side,0],
                             offset_points[ji,side,1],
                             labeler(j,side) )
        return coll
    
    def scalar_contour(self,scalar,V=10,smooth=True):
        """ Generate a collection of edges showing the contours of a
        cell-centered scalar.

        V: either an int giving the number of contours which will be
        evenly spaced over the range of the scalar, or a sequence
        giving the exact contour values.

        smooth: control whether one pass of 3-point smoothing is
        applied.

        returns a LineCollection 
        """
        if isinstance(V,int):
            V = np.linspace( np.nanmin(scalar),np.nanmax(scalar),V )

        disc = np.searchsorted(V,scalar) # nan=>last index

        e2c=self.edge_to_cells()
        nc1 = e2c[:,0]
        nc2 = e2c[:,1].copy()
        nc2[nc2<0] = nc1[nc2<0]

        to_show = (disc[nc1]!=disc[nc2]) & np.isfinite(scalar[nc1]+scalar[nc2]) 

        segs = self.nodes['x'][ self.edges[to_show]['nodes'], :]

        # goofy work around to prevent extraneous output.
        pm=join_features.progress_message
        def nop(*a,**k):
            pass
        join_features.progress_message=nop
        joined_segs = join_features.merge_lines(segments=segs)
        join_features.progress_message=pm
        
        # Smooth those out some...
        def smooth_seg(seg):
            seg = seg.copy()
            seg[1:-1,:] = (2*seg[1:-1,:] + seg[0:-2,:] + seg[2:,:])/4.0
            return seg

        if smooth:
            simple_segs = [smooth_seg(seg) for seg in joined_segs]
        else:
            simple_segs = joined_segs

        from matplotlib import collections
        ecoll = collections.LineCollection(simple_segs)
        ecoll.set_edgecolor('k')

        return ecoll

    def make_triangular(self):
        if self.max_sides==3:
            return # nothing to do

        for c in self.valid_cell_iter():
            nodes=np.array(self.cell_to_nodes(c))

            if len(nodes)==3:
                continue
            self.delete_cell(c)
            self.add_cell_and_edges(nodes=nodes[ [0,1,2] ] )
            if len(nodes)>=4:
                self.add_cell_and_edges(nodes=nodes[ [0,2,3] ] )
            if len(nodes)>=5: # a few of these...
                self.add_cell_and_edges(nodes=nodes[ [0,3,4] ] )
            # too lazy to be generic about it...
            # also note that the above only work for convex cells.

        self.renumber()

    def mpl_triangulation(self):
        """
        Return a matplotlib triangulation for the cells of the grid.
        Only guarantees that the nodes retain their order
        """
        tris=[] # [ (n1,n2,n3), ...]

        for c in self.valid_cell_iter():
            nodes=np.array(self.cell_to_nodes(c))

            # this only works for convex cells
            for i in range(1,len(nodes)-1):
                tris.append( nodes[ [0,i,i+1] ] )

        tris=np.array(tris)
        tri=Triangulation(self.nodes['x'][:,0],self.nodes['x'][:,1],
                          triangles=tris )
        return tri
        
    def contourf_node_values(self,values,*args,**kwargs):
        """
        Plot a smooth contour field defined by values at nodes and topology of cells.

        More involved than you might imagine:
         1. Fabricate a triangular version of the grid
         2. 
        """
        ax=kwargs.pop('ax',None) or plt.gca()
        tri=self.mpl_triangulation()
        return ax.tricontourf(tri,values,*args,**kwargs)
        
    def edge_clip_mask(self,xxyy):
        centers=self.edges_center()
        return (centers[:,0] > xxyy[0]) & (centers[:,0]<xxyy[1]) & \
            (centers[:,1] > xxyy[2]) & (centers[:,1]<xxyy[3])
        
    def cell_clip_mask(self,xxyy):
        centers=self.cells_center()
        return  (centers[:,0] > xxyy[0]) & (centers[:,0]<xxyy[1]) & \
            (centers[:,1] > xxyy[2]) & (centers[:,1]<xxyy[3])
        
    def plot_cells(self,ax=None,mask=None,values=None,clip=None,centers=False,labeler=None,
                   centroid=False,**kwargs):
        """
        centers: scatter plot of cell centers.  otherwise polygon plot
        labeler: f(cell_idx,cell_record) => string for labeling.
        centroid: if True, use centroids instead of centers
        """
        ax = ax or plt.gca()
        
        if values is not None:
            # asanyarray allows for masked arrays to pass through unmolested.
            values = np.asanyarray(values)

        if clip is not None: # convert clip to mask
            mask=self.cell_clip_mask(clip)

        if mask is None:
            mask=~self.cells['deleted']

        if values is not None and len(values)==self.Ncells():
            values = values[mask]

        if centers or labeler:
            if centroid:
                xy=self.cells_centroid()
            else:
                xy=self.cells_center()
        else:
            xy=None # unused
            
        if not centers:
            if values is not None:
                kwargs['array'] = values
            cell_nodes = self.cells['nodes'][mask]
            polys = self.nodes['x'][cell_nodes]
            missing = cell_nodes<0
            polys[missing,:] = np.nan # seems to work okay for triangles
            coll = PolyCollection(polys,**kwargs)
            ax.add_collection(coll)
        else:
            args=[]
            if values is not None:
                args.append(values)
            coll = ax.scatter(xy[mask,0],xy[mask,1],20,*args,**kwargs)

        if labeler is not None:
            for c in np.nonzero(mask)[0]:
                ax.text(xy[c,0],xy[c,1],labeler(c,self.cells[c]))

        request_square(ax)
        return coll
    
    def edges_length(self):
        p1 = self.nodes['x'][self.edges['nodes'][:,0]]
        p2 = self.nodes['x'][self.edges['nodes'][:,1]]
        return mag( p2-p1 )

    def cells_area(self):
        sel = np.isnan(self.cells['_area']) & (~self.cells['deleted'])

        if sum(sel)>0:
            for c in np.nonzero(sel)[0]:
                self.cells['_area'][c] = signed_area(self.nodes['x'][self.cell_to_nodes(c)])
                
        return self.cells['_area']

    #-# Selection methods:
    #  various methods which return a bitmask over cells, edges or nodes
    #  though in some cases it's more efficient to deal with index lists, it's much
    #  easier to compose selections with bitmasks, so there it is.
    #  when implementing these, note that all selections should avoid 'deleted' elements.
    #   unless the selection criteria explicitly includes them.
    def select_edges_intersecting(self,geom,invert=False):
        """
        geom: a shapely geometry
        returns: bitmask overcells, with non-deleted, selected edges set and others False.
        if invert is True, select edges which do not intersect the the given geometry.  
        """
        sel = np.zeros(self.Nedges(),np.bool8) # initialized to False
        for j in range(self.Nedges()):
            if self.edges['deleted'][j]:
                continue
            edge_line = geometry.LineString(self.nodes['x'][self.edges['nodes'][j]])
            sel[j] = geom.intersects(edge_line)
            if invert:
                sel[j] = ~sel[j]
        return sel
    
    def cell_polygon(self,c):
        return geometry.Polygon(self.nodes['x'][self.cell_to_nodes(c)])

    def node_point(self,n):
        return geometry.Point( self.nodes['x'][n] )

    def boundary_linestrings(self):
        # could be much smarter and faster, directly traversing boundary edges
        # but this way is easy
        e2c=self.edge_to_cells()
        # some grids don't abide by the boundary always being on the "right"
        # so use any()
        boundary_edges=(np.any(e2c<0,axis=1))&(~self.edges['deleted'])

        if 0: # old, slow implementation
            segs=self.nodes['x'][self.edges['nodes'][boundary_edges]]
            lines=join_features.merge_lines(segments=segs)
        else:
            marked=np.zeros(self.Nedges(),np.bool8)
            lines=[]
            for j in np.nonzero(boundary_edges)[0]:
                if marked[j]:
                    continue
                trav=self.halfedge(j,0)
                if trav.cell()>=0:
                    trav=trav.opposite()
                assert trav.cell()<0
                start=trav
                this_line_nodes=[trav.node_rev(),trav.node_fwd()]
                while 1:
                    this_line_nodes.append(trav.node_fwd())
                    marked[trav.j]=True
                    trav=trav.fwd()
                    if trav==start:
                        break
                lines.append( self.nodes['x'][this_line_nodes] )

        return lines

    def boundary_polygon_by_edges(self):
        """ return polygon, potentially with holes, representing the domain.
        equivalent to unioning all cell_polygons, but hopefully faster.
        in one test, this method was 3.9 times faster than union.  This is 
        certainly depends on the complexity and size of the grid, though.
        """
        lines=self.boundary_linestrings()
        polys=join_features.lines_to_polygons(lines,close_arc=False)
        if len(polys)>1:
            raise GridException("somehow there are multiple boundary polygons")
        return polys[0]

    def boundary_polygon_by_union(self):
        """ Compute a polygon encompassing the full domain by unioning all
        cell polygons.
        """
        cell_geoms = [None]*self.Ncells()

        for i in self.valid_cell_iter():
            xy = self.nodes['x'][self.cell_to_nodes(i)]
            cell_geoms[i] = geometry.Polygon(xy)
        return ops.cascaded_union(cell_geoms) 

    def boundary_polygon(self):
        """ return polygon, potentially with holes, representing the domain.
        This method tries an edge-based approach, but will fall back to unioning
        all cell polygons if the edge-based approach fails.
        """
        try:
            return self.boundary_polygon_by_edges()
        except Exception as exc:
            self.log.warning('Warning, boundary_polygon() failed using edges!  Trying polygon union method')
            self.log.warning(exc,exc_info=True)
            return self.boundary_polygon_by_union()

    def extract_linear_strings(self):
        """
        extract contiguous linestrings as sequences of nodes.
        """ 
        # there are at least three choices of how greedy to be.
        #  min: each edge is its own feature
        #  max: extract features as long as possible, and allow for 'T' junctions.
        #  mid: break features at nodes with degree>2.
        # go with mid
        strings=[]
        edge_marks=np.zeros( self.Nedges(),'b1')

        for j in self.valid_edge_iter():
            if edge_marks[j]:
                continue
            edge_marks[j]=True

            trav=tuple(self.edges['nodes'][j])
            node_fwd=self.edges['nodes'][j,1]
            node_rev=self.edges['nodes'][j,0]

            node_string=[node_fwd,node_rev]

            for trav in [ (node_fwd,node_rev),
                          (node_rev,node_fwd) ]:
                while 1:
                    js = self.node_to_edges(trav[1])

                    if len(js)!=2:
                        break

                    for j in js:
                        jnodes=self.edges['nodes'][j]
                        if trav[0] in jnodes:
                            continue
                    if edge_marks[j]:
                        # possible if we go all the way around a ring.
                        break
                    edge_marks[j]=True
                    nxt=[n for n in jnodes if n!=trav[1]][0]
                    node_string.append(nxt)
                    trav=(trav[1],nxt)
                node_string=node_string[::-1]

            feat_nodes=np.array( node_string )
            strings.append( feat_nodes )
        return strings

    def select_nodes_boundary_segment(self, coords, ccw=True):
        """
        bc_coords: [ [x0,y0], [x1,y1] ] coordinates, defining
        start and end of boundary segment, traversing CCW boundary of
        grid.

        if ccw=False, then traverse the boundary CW instead of CCW.

        returns [n0,n1,...] nodes along boundary between those locations.
        """
        self.edge_to_cells()
        start_n,end_n=[ self.select_nodes_nearest(xy) 
                        for xy in coords]
        cycle=np.asarray( self.boundary_cycle() )
        start_i=np.nonzero( cycle==start_n )[0][0]
        end_i=np.nonzero( cycle==end_n )[0][0]

        if start_i<end_i:
            boundary_nodes=cycle[start_i:end_i+1]
        else:
            boundary_nodes=np.r_[ cycle[start_i:], cycle[:end_i]]
        return boundary_nodes

    def select_nodes_intersecting(self,geom=None,xxyy=None,invert=False,as_type='mask'):
        sel = np.zeros(self.Nnodes(),np.bool8) # initialized to False

        assert (geom is not None) or (xxyy is not None)

        if xxyy is not None:
            geom=geometry.box(xxyy[0], xxyy[2],xxyy[1],xxyy[3])
            
        for n in range(self.Nnodes()):
            if self.nodes['deleted'][n]:
                continue
            test = geom.intersects(self.node_point(n))
            if invert:
                test = ~test
            sel[n] = test
        if as_type!='mask':
            sel=np.nonzero(sel)[0]
        return sel

    def select_cells_intersecting(self,geom,invert=False,as_type="mask",by_center=False):
        """
        geom: a shapely geometry
        invert: select cells which do not intersect.
        as_type: 'mask' returns boolean valued mask, 'indices' returns array of indices
        by_center: if true, test against the cell center.  By default, tests against the 
        finite cell.
        """
        if as_type is 'mask':
            sel = np.zeros(self.Ncells(),np.bool8) # initialized to False
        else:
            sel = []

        if by_center:
            centers=self.cells_center()
            
        for c in range(self.Ncells()):
            if self.cells['deleted'][c]:
                continue
            if by_center:
                test=geom.intersects( geometry.Point(centers[c]) )
            else:
                test=geom.intersects( self.cell_polygon(c) )
            if invert:
                test = ~test
            if as_type is 'mask':
                sel[c] = test
            else:
                if test:
                    sel.append(c)
        return sel

    def select_cells_by_cut(self,line,start=0,side='left',delta=1.0):
        """
        Split the cells by a linestring.  By default, returns
        a bitmask set to True for cells falling on the "left"
        side of the linestring.  

        Uses basic graph traversal, and will be faster when start
        is set to a cell index on the smaller side of the linestring.

        delta: size of finite difference used in determining the orientation
        of the cut.
        """
        marks=np.zeros(self.Ncells(),np.bool8)

        def test_edge(j):
            cells=self.edges['cells'][j]
            if cells[0]<0 or cells[1]<0:
                return True # don't traverse
            seg=geometry.LineString( cc[cells] )
            return line.intersects(seg)

        stack=[start]
        count=0

        start_on_left=None

        cc=self.cells_center()
        e2c=self.edge_to_cells()

        while stack:
            count+=1
            if count%5000==0:
                self.log.info("checked on %d/%d edges"%(count,self.Nedges()))

            c=stack.pop()

            marks[c]=True
            for j in self.cell_to_edges(c):
                if test_edge(j):
                    if start_on_left is None:
                        # figure out the orientation
                        cells=e2c[j]
                        if cells[0]>=0 and cells[1]>=0:
                            if cells[0]==c:
                                seg=geometry.LineString( cc[cells] )
                            else:
                                seg=geometry.LineString( cc[cells[::-1]] )
                            orientation=orient_intersection(seg,line)
                            if orientation>0:
                                start_on_left=True
                            else:
                                start_on_left=False
                    continue
                for nbr in self.edges['cells'][j]:
                    # redundant but cheap check on nbr sign.
                    if nbr>=0 and marks[nbr]==0:
                        stack.append(nbr)

        # make sure we eventually had a real edge crossing
        assert start_on_left is not None

        # would like to know which side of the cut we are on...
        # and invert marks if the request was for the other side
        if (side=='left') != (start_on_left==True):
            marks=~marks

        return marks

    ### NB: refreshing indices after modifying the grid is not yet implemented
    #       the scipy KDtree is easy to install, but does not allow for online
    #       changes.  long-term solution may involve libspatialindex/RTree, or
    #       a CGAL-derived index.
    _node_index=None
    def node_index(self):
        if self._node_index is None:
            tuples = [(i,self.nodes['x'][i,self.xxyy],None) 
                      for i in range(self.Nnodes()) 
                      if not self.nodes['deleted'][i] ]
            self._node_index = gen_spatial_index.PointIndex(tuples,interleaved=False)
        return self._node_index

    def select_nodes_nearest(self,xy,count=None):
        """ count is None: return a scalar, the closest.
        otherwise, even if count==1, return a list
        """
        xy=np.asarray(xy)
        real_count=count
        if count is None:
            real_count=1
        hits = self.node_index().nearest(xy[self.xxyy],real_count)

        if isinstance( hits, types.GeneratorType): # usual for recent versions
            results=[]
            for hit in hits:
                results.append(hit)
                if len(results)==real_count:
                    break
            hits=results

        if count is None:
            # print "ug: select_nodes_nearest(%s)=%s"%(xy,hits[0])
            if len(hits):
                return hits[0]
            else:
                return None
        else:
            return hits

    def shortest_path(self,n1,n2,return_type='nodes',
                      edge_selector=lambda j: True):
        """ dijkstra on the edge graph from n1 to n2
        returns list of node indexes
        selector: given an edge index, return True if the edge should be considered
        """
        if pq is None:
            raise GridException("shortest_path requires the  priority_queue module.")

        queue = pq.priorityDictionary()
        # 
        queue[n1] = [0,None] # distance, predecessor

        done = {} # likewise

        while 1:
            # find the queue-member with the lowest cost:
            if len(queue)==0:
                return None
            best = queue.smallest()
            best_cost,pred = queue[best]
            del queue[best]

            done[best] = [best_cost,pred]

            if best == n2:
                break

            # figure out its neighbors
            nbrs=[]
            for j in self.node_to_edges(n=best):
                if edge_selector(j):
                    ne1,ne2=self.edges['nodes'][j]
                    if ne1==best:
                        nbrs.append(ne2)
                    else:
                        nbrs.append(ne1)

            for nbr in nbrs:
                if nbr in done:
                    continue

                dist = mag( self.nodes['x'][nbr] - self.nodes['x'][best] )
                new_cost = best_cost + dist

                if nbr not in queue:
                    queue[nbr] = [np.inf,None]

                if queue[nbr][0] > new_cost:
                    queue[nbr] = [new_cost,best]

        # reconstruct the path:
        path = [n2]

        while 1:
            pred=done[path[-1]][1]
            if pred is None:
                break
            else:
                path.append(pred)

        path = np.array(path[::-1]) # reverse it so it goes from n1 to n2

        if return_type=='nodes':    
            return path
        elif return_type in ('edges','sides'):
            return np.array( [self.nodes_to_edge(path[i],path[i+1])
                              for i in range(len(path)-1)] )


    def create_dual(self,center='centroid',create_cells=False):
        """
        Very basic dual-grid construction.  Does not yet create cells,
        just being used to simplify connectivity of an aggregation grid.
        """
        assert not create_cells,"Not yet supported"
        gd=UnstructuredGrid()

        if center=='centroid':
            cc=self.cells_centroid()
        else:
            cc=self.cells_center()

        gd.add_node_field('dual_cell',np.zeros(0,'i4'))
        gd.add_edge_field('dual_edge',np.zeros(0,'i4'))

        for c in self.valid_cell_iter():
            # redundant, but we both force the index of this
            # to be c, but also store a dual_cell index.  This
            # be streamlined once it's clear that dual_cell is not needed.
            gd.add_node(_index=c,x=cc[c],dual_cell=c)

        e2c=self.edge_to_cells()

        for j in self.valid_edge_iter():
            if e2c[j].min() < 0:
                continue # boundary
            dj_exist=gd.nodes_to_edge(e2c[j])
            if dj_exist is None:
                dj=gd.add_edge(nodes=e2c[j],dual_edge=j)

        return gd
        
    def cells_connected_components(self,edge_mask,cell_mask=None,randomize=True):
        """
        Label the cells of the grid based on connections. 
        edge_mask: boolean array, true for edges which should be considered a connection.
        cell_mask: optional boolean array, true for cells which should be considered
          active.  Currently this is taken into account as a post-processing step - 
          connectivity is defined based on edges, and inactive cells are trimmed from
          the output.  This may change in the future, though.
        randomize: the returned labels are randomly assigned.  This can help with 
          color plotting of the labels by generally creating more contrast between 
          adjacent components.

        Returns: labels, masked integer array of size self.Ncells().  Inactive cells
          masked out, other cells labeled with the component to which they belong.
          
        """
        # further constrain to edges which are not on the boundary
        edge_mask=edge_mask & np.all( self.edges['cells']>=0,axis=1)

        cell_pairs = self.edges['cells'][edge_mask]

        # use scipy graph algorithms to find the connections
        from scipy import sparse

        graph=sparse.csr_matrix( (np.ones(len(cell_pairs)), 
                                  (cell_pairs[:,0],cell_pairs[:,1])),
                                 shape=(self.Ncells(),self.Ncells()) )

        n_comps,labels=sparse.csgraph.connected_components(graph,directed=False)

        if cell_mask is None:
            cell_mask=slice(None)

        unique_labels=np.unique( labels[cell_mask] ) 
        labels[~cell_mask]=-1 # mark dry cells as -1

        # create an array which takes the original label, maps it to small, sequential
        # label.

        if not randomize:
            new_labels=np.arange(len(unique_labels)) # -1 will be handled separately
        else:
            new_labels=np.argsort(np.random.random(len(unique_labels)))

        mapper=np.zeros( 1+unique_labels.max() ) - 1 # map original labels to compressed labels
        mapper[unique_labels]=new_labels
        labels=mapper[labels]
        labels[~cell_mask] = -1
        labels=np.ma.array(labels,mask=~cell_mask)
        return labels

        
    _cell_center_index=None
    cell_center_index_point='centroid' # 'centroid' or 'circumcenter'
    def cell_center_index(self):
        """ TODO: this is not wired into the update,listen, etc.
        framework.  Do not use yet for a grid which will be modified.
        """
        if self._cell_center_index is None:
            if self.cell_center_index_point=='circumcenter':
                cc=self.cells_center()
            else:
                cc=self.cells_centroid()
            tuples = [(i,cc[i,self.xxyy],None) 
                      for i in range(self.Ncells()) 
                      if not self.cells['deleted'][i] ]
            self._cell_center_index = gen_spatial_index.PointIndex(tuples,interleaved=False)
        return self._cell_center_index

    def select_edges_nearest(self,xy,count=None,fast=True):
        xy=np.asarray(xy)

        real_count=count
        if count is None:
            real_count=1

        if fast:
            # TODO: maintain a proper edge index
            # For now, brute force it.
            ec=self.edges_center()
            dists=mag(xy-ec)
            dists[self.edges['deleted']]=np.inf
            hits=np.argsort(dists)[:real_count]
        else:
            # actually query against the finite geometry of the edge
            # still not exact, but okay in most cases.  an exact solution
            # will have to wait until exact_delaunay is faster.
            j_near = self.select_edges_nearest(xy,count=real_count*5,fast=True)

            geoms=[geometry.LineString(self.nodes['x'][self.edges['nodes'][j]])
                  for j in j_near]
            P=geometry.Point(xy)

            dists=[g.distance(P) for g in geoms]
            order=np.argsort(dists)
            hits=j_near[order[:real_count]]

        if count is None:
            return hits[0]
        else:
            return hits

    def select_cells_nearest(self,xy,count=None,inside=False):
        """
        xy: coordinate to query around
        count: if None, return a single index, otherwise return an array of
          indices (even if count==1)
        inside: 
          False => use faster query, but no guarantee that xy is inside the returned cell
          True  => verify that xy is inside the returned cell, may return None.  cannot
           be used with count.  Even with True, the search isn't exhaustive and a really
           bizarre grid might yield erroneous results.  In this case, count is used to
           specify how many cells to test for containing the query point, and the
           return value will be a single index.
        """
        xy=np.asarray(xy)
        real_count=count
        if count is None:
            real_count=1

        if inside:
            # Now use count t
            # assert real_count==1
            # give ourselves a better shot at finding the right cell
            # figure that 10 is greater than the degree of
            # any nodes
            if count is None:
                real_count=10
            count=1

        hits = self.cell_center_index().nearest(xy[self.xxyy],real_count)

        if isinstance( hits, types.GeneratorType): # usual for recent versions
            results=[]
            for hit in hits:
                results.append(hit)
                if len(results)==real_count:
                    break
            hits=results
        
        if inside: # check whether the point is truly inside the cell
            # -- using shapely to determine contains, may be slower than matplotlib
            #  pnt=geometry.Point(xy[0],xy[1])
            #  for hit in hits:
            #      if self.cell_polygon(hit).contains(pnt):
            # -- using matplotlib to determine contains
            for hit in hits:
                if self.cell_path(hit).contains_point(xy):
                    return hit            
            return None

        if count is None:
            if len(hits):
                return hits[0]
            else:
                return None
        else:
            return hits
    
    def point_to_cell(self,xy):
        """ like select_cells_nearest, but verifies that the point
        is inside the cell.
        returns None if the point is not inside a cell
        """
        pnt = geometry.Point(xy)

        for c in self.select_cells_nearest(xy,10):
            if self.cell_polygon(c).intersects(pnt):
                return c
        return None

    def circum_errors(self):
        rel_errors=np.zeros(self.Ncells(),'f8')
        centers = self.cells_center()
        for nsi in range(3,self.max_sides+1):
            sel = (self.cells['nodes'][:,nsi-1]>=0)
            if nsi<self.max_sides:
                sel = sel&(self.cells['nodes'][:,nsi]<0)

            if nsi==3:
                rel_errors[sel]=0.0
                print( "%7d cells with 3 sides."%np.sum(sel) )
            else:
                offsets = self.nodes['x'][self.cells['nodes'][sel,:nsi]] - centers[sel,None,:]
                dists = mag(offsets)
                error = np.std(dists,axis=1) / np.mean(dists,axis=1)
                rel_errors[sel]=error
        return rel_errors
        
    def report_orthogonality(self):
        # circumcenter errors:
        errs=self.circumcenter_errors(radius_normalized=True)
        print("  Mean circumcenter error: %g"%(errs.mean()) )
        print("  Max circumcenter error: %g"%(errs.max()) )

        # angle errors:
        errs=np.abs( self.angle_errors() ) * 180/np.pi
        sel=np.isfinite(errs)
        print("  Mean angle error: %.2f deg"%( errs[sel].mean() ) ) 
        print("  Max angle error: %.2f deg"%( errs[sel].max() ) )

    def circumcenter_errors(self,radius_normalized=False,cells=None):
        if cells is None:
            cells=slice(None)
            
        centers = self.cells_center()[cells]
        errors=np.zeros( len(self.cells[cells]),'f8')

        for nsi in range(3,self.max_sides+1):
            sel = (self.cells['nodes'][cells,nsi-1]>=0) & (~self.cells['deleted'][cells])
            if nsi<self.max_sides:
                sel = sel&(self.cells['nodes'][cells,nsi]<0)

            if nsi==3:
                pass # print "%7d cells with 3 sides."%np.sum(sel)
            else:
                # A little goofy here trying to calculate just a subset..
                offsets = self.nodes['x'][self.cells['nodes'][cells][sel,:nsi]] - centers[sel,None,:]
                dists = mag(offsets)
                # print "%7d cells with %d sides."%(np.sum(sel),nsi)
                errs=np.std(dists,axis=1)
                if radius_normalized:
                    errs /=np.mean(dists,axis=1)
                errors[sel]=errs
        return errors
    def angle_errors(self):
        centers = self.cells_center()
        e2c=self.edge_to_cells(recalc=True)
        edge_vecs=( self.nodes['x'][ self.edges['nodes'][:,1] ]
                    - self.nodes['x'][ self.edges['nodes'][:,0] ] )
        link_vecs=( centers[e2c[:,1]] -
                    centers[e2c[:,0]] )
        boundary=e2c.min(axis=1)<0
        link_vecs[boundary,:]=np.nan
        dots=( (edge_vecs*link_vecs).sum(axis=1) 
               / mag(link_vecs)
               / mag(edge_vecs) )
        angle_err=(np.arccos(dots) - np.pi/2) 
        return angle_err



    @staticmethod
    def from_pickle(fn):
        with open(fn,'rb') as fp:
            return pickle.load(fp)
        
    def write_pickle(self,fn):
        with open(fn,'wb') as fp:
            pickle.dump(self,fp,-1)
            
    def __getstate__(self):
        # Mostly just clear out elements which can be easily recreated,
        # or objects which don't pickle well.
        d=super(UnstructuredGrid,self).__getstate__()

        d['_node_to_edges']=None
        d['_node_to_cells']=None
        d['_node_index'] = None
        d['_cell_center_index'] = None
        d['log']=None

        return d
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.init_log()

        print( "May need to rewire any internal listeners" )

    def init_log(self):
        self.log = logging.getLogger(self.__class__.__name__)
        
    def write_suntans(self,path):
        with open(os.path.join(path,'points.dat'),'wt') as fp:
            print( "Writing SUNTANS: %d nodes"%self.Nnodes() )
            for n in range(self.Nnodes()):
                fp.write("%.5f %.5f 0\n"%(self.nodes['x'][n,0],
                                          self.nodes['x'][n,1]))

        ptm_sides=self.edges_as_nodes_cells_mark()
        with open(os.path.join(path,'edges.dat'),'wt') as fp:
            for j in range(self.Nedges()):
                fp.write("%d %d %d %d %d\n"%( ptm_sides[j,0],ptm_sides[j,1],
                                              ptm_sides[j,4],
                                              ptm_sides[j,2],ptm_sides[j,3]) )

        with open(os.path.join(path,'cells.dat'),'wt') as fp:
            vc=self.cells_center()
            for i in range(self.Ncells()):
                fp.write("%.5f %.5f "%(vc[i,0],vc[i,1])) # voronoi centers
                nodes=self.cell_to_nodes(i)
                assert len(nodes)==3
                fp.write("%d %d %d "%(nodes[0],nodes[1],nodes[2]))
                nbrs=self.cell_to_cells(i,ordered=True) 
                fp.write("%d %d %d\n"%(nbrs[0],nbrs[1],nbrs[2]))
        
    def write_untrim08(self,fn):
        """ write this grid out in the untrim08 format.  Since untrim grids have
        some extra fields, this just delegates to the Untrim grid class, converting
        this grid to untrim then writing.
        """
        UnTRIM08Grid(grid=self).write_untrim08(fn)


    #-# Grid refinement
    # as ugly as it is having this in here, there's not a very clean way to
    # put this sort of code elsewhere, and still be able to call super()
    def global_refine(self):
        # # Some helper functions - stuffed inside here to avoid
        #  polluting namespace
        def copy_edge_attributes_to_refined(g_orig,g_new):
            """ find edges of g_orig which were subdivided to form edges in g_new,
            then copy relevant attributes to the subdivided edges.  No changes are
            made to edges which are internal to refined cells.

            Note that this copies all non-topology attributes, which may not be
            correct.  Probably have to clean that up after the fact.

            Also copies are shallow - array data will be copied, but if there are
            objects in the arrays, these will not be copied, just referenced.
            """
            def copy_attributes(j_orig,j_new):
                for field,type_ in g_new.edge_dtype:
                    if field not in ['nodes','cells'] and not field.startswith('_'):
                        g_new.edges[field][j_new] = g_orig.edges[field][j_orig]

            # loop through setting edge marks
            #  edges between a midpoint and one of the
            #  endpoints
            for j in range(g_new.Nedges()):
                a,b = g_new.edges['nodes'][j]
                if a>b:
                    a,b = b,a
                # only care about edges where one node is original,
                # the other is a midpoint.
                # there are no edges where both are original, and
                # edges with both as midpoints or with one as a center
                # always internal.
                if a < g_orig.Nnodes() and b>=g_orig.Nnodes() and b<g_orig.Nnodes()+g_orig.Nedges():
                    j_orig = b - g_orig.Nnodes()
                    copy_attributes(j_orig,j)

        def copy_cell_attributes_to_refined(g_orig,g_new):
            def copy_attributes(c_orig,c_new):
                for field,type_ in g_new.cell_dtype:
                    if field not in ['nodes','edges'] and not field.startswith('_'):
                        g_new.cells[field][c_new] = g_orig.cells[field][c_orig]

            for c_new in range(g_new.Ncells()):
                copy_attributes(c_new//4,c_new)
        
        # Refining:
        # 1. add the new points, first the edge-midpoint, second the cell centers
        new_points = [self.nodes['x'],
                      self.edges_center(),
                      self.cells_center()]
        new_points = np.concatenate( new_points )

        # 2. build up cells, just using the nodes
        new_cells = np.zeros( (self.max_sides*self.Ncells(),self.max_sides), np.int32) - 1

        for c in range(self.Ncells()):
            cn = self.cell_to_nodes(c)

            midpoints = []
            for i,(a,b) in enumerate(circular_pairs(cn)):
                j = self.nodes_to_edge(a,b)
                mab = self.Nnodes() + j
                midpoints.append(mab)

            if len(cn)==3: # tri
                for i in range(3):
                    new_cells[4*c+i,:3] = [ midpoints[(i-1)%3], cn[i], midpoints[i]]
                # plus the one in the center:
                new_cells[4*c+3,:3] = midpoints
            else: # quad
                n_ctr = self.Nnodes() + self.Nedges() + c
                for i in range(4):
                    new_cells[4*c+i] =  [ midpoints[(i-1)%4], cn[i], midpoints[i], n_ctr]

        # 3. generic construction of edges from cells
        # try to use the same subclass as the original grid, so we'll have the same
        # fields available.  
        # OLD: gr = ug.UnstructuredGrid()
        gr = self.__class__()
        gr.from_simple_data(points = new_points,
                            cells = new_cells)

        gr.make_edges_from_cells()
        copy_edge_attributes_to_refined(self,gr)
        copy_cell_attributes_to_refined(self,gr)

        # Not sure whether it's better to put this here, or in the
        # untrim subclass.  Really the whole approach is specific to
        # orthogonal grids, but so far the only grids using this code
        # are orthogonal.

        # quad centers may be very close to a (typ. LAND) edge
        # this will turn the resulting cells into triangles.
        gr.collapse_short_edges()
        # which deletes some edges and nodes, so renumber
        gr.renumber()
        
        return gr

    def edges_as_nodes_cells_mark(self):
        """ returns the edges definition int the legacy PTM grid format,
            in the form of an array:
            [left node, right node, left cell, right cell, boundary marker]
        """
        ptm_sides = np.zeros([len(self.edges),5],np.int32)
        ptm_sides[:,0:2] = self.edges['nodes']
        ptm_sides[:,2:4] = self.edges['cells']
        ptm_sides[:,4] = self.edges['mark']
        return ptm_sides       

    def write_cells_shp(self,shpname,extra_fields=[],overwrite=True):
        """ extra_fields is a list of lists, 
            with either 3 items:
              # this was is obsolete.
              extra_fields[i] = (field_name,field_type, lambda cell_index: field_value)
              field type must be a numpy type - int32,float64, etc.     
            or 2 items:
              extra_fields[i] = (field_name,numpy array)
        """
        # assemble a numpy struct array with all of the info 
        # seems that having an object references in there is unstable,
        # so pass geometries in a list separately.
        base_dtype =[('poly_id1',np.int32),
                     ('area',np.float64),
                     ('volume',np.float64),
                     ('depth_mean',np.float64)]

        try:
            cell_depths_max = self.cell_depths_max()
            extra_fields.append( ('depth_max',np.float64, lambda i: cell_depths_max[i]) )
        except:
            pass

        for efi in range(len(extra_fields)):
            fname,fdata=extra_fields[efi]
            base_dtype.append( (fname,fdata.dtype) )

        cell_data = np.zeros(self.Ncells(), dtype=base_dtype)

        for efi in range(len(extra_fields)):
            fname,fdata=extra_fields[efi]
            cell_data[fname]=fdata

        self.update_cell_edges()

        cell_geoms = [None]*self.Ncells()
        
        cell_data['depth_mean'] = self.cell_depths()
        cell_data['area']=self.cells_area()
        cell_data['volume']=cell_data['depth_mean']*cell_data['area']
        cell_data['poly_id1'] = 1+np.arange(self.Ncells())

        for poly_id in range(self.Ncells()):
            if poly_id % 500 == 0:
                print( "%0.2g%%"%(100.*poly_id/self.Ncells()) )

            # older code put this together manually.
            cell_geoms[poly_id]=self.cell_polygon(poly_id)

        print( cell_data.dtype )
        wkb2shp.wkb2shp(shpname,input_wkbs=cell_geoms,fields=cell_data,
                        overwrite=overwrite)

    def write_shore_shp(self,shpname,geom_type='polygon'):
        poly=self.boundary_polygon()
        if geom_type=='polygon':
            geoms=[poly]
        elif geom_type=='linestring':
            geoms=list(poly.boundary.geoms)
        wkb2shp.wkb2shp(shpname,geoms)

    def init_shp(self,shpname,geom_type):
        drv = ogr.GetDriverByName('ESRI Shapefile')
        if shpname[-4:] == '.shp':
            # remove any matching files:
            if os.path.exists(shpname):
                print( "Removing the old to make way for the new" )
                os.unlink(shpname)

        new_ds = drv.CreateDataSource(shpname)
        srs = osr.SpatialReference()
        
        # for now, assume that it's UTM Zone 10, NAD83 - 
        srs.SetFromUserInput('EPSG:26910')
        
        base,ext = os.path.splitext(os.path.basename(shpname))

        new_layer = new_ds.CreateLayer(base,srs=srs,geom_type=geom_type)
        return new_ds,new_layer


    def edge_depths(self):
        
        try:
            return self._edge_depth
        except:
            undefined_edge_depth = np.zeros(len(self.edges),np.float64)
            return undefined_edge_depth

    def write_edges_shp(self,shpname,extra_fields=[]):
        """ Write a shapefile with each edge as a polyline.
        see write_cells_shp for description of extra_fields        
        """ 
        base_dtype = [('edge_id1',np.int32),
                      ('length',np.float64),
                      ('depth_mean',np.float64)]
                      
        side_depths_mean = self.edge_depths()
        
        try:
            side_depths_max = self.side_depths_max()
            extra_fields.append( ('depth_max',np.float64, lambda e: side_depths_max[e]) )
        except:
            pass
        
                      
        for efi in range(len(extra_fields)):
            fname,ftype,ffunc = extra_fields[efi]
            if ftype == int:
                ftype = np.int32
            base_dtype.append( (fname,ftype) )
            
        edges = self.edges_as_nodes_cells_mark()
        vertices = self.nodes['x']
        
        edge_data = np.zeros(len(edges), dtype=base_dtype)
        edge_geoms = [None]*len(edges)
          
        for edge_id in range(edges.shape[0]):
            if edge_id % 500 == 0:
                print("%0.2g%%"%(100.*edge_id/edges.shape[0]))
            
            nodes = vertices[edges[edge_id,:2]]
            g = geometry.LineString(nodes)
            edge_geoms[edge_id] = g
            edge_data[edge_id]['length'] = g.length
            edge_data[edge_id]['edge_id1'] = edge_id + 1
            edge_data[edge_id]['depth_mean'] = side_depths_mean[edge_id]

            for fname,ftype,ffunc in extra_fields:
                edge_data[edge_id][fname] = ffunc(edge_id)
            
        wkb2shp.wkb2shp(shpname,input_wkbs=edge_geoms,fields=edge_data,
                        overwrite=True)

    def write_node_shp(self,shpname,extra_fields=[]):
        """ Write a shapefile with each node.  Fields will attempt to mirror
        self.nodes.dtype

        extra_fields: goal is similar to write_cells_shp and write_edges_shp, 
        but not yet supported.
        """
        assert len(extra_fields)==0 # not yet supported!

        # zero-based index of node (why does write_edge_shp create 1-based ids?)
        base_dtype = [('node_id',np.int32)]

        node_geoms=[geometry.Point( self.nodes['x'][i] )
                    for i in self.valid_node_iter() ]

        node_data=self.nodes[~self.nodes['deleted']].copy()

        # don't need to write all of the original fields out:
        node_data=utils.recarray_del_fields(node_data,['x','deleted'])

        wkb2shp.wkb2shp(shpname,input_wkbs=node_geoms,fields=node_data,
                        overwrite=True)
        
    def write_ptm_gridfile(self,fn):
        """ write this grid out in the ptm grid format.
        """
        vertex_hdr = " Vertex Data: vertex_number, x, y"
        poly_hdr = " Polygon Data: polygon_number, number_of_sides,center_x, center_y, center_depth, side_indices(number_of_sides), marker(0=internal,1=open boundary)"
        side_hdr = " Side Data: side_number, side_depth, node_indices(2), cell_indices(2), marker(0=internal,1=external,2=flow boundary,3=open boundary)"

        with open(fn,'wt') as fp:
            # write header counts
            fp.write(" Number of Vertices\n")
            fp.write(" %20d\n"%self.Nnodes())
            fp.write(" Number of Polygons\n")
            fp.write(" %20d\n"%self.Ncells())
            fp.write(" Number of Sides\n")
            fp.write(" %20d\n"%self.Nedges())
            fp.write(" NODATA (land) value\n")
            fp.write(" -9999.000000000\n")

            # write vertex info
            fp.write(vertex_hdr+"\n")
            for v in range(self.Nnodes()):
                fp.write(" %10d %16.7f %16.7f\n"%(v+1,
                                                 self.nodes['x'][v,0],
                                                 self.nodes['x'][v,1]))

            # write polygon info
            fp.write(poly_hdr+"\n")
            cell_write_str1 = " %10d %10d %16.7f %16.7f %16.7f "
            cell_depths = self.cell_depths()
            for e in range(self.Ncells()):
                edges = self.cells['edges'][e,:]
                edges[edges<0] = -1
                edge_str = " ".join( ["%10d"%(s+1) for s in edges] )
                edge_str = edge_str+" %10d\n"%(self.cells['mark'][e])
                nsides = sum(edges>=0)
                fp.write(cell_write_str1%(e+1,
                                          nsides,
                                          self.cells['_center'][e,0],
                                          self.cells['_center'][e,1],
                                          cell_depths[e]))
                fp.write(edge_str)
            
            # write side info
            fp.write(side_hdr+"\n")
            edge_depths = self.edge_depths()
            edge_write_str = " %10d %16.7f %10d %10d %10d %10d %10d\n"
            for s in range(self.Nedges()):
                edges = self.edges['cells'][s,:]
                edges[edges<0] = -1          
                nodes = self.edges['nodes'][s,:]
                nodes[nodes<0] = -1
                fp.write(edge_write_str%(s+1,
                                          edge_depths[s],
                                          nodes[0]+1,
                                          nodes[1]+1,
                                          edges[0]+1,
                                          edges[1]+1,
                                          self.edges['mark'][s])) 

    def cell_depths(self):
        try:
            return self._cell_depth
        except:
            undefined_cell_depth = np.zeros(len(self.cells),np.float64)
            return undefined_cell_depth

    #--# generation methods
    def add_rectilinear(self,p0,p1,nx,ny):
        """
        add nodes,edges and cells defining a rectilinear grid.
        nx gives the number of nodes in the x-direction (nx-1 cells)

        returns a dict with nodes=> [nx,ny] array of node indices
           cells=>[nx-1,ny-1] array of cell indices.
           currently does not return edge indices
        """
        assert self.max_sides>=4

        node_ids=np.zeros( (nx,ny), int)-1
        xs=np.linspace(p0[0],p1[0],nx)
        ys=np.linspace(p0[1],p1[1],ny)

        # create the nodes
        for xi,x in enumerate(xs):
            for yi,y in enumerate(ys):
                node_ids[xi,yi] = self.add_node(x=[x,y])

        cell_ids=np.zeros( (nx-1,ny-1), int)-1

        # create the cells
        for xi in range(nx-1):
            for yi in range(ny-1):
                nodes=[ node_ids[xi,yi],
                        node_ids[xi+1,yi],
                        node_ids[xi+1,yi+1],
                        node_ids[xi,yi+1] ]
                cell_ids[xi,yi]=self.add_cell_and_edges(nodes=nodes) 
        return {'cells':cell_ids,
                'nodes':node_ids}

    # Half-edge interface
    def halfedge(self,j,orient):
        return HalfEdge(self,j,orient)
    def nodes_to_halfedge(self,n1,n2):
        return HalfEdge.from_nodes(self,n1,n2)
    def cell_to_halfedge(self,c,i):
        j=self.cell_to_edges(c)[i]
        if self.edges['cells'][j,0]==c:
            return HalfEdge(self,j,0)
        else:
            return HalfEdge(self,j,1)

    def cell_containing(self,xy,neighbors_to_test=4):
        """ Compatibility wrapper for select_cells_nearest.  This
        may disappear in the future, depending...
        """ 
        hit = self.select_cells_nearest(xy, count=neighbors_to_test, inside=True)
        if hit is None:
            return -1
        else:
            return hit

    def cell_path(self,i):
        """
        Return a matplotlib Path object representing the closed polygon of
        cell i
        """
        cell_nodes = self.cell_to_nodes(i)
        cell_codes = np.ones(len(cell_nodes),np.int32)*Path.LINETO
        cell_codes[0] = Path.MOVETO    
        cell_codes[-1] = Path.CLOSEPOLY
        return Path(self.nodes['x'][cell_nodes])

class UGrid(UnstructuredGrid):
    def __init__(self,nc=None,grid=None):
        """
        nc: Read from a ugrid netcdf file, or netcdf object
        grid: initialize from existing UnstructuredGrid (though not necessarily an untrim grid)
        *** currently hardwired to use 1st mesh found
        """    
        super(UGrid,self).__init__()
        if nc is not None:
            if isinstance(nc,str):
                self.nc_filename = nc
                self.nc = netCDF4.Dataset(self.nc_filename,'r')
            else:
                self.nc_filename = None
                self.nc = nc
            
            self.meshes = self.mesh_names()
            # hardwired to use 1st mesh found
            self.read_from_nc_mesh(self.meshes[0])
        elif grid is not None:
            self.copy_from_grid(grid)

    def read_from_nc_mesh(self,mesh_name):
        """
        mesh_name: string with ugrid netCDF mesh name
        """
        # read number of nodes, then copy xy locations
        n_nodes = len(self.nc.dimensions['n'+mesh_name+'_node'])
        self.nodes = np.zeros(n_nodes,self.node_dtype)
        self.nodes['x'][:,0] = self.nc.variables[mesh_name+'_node_x'][...]
        self.nodes['x'][:,1] = self.nc.variables[mesh_name+'_node_y'][...]
        

        n_cells = len(self.nc.dimensions['n'+mesh_name+'_face'])
        self.cells = np.zeros(n_cells,self.cell_dtype)
        self.cells['nodes'] = self.nc.variables[mesh_name+'_face_nodes'][...]
        self._cell_depths = self.nc.variables[mesh_name+'_face_depth'][...]
        self.cells['mark'] = self.nc.variables[mesh_name+'_face_bc'][...]
        self.cells['edges'] =self.nc.variables[mesh_name+'_face_edges'][...]
        self.cells['_center'][:,0] =self.nc.variables[mesh_name+'_face_x'][...]
        self.cells['_center'][:,1] =self.nc.variables[mesh_name+'_face_y'][...]
        
        # optional cell reads
        self.cells['_area'] = np.nan   # signal stale

        n_edges = len(self.nc.dimensions['n'+mesh_name+'_edge'])
        self.edges = np.zeros(n_edges,self.edge_dtype)
        self.edges['nodes'] = self.nc.variables[mesh_name+'_edge_nodes'][...]
        self.edges['cells'] = self.nc.variables[mesh_name+'_edge_faces'][...]
        self.edges['mark'] = self.nc.variables[mesh_name+'_edge_bc'][...]
        self._edge_depth = self.nc.variables[mesh_name+'_edge_depth'][...]
                           
        self.refresh_metadata()
        self.build_node_to_cells()
        self.build_node_to_edges()
       
    #def cell_depths(self):
    #    return self._cell_depths   

    def cell_depths(self):
        
        try:
            return self._cell_depth
        except:
            undefined_cell_depth = np.zeros(len(self.cells),np.float64)
            return undefined_cell_depth

    def data_variable_names(self):
        """ return list of variables which appear to have real data (i.e. not just mesh
        geometry / topology)
        """
        data_names = []
        mesh = self.mesh_names()
        prefix = mesh[0]+'_'
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
       
        
class UnTRIM08Grid(UnstructuredGrid):
    hdr_08 = '&GRD_2008'
    hdr_old = '&LISTGRD'
    DEPTH_UNKNOWN = np.nan

    angle = 0.0
    location = "'n/a'"
    
    def __init__(self,grd_fn=None,grid=None,extra_cell_fields=[],extra_edge_fields=[]):
        """
        grd_fn: Read from an untrim .grd file
        grid: initialize from existing UnstructuredGrid (though not necessarily an untrim grid)
        """
        # NB: these depths are as soundings - positive down.
        super(UnTRIM08Grid,self).__init__( extra_cell_fields = extra_cell_fields + [('depth_mean',np.float64),
                                                                                    ('depth_max',np.float64),
                                                                                    ('red',np.bool8),
                                                                                    ('subgrid',object)],
                                           extra_edge_fields = extra_edge_fields + [('depth_mean',np.float64),
                                                                                    ('depth_max',np.float64),
                                                                                    ('subgrid',object)] )
        if grd_fn is not None:
            self.read_from_file(grd_fn)
        elif grid is not None:
            self.copy_from_grid(grid)
        
    def Nred(self):
        # nothing magic - just reads the cell attributes
        return sum(self.cells['red'])
    
    def renumber_cells_ordering(self): # untrim version
        # not sure about placement of red cells, but presumably something like this:

        # so marked, red cells come first, then marked black cells (do these exist?)
        # then unmarked red, then unmarked black.
        # mergesort is stable, so if no reordering is necessary it will stay the same.
        Nactive = sum(~self.cells['deleted'])
        return np.argsort( -self.cells['mark']*2 - self.cells['red'] + 10*self.cells['deleted'],
                           kind='mergesort')[:Nactive]

    def renumber_edges_ordering_without_delete(self): # untrim version
        # want marks==0, marks==self.FLOW, marks==self.LAND
        mark_order = np.zeros(3,np.int32)
        mark_order[0] = 0 # internal comes first
        mark_order[self.FLOW] = 1 # flow comes second
        mark_order[self.LAND] = 2 # land comes last
        return np.argsort(mark_order[self.edges['mark']],kind='mergesort')

    def copy_from_grid(self,grid):
        super(UnTRIM08Grid,self).copy_from_grid(grid)

        # now fill in untrim specific things:
        if isinstance(grid,UnTRIM08Grid):
            for field in ['depth_mean','depth_max','red']:
                self.cells[field] = grid.cells[field]
            for field in ['depth_mean','depth_max']:
                self.edges[field] = grid.edges[field]
    def renumber_edges_ordering(self): # untrim version
        # want marks==0, marks==self.FLOW, marks==self.LAND
        mark_order = np.zeros(3,np.int32)
        mark_order[0] = 0 # internal comes first
        mark_order[self.FLOW] = 1 # flow comes second
        mark_order[self.LAND] = 2 # land comes last
        Nactive = sum(~self.edges['deleted'])
        return np.argsort(mark_order[self.edges['mark']]+10*self.edges['deleted'],
                          kind='mergesort')[:Nactive]

    def copy_from_grid(self,grid):
        super(UnTRIM08Grid,self).copy_from_grid(grid)

        # now fill in untrim specific things:
        if isinstance(grid,UnTRIM08Grid):
            for field in ['depth_mean','depth_max','red']:
                self.cells[field] = grid.cells[field]
            for field in ['depth_mean','depth_max']:
                self.edges[field] = grid.edges[field]
            # Subgrid is separate
            self.cells['subgrid'] = copy.deepcopy(grid.cells['subgrid'])
            self.edges['subgrid'] = copy.deepcopy(grid.edges['subgrid'])
        else:
            # The tricky part - fabricating untrim data from a non-untrim grid:
            if 'depth' in grid.cells.dtype.names:
                d = grid.cells['depth']
            else:
                d = self.DEPTH_UNKNOWN
            self.cells['depth_mean'] = self.cells['depth_max'] = d

            if 'depth' in grid.edges.dtype.names:
                d = grid.edges['depth']
            else:
                d = self.DEPTH_UNKNOWN
            self.edges['depth_mean'] = self.edges['depth_max'] = d

            self.infer_depths()

    def infer_depths(self):
        """ fill in depth and subgrid depth as much as possible, using
        cell/edge depth information.  If cells have depth but not edges,
        copy cell to edges, or vice versa.  Then fill in subgrid for both
        cells and edges
        """
        self.infer_depths_edges_from_cells()
        self.infer_depths_cells_from_edges()
        self.copy_depths_to_subgrid(depth_stat='depth_mean')
        
    def infer_depths_edges_from_cells(self):
        """ edge depths are set to shallowest neighboring cell
        """
        sel_edges = np.isnan(self.edges['depth_mean'])

        edge_from_cell = self.cells['depth_mean'][self.edges['cells'][sel_edges]].min(axis=1)
        self.edges['depth_mean'][sel_edges] = edge_from_cell
        self.edges['depth_max'][sel_edges] = edge_from_cell
        
    def infer_depths_cells_from_edges(self):
        """ cell depths are set as deepest neighboring edge
        """
        sel_cells = np.nonzero(np.isnan(self.cells['depth_mean']))[0]
        # iterate, since number of sides varies
        edges = self.cells['edges'][sel_cells]
        edge_depths = self.edges['depth_mean'][edges]
        edge_depths[edges<0] = -np.inf # to avoid missing edges for triangles
        
        self.cells['depth_mean'][sel_cells] = edge_depths.max(axis=1)
        self.cells['depth_max'][sel_cells] = self.cells['depth_mean'][sel_cells]

    def copy_mean_depths_to_subgrid(self,overwrite=True,cells=True,edges=True):
        """
        copy all depth_mean values (all edges, all cells), to single-entry
        subgrid.

        thought mostly not implemented yet, the intended meaning of the options:
        overwrite: set to False to skip cells/edges which already have subgrid
        cells: False to skip cells, (not yet: or a bool bitmap to select a subset of cells)
        edges: same as for cells.
        """
        if cells:
            area = self.cells_area()
            depth = self.cells['depth_mean']
            # funny indexing to add unit dimension,
            # and zip to make these into tuples like ([area[0]],[depth[0]])
            for c in range(self.Ncells()):
                if overwrite or self.cells['subgrid'][c]==0:
                    self.cells['subgrid'][c] = (area[c,None],depth[c,None])
        if edges:
            length = self.edges_length()
            depth = self.edges['depth_mean']
            # funny indexing to add unit dimension,
            # and zip to make these into tuples like ([area[0]],[depth[0]])
            for j in range(self.Nedges()):
                if overwrite or self.edges['subgrid'][j]==0:
                    self.edges['subgrid'][j] = (length[j,None],depth[j,None])
        
    def copy_mean_depths_to_subgrid_outside_polygon(self,polygon_shp,overwrite=True,cells=True,edges=True):
        """
        copy all depth_mean values (all edges, all cells), to single-entry
        subgrid.

        thought mostly not implemented yet, the intended meaning of the options:
        overwrite: set to False to skip cells/edges which already have subgrid
        cells: False to skip cells, (not yet: or a bool bitmap to select a subset of cells)
        edges: same as for cells.
        """

        subgrid_regions = self.read_polygon_shp(polygon_shp)

        if cells:
            area = self.cells_area()
            depth = self.cells['depth_mean']
            # funny indexing to add unit dimension,
            # and zip to make these into tuples like ([area[0]],[depth[0]])
            for c in range(self.Ncells()):
                if overwrite or self.cells['subgrid'][c]==0:
                    cell_nodes = self.cells['nodes'][c]
                    print( cell_nodes )
                    print( self.nodes['x'][cell_nodes] )
                    cell_poly = geometry.Polygon(np.asarray(self.nodes['x'][cell_nodes]))
                    # check for intersection (boolean)
                    intersect = self.check_for_intersection(cell_poly,subgrid_regions)
                    # reset depths OUTSIDE of intersection region
                    if not intersect:
                        self.cells['subgrid'][c] = (area[c,None],depth[c,None])

        if edges:
            length = self.edges_length()
            depth = self.edges['depth_mean']
            # funny indexing to add unit dimension,
            # and zip to make these into tuples like ([area[0]],[depth[0]])
            for j in range(self.Nedges()):
                if overwrite or self.edges['subgrid'][j]==0:
                    edge_nodes = self.edges['nodes'][j]
                    edge_line = geometry.LineString(self.nodes['x'][edge_nodes])
                    # check for intersection (boolean)
                    intersect = self.check_for_intersection(edge_line,subgrid_regions)
                    # reset depths OUTSIDE of intersection region
                    if not intersect:
                        self.edges['subgrid'][j] = (length[j,None],depth[j,None])

    def check_for_intersection(self, geom, regions):

        intersect_sum = 0.0
        for region in regions:
            intersect = geom.intersection(region).area
            if (type(geom) == 'Polygon'):
                intersect_sum += geom.intersection(region).area
            else: # assume line
                intersect_sum += geom.intersection(region).length
        
        if intersect_sum > 0.0:
            intersect = True
        else:
            intersect = False

        return intersect
        
    def read_polygon_shp(self,fn):
        ods = ogr.Open(fn)
        layer = ods.GetLayer(0)

        polygons = []  

        while 1:
            feat = layer.GetNextFeature()
            if feat is None:
                break
            geo = feat.GetGeometryRef()

            if geo.GetGeometryName() != 'POLYGON':
                raise GridException("All features must be polygons")
            poly = wkb.loads( geo.ExportToWkb() )
            if len(poly.interiors) > 0:
                raise GridException("No support for interior rings")
            polygons.append( poly )

        polygons=polygons
            
        return polygons

    def overwrite_field(self,cells=None,edges=None,source='depth_max',target='depth_mean'):
        """
        overwrite one field in structure with another

        the initial purpose of this was to overwrite the mean depth with the max
          depth along edges intersected by thalweg line. 
        """
        if cells is not None:
            self.cells[target][cells]=self.cells[source][cells]
        if edges is not None:
            self.edges[target][edges]=self.edges[source][edges]

    def copy_depths_to_subgrid(self,cells='missing',edges='missing',depth_stat='depth_mean'):
        """
        copy depth_mean values to single-entry subgrid.

        cells/edges possible values:
          'missing': only fill in subgrid entries which are either 0 or empty lists
          array of integers: select specific elements by index
          array of booleans: select specific elements by bitmask
          False: skip entirely
        """
        def depth_stat_to_subgrid(selector,elements,extent,depth):
            """ slightly confusing generic operation for cells and edges
            selector is 'missing',an index array, or a bitmask array
            elements is self.cells or self.edges
            extent is area or length
            depth is ... depth_mean or depth_max
            """
            if selector is 'missing':
                overwrite=False
                selector = range(len(elements))
            else:
                overwrite=True
                selector = np.asarray(selector)
                if selector.dtype == np.bool8:
                    selector = np.nonzero( selector )[0]

            # so now selector is iterable, but still doesn't account for deleted elements
            for s in selector:
                if elements['deleted'][s]:
                    continue
                if overwrite or elements['subgrid'][s]==0 or elements['subgrid'][s] == ([],[]):
                    # funny indexing to add unit dimension,
                    # and zip to make these into tuples like ([area[0]],[depth[0]])
                    elements['subgrid'][s] = (extent[s,None],depth[s,None])

        if cells is not False:
            depth_stat_to_subgrid(cells,self.cells,self.cells_area(),self.cells[depth_stat])
        if edges is not False:
            depth_stat_to_subgrid(edges,self.edges,self.edges_length(),self.edges[depth_stat])
        
    def read_from_file(self,grd_fn):
        """ 
        read untrim format from the the given filename

        when called, from __init__, self.fp is already pointing
        to the grid file, but nothing has been read
        """
        self.grd_fn = grd_fn
        self.fp = open(self.grd_fn,'rt')
        hdr = self.fp.readline().strip() #header &GRD_2008 or &LISTGRD

        if hdr == self.hdr_08:
            print( "Will read 2008 format for grid" )
            n_parms = 11
        elif hdr == self.hdr_old:
            print( "Will read old UnTRIM grid format" )
            n_parms = 10

        for i in range(n_parms):  # ignore TNE and TNS in new format files
            l = self.fp.readline()
            lhs,rhs = l.split('=')
            val = rhs.strip().strip(',')
            varname = lhs.strip()
            print( "%s=%s"%(varname,val) )

            if varname=='NV':
                Nvertices = int(val)
            elif varname=='NE':
                Npolys = int(val)
            elif varname=='NS':
                Nsides = int(val)
            elif varname=='NBC':
                Nboundary_poly = int(val)
            elif varname=='NSI':
                Ninternal_sides = int(val)
            elif varname=='NSF':
                Nflow_sides = int(val)
            elif varname=='NBC':
                Nbc = int(val)
            elif varname=='ANGLE':
                self.angle = float(val)
            elif varname=='LOCATION':
                self.location = val
            elif varname=='NR':  ## these are read, but not used
                Nred = int(val)
            elif varname=='TNE':
                TNE=int(val)
            elif varname=='TNS':
                TNS=int(val)
            # others: HLAND for older fmt.
        
        while 1:
            s = self.fp.readline().strip() # header:  /
            if s == '/':
                break

        # We know the size of everything, and can ask UnstructuredGrid to allocate
        # arrays now, with the 'special' meaning that passing an integer means allocate
        # the array of that size, full of zeros.
        # this allocates
        #  self.nodes, self.edges, self.cells
        self.from_simple_data(points = Nvertices,edges = Nsides, cells = Npolys)

        for v in range(Nvertices):
            Cv = self.fp.readline().split()
            if hdr == self.hdr_08:
                vertex_num = int(Cv.pop(0))
                if vertex_num != v+1:
                    print( "Mismatched vertex numbering: %d != %d"%(vertex_num,v+1) )
            self.nodes['x'][v,0] = float(Cv[0])
            self.nodes['x'][v,1] = float(Cv[1])
            
        print( "Npolys",Npolys )
        self.cells['edges'] = self.UNKNOWN # initialize all
        self.cells['nodes'] = self.UNKNOWN
        
        for c in range(Npolys):
            l = self.fp.readline()
            Cp = l.split()
            if hdr == self.hdr_08:
                poly_num = int(Cp.pop(0))
                if poly_num-1 != c:
                    print( "Mismatched polygon id: %fd != %d"%(poly_num,c+1) )
                
            numsides = int(Cp[0])

            self.cells['_center'][c,0] = float(Cp[1])
            self.cells['_center'][c,1] = float(Cp[2])

            if hdr == self.hdr_old:
                # vertex index is Cp[3,5,7,9]
                # the others, 4,6,8,10, are edges, right?
                # convert to 0 based indices here

                # This is probably wrong! I think it's actually reading the
                # sides
                self.cells['edges'][c,0] = int(Cp[4]) - 1
                self.cells['edges'][c,1] = int(Cp[6]) - 1 
                self.cells['edges'][c,2] = int(Cp[8]) - 1
                if numsides == 4:
                    self.cells['edges'][c,3] = int(Cp[10]) - 1 
                else:
                    self.cells['edges'][c,3]=self.UNDEFINED
                #HERE - need to copy that to self.cells['nodes']
            else:
                for ei in range(numsides):
                    self.cells['nodes'][c,ei] = int(Cp[3+ei]) - 1
                    self.cells['edges'][c,ei] = int(Cp[3+numsides+ei]) - 1
                self.cells['nodes'][c,numsides:]=self.UNDEFINED
                self.cells['edges'][c,numsides:]=self.UNDEFINED
        
        # choose some large, above-sea-level depth
        self.cells['depth_mean'] = -1000 # not sure this is doing anything...

        for e in range(Nsides):
            Cs = self.fp.readline().split()
            if hdr == self.hdr_08:
                # side num = int(Cs.pop(0))
                Cs.pop(0)
            elif hdr == self.hdr_old:
                # side depth?
                edge_depth = self.edges['depth_mean'][e] = float(Cs.pop(0))
                
            self.edges['nodes'][e,0] = int(Cs[0])-1  # vertex indices
            self.edges['nodes'][e,1] = int(Cs[1])-1
            
            self.edges['cells'][e,0] = int(Cs[2])-1  # cell neighbors
            self.edges['cells'][e,1] = int(Cs[3])-1

            if hdr == self.hdr_old:
                for nc in self.edges['cells'][e]:
                    if nc >= 0 and edge_depth > self.cells['depth_mean'][nc]:
                        self.cells['depth_mean'][nc] = edge_depth

        if hdr==self.hdr_old:
            # old format - have to infer cell nodes from edges
            self.make_cell_nodes_from_edge_nodes()

        # Try to make sense of the marks and red/black:
        self.cells['red'][:Nred] = True
        self.cells['mark'][:Nboundary_poly] = self.BOUNDARY
        self.edges['mark'][:Ninternal_sides] = 0
        self.edges['mark'][Ninternal_sides:Nflow_sides] = self.FLOW
        self.edges['mark'][Nflow_sides:] = self.LAND

        # Bathymetry:
        if hdr == self.hdr_08:
            # make a cheap tokenizer to read floats across lines
            # note that it's up to the user to know that all values from
            # the line are read, and not to get the iterator until you're
            # ready for some values to be read
            def tokenizer():
                while True:
                    for item in self.fp.readline().split():
                        yield item
            for c in range(Npolys):
                check_c,nis = [int(s) for s in self.fp.readline().split()]
                if check_c != c+1:
                    print("ERROR: while reading cell subgrid, cell index mismatch: %s vs. %d"%(c+1,check_c))
                
                next_token = tokenizer().next
                areas = np.array( [float(next_token()) for sg in range(nis)] )
                depths = np.array( [float(next_token()) for sg in range(nis)] )
                    
                self.cells['depth_mean'][c] = np.sum(areas*depths) / np.sum(areas)
                self.cells['_area'][c] = np.sum(areas)
                self.cells['depth_max'][c] = depths.max()
                self.cells['subgrid'][c] = (areas,depths)
            for e in range(Nflow_sides):
                l = self.fp.readline()
                # print "%d/%d - Read line: %s"%(e,self.Nsides,l)
                check_e,nis = [int(s) for s in l.split()]
                if check_e != e+1:
                    print( "ERROR: While reading edge subgrid, edge index mismatch: %s vs. %s"%(e+1,check_e) )
                next_token = tokenizer().next
                lengths = np.array( [float(next_token()) for sg in range(nis)] )
                depths =  np.array( [float(next_token()) for sg in range(nis)] )
                if sum(lengths)<=0:
                    print( "edge %d has bad lengths"%e )
                self.edges['depth_mean'][e] = np.sum(lengths*depths) / sum(lengths)
                self.edges['depth_max'][e]  = depths.max()
                self.edges['subgrid'][e] = (lengths,depths)
            # and land boundaries get zeros.
            for e in range(Nflow_sides,Nsides):
                self.edges['depth_mean'][e] = 0.0
                self.edges['depth_max'][e] = 0.0
                self.edges['subgrid'][e] = ([],[])
                
    # Some subgrid specific stuff:
    def Nsubgrid_cells(self):
        # equivalent to TNE
        return sum( [len(sg[0]) for sg in self.cells['subgrid'] if sg!=0] )
    
    def Nsubgrid_edges(self):
        # equivalent to TNS
        return sum( [len(sg[0]) for sg in self.edges['subgrid'] if sg!=0] )

    def write_untrim08(self,fn):
        """ write this grid out in the untrim08 format.
        Note that for some fields (red/black, subgrid depth), if this
        grid doesn't have that field, this code will fabricate the data
        and probably not very well.
        """
        with open(fn,'wt') as fp:
            fp.write(self.hdr_08+"\n")

            n_parms = 11

            Nland = sum(self.edges['mark']==self.LAND)
            Nflow = sum(self.edges['mark']==self.FLOW)
            Ninternal = sum(self.edges['mark']==0)
            Nbc = sum(self.cells['mark'] == self.BOUNDARY)


            fp.write("NV      =%d,\n"%self.Nnodes())
            fp.write("NE      =%d,\n"%self.Ncells())
            fp.write("NR      =%d,\n"%self.Nred()) 
            fp.write("NS      =%d,\n"%self.Nedges())
            fp.write("NSI     =%d,\n"%Ninternal)
            fp.write("NSF     =%d,\n"%(Ninternal+Nflow))
            fp.write("NBC     =%d,\n"%Nbc)
            fp.write("TNE     =%d,\n"%self.Nsubgrid_cells()) 
            fp.write("TNS     =%d,\n"%self.Nsubgrid_edges())  
            fp.write("ANGLE   =%.4f,\n"%self.angle)
            fp.write("LOCATION=%s\n"%self.location)

            fp.write("/\n")

            for v in range(self.Nnodes()):
                fp.write("%10d %13.4f %15.4f\n"%(v+1,
                                                 self.nodes['x'][v,0],
                                                 self.nodes['x'][v,1]))
            # cell lines are like:
            #  1  4  490549.7527  4176428.3398   31459  30777  31369  31716    3  1  49990 2
            # idx     center_x     center_y      nodes---------------------    edges--------
            #    Nsides

            # Edge lines are like:
            #     49990  31369  31716   1   0
            #     idx    nodes-------   cells-- 0 if boundary
            centers = self.cells_center()
            
            for c in range(self.Ncells()):
                edges = self.cell_to_edges(c)
                nodes = self.cell_to_nodes(c)

                nsides = len(edges)

                fp.write("%10d %14d %13.4f %17.4f "%(c+1,nsides,centers[c,0],centers[c,1]))
                edge_str = " ".join( ["%14d"%(e+1) for e in edges] )
                node_str = " ".join( ["%14d"%(n+1) for n in nodes] )
                fp.write(node_str+" "+edge_str+"\n")

            for e in range(self.Nedges()):
                fp.write("%10d %14d %14d %14d %14d\n"%(e+1,
                                                       self.edges['nodes'][e,0]+1,self.edges['nodes'][e,1]+1,
                                                       self.edges['cells'][e,0]+1,self.edges['cells'][e,1]+1))

            # since we have to do this 4 times, make a helper function
            def fmt_wrap_lines(fp,values,fmt="%14.4f ",per_line=10):
                """ write values out to file fp with the given string format, but break
                the lines so no more than per_line values on a line
                ends with a newline
                """
                for i,a in enumerate(values):
                    if i>0 and i%10==0:
                        fp.write("\n")
                    fp.write("%14.4f "%a)
                fp.write("\n")
                
            # subgrid bathy
            for c in range(self.Ncells()):
                areas,depths = self.cells['subgrid'][c]
                nis = len(areas)

                fp.write("%14d %14d\n"%(c+1,nis))
                fmt_wrap_lines(fp,areas)
                fmt_wrap_lines(fp,depths)

            edge_lengths = self.edges_length()

            for e in range(Ninternal+Nflow):
                lengths,depths = self.edges['subgrid'][e]
                nis = len(lengths)

                fp.write("%10d %9d\n"%(e+1,nis))
                fmt_wrap_lines(fp,lengths)
                fmt_wrap_lines(fp,depths)

    def global_refine(self):
        """
        UnTRIM specialization for grid refinement
        """
        gr = super(UnTRIM08Grid,self).global_refine()
        gr.infer_depths()
        gr.location = self.location
        gr.angle = self.angle
        gr.cells['red'][:] = True
        return gr

class SuntansGrid(UnstructuredGrid):
    """ Read/write suntans formatted grids
    """
    max_sides=3
    
    def __init__(self,suntans_path,elev2depth=False):
        super(SuntansGrid,self).__init__()
        self.read_from_file(suntans_path)
    def read_from_file(self,suntans_path):
        points=np.loadtxt(os.path.join(suntans_path,'points.dat'))
        points=points[:,:2]
        
        # node,node,mark,cell,cell
        edges=np.loadtxt(os.path.join(suntans_path,'edges.dat'),
                           dtype=np.int32)
        cell_dtype=[ ('center',np.float64,2),
                     ('nodes',np.int32,3),
                     ('nbrs',np.int32,3) ]
        cells=np.loadtxt(os.path.join(suntans_path,'cells.dat'),
                         dtype=cell_dtype)

        self.from_simple_data(points=points,edges=edges[:,:2],cells=cells['nodes'])
        self.edges['mark']=edges[:,2]
        self.edges['cells']=edges[:,3:]
        self.cells['_center']=cells['center']
        # assume that the edges are present, and we can figure out cell adjacency
        # from there

class Sms2DM(UnstructuredGrid):
    """ 
    Handling specific to SMS 2dm files, such as the ones used
    in Bay Delta SCHISM

    Parsing taken from sms2gr3.py, DWR Bay Delta Schism.
    """
    max_sides=3
    def __init__(self,grd_fn,elev2depth=False):
        super(Sms2DM,self).__init__( extra_cell_fields=[('depth',np.float64)],
                                     extra_node_fields=[('depth',np.float64)],
                                     extra_edge_fields=[('depth',np.float64)] )
        self.read_from_file(grd_fn,elev2depth=elev2depth)

    def read_from_file(self,grd_fn,elev2depth=False):
        self.grd_fn = grd_fn

        with open(self.grd_fn,'rt') as fp:
            all_lines = fp.readlines()

        print( "Total lines in input file: %s" % len(all_lines) )

        # E3T <id> <n1> <n2> <n3> <marker>
        elementlines = [line.strip().split()[1:5] for line in all_lines 
                        if line.startswith("E3T")]
        nelement = len(elementlines)

        # ND <id> <x> <y> <z>
        nodelines = [line.strip().split()[1:5] for line in all_lines[nelement:] 
                     if line.startswith("ND")]
        nnode  = len(nodelines)
        last_possible_node = nnode + nelement

        # none of these in bay_delta_74.2dm
        nodestrlines = [line.strip() for line in all_lines[last_possible_node:] 
                        if line.startswith("NS")]
        nnodestrlines = len(nodestrlines)

        # none of these in bay_delta_74.2dm
        boundlines = [line.strip() 
                      for line in all_lines[last_possible_node+nnodestrlines:] 
                      if line.startswith("BC")]

        print( "N nodes: %s" % nnode )
        print( "N element: %s" % nelement )

        # allocates the arrays:
        self.from_simple_data(points=nnode,cells=nelement)

        for i,nodeinfo in enumerate(nodelines):
            xyz=[float(x) for x in nodeinfo[1:]]
            self.nodes['x'][i,:] = xyz[:2]
            self.nodes['depth'][i]= xyz[2]
            node_id = int(nodeinfo[0])        
            assert node_id == (i+1)
        if elev2depth:
            self.nodes['depth'][:] *= -1
        #else:  # this isn't defined in the original code.
        #    adjust_height(nodes)   

        for i,eleminfo in enumerate(elementlines):
            elem_id, n0, n1, n2 = [int(x) for x in eleminfo]
            self.cells['nodes'][i,:3] = n0,n1,n2
        # collectively make those 0-based:
        self.cells['nodes'][:,:3] -= 1
        if self.max_sides>3:
            self.cells['nodes'][:,3:] = self.UNDEFINED # all triangles

        # additional aspects of the file are not yet implemented.

        # boundnodestrings = []
        # boundid = []
        # startnew = True
        # for line in nodestrlines:
        #     if startnew:
        #         latest = []
        #         boundnodestrings.append(latest)
        #         startnew = False
        #     items = [int(x) for x in line.split()[1:]]
        #     if items[-2] < 0:
        #         startnew = True
        #         items[-2] *= -1
        #         latest += items[0:-1]
        #         boundid.append(items[-1])
        #     else: 
        #         latest += items
        # nboundsegs = len(boundnodestrings)

        # bc_regex = re.compile(r"""BC\s+\d+\s+\"(land|open|island)\"\s+(\d+).*\nBC_DEF\s+\d+\s+\d+\s+\"(.*)\"\s+\d+\s+\"(.*)\".*""")

        # boundary_defs = {}
        # for m in bc_regex.finditer(string.join(boundlines,"\n")):
        #     btype = m.group(1)
        #     bdef_id = int(m.group(2))
        #     assert m.group(3) == "name"
        #     name = m.group(4)
        #     boundary_defs[bdef_id] = (bdef_id,btype,name)

        # boundaries = []

        # for line in boundlines:
        #     if line.startswith("BC_VAL"):
        #         items = string.split(line)[1:]
        #         entity_id, def_id, param_id = [int(x) for x in items[1:-1]]
        #         name = items[-1]
        #         boundary_def = boundary_defs[def_id]
        #         bc = Boundary(name, boundary_def[1], np.array(boundnodestrings[entity_id-1], dtype = "int"))
        #         boundaries.append(bc)


    
class PtmGrid(UnstructuredGrid):
    def __init__(self,grd_fn):
        super(PtmGrid,self).__init__( extra_cell_fields=[('depth',np.float64)],
                                      extra_edge_fields=[('depth',np.float64)] )
        
        self.read_from_file(grd_fn)

    def read_from_file(self,grd_fn):
        self.grd_fn = grd_fn
        self.fp = open(self.grd_fn,'rt')
        
        while True:
            line = self.fp.readline()
            if line == '':
                break
            line = line.strip()

            if line.find('Number of Vertices')>= 0:
                Nvertices = int(self.fp.readline().strip())
            elif line.find('Number of Polygons')>=0:
                Npolys = int(self.fp.readline().strip())
            elif line.find('Number of Sides')>=0:
                Nsides = int(self.fp.readline().strip())
            elif line.find('NODATA (land) value')>=0:
                nodata_value = float(self.fp.readline().strip())
            elif line.find('Vertex Data:') >= 0:
                vertex_data_offset = self.fp.tell()
                for i in range(Nvertices):
                    self.fp.readline()
            elif line.find('Polygon Data:') >= 0:
                polygon_data_offset = self.fp.tell()
                for i in range(Npolys):
                    self.fp.readline()
            elif line.find('Side Data:') >= 0:
                side_data_offset = self.fp.tell()
                for i in range(Nsides):
                    self.fp.readline()
            else:
                pass # print "Skipping line: ",line

        # allocate
        self.from_simple_data(points=Nvertices,edges=Nsides,cells=Npolys)

        self.read_vertices(vertex_data_offset)
        self.read_polygons(polygon_data_offset)
        self.read_sides(side_data_offset)
        self.update_cell_nodes()
            
    def read_vertices(self,vertex_data_offset):
        print( "Reading vertices" )
        self.fp.seek(vertex_data_offset)
        
        for i in range(self.Nnodes()):
            line = self.fp.readline().split()
            self.nodes['x'][i,:] = map(float,line[1:])
    
    def read_polygons(self,polygon_data_offset):
        print( "Reading polygons" )

        self.fp.seek(polygon_data_offset)

        self.cells['nodes'][:,:] = -1
        self.cells['edges'][:,:] = -1
        
        # polygons stored as indices into edge array,
        # and triangles have the 4th index set to -1
        # grd numbers polygons starting with 1, so these
        # indices will be off by 1.
        
        for i in range(self.Ncells()):
            line = self.fp.readline().split()
            # polygon_number, number_of_sides,center_x, center_y, center_depth, side_indices(number_of_sides), marker(0=internal,1=open boundary)
            poly_id = int(line[0])
            nsides_this_poly = int(line[1])
            self.cells['depth'][i] = float(line[4])

            # grab all of the edge indices:
            self.cells['edges'][i,:nsides_this_poly] = [int(s)-1 for s in line[5:5+nsides_this_poly]]

    def read_sides(self,side_data_offset):
        print( "Reading sides" )
        # Side Data: side_number, side_depth, node_indices(2), cell_indices(2), marker(0=internal,1=external,2=flow boundary,3=open boundary)

        self.fp.seek(side_data_offset)
        
        # store nodeA,nodeB, cell1,cell2, marker
        for i in range(self.Nedges()):
            line = self.fp.readline().split()
            side_id = int(line[0])
            self.edges['depth'][i] = float(line[1])
            self.edges['nodes'][i,:] = [int(s)-1 for s in line[2:4]]
            self.edges['cells'][i,:] = [int(s)-1 for s in line[4:6]]
            self.edges['mark'][i] = int(line[6])
