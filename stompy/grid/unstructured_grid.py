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
                     orient_intersection,array_append,within_2d, to_unit, progress,
                     dist_along, recarray_add_fields,recarray_del_fields,
                     point_segment_distance)

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

class AmbiguousMeshException(GridException):
    def __init__(self,*a,meshes=[],**k):
        super().__init__(*a,**k)
        self.meshes=meshes

def request_square(ax,max_bounds=None):
    """
    Attempt to set a square aspect ratio on matplotlib axes ax
    max_bounds: if specified, adjust axes to include that area,
     unless ax is already zoomed in smaller than bounds
    """
    # in older matplotlib, this was sufficient:
    # ax.axis('equal')
    # But in newer matplotlib, if the axes are shared,
    # that fails.
    # Maybe this is better?  But it artificially squeezes the
    # plot box sometimes.
    # plt.setp(ax,aspect=1.0,adjustable='box-forced')

    #plt.setp(ax,aspect=1.0,adjustable='datalim')
    # this is rearing its head again.  see this thread:
    # https://github.com/matplotlib/matplotlib/issues/11416
    # not sure if there is a nice way around it at this point.
    
    plt.setp(ax,aspect='equal')
    if max_bounds is not None:
        bounds=ax.axis()
        if ( (bounds[0]>=max_bounds[0]) and
             (bounds[1]<=max_bounds[1]) and
             (bounds[2]>=max_bounds[2]) and
             (bounds[3]<=max_bounds[3]) ):
            pass #
        else:
            ax.axis(max_bounds)


def find_common_nodes(gA,gB,tol=0.0):
    """
    Return a list of [ (nA0, nB0), ... ]
    for nodes in A and B with exactly the same xy coordinates (default,
    or within tol distance of each other for tol>0.

    Used for merging grids.
    """
    dupes=[]
    xys={}
    if tol==0.0:
        for n in gA.valid_node_iter():
            k=tuple(gA.nodes['x'][n])
            xys[k]=n
        for n in gB.valid_node_iter():
            k=tuple(gB.nodes['x'][n])
            if k in xys:
                dupes.append( [xys[k],n] )
    else:
        # Much slower, but can allow a bit of slop:
        for n_B in gB.valid_node_iter():
            n_A=gA.select_nodes_nearest(gB.nodes['x'][n_B],max_dist=tol)
            if n_A is not None:
                dupes.append( [n_A,n_B] )
    return dupes

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

    def normal(self):
        """
        Unit vector perpendicular to edge, pointing toward cell.  
        """
        # for orient==0, the sign of the normal is opposite of the grid's
        # edge normal (grid normal is from c1 to c2, but here orient=0 means
        # towards c1).
        return self.grid.edges_normals(self.j) * (-1)**(1-self.orient)

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
    OPEN=4  # edge mark - typ. ocean boundary.

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

    node_dtype = [ ('x',(np.float64,2)),('deleted',np.bool_) ]
    node_defaults=None
    cell_dtype  = [ # edges/nodes are set dynamically in __init__ since max_sides can change
                    ('_center',(np.float64,2)),  # typ. voronoi center
                    ('mark',np.int32),
                    ('_area',np.float64),
                    ('deleted',np.bool_)]
    cell_defaults=None
    edge_dtype = [ ('nodes',(np.int32,2)),
                   ('mark',np.int32),
                   ('cells',(np.int32,2)),
                   ('deleted',np.bool_)]
    edge_defaults=None

    ##
    
    filename=None

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

        self.update_element_defaults()

        self.edge_defaults['cells']=self.UNMESHED
        self.cell_defaults['edges']=self.UNKNOWN

        if grid is not None:
            self.copy_from_grid(grid)
        else:
            self.from_simple_data(points=points,edges=edges,cells=cells)
            
    def update_element_defaults(self):
        """
        Allocate or update default 'template' values for the 3 types of elements.
        Existing default values are copied based on name.
        New default values get whatever np.zeros gives them, with the exception of
        floating that get nan and object which gets None.
        """
        def copy_and_update(default,new_dtype):
            if default is None:
                return np.zeros( (), new_dtype)
            if default.dtype == new_dtype:
                return default # already fine.
            new_def=np.zeros( (), new_dtype )
            for name in new_def.dtype.names:
                if name in default.dtype.names:
                    if new_def[name].shape == default[name].shape:
                        new_def[name]=default[name]
                    else:
                        # For now just allow differences in the length of a vector
                        # i.e. when changing max_sides
                        assert new_def[name].ndim==default[name].ndim,"ndim changed: %s"%name
                        max_len=min(len(new_def[name]),len(default[name]))
                        # in case new_def is longer -- will fill with the first element
                        # of default
                        new_def[name][:] = default[name][0]
                        new_def[name][:max_len] = default[name][:max_len]
                elif np.issubdtype(new_def[name].dtype,np.floating):
                    new_def[name]=np.nan
                elif np.issubdtype(new_def[name].dtype,np.object_):
                    new_def[name]=None
                # otherwise whatever np.zeros serves up.
            return new_def
                    
        self.node_defaults=copy_and_update(self.node_defaults,self.node_dtype)
        self.edge_defaults=copy_and_update(self.edge_defaults,self.edge_dtype)
        self.cell_defaults=copy_and_update(self.cell_defaults,self.cell_dtype)

    def copy(self):
        # maybe subclasses shouldn't be used here - for example,
        # this requires that every subclass include 'grid' in its
        # __init__.  Maybe more subclasses should just be readers?

        # Old approach
        # return UnstructuredGrid(grid=self)

        # But I want to preserve more of the structures
        g=UnstructuredGrid(max_sides=self.max_sides)
        
        g.cell_dtype=self.cell_dtype
        g.edge_dtype=self.edge_dtype
        g.node_dtype=self.node_dtype

        g.cells=self.cells.copy()
        g.edges=self.edges.copy()
        g.nodes=self.nodes.copy()

        g.cell_defaults=self.cell_defaults.copy()
        g.edge_defaults=self.edge_defaults.copy()
        g.node_defaults=self.node_defaults.copy()

        g.refresh_metadata()
        return g

    def copy_from_grid(self,grid):
        """
        Copy topology from grid to self, replacing any existing
        grid information.
        In the future this may also handle copying additional node/edge/cell
        fields, but right now does NOT.
        """
        # this takes care of allocation, and setting the most basic topology
        cell_nodes=grid.cells['nodes']
        if self.max_sides < grid.max_sides:
            assert np.all(cell_nodes[:,self.max_sides:]<0),"Trying to copy from grid with greater max_sides"
            cell_nodes=cell_nodes[:,:self.max_sides]
        
        self.from_simple_data(points=grid.nodes['x'],
                              edges=grid.edges['nodes'],
                              cells=cell_nodes)
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

        TODO: some grids (DFM) have additional fields that depend on 
        max_sides, such as face_x_bnd. Would be nice to have a way of
        detecting that and resizing those fields as well.
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
        # this will handle the change in shape:
        self.update_element_defaults()
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

    def remove_duplicates(self):
        cc=self.cells_center()

        to_delete=[]
        # have a duplicate triangle:
        for c in range(self.Ncells()):
            other=self.select_cells_nearest(cc[c])
            if other!=c:
                logging.warning("deleting duplicate cell %d"%c)
                to_delete.append(c)

        if to_delete:
            for n in to_delete:
                self.delete_cell(n)
            self.renumber()

    @staticmethod
    def from_trigrid(g):
        return UnstructuredGrid(edges=g.edges,points=g.points,cells=g.cells)

    @staticmethod
    def read_gmsh(fn):
        """
        Limited read support for ASCII gmsh output
        """
        with open(fn,'rt') as fp:
            head_start=fp.readline()
            assert head_start.strip()=='$MeshFormat'
            ver,asc_bin,data_size = fp.readline().strip().split()[:3]
            assert asc_bin=='0',"Not ready for binary"
            head_end=fp.readline()
            assert head_end.strip()=='$EndMeshFormat'

            while 1:
                blk_start=fp.readline().strip()
                if blk_start=='':
                    break
                logging.info("Reading %s"%blk_start)
                assert blk_start[0]=='$'

                blk_end=blk_start.replace('$','$End')

                if blk_start=='$Entities':
                    npoints,ncurves,nsurf,nvols=[int(s) for s in fp.readline().strip().split()]
                    # Not worrying about these now...
                elif blk_start=='$Nodes':
                    nblocks,nnodes,min_node_tag,max_node_tag = [int(s) for s in fp.readline().strip().split()]
                    node_xy=np.nan*np.zeros((max_node_tag-min_node_tag+1,2))

                    for blk_i in range(nblocks):
                        ent_dim,ent_tag,param,blk_nnodes=[int(s) for s in fp.readline().strip().split()]
                        tags=[]
                        xyzs=[]
                        for blk_n in range(blk_nnodes):
                            tags.append(int(fp.readline().strip()))
                        for blk_n in range(blk_nnodes):
                            xyz=[float(s) for s in fp.readline().strip().split()]
                            xyzs.append(xyz)
                        for tag,xyz in zip(tags,xyzs):
                            node_xy[tag-min_node_tag,:]=xyz[:2]
                elif blk_start=='$Elements':
                    nblocks,nelts,min_tag,max_tag = [int(s) for s in fp.readline().strip().split()]

                    tris=[]

                    for blk_i in range(nblocks):
                        ent_dim,ent_tag,ent_typ,blk_nelts=[int(s) for s in fp.readline().strip().split()]
                        elt_nnodes={15:1, # nodes
                                    1:2,  # edges
                                    2:3}  # triangles
                        tags=[]

                        for blk_nelt in range(blk_nelts):
                            tags=[int(s) for s in fp.readline().strip().split()]
                            elt_tag=tags[0]
                            node_tags=tags[1:]
                            assert len(node_tags)==elt_nnodes[ent_typ]
                            if ent_typ==2:
                                tris.append(node_tags)
                    tris=np.array(tris)-min_node_tag

                while fp.readline().strip()!=blk_end:
                    pass
        g=UnstructuredGrid(max_sides=3)
        g.from_simple_data(points=node_xy,cells=tris)
        g.make_edges_from_cells_fast()
        g.update_cell_edges() # or could set to unknown
        return g

    def write_gmsh_geo(self,fn):
        """
        Limited writing of gmsh geometry input file
        This will likely acquire more parameters.  For now, it
        writes nodes, edges, boundary cycles, and a single plane
        surface for the whole domain.  It's not for writing out
        cells (except one large cell for the area).
        """
        with open(fn,'wt') as fp:
            fp.write("// Boundary geometry\n")
            el=self.edges_length()
            for n in range(self.Nnodes()):
                p=self.nodes['x'][n]
                #  s=scale(p)
                js=self.node_to_edges(n)
                s=el[js].max()
                # x,y,z, scale
                fp.write("Point(%d) = {%.4f, %.4f, 0, %g};\n"%
                         (n+1,p[0],p[1],s))
            for j in range(self.Nedges()):
                n12=self.edges['nodes'][j]
                fp.write("Line(%d) = {%d, %d};\n"%(j+1,n12[0]+1,n12[1]+1))

            cycles=self.find_cycles(max_cycle_len=self.Nnodes())

            for cyc_i,cycle in enumerate(cycles):
                cycle=cycles[0]

                j1_list=[]
                for a,b in zip(cycle,np.roll(cycle,-1)):
                    j=self.nodes_to_edge(a,b)
                    if self.edges['nodes'][j,0]==a:
                        j1=j+1
                    else:
                        j1=-(j+1)
                    j1_list.append(j1)

                fp.write("Curve Loop(%d) = {%s};\n"%
                         (cyc_i+1,", ".join(["%d"%j1 for j1 in j1_list])))

            fp.write("Plane Surface(1) = {%s};\n"%
                     ", ".join([str(c+1) for c in range(len(cycles))]))
    
    @staticmethod
    def read_ugrid(*a,**kw):
        return UnstructuredGrid.from_ugrid(*a,**kw)

    @staticmethod
    def read_untrim(*a,**kw):
        return UnTRIM08Grid(*a,**kw)
    
    @staticmethod
    def from_ugrid(nc,mesh_name=None,skip_edges=False,fields='auto',
                   auto_max_bytes_per_element=256,dialect=None):
        """ extract 2D grid from netcdf/ugrid
        nc: either a filename or an xarray dataset.
        fields: 'auto' [new] populate additional node,edge and cell fields
        based on matching dimensions.  In case the source file is a full 
        run, 'large' variables are not included in auto, determined by
          whether each element has a size greater than auto_max_bytes_per_element.
          'all' will load all fields, regardless of size.
          a list of names will load those specific variables.

        dialect: ad-hoc support for slight variants on the format.
          'fishptm' for reading ptm hydro files in nc format where the names
          are standardized but a mesh variable is not set.

        Also sets grid.nc_meta to be a dictionary with the relevant attributes
        from the mesh variable (i.e. node_dimension, edge_dimension, face_dimension,
         face_node_connectivity, edge_node_connectivity, and optionally face_edge
         connectivity, edge_face_connectivity, etc.)
        """
        if isinstance(nc,six.string_types):
            filename=nc
            if dialect=='fvcom':
                # with SFBOFS output, ran into dimension/variable naming
                # clashes, which are skirted by these options:
                nc=xr.open_dataset(nc,
                                   decode_times=False,decode_coords=False,
                                   drop_variables=['siglay','siglev'])
            else:
                nc=xr.open_dataset(nc)
        else:
            filename=None

        # fields which will be ignored for fields='auto', presumably because
        # they are grid geometry/topology, and already handled.
        ignore_fields=[]
            
        if dialect=='fishptm':
            mesh_name='Mesh2'
            nc[mesh_name]=(),1
            nc[mesh_name].attrs.update(dict(cf_role='mesh_topology',
                                            node_coordinates='Mesh2_node_x Mesh2_node_y',
                                            face_node_connectivity='Mesh2_face_nodes',
                                            edge_node_connectivity='Mesh2_edge_nodes',
                                            face_edge_connectivity='Mesh2_face_edges',
                                            edge_face_connectivity='Mesh2_edge_faces',
                                            node_dimension='nMesh2_node',
                                            edge_dimension='nMesh2_edge',
                                            face_dimension='nMesh2_face',
                                            face_coordinates='Mesh2_face_x Mesh2_face_y',
                                            edge_coordinates='Mesh2_edge_x Mesh2_edge_y'))
            # This is straight from the output, so no need to add bathy offset
            #grid.add_cell_field('z_bed',-grid.cells['Mesh2_face_depth'])

        if dialect=='fvcom': # specifically the SFBOFS field output
            ds['mesh']=1
            ds['faces']=ds['nv'].transpose()
            ds['faces'].attrs['start_index']=1
            ds['mesh'].attrs.update( dict(cf_role='mesh_topology',
                                          node_coordinates='x y',
                                          face_node_connectivity='faces', 
                                          #edge_node_connectivity='Mesh2_edge_nodes',
                                          #face_edge_connectivity='Mesh2_face_edges',
                                          #edge_face_connectivity='Mesh2_edge_faces',
                                          node_dimension='node',
                                          #edge_dimension='nMesh2_edge',
                                          face_dimension='nele',
                                          face_coordinates='xc yc',
                                          #edge_coordinates='Mesh2_edge_x Mesh2_edge_y'
            ))
            
        if mesh_name is None:
            meshes=[]
            for vname in nc.variables.keys():
                if nc[vname].attrs.get('cf_role',None) == 'mesh_topology':
                    meshes.append(vname)
            if len(meshes)!=1:
                raise AmbiguousMeshException("Could not uniquely determine mesh variable",
                                             meshes=meshes)
            mesh_name=meshes[0]

        mesh = nc[mesh_name]

        node_x_name,node_y_name = mesh.node_coordinates.split()

        ignore_fields.extend([node_x_name,node_y_name])
        node_x=nc[node_x_name]
        node_dimension=node_x.dims[0] # save for tracking metadata
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
            ignore_fields.append(varname)
            ncvar=nc[varname]
            idxs=ncvar.values
            try:
                start_index=ncvar.attrs['start_index']
            except KeyError:
                start_index=None

            if start_index not in [0,1]:
                max_idx=np.nanmax(idxs)
                if max_idx==len(node_xy):
                    start_index=1
                    logging.warning("Variable %s has bad start_index, assume %d"%(varname,start_index))
                elif max_idx==len(node_xy)-1:
                    if start_index is not None:
                        # This is the default, so only complain if something erroneous was specified.
                        logging.warning("Variable %s has bad start_index, assume 0"%(varname))
                    start_index=0
                else:
                    assumed=0
                    if start_index is not None:
                        logging.warning("Variable %s has bad start_index, punting with %d"%(varname,assumed))
                    start_index=assumed

            idxs = idxs - start_index
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

        # remember the cell dimension from that:
        cell_dimension=nc[mesh.face_node_connectivity].dims[0]

        
        # suntans has a nonstandard, but not always specified, fill value.
        faces[faces>=len(node_x)]=UnstructuredGrid.UNDEFINED
        if 'edge_node_connectivity' in mesh.attrs:
            edges = process_as_index(mesh.edge_node_connectivity) # [N,2]
            edge_dimension=nc[mesh.edge_node_connectivity].dims[0]
            ug=UnstructuredGrid(points=node_xy,cells=faces,edges=edges)
        else:
            ug=UnstructuredGrid(points=node_xy,cells=faces)
            ug.make_edges_from_cells()
            edge_dimension=None

        # When the incoming netcdf supplies additional topology, use it
        if 'face_edge_connectivity' in mesh.attrs:
            v=mesh.attrs['face_edge_connectivity']
            if v in nc:
                # Some files advertise variable they don't have. Trust no one.
                ug.cells['edges'] = nc[mesh.attrs['face_edge_connectivity']].values
        if 'edge_face_connectivity' in mesh.attrs:
            v=mesh.attrs['edge_face_connectivity']
            if v in nc:
                ug.edges['cells'] = nc[mesh.attrs['edge_face_connectivity']].values
        
        if dialect=='fishptm':
            ug.cells_center()
            ug.cells_area()

            mark = ug.cells['mark'] = nc['Mesh2_face_bc'].values
            ug.cells['mark'][mark<3] = 0  # different markers for suntans
            ug.cells['mark'][mark==3] = 1 # different markers for suntans
            ug.edges['mark'] = nc['Mesh2_edge_bc'].values
            ug.add_cell_field('cell_depth',nc['Mesh2_face_depth'].values)
            ug.add_edge_field('edge_depth',nc['Mesh2_edge_depth'].values)

        if 'face_coordinates' in mesh.attrs:
            face_x,face_y = mesh.face_coordinates.split()
            ug.cells['_center'][:,0] = nc[face_x].values
            ug.cells['_center'][:,1] = nc[face_y].values
            ignore_fields.extend([face_x,face_y])
            
        if fields is not None: # fields=='auto':
            # doing this after the fact is inefficient, but a useful
            # simplification during development
            for dim_attr,struct,adder in [('node_dimension',ug.nodes,ug.add_node_field),
                                          ('edge_dimension',ug.edges,ug.add_edge_field),
                                          ('face_dimension',ug.cells,ug.add_cell_field)]:
                dim_name=mesh.attrs.get(dim_attr,None)
                if not dim_name and dim_attr=='node_dimension':
                    # Should be able to make a good guess based on node coordinates
                    dim_name=nc[node_x_name].dims[0]
                if dim_name:
                    prefix='_'+dim_name+'_' # see below, for uniquifying fields
                    
                    for vname in nc.data_vars:
                        if vname in ignore_fields:
                            continue # skip things like node_x
                        if len(nc[vname].dims)==0:
                            continue
                        if nc[vname].dims[0]!=dim_name:
                            continue # might be too restrictive

                        if fields=='all':
                            pass
                        elif fields=='auto':
                            # Check size
                            if nc[vname].size / len(struct) > auto_max_bytes_per_element:
                                continue
                        elif vname not in fields:
                            continue
                        
                        struct_vname=vname
                        # Undo the uniquifying code in write_ugrid
                        # This allows for a field like 'mark' to
                        # exist in UnstructuredGrid from both edges
                        # and cells, but on writing to netcdf
                        # one or both will get _<dim_name>_ prefix
                        if struct_vname.startswith(prefix):
                            struct_vname=struct_vname[len(prefix):]
                            
                        if struct_vname in struct.dtype.names:
                            # already exists, just copy, and hope data type and shape
                            # are ok.  
                            struct[struct_vname]=nc[vname].values
                        else:
                            adder( struct_vname, nc[vname].values )
                    

        if dialect=='fishptm':
            ug.add_cell_field('z_bed',-ug.cells['Mesh2_face_depth'])

        # Set the ugrid metadata on the grid itself to help downstream processing
        ug.nc_meta={k: mesh.attrs.get(k,None)
                    for k in ['node_dimension','edge_dimension','face_dimension',
                              'face_node_connectivity','edge_node_connectivity',
                              'face_edge_connectivity','edge_face_connectivity',
                              'node_coordinates','face_coordinates','edge_coordinates']}
        # node_dimension is often omitted, but easy to figure out
        if ug.nc_meta['node_dimension'] is None:
            ug.nc_meta['node_dimension']=node_dimension
        if ug.nc_meta['face_dimension'] is None:
            ug.nc_meta['face_dimension']=cell_dimension
        if ug.nc_meta['edge_dimension'] is None:
            ug.nc_meta['edge_dimension']=edge_dimension
        
        ug.filename=filename
        return ug

    @staticmethod
    def read_tin_xml(fn):
        from ..io import landxml
        P,F = landxml.read_tin_xml_mmap(fn)
        gtin=UnstructuredGrid(points=P[:,:2],
                              cells=F)
        gtin.make_edges_from_cells_fast()
        gtin.add_node_field('elev',P[:,2])
        return gtin
    
    # COMING SOON
    #@staticmethod
    #def read_rgfgrid(grd_fn):
    #    HERE

    @staticmethod
    def read_gr3(fn):
        return SchismGrid.read_gr3(fn)
    
    @staticmethod
    def read_ras2d(hdf_fname, twod_area_name=None, elevations=True, subedges=None):
        """
        Read a RAS2D grid from HDF.
        hdf_fname: path to hdf file
        twod_area_name: string naming the 2D grid in the HDF file.  If omitted,
          attempt to detect this, but will fail if the number of valid names
          is not exactly 1.
        subedges: check for face internal points and populate coordinate strings in this
         field. stompy stores the full geometry here. If subedges is None, skip processing.
         otherwise all edges will get an entry, and faces with no internal points will 
         just get a copy of the simple face geometry.

        Acknowledgements: Original code by Stephen Andrews.

        elevations: If elevation-related fields (currently just min elevation in 
          cells and faces) are found, add them to the grid.
        """
        import h5py # import here, since most of stompy does not rely on h5py
        h = h5py.File(hdf_fname, 'r')

        try:
            cc_suffixes=None
            if twod_area_name is None:
                names=list(h['Geometry/2D Flow Areas'].keys())
                twod_names={} # name => dict of info
                for name in names:
                    for version,suffix in [('v6','Cells Center Coordinate'),
                                           ('v2025','Cell Coordinates')]:
                        try:
                            cc_variable='Geometry/2D Flow Areas/'+name+'/'+suffix
                            h[cc_variable]
                            twod_names[name]=dict(cc_variable=cc_variable,version=version)
                            break
                        except KeyError:
                            continue
                if len(twod_names)==1:
                    twod_area_name=list(twod_names)[0]
                elif len(twod_names)>1:
                    raise Exception("Must specify twod_area_name from %s"%( ", ".join(twod_names) ))
                else:
                    raise Exception("No viable twod_area_name values")

            twod_info = twod_names[twod_area_name]
            cell_center_xy = h[twod_info['cc_variable']]
            cell_center_xy=np.array(cell_center_xy) # a bit faster?

            ncells = len(cell_center_xy)
            ccx = np.array([cell_center_xy[i,0] for i in range(ncells)])
            ccy = np.array([cell_center_xy[i,1] for i in range(ncells)])

            if twod_info['version']=='v6':
                points_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/FacePoints Coordinate']
            elif twod_info['version']=='v2025':
                points_xy = h['Geometry/2D Flow Areas/' + twod_area_name + '/Node Coordinates']
            else:
                raise Exception("Unknown version: "+ twod_info['version'])
                
            points_xy=np.array(points_xy)

            npoints = len(points_xy)
            points = np.zeros((npoints, 2), np.float64)
            for n in range(npoints):
                points[n, 0] = points_xy[n,0]
                points[n, 1] = points_xy[n,1]

            if twod_info['version']=='v6':
                edge_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces FacePoint Indexes']
                edge_nodes=np.array(edge_nodes)
            elif twod_info['version']=='v2025':
                edge_data = np.array(h['Geometry/2D Flow Areas/' + twod_area_name + '/Face Data'])
                edge_nodes = edge_data[:,2:] # cellA,cellB, facepoint A, facepoint B
            else:
                raise Exception("Unknown version: "+ twod_info['version'])
                
            nedges = len(edge_nodes)
            edges = -1 * np.ones((nedges, 2), dtype=int)
            for j in range(nedges):
                edges[j][0] = edge_nodes[j,0]
                edges[j][1] = edge_nodes[j,1]

            if subedges is not None:
                extra_edge_fields=[(subedges,object)]
            else:
                extra_edge_fields=[]

            if twod_info['version']=='v6':
                cell_nodes = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cells FacePoint Indexes']
                cell_nodes=np.array(cell_nodes)
                max_cell_faces=cell_nodes.shape[1]
                ncells=len(cell_nodes) # can be smaller than len(cell_center_xy) ?
            elif twod_info['version']=='v2025':
                # I think this is a list of face indexes
                # and it has Start and Count attributes giving the data for each cell.
                # There is also Node Data, similarly formatted, associating nodes to cells.
                # Neither of these directly give order of nodes within a cell.
                ragged_cell_faces = h['Geometry/2D Flow Areas/' + twod_area_name + '/Cell Data']
                cell_face_start = ragged_cell_faces.attrs['Start']
                cell_face_count = ragged_cell_faces.attrs['Count']
                # convert to non-ragged.
                max_cell_faces=cell_face_count.max()
                ncells=len(cell_face_count)
                cell_nodes=np.full((ncells,max_cell_faces),-1)
                # This feels very slow
                for i in range(ncells):
                    faces = ragged_cell_faces[cell_face_start[i]:cell_face_start[i]+cell_face_count[i]]
                    for f in range(len(faces)):
                        f_this = faces[f]
                        f_next = faces[(f+1)%len(faces)]
                        n_this = edge_nodes[f_this]
                        n_next = edge_nodes[f_next]
                        if n_this[0] in n_next:
                            assert n_this[1] not in n_next
                            cell_nodes[i,f] = n_this[1]
                        else:
                            cell_nodes[i,f] = n_this[0]
            else:
                raise Exception("Unknown version: "+ twod_info['version'])
                
            for i in range(len(cell_nodes)):
                if cell_nodes[i,2] < 0:  # first ghost cell (which are sorted to end of list)
                    ncells=i # don't count ghost cells
                    break
            cells = -1 * np.ones((ncells, max_cell_faces), dtype=int)
            for i in range(ncells):
                for k in range(max_cell_faces):
                    cells[i][k] = cell_nodes[i][k]

            grd = UnstructuredGrid(edges=edges, points=points,
                                   cells=cells, max_sides=max_cell_faces,
                                   extra_edge_fields=extra_edge_fields)
            grd.twod_area_name = twod_area_name

            if len(ccx) > grd.Ncells():
                N=grd.Ncells()
                print(f"{len(ccx)-N} apparent ghost cell centers")
                grd.ghost_cc=np.c_[ ccx[N:], ccy[N:]]
                ccx=ccx[:N]
                ccy=ccy[:N]
            grd.cells['_center'][:,0] = ccx
            grd.cells['_center'][:,1] = ccy
            
            if elevations:
                cell_key='Geometry/2D Flow Areas/' + twod_area_name + '/Cells Minimum Elevation'
                edge_key='Geometry/2D Flow Areas/' + twod_area_name + '/Faces Minimum Elevation'
                if cell_key in h:
                    Nc=grd.Ncells() # to trim out trailing ghost cells
                    grd.add_cell_field('cell_z_min',
                                       h[cell_key][:Nc])
                if edge_key in h:
                    grd.add_edge_field('edge_z_min',
                                       h[edge_key])
            if subedges is not None:
                if twod_info['version']=='v6':
                    try:
                        # index,count for each face into perimeter values
                        internal_info  = h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces Perimeter Info']
                        internal_start = internal_info[:,0]
                        internal_count = internal_info[:,1]
                        # combined array of points, [N,{x,y}]
                        internal_values= h['Geometry/2D Flow Areas/' + twod_area_name + '/Faces Perimeter Values']
                    except KeyError:
                        # File not written with subedge geometry
                        internal_info=None
                        internal_values=None
                elif twod_info['version']=='v2025':
                    try:
                        # combined array of points, [N,{x,y}]
                        internal_values  = h['Geometry/2D Flow Areas/' + twod_area_name + '/Face Internal Points']
                        internal_count = internal_values.attrs['Count']
                        internal_start = internal_values.attrs['Start']
                    except KeyError:
                        print("Failed to find subedge info")
                        # File not written with subedge geometry
                        internal_info=None
                        internal_values=None
                else:
                    raise Exception("Unknown version: "+ twod_info['version'])
                        
                    
                edge_nodes=np.array(edge_nodes)
                nedges = len(edge_nodes)
                edges = -1 * np.ones((nedges, 2), dtype=int)
                for j in range(grd.Nedges()):
                    points = grd.nodes['x'][grd.edges['nodes'][j]]
                    if internal_values is not None:
                        start=internal_start[j]
                        count=internal_count[j]
                        if count>0:
                            points=np.concatenate( [points[:1,:],
                                                    internal_values[start:start+count],
                                                    points[1:,:]],axis=0)
                    grd.edges[subedges][j]=points
                    
            return grd
        finally:
            h.close()

    @staticmethod
    def read_dfm(nc=None,fn=None,
                 cells_from_edges='auto',max_sides=6,cleanup=False):
        """
        Migrating this to here from model.delft.dfm_grid

        nc: An xarray dataset or path to netcdf file holding the grid
        fn: path to netcdf file holding the grid (redundant with nc)
        cells_from_edges: 'auto' create cells based on edges if cells do not exist in the dataset
          specify True or False to force or disable this.
        max_sides: maximum number of sides per cell, used both for initializing datastructures, and
          for determining cells from edge connectivity.
        cleanup: for grids created from multiple subdomains, there are sometime duplicate edges and nodes.
          this will remove those duplicates, though there are no guarantees that indices are
          preserved.

        todo: populate nc_meta similar to how from_ugrid() does.
        """
        filename=None

        if nc is None:
            assert fn
            filename=fn
            nc=xr.open_dataset(fn)

        if isinstance(nc,str):
            filename=nc
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
                cells[bad]=-1.0 # used to be 0, and after the cast
                cells=cells.astype(np.int32)

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
            kwargs['max_sides']=max_sides

        # Partition handling - at least the output of map_merge
        # does *not* remap indices in edges and cells
        if 'partitions_node_start' in nc.variables:
            nodes_are_contiguous = np.all( np.diff(nc.partitions_node_start.values)
                                           == nc.partitions_node_count.values[:-1] )
            assert nodes_are_contiguous, "Merged grids can only be handled when node indices are contiguous"
        else:
            nodes_are_contiguous=True

        if 'partitions_edge_start' in nc.variables:
            edges_are_contiguous = np.all( np.diff(nc.partitions_edge_start.values)
                                           == nc.partitions_edge_count.values[:-1] )
            assert edges_are_contiguous, "Merged grids can only be handled when edge indices are contiguous"
        else:
            edges_are_contiguous=True

        if 'partitions_face_start' in nc.variables:
            faces_are_contiguous = np.all( np.diff(nc.partitions_face_start.values)
                                           == nc.partitions_face_count.values[:-1] )
            assert faces_are_contiguous, "Merged grids can only be handled when face indices are contiguous"
            if cleanup:
                log.warning("Some MPI grids have duplicate cells, which cannot be cleaned, but cleanup=True")
        else:
            face_are_contiguous=True

        g=UnstructuredGrid(**kwargs)
        g.filename=filename

        if cells_from_edges:
            print("Making cells from edges")
            g.make_cells_from_edges()

        if var_depth in nc.variables: # have depth at nodes
            g.nodes['depth']=nc[var_depth].values.copy()

        if 'NetLinkType' in nc.variables:
            # typ: 0: 2D closed, 1: 2D open, 2: 1D open
            g.add_edge_field('NetLinkType',nc['NetLinkType'].values)

        if 'iglobal_s' in nc.variables:
            g.add_cell_field('iglobal_s',nc['iglobal_s'].values)

        if cleanup:
            # defined below
            cleanup_dfm_multidomains(g)
        # I think this helps on windows.
        nc.close()
        return g

    @staticmethod
    def read_delft_curvilinear(fn,**kw):
        return RgfGrid(fn,**kw)
    
    @staticmethod
    def read_suntans_hybrid(path='.',points='points.dat',edges='edges.dat',cells='cells.dat'):
        """
        For backwards compatibility.  Better to use read_suntans which auto-detects
        format.
        """
        return UnstructuredGrid.read_suntans(path=path,points=points,edges=edges,cells=cells,
                                             dialect='hybrid')
    @staticmethod
    def read_suntans_classic(path='.',points='points.dat',edges='edges.dat',cells='cells.dat'):
        """
        For backwards compatibility.  Better to use read_suntans which auto-detects
        format.
        """
        return UnstructuredGrid.read_suntans(path=path,points=points,edges=edges,cells=cells,
                                             dialect='classic')

    @staticmethod
    def read_suntans(path='.',points='points.dat',edges='edges.dat',cells='cells.dat',
                     edgedata=True,celldata=True,subdomain=None,
                     dialect='auto'):
        """
        Read text-based suntans format which can accomodate arbitrary numbers of sides.
        This can be read/written by Janet.
        dialect: 'auto' = attempt classic fall back to hybrid
         'classic': assume classic, fail otherwise
         'hybrid': assume hybrid, fail otherwise

        edgedata: True: try to find and load an edgedata file.
        celldata: same, but not yet implemented
        subdomain: if an integer, append that to the filenames to load a specific subdomain grid.
        """
        points_fn=os.path.join(path,points)
        edges_fn=os.path.join(path,edges)
        cells_fn=os.path.join(path,cells)

        if subdomain is not None:
            suffix=".%d"%subdomain
            # points_fn is shared
            edges_fn+=suffix
            cells_fn+=suffix

        xyz=np.loadtxt(points_fn)
        edge_nnmcc=np.loadtxt(edges_fn).astype('i4')

        if edge_nnmcc.shape[1]==6:
            # partitioned output has an extra colum - all zero in my
            # one glance.
            logging.debug("Dropping 6th column from edge data")
            edge_nnmcc=edge_nnmcc[:,:5]

        # if we read classic, this gets reset to 3
        # otherwise, why not leave some breathing room?
        max_sides=8
        cell_nodes=[]
        cell_ccs=[]
        cell_nbrs=[]

        # classic suntans has lines ctrx ctry node1 node2 node3 nbrcell1 nbrcell2 nbrcell3
        if dialect in ['auto','classic']:
            cell_dtype=[ ('center',np.float64,2),
                         ('nodes',np.int32,3),
                         ('nbrs',np.int32,3) ]
            try:
                # seems that numpy will not reliably fail when there is an
                # extra column of data.
                raw_cells=np.loadtxt(cells_fn)
                if raw_cells.shape[1]!=8:
                    raise ValueError("Looks like a hybrid file with all triangles")
                cells=np.zeros(len(raw_cells),dtype=cell_dtype)
                cells['center']=raw_cells[:,0:2]
                cells['nodes']=raw_cells[:,2:5]
                cells['nbrs']=raw_cells[:,5:8]
            except ValueError:
                if dialect=='auto':
                    cells=None
                else:
                    raise Exception("Failed to parse %s as classic suntans"%cells_fn)
            if cells is not None:
                max_sides=3
                dialect='classic' # prevent fall-through to hybrid
                all_cells=cells['nodes']
                all_cc=cells['center']

        if dialect in ['auto','hybrid']:
            with open(cells_fn,'rt') as fp:
                for line in fp:
                    parts=line.strip().split()
                    try:
                        nsides=int(parts[0])
                    except ValueError:
                        logging.error("Error parsing nsides in suntans hybrid -- maybe a classic suntans grid?",
                                      exc_info=True)
                        raise
                    max_sides=max(nsides,max_sides)
                    cc=[float(p) for p in parts[1:3]]
                    nodes=[int(p) for p in parts[3:3+nsides]]
                    nbrs=[int(p) for p in parts[3+nsides:3+2*nsides]]
                    cell_nodes.append(nodes)
                    cell_ccs.append(cc)
                    cell_nbrs.append(nbrs)

            for n,nbr in zip(cell_nodes,cell_nbrs):
                extra=max_sides-len(n)
                if extra:
                    n.extend([-1]*extra)
                    nbr.extend([-1]*extra)
            all_cells=np.array(cell_nodes)
            all_cc=np.array(cell_ccs)

        g=UnstructuredGrid(max_sides=max_sides)
        g.from_simple_data(points=xyz[:,:2],
                           edges=edge_nnmcc,
                           cells=all_cells)
        g.cells['_center']=all_cc

        if edgedata:
            if edgedata is True:
                edgedata=edges_fn.replace('edges','edgedata')
            if os.path.exists(edgedata):
                edata=np.loadtxt(edgedata)
                # should be same as calculated edges_length()
                g.add_edge_field('df',edata[:,0])
                # voronoi spacing
                g.add_edge_field('dg',edata[:,1])
                # edge normal
                g.add_edge_field('n1',edata[:,2])
                g.add_edge_field('n2',edata[:,3])
                # number of edge layers with cells on both sides
                g.add_edge_field('Nke',edata[:,6].astype(np.int32))
                # number of edge layers with cells on at least one side.
                g.add_edge_field('Nkc',edata[:,7].astype(np.int32))
                # these are technically redundant with the internally calculated
                # edges['cells'].
                g.add_edge_field('nc1',edata[:,8].astype(np.int32))
                g.add_edge_field('nc2',edata[:,9].astype(np.int32))
                g.add_edge_field('gradf1',edata[:,10].astype(np.int32))
                g.add_edge_field('gradf2',edata[:,11].astype(np.int32))
                g.edges['mark']=edata[:,12].astype(np.int32)
        if celldata:
            if celldata is True:
                celldata=cells_fn.replace('cells','celldata')
            if os.path.exists(celldata):
                with open(celldata,'rt') as fp:
                    dv=np.zeros(g.Ncells(),np.float64)
                    Nk=np.zeros(g.Ncells(),np.int32)

                    for i,line in enumerate(fp):
                        parts=line.strip().split()
                        # too lazy to make this dynamic, but note that
                        # there isn't a big change -- just the first entry
                        if 1: # hybrid
                            nsides=int(parts.pop(0))
                        else:
                            nsides=3 # triangles for always

                        # xv, yv...
                        dv[i]=float(parts[3])
                        Nk[i]=int(parts[4])
                        # for now ignore the rest since they are
                        # redundant

                # depth at voronoi point
                g.add_cell_field('dv',dv)
                # layers
                g.add_cell_field('Nk',Nk)
        return g

    @staticmethod
    def read_sms(fname,open_marker=2,land_marker=1):
        """
        Ported from https://github.com/rustychris/stomel/blob/master/src/trigrid.py
        """
        fp = open(fname)

        # skip leading blank lines
        while 1:
            line = fp.readline().strip()
            if line != "":
                break

        # first non-blank line has number of cells and edges:
        Ncells,Npoints = map(int,line.split())

        # each point has three numbers, though the third is apparently
        # always 0

        points = np.zeros((Npoints,2),np.float64)
        for i in range(Npoints):
            line = fp.readline().split()
            # pnt_id is 1-based
            pnt_id = int(line[0])

            points[pnt_id-1] = [float(s) for s in line[1:3]]

        logging.info("Reading cells")
        max_sides=12 # aim high...
        real_max_sides=3 # but track what the real value is
        cells = np.zeros((Ncells,max_sides),np.int32) # store zero-based indices, and assume
        cells[:]=-1 
        cell_mask = np.ones( len(cells) )

        # for now, everything is a triangle. need sample input with quads to test
        # non-triangle code.
        for i in range(Ncells):
            parsed = [int(s) for s in fp.readline().split()]
            cell_id = parsed[0]
            nvertices = parsed[1]
            pnt_ids = np.array(parsed[2:])

            if nvertices>real_max_sides: real_max_sides=nvertices
            
            # store them as zero-based.  Point indices beyond nvertices
            # already initialized to -1 above.
            cells[cell_id-1,:nvertices] = pnt_ids - 1

        cells=cells[:,:real_max_sides]
        
        g=UnstructuredGrid(max_sides=real_max_sides)
        g.from_simple_data(points=points,cells=cells)

        # At this point we have enough info to create the edges
        g.make_edges_from_cells()

        # And then go back and switch the marker for some of the edges:
        logging.info("Reading boundaries")
        def read_first_int():
            return int(fp.readline().split()[0])

        for btype in ['open','land']:
            if btype == 'open':
                marker = open_marker # open - not sure if this is 2 or 3...
            else:
                marker = land_marker # closed
                
            n_boundaries = read_first_int()
            logging.info("Number of %s boundaries: %d"%(btype,n_boundaries))
            tot_boundary_nodes = read_first_int() # who cares...

            for boundary_i in range(n_boundaries):
                logging.info("Reading %s boundary %d"%(btype,boundary_i+1))
                n_nodes_this_boundary = read_first_int()
                for i in range(n_nodes_this_boundary):
                    node_i = read_first_int() - 1 # zero-based
                    if i>0:
                        # update the marker in edges
                        if node_i < last_node_i:
                            pa,pb = node_i,last_node_i
                        else:
                            pa,pb = last_node_i,node_i
                            
                        edge_i = g.nodes_to_edge((pa,pb))
                        if edge_i is None:
                            logging.error("Couldn't find edge %d-%d"%(pa,pb))
                            raise
                        g.edges['mark'][edge_i] = marker
                            
                    last_node_i = node_i

        logging.debug("Done reading sms grid")
        return g

    def write_gr3(self,fn,z_flip='node_z_bed',z='depth',bc_marks='auto'):
        """
        Write Schism-compatible gr3 grid to fn
        z: ndarray of values to write, or name of node field to use.
        z_flip: name of positive-up node field, which will be negated and
          written out

        bc_marks: 'auto' write boundary markers if self['bc_id'] exists
         True: write bc_marks, and fail if field doesn't exist
         False: skip marks even if present
        """
        import pandas as pd

        has_bc_data=('bc_id' in self.edges.dtype.names)
        if bc_marks=='auto':
            bc_marks=has_bc_data
        elif bc_marks:
            assert has_bc_data
            
        fp=open(fn,'wt')
        name=getattr(self,'name','unnamed')
        fp.write(name+"\n")
        fp.write("%d %d\n"%(self.Ncells(),self.Nnodes()))
        node_df=pd.DataFrame()
        node_df['id']=1+np.arange(self.Nnodes())
        node_df['x']=self.nodes['x'][:,0]
        node_df['y']=self.nodes['x'][:,1]
        if isinstance(z,np.ndarray) or np.isreal(z):
            node_df['z']=z
        elif z in self.nodes.dtype.names:
            node_df['z']=self.nodes[z] # pretty sure these are positive down
        elif z_flip in self.nodes.dtype.names:
            node_df['z']=-self.nodes[z_flip]

        fp.write( node_df.to_csv(None,index=False,sep=' ',header=False,float_format="%.6f") )

        cell_df=pd.DataFrame(self.cells['nodes']+1).reset_index() # to 1-based
        cell_df['index']+=1 # back to 1-based
        cell_df['nnodes']=(self.cells['nodes']>=0).sum(axis=1)
        cell_df=cell_df[ ['index','nnodes',0,1,2,3] ] # reorder columns
        cell_df[3] = np.where( cell_df[3]>0, cell_df[3], np.nan)
        fp.write( cell_df.to_csv(None,index=False,sep=' ',header=False,
                                 float_format='%d') )

        # 1 = Number of open boundaries
        # 13 = Total number of open boundary nodes
        # 13 = Number of nodes for open boundary 1
        if bc_marks:
            for bc_type,bc_label in [ (self.OPEN,'open'),
                                      (self.LAND,'land') ]:
                bc_mask=self.edges['mark']==bc_type
                if np.any(bc_mask):
                    n_bc=self.edges['bc_id'][bc_mask].max() + 1
                else:
                    n_bc=0
                fp.write("%d = Number of %s boundaries\n"%(n_bc,bc_label))

                #          number of edges    1 extra node per linestring
                nodes_total = bc_mask.sum() + n_bc
                fp.write("%d = Total number of %s boundary nodes\n"%(nodes_total,bc_label))
                self.edge_to_cells()
                exterior_nodes=self.boundary_cycle()
                for bc_id in range(n_bc):
                    bc_id_mask=bc_mask & (self.edges['bc_id']==bc_id)
                    n_bc_id=bc_id_mask.sum() + 1

                    # Now the goal is to look at the nodes
                    # grab an edge, scan to the start of the linestring, and output in order
                    j0=np.nonzero(bc_id_mask)[0][0]
                    he=self.halfedge(j0,1) # have it face out so we traverse the outside
                    # sanity checks
                    assert he.cell_opp()>=0
                    assert he.cell()<0

                    if bc_type==self.OPEN:
                        is_interior=""
                    else:
                        # Have to figure out whether this falls on the exterior boundary
                        # or an island
                        if he.node_fwd() in exterior_nodes:
                            is_interior="0 "
                        else:
                            is_interior="1 "

                    fp.write("%d %s= number of nodes for %s boundary %d\n"%(n_bc_id,is_interior,bc_label,bc_id+1))

                    while bc_id_mask[he.j]:
                        he=he.fwd()
                    n_written=0
                    while 1:
                        fp.write("%d\n"%(1+he.node_rev())) # to 1-based
                        n_written+=1
                        he=he.rev()
                        if not bc_id_mask[he.j]:
                            break
                    assert n_written==n_bc_id
            
        fp.close()
        
    @classmethod
    def read_gr3(cls,fn):
        fp=open(fn,'rt')
        g=cls(max_sides=4)
        
        g.filename=fn
        g.name=fp.readline().strip()
        
        # 108 130
        Nc,Nn = [int(s) for s in fp.readline().split()]

        xyzs=[]


        for n in range(Nn):
            fields=fp.readline().split()
            idx=int(fields[0]) # 1-based
            assert n+1==idx

            x,y,z=[float(s) for s in fields[1:]]
            xyzs.append([x,y,z])
        xyzs=np.array(xyzs)
        g.from_simple_data(points=xyzs)
        g.add_node_field('depth',xyzs[:,2])
        g.add_node_field('node_z_bed',-xyzs[:,2])

        cells=[]

        for c in range(Nc):
            fields=fp.readline().split()
            idx=int(fields[0]) # 1-based
            nnodes=int(fields[1])
            nodes=np.array([int(s) for s in fields[2:]])

            g.add_cell_and_edges(nodes=nodes-1)

        # So far so good
        def value_key(caster=int): # parse <int> = <string>
            line=fp.readline()
            if line=='': return None,None
            parts=line.split('=')
            return caster(parts[0]),parts[1].strip()
        def value_value_key(caster=int): # parse <int> <int> = <string>
            line=fp.readline()
            if line=='': return None,None
            parts=line.split('=')
            vals=parts[0].split()
            return caster(vals[0]),caster(vals[1]),parts[1].strip()

        # These are linestrings
        # 0: not a bc
        # 1: open boundary
        # 2: land boundary
        g.edges['mark']=cls.LAND
        # which bc of the corresponding type.
        g.add_edge_field('bc_id',-np.ones(g.Nedges(),np.int32))

        while 1:
            Nbc,bc_type=value_key() # 1 = Number of open boundaries
            if Nbc is None:
                break

            if bc_type.lower()=='number of open boundaries':
                mark=cls.OPEN
            elif bc_type.lower()=='number of land boundaries':
                mark=cls.LAND
            else:
                raise Exception("Unknown boundary type '%s'"%bc_type)

            # not using this
            Nbc_nodes,_=value_key() # 13 = Total number of open boundary nodes

            for bc_id in range(Nbc):
                if mark==cls.OPEN:
                    N_this_bc,_=value_key() # 13 = Number of nodes for open boundary 1
                elif mark==cls.LAND:
                    # interior: 0 for external boundary
                    #           1 for island boundary
                    N_this_bc,interior,_=value_value_key() # 13 0 = Number of nodes for land boundary

                nodes=[int(fp.readline().strip()) for _ in range(N_this_bc)]
                nodes=np.array(nodes)-1 # to 0-based
                for a,b in zip(nodes[:-1],nodes[1:]):
                    j=g.nodes_to_edge(a,b)
                    g.edges['mark'][j]=mark
                    g.edges['bc_id'][j]=bc_id

        fp.close()
        return g

    def write_cells_geopandas(self,crs="+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs"):
        """
        Copy cell geometry to a geopandas dataframe.
        Original code credit to Zhenlin Zhang.

        crs: specify a proj projection string.
        """
        # stompy generally does not depend on geopandas,        
        # so use local import
        import pandas as pd
        import geopandas as gpd
        polys=[self.cell_polygon(c) for c in range(self.Ncells())]
            
        df = pd.DataFrame()
        df['geometry'] = polys
        gdf = gpd.GeoDataFrame(df,geometry='geometry')
        gdf.crs = crs
        return gdf

    def write_to_xarray(self,*a,**kw):
        return self.write_xarray(*a,**kw)
    
    def write_xarray(self,ds=None,mesh_name='mesh',
                     node_coordinates='node_x node_y',
                     face_node_connectivity='face_node',
                     edge_node_connectivity='edge_node',
                     face_dimension='face',
                     edge_dimension='edge',
                     node_dimension='node'):
        """ write grid definition, ugrid-ish, to a new xarray dataset
        """
        import xarray as xr
        if ds is None:
            ds=xr.Dataset()

        ds[mesh_name]=1
        ds[mesh_name].attrs['cf_role']='mesh_topology'
        ds[mesh_name].attrs['node_coordinates']=node_coordinates
        ds[mesh_name].attrs['face_node_connectivity']=face_node_connectivity
        ds[mesh_name].attrs['edge_node_connectivity']=edge_node_connectivity
        ds[mesh_name].attrs['face_dimension']=face_dimension
        ds[mesh_name].attrs['edge_dimension']=edge_dimension

        node_x,node_y=node_coordinates.split()
        ds[node_x]= ( (node_dimension,),self.nodes['x'][:,0])
        ds[node_y]= ( (node_dimension,),self.nodes['x'][:,1])

        ds[face_node_connectivity]=( (face_dimension,'maxnode_per_face'), self.cells['nodes'] )
        ds[edge_node_connectivity]=( (edge_dimension,'node_per_edge'), self.edges['nodes'] )

        return ds

    def write_ugrid(self,
                    fn,
                    mesh_name='mesh',
                    fields='auto',
                    overwrite=False,
                    centers=True,
                    dialect=None):
        """
        rough ugrid writing - doesn't set the full complement of
        attributes (missing_value, edge-face connectivity, others...)

        mesh_name: the name of the variable in the netcdf file holding the attributes
         which define the collection of variables describing the grid.
        fields: 'auto' -- write fields of the UnstructuredGrid.  e.g. if self.cells['depth']
          exists, it will be written to the netcdf.  clashes between names are resolved by
          prefixing with 'cells_', 'edges_' or 'nodes_'.
          currently the only other option is to specify False, which will not write any
          additional data beyond the grid topology.
        overwrite: if False and the output exists, fail.
        centers: write cell centers.  this will trigger a call to cells_center()
          to make sure that any nan-valued centers are recalculated.

        dialect: 
          'mdal': As of QGIS 3.14, there are some limitations and strict 
           requirements for a mesh to be read in via mdal. Notably node and cell
           data can only be float or double.  Earlier mdal also required that face_node
           be float (something related to _FillValue, nan's).  That was the case with QGIS 3.6
        """
        if os.path.exists(fn):
            if overwrite:
                os.unlink(fn)
            else:
                raise GridException("File %s exists"%(fn))

        ds=xr.Dataset()
        ds[mesh_name]=1

        mesh_var=ds[mesh_name]
        # required: (this seems more robust than mesh_var.attrs[...] = '...'
        ds[mesh_name].attrs['cf_role']='mesh_topology'
        ds[mesh_name].attrs['standard_name']='mesh_topology'
        ds[mesh_name].attrs['topology_dimension']=2
        ds[mesh_name].attrs['dimension']=2
        ds[mesh_name].attrs['face_node_connectivity']='face_node'
        ds[mesh_name].attrs['node_coordinates']='node_x node_y'
        ds[mesh_name].attrs['long_name']='mesh topology'
        # "optionally required"
        ds[mesh_name].attrs['edge_dimension']='edge'
        ds[mesh_name].attrs['face_dimension']='face'
        ds[mesh_name].attrs['edge_node_connectivity']='edge_node'
        # optional
        ds[mesh_name].attrs['node_dimension']='node'
        ds[mesh_name].attrs['max_face_nodes_dimension']='maxnode_per_face'

        ds['node_x'] = ('node',),self.nodes['x'][:,0]
        ds['node_y'] = ('node',),self.nodes['x'][:,1]
        ds['face_node'] = ('face','maxnode_per_face'),self.cells['nodes']
        ds['edge_node']=('edge','node_per_edge'),self.edges['nodes']
        ds['edge_node'].attrs['start_index']=0

        ds['node_x'].attrs['standard_name']='projection_x_coordinate'
        ds['node_y'].attrs['standard_name']='projection_y_coordinate'

        ds['node_x'].attrs['location']='node'
        ds['node_y'].attrs['location']='node'

        ds['face_node'].attrs['cf_role']='face_node_connectivity'
        ds['face_node'].attrs['standard_name']='face_node_connectivity'
        ds['face_node'].attrs['start_index']=0
        ds['face_node'].attrs['long_name']='Vertex nodes of mesh faces'
        ds['face_node'].attrs['units']='nondimensional'
        
        if centers:
            cc=self.cells_center()
            ds[mesh_name].attrs['face_coordinates']='face_x face_y'
            ds['face_x']=('face',),self.cells['_center'][:,0]
            ds['face_y']=('face',),self.cells['_center'][:,1]

            ds['face_x'].attrs['location']='face'
            ds['face_y'].attrs['location']='face'
            ds['face_x'].attrs['standard_name']='projection_x_coordinate'
            ds['face_y'].attrs['standard_name']='projection_y_coordinate'

        if fields=='auto':
            for src_data,dim_name in [ (self.cells,'face'),
                                       (self.edges,'edge'),
                                       (self.nodes,'node') ]:
                for field in src_data.dtype.names:
                    if field.startswith('_'):
                        continue # presumed to be private, probably auto-calculated.
                    if field in ['cells','nodes','edges','deleted','face_x','face_y']:
                        continue # already included
                    if (src_data is self.nodes) and field=='x':
                        continue
                    if np.issubdtype(src_data[field].dtype,np.object_):
                        logging.warning("write_ugrid: will drop %s"%field)
                        continue
                    if (dialect=='mdal' and
                        not np.issubdtype(src_data[field].dtype,np.floating)):
                        logging.warning("write_ugrid: mdal dialect will drop non-floating %s"%field)
                        continue
                    if field in ds: # avoid duplicate names
                        # This is messy. There are fields like 'mark'
                        # that may exist on cells, edges and nodes.
                        # in the netcdf these have to have unique names.
                        prefix='_'+dim_name
                        if field.startswith(prefix):
                            logging.warning("Duplicate field already has prefix: %s"%field)
                            continue
                        out_field = prefix + "_" + field
                    else:
                        out_field=field

                    # simple handling to multidimensional arrays.
                    extra_dims=['d%d'%d for d in src_data[field].shape[1:]]
                    out_dims=(dim_name,) + tuple(extra_dims)
                    ds[out_field] = out_dims,src_data[field]

        ds.attrs['Conventions']='CF-1.6, UGRID-1.0'
        ds=ds.set_coords(['node_x','node_y'])
        
        ds.to_netcdf(fn)
        return ds

    def write_dfm(self,nc_fn,overwrite=False,node_elevation=None,check_depth=True):
        """
        nc_fn: netcdf file to write to
        overwrite: if True, allow overwriting an existing file
        node_elevation: if specified, the field used for node elevations, assumed positive-up.
          if None, will check for 'node_z_bed' and 'depth' as fields for nodes.

        check_depth: raise an exception if any node depth values are nan.
        """
        # use outdated netcdf wrapper
        # TODO: migrate the xarray or netCDF4
        from ..io import qnc
        nc=qnc.empty(fn=nc_fn,overwrite=overwrite,format='NETCDF3_CLASSIC')

        # schema copied from r17b_net.nc as written by rgfgrid
        nc.createDimension('nNetNode',self.Nnodes())
        nc.createDimension('nNetLink',self.Nedges())
        nc.createDimension('nNetLinkPts',2)

        node_x=nc.createVariable('NetNode_x','f8',('nNetNode'))
        node_x[:] = self.nodes['x'][:,0]
        node_x.units='m'
        node_x.standard_name = "projection_x_coordinate"
        node_x.long_name="x-coordinate of net nodes"
        node_x.grid_mapping = "projected_coordinate_system"

        node_y=nc.createVariable('NetNode_y','f8',('nNetNode'))
        node_y[:] = self.nodes['x'][:,1]
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

        if ('lon' in self.nodes.dtype.names) and ('lat' in self.nodes.dtype.names):
            print("Will include longitude & latitude")
            node_lon=nc.createVariable('NetNode_lon','f8',('nNetNode'))
            node_lon[:]=self.nodes['lon'][:]
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

        if node_elevation is None:
            if 'node_z_bed' in self.nodes.dtype.names:
                node_elevation='node_z_bed'
            elif 'depth' in self.nodes.dtype.names:
                node_elevation='depth'

        if check_depth:
            # DFM does not gracefully deal with nan depths -- preemptively fail
            if np.any(~np.isfinite(self.nodes[node_elevation])):
                raise Exception("Grid has nan or infinite node elevations")
            
        if node_elevation in self.nodes.dtype.names:
            node_z = nc.createVariable('NetNode_z','f8',('nNetNode'))
            node_z[:] = self.nodes[node_elevation][:]
            node_z.units = "m"
            node_z.positive = "up"
            node_z.standard_name = "sea_floor_depth"
            node_z.long_name = "Bottom level at net nodes (flow element\'s corners)"
            node_z.coordinates = "NetNode_x NetNode_y"
            node_z.grid_mapping = "projected_coordinate_system"

        links = nc.createVariable('NetLink','i4',('nNetLink','nNetLinkPts'))
        links[:,:]=self.edges['nodes'] + 1 # to 1-based!
        links.standard_name = "netlink"
        links.long_name = "link between two netnodes"

        link_types=nc.createVariable('NetLinkType','i4',('nNetLink'))
        link_types[:] = 2 # always seems to be 2 for these grids
        link_types.long_name = "type of netlink"
        link_types.valid_range = [0, 2]
        link_types.flag_values = [0, 1, 2]
        link_types.flag_meanings = "closed_link_between_2D_nodes link_between_1D_nodes link_between_2D_nodes"

        # global attributes - probably ought to allow passing in values for these...
        nc.institution = "stompy"
        nc.references = "http://github.com/rustychris/stompy"
        nc.history = "stompy unstructured_grid"

        nc.source = "Deltares, D-Flow FM Version 1.1.135.38878MS, Feb 26 2015, 17:00:33, model"
        nc.Conventions = "CF-1.5:Deltares-0.1"

        if 1:
            # add the complines to encode islands
            lines=self.boundary_linestrings()
            nc.createDimension('nNetCompLines',len(lines))

            # And add the cells:
            nc.createDimension('nNetElemMaxNode',self.max_sides)
            nc.createDimension('nNetElem',self.Ncells())
            missing=-2147483647 # DFM's preferred missing value

            cell_var=nc.createVariable('NetElemNode','i4',('nNetElem','nNetElemMaxNode'),
                                       fill_value=missing)
            # what to do about missing nodes?
            cell_nodes=self.cells['nodes'] + 1 #make it 1-based
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
        
    @staticmethod
    def from_shp(shp_fn,tolerance=0.0,**kw):
        # bit of extra work to find the number of nodes required
        feats=wkb2shp.shp2geom(shp_fn)

        if 'max_sides' not in kw:
            def all_rings():
                for geom in feats['geom']:
                    if geom.geom_type=='Polygon':
                        yield geom.exterior.coords
                        for ring in geom.interiors:
                            yield ring.coords
                            
            nsides=[len(ring)
                    for ring in all_rings()]
            if nsides:
                kw['max_sides']=np.max(nsides)

        g=UnstructuredGrid(**kw)
        g.add_from_shp(features=feats,tolerance=tolerance)
        return g
    
    def add_from_shp(self,shp_fn=None,features=None,linestring_field=None,
                     check_degenerate_edges=True, tolerance=0.0):
        """ Add features in the given shapefile to this grid.
        Limited support: 
        polygons must conform to self.max_sides
        
        linestring_field: incoming linestring are stuffed into an edge field
         as [N,2] coordinate arrays. mesh topology uses only start/end of 
         each linestring. If None (default), linestrings are handled segment
         by segment as distinct mesh edges.
         
        allow_degenerate_edges: permit edges that start/end on same node.
        """
        if features is None:
            features=wkb2shp.shp2geom(shp_fn)
            
        if linestring_field is not None:
            if linestring_field is self.edges.dtype.names:
                assert np.issubdtype(object,self.edges.dtype),"Linestring field has weird type"
            
            self.add_edge_field(linestring_field,np.zeros(self.Nedges(),dtype=object),
                                on_exists='pass')
            
        for geo in features['geom']:
            if geo.geom_type =='Polygon':
                def all_rings(geom):
                    if geom.geom_type=='Polygon':
                        yield np.array(geom.exterior.coords)
                        for ring in geom.interiors:
                            yield np.array(ring.coords)

                for coords in all_rings(geo):
                    if np.all(coords[-1] ==coords[0] ):
                        coords=coords[:-1]

                    # also check for ordering - force CCW.
                    if signed_area(coords)<0:
                        coords=coords[::-1]

                    # used to always return a new node - bad!
                    nodes=[self.add_or_find_node(x=x,tolerance=tolerance)
                           for x in coords]
                    self.add_cell_and_edges(nodes=nodes)
            elif geo.geom_type=='LineString':
                coords=np.array(geo.coords)
                if linestring_field is None:
                    self.add_linestring(coords,tolerance=tolerance)
                else:
                    nA=self.add_or_find_node(coords[0,:], tolerance=tolerance)
                    nB=self.add_or_find_node(coords[-1,:], tolerance=tolerance)
                    j=self.add_edge(nodes=[nA,nB],_check_degenerate=check_degenerate_edges)
                    self.edges[linestring_field][j]=coords                    
            else:
                raise GridException("Not ready for geometry type %s"%geo.geom_type)
        # still need to collapse duplicate nodes
        
    def add_linestring(self,coords,closed=False, tolerance=0.0):
        nodes=[self.add_or_find_node(x=x, tolerance=tolerance)
               for x in coords]
        edges=[]

        if not closed:
            ABs=zip(nodes[:-1],nodes[1:])
        else:
            ABs=zip(nodes, np.roll(nodes,-1))
            
        for a,b in ABs:
            j=self.nodes_to_edge(a,b)
            if j is None:
                j=self.add_edge(nodes=[a,b])
            edges.append(j)
            
        return nodes,edges
        
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

    def update_cell_edges(self,select='missing'):
        """ from edges['nodes'] and cells['nodes'], set cells['edges']
        select: 'all':  force an update on all cells.
        'missing': only cells for which the number of set edges does not
         match the number of nodes will be updated.
        """
        if select=='all':
            cells=self.valid_cell_iter()
        else:
            edge_per_cell=np.sum(self.cells['edges']<0,axis=1)
            node_per_cell=np.sum(self.cells['nodes']<0,axis=1)
            cells=np.nonzero( edge_per_cell != node_per_cell )[0]

        for c in cells:
            self.cells['edges'][c,:]=self.UNDEFINED # == -1
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
        self.update_element_defaults()
        
    def delete_edge_field(self,*names):
        self.edges=recarray_del_fields(self.edges,names)
        self.edge_dtype=self.edges.dtype
        self.update_element_defaults()

    def add_node_field(self,name,data,on_exists='fail'):
        """ add a new field to nodes, amend node_dtype
        on_exists: what to do if the name is already a node field.
         'fail'  raise exception
         'pass'  leave existing
         'overwrite' replace existing -- note that this does not currently
           change the type of the field, but may do so in the future.
        """
        if name in np.dtype(self.node_dtype).names:
            if on_exists == 'fail':
                raise GridException("Node field %s already exists"%name)
            elif on_exists == 'pass':
                return
            elif on_exists == 'overwrite':
                self.nodes[name] = data
            else:
                raise Exception("Bad on_exists option: %s"%on_exists)
        else:
            self.nodes=recarray_add_fields(self.nodes,
                                           [(name,data)])
            self.node_dtype=self.nodes.dtype
        self.update_element_defaults()
        
    def delete_node_field(self,*names):
        self.nodes=recarray_del_fields(self.nodes,names)
        self.node_dtype=self.nodes.dtype
        self.update_element_defaults()

    def add_cell_field(self,name,data,on_exists='fail'):
        """
        modifies cell_dtype to include a new field given by name,
        initialize with data.  NB this requires copying the cells
        array - not fast!
        """
        # will need to get fancier to discern vector dtypes
        # assert data.ndim==1  - maybe no need to be smart?
        data=np.asarray(data)
        if name in np.dtype(self.cell_dtype).names:
            if on_exists == 'fail':
                raise GridException("Cell field %s already exists"%name)
            elif on_exists == 'pass':
                return
            elif on_exists == 'overwrite':
                self.cells[name] = data
        else:
            self.cells=recarray_add_fields(self.cells,
                                           [(name,data)])
            self.cell_dtype=self.cells.dtype
        self.update_element_defaults()
        
    def delete_cell_field(self,*names):
        self.cells=recarray_del_fields(self.cells,names)
        self.cell_dtype=self.cells.dtype
        self.update_element_defaults()

    def match_to_grid(self,other,tol=1e-3):
        """
        Return node_map,edge_map,cell_map
        Each is indexed by self's items, and is either an index
        into other's items, or -1 if no match was found. 
         
        I.e.:
        self.nodes['x'][n] == other.nodes['x'][node_map[n]]
        """
        import scipy.spatial
        
        kdt=scipy.spatial.KDTree( other.nodes['x'] )
        dists_idxs=[ kdt.query(xy,1) for xy in self.nodes['x'] ]
        dists_idxs=np.array(dists_idxs)
        node_map=dists_idxs[:,1].astype(np.int32)
        node_map[ dists_idxs[:,0]>tol ] = -1

        # Edges can be matched, but they don't necessarily have the same
        # orientation.
        kdt=scipy.spatial.KDTree( other.edges_center() )
        dists_idxs=[ kdt.query(xy,1) for xy in self.edges_center() ]
        dists_idxs=np.array(dists_idxs)
        edge_map=dists_idxs[:,1].astype(np.int32)
        edge_map[ dists_idxs[:,0]>tol ] = -1

        kdt=scipy.spatial.KDTree( other.cells_centroid() )
        dists_idxs=[ kdt.query(xy,1) for xy in self.cells_centroid() ]
        dists_idxs=np.array(dists_idxs)
        cell_map=dists_idxs[:,1].astype(np.int32)
        cell_map[ dists_idxs[:,0]>tol ] = -1

        return node_map,edge_map,cell_map
        
    def renumber(self,reorient_edges=True,reorient_cells=True):
        """
        Renumber all nodes, edges and cells to omit
        deleted items.
        reorient_edges: also flip edges to keep interior of domain to the
          left (see orient_edges)
        reorient_cells: force cells to have positive area by reversing node
           order as needed
        """
        node_map=self.renumber_nodes()
        edge_map=self.renumber_edges()
        cell_map=self.renumber_cells()

        if reorient_edges:
            self.orient_edges()

        if reorient_cells:
            self.orient_cells()
            
        return dict(node_map=node_map,edge_map=edge_map,cell_map=cell_map)

    def orient_edges(self,on_bare_edge='fail'):
        """
        Flip any boundary edges with the exterior on the left.
        on_bare_edge: what to do if an edge is missing cells on both sides.
          'fail': raise GridException
          'pass': ignore.
        """
        e2c=self.edge_to_cells()
        to_flip=np.nonzero( (e2c[:,0]<0) & (e2c[:,1]>=0) )[0]

        bare=np.nonzero( (e2c[:,0]<0) & (e2c[:,1]<0) )[0]
        if len(bare) and on_bare_edge=='fail':
            raise GridException("orient_edges: edges with no cells: %s"%bare)
        
        if len(to_flip):
            self.log.info("Will flip %d edges"%len(to_flip))
            for j in to_flip:
                rec=self.edges[j]
                self.modify_edge( j=j,
                                  nodes=[rec['nodes'][1],rec['nodes'][0]],
                                  cells=[rec['cells'][1],rec['cells'][0]] )
        if on_bare_edge=='return':
            return bare

    def orient_cells(self,cw_islands=False, subedges=None):
        A=self.cells_area(subedges=subedges)

        flip_mask=A<0

        if cw_islands:
            # look for cells inside other cells, and make
            islands=self.select_island_cells(subedges=subedges)
            flip_mask[islands] = ~flip_mask[islands]

        flip=np.nonzero(flip_mask)[0]
        for c in flip:
            nodes=self.cell_to_nodes(c)
            self.cells['nodes'][c,:len(nodes)] = nodes[::-1]
        # Might be able to be smarter about which edges
        self.edge_to_cells(recalc=True)
        self.cells_area(sel=flip)

    def explode_subedges(self,subedges):
        """
        Convert subedges to regular edges, and return the set of original nodes.
        This ignores cells.
        """
        original_nodes=list(self.valid_node_iter())
        for j in self.valid_edge_iter():
            coords=self.edges[subedges][j]
            if coords is None:
                continue
            coords=np.asarray(coords)
            if coords.ndim==0 or coords.shape[0]==2:
                continue
            
            nodes=self.edges['nodes'][j].copy()
            coords=coords.copy()
            self.delete_edge(j)
            inner_nodes,inner_edges=self.add_linestring(coords[1:-1,:],closed=False)
            self.add_edge(nodes=[nodes[0],inner_nodes[0]])
            self.add_edge(nodes=[inner_nodes[-1],nodes[1]])
            
        return original_nodes

    def get_subedge(self,j,subedges):
        subedge=np.asarray(self.edges[subedges][j])
        if (subedge.ndim!=2):
            subedge=self.nodes['x'][ self.edges['nodes'][j] ]
            self.edges[subedges][j]=subedge
        return subedge
    
    def implode_subedges(self,subedges,break_nodes,allow_duplicates=False):
        """
        Glue edges back into complex edges except at break_nodes. This has to be aware of 
        cells!
        Start with slow implementation since we can reuse some other methods
        """
        break_nodes={n:True for n in break_nodes}
        
        # Ensure that all subedge fields are valid.
        for j in self.valid_edge_iter():
            self.get_subedge(j,subedges)
        
        for n in self.valid_node_iter():
            if n in break_nodes: continue
            edges=self.node_to_edges(n)
            if len(edges)!=2: continue
            #hes=self.node_to_halfedges(n)
            #if len(hes)!=2: continue

            sub0=self.edges[subedges][edges[0]]
            sub1=self.edges[subedges][edges[1]]
            
            # n_left ---edges[0]--- n ---edges[1]--- n_right
            e0nodes=self.edges['nodes'][edges[0]]
            e1nodes=self.edges['nodes'][edges[1]]
            if e0nodes[0]==n:
                n_left=e0nodes[1]
                sub0=sub0[::-1]
            elif e0nodes[1]==n:
                n_left=e0nodes[0]
            else: assert False
            if e1nodes[0]==n:
                n_right=e1nodes[1]
            elif e1nodes[1]==n:
                n_right=e1nodes[0]
                sub1=sub1[::-1]
            else: assert False
            
            # combine...
            new_edge=self.merge_edges(edges,_check_existing=not allow_duplicates)
            # Could make this dependent on whether end points match.
            sub=np.concatenate([sub0,sub1[1:]])
            if self.edges['nodes'][new_edge,0]==n_left:
                pass
            elif self.edges['nodes'][new_edge,0]==n_right:
                sub=sub[::-1]
            else: assert False  
            
            self.edges[subedges][new_edge] = sub


    def select_island_cells(self,subedges=None):
        """
        Return bitmask over cells with True for cells that are 
        entirely inside other cells. Does not attempt to go
        deeper than one layer (so a pond on an island in a lake
        will be marked as an island).
        This is used to orient cell areas, so the test is performed
        on the un-oriented cell geometry.

        Current implementation is not fast! quadratic and slow tests!
        """
        valid_cells=np.nonzero(~self.cells['deleted'])[0]
        A=np.abs(self.cells_area(subedges=subedges))[valid_cells]

        # from largest to smallest
        order = np.argsort(-A)

        is_island=np.zeros(self.Ncells(), bool)
        is_island[:] = False

        polys=[self.cell_polygon(c,subedges=subedges) for c in order]
        
        for ci,c in enumerate(order):
            if ci==0: continue # skip first cell
            
            for candidate in range(ci):
                if polys[candidate].contains( polys[ci] ):
                    is_island[c]=True
                    break
        return is_island
        
    def renumber_nodes_ordering(self):
        return np.argsort(self.nodes['deleted'],kind='mergesort')

    def renumber_nodes(self,order=None):
        """
        Renumber node indices, dropping nodes which are deleted.
        Updates 'nodes' fields for edges and cells
        Returns an array node_map which is indexed by by the old
        node indices, and maps those to new node indices or -1
        if the old node has been dropped.
        """
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

        enodes=self.edges['nodes'].copy()
        enodes[self.edges['deleted'],:]=-1
        self.edges['nodes'] = node_map[enodes]
        cnodes=self.cells['nodes'].copy()
        cnodes[self.cells['deleted'],:]=-1
        self.cells['nodes'] = node_map[cnodes]

        self._node_to_edges = None
        self._node_to_cells = None
        self._node_index = None
        return node_map

    def delete_orphan_edges(self):
        self.delete_naked_edges()
    def delete_naked_edges(self):
        """
        Delete edges which have no cell neighbors
        """
        e2c=self.edge_to_cells()

        to_delete=np.all(e2c<0,axis=1) & (~self.edges['deleted'])
        for j in np.nonzero(to_delete)[0]:
            self.delete_edge(j)

    def delete_orphan_nodes(self):
        self.delete_naked_nodes()
    def delete_naked_nodes(self):
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

        self.log.debug("%d nodes found to be orphans"%np.sum(~used))

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
        """
        Renumber cell indices, dropping deleted cells.
        Update edges['cells'], preserving negative values (e.g.
        an edge marker, boundary, etc.).
        Returns cell_map, which can be indexed by old cell indices
        to get new cell indices.  Cell map is actually slightly
        larger than the number of old cells, to accomodate negative
        indexing.  For example, cell_map[-2]=-2
        """
        if order is None:
            csort = self.renumber_cells_ordering()
        else:
            csort= order
        Nneg=-min(-1,self.edges['cells'].min())
        cell_map = np.zeros(self.Ncells()+Nneg,np.int32) # do this before truncating cells
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
        return cell_map

    def renumber_edges_ordering(self):
        Nactive = sum(~self.edges['deleted'])
        return np.argsort( self.edges['deleted'],kind='mergesort')[:Nactive]

    def renumber_edges(self,order=None):
        """
        Renumber edge indices, dropping deleted edges.
        Returns edge_map, which can be indexed by old edge indices
        to get new edge indices.
        edge_map has an extra entry to all undefined edges (i.e. the 4th edge of
        a triangle, marked with cells['edges'][n,3]==-1) to map back to -1.
        """
        if order is None:
            esort=self.renumber_edges_ordering()
        else:
            esort=order

        # edges take a little extra work, for handling -1 missing edges
        # Follows same logic as for cells
        if self.Ncells():
            Nneg=-min(-1,self.cells['edges'].min())
        else:
            Nneg=1
        edge_map = np.zeros(self.Nedges()+Nneg) # do this before truncating
        self.edges = self.edges[esort]

        edge_map[:] = -999 # these should only remain for deleted edges, which won't show up in the output
        edge_map[:-Nneg][esort] = np.arange(self.Nedges()) # and this after truncating
        #edge_map[-1] = -1 # triangles have a -1 -> -1 edge mapping
        edge_map[-Nneg:] = np.arange(-Nneg,0)

        self.cells['edges'] = edge_map[self.cells['edges']]
        return edge_map

    def add_grid(self,ugB,merge_nodes=None,log=None,tol=0.0):
        """
        Add the nodes, edges, and cells from another grid to this grid.
        Copies fields with common names, any other fields are dropped from ugB.
        If self.max_sides is smaller than ugB.max_sides, it will be increased.

        merge_nodes: [ (self_node,ugB_node), ... ]
          Nodes which overlap and will be mapped instead of added.
        or 'auto' in which case duplicate nodes by coordinate will be chosen for merging,
         optionally with a non-zero tolerance.
        """
        if self.max_sides<ugB.max_sides:
            # This could be smarter and only increase if ugB is actually
            # using the additional sides.
            self.log.warning("Increasing max_sides from %d to %d"%(self.max_sides,ugB.max_sides))
            self.modify_max_sides(ugB.max_sides)
        else:
            self.log.debug("max_sides is okay (%d)"%(self.max_sides))
            
        node_map=np.zeros( ugB.Nnodes(), 'i4')-1
        edge_map=np.zeros( ugB.Nedges(), 'i4')-1
        cell_map=np.zeros( ugB.Ncells(), 'i4')-1

        if merge_nodes is not None:
            if isinstance(merge_nodes,str) and merge_nodes=='auto':
                merge_nodes=find_common_nodes(self,ugB,tol=tol)
            for my_node,B_node in merge_nodes:
                node_map[B_node]=my_node

            # Also need to be careful about overlapping cells when there
            # is the possibility of merging nodes, and that means having
            # up to date edges['cells']
            self.edge_to_cells()
            ugB.edge_to_cells()

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

            # avoid mutating ugB, and pass on only the valid
            # nodes and edges
            orig_nodes=kwargs['nodes']
            kwargs['nodes'] = orig_nodes[orig_nodes>=0]
            orig_edges=kwargs['edges']
            kwargs['edges'] = orig_edges[orig_edges>=0]

            for i,node in enumerate(kwargs['nodes']):
                kwargs['nodes'][i]=node_map[node]

            # less common, but still need to check for duplicated cells
            # when merge_nodes is used.
            if merge_nodes is not None:
                c=self.nodes_to_cell( kwargs['nodes'], fail_hard=False)
                if c is not None:
                    cell_map[n]=c
                    if log: log.info("Skipping existing cell: %d: %s => %d: %s"%( n,str(orig_nodes),
                                                                                  c,str(kwargs['nodes'])))
                    continue

            for i,edge in enumerate(kwargs['edges']):
                kwargs['edges'][i]=edge_map[edge]

            cell_map[n]=self.add_cell(**kwargs)

        return node_map,edge_map,cell_map

    def boundary_cycle(self):
        """
        Return list of nodes in the outermost boundary
        """
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

    def find_cycles(self,max_cycle_len='auto',starting_edges=None,check_area=True):
        """ traverse edges, returning a list of lists, each list giving the
        CCW-ordered node indices which make up a 'facet' or cycle in the graph
        (i.e. potentially a cell).
        starting_edges: iterable of edge indices from which to start the cycle
          traversal.
        check_area: if True, make sure that any returned cycles form a polygon
         with positive area.   This can be an issue if the outer ring of the grid is
         short enough to be a cycle itself.
        """
        if max_cycle_len=='auto':
            max_cycle_len=self.max_sides

        visited=set() # directed tuple of nodes

        cycles=[]

        if starting_edges is None:
            starting_edges=self.valid_edge_iter()

        for ji,j in enumerate(starting_edges):
            if ji>0 and ji % 10000==0:
                logging.info("Edge %d/%d, %d cycles"%(ji,self.Nedges(),len(cycles)))

            if isinstance(j,HalfEdge):
                halfedges=[j]
            else:
                # iterate over the two half-edges
                halfedges=[self.halfedge(j,0),
                           self.halfedge(j,1)]
            # for A,B in (self.edges['nodes'][j], self.edges['nodes'][j,::-1]):
            # Skip half edges already visited:
            halfedges=[ he for he in halfedges if (he.node_rev(),he.node_fwd()) not in visited]
            for he in halfedges:
                cycle=[he.node_rev()]
                trav=he
                while trav.node_fwd() != he.node_rev() and len(cycle)<=max_cycle_len:
                    cycle.append(trav.node_fwd())
                    visited.add((trav.node_rev(),trav.node_fwd()))
                    trav=trav.fwd()
                visited.add((trav.node_rev(),trav.node_fwd()))
                    
                if len(cycle)<=max_cycle_len:
                    # potential cycle:
                    if check_area:
                        A=signed_area( self.nodes['x'][cycle] )
                        if A>0:
                            cycles.append(cycle)
                    else:
                        cycles.append(cycle)
        return cycles
    
    def make_cells_from_edges(self,max_sides=None):
        """
        Traverse edges and create cells for edge cycles
        of len<=max_sides.
        max_sides=None will use the grid's existing max_sides.
        max_sides='auto' will use max_sides large enough to
        includes all cycles.
        """
        if max_sides=='auto':
            cycles=self.find_cycles(max_cycle_len=self.Nedges())
            if len(cycles)==0:
                return
            max_sides = max([len(cycle) for cycle in cycles])
            if max_sides>self.max_sides:
                self.modify_max_sides(max_sides)
        else:
            max_sides=max_sides or self.max_sides
            cycles=self.find_cycles(max_cycle_len=max_sides)
        assert max_sides<=self.max_sides
        ncells=len(cycles)
        if ncells:
            self.cells = np.zeros( ncells, self.cell_dtype)
            self.cells['nodes'][...] = -1
            self.cells['_center'] = np.nan # signal stale
            self.cells['_area'] = np.nan   # signal stale
            self.cells['edges'] = self.UNKNOWN # and no data here
            for i,cycle in enumerate(cycles):
                self.cells['nodes'][i,:len(cycle)]=cycle
            # This is now stale
            self._cell_center_index=None
            self.edge_to_cells(recalc=True)

    def interp_cell_to_node(self,cval):
        result=np.zeros(self.Nnodes(),cval.dtype)
        for n in range(self.Nnodes()):
            result[n]=cval[self.node_to_cells(n)].mean()
        return result

    def interp_cell_to_node_matrix(self):
        from scipy import sparse
        M=sparse.dok_matrix( (self.Nnodes(),self.Ncells()), np.float64)
        for n in range(self.Nnodes()):
            cells=self.node_to_cells(n)
            weight=1./len(cells)
            for c in cells:
                M[n,c]=weight
        return M.tocsr()
    
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

    def interp_cell_to_raster_function(self,*a,**kw):
        """
        Bundle interp_cell_to_raster_matrix into a simple function that
        takes an array of cell values and returns a raster field.SimpleGrid.
        Handles the reshaping and boundary filling, and with precomputing
        the sparse matrix this can resample 50k cell values to a 250m grid
        in 20ms.

        Can be called with data array set to None in order to get a SimpleGrid
        of the right size/extent for metadata purposes.
        """
        from ..spatial import field
        fld,M=self.interp_cell_to_raster_matrix(*a,**kw)
        invalid=M.A.sum(axis=1)==0.0
        
        def to_field(cell_values,fld=fld,M=M,invalid=invalid):
            if cell_values is None:
                # shorthand to get a field with the extents and shape of
                # the output.
                cell_values=np.zeros(M.shape[1])
            pix_values=M.dot(cell_values)
            pix_values[invalid]=np.nan
            f_result=field.SimpleGrid(extents=fld.extents,F=pix_values.reshape(fld.F.shape))
            f_result.fill_by_convolution(iterations=1)
            return f_result
        return to_field
            
    def interp_cell_to_raster_matrix(self,fld=None, dx=None, dy=None):
        """
        return field.SimpleGrid and sparse matrix that interpolates cell-centered values to
        the given field.SimpleGrid, such that

        fld,M = g.interp_cell_to_raster_matrix(dx=50,dy=50)
        pix_values=M.dot(cell_values).reshape(fld.F.shape)
        interped=field.SimpleGrid(extents=fld.extents,F=pix_values)

        sets interped as a georeferenced raster that interpolates cell-centered
        values.

        As-is, pixels that do not overlap the grid (specifically that do not overlap
        the bounding box of any cell) end up with 0.
        This can be troubling if the pixels are then used for interpolation, since
        that zero can creep back in even when the zero pixel does not overlap the grid

        This could be solved 'in post', by masking and fill_by_convolution. Better yet
        would be to include that filling in the matrix, but that may be more work than
        its worth. Punt to a wrapper function that will do this in post.
        """
        from scipy import sparse
        from ..spatial import field
        
        if fld is None:
            g_xxyy=self.bounds()
            assert dx is not None
            assert dy is not None
            fld=field.SimpleGrid.zeros(extents=[g_xxyy[0]-dx, g_xxyy[1]+dx,
                                                g_xxyy[2]-dy, g_xxyy[3]+dy],
                                       dx=dx,dy=dy)
        
        n_rows,n_cols=fld.F.shape
        n_pix=fld.F.size
        
        # columns are mesh cells
        # rows are pixels.
        M=sparse.dok_matrix( (n_pix, self.Ncells()), np.float32)

        # iterate over mesh cells and update pixels.
        #   can be fairly fast -- bounds of the cell, evenly distribute pixels.
        #   then normalize rows to sum to 1.
        #   if needed this could be more precise, intersecting pixels with each
        #   cell. Too expensive for the current application.
        for cell in range(self.Ncells()):
            pnts = self.nodes['x'][self.cell_to_nodes(cell)]
            pmin=pnts.min(axis=0)
            pmax=pnts.max(axis=0)
            rrcc=fld.rect_to_indexes([pmin,pmax])
            for row in range(rrcc[0],rrcc[1]):
                for col in range(rrcc[2],rrcc[3]):
                    lidx=col+row*n_cols
                    M[lidx,cell]=1.0
                    
        cell_counts=M.sum(axis=1).A[:,0]
        valid=cell_counts>0
        cell_counts[valid] = 1./cell_counts[valid]
        # Appears that nan's don't propagate as expected.
        # cell_counts[~valid] = np.nan # blank out non-overlapping pixels
        cell_diag = sparse.dok_matrix( (n_pix,n_pix), np.float32)
        cell_diag.setdiag(cell_counts)
        M = cell_diag * M 
                    
        return fld,M
    
    def interp_edge_to_cell(self,values):
        """
        Average edge-centered values to get cell centered values.
        """
        # Could be vectorized, but I don't remember the call to get
        # numpy to sum over repeated indices.
        cvals=np.zeros(self.Ncells(),values.dtype)
        counts=np.zeros(self.Ncells(),np.int32)
        for c in self.valid_cell_iter():
            cvals[c]=np.mean(values[self.cell_to_edges(c)])
        return cvals

    def interp_perot(self,values,edge_normals=None):
        """
        Interpolate edge-normal vector components to cell-centered
        vector value.
        edge_normals can be supplied in case data uses a different convention
        than self.
        """
        # With some preprocessing, this could be turned into a sparse matrix
        # multiplication and be much faster.
        cc=self.cells_center()
        ec=self.edges_center()

        if edge_normals is None: edge_normals=self.edges_normals()
                
        e2c=self.edge_to_cells()
        el=self.edges_length()
        Uc=np.zeros((self.Ncells(),2),np.float64)

        for c in np.arange(self.Ncells()):
            js=self.cell_to_edges(c)
            for nf,j in enumerate(js):
                de2f=mag(cc[c]-ec[j])
                # Uc ~ m3/s * m
                Uc[c,:] += values[j]*edge_normals[j]*de2f*el[j]
        Uc /= self.cells_area()[:,None]
        return Uc

    def interp_perot_matrix(self,edge_normals=None):
        """
        preprocessed version of perot interp. 1000x faster on 50k cells
        than looping version above.
        """
        if edge_normals is None:
            edge_normals=self.edges_normals()

        from scipy import sparse
        # rows are cell0u,cell0v,cell1u,cell1v, ...
        M=sparse.dok_matrix( (2*self.Ncells(),self.Nedges()), np.float64)
            
        cc=self.cells_center()
        ec=self.edges_center()

                
        e2c=self.edge_to_cells()
        el=self.edges_length()
        Uc=np.zeros((self.Ncells(),2),np.float64)

        Ac=self.cells_area()
        
        for c in np.arange(self.Ncells()):
            js=self.cell_to_edges(c)
            for nf,j in enumerate(js):
                de2f=mag(cc[c]-ec[j])
                M[2*c+0,j] = edge_normals[j,0]*de2f*el[j] / Ac[c]
                M[2*c+1,j] = edge_normals[j,1]*de2f*el[j] / Ac[c]
                
        return M

    def cell_gradient(self,cell_values):
        """
        Use Perot interp to get cell-centered gradients from cell-centered values.
        No attempt to be clever at boundaries.
        """
        edge_gradient = np.zeros(self.Nedges(),np.float64)
        cc=self.cells_center()
        e2c=self.edge_to_cells()
        c1=np.where(e2c[:,0]>=0,e2c[:,0],e2c[:,1])
        c2=np.where(e2c[:,1]>=0,e2c[:,1],e2c[:,0])
        edge_slopes= (cell_values[c2] - cell_values[c1]) / np.where(c1==c2,1.0, mag(cc[c2]-cc[c1]))
        return self.interp_perot(edge_slopes)
    
    def cells_to_edge(self,a,b):
        j1=self.cell_to_edges(a)
        j2=self.cell_to_edges(b)
        for j in j1:
            if j in j2:
                return j
        return None

    def edge_to_cells_reflect(self,*a,**k):
        """
        Like edge_to_cells, but when a cell is missing (<0), replace
        with the opposite cell. 
        """
        e2c=self.edge_to_cells(*a,**k).copy()
        left_missing=e2c[:,0]<0
        right_missing=e2c[:,1]<0
        e2c[left_missing,0] = e2c[left_missing,1]
        e2c[right_missing,1] = e2c[right_missing,0]
        return e2c
        
    def edge_to_cells(self,e=slice(None),recalc=False,
                      on_missing='error'):
        """
        recalc: if True, forces recalculation of all edges.

        e: limit output to a single edge, a slice or a bitmask
          may also limit updates to the requested edges

        on_missing: Invalid mesh may have cells which do not
          have corresponding edges.
         'error': raise Exception
         
        """
        # try to be a little clever about recalculating -
        # it can be very slow to check all edges for UNKNOWN
        if recalc:
            self.edges['cells'][:,:]=self.UNMESHED
            self.log.info("Recalculating edge to cells" )
            all_c=np.nonzero( ~self.cells['deleted'] )[0]
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
                    # handle e=[1,45,321], e=[True,False,True, True,...]
                    e=np.asarray(e)
                    L=len(e)
                    if L==self.Nedges() and np.issubdtype(np.bool_,e.dtype):
                        js=np.nonzero(e)[0]
                    else:
                        js=e
                except TypeError:
                    # handle e is an int, int32, int64, etc scalar.
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
                    msg="Failed to find an edge, c=%d, nodes=%d,%d"%(c,a,b) 
                    if on_missing=='error':
                        raise Exception(msg)
                    elif on_missing=='add':
                        print(msg+" -- adding edge")
                        j=self.add_edge(nodes=[a,b])
                    elif on_missing=='delete':
                        print(msg+" -- deleting cell")
                        self.delete_cell(c)
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
        self._node_to_edges=None

        cell_count=0
        for c in self.valid_cell_iter():
            cell_count+=1
            
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
        if len(new_edges):
            self.edges['nodes'] = new_edges[:,:2]
            self.edges['cells'] = new_edges[:,2:4]

    def make_edges_from_cells_fast(self):
        """
        vectorized version.  might be buggy.
        new June 2020
        """
        self._node_to_edges=None
        valid_cells=~self.cells['deleted']

        def pair_iter():
            all_c=np.arange(self.Ncells())

            for face in range(self.max_sides):
                a=self.cells['nodes'][:,face]
                if face+1==self.max_sides:
                    b=self.cells['nodes'][:,0]
                else:
                    b=self.cells['nodes'][:,face+1]
                    b=np.where( b<0, self.cells['nodes'][:,0], b )
                valid=(a>=0)&(~self.cells['deleted'])
                yield (a[valid],b[valid],all_c[valid])

        edge_sets=[]
        for a,b,cell in pair_iter():
            flip=a>b
            new_edges=np.zeros( len(a), [('a',np.int32),
                                         ('b',np.int32),
                                         ('c1',np.int32),
                                         ('c2',np.int32)])
            new_edges['a']=np.where(flip,b,a)
            new_edges['b']=np.where(flip,a,b)
            new_edges['c1']=np.where(flip,-1,cell)
            new_edges['c2']=np.where(flip,cell,-1)

            edge_sets.append(new_edges)
        new_edges=np.concatenate(edge_sets)

        if len(new_edges)==0:
            assert valid_cells.sum()==0
            print("Careful -- no edges found!")
            
        #order=np.argsort( new_edges, order=['a','b'], kind='stable') # 2s
        # 5x faster
        # kind='stable' is only in numpy >=1.15.0
        # 'mergesort' is also stable (and probably internally they are identical)
        orderb=np.argsort( new_edges['b'], kind='mergesort') # 0.3s
        ordera=np.argsort( new_edges['a'][orderb], kind='mergesort' )
        order=orderb[ordera]

        new_edges=new_edges[order]

        # indices of first part of an edge
        breaks=np.r_[False, (np.diff(new_edges['a'])!=0) | (np.diff(new_edges['b'])!=0)]
        edge=np.r_[0,np.nonzero(breaks)[0] ]
        j=np.cumsum(breaks)

        self.edges = np.zeros( j[-1]+1,self.edge_dtype )
        self.edges['nodes'][:,0] = new_edges['a'][edge]
        self.edges['nodes'][:,1] = new_edges['b'][edge]
        self.edges['cells']=-1
        has_left=new_edges['c1']>=0
        has_right=new_edges['c2']>=0
        self.edges['cells'][j[has_left],0]=new_edges['c1'][has_left]
        self.edges['cells'][j[has_right],1]=new_edges['c2'][has_right]

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

    def edge_center(self,j):
        return self.nodes['x'][self.edges['nodes'][j]].mean(axis=0)
    def edges_center(self,edges=None):
        """
        edges: sequence of edge indices to compute subset
        """
        if edges is None:
            centers=np.zeros( (self.Nedges(), 2), np.float64)
            valid=~self.edges['deleted']
            centers[valid,:] = self.nodes['x'][self.edges['nodes'][valid,:]].mean(axis=1)
            centers[~valid,:]=np.nan
        else:
            edges=np.asarray(edges)
            centers=np.zeros( (len(edges), 2), np.float64)
            valid=~self.edges['deleted'][edges]
            centers[valid,:] = self.nodes['x'][self.edges['nodes'][edges[valid],:]].mean(axis=1)
            centers[~valid,:]=np.nan

        return centers

    def cells_centroid(self,ids=None,method='py'):
        """
        Calculate cell centroids.  Defaults to faster python method,
        though subsetting with ids is not optimized with the python method.
        """
        if method=='py':
            return self.cells_centroid_py(ids)
        else:
            return self.cells_centroid_shapely(ids=ids)

    def cells_representative_point(self,ids=None):
        if ids is None:
            ids=np.arange(self.Ncells())

        points=np.zeros( (len(ids),2),np.float64)*np.nan

        for ci,c in enumerate(ids):
            if not self.cells['deleted'][c]:
                # 2023-08-25 RH: use coords accessor as direct array conversion
                # deprecated.
                points[ci]=np.array(self.cell_polygon(c).representative_point().coords)[0]
        return points
    
    def cell_coords_subedges(self,c,subedges):
        """
        Coordinate sequence [N,2] pulling sub-edge linestring from
        self.edges[subedges].
        """
        coords=[]
        for j in self.cell_to_edges(c):
            seg_coords=self.edges[subedges][j]
            if self.edges['cells'][j,0]==c:
                pass
            elif self.edges['cells'][j,1]==c:
                seg_coords=seg_coords[::-1,:]
            else:
                assert False
            coords.append(seg_coords[1:,:])
        return np.concatenate(coords)
        
    def cells_centroid_shapely(self,ids=None):
        """
        Calculate cell centroids using shapely/libgeos library.
        Possibly more robust than python code, but 2 orders of magnitude 
        slower.
        """
        if ids is None:
            ids=np.arange(self.Ncells())

        centroids=np.zeros( (len(ids),2),np.float64)*np.nan

        for ci,c in enumerate(ids):
            if not self.cells['deleted'][c]:
                centroids[ci]= np.array(self.cell_polygon(c).centroid)
        return centroids

    def cells_centroid_py(self,ids=None):
        """
        Vectorized python calculation of centroids.  Returns centroids
        [Ncells,{x,y}], with nan for deleted cells.
        if ids is included, return only the values for the given list of
        cells
        """
        A=self.cells_area(sel=ids)
        cxy=np.zeros( (len(A),2), np.float64)

        if ids is None:
            sel=slice(None)
        else:
            sel=ids

        nodes=self.cells['nodes'][sel,:]
        refs=self.nodes['x'][self.cells['nodes'][sel,0]]

        # replace missing nodes with first node
        cnodes=np.where( nodes>=0,
                         nodes,
                         nodes[:,0][:,None] )
        all_pnts=self.nodes['x'][cnodes] - refs[:,None,:]

        i=np.arange(cnodes.shape[1])
        ip1=(i+1)%cnodes.shape[1]

        xA=all_pnts[:,i,:]
        xB=all_pnts[:,ip1,:]
        tmp=xA[:,:,0]*xB[:,:,1] - xB[:,:,0]*xA[:,:,1]

        cxy[:,0] = ( (xA[:,:,0]+xB[:,:,0])*tmp).sum(axis=1)
        cxy[:,1] = ( (xA[:,:,1]+xB[:,:,1])*tmp).sum(axis=1)

        valid=~self.cells['deleted'][sel]
        cxy[valid] /= 6*A[valid,None]
        cxy[valid] += refs[valid,:]

        cxy[~valid,:]=np.nan

        return cxy

    default_cells_center_mode='first3'
    def cells_center(self,refresh=False,mode=None):
        """ calling this method is preferable to direct access to the
        array, since cell centers can possibly be stale if the grid has been
        modified (though no such checking exists yet).

        For now, circumcenter is calculated only from 1st 3 points, even if it's
        a quad.

        refresh: must be True, False, or something slice-like (slice, bitmap, integer array)

        mode: first3 - estimate circumcenter from the first 3 nodes
         sequential - estimate center from all consecutive triples of nodes
        """
        mode=mode or self.default_cells_center_mode
        if refresh is True:
            to_update=slice(None)
            do_update=True
        elif refresh is not False:
            to_update=refresh
            if isinstance(to_update,slice):
                do_update=True
            else:
                to_update=np.asarray(to_update)
                if to_update.dtype==np.bool_:
                    do_update=to_update.sum()
                else:
                    do_update=len(to_update)
        else:
            to_update = np.isnan(self.cells['_center'][:,0]) & (~self.cells['deleted'])
            do_update=to_update.sum()

        # yeah, it's sort of awkward to handle the different ways that refresh
        # can be specified. maybe bitmasks aren't a great approach. doesn't
        # scale that well.
        if do_update: # np.sum(to_update) > 0:
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

    def edges_normals(self,edges=slice(None),force_inward=False,update_e2c=True,
                      cache=False,update=False):
        """
        Calculate unit normal vectors for all edges, or a subset if edges
        is specified.
        force_inward: for edges with only one adjacent cell, modify the
           sign of the normal such that it is positive towards the adjacent
           cell, i.e. positive into the domain.  Otherwise, all normals
           are positive left-to-right (i.e. from cell 0 to cell 1)

        update_e2c: whether to recalculate edges['cells'] for the required edges.

        cache: if a string, normals are saved on the grid in the given variable.
        if the grid is edited, these are not updated!
        update: only relevant when cache is set. Force recalculation.
        """
        # does not assume the grid is orthogonal - normals are found by rotating
        # the tangent vector
        # I think this is positive towards c1, i.e. from left to right looking
        # from n0 to n1

        # starts as the vector from node1 to node2
        # say c0 was on the left, c1 on the right.  so n0 -> n1 is (0,1)
        # then that is changed to (1,0), then (1,-0)
        # so this pointing left to right, and is in fact pointing towards c1.
        # had been axis=1, and [:,0,::-1]
        # but with edges possibly a single index, make it more general
        if cache and not update and cache in self.edges.dtype.names:
            return self.edges[cache]
        
        normals = np.diff(self.nodes['x'][self.edges['nodes'][edges]],axis=-2)[...,0,::-1]
        normals[...,1] *= -1
        normals /= mag(normals)[...,None]
        if force_inward:
            if update_e2c:
                self.edge_to_cells(e=edges)
            e2c=self.edges['cells'][edges]

            to_flip=(e2c[...,1]<0)&(e2c[...,0]>=0) # c1 is 'outside', c0 is 'inside'
            # This feels a bit sketch when edges is a single index.  Tested with
            # numpy 1.14.0 and it does the right thing
            normals[to_flip] *= -1

        if cache:
            self.add_edge_field(cache,normals,on_exists='overwrite')
            
        return normals

    # Variations on access to topology
    _node_to_cells = None
    def node_to_cells(self,n):
        # almost certainly a sign of an upstream error
        assert n>=0,"Query for cells containing a negative node is not allowed"
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
        """
        return edge index for the edge joining nodes n1,n2
        n1: node index, or if n2 is None, a sequence of 2 node indices
        n2: node index
        """
        if n2 is None:
            n1,n2=n1

        # Some slight "exotic" cases like conceptual meshes
        # and have edges that start/end on the same node.
        #assert n1!=n2,"Duplicate node %d in nodes_to_edge"%n1
        candidates1 = self.node_to_edges(n1)
        if n1!=n2:
            candidates2 = self.node_to_edges(n2)
    
            # about twice as fast to loop this way
            for e in candidates1:
                if e in candidates2:
                    return e
            return None
        else:
            for e in candidates1:
                if ( (self.edges['nodes'][e,0]==n1)
                    and (self.edges['nodes'][e,1]==n2) ):
                    return e
            return None
                
        # # this way has nodes_to_edge taking 3x longer than just the node_to_edges call
        # for e in candidates1:
        #     if n2 in self.edges['nodes'][e]:
        #         return e

    def nodes_to_cell(self,ns,fail_hard=True):
        cells=self.node_to_cells(ns[0])
        for n in ns[1:]:
            if n<0:
                # allow for somebody including missing nodes in the input
                # though assume that all negative node indices are at the
                # end of ns.
                break
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

    def cell_to_edges(self,c=None,ordered=False,pad=False):
        """ returns the indices of edges making up this cell -
        including trimming to the number of sides for this cell.

        ordered: return edges ordered by the node ordering,
            with the first edge connecting the first two nodes

        pad: pad out to max_sides with -1.
        """
        if ordered:
            e=[]
            # okay to be UNDEFINED, but not UNKNOWN.
            assert np.all(self.cells['nodes'][c]!=self.UNKNOWN)
            nodes=self.cell_to_nodes(c)
            N=len(nodes)

            for na in range(N):
                nb=(na+1)%N
                this_edge=self.nodes_to_edge( nodes[na],nodes[nb] )
                if this_edge is None:
                    self.log.warning("cell %d was missing its edges"%c)
                    this_edge=-1
                e.append(this_edge)
            if pad and N<self.max_sides:
                e.extend([-1]*(self.max_sides-N))
            return np.array(e,np.int32)
        else:
            e = self.cells['edges'][c]
            if np.any( e==self.UNKNOWN ):
                e=self.cell_to_edges(c,ordered=True,pad=True)
                self.cells['edges'][c]=e
            if pad:
                return e
            else:
                return e[e>=0]

    def cell_to_nodes(self,c):
        """ returns nodes making up this cell, including trimming
        to number of nodes in this cell.
        """
        n = self.cells['nodes'][c]
        return n[n>=0]

    def cell_to_cells(self,c,ordered=False,pad=False):
        """ Return adjacent cells for c. if ordered, follow suntans convention for
        order of neighbors.  cells[0] has nodes
        self.cells['nodes'][c,0]
        and
        self.cells['nodes'][c,1]

        if pad is True, add -1 up to max_sides.
        """
        js=self.cell_to_edges(c,ordered=ordered,pad=pad)

        e2c=self.edge_to_cells(js) # make sure it's fresh

        nbrs=[]
        for j,(c1,c2) in zip(js,e2c):
            if j<0:
                nbrs.append(-1)
            elif c1==c:
                nbrs.append(c2)
            else:
                nbrs.append(c1)
        return nbrs

    def cell_to_cell_node_neighbors(self,c):
        """ Return adjacent cells for c sharing at least one node
            Include the cell itself in list returned
        """
        nodes=self.cell_to_nodes(c)

        nbrs = []
        for node in nodes[np.where(nodes >= 0)[0]]:
            n2c=self.node_to_cells(node) # make sure it's fresh
            nbrs += n2c
        nbrs = np.unique(np.asarray(nbrs))

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
    def is_boundary_node(self,n):
        for j in self.node_to_edges(n):
            if self.is_boundary_edge(j):
                return True
        return False

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
            # RH 2020-07-16: new code for default values.  maybe works?
            n=self.node_defaults # np.zeros( (), dtype=self.node_dtype)
            self.nodes=array_append(self.nodes,n)
            i=len(self.nodes)-1
        else:
            self.nodes[i]=self.node_defaults

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
    def add_edge(self,_check_existing=True,_check_degenerate=True,**kwargs):
        """
        Does *not* check topology / planarity
        _check_existing: fail if the edge (or its reverse) already exists
        _check_degenerate: fail if the two nodes are equal
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
        if _check_degenerate:
            if self.edges[j]['nodes'][0]==self.edges[j]['nodes'][1]:
                raise self.InvalidEdge('duplicate nodes')

        if self._node_to_edges is not None:
            n1,n2=self.edges['nodes'][j]
            self._node_to_edges[n1].append(j)
            if n1!=n2:
                self._node_to_edges[n2].append(j)

        self.push_op(self.unadd_edge,j)
        return j

    def unadd_edge(self,j):
        self.delete_edge(j)

    # the delete_* operations require that there are no dependent
    # entities, while the delete_*_cascade operations will check
    # and remove dependent entitites
    @listenable
    def delete_edge(self,j,check_cells=True):
        if check_cells and np.any(self.edges['cells'][j]>=0):
            raise GridException("Edge %d has cell neighbors"%j)
        self.edges['deleted'][j] = True
        if self._node_to_edges is not None:
            n1,n2 = self.edges['nodes'][j]
            self._node_to_edges[n1].remove(j)
            if n1!=n2:
                self._node_to_edges[n2].remove(j)

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
        # This used to add recalc=True, but that actually
        # forces recalculation of *all* edges, not just j.
        # If that becomes a problem, need to root out where
        # edges['cells'] gets corrupted, and not just blindly
        # recalculated all the time.
        c1,c2=self.edge_to_cells(j)
        if c1>=0:
            self.delete_cell(c1)
        if c2>=0 and c1!=c2:
            # rare -- but there can be a degenerate edge poking into a cell
            # interior.
            self.delete_cell(c2)
        self.delete_edge(j)

    def merge_edges(self,edges=None,node=None,_check_existing=True):
        """ Given a pair of edges sharing a node,
        with no adjacent cells or additional edges,
        remove/delete the nodes, combined the edges
        to a single edge, and return the index of the
        resulting edge.
        _check_existing: True: usual check that the newly created
          edge does not already exist. False skips that check. Failing
          this test leaves the grid in a partially updated state!
        """
        if edges is None:
            edges=self.node_to_edges(node)
            assert len(edges)==2
        if node is None:
            Na=self.edges['nodes'][edges[0]]
            Nb=self.edges['nodes'][edges[1]]
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
                # Marks the edge information as stale. Just for safety --
                # we patch it up below.
                self.modify_cell(c,nodes=c_nodes,edges=[self.UNKNOWN])

        # Edge A will be the one to keep
        # modify_edge knows about changes to nodes
        new_nodes=[ self.edges['nodes'][A,1-Ab],
                    self.edges['nodes'][C,1-Cb] ]
        if Ab==0: # take care to preserve orientation
            new_nodes=new_nodes[::-1]

        # A bit sneaky, but rather than removing the cell and adding it back in,
        # just disconnect edge C from the cell so that delete_edge does not complain.
        # are there other places to update this?
        # self.edges['cells'][C,:]=-1
        # if modify_cell were smarter this wouldn't be an issue.
        # as it stands this may not undo correctly
        self.delete_edge(C,check_cells=False)
        # expanding modify_edge into a delete/add allows
        # a ShadowCDT to maintain valid state
        # self.modify_edge(A,nodes=new_nodes)
        # be careful to copy A's entries, as they will get overwritten
        # during the delete/add process.
        edge_data=rec_to_dict(self.edges[A].copy())
        # similar to above.
        #self.edges['cells'][A,:]=-1
        self.delete_edge(A,check_cells=False)
        self.delete_node(B)
        edge_data['nodes']=new_nodes
        self.add_edge(_index=A,_check_existing=_check_existing,**edge_data)
        for c in edge_data['cells']:
            if c>=0:
                self.modify_cell(c,edges=self.cell_to_edges(edge_data['cells'][0], ordered=True))
        return A
    def split_edge_basic(self,j,**node_args):
        """
        The opposite of merge_edges, take an existing edge and insert
        a node into the middle of it.
        Does not allow for any cells to be involved.
        Defaults to midpoint.
        Returns index of new edge and index of new node
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

    def split_edge(self,j=None,x=None,split_cells=True,merge_thresh=-1,**node_args):
        """
        Split an edge, optionally with cells, too.

        split_cells: True: split cells, creating two
         or three cells from the original.
        False: just insert the new node into cells
          Note that this may be ignored if self.max_sides
          is not large enough!

        merge_thresh: if non-negative, check new cells for being joined
        with neighbors to make near-orthogonal quads. a threshold of 0.1
        is reasonable.

        returns 3-tuple:
          j_new - the edge created by splitting j
          n_new - new node at the split
          edges_next_split - list of edge indices that could be split next
          (special case for strip of quads)
        """
        # This edge will get a point inserted at the midpoint
        if j is None:
            j=self.select_edges_nearest(x)

        j_nodes=self.edges['nodes'][j].copy()

        c_nbrs=[]
        for ci,c in enumerate(self.edge_to_cells(j)):
            if c<0:
                continue
            nodes=self.cell_to_nodes(c)
            saved=rec_to_dict(self.cells[c])
            self.delete_cell(c)
            c_nbrs.append( dict(nodes=nodes,c=c) )

        kw={}
        if x is not None:
            kw['x']=x
        kw.update(node_args)
        j_new,n_new=self.split_edge_basic(j,**kw)

        # edges that would be the logical next edge to split if this
        # split is propagating across a series of quads. 0,1 or 2
        # edge indexes
        edges_next_split=[]
        for c_nbr in c_nbrs:
            # the new ring of nodes, with n_new in the right spot
            nodes=list(c_nbr['nodes'])
            na_i=nodes.index(j_nodes[0])
            nb_i=nodes.index(j_nodes[1])

            # then na_i,nb_i are in sequence
            if (nb_i-na_i) % len(nodes) == 1:
                new_i=na_i+1
            elif (na_i-nb_i) % len(nodes) == 1:
                new_i=nb_i+1
            else:
                raise Exception("Failed to insert node in order either way")
            nodes[new_i:new_i]=[n_new]

            if not split_cells:
                if len(c_nbr['nodes'])+1 > self.max_sides:
                    print("Will not insert node into cell because max_sides is too small")
                    split_cells=True
                else:
                    # probably ought to copy more stuff from the original
                    self.add_cell(nodes=nodes)
                    continue
            if split_cells:
                # one triangle going forward:
                new_cells=[]
                rolled=(nodes[new_i:]+nodes[:new_i])
                new_cells.append( rolled[:3] ) # tri_fwd
                new_cells.append( rolled[-2:] + rolled[:1]) # tri_rev

                mid=rolled[2:-1]
                if len(mid)>1:
                    new_cells.append(mid+[n_new]) # poly_mid
                    if len(mid)==2: # special case for quads, queue up next splits
                        j_next=self.nodes_to_edge(mid[0],mid[1])
                        if j_next is not None:
                            logging.info("Queueing up next edge for split %d"%j_next)
                            edges_next_split.append(j_next)

            for new_cell in new_cells:
                self.add_cell_and_edges(nodes=new_cell)

        if merge_thresh>=0:
            self.automerge_cells(n_new,thresh=merge_thresh)

        return j_new,n_new,edges_next_split

    def automerge_cells(self,n,thresh=0.1):
        """
        Check cells around n for potential tri+tri=>quad
        merges opposite n.
        This will not merge two cells adjacent to n
        """
        # get half edge from n_new along j_new
        n_nbrs=self.node_to_nodes(n)
        if len(n_nbrs)==0: return

        cc=self.cells_center()
        A=self.cells_area()

        j_to_merge=[]

        # which cells are about to get merged. with 'realistic'
        # values of thresh, this wouldn't be necessary, but this
        # way avoid outright failure if thresh is so loose as to allow
        # multiple merges on the same cell.
        dirty_cells={}

        for i,n_nbr in enumerate(n_nbrs):
            he=self.nodes_to_halfedge(n,n_nbr)

            c=he.cell()
            if c<0 or c in dirty_cells: continue
            if len(self.cell_to_nodes(c))!=3: continue

            # check both cell opposite n, and successive cells adjacent
            # to n
            for he_check in [he,he.fwd()]:
                c_nbr=he_check.cell_opp()
                if c_nbr<0 or c_nbr in dirty_cells: continue
                if len(self.cell_to_nodes(c_nbr))!=3: continue

                ccA=cc[c]
                ccB=cc[c_nbr]
                coinc=mag(ccA-ccB) / np.sqrt( A[c] + A[c_nbr] )
                if coinc<thresh:
                    j=he_check.j
                    j_to_merge.append(j)
                    # record that these cells will be modified to avoid
                    # conflicts
                    dirty_cells[c]=j
                    dirty_cells[c_nbr]=j
                    break #no need to check other he option

        for j in j_to_merge:
            self.log.info("auto-merging j=%d"%j)
            self.merge_cells(j)

    def add_quad_from_edge(self,j,orthogonal='edge'):
        """
        Add a quad extending from the given edge, with some heuristics 
        on where to place the new node.
        returns {'j_next':next edge up for a step}

        orthogonal: 
          'edge': make the new edge perpendicular to the sequence of edges
           we're running along.  If you're paving a long row, this is probably
           better as it avoids the flip-flop trapezoids
          'cell': make the new cell orthogonal.  For a single new quad, this
           will give a perfectly orthogonal cell.
        """
        assert orthogonal in ['edge','cell']
        
        nodes=self.edges['nodes'][j]
        cells=self.edge_to_cells(j)
        if (cells<0).sum()!=1:
            raise self.GridException("Must have exactly one side edge unpaved")

        # he gets the outward facing half-edge
        he=self.halfedge(j,0)
        if he.cell()>=0:
            he=he.opposite()
        assert he.cell()<0

        he_fwd=he.fwd()
        he_rev=he.rev()

        a,b,c,d=abcd=[he_rev.node_rev(),
                      he.node_rev(),
                      he.node_fwd(),
                      he_fwd.node_fwd()]

        pnts=self.nodes['x'][abcd]
        dpnts=np.diff(pnts,axis=0)
        angles=np.arctan2(dpnts[:,1],dpnts[:,0])
        int_angles=np.diff(angles)

        # wrap it to be in [-180,180]
        int_angles=(int_angles+np.pi)%(2*np.pi) - np.pi

        # In creating the new point, the most obvious choices are to
        # make a trapezoid and the remaining faces are symmetric.
        # there is a choice of which edges to make parallel.  The 
        # more common usage is probably adding a row along the
        # length of a channel, so the selected edge is one of the symmetric
        # edges, not the parallel edge

        j_next=None

        min_int_angle=60*np.pi/180.
        if (int_angles[0]>int_angles[1]) and (int_angles[0]>min_int_angle):
            # quad will be a,b,c,N
            # calculate new 'd'
            # start with symmetric trapezoid
            if orthogonal=='edge':
                z=he_rev.rev().node_rev()
                if z==b:
                    orthogonal='cell' # can't do edge, fall through.
                else:
                    # unit vector at point a along which we place the
                    # new node.
                    a_perp=to_unit(he_rev.rev().normal() + he_rev.normal())
                    # distance perpendicular to ab
                    width= np.dot( he_rev.normal(), pnts[2]-pnts[1])
                    a_dist=width/np.dot( a_perp, he_rev.normal())
                    new_x_d=pnts[0]+a_perp*a_dist
            if orthogonal=='cell':
                new_x_d=pnts[2]+pnts[0]-pnts[1] # parallelogram
                para=to_unit(pnts[0]-pnts[1])
                new_x_d-= 2 * para*np.dot(para,pnts[2]-pnts[1])
                
            new_x=new_x_d
            new_n=self.add_node(x=new_x)
            self.add_edge(nodes=[new_n,c])
            j_next=self.add_edge(nodes=[new_n,a])
            self.add_cell(nodes=[a,b,c,new_n])

        elif (int_angles[1]>int_angles[0]) and (int_angles[1]>min_int_angle):
            if orthogonal=='edge':
                e=he_fwd.fwd().node_fwd()
                if e==c:
                    orthogonal='cell' # folds back on itself.  no go.
                else:
                    d_perp=to_unit(he_fwd.fwd().normal() + he_fwd.normal())
                    width=np.dot(he_fwd.normal(), pnts[1]-pnts[2])
                    d_dist=width/np.dot( d_perp, he_fwd.normal())
                    new_x_a=pnts[3]+d_perp*d_dist

            if orthogonal=='cell':
                # quad will be N,b,c,d
                # calculate new 'a'
                new_x_a=pnts[1]+pnts[3]-pnts[2] # parallelogram
                para=to_unit(pnts[3]-pnts[2])
                new_x_a-=2 * para*np.dot(para,pnts[1]-pnts[2])
                
            new_x=new_x_a
            new_n=self.add_node(x=new_x)
            self.add_edge(nodes=[new_n,b])
            j_next=self.add_edge(nodes=[new_n,d])
            self.add_cell(nodes=[new_n,b,c,d])

        return dict(j_next=j_next)

    def merge_cells_by_circumcenter(self,d_min=None, d_min_rel=0.01):
        """
        Merge adjacent cells when circumcenters are within d_min
        of each other, or d/sqrt(Asum) < d_min_rel
        """
        # signed distance
        e2c=self.edge_to_cells()
        c1=e2c[:,0].copy()
        c2=e2c[:,1].copy()
        c1[c1<0]=c2[c1<0]
        c2[c2<0]=c1[c2<0]
        cc=self.cells_center()
        n=self.edges_normals()
        d=((cc[c2] - cc[c1]) * n).sum(axis=1)
        A=self.cells_area()
        L=np.sqrt( A[c1] + A[c2] )

        sel=False
        if d_min is not None:
            sel=sel | (d<=d_min)
        if d_min_rel is not None:
            sel=sel | (d<=d_min_rel * L)

        sel=sel & (c1!=c2)

        hit_side_limit=0
        for j in np.nonzero(sel)[0]:
            if self.edges['deleted'][j]: continue
            if self.cell_Nsides(c1[j]) + self.cell_Nsides(c2[j]) - 1 >self.max_sides:
                hit_side_limit+=1
                continue
            self.merge_cells(j=j)

        if hit_side_limit:
            print("%d edges could not be merged due to max_sides"%hit_side_limit)
    def merge_cells(self,j=None,x=None):
        """
        Given an edge or point near an edge midpoint,
        merge cells on either side and return the new index.
        Raises exception if there are not cells on both sides.
        """
        if j is None:
            j=self.select_edges_nearest(x)

        # the two cells:
        cells=self.edge_to_cells(j)
        if cells.min()<0:
            raise Exception("merge_cells on edge without two neighors")

        # not the fastest, but easy!
        x=self.edge_center(j)
        self.delete_edge_cascade(j)
        c_new=self.add_cell_at_point(x)
        assert c_new is not None,"Might have a max_sides problem"
        return c_new

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
                    # is this where edges['cells'] should be updated?

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
                        plt.figure(1).clf()
                        self.plot_edges(color='k',lw=0.4)
                        self.plot_edges(mask=[j,jo],color='r',lw=1,labeler='id')
                        self.plot_cells(mask=[c],color='r',alpha=0.3,labeler='id')
                        plt.axis('tight')
                        plt.axis('equal')
                        plt.axis((552453., 552569, 4123965., 4124066.) )
                        raise Exception("Failed in some tedium")
                    if jo_cells[jo_cells_side]>=0:
                        plt.figure(1).clf()
                        self.plot_edges(color='k',lw=0.4)
                        self.plot_edges(mask=[j,jo],color='r',lw=1,labeler='id')
                        self.plot_cells(mask=[c],color='r',alpha=0.3,labeler='id',centroid=True)
                        plt.axis('tight')
                        plt.axis('equal')
                        # plt.axis((552453., 552569, 4123965., 4124066.) )
                        raise Exception("jo_cells[%d]=%s, expected <0. Verify CELL AREA!"%(jo_cells_side,
                                                                        jo_cells[jo_cells_side]))
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
        if check_active and (i>=self.Ncells() or self.cells['deleted'][i]!=False):
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
        for j in self.cell_to_edges(i):
            assert j>=0,"Used to test, but this should always be true"
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
        self.log.info("%s: toggle_cell_at_point()"%self)
        c=self.delete_cell_at_point(x)
        self.log.info("%s: toggle_cell_at_point() deletion yielded %s"%(self,c))

        if c is None:
            c=self.add_cell_at_point(x,**kw)
            self.log.info("%s: toggle_cell_at_point() add yielded %s"%(self,c))
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
        if max_nodes is None:
            max_nodes=self.max_sides
        elif max_nodes<0:
            max_nodes=self.Nnodes()

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
            elif self.cells[k].ndim==2: # catch-all when lengths don't match
                # Need this in cases where there are other per-edge or per-node fields
                # and the source of the data has fewer maxsides than self.
                # Have to punt on the undefined value (though could go with nan if
                # if it's float-valued)
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
        edges=self.cell_to_edges(i,ordered=True)
        nodes=self.cell_to_nodes(i)

        # if 'edges' not in kwargs: 
        # Been having trouble with calls that edges
        # but it's empty. Safer just to set them.
        self.cells['edges'][i,:len(edges)]=edges
        self.cells['edges'][i,len(edges):]=self.UNDEFINED

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
                print("side: %d j=%d n1=%d  n2=%d  edges[nodes][j]=%s"%
                      (side,j,n1,n2,self.edges['nodes'][j,:]))
                assert False # umbra fails here

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
                #raise GridException("not implemented")
                return False
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
    def plot_boundary(self,select_by='cells',**kwargs):
        """
        select_by: 'mark' chooses boundary edges by non-zero edge mark.
        'cells' choose boundary edges by a negative cell neighbor
        """
        if select_by=='cells':
            sel=(self.edge_to_cells().min(axis=1)<0)
        else: # 'mark'
            sel=self.edges['mark']>0

        sel=sel&(~self.edges['deleted'])

        return self.plot_edges(mask=sel,**kwargs)

    def node_clip_mask(self,clip):
        return within_2d(self.nodes['x'],clip)

    def plot_nodes(self,ax=None,mask=None,values=None,sizes=20,labeler=None,clip=None,
                   masked_values=None,label_jitter=0.0,
                   **kwargs):
        """ plot nodes as scatter
        labeler: callable taking (node index, node record), return string
        """
        ax=ax or plt.gca()

        if mask is None:
            mask=~self.nodes['deleted']

        if clip is not None: # convert clip to mask
            mask=mask & self.node_clip_mask(clip)

        if masked_values is not None:
            values=masked_values
        elif values is not None:
            if isinstance(values,six.string_types):
                values=self.nodes[values]
            else:
                values=np.asanyarray(values)

            if len(values)==self.Nnodes():
                values=values[mask]

        kwargs['c']=values

        if labeler is not None:
            if labeler=='id':
                labeler=lambda n,rec: str(n)
            elif labeler in self.nodes.dtype.names:
                field=labeler
                labeler=lambda i,r: str(r[field])

            x=self.nodes['x']
            if label_jitter!=0.0:
                x=x+label_jitter*(np.random.random( (self.Nnodes(),2) )-0.5)
            # weirdness to account for mask being indices vs. bitmask
            for n in np.arange(self.Nnodes())[mask]: # np.nonzero(mask)[0]:
                ax.text(x[n,0],x[n,1], labeler(n,self.nodes[n]))

        coll=ax.scatter(self.nodes['x'][mask][:,0],
                        self.nodes['x'][mask][:,1],
                        sizes,
                        **kwargs)
        bounds=[ self.nodes['x'][mask][:,0].min(),
                 self.nodes['x'][mask][:,0].max(),
                 self.nodes['x'][mask][:,1].min(),
                 self.nodes['x'][mask][:,1].max()]
        if (bounds[0]<bounds[1]) and (bounds[2]<bounds[3]):
            request_square(ax,bounds)
        else:
            # in case the bounds are degenerate
            request_square(ax)

        return coll

    def plot_edges(self,ax=None,mask=None,values=None,clip=None,labeler=None,
                   label_jitter=0.0,lw=0.8,return_mask=False,
                   subedges=None,**kwargs):
        """
        plot edges as a LineCollection.
        optionally select a subset of edges with boolean array mask.
        Note that mask is over all edges, even deleted ones, and overrides
          internal masking of deleted edges.
        and set scalar values on edges with values
         - values can have size either Nedges, or sum(mask)
        labeler: function(id,rec) => string for adding text labels.  Specify 'id'
          for the common case of labeling edges by id, or the name of an edge field
          to label by str(field_value)
        lw: defaults to a thin line, usually more useful with grids, instead of 
          modern matplotlib default which is thick for data plots.
        return_mask: return the line collection and the mask array
        subedges: specify a field name (in the future maybe a lambda) for an
         alternative geometry in the form of an [N,2] coord string.

        Returns: LineCollection, and if return_mask is True then also the mask
        """
        ax = ax or plt.gca()

        edge_nodes = self.edges['nodes']
        if mask is None:
            mask = ~self.edges['deleted']

        if values is not None:
            if isinstance(values,six.string_types):
                values=self.edges[values]
            else:
                # asanyarray allows for masked arrays to pass through unmolested.
                values = np.asanyarray(values)

        if clip is not None:
            mask=mask & self.edge_clip_mask(clip)

        edge_nodes = edge_nodes[mask]
        # try to be smart about when to slice the edge values and when
        # they come pre-sliced
        if values is not None and len(values)==self.Nedges():
            values = values[mask]

        segs = self.nodes['x'][edge_nodes]
        if len(segs):
            # Compute bounds before checking segments. Could be bad
            # if at some point segments is valid but edges are not.
            # But more of the logic in here has to change if segments
            # are totally unrelated to the simple edge (spatial clipping).
            bounds=[segs[...,0].min(),
                    segs[...,0].max(),
                    segs[...,1].min(),
                    segs[...,1].max()]
        else:
            bounds=None
        if isinstance(subedges,six.string_types):
            segs = self.edges[subedges][mask]
             
        if values is not None:
            kwargs['array'] = values

        lcoll = LineCollection(segs,lw=lw,**kwargs)

        if labeler is not None:
            if labeler=='id':
                labeler=lambda i,r: str(i)
            elif labeler in self.edges.dtype.names:
                field=labeler
                labeler=lambda i,r: str(r[field])
                
            ec=self.edges_center()
            if label_jitter!=0.0:
                ec=ec+label_jitter*(np.random.random( (self.Nedges(),2) )-0.5)
                
            # weirdness to account for mask being indices vs. bitmask
            for n in np.arange(self.Nedges())[mask]:
                if subedges is not None:
                    arc=self.edges[subedges][n]
                    pnt=0.5*( arc[arc.shape[0]//2,:] + arc[arc.shape[0]//2-1,:])
                else:
                    pnt=ec[n,:]
                ax.text(pnt[0], pnt[1], labeler(n,self.edges[n]))

        ax.add_collection(lcoll)
        if bounds is not None:
            request_square(ax,bounds)

        if return_mask:
            return lcoll,mask
        else:
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

    def fields_to_xy(self,target,node_fields,x0,eps=1e-6):
        """
        Special purpose method to traverse a pair of node-centered
        fields from x0 to find the point x that would linearly interpolate
        those fields to the value of target.

        target: values of node_fields to locate
        x0: starting point

        NB: edges['cells'] must be up to date before calling
        """
        c=self.select_cells_nearest(x0)

        for loop in range(self.Ncells()):
            # if loop>self.Ncells()-10:
            #     import pdb
            #     pdb.set_trace()
                
            c_nodes=self.cell_to_nodes(c)
            M=np.array( [ node_fields[0][c_nodes],
                          node_fields[1][c_nodes],
                          [1,1,1] ] )
            b=[target[0],target[1],1.0]

            weights=np.linalg.solve(M,b)
            if min(weights)<-eps: # not there yet.
                min_w=np.argmin(weights)
                c_edges=self.cell_to_edges(c,ordered=True)# nodes 0--1 is edge 0, ...
                sel_j=c_edges[ (min_w+1)%(len(c_edges)) ]
                edges=self.edges['cells'][sel_j]
                if edges[0]==c:
                    next_c=edges[1]
                elif edges[1]==c:
                    next_c=edges[0]
                else:
                    raise Exception("Fail.")
                if next_c<0:
                    if weights.min()<-1e-5:
                        print("Left triangulation (min weight: %f)"%weights.min())
                        # Either the starting cell didn't allow a simple path
                        # to the target, or the target doesn't fall inside the
                        # grid (e.g. ragged edge)
                        return [np.nan,np.nan]
                    # Clip the answer to be within this cell (will be on an edge
                    # or node).
                    weights=weights.clip(0)
                    weights=weights/weights.sum()
                    break
                c=next_c
                continue
            else:
                weights=weights.clip(0)
                weights=weights/weights.sum()
                break
        else:
            raise Exception("Failed to terminate in fields_to_xy()")
        x=(self.nodes['x'][c_nodes]*weights[:,None]).sum(axis=0)
        return x
    
    def trace_node_contour(self,cval,node_field,pos_side,
                           n0=None,loc0=None,
                           return_full=False):
        """
        Specialized contour tracing:
         Trace a node-centered contour cval, starting from either node n0
         or an arbitrary point p0, and keeping
         the increasing direction of node_field to pos_side of the
         trace.
        
        n0: starting node (node_field[n0]==cval)
        loc0: explicit starting element, including ('point',None,[x,y])

        pos_side: 'left' or 'right'
        cval: value of the contour to trace.
        node_field: value of field on the nodes.

        return_full: False=> return just the array of point locations.
        True=> return a list of [element type, element id, point].
        element_type: 'cell','edge','node'
        element_id: index of the corresponding element.
        point: coordinate for 0-dimensional intersections, else None
        """
        def he_to_point(he):
            """ 
            Return point along given half-edge that intersects the contour
            """
            nbr_a=he.node_fwd()
            nbr_b=he.node_rev()
            alpha=(cval-node_field[nbr_a])/(node_field[nbr_b]-node_field[nbr_a])

            assert alpha>=0
            assert alpha<=1.0

            pnt=(1-alpha)*self.nodes['x'][nbr_a] + alpha*self.nodes['x'][nbr_b]
            return pnt

        if n0 is not None:
            path=[('node',n0,self.nodes['x'][n0])]
        else:
            path=[loc0]

        def oriented_edge_intersection(nbr_a,nbr_b):
            """
            nbr_a,nbr_b: node indices, with nbr_a to the
            right of nbr_b when looking that direction.
            Check orientation and values. If good, add 
            items to path and return True.
            Else return False.
            """
            # This is problematic when hitting a corner
            # that has only 1 triangle.
            # at 5034.  Just came from 5035, and 5033
            # is off to the left.
            # nbr_a=5033, nbr_b=5035
            #  This is looking back into the domain.
            # nbr_a=5035, nbr_b=5033
            #  This is looking out of the domain.

            j=self.nodes_to_edge([nbr_a,nbr_b])
            if j is None:
                return False # that's not into a cell

            # Are we looking in the correct direction?
            if (pos_side=='right') and not (node_field[nbr_a]>node_field[nbr_b]):
                return False # nope
            if (pos_side=='left') and not (node_field[nbr_a]<node_field[nbr_b]):
                return False # nope

            if node_field[nbr_a]==cval:
                if path[-1][0]=='node':
                    n=path[-1][1]
                    path.append( ('edge',self.nodes_to_halfedge(n,nbr_a),None) )
                path.append( ('node',nbr_a,self.nodes['x'][nbr_a]) )
                return True
            elif node_field[nbr_b]==cval:
                if path[-1][0]=='node':
                    n=path[-1][1]
                    path.append( ('edge',self.nodes_to_halfedge(n,nbr_b),None) )
                path.append( ('node',nbr_b,self.nodes['x'][nbr_b]) )
                return True
            elif (node_field[nbr_a]<cval) == (node_field[nbr_b]>cval):
                he=self.nodes_to_halfedge(nbr_b,nbr_a)
                path.append(('cell',he.cell_opp(),None))
                path.append(('edge',he,he_to_point(he)))
                return True
            else:
                return False
            
        for _ in range(self.Nnodes()+self.Nedges()):
            loc=path[-1]
            
            if loc[0]=='node':
                # Check for adjacent nodes, and adjacent cells.
                n=loc[1]
                n_nbrs=self.angle_sort_adjacent_nodes(loc[1])

                # Check for adjacent cell
                # nbrs are in CCW order.
                for nbr_a,nbr_b in zip(n_nbrs,np.roll(n_nbrs,-1)):
                    # In the case of a corner with 1 triangle, possible
                    # that we've wrapped around.  So check area:
                    A=signed_area( self.nodes['x'][ [n,nbr_a,nbr_b] ])
                    if A<0: continue
                    if oriented_edge_intersection(nbr_a,nbr_b):
                        break # and continue OUTER loop
                else:
                    print("didn't get it")
                    # EXIT OUTER loop
                    break 
            elif loc[0]=='edge':
                he=loc[1]
                if he.cell()<0:
                    break # EXIT OUTER loop
                path.append( ('cell',he.cell(),None) )

                n_opp=he.fwd().node_fwd()

                if node_field[n_opp]==cval:
                    path.append( ('node',n_opp,self.nodes['x'][n_opp]) )
                elif (node_field[n_opp]<cval)==(node_field[he.node_fwd()]<cval):
                    # opp and fwd fall on the same side of the contour
                    he_next=he.rev().opposite()
                    path.append( ('edge', he_next, he_to_point(he_next)) )
                elif (node_field[n_opp]<cval)==(node_field[he.node_rev()]<cval):
                    he_next=he.fwd().opposite()
                    path.append( ('edge',he_next,he_to_point(he_next)) )
                else:
                    ax.plot( [pnts[-1][0]],[pnts[-1][1]],'ro')
                    raise Exception("Failed to find a way out of this cell")
            elif loc[0]=='point':
                pnt=loc[2]

                # The point could be coincident with a node, lie on an edge,
                # or fall within a cell.

                # Point in node:
                n=self.select_nodes_nearest(pnt)
                if mag( pnt - self.nodes['x'][n] )<1e-10:
                    # Might want to replace it in path? ... 
                    path[-1] = ('node',n,self.nodes['x'][n])
                    # Might not...
                    # path.append( ('node',n,self.nodes['x'][n]) )
                    continue
                
                # Point in edge:
                j=self.select_edges_nearest(pnt,fast=False)
                seg=self.nodes['x'][self.edges['nodes'][j]]
                # This epsilon includes a little bit of slop for points constructed
                # along a line with UTM-scaled coordinates ( ~ 1e6 )
                if point_segment_distance(pnt,seg)<1e-8:
                    # Still have to orient the half-edge
                    na,nb=self.edges['nodes'][j]

                    # Are we looking in the correct direction?
                    if (pos_side=='right') and (node_field[na]>node_field[nb]):
                        na,nb=nb,na
                    if (pos_side=='left') and (node_field[na]<node_field[nb]):
                        na,nb=nb,na
                    
                    he=self.nodes_to_halfedge(na,nb)
                    # path.append(('edge',he,pnt)) # add... 
                    path[-1]= ('edge',he,pnt) # ... or replace
                    continue

                # Point in cell:
                c=self.select_cells_nearest(pnt,inside=True)
                if c is None:
                    raise Exception("Couldn't figure out how to start")
                c_nodes=self.cell_to_nodes(c)
                for a,b in zip(c_nodes,np.roll(c_nodes,-1)):
                    if oriented_edge_intersection(a,b):
                        break # and continue OUTER loop
                else:
                    break # EXIT OUTER loop
            else:
                raise Exception("Bad element type %s"%loc[0])
        else:
            self.path_fail=path
            raise Exception("Failed to exit trace contour. see path_fail")

        if return_full:
            return path
        else:
            points=[pnt
                    for typ,idx,pnt in path
                    if pnt is not None]
            trace=np.array(points)
            return trace
    
    def scalar_contour(self,scalar,V=10,smooth=True,boundary='reflect',
                       return_segs=False):
        """ Generate a collection of edges showing the contours of a
        cell-centered scalar.

        V: either an int giving the number of contours which will be
        evenly spaced over the range of the scalar, or a sequence
        giving the exact contour values.

        smooth: control whether one pass of 3-point smoothing is
        applied.

        boundary:
          'reflect' assumes zero gradient at boundaries, such that
             contours will never fall on a boundary.
          numeric value: apply the given constant as the out-of-domain value.

        returns a LineCollection, or a list of segments if return_segs
        """
        if isinstance(V,int):
            V = np.linspace( np.nanmin(scalar),np.nanmax(scalar),V )

        # bin the scalar values
        disc = np.searchsorted(V,scalar) # nan=>last index

        # Start with the 'reflect" approach for boundaries:
        e2c=self.edge_to_cells()
        nc1 = e2c[:,0].copy() # be sure we don't muck with grid internals
        nc2 = e2c[:,1].copy()
        nc2[nc2<0] = nc1[nc2<0]
        nc1[nc1<0] = nc2[nc1<0]
        disc_nc1=disc[nc1] # per-edge discretized scalar value on 'left'
        disc_nc2=disc[nc2] # per-edge discretized scalar value on 'right'

        if boundary!='reflect':
            disc_boundary=np.searchsorted(V,boundary)
            # Edges with cell 0 outside..
            disc_nc1[ (e2c[:,0]<0) ] = disc_boundary
            disc_nc2[ (e2c[:,1]<0) ] = disc_boundary

        to_show = (disc_nc1!=disc_nc2) & np.isfinite(scalar[nc1]+scalar[nc2])

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

        if return_segs:
            return simple_segs
        else:
            from matplotlib import collections
            ecoll = collections.LineCollection(simple_segs)
            ecoll.set_edgecolor('k')

            return ecoll

    def make_triangular(self,record_original=True):
        """
        Adds edges, splits cells, to make the grid only
        triangles.
        record_original: original cell and edge indices are stored
        on the new cells and edges
        """
        if record_original:
            self.add_cell_field('orig_cell',np.arange(self.Ncells()),
                                on_exists='overwrite')
            self.add_edge_field('orig_edge',np.arange(self.Nedges()),
                                on_exists='overwrite')
        if self.max_sides==3:
            return # nothing to do

        splits=[]
        # like self.add_cell_and_edges, but careful to set
        # orig_edge to -1
        def record_add_cell_and_edges(nodes,**kws):
            for a,b in circular_pairs(nodes):
                j=self.nodes_to_edge(a,b)
                if j is None:
                    self.add_edge(nodes=[a,b],orig_edge=-1)
            return self.add_cell(nodes=nodes,**kws)
        
        for c in self.valid_cell_iter():
            nodes=np.array(self.cell_to_nodes(c))

            if len(nodes)==3:
                continue
            self.delete_cell(c)
            record_add_cell_and_edges(nodes=nodes[ [0,1,2] ], orig_cell=c)
            if len(nodes)>=4:
                record_add_cell_and_edges(nodes=nodes[ [0,2,3] ],orig_cell=c )
            if len(nodes)>=5: # a few of these...
                record_add_cell_and_edges(nodes=nodes[ [0,3,4] ],orig_cell=c )
            assert len(nodes)<6,"was lazy about making this generic"

        self.renumber()

    _mpl_tri=None # (tri,srcs) or None
    def mpl_triangulation(self,cell_mask=None,offset=[0,0],return_sources=False,
                          refresh=True):
        """
        Return a matplotlib triangulation for the cells of the grid.
        Only guarantees that the nodes retain their order

        offset: remove the given coordinates (for centering around 0)
        refresh: force recalculation of any cached state

        cell_mask: either bool array with True for included cells, or
         an array of cell indices.
        """
        cacheable=(cell_mask is None) and (offset[0]==0) and (offset[1]==0)
        if ( cacheable
             and (not refresh)
             and (self._mpl_tri is not None)
        ):
            if return_sources:
                return self._mpl_tri
            else:
                return self._mpl_tri[0]
            
        tris=[] # [ (n1,n2,n3), ...]
        srcs=[] # [ c1, c2, c3, c3, c4, ...]
        
        if cell_mask is None:
            cell_mask=np.nonzero( ~self.cells['deleted'] )[0]
        else:
            cell_mask=np.asarray(cell_mask)
            if np.issubdtype(cell_mask.dtype,np.bool_):
                cell_mask=np.nonzero(cell_mask)[0]

        if self.max_sides>3:
            for c in cell_mask:
                nodes=np.array(self.cell_to_nodes(c))

                # this only works for convex cells
                for i in range(1,len(nodes)-1):
                    tris.append( nodes[ [0,i,i+1] ] )
                    srcs.append(c)

            tris=np.array(tris)
        else:
            tris=self.cells['nodes'][cell_mask]
            srcs=np.nonzero(cell_mask)[0]

        x = self.nodes['x'][:,0]
        y = self.nodes['x'][:,1]
        
        if offset is not None:
            x=x-offset[0]
            y=y-offset[1]
            
        tri=Triangulation(x, y, triangles=tris)

        srcs=np.array(srcs)

        if cacheable:
            self._mpl_tri=(tri,srcs)
            
        if return_sources:
            return tri,srcs
        else:
            return tri

    def contourf_node_values(self,values,*args,**kwargs):
        """
        Plot a smooth contour field defined by values at nodes and topology of cells.

        More involved than you might imagine:
         1. Fabricate a triangular version of the grid
         2.
        """
        ax=kwargs.pop('ax',None) or plt.gca()
        tri_kwargs=kwargs.pop('tri_kwargs',{})
        tri=self.mpl_triangulation(**tri_kwargs)
        return ax.tricontourf(tri,values,*args,**kwargs)
    
    def contour_node_values(self,values,*args,**kwargs):
        """
        Plot a smooth contour field defined by values at nodes and topology of cells.
        """
        ax=kwargs.pop('ax',None) or plt.gca()
        tri_kwargs=kwargs.pop('tri_kwargs',{})
        tri=self.mpl_triangulation(**tri_kwargs)
        return ax.tricontour(tri,values,*args,**kwargs)
    
    def average_matrix(self,f=1.0,normalize='area'):
        """
        Smoothing on the grid.  Returns a sparse matrix suitable for repeated
        application to a cell-centered scalar field, each time replacing
        a cell with the average of its neighbors.

        Assume that grid scale is an okay proxy for diffusion rate, so that
        it's better to just average within the neighborhood rather than compute
        diffusivities.

        This is *not* a proper, finite volume diffusion.  It does not conserve mass.

        f: diffusion factor.  1 means replaces each cell with the average of its neighbors.
          0.5 would be to 50% original value, 50% average of neighbors.

        """
        from scipy import sparse
        from scipy.sparse import linalg

        N=self.Ncells()
        D=sparse.dok_matrix((N,N),np.float64)

        for c in range(self.Ncells()):
            nbrs=np.array( self.cell_to_cells(c) )
            nbrs=nbrs[nbrs>=0]
            D[c,c]=1-f
            for nbr in nbrs:
                D[c,nbr] = f/float(len(nbrs))

        return D.tocsr()

    def smooth_matrix(self,f=0.5,K='scaled',dx='grid',V='grid',A='grid',dt=None):
        """
        Smoothing on the grid following finite volume diffusion.
        Returns a sparse matrix suitable for repeated
        application to a cell-centered scalar field, each time replacing
        a cell with the average of its neighbors.

        f: in the default case, a non-dimensional time step
        dx: either an array of cell spacings, 'grid' to pull cell spacings from the
          grid, or None to use 1.0.
        V: either an array of cell 'volumes', 'grid' to pull cell *area* from the grid, 
          or None to use 1.0.
        A: either an array of flux 'areas', 'grid' to pull edge *length* from the grid,
          or None to use 1.0.
        dt: user supplied time step.
        
        The default choices will create a matrix suitable for smoothing a 2D concentration
        field.  K will be scaled such that coarse regions of the grid have a higher effective
        diffusion coefficient.

        No attempt is made to include depth information -- to properly diffuse a 3D scalar
        field requires passing in the proper depth-inclusive values for V and A.
        """
        from scipy import sparse
        from scipy.sparse import linalg

        Nc=self.Ncells()
        Nj=self.Nedges()
        D=sparse.dok_matrix((Nc,Nc),np.float64)

        e2c=self.edge_to_cells()
        all_j=np.arange(Nj)
        internal=np.all(e2c>=0,axis=1)

        # the pure smoothing case
        # this is already more mass conservative than the simple averaging, as
        # fluxes at least balance.
        if isinstance(dx,np.ndarray):
            pass # user supplied data
        elif dx is None:
            dx=np.ones(Nj,np.float64)
        elif dx=='grid':
            cc=self.cells_center()
            dx=np.ones(Nj,np.float64)
            dx[internal]=mag(cc[e2c[internal,0]] - cc[e2c[internal,1]])
        else:
            raise Exception("did not understand dx argument %s"%str(dx))

        if isinstance(A,np.ndarray):
            pass
        elif A=='dx':
            A=dx
        elif A=='grid':
            A=self.edges_length()
        elif A is None:
            A=np.ones(Nj,np.float64)
        else:
            raise Exception("did not understand A argument %s"%str(A))

        if isinstance(V,np.ndarray):
            pass
        elif V=='grid':
            V=self.cells_area()
        elif V is None:
            V=np.ones(Nc,np.float64)
        else:
            raise Exception("Did not understand V argument %s"%str(V))

        if isinstance(K,np.ndarray):
            pass
        elif K=='scaled':
            K=np.ones(Nj,np.float64)
            # some judgement call here in whether to scale with the average of the
            # volumes, or minimum.
            K[internal]=0.5*dx[internal]*(V[e2c[internal,0]] + V[e2c[internal,1]])/A[internal]
        elif K is None:
            K=np.ones(Nj,np.float64)

        if dt is None:
            dt=float(f)/self.max_sides # eh

        for c in range(self.Ncells()):
            D[c,c]=1

        for j in progress(all_j[internal]):
            nc1,nc2=e2c[j]
            fac=dt*K[j]*A[j]/dx[j]
            D[nc1,nc1] += -fac/V[nc1]
            D[nc2,nc2] += -fac/V[nc2]
            D[nc1,nc2] += fac/V[nc1]
            D[nc2,nc1] += fac/V[nc2]

        for c in range(self.Ncells()):
            if D[c,c]<0:
                print("Matrix is not positive-definite. May be unstable - decrease f.")

        return D.tocsr()

    def edge_clip_mask(self,xxyy,ends=False):
        """
        return a bitmask over edges falling in the boundiny box.
        if ends is True, test against the bbox of the edge, not
        just its center.
        """
        if not ends:
            centers=self.edges_center()
            xmin=xmax=centers[:,0]
            ymin=ymax=centers[:,1]
        else:
            nxy=self.nodes['x'][self.edges['nodes']]
            xmin=nxy[:,:,0].min(axis=1)
            xmax=nxy[:,:,0].max(axis=1)
            ymin=nxy[:,:,1].min(axis=1)
            ymax=nxy[:,:,1].max(axis=1)

        return (xmax>xxyy[0]) & (xmin<xxyy[1]) & \
            (ymax > xxyy[2]) & (ymin<xxyy[3])

    def cell_clip_mask(self,xxyy,by_center=True):
        """
        Calculate boolean mask of cells falling within the bounds xxyy.
        xxyy: [xmin,xmax,ymin,ymax]
        by_center: by default cell centers (circumcenters) are tested against
         the bounds. If False, test the cell envelope against xxyy. For a precise 
         test of the exact cell geometry, use select_cells_intersecting.

        Note that there is some precalculation below which can be slow and is
        not currently cached. As such, if many repeated calls are expected it may
        be *much* faster to deconstruct this method (or update the code here to
        cache xmin,xmax,ymin,ymax).
        """
        if by_center:
            centers=self.cells_center()
            return  (centers[:,0] > xxyy[0]) & (centers[:,0]<xxyy[1]) & \
                (centers[:,1] > xxyy[2]) & (centers[:,1]<xxyy[3])
        else:
            # test cell bounds
            nodes=self.cells['nodes']
            x=np.where( nodes>=0, self.nodes['x'][nodes,0], np.nan )
            y=np.where( nodes>=0, self.nodes['x'][nodes,1], np.nan )

            xmin=np.nanmin(x,axis=1)
            ymin=np.nanmin(y,axis=1)
            xmax=np.nanmax(x,axis=1)
            ymax=np.nanmax(y,axis=1)

            cell_valid=(xmin<xxyy[1])&(xmax>xxyy[0])&(ymin<xxyy[3])&(ymax>xxyy[2])
            return cell_valid

    def tripcolor_cell_values(self,values,ax=None,**kw):
        """
        Plot cell values using matplotlib's tripcolor. The main advantage
        compared to plot_cells is that the result is truly seamless, without having
        to use finite thickness edges.
        """
        tri,sources = self.mpl_triangulation(return_sources=True,refresh=False)
        if ax is None:
            ax=plt.gca()
        return ax.tripcolor(tri, values[sources], **kw)
    
    def plot_cells(self,ax=None,mask=None,values=None,clip=None,centers=False,labeler=None,
                   masked_values=None,ragged_edges=None,
                   centroid=False,subedges=None,**kwargs):
        """
        values: color cells based on the given values.  can also be
          the name of a field in self.cells.
        masked_values: same as values, but just for elements in mask
        centers: scatter plot of cell centers.  otherwise polygon plot
        labeler: f(cell_idx,cell_record) => string for labeling.
        centroid: if True, use centroids instead of centers.  if an array,
          use that as the center point rather than circumcenters or centroids
        ragged_edges: controls how cells with fewer than max_sides edges are handled.
          False: nan-mask extra edges (but edges won't plot correctly), True: pass
          a ragged list of arrays. None: choose based on whether edgecolor is specified.
        subedges: a field on edges that provides an alternate geometry as a linestring 
        [N,2].
        """
        ax = ax or plt.gca()

        if values is not None:
            if isinstance(values,six.string_types):
                values=self.cells[values]
            else:
                # asanyarray allows for masked arrays to pass through unmolested.
                values = np.asanyarray(values)

        if ragged_edges is None:
            ragged_edges='edgecolor' in kwargs

        if mask is None:
            mask=~self.cells['deleted']
        else:
            # force to a bitmask, even though could be inefficient
            mask=np.asarray(mask)
            if np.issubdtype(mask.dtype,np.integer):
                bitmask=np.zeros(self.Ncells(),bool) # np.bool deprecated
                bitmask[mask]=True
                if masked_values is not None:
                    masked_values=masked_values[ np.argsort(mask) ]
                mask=bitmask

        if clip is not None: # convert clip to mask
            mask=mask & self.cell_clip_mask(clip,by_center=False)

        if len(mask)==0 or np.all(~mask):
            return
        
        if values is not None and len(values)==self.Ncells():
            values = values[mask]
        elif masked_values is not None:
            values = masked_values

        if values is not None and not np.issubdtype(values.dtype,np.number):
            # Hack to scalarize categorical data. This will still fail
            # if the type is not comparable.
            uniq = np.unique(values)
            values = np.searchsorted(uniq,values)
            
        if centers or labeler:
            if isinstance(centroid,np.ndarray):
                xy=centroid
            elif centroid:
                xy=self.cells_centroid()
            else:
                xy=self.cells_center()
        else:
            xy=None # unused

        if not centers:
            if values is not None:
                kwargs['array'] = values
            cell_nodes = self.cells['nodes'][mask]

            # do this regardless of settings in order to get bounds.
            polys = self.nodes['x'][cell_nodes]
            missing = cell_nodes<0
            polys[missing,:] = np.nan # seems to work okay for triangles

            if subedges is None:                
                if ragged_edges:
                    # slower, but properly shows edges
                    plot_polys = [ self.nodes['x'][cn[ cn>=0 ]]
                                   for cn in cell_nodes]
                else:
                    plot_polys=polys
            else:
                plot_polys = [self.cell_coords_subedges(c,subedges)
                              for c in np.nonzero(mask)[0]]
                if not isinstance(centroid,np.ndarray) and centroid:
                    # override with representative point
                    xy = np.concatenate( [geometry.Polygon(poly).representative_point().coords
                                          for poly in plot_polys])
                
            coll = PolyCollection(plot_polys,**kwargs)
            ax.add_collection(coll)
            
            # We're
            bounds=[np.nanmin( polys[...,0]),
                    np.nanmax( polys[...,0]),
                    np.nanmin( polys[...,1]),
                    np.nanmax( polys[...,1])]
        else:
            args=[]
            if values is not None:
                args.append(values)
            coll = ax.scatter(xy[mask,0],xy[mask,1],20,*args,**kwargs)
            
            bounds=[np.nanmin( xy[mask,0]),
                    np.nanmax( xy[mask,0]),
                    np.nanmin( xy[mask,1]),
                    np.nanmax( xy[mask,1])]

        if labeler is not None:
            if labeler=='id':
                labeler=lambda i,r: str(i)
            elif labeler in self.cells.dtype.names:
                field=labeler
                labeler=lambda i,r: str(r[field])
                
            for c in np.nonzero(mask)[0]:
                ax.text(xy[c,0],xy[c,1],labeler(c,self.cells[c]))

        
        if (bounds[0]<bounds[1]) and (bounds[2]<bounds[3]):
            request_square(ax,bounds)
        else:
            # in case the bounds are degenerate
            request_square(ax)
                
        return coll

    def tripcolor_cell_values(self,values,ax=None,refresh=False,**kw):
        tri,sources = self.mpl_triangulation(return_sources=True,refresh=refresh)
        if ax is None: ax=plt.gca()

        return ax.tripcolor(tri,values[sources],**kw)
    
    def edges_length(self,sel=None):
        if sel is None:
            lengths=np.full(self.Nedges(),np.nan,np.float64)
            sel=~self.edges['deleted']
            p1 = self.nodes['x'][self.edges['nodes'][sel,0]]
            p2 = self.nodes['x'][self.edges['nodes'][sel,1]]
            lengths[sel]=mag( p2-p1 )
        else:
            p1 = self.nodes['x'][self.edges['nodes'][sel,0]]
            p2 = self.nodes['x'][self.edges['nodes'][sel,1]]
            lengths=mag( p2-p1 )
        return lengths

    def cells_area(self,sel=None,subedges=None):
        """
        sel: list/array of cell indices to calculate.  Can be multidimensional,
          but cannot be a bitmask.
        defaults to cells which have a nan area
        subedges: include sub-edge geometry given by edges[subedges]. Disables updates to _area.
        """
        if subedges is not None:
            areas=np.full(self.Ncells(), np.nan) # could be wasteful...
            if sel is None:
                sel=np.arange(self.Ncells())
                cells=self.valid_cell_iter()
            else:
                cells=sel
            for c in cells:
                coords=self.cell_coords_subedges(c,subedges=subedges)
                areas[c] = signed_area(coords)
            return areas[sel]

        
        if sel is None:
            recalc=np.nonzero( np.isnan(self.cells['_area']) & (~self.cells['deleted']))[0]
        else:
            recalc=np.asarray(sel).ravel()

        for c in recalc:
            self.cells['_area'][c] = signed_area(self.nodes['x'][self.cell_to_nodes(c)])

        if sel is None:
            sel=slice(None)
        return self.cells['_area'][sel]

    #-# Selection methods:
    #  various methods which return a bitmask over cells, edges or nodes
    #  though in some cases it's more efficient to deal with index lists, it's much
    #  easier to compose selections with bitmasks, so there it is.
    #  when implementing these, note that all selections should avoid 'deleted' elements.
    #   unless the selection criteria explicitly includes them.
    def select_edges_by_polyline(self,geom,rrtol=3.0,update_e2c=True,
                                 boundary=True,return_nodes=False):
        """
        same as dfm_grid.polyline_to_boundary_edges:

        Mimic FlowFM boundary edge selection from polyline to edges.
        Identifies boundary edges which would be selected as part of the
        boundary group.

        linestring: [N,2] polyline data, or shapely LineString geometry
        rrtol: controls search distance away from boundary. Defaults to
        roughly 3 cell length scales out from the boundary.

        update_e2c: if True, will call edge_to_cells() first.  This is safest, but
        can be slow.  If the caller can guarantee that edges['cells'] is up to date,
        set this to False to avoid the overhead. When boundary=False, e2c isn't used 
        and this parameter

        boundary: if True follow the DFM intention, limiting the search to boundary
        edges. if False, then this extracts paths within the grid, based on the shortest
        path between points on the linestring

        returns ndarray of edge indices, unless return_nodes is given in which case 
        in-order node indices are returned.
        """
        if isinstance(geom,geometry.LineString):
            linestring=np.array(geom.coords)
        else:
            linestring=np.asanyarray(geom)


        if boundary:
            if update_e2c:
                self.edge_to_cells()
            
            boundary_edges=np.nonzero( np.any(self.edges['cells']<0,axis=1) )[0]
            adj_cells=self.edges['cells'][boundary_edges].max(axis=1)

            adj_centers=self.cells_center()[adj_cells]
            edge_centers=self.edges_center()[boundary_edges]
            cell_to_edge=edge_centers-adj_centers
            cell_to_edge_dist=mag(cell_to_edge)
            outward=-self.edges_normals(edges=boundary_edges,force_inward=True,update_e2c=update_e2c)

            dis=np.maximum( 0.5*np.sqrt(self.cells_area()[adj_cells]),
                            cell_to_edge_dist )
            probes=edge_centers+(2*rrtol*dis)[:,None]*outward
            segs=np.array([adj_centers,probes]).transpose(1,0,2)
            linestring_geom=geometry.LineString(linestring)

            probe_geoms=[geometry.LineString(seg) for seg in segs]

            hits=[idx
                  for idx,probe_geom in enumerate(probe_geoms)
                  if linestring_geom.intersects(probe_geom)]
            edge_hits=boundary_edges[hits]
        else:
            # Map linestring vertices to nodes
            ls_nodes=[self.select_nodes_nearest(p) for p in linestring]
            all_nodes=[]
            edge_hits=[]
            for a,b in zip(ls_nodes[:-1],ls_nodes[1:]):
                if a==b: continue
                edge_hits.extend( self.shortest_path(a,b,return_type='edges') )

        if return_nodes:
            pieces = {} # ending nodes => node string
            def pop_oriented(n): # remove both entries for pieces[n], return ns starting with n
                ns = pieces.pop(n)
                if ns[0]!=n:
                    ns=ns[::-1]
                del pieces[ns[-1]]
                return ns
                
            def add_node_string(ns):
                ns=list(ns)
                if ns[0] in pieces:
                    # flips keep the other end of ns at the other end
                    ns = pop_oriented(ns[0])[::-1] + ns[1:]
                if ns[-1] in pieces:
                    ns = ns[:-1] + pop_oriented(ns[-1])
                pieces[ns[0]]  = ns
                pieces[ns[-1]] = ns
                    
            for j in edge_hits:
                add_node_string(self.edges['nodes'][j])

            if len(edge_hits)==0:
                return []
            else:
                # probably too strict, but caller probably doesn't expect
                # disconnected segments.
                assert len(pieces)==2
                return pieces.popitem()[1]
            return nodes
        
        return edge_hits
    
    def select_edges_intersecting(self,geom,invert=False,mask=slice(None),
                                  by_center=False,as_type='mask'):
        """
        geom: a shapely geometry
        returns: bitmask over edges, with non-deleted, selected edges set and others False.
        if invert is True, select edges which do not intersect the the given geometry.
        mask: bitmask or index array to limit the selection to a subset of edges.
        note that the return value is still a bitmask over the whole set of edges, not just
        those in the mask
        by_center: test the midpoint, rather than the line segment
        """
        sel = np.zeros(self.Nedges(),np.bool_) # initialized to False
        if by_center:
            centers=self.edges_center()
            
        for j in np.arange(self.Nedges())[mask]:
            if self.edges['deleted'][j]:
                continue
            if by_center:
                edge_line = geometry.Point(centers[j])
            else:
                edge_line = geometry.LineString(self.nodes['x'][self.edges['nodes'][j]])
            sel[j] = geom.intersects(edge_line)
            if invert:
                sel[j] = ~sel[j]
                
        if as_type=='indices':
            return np.nonzero(sel)[0]
        else:
            return sel

    def cell_polygon(self,c,subedges=None):
        if subedges is None:
            coords=self.nodes['x'][self.cell_to_nodes(c)]
        else:
            coords=self.cell_coords_subedges(c,subedges=subedges)
        return geometry.Polygon(coords)

    def edge_line(self,e):
        return geometry.LineString(self.nodes['x'][self.edges['nodes'][e]])

    def node_point(self,n):
        return geometry.Point( self.nodes['x'][n] )

    def boundary_linestrings(self,return_nodes=False,sort=False):
        """
        Extract line strings for all boundaries, as determined by
        edges having less than two adjacent cells.

        returns a list of arrays.
        defaults to arrays of xy coordinates
        return_nodes: return arrays of node indices

        sort: order the linestrings by absolute area, with the
         largest having positive CCW area, and all others having
         positive CW area. Does not attempt to resolve pond-on-an-island
         nested boundaries.
        """
        # could be much smarter and faster, directly traversing boundary edges
        # but this way is easy
        e2c=self.edge_to_cells()
        # some grids don't abide by the boundary always being on the "right"
        # so use any()
        boundary_edges=(np.any(e2c<0,axis=1))&(~self.edges['deleted'])

        marked=np.zeros(self.Nedges(),np.bool_)
        lines=[]
        for j in np.nonzero(boundary_edges)[0]:
            if marked[j]:
                continue
            trav=self.halfedge(j,0)
            if trav.cell()>=0:
                trav=trav.opposite()
            assert trav.cell()<0
            start=trav
            this_line_nodes=[trav.node_rev()]
            while 1:
                if marked[trav.j]:
                    print("maybe hit a dead end -- boundary maybe not closed")
                    print("edge centered at %s traversed twice"%(self.edges_center()[trav.j]))
                    #import pdb
                    #pdb.set_trace()
                    raise Exception("Hit a dead end -- boundary maybe not closed."
                                    + "edge centered at %s traversed twice"%(self.edges_center()[trav.j]))
                this_line_nodes.append(trav.node_fwd())
                marked[trav.j]=True
                trav=trav.fwd()
                # in grids with no cell marks, trav test appears
                # unreliable
                if trav==start:
                    break
                if this_line_nodes[-1]==this_line_nodes[0]:
                    print("trav was different, but nodes are the same")
                    break
            lines.append( np.array(this_line_nodes) )
            assert np.all( np.diff(lines[-1])!=0 )
        if sort:
            areas=np.array( [signed_area(self.nodes['x'][l]) for l in lines] )
            order=np.argsort(-np.abs(areas))
            areas=areas[order]
            lines=[ lines[i] for i in order ]
            for i,line in enumerate(lines):
                if (i==0)!=(areas[i]>0):
                    lines[i]=lines[i][::-1]
                    
        if not return_nodes:
            lines=[self.nodes['x'][line] for line in lines]
            
        return lines

    def boundary_polygon_by_edges(self,allow_multiple=False):
        """ return polygon, potentially with holes, representing the domain.
        equivalent to unioning all cell_polygons, but hopefully faster.
        in one test, this method was 3.9 times faster than union.  This is
        certainly depends on the complexity and size of the grid, though.

        allow_multiple: generally the grid is contiguous, and there can only be
         one boundary polygon (possibly with holes).  when multiple polygons
         are found, that is usually interpreted as an error in topology.  allow_multiple
         will instead return a list of polygons.
        """
        lines=self.boundary_linestrings()
        polys,extras=join_features.lines_to_polygons(lines,close_arc=False,single_feature=False)
        if len(polys)>1 and not allow_multiple:
            raise GridException("somehow there are multiple boundary polygons")
        if len(extras)>1:
            raise GridException("not all boundary edges were in a boundary polygon?")
        if allow_multiple:
            return polys
        else:
            return polys[0]

    def boundary_polygon_by_union(self,cells=None):
        """ Compute a polygon encompassing the full domain by unioning all
        cell polygons.
        cells: optionally a sequence or bitmask of cell ids to be used, otherwise
        all non-deleted cells.
        """
        if cells is None:
            cell_geoms = [None]*self.Ncells()
            cells=self.valid_cell_iter()
        else:
            if np.issubdtype(cells.dtype, np.bool_):
                cells=np.nonzero(cells)[0]
            cell_geoms = [None]*len(cells)

        for idx,i in enumerate(cells):
            xy = self.nodes['x'][self.cell_to_nodes(i)]
            cell_geoms[idx] = geometry.Polygon(xy)
        del cell_geoms[(idx+1):]
        #return ops.cascaded_union(cell_geoms)
        # Updated api as of 2022-02-22
        return ops.unary_union(cell_geoms)

    def boundary_polygon(self):
        """ return polygon, potentially with holes, representing the domain.
        This method tries an edge-based approach, but will fall back to unioning
        all cell polygons if the edge-based approach fails.
        """
        try:
            # TODO: why does this fail so often? Is it stale edge_to_cells? 
            return self.boundary_polygon_by_edges()
        except Exception as exc:
            self.log.info('Warning, boundary_polygon() failed using edges!  Trying polygon union method')
            # self.log.warning(exc,exc_info=True)
            return self.boundary_polygon_by_union()

    def area_total(self):
        """
        Total area of the grid cells.  Currently computed by summing per-cell
        areas, though there are more efficient ways (probably).
        """
        return self.cells_area().sum()

    def extract_linear_strings(self,edge_select=None,end_func=None):
        """
        extract contiguous linestrings as sequences of nodes.
        
        end_func: lambda node, [list of edges]: True/False
           test function to say whether the given node should be the end 
           of a linear string.
        """
        # there are at least three choices of how greedy to be.
        #  min: each edge is its own feature
        #  max: extract features as long as possible, and allow for 'T' junctions.
        #  mid: break features at nodes with degree>2.
        # go with mid
        strings=[]
        edge_marks=np.zeros(self.Nedges(), np.bool_)

        def degree2_predicate(n,js):
            """
            returns true if the node n, with selected edges js,
            should be considered an end of a string.
            """
            return len(js)!=2

        if end_func is None:
            end_func=degree2_predicate

        for j0 in self.valid_edge_iter():
            if (edge_select is not None) and (not edge_select[j0]):
                continue
            if edge_marks[j0]:
                continue
            edge_marks[j0]=True

            # trav=tuple(self.edges['nodes'][j])
            node_fwd=self.edges['nodes'][j0,1]
            node_rev=self.edges['nodes'][j0,0]

            node_string=[node_fwd,node_rev]

            for trav in [ (node_fwd,node_rev),
                          (node_rev,node_fwd) ]:
                while 1:
                    js = self.node_to_edges(trav[1])

                    if edge_select is not None:
                        js=[j for j in js if edge_select[j]]

                    # if len(js)!=2:
                    if end_func(trav[1],js):
                        break

                    for j in js:
                        jnodes=self.edges['nodes'][j]
                        if trav[0] not in jnodes:
                            break
                    else:
                        assert False
                        
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

    def select_quad_subset(self,ctr,max_cells=None,max_radius=None,node_set=None,
                           return_full=False):
        """
        Starting from ctr, select a contiguous set of nodes connected 
        by quads, up to max_cells and within max_radius of the starting
        point.

        Alternatively, specify a list of node indices in node_set, and the search
        is limited to traversal among that set of nodes.

        if return_full is True, return a single 2D array with node, edge and cell indices.
        [even,even] indices are nodes, [even,odd] and [odd,even] are edges, and [odd,odd]
        are cells.

        Returns (node_idxs,ij)
          node_idxs: array of indices into self.nodes
          ij: cartesian indices for each of those nodes.

        The orientation of ij is arbitrary.  The origin is near ctr (but not
        necessarily that the node closest to ctr is (0,0)
        ij values include negatives
        """
        rotL=np.array( [[0,-1],[1,0]] )
        rotR=np.array( [[0,1],[-1,0]] )

        stack=[]

        node_ij={} # map nodes to their ij index
        visited_cells={} # cell index => the index, 0..3, of its node that has the min i and min j.

        if node_set is None:
            j=self.select_edges_nearest(ctr)
            he=self.halfedge(j,0)
            if he.cell()<0:
                he=he.opposite()
            assert he.cell()>=0
        else:
            n=node_set[0]
            node_set=set(node_set)
            he=None
            for j in self.node_to_edges(n):
                if ( (self.edges['nodes'][j,0] in node_set)
                      and
                     (self.edges['nodes'][j,1] in node_set) ):
                    he=self.halfedge(j,0)
                    if he.cell()<0:
                        he=he.opposite()
                    if he.cell()<0:
                        he=None
                        continue
                    break
            if he is None:
                # could try harder with other nodes..
                print("Failed to find an edge with first node of set.")
                return np.zeros(0,np.int32),np.zeros((0,2),np.int32)
                
        node_ij[ he.node_rev() ] = np.array([0,0])

        # stack is a half edge, meaning visit the cell the half edge is facing.
        # node_ij[node_rev()] is guaranteed to be populated.
        # and dir gives the ij vector for the edge normal (into the new cell)
        # if node_set is given, then both nodes of the half edge are guaranteed to be in node_set
        self.edge_to_cells()

        stack.append( (he, np.array([1,0]) ) )

        cc=self.cells_center()

        while stack:
            he,vecnorm = stack.pop(0)
            c=he.cell()
            if (c in visited_cells) or (c<0):
                continue

            if (max_radius is not None) and (mag(cc[c]-ctr)>max_radius):
                continue

            visited_cells[c]=True

            # Be sure search is breadth first, and we stop
            # with a given count.
            if (max_cells is not None) and (len(visited_cells)>max_cells):
                break

            assert he.node_rev() in node_ij

            if self.cell_Nsides(c)!=4:
                continue

            he_trav=he
            ij_norm=vecnorm
            # node_ij[nrev] may not be saved if nrev isn't valid
            # so traverse ij_rev manually
            ij_rev=node_ij[he.node_rev()]
            
            for i in range(4):
                nrev=he_trav.node_rev()
                nfwd=he_trav.node_fwd()
                ij_fwd=ij_rev + rotR.dot(ij_norm)

                if ((node_set is None) or (nfwd in node_set)):
                    # valid  node
                    if nfwd in node_ij:
                        # probably will have to relax this.
                        assert np.all(node_ij[nfwd]==ij_fwd)
                    else:
                        node_ij[nfwd]=ij_fwd

                    if nrev in node_ij:
                        # both ends of the half edge have ij,
                        # and both are valid.
                        # queue a visit to trav's opposite
                        he_opp=he_trav.opposite()
                        stack.append( (he_opp,-ij_norm) )

                # And move to next face of quad
                he_trav=he_trav.fwd()
                ij_norm=rotL.dot(ij_norm)
                ij_rev=ij_fwd

        node_idxs=np.array( list(node_ij.keys()) )
        ij=np.array( [node_ij[n] for n in node_idxs] )

        if not return_full:
            return node_idxs, ij

        # Convert to full index representation.
        # shift so ij are zero-based.
        ij[:,:] -= ij.min(axis=0)
        rows,cols=ij.max(axis=0) # This will be number of cells
        IJ=-np.ones([2*rows+1,2*cols+1],np.int32)
        for n,n_ij in zip(node_idxs,ij): IJ[2*n_ij[0],2*n_ij[1]]=n
        for i in range(2*rows+1):
            for j in range(2*cols+1):
                if i%2==0 and j%2==0: continue
                if i%2==1 and j%2==0:
                    n1=IJ[i-1,j]
                    n2=IJ[i+1,j]
                    if n1<0 or n2<0: continue
                    IJ[i,j]=self.nodes_to_edge(n1,n2)
                elif i%2==0 and j%2==1:
                    n1=IJ[i,j-1]
                    n2=IJ[i,j+1]
                    if n1<0 or n2<0: continue
                    IJ[i,j]=self.nodes_to_edge(n1,n2)
                else:
                    nodes=np.r_[ IJ[i-1,j-1],
                                IJ[i-1,j+1],
                                IJ[i+1,j-1],
                                IJ[i+1,j+1] ]
                    if np.all(nodes>=0):
                        c = self.nodes_to_cell(nodes,fail_hard=False)
                        if c is not None:
                            IJ[i,j] = c
        return IJ
        
    def select_nodes_boundary_segment(self, coords, ccw=True):
        """
        bc_coords: [ [x0,y0], [x1,y1] ] coordinates, defining
        start and end of boundary segment, traversing CCW boundary of
        grid.

        if ccw=False, then traverse the boundary CW instead of CCW.

        returns [n0,n1,...] nodes along boundary between those locations.

        This does not currently support islands.
        """
        self.edge_to_cells()
        cycle=np.asarray( self.boundary_cycle() )
        #start_n,end_n=[ self.select_nodes_nearest(xy)
        #                for xy in coords]

        nodes=[]
        for xy in coords:
            dists=mag( self.nodes['x'][cycle] - xy)
            nodes.append( cycle[ np.argmin(dists) ] )
        start_n,end_n=nodes
        
        start_i=np.nonzero( cycle==start_n )[0][0]
        end_i=np.nonzero( cycle==end_n )[0][0]

        if start_i<end_i:
            boundary_nodes=cycle[start_i:end_i+1]
        else:
            boundary_nodes=np.r_[ cycle[start_i:], cycle[:end_i]]
        return boundary_nodes

    def select_nodes_intersecting(self,geom=None,xxyy=None,invert=False,as_type='mask'):
        sel = np.zeros(self.Nnodes(),np.bool_) # initialized to False

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

    def select_cells_intersecting(self,geom,invert=False,as_type="mask",by_center=False,
                                  order=False, return_distance=False):
        """
        geom: a shapely geometry
        invert: select cells which do not intersect.
        as_type: 'mask' returns boolean valued mask, 'indices' returns array of indices
        by_center: if True, test against the cell center.  By default, tests against the
        finite cell.
         if 'centroid', test against the centroid
        order: if True and the input geometry is a linestring, order the cells
         by distance along the linestring. force as_type='indices'
        return_distance: with order -- return the distance along the transect too
        """        
        if geom.geom_type=='LineString' and order:
            as_type='indices'

        if isinstance(as_type,str) and as_type=='mask':
            sel = np.zeros(self.Ncells(),np.bool_) # initialized to False
        else:
            sel = []

        if by_center=='centroid':
            centers=self.cells_centroid()
        elif by_center=='representative':
            centers=self.cells_representative_point()
        elif by_center:
            centers=self.cells_center()

        for c in range(self.Ncells()):
            if self.cells['deleted'][c]:
                continue
            if by_center:
                test=geom.intersects( geometry.Point(centers[c]) )
            else:
                test=geom.intersects( self.cell_polygon(c) )
            if invert:
                test = not test # not in numpy land, so don't invert with ~
            if isinstance(as_type,str) and as_type=='mask':
                sel[c] = test
            else:
                if test:
                    sel.append(c)

        if geom.geom_type=='LineString' and order:                    
            sel=np.array(sel)
            centers=self.cells_center()[sel]
            dist_along=np.array( [geom.project(geometry.Point(center))
                                  for center in centers] )
            ordering=np.argsort(dist_along)
            sel=sel[ordering]
            if return_distance:
                return sel, dist_along[ordering]

        return sel

    def points_to_cells(self,points,method='kdtree'):
        """
        Map a large number of points to the containing cells.
        This can achieve some significant speedups, but much is
        in the details.

        method: 
        'point_index': construct a point index (via gen_spatial_index,
           which is usually based on libspatialindex r-trees), which 
           is then queried per cell, and refined via cell_path.
        'cell_index': build a rectangular index of cell bounding boxes, 
           then iterate over points and query the cell index.
        'cells_nearest':
           iterate and use self.select_cells_nearest. This allows near
           matches, unlike the other methods which require points to fall
           inside the cells
        'kdtree': build a kdtree of the points, query by circumcenter and
           circumradius of each cell, and refine via cell_path.
        'mpl': build a matplotlib triangulation (which is cached on self)
           and matplotlib's TriFinder. This is potentially very fast and
           accurate, assuming matplotlib is available, but has also been
           unstable in some cases, causing python to crash.

        For small inputs, 'cells_nearest' may be fastest, especially if
        a cell index is already built.
        
        For large inputs, kdtree is fast. A test with 500k points and a
        grid with 57k cells:
          point_index: 10.3s
          cell_index: 23.2s
          cells_nearest: 85.9s
          kdtree: 3.3s
        """
        cells=-np.ones(len(points),np.int32)

        if method=='cell_index':
            # and compare to building a finite-rectangle index for
            # the cells
            boxes=np.zeros( (self.Ncells(),4), np.float64)
            paths=[None]*self.Ncells()

            # the setup part:
            for c in range(self.Ncells()):
                p=paths[c]=self.cell_path(c)
                verts=p.vertices
                boxes[c,:]=[verts[:,0].min(), verts[:,0].max(),
                            verts[:,1].min(), verts[:,1].max()]
            cell_index=gen_spatial_index.RectIndex(zip(range(self.Ncells()),
                                                       boxes,
                                                       [None]*self.Ncells()),
                                                   interleaved=False)
            for p,x in progress(enumerate(points)):
                for c in cell_index.intersection([x[0],x[0],x[1],x[1]]):
                    if paths[c].contains_point(x):
                        cells[p]=c
                        break
        elif method=='point_index':
            xxyy=np.c_[points[:,0], points[:,0], points[:,1], points[:,1]]
            # building in one go is several times faster than incremental.
            self.log.info("Building point index (%d points)"%len(points))
            idx=gen_spatial_index.PointIndex(zip(range(len(points)),
                                                 xxyy,
                                                 [None]*len(points)),
                                             interleaved=False)
            self.log.info("Querying point index (%d cells)"%self.Ncells())
            for c in range(self.Ncells()):
                p=self.cell_path(c)
                verts=p.vertices
                box=[verts[:,0].min(), verts[:,0].max(),
                     verts[:,1].min(), verts[:,1].max()]
                for hit in idx.intersection(box):
                    if p.contains_point(points[hit]):
                        cells[hit]=c
        elif method=='cells_nearest':
            for i in progress(range(len(points))):
                # this step is the slowest.
                # inverting the loop to be over cells, and check all points against
                # a single cell at a time was equally slow with 500k points and 50k
                # cells.
                c=self.select_cells_nearest(points[i], inside=False)
                if c is None: continue
                cells[i]=c
        elif method=='kdtree':
            from scipy.spatial import cKDTree
            
            kdt=cKDTree(points)
            cc=self.cells_center()
            radii=mag( cc - self.nodes['x'][self.cells['nodes'][:,0]])
            # A little slop on radius in case cells are not perfectly
            # orthogonal.
            pnts=kdt.query_ball_point(cc,1.05*radii)

            for c,candidates in enumerate(pnts):
                p=self.cell_path(c)
                hits=[cand for cand in candidates 
                      if p.contains_point(points[cand])]
                cells[hits]=c
        elif method=='mpl':
            tri,tsrcs=self.mpl_triangulation(return_sources=True)
            tf=tri.get_trifinder()
            tcells=tf(points[:,0],points[:,1])
            valid=tcells>=0
            cells[valid]=tsrcs[tcells[valid]]
        else:
            raise Exception("bad method %s"%method)
            
        return cells

    def select_cells_by_cut(self,line,start=0,side='left',delta=1.0):
        """
        Split the cells by a linestring.  By default, returns
        a bitmask set to True for cells falling on the "left"
        side of the linestring.

        line: shapely LineString object

        Uses basic graph traversal, and will be faster when start
        is set to a cell index on the smaller side of the linestring.

        delta: size of finite difference used in determining the orientation
        of the cut.
        """
        marks=np.zeros(self.Ncells(),np.bool_)

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

    def select_nodes_nearest(self,xy,count=None,max_dist=None):
        """ count is None: return a scalar, the closest.
        otherwise, even if count==1, return a list
        """
        xy_orig=xy
        xy=np.asarray(xy)
        assert xy.shape==(2,),"select nodes nearest was given '%s'"%str(xy_orig)
        
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

        if max_dist is not None:
            hits=[ hit for hit in hits
                   if mag(xy - self.nodes['x'][hit])<=max_dist ]
            
        if count is None:
            if len(hits):
                return hits[0]
            else:
                return None
        else:
            return hits

    def shortest_path(self,n1,n2,return_type='nodes',
                      edge_weight=None,max_return=None,
                      edge_selector=lambda j,direc: True,
                      directed=False,
                      traverse='nodes'):
        """ dijkstra on the edge graph from n1 to n2

        return_type:
          'nodes': list of node indexes
          'edges' or 'sides': array of edge indices
          'cost': just the total cost
        selector: given an edge index and direction, return True if the edge should be considered.
        edge_weight: None: use euclidean distance, otherwise function taking an
          edge index and returning its weight.

        n2 is typically a single node index, but can also be a collection.
        in that case, return values will be a list of (n,value) tuples, in order
        of smallest to largest cost.

        directed is no longer necessary, and edge_weight and edge_selector now always take
        direction (+1 for 'forward' on an edge, -1 for 'backward') as a second argument.  
        For nondirected graphs simply ignore the 
        direction.

        If max_return is not None, it sets the maximum number of nodes in n2 to return.
        If n2 is a single scalar index, then the return value is just the requested value

        traverse:
          'nodes' path is nodes connected by edges.  
          'cells' path is cells connected by edges
            this changes the interpretation of n1,n2 and return_type, such that these
            are all cell indexes instead of node indexes.
        """
        if pq is None:
            raise GridException("shortest_path requires the  priority_queue module.")
        queue = pq.priorityDictionary()
        queue[n1] = [0,None] # distance, predecessor
        done = {} # likewise

        try:
            dests=set(n2) # exception if n2 is a scalar index
            return_scalar=False
        except TypeError:
            dests=set([n2])
            return_scalar=True

        if max_return is None:
            max_return=len(dests)

        results=[] # list of dicts, return values per target node in dests

        if traverse=='cells':
            e2c=self.edge_to_cells()
            # safer than using circumcenters
            x=self.cells_centroid()
        elif traverse=='nodes':
            x=self.nodes['x']
            
        while 1:
            # find the queue-member with the lowest cost:
            if len(queue)==0:
                break
            best = queue.smallest()
            best_cost,pred = queue[best]
            del queue[best]

            done[best] = [best_cost,pred]

            if best in dests:
                # use the key 'n', but if traverse=='cells', that is actually
                # a cell index
                results.append( dict(n=best,cost=best_cost) )
                if len(results)>=max_return:
                    break # done!

            # figure out its neighbors
            nbrs=[] # tuples of nbr node, edge, +1/-1 direction

            if traverse=='nodes':
                for j in self.node_to_edges(n=best):
                    ne1,ne2=self.edges['nodes'][j]
                    if ne1==best and edge_selector(j,1):
                        nbrs.append( (ne2,j,1) )
                    elif edge_selector(j,-1):
                        nbrs.append( (ne1,j,-1) )
            elif traverse=='cells':
                for j in self.cell_to_edges(best):
                    c1,c2=e2c[j]
                    if c1==best and c2>=0 and edge_selector(j,1):
                        nbrs.append( (c2,j,1) )
                    elif c2==best and c1>=0 and edge_selector(j,-1):
                        nbrs.append( (c1,j,-1) )
                        
            for nbr,j,direc in nbrs:
                if nbr in done:
                    continue

                if edge_weight is None:
                    dist = mag( x[nbr] - x[best] )
                else:
                    dist = edge_weight(j,direc)
                if not np.isfinite(dist):
                    continue # second way of ignoring edges

                new_cost = best_cost + dist

                if nbr not in queue:
                    queue[nbr] = [np.inf,None]

                if queue[nbr][0] > new_cost:
                    queue[nbr] = [new_cost,best]

        # update/replace the return values in results based on return_type
        return_values=[] #  (node/cell, <return value>)
        for i,result in enumerate(results):
            if return_type in ['nodes','edges','sides','cells']:
                # reconstruct the path:
                path = [result['n']] # will be a cell if traverse=='cells'
                while 1:
                    pred=done[path[-1]][1]
                    if pred is None:
                        break
                    else:
                        path.append(pred)
                path=np.array(path[::-1]) # reverse it so it goes from n1 to n2
                result['path']=path

            if return_type in ['nodes','cells']:
                return_value=path
            elif return_type in ('edges','sides'):
                if traverse=='nodes':
                    return_value=np.array( [self.nodes_to_edge(path[i],path[i+1])
                                            for i in range(len(path)-1)] )
                elif traverse=='cells':
                    return_value=np.array( [self.cells_to_edge(path[i],path[i+1])
                                            for i in range(len(path)-1)] )
                else:
                    raise Exception("Bad value for traverse: %s"%traverse)
            elif return_type=='cost':
                return_value=done[result['n']][0]
            return_values.append( (result['n'],return_value) )
        if return_scalar:
            return return_values[0][1]
        else:
            return return_values

    def select_cells_along_ray(self,x0,vec):
        """
        From the point x0, march in the direction given by vec
        and report all cells encountered along the way.
        if x0 falls outside the grid return None
        """
        edge_norm=self.edges_normals(cache='norm')
        
        p_dtype=[ ('x',np.float64,2),
                 ('j_last',np.int32),
                 ('c',np.int32)]
        p=np.zeros((),p_dtype)
        p['x']=x0
        p['j_last']=-1 # particle state
        c=self.select_cells_nearest(p['x'],inside=True)
        if c is None:
            return None # starting point is not in grid
        p['c']=c
        path=[p.copy()]

        while True:
            # move point to the boundary of the next cell in
            # the vec direction

            # code taken from pypart/basic.py
            dt=np.inf # 
            j_cross=None
            j_cross_normal=None

            for j in self.cell_to_edges(p['c']):
                if j==p['j_last']:
                    continue # don't cross back
                # get an outward normal:
                normal=edge_norm[j]
                if self.edges['cells'][j,1]==p['c']: # ~checked
                    normal=-normal
                # vector from xy to a point on the edge
                d_xy_n = self.nodes['x'][self.edges['nodes'][j,0]] - p['x']
                # perpendicular distance
                dp_xy_n=d_xy_n[0] * normal[0] + d_xy_n[1]*normal[1]
                assert dp_xy_n>=0 #otherwise sgn probably wrong above

                closing=vec[0]*normal[0] + vec[1]*normal[1]

                #if closing<0: 
                #    continue
                #else:
                dt_j=dp_xy_n/closing
                if dt_j>0 and dt_j<dt:
                    j_cross=j
                    dt=dt_j
                    j_cross_normal=normal

            # Take the step
            delta=vec*dt
            # see if we're stuck
            if mag(delta) / (mag(delta) + mag(p['x'])) < 1e-14:
                print("Steps are too small")
                break

            p['x'] += delta
            p['j_last'] = j_cross

            assert j_cross is not None

            # print "Cross edge"
            e_cells=self.edges['cells'][j_cross]
            if e_cells[0]==p['c']:
                new_c=e_cells[1]
            elif e_cells[1]==p['c']:
                new_c=e_cells[0]
            else:
                assert False

            if new_c<0:
                path.append(p.copy()) # will repeat the cell
                break # done - hit the boundary
            p['c']=new_c
            path.append(p.copy())
        return np.array(path)
        
    def distance_transform_nodes(self,nodes,max_distance=None):
        """
        Akin to image processing distance transform, centered on nodes
        and connceted by edges.
        nodes: a sequence of node indices or a bitmask
        max_distance: stop seaching when distance reaches this threshold.
          This is not currently an efficient implementation, and you will almost
          certainly want to supply max_distance.
        """

        if np.issubdtype(nodes.dtype,np.bool_):
            nodes=np.nonzero(nodes)[0]

        dists=np.zeros(self.Nnodes(),np.float64)
        dists[:]=np.inf
        dists[nodes]=0.0
        #L=15.0

        queue=list(nodes)

        while len(queue):
            n=queue.pop(0)
            nval=dists[n]
            for nbr in self.node_to_nodes(n):
                if dists[nbr]<=nval:
                    continue
                d=mag(self.nodes['x'][n] - self.nodes['x'][nbr])
                if dists[nbr]>nval+d:
                    dists[nbr]=nval+d
                    if (nbr not in queue) and (dists[nbr]<max_distance):
                        queue.append(nbr)
        return dists

        
    def remove_disconnected_components(self,renumber=True):
        """
        Clean up disconnected portions of the grid.  Disconnected
        portions are chosen based on area -- the largest contiguous
        area is kept, and all others removed.

        renumber: if False, skip renumbering afterwards
        """
        node_starts=[]
        boundary_polys=self.boundary_polygon_by_edges(allow_multiple=True)
        areas=[p.area for p in boundary_polys]
        to_keep=np.argmax(areas)
        for i,p in enumerate(boundary_polys):
            if i==to_keep: continue
            pnt=np.array(p.exterior)[0] # grab 1st point
            node_starts.append(self.select_nodes_nearest(pnt))

        stack=node_starts
        while stack:
            n=stack.pop(0)
            if self.nodes['deleted'][n]: continue # already visited
            nbrs=self.node_to_nodes(n)
            self.delete_node_cascade(n) # deleted cells/edges, too
            stack.extend(nbrs)

        if renumber:
            self.renumber()

    def create_dual(self,center='centroid',create_cells=False,
                    remove_disconnected=False,remove_1d=True,
                    extend_to_boundary=False):
        """
        Return a new grid which is the dual of this grid. This
        is robust for triangle->'hex' grids, and provides reasonable
        results for other grids.

        remove_1d: avoid creating edges in the dual which have no
          cell.  This happens when an input cell has all of its nodes
          on the boundary.

        extend_to_boundary: in contrast to remove_1d, this adds 
          edges to these 1d features to make them 2d, more or less
          taking the implied rays and intersecting them with the
          boundary of the original grid.
        """
        if remove_disconnected and not create_cells:
            # anecdotal, but remove_disconnected calls boundary_linestrings,
            # which in turn needs cells.
            raise Exception("Creating the dual without cells is not compatible with removing disconnected")
        gd=UnstructuredGrid()

        if center=='centroid':
            cc=self.cells_centroid()
        else:
            cc=self.cells_center()

        gd.add_node_field('dual_cell',np.zeros(0,np.int32))
        if extend_to_boundary: # expand_boundary:
            gd.add_node_field('dual_edge',np.zeros(0,np.int32))
        gd.add_edge_field('dual_edge',np.zeros(0,np.int32))

        # precalculate mask of boundary nodes
        if remove_1d or extend_to_boundary:
            e2c=self.edge_to_cells()
            boundary_edge_mask=e2c.min(axis=1)<0
            boundary_nodes=np.unique(self.edges['nodes'][boundary_edge_mask])
            boundary_node_mask=np.zeros(self.Nnodes(),np.bool_)
            boundary_node_mask[boundary_nodes]=True
            # below, if a cell's nodes are all True in boundary_node_mask,
            # it will be skipped

        cell_to_dual_node={}
        for c in self.valid_cell_iter():
            # dual_cell is redundant if remove_1d is False.
            if remove_1d:
                nodes=self.cell_to_nodes(c)
                if np.all(boundary_node_mask[nodes]):
                    continue
            node_idx=gd.add_node(x=cc[c],dual_cell=c)
            cell_to_dual_node[c]=node_idx

        e2c=self.edge_to_cells()

        if extend_to_boundary:
            boundary_edge_to_dual_node=-np.ones(self.Nedges(),np.int64)
            edge_center=self.edges_center()
        
        for j in self.valid_edge_iter():
            if e2c[j].min() < 0:
                if extend_to_boundary:
                    # Boundary edges *also* get nodes at their midpoints
                    boundary_edge_to_dual_node[j] = dnj = gd.add_node(x=edge_center[j],
                                                                      dual_edge=j)
                    # And induce a dual edge from the neighboring cell's dual
                    # node to this edge's midpoint
                    dnc=cell_to_dual_node[e2c[j,:].max()]
                    dj=gd.add_edge(nodes=[dnj,dnc],dual_edge=j)
                else:
                    continue # boundary
            elif remove_1d and np.all(boundary_node_mask[self.edges['nodes'][j]]):
                continue # would create a 1D link
            else:
                # Regular interior edge
                dn1=cell_to_dual_node[e2c[j,0]]
                dn2=cell_to_dual_node[e2c[j,1]]

                dj_exist=gd.nodes_to_edge([dn1,dn2]) 
                if dj_exist is None:
                    dj=gd.add_edge(nodes=[dn1,dn2],dual_edge=j)

        if extend_to_boundary:
            # Nodes also imply an edge in the dual -- and maybe even two
            # edges if we want this edge to go through the node
            for n in np.nonzero(boundary_node_mask)[0]:
                jbdry=[j for j in self.node_to_edges(n) if boundary_edge_mask[j]]
                # jbdry could be a multiple of 2 if there are multiple boundaries
                # tangent to each other at n. Not going to handle that right now.
                assert len(jbdry)==2,"Not ready for coincident boundaries"
                dnodes=boundary_edge_to_dual_node[jbdry]
                assert np.all(dnodes>=0)
                gd.add_edge(nodes=dnodes)

        if create_cells:
            # to create cells in the dual -- these map to interior nodes
            # of self.
            max_degree=10 # could calculate this.
            e2c=self.edge_to_cells()
            gd.modify_max_sides(max_degree)

            gd.add_cell_field('source_node',np.zeros(gd.Ncells(),np.int32)-1,
                              on_exists='overwrite')

            for n in self.valid_node_iter():
                if n%1000==0:
                    print("%d/%d dual cells"%(n,self.Nnodes())) # not exact
                js=self.node_to_edges(n)
                if np.any(e2c[js,:]<0): # boundary
                    continue
                tri_cells=self.node_to_cells(n) # i.e. same as dual nodes
                dual_nodes=np.array([cell_to_dual_node[c] for c in tri_cells])

                # but those have to be sorted.  sort the tri cell centers, same
                # as dual nodes, relative to tri node
                diffs=gd.nodes['x'][dual_nodes] - self.nodes['x'][n]
                angles=np.arctan2(diffs[:,1],diffs[:,0])
                dual_nodes=dual_nodes[np.argsort(angles)]
                gd.add_cell(nodes=dual_nodes,source_node=n)

            # flip edges to keep invariant that external cells are always
            # second.
            e2c=gd.edge_to_cells()
            to_flip=e2c[:,0]<0
            for fld in ['nodes','cells']:
                gd.edges[fld][to_flip] = gd.edges[fld][to_flip][:,::-1]

            # the original node locations are a more accurate orthogonal
            # center than what we can calculate currently.
            gd.cells['_center']=self.nodes['x'][gd.cells['source_node']]

        if remove_disconnected:
            gd.remove_disconnected_components()

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
        self.edge_to_cells()
        edge_mask=edge_mask & np.all( self.edges['cells']>=0,axis=1)

        cell_pairs = self.edges['cells'][edge_mask]

        # use scipy graph algorithms to find the connections
        from scipy import sparse
        from scipy.sparse import csgraph

        graph=sparse.csr_matrix( (np.ones(len(cell_pairs)),
                                  (cell_pairs[:,0],cell_pairs[:,1])),
                                 shape=(self.Ncells(),self.Ncells()) )

        n_comps,labels=csgraph.connected_components(graph,directed=False)

        if cell_mask is None:
            cell_mask=np.ones(self.Ncells(), np.bool_)

        labels[~cell_mask]=-1 # mark dry cells as -1
        unique_labels=np.unique( labels[cell_mask] )

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

    def select_cells_nearest(self,xy,count=None,inside=False,method='auto',
                             subedges=None):
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
          'try' => first try to find a cell containing xy, but if that fails, return the
           'closest' as if inside=False
        method: 
          'cc_index': find candidate cells from cell center index, then check candidates
          'node_index': get candidate cells from nearby nodes
          'auto': heuristic.  If inside is True, try cell centers, fallback to nodes. If
           inside is False, just use cell centers and hope.
        subedges: partial support for sub-edge geometry. Specifies a field on edges with
          coordinates. Only relevant for inside test.
        """
        xy=np.asarray(xy)
        real_count=count
        if count is None:
            real_count=1

        if method=='auto':
            if inside:
                c=self.select_cells_nearest(xy,inside=True,method='cc_index')
                if c is None:
                    c=self.select_cells_nearest(xy,inside=True,method='node_index')
                return c
            else:
                method='cc_index'
            
        if inside:
            # Now use count t
            # assert real_count==1
            # give ourselves a better shot at finding the right cell
            # figure that 10 is greater than the degree of
            # any nodes
            if count is None:
                real_count=10
            count=1

        if method=='cc_index':
            hits = self.cell_center_index().nearest(xy[self.xxyy],real_count)
            if isinstance( hits, types.GeneratorType): # usual for recent versions
                results=[]
                for hit in hits:
                    results.append(hit)
                    if len(results)==real_count:
                        break
                hits=results
        elif method=='node_index':
            cells=set()
            for n in self.select_nodes_nearest(xy,real_count):
                cells.update(self.node_to_cells(n))
            hits=list(cells)

        # this is only necessary with spatial indexes that don't deal with
        # deletion.  bandaid.
        Nc=self.Ncells()
        hits=[hit for hit in hits
              if (hit<Nc) and not self.cells['deleted'][hit]]
                
        if inside: # check whether the point is truly inside the cell
            # -- using shapely to determine contains, may be slower than matplotlib
            #  pnt=geometry.Point(xy[0],xy[1])
            #  for hit in hits:
            #      if self.cell_polygon(hit).contains(pnt):
            # -- using matplotlib to determine contains
            for hit in hits:
                if self.cell_path(hit,subedges=subedges).contains_point(xy):
                    return hit
            if inside=='try':
                return hits[0]
            else:
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

    def circumcenter_errors(self,radius_normalized=False,cells=None,
                            cc=None):
        """
        cc: if provided, an up-to-date circumcenter array for the whole
        grid.
        """
        if cells is None:
            cells=slice(None)

        if cc is None:
            centers = self.cells_center()[cells]
        else:
            centers=cc[cells]
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
    
    def edge_clearance(self,edges=None,mode='min',cc=None,Ac=None,
                       eps_Ac=1e-5,recalc_e2c=False):
        """
        For each edge, calculate signed distance between
        adjacent circumcenters and the edge center.
        Normalize by sqrt(cell area).
        By default, does *not* recalculate edge_to_cells.

        mode: 'min': return the smallest of the two clearances.
        'double': return half of the signed distance between the two cell
        centers. (half so that the result is same scale as 'min').

        cc,Ac: in performance-critical loops, pass in cc (circumcenters)
        and/or Ac (cell areas). Both should be for the full grid, regardless
        of edges selected.

        eps_Ac: lower bound for areas, to protect against negative areas.
        """
        ec=self.edges_center(edges=edges)

        if edges is not None:
            en=self.edges_normals(edges)
            e_sel=edges
        else:
            en=self.edges_normals()
            e_sel=slice(None)

        if recalc_e2c:
            if edges is not None:
                e2c=self.edge_to_cells(edges)
            else:
                e2c=self.edge_to_cells()
        else:
            e2c=self.edges['cells'][e_sel]

        valid=(e2c[:,:]>=0)
        if cc is None:
            if edges is not None:
                # not really much of a speedup.
                cells=np.unique(e2c[e2c>=0])
                cc=self.cells_center(refresh=cells) # still returns alls
            else:
                cc=self.cells_center()

        cc0=cc[e2c[:,0]]
        cc1=cc[e2c[:,1]]

        d0=((ec-cc0)*en).sum(axis=1)
        d1=((cc1-ec)*en).sum(axis=1)
        d=np.c_[d0,d1]

        if Ac is None:
            Ac=self.cells_area()
        L=np.sqrt(Ac[e2c[:,:]].clip(eps_Ac))

        nd=d/L
        nd[~valid]=np.nan
        nd_double=np.sum(nd,axis=1)

        orig=np.seterr()
        np.seterr(invalid='ignore')
        edge_quality_single=np.nanmin(nd,axis=1)
        np.seterr(**orig)
        
        
        edge_quality_double=nd_double/2
        if (mode=='min') or (mode=='single'):
            edge_quality=edge_quality_single
        elif mode=='double':
            edge_quality=edge_quality_double
        else:
            assert False,"what do you want, friend?"

        return edge_quality
    
    @staticmethod
    def read_pickle(fn):
        with open(fn,'rb') as fp:
            g=pickle.load(fp)
        # Might be an older grid
        g.update_element_defaults()
        return g
        
    @staticmethod
    def from_pickle(fn):
        return UnstructuredGrid.read_pickle(fn)
    
    def write_pickle(self,fn,overwrite=False):
        if os.path.exists(fn) and not overwrite:
            raise Exception("File %s exists, and overwrite is False"%fn)
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

        if 'fp' in d:
            # some readers hang on to a file point and that won't pickle.
            d['fp']=None

        return d
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.init_log()

        logging.debug( "May need to rewire any internal listeners" )

    def init_log(self):
        self.log = logging.getLogger(self.__class__.__name__)

    def write_suntans(self,path,overwrite=False):
        def check(fn):
            if not overwrite:
                assert not os.path.exists(fn),"File %s already exists"%fn
        fn_pnts=os.path.join(path,'points.dat')
        check(fn_pnts)
        with open(fn_pnts,'wt') as fp:
            print( "Writing SUNTANS: %d nodes"%self.Nnodes() )
            for n in range(self.Nnodes()):
                fp.write("%.5f %.5f 0\n"%(self.nodes['x'][n,0],
                                          self.nodes['x'][n,1]))

        ptm_sides=self.edges_as_nodes_cells_mark()
        fn_edges=os.path.join(path,'edges.dat')
        check(fn_edges)
        with open(fn_edges,'wt') as fp:
            for j in range(self.Nedges()):
                fp.write("%d %d %d %d %d\n"%( ptm_sides[j,0],ptm_sides[j,1],
                                              ptm_sides[j,4],
                                              ptm_sides[j,2],ptm_sides[j,3]) )

        fn_cells=os.path.join(path,'cells.dat')
        check(fn_cells)
        with open(fn_cells,'wt') as fp:
            vc=self.cells_center()
            for i in range(self.Ncells()):
                fp.write("%.5f %.5f "%(vc[i,0],vc[i,1])) # voronoi centers
                nodes=self.cell_to_nodes(i)
                assert len(nodes)==3
                fp.write("%d %d %d "%(nodes[0],nodes[1],nodes[2]))
                nbrs=self.cell_to_cells(i,ordered=True)
                fp.write("%d %d %d\n"%(nbrs[0],nbrs[1],nbrs[2]))

    def write_suntans_hybrid(self,path='.',points='points.dat',edges='edges.dat',cells='cells.dat',
                             z_offset=0.0,overwrite=False):
        """
        Write text-based suntans format which can accomodate arbitrary numbers of sides.
        This can be read by Janet.
        z_offset is added to elevations, which are assumed to come in positive-up.
        It doesn't really matter, because depth is not even used here.
        """
        xy=self.nodes['x']
        z=np.zeros(len(xy)) + z_offset
        xyz=np.c_[ xy, z]

        def check(fn):
            if not overwrite:
                assert not os.path.exists(fn),"File %s already exists"%fn

        point_fn=os.path.join(path,points)
        check(point_fn)
        np.savetxt(point_fn,xyz,fmt="%.12e")

        edge_nodes=self.edges['nodes']

        e2c=self.edge_to_cells().copy()
        e2c[e2c<0]=-1

        if 'mark' in self.edges.dtype.names:
            marks=self.edges['mark']
        else:
            log.info("No mark field, setting mark=1 on boundary edges.")
            marks=np.zeros(len(e2c),np.int32)
            marks[e2c[:,1]<0]=1

        edge_fn=os.path.join(path,edges)
        check(edge_fn)

        edge_nnmcc=np.c_[edge_nodes,marks,e2c]
        np.savetxt(edge_fn,edge_nnmcc,fmt='%d')

        cell_fn=os.path.join(path,cells)
        check(cell_fn)
        cc=self.cells_center()

        with open(cell_fn,'wt') as fp:
            for c in range(self.Ncells()):
                nsides=self.cell_Nsides(c)
                txt=["%d %.12e %.12e"%(nsides,cc[c,0],cc[c,1])]

                for n in self.cells['nodes'][c,:nsides]:
                    txt.append(str(n))

                # Important to get the right order
                for nbr in self.cell_to_cells(c,ordered=True):
                    if nbr<0:
                        txt.append("-1")
                    else:
                        txt.append(str(nbr))
                txt.append("\n")
                fp.write(" ".join(txt))

    def write_untrim08(self,fn,overwrite=False):
        """ write this grid out in the untrim08 format.  Since untrim grids have
        some extra fields, this just delegates to the Untrim grid class, converting
        this grid to untrim then writing.
        """
        UnTRIM08Grid(grid=self).write_untrim08(fn,overwrite=overwrite)


    #-# Grid refinement
    # as ugly as it is having this in here, there's not a very clean way to
    # put this sort of code elsewhere, and still be able to call super()
    def global_refine(self,l_thresh=1.0):
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
        new_cells = np.zeros( (4*self.Ncells(),self.max_sides), np.int32) - 1

        for c in self.valid_cell_iter():
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
            elif len(cn)==4: # quad
                n_ctr = self.Nnodes() + self.Nedges() + c
                for i in range(4):
                    new_cells[4*c+i,:4] =  [ midpoints[(i-1)%4], cn[i], midpoints[i], n_ctr]
            else:
                raise Exception("global_refine() can only handle triangles and quads")

        # 3. generic construction of edges from cells
        # try to use the same subclass as the original grid, so we'll have the same
        # fields available.
        # OLD: gr = ug.UnstructuredGrid()
        gr = self.__class__(max_sides=self.max_sides)
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
        gr.collapse_short_edges(l_thresh=l_thresh)
        # and drop centers that didn't get used
        gr.delete_naked_nodes()
        # make it nice.
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

    def write_cells_shp(self,shpname,extra_fields=[],overwrite=False):
        """ extra_fields is a list of lists,
            with either 3 items:
              # this way is obsolete.
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
        result=wkb2shp.wkb2shp(shpname,input_wkbs=cell_geoms,fields=cell_data,
                               overwrite=overwrite)
        return result

    def write_shore_shp(self,shpname,geom_type='polygon',overwrite=False):
        poly=self.boundary_polygon()
        if geom_type=='polygon':
            geoms=[poly]
        elif geom_type=='linestring':
            geoms=list(poly.boundary.geoms)
        wkb2shp.wkb2shp(shpname,geoms,overwrite=overwrite)

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


    def write_edges_shp(self,shpname,extra_fields=[],overwrite=False):
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
            if edge_id % 5000 == 0:
                print("%0.3g%%"%(100.*edge_id/edges.shape[0]))

            nodes = vertices[edges[edge_id,:2]]
            g = geometry.LineString(nodes)
            edge_geoms[edge_id] = g
            edge_data[edge_id]['length'] = g.length
            edge_data[edge_id]['edge_id1'] = edge_id + 1
            edge_data[edge_id]['depth_mean'] = side_depths_mean[edge_id]

            for fname,ftype,ffunc in extra_fields:
                edge_data[edge_id][fname] = ffunc(edge_id)

        wkb2shp.wkb2shp(shpname,input_wkbs=edge_geoms,fields=edge_data,
                        overwrite=overwrite)

    def write_node_shp(self,*a,**kw):
        """
        see write_nodes_shp.  This function here because of inconsistent naming
        in the past.
        """
        return self.write_nodes_shp(*a,**kw)
    def write_nodes_shp(self,shpname,extra_fields=[],overwrite=False):
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
        node_data=recarray_del_fields(node_data,['x','deleted'])

        wkb2shp.wkb2shp(shpname,input_wkbs=node_geoms,fields=node_data,
                        overwrite=overwrite)

    def write_ptm_gridfile(self,fn,overwrite=False,subgrid=True):
        """ write this grid out in the ptm grid format.
        subgrid: append cell and edge subgrid data.  this is
        really just faked, with a single sub-element per item.

        2020-12-31: Force update of all cells['edges']. Not clear
         where edges are getting out of order in cells, but it
         happens and in PTM grd file that's bad.
        """
        vertex_hdr = " Vertex Data: vertex_number, x, y"
        poly_hdr = " Polygon Data: polygon_number, number_of_sides,center_x, center_y, center_depth, side_indices(number_of_sides), marker(0=internal,1=open boundary)"
        side_hdr = " Side Data: side_number, side_depth, node_indices(2), cell_indices(2), marker(0=internal,1=external,2=flow boundary,3=open boundary)"

        if not overwrite:
            assert not os.path.exists(fn),"File %s already exists"%fn

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
            try:
                cell_depths = self.cells['cell_depth']
            except ValueError:
                # 2022-05-16 RH: too lazy to chase down who decides the name
                # here. So handle either one.
                cell_depths = self.cells['depth']
            # OLD: Make sure cells['edges'] is set, but don't replace
            #   data if it is already there.
            # 2020-12-31: Some issues with grids having out of order
            #  cells['edges']. Force full update.
            self.update_cell_edges(select='all')
            cc=self.cells_center()
            for c in range(self.Ncells()):
                edges = self.cells['edges'][c,:]
                edges[edges<0] = -1
                edge_str = " ".join( ["%10d"%(s+1) for s in edges] )
                edge_str = edge_str+" %10d\n"%(self.cells['mark'][c])
                nsides = sum(edges>=0)
                # RH: cell_depths is positive:up, but ptm expects positive:down
                #     as far as I can tell.
                fp.write(cell_write_str1%(c+1,
                                          nsides,
                                          cc[c,0],
                                          cc[c,1],
                                          cell_depths[c]))
                fp.write(edge_str)

            # write side info
            fp.write(side_hdr+"\n")
            # likewise, ptm expects edge depths to be positive down.
            edge_depths = self.edges['edge_depth']
            edge_write_str = " %10d %16.7f %10d %10d %10d %10d %10d\n"
            for s in range(self.Nedges()):
                edges = self.edges['cells'][s,:]
                edges[edges<0] = -1
                nodes = self.edges['nodes'][s,:]
                nodes[nodes<0] = -1
                # RH 2019-01-14: flip sign on edge depth.
                # RH 2020-05-23: reverting.  trying to keep 'depth' to mean positive
                #  down, and 'z' to mean positive up.
                fp.write(edge_write_str%(s+1,
                                         edge_depths[s],
                                         nodes[0]+1,
                                         nodes[1]+1,
                                         edges[0]+1,
                                         edges[1]+1,
                                          self.edges['mark'][s]))
            if subgrid:
                Ac=self.cells_area()
                for c in range(self.Ncells()):
                    cell_n_sub=1
                    fp.write("%10d %10d\n"%(c+1,cell_n_sub))
                    fp.write("%15.4f\n"%Ac[c])
                    fp.write("%15.4f\n"%(-cell_depths[c]))
                Le=self.edges_length()
                for s in range(self.Nedges()):
                    edge_n_sub=1
                    fp.write("%10d %10d\n"%(s+1,edge_n_sub))
                    fp.write("%15.4f\n"%Le[s])
                    fp.write("%15.4f\n"%(-edge_depths[s]))

    def cell_depths(self):
        """
        DEPRECATED. 
        Moving away from having unstructured_grid try to infer how depth information
        should be handled.  Instead, users should strive to keep positive-up values in
        'z_*' fields, and positive-down values in 'depth_*'  fields, and where possible
        keep names of fields unique between cells, edges, and nodes to facilitate netcdf
        output.

        Return an array of cell-centered depths.  This *should* be
        a positive:up quantity.
        TODO: make naming consistent so that this is elevation, indicating
        the sign convention.

        Returns all zeros if no edge depth data is found
        """
        try:
            return self.cells['depth']
        except ValueError:
            pass

        try:
            return self._cell_depth
        except AttributeError:
            pass

        return np.zeros(len(self.cells),np.float64)

    def edge_depths(self):
        """
        Return an array of edge-centered depths.  This *should* be
        a positive:up quantity.

        Returns all nan if no edge depth data is found. (used to return
        all zero)
        """
        try:
            return self.edges['depth']
        except ValueError:
            pass

        try:
            return self._edge_depth
        except AttributeError:
            pass

        return np.nan*np.ones(len(self.edges),np.float64)

    #--# generation methods
    def add_rectilinear(self,p0,p1,nx,ny,reverse_cells=False):
        """
        add nodes,edges and cells defining a rectilinear grid.
        nx gives the number of nodes in the x-direction (nx-1 cells)

        reverse_cells: If True, the order of nodes in each cell will be reversed.
          With a typical call where p0<p1, the default behavior creates proper 
          cells with positive area / CCW nodes.  But if the grid is going to be 
          mapped through a reflection, this option allows reversing the nodes.

        returns a dict with nodes=> [nx,ny] array of node indices
           cells=>[nx-1,ny-1] array of cell indices.
           currently does not return edge indices
        """
        assert self.max_sides>=4

        node_ids=np.zeros( (nx,ny), int)-1
        xs=np.linspace(p0[0],p1[0],nx)
        ys=np.linspace(p0[1],p1[1],ny)

        # create the nodes
        if self.Nnodes()>0:
            for xi,x in enumerate(xs):
                for yi,y in enumerate(ys):
                    node_ids[xi,yi] = self.add_node(x=[x,y])
        else: # faster, but untested
            node_ids=np.arange(nx*ny).reshape( (nx,ny) )
            self.nodes=np.zeros( nx*ny, self.node_dtype)
            Y,X=np.meshgrid( ys,xs)
            self.nodes['x'][:,0]=X.ravel()
            self.nodes['x'][:,1]=Y.ravel()

        if self.Ncells()>0:
            cell_ids=np.zeros( (nx-1,ny-1), int)-1
            # slower, but probably plays nicer with existing topology
            for xi in range(nx-1):
                for yi in range(ny-1):
                    nodes=[ node_ids[xi,yi],
                            node_ids[xi+1,yi],
                            node_ids[xi+1,yi+1],
                            node_ids[xi,yi+1] ]
                    if reverse_cells:
                        nodes=nodes[::-1]
                    cell_ids[xi,yi]=self.add_cell_and_edges(nodes=nodes)
        else:
            # Blank canvas - just create all in one go
            cells=np.c_[ node_ids[:-1,:-1].ravel(),
                         node_ids[1:,:-1].ravel(),
                         node_ids[1:,1:].ravel(),
                         node_ids[:-1,1:].ravel() ]
            if reverse_cells:
                cells=cells[:,::-1]
            self.cells=np.zeros(len(cells),self.cell_dtype)
            self.cells[:]=self.cell_defaults
            self.cells['nodes'][:,:4]=cells
            if self.max_sides>4:
                self.cells['nodes'][:,4:]=-1
            self.make_edges_from_cells_fast()
            # May need to flip this:
            cell_ids=np.arange((nx-1)*(ny-1)).reshape( [nx-1,ny-1] )
            self.cells['_center'][:]=np.nan
            self.cells['_area'][:]=np.nan
            
        return {'cells':cell_ids,
                'nodes':node_ids}

    def add_rectilinear_on_line(self,centerline,profile,add_streamwise=True):
        """
        add nodes,edges and cells defining a quad grid.
        centerline: [N,2] centerline points, already at desired resolution.
        profile: function(x,s,perp) => [M] values for lateral spacing.
          x: [x,y] point on centerline
          s: nondimensional location on centerline, 0 to 1
          perp: unit normal perpendicular to left of centerline

        add_streamwise: d_lat,d_long will be set for nodes and cells, holding
         dimensional streamwise coordinates

        returns a dict with nodes=> [nx,ny] array of node indices
           cells=>[nx-1,ny-1] array of cell indices.
           currently does not return edge indices
        """
        assert self.max_sides>=4

        if add_streamwise:
            # so we can track cell's in streamwise coordinates
            zero=np.float64(0)
            self.add_cell_field('d_lat',zero,on_exists='pass')
            self.add_cell_field('d_long',zero,on_exists='pass')
            self.add_node_field('d_lat',zero,on_exists='pass')
            self.add_node_field('d_long',zero,on_exists='pass')

        nx=len(centerline)

        centerline_s=dist_along(centerline)

        center_deltas=centerline[2:]-centerline[:-2]
        center_deltas=np.concatenate( [ center_deltas[:1],
                                        center_deltas,
                                        center_deltas[-1:] ] )
        # left-pointing normals
        center_norms=np.c_[ -center_deltas[:,1], center_deltas[:,0] ]
        center_norms=to_unit(center_norms)

        prof0=profile(x=centerline[0],s=0,perp=center_norms[0])
        ny=len(prof0)
        node_ids=np.zeros( (nx,ny), int)-1
        
        # create the nodes
        for xi in range(nx):
            ctr=centerline[xi]
            s=centerline_s[xi]
            prof=profile(x=ctr,s=s,perp=center_norms[xi])

            for yi in range(ny):
                X=ctr+prof[yi]*center_norms[xi]
                kw={}
                if add_streamwise:
                    kw['d_lat']=prof[yi]
                    kw['d_long']=s
                node_ids[xi,yi] = self.add_node(x=X,**kw)

        cell_ids=np.zeros( (nx-1,ny-1), int)-1

        # create the cells
        for xi in range(nx-1):
            for yi in range(ny-1):
                nodes=[ node_ids[xi,yi],
                        node_ids[xi+1,yi],
                        node_ids[xi+1,yi+1],
                        node_ids[xi,yi+1] ]
                kw={}
                if add_streamwise:
                    kw['d_lat']=self.nodes['d_lat'][nodes].mean()
                    kw['d_long']=self.nodes['d_long'][nodes].mean()
                cell_ids[xi,yi]=self.add_cell_and_edges(nodes=nodes,**kw)

        return {'cells':cell_ids,
                'nodes':node_ids}

    def add_quad_ring(self,r_in,r_out=None,nrows=None,nspokes=None,
                      sides=4,stagger0=0,scale=None):
        """
        Add rings, with option to double resolution, add multiple rows,
        add a triangle-strip or quad strip.
        Interface subject to change.
        """
        if scale is None:
            assert nspokes is not None
            scale=r_in*2*np.pi/nspokes
        tol=0.01*scale
        nspokes=nspokes or int(2*np.pi*r_in/scale)
        theta0=stagger0*2*np.pi/nspokes

        def ring(n,r,stagger=0):
            theta=theta0 + stagger*2*np.pi/n + np.linspace( 0,2*np.pi,n+1 )[:-1]
            return r*np.c_[ np.cos(theta),np.sin(theta)]

        if nrows=='stitch':
            radii=np.linspace(r_in,r_out,2)
        elif nrows is not None:
            radii=np.linspace(r_in,r_out,nrows)
        else:
            radii=[r_in]

        # circumference
        rings=[]
        for ri,r in enumerate(radii):
            if nrows=='stitch' and ri==1:
                pnts=ring(2*nspokes,r)
            elif sides==3:
                pnts=ring(nspokes,r,stagger=0.5*(ri%2))
            else:
                pnts=ring(nspokes,r)
            rings.append(pnts)
        for pnts in rings:
            for a,b in circular_pairs(pnts):
                na=self.add_or_find_node(x=a,tolerance=tol)
                nb=self.add_or_find_node(x=b,tolerance=tol)
                try:
                    if na!=nb:
                        self.add_edge(nodes=[na,nb])
                except self.GridException:
                    pass
        for ra,rb in zip(rings[:-1],rings[1:]):
            if len(ra)==len(rb):
                if sides==4:
                    segss=[ zip(ra,rb)]
                else:
                    segss=[ zip(ra,rb),
                            zip(np.roll(ra,-1,axis=0),rb)]
            else: # stitch
                segss=[ zip(ra,rb[::2]),
                       zip(ra,rb[1::2]),
                       zip(np.roll(ra,-1,axis=0),rb[1::2]) ]
            for segs in segss:
                for pa,pb in segs:
                    na=self.add_or_find_node(x=pa,tolerance=tol)
                    nb=self.add_or_find_node(x=pb,tolerance=tol)
                    try:
                        if na!=nb:
                            self.add_edge(nodes=[na,nb])
                    except self.GridException:
                        pass

    # Half-edge interface
    def halfedge(self,j,orient):
        return HalfEdge(self,j,orient)
    def nodes_to_halfedge(self,n1,n2):
        return HalfEdge.from_nodes(self,n1,n2)
    def node_to_halfedges(self,n):
        return [self.nodes_to_halfedge(n,nbr) 
                for nbr in self.angle_sort_adjacent_nodes(n)]
    def cell_to_halfedge(self,c,i):
        j=self.cell_to_edges(c,ordered=True)[i]
        if self.edges['cells'][j,0]==c:
            return HalfEdge(self,j,0)
        else:
            return HalfEdge(self,j,1)
        
    def cell_to_halfedges(self,c):
        return [HalfEdge(self,j,1-int(self.edges['cells'][j,0]==c))
                for j in self.cell_to_edges(c,ordered=True) if j>=0]

    def cell_containing(self,xy,neighbors_to_test=4):
        """ Compatibility wrapper for select_cells_nearest.  This
        may disappear in the future, depending...
        """
        hit = self.select_cells_nearest(xy, count=neighbors_to_test, inside=True)
        if hit is None:
            return -1
        else:
            return hit

    def cell_path(self,i,subedges=None):
        """
        Return a matplotlib Path object representing the closed polygon of
        cell i
        """
        if subedges is None:
            cell_nodes = self.cell_to_nodes(i)
            return Path(self.nodes['x'][cell_nodes])
        else:
            return Path(self.cell_coords_subedges(i,subedges))

    
class UGrid(UnstructuredGrid):
    def __init__(self,nc=None,grid=None):
        """
        nc: Read from a ugrid netcdf file, or netcdf object
        grid: initialize from existing UnstructuredGrid (though not necessarily an untrim grid)
        *** currently hardwired to use 1st mesh found
        """
        logging.warning("UGrid will be deprecated.  Use UnstructuredGrid.from_ugrid")
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
    DEPTH_UNKNOWN = np.nan # used when no incoming depth is given, or if incoming depth is nan.

    angle = 0.0
    location = "''" # don't use a slash in here!

    def __init__(self,grd_fn=None,grid=None,extra_cell_fields=[],extra_edge_fields=[],
                 clean=False):
        """
        grd_fn: Read from an untrim .grd file
        grid: initialize from existing UnstructuredGrid (though not necessarily an untrim grid)
        clean: if initializing from another grid and this is True, fix up edge marks, order, and
        orientation to follow conventions.
          This had defaulted to True, but that can be surprising when trying to load both a grid
        and data. 
        """
        # NB: these depths are as soundings - positive down.
        super(UnTRIM08Grid,self).__init__( extra_cell_fields = extra_cell_fields + [('depth_mean',np.float64),
                                                                                    ('depth_max',np.float64),
                                                                                    ('red',np.bool_),
                                                                                    ('subgrid',object)],
                                           extra_edge_fields = extra_edge_fields + [('depth_mean',np.float64),
                                                                                    ('depth_max',np.float64),
                                                                                    ('subgrid',object)] )
        if grd_fn is not None:
            self.read_from_file(grd_fn)
        elif grid is not None:
            self.copy_from_grid(grid)
            if clean:
                self.edges['mark']=self.inferred_edge_marks()
                # Even if the incoming grid had been renumbered(), untrim
                # has a specific order
                self.renumber()

    def Nred(self):
        if 'red' in self.cells.dtype.names:
            # nothing magic - just reads the cell attributes
            return sum(self.cells['red'])
        else:
            # in case somebody deleted the red field.
            return 0

    def renumber_cells_ordering(self): # untrim version
        # not sure about placement of red cells, but presumably something like this:

        # so marked, red cells come first, then marked black cells (do these exist?)
        # then unmarked red, then unmarked black.
        # mergesort is stable, so if no reordering is necessary it will stay the same.
        if 'red' not in self.cells.dtype.names:
            return super().renumber_cells_ordering()

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

    def renumber_edges_ordering(self): # untrim version
        # want marks==0, marks==self.FLOW, marks==self.LAND
        mark_order = np.zeros(3,np.int32)
        mark_order[0] = 0 # internal comes first
        mark_order[self.FLOW] = 1 # flow comes second
        mark_order[self.LAND] = 2 # land comes last
        Nactive = sum(~self.edges['deleted'])
        return np.argsort(mark_order[self.edges['mark']]+10*self.edges['deleted'],
                          kind='mergesort')[:Nactive]

    def copy(self):
        # Deep copy
        # details of this interface have morphed over time.
        # In theory UnstructuredGrid.copy() do this, and there's no
        # need for a specific untrim version. But having
        # UnstructuredGrid.copy() call a subclass constructor gets into
        # issues when the interface is not standardized. That could be
        # dealt with by requiring subclasses to either support a constructor
        # that allows copying, or to reimplement copy(). Rather than thinking
        # the big thoughts, I'm just overriding copy().
        g=UnTRIM08Grid()
        
        g.cell_dtype=self.cell_dtype
        g.edge_dtype=self.edge_dtype
        g.node_dtype=self.node_dtype

        g.cells=self.cells.copy()
        g.edges=self.edges.copy()
        g.nodes=self.nodes.copy()

        g.cell_defaults=self.cell_defaults.copy()
        g.edge_defaults=self.edge_defaults.copy()
        g.node_defaults=self.node_defaults.copy()

        # Subgrid is stored as references to ragged objects, which
        # need to be copied explicitly
        g.cells['subgrid'] = copy.deepcopy(self.cells['subgrid'])
        g.edges['subgrid'] = copy.deepcopy(self.edges['subgrid'])

        g.refresh_metadata()
        return g
    
    def copy_from_grid(self,grid):
        super(UnTRIM08Grid,self).copy_from_grid(grid)

        # now fill in untrim specific things:
        if isinstance(grid,UnTRIM08Grid):
            for field in ['depth_mean','depth_max','red']:
                if field in self.cells.dtype.names:
                    self.cells[field] = grid.cells[field]
            for field in ['depth_mean','depth_max']:
                if field in self.edges.dtype.names:
                    self.edges[field] = grid.edges[field]

            # Subgrid is separate
            if 'subgrid' in self.cells.dtype.names:
                self.cells['subgrid'] = copy.deepcopy(grid.cells['subgrid'])
            if 'subgrid' in self.edges.dtype.names:
                self.edges['subgrid'] = copy.deepcopy(grid.edges['subgrid'])
            # ideally should be smarter -- if those fields are missing, we should
            # probably revert to non-Untrim specific code below
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

    def infer_depths_cells_from_edges(self,valid=None):
        """ cell depths are set as max of neighboring edge depth.
        sets cells['depth_mean'] and cells['depth_max']
        both to the max depth of neighboring edge['depth_mean'].

        valid: optional bitmask to consider only a subset of edges
        """
        sel_cells = np.nonzero(np.isnan(self.cells['depth_mean']))[0]
        # iterate, since number of sides varies
        edges = self.cells['edges'][sel_cells]

        edge_depths=self.edges['depth_mean']
        if valid is not None:
            valid_depths=-np.inf*np.ones(self.Nedges())
            valid_depths[valid]=edge_depths[valid]
            edge_depths=valid_depths
        edge_depths=edge_depths[edges]
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
            if isinstance(selector,str) and selector=='missing':
                overwrite=False
                selector = range(len(elements))
            else:
                overwrite=True
                selector = np.asarray(selector)
                if selector.dtype == np.bool_:
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
        fp = open(self.grd_fn,'rt')
        hdr = fp.readline().strip() #header &GRD_2008 or &LISTGRD

        if hdr == self.hdr_08:
            print( "Will read 2008 format for grid" )
            n_parms = 11
        elif hdr == self.hdr_old:
            print( "Will read old UnTRIM grid format" )
            n_parms = 10
        else:
            raise Exception("hdr '%s' not recognized"%hdr)

        for i in range(n_parms):  # ignore TNE and TNS in new format files
            l = fp.readline()
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
            s = fp.readline().strip() # header:  /
            if s == '/':
                break

        # We know the size of everything, and can ask UnstructuredGrid to allocate
        # arrays now, with the 'special' meaning that passing an integer means allocate
        # the array of that size, full of zeros.
        # this allocates
        #  self.nodes, self.edges, self.cells
        self.from_simple_data(points = Nvertices,edges = Nsides, cells = Npolys)

        for v in range(Nvertices):
            Cv = fp.readline().split()
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
            l = fp.readline()
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
                # any reason not to just read the nodes directly here?
                # used to rely on copying the sides back to cells['nodes'],
                # but try just reading them directly:
                self.cells['nodes'][c,0] = int(Cp[3]) - 1
                self.cells['nodes'][c,1] = int(Cp[5]) - 1
                self.cells['nodes'][c,2] = int(Cp[7]) - 1

                if numsides == 4:
                    self.cells['edges'][c,3] = int(Cp[10]) - 1
                    self.cells['nodes'][c,3] = int(Cp[9]) - 1
                else:
                    self.cells['edges'][c,3]=self.UNDEFINED
                    self.cells['nodes'][c,3]=self.UNDEFINED

            else:
                for ei in range(numsides):
                    self.cells['nodes'][c,ei] = int(Cp[3+ei]) - 1
                    self.cells['edges'][c,ei] = int(Cp[3+numsides+ei]) - 1
                self.cells['nodes'][c,numsides:]=self.UNDEFINED
                self.cells['edges'][c,numsides:]=self.UNDEFINED

        # choose some large, above-sea-level depth
        self.cells['depth_mean'] = -1000 # not sure this is doing anything...

        for e in range(Nsides):
            Cs = fp.readline().split()
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
            # self.make_cell_nodes_from_edge_nodes()
            # Not sure why I thought that.  When reading a 2004 file, the nodes
            # were there, although there was a discrepancy between
            # edges['nodes'][ cells['edges'] ]
            # and cells['nodes']
            pass

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
                    for item in fp.readline().split():
                        yield item
            token_gen=tokenizer()
            # py2/py3 compatibility
            def itok(): return int(six.next(token_gen))
            def ftok():
                # Some Janet files come back with ?, presumably for missing depth
                # data
                s=six.next(token_gen).strip()
                if s=='?':
                    return np.nan
                else:
                    return float(s)

            for c in range(Npolys):
                check_c=itok()
                nis    =itok()

                if check_c != c+1:
                    print("ERROR: while reading cell subgrid, cell index mismatch: %s vs. %d"%(c+1,check_c))

                areas = np.array( [ftok() for sg in range(nis)] )
                depths = np.array( [ftok() for sg in range(nis)] )

                self.cells['depth_mean'][c] = np.sum(areas*depths) / np.sum(areas)
                self.cells['_area'][c] = np.sum(areas)
                self.cells['depth_max'][c] = depths.max()
                self.cells['subgrid'][c] = (areas,depths)
            for e in range(Nflow_sides):
                check_e=itok()
                nis=itok()

                if check_e != e+1:
                    print( "ERROR: While reading edge subgrid, edge index mismatch: %s vs. %s"%(e+1,check_e) )

                lengths = np.array( [ftok() for sg in range(nis)] )
                depths =  np.array( [ftok() for sg in range(nis)] )
                if sum(lengths)<=0:
                    if len(lengths)>1:
                        print( "edge %d has bad lengths"%e )
                    else:
                        # sometimes an edge with no subgrid just has a depth,
                        # no lengths, and it's not really necessary
                        pass
                    self.edges['depth_mean'][e] = np.mean(depths)
                else:
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
        # Note that this should not count land cells!
        # just internal and flow edges
        return sum( [len(sg[0])
                     for sg,mark in zip(self.edges['subgrid'],self.edges['mark'])
                     if sg!=0 and mark!=self.LAND] )

    def inferred_edge_marks(self):
        """
        Generate an edge marks array that makes sure any mark=0 edges
        get labelled as land. This does not alter the grid -- just returns
        a mark array. The returned array may be the existing marks if no
        changes are required, so don't modify the array unless you don't
        care about edges['marks'].
        """
        # Force recalc, as otherwise we'll write everything out as LAND.
        e2c=self.edge_to_cells(recalc=True)
        boundary=e2c.min(axis=1)<0
        marks=self.edges['mark']
        sel=(marks==0) & boundary
        if np.any(sel):
            return np.where(sel,self.LAND,marks)
        else:
            # Already consistent
            return marks
    
    def write_untrim08(self,fn,overwrite=False):
        """ write this grid out in the untrim08 format.
        Note that for some fields (red/black, subgrid depth), if this
        grid doesn't have that field, this code will fabricate the data
        and probably not very well.
        """
        if not overwrite:
            assert not os.path.exists(fn),"Output file %s already exists"%fn

        with open(fn,'wt') as fp:
            fp.write(self.hdr_08+"\n")

            n_parms = 11

            # Commonly marks have not been set, but we at least know where
            # boundaries are.
            edge_marks=self.inferred_edge_marks()
            
            Nland = sum(edge_marks==self.LAND)
            Nflow = sum(edge_marks==self.FLOW)
            Ninternal = sum(edge_marks==0)
            Nbc = sum(self.cells['mark'] == self.BOUNDARY)

            # 2018-08-10 RH: reorder this to match how things come
            # out of Janet, in hopes of making this file readable
            # by Janet.
            # 2021-12-01 RH: reorder again?
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

            e2c=self.edges['cells']
            # During grid generation, may have some -2 or other values here.
            # Make them all 0 in hopes of keeping Janet happy
            e2c1=np.where(e2c>=0,1+e2c,0)
            for e in range(self.Nedges()):
                fp.write("%10d %14d %14d %14d %14d\n"%(e+1,
                                                       self.edges['nodes'][e,0]+1,self.edges['nodes'][e,1]+1,
                                                       e2c1[e,0],e2c1[e,1]))

            # since we have to do this 4 times, make a helper function
            def fmt_wrap_lines(fp,values,fmt="%14.4f ",per_line=10):
                """ write values out to file fp with the given string format, but break
                the lines so no more than per_line values on a line
                ends with a newline
                """
                for i,a in enumerate(values):
                    if i>0 and i%10==0:
                        fp.write("\n")

                    if np.isnan(a): a=self.DEPTH_UNKNOWN
                    
                    if np.isfinite(a):
                        fp.write("%14.4f "%a)
                    else:
                        # Janet maybe prefers '?' over 'nan'
                        fp.write("             ? ")
                fp.write("\n")

            # subgrid bathy
            cA=self.cells_area()
            for c in range(self.Ncells()):
                try:
                    areas,depths = self.cells['subgrid'][c]
                except TypeError:
                    # GIS editing might leave some cells with no subgrid
                    areas=[cA[c]]
                    depths=[0.0]
                    
                nis = len(areas)

                fp.write("%14d %14d\n"%(c+1,nis))
                fmt_wrap_lines(fp,areas)
                fmt_wrap_lines(fp,depths)

            edge_lengths = self.edges_length()

            for e in range(Ninternal+Nflow):
                try:
                    lengths,depths = self.edges['subgrid'][e]
                    nis = len(lengths)
                except TypeError:
                    # GIS editing might leave some edges with no subgrid
                    nis=0
                    
                if nis==0: # causes issues to have nothing here...
                    nis=1
                    lengths=[edge_lengths[e]]
                    depths=[self.DEPTH_UNKNOWN]

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
        logging.warning("SuntansGrid will be deprecated.  Use UnstructuredGrid.read_suntans")
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
        self.edges['cells']=edges[:,3:5]
        # self.edges['index']=edges[:,5] # not sure the purpose of the last column
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
        # 2021-01-06: Had trouble on windows in python 2 using
        # text mode. Seems that fp.tell() and fp.seek() are not
        # consistent. In binary mode, we'll get /r/n line endings
        # if the file turns out to have been written on windows,
        # but readline() and strip() will handle this okay.
        self.fp = open(self.grd_fn,'rb')

        while True:
            line = self.fp.readline()
            if line == b'':
                break
            line = line.strip()

            if line.find(b'Number of Vertices')>= 0:
                Nvertices = int(self.fp.readline().strip())
            elif line.find(b'Number of Polygons')>=0:
                Npolys = int(self.fp.readline().strip())
            elif line.find(b'Number of Sides')>=0:
                Nsides = int(self.fp.readline().strip())
            elif line.find(b'NODATA (land) value')>=0:
                nodata_value = float(self.fp.readline().strip())
            elif line.find(b'Vertex Data:') >= 0:
                vertex_data_offset = self.fp.tell()
                for i in range(Nvertices):
                    self.fp.readline()
            elif line.find(b'Polygon Data:') >= 0:
                polygon_data_offset = self.fp.tell()
                for i in range(Npolys):
                    self.fp.readline()
            elif line.find(b'Side Data:') >= 0:
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
            self.nodes['x'][i,:] = [float(s) for s in line[1:]]

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
            # polygon_number, number_of_sides,center_x, center_y, center_depth,
            #   side_indices(number_of_sides), marker(0=internal,1=open boundary)
            poly_id = int(line[0])
            nsides_this_poly = int(line[1])
            self.cells['depth'][i] = float(line[4])

            # grab all of the edge indices:
            self.cells['edges'][i,:nsides_this_poly] = [int(s)-1 for s in line[5:5+nsides_this_poly]]

    def read_sides(self,side_data_offset):
        print( "Reading sides" )
        # Side Data: side_number, side_depth, node_indices(2), cell_indices(2),
        #    marker(0=internal,1=external,2=flow boundary,3=open boundary)

        self.fp.seek(side_data_offset)

        # store nodeA,nodeB, cell1,cell2, marker
        for i in range(self.Nedges()):
            line = self.fp.readline().split()
            side_id = int(line[0])
            self.edges['depth'][i] = float(line[1])
            self.edges['nodes'][i,:] = [int(s)-1 for s in line[2:4]]
            self.edges['cells'][i,:] = [int(s)-1 for s in line[4:6]]
            self.edges['mark'][i] = int(line[6])


class RgfGrid(UnstructuredGrid):
    """
    Read structured (curvilinear) Delft3D grids
    """
    max_sides=4
    class GrdTok(object):
        def __init__(self,grd_fn):
            self.fp=open(grd_fn,'rt')
            self.buff=None # unprocessed data
        def read_key_value(self):
            key,value = self.try_read_key_value()
            assert key is not None
            return key,value
        
        def try_read_key_value(self):
            while self.buff is None:
                self.buff=self.fp.readline().strip()
                if self.buff[0]=='*':
                    self.buff=None
            if '=' not in self.buff:
                return None,None
            key,value=self.buff.split('=',1)
            self.buff=None
            key=key.strip()
            value=value.strip()
            return key,value
        def read_token(self):
            while self.buff is None:
                self.buff=self.fp.readline().strip()
                if self.buff[0]=='*':
                    self.buff=None
            parts=self.buff.split(None,1)
            if len(parts)==0:
                self.buff=None
                return None
            if len(parts)==1:
                self.buff=None
                return parts[0]
            if len(parts)==2:
                self.buff=parts[1]
                return parts[0]
            raise Exception("not reached")

    def __init__(self,grd_fn,dep_fn='infer',enc_fn='infer'):
        super(RgfGrid,self).__init__()
        
        tok=self.GrdTok(grd_fn)

        metadata={
            'Missing Value':np.nan, # probably could find a better default
            'Coordinate System':None # probably not the right string
        }
        
        while 1: # read key-value pairs
            key,value = tok.try_read_key_value()
            if key is not None:
                metadata[key] = value
            else:
                break
        #_,coord_sys=tok.read_key_value()
        #_,missing_val=tok.read_key_value()
        missing_val=float(metadata['Missing Value'])

        m_count=int(tok.read_token())
        n_count=int(tok.read_token())
        [tok.read_token() for _ in range(3)] # docs say they aren't used

        xy=np.zeros( (n_count,m_count,2), np.float64)

        def read_coord():
            v=float(tok.read_token())
            if v==missing_val:
                return np.nan
            else:
                return v

        for comp in [0,1]:
            for row in range(n_count):
                tok.read_token()  # ETA=
                row_num=int(tok.read_token())
                assert row_num==row+1

                for col in range(m_count):
                    xy[row,col,comp]=read_coord()

        self.add_node_field('row',np.zeros(0,np.int32))
        self.add_node_field('col',np.zeros(0,np.int32))
        self.add_cell_field('row',np.zeros(0,np.int32))
        self.add_cell_field('col',np.zeros(0,np.int32))

        # Add nodes:
        node_idxs=np.zeros( (n_count,m_count), np.int32)-1
        cell_idxs=np.zeros( (n_count-1,m_count-1), np.int32)-1

        for row in range(n_count):
            for col in range(m_count):
                if np.isfinite(xy[row,col,0]):
                    node_idxs[row,col]=self.add_node(x=xy[row,col],row=row,col=col)

        # Add cells, filling in edges as needed
        for row in range(n_count-1):
            for col in range(m_count-1):
                nodes=[ node_idxs[row,col],
                        node_idxs[row,col+1],
                        node_idxs[row+1,col+1],
                        node_idxs[row+1,col] ]
                if np.any(np.array(nodes)<0): continue
                cell_idxs[row,col]=self.add_cell_and_edges(nodes=nodes,row=row,col=col)

        # Fast lookup -- but might become stale...
        self.rowcol_to_node=node_idxs
        self.rowcol_to_cell=cell_idxs
        self.grd_filename=grd_fn

        if dep_fn=='infer':
            dep_fn=grd_fn.replace('.grd','.dep')
        if dep_fn is not None:
            self.read_depth(dep_fn)

        if enc_fn=='infer':
            enc_fn=grd_fn.replace('.grd','.enc')
        if enc_fn is not None:
            self.read_enclosure(enc_fn)

    def read_depth(self,dep_fn):
        # And the depth file?
        # Hmm - have a staggering issue.  This file is 1 larger.
        # docs say that the grd file has coordinates for the "depth points".
        # maybe depth is given at nodes, but "depth points" is like arakawa
        # C, and cell-centered?

        dep_data=np.fromfile(dep_fn,sep=' ')
        dep2d=dep_data.reshape( (self.rowcol_to_node.shape[0]+1,
                                 self.rowcol_to_node.shape[1]+1) )
        # Seems like the staggering is off, but when I try to average down to
        # the number of nodes I have, the values are bad.  Suggests that even though
        # the depth data is 1 larger in each coordinate direction, it is still just
        # node centered (or at least centered on what I have claimed to be nodes...)
        #dep2d_centered=0.25*(dep2d[1:,1:] + dep2d[:-1,:-1] + dep2d[1:,:-1] + dep2d[:-1,1:])
        dep2d_centered=dep2d[:-1,:-1]
        dep_node_centered=dep2d_centered[ self.nodes['row'], self.nodes['col']]

        self.add_node_field('depth_node',dep_node_centered)
        self.add_cell_field('depth_cell',self.interp_node_to_cell(dep_node_centered))
        
    def read_enclosure(self,enc_fn):
        """
        Read the enclosure file. Saves the list of row/col indices, 0-based,
        to self.enclosure.

        Note that this is just for logical comparisons on the
        grid, not for geographic representation. The range of indices is 1 greater
        than the grid indices for nodes and 2 greater than grid indices for cells.
        This is because the vertices of the enclosure are on "ghost" cell centers
        outside the actual domain. 
        """
        with open(enc_fn,'rt') as fp:
            ijs=[]
            for line in fp:
                line=line.strip().split('*')[0]
                if not line: continue
                row,col=[int(s) for s in line.split()]
                ijs.append( [row,col] )
        self.enclosure=np.array(ijs)-1

def cleanup_dfm_multidomains(grid):
    """
    Given an unstructured grid which was the product of DFlow-FM
    multiple domains stitched together, fix some of the extraneous
    geometries left behind.
    Grid doesn't have to have been read as a DFMGrid.

    Cell indices are preserved, but node and edge indices are not.
    """
    grid.log.debug("Regenerating edges")
    grid.make_edges_from_cells()
    grid.log.debug("Removing orphaned nodes")
    grid.delete_orphan_nodes()
    grid.log.debug("Removing duplicate nodes")
    grid.merge_duplicate_nodes() # this can delete edges
    
    # To avoid downstream errors when the 'deleted' flags
    # are not handled, renumber.
    
    grid.log.debug("Renumbering nodes")
    grid.renumber_nodes()
    grid.log.debug("Renumbering edges") 
    grid.renumber_edges()
    
    grid.log.debug("Extracting grid boundary")
    return grid


