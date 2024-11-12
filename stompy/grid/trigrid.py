"""
Class for representing and manipulating triangular grids.
This has largely been superceded by unstructured_grid.py, but remains
here for compatibility with some SUNTANS code, as well as the
foundation for tom.py, which is still better than the unstructured_grid.py-based
grid generation.
"""
from __future__ import print_function

import six
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import collections
except:
    pass

try:
    # moved around a while back
    nanmean=np.nanmean
except AttributeError:
    from scipy.stats import nanmean

# qgis clashes with the Rtree library (because it includes its own local copy).
# fall back to a wrapper around the qgis spatial index if it looks like we're running
# under qgis.
# from safe_rtree import Rtree
# 2015-11-18: should be history, qgis was patched at some point, iirc

# updated version of safe_rtree which tries several approaches
from ..spatial.gen_spatial_index import PointIndex as Rtree

try:
    from shapely import geometry
    import shapely.predicates
except ImportError:
    print("Shapely is not available!")
    geometry = "unavailable"
    
from .. import priority_queue as pq
from ..spatial import join_features

import os, types

try:
    try:
        from osgeo import ogr, osr
    except ImportError:
        import ogr, osr
except ImportError:
    print("GDAL failed to load")
    ogr = "unavailable"
    osr = ogr
    
from ..utils import array_append

# edge markers:
CUT_EDGE = 37 # the marker for a cut edge
OPEN_EDGE = 3
LAND_EDGE = 1
DELETED_EDGE = -1

# edge-cell markers ( the cell ids that reside in the edge array
BOUNDARY = -1 # cell marker for edge of domain
UNMESHED = -2 # cell marker for edges not yet meshed

xxyy = np.array([0, 0, 1, 1])
xyxy = np.array([0, 1, 0, 1])


def dist(a, b):
    return np.sqrt(np.sum((a-b)**2,axis=-1))

# rotate the given vectors/points through the CCW angle in radians
def rot(angle, pnts):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return np.tensordot(R, pnts, axes=(1, -1) ).transpose() # may have to tweak this for multiple points

def signed_area(points):
    i = np.arange(points.shape[0])
    ip1 = (i+1)%(points.shape[0])
    return 0.5*(points[i, 0]*points[ip1, 1] - points[ip1, 0]*points[i, 1]).sum()
    
def is_ccw(points):
    return signed_area(points) > 0
    
def ensure_ccw(points):
    if not is_ccw(points):
        # print "Hey - you gave me CW points.  I will reverse"
        points = points[::-1]
    return points

def ensure_cw(points):
    if is_ccw(points):
        # print "Hey - you gave me CCW points.  I will reverse"
        points = points[::-1]
    return points


def outermost_rings( poly_list ):
    """ 
    given a list of Polygons, return indices for those that are not inside
    any other polygon
    """
    areas = np.array( [p.area for p in poly_list])
    order = np.argsort(-1 * areas) # large to small
    outer = []

    for i in range(len(order)):
        ri = order[i]
        # print "Checking to see if poly %d is an outer polygon"%ri

        is_exterior = 1
        # check polygon ri (the ith largest) against all polygons
        # larger than it.
        for j in range(i):
            rj = order[j]

            if poly_list[rj].contains( poly_list[ri] ):
                # print "%d contains %d"%(rj,ri)
                is_exterior = 0 # ri is contained by rj, so not exterior
                break

        if is_exterior:
            # print "%d is exterior"%ri
            outer.append(ri)
    return outer


def circumcenter(p1,p2,p3):
    ref = p1
    
    p1x = p1[...,0] - ref[...,0] # ==0.0
    p1y = p1[...,1] - ref[...,1] # ==0.0
    p2x = p2[...,0] - ref[...,0]
    p2y = p2[...,1] - ref[...,1]
    p3x = p3[...,0] - ref[...,0]
    p3y = p3[...,1] - ref[...,1]

    vc = np.zeros( p1.shape, np.float64)
    
    # taken from TRANSFORMER_gang.f90
    dd=2.0*((p1x-p2x)*(p1y-p3y) -(p1x-p3x)*(p1y-p2y))
    b1=p1x**2+p1y**2-p2x**2-p2y**2
    b2=p1x**2+p1y**2-p3x**2-p3y**2 
    vc[...,0]=(b1*(p1y-p3y)-b2*(p1y-p2y))/dd + ref[...,0]
    vc[...,1]=(b2*(p1x-p2x)-b1*(p1x-p3x))/dd + ref[...,1]
    
    return vc


class TriGridError(Exception):
    pass

class NoSuchEdgeError(TriGridError):
    pass

class NoSuchCellError(TriGridError):
    pass


# cache the results of reading points.dat files for suntans grid files
# maps filenames to point arrays - you should probably
# just copy the array, though, since there is the possibility
# of altering the points array
points_dat_cache = {}


class TriGrid(object):
    index = None
    edge_index = None
    _vcenters = None
    verbose = 0
    default_clip = None
    
    def __init__(self,sms_fname=None,
                 tri_basename=None,
                 suntans_path=None,processor=None,
                 suntans_reader=None,
                 tec_file=None,
                 gmsh_basename=None,
                 edges=None,points=None,cells=None,
                 readonly=False):
        self.sunreader = None
        self._pnt2cells = None
        self.readonly = readonly

        self.init_listeners()
        
        if sms_fname:
            self.read_sms(sms_fname)
        elif tri_basename:
            self.read_triangle(tri_basename)
        elif gmsh_basename:
            self.read_gmsh(gmsh_basename)
        elif suntans_path:
            self.processor = processor
            self.suntans_path = suntans_path
            self.read_suntans()
        elif suntans_reader:
            self.processor = processor
            self.sunreader = suntans_reader
            self.read_suntans()
        elif tec_file:
            self.read_tecplot(tec_file)
        elif points is not None:
            self.from_data(points,edges,cells)
        else:
            # This will create zero-length arrays for everyone.
            self.from_data(None,None,None)

    def file_path(self,conf_name):
        if self.sunreader:
            return self.sunreader.file_path(conf_name,self.processor)
        else:
            if conf_name == 'points':
                basename = 'points.dat'
            elif conf_name == 'edges':
                basename = 'edges.dat'
            elif conf_name == 'cells':
                basename = 'cells.dat'
            else:
                raise Exception("Unknown grid conf. name: "+conf_name)
            if self.processor is not None and conf_name != 'points':
                basename = basename + ".%i"%self.processor
            return self.suntans_path + '/' + basename
    def from_data(self,points,edges,cells):
        if points is None:
            self.points = np.zeros( (0,2), np.float64 )
        else:
            self.points = points[:,:2]  # discard any z's that come in

        if cells is None:
            self.cells = np.zeros( (0,3), np.int32)
        else:
            self.cells = cells
        
        if edges is None:
            self.edges = np.zeros((0,5),np.int32)
        else:
            ne = len(edges)
            # incoming edges may just have connectivity
            if edges.shape[1] == 2:
                self.edges = np.zeros( (ne,5), np.int32)
                self.edges[:,:2] = edges

                # defaults:
                self.edges[:,2] = LAND_EDGE 
                self.edges[:,3] = UNMESHED
                self.edges[:,4] = BOUNDARY

                # update based on cell information:
                self.set_edge_neighbors_from_cells()

                # And make a better guess at edge marks
                internal = (self.edges[:,3]>=0) & (self.edges[:,4]>=0)
                self.edges[internal,2] = 0 
            elif edges.shape[1] == 5:
                self.edges = edges
            else:
                raise Exception("Edges should have 2 or 5 entries per edge")

            
    def set_edge_neighbors_from_cells(self):
        iip = np.array([[0,1],[1,2],[2,0]])

        for c in six.moves.range(self.Ncells()):
            for pair in iip:
                nodes = self.cells[c,pair]
                j = self.find_edge(nodes)
                if nodes[0] == self.edges[j,0]:
                    self.edges[j,3] = c
                else:
                    self.edges[j,4] = c

    def refresh_metadata(self):
        """ 
        Call this when the cells, edges and nodes may be out of sync with indices
        and the like.
        """
        self.index = None
        self.edge_index = None
        self._vcenters = None

    def read_suntans(self,use_cache=1):
        self.read_from = "Suntans"

        # read the points:
        points_fn = os.path.abspath( self.file_path("points") )
        if use_cache and points_fn in points_dat_cache:
            self.points = points_dat_cache[points_fn]
            if not self.readonly:
                self.points = self.points.copy()
        else:
            points_fp = open(points_fn)

            pnts = []
            for line in points_fp:
                coords = list([float(s) for s in line.split()])
                if len(coords) >= 2:
                    pnts.append(coords[:2])
            self.points = np.array(pnts)
            if use_cache:
                if self.readonly:
                    points_dat_cache[points_fn] = self.points
                else:
                    points_dat_cache[points_fn] = self.points.copy()

        # read the cells:
        cell_fname = self.file_path("cells")
            
        cells_fp = open(cell_fname)
        vcenters = []
        cells = []
        for line in cells_fp:
            line = line.split()
            if len(line)==8:
                # first two are voronoi center coordinates
                vcenters.append( list([float(s) for s in line[:2]]))
                # then three point indices:
                cells.append( list([int(s) for s in line[2:5]]) )
        self._vcenters = np.array(vcenters)
        self.cells = np.array(cells)

        self.cell_mask = np.ones( len(self.cells) )
        
        # Edges!
        # Each line is endpoint_i endpoint_i  edge_marker  cell_i cell_i
        edge_fname = self.file_path('edges')
        # print "Reading edges from %s"%edge_fname

        # edges are stored just as in the data file:
        #  point_i, point_i, marker, cell_i, cell_i
        edges_fp = open(edge_fname,"rt")

        edges = []
        for line in edges_fp:
            line = line.split()
            if len(line) == 5:
                edges.append( list([int(s) for s in line]) )
        self.edges = np.array(edges)

    def read_gmsh(self,gmsh_basename):
        """ 
        reads output from gmsh  - gmsh_basename.{nod,ele}
        """
        self.fname = gmsh_basename
        self.read_from = "GMSH"
        
        self._vcenters = None # will be lazily created

        points = np.loadtxt( self.fname +".nod")
        id_offset = int(points[0,0]) # probably one-based

        self.points = points[:,1:3]
        
        print("Reading cells")
        elements = np.loadtxt( self.fname +".ele")

        self.cells = elements[:,1:4].astype(np.int32) - id_offset
        self.cell_mask = np.ones( len(self.cells) )
        self.make_edges_from_cells()

        print( "Done")

    def read_triangle(self,tri_basename):
        """ 
        reads output from triangle, tri_basename.{ele,poly,node}
        """
        self.fname = tri_basename
        self.read_from = "Triangle"
        
        self._vcenters = None # will be lazily created

        
        points_fp = open(tri_basename + ".node")

        Npoints,point_dimension,npoint_attrs,npoint_markers = [int(s) for s in points_fp.readline().split()]

        self.points = np.zeros((Npoints,2),np.float64)
        id_offset = 0
        for i in range(self.Npoints()):
            line = points_fp.readline().split()
            # pnt_id may be 0-based or 1-based.
            pnt_id = int(line[0])
            if i == 0:
                id_offset = pnt_id
                print( "Index offset is ",id_offset)

            # let z component stay 0
            self.points[i,:2] = list( [float(s) for s in line[1:3]] )
        points_fp.close()

        print("Reading cells")
        elements_fp = open(tri_basename + ".ele")
        Ncells,node_per_tri,ncell_attrs = [int(s) for s in elements_fp.readline().split()]

        if node_per_tri != 3:
            raise Exception("Please - just use 3-point triangles!")
        
        self.cells = np.zeros((Ncells,3),np.int32) 
        self.cell_mask = np.ones( len(self.cells) )

        for i in range(self.Ncells()):
            parsed = [int(s) for s in elements_fp.readline().split()]
            cell_id = parsed[0]
            self.cells[i] = np.array(parsed[1:]) - id_offset

        edges_fn = tri_basename + ".edge"
        if os.path.exists(edges_fn):
            edges_fp = open()

            Nedges,nedge_markers = [int(s) for s in edges_fp.readline().split()]

            self.edges = np.zeros((Nedges,5),np.int32)

            # each edge is stored as:  (pnt_a, pnt_b, default_marker,node_1,node_2)
            for i in range(Nedges):
                idx,pnta,pntb,marker = [int(s) for s in edges_fp.readline().split()]
                # and a bit of work to figure out which cells border this edge:
                pnta -= id_offset
                pntb -= id_offset

                cells_a = self.pnt2cells(pnta)
                cells_b = self.pnt2cells(pntb)

                adj_cells = list(cells_a.intersection(cells_b))
                neighbor1 = adj_cells[0]

                if len(adj_cells) == 1:
                    neighbor2 = -1
                else:
                    neighbor2 = adj_cells[1]

                self.edges[i] = [pnta,pntb,marker,neighbor1,neighbor2]
        else:
            print( "No edges - will recreate from cells")
            self.make_edges_from_cells()
            

        print( "Done")
        

    def read_tecplot(self,fname):
        self.read_from = 'tecplot'
        self.fname = fname

        self._vcenters = None # lazy creation

        fp = open(fname)

        while 1:
            line = fp.readline()
            if line.find('ZONE') == 0:
                break

        import re
        m = re.search(r'\s+N=\s*(\d+)\s+E=\s*(\d+)\s',line)
        if not m:
            print("Failed to parse: ")
            print( line)
            raise Exception("Tecplot parsing error")

        
        # first non-blank line has number of cells and edges:
        Ncells = int( m.group(2) )
        Npoints = int( m.group(1) )

        self.points = np.zeros((Npoints,2),np.float64)
        for i in range(Npoints):
            self.points[i,:] = [float(s) for s in fp.readline().split()]

        print("Reading cells")
        self.cells = np.zeros((Ncells,3),np.int32) # store zero-based indices
        self.cell_mask = np.ones( len(self.cells) )

        # we might be reading in the output from ortho, in which
        # it reports the number of unique cells, not the real number
        # cells

        i=0
        cell_hash = {}
        for line in fp:
            pnt_ids = np.array( [int(s) for s in line.split()] )

            my_key = tuple(np.sort(pnt_ids))
            if my_key not in cell_hash:
                cell_hash[my_key] = i
                
                # store them as zero-based
                self.cells[i] = pnt_ids - 1
                i += 1

        if i != Ncells:
            print( "Reading %i cells, but expected to get %i"%(i,self.Ncells))
            self.cells = self.cells[:i,:]

        # At this point we have enough info to create the edges
        self.make_edges_from_cells()

    # these are used in some gui code
    _cell_centers = None
    def cell_centers(self):
        if self._cell_centers is None:
            self._cell_centers = self.points[self.cells].mean(axis=1)
        return self._cell_centers
    
    _edge_centers = None
    def edge_centers(self):
        if self._edge_centers is None:
            self._edge_centers = self.points[self.edges[:,:2]].mean(axis=1)

        return self._edge_centers

    def ghost_cells(self):
        """ 
        Return a bool array, with ghost cells marked True
        Ghost cells are determined as any cell with an edge that has marker 6
        """
        ghost_edge = self.edges[:,2] == 6
        ghost_cells = self.edges[ghost_edge,3:5].ravel()

        bitmap = np.zeros( self.Ncells(), np.bool_ )
        bitmap[ ghost_cells ] = True
        return bitmap

    def delete_unused_nodes(self):
        """ 
        any nodes which aren't in any cells or edges will be removed.
        """
        all_nodes = np.arange(self.Npoints())

        cell_nodes = np.unique(self.cells.ravel())
        edge_nodes = np.unique(self.edges[:,:2].ravel())
        deleted_nodes = np.nonzero(np.isnan(self.points[:,0]))[0]

        okay_nodes = np.unique( np.concatenate( (cell_nodes,edge_nodes,deleted_nodes) ) )

        unused = np.setdiff1d(all_nodes,okay_nodes)

        for n in unused:
            self.delete_node(n)
        
    def renumber(self):
        """
        removes duplicate cells and nodes that are not
        referenced by any cell, as well as cells that have been deleted (==-1)
        """
        cell_hash = {} # sorted tuples of vertices
        new_cells = [] # list of indexes into the old ones
        for i in range(self.Ncells()):
            my_key = tuple( np.sort(self.cells[i]) )

            if my_key not in cell_hash and self.cells[i,0] >= 0:
                # we're original and not deleted
                cell_hash[my_key] = i # value is ignored...
                new_cells.append( i )

        self.cells = self.cells[new_cells]

        # remove lonesome nodes
        active_nodes = np.unique(self.cells.ravel())
        if np.any(active_nodes) <= 0:
            raise Exception("renumber: Active nodes includes some negative indices")

        old_indices = -np.ones(self.Npoints(),np.int32)
        
        self.points = self.points[active_nodes]
        if np.any(np.isnan(self.points)):
            raise Exception("renumber: some points have NaNs!")
        
        # need a mapping from active node to its index -
        # explicitly ask for int32 for consistency
        new_indices = np.arange(active_nodes.shape[0],dtype=np.int32)
        old_indices[active_nodes] = new_indices
        # map onto the new indices
        self.cells = old_indices[self.cells]

        if np.any(self.cells) < 0:
            raise Exception("renumber: after remapping indices, have negative node index in cells")

        # clear out stale data
        self._pnt2cells = None
        self.index = None
        self.edge_index = None
        self._pnt2edges = None
        self._vcenters = None

        # rebuild the edges
        self.make_edges_from_cells()

        # return the mappings so that subclasses can catch up
        return {'valid_cells':new_cells,'pointmap':old_indices,
                'valid_nodes':active_nodes}

    def write_Triangle(self,basename,boundary_nodes=None):
        """  
        duplicate some of the output of the Triangle program -
        particularly the .node and .ele files

        note that node and cell indices are taken as 1-based.

        if boundary_nodes is supplied, it should be an integer valued array of length Npoints,
        and give the boundary marker for each node (usually 0 for internal, nonzero for boundary).
        this can be used to specify a subset of the boundary nodes for a BC in SWAN.

        if not specified, boundary markers will be 0 for internal, 1 for external nodes.
        """
        node_fp = open(basename + ".node",'wt')
        node_fp.write("%d 2 0 1\n"%(self.Npoints()))
        for n in range(self.Npoints()):
            if boundary_nodes is not None:
                bmark = boundary_nodes[n]
            else:
                # id x y boundary marker
                bmark = 0
                if self.boundary_angle(n) != 0:
                    bmark = 1
            node_fp.write("%d %f %f %d\n"%(n+1,self.points[n,0],self.points[n,1], bmark ) )
        node_fp.close()

        ele_fp = open(basename + ".ele",'wt')
        ele_fp.write("%d 3 0\n"%(self.Ncells()))
        for i in range(self.Ncells()):
            ele_fp.write("%d %d %d %d\n"%(i+1,self.cells[i,0]+1,self.cells[i,1]+1,self.cells[i,2]+1))
        ele_fp.close()

    def write_obj(self,fname):
        """ 
        Output to alias wavefront
         - scales points to fall within [0,10]
        """
        fp = open(fname,'wt')

        pmax = self.points.max(axis=0)
        pmin = self.points.min(axis=0)
        rng = (pmax-pmin).max()

        scaled_points = (self.points - pmin)*(10/rng)

        for i in six.moves.range(self.Npoints()):
            fp.write("v %f %f 0.0\n"%(scaled_points[i,0],scaled_points[i,1]))

        for i in six.move.range(self.Ncells()):
            fp.write("f %d %d %d\n"%(self.cells[i,0]+1,
                                     self.cells[i,1]+1,
                                     self.cells[i,2]+1))

        fp.close()
        
    def write_tulip(self,fname):
        """ 
        Write a basic representation of the grid to a tulip
        compatible file
        """
        fp = open(fname,'wt')

        fp.write("(tlp \"2.0\"\n")
        
        fp.write("(nodes ")

        for i in six.moves.range(self.Npoints()):
            if not np.isnan(self.points[i,0]):
                fp.write(" %i"%i )

        fp.write(")\n")

        for e in six.moves.range(self.Nedges()):
            if self.edges[e,0] >= 0:
                fp.write("(edge %i %i %i)\n"%(e,self.edges[e,0],self.edges[e,1]))

        # and the locations of the nodes
        fp.write("(property 0 layout \"viewLayout\" \n")
        for i in six.moves.range(self.Npoints()):
            if not np.isnan(self.points[i,0]):
                fp.write("  (node %i \"(%f,%f,0)\")\n"%(i,self.points[i,0],self.points[i,1]))

        fp.write(")\n")
        
        fp.write(")\n")
        fp.close()
        
    def write_sms(self,fname):
        fp = open(fname,'wt')

        fp.write("\n") # seems to start with blank line.
        fp.write("%i  %i\n"%(self.Ncells(),self.Npoints()))
        
        # each point has three numbers, though the third is apparently
        # always 0

        for i in range(self.Npoints()):
            fp.write("%10i %.11f %.11f %.11f\n"%(i+1,
                                                 self.points[i,0],
                                                 self.points[i,1],
                                                 0.0 ))
            
        # everything is a triangle

        # compute area, positive means CCW
        #  - turns out SMS wants the order to be consistent, but it always *creates* CCW
        # triangles.  so best to create CCW triangles
        bad = self.areas() < 0
        n_bad = np.sum(bad)
        
        if n_bad > 0:
            print( "Found %i CW triangles that will be reversed"%n_bad )

            self.cells[bad,: ] = self.cells[bad,::-1]

        for i in range(self.Ncells()):
            fp.write("%i 3 %i %i %i\n"%(i+1,
                                        self.cells[i,0]+1,
                                        self.cells[i,1]+1,
                                        self.cells[i,2]+1) )

        # And then go back and switch the marker for some of the edges:
        print( "SMS output: omitting boundary information")
        fp.write("0 = Number of open boundaries\n")
        fp.write("0 = Total number of open boundary nodes\n")
        fp.write("0 = Number of land boundaries\n")
        fp.write("0 = Total number of land boundary nodes\n")
        fp.close()

    def areas(self):
        """ 
        returns signed area, CCW is positive
        """
        i = np.array([0,1,2])
        ip = np.array([1,2,0])
        xi = self.points[self.cells[:,i],0]
        yi = self.points[self.cells[:,i],1]
        xip = self.points[self.cells[:,ip],0]
        yip = self.points[self.cells[:,ip],1]

        A = 0.5 * (xi*yip-xip*yi).sum(axis=1)

        return A

    def angles(self):
        """ 
        returns [Nc,3] array of internal angles, in radians
        """ 
        triples=np.array( [[0,1,2],[1,2,0],[2,0,1] ] )
        all_triples=self.points[self.cells[:,triples]]
        delta=np.diff(all_triples,axis=2)
        abs_angles=np.arctan2(delta[...,1],delta[...,0])
        rel_angles=(abs_angles[...,1] - abs_angles[...,0])
        int_angles= np.pi - (rel_angles%(2*np.pi))
        return int_angles
            
    def read_sms(self,fname):
        self.fname = fname
        self.read_from = "SMS"
        self._vcenters = None # will be lazily created

        fp = open(fname)

        # skip leading blank lines
        while 1:
            line = fp.readline().strip()
            if line != "":
                break

        # first non-blank line has number of cells and edges:
        Ncells,Npoints = [int(s) for s in line.split()]

        # each point has three numbers, though the third is apparently
        # always 0

        self.points = np.zeros((Npoints,2),np.float64)
        for i in range(Npoints):
            line = fp.readline().split()
            # pnt_id is 1-based
            pnt_id = int(line[0])

            self.points[pnt_id-1] = [float(_s) for _s in line[1:3]]

        print("Reading cells")
        self.cells = np.zeros((Ncells,3),np.int32) # store zero-based indices, and assume
        self.cell_mask = np.ones( len(self.cells) )

        # everything is a triangle
        for i in range(Ncells):
            parsed = [int(_s) for _s in fp.readline().split()]
            cell_id = parsed[0]
            nvertices = parsed[1]
            pnt_ids = np.array(parsed[2:])

            if nvertices != 3:
                raise Exception("Assumption of all triangles is not true!")
            # store them as zero-based
            self.cells[cell_id-1] = pnt_ids - 1

        # At this point we have enough info to create the edges
        self.make_edges_from_cells()

        # And then go back and switch the marker for some of the edges:
        print("Reading boundaries")
        def read_first_int():
            return int(fp.readline().split()[0])

        for btype in ['open','land']:
            if btype == 'open':
                marker = 3 # open - not sure if this is 2 or 3...
            else:
                marker = 1 # closed
                
            n_boundaries = read_first_int()
            print( "Number of %s boundaries: %d"%(btype,n_boundaries) )
            tot_boundary_nodes = read_first_int() # who cares...

            for boundary_i in range(n_boundaries):
                print( "Reading %s boundary %d"%(btype,boundary_i+1))
                n_nodes_this_boundary = read_first_int()
                for i in range(n_nodes_this_boundary):
                    node_i = read_first_int() - 1 # zero-based
                    if i>0:
                        # update the marker in edges
                        if node_i < last_node_i:
                            pa,pb = node_i,last_node_i
                        else:
                            pa,pb = last_node_i,node_i
                            
                        try:
                            edge_i = self.find_edge((pa,pb))
                            self.edges[edge_i,2] = marker
                        except NoSuchEdgeError:
                            print( "Couldn't find edge",(pa,pb))
                            print(self.points[ [pa,pb] ])
                            raise
                            
                    last_node_i = node_i

        print( "Done")


    def pnt2cells(self,pnt_i):
        if self._pnt2cells is None:
            # build hash table for point->cell lookup
            self._pnt2cells = {}
            for i in range(self.Ncells()):
                for j in range(3):
                    if self.cells[i,j] not in self._pnt2cells:
                        self._pnt2cells[self.cells[i,j]] = set()
                    self._pnt2cells[self.cells[i,j]].add(i)
        return self._pnt2cells[pnt_i]

    def Nedges(self):
        return len(self.edges)
    def Ncells(self):
        return len(self.cells)
    def Npoints(self):
        return len(self.points)
    

    _pnt2edges = None
    def pnt2edges(self,pnt_i):
        if self._pnt2edges is None:
            # print "building pnt2edges"
            
            p2e = {}
            for e in range(self.Nedges()):
                # skip deleted edges
                if self.edges[e,2] == DELETED_EDGE:
                    continue
                
                for p in self.edges[e,:2]:
                    if p not in p2e:
                        p2e[p] = []
                    p2e[p].append(e)
            self._pnt2edges = p2e

        return self._pnt2edges.get(pnt_i,[])

    def boundary_angle(self,pnt_i):
        """ 
        returns the interior angle in radians, formed by the
        boundary at the given point
        """

        edges = self.pnt2edges(pnt_i)

        # find the absolute angle of each edge, as an angle CCW from
        # east

        angle_right=None # the angle of the edge with the domain on the right
        angle_left =None # angle of the edge with the domain on the left

        for edge in edges:
            # only care about the edges on the boundary:
            if self.edges[edge,4] != BOUNDARY:
                continue
            segment = self.edges[edge,:2]
            seg_reversed = 0
            if segment[0] != pnt_i:
                segment = segment[::-1]
                seg_reversed = 1
                
            # sanity check
            if segment[0] != pnt_i:
                raise Exception( "Well, where is %d in %s"%(pnt_i,segment) )

            delta = self.points[segment[1]] - self.points[segment[0]]

            angle = np.arctan2(delta[1],delta[0])
            # print "Edge %i to %i has angle %g degrees"%(edge,segment[1],180*angle/pi)

            # on which side of this edge is the domain?
            my_cell = self.edges[edge,3]

            if my_cell == UNMESHED:
                # the paver enforces that cell markers are 3=>left,4=>right
                # so with the stored order of the edge, the pretend cell center
                # is always to the left
                if not seg_reversed:
                    xprod = -1
                else:
                    xprod = 1
            else:
                my_cell_middle = np.mean( self.points[ self.cells[my_cell] ] , axis=0 )

                delta_middle = my_cell_middle - self.points[pnt_i]

                # and cross-product:
                xprod = np.cross(delta_middle,delta)
                
            # print "Cross-product is: ",xprod
            if xprod > 0:
                # the cell center lies to the right of this edge,
                # print "Edge to %i has domain to the right"%segment[1]
                angle_right = angle
            else:
                # print "Edge to %i has domain to the left"%segment[1]
                angle_left = angle
        if angle_left is None and angle_right is None:
            # it's an interior node, so no boundary angle...
            return 0.0

        if angle_left is None:
            print( "Angle from point %i with domain to left is None!"%pnt_i )
        if angle_right is None:
            print( "Angle from point %i with domain to right is None!"%pnt_i )
        
        boundary_angle = (angle_right - angle_left) % (2*np.pi)
        return boundary_angle
        

    def plot_bad_bcs(self):
        bad_bcs = ((self.edges[:,2] == 0) != (self.edges[:,4] >= 0))
        self.plot(edge_mask = bad_bcs)

    def plot_nodes(self,ids=None):
        if ids is None:
            ids = np.arange(self.Npoints())
            if self.default_clip is not None:
                c = self.default_clip
                valid = (self.points[:,0] > c[0]) & (self.points[:,0]<c[1]) & \
                        (self.points[:,1] > c[2]) & (self.points[:,1]<c[3])
                ids= ids[valid]
            
        [plt.annotate(str(i),self.points[i]) for i in ids if not np.isnan(self.points[i,0])]
        
    def plot_edge_marks(self,edge_mask=None,clip=None):
        """ 
        label edges with c[nc1]-j[j],mark-c[nc2],
        rotated so it reads in the correct orientation for nc1, nc2
        edge_mask should be a boolean array of size Nedges()
        clip can be a list like matplotlib axis() - [xmin,xmax,ymin,ymax]
        """
        if clip is None:
            clip = self.default_clip

        if edge_mask is None and clip:
            ec = self.edge_centers()
            edge_mask = self.edges[:,0] >=0 & ((ec[:,0] >= clip[0]) & (ec[:,0]<=clip[1]) \
                            & (ec[:,1] >= clip[2]) & (ec[:,1]<=clip[3]) )
        else:
            edge_mask = self.edges[:,0] >= 0

        for e in np.nonzero(edge_mask)[0]:
            delta = self.points[ self.edges[e,1]] - self.points[self.edges[e,0]]
            angle = np.arctan2(delta[1],delta[0])
            plt.annotate("c%d-j%d,%d-c%d"%(self.edges[e,3],e,self.edges[e,2],self.edges[e,4]),
                         ec[e],rotation=angle*180/np.pi - 90,ha='center',va='center')

    def plot(self,voronoi=False,line_collection_args={},
             all_cells=True,edge_values=None,
             edge_mask=None,vmin=None,vmax=None,ax=None,
             clip=None):
        """ 
        vmin: if nan, don't set an array at all for the edges

        clip=[xmin,xmax,ymin,ymax]: additionally mask edges which are not within the given rectangle

        edge_values: defaults to the edge marker.
        
        """
        if ax is None:
            ax = plt.gca()

        if self.Ncells() == 0:
            voronoi = False
            
        if voronoi:
            self.vor_plot = ax.plot(self.vcenters()[:,0],self.vcenters()[:,1],".")

        if self.Nedges() == 0:
            return

        if edge_mask is None:
            if not all_cells:
                edge_mask = self.edges[:,4] < 0
            else:
                edge_mask = self.edges[:,0] >= 0 # np.ones( self.edges[:,2].shape ) == 1

        if np.sum(edge_mask) == 0:
            return
        
        # g.edges[:,:2] pulls out every edge, and just the endpoint
        #   indices.
        # indexing points by this maps the indices to points
        # which then has the z-values sliced out

        segments = self.points[self.edges[edge_mask,:2]]

        clip=clip or self.default_clip
        
        # Apply clip only to valid edges
        if clip is not None:
            # segments is Nedges * {a,b} * {x,y}
            points_visible = (segments[...,0] >= clip[0]) & (segments[...,0]<=clip[1]) \
                             & (segments[...,1] >= clip[2]) & (segments[...,1]<=clip[3])
            # so now clip is a bool array of length Nedges
            clip = any( points_visible, axis=1)
            segments = segments[clip,...]
        
        line_coll = collections.LineCollection(segments,**line_collection_args)

        if vmin is not None and np.isnan(vmin):
            print( "Skipping the edge array" )
        else:
            # allow for coloring the edges
            if edge_values is None:
                edge_values = self.edges[:,2]
                
            edge_values = edge_values[edge_mask]
            if clip is not None:
                edge_values = edge_values[clip]

            line_coll.set_array(edge_values)

            if vmin:
                line_coll.norm.vmin = vmin
            if vmax:
                line_coll.norm.vmax = vmax
        
        ax.add_collection(line_coll)

        self.edge_collection = line_coll

        ax.axis('equal')
        if not voronoi:
            # the collections themselves do not automatically set the
            # bounds of the axis
            ax.axis(self.bounds())
        return line_coll

    def plot_scalar(self,scalar,pdata=None,clip=None,ax=None,norm=None,cmap=None):
        """ Plot the scalar assuming it sits at the center of the
        cells (i.e. use the voronoi centers)
        scalar should be a 1d array, with length the same as the
        number of cells

        to mask out values, set scalar to nan
        """
        if ax is None:
            ax = plt.gca()
        
        if not pdata:
            # create a numpy array for all of the segments:
            # each segment has 4 points so that it closes the triangle
            segments = np.zeros((self.Ncells(),4,2),np.float64)
            for i in range(self.Ncells()):
                for j in range(4):
                    segments[i,j,:] = self.points[self.cells[i,j%3]]
            clip=clip or self.default_clip
            if clip:
                good_points = (self.points[:,0] > clip[0]) & \
                              (self.points[:,0] < clip[1]) & \
                              (self.points[:,1] > clip[2]) & \
                              (self.points[:,1] < clip[3])
                # how to map that onto segments?
                good_verts = good_points[self.cells]
                good_cells = good_verts.sum(axis=1) == 3
                segments = segments[good_cells]
                scalar = scalar[good_cells]
                if len(scalar) == 0:
                    return None
                
            mask = np.isnan(scalar)
            if np.any(mask):
                segments = segments[~mask]
                scalar = scalar[~mask]
                if len(scalar) == 0:
                    return None
                
            patch_coll = collections.PolyCollection(segments,edgecolors='None',antialiaseds=0,norm=norm,cmap=cmap)
            # is this sufficient for coloring? YES
            patch_coll.set_array(scalar)
            pdata = patch_coll
            ax.add_collection(patch_coll)

            ax.axis('equal')

            ax.axis(self.bounds())
        else:
            pdata.set_array(scalar)
            plt.draw()
        return pdata
    def animate_scalar(self,scalar_frames,post_proc=None):
        plt.clf() # clear figure, to get rid of colorbar, too
        vmin = scalar_frames.min()
        vmax = scalar_frames.max()
        print( "Max,min: ",vmax,vmin)
        
        pdata = self.plot_scalar(scalar_frames[0])
        plt.title("Step 0")
        pdata.norm.vmin = vmin
        pdata.norm.vmax = vmax
        plt.colorbar(pdata)

        plt.show()

        for i in range(1,scalar_frames.shape[0]):
            plt.title("Step %d"%i)
            self.plot_scalar(scalar_frames[i],pdata)
            if post_proc:
                post_proc()

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

        nc1 = self.edges[:,3]
        nc2 = self.edges[:,4].copy() 
        nc2[nc2<0] = nc1[nc2<0]

        to_show = (disc[nc1]!=disc[nc2]) & np.isfinite(scalar[nc1]+scalar[nc2]) 

        segs = self.points[ self.edges[to_show,:2], :]

        joined_segs = join_features.merge_lines(segments=segs)

        # Smooth those out some...
        def smooth_seg(seg):
            seg = seg.copy()
            seg[1:-1,:] = (2*seg[1:-1,:] + seg[0:-2,:] + seg[2:,:])/4.0
            return seg

        if smooth:
            simple_segs = [smooth_seg(seg) for seg in joined_segs]
        else:
            simple_segs = joined_segs

        ecoll = collections.LineCollection(simple_segs)
        ecoll.set_edgecolor('k')

        return ecoll

    def bounds(self):
        valid = np.isfinite(self.points[:,0])
        return (self.points[valid,0].min(),self.points[valid,0].max(),
                self.points[valid,1].min(),self.points[valid,1].max() )

    def vcenters(self):
        if self._vcenters is None:
            p1 = self.points[self.cells[:,0]]
            p2 = self.points[self.cells[:,1]]
            p3 = self.points[self.cells[:,2]]

            self._vcenters = circumcenter(p1,p2,p3)

        return self._vcenters
    
    def faces(self,i):
        # returns an 3 element array giving the edge indices for the
        # cell i
        # the 0th edge goes from the 0th vertex to the 1st.
        f = np.array([-1,-1,-1])
        for nf in range(3):
            f[nf] = self.find_edge( (self.cells[i,nf],self.cells[i,(nf+1)%3]) )
        return f

    def write_cells_shp(self,shpname,cell_mask=None,overwrite=False,fields=None):
        """
        fields: a structure array of fields to write out - see wkb2shp
        """ 
        from ..spatial import wkb2shp
        if cell_mask is None:
            cell_mask = slice(None)
        polys = self.points[self.cells[cell_mask,:],:]

        tris = [geometry.Polygon(p) for p in polys]
        wkb2shp.wkb2shp(shpname,tris,overwrite=overwrite,fields=fields[cell_mask])
        
    def write_shp(self,shpname,only_boundaries=1,edge_mask=None,overwrite=0):
        """ Write some portion of the grid to a shapefile.
        If only_boundaries is specified, write out only the edges that have non-zero marker

        For starters, this writes every edge as a separate feature, but at some point it
        may make polygons out of the edges.
        """
        if edge_mask is None:
            if only_boundaries:
                edge_mask = (self.edges[:,2] != 0)
            else:
                edge_mask = (self.edges[:,0]>=0)

        if overwrite and os.path.exists(shpname):
            # hopefully it's enough to just remove the .shp, and not worry about
            # the other files.
            os.unlink(shpname)

        # Create the shapefile
        drv = ogr.GetDriverByName('ESRI Shapefile')
        ods = drv.CreateDataSource(shpname)
        srs = osr.SpatialReference()
        srs.SetFromUserInput('EPSG:26910')

        olayer = ods.CreateLayer(shpname,
                                 srs=srs,
                                 geom_type=ogr.wkbLineString)
        edge_field = olayer.CreateField(ogr.FieldDefn('edge',ogr.OFTInteger))
        marker_field = olayer.CreateField(ogr.FieldDefn('marker',ogr.OFTInteger))
        
        fdef = olayer.GetLayerDefn()

        for j in np.nonzero(edge_mask)[0]:
            e = self.edges[j]
            geo = geometry.LineString( [self.points[e[0]], self.points[e[1]]] )
            
            new_feat_geom = ogr.CreateGeometryFromWkb( geo.wkb )

            feat = ogr.Feature(fdef)
            feat.SetGeometryDirectly(new_feat_geom)
            # force to python int, as numpy types upset swig.
            feat.SetField('edge',int(j))
            feat.SetField('marker',int(e[2]))

            olayer.CreateFeature(feat)
        olayer.SyncToDisk()

    def write_contours_shp(self,shpname,cell_depths,V,overwrite=False):
        """ like write_shp, but collects edges for each depth in V.
        """
        # because that's how suntans reads depth - no sign
        V = abs(V)
        cell_depths = abs(cell_depths)

        if overwrite and os.path.exists(shpname):
            os.unlink(shpname)

        # Create the shapefile
        drv = ogr.GetDriverByName('ESRI Shapefile')
        ods = drv.CreateDataSource(shpname)
        srs = osr.SpatialReference()
        srs.SetFromUserInput('EPSG:26910')

        olayer = ods.CreateLayer(shpname,
                                 srs=srs,
                                 geom_type=ogr.wkbLineString)

        # create some fields:
        olayer.CreateField(ogr.FieldDefn('depth',ogr.OFTReal))
        olayer.CreateField(ogr.FieldDefn('edge',ogr.OFTInteger))
        
        fdef = olayer.GetLayerDefn()

        internal = (self.edges[:,4] >= 0)

        for v in V:
            print( "Finding contour edges for depth=%f"%v)

            # These could be tweaked a little bit to get closed polygons
            on_contour = (cell_depths[self.edges[:,3]] <= v ) != (cell_depths[self.edges[:,4]] <= v)
            edge_mask = on_contour & internal

            for j in np.nonzero(edge_mask)[0]:
                e = self.edges[j]
                geo = geometry.LineString( [self.points[e[0]], self.points[e[1]]] )

                new_feat_geom = ogr.CreateGeometryFromWkb( geo.wkb )

                feat = ogr.Feature(fdef)
                feat.SetGeometryDirectly(new_feat_geom)
                feat.SetField('depth',float(v))
                feat.SetField('edge',int(j))

                olayer.CreateFeature(feat)
            olayer.SyncToDisk()

    def carve_thalweg(self,depths,threshold,start,mode,max_count=None):
        """  Ensures that there is a path of cells from the given start edge
        to deep water with all cells of at least threshold depth.

        start: edge index
        
        depths and threshold should all be as *soundings* - i.e. positive 

        mode is 'cells' - cell-centered depths
           or 'edges' - edge-centered depths

        max_count: max number of cells/edges to deepen along the path (starting
          at start).

        Modifies depths in place.
        """
        c = self.edges[start,3]

        # approach: breadth-first search for a cell that is deep enough.

        # Track who's been visited -
        # this records the index of the cell from which this cell was visited.
        visitors = -1 * np.ones(self.Ncells(),np.int32)

        # Initialize the list of cells to visit
        stack = [c]
        visitors[c] = c # sentinel - visits itself

        gold = None
        
        try:
            while 1:
                new_stack = []

                for c in stack:
                    # find the neighbors of this cell:
                    edges = self.cell2edges(c)
                    for e in edges:
                        if mode == 'edges' and depths[e] > threshold:
                            gold = c
                            raise StopIteration

                        # find the neighbor cell
                        if self.edges[e,3] == c:
                            nc = self.edges[e,4]
                        else:
                            nc = self.edges[e,3]
                            
                        # have the neighbor, but should we visit it?
                        if nc < 0 or visitors[nc] >= 0:
                            continue
                        visitors[nc] = c
                        new_stack.append(nc)

                        if mode == 'cells' and depths[nc] > threshold:
                            gold = nc
                            raise StopIteration

                # everyone at this level has been visited and we haven't hit gold.
                # on to the next ring of neighbors:
                stack=new_stack
        except StopIteration:
            pass

        # then trace back and update all the depths that are too small
        c = gold

        along_the_path = []
        while c != visitors[c]:
            if mode == 'edges':
                e = self.cells2edge(c,visitors[c])
                along_the_path.append(e)
                #if depths[e] < threshold:
                #    depths[e] = threshold
                
            c=visitors[c]
            if mode == 'cells':
                along_the_path.append(c)
                #if depths[c] < threshold:
                #    depths[c] = threshold
        if max_count is None or max_count > len(along_the_path):
            max_count = len(along_the_path)
        for item in along_the_path[-max_count:]:
            if depths[item] < threshold:
                depths[item] = threshold
                    
        # Take care of starting edge
        if mode == 'edges' and depths[start] < threshold:
            depths[start] = threshold

    def write_mat(self,fn,order='ccw'):
        from scipy.io import savemat

        if order == 'ccw':
            cslice=slice(None)
        elif order == 'cw':
            cslice=slice(None,None,-1)
        else:
            raise Exception("Bad order: %s"%order)

        d={}

        d['points'] = self.points
        # to 1-based
        d['cells'] = 1+self.cells[:,cslice]
        d['edges'] = 1+self.edges[:,:2]
        d['edge_to_cells'] = 1+self.edges[:,3:5]
        d['edge_mark']=self.edges[:,2]
        d['cell_circumcenters']=self.vcenters()
        d['readme']="\n".join(["points: [Npoints,2] node locations",
                               "cells: [Ncells,3] - one-based index into points, %s order"%order,
                               "edges: [Nedges,2] - one-based nodes for each edge",
                               "edge_to_cells: [Nedges,2] - left/right cell index for each edge.",
                               "   right_cell=-1 if on the border",
                               "edge_mark: [Nedges] - 0 for internal edge, 1 for boundary",
                               "cell_circumcenters: [Ncells,2] x/y location of circumcenter (i.e. Delaunay center)"])
        savemat(fn,d)
        
    def write_suntans(self,pathname):
        """ create cells.dat, edges.dat and points.dat
        from the TriGrid instance, all in the directory
        specified by pathname
        """
        if not os.path.exists(pathname):
            print( "Creating folder ",pathname)
            os.makedirs(pathname)
        
        # check for missing BCs
        missing_bcs = (self.edges[:,2]==0) & (self.edges[:,4]<0)
        n_missing = missing_bcs.sum()
        if n_missing > 0:
            print( "WARNING: %d edges are on the boundary but have marker==0"%n_missing )
            print( "Assuming they are closed boundaries!" )
            # make a copy so that somebody can plot the bad cells afterwards
            # with plot_missing_bcs()
            my_edges = self.edges.copy()
            my_edges[missing_bcs,2] = 1
        else:
            my_edges = self.edges
        
        cells_fp = open(pathname + "/cells.dat","w")
        edges_fp = open(pathname + "/edges.dat","w")
        points_fp= open(pathname + "/points.dat","w")

        for i in range(self.Npoints()):
            points_fp.write("%.5f %.5f 0\n"%(self.points[i,0],self.points[i,1]))
        points_fp.close()
        

        # probably this can be done via the edges array
        for i in range(self.Ncells()):
            # each line in the cell output is
            # x, y of voronoi center
            # zero-based point-indices x 3
            # zero-based ?cell? indices x 3, for neighbors?

            # find the neighbors:
            # the first neighbor: need another cell that has
            # both self.cells[i,0] and self.cells[i,1] in its
            # list.
            my_set = set([i])
            n = [-1,-1,-1]
            for j in 0,1,2:
                adj1 = self.pnt2cells(self.cells[i,j])
                adj2 = self.pnt2cells(self.cells[i,(j+1)%3])
                neighbor = adj1.intersection(adj2).difference(my_set)
                if len(neighbor) == 1:
                    n[j] = neighbor.pop()
            
            cells_fp.write("%.5f %.5f %i %i %i %i %i %i\n"%(
                    self.vcenters()[i,0],self.vcenters()[i,1],
                    self.cells[i,0],self.cells[i,1],self.cells[i,2],
                    n[0],n[1],n[2]))
        cells_fp.close()

        for edge in my_edges:
            # point_id, point_id, edge_type, cell, cell
            edges_fp.write("%i %i %i %i %i\n"%(
                edge[0],edge[1],
                edge[2],
                edge[3],edge[4]))
            

        edges_fp.close()

    def find_edge(self,nodes):
        # this way is slow - most of the time in the array ops
        
        # try:
        #     e = np.intersect1d( np.unique(self.pnt2edges(nodes[0])),
        #                         np.unique(self.pnt2edges(nodes[1])) )[0]
        # except IndexError:
        #     raise NoSuchEdgeError,str(nodes)
        # return e

        el0 = self.pnt2edges(nodes[0])
        el1 = self.pnt2edges(nodes[1])
        for e in el0:
            if e in el1:
                return e
        raise NoSuchEdgeError(str(nodes))

    def find_cell(self,nodes):
        """ return the cell (if any) that is made up of the given nodes
        depends on pnt2cells
        """
        try:
            cells_a = self.pnt2cells(nodes[0])
            cells_b = self.pnt2cells(nodes[1])
            cells_c = self.pnt2cells(nodes[2])

            c = cells_a.intersection(cells_b).intersection(cells_c)

            if len(c) == 0:
                raise NoSuchCellError()
            elif len(c) > 1:
                raise Exception("Nodes %s mapped to cells %s"%(nodes,c))
            else:
                return list(c)[0]
        except KeyError:
            raise NoSuchCellError()
            
    def cell_neighbors(self,cell_id,adjacent_only=0):
        """ return array of cell_ids for neighbors of this
        cell.  here neighbors are defined by sharing a vertex,
        not just sharing an edge, unless adjacent_only is specified.
        (in which case it only returns cells sharing an edge)
        """
        if not adjacent_only:
            neighbors = [list(self.pnt2cells(p)) for p in self.cells[cell_id]]
            return np.unique(six.moves.reduce(lambda x,y: x+y,neighbors))
        else:
            nbrs = []
            for nc1,nc2 in self.edges[self.cell2edges(cell_id),3:5]:
                if nc1 != cell_id and nc1 >= 0:
                    nbrs.append(nc1)
                if nc2 != cell_id and nc2 >= 0:
                    nbrs.append(nc2)
            return np.array(nbrs)
        
    def make_edges_from_cells(self):
        # iterate over cells, and for each cell, if it's index
        # is smaller than a neighbor or if no neighbor exists,
        # write an edge record
        edges = []
        default_marker = 0

        # this will get built on demand later.
        self._pnt2edges = None
        
        for i in range(self.Ncells()):
            # find the neighbors:
            # the first neighbor: need another cell that has
            # both self.cells[i,0] and self.cells[i,1] in its
            # list.
            my_set = set([i])
            n = [-1,-1,-1]
            for j in 0,1,2:
                pnt_a = self.cells[i,j]
                pnt_b = self.cells[i,(j+1)%3]
                    
                adj1 = self.pnt2cells(pnt_a) # cells that use pnt_a
                adj2 = self.pnt2cells(pnt_b) # cells that use pnt_b

                # the intersection is us and our neighbor
                #  so difference out ourselves...
                neighbor = adj1.intersection(adj2).difference(my_set)
                # and maybe we ge a neighbor, maybe not (we're a boundary)
                if len(neighbor) == 1:
                    n = neighbor.pop()
                else:
                    n = -1
                    
                if n==-1 or i<n:
                    # we get to add the edge:
                    edges.append((pnt_a,
                                  pnt_b,
                                  default_marker,
                                  i,n))

        self.edges = np.array(edges,np.int32)

    def verify_bc(self,do_plot=True):
        """ check to make sure that all grid boundaries have a BC set 
        """
        #  point_i, point_i, marker, cell_i, cell_i

        # marker: 0=> internal,1=> closed, 3=> open
        #  make sure that any internal boundary has a second cell index
        #  assumes that all edges have the first cell index != -1
        bad_edges = np.nonzero( (self.edges[:,2]==0) & (self.edges[:,4]==-1 ) )[0]

        if do_plot:
            for e in bad_edges:
                bad_points = self.edges[e,0:2]
            
                plt.plot(self.points[bad_points,0],
                         self.points[bad_points,1],'r-o')
        
        if len(bad_edges) > 0:
            print( "BAD: there are %d edges without BC that have only 1 cell"%len(bad_edges))
            return 0
        else:
            return 1

    def cell2edges(self,cell_i):
        if self.cells[cell_i,0] == -1:
            raise Exception("cell %i has been deleted"%cell_i)
        
        # return indices to the three edges for this cell:
        pnts = self.cells[cell_i] # the three vertices

        # the k-th edge is opposite the k-th point, like in CGAL
        edges = [ self.find_edge( (pnts[(i+1)%3], pnts[(i+2)%3]) ) for i in range(3) ]
        return edges

    _cell_edge_map = None
    def cell_edge_map(self):
        """ cell2edges for the whole grid
        return an integer valued [Nc,3] array, where [i,k] is the edge index
        opposite point self.cells[i,k]

        N.B. this is not kept up to date when modifying the grid.
        """
        if self._cell_edge_map is None:
            cem = np.zeros( (self.Ncells(),3), np.int32)

            for i in six.moves.range(self.Ncells()):
                cem[i,:] = self.cell2edges(i)
            self._cell_edge_map = cem
        return self._cell_edge_map

    def interp_cell_to_edge(self,F):
        """ given a field [Nc,...], linearly interpolate
        to edges and return [Ne,...] field.
        """
        ec = self.edge_centers()
        vc = self.vcenters()

        nc1 = self.edges[:,3]
        nc2 = self.edges[:,4]
        nc2[nc2<0] = nc1[nc2<0]

        df1 = dist(ec,vc[nc1])
        df2 = dist(ec,vc[nc2])

        nc1_weight = df2/(df1+df2)
        if F.ndim == 2:
            nc1_weight = nc1_weight[:,None]

        return nc1_weight * F[nc1] + (1-nc1_weight) * F[nc2]

    def interp_cell_to_node(self,F):
        vals=np.zeros(self.Npoints(),'f8')

        for i in range(self.Npoints()):
            cells=list(self.pnt2cells(i))
            vals[i]=F[cells].mean()
        return vals

    def cell_divergence_of_edge_flux(self,edge_flux):
        """ edge_flux is assumed to be depth integrated, but not
        horizontally integrated - so something like watts per meter
        """
        cell_to_edges = self.cell_edge_map() # slow! 30s

        ec = self.edge_centers()
        vc = self.vcenters()
        
        dxy = ec[cell_to_edges] - vc[:,None,:]
        dxy_norm = dist(dxy,0*dxy)
        # should be outward normals, [Nc,3 edges,{x,y}]
        nxy = dxy / dxy_norm[:,:,None]
        # got depth from the start, but need edge length
        edge_len = dist( self.points[self.edges[:,0]], self.points[self.edges[:,1]])
        flux_divergence = np.sum(np.sum(nxy * (edge_len[:,None] * edge_flux)[cell_to_edges],
                                        axis=1),axis=1) # maybe...
        return flux_divergence / self.areas()

    def smooth_scalar(self,cell_value):
        """
        simple method for smoothing a scalar field.  note that this is not
        conservative of anything! and the degree of smoothing is per cell, not
        per area, so results may be misleading.

        it does take care not to corrupt valid values with nans during the
        smoothing
        """
        
        nc1 = self.edges[:,3]
        nc2 = self.edges[:,4]
        nc2[nc2<0] = nc1[nc2<0]

        nc12 = self.edges[:,3:5].copy()
        boundary = nc12[:,1]<0
        nc12[boundary,1] = nc12[boundary,0]
        
        edge_mean = nanmean(cell_value[nc12],axis=1)
        # 0.5*(cell_value[nc1] + cell_value[nc2])
        
        # new_values = edge_mean[self.cell_edge_map()].mean(axis=1)
        new_values = nanmean(edge_mean[self.cell_edge_map()],axis=1)
        # but don't turn nan values into non-nan values
        new_values[np.isnan(cell_value)]=np.nan
        return new_values

    def cells2edge(self,nc1,nc2):
        e1 = self.cell2edges(nc1)
        e2 = self.cell2edges(nc2)
        for e in e1:
            if e in e2:
                return e
        raise Exception("Cells %d and %d don't share an edge"%(nc1,nc2))

    def build_index(self):
        if self.index is None:
            # assemble points into list of (id, [x x y y], None)
            if self.verbose > 1:
                print( "building point index")
            # old rtree required that stream inputs have non-interleaved coordinates,
            # but new rtree allows for interleaved coordinates all the time.
            # best solution probably to specify interleaved=False
            tuples = [(i,self.points[i,xxyy],None) for i in range(self.Npoints()) if np.isfinite(self.points[i,0]) ]
                
            self.index = Rtree(tuples,interleaved=False)
            if self.verbose > 1:
                print("done")
            
    def build_edge_index(self):
        if self.edge_index is None:
            print( "building edge index")
            ec = self.edge_centers()
            tuples = [(i,ec[i,xxyy],None) for i in range(self.Nedges())]
            self.edge_index = Rtree(tuples,interleaved=False)
            print( "done")

    def closest_point(self,p,count=1,boundary=0):
        """ Returns the count closest nodes to p
        boundary=1: only choose nodes on the boundary.
        """
        if boundary:
            # print "Searching for nearby boundary point"
            # this is slow, but I'm too lazy to add any sort of index specific to
            # boundary nodes.  Note that this will include interprocessor boundary
            # nodes, too.
            boundary_nodes = np.unique( self.edges[self.edges[:,2]>0,:2] )
            dists = np.sum( (p - self.points[boundary_nodes])**2, axis=1)
            order = np.argsort(dists)
            closest = boundary_nodes[ order[:count] ]
            # print "   done with boundary node search"
            
            if count == 1:
                return closest[0]
            else:
                return closest
        else:
            if self.index is None:
                self.build_index()

            p = np.array(p)

            # returns the index of the grid point closest to the given point:
            hits = self.index.nearest( p[xxyy], count)

            # newer versions of rtree return a generator:
            if isinstance( hits, types.GeneratorType):
                # so translate that into a list like we used to get.
                hits = [next(hits) for i in range(count)]
                        
            if count > 1:
                return hits
            else:
                return hits[0]

    def closest_edge(self,p):
        if self.edge_index is None:
            self.build_edge_index()
            
        hits = self.edge_index.nearest( p[xxyy], 1)
        
        # newer versions of rtree return a generator:
        if isinstance( hits, types.GeneratorType):
            # so translate that into a list like we used to get.
            return hits.next()
        else:
            return hits[0]

    
    def closest_cell(self,p,full=0,inside=False):
        """
        full=0: return None if the closest *point* is not in a cell on this subdomain
        full=1: exhaustively search all cells, even if the nearest point is not on this subdomain

        inside: require that the returned cell contains p, otherwise return None

        DANGER: this method is not robust!  in particular, a nearby but disconnected
        cell could have a vertex close to the query point, and we'd never see the
        right cell.
        """
        # rather than carry around another index, reuse the point index
        i = self.closest_point(p)
        try:
            cells = list( self.pnt2cells(i) )
        except KeyError:
            if not full:
                return None
            else:
                print( "This must be on a subdomain.  The best point wasn't in one of our cells")
                cells = range(self.Ncells())

        if inside:
            pnt = geometry.Point(p[0],p[1])
            for c in cells:
                tri = geometry.Polygon(self.points[self.cells[c]])
                if tri.contains(pnt):
                    return c
            return None
        else:
            cell_centers = self.vcenters()[cells]
            dists = ((p-cell_centers)**2).sum(axis=1)
            chosen = cells[np.argmin(dists)]

            dist = np.sqrt( ((p-self.vcenters()[chosen])**2).sum() )
            # print "Closest cell was %f [m] away"%dist
        return chosen
        
    def set_edge_markers(self,pnt1,pnt2,marker):
        """ Find the nodes closest to each of the two points,
        Search for the shortest path between them on the boundary.
        Set all of those edges' markers to marker
        """
        n1 = self.closest_point(pnt1)
        n2 = self.closest_point(pnt2)

        path = self.shortest_path(n1,n2,boundary_only=1)

        for i in range(len(path)-1):
            e = self.find_edge( path[i:i+2] )
            self.edges[e,2] = marker

    def shortest_path(self,n1,n2,boundary_only=0,max_cost = np.inf):
        """ dijkstra on the edge graph from n1 to n2
        boundary_only: limit search to edges on the boundary (have
        a -1 for cell2)
        """
        queue = pq.priorityDictionary()
        queue[n1] = 0

        done = {}

        while 1:
            # find the queue-member with the lowest cost:
            if len(queue)==0:
                return None # no way to get there from here.
            best = queue.smallest()
            best_cost = queue[best]
            if best_cost > max_cost:
                print( "Too far" )
                return None
            
            del queue[best]

            done[best] = best_cost

            if best == n2:
                # print "Found the ending point"
                break

            # figure out its neighbors
            # This used to use cells, but this query is valid even when there are no cells,
            # so don't rely on cells.
            #cells = list(self.pnt2cells(best))
            #all_points = unique( self.cells[cells] )
            edges = self.pnt2edges(best)
            all_points = np.unique( self.edges[edges,:2] )

            for p in all_points:
                if p in done:
                    # both for p and for points that we've already done
                    continue
                
                if boundary_only:
                    e = self.find_edge( (best,p) )
                    if self.edges[e,4] != BOUNDARY:
                        continue

                dist = np.sqrt( ((self.points[p] - self.points[best])**2).sum() )
                new_cost = best_cost + dist

                if p not in queue:
                    queue[p] = np.inf

                if queue[p] > new_cost:
                    queue[p] = new_cost

        # reconstruct the path:
        path = [n2]

        while 1:
            p = path[-1]
            if p == n1:
                break

            # figure out its neighbors
            edges = self.pnt2edges(p)
            all_points = np.unique( self.edges[edges,:2] )

            found_prev = 0
            for nbr in all_points:
                if nbr == p or nbr not in done:
                    continue

                dist = np.sqrt( ((self.points[p] - self.points[nbr])**2).sum() )

                if done[p] == done[nbr] + dist:
                    path.append(nbr)
                    found_prev = 1
                    break
            if not found_prev:
                return None

        return np.array( path[::-1] )

    def cells_on_line(self,xxyy):
        """ Return cells intersecting the given line segment
        cells are found based on having vertices which straddle
        the line, and cell centers which are within the segment's
        extent
        """
        m=np.array([ [xxyy[0],xxyy[2],1],
                  [xxyy[1],xxyy[3],1],
                  [1,1,1] ])
        b=np.array([0,0,abs(xxyy).mean()])
        line_eq=np.linalg.solve(m,b)

        hom_points=np.concatenate( (self.points,np.ones((self.Npoints(),1))),axis=1)

        pnt_above=np.dot(hom_points,line_eq)>0
        cell_sum=np.sum(pnt_above[self.cells],axis=1)
        straddle=np.nonzero((cell_sum>0)&(cell_sum<3))[0]

        # further limit that to the lateral range of the transect
        A=np.array([xxyy[0],xxyy[2]])
        B=np.array([xxyy[1],xxyy[3]])
        vec=B-A
        d_min=0
        d_max=np.norm(vec)
        vec/=d_max

        straddle_dists=(vec[None,:]*(self.vcenters()[straddle]-A)).sum(axis=1)
        on_line=(straddle_dists>=d_min)&(straddle_dists<=d_max)
        cells_on_line=straddle[on_line]
        return cells_on_line

    ### graph modification api calls

    def delete_node_and_merge(self,n):
        """ For a degree 2 node, remove it and make one edge out its two edges.
        this used to be in paver, but I don't think there is any reason it can't
        be here in trigrid.
        """
        edges = self.pnt2edges(n)
        
        if self.verbose > 1:
            print( "Deleting node %d, with edges %s"%(n,edges))

        if len(edges) == 2:
            if self.verbose > 1:
                print( "  deleting node %d, will merge edges %d and %d"%(n,edges[0],edges[1]))
            e = self.merge_edges( edges[0], edges[1] )
        elif len(edges) != 0:
            print("Trying to delete node",n)
            plt.annotate("del",self.points[n])
            print("Edges are:",self.edges[edges])
            
            raise Exception("Can't delete node with %d edges"%len(edges))

        edges = self.pnt2edges(n)

        if len(edges) != 0:
            print("Should have removed all edges to node %d, but there are still some"%n)
            
        self.delete_node(n)
        return e

    
    def unmerge_edges(self,e1,e2,e1data,e2data):
        self.edges[e1] = e1data
        self.edges[e2] = e2data

        # too lazy to do this right now, so to be safe just kill it
        self._pnt2edges = None
        
    def merge_edges(self,e1,e2):
        """ returns the id of the new edge, which for now will always be one of e1 and e2
        (and the other will have been deleted
        """
        if self.verbose > 1:
            print("Merging edges %d %d"%(e1,e2) )
            print(" edge %d: nodes %d %d"%(e1,self.edges[e1,0],self.edges[e1,1]))
            print(" edge %d: nodes %d %d"%(e2,self.edges[e2,0],self.edges[e2,1]))

        
        B = np.intersect1d( self.edges[e1,:2], self.edges[e2,:2] )[0]

        # try to keep ordering the same (not sure if this is necessary)
        if self.edges[e1,0] == B:
            e1,e2 = e2,e1

        # push the operation with the re-ordered edge nodes, so that we know (i.e.
        # live_dt knows) which of the edges is current, and which is being undeleted.
        self.push_op(self.unmerge_edges, e1, e2, self.edges[e1].copy(), self.edges[e2].copy() )
            

        # pick C from e2
        if self.edges[e2,0] == B:
            C = self.edges[e2,1]
        else:
            C = self.edges[e2,0]

        if self.edges[e1,0] == B:
            self.edges[e1,0] = C
            A = self.edges[e1,1]
        else:
            self.edges[e1,1] = C
            A = self.edges[e1,0]

        # print "  nodes are %d %d %d"%(A,B,C)

        # this removes e2 from _pnt2edges for B & C
        # because of mucking with the edge data, better to handle the
        # entire rollback in merge_edges
        self.delete_edge(e2,rollback=0)
        
        # fix up edge lookup tables:
        if self._pnt2edges is not None:
            self._pnt2edges[C].append(e1)

            # B is still listed for e1
            b_edges = self._pnt2edges[B]
            if b_edges != [e1]:
                print("Merging edges.  Remaining pnt2edges[B=%d] = "%B,b_edges ) 
                print("is not equal to e1 = ",[e1] )
            self._pnt2edges[B] = []

        # and callbacks:
        self.updated_edge(e1)
        return e1
        
    def undelete_node(self,i,p):
        self.points[i] = p

        if self.index is not None:
            self.index.insert(i, self.points[i,xxyy] )

    
    def delete_node(self,i,remove_edges=1):
        if self.verbose > 1:
            print("delete_node: %d, remove_edges=%s"%(i,remove_edges))
            
        if remove_edges:
            # make a copy so that as delete_edge modifies 
            # _pnt2edges we still have the original list
            nbr_edges = list(self.pnt2edges(i))

            for e in nbr_edges:
                self.delete_edge(e)

        self.push_op(self.undelete_node,i,self.points[i].copy())
            
        # nodes are marked as deleted by setting the x coordinate
        # to NaN, and remove from index
        if self.index is not None:
            coords = self.points[i,xxyy]
            self.index.delete(i, coords )
            
        self.points[i,0] = np.nan
        self.deleted_node(i)


    def undelete_cell(self,c,nodes,edge_updates):
        self.cells[c] = nodes
        self._vcenters = None # lazy...        

        for e,vals in edge_updates:
            self.edges[e] = vals
            
        if self._pnt2cells is not None: 
            for i in nodes:
                if i not in self._pnt2cells:
                    self._pnt2cells[i] = set()
                self._pnt2cells[i].add(c)
        
    def delete_cell(self,c,replace_with=-2,rollback=1):
        """ 
        replace_with: the value to set on edges that used to reference
        this cell.
        -2 => leave an internal hole
        -1 => create an 'island'
        """

        nA,nB,nC = self.cells[c]
        ab = self.find_edge([nA,nB])
        bc = self.find_edge([nB,nC])
        ca = self.find_edge([nC,nA])

        edge_updates = [ [ab,self.edges[ab].copy()],
                         [bc,self.edges[bc].copy()],
                         [ca,self.edges[ca].copy()] ]
                     
        self.push_op(self.undelete_cell,c,self.cells[c].copy(),edge_updates)

        for e in [ab,bc,ca]:
            if self.edges[e,3] == c:
                check = 3
            elif self.edges[e,4] == c:
                check = 4
            else:
                print( "Cell: %d  check on edge %d  with nbrs: %d %d"%(
                    c,e,self.edges[e,3],self.edges[e,4]) )
                
                raise Exception("Deleting cell, but edge has no reference to it")
            self.edges[e,check] = replace_with
            
            # optional - update edge marker, and for now just assume it will
            # be a land edge (other BC types are generally handled later anyway)
            if replace_with == -1:
                # print "Deleting cell and replace_with is",replace_with
                if self.edges[e,2] == 0:
                    # print "Internal edge becoming a land edge"
                    self.edges[e,2] = LAND_EDGE
                    self.updated_edge(e)
        
        self.cells[c,:] = -1
        if self._vcenters is not None:
            self._vcenters[c] = np.nan
            
        if self._pnt2cells is not None:
            for n in [nA,nB,nC]:
                self._pnt2cells[n].remove(c)
            
        self.deleted_cell(c)

    def undelete_edge(self,e,e_data):
        self.edges[e] = e_data

        # fix up indexes:
        if self._pnt2edges is not None:
            for n in self.edges[e,:2]:
                if n not in self._pnt2edges:
                    self._pnt2edges[n] = []
                self._pnt2edges[n].append(e)

        if self.edge_index is not None:
            coords = self.edge_centers()[e][xxyy]
            self.edge_index.insert(e,coords)
        
    def delete_edge(self,e,rollback=1):
        """ for now, just make it into a degenerate edge
        specify rollback=0 to skip recording the undo information
        """
        if self.verbose > 1:
            print( "Deleting edge %d:"%e)

        # remove any neighboring cells first
        cell_nbrs = self.edges[e,3:5]

        if any(cell_nbrs == -1):
            replace_with = -1
        else:
            replace_with = -2
        
        for c in cell_nbrs:
            if c >= 0:
                self.delete_cell(c,replace_with=replace_with,rollback=rollback)

        # clear out indexes
        if self._pnt2edges is not None:
            self._pnt2edges[self.edges[e,0]].remove(e)
            self._pnt2edges[self.edges[e,1]].remove(e)

        if self.edge_index is not None:
            coords = self.edge_centers()[e][xxyy]
            self.edge_index.delete(e,coords)

        if rollback:
            self.push_op(self.undelete_edge,e,self.edges[e].copy())
        
        # mark edge deleted
        self.edges[e,:2] = -37
        self.edges[e,2] = DELETED_EDGE # DELETED
        self.edges[e,3:5] = -37

        # signal to anyone who cares
        self.deleted_edge(e)

    def valid_edges(self):
        """ returns an array of indices for valid edges - i.e. not deleted"""
        return np.nonzero(self.edges[:,2]!=DELETED_EDGE)[0]
    
    def split_edge(self,nodeA,nodeB,nodeC):
        """ take the existing edge AC and insert node B in the middle of it

        nodeA: index to node on one end of the existing edge
        nodeB: (i) index to new new node in middle of edge,
               (ii) tuple (coords, dict for add_node options)
               may be extended to allow arbitrary options for point
        nodeC: index to node on other end of existing edge
        """
        e1 = self.find_edge([nodeA,nodeC])

        if isinstance(nodeB,tuple):
            pntB,pntBopts=nodeB
            nodeB=None
        else:
            pntB=self.points[nodeB]
                
        if any( self.edges[e1,3:5] >= 0 ):
            print( "While trying to split the edge %d (%d-%d) with node %s"%(e1,nodeA,nodeC,nodeB))
            plt.annotate(str(nodeA),self.points[nodeA])
            plt.annotate(str(nodeB),pntB)
            plt.annotate(str(nodeC),self.points[nodeC])
            print("The cell neighbors of the edge are:",self.edges[e1,3:5])
            raise Exception("You can't split an edge that already has cells")

        # 2011-01-29: this used to be in the opp. order - but that implies
        #   an invalid state
        self.push_op(self.unmodify_edge,e1,self.edges[e1].copy())

        # 2014-11-06: for a nodeB colinear with nodeA-nodeC, this is the
        #   more appropriate time to create nodeB
        if nodeB is None:
            nodeB=self.add_node(pntB,**pntBopts)

        self.push_op(self.unadd_edge,self.Nedges())
        
        self.edges = array_append( self.edges, self.edges[e1] )
        e2 = self.Nedges() - 1
        
        # first make the old edge from AC to AB
        if self.edges[e1,0] == nodeC:
            self.edges[e1,0] = nodeB
            self.edges[e2,1] = nodeB
        else:
            self.edges[e1,1] = nodeB
            self.edges[e2,0] = nodeB

        # handle updates to indices
        #   update pnt2edges
        if self._pnt2edges is not None:
            # nodeA is fine.
            # nodeB has to get both edges:
            self._pnt2edges[nodeB] = [e1,e2]
            # nodeC 
            i = self._pnt2edges[nodeC].index(e1)
            self._pnt2edges[nodeC][i] = e2
            

        self.updated_edge(e1)
        self.created_edge(e2)

        return e2


    def unadd_edge(self,old_length):
        #print "unadding edge %d"%old_length
        new_e = old_length

        if self._pnt2edges is not None:
            for n in self.edges[new_e,:2]:
                self._pnt2edges[n].remove(new_e)
        
        self.edges = self.edges[:old_length]

    def unmodify_edge(self, e, old_data):
        # print "unmodifying edge %d reverting to %s"%(e,old_data)
        if self._pnt2edges is not None:
            a,b = self.edges[e,:2]
            self._pnt2edges[a].remove(e)
            self._pnt2edges[b].remove(e)

            a,b = old_data[:2]
            self._pnt2edges[a].append(e)
            self._pnt2edges[b].append(e)
            
        self.edges[e] = old_data
            

    def add_edge(self,nodeA,nodeB,marker=0,cleft=-2,cright=-2,coerce_boundary=None):
        """ returns the number of the edge
        for cells that are marked -2, this will check to see if a new cell can
        be made on that side with other unmeshed edges
        """
        
        # print "trigrid: Adding an edge between %s and %s"%(nodeA,nodeB)

        try:
            e = self.find_edge([nodeA,nodeB])
            raise Exception("edge between %d and %d already exists"%(nodeA,nodeB))
        except NoSuchEdgeError:
            pass

        # dynamic resizing for edges:
        self.push_op(self.unadd_edge,len(self.edges))
        self.edges = array_append( self.edges, [nodeA,nodeB,marker,cleft,cright] )

        this_edge = self.Nedges()-1
        edge_ab = this_edge # for consistency in the mess of code below
        # print "This edge: ",this_edge

        self.cells_from_last_new_edge = []
        
        if cleft == -2 or cright == -2:
            # First get any candidates, based just on connectivity
            edges_from_a = self.pnt2edges(nodeA) 
            edges_from_b = self.pnt2edges(nodeB) 

            neighbors_from_a = np.setdiff1d( self.edges[edges_from_a,:2].ravel(), [nodeA,nodeB] )
            neighbors_from_b = np.setdiff1d( self.edges[edges_from_b,:2].ravel(), [nodeA,nodeB] )

            # nodes that are connected to both a and b
            candidates = np.intersect1d( neighbors_from_a, neighbors_from_b )

            if len(candidates) > 0:
                # is there a candidate on our right?
                ab = self.points[nodeB] - self.points[nodeA]
                ab_left = rot(np.pi/2,ab)

                new_cells = []

                for c in candidates:
                    ac = self.points[c] - self.points[nodeA]
                     
                    if np.dot(ac,ab_left) < 0: # this one is on the right of AB
                        # make a stand-in A & B that are in CCW order in this cell
                        ccwA,ccwB = nodeB,nodeA
                        check_cell_ab = 4 # the relevant cell for the new edge
                    else:
                        ccwA,ccwB = nodeA,nodeB
                        check_cell_ab = 3
                        
                        
                    edge_ac = self.find_edge((ccwA,c))
                    if self.edges[edge_ac,0] == ccwA:
                        # then the edge really is stored ac
                        check_cell_ac = 4
                    else:
                        check_cell_ac = 3

                    edge_bc = self.find_edge((ccwB,c))
                    if self.edges[edge_bc,0] == ccwB:
                        check_cell_bc = 3
                    else:
                        check_cell_bc = 4

                    # so now we have edge_ab, edge_ac, edge_bc as edge ids for the
                    # edges that make up a new cell, and corresponding check_cell_ab
                    # check_cell_ac and check_cell_bc that index the adj. cell that is
                    # facing into this new cell.

                    ccw_edges = [edge_ab,edge_bc,edge_ac]
                    check_cells = [check_cell_ab, check_cell_bc, check_cell_ac]
                    
                    adj_ids = [ self.edges[e,check]
                                for e,check in zip(ccw_edges,check_cells) ]
                    adj_ids = np.array( adj_ids )

                    if any(adj_ids >= 0) and any( adj_ids != adj_ids[0]):
                        # bad.  one edge thinks there is already a cell here, but
                        # the others doesn't agree.
                        print("During call to add_edge(nodeA=%d nodeB=%d marker=%d cleft=%d cright=%d coerce=%s"%(nodeA,
                                                                                                                  nodeB,
                                                                                                                  marker,
                                                                                                                  cleft,
                                                                                                                  cright,
                                                                                                                  coerce_boundary))
                        raise Exception("cell neighbor values for new cell using point %d are inconsistent: %s"%(c,adj_ids))
                    elif all(adj_ids == -1):
                        # leave them be, no new cell, all 3 edges are external
                        pass
                    elif coerce_boundary == -1:
                        # no new cell - everybody gets -1
                        self.edges[edge_ab,check_cell_ab] = -1
                        self.edges[edge_ac,check_cell_ac] = -1
                        self.edges[edge_bc,check_cell_bc] = -1
                    elif all( adj_ids == -2 ) or coerce_boundary == -2:
                        # make new cell, everybody gets that cell id
                        # Create the cell and get it's id:
                        new_cells.append( self.add_cell([ccwA,ccwB,c]) )

                        # update everybody's cell markers:
                        self.push_op(self.unmodify_edge, edge_ac, self.edges[edge_ac].copy() )
                        self.push_op(self.unmodify_edge, edge_bc, self.edges[edge_bc].copy() )
                        self.edges[edge_ac,check_cell_ac] = new_cells[-1]
                        self.edges[edge_bc,check_cell_bc] = new_cells[-1]
                        self.edges[edge_ab,check_cell_ab] = new_cells[-1]

                        # extend boundary - the fun one
                        # only when there was an external edge that now falls inside
                        # the new cell => mark the *other* side of the other edges to
                        # -1
                        if any(adj_ids==-1):
                            for i in range(3):
                                if adj_ids[i] == -2:
                                    # make its outside cell a -1
                                    # the 7-check gives us the outside cell nbr
                                    self.edges[ccw_edges[i],7-check_cells[i]] = -1
                                    # go ahead and set a closed edge, too
                                    self.edges[ccw_edges[i],2] = LAND_EDGE
                                else:
                                    # as long as this edge wasn't originally -1,-1
                                    # (which ought to be illegal), it's safe to say
                                    # that it is now internal
                                    self.edges[ccw_edges[i],2] = 0 # internal
                                # either way, let people know that markers have changed,
                                # but wait until later to signal on the new edge since
                                # it is not in the indices yet
                                if ccw_edges[i] != edge_ab:
                                    self.updated_edge(ccw_edges[i])

                self.cells_from_last_new_edge = new_cells

        # update pnt2edges
        if self._pnt2edges is not None:
            for n in [nodeA,nodeB]:
                if n not in self._pnt2edges:
                    self._pnt2edges[n] = []
                # print "Adding edge %d to list for node %d"%(this_edge,n)
                self._pnt2edges[n].append(this_edge)

        self.created_edge(this_edge)
                
        return this_edge

    def unadd_node(self,old_length):
        if self.index is not None:
            curr_len = len(self.points)
            for i in range(old_length,curr_len):
                coords = self.points[i,xxyy]
                self.index.delete(i, coords )
        
        self.points = self.points[:old_length]
        
    def add_node(self,P):
        P = P[:2]
        
        self.push_op(self.unadd_node,len(self.points))
        self.points = array_append( self.points, P )

        new_i = self.Npoints() - 1
        
        if self.index is not None:
            # print "Adding new node %d to index at "%new_i,self.points[new_i,xxyy]
            self.index.insert(new_i, self.points[new_i,xxyy] )

        self.created_node(new_i)
        
        return new_i

    def unadd_cell(self,old_length):
        # remove entries from _pnt2cells
        #  the cell that was added is at the end:
        if self._pnt2cells is not None:
            new_c = old_length
            for n in self.cells[new_c]:
                self._pnt2cells[n].remove(new_c)
        
        self.cells = self.cells[:old_length]
                                
    def add_cell(self,c):
        self.push_op(self.unadd_cell,len(self.cells))
        c = np.array(c,np.int32)
        
        i = np.array([0,1,2])
        ip = np.array([1,2,0])
        xi = self.points[c[i],0]
        yi = self.points[c[i],1]
        xip = self.points[c[ip],0]
        yip = self.points[c[ip],1]
        
        A = 0.5 * (xi*yip-xip*yi).sum()
        if A < 0:
            print( "WARNING: attempt to add CW cell.  Reversing")
            c = c[::-1]
        
        # self.cells = concatenate( (self.cells, [c]) )
        self.cells = array_append( self.cells, c )
        self._vcenters = None
        
        this_cell = self.Ncells() - 1

        if self._pnt2cells is not None: # could be smarter and actually update.
            for i in c:
                if i not in self._pnt2cells:
                    self._pnt2cells[i] = set()
                self._pnt2cells[i].add(this_cell)

        self.created_cell(this_cell)
                
        return this_cell


    def edges_to_rings(self, edgemask=None, ccw=1):
        """ using only the edges for which edgemask is true,
        construct rings.  if edgemask is not given, use all of the
        current edges

        if ccw is 1, only non-intersecting ccw rings will be return
        if ccw is 0, only non-intersecting cw rings will be return
        """
        if edgemask is not None:
            edges = self.edges[edgemask,:2]
            masked_grid = TriGrid(points=self.points,edges=edges)
            return masked_grid.edges_to_rings(edgemask=None)

        # remember which edges have already been assigned to a ring
        edges_used = np.zeros( self.Nedges(), np.int8 )

        rings = []
        
        for start_e in six.moves.range(self.Nedges()):
            if edges_used[start_e]:
                continue

            # start tracing with the given edge -
            # it's hard to know beforehand which side of this edge is facing into
            # the domain, so start with the assumption that it obeys our convention
            # that going from edge[i,0] to edge[i,1] the interior is to the left
            # once a ring has been constructed, check to see if it has negative area
            # in which case we repeat the process with the opposite ordering.

            # one problem, though, is that interior rings really should have negative
            # area, since they will become holes.

            # at least the one with the largest area is correct.
            # Then any that are inside it should have negative areas...
            
            # what if we found all rings with positive area and all with negative
            # area.  then we'd have all the information ready for choosing who is
            # inside whom, and which orientation is correct?

            failed_edges_used1 = None
            failed_edges_used2 = None
            
            for flip in [0,1]:
                e = start_e
                edges_used[e] = 1 # tentatively used.
                a,b = self.edges[e,:2]
                if flip:
                    a,b = b,a

                if self.verbose > 1:
                    print( "Starting ring trace with nodes ",a,b)
                ring = [a,b] # stores node indices

                node_count = 1
                while 1:
                    node_count += 1

                    # used to be node_count > self.Npoints(), but since we step
                    # one extra bit around the circle, then go back and remove one
                    # node, I think it should be 1+.
                    if node_count > 1+self.Npoints():
                        # debug
                        self.plot()
                        pnts = self.points[ring] 
                        plt.plot(pnts[:,0],pnts[:,1],'ro')
                        # /debug
                        raise Exception("Traced too far.  Something is wrong.  bailing")
                    
                    b_edges = self.pnt2edges(b)

                    if len(b_edges) == 2:
                        # easy case - one other edge leaves.
                        # new edge e
                        if b_edges[0] == e:
                            e = b_edges[1]
                        else:
                            e = b_edges[0]

                        # # setdiff1d isn't very fast...
                        # e = setdiff1d(b_edges,[e])[0]
                    else:
                        # calculate angles for all the edges, CCW relative to the
                        # x-axis (atan2 convention)
                        angles = []
                        for next_e in b_edges:
                            c = np.setdiff1d(self.edges[next_e,:2],[b])[0]
                            d = self.points[c] - self.points[b]
                            angles.append( np.arctan2(d[1],d[0]) )
                        angles = np.array(angles)
                        
                        e_idx = b_edges.index(e)
                        e_angle = angles[e_idx]
                        angles = (angles-e_angle) % (2*np.pi)
                        next_idx = np.argsort(angles)[-1]
                        e = b_edges[next_idx]
                        
                    # # setdiff1d is slow.  do this manually
                    # c = setdiff1d(self.edges[e,:2],[b])[0]
                    if self.edges[e,0] == b:
                        c = self.edges[e,1]
                    else:
                        c = self.edges[e,0]

                    # print " next node in trace: ",c

                    # now we have a new edge e, and the next node in the ring c
                    if edges_used[e] == 0:
                        edges_used[e] = 1 # mark as tentatively used
                    else:
                        # we guessed wrong, and now we should just bail and flip the
                        # other way but it's not so slow just to keep going and figure it
                        # out later.
                        # print "Could be smarter and abort now."
                        pass

                    if len(ring) >= 2 and b==ring[0] and c == ring[1]:
                        #print " %d,%d == %d,%d we've come full circle.  well done"%(b,c,
                        #                                                            ring[0],ring[1])
                        break
                    ring.append(c)
                    a,b = b,c

                # remove that last one where we figured out that we were really all the way
                # around.
                ring = ring[:-1]
                points = self.points[ring]
                if bool(ccw) == bool(is_ccw(points)):
                    # print "great, got correctly oriented ring (ccw=%s)"%ccw
                    edges_used[ edges_used==1 ] = 2 # really used
                    rings.append( np.array(ring) )
                    break # breaks out of the flip loop
                else:
                    # print "ring orientation wrong, wanted ccw=%s"%ccw
                    if flip:
                        area = signed_area(points)
                        
                        if self.verbose > 1:
                            print("Failed to get positive area either way:" )
                            print("Ring area is ",area )

                        if np.isnan(area):
                            print("Got nan area:" )
                            print(points)
                            raise Exception("NaN area trying to figure out rings")
                        
                        # raise Exception,"Failed to make positive area ring in either direction"
                        # I think this is actually valid - when Angel Island gets joined to
                        # Tiburon, if you start on Angel island either way you go you trace
                        # a region CCW.
                        
                        # however, nodes that were visited in both directions
                        # should 'probably' be marked so we don't visit them more.
                        # really not sure how this will fair with a multiple-bowtie
                        # issue...
                        failed_edges_used2 = np.where(edges_used==1)[0]
                        
                        edges_used[ np.intersect1d( failed_edges_used1,
                                                    failed_edges_used2 ) ] = 2
                    
                    # otherwise try again going the other direction,
                    # unmark edges, but remember them
                    failed_edges_used1 = np.where(edges_used==1)[0]
                    edges_used[ edges_used==1 ] = 0 # back into the pool

        if self.verbose > 0:
            print("Done creating rings: %d rings in total"%len(rings))
        return rings

    def edges_to_polygons(self,edgemask):
        """ use the edges (possibly masked by given edgemask) to create
        a shapely.geometry.Polygon() for each top-level polygon, ordered
        by decreasing area
        """
        rings_and_holes = self.edges_to_rings_and_holes(edgemask)

        polys = []
        for r,inner_rings in rings_and_holes:
            outer_points = self.points[r]
            inner_points = [self.points[ir] for ir in inner_rings]
            polys.append( geometry.Polygon( outer_points, inner_points) )

        areas = np.array([p.area for p in polys])
        order = np.argsort(-1 * areas)

        return [polys[i] for i in order]
        
        
    def edges_to_rings_and_holes(self,edgemask):
        """ using only the edges for which edgemask is true,
        construct polygons with holes.  if edgemask is not given, use all of the
        current edges

        This calls edges_to_rings to get both ccw rings and cw rings, and
        then determines which rings are inside which.

        returns a list [ [ outer_ring_nodes, [inner_ring1,inner_ring2,...]], ... ]
        """
        if edgemask is not None:
            edges = self.edges[edgemask,:2]
            masked_grid = TriGrid(points=self.points,edges=edges)
            return masked_grid.edges_to_rings_and_holes(edgemask=None)

        # print "calling edges_to_rings (ccw)"
        ccw_rings = self.edges_to_rings(ccw=1)
        # print "calling edges_to_rings (cw)"
        cw_rings  = self.edges_to_rings(ccw=0)

        # print "constructing polygons"
        # make single-ring polygons out of each:
        ccw_polys = [geometry.Polygon(self.points[r]) for r in ccw_rings]
        cw_polys  = [geometry.Polygon(self.points[r]) for r in cw_rings]

        # assume that the ccw poly with the largest area is the keeper.
        # technically we should consider all ccw polys that are not inside
        # any other poly
        ccw_areas = [p.area for p in ccw_polys]

        outer_rings = outermost_rings(ccw_polys)

        # Then for each outer_ring, search for cw_polys that fit inside it.
        outer_polys = []  # each outer polygon, followed by a list of its holes

        # print "finding the nesting order of polygons"
        for oi in outer_rings:
            outer_poly = ccw_polys[oi]

            # all cw_polys that are contained by this outer ring.
            # This is where the predicate error is happening -
            possible_children_i = []
            for i in range(len(cw_polys)):
                try:
                    if i!=oi and outer_poly.contains(cw_polys[i]):
                        if not cw_polys[i].contains(outer_poly):
                            possible_children_i.append(i)
                        else:
                            print("Whoa - narrowly escaped disaster with a congruent CW poly")
                except shapely.predicates.PredicateError:
                    print("Failed while comparing rings - try negative buffering")
                    d = np.sqrt(cw_polys[i].area)
                    inner_poly = cw_polys[i].buffer(-d*0.00001, 4)

                    if outer_poly.contains( inner_poly ):
                        possible_children_i.append(i)

            # the original list comprehension, but doesn't handle degenerate
            # case
            
            # possible_children_i = [i for i in range(len(cw_polys)) \
            #                        if outer_poly.contains( cw_polys[i] ) and i!=oi ]
                
            possible_children_poly = [cw_polys[i] for i in possible_children_i]

            # of the possible children, only the ones that are inside another child are
            # really ours.  outermost_rings will return indices into possible_children, so remap
            # those back to proper cw_poly indices to get children.
            children = [possible_children_i[j] for j in outermost_rings( possible_children_poly )]

            outer_polys.append( [ccw_rings[oi],
                                 [cw_rings[i] for i in children]] )
            
        return outer_polys

    def select_edges_by_polygon(self,poly):
        ecs=self.edge_centers()
        return np.nonzero( [poly.contains( geometry.Point(ec)) for ec in ecs] )[0]

    def trim_to_left(self, path):
        """ Given a path, trim all cells to the left of it.
        """
        # mark the cut edges:
        for i in range(len(path)-1):
            e = self.find_edge( path[i:i+2] )

            if self.edges[e,2] == 0 or self.edges[e,2] == CUT_EDGE:
                # record at least ones that is really cut, in case some of
                # of the cut edges are actually on the boundary
                cut_edge = (path[i],path[i+1],e)

                self.edges[e,2] = CUT_EDGE

        # choose the first cell, based on the last edge that was touched above:

        # the actual points:
        a = self.points[cut_edge[0]]
        b = self.points[cut_edge[1]]
        # the edge index
        edge = cut_edge[2]

        # the two cells that form this edge:
        cell1,cell2 = self.edges[edge,3:]
        other_point1 = np.setdiff1d( self.cells[cell1], cut_edge[:2] )[0]
        other_point2 = np.setdiff1d( self.cells[cell2], cut_edge[:2] )[0]

        parallel = (b-a)
        # manually rotate 90deg CCW
        bad = np.array([ -parallel[1],parallel[0]] )

        if np.dot(self.points[other_point1],bad) > np.dot(self.points[other_point2],bad):
            bad_cell = cell1
        else:
            bad_cell = cell2

        print("Deleting")
        self.recursive_delete(bad_cell)
        print("Renumbering")
        self.renumber()

    def recursive_delete(self,c,renumber = 1):
        del_count = 0
        to_delete = [c]
        

        # things the queue have not been processed at all...

        while len(to_delete) > 0:
            # grab somebody:
            c = to_delete.pop()
            if self.cells[c,0] == -1:
                continue

            # get their edges
            nodea,nodeb,nodec = self.cells[c]

            my_edges = [self.find_edge( (nodea,nodeb) ),
                        self.find_edge( (nodeb,nodec) ),
                        self.find_edge( (nodec,nodea) ) ]

            # mark it deleted:
            self.cells[c,0] = -1
            del_count += 1

            # add their neighbors to the queue to be processed:
            for e in my_edges:
                if self.edges[e,2] == 0:# only on non-cut, internal edges:
                    c1,c2 = self.edges[e,3:]
                    if c1 == c:
                        nbr = c2
                    else:
                        nbr = c1

                    if nbr >= 0:
                        to_delete.append(nbr)
        print("Deleted %i cells"%del_count)


    # ## Experimental stitching - started to backport from trigrid2.py, not
    # ## complete.  See paver.py:splice_in_grid
    # @staticmethod
    # def stitch_grids(grids,use_envelope=False,envelope_tol=0.01,join_tolerance=0.25):
    #     """ grids: an iterable of TriGrid instances
    #     combines all grids together, removing duplicate points, joining coincident vertices
    # 
    #     use_envelope: use the rectangular bounding box of each grid to determine joinable
    #       nodes.   
    #     envelope_tol: if using the grid bounds to determine joinable leaf nodes, the tolerance
    #       for determining that a node does lie on the boundary.
    #     join_tolerance: leaf nodes from adjacent grids within this distance range will be
    #       considered coincident, and joined.
    #       
    #     """
    #     # for each grid, an array of node indices which will be considered
    #     all_leaves = []
    # 
    #     accum_grid = None
    # 
    #     for i,gridB in enumerate(grids):
    #         if i % 100 == 0:
    #             print "%d / %d"%(i,len(grids)-1)
    # 
    #         if gridB.Npoints() == 0:
    #             print "empty"
    #             continue
    # 
    #         gridB.verbose = 0
    #         gridB.renumber()
    # 
    #         Bleaves = array(gridB.leaf_nodes(use_envelope=use_envelope,
    #                                          tolerance=envelope_tol,
    #                                          use_degree=False),
    #                         np.int32)
    # 
    #         if i == 0:
    #             accum_grid = gridB
    #             if len(Bleaves):
    #                 all_leaves.append( Bleaves )
    #         else:
    #             accum_grid.append_grid(gridB)
    #             if len(Bleaves):
    #                 all_leaves.append( Bleaves + gridB.node_offset)
    # 
    #     all_leaves = concatenate(all_leaves)
    # 
    #     # build an index of the leaf nodes to speed up joining
    #     lf = field.XYZField(X=accum_grid.nodes['x'][all_leaves],
    #                         F=all_leaves)
    #     lf.build_index()
    # 
    #     to_join=[] # [ (i,j), ...] , with i<j, and i,j indexes into accum_grid.nodes
    #     for i,l in enumerate(all_leaves):
    #         nbrs = lf.nearest(lf.X[i],count=4)
    # 
    #         for nbr in nbrs:
    #             if nbr <= i:
    #                 continue
    #             dist = norm(lf.X[i] - lf.X[nbr])
    #             if dist < join_tolerance:
    #                 print "Joining with distance ",dist
    #                 to_join.append( (all_leaves[i], all_leaves[nbr]) )
    # 
    #     # okay - so need to allow for multiple joins with a single node.
    #     # done - but is the joining code going to handle that okay?
    # 
    #     _remapped = {} # for joined nodes, track who they became
    #     def canonicalize(n): # recursively resolve remapped nodes
    #         while _remapped.has_key(n):
    #             n = _remapped[n]
    #         return n
    # 
    #     for a,b in to_join:
    #         a = canonicalize(a)
    #         b = canonicalize(b)
    #         if a==b:
    #             continue
    # 
    #         a_nbrs = accum_grid.node_neighbors(a)
    #         accum_grid.delete_node(a)
    #         for a_nbr in a_nbrs:
    #             # with edges along a boundary, it's possible that
    #             # the new edge already exists
    #             try:
    #                 accum_grid.nodes_to_edge( [a_nbr,b] )
    #             except NoSuchEdgeError:
    #                 accum_grid.add_edge(a_nbr,b)
    #     return accum_grid


    # Undo-history management - very generic.
    op_stack_serial = 10
    op_stack = None
    def checkpoint(self):
        if self.op_stack is None:
            self.op_stack_serial += 1
            self.op_stack = []
        return self.op_stack_serial,len(self.op_stack)

    def revert(self,cp):
        serial,frame = cp
        if serial != self.op_stack_serial:
            raise ValueError("The current op stack has serial %d, but your checkpoint is %s"%(self.op_stack_serial,
                                                                                              serial))
        while len(self.op_stack) > frame:
            self.pop_op()

    def commit(self):
        self.op_stack = None
        self.op_stack_serial += 1
    
    def push_op(self,meth,*data,**kwdata):
        if self.op_stack is not None:
            self.op_stack.append( (meth,data,kwdata) )

    def pop_op(self):
        f = self.op_stack.pop()
        if self.verbose > 3:
            print("popping: ",f)
        meth = f[0]
        args = f[1]
        kwargs = f[2]
        
        meth(*args,**kwargs)
    ###
        
    def unmove_node(self,i,orig_val):
        # update point index:
        if self.index is not None:
            curr_coords = self.points[i,xxyy]
            orig_coords = orig_val[xxyy]
            
            self.index.delete(i, curr_coords )
            self.index.insert(i, orig_coords )

        self.points[i] = orig_val

    def move_node(self,i,new_pnt):
        self.push_op(self.unmove_node,i,self.points[i].copy())
                     
        # update point index:
        if self.index is not None:
            old_coords = self.points[i,xxyy]
            new_coords = new_pnt[xxyy]
            self.index.delete(i, old_coords )
            self.index.insert(i, new_coords )

        self.points[i] = new_pnt

        self.updated_node(i)
        
        for e in self.pnt2edges(i):
            new_ec = self.points[self.edges[e,:2]].mean(axis=0)
            
            if self._edge_centers is not None:
                old_ec = self._edge_centers[e]
                self._edge_centers[e] = new_ec
            if self.edge_index is not None:
                self.edge_index.delete(e,old_ec[xxyy])
                self.edge_index.insert(e,new_ec[xxyy])
            
            self.updated_edge(e)

    def updated_node(self,i):
        for cb in self._update_node_listeners.values():
            cb(i)

    def updated_edge(self,e):
        for cb in self._update_edge_listeners.values():
            cb(e)

    def updated_cell(self,c):
        for cb in self._update_cell_listeners.values():
            cb(c)

    def created_node(self,i):
        for cb in self._create_node_listeners.values():
            cb(i)

    def created_edge(self,e):
        # fix up the edge index
        ec = self.points[self.edges[e,:2]].mean(axis=0)
        
        if self._edge_centers is not None:
            if e != len(self._edge_centers):
                # ideally should know where this is getting out of sync and
                # fix it there.
                print("Edge centers is out of sync.  clearing it.")
                self._edge_centers = None
                self.edge_index = None
            else:
                self._edge_centers = array_append(self._edge_centers,ec)

        if self.edge_index is not None:
            print("edge_index: inserting new edge center %i %s"%(e,ec))
            self.edge_index.insert( e, ec[xxyy] )
            
        for cb in self._create_edge_listeners.values():
            cb(e)

    def created_cell(self,c):
        for cb in self._create_cell_listeners.values():
            cb(c)

    def deleted_cell(self,c):
        for cb in self._delete_cell_listeners.values():
            cb(c)

    def deleted_node(self,i):
        for cb in self._delete_node_listeners.values():
            cb(i)
            
    def deleted_edge(self,e):
        for cb in self._delete_edge_listeners.values():
            cb(e)
        

    # subscriber interface for updates:
    listener_count = 0
    def init_listeners(self):
        self._update_node_listeners = {}
        self._update_edge_listeners = {}
        self._update_cell_listeners = {}
        self._create_node_listeners = {}
        self._create_edge_listeners = {}
        self._create_cell_listeners = {}
        self._delete_node_listeners = {}
        self._delete_edge_listeners = {}
        self._delete_cell_listeners = {}
    
    def listen(self,event,cb):
        cb_id = self.listener_count
        if event == 'update_node':
            self._update_node_listeners[cb_id] = cb
        elif event == 'update_edge':
            self._update_edge_listeners[cb_id] = cb
        elif event == 'update_cell':
            self._update_cell_listeners[cb_id] = cb
        elif event == 'create_node':
            self._create_node_listeners[cb_id] = cb
        elif event == 'create_edge':
            self._create_edge_listeners[cb_id] = cb
        elif event == 'delete_node':
            self._delete_node_listeners[cb_id] = cb
        elif event == 'delete_edge':
            self._delete_edge_listeners[cb_id] = cb
        elif event == 'delete_cell':
            self._delete_cell_listeners[cb_id] = cb
        else:
            raise Exception("unknown event %s"%event)
            
        self.listener_count += 1
        return cb_id

    def unlisten(self,cb_id):
        for l in [ self._update_node_listeners,
                   self._update_edge_listeners,
                   self._update_cell_listeners,
                   self._create_node_listeners,
                   self._create_edge_listeners,
                   self._create_cell_listeners,
                   self._delete_node_listeners,
                   self._delete_edge_listeners,
                   self._delete_cell_listeners]:
            if cb_id in l:
                del l[cb_id]
                return
        print("Failed to remove cb_id %d"%cb_id)

        


