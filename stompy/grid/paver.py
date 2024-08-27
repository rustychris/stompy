from __future__ import print_function

import logging
log=logging.getLogger('stompy.grid.paver')

import sys, os

import pickle

import pdb

import numpy as np
from numpy.linalg import norm
from ..utils import array_append

from scipy import optimize as opt

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from shapely import geometry as geo
from shapely import wkb 

try:
    from osgeo import ogr
except ImportError:
    import ogr
    
import stompy.priority_queue as pq

from stompy.spatial import field

from .smoother import remove_small_islands, adjust_scale, apollonius_scale

from ..spatial.linestring_utils import upsample_linearring,downsample_linearring,resample_linearring

# if CGAL is unavailable, this will fall back to OrthoMaker
from . import live_dt
paving_base = live_dt.LiveDtGrid

from .paver_opt_mixin import OptimizeGridMixin

def my_fmin(f,x0,args=(),xtol=1e-4,disp=0):
    # The original, basic fmin -
    # uses some sort of simplex algorithm, and tends to find
    # pseudo-global minima
    return opt.fmin(f,x0,args=args,xtol=xtol,disp=disp)
    #  in at least one case, this doesn't terminate...
    # return opt.fmin_cg(f,x0,args=args,gtol=xtol,disp=disp)

from . import trigrid
from . import orthomaker as om

from .trigrid import rot,ensure_ccw,ensure_cw,is_ccw

class StrategyFailed(Exception):
    pass

class FillFailed(Exception):
    pass


# Try wrapping up linked-list elements into a class so we can
# define a hash function and put them into the
def CIter_expand(clist,d_prv,d,d_nxt):
    return clist.find_iter(d_prv,d,d_nxt)
CIter_expand.__safe_for_unpickling__ = True

class CIter(object):
    def __init__(self,data,prv,nxt,clist):
        self.data = data
        self.prv = prv
        self.nxt = nxt
        self.clist = clist

    def append(self,*args,**kwargs):
        """ proxies to clist append """
        return self.clist.append(*args,after=self,**kwargs)
    def prepend(self,*args,**kwargs):
        return self.clist.prepend(*args,before=self,**kwargs)

    def __str__(self):
        return "[%d-%d-%d]"%(self.prv.data,self.data,self.nxt.data)

    ### Pickle API
    # really we want the clist to exist, and then we just need to pick
    # the right CIter out of the clist.
    def __getstate__(self):
        # have to break the recursive cycle, so no references to other
        # iters
        return

    def __setstate__(self,d):
        self.data = d[1]

    def __reduce__(self):
        if self.prv:
            pdata = self.prv.data
        else:
            pdata = None
        if self.nxt:
            ndata = self.nxt.data
        else:
            ndata = None

        return (CIter_expand,(self.clist,pdata,self.data,ndata))

    def __lt__(self,other):
        """these instances are put in a priority_queue, which in python 3
        will break ties by comparing not just the key but also the value (self).
        """
        return id(self)<id(other)

    def __le__(self,other):
        return id(self)<=id(other)


class CList(object):
    """ Keep track of a linked list of nodes on the boundary of the paved region of
    a Paving.

    dynamically tracks the tightest angle, for picking the next node to fill.
    """

    # if true, then metrics for neighbors will be zeroed on insert and delete
    # of an element.
    
    clear_neighbor_metrics = 1
    
    def __init__(self):
        self.clear()

    def prepend(self,d,before,metric=0.0):
        return self.append(d,after=self.prv(before),metric=metric)
        
    def append(self,d,after=None,metric=0.0):
        """ Insert a new element with data d after the given element
        (defaults to inserting just before head)
        
        clear_neighbor_metrics: zero out the metrics for both neighbors of
        the new element
        """
        # data object, next, prev
        #   so that elt[-1] is the prev pointer and elt[1] is the next pointer.

        new_elt = CIter(d,None,None,self)
        
        if self.head is None:
            if after:
                raise Exception("You specified after, but the list is empty")
            
            self.head = new_elt
            new_elt.prv = new_elt
            new_elt.nxt = new_elt
        else:
            if after is None:
                after = self.head.prv
            elif not isinstance(after,CIter):
                raise Exception("after was specified but isn't a CIter")

            before = after.nxt
            
            after.nxt = new_elt
            new_elt.prv = after

            before.prv = new_elt
            new_elt.nxt = before

        # update related datastructures
        self.heap[new_elt] = metric
        
        if self.clear_neighbor_metrics and new_elt.nxt != new_elt:
            self.heap[new_elt.prv] = 0.0
            self.heap[new_elt.nxt] = 0.0

        if d not in self.node_to_iters:
            self.node_to_iters[d] = []
        self.node_to_iters[d].append(new_elt)
        
        self.count += 1
        return new_elt
        
    def __len__(self):
        return self.count
    
    def __contains__(self,node):
        node = int(node)
        
        return node in self.node_to_iters
    
    def iter_smallest_metric(self):
        return self.heap.smallest()

    def nxt(self,it):
        """ return next iter"""
        return it.nxt
    
    def prv(self,it):
        return it.prv

    def update_metric(self,it,metric):
        self.heap[it] = metric

    def metric(self,it):
        return self.heap[it]

    def to_array(self,return_weights=False):
        """ traverse the ring and return the data values in an array
        if return weights is true, returns a second array of the values
        in the heap
        """
        a = np.zeros(len(self),np.int32)
        if return_weights:
            w = np.zeros(len(self),np.float64)

        trav = self.head
        for i in range(len(self)):
            a[i] = trav.data
            if return_weights:
                w[i] = self.metric(trav)
            trav = trav.nxt
        if trav != self.head:
            raise Exception("Thought we were making the full loop, but ended up elsewhere")
        
        if return_weights:
            return a,w
        else:
            return a
    
    def clear(self):
        """ remove all elements """
        self.count = 0
        self.head = None
        self.heap = pq.priorityDictionary()
        self.node_to_iters = {}

    ### Pickle API
    # CList causes the regular pickle approach to crash with too many recursive calls
    def __getstate__(self):
        d_array,w_array = self.to_array(return_weights=True)
        return {'data':d_array,
                'weights':w_array}

    def __setstate__(self,d):
        d_array = d['data']
        w_array = d['weights']

        self.clear()
        for i in range(len(d_array)):
            self.append(d_array[i],metric=w_array[i])

    def find_iter(self,d_prv,d,d_nxt):
        """ Used for unpickling CIters - return the iter that matches the given
        triple of nodes
        """
        if d in self.node_to_iters:
            for it in self.node_to_iters[d]:
                if it.prv.data == d_prv and it.nxt.data == d_nxt:
                    return it
            raise Exception("Iter %d-%d-%d not found in CList"%(d_prv,d,d_nxt))
        else:
            return CIter(-1,None,None,"STALE")

    ###
            

    def fwd_edge_iter(self,start_elt=None):
        if start_elt is None:
            start_elt = self.head

        trav = start_elt
        
        while 1:
            yield trav.data,trav.nxt.data
            
            trav = trav.nxt
            
            if trav == start_elt:
                break

    def remove_nodes(self,*nodes):
        for n in nodes:
            self.remove_iters(*self.node_to_iters[n])

    def unremove_iter(self,elt,metric=0.0):
        """ paste an existing iter back into the list using
        its prv and nxt pointers
        note that this will not restore the head pointer unless
        this is the only element in the list"""

        # sanity check:
        if elt.prv.nxt != elt.nxt:
            raise Exception("Tried to unremove_iter, but it's state is inconsistent with the current list")
        if elt.clist != self:
            raise Exception("Tried to unremove_iter, but it belongs to a different clist")

        #  print "re-inserting an iter.  hold on"
        elt.prv.nxt = elt
        elt.nxt.prv = elt

        if self.head is None:
            self.head = elt

        self.count += 1
        
        # update related datastructures
        self.heap[elt] = metric

        d=elt.data
        if d not in self.node_to_iters:
            self.node_to_iters[d] = []
        self.node_to_iters[d].append(elt)
        
        
    def remove_iters(self,*iters):
        for elt in iters:
            # Fix up the mapping first
            d = elt.data
            self.node_to_iters[d].remove(elt)
            
            if len(self.node_to_iters[d]) == 0:
                del self.node_to_iters[d]

            # Remove us from the linked list
            elt.nxt.prv = elt.prv
            elt.prv.nxt = elt.nxt
            self.count -= 1

            # Take care of self.head
            if self.count == 0:
                self.clear()
                return
            if self.head == elt:
                self.head = elt.nxt

            if self.clear_neighbor_metrics:
                self.heap[elt.prv] = 0.0
                self.heap[elt.nxt] = 0.0

            # and update extra datastructures
            del self.heap[elt]
            
def line_eq(pa,pb):
    """ returns coefficients a,b,c that describe the
    line through pa and pb as ax+by+c=0
    """
    d = pa-pb

    # careful not to divide them -
    # whichever is smaller in abs. value, dx or dy, make the corresponding
    # coefficient 1, then solve for the other as

    # 
    # b = -a*dx/dy

    if abs(d[0]) > abs(d[1]): # dx is greater
        b = 1
        a = -b*d[1]/d[0]
    else:
        a = 1
        b = -a*d[0]/d[1]

    # then solve for c using one of the input points
    c = -(a*pa[0] + b*pa[1])
    return a,b,c


def point_in_triangle(pnt,tri):
    """ Return true if pnt is on the triangle tri (in its interior or on one
    of its edges)  tri must be CCW ordered.
    """
    # Given a triangle with CCW vertices, return true if the given point is inside
    # the triangle
    # 
    
    areas = []

    all_points = np.concatenate( ([pnt],tri) )


    tris = [[0,1,2],
            [0,2,3],
            [0,3,1]]
    
    i = np.arange(3)
    ip1 = (i+1)%3

    for t in tris:
        points = all_points[t]
        areas.append( 0.5*(points[i,0]*points[ip1,1] - points[ip1,0]*points[i,1]).sum() )
    return min(areas) >= 0.0


def one_point_cost(pnt,edges,target_length=5.0):
    """
    pnt is intended to complete a triangle with each
    pair of points in edges, and should be to the left
    of each edge
    """
    penalty = 0
    
    max_angle = 85.0*np.pi/180.

    # all_edges[triangle_i,{ab,bc,ca},{x,y}]
    all_edges = np.zeros( (edges.shape[0], 3 ,2), np.float64 )
    
    # get the edges:
    all_edges[:,0,:] = edges[:,0] - pnt  # ab
    all_edges[:,1,:] = edges[:,1] - edges[:,0] # bc
    all_edges[:,2,:] = pnt - edges[:,1] # ca

    i = np.arange(3)
    im1 = (i-1)%3
    
    ## cost based on angle:
    abs_angles = np.arctan2( all_edges[:,:,1], all_edges[:,:,0] )
    all_angles = (np.pi - (abs_angles[:,i] - abs_angles[:,im1]) % (2*np.pi)) % (2*np.pi)
        
    # a_angles = (pi - (ab_angles - ca_angles) % (2*pi)) % (2*pi)
    # b_angles = (pi - (bc_angles - ab_angles) % (2*pi)) % (2*pi)
    # c_angles = (pi - (ca_angles - bc_angles) % (2*pi)) % (2*pi)

    if 1:
        # 60 is what it's been for a while, but I think in one situation
        # this put too much weight on small angles.
        # tried considering just large angles, but that quickly blew up.
        # even just changing this to 50 still blows up.
        #  how about a small tweak - s/60/58/ ??
        worst_angle = abs(all_angles - 60*np.pi/180.).max() 
        alpha = worst_angle /(max_angle - 60*np.pi/180.0)

        # 10**alpha: edges got very short...
        # 5**alpha - 1: closer, but overall still short edges.
        # alpha**5: angles look kind of bad
        angle_penalty = 10*alpha**5

        # Seems like it doesn't try hard enough to get rid of almost bad angles.
        # in one case, there is a small angle of 33.86 degrees, and another angle
        # of 84.26 degrees. so the cost function only cares about the small angle
        # because it is slightly more deviant from 60deg, but we may be in a cell
        # where the only freedom we have is to change the larger angles.

        # so add this in:
        if 1:
            # extra exponential penalty for nearly bad triangles:
            # These values mean that 3 degrees before the triangle is invalid
            # the exponential cuts in and will add a factor of e by the time the
            # triangles is invalid.

            scale_rad = 3.0*np.pi/180. # radians - e-folding scale of the cost
            # max_angle - 2.0*scale_rad works..
            thresh = max_angle - 1.0*scale_rad # angle at which the exponential 'cuts in'
            big_angle_penalty = np.exp( (all_angles.max() - thresh) / scale_rad)
    else:
        alphas = (all_angles - 60*np.pi/180.) / (max_angle - 60*np.pi/180.)
        alphas = 10*alphas**4
        angle_penalty = alphas.sum()
    
    penalty += angle_penalty + big_angle_penalty

    ## Length penalties:
    ab_lens = (all_edges[:,0,:]**2).sum(axis=1)
    ca_lens = (all_edges[:,2,:]**2).sum(axis=1)

    if 1:  # the usual..
        min_len = min(ab_lens.min(),ca_lens.min())
        max_len = max(ab_lens.min(),ca_lens.min())

        undershoot = target_length**2 / min_len
        overshoot  = max_len / target_length**2

        length_penalty = 0

        length_factor = 2
        length_penalty += length_factor*(max(undershoot,1) - 1)
        length_penalty += length_factor*(max(overshoot,1) - 1)
    elif 1:
        # Try an exponential
        rel_len_ab = ab_lens / target_length**2
        rel_len_ca = ca_lens / target_length**2

        # So we want to severely penalize things that are more than double
        # the target length or less than half the target length.
        # just a wild guess here, that maybe the threshold needs to be larger.
        # well, how about the penalty kicks in at 3x
        thresh = 9.0 # 2.5*2.5
        length_penalty = np.exp( rel_len_ab - thresh ).sum() + np.exp( 1.0/rel_len_ab - thresh).sum()
        length_penalty += np.exp( rel_len_ca - thresh ).sum() + np.exp( 1.0/rel_len_ca - thresh).sum()

    else:
        rel_errs_ab = (ab_lens - target_length**2) / target_length**2
        rel_errs_ca = (ca_lens - target_length**2) / target_length**2

        length_penalty = ( (rel_errs_ab**2).sum() + (rel_errs_ca**2).sum() )
        
    penalty += length_penalty

    # print "angle_penalty=%g  big_angle(%g)=%g length_penalty=%g"%(angle_penalty,biggest_angle,big_angle_penalty,length_penalty)
        
    return penalty

        


def compute_ring_normals(ring,closed=1):
    """ Given a closed ring (unless closed=0) (but unrepeated node) of points in ring,
    return a similar structure, but has unit vectors perpendicular
    (positive to left of the boundary)
    """
    i = np.arange(len(ring))
    ip1 = (i+1)%len(ring)
    im1 = (i-1)%len(ring)

    diffs = ring[ip1] - ring[i]
    # rotation matrix for angle=pi/2
    R = np.array( [[0,1],[-1,0]] )
    
    seg_normals = np.dot(diffs,R)
    
    mags = np.sqrt( np.sum(seg_normals**2, axis=1) )
    seg_unit_normals = seg_normals / mags[:,None]

    # now seg_unit_normals[i] is the normal for the edge between
    # i and ip1
    
    vertex_normals = seg_unit_normals[im1] + seg_unit_normals[i]
    
    # with open, degenerate lines, it is possible to wrap around the
    # end of a line, and vertex_normals will be 0.
    degen = np.all(vertex_normals == 0,axis=1)
    vertex_normals[degen] = seg_unit_normals[im1][degen]

    mags = np.sqrt( np.sum(vertex_normals**2, axis=1) )
    vertex_unit_normals = vertex_normals / mags[:,None]
    
    return vertex_unit_normals


def intersect_lines( AB, CD ):
    """ Given a pair of line segments
    return the point describing their intersection.

    NOTE: this is not guaranteed to behave nicely when AB and CD
    don't intersect.
    """
    AB = geo.LineString( AB )
    CD = geo.LineString( CD )

    return array( AB.intersection( CD ) )

    
class Paving(paving_base,OptimizeGridMixin):
    """  Tracks state as paving progresses.
    """
    # constants:
    # Node data fields:
    STAT = 0
    ORIG_RING = 1 # only set for nodes on the original_ring
    ALPHA = 2
    BETA = 3

    # Node statuses:
    HINT = 555  # node is on boundary, but just a hint of where the boundary is
    SLIDE = 666  #  attached, but allowed to move along boundary
    FREE =  777  # internal node, can move in 2D
    RIGID = 888 #  may not be moved
    DELETED = 999

    # Beta rules:
    BETA_NEVER=0
    BETA_RESCUE=1
    BETA_ALWAYS=2

    # Nonlocal methods:
    PROACTIVE_DELAUNAY=1
    SHEEPISH=2
    
    # subject to tuning...
    # 0.22 is just high enough to even out angle creep from
    # channel ends.
    cost_threshold = 0.22
    dyn_zoom = 0
    label = None # convenience for keeping track of tests

    # how many times to try relaxing the neighborhood around edits
    relaxation_iterations=4
    
    beta_rule = BETA_RESCUE # BETA_NEVER, BETA_RESCUE, BETA_ALWAYS
    
    nonlocal_method = PROACTIVE_DELAUNAY

    # the cost when sliding a boundary node with beta enabled will be multiplied
    # by exp( beta_cost_factor * beta / local_scale )
    # so when beta= local_scale/beta_cost_factor, the exponential kicks in.
    beta_cost_factor = 4.0

    # if true, non-end nodes on internal lines will be initialized as HINT nodes,
    # otherwise the internal line will be resampled at local densities and set to
    # RIGID.
    slide_internal_guides = 1

    def __init__(self,rings=None,density=None,
                 shp=None,geom=None,min_density=None,
                 initial_node_status=HINT,resample=None,degenerates=[],
                 label=None,slide_internal_guides = 1,
                 **kwargs):
        """ rings:
              a list of Ni*2 arrays where the first is the outermost exterior ring
                and all others are interior rings

            density: a DensityField object that defines nominal edge-length

            shp: path to a shapefile with the shoreline as one polygon (possibly with holes)
               exactly one of rings or shp must be specified

            geom: a shapely polygon object for initializing the shoreline

            min_density: an alternate field which is used to remove triangles from the delaunay
              triangulation of the shoreline.  if not specified, density will be used for this
              purpose. 
            
            initial_node_status: whether nodes in rings() can be moved.
               either HINT (can be moved) or RIGID (may not be moved)

            degenerates: a list of point arrays that are degenerate (zero area)
              curves.  For now, these are assumed to be sampled properly, and
              will be marked as rigid (soon to change to SLIDE, right?).

        if points are not domain-to-the-left, they will be reversed.
        """
        super(Paving,self).__init__(**kwargs)

        if resample == 'upsample' and initial_node_status == self.HINT:
            print("Taking 'upsample' strategy to mean initial nodes are rigid")
            initial_node_status = self.RIGID
            
        self.label = label
        self.slide_internal_guides = slide_internal_guides
        
        self.density = density
        if min_density is not None:
            self.min_density = min_density
        else:
            self.min_density = density

        self.step = 0

        self.plot_history = []
        
        self.hopeless_cells = np.array([])
        self.updated_cells = []
        
        self._vcenters = None

        # if we're given a shapefile, load it into rings now:
        if shp is not None:
            if rings is not None:
                raise Exception("exactly one of rings and shp must be specified")
            rings = self.shp_to_rings(shp)
        elif geom is not None:
            rings = self.geom_to_rings(geom)

        self.initial_node_status = initial_node_status
        self.degenerate_rings = [] # ids into original_rings for degenerate rings.

        if rings is not None:
            self.cells = np.zeros( (0,3), np.int32 )
            self.initialize_rings(rings,degenerates)
        else:
            # probably loaded a pre-existing grid
            self.initialize_from_existing_grid()
        
    def initialize_from_existing_grid(self):
        """ Try to piece together paving data from only a pre-existing suntans grid.
        """

        # This is mainly for doctoring up an existing grid -
        # without any extra information, we assume that boundary nodes are RIGID and
        # everybody else is FREE.  This can be doctored up afterwards by 
        # self.infer_original_rings()
        self.degenerate_rings = []

        # self.freeze=1            

        # self.initialize_boundaries()
        self.poly = "N/A"

        self.node_data = np.zeros( (self.Npoints(),4), np.float64 )

        # internal nodes are easy:
        self.node_data[:,self.STAT] = self.FREE
        self.node_data[:,self.ORIG_RING] = -1
        self.node_data[:,self.ALPHA] = np.nan
        self.node_data[:,self.BETA] = np.nan

        # boundary nodes -
        rings_and_children = self.edges_to_rings_and_holes(edgemask=(self.edges[:,4] == -1))
        if len(rings_and_children) != 1:
            print("Loading existing grid - expected edges_to_rings_and_holes to return exactly 1, got %d"%len(rings_and_children))
            raise Exception("Bad result for rings_and_holes")
        
        node_rings = [rings_and_children[0][0]] + rings_and_children[0][1]

        for i in range(len(node_rings)):
            for j in range(len(node_rings[i])):
                n = node_rings[i][j]
                self.node_data[n,self.STAT] = self.RIGID
                self.node_data[n,self.ORIG_RING] = i # even though they are really from many rings
                self.node_data[:,self.ALPHA] = j # 
                self.node_data[:,self.BETA] = 0 # should be safe.

        # And translate rings to actual coordinates, rather than node indices
        self.rings = [self.points[r] for r in node_rings]
        
        self.original_rings = [r.copy() for r in self.rings]
        self.oring_normals = [compute_ring_normals(r) for r in self.original_rings]

        self.unpaved = []
        
        # edge_data:
        #   0: step at which the edge was created
        #   1: original ring to which the edge belongs (only used for degenerate edges right now)
        self.edge_data = np.zeros( (self.Nedges(),2), np.int32)
        
        # actually, I think this isn't necessary
        # no topology is changed here, only Paving-specific bookkeeping.
        # self.freeze=0
        # self.refresh_metadata()

    def initialize_rings(self,rings,degenerates):
        """ cleans up rings, making sure they are in a list, properly ordered,
        without repeating endpoint, and include degenerate rings
        """
        if not isinstance(rings,list):
            rings = [rings]

        if len(rings) > 0: # in weird special cases may not start with any rings
            rings[0] = ensure_ccw(rings[0])
            if all( rings[0][-1] == rings[0][0]):
                # print "Removing repeated first vertex"
                rings[0] = rings[0][:-1]

            for i in range(1,len(rings)):
                # these tests only make sense for non-degenerate rings
                rings[i] = ensure_cw(rings[i])
                if all( rings[i][-1] == rings[i][0] ):
                    # print "Removing repeated first vertex"
                    rings[i] = rings[i][:-1]

            # Drop any rings that don't have enough points
            good_rings = []
            for r in rings:
                if len(r) > 2:
                    good_rings.append(r)
                else:
                    print("WARNING: dropping a ring because it didn't have enough unique points")
            rings = good_rings


        # Add in degenerate rings, but note that they are degenerate
        # to flag special handling later
        self.degenerate_rings = []

        self.rings = rings

        self.freeze=1            

        self.initialize_boundaries()
        self.initialize_edges()

        self.freeze=0
        self.refresh_metadata()

        for i in range(len(degenerates)):
            # important to do this after thawing above, b/c it will
            # use the DT for clipping.
            self.clip_and_add_degenerate_ring( degenerates[i] )

    def clip_and_add_degenerate_ring(self,degen):
        """
        prepare degenerate rings for addition to the current graph.
        For each segment of the polyline, check it against for intersections
        with any existing edge.  Intersections with internal edges will create a new
        node at the intersection point (if a node isn't already very near to there),
        and proceed as if two separate degenerate lines were given.
        Segments falling outside the domain will be clipped off.
        """
        degen = degen.copy()

        okay_points = [degen[0]]

        i = 1
        while i < len(degen):
            p1 = okay_points[-1]
            p2 = degen[i]

            # if either of these are already nodes, go with the node
            check_n1 = self.closest_point(p1)
            if not all(self.points[check_n1] == p1):
                check_n1 = None
                check_p1 = p1
            else:
                check_p1 = None
                
            check_n2 = self.closest_point(p2)
            if not all(self.points[check_n2] == p2):
                check_n2 = None
                check_p2 = p2
            else:
                check_p2 = None

            # Here we should give it nodes if we have them
            conflicts = self.check_line_is_clear_new(n1=check_n1,n2=check_n2,
                                                     p1=check_p1,p2=check_p2)

            if len(conflicts) == 0:
                # boring.
                okay_points.append( degen[i] )
                i += 1
            else:
                # but now we need to have the real points, even if they came from nodes-
                if p1 is None:
                    p1 = self.points[n1]
                if p2 is None:
                    p2 = self.points[n2]

                # the conflicts should come back in order, so just take the first one:
                print("okay, so we got a conflict from the degenerate edge")
                print("it's ", conflicts[0])

                conflict = conflicts[0]

                if conflict[0] == 'v':
                    print("It's a vertex - let's join")
                    n_intersect = conflict[1][0]
                    p_intersect = self.points[n_intersect]
                else:
                    print("It's an edge - have to split it")
                    conf_n1,conf_n2 = conflict[1]

                    p_intersect = intersect_lines( (self.points[conf_n1],self.points[conf_n2]),
                                                   (p1,p2) )

                    # could populate ring, alpha, beta, too.  But it's rigid, why bother?
                    # because there are some sanity checks that at least want to know that
                    # the ring matches while marching along...
                    orig_ring1 = int(self.node_data[conf_n1,self.ORIG_RING])
                    orig_ring2 = int(self.node_data[conf_n2,self.ORIG_RING])
                    
                    # might as well fabricate an alpha, too.
                    if orig_ring1 >= 0 and (orig_ring1 == orig_ring2):
                        print("Figuring alpha from intersection with edge %d-%d"%(conf_n1,conf_n2))
                        orig_ring = orig_ring1

                        # assume that the direction between the two nodes is the same as the
                        # direction of smaller alpha span (fairly safe, but not bulletproof)
                        
                        span12 = (self.node_data[conf_n2,self.ALPHA] - self.node_data[conf_n1,self.ALPHA]) % len(self.original_rings[orig_ring])
                        span21 = (self.node_data[conf_n1,self.ALPHA] - self.node_data[conf_n2,self.ALPHA]) % len(self.original_rings[orig_ring])

                        if span12 > span21:
                            conf_n1,conf_n2 = conf_n2,conf_n1
                            span12 = span21
                        
                        alpha = self.node_data[conf_n1,self.ALPHA] + \
                                span12*  (norm(p_intersect-self.points[conf_n1]) / norm(self.points[conf_n2]-self.points[conf_n1]))
                        print("n1 alpha=%g  new alpha=%g  n2 alpha=%g"%(self.node_data[conf_n1,self.ALPHA],
                                                                        alpha,
                                                                        self.node_data[conf_n2,self.ALPHA]))
                    else:
                        orig_ring = -1 
                        alpha = -1
                    
                    n_intersect = self.add_node( p_intersect, stat=self.RIGID,orig_ring=orig_ring,alpha=alpha )
                    self.split_edge(conf_n1,n_intersect,conf_n2)
                    

                okay_points.append( p_intersect )
                sub_degen = np.array( okay_points )
                print("Might be adding sub-degenerate ring: ",sub_degen)

                # But it's possible that this polyline is not in the domain -
                interior = 0.5*(sub_degen[0] + sub_degen[1])
                interior_pnt = geo.Point( interior )
                if self.poly.contains( interior_pnt ):
                    print("Looks like it's in the domain")
                    self.add_degenerate_ring( sub_degen )
                else:
                    print("Discarding portion of polyline not in the domain")
                okay_points = [p_intersect]
                # and we don't advance i, since we're still trying to get to that next
                # node.

        # Take care of any left over points in okay_points -
        if len(okay_points) > 1:
            sub_degen = np.array( okay_points )
            interior = 0.5*(sub_degen[0] + sub_degen[1])
            interior_pnt = geo.Point( interior )
            if self.poly.contains( interior_pnt ):
                print("Looks like it's in the domain")
                self.add_degenerate_ring( sub_degen )
            else:
                print("Discarding portion of polyline not in the domain")
            
            
    def add_degenerate_ring(self,degen):
        """ Handles adding a degenerate (non-closed) ring -
        This actually uses the regular graph-editing API, building up the edges
        manually.

        There is a corner case where a non-local connection may find a node near
        the end of the degenerate ring.  Since the end point, for now, has to be
        rigid, it cannot be resampled away.  This leaves us with a little pigtail that
        may be hard to get around.  For now, the patch is to 
        """
        # Three cases:
        # 1. the degenerate line connects to existing lines on both ends
        #  -> adding the last node has to be handled as a non-local connection
        # 2. the degenerate line connects to existing lines on one end
        #  -> be careful to add the new edges, on both sides of the degenerate line,
        #     to the corresponding unpaved clist
        # 3. the degenerate line is unconnected -
        #  -> create a new clist going all the way around it

        # Steps:

        #  0. remove any interior nodes that are too close to the ends, assuming that
        #      slide_interiors is true.
        #  1. Figure out which of the above cases we're talking about
        #  1a. For now, resample the line according to the density.
        #  2. Add the new original ring (see initialize_boundaries for how this is done for regular
        #     rings).
        #  3. Add the new nodes, giving them alpha values, but for now setting everyone to RIGID.
        #     For endpoint nodes, don't add a new node, but set the existing one to RIGID.
        #  4. Add the edges, doctoring up the clists as we go.

        if self.slide_internal_guides:
            valid = np.ones(len(degen),np.bool_)
            l = self.density(degen[0])
            for i in range(1,len(degen)):
                if norm(degen[0] - degen[i]) < l:
                    valid[i] = False
                else:
                    break
            for i in range(len(degen)-1,0):
                if norm(degen[-1] - degen[i]) < l:
                    valid[i] = False
                else:
                    break
            if (~valid).sum() > 0:
                print("Removing %d points from internal guide because they're too close to the end"%( (~valid).sum() ))
            degen = degen[valid]
            if len(degen) < 2:
                print("... and that made the line too short.  will omit this one.")
                return
        
        p_start = degen[0]
        p_end = degen[-1]

        start_twin = self.closest_point(p_start)
        end_twin   = self.closest_point(p_end)

        if np.all(self.points[start_twin] == p_start):
            print("Internal edge starts on existing node.")
            start_connects = 1
        else:
            start_connects = 0
            start_twin = None

        if np.all(self.points[end_twin] == p_end):
            print("Internal edge ends on existing node.")
            end_connects = 1
        else:
            end_connects = 0
            end_twin = None

        if not start_connects and end_connects:
            degen = degen[::-1]
            start_connects,end_connects = end_connects,start_connects
            start_twin,end_twin = end_twin,start_twin


        ## Resample only when we're making everything rigid - 
        if not self.slide_internal_guides:
            degen_sampled,degen_alpha = upsample_linearring(degen,self.density,closed_ring=0,return_sources=True)
        else:
            degen_sampled = degen
            degen_alpha = np.arange(float(len(degen)))
        
        ## Create an original ring (slider) for this line
        oring_id = len(self.original_rings)
        self.original_rings.append( degen.copy() )
        # this function probably doesn't really understand non-closed rings...  beware
        self.oring_normals.append( compute_ring_normals( self.original_rings[-1] ) )
        self.degenerate_rings.append(oring_id)


        ## Add the new nodes, setting them all to RIGID/HINT depending on self.slide_internal_guides
        i_to_add = np.arange(len(degen_sampled))
        node_ids = -1*np.ones( len(degen_sampled), np.int32)
        
        if start_connects:
            i_to_add = i_to_add[1:]
            node_ids[0] = start_twin
            # will be forced RIGID below
        if end_connects:
            i_to_add = i_to_add[:-1]
            node_ids[-1] = end_twin
            # will be forced RIGID below

        if self.slide_internal_guides:
            internal_node_stat = self.HINT
        else:
            internal_node_stat = self.RIGID
        for i in i_to_add:
            node_ids[i] = self.add_node(degen_sampled[i],stat=internal_node_stat,
                                        orig_ring=oring_id,alpha=degen_alpha[i])

        # endpoints are rigid regardless:
        self.node_data[node_ids[0],self.STAT] = self.RIGID
        self.node_data[node_ids[-1],self.STAT] = self.RIGID
        
        if not start_connects:
            # have to create a new unpaved clist for the degenerate edge
            cl = CList()
            self.unpaved.append(cl)
            cl.append(node_ids[0])

        # Add edges, fixing up the clist at the same time.
        insert_iters = self.all_iters_for_node( node_ids[0] )
        if len(insert_iters) != 1:
            # annoying - if there are already multiple edges coming into our starting point,
            # we have to figure out which clist is right
            insert_iter = None
            for ii in insert_iters:
                # looking for an iter where the ray going to our next node falls between
                # rays to next and prev.
                asort = self.angle_sort_nodes(node_ids[0],
                                              np.array( [ii.nxt.data,ii.prv.data,node_ids[1]]) )
                asort = asort.tolist()
                nxt_i = asort.index( ii.nxt.data )
                if asort[ (nxt_i+1)%3 ] == node_ids[1]:
                    insert_iter = ii
                    break
            if insert_iter is None:
                raise Exception("Failed to find a good iter in which to insert the degenerate edge")
                 
        else:
            insert_iter = insert_iters[0]
            
        for i in range(1,len(node_ids)):
            ## Maybe not right for the last step...

            # note that this edge belongs to this particular original ring
            e = self.add_edge( node_ids[i-1], node_ids[i], oring=oring_id )

            # Special handling of the last one if it's connected:
            if end_connects and i == len(node_ids)-1:
                print("Connecting the end of an interior edge")
                self.update_unpaved_boundaries(e)
            else:
                # Otherwise we manually take care of the unpaved clist here
                # coming into this, insert_iter points to node_ids[i-1] -
                insert_iter = insert_iter.append( node_ids[i], metric=0 )

                # in all but the case of an unconnected degenerate edge, on the
                # first step, we need to add the "other side" of the edge we just
                # inserted.
                if insert_iter.nxt.data != insert_iter.prv.data:
                    insert_iter.append( insert_iter.prv.data, metric=0 )

    def infer_original_rings(self,mask=None,new_stat=SLIDE):
        """
        Assuming self is a completed grid, build self.rings, self.original_rings,
        self.oring_normals, and populate node_data with ALPHA,STAT,BETA,ORIG_RING
        """
        if mask is None:
            mask=self.edges[:,4]==-1

        rings_and_holes=self.edges_to_rings_and_holes( mask )[0]
        ring_nodes=[rings_and_holes[0]] + rings_and_holes[1]

        self.rings=[]
        for ridx,rnodes in enumerate(ring_nodes):
            self.rings.append( self.points[rnodes] )
            for nidx,n in enumerate(rnodes):
                self.node_data[n,self.ALPHA]=nidx
                self.node_data[n,self.ORIG_RING] = ridx
                self.node_data[n,self.STAT]=new_stat
                self.node_data[n,self.BETA]=0.0

        self.original_rings = [r.copy() for r in self.rings]
        self.oring_normals = [compute_ring_normals(r) for r in self.original_rings]

    def shp_to_rings(self,shp):
        """ Load rings out of a shapefile.  Only the first feature will be
        read, and it must be a polygon.  It can have holes, though.
        """
        
        ods = ogr.Open(shp)
        feat = ods.GetLayer(0).GetNextFeature()
        geom = wkb.loads( feat.GetGeometryRef().ExportToWkb() )

        return self.geom_to_rings(geom)

    def geom_to_rings(self,geom):
        def dedupe(ring):
            # remove repeated vertices
            jp1 = (np.arange(len(ring)) + 1)%len(ring)
            dupes = np.all(ring == ring[jp1],axis=1)
            good = ~dupes
            return ring[~dupes]
            
        if geom.type == 'MultiPolygon':
            print("Geometry is a multipolygon - will use the polygon with greatest area")
            areas = [g.area for g in geom.geoms]
            best = np.argmax(areas)
            old_geom = geom # don't dereference too hastily...
            geom = old_geom.geoms[best]
        
        rings = []
        rings.append( dedupe(np.array( geom.exterior.coords )) )

        for i in range(len(geom.interiors)):
            rings.append( dedupe( np.array( geom.interiors[i].coords )) )

        return rings

    def write_complete(self,fn):
        """ Save the complete (hopefully) state of the Paving to a file.
        The hope is that this will be useful for checkpointing a paving
        and restarting it later.
        The first attempt here is to use the Pickle protocol, and take care of
        all the gritty details in __getstate__ and __setstate__

        
        """
        fp = open(fn,'wb')
        pickle.dump(self,fp,-1)
        fp.close()

    @staticmethod
    def load_complete(fn):
        fp = open(fn,'rb')
        obj = pickle.load(fp)
        fp.close()
        return obj

    def __getstate__(self):
        d = self.__dict__.copy()

        # Clear out things that we don't need
        d['_create_node_listeners'] = {}
        d['_create_cell_listeners'] = {}
        d['_create_edge_listeners'] = {}
        d['_update_node_listeners'] = {}
        d['_update_edge_listeners'] = {}
        d['_delete_edge_listeners'] = {}
        d['_delete_cell_listeners'] = {}
        d['_delete_node_listeners'] = {}
        d['op_stack_serial'] = 10
        d['op_stack'] = None
        d['last_fill_iter'] = None

        # probably ought to be in live_dt
        d['DT'] = 'rebuild'
        d['vh'] = 'rebuild'
        d['vh_info'] = 'rebuild'

        d['poly'] = 'rebuild'

        d['plot_history'] = []
        d['click_handler_id'] = None
        d['showing_history'] = None
        d['index'] = None

        return d
    def __setstate__(self,d):
        self.__dict__.update(d)

        # probably ought to be in live_dt
        try:
            if len(self.shapely_rings) > 0:
                self.poly = geo.Polygon( self.shapely_rings[0],self.shapely_rings[1:] )
            else:
                self.poly = None
        except AttributeError:
            self.shapely_rings = []
            self.poly = None
            
        self.refresh_metadata()
    
    def smooth(self):
        """  Call the smoother and re-initialize the shoreline with the result.
        Note: smoothing does not know how to handle degenerate edges - they will
        be left in, but no check is made that they are inside the smoothed boundaries.
        """
        self.old_poly = self.poly
        
        if len(self.degenerate_rings):
            # With the new CGAL-based code, to handle degenerates, need to eventually
            # figure out how to adapt the smoothing algorithm to understand degenerates
            print("WARNING: dangerous territory.  Smoothing does not know about degenerate rings :WARNING")
            saved_degenerates = [self.original_rings[i] for i in self.degenerate_rings]
            rings = self.geom_to_rings(self.poly)
            self.initialize_rings(rings,[])
        else:
            saved_degenerates = []
            
        if self.verbose > 0:
            print("Smoothing")
            
        new_poly = self.smoothed_poly(self.min_density)
        self.new_poly = new_poly
        if self.verbose > 0:
            print("Done smoothing")

        # kludge: to avoid further headaches, negative buffer the entire polygon
        #   to remove degenerate self-tangent connections
        new_poly2 = new_poly.buffer(-0.25,3)

        print("buffered")
        
        # and that is not very well-behaved - occasionally creates slivers.
        if new_poly2.type == 'MultiPolygon':
            best_i = 0
            for i in range(1,len(new_poly2.geoms)):
                if new_poly2.geoms[best_i].area < new_poly2.geoms[i].area:
                    best_i = i
            # we have to save this because shapely sometimes segfaults
            # on freeing things.
            self.mp_poly = new_poly2
            new_poly2 = new_poly2.geoms[best_i]

        print("selected")
        
        self.new_poly2 = new_poly2

        #if self.verbose > 0:
        print("Removing small islands")

        new_poly3 = remove_small_islands(new_poly2,self.min_density)
        self.new_poly3 = new_poly3

        #if self.verbose > 0:
        print("Done removing small islands")
        
        rings = self.geom_to_rings(new_poly3)

        print("made rings from new_poly3")
        # The buffering can introduce very short edges - filter those out now...
        factor = 0.01 # multiplier for min_density to determine overly short edges
        rings = [ downsample_linearring(r,factor*self.min_density,
                                        closed_ring=1) for r in rings]

        print("Downsampled...")
        # re-initialize with the new rings
        self.initialize_rings(rings,saved_degenerates)
        print("Re-initialized.")

    telescope_rate = 1.1
    def adjust_density(self):
        """ Calculate clearances from the shoreline and decrease the requested
        scale where the clearance is smaller then the scale (or by some factor)

        For now, this can only deal with an XYZField being used for the density.
        """
        self.requested_density = self.density

        if self.verbose>0:
            print("Adjusting scale based on shoreline")
            
        self.density = adjust_scale(self.poly,self.requested_density,r=self.telescope_rate)

        if self.verbose>0:
            print("done with adjust_scale")

    def adjust_density_by_apollonius(self):
        """
        Replace self.density with an instance that uses the Apollonius Graph 
        """
        self.requested_density = self.density

        self.ag_density = self.apollonius_scale(r=self.telescope_rate)

        # The realy density is then the lesser of the requested and the telescoped
        self.density = field.BinopField( self.ag_density,np.minimum,self.requested_density )
        if self.verbose>0:
            print("done with adjust_scale")

    def initialize_boundaries(self):
        """ set up sliders for each of the rings
            set up a boundary and unpaved CLists for each of the rings 
        """
        # initialize all of the points:
        self.points = np.concatenate( self.rings )
        # make sure it's floating point:
        self.points = np.asarray(self.points,dtype=np.float64)

        if self.verbose > 1:
            print("Number of points: ",self.Npoints())

        shapely_rings = []
        for ri in range(len(self.rings)):
            assert ri not in self.degenerate_rings
            
            r = self.rings[ri]
            # oddly, shapely likes outer *and* inner rings to be
            # CCW
            if not is_ccw(r):
                r = r[::-1]

            # also shapely likes rings to have the closing point.
            # (though it doesn't complain if it's missing, it just
            # gets flaky)
            r = np.concatenate( (r,[r[0]]) )

            if self.verbose > 1:
                print("Ring is ",r)
                # plot( r[:,0], r[:,1] )

            shapely_rings.append(r.copy())

        self.shapely_rings = shapely_rings
        # print self.shapely_rings
        
        self.poly = geo.Polygon( shapely_rings[0],shapely_rings[1:] )

        if self.verbose:
            print("poly area:",self.poly.area)
        
        # node type, ring source, alpha, beta
        self.node_data = np.zeros( (self.Npoints(),4), np.float64 )
        
        # data for the slider:
        self.original_rings = [r.copy() for r in self.rings]
        # data for the cross-boundary slider
        self.oring_normals = [compute_ring_normals(r) for r in self.original_rings]
        
        # dllist of the portion that has not been paved
        self.unpaved = [None] * len(self.rings)

        ni = 0
        for ri in range(len(self.rings)):
            degen = ri in self.degenerate_rings
            if degen:
                print("Degenerate ring %i will be set to all RIGID")
            
            cl = CList()
            for i in range(len(self.rings[ri])):
                cl.append( ni, metric=0 )
                if not degen:
                    self.node_data[ni,self.STAT] = self.initial_node_status
                else:
                    # new code! make them rigid for now.
                    self.node_data[ni,self.STAT] = self.RIGID
                    
                self.node_data[ni,self.ORIG_RING] = ri
                self.node_data[ni,self.ALPHA] = i
                self.node_data[ni,self.BETA] = 0.0
                
                ni += 1
            if degen:
                print("Return loop on degenerate ring")
                # ni is the last node + 1
                # say we just put nodes 0,1,2,...,9
                # in.  then ni = 10, and we want to add
                # in nodes 8,7,6,...,1
                # so the first node is ni - 2, and there
                # are
                for i in range(1,len(self.rings[ri])-1):
                    cl.append(ni-i-1) #?
            
            self.unpaved[ri] = cl

    def initialize_edges(self):
        # set the stage:
        self.edges = np.zeros( (0,5), np.int32 )
        self.edge_data = np.zeros( (0,2), np.int32)

        self._pnt2edges = {}
        
        # loop through the unpaved rings, making edges from everything
        for ri in range(len(self.unpaved)):
            uring = self.unpaved[ri]
            
            for n1,n2 in uring.fwd_edge_iter():
                marker = 1 # boundary edge
                c1 = -2 # to be meshed
                c2 = -1 # outside the domain

                self.add_edge(n1,n2,marker,c1,c2, oring = self.node_data[n1,self.ORIG_RING] )

    def boundary_slider(self,ri,alpha,beta=0.0):
        len_b = len(self.original_rings[ri])

        i = int(np.floor(alpha)) % len_b
        frac = (alpha - i) % 1.0
        
        ip1 = (i+1) % len_b

        normal = (1-frac)*self.oring_normals[ri][i] + frac*self.oring_normals[ri][ip1]
        offset = beta * normal;

        return offset + (1-frac)*self.original_rings[ri][i] + frac*self.original_rings[ri][ip1]

    ## methods for GUI editing:
    def toggle_cell(self,p=None,nodes=None):
        """ given a point p, if it is inside a cell, delete the cell and replace with
        solid boundaries.
        if it is not inside a cell, but is bounded by exactly three existing edges,
        add a new cell using those edges.
        """
        if p is not None:
            nodes = self.delaunay_face(p)
            if None in nodes:
                # it's not bounded by a cell - outside the convex hull
                return False
        else:
            p = self.points[nodes].mean(axis=0)
            
        # Is it already a cell?
        
        try:
            c = self.find_cell(nodes)
        except trigrid.NoSuchCellError:
            c = None

        if c is None:
            self.create_cell_and_fix_edges(p,nodes)
        else: # cell exists - so remove it
            # it's possible that if a grid is in an inconsistent state we need
            # to be more careful about this, and do it manually.
            self.delete_cell(c,replace_with=-1)

    def create_cell_and_fix_edges(self,p,nodes):
        # Can it be made into a cell?
        edges = []
        for i in [0,1,2]:
            try:
                edges.append( self.find_edge( (nodes[i],nodes[(i+1)%3]) ) )
            except trigrid.NoSuchEdgeError:
                pass
        if len(edges) == 3:
            print("Okay - looks like we can make a cell of it")
            # For each of those edges, figure out which side of it this new cell
            # is on:
            sides = []
            for e in edges:
                A = self.points[ self.edges[e,0] ]
                B = self.points[ self.edges[e,1] ]
                AB = B-A
                # rotate that vector 90d CCW:
                ABleft = AB[::-1]
                ABleft[0] *= -1
                Ap = p - A
                dp = dot(ABleft,Ap)
                if dp > 0:
                    sides.append(0) # new cell is on left of edge
                else:
                    sides.append(1) # new cell is on right of edge

            new_c = self.add_cell( nodes )
            for i in range(len(edges)):
                self.edges[ edges[i], 3+sides[i] ] = new_c
                # And if the other side is not closed, set marker to 0
                if self.edges[ edges[i],4-sides[i] ] != -1: 
                    self.edges[ edges[i],2] = 0 # internal edge
                self.updated_edge( edges[i] )
            return True
        else:
            print("Found only %d edges - not a valid cell"%(len(edges)))
            return False

    ##

    click_handler_id = None
    def install_click_handler(self):
        fig = gcf()
        # remember what figure we installed it on, too.
        if self.click_handler_id:
            (old_fig, old_cid) = self.click_handler_id
            if fig==old_fig:
                return
            
        self.click_handler_id = (fig,fig.canvas.mpl_connect('button_press_event', self.handle_click))
    
    def handle_click(self,event):
        # print "Got a click"
        f = gcf()
        # print "Active: ",f.canvas.toolbar._active
        if f.canvas.toolbar._active is None:
            # print "button",event.button
            if event.button == 1: # go back in time:
                if self.showing_history is None:
                    new_step = len(self.plot_history)-1
                else:
                    new_step = self.showing_history - 1
            else: # forward in time
                if self.showing_history is None or self.showing_history==-1:
                    return
                else:
                    new_step = self.showing_history+1
            self.plot(plot_step=new_step)
                
    showing_history = None
    def plot_boundary(self):
        for r in self.original_rings:
            plt.fill(r[:,0], r[:,1], fc='none',ec='gray')

    def plot_unpaved(self):
        for unpaved in self.unpaved:
            a = unpaved.to_array()
            # print "This ring is ",a
            if len(a) > 0:
                a = np.concatenate( (a, [a[0]]) )
                plt.plot(self.points[a,0],self.points[a,1],'m-')
                plt.scatter(self.points[a,0],self.points[a,1], s=60,
                            c=self.node_data[a,self.STAT], edgecolors='none')

    def tg_plot(self,*args,**kwargs):
        return trigrid.TriGrid.plot(self,*args,**kwargs)
        
    n_history=0
    def plot(self,just_save=0,plot_step=None,dyn_zoom=None,plot_title=None,clip=None,line_collection_args={},
             boundary=False,hold=False):
        seg_coll=None

        if plot_title is None:
            plot_title = "Step %i"%self.step
        else:
            print("plotting '%s'"%plot_title)
            
        if dyn_zoom is None:
            dyn_zoom = self.dyn_zoom

        if clip is None:
            clip = self.default_clip

        if self.n_history==0 or plot_step is None:
            self.showing_history = None

            good_edges = self.edges[:,2] != trigrid.DELETED_EDGE
            if boundary:
                good_edges = good_edges & (self.edges[:,4]<0)

            segments = self.points[self.edges[good_edges,:2]]

            # Apply clip only to valid edges
            if clip is not None:
                # segments is Nedges * {a,b} * {x,y}
                points_visible = (segments[...,0] >= clip[0]) & (segments[...,0]<=clip[1]) \
                                 & (segments[...,1] >= clip[2]) & (segments[...,1]<=clip[3])
                # so now clip is a bool array of length Nedges
                clip = np.any( points_visible, axis=1)
                segments = segments[clip,...]

            if len(segments) == 0:
                print("WARNING: no segments to display - maybe clip=%s is too strict"%str(clip))
                return
            lc_args = {'lw':0.5*np.ones(len(segments))}
            lc_args.update(line_collection_args)
            seg_coll = LineCollection( segments, **lc_args )

            # color things based on how old they are:
            # kludge: if add_edge() failed, it will leave edge_data one short
            if len(self.edge_data) < len(self.edges):
                self.edge_data = array_append(self.edge_data,[self.step,-1])
                
            age = self.edge_data[good_edges,0]
            if clip is not None:
                age = age[clip]
                
            seg_coll.set_array( np.log(1+self.step - age) )

            if self.n_history>0: # save for later
                self.plot_history.append([seg_coll,plot_title])
                if len(self.plot_history) > self.n_history:
                    self.plot_history = self.plot_history[-self.n_history:]
        else:
            if plot_step < 0:
                plot_step = 0
            elif plot_step >= len(self.plot_history):
                plot_step = len(self.plot_history) - 1
                
            seg_coll,plot_title = self.plot_history[plot_step]
            self.showing_history = plot_step

        if not just_save:
            ax = plt.gca()
            if not hold:
                ax.collections = []
        
            ax.add_collection( seg_coll )
            ax.set_title(plot_title)

            if plot_step is None and dyn_zoom and len(self.cells)>0:
                p = self.points[self.cells[-1,0],:2]
                scale = self.density(p)
                fact = 15 # the zoom size relative to the local scale

                cur = ax.axis()
                if p[0] > cur[0] and p[0] < cur[1] and p[1] > cur[2] and p[1] < cur[3] and (cur[1] - cur[0]) < 3.5*fact*scale:
                    pass
                else:
                    ax.axis([p[0]-fact*scale,p[0]+fact*scale,
                             p[1]-fact*scale,p[1]+fact*scale])
                
            plt.draw()
            if self.n_history > 0:
                self.install_click_handler()
        return seg_coll

    def unadd_edge(self,old_length):
        super(Paving,self).unadd_edge(old_length)
        
        self.edge_data = self.edge_data[:old_length]
        
    def add_edge(self,*args,**kwargs):
        if 'oring' in kwargs:
            oring = kwargs['oring']
            del kwargs['oring']
        else:
            oring = -1
        index = super(Paving,self).add_edge(*args,**kwargs)
        if index < 0:
            print("Paver got a failed add_edge.  Relaying to caller")
            return index

        self.edge_data = array_append( self.edge_data, [self.step,oring] )

        if len(self.edge_data) != self.Nedges():
            raise Exception("Somehow edge_data and edges are out of sync")

        if self.edges[index,3] == -1:
            raise Exception("Somebody left an internal edge as boundary")
        
        return index

    def split_edge(self,nodeA,nodeB,nodeC):
        """ per updates to trigrid.py and live_dt.py, nodeB may be 
        an index or an iterable giving coordinates
        """
        e1 = self.find_edge([nodeA,nodeC])
        
        index = super(Paving,self).split_edge(nodeA,nodeB,nodeC)

        # reversing this is taken care of through TriGrid.unadd_edge, which
        # will call our unadd_edge.  There is no unsplit_edge method.
        self.edge_data = array_append( self.edge_data, [self.step,self.edge_data[e1,1]] )

        if len(self.edge_data) != self.Nedges():
            raise Exception("Somehow edge_data and edges are out of sync")

        if isinstance(nodeB,Iterable):
            pntB=nodeB
            if self.edges[e1,0] in [nodeA,nodeC]:
                nodeB=self.edges[e1,1]
            else:
                nodeB=self.edges[e1,0]
        else:
            pntB=None

        # Doctor any unpaved rings that were affected:
        # look for an iter going from a to c, and one from c to a
        for it in self.all_iters_for_node(nodeA):
            # in the case of an internal guide, if nodeA is the endpoint,
            # then both nxt and prv are nodeC, and both need to be updated,
            # thus both clauses are if's, not if...elif
            if it.nxt.data == nodeC:
                elt = it.clist.append(nodeB,after=it)
                self.push_op(self.unadd_to_unpaved,elt)                
            if it.prv.data == nodeC:
                elt = it.clist.prepend(nodeB,before=it)
                self.push_op(self.unadd_to_unpaved,elt)                
                
        return index

    def unadd_node(self,old_length):
        super(Paving,self).unadd_node(old_length)

        self.node_data = self.node_data[:old_length]
        
    def add_node(self,new_point,stat,orig_ring=-1,alpha=-1,beta=0.0):
        index = super(Paving,self).add_node(new_point)

        self.node_data = array_append( self.node_data, [stat,orig_ring,alpha,beta] )
        
        return index

    def clean_unpaved(self):
        """Remove any unpaved rings that have 0-3 nodes.
        Ideally fill() would take care of this, but it's easier to handle it in
        a more brute force way, and probably isn't a big time sink
        """
        i = 0 
        while i < len(self.unpaved):
            if len(self.unpaved[i]) <= 3:
                if len(self.unpaved[i]) == 2:
                    e = self.find_edge( (self.unpaved[i].head.data,
                                         self.unpaved[i].head.nxt.data) )
                    if self.edges[e,3] == trigrid.UNMESHED:
                        # with the internal guides code, it's possible that an internal line
                        # has only 2 nodes, yet hasn't been paved at all.
                        i += 1
                        continue
                elif len(self.unpaved[i]) == 3:
                    # could be a triangular island, totally unpaved
                    nodes = [self.unpaved[i].head.data,
                             self.unpaved[i].head.nxt.data,
                             self.unpaved[i].head.nxt.nxt.data]
                    xy = self.points[nodes]
                    if not trigrid.is_ccw(xy):
                        if self.verbose>1:
                            print("Triangle is really an unpaved island")
                        i+=1
                        continue
                if self.verbose > 1:
                    print("Removing unpaved ring that has only %d nodes"%len(self.unpaved[i]))
                del self.unpaved[i]
            else:
                i += 1

    # If set to true, node metrics will put all original-ring nodes before any interior
    # nodes.  Currently this doesn't work well, as it misses joins in tight channels, and
    # the would-be-joined node is found as a non-local neighbor.
    prefer_original_ring_nodes=False
    def choose_and_fill(self,ri=None):
        """ choose a center node from the given unpaved ring and
        try to fill it in with new edges/nodes
        """
        if self.verbose > 0:
            print(self.step)

        while 1: # loop, because we may make structural changes and have to re-choose
            center_elt = None
            
            if ri is None:
                if self.prefer_original_ring_nodes:
                    # loop through to find a ring that will give us an original node
                    for i in range(len(self.unpaved)):
                        if len(self.unpaved[i]) <= 3:
                            continue
                        maybe_elt = self.elt_smallest_internal_angle(i)
                        if self.node_data[maybe_elt.data,self.ORIG_RING] >= 0:
                            ri = i
                            center_elt = maybe_elt
                            break
                        
                # either we don't want original ring nodes, or there aren't any left
                if center_elt is None: 
                    for i in range(len(self.unpaved)):
                        if len(self.unpaved[i]) > 3:
                            ri = i
                            center_elt = self.elt_smallest_internal_angle(ri)
                            break
                        
            if ri is None:
                log.info("No more rings need paving -- we're done")
                return False

            if self.verbose > 1:
                print("Choosing from ring ",ri)

            metric = center_elt.clist.heap[center_elt]

            if self.verbose > 1:
                print("Chosen iter: ",center_elt.prv.data,center_elt.data,center_elt.nxt.data)

            # dangerous territory - if this node appears more than once
            # in the ring or in multiple rings, seems like it could get
            # nasty if it or its neighbors are hint nodes.  beware...
            if self.resample_neighbors(center_elt,new_node_stat=self.SLIDE) < 0:
                print("resample_neighbors() made structural changes.  Choosing again")
                continue

            if self.beta_rule == self.BETA_NEVER:
                use_betas = [0]
            elif self.beta_rule == self.BETA_ALWAYS:
                use_betas = [1]
            else: # BETA_RESCUE
                use_betas = [0,1]
                
            for use_beta in use_betas:
                if self.verbose > 0:
                    if use_beta == 1 and self.beta_rule == self.BETA_RESCUE:
                        print("HEY!! RESCUE, maybe?!")

                fill_status = self.fill(center_elt = center_elt,use_beta=use_beta)

                # rare case, but sometimes we can just shift over in a len=4 ring
                if not fill_status and len(self.unpaved[ri]) == 4:
                    print("Trying a shift left in choose_and_fill()")
                    fill_status = self.fill(center_elt = center_elt.nxt,use_beta=use_beta)
                
                if fill_status:
                    break
                
            if not fill_status:
                print("FAILED")
                raise FillFailed()

            if self.verbose > 0:
                print("*")
                
            self.clean_unpaved()
            
            self.step += 1
            return fill_status
        
    stop_at = None
    def resample_neighbors(self,center_elt,new_node_stat=SLIDE):
        """  The heart of the dynamic boundary sampling.
        We've chosen center_node as the next thing to fix, and
        need to make sure that it has good neighbor nodes

        this only applies when center_node is on the boundary and
        only to neighbors that are currently on the boundary as well.

        new_node_stat: the status to assign the new node

        if it had to make structural changes, returns -1, otherwise
        0

        """
        status = 0 # default to all okay
        
        center_node = center_elt.data
        if not self.node_on_boundary(center_node):
            return status

        ring_i = int( self.node_data[center_node,self.ORIG_RING] )

        # print "top of resample neighbors"
        
        local_scale = self.density( self.points[center_node] )

        if self.node_data[center_node,self.STAT] == self.HINT:
            self.node_data[center_node,self.STAT] = self.SLIDE

        if self.stop_at =='resample_neighbors':
            pdb.set_trace()
        
        for direc in ['forward','backward']:
            # this is the distance along the boundary, in each direction
            # before we get to an edge that has been meshed or a node that
            # cannot be moved
            # we are allowed to subdivide to get a good local scale

            # This just got changed from 'original' to 'unpaved'
            # This really does more work than necessary -
            #   it would be sufficient to stop walking the boundary once it's more
            #   than say 10 local_scale's away.  if we stop early, then end_elt will
            #   not be set.  the upper bound is compared to max_dist, not the along
            #   path distance.
            free_boundary_length,end_elt,max_dist = self.free_distance_along_ring(center_elt,
                                                                                  direc,
                                                                                  'unpaved',
                                                                                  upper_bound=10*local_scale)
            if self.verbose > 1:
                if end_elt is not None:
                    print("resample neighbors: free distance=%g from node %d -> %d %s"%(free_boundary_length,
                                                                                        center_elt.data,
                                                                                        end_elt.data,
                                                                                        direc))
                else:
                    print("resample neighbors: max_dist reached upper bound")
                print("      max_dist=%f  local_scale=%f"%(max_dist,local_scale))

            if direc == 'forward':
                prev_elt = center_elt.prv
            else:
                prev_elt = center_elt.nxt

            # end_elt is None if free_distance.. stopped early
            if end_elt and (end_elt == prev_elt or end_elt == center_elt):
                if self.verbose > 0 and free_boundary_length>0:
                    # this is getting printed way too often - seems like maybe end_elt is
                    # no longer set to None in the case that there is no free distance.
                    print("Next node forward is same as one node back.  proceed with caution")
                closed_loop = 1
            else:
                closed_loop = 0

            if self.verbose > 2:
                self.plot(plot_title='resample: %s, clsd=%d ctr=%s'%(direc,closed_loop, center_elt))
            
            if free_boundary_length > 0:
                if end_elt is not None:
                    n_segments = round(free_boundary_length / local_scale)
                    n_segments = max(1,n_segments)

                    if end_elt == center_elt:
                        log.info("We're on a very small loop. Forcing 3 edges")
                        n_segments = max(3,n_segments)
                    elif end_elt == prev_elt:
                        log.info("We're on a very small loop, with one real edge.  Forcing remained to be 2 edges")
                        n_segments = max(2,n_segments)

                    # like local_scale, but "evenly" divides the remaining length
                    quant_scale = free_boundary_length / n_segments

                    # look out for close quarters
                    if quant_scale >= max_dist:
                        if self.verbose > 0 and quant_scale >= 1.05*max_dist:
                            log.info("Too tight.  We wanted to resample at a scale of %f, but the"%quant_scale)
                            log.info("farthest away free point is %f away"%max_dist)
                        n_segments = 1

                    if self.verbose > 1:
                        log.info("resample_neighbors: will try to make that into %d segments"%n_segments)
                else:
                    if self.verbose > 1:
                        log.info("resample neighbors: looks wide open")
                    n_segments=10 # just > 2
                    quant_scale=local_scale
                    
                if n_segments > 1:
                    # print "resample_neighbors: calling resample_boundary"
                    new_elt = self.resample_boundary(center_elt,direc,local_scale=quant_scale,
                                                     new_node_stat=new_node_stat)
                    if not new_elt:
                        log.warning("resample didn't return an element.  could be a problem")
                    elif n_segments == 2:
                        if self.verbose > 1:
                            log.info("resample_neighbors: n_segments is 2, so remove all nodes between new and end_elt")
                            log.info("  resample_boundary, from %s in direction %s, returned %s"%(center_elt,direc,new_elt))
                        to_delete = []
                        trav = new_elt
                        while 1:
                            if direc == 'forward':
                                trav = trav.nxt
                            else:
                                trav = trav.prv

                            if trav == end_elt:
                                break
                            to_delete.append(trav)
                                
                        for elt in to_delete:
                            if self.verbose > 1:
                                print("  resample_neighbors: deleting node %d on ring %s"%(elt.data,elt.clist))
                            self.delete_node_and_merge(elt.data)
                            for all_elt in self.all_iters_for_node(elt.data):
                                self.push_op(self.unremove_from_unpaved,all_elt)
                                all_elt.clist.remove_iters(all_elt)

                        if self.verbose > 2:
                            print("OK.  Hopefully that worked.")
                            self.plot(plot_title='step %d: just cleared free boundary'%self.step)
                else:
                    if n_segments < 1:
                        print("Dynamic boundary sampling - free, but short! still trying...")
                        n_segments = 1

                    # Theoretically, this will be handled in close_free_boundary()
                    # if closed_loop:
                    #     print "There is only room for one edge on the free boundary, but"
                    #     print "this is a closed loop, so bailing out because that would"
                    #     print "make a duplicate edge"
                    #     raise Exception,"would be making duplicate edge"
                        
                    # in this case there is some free boundary out there, but it may
                    # have too many points and is almost certainly marked HINT.
                    # here we convert the remaining span of free boundary to one edge
                    if self.verbose>1:
                        print("calling close_free_boundary")
                    status = self.close_free_boundary(center_elt,end_elt,direc)

            if self.verbose > 2:
                self.plot(plot_title='step %d after resampling %d in dir %s'%(self.step,
                                                                              center_elt.data,
                                                                              direc))
        # -1: structural changes (node degree has changed)
        # 0: all good
        return status
      
                        

    # helper functions for dynamic resampling that have not been written:
    # it may be best to create a structure like the unpaved linked list
    # that will handle the boundary.  This would provide for consistent
    # ways to track sources, boundary membership, and insert new nodes.
    def node_on_boundary(self,n):
        return self.node_data[n,self.ORIG_RING] >= 0

    ## Self-intersection tests
    # 1e-12 is okay for domains O(100) units across with
    # edge lengths O(10).
    # for a UTM domain, though, what should be 0 can be
    # as large as 1e-7.
    TEST_SMALL = 1e-10
    def check_edge_against_boundary(self,e):
        # for starters, just loop over everything
        a,b = self.edges[e,:2]
        ab = [a,b]

        # choose a reference point and shift all other
        # points to be relative to here.  This hopefully
        # avoids some floating point precision issues
        ref = self.points[a]

        pa = self.points[a] - ref
        pb = self.points[b] - ref

        a_ab,b_ab,c_ab = line_eq(pa,pb)

        for unpaved in self.unpaved:
            # at least do the first step as vectorized ops
            n_list = unpaved.to_array()
            i = np.arange(len(n_list))
            ip1 = (1+i) % len(n_list)
            N1 = n_list[i]
            N2 = n_list[ip1]

            # these were i and ip1 - seems totally wrong.
            P1 = self.points[N1] - ref
            P2 = self.points[N2] - ref

            D1 = a_ab*P1[:,0] + b_ab*P1[:,1] + c_ab
            D2 = a_ab*P2[:,0] + b_ab*P2[:,1] + c_ab
            
            PROD1 = D1*D2

            possible = np.where(PROD1 < -self.TEST_SMALL)[0]
            
            for pi in possible:
                n1 = N1[pi]
                n2 = N2[pi]

                p1 = P1[pi]
                p2 = P2[pi]
                prod1 = PROD1[pi]
                
                # print "One-way intersection w/product=",d1*d2
                # could be an intersection
                # see if the original edge spans
                # both sides of this edge
                a_12,b_12,c_12 = line_eq(p1,p2)
                d1 = a_12*pa[0] + b_12*pa[1] + c_12
                d2 = a_12*pb[0] + b_12*pb[1] + c_12
  
                if d1*d2 < -self.TEST_SMALL:
                    print("two-way intersection w/products %g and %g"%(prod1,d1*d2))

                    # possible for a pair of lines that share a node
                    # to get enough numerical roundoff to appear
                    # to intersect, so double check for any shared nodes
                    if a in [n1,n2] or b in [n1,n2]:
                        print("they share a node - consider increasing TEST_SMALL or learning to write robust code")
                        continue

                    e_bad = self.find_edge( (n1,n2) )
                    # print "pi was",pi
                    print("nodes are ",n1,n2)
                    print("Intersection found between edges %d and %d"%(e,e_bad))
                    
                    [plt.annotate(str(i),self.points[i]) for i in (n1,n2,a,b)]
                    #annotate('p1',p1+ref)
                    #annotate('p2',p2+ref)
                    
                    plt.plot([pa[0]+ref[0],pb[0]+ref[0]],
                             [pa[1]+ref[1],pb[1]+ref[1]],'m',lw=3)
                    plt.plot([p1[0]+ref[0],p2[0]+ref[0]],
                             [p1[1]+ref[1],p2[1]+ref[1]],'c',lw=3)

                    return n1,n2
        return None

    def free_distance_along_ring(self,from_elt,direc,path='original',upper_bound=None):
        """
        this is the distance along the boundary, in the given
        direction 'forward' | 'backward'
        before we get to an edge that has been meshed or a node that
        cannot be moved
        
        adding up segments lengths as long as the edges are unmeshed (have
        no cells on either side)

        if path == 'original', the distance is computed following the original
        ring.
        if path == 'unpaved', the distance is computed following the edges of
        unpaved.

        upper_bound: if set, the tracing will stop if max_dist reaches the upper_bound.
          in this case, end_elt is set to None.

        return distance,end_elt,max_dist
          where max_dist is the distance to the node on the path farthest
          from from_elt 
        """
        
        orig_ring_i = int(self.node_data[from_elt.data,self.ORIG_RING])

        unpaved = from_elt.clist
        
        #print "top of free_distance_along_boundary(ri = %d,%s,%s)"%(orig_ring_i,
        #                                                            from_elt.data,
        #                                                            direc)
        
        # figure out which direction we're going
        if direc == 'forward':
            stepper = unpaved.nxt
        elif direc == 'backward':
            stepper = unpaved.prv
        else:
            raise Exception("direc should be forward or backward")

        # we'll get an alpha value for from_node and the node defining the end
        # of the free interval, then ask the boundary slider code to get a
        # distance along the original_ring joining them.

        A = from_elt
        B = stepper(A)

        n_edges_walked = 0
        nodes = [A.data]

        stopped_early = 0
        
        while 1:
            ## I'm almost positive that some of these checks are redundant...

            # print " free_dist: A=%d, B=%d"%(A.data,B.data)

            ## First, checks that would cause us to stop at A:
            
            # get the edge between from_node and to_node
            e = self.find_edge((A.data,B.data))
            
            # is it free? both sides of the edge must either be
            # boundary or unmeshed (-1 or -2)
            if np.any( self.edges[e,3:5] >= 0 ):
                # this edge should not be counted
                # print "stopping: edge AB has cells"
                to_elt = A
                break

            # Are there times that it's okay to count edges that are
            # unmeshed on both sides?  I'm going to say no...  may have
            # change if/when we are allowed to resample degenerate edges

            # With the new code that resamples degenerate edges, this check
            # is now disabled in favor of checking whether the edge belongs
            # to an original ring.  Note that there are probably some places
            # where edge_data is not populated correctly
            ## if all( self.edges[e,3:5] == -2 ):
            ##     print "Stopping free_distance because both sides are unmeshed."
            ##     to_elt = A
            ##     break
            if self.edge_data[e,1] < 0:
                print("Stopping free_distance because hit an edge that is not on the original ring")
                to_elt = A
                break

            # the edge check above isn't foolproof - we can have
            # edges that are internal, with -2 on both sides
            if self.node_data[B.data,self.STAT] == self.FREE:
                # print "stopping - node A is FREE, discard AB"
                to_elt = A
                break

            # also, we may have jumped to another ring
            if self.node_data[B.data,self.ORIG_RING] != orig_ring_i:
                # print "stopping - B is on a different ring"
                to_elt = A
                break

            # check to see if A is already far enough away:
            dist_sqr = ((self.points[from_elt.data] - self.points[A.data])**2).sum()
            if upper_bound and dist_sqr > upper_bound*upper_bound:
                if self.verbose > 2:
                    print("stopping a free_distance_query early")
                to_elt = None
                stopped_early = 1
                break

            ## Second, checks that would cause us to stop at B
            
            # is the other end of this edge movable?
            # we have to stop at rigid  and
            # be sure to stop if we make it all the way around
            # 2011-01-29: shouldn't we stop at SLIDE nodes, too?
            #   adding that in, though I wonder why it wasn't there before.
            if self.node_data[B.data,self.STAT] in [self.RIGID,self.SLIDE] or \
               B==from_elt:
                # print "Stopping - B is rigid"
                to_elt = B
                n_edges_walked+=1
                nodes.append(B.data)
                break

            # proceed to next edge
            A = B
            B = stepper(B)
            n_edges_walked += 1
            nodes.append(A.data)

        # NB: to_elt may be None here.
        if to_elt == from_elt:
            if n_edges_walked == 0:
                # print "free_distance_along_original_ring() got nowhere."
                return 0.0,to_elt,0.0
            
        # print "Nodes walked:",nodes
        # print "Ending iter: ",to_elt
        
        if path=='unpaved':
            # first, make sure we got the last node - this should now be taken care of by
            # the code above.
            # if nodes[-1] != to_elt.data:
            #     nodes.append( to_elt.data )
            plist = self.points[nodes]
            dist_unpaved = np.sqrt( (np.diff(plist,axis=0)**2).sum(axis=1) ).sum()

            dist_sqr = ((self.points[from_elt.data] - self.points[nodes])**2).sum(axis=1)
            
            return dist_unpaved,to_elt,np.sqrt(dist_sqr.max())
        else:
            raise Exception("stale code")
            metrics = [ self.node_data[from_elt.data,self.ALPHA],
                        self.node_data[to_elt.data,self.ALPHA] ]
            if direc == 'backward':
                metrics = metrics[::-1]

            # too lazy right now to compute max_dist when walking the original ring
            # no code should be using it right now anyway.
            return (self.coarse_boundary_length( orig_ring_i, metrics[0], metrics[1] ),
                    to_elt,
                    np.nan)

    def close_free_boundary(self,start_elt,end_elt,direction):
        """
        kind of like clear_boundary_by_alpha, although the endpoint already
        exists in this case so we're careful to delete exactly the ones
        before it.

        direction='forward' | 'backward'
        returns -1 if degree of start_elt was changed
           this happens when start_elt is already adjacent to end_elt in the
           other direction.  closing the boundary then means collapsing the
           loop.
        """
        # print "Call to close_free_boundary, node %d to %d, direction=%s"%(start_elt.data,
        #                                                                   end_elt.data,
        #                                                                   direction)
        status = 0            
        
        if direction == 'forward':
            direction = 1
            stepper = lambda e: e.nxt
            opp_nbr = start_elt.prv
        elif direction == 'backward':
            direction = -1
            stepper = lambda e: e.prv
            opp_nbr = start_elt.nxt
        else:
            raise "Bad direction %s"%direction

        if end_elt == opp_nbr:
            if self.verbose > 0:
                print("Ahhh - this is going to be a collapse.  Hold on")

            # ultimately we want the edge [start_elt.data,end_elt.data]
            # to have -1 on the side that is to the inside of the loop.
            # it's safe to delete those edges then - they shouldn't have
            # any cell neighbors (since we checked before that they were
            # free).

            # First, delete the edge going the other way
            e = self.find_edge([start_elt.data,end_elt.data])
            if self.verbose > 0:
                print("Delete edge %d, with data: %s"%(e,self.edges[e]))

            a,b,xx,c1,c2 = self.edges[e]
            # so a->b has c1 on left, c2 on right

            if a==start_elt.data:
                # edge e is oriented opp. the direction we're going
                if direction==-1:
                    old_cell_nbr = c2
                    unpaved_nbr = c1
                else:
                    old_cell_nbr = c1
                    unpaved_nbr = c2
            else: # b==start_elt.data
                if direction==-1:
                    old_cell_nbr = c1
                    unpaved_nbr = c2
                else:
                    old_cell_nbr = c2
                    unpaved_nbr = c1

            if unpaved_nbr != -2:
                raise Exception("Expected unpaved, but interior nbr cell was %s"%unpaved_nbr)

            if old_cell_nbr >= 0:
                old_cell = self.cells[old_cell_nbr] # remember the nodes
            else:
                old_cell = None
            
            self.delete_edge(e)
            status = -1

        # walk along the boundary
        to_delete = []
        trav = stepper(start_elt)
        while trav != end_elt:
            to_delete.append(trav)
            trav = stepper(trav)

        unpaved = start_elt.clist
        for elt in to_delete:
            self.delete_node_and_merge(elt.data)
            # be warned - this is delicate...
            # node may belong to multiple iters, if it's an internal guide.
            for all_elt in self.all_iters_for_node(elt.data):
                self.push_op(self.unremove_from_unpaved,all_elt)
                all_elt.clist.remove_iters(all_elt)

        if status == -1:
            # We've removed all the extra edges, should be back to having an edge
            # between a,b
            new_e = self.find_edge([a,b])
            print("New edge %d-%d => %s"%(a,b,self.edges[new_e]))

            if old_cell_nbr >= 0:
                if self.verbose > 0:
                    print("Would be adding a cell back in, using the same 3rd node as before")
                ci = self.add_cell(old_cell)

                # one of the edges here should be marked unpaved, which we replace with
                # the newly created cell
                if self.edges[new_e,3] == trigrid.BOUNDARY:
                    if self.edges[new_e,4] != trigrid.UNMESHED:
                        raise Exception("This should have been unmeshed!")
                    self.edges[new_e,4] = ci
                elif self.edges[new_e,4] == trigrid.BOUNDARY:
                    if self.edges[new_e,3] != trigrid.UNMESHED:
                        raise Exception("This should have been unmeshed!")
                    self.edges[new_e,3] = ci
                else:
                    raise Exception("Neither side of the new edge is a boundary. wrong!")
                
        return status

    def pnt2nbrs(self,n):
        e = self.pnt2edges(n) # all of our edges
        n_is_first = 1 * (self.edges[e,0] == n) 
        other = self.edges[e,n_is_first]
        return other
        
    def resample_boundary(self,start_elt,direction,local_scale,
                          new_node_stat=SLIDE):
        """
        new_node_stat:  status to be assigned to any new nodes created
        returns an iterator for the new element neighboring start_elt.
        """
        # Starting angle-sort of the neighbors 
        start_nbrs = self.angle_sort_neighbors(start_elt.data)
        # Which of those is the one being moved?
        if direction =='forward':
            to_move = start_elt.nxt.data
        else:
            to_move = start_elt.prv.data

        ## Special handling of internal guides [degenerates]
        e = self.find_edge( (start_elt.data,to_move) )
        
        if self.slide_internal_guides and self.verbose > 1:
            print("Resampling on edge e=%d,  start node=%d,  node to move=%d\n"%(e,start_elt.data,to_move))
            print("Resampling on edge e=%d,  %d->%d"%(e,start_elt.data,to_move))
            print("is %d in degenerate_rings: "%(self.edge_data[e,1]),self.degenerate_rings)

        # internal guides may flip direction in order to make alpha consistent
        # with the direction - remember the original direction so we can return the
        # correct iterator at the end
        original_direction = direction
        original_start_elt = start_elt
        
        if self.edge_data[e,1] in self.degenerate_rings:
            if self.verbose > 1:
                print("Okay - this is a degenerate edge")
            
            # if we're resampling a degenerate edge, alpha may not increase in the
            # expected way, but we can flip to the other side to make it work out
            need_to_flip = 0
            
            if (direction == 'forward') != (self.node_data[start_elt.data,self.ALPHA] < self.node_data[to_move,self.ALPHA]):
                if self.verbose > 1:
                    print("Resampling degenerate: need to flip-flip")

                # [at least] two cases:
                #  -we're in the middle of a degenerate edge - in this case, we can flip the direction, and
                #   grab an elt from the other side
                #  -we're at the end of the degenerate edge - in this case, we only have to flip the direction.
                if direction == 'forward':
                    direction = 'backward'
                else:
                    direction = 'forward'

                # and in all cases we don't have to update to_move
                if start_elt.nxt.data != start_elt.prv.data:
                    # find another start_elt
                    its = self.all_iters_for_node(start_elt.data)
                    opp_it = None
                    for it in its:
                        if direction == 'forward' and it.nxt.data == to_move:
                            opp_it = it
                            break
                        elif direction == 'backward' and it.prv.data == to_move:
                            opp_it = it
                            break
                    if opp_it is None:
                        raise Exception("Tried to find mirror iter, but failed")
                    start_elt = opp_it
                else:
                    print("End of the degenerate line, just flipping direction")
            
        to_move_i = np.where(start_nbrs==to_move)[0][0]

        node_bounds = start_nbrs[ [(to_move_i-1)%len(start_nbrs),
                                   (to_move_i+1)%len(start_nbrs) ] ]
        deltas = self.points[node_bounds] - self.points[start_elt.data]
        cw_ccw_angles = np.arctan2( deltas[:,1],deltas[:,0] )
        cw_angle_bound = cw_ccw_angles[0]
        angle_range = (cw_ccw_angles[1] - cw_ccw_angles[0]) % (2*np.pi)

        #-- This block used to be inside the try:except: block, but I don't think
        #   there's any reason for that.
        start_node = start_elt.data
        ring_i = int( self.node_data[start_node,self.ORIG_RING] )

        if self.verbose > 1:
            print("resample_boundary: start_elt=%s dir=%s scale=%f"%(start_elt,direction,
                                                                     local_scale))

        len_b = len(self.original_rings[ring_i])
        if direction == 'forward':
            direction = 1
        elif direction == 'backward':
            direction = -1
        else:
            raise "Bad direction %s"%direction

        alpha = self.node_data[start_node, self.ALPHA]
        beta  = self.node_data[start_node, self.BETA]

        if self.slide_internal_guides and self.verbose > 1:
            print("Starting alpha = %f  direction=%d"%(alpha,direction))

        # force new node to be on the original ring
        # setting this to beta will place the new node on the same beta offset,
        # but then we have to worry about running into a neighbor that has a different
        # beta
        sliding_beta = 0.0 

        # slide alpha in the direction given by direction until we've gone
        # far enough along the coarse boundary

        # state:
        # the previous point in our interpolation, along with it's alpha value

        # Rather than trying to make the traversal robust to a starting point
        # that isn't on the boundary, just do everything based off a beta=0
        # point.  This will make the resulting point slightly farther from
        # the real starting point, but as long as beta is small, this shouldn't
        # be a problem.

        # start_point = self.points[start_node] # the real starting point
        start_point = self.boundary_slider(ring_i,alpha,0.0)  # the easier one.

        # i tracks the point that we're moving towards
        if direction > 0:
            # always the next higher whole number
            i = (np.floor(alpha) + 1) % len_b
        else:
            i = (np.ceil(alpha) - 1) % len_b

        i=int(i)
        start_i = i 
        oring = self.original_rings[ring_i]
        nring = self.oring_normals[ring_i]

        # as long as we need to go farther than the length of this segment,
        # iterate to the next segment
        # in case we can't get as far away as we'd like, remember
        # the farthest away and it's alpha
        max_dist = 0
        max_dist_alpha = None
        full_loop = 0
        while 1:
            # print "Trying point i=%d"%i
            d = norm( (oring[i]+nring[i]*sliding_beta) - start_point )
            if d > local_scale:
                # normal, i is the first oring index that is farther away
                # than we want
                break
            if d > max_dist:
                max_dist = d
                max_dist_alpha = i

            i = (i + direction) % len_b
            if i==start_i:
                print("resample_boundary: couldn't get far enough away from node")
                # so now we really just want to use max_dist_alpha
                # probably ought to also detect if we've run into our neighbor
                # in the opposite direction.
                full_loop = 1
                break

        im1 = (i - direction) % len_b

        if self.slide_internal_guides and self.verbose > 1:
            print("Stopped at im1=%d  i=%d"%(im1,i))

        # so now i indexes the point along the coarse boundary just beyond the length we want
        # unless we made a full loop, in which case use the farthest-away point we encountered.
        if full_loop:
            print("Went all the way around, will use max_dist_alpha")
            new_alpha = max_dist_alpha
        elif i == start_i:
            # just need to interpolate along the segment that we're already on.
            #  the new alpha is wherever we started on this segment, modified by the relative
            #  length of the segment we had to move

            new_alpha = alpha + direction*( local_scale / \
                                            norm( (oring[i]+sliding_beta*nring[i]) - (oring[im1]+sliding_beta*nring[im1])) )

            # multiply by direction b/c if we are moving backwards, starting at
            # alpha = 1, it's okay that we get new_alpha = 0.999
            if np.floor(direction*new_alpha) != np.floor(direction*alpha):
                print(new_alpha, alpha)
                raise Exception("Thought that we were on the same segment, but alphas are different")
        else:
            # approximate this, because we don't really want to solve the quadratic equation here.
            near_dist = norm( (oring[im1]+nring[im1]*sliding_beta) - start_point )
            far_dist  = norm( (oring[i]+nring[i]*sliding_beta)   - start_point )

            if near_dist > local_scale or far_dist < local_scale:
                raise Exception("Sanity lost.  walking didn't work.")

            # new alpha is the near end of the segment
            new_alpha = im1 + direction* (local_scale - near_dist) / (far_dist - near_dist)

        new_alpha = new_alpha % len_b

        # print "resample_boundary: new_alpha=%f"%new_alpha

        new_point = self.boundary_slider(ring_i,new_alpha,sliding_beta)

        if full_loop and verbose > 1:
            print("Boundary sliding: wanted length = %g, got length = %g"%(local_scale,norm(new_point - start_point )))

        # A few geometry checks here signal whether we can resample, or we should just bail.
        new_node_passes_checks = 1
        
        ## ANGLE CHECK
        if len(start_nbrs) > 2:
            new_point_angle = np.arctan2( new_point[1] - start_point[1],
                                          new_point[0] - start_point[0] )
            past_cw = (new_point_angle - cw_angle_bound) % (2*np.pi)
            if past_cw > angle_range:
                print("Nope - the angle of the new point was bad relative to neighbors.")
                print("  cw_angle_bound: %f"%(180*cw_angle_bound/np.pi))
                print("     angle_range: %f"%(180*angle_range/np.pi))
                print("     center node: %d"%(start_elt.data))
                plt.annotate('center',start_point)
                plt.annotate('new_point',new_point)

                new_node_passes_checks = 0
                
        ## /ANGLE CHECK

        ## INTERSECTION CHECK
        # if we probe with a point that is exactly on top of an existing point we get
        # into CGAL trouble.  So figure out if this
        closest_existing_node = self.find_closest_node_to_alpha(start_elt,direction,new_alpha)
        if abs(self.points[closest_existing_node] - new_point).sum() < 1e-5:
            # print "will use a pre-existing node rather than a probe point"
            constr_edges = self.check_line_is_clear_new(n1=original_start_elt.data,
                                                        n2=closest_existing_node)
        elif abs(live_dt.distance_left_of_line(new_point,
                                               self.points[original_start_elt.data],
                                               self.points[closest_existing_node] ))  < 1e-7:
            # if the probe edge is coincident with a real edge, CGAL may ditch
            # the constrained edge, which causes headaches later.
            # print "Intersection test must be safe, since it's already colinear, right?"
            # it's possible that this is too permissive - the 3 points could be colinear
            # but not actually connected - if that does turn out to be an issue, I think
            # it would be sufficient to do the same test as the previous clause - even
            constr_edges = []
        else:
            # on thistle, with CGAL 4.2, SWIG 2.0.10, it fails somewhere in here.
            constr_edges = self.check_line_is_clear_new(n1=original_start_elt.data,
                                                        p2=new_point)
        # print "Got constrained edges ",constr_edges
        
        # some of these edges could be edges that we're about to be deleting anyway -
        # weed those out by consulting alpha values, and ignoring edges between nodes
        # that fall within the alpha range that we'll be tossing out
        if direction>0:
            int_min_alpha = self.node_data[start_node, self.ALPHA]
            alpha_span = (new_alpha - int_min_alpha) % len_b
        else:
            int_min_alpha = new_alpha
            alpha_span = (self.node_data[start_node, self.ALPHA] - int_min_alpha) % len_b
            
        for constr_type,nodes in constr_edges:
            if constr_type != 'e':
                continue # potentially an issue, but I'm not going to get into dealing with it now.
            e = self.find_edge(nodes)
            # print "Checking on potential conflict with edge %d"%e
            alphas = self.node_data[nodes,self.ALPHA]
            if np.all( (alphas - int_min_alpha) % len_b > alpha_span):
                print("WARNING: looks like this would cause a self-intersection")
                new_node_passes_checks = 0
        ## /INTERSECTION CHECK

        if not new_node_passes_checks:
            print("resample_neighbors: node failed some geometry checks - will stick with the current neighbor")
            # our neighbor is left unchanged in position, but we may still
            # need to update its status
            n = to_move
            # < just to avoid changing a SLIDE back into a HINT
            if self.node_data[n,self.STAT] < new_node_stat:
                print("Changing its status from %s to %s"%(self.node_data[n,self.STAT],
                                                           new_node_stat))
                self.change_node_stat(n,new_node_stat)
            print("Neighbor will be left as node %d"%n)

            # any reason not to return the corresponding bounday iter?
            if original_direction == 'forward':
                return original_start_elt.nxt
            else:
                return original_start_elt.prv
            

        # hold DT updates because we may go through some inconsistent states
        # while deleting nodes
        self.hold()
        try:
            # this makes sure that there aren't any nodes that are too close (any that are
            # will be deleted)
            if self.verbose > 1:
                print("resample_boundary: calling clear_boundary_by_alpha_range")

            # it's possible that we will try to clear too much -
            # currently, we can't handle even temporarily making a ring into a 1 or
            # 2 node ring.  to avoid that, this may return a node that can then be slid
            # into place, rather than being deleted and then inserting a new node.  
            n = self.clear_boundary_by_alpha_range(start_elt,direction,new_alpha,keep_last=1)

            if self.verbose > 1:
                print("resample_boundary: clear_boundary_by_alpha range gave back node n=%s"%n)

            if self.verbose > 2:
                self.plot(plot_title="step %d: after clear_boundary_by_alpha"%self.step)

            if n is not None:
                if self.verbose > 1:
                    print("resample_boundary: Sliding node %d instead of deleting and re-inserting"%n)

                if self.node_data[n,self.STAT] != new_node_stat:
                    self.change_node_stat(n,new_node_stat)

                if self.check_new_alpha(n,new_alpha):
                    self.slide_node(n,new_alpha,sliding_beta)
                else:
                    raise Exception("Alpha from node from clear_boundary_by_alpha() is invalid")

                # this is also nice because the topology is already there.  we're done.
                if original_direction == 'forward':
                    return original_start_elt.nxt
                else:
                    return original_start_elt.prv
            else:
                # Figure out which old edge is going to be split by the new node.
                if original_direction == 'forward':
                    old_nbr = original_start_elt.nxt.data
                else:
                    old_nbr = original_start_elt.prv.data

                # One particular corner case:
                # when the ring is small, it is possible that clear_boundary_by_alpha
                # cleared all the way back to our neighbor in the other direction.
                # for the moment, detect that and complain.  Two possibilities for what
                # should happen: either the new node is on top of our other neighbor,
                # and there is no use in splitting the old edge - just merge it
                # or the new node is good and we should be making a triangle with the
                # new node and our old neighbor.

                # NB: this test is not complete!  the start_elt iterator follows the unpaved
                # boundary, but the problem here is really along the original ring.

                if original_start_elt.nxt.data == original_start_elt.prv.data:
                    # This used to be a bailout:
                    # raise Exception,"Cleared entire boundary except for old neighbor"
                    # but with internal guides, it's entirely possible that a ring has
                    # exactly two nodes.
                    print("WARNING: Cleared entire boundary except for old neighbor")
                    print("         This could just be a 2-node internal guide")
                    
                # print "About to split edge %d-%d to add our new resampled node %d"%(start_elt.data,
                #                                                                     old_nbr,
                #                                                                     n)

                # e = self.find_edge((start_node,old_nbr))

                # eventually these new nodes can be stat=SLIDE, but at the moment
                # that appears to be broken.
                # n = self.add_node(new_point,stat=new_node_stat,orig_ring=ring_i,alpha=new_alpha,beta=sliding_beta)
                # a new node could upset the live_dt data structure - so pass the info on up the chain
                # and trigrid will create it for us when it's safe.
                ninfo=(new_point,dict(stat=new_node_stat,orig_ring=ring_i,alpha=new_alpha,beta=sliding_beta))

                # This probably needs to either take elements, or be smart enough
                # to look in unpaved[] for element pairs that must be updated.
                # the latter would be preferable, since it would allow resampling
                # of shared edges (think degenerate islands)
                self.split_edge(start_node,ninfo,old_nbr)

                # split_edge is now smart enough to handle this automatically -
                if original_direction=='forward':
                    return original_start_elt.nxt
                else:
                    return original_start_elt.prv
        finally:
            self.release()
            
    def find_closest_node_to_alpha(self,start_elt,direction,new_alpha):
        """ Return the node index for the node found by walking along the
        elt in the given direction, finding the new with an alpha closest to
        new_alpha.
        """
        oring = int(self.node_data[start_elt.data,self.ORIG_RING])
        len_b = len(self.original_rings[oring])

        # have to be careful with modulo distances
        def alpha_diff(alpha):
            d1 = (alpha - new_alpha) % len_b
            d2 = (new_alpha - alpha) % len_b
            return min(d1,d2)
        
        best_n = start_elt.data
        best_diff = alpha_diff(self.node_data[best_n,self.ALPHA])
        trav = start_elt
        
        while 1:
            if direction>0:
                trav = trav.nxt
            else:
                trav = trav.prv
            this_diff = alpha_diff( self.node_data[trav.data,self.ALPHA] )
            if this_diff < best_diff:
                best_diff = this_diff
                best_n = trav.data
            else:
                break
            if trav == start_elt:
                break
        return best_n
                
            

    def clear_boundary_by_alpha_range(self,start_elt,direction,end_alpha,keep_last=0):
        """ this makes sure that there aren't any nodes that are too close (any that are
        will be deleted)

        if keep_last is true:
          when possible, it leaves the last node there and returns its index
          such that it can just be slid.
        """
        start_node = start_elt.data
        start_alpha = self.node_data[start_node,self.ALPHA]
        ring_i = int( self.node_data[start_node,self.ORIG_RING] )

        len_b = len( self.original_rings[ring_i] )

        # span is the length of the range of alphas that are to be discarded.
        span = (direction*(end_alpha - start_alpha)) % len_b

        to_delete = []

        if direction == 1:
            stepper = lambda e: e.nxt
        elif direction == -1:
            stepper = lambda e: e.prv
        else:
            raise ValueError("Direction must be 1 or -1")

        trav = stepper(start_elt)

        to_delete = []

        last_dist = 0
        while 1:
            n = trav.data
            n_stat=self.node_data[n,self.STAT]
            if n_stat != self.HINT:
                # This is generally not a problem.  Used to be reported
                # as a warning, but frequently this just means that the "right"
                # node is in to_delete, and will be chosen below.  It's possible
                # that if keep_last is False, then we're going to try to monkey
                # with this node and it's more fixed than a HINT
                log.debug("While traversing the boundary in clear_boundary_range_by_alpha")
                log.debug("   encountered non-HINT node %d before alpha was reached"%n )
                if (not keep_last) and (n_stat != self.SLIDE):
                    log.warning("Actually it's worse - this node has stat=%s and can't be moved at all"%n_stat)
                    log.warning("   and keep_last is not set, so we're going to try to move it")
                break
                
            # Actually this could happen, once two rings are joined
            # by non-local connection, we could take one step and be
            # on another ring.  the only reason it shouldn't is that
            # the free_distance_along_...() call should have figured
            # out that it's not valid to request an alpha this far away
            if self.node_data[n,self.ORIG_RING] != ring_i:
                log.error("Ring mismatch in clear_boundary_by_alpha")
                log.error("  clearing from %f to %f"%(start_alpha, end_alpha))
                log.error("  direction is ",direction)
                log.error("  start node is ",start_elt.data)
                log.error("  current node is",n)
                log.error("  starting ring is ",ring_i)
                log.error("  current ring is",self.node_data[n,self.ORIG_RING])

                raise Exception("rings don't match")
            
            this_alpha = self.node_data[n,self.ALPHA]

            # Similar to how span is calculated
            dist = (direction*(this_alpha-start_alpha)) % len_b

            if self.verbose > 1:
                log.debug("  stepping, got alpha=%f"%this_alpha)
                log.debug("  dist = %f, span = %f"%(dist,span))

            # reach every so slightly farther to take care of some
            # floating point roundoff
            # I think that this is saying that we only stop when the
            # distance traversed in alpha is 1.001 farther than the section
            # we want to ensure is clear.  So if a node is at 1.000001*span
            # this won't stop, will include it in the ones to delete, and
            # the new node won't be on top of it.
            if dist > 1.001*span:
                break
            elif self.verbose > 1 and dist > span:
                log.debug("clear_boundary_by_alpha: fudged the span to include an extra node")

            if dist < last_dist:
                raise Exception("Inf loop - distance along ring got smaller")
            last_dist = dist
            
            to_delete.append( trav )
            trav = stepper(trav)

        unpaved = start_elt.clist

        if keep_last and len(to_delete) > 0:
            kept_last = to_delete[-1].data
            to_delete = to_delete[:-1]
        else:
            kept_last = None
            
        for elt in to_delete:
            # print "clear_boundary_by_alpha_range: deleting node %d (elt %s)"%(elt.data,elt)
            self.delete_node_and_merge(elt.data)

            # This used to assume elt was the only iter, but with internal_guides, we
            # might be removing two iters for one node.
            for each_elt in self.all_iters_for_node(elt.data):
                # print "  including unpaved iter %s"%each_elt
                self.push_op(self.unremove_from_unpaved,each_elt)
                each_elt.clist.remove_iters(each_elt)
        return kept_last

    def unremove_from_unpaved(self,elt):
        # this is a bit hairy, and may blow up, but worth a shot.
        # note that we can't just append, because that would create
        # a new iter and future undeletes wouldn't be able to find
        # this iter.  instead reuse this iter, and have the clist
        # simply paste it back in line.

        # let it reset the metric
        elt.clist.unremove_iter(elt)

    def unadd_to_unpaved(self,elt):
        elt.clist.remove_iters(elt)
        
    def undelete_node_paver(self, n, node_data):
        self.node_data[n] = node_data

    def delete_node(self,n,remove_edges=1):
        self.push_op(self.undelete_node_paver,n,self.node_data[n].copy())
        self.node_data[n,self.STAT] = self.DELETED
                     
        super(Paving,self).delete_node(n,remove_edges)

    def undelete_edge_unpaved_handler(self,added_clists,removed_clists,added_iters):
        """ a bit dicey... we'll see..
        """
        for clist in added_clists:
            self.unpaved.remove(clist)
        for clist in removed_clists:
            self.unpaved.append(clist)
        for it in added_iters:
            it.clist.remove_iters(it)
        
    def delete_edge_unpaved_handler(self,e,rollback,handle_unpaved):
        """ The Paving specific parts of delete_edge, for updating unpaved
        Called from delete_edge when handle_unpaved is set.
        """
        if self.verbose > 1:
            print("Patching unpaved on delete_edge")
            
        # Cases for cells:
        #  (a) both sides are unpaved => need to join the two unpaved rings
        #  (b) one side is unpaved => the 3rd node on the paved side gets inserted
        #      into unpaved
        #  (c) both sides are paved => create a new unpaved with these four nodes

        # The other cases don't involve unpaved, and so aren't considered:
        #  (d) one side is paved, the other is boundary => make the two other edges
        #      of the paved side boundary.  But what if one of those edges is already
        #      boundary? Caller's responsibility, but we can flag the edge in dangling_boundary_edges
        #  (e) one side is unpaved, the other exterior => Error.  This would imply
        #      that we're breaching an unpaved ring.  probably not what we'd want (would
        #      have to make the entire unpaved ring then into boundary.

        # Record what we do, so that it can be rolled back later.
        added_clists = []
        removed_clists = []
        added_iters = []
        
        left,right = self.edges[e,3:5]
        left_iter,right_iter = self.iters_for_edge(e)

        if left==trigrid.UNMESHED and right==trigrid.UNMESHED:

            left_clist = left_iter.clist
            right_clist = right_iter.clist

            new_clist = CList()
            # merge them -
            trav = right_iter.nxt
            while trav != right_iter:
                if trav == left_iter:
                    # so they are from the same ring, and removing this edge
                    # is going to create 2 new unpaved rings.
                    if len(new_clist):
                        self.unpaved.append(new_clist)
                        added_clists.append(new_clist)
                    
                    new_clist = CList()
                    break
                # starts with right.nxt (same *node* as left)
                new_clist.append( trav.data )

                trav = trav.nxt

                    
            trav = left_iter.nxt
            while trav != left_iter:
                if trav == right_iter:
                    # they were from the same ring - we're done.
                    break
                new_clist.append( trav.data )
                trav = trav.nxt
            if len(new_clist):
                self.unpaved.append(new_clist)
                added_clists.append(new_clist)
            
            # remove the old ones:
            self.unpaved.remove(left_clist)
            removed_clists.append( left_clist )
            if right_clist != left_clist:
                self.unpaved.remove(right_clist)
                removed_clists.append( right_clist )
                
        elif left==trigrid.UNMESHED and right>=0:
            node_c = np.setdiff1d(self.cells[right],self.edges[e,:2])[0]
            new_iter = left_iter.clist.append(node_c,after=left_iter)
            added_iters.append(new_iter)
        elif right==trigrid.UNMESHED and left>=0:
            node_c = np.setdiff1d(self.cells[left],self.edges[e,:2])[0]
            new_iter = right_iter.clist.append(node_c,after=right_iter)
            added_iters.append(new_iter)
        elif left>=0 and right>=0:
            node_c_right = np.setdiff1d(self.cells[right],self.edges[e,:2])[0]
            node_c_left  = np.setdiff1d(self.cells[left],self.edges[e,:2])[0]
            new_clist = CList()
            new_clist.append(self.edges[e,0])
            new_clist.append(node_c_right)
            new_clist.append(self.edges[e,1])
            new_clist.append(node_c_left)
            self.unpaved.append( new_clist )
            added_clists.append( new_clist )

        self.push_op(self.undelete_edge_unpaved_handler,added_clists,removed_clists,added_iters)
                     
            
    def undelete_edge_paver(self,e,e_data):
        self.edge_data[e] = e_data

        
    def delete_edge(self,e,rollback=1,handle_unpaved=0):
        if handle_unpaved:
            self.delete_edge_unpaved_handler(e,rollback,handle_unpaved)
        
        if rollback:
            self.push_op(self.undelete_edge_paver,e,self.edge_data[e].copy())

        # so far edge data is just the age, no need to clear that out,
        # but leave the undelete code in place for when edge data has
        # more
        # self.edge_data[e] = LEAVE IT

        super(Paving,self).delete_edge(e,rollback)
        


    def coarse_boundary_length(self, ring_i, alpha1, alpha2 ):
        """ given the two metrics (fractional indices into coarse_boundary points)
        find the length along the coarse boundary.
        """
        start_i = int(np.ceil(alpha1))
        end_i   = int(np.floor(alpha2))

        # those are indices into coarse_boundary
        # construct an index sequence that hits all of the points
        # in coarse_boundary that we need
        len_b = len(self.original_rings[ring_i])
        
        if alpha1 >= alpha2: # wraps around:
            end_i += len_b

        ilist = np.arange(start_i,end_i) % len_b

        coarse_points = self.original_rings[ring_i][ilist]

        # add fractional points at either end - possible that they
        # are duplicates, but easier to just let that go
        coarse_points = np.concatenate( ( [self.boundary_slider(ring_i, alpha1)],
                                          coarse_points,
                                          [self.boundary_slider(ring_i, alpha2)] ) )
        
        deltas = np.diff(coarse_points,axis=0)
        return np.sqrt((deltas**2).sum(axis=1)).sum()

    def elt_smallest_internal_angle(self,ring_i):
        while 1:
            e = self.unpaved[ring_i].iter_smallest_metric()
            if self.verbose > 2:
                print("HEAP: pulled %s with metric %f=%.2fdeg"%(e,
                                                                self.unpaved[ring_i].metric(e),
                                                                180*self.unpaved[ring_i].metric(e)/np.pi))
                
            if self.unpaved[ring_i].metric(e) == 0.0: #need to recompute the angle
                new_metric = self.internal_angle(apex=e)
                if self.prefer_original_ring_nodes and self.node_data[e.data,self.ORIG_RING]<0:
                    new_metric += 10
                self.unpaved[ring_i].update_metric(e,new_metric)
            else:
                if self.verbose > 2:
                    print("smallest angle chosen was: ",e)
                #print "Chose elt with metric=%f"%self.unpaved[ring_i].metric(e)
                #print "Smallest in heap is: %f"%min( self.unpaved[ring_i].heap.values() )
                return e

    def relax_around_cell(self,c,n_radius=2,use_beta=0):
        c_list = np.array([c])
        
        while n_radius > 0:
            new_c_list = c_list
            
            for cell in c_list:
                new_c_list = np.concatenate([new_c_list,self.cell_neighbors(cell)])
            c_list = np.unique(new_c_list)
            n_radius -= 1

        nbr_points = unique(self.cells[c_list].ravel())

        for n in nbr_points:
            self.safe_relax_one(n,use_beta=use_beta)

    def safe_relax_one(self,n,use_beta=0,max_cost=np.inf):
        stat = self.node_data[n,self.STAT]

        if stat == self.RIGID:
            return False
        elif stat == self.SLIDE:
            self.relax_one(n,boundary=True,use_beta=use_beta,max_cost=max_cost)
        elif stat == self.HINT:
            raise Exception("I don't think we should be relaxing HINT nodes.  Maybe forgot to update stat?")
        else:
            self.relax_one(n,max_cost=max_cost)
        
    def internal_angle(self, apex, do_plot=0):
        """ returns the internal angle on the unpaved boundary at the given
        apex node, which is an index into unpaved[ring_i]
        """
        
        seq = np.array( [apex.prv.data,apex.data,apex.nxt.data] )

        diffs = self.points[seq[1:]] - self.points[seq[:-1]]
        angles = np.arctan2( diffs[:,1], diffs[:,0] )
        # the difference gives the outside angle, so subtract from 180deg
        d_angle = (np.pi - (angles[1] - angles[0])) % (2*np.pi)

        # degenerate rings can have 360 degree angles - 
        if d_angle == 0.0:
            d_angle = 2*np.pi

        return d_angle
    
    def angle_to_left(self,pnts,do_plot=0):
        ab = pnts[1] - pnts[0]
        ab = ab/ norm(ab)
        bc = pnts[2] - pnts[1]
        bc = bc/norm(bc)

        ab_parallel = ab

        ab_perp = rot(np.pi/2,ab)

        theta = np.arctan2( np.dot(bc,ab_perp), np.dot(bc,ab_parallel) )

        int_angle = np.pi - theta
        
        if do_plot:
            plt.annotate("%.3g"%(180*int_angle/np.pi),pnts[1])

        return int_angle

    def internal_delaunay_neighbors(self,elt):
        """ return the DT neighbors of the given elt, sorted CCW, and
        limited to neighbors falling within the internal angle of the elt
        """
        n = elt.data
        dt_neighbors = self.delaunay_neighbors(n)
        dt_ordered = self.angle_sort_nodes(n,dt_neighbors)
        # we can narrow the list by excluding anything that wouldn't
        # subtend our angle (is that the right way to use 'subtend'?)
        nxt_i = np.where( dt_ordered==elt.nxt.data )[0][0]
        indices = (np.arange(len(dt_ordered)) + nxt_i)%len(dt_ordered)

        dt_ordered = dt_ordered[indices]

        if np.where( dt_ordered==elt.nxt.data )[0][0] != 0:
            raise Exception("silly modulo trick didn't work")

        # dt_ordered puts the nodes in CCW order, and now we have
        # center_elt.nxt as the first one.

        prv_i = np.where( dt_ordered==elt.prv.data )[0][0]

        dt_ordered = dt_ordered[1:prv_i]
        return dt_ordered


    def nonlocal_wall(self,left_elt,right_elt,local_scale,shoot_ray=1):
        # This used to just call nonlocal_bisector, but it can be
        # a bit smarter than that and try to find nodes that actually
        # make sense for a wall connection - 

        ## First: check on nodes that are dt_neighbors of both endpoints
        left_neighbors = self.internal_delaunay_neighbors(left_elt)
        right_neighbors = self.internal_delaunay_neighbors(right_elt)

        dt_ordered = [n for n in left_neighbors if n in right_neighbors]
        
        # If the actual length is longer than local_scale, use it
        scale_avg = np.sqrt( np.sum( (self.points[left_elt.data] - self.points[right_elt.data])**2))
        if scale_avg > local_scale:
            local_scale = scale_avg

        left_local_nodes = self.local_nodes(left_elt.data,2*local_scale,steps=2)
        right_local_nodes = self.local_nodes(right_elt.data,2*local_scale,steps=2)

        # Combine the two - 
        local_nodes = dict( left_local_nodes )
        for n,dist in right_local_nodes:
            if n not in local_nodes:
                local_nodes[n] = dist
            else:
                local_nodes[n] = min(local_nodes[n],dist)

        best_dist = np.inf
        best_nbr = None

        # Define vector along the edge, parallel to LR
        pR = self.points[right_elt.data]
        pL = self.points[left_elt.data]
        
        vec = pR - pL

        for nbr in dt_ordered:
            # Straight-line distance
            dist = max( norm( pR - self.points[nbr] ),
                        norm( pL - self.points[nbr] ) )
            # Distance along local edges
            if nbr in local_nodes:
                d_boundary = local_nodes[nbr]
                # for a perfect, equilateral, this would be 2.0
                # for a right-triangle, sqrt(2)
                if d_boundary / dist < 1.25:
                    log.debug("Discarding local node because boundary distance less than 25% longer than straight line")
                    continue
                else:
                    log.debug("d_boundary %d => %d = %g vs line %g"%(n,nbr,d_boundary,dist))
                    log.debug("CAREFUL: allowing a nonlocal connection to a nearby node, but okay(?) b/c of distances")

            # For bisectors:
            #   1.3 seemed to reach a bit far.
            #   1.0 was too short
            #   1.15 was too short in some situations. It's not clear
            # that going just from local_scale is right.
            if dist > best_dist or dist > 1.25*local_scale:
                continue

            # make sure that it falls within the perpendicular range of this edge -
            if np.dot(vec,self.points[nbr] - pL) < 0:
                # too far back
                continue
            if np.dot(vec,self.points[nbr] - pR) > 0:
                # too far ahead
                continue

            best_dist = dist
            best_nbr = nbr

        if self.verbose > 2:
            if best_nbr is not None:
                plt.annotate('NL:%d'%best_nbr,self.points[best_nbr])

        if best_nbr is None and shoot_ray:
            ## Try the ray shooting method
            #  here,knowing that the connection will be made to right_elt, 
            #  we shoot the ray from right elt, perpendicular to the edge
            
            # first get the direction -
            vec = np.array([-vec[1],vec[0]])

            # 1.5 seems too far - we don't currently optimize this point to be
            # any closer after finding it, and we are already looking at a local
            # scale that is larger than 1 if not both of the neighboring edges.
            # I'm trying 1.33 here...
            e,pnt = self.shoot_ray(right_elt.data,vec,1.33*local_scale)

            if e is not None:
                print("Ray shooting found a possible non-local wall connector")
                
                # have to make sure that the edge we found is a boundary edge that can
                # be divided or slid.  This means it must be unmeshed, and on the boundary
                # (maybe someday we'd allow to be unmeshed on both sides)
                if (self.edges[e,3] == trigrid.UNMESHED) and (self.edges[e,4] == trigrid.BOUNDARY):
                    # for now, ignore whether the nodes are free or not.
                    # there is also the issue that it needs to be on the boundary, but we found it by
                    # assuming the boundary is locally a straight line between the two nodes that are
                    # nearby.  could get some surprises...

                    # There may already be a point that is close enough - calculate distance
                    # from our intersecting point 
                    
                    da = norm( pnt-self.points[self.edges[e,0]])
                    db = norm( pnt-self.points[self.edges[e,1]])

                    # so if the point that we hit is within 10% of the local scale of an existing
                    # point, take that point even if it didn't pass the locality test above.
                    nbrs = [right_elt.nxt.data,right_elt.prv.data]
                    
                    close_fac=0.1 # could consider increasing this
                    if (da / local_scale < close_fac) and (self.edges[e,0] not in nbrs):
                        print("ray found an existing node %d"%self.edges[e,0])
                        return self.edges[e,0]
                    if (db / local_scale < close_fac) and (self.edges[e,1] not in nbrs):
                        print("ray found an existing node %d"%self.edges[e,1])
                        return self.edges[e,1]

                    # so resample_boundary will create the new node, but we need to find the right
                    # ring to put it on.
                    for b_iter in self.all_iters_for_node( self.edges[e,0] ):
                        if b_iter.nxt.data == self.edges[e,1]:
                            start_elt = b_iter
                            break

                    new_elt = self.resample_boundary(start_elt,'forward',da)
                    print("ray shooting created new node %d"%new_elt.data)

                    # And check to see if the new node is a DT neighbor - note that this can only be
                    # done if the DT is active - otherwise assume that it's okay...
                    if self.freeze or self.holding:
                        print("WARNING: DT is frozen or holding - cannot double check validity of nonlocal bisector")
                        return new_elt.data

                    # had been v, probably left over from recent edits
                    # pretty sure that should be new_elt.data
                    if new_elt.data in self.delaunay_neighbors(right_elt.data):
                        return new_elt.data
                    print("ray shooting created new node but it's not a DT neighbor.")

        return best_nbr


    def nonlocal_bisector(self,center_elt,local_scale,shoot_ray=1):
        """ Look for a nonlocal node approximately with local_scale of
        center_elt, and approximately bisecting the angle at center_elt.

        This will first consult the Delauanay triangulation and consider
        nodes that are connected to center_elt.

        shoot_ray: If no existing nodes are found via the DT, shoot a ray
          out of center_elt - if it intersects an edge that we can resample
          then add a node - but make sure that the new node shows up in the
          DT.
        """
        # only works if we have the live triangulation
        n = center_elt.data

        best_dist = np.inf
        best_nbr = None
        
        # If the actual lengths are longer than local_scale, use their average
        scale_left  = np.sqrt( np.sum( (self.points[center_elt.prv.data] - self.points[center_elt.data])**2))
        scale_right = np.sqrt( np.sum( (self.points[center_elt.nxt.data] - self.points[center_elt.data])**2))
        scale_avg = 0.5*(scale_left+scale_right)
        if scale_avg > local_scale:
            local_scale = scale_avg

        # get the list of nodes that the triangulation tells us
        # are good..
        dt_ordered = self.internal_delaunay_neighbors(center_elt)
        if len(dt_ordered) > 0:
            local_nodes = self.local_nodes(n,2*local_scale,steps=2)

            for nbr in dt_ordered:
                # Straight-line distance
                dist = norm( self.points[n] - self.points[nbr] )
                # Distance along local edges
                d_boundary = local_nodes[ local_nodes[:,0] == nbr,1]

                if len(d_boundary):
                    d_boundary = d_boundary[0]
                    # for a perfect, equilateral, this would be 2.0
                    # for a right-triangle, sqrt(2)
                    if d_boundary / dist < 1.25:
                        log.debug("Discarding local node because boundary distance less than 25% longer than straight line")
                        continue
                    else:
                        log.debug("d_boundary %d => %d = %g vs line %g"%(n,nbr,d_boundary,dist))
                        log.debug("CAREFUL: allowing a nonlocal connection to a nearby node, but okay(?) b/c of distances")


                # 1.3 seemed to reach a bit far.
                # 1.0 was too short
                # 1.15 was too short in some situations. It's not clear
                #  that going just from local_scale is right.
                if dist > best_dist or dist > 1.25*local_scale:
                    if self.verbose>1:
                        log.debug("Discarding nonlocal because relative distance is %.3f, best is %.3f"%(dist/local_scale,
                                                                                                         best_dist/local_scale))
                    continue

                best_dist = dist
                best_nbr = nbr

        if self.verbose > 2:
            if best_nbr is not None:
                plt.annotate('NL:%d'%best_nbr,self.points[best_nbr])

        if best_nbr is None and shoot_ray:
            ## Try the ray shooting method
            # first get the direction -
            a = self.points[center_elt.prv.data]
            b = self.points[center_elt.data]
            c = self.points[center_elt.nxt.data]

            theta_ba = np.arctan2(a[1]-b[1],a[0]-b[0])
            theta_bc = np.arctan2(c[1]-b[1],c[0]-b[0])
            
            theta = theta_bc + ((theta_ba - theta_bc)%(2*np.pi))/2.0
            vec = np.array([np.cos(theta),np.sin(theta)])

            # 1.5 seems too far - we don't currently optimize this point to be
            # any closer after finding it, and we are already looking at a local
            # scale that is larger than 1 if not both of the neighboring edges.
            # I'm trying 1.33 here...
            e,pnt = self.shoot_ray(center_elt.data,vec,1.33*local_scale)

            if e is not None:
                print("Ray shooting found a possible non-local bisector")
                
                # have to make sure that the edge we found is a boundary edge that can
                # be divided or slide.  This means it must be unmeshed, and on the boundary
                valid_external = (self.edges[e,3] == trigrid.UNMESHED) and (self.edges[e,4] == trigrid.BOUNDARY)
                valid_internal = (self.edges[e,3] == trigrid.UNMESHED) and (self.edges[e,4] == trigrid.UNMESHED) \
                                 and (self.edge_data[e,1] >= 0)
                
                if valid_external or valid_internal:
                    # for now, ignore whether the nodes are free or not.
                    # there is also the issue that it needs to be on the boundary, but we found it by
                    # assuming the boundary is locally a straight line between the two nodes that are
                    # nearby.  could get some surprises...

                    # There may already be a point that is close enough - calculate distance
                    # from our intersecting point 
                    
                    da = norm( pnt-self.points[self.edges[e,0]])
                    db = norm( pnt-self.points[self.edges[e,1]])

                    # so if the point that we hit is within 10% of the local scale of an existing
                    # point, take that point even if it didn't pass the locality test above.
                    nbrs = [center_elt.nxt.data,center_elt.prv.data]
                    close_fac=0.1 # larger??

                    if (da / local_scale < close_fac) and (self.edges[e,0] not in nbrs):
                        print("ray found an existing node %d"%self.edges[e,0])
                        return self.edges[e,0]
                    if (db / local_scale < close_fac) and (self.edges[e,1] not in nbrs):
                        print("ray found an existing node %d"%self.edges[e,1])
                        return self.edges[e,1]

                    # so resample_boundary will create the new node, but we need to find the right
                    # ring to put it on.
                    for b_iter in self.all_iters_for_node( self.edges[e,0] ):
                        if b_iter.nxt.data == self.edges[e,1]:
                            start_elt = b_iter
                            break

                    new_elt = self.resample_boundary(start_elt,'forward',da)
                    print("ray shooting created new node %d"%new_elt.data)

                    # And check to see if the new node is a DT neighbor - note that this can only be
                    # done if the DT is active - otherwise assume that it's okay...
                    if self.freeze or self.holding:
                        print("WARNING: DT is frozen or holding - cannot double check validity of nonlocal bisector")
                        return new_elt.data

                    if new_elt.data in self.delaunay_neighbors(center_elt.data):
                        return new_elt.data
                    print("ray shooting created new node but it's not a DT neighbor.")

        return best_nbr
            

    def fill(self,center_elt,use_beta=0):
        n_edges = 3 # new code always considers exactly 3

        theta = self.internal_angle(apex=center_elt)

        # in-order node indices of us and immediate neighbors
        ordered = np.array( [center_elt.prv.data,center_elt.data,center_elt.nxt.data] )

        local_length = self.density( self.points[ordered[1],:2] )

        edge_diffs = self.points[ordered[:-1]] - self.points[ordered[1:]]

        edge_lengths = np.sqrt( (edge_diffs**2).sum(axis=1) )
        scale_factor = edge_lengths.mean() / local_length

        unpaved = center_elt.clist

        if self.verbose > 1:
            print("Desired length: ",local_length)
            print("Current edge length: ",edge_lengths)
            print("Scale factor: ",scale_factor)
            print("theta: %f"%(theta*180/np.pi))
            print("len(unpaved): ",len(unpaved))
            if len(unpaved) < 10:
                print("  nodes:",unpaved.to_array())

        # Strategies:
        #   wall: one segment, one point. scale neutral
        #   bisect: two segments, add one point approx. bisecting the angle
        #   cutoff: two segments, join the endpoints with one new segment

        join_okay = (self.FREE in [self.node_data[ordered[0],self.STAT],self.node_data[ordered[2],self.STAT]])

        # figure out the prioritized list of possible next cells:
        if len(unpaved)==4:
            if self.verbose > 0:
                print("Forcing a cutoff because only 4 nodes left on unpaved")
                print("  or it could be a join, if you're really unlucky")
            # if this fails, it's up to choose_and_fill to notice that we're on a 4-ring,
            # and to try again with a shift.
            
            # EXTRA: make sure the 4th node on the ring is at least SLIDE
            #   this could be done right before the optimization, too.
            #   make sure that all the nodes that made cells with are at least
            #   SLIDE.  Not sure which would be better.  If we do it here, then
            #   I don't think we have to worry about reverting it.  In a 4-node ring,
            #   sooner or later the 4th guy is going to get set to SLIDE 
            fourth = center_elt.nxt.nxt.data
            if self.node_data[fourth,self.STAT] == self.HINT:
                if self.verbose > 1:
                    print("Making 4th node on ring into SLIDE")
                self.node_data[fourth,self.STAT] = self.SLIDE
            
            strategies = ['cutoff','join']
        elif len(unpaved)==3:
            # very rare - but sometimes the resampling can take a quad or larger poly and
            # turn it into a triangle by the time we get to this point
            # but in that case it hasn't been optimized, and may not even be a cell
            # yet.
            strategies = ['close']
                
        # 160 - (scale_factor-1.)*20 seemed a bit inadequate
        # 160 - (scale_factor-1.)*30 has worked well, but in the full-bay grid allows some growth
        # see how bad it gets if we go to *50 - wall is really just breaking the bisect process
        # into two steps, so results are probably robust against most changes here. 
        elif theta > np.pi/180. * ( 160. - (scale_factor - 1.0)*50):
            strategies = ['wall']
        # if we're getting too fine, scale_factor is small, so we
        # push towards cutoffs over bisection
        # New change: 1.5 exponent on scale factor.  probably should just be linear.
        # 6/15/2010: trying linear, in hopes that it won't be so stiff when transitioning.
        # 7/8/2010: go back to 1.5 - it's too loose now, and while oscillations can be dealt with by
        #   repaving, regions of different scale are very hard to join.
        elif theta > 85*np.pi/180. * ( 1.0 / scale_factor )**1.5:
            strategies = ['bisect','cutoff','wall']
        elif theta > 30*np.pi/180.:
            # in general, we want to try a cutoff
            strategies = ['cutoff','join','bisect']

            # BUT: if we're up against a wall, try to bisect before cutoff
            # need to test whether ordered[1] is on the boundary
            on_boundary = self.node_on_boundary(ordered[1])

            if on_boundary:
                edges = self.pnt2edges(ordered[1])
                current_degree = len(edges)

                # the specific case I'm after right now: node is degree 3, boundary angle
                # is close to 2*max_angle.  then *don't* cutoff
                if current_degree == 3:
                    # This is the internal angle formed by the edges that lie along the
                    # boundary.
                    boundary_angle = self.boundary_angle(ordered[1])
                    # e.g. if the boundary angle is 170deg and max_angle is 85deg
                    #   then with two cells it hits the threshold exactly.  two cells
                    #   implies 3 edges, so the node has to be at least degree 3.
                    # of course that assumes that we can exactly bisect the angle,
                    # which we won't, so include some wiggle room - wiggle.
                    wiggle = 5*np.pi/180.
                    min_degree = 1.0 + boundary_angle / (self.max_angle - wiggle)

                    # more of the example: the angle is 170, but we specify 5 degrees
                    # of wiggle room with 85 degree max_angle.  so 170/(85-5)=2.125
                    # and min_degree = 3.125
                    # we're coming into this with 3 nodes already.  A cutoff will
                    # preserve the degree, 

                    if current_degree < min_degree:
                        if self.verbose > 2:
                            print("HEY! giving bisect preference because of a boundary angle")
                        strategies = ['bisect','cutoff','join']                        
        else:
            strategies = ['join','cutoff','bisect']

        if not join_okay and 'join' in strategies:
            strategies.remove('join')

        if self.nonlocal_method & self.PROACTIVE_DELAUNAY:
            new_strategies = []
            for s in strategies:
                if s == 'bisect':
                    new_strategies += ['bisect_nonlocal','bisect']
                elif s == 'wall':
                    new_strategies += ['wall_nonlocal','wall']
                else:
                    new_strategies.append(s)
            strategies=new_strategies

        boundary_diffs = None # a record of the changes that will need to be made to unpaved

        if self.verbose > 1:
            print("theta=%f  strategies=%s"%(theta*180/np.pi, strategies))

        # currently it is leaving an extra edge, and that edge along with 0 and 1
        #  have had their -2 flags set to 0

        self.last_fill_iter = center_elt
        self.last_strategies = strategies

        while len(strategies):  #  for strategy in strategies:
            strategy = strategies.pop(0)

            # a function that is run after relaxation - if it returns false,
            # something is amiss and the strategy is aborted
            post_relax_check = None
            preemptive = False
            
            try:
                if self.verbose > 1:
                    print("strategy: %s, nodes=%d %d %d"%(strategy,ordered[0],ordered[1],ordered[2]))

                # assume clean coming in, and that the rollback is successful
                # so on successive strategies start with no updated_cells
                self.updated_cells = [] 

                checkpoint = self.checkpoint()

                strategy_failed = 0 # set to 1 when a strategy fails to complete

                if strategy == 'cutoff':
                    # It's possible for the cutoff to intersect existing edges in
                    # tight conditions - check and fail if that's the case:
                    if len(self.check_line_is_clear(ordered[0],ordered[2])) > 0:
                        print("EASY There.  Cutoff wasn't clear - will try something else")
                        print("Will make sure that bisect_nonlocal is tried next")
                        strategies.insert(0,'bisect_nonlocal')
                        raise StrategyFailed('Cutoff intersected existing edges')
                    # if stubs are removed, need to use the new center_elt
                    # in order to get the right topology
                    # hopefully it's not problematic to be changing this in the middle
                    # of fill()
                    center_elt_mod=self.prepare_cutoff_stubs(center_elt)
                    unpaved_mod=center_elt_mod.clist
                    
                    tmp_edge = self.add_edge( ordered[0], ordered[2])
                    self.updated_cells += self.cells_from_last_new_edge

                    # if these edges do work out, this will be the new unpaved:
                    if len(unpaved) == 4:
                        # remove all of them...
                        new_unpaved = [(unpaved_mod.clear, )]
                    else:
                        new_unpaved = [(unpaved_mod.remove_iters, center_elt_mod),
                                       (unpaved_mod.update_metric, center_elt_mod.prv, 0.0),
                                       (unpaved_mod.update_metric, center_elt_mod.nxt, 0.0)]

                    if self.verbose > 2:
                        print("Did a cutoff, and the update cells are: ",self.updated_cells)
                elif strategy == 'close':
                    # The cell is basically there -
                    # for development purposes - make a few extra checks.  plus this is rare,
                    # so no worry about it being slower.

                    if self.verbose > 0:
                        print("processing a 'close' strategy on ",ordered)
                    
                    # will throw an exception if the sanity check fails:
                    j = self.find_edge( (ordered[0],ordered[2]) )

                    try:
                        i = self.find_cell( ordered )
                    except trigrid.NoSuchCellError:
                        i = None

                    if i is None:
                        i = self.add_cell( ordered )
                    # not sure what should go here - maybe we don't need to do anything,
                    # since at this point this unpaved ring has 3 nodes, and will be skipped
                    # in the future.
                    new_unpaved = []
                    
                    # Either way, go ahead and include it for optimization:
                    self.updated_cells.append(i)
                        
                elif strategy == 'bisect_nonlocal':
                    n = self.nonlocal_bisector(center_elt,local_length)
                    if n is None:
                        if self.verbose > 1:
                            print("No good bisector from %s, length=%g"%(str(center_elt),local_length))
                        raise StrategyFailed('No good bisector found from Delaunay')

                    log.debug("bisect_nonlocal is about to resample preemptively")
                    for it in self.all_iters_for_node(n):
                        # not entirely sure whether this should be HINT or SLIDE...
                        # used to be HINT, but it was possible to revert a SLIDE node
                        # into a HINT node that way..
                        self.resample_neighbors(it,self.SLIDE)
                        
                    log.debug("bisect_nonlocal is about to add an edge")
                    e=self.add_edge(center_elt.data,n)
                    self.updated_cells += self.cells_from_last_new_edge
                    log.debug("bisect_nonlocal got cells %s from new edge"%(self.cells_from_last_new_edge))
                    
                    # I was hoping to do this via new_unpaved, but the relaxing
                    # comes first, and we need to toggle these statuses before relaxing.
                    if self.node_data[n,self.STAT] == self.HINT:
                        self.node_data[n,self.STAT] = self.SLIDE
                    
                    # not sure about the resample stuff - it does need to be before
                    # update_unpaved_boundaries, though, b/c center_elt is not going
                    # to be meaningful after that call.
                    # check_nonlocal was all about resampling everybody, though.
                    # I'm not sure why I didn't before put some resampling in here.
                    # it's currently getting thrown off by a non-local edge being
                    # cramped during a subsequent join and relax
                    new_unpaved = []

                    # Was there a reason for not doing this before?
                    for n_iter in self.all_iters_for_node(n):
                        new_unpaved.append( (self.resample_neighbors,n_iter,self.SLIDE) )

                    new_unpaved.append( (self.update_unpaved_boundaries,e) )
                    
                elif strategy == 'bisect':
                    # choose a point based just on the first three ordered vertices
                    pnts = self.points[ordered[:3],:2]

                    bisect_angle = theta / 2.0

                    # New check to use the shorter of two edges for the bisect point
                    # Used to always used BC, the first case here.
                    if norm(pnts[2] - pnts[1]) < norm(pnts[0]-pnts[1]):
                        # takes the length from BC, rotates through to bisect the angle.
                        newB = pnts[1] + rot(bisect_angle,pnts[2] - pnts[1])
                    else:
                        newB = pnts[1] + rot(-bisect_angle,pnts[0] - pnts[1])

                    new_node = self.add_node(newB,stat=self.FREE)
                    
                    # and we just created edges.  If we're closing
                    # the ring then the first node in ordered is repeated and
                    # we don't want to add that edge twice:
                    if ordered[0] == ordered[-1]:
                        # This hasn't be double checked for node_is_nonlocal
                        # not sure that this even gets called ever.
                        n_new_edges = len(ordered) - 1
                    else:
                        n_new_edges = len(ordered)


                    # 2019-04-28: consider pre-testing these
                    for i in range(n_new_edges):
                        if len(self.check_line_is_clear(ordered[i], new_node ))>0:
                            print("EASY There.  Bisect wasn't clear")
                            raise StrategyFailed('Bisect intersected existing edges')
                    # /2019-04-28
                               
                    for i in range(n_new_edges):
                        self.add_edge( ordered[i], new_node )
                        self.updated_cells += self.cells_from_last_new_edge 

                    # replace the old node with the new node - be sure to do this
                    # after getting the edges taken care of.

                    if len(ordered) != 3:
                        raise Exception("How can ordered not be three nodes for a bisect?")
                    else:
                        new_unpaved = [(unpaved.update_metric,center_elt.prv,0.0),
                                       (unpaved.update_metric,center_elt.nxt,0.0),
                                       (unpaved.remove_iters,center_elt),
                                       (unpaved.append,new_node, center_elt.prv),
                                       ]
                elif strategy == 'wall_nonlocal':
                    # Taken mostly from bisect_nonlocal - see that for more notes
                    n = self.nonlocal_wall(center_elt.prv,center_elt,local_length)
                    if n is None:
                        raise StrategyFailed('No good wall point from Delaunay')

                    print("HEY - wall_nonlocal got a hit")
                    
                    for it in self.all_iters_for_node(n):
                        # not entirely sure whether this should be HINT or SLIDE...
                        # see notes in bisect_nonlocal
                        self.resample_neighbors(it,self.SLIDE)

                    e=self.add_edge(center_elt.data,n)
                    self.updated_cells += self.cells_from_last_new_edge
                    
                    if self.node_data[n,self.STAT] == self.HINT:
                        self.node_data[n,self.STAT] = self.SLIDE
                    
                    new_unpaved = []

                    for n_iter in self.all_iters_for_node(n):
                        new_unpaved.append( (self.resample_neighbors,n_iter,self.SLIDE) )

                    new_unpaved.append( (self.update_unpaved_boundaries,e) )

                elif strategy == 'wall':
                    # only look at the first two points:
                    pnts = self.points[ordered[:2],:2]

                    # takes the length from AB, rotate through 90, scale down to
                    # make the triangle equi
                    newB = pnts[0] + rot(60*np.pi/180.0,pnts[1] - pnts[0])

                    new_node = self.add_node(newB,stat=self.FREE)

                    # and we just created edges:
                    self.add_edge(ordered[0], new_node)
                    self.add_edge(ordered[1], new_node)
                    self.updated_cells += self.cells_from_last_new_edge

                    new_unpaved = [(unpaved.append,new_node,center_elt.prv),
                                   (unpaved.update_metric,center_elt.prv,0.0),
                                   (unpaved.update_metric,center_elt,0.0)]
                elif strategy == 'join':
                    if self.verbose > 1:
                        print("hang on - trying a join")

                    # goal is to merge ordered[0] and ordered[2]
                    # only valid if at least one of them is FREE
                    # choose the one that will get deleted:
                    a,b,c = ordered

                    to_delete = None

                    # here at_risk is the node from to_delete closest to the gap
                    # and at risk of intersecting neighbors of to_keep.
                    # intervener is the node adjacent to to_keep that is the first
                    # in line to cause a self-intersection with at_risk
                    if self.node_data[a,self.STAT] == self.FREE:
                        to_delete = a
                        to_keep = c
                        at_risk = center_elt.prv.prv.data
                        intervener = center_elt.nxt.nxt.data
                        keep_dir = 'forward'
                    elif self.node_data[c,self.STAT] == self.FREE:
                        to_delete = c
                        to_keep = a
                        at_risk = center_elt.nxt.nxt.data
                        intervener = center_elt.prv.prv.data
                        keep_dir = 'backward'
                    else:
                        raise StrategyFailed("Joining cannot work - neither node is free")

                    # It's possible that the edge from to_keep to the first neighbor
                    # of to_delete is not free of the edges from to_keep to its current
                    # neighbors.
                    # The most immediate solution is to resample the boundary leaving
                    # to_keep.  This only makes sense when to_keep is a boundary node
                    if self.node_on_boundary(to_keep):
                        at_risk_length = norm( self.points[at_risk] - self.points[to_keep] )
                        # the problem is that a neighbor of to_keep may be inside the
                        # triangle at_risk-to_keep-to_delete, which is the region that we
                        # are about to "annex" by moving edges from to_delete to to_keep.
                        # walk away from to_keep at least at_risk_length, and see if any
                        # nodes are inside the annex.  if they are, we have to either remove
                        # them or fail the join.
                        elts_to_delete = []
                        dist_to_resample = 0.0
                        
                        if keep_dir == 'forward':
                            annex = self.points[ [to_keep,at_risk,to_delete] ] # CCW
                            trav = center_elt.nxt.nxt
                            while 1:
                                d = norm( self.points[trav.data] - self.points[to_keep] ) 
                                if d >= at_risk_length:
                                    break # far enough away to be safe
                                if trav.data == at_risk:
                                    break # closed loop - should be safe.
                                if point_in_triangle(self.points[trav.data],annex):
                                    elts_to_delete.append(trav) # NB: may not be able to delete this...
                                    dist_to_resample = max(dist_to_resample,d)
                                trav = trav.nxt
                        else:
                            annex = self.points[ [to_keep,to_delete,at_risk] ] # CCW
                            trav = center_elt.prv.prv # aka intervener
                            while 1:
                                d = norm( self.points[trav.data] - self.points[to_keep] ) 
                                if d >= at_risk_length:
                                    break
                                # in tight quarters, possible for the loop to be closed -
                                # should be fine if we reach that..
                                if trav.data == at_risk:
                                    break
                                if point_in_triangle(self.points[trav.data],annex):
                                    elts_to_delete.append(trav)
                                    dist_to_resample = max(dist_to_resample,d)
                                trav = trav.prv

                        # check for any that cannot be deleted - if so, FAIL
                        undeletable = []
                        for etd in elts_to_delete:
                            # This could possibly be more aggressive - allowing the deletion
                            # of any degree 2 node.  for now, though, I want to use resample_boundary
                            # and that means 
                            if not self.node_on_boundary(etd.data) or len(self.pnt2edges(etd.data)) != 2:
                                undeletable.append(etd.data)

                        if len(undeletable) > 0:
                            print("Was trying a join, found nodes that needed to be cleared out")
                            print(", ".join([str(e) for e in elts_to_delete]))
                            print("But some are not valid for deletion: ")
                            print(undeletable)
                            raise StrategyFailed("Couldn't clear nodes for a safe join")
                        elif dist_to_resample > 0.0:
                            # really just want to clear out all nodes up to and including the
                            # last of elts_to_delete
                            last_alpha = self.node_data[elts_to_delete[-1].data,self.ALPHA]
                            
                            print("Okay - found some conflicting nodes ahead of a join.  Will clear them out")
                            for e in elts_to_delete:
                                print(str(e))
                                
                            if keep_dir == 'forward':
                                self.close_free_boundary(center_elt.nxt,
                                                         elts_to_delete[-1].nxt,
                                                         'forward')
                            else:
                                self.close_free_boundary(center_elt.prv,
                                                         elts_to_delete[-1].prv,
                                                         'backward')
                            
                    # figure out its neighbors.  Do the same for the neighbors of to_keep
                    # since we need that to check the angle-sort of all of the eventual
                    # neighbors of to_keep
                    td_nbrs = self.edges[self.pnt2edges(to_delete),:2].ravel()
                    tk_nbrs = self.edges[self.pnt2edges(to_keep),:2].ravel()

                    # This is all the neighbors of the node that will be deleted
                    td_nbrs = np.setdiff1d(td_nbrs,[to_delete])
                    tk_nbrs = np.setdiff1d(tk_nbrs,[to_keep,b]) # remove common nbr, too

                    # get the starting angle-order for all_nbrs, starting with b
                    # in the old (<1/10/2013) code, all_nbrs was allowed to contain
                    # a duplicate, if intervener and at_risk were the same.  to allow
                    # for double joins, that's no good.  so check, and note
                    # it in double_join
                    if at_risk == intervener:
                        double_join = True
                        all_nbrs = np.concatenate( (np.setdiff1d(td_nbrs,[at_risk]),
                                                    tk_nbrs) )
                    else:
                        double_join = False
                        all_nbrs = np.concatenate( (td_nbrs,tk_nbrs) )

                    all_nbrs,all_angles = self.angle_sort_nodes(to_keep,all_nbrs,return_angles=True)

                    # Check right away for any duplicate angles - these probably mean that
                    # some pair of potential nbrs are colinear with to_keep, and that means trouble.
                    # but it's a bit more subtle than that - an angle may be repeated because of a
                    # repeated neighbor - i.e. when intervener and at_risk are the same.
                    # this is necessary in addition to the angle sort test below because often
                    # a number of nodes are lined up on a single straight edge of the boundary,
                    if np.any( np.diff(all_angles)== 0.0 ):
                        raise StrategyFailed("join would form degenerate triangle")
                    
                    bi = np.where( all_nbrs == b)[0][0]
                    all_nbrs = np.concatenate( (all_nbrs[bi:],all_nbrs[:bi]) )
                    all_angles = np.concatenate( (all_angles[bi:],all_angles[:bi]) )

                    if self.verbose > 1:
                        print("angle-sorted neighbors before join: ",all_nbrs)
                        print("                            angles: ",all_angles)

                    def check_relaxation():
                        sorted_nbrs = self.angle_sort_nodes(to_keep,all_nbrs)
                        bi = np.where( sorted_nbrs == b )[0][0]
                        new_all_nbrs = np.concatenate( (sorted_nbrs[bi:],sorted_nbrs[:bi]) )
                        if self.verbose > 1:
                            print("Checking the relaxation")
                            print("---Comparing---")
                            print(new_all_nbrs)
                            print(all_nbrs)
                            print("---")
                        return np.all(new_all_nbrs==all_nbrs)
                    post_relax_check = check_relaxation

                    ## Trying a stricter test for angles -
                    #  using intervener to identify bad angles is problematic because in some
                    #  cases intervener == b, and the test cannot be completed.  There are also
                    #  cases where the intervener is on a really short leash
                    #  This test may be overly strict - not really sure yet...
                    #  It enforces that the angle sort of all_nbrs is unchanged whether taken from
                    #  a or c.
                    #  we already have all_nbrs from A
                    if 1:
                        all_nbrs_from_tk = self.angle_sort_nodes(to_delete,all_nbrs,return_angles=False)
                        bi = np.nonzero( all_nbrs_from_tk == b)[0][0]
                        all_nbrs_from_tk = np.concatenate( (all_nbrs_from_tk[bi:], all_nbrs_from_tk[:bi]) )
                        if np.any( all_nbrs_from_tk != all_nbrs ):
                            if self.verbose > 1:
                                print("New angle sorting check raised the red flag")
                                self.plot()
                                self.plot_nodes([to_keep,to_delete] + all_nbrs.tolist())
                            raise StrategyFailed("join would alter the angle sort")

                    if self.verbose > 2:
                        print("joining - need to move these neighbors over:",td_nbrs)
                        self.plot('step %d: about to join'%self.step)

                    for nbr in td_nbrs:
                        e = self.find_edge([to_delete,nbr])
                        if self.verbose > 1:
                            print("for join, delete_edge(%d) nodes %d-%d"%(e,self.edges[e,0],self.edges[e,1]))
                        self.delete_edge(e)
                    self.delete_node(to_delete)

                    if self.verbose > 2:
                        self.plot('step %d: cleared edges, about to add new'%self.step)

                    for nbr in td_nbrs:
                        # the edge back to b is not moved, just deleted
                        # Also to_delete and to_keep may share a neighbor
                        # in this direction, too (i.e. a small closed loop).
                        # so don't add that edge, either.
                        if nbr != b and nbr not in tk_nbrs:
                            self.add_edge(nbr,to_keep)
                            self.updated_cells += self.cells_from_last_new_edge

                    # And then the changes to unpaved -
                    # for the original ring, we remove both the center_elt and to_delete.
                    # for any other iters that to_delete had, replace to_delete with to_keep

                    # we have now removed both b and to_delete
                    if center_elt.nxt.data == to_delete:
                        elt_to_delete = center_elt.nxt
                    elif center_elt.prv.data == to_delete:
                        elt_to_delete = center_elt.prv
                    else:
                        raise Exception("now where did that iter go?")

                    new_unpaved = [(unpaved.remove_iters,center_elt,elt_to_delete)]

                    # handle other iters that to_delete has:
                    for it in self.all_iters_for_node(to_delete):
                        # if it is the one we just dealt with, don't worry about it.
                        # be careful about testing equality though -
                        if it.nxt == center_elt or it.prv == center_elt:
                            # this is the one we've already handled
                            continue
                        print("Join: updating iter %s to replace %d with %d"%(it,to_delete,to_keep))
                        # replace to_delete with to_keep
                        new_unpaved.append( (it.clist.append, to_keep, it) )
                        new_unpaved.append( (it.clist.remove_iters, it) )
                else:
                    raise Exception("Unknown strategy %s"%strategy)


                # up to 4 times, tweak all the nodes in the neighborhood.
                # first we tweak only the nodes of newly created cells.
                # those tweaks then change some points, and if necessary
                # we look for ways to change those points and their neighbors
                #   on entry to the loop, self.updated_cells is a list of cell ids
                #   that have been tweaked/created/etc. above.
                # on each loops we take updated_cells -> nodes, tune those nodes,
                #   and for nodes that we moved, add back into updated_cells their
                #   adjacent cells.
                # in this way each loop covers one more radius

                if self.verbose > 2:
                    self.plot(plot_title="strategy %s, about to optimize"%strategy )

                # self.hold() # don't update the DT while relaxing.
                try:
                    for i in range(self.relaxation_iterations):
                        cells_to_check = np.unique( np.array( self.updated_cells,np.int32) )
                        nodes_to_check = np.unique( self.cells[cells_to_check,:] )

                        if self.verbose > 1: ## pre-check
                            cells_to_verify = cells_to_check

                            print(" Checking cells *before* more optimization: ",cells_to_verify)

                            angles = self.tri_angles(cells_to_verify)

                            failed = cells_to_verify[ np.any(angles>self.max_angle,axis=1) ]

                            print(" precheck bad cells: ",failed)

                        # not sure what the best ordering is here, but probably
                        # not random.
                        #   relaxing old nodes first didn't seem that great.  Large
                        #   triangles would crop up a lot...  overall it does better
                        #   when relaxing new nodes first, then old
                        nodes_to_check = sorted(nodes_to_check)[::-1]

                        for n in nodes_to_check:
                            if n < 0:
                                print("relaxation round=%d"%(i))
                                print("cells to check",cells_to_check)
                                raise Exception("somehow a negative node got in here")

                            cost = self.cost_for_point(n) # accounts for all adjacent cells.
                            if cost < self.cost_threshold:
                                pass
                            else:
                                if self.verbose > 0:
                                    sys.stdout.write('.') ; sys.stdout.flush()

                                # note that at this point we've added new edges but haven't
                                # updated self.unpaved, so things are not entirely consistent
                                cp = self.checkpoint()
                                # the 200.0 here is just a guess....
                                # this used give max_cost=min(cost,200.0)
                                # but in some bad situations that disallows intermediate
                                # bad-but-less-bad situations.
                                # what's the problem with just saying that it has to improve
                                # the cost?
                                self.safe_relax_one(n,use_beta=use_beta,max_cost=cost) # min(cost,200.0) )

                                if post_relax_check is not None and not post_relax_check():
                                    print("Post-relax check failed.  Reverting the relaxation")
                                    # Note: as it stands, this will leave any fake vertices
                                    # in the triangulation which will almost certainly cause problems
                                    # down the line.
                                    self.revert(cp)
                                else:
                                    # and make a note that we altered a cell:
                                    self.updated_cells += list( self.pnt2cells(n) )

                            # check all edges adjacent to this node to see if they
                            # intersect.  For now don't worry about the possibility of
                            # checking an edge twice - the above may have moved this node
                            # and we'd need to check it twice anyway.
                            for e in self.pnt2edges(n):
                                # print "Checking edge %d from node %dfor self-intersections"%(e,n)
                                ## DANGER: this is really slow on large inputs, so I'm disabling it
                                #  but need to find a way to check this more efficiently.
                                violations = None # self.check_edge_against_boundary(e)
                                if violations is not None:
                                    raise Exception("Self-intersection!")

                        # check angles for any cells that were just tweaked, and including any
                        cells_to_verify = np.union1d(cells_to_check,self.updated_cells)

                        if self.verbose > 1:
                            print("Cells to verify: ",cells_to_verify)

                        angles = self.tri_angles(cells_to_verify)

                        if np.any(angles > self.max_angle):
                            if self.verbose > 1:
                                failed = cells_to_verify[ np.any(angles>self.max_angle,axis=1) ]
                                print(" after optimization: bad cells: ",failed)
                            if self.verbose > 2:
                                self.plot( plot_title="after optimization, still have some bad cells" )

                            sys.stdout.write('!') ; sys.stdout.flush()
                            all_okay = 0
                        else:
                            all_okay = 1
                            break
                finally:
                    # Is this the right time to release?
                    # self.release()
                    pass
            except StrategyFailed as exc:
                preemptive=True
                if self.verbose > 1:
                    print("preemptive fail.  will revert and try next strategy")
                if self.verbose > 2:
                    print("Preemptive fail details:")
                    print(exc)
                all_okay = 0

            if not all_okay:
                if self.verbose > 2:
                    # sometimes this will fail because things are in an
                    # inconsistent state.
                    try:
                        self.plot(plot_title='%s failed, about to revert'%strategy)
                    except:
                        print("plot failed, probably we're in an inconsistent state")

                if not preemptive and self.verbose > 0:
                    print("%s failed.  Reverting"%strategy)
                self.revert(checkpoint)

                if self.verbose > 2:
                    self.plot(plot_title='%s failed, just reverted'%strategy )
            else:
                # print "Excellent. moving on"
                for action in new_unpaved:
                    action[0](*action[1:])
                break

        self.commit()

        if not all_okay:
            print("fill() failed, after all strategies were tried")
            return False
        else:
            if self.verbose > 2:
                self.plot(plot_title='step %d: about to call post_fill'%self.step)

            # Try the post-fill non-local connection here:
            self.post_fill(ordered=ordered,
                           modified_nodes=nodes_to_check)

            return True

    def plot_last_fill(self):
        """ some helpful (?) plots after fill() 
        """
        center_elt = self.last_fill_iter
        if center_elt is None:
            print("No data for last fill")
            return
        print("Strategies were ",self.last_strategies)
        n = center_elt.data
        scale = self.density( self.points[n] )
        ctr =   self.points[n]
        
        self.default_clip = ( ctr[0] - 20*scale, ctr[0] + 20*scale,
                              ctr[1] - 20*scale, ctr[1] + 20*scale )
        self.plot()
        
        self.plot_nodes( [center_elt.prv.data,center_elt.data,center_elt.nxt.data] )

        plt.axis('equal')


    def post_fill(self,ordered,modified_nodes):
        #print "Top of post_fill"

        # for non-local to work, the boundary needs to be resampled
        # print "Resampling neighbors after fill: ",ordered[0], ordered[-1]

        # try HINTing these, so that they can still be moved around freely
        # now that one node can be in multiple rings, and multiple times in
        # a ring, this gets a bit more complicated.  Before calling nonlocal,
        # though, need to be sure that there aren't any nodes sitting nearby
        # and connected that will appear to be good non-local candidates

        nodes_to_resample = [ordered[0],ordered[-1]]
        for node in nodes_to_resample:
            # print "Looking for node %d"%node
            # self.plot(plot_title="post_fill: about to resample node %d"%node)
            for elt in self.all_iters_for_node(node):
                # print "  post_fill: around iter %s"%elt
                self.resample_neighbors(elt,new_node_stat = self.HINT )
                # self.plot(plot_title="post_fill: resampled elt %s"%elt)


        # The old resampling was easier:
        # self.resample_neighbors(ordered[0],new_node_stat = self.HINT)
        # self.resample_neighbors(ordered[-1],new_node_stat = self.HINT)

        if self.nonlocal_method & self.SHEEPISH:
            if self.verbose > 0:
                print("Checking non-local connections to nodes: ",modified_nodes)

            for n in modified_nodes:
                self.check_nonlocal_connection(n)
        # print "End of post_fill"


    def prepare_cutoff_stubs(self,elt):
        """ In the presence of degenerate edges, it's possible to shoot for a cutoff,
        but wind up with a stub poking into the resulting cell.  This function
        will check for and remove those stubs, ahead of fill() trying to make the cutoff
        returns an updated iter to replace elt
        """
        a=elt.prv.data
        b=elt.data
        c=elt.nxt.data

        tri=geo.Polygon(self.points[ [a,b,c] ])
        def n_inside(n):
            return tri.contains( geo.Point(self.points[n,0],self.points[n,1]) )

        pairs_to_del=[] # node index pairs, with the first being the expendable one

        if self.node_data[a][1] in self.degenerate_rings:
            trav=elt.prv.prv
            while trav.data!=a and n_inside(trav.data):
                pairs_to_del.append([trav.data,trav.nxt.data])
                if trav.nxt.data==trav.prv.data:
                    break
                trav=trav.prv

        if self.node_data[c][1] in self.degenerate_rings:
            trav=elt.nxt.nxt
            while trav.data!=c and n_inside(trav.data):
                pairs_to_del.append([trav.data,trav.prv.data])
                if trav.nxt.data==trav.prv.data:
                    break
                trav=trav.nxt
        
        for n1,n2 in pairs_to_del[::-1]:
            print(" Deleting stub edges inside cutoff [%d-%d]"%(n1,n2))
            j=self.find_edge([n1,n2])
            self.delete_edge(j,handle_unpaved=1)
            self.delete_node(n1)
        if len(pairs_to_del):
            # have to get an updated elt
            for elt in self.all_iters_for_node(b):
                if elt.prv.data==a:
                    break
            else:
                raise Exception("how did you fall out of that loop?")
        return elt

    def check_nonlocal_connection(self,n):
        if self.verbose > 1:
            print("Checking for non-local connections to node %d"%n)

        # query enough of the closest points to get the nearest that
        # is not connected to us.  
        nearest = self.closest_point(self.points[n],10)

        if self.verbose > 1:
            print("Got %d closest nodes"%len(nearest))
        
        not_connected = []
        
        for other in nearest:
            if self.node_data[other,self.STAT] == self.DELETED:
                continue
            try:
                self.find_edge([n,other])
            except trigrid.NoSuchEdgeError:
                not_connected.append(other)

        # print "%d of them are not connected to us"%len(not_connected)

        if len(not_connected) == 0:
            # print "No non-connected, non-deleted nodes"
            return

        local_scale = self.density( self.points[n] )

        # this 1.3 should probably just be larger than the 0.95 below
        # actually, maybe make it significantly larger...
        # !!! Try making this more like 3 or 4, but first need to debug
        #  cell-neighbor inconsistency bug
        local_nodes = self.local_nodes(n,1.3*local_scale)

        # remove any that are too close along the same unpaved ring
        not_local = []

        if self.verbose > 2:
            print("Checking nodes that are not connected to %d"%n)

        for other in not_connected:
            if other not in local_nodes:
                if self.verbose > 2:
                    print("  %d is not connected.  will consider it."%other)
                not_local.append(other)

        if len(not_local) == 0:
            return 

        dists = [norm( self.points[n] - self.points[other] ) for other in not_local]

        order = argsort(dists)

        # starting from closest, see if anybody is connectable and close
        # enough
        for i in order:
            min_dist = dists[i]
            best_nbr = not_local[i]

            # really touchy about this 0.95 right now.  It should be able
            # to withstand values like 1.33, but that blows up...
            # will having it at 0.85 help anything out, at least for the
            # test case (!?) - 0.85 doesn't look as good, and ends up with
            # some small triangles
            # how about 1.1, now that we kind of have a guard against superfluous
            # non-local connections? well, the boundary_resampling doesn't work with
            # that, it tried to relax a HINT node.

            # the tests for whether a node is a good match for a non-local connection
            # still need some work.  this used to be 0.95 - I'm changing it to 0.90
            # in hopes of getting further along in the delta triangulation.
            if min_dist > 0.95 * local_scale:
                break
            
            nn = np.array([n,best_nbr])
            xy = self.points[nn]

            ### There was at least one case of trying to make an edge with a node that
            # wasn't even part of the unpaved boundary.
            if self.verbose > 2:
                print("Checking to make sure that node %d is part of the unpaved boundary")
            n_iters = self.all_iters_for_node(best_nbr)
            if len(n_iters) == 0:
                if self.verbose > 1:
                    print("Almost tried to make a non-local connection to a node that isn't even in unpaved")
                continue

            ## Another big check - don't make non-local connection over land.
            if self.verbose > 2:
                print("Checking if non-local edge [%d,%d], with dist=%g is within the domain"%(n,best_nbr,min_dist))
                
            pnts = np.array( [0.999*xy[0] + 0.001*xy[1],
                              0.001*xy[0] + 0.999*xy[1]] )
            l = geo.LineString( pnts )

            if not self.poly.contains(l):
                if self.verbose>2:
                    print("Nope, that edge wasn't within the domain")
                continue

            ## And an annoying one - when the scale is changing rapidly and we are squished,
            #  this tends to pop up.  Need to make sure that the line doesn't intersect any
            #  edges.
            if not self.check_line_intersections(n,best_nbr):
                if self.verbose > 1:
                    print("Narrowly avoided making a bad non-local connection")
                continue
            
            print("Trying non-local connection %d and %d"%(n,best_nbr))
            e = self.add_edge(nn[0],nn[1])
            print("Just added non-local edge %d"%e)
            # I wish this weren't needed, but it can
            # leave over-sampled boundary nodes that get
            # picked up as non-local connections later.

            # maybe overkill, but make sure we are okay on
            # all unpaved boundaries:
            for elt in self.all_iters_for_node(best_nbr):
                # leaving these as HINT is problematic sometimes.  The
                # new edge is often too short and in the next step will
                # probably be optimized.  this means it may move substantially
                # This is particularly a problem join to the end of an island.
                # if the resampled neighbors are HINTs, then this node may be
                # slid around to the other side of the island, making our non-local
                # edge actually cross over the island.  
                self.resample_neighbors(elt,new_node_stat=self.SLIDE)
            if 0:
                # slow - this completely recomputes the unpaved boundaries from scratch
                # remove soon.
                self.regenerate_unpaved_boundaries()
            else:
                self.update_unpaved_boundaries(e)
            return

    def check_line_intersections(self,a,b):
        """ check that a line segment between points[a] and points[b]
          doesn't intersect any nearby edges
        """
        ref = 0.5*(self.points[a] + self.points[b])
        pa = self.points[a] - ref
        pb = self.points[b] - ref
        
        a_ab,b_ab,c_ab = line_eq(pa,pb)

        # gather some nearby points
        nearest = self.closest_point(ref,20)

        # and their edges:
        edge_lists = [self.pnt2edges(n) for n in nearest]
        edges = np.unique( np.concatenate( edge_lists ) )

        N1 = self.edges[edges,0]
        N2 = self.edges[edges,1]

        # these were i and ip1 - seems totally wrong.
        P1 = self.points[N1] - ref
        P2 = self.points[N2] - ref

        D1 = a_ab*P1[:,0] + b_ab*P1[:,1] + c_ab
        D2 = a_ab*P2[:,0] + b_ab*P2[:,1] + c_ab
            
        PROD1 = D1*D2

        possible = np.where(PROD1 < -self.TEST_SMALL)[0]
            
        for pi in possible:
            n1 = N1[pi]
            n2 = N2[pi]

            p1 = P1[pi]
            p2 = P2[pi]
            prod1 = PROD1[pi]

            a_12,b_12,c_12 = line_eq(p1,p2)
            d1 = a_12*pa[0] + b_12*pa[1] + c_12
            d2 = a_12*pb[0] + b_12*pb[1] + c_12

            if d1*d2 < -self.TEST_SMALL:
                print("two-way intersection w/products %g and %g"%(prod1,d1*d2))
                if a in [n1,n2] or b in [n1,n2]:
                    print("they share a node - consider increasing TEST_SMALL or learning to write robust code")
                    continue

                e_bad = self.find_edge( (n1,n2) )
                return False
        return True

        
    def local_nodes(self,n,local_radius,steps=2):
        """ find nodes that are nearby along unpaved boundaries
        steps is a lower-bound on how many steps we take along
        boundaries.  So with steps=2, we force inclusion of
        neighbors of neighbors.

        local_radius: trace at least enough to get this far away

        returns an Nx2 array of nodes and distance along boundary to each.
          if a node is reachable in multiple ways, report the shortest distance.
        """
        local_nodes = {}
        
        p = self.points[n]

        tnxt = lambda t: t.nxt
        tprv = lambda t: t.prv
        
        for elt in self.all_iters_for_node(n):
            for stepper in tnxt,tprv:
                trav = elt
                s = 0 # steps taken
                dsum = 0.0 # cumulative distance from p
                last_point = p
                while 1:
                    trav = stepper(trav)
                    s+=1
                    if trav == elt:
                        break
                    #
                    straightline = norm( self.points[trav.data] - p )
                    dsum += norm( self.points[trav.data] - last_point )
                    last_point = self.points[trav.data]
                    if straightline < local_radius or s<=steps:
                        if trav.data not in local_nodes:
                            local_nodes[trav.data] = dsum
                        else:
                            local_nodes[trav.data] = min(local_nodes[trav.data],
                                                         dsum )
                    else:
                        break
                    
        return np.array(list(local_nodes.items()))
        
    def all_iters_for_node(self,node):
        iters = []
        for unpaved in self.unpaved:
            if node in unpaved:
                for elt in unpaved.node_to_iters[node]:
                    iters.append(elt)
        return iters

    def iters_for_edge(self,edge):
        """ return left and right iters for the given edge.  one or both may be None
        if that side isn't on an unpaved ring.  Each iter is for the more CW node
        (i.e. i.nxt.data gives the other node of the edge)
        """
        left_iter,right_iter = None,None
        if self.edges[edge,3] == trigrid.UNMESHED:
            left_iter = [li for li in self.all_iters_for_node(self.edges[edge,0]) \
                         if li.nxt.data == self.edges[edge,1]][0]
        if self.edges[edge,4] == trigrid.UNMESHED:
            right_iter = [ri for ri in self.all_iters_for_node(self.edges[edge,1]) \
                          if ri.nxt.data == self.edges[edge,0]][0]
        return left_iter,right_iter
                          
    def update_unpaved_boundaries(self,e):
        """ Update the unpaved boundaries after an edge completes a non-local connection
        """
        # we don't respect degenerate_rings, which shouldn't be a problem by the
        # time this gets called, but nonetheless, clear it out in case somebody
        # tries to call smooth() after this.
        # actually, now that degenerate edges can have sliding nodes, we need to remember
        # which original rings were degenerate.
        # self.degenerate_rings = "I don't think we need this after initialization"
        
        a,b = self.edges[e,:2]

        # At this point, the boundary iters haven't been updated at all.
        def choose_iter(m,n):
            """ find the iter that m is on which faces n """
            m_iters = self.all_iters_for_node(m)

            neighbors = self.angle_sort_neighbors(m).tolist()
            n_from_m = neighbors.index(n)

            prv = neighbors[(n_from_m+1)%len(neighbors)]
            nxt = neighbors[(n_from_m-1)%len(neighbors)]

            for m_iter in m_iters:
                if m_iter.nxt.data == nxt and m_iter.prv.data == prv:
                    return m_iter

            print("Failed to find an iter for node %d which faces node %d"%(m,n))
            raise Exception("Bad mojo trying to find iters in update_unpaved_boundaries")
            
        a_iter = choose_iter(a,b)
        b_iter = choose_iter(b,a)

        # There are 2 cases, which can probably be reduced even further
        # 1) a,b on same ring.  e cuts the ring into two rings.
        #    in this case, a_iter->b_iter + e is one ring
        #   and b_iter->a_iter + e is the other ring.
        # 2) a,b on different rings.  makes on ring, a splice of the two
        #    rings at a_iter and b_iter.

        if a_iter.clist == b_iter.clist:
            old_clist = a_iter.clist
            log.debug("Iters are from the same ring - will split that ring in two")

            # build up a new clist, starting at a_iter, 
            new_clist = CList()
            new_head = new_clist.append(a_iter.data)
            trav = a_iter.nxt
            while 1:
                new_head = new_clist.append(trav.data,after=new_head)
                if trav == b_iter:
                    print("Splitting boundary, made it back to b_iter.  good.")
                    break
                else:
                    old_clist.remove_iters(trav)
                    # even iters that have been removed still retain their own pointers,
                    # it's just that their old neighbors no longer point to them.
                    trav = trav.nxt
            # sanity check:
            if trav != b_iter:
                raise Exception("Trav should have ended up at b_iter")
            if trav.prv != a_iter:
                raise Exception("b_iter should have ended up just after a_iter")

            # now we have a new unpaved ring, new_clist, and
            # we stick back into unpaved right with the old one
            ri = self.unpaved.index( old_clist )

            # Order the smaller ring first - with the current (4/7/10) nonlocal
            # code, it is a bit hasty in making nonlocal connections that could
            # have been made locally - this creates a small ring that may be 
            # difficult to pave - better to treat it first before there are
            # cells on the other side which limit maneuverability

            if len(old_clist) < len(new_clist):
                insertion = [old_clist,new_clist]
            else:
                insertion = [new_clist,old_clist]
                    
            new_unpaved_list = self.unpaved[:ri] + insertion + self.unpaved[ri+1:]
            self.unpaved = new_unpaved_list
            
        else:
            log.debug("Iters are from different rings - will join rings together")

            if len(a_iter.clist) < len(b_iter.clist):
                a_iter,b_iter = b_iter,a_iter
            a_clist = a_iter.clist
            b_clist = b_iter.clist

            # So a_iter is the larger one.
            a_head = a_iter
            # insert b_iter before the loop, so we don't delete it yet.
            # b_iter.data is in b_iter presumably only once, but it will get
            # included in the new loop of a_clist twice.
            a_head = a_clist.append(b_iter.data,after=a_head)
            trav = b_iter.nxt
            while len(b_clist) > 0:
                # move the current node over.
                a_head = a_clist.append(trav.data,after=a_head)
                # remove and step
                b_clist.remove_iters(trav)
                trav = trav.nxt
            # b_clist gets left in self.unpaved, but it's empty.
            # if I can verify that nobody remembers indices into unpaved, could remove it
            # from the list.
                
            # And it goes back through a when it rejoins
            a_head = a_clist.append(a_iter.data,after=a_head)

            # Rearrange in self.unpaved:
            a_ri = self.unpaved.index( a_clist )
            b_ri = self.unpaved.index( b_clist )
            if a_ri > b_ri:
                # probably doesn't matter, but shift the non-empty clist to be in the lower
                # index of unpaved
                self.unpaved[b_ri] = a_clist
                self.unpaved[a_ri] = b_clist


    
    def angle_sort_neighbors(self,n):
        """ return neighbor nodes of n in ccw order, starting with the most clockwise
        relative to +x
        """
        nbrs = self.pnt2nbrs(n)
        return self.angle_sort_nodes(n,nbrs)

    def angle_sort_nodes(self,n,nbrs,return_angles=False):
        """ if return_angles is true, returns a tuple, the first element
        being the array of reordered nbr nodes, and the second their
        angles relative to n.
        """
        # delta along each edge
        diffs = self.points[nbrs] - self.points[n]
        
        angles = np.arctan2(diffs[:,1],diffs[:,0]) % (2*np.pi)
        ordering = np.argsort(angles)
        if return_angles:
            return nbrs[ordering], angles[ordering]
        else:
            return nbrs[ordering]

    def regenerate_unpaved_boundaries(self):
        """ Re-initializes unpaved rings based on searching edges/cells
        for unpaved regions.
        """
        # print "Top of regenerate_unpaved_boundaries()"

        candidate_edges = np.any( self.edges[:,3:5] == trigrid.UNMESHED, axis=1 )
        candidate_edges = candidate_edges & (self.edges[:,2] != trigrid.DELETED_EDGE)
        rings_and_holes = self.edges_to_rings_and_holes(candidate_edges)

        rings = []
        
        for exterior,interiors in rings_and_holes:
            rings.append( exterior )
            rings += interiors

        # print "%d rings in total"%len(rings)

        # dllist of the portion that has not been paved
        self.unpaved = [None] * len(rings)

        for ri in range(len(rings)):
            cl = CList()
            for i in rings[ri]:
                cl.append( i, metric=0 )
            
            self.unpaved[ri] = cl
        # print "Done with regenerating unpaved boundaries"

    ## Some interfaces for manual editing
    def flip_edge(self,e):
        """ Tries an edge flip - returns True if successful, False if the edge
        is on the boundary, or if the resulting geometry would cause a self-intersection
        """
        nc1,nc2 = self.edges[e,3:]
        if nc1 < 0 or nc2 < 0:
            "Boundary edge"
            return False

        all_points = np.unique( np.concatenate( [self.cells[nc1],
                                                 self.cells[nc2]] ) )

        new_edge_points = [v for v in all_points if v not in self.edges[e,:2]]

        if len(new_edge_points) == 2:
            print("Will swap edge %d-%d for edge %d-%d"%(self.edges[e,0],self.edges[e,1],
                                                         new_edge_points[0],new_edge_points[1]))
        else:
            print("Instead of 2 points to swap, found: ",new_edge_points)
            return False

        self.delete_edge(e)
        self.add_edge(new_edge_points[0],new_edge_points[1])
        return True

        
    def cost_args_for_node(self,i):
        # find all of the cells that we're in:
        my_cells = self.pnt2cells(i)

        if len(my_cells) == 0:
            return

        my_cells = np.array(list(my_cells))
        
        edges = self.cells[my_cells]

        # pack our neighbors from the cell list into an edge
        # list that respects the CCW condition that pnt must be on the
        # left of each segment
        for j in range(len(edges)):
            if edges[j,0] == i:
                edges[j,:2] = edges[j,1:]
            elif edges[j,1] == i:
                edges[j,1] = edges[j,0]
                edges[j,0] = edges[j,2]

        edges = edges[:,:2]

        edge_points = self.points[edges]

        # my cost:
        local_length = self.density( self.points[i,:2] )

        return edge_points,local_length
    
    def relax_one(self,i,boundary=False,use_beta=0,max_cost=np.inf):
        """
        boundary: node will be moved only along boundary (but see use_beta)

        use_beta: allow tuning boundary nodes both along and across the boundary,
          but self.beta_cost_factor will impose a penalty on moving off the boundary.

        max_cost: if the new cost is larger than this value, don't commit the change.  
        """
        if self.verbose > 1:
            print("Relaxing node %d, boundary=%s use_beta=%s"%(i,boundary,use_beta))
            
        # edge_points, local_length
        edge_points,local_length = self.cost_args_for_node(i)
        if not boundary:
            #print "Relaxing node %d, no boundary"%i
            xtol = 0.01*local_length

            new_pnt_i = my_fmin(one_point_cost,self.points[i],
                                 args=(edge_points,local_length),disp=0, 
                                 xtol=xtol)

            new_cost = one_point_cost(new_pnt_i,edge_points,local_length)
            if new_cost > max_cost:
                if self.verbose > 0:
                    print("post-relax cost was %f - don't commit"%new_cost)
            else:
                if abs( (new_pnt_i-self.points[i])).sum() > 0.1*xtol:
                    self.move_node(i, new_pnt_i )
        else:
            # xtol is a bit harder to quantify here -
            #  alpha scale varies with distance between nodes
            
            ring_i = int(self.node_data[i,self.ORIG_RING])
            starting_beta = self.node_data[i,self.BETA]
            starting_alpha = self.node_data[i,self.ALPHA]
            
            if not use_beta:
                xtol = 0.01

                # non-dimensionalize alpha to be 1 at the starting point
                # 
                def bdry_point_cost_a(nd_alpha):
                    alpha = nd_alpha-1+starting_alpha
                    new_point = self.boundary_slider(ring_i, alpha,starting_beta)
                    return one_point_cost(new_point,edge_points,local_length)

                # 2014-11-06: used to omit the [0], maintaining an array value
                new_nd_alpha =my_fmin(bdry_point_cost_a,1,disp=0,
                                      xtol=xtol)[0]
                new_cost = bdry_point_cost_a(new_nd_alpha)
                new_alpha = new_nd_alpha - 1 + starting_alpha
                new_beta = starting_beta

                if self.verbose > 1:
                    print("relax_one(%d) max_cost=%f new_cost=%f  alpha %f -> %f "%(i,max_cost,new_cost,starting_alpha,new_alpha))
            else:
                xtol = min( 0.01, 0.01*local_length )
                
                if self.verbose > 0:
                    print("Relaxing with alpha AND beta...")

                # beta is scaled up by the local length - this is a vague attempt to put it on
                # more equal footing with alpha.  fmin() will start with steps that are either
                # 5% of the value, or 0.00025 if the starting value is 0.
                # beta is already underscaled - so arbitrarily scale it up by local_length
                # in the eyes fmin(), thus sbeta = beta / local_length + local_length
                # so if alpha is changed by 1, it gives a physical change O(local_length),
                # if sbeta is changed by 1, I want the point to move O(local_length)
                # also, I want the starting sbeta to be O(1), so it will try to move it by
                # 5%, which will move the point ~5% local_length

                # Updated to get the same treatment as the alpha-only case, where
                # both are non-dimensionalized to start at 1.0, and a change of O(1) moves the
                # point O(local_length).

                # sbeta = 1 + beta/local_length
                # beta = (sbeta-1)*local_length
                
                def bdry_point_cost_ab(nd_alpha_beta):
                    alpha = nd_alpha_beta[0] - 1 + starting_alpha
                    beta  = (nd_alpha_beta[1] - 1) * local_length
                    
                    new_point = self.boundary_slider(ring_i, alpha,beta )
                    c = one_point_cost(new_point,edge_points,local_length)
                    # print "cost=%f   at alpha=%g beta=%g"%(c,alpha_sbeta[0],local_length * (alpha_sbeta[1]-1.0) )
                    return c
                starting_nd_alpha_beta = self.node_data[i,self.ALPHA:(self.BETA+1)].copy()
                starting_nd_alpha_beta[0] -= (starting_alpha - 1)
                starting_nd_alpha_beta[1] = 1 + starting_beta / local_length 
                new_nd_alpha_beta = my_fmin(bdry_point_cost_ab,starting_nd_alpha_beta,disp=0,
                                            xtol=xtol)
                new_cost = bdry_point_cost_ab(new_nd_alpha_beta)

                new_alpha = new_nd_alpha_beta[0] - 1 + starting_alpha
                new_beta = (new_nd_alpha_beta[1] - 1.0)*local_length
                if self.verbose > 0:
                    print("Result was: ",new_alpha,new_beta)

            if new_cost > max_cost:
                if self.verbose > 0:
                    print("Sliding, new cost was %g - don't commit"%new_cost)
            else:
                if abs( new_beta-starting_beta ) + abs( new_alpha - starting_alpha ) > 1e-4:
                    # print "Slid node on boundary %.4f -> %.4f"%(starting_alpha,new_alpha)
                    # make sure that the node hasn't gone beyond either of it's neighbors
                    # HERE: the node may or may not be on an unpaved boundary.  If it is
                    # not on an unpaved, then there shouldn't be any HINT nodes around
                    # and our alpha range is limited by immediate neighbors.
                    # if it is on an unpaved, then alpha may be limited by non-immediate
                    # neighbors.
                    if self.check_new_alpha(i,new_alpha):
                        self.slide_node(i,new_alpha,new_beta)
                        # HERE is a problem - for now, maybe we don't even have to resample??
                        # self.resample_neighbors(i,new_node_stat=self.HINT)
                    else:
                        if self.verbose > 0:
                            print("Not committing new alpha - it was out of bounds")

    def check_new_alpha(self,i,new_alpha):
        """ really for inner rings we should do one more check -
         see if the new alpha puts the centroid of the ring on the
         wrong side of any boundary edges connected to i.  not sure
         how to do this robustly, though, since there would still be
         hint nodes in the way.
        """
        
        lower_bound, upper_bound = self.node_alpha_bounds(i)
        # print "Node %d, new alpha = %.4f, alpha bounds = %.4f %.4f"%(i,new_alpha,lower_bound,upper_bound)

        ri = int(self.node_data[i,self.ORIG_RING])
        
        len_b = len(self.original_rings[ri])

        if lower_bound == upper_bound:
            # should only happen when exactly one other node is on the
            # ring.  In this case we can move wherever we want (though
            # there may be some intersection issues if we move too much)
            return True
            
        if (new_alpha - lower_bound) % len_b > (upper_bound - lower_bound) % len_b:
            if 0: # enable if alpha checks are failing a lot
                print("Alpha check failed:  %g  ?%g?  %g"%(lower_bound,new_alpha,upper_bound))
                lower_pnt = self.boundary_slider(ri,lower_bound)
                upper_pnt = self.boundary_slider(ri,upper_bound)
                new_pnt   = self.boundary_slider(ri,new_alpha)
                cur_pnt   = self.points[i]

                plt.annotate('lower',lower_pnt)
                plt.annotate('upper',upper_pnt)
                plt.annotate('new',new_pnt)
                plt.annotate('cur',cur_pnt)

                raise Exception("Stop and take a look")
            return False
        return True
    

    def node_alpha_bounds(self,i):
        """ return a lower and upper bound on valid values for a node's
        alpha.
        the given node must already lie on an original ring.
        """
        # print "Searching for bounds for node %d"%i

        # new approach purely from node_data
        ring_i = int( self.node_data[i,self.ORIG_RING] )

        if ring_i < 0:
            raise Exception("Node alpha bounds called, but node not on original ring")

        my_alpha = self.node_data[i,self.ALPHA]

        # a cheap hack - in case node i is a hint node
        saved_node_stat = self.node_data[i,self.STAT]
        self.node_data[i,self.STAT] = self.RIGID # just temporary
        
        # well, it's going to be slow...
        # this could probably be rewritten, if it ever proves to be a performance
        # bottleneck.
        same_ring_nodes = np.where( (self.node_data[:,self.ORIG_RING] == ring_i) & \
                                    (self.node_data[:,self.STAT] != self.HINT) & \
                                    (self.node_data[:,self.STAT] != self.DELETED) )[0]

        self.node_data[i,self.STAT] = saved_node_stat
        
        same_ring_alphas = self.node_data[ same_ring_nodes, self.ALPHA ]

        # look for the one closest to us in the CCW direction:
        len_b = len(self.original_rings[ring_i])
        diff_alphas = (same_ring_alphas - my_alpha) % len_b
        # sort them -- in-place
        diff_alphas.sort()

        # we should be in there - check to be sure
        if diff_alphas[0] != 0:
            raise Exception("The first diff should be us, unless we are somehow a HINT node")


        if len(diff_alphas) == 1:
            print("Wow - node %d on ring %d has no fixed neighbors"%(i,ring_i))
            # make up an alpha range - could also change the calling code
            # but this way we can kind of signify that the node shouldn't
            # get slid all the way around
            # assumes that the ring is evenly sampled, but really this is
            # just a rough guide.
            min_alpha = (my_alpha - 0.25*len_b) % len_b
            max_alpha = (my_alpha + 0.25*len_b) % len_b
        else:
            max_alpha = (my_alpha + diff_alphas[1]) % len_b
            # and the farthest around is the one just CW of us
            min_alpha  = (my_alpha + diff_alphas[-1]) % len_b

        return min_alpha,max_alpha

        # 
        # # offlimits since we don't have elements, just node index
        # # dist_back,node_back = self.free_distance_along_original_ring(i,'backward')
        # # dist_for,node_for = self.free_distance_along_original_ring(i,'forward')
        # 
        # 
        # # get some info on our adjacent edges
        # my_edges = array( self.pnt2edges(i) )
        # 
        # # !!! this will be a problem with degenerate rings
        # edge_on_boundary = any( self.edges[my_edges,3:] == -1, axis=1 )
        # my_edges = my_edges[edge_on_boundary]
        # 
        # # print "Node has these boundary edges: ",my_edges
        # if len(my_edges) != 2:
        #     raise Exception,"Really ought to find exactly two edges on boundary"
        # 
        # # which of those edges is backwards along the boundary?
        # 
        # # is it the first edge?
        # a,b = self.edges[my_edges[0],:2]
        # # convention is that a,b has edges[:,3] cell on left, and [:,4]
        # # on right.  
        # if self.edges[my_edges[0],3] == -1:
        #     a,b = b,a
        # # now a,b is oriented CCW, domain to the left.
        # 
        # if b == i:
        #     back_edge = my_edges[0]
        #     for_edge = my_edges[1]
        # else:
        #     back_edge = my_edges[1]
        #     for_edge = my_edges[0]
        # 
        # # two possibilities - an alpha bound may be from the next non-HINT
        # # node on the boundary, or a single edge to another non-HINT node.
        # 
        # if dist_back == 0:
        #     node_back = setdiff1d( self.edges[back_edge,:2], [i] )[0]
        # if dist_for == 0:
        #     node_for  = setdiff1d( self.edges[for_edge,:2], [i] )[0]
        # 
        # # print "Back node=%d  Forward node=%d"%(node_back,node_for)
        # 
        # if self.node_data[node_back,self.ORIG_RING] != self.node_data[i,self.ORIG_RING]:
        #     raise Exception,"Somehow we got a node on a different ring"
        # if self.node_data[node_for,self.ORIG_RING] != self.node_data[i,self.ORIG_RING]:
        #     raise Exception,"Somehow we got a node on a different ring"
        #     
        # return self.node_data[node_back,self.ALPHA],self.node_data[node_for,self.ALPHA]
        
    def slide_node(self,n,new_alpha,new_beta=None):
        """ update the location of a boundary node by it's alpha value.
        Takes care of calling move_node(), updating node_data, and
        removing any HINT nodes that are in the way.

        defaults to keeping the old beta
        """
        old_alpha = self.node_data[n,self.ALPHA]
        old_beta = self.node_data[n,self.BETA]
        if new_beta is None:
            new_beta = old_beta

        # figure out which unpaved ring, if any, we are in
        all_ui = []
        for i in range(len(self.unpaved)):
            if n in self.unpaved[i]:
                all_ui.append(i)
            
        # only worry about neighbors if we are still on an unpaved
        # boundary

        if len(all_ui)>1 and self.verbose > 1:
            print("WARNING: sliding a node (%d) that belongs to multiple unpaved rings. "%n)
            
        ri = int(self.node_data[n,self.ORIG_RING])
        len_b = len(self.original_rings[ri])
        
        # assumes that we slid in the shorter direction around the
        # ring:
        delta_alpha = (new_alpha - old_alpha + 0.5*len_b) % len_b - 0.5*len_b

        for ui in all_ui:
            for elt in self.unpaved[ui].node_to_iters[n]:

                # figure out which way to look based on which neighbor in
                # unpaved is a HINT node.
                next_n = elt.nxt.data
                prev_n = elt.prv.data

                forward_is_free = self.node_data[next_n,self.STAT] == self.HINT
                backward_is_free = self.node_data[prev_n,self.STAT] == self.HINT

                # print "Forward is free: ",forward_is_free
                # print "Backward is free: ",backward_is_free

                if (not forward_is_free) and (not backward_is_free):
                    # print "slide_node: neither direction is free, so hopefully we're okay"
                    pass
                elif forward_is_free and backward_is_free:
                    # *maybe* this would be okay for the very first node, but why
                    # would we be calling slide_node for the first node, before any
                    # other existed?
                    # raise Exception,"Shouldn't have HINT nodes in both directions"
                    
                    # I think this is actually okay, so use the value of delta_alpha
                    # to choose which way to move (that may be a bit dicy, but we'll
                    # see...
                    if delta_alpha > 0:
                        self.clear_boundary_by_alpha_range(elt,1,new_alpha)
                    else:
                        self.clear_boundary_by_alpha_range(elt,-1,new_alpha)
                elif forward_is_free and delta_alpha > 0:
                    self.clear_boundary_by_alpha_range(elt,1,new_alpha)
                elif backward_is_free and delta_alpha < 0:
                    self.clear_boundary_by_alpha_range(elt,-1,new_alpha)

        self.push_op(self.unslide_node, n, old_alpha, old_beta )
        
        self.node_data[n,self.ALPHA] = new_alpha
        self.node_data[n,self.BETA] = new_beta
        new_pnt_i = self.boundary_slider(ri, new_alpha, new_beta)

        # new checking for collapsing endpoints on internal guides
        if ri in self.degenerate_rings:
            replace = None
            if new_alpha == 0 or new_alpha == len_b-1:
                # if the endpoint is alone, we can (maybe?) just kill it?
                replace = self.closest_point(new_pnt_i)
                nbrs = self.pnt2nbrs(replace)
                if n==replace:
                    # this happens because in rare cases a guide can have its end snipped
                    # if that end winds up inside a cutoff. 
                    pass
                else:
                    print("%d is trying to slide on top of the end node %d, with nbrs %s"%(n,replace,nbrs))
                    if len(nbrs)==1 and nbrs[0]==n:
                        loner_stat = self.node_data[replace,self.STAT]
                        # Remove replace from unpaved:
                        # note that we should get exactly two iters that point to
                        # replace
                        for each_elt in self.all_iters_for_node(n):
                            if each_elt.nxt.data == replace:
                                break
                        clist = each_elt.clist

                        self.push_op(self.unremove_from_unpaved,each_elt.nxt.nxt)
                        clist.remove_iters(each_elt.nxt.nxt)
                        self.push_op(self.unremove_from_unpaved,each_elt.nxt)
                        clist.remove_iters(each_elt.nxt)


                        # something is making the index very unhappy -
                        # try just killing it preemptively:
                        self.index = None
                        print("deleting %d"%replace)
                        self.delete_node(replace)
                        print("done")
                        self.change_node_stat(n,loner_stat)
                    else:
                        print("Whoa - this is bad - trying to slide a node on top of another node, but we can't")
                        print("get rid of the other node because it has other neighbors")
                    
        # print "Moving %d to %s"%(n,new_pnt_i)
        self.move_node(n, new_pnt_i )
        # print "done"

    def unslide_node(self,n,old_alpha,old_beta):
        self.node_data[n,self.ALPHA] = old_alpha
        self.node_data[n,self.BETA]  = old_beta

    def unchange_node_stat(self,n,old_stat):
        self.node_data[n,self.STAT] = old_stat
        
    def change_node_stat(self,n,new_stat):
        self.push_op(self.unchange_node_stat, n, self.node_data[n,self.STAT] )
        self.node_data[n,self.STAT] = new_stat
                     
    def unmove_node(self,i,old_val):
        super(Paving,self).unmove_node(i,old_val)

        for elt in self.all_iters_for_node(i):
            elt.clist.update_metric(elt,0.0)
        
    def move_node(self,i,new_pnt):
        # good place to zero out internal angles
        # note that this is a parallel but separate way of updating points than
        #  orthomaker::push_node().  maybe someday they will merge, but for now
        #  the goal is optimization.

        # slow, but should work for now - we have to figure out if this node
        # lies on a boundary
        for elt in self.all_iters_for_node(i):
            elt.clist.update_metric(elt,0.0)
        
        return super(Paving,self).move_node(i,new_pnt)
    
    def cost_for_point(self,i):
        edge_points,local_length = self.cost_args_for_node(i)
        return one_point_cost( self.points[i,:2], edge_points, local_length )
        
    def relax(self,plot_progress=False,threshold=None):
        # loop over all the points and try to optimize them with the minimization routines

        if threshold is None:
            threshold = self.cost_threshold
            
        # r = random(self.Npoints())
        # ordering = argsort(r)
        # ordering = range(self.Npoints())[::-1]
        costs = [self.cost_for_point(i) for i in range(self.Npoints())]
        
        ordering = argsort(costs)[::-1]
        
        for i in ordering:
            if costs[i] < threshold:
                continue
            
            if len(self.pnt2edges(i)) == 0:
                continue
            
            self.safe_relax_one(i)
            if plot_progress:
                self.plot()
            
        self.step += 1

    def splice_in_grid(self,gridB,join_tolerance=0.25):
        """
        gridB: TriGrid instance to be inserted into self
        join_tolerance: nodes within this distance will be merged.
        """
        
        # Try a simpler approach to stitching, relying on step
        # by step grid modifications.  gridA should be a Paving
        # instance, but gridB need only be a TriGrid.

        ## NODES
        nodeBtoA = {}

        # Copy the nodes over:
        for nB in range(gridB.Npoints()):
            if np.isnan(gridB.points[nB,0]):
                continue # node has been deleted...
            close_nA = self.closest_point( gridB.points[nB] )
            dist = norm( self.points[close_nA] - gridB.points[nB] )
            if dist < join_tolerance:
                print("Would be joining %d - %d"%(nB,close_nA))
                nodeBtoA[nB] = close_nA
            else:
                nodeBtoA[nB] = self.add_node( gridB.points[nB],
                                              stat=self.RIGID )

        ## EDGES
        edgeBtoA = {} # probably don't need this
        for jB in range(gridB.Nedges()):
            if gridB.edges[jB,0] < 0:
                continue # edge has been deleted
            n1 = nodeBtoA[gridB.edges[jB,0]]
            n2 = nodeBtoA[gridB.edges[jB,1]]

            # May be a duplicate edge:
            try:
                jA = self.find_edge( (n1,n2) )
                edgeBtoA[jB] = jA
            except trigrid.NoSuchEdgeError:
                edgeBtoA[jB] = self.add_edge(n1,n2)

        ## CELLS
        # Most cells come over okay, but at the boundaries, can lose some.        
        cellBtoA = {}
        for cB in range(gridB.Ncells()):
            if gridB.cells[cB,0] < 0:
                continue # it's been deleted
            abc_A = [ nodeBtoA[i] for i in gridB.cells[cB]]
            try:
                cellBtoA[cB] = self.find_cell(abc_A)
            except trigrid.NoSuchCellError:
                print("had to create a cell")
                cellBtoA[cB] = self.add_cell(abc_A)
                
        return self

    # save_stride_fmt = "%(label)s-%(step)06d.pav"
    save_stride_fmt = "%(label)s.pav"
    def save_stride_filename(self):
        d={'label':self.label,
           'step':self.step}
        if d['label'] is None:
            d['label'] = "checkpoint"
            
        return self.save_stride_fmt%d

    def pave_all(self,plot_stride=None,n_steps=None,save_stride=None):
        if n_steps is None:
            self.init_stats()
        while 1: 
            if n_steps and self.step >= n_steps:
                break
            if not self.choose_and_fill():
                break
            if plot_stride and self.step%plot_stride == 0:
                self.plot()
                plt.draw()
            if save_stride and self.step%save_stride == 0:
                fn = self.save_stride_filename()
                print("Saving current state to %s"%fn)
                self.write_complete(fn)
                print("Done with save")
                

        if n_steps is None:
            self.finish_stats()

        if plot_stride:
            self.plot()    

        if len(self.unpaved) == 0:
            print("Done!")

    def renumber(self):
        # copy edge data - make a hash based on the sorted node indices
        tmp_edge_data = {}
        for e in range(self.Nedges()):
            if self.edges[e,2] == trigrid.DELETED_EDGE:
                continue
            n1,n2 = self.edges[e,:2]
            if n1 > n2:
                n1,n2 = n2,n1
            tmp_edge_data[ (n1,n2) ] = self.edge_data[e]
        
        mappings = super(Paving,self).renumber()

        # try to map edge data
        self.edge_data = np.zeros( (self.Nedges(),self.edge_data.shape[1]), np.float64)
        self.edge_data[:,:] = -1 # in case the remapping doesn't work out.
        for e in range(self.Nedges()):
            n1,n2 = mappings['valid_nodes'][ self.edges[e,:2] ]
            if n1 > n2:
                n1,n2 = n2,n1
            self.edge_data[e] = tmp_edge_data[ (n1,n2) ]

        # but node data can be mapped:
        self.node_data = self.node_data[ mappings['valid_nodes'] ]
        return mappings

        
    #-#-# Reporting
    times = None
    def init_stats(self):
        self.start_times = np.array( os.times() )
    def finish_stats(self):
        """
        Timing
        """
        self.finish_times = np.array( os.times() )
        self.times = self.finish_times - self.start_times

        print("Renumbering in preparation for statistics")
        self.renumber()

        print("------ Statistics ------")
        print("[%s]"%self.label)
        print("CPU user time (unix only): ",self.times[0])
        print("CPU sys time  (unix only): ",self.times[1])
        print("Wall time                : ",self.times[4])

        print("  cells: ",self.Ncells())
        print("  edges: ",self.Nedges())
        print(" points: ",self.Npoints())
        print(" cells/cpu-second: ", self.Ncells()/self.times[0] )

        ### Metrics for how good the triangles are:
        vcenters = self.vcenters()
        scales = self.density(vcenters)
        # areas  = self.areas()
        a = self.points[self.cells[:,0]]
        b = self.points[self.cells[:,1]]
        c = self.points[self.cells[:,2]]
        ab = np.sqrt(((a-b)**2).sum( axis=1 ))
        bc = np.sqrt(((c-b)**2).sum( axis=1 ))
        ca = np.sqrt(((a-c)**2).sum( axis=1 ))
        mean_length = (ab+bc+ca)/3
        scale_rel_error = mean_length / scales - 1.0

        self.mean_scale_rel_error = scale_rel_error.mean()
        self.min_scale_rel_error  = scale_rel_error.min()
        self.max_scale_rel_error  = scale_rel_error.max()
        self.std_scale_rel_error  = np.std(scale_rel_error)

        # and the clearances - go by edges
        # NB: this is the dg distance - distance across the edge
        # between the two voronoi centers. not exactly the right
        # metric, but it's what I've got already in the code...
        edge_clearances = self.edge_clearances() / 2.0
        a = self.points[self.edges[:,0]]
        b = self.points[self.edges[:,1]]
        
        edge_lengths = np.sqrt( ((a-b)**2).sum(axis=1) )
        clearance_rel_error = edge_clearances / ( edge_lengths/2.0 / np.sqrt(3) ) - 1.0
        self.mean_clearance_rel_error = clearance_rel_error.mean()
        self.min_clearance_rel_error = clearance_rel_error.min()
        self.max_clearance_rel_error = clearance_rel_error.max()
        self.std_clearance_rel_error = np.std( clearance_rel_error )
        
        print("Relative scale error:      % .3f ---  % .3f+-% .3f  --- % .3f"%(self.min_scale_rel_error,
                                                                               self.mean_scale_rel_error,
                                                                               self.std_scale_rel_error,
                                                                               self.max_scale_rel_error))
        
        print("Relative clearance error:  % .3f ---  % .3f+-% .3f  --- % .3f"%(self.min_clearance_rel_error,
                                                                               self.mean_clearance_rel_error,
                                                                               self.std_clearance_rel_error,
                                                                               self.max_clearance_rel_error))
        print(" ----- ---------- ----- ")
        
        

def plot_cost_field(p,node,scale=None,samples=300):
    """ Plot the cost field around the given node, in a square area
    of side-length scale, and with the given number of samples in
    each direction.
    """
    if scale is None:
        # choose the length of the longest edge adjacent to node
        edges = p.pnt2edges(node)
        points = p.points[ p.edges[edges,:2] ]
        lengths = np.sqrt( np.sum( np.diff(points,axis=1)**2, axis=2) )
        scale = lengths.max()
        
    edge_points,local_length = p.cost_args_for_node(node)

    return plot_cost_field_at_point(p.points[node],edge_points,local_length,scale,samples)

def plot_cost_field_at_point(pnt,edge_points,local_length,scale,samples):
    x = np.linspace( pnt[0]-scale,pnt[0]+scale,samples)
    y = np.linspace( pnt[1]-scale,pnt[1]+scale,samples)

    costs = np.zeros( (len(x),len(y)), np.float64 )
    for r in range(len(y)):
        for c in range(len(x)):
            costs[r,c] = one_point_cost( np.array([x[c],y[r]]),
                                         edge_points, local_length )

    return plt.imshow(log(costs),extent=(x[0],x[-1],y[0],y[-1]),
                      origin='bottom',interpolation='nearest',
                      vmax=10)

    
def plot_density( density, center=None, scale=None, samples=200 ):
    if center is None:
        z = plt.axis()
        center = [0.5*(z[1]+z[0]),
                  0.5*(z[3]+z[2])]
        scale = 0.5*(z[1]-z[0])
    x = np.linspace( center[0] - scale, center[0] + scale, samples)
    y = np.linspace( center[1] - scale, center[1] + scale, samples)
    X,Y = np.meshgrid(x,y)
    XY = np.concatenate( (X[:,:,None], Y[:,:,None]), axis=2 )
    
    D = density(XY)
    plt.imshow(D,extent=[x[0],x[-1],y[0],y[-1]],
               origin='bottom',interpolation='nearest')




