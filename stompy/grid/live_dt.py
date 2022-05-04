from __future__ import print_function

# Maintain a live constrained delaunay triangulation of the grid.
# designed as a mixin

from collections import defaultdict

try:
    from collections.abc import Iterable
except:
    from collections import Iterable

import logging
log = logging.getLogger('stompy.live_dt')

import pdb

import numpy as np
from numpy.linalg import norm,solve

from matplotlib import collections
import matplotlib.pyplot as plt

from .. import utils
from ..utils import array_append
from ..spatial import field,robust_predicates
from . import orthomaker,trigrid,exact_delaunay

def ray_intersection(p0,vec,pA,pB):
    d1a = np.array([pA[0]-p0[0],pA[1]-p0[1]])

    # alpha * vec + beta * ab = d1a
    # | vec[0] ab[0]   | | alpha | = |  d1a[0]  |
    # | vec[1] ab[1]   | | beta  | = |  d1a[1]  |

    A = np.array( [[vec[0],  pB[0] - pA[0]],
                   [vec[1],  pB[1] - pA[1]]] )
    alpha_beta = solve(A,d1a)
    return p0 + alpha_beta[0]*np.asarray(vec)


class MissingConstraint(Exception):
    pass

def distance_left_of_line(pnt, qp1, qp2):
    # return the signed distance for where pnt is located left of
    # of the line qp1->qp2
    #  we don't necessarily get the real distance, but at least something
    #  with the right sign, and monotonicity
    vec = qp2 - qp1
    
    left_vec = np.array( [-vec[1],vec[0]] )
    
    return (pnt[0] - qp1[0])*left_vec[0] + (pnt[1]-qp1[1])*left_vec[1]



class LiveDtGridNull(orthomaker.OrthoMaker):
    """ absolute minimum, do-nothing stand-in for LiveDtGrid implementations.
    probably not useful, and definitely not complete
    """
    has_dt = 0
    pending_conflicts = []

    def hold(self):
        pass
    def release(self):
        pass
    def delaunay_neighbors(self,n):
        return []

LiveDtGrid=LiveDtGridNull # will be set to the "best" implementation below

class LiveDtGridBase(orthomaker.OrthoMaker):
    """
    A mixin which adds a live-updated constrained Delaunay
    triangulation to shadow the grid, aiding in various geometric
    queries.  Similar in spirit to ShadowCDT, but this is an older
    code designed explicitly for use with paver, and originally
    only using CGAL for the triangulation.

    This is the abstract base class, which can either use a CGAL
    implementation or a pure-python implementation in subclasses
    below.

    This mixin maintains the mapping between nodes in self, and
    the vertices in the shadow Delaunay triangulation.  

    self.vh[n] maps a trigrid node n to the "handle" for a Delaunay
    vertex.

    self.vh_info[vh] provides the reverse mapping, from a Delaunay
    vertex handle to a node
    """
    has_dt = 1
    # if true, skips graph API handling
    freeze=0 

    # if true, stores up modified nodes and edges, and
    # updates all at once upon release
    holding = 0

    # queue of conflicting edges that have been un-constrained to allow for
    # an add_edge() to proceed
    pending_conflicts = []
    edges_to_release = None

    # triangles in the "medial axis" with a radius r < density/scale_ratio_for_cutoff
    # will be removed.
    # the meaning of this has changed slightly - 1/9/2013
    # now it is roughly the number of cells across a channel to make room for.
    # so at 1.0 it will retain channels which are one cell wide (actually there's a
    # bit of slop - room for 1.3 cells or so).
    # at 2.0, you should get 2-3 cells across.
    scale_ratio_for_cutoff = 1.0

    # even though in some cases the vertex handle type is more like int32,
    # leave this as object so that None can be used as a special value.
    vh_dtype='object' # used in allocating 
    
    def __init__(self,*args,**kwargs):
        super(LiveDtGridBase,self).__init__(*args,**kwargs)

        self.populate_dt()

    check_i = 0
    def check(self):
        return 
        print("    --checkplot %05i--"%self.check_i)

        plt.figure(10)
        plt.clf()
        self.plot_dt()
        if self.default_clip is not None:
            self.plot_nodes()
            plt.axis(self.default_clip)

        plt.title("--checkplot %05i--"%self.check_i)
        plt.savefig('tmp/dtframe%05i.png'%self.check_i)
        self.check_i += 1

        plt.close(10)

    def refresh_metadata(self):
        """ Should be called when all internal state is changed outside
        the mechanisms of add_X, delete_X, move_X, etc.
        """
        super(LiveDtGridBase,self).refresh_metadata()

        self.populate_dt()

    def populate_dt(self):
        """ Initialize a triangulation with all current edges and nodes.
        """
        # print("populate_dt: top")

        self.dt_allocate()
        self.vh = np.zeros( (self.Npoints(),), self.vh_dtype)
        self.vh[:]=None # 0 isn't actually a good mark for unused.
        
        # sometimes CGAL creates vertices automatically, which are detected by
        # having info == None
        self.vh_info = defaultdict(lambda:None)

        # print("populate_dt: adding points")
        for n in range(self.Npoints()):
            if n % 50000==0:
                log.info("populate_dt: %d/%d"%(n,self.Npoints()))
            # skip over deleted points:
            if np.isfinite(self.points[n,0]):
                self.dt_insert(n)
                
        # print("populate_dt: add constraints")
        for e in range(self.Nedges()):
            if e % 50000==0:
                log.info("populate_dt: %d/%d"%(e,self.Nedges()))
            a,b = self.edges[e,:2]
            if a>=0 and b>=0: # make sure we don't insert deleted edges
                self.safe_insert_constraint(a,b)
        # print("populate_dt: end")

    def safe_insert_constraint(self,a,b):
        """ adds a constraint to the DT, but does a few simple checks first
        if it's not safe, raise an Exception
        """
        if a < 0 or b < 0 or a==b:
            raise Exception("invalid node indices: %d %d"%(a,b))
        if all(self.points[a] == self.points[b]):
            raise Exception("invalid constraint: points[%d]=%s and points[%d]=%s are identical"%(a,self.points[a],
                                                                                                 b,self.points[b]))
        if self.verbose > 2:
            print("    Inserting constraint (populate_dt): %d %d  %s %s"%(a,b,self.vh[a],self.vh[b]))
            print("      node A=%s   node B=%s"%(self.points[a],self.points[b]))
            print("      A.point=%s  B.point=%s"%(self.vh[a].point(), self.vh[b].point()))

        self.dt_insert_constraint(a,b)

        # double check to make sure that it's actually in there...
        found_it=0
        for edge in self.dt_incident_constraints(self.vh[a]):
            v1,v2 = edge.vertices()
            if v1==self.vh[b] or v2==self.vh[b]:
                found_it = 1
                break
        if not found_it:
            # we have a conflict - search from a to b
            msg="Just tried to insert a constraint %d-%d (%s - %s), but it's not there!"%(a,b,
                                                                                          self.points[a],
                                                                                          self.points[b])
            raise MissingConstraint(msg)
        
    ## Hold/release
    def hold(self):
        if self.holding == 0:
            self.holding_nodes = {}

        self.holding += 1

    def release(self):
        if self.holding == 0:
            raise Exception("Tried to release, but holding is already 0")

        self.holding -= 1

        if self.holding == 0:
            # First, make sure that we have enough room for new nodes:
            while len(self.vh) < self.Npoints():
                # This used to extend with 0, but None is a better option
                self.vh = array_append(self.vh,None)

            held_nodes = list(self.holding_nodes.keys())

            # Remove all of the nodes that were alive when we started
            # the hold:
            for n in held_nodes:
                # first, it was != 0.
                # then it was "is not 0"
                # but with exact_delaunay, 0 is a valid vertex handle.
                if self.vh[n] is not None: # used to != 0
                    self.dt_remove_constraints(self.vh[n]) 
                    self.dt_remove(n) # that's correct, pass trigrid node index

            # Add back the ones that are currently valid
            for n in held_nodes:
                if np.isfinite(self.points[n,0]):
                    self.dt_insert(n)

            # Add back edges for each one
            held_edges = {}
            for n in held_nodes:
                for e in self.pnt2edges(n):
                    held_edges[e] = 1

            self.edges_to_release = list(held_edges.keys())
            while len(self.edges_to_release) > 0:
                e = self.edges_to_release.pop()
                # call dt_add_edge to get all of the conflicting-edge-detecting
                # functionality.  
                self.dt_add_edge(e)

            self.edges_to_release = None
            self.holding_nodes=0

        return self.holding

    def dt_update(self,n):
        if self.verbose > 2:
            print("    dt_update TOP: %d"%n)
            self.check()

        # have to remove any old constraints first:
        n_removed = 0
        to_remove = []

        # probably unnecessary, but queue the deletions to avoid any possibility
        # of confusing the iterator
        for edge in self.dt_incident_constraints(self.vh[n]):
            n_removed += 1
            v1,v2 = edge.vertices()

            vi1 = self.vh_info[v1]
            vi2 = self.vh_info[v2]

            to_remove.append( (edge, vi1, vi2) )
            if self.verbose > 2:
                # weird stuff is happening in here, so print out some extra
                # info
                print("    dt_update: found old constraint %s-%s"%(vi1,vi2))

        if n_removed != len(self.pnt2edges(n)):
            print("    WARNING: was going to remove them, but n_removed=%d, but pnt2edges shows"%n_removed)
            # How many of this point's edges are in the queue to be added?
            count_unreleased = 0
            if self.edges_to_release:
                for e in self.pnt2edges(n):
                    if e in self.edges_to_release:
                        count_unreleased += 1
            if n_removed + count_unreleased != len(self.pnt2edges(n)):    
                print(self.edges[self.pnt2edges(n),:2])
                print("Even after counting edges that are queued for release, still fails.")
                raise Exception("Something terrible happened trying to update a node")

        for edge,a,b in to_remove:
            self.dt_remove_constrained_edge(edge)

        self.dt_remove(n)
        self.dt_insert(n)
        # add back any of the constraints that we removed.
        # This used to add all constraints involving n, but if we are in the middle
        # of a release, pnt2edges() will not necessarily give the same edges as
        # constraints
        all_pairs = []
        for edge,a,b in to_remove:
            all_pairs.append( (a,b) )

            self.safe_insert_constraint(a,b)
            n_removed -= 1

        if n_removed != 0:
            print("    WARNING: in updating node %d, removed-added=%d"%(n,n_removed))
            print("    Inserted edges were ",all_pairs)
            raise Exception("why does this happen?")

        if self.verbose > 2:
            print("    dt_update END: %d"%n)
            self.check()

    def dt_add_edge(self,e):
        """ Add the edge, indexed by integer e and assumed to exist in the grid,
        to the Delaunay triangulation.  This method is a bit sneaky, and in dire
        situations tries to adjust the geometry to allow an otherwise invalid
        edge to be inserted.  

        Not a good design, but this is old code.
        """
        a,b = self.edges[e,:2]

        #-#-# Try to anticipate unsafe connections 
        for i in range(3): # try a few times to adjust the conflicting nodes
            constr_edges = self.check_line_is_clear(a,b)
            if len(constr_edges)>0:
                print("--=-=-=-=-=-= Inserting this edge %d-%d will cause an intersection -=-=-=-=-=-=-=--"%(a,b))
                for v1,v2 in constr_edges:
                    # use %s formats as values could be None
                    print("  intersects constrained edge: %s - %s"%(self.vh_info[v1],self.vh_info[v2]))

                if self.verbose > 1:
                    if i==0:
                        self.plot(plot_title="About to prepare_conflicting_edges")
                        plt.plot(self.points[[a,b],0],
                                 self.points[[a,b],1],'m')

                # Debugging:
                # raise Exception("Stopping before trying to fix conflicting edges")
                self.prepare_conflicting_edges(e,constr_edges)
            else:
                break
        #-#-#

        self.safe_insert_constraint(a,b)

        if a>b:
            a,b=b,a

        if self.verbose > 2:
            print("    dt_add_edge: adding constraint %d->%d"%(a,b))
            self.check()

    def prepare_conflicting_edges(self,e,constr_edges):
        """
        If an edge to be inserted is not valid, try to adjust one or more nodes
        in a small way to make it clear.  This approach is clearly stepping outside
        the bounds of good program flow. 
        """
        # First figure out which side is "open"
        # We should only be called when the data in self.edges has already
        # been taken care of, so it should be safe to just consult our cell ids.
        a,b = self.edges[e,:2]

        # arrange for a -> b to have the open side to its right
        if self.edges[e,3] >= 0 and self.edges[e,4] >= 0:
            print("prepare_conflicting_edges: both sides are closed!")
            return
        if self.edges[e,3] == -1 and self.edges[e,4] != -1:
            a,b = b,a
        elif self.edges[e,4] == -1:
            pass
        elif self.edges[e,3] == -2 and self.edges[e,4] != -2:
            a,b = b,a
        # otherwise it's already in the correct orientation

        print("prepare_conflicting_edges: proceeding for edge %d-%d"%(a,b))

        AB = self.points[b] - self.points[a]
        open_dir = np.array( [AB[1],-AB[0]] )
        mag = np.sqrt(AB[0]**2+AB[1]**2)
        AB /= mag
        open_dir /= mag

        to_move = [] # node index for nodes that need to be moved.

        for cgal_edge in constr_edges:
            vh_c,vh_d = cgal_edge

            c = self.vh_info[vh_c]
            d = self.vh_info[vh_d]
            if c is None:
                print("No node data for conflicting vertex %s"%vh_c)
                continue
            if d is None:
                print("No node data for conflicting vertex %s"%vh_d)
                continue

            # 2. which one is on the closed side?
            c_beta = np.dot( self.points[c] - self.points[a],
                             open_dir)
            d_beta = np.dot( self.points[d] - self.points[a],
                             open_dir)
            if c_beta < 0 and d_beta >= 0:
                to_move.append(c)
            elif d_beta < 0 and c_beta >= 0:
                to_move.append(d)
            else:
                print("Neither node in conflicting edge appears to be on the closed side")

        to_move = np.unique(to_move)
        eps = mag / 50.0

        for n in to_move:
            beta = np.dot( self.points[n] - self.points[a], open_dir)
            if beta >= 0:
                raise Exception("Only nodes with beta<0 should be in this list!")
            new_point = self.points[n] - (beta-eps)*open_dir
            print("prepare_conflicting_edges: Moving node %d to %s"%(n,new_point))
            self.move_node(n,new_point)

    def dt_remove_edge(self,e,nodes=None):
        """  Remove the given edge from the triangulation.  In cases
        where the edge e has already been updated with different nodes,
        pass in nodes as [a,b] to remove the edge as it was.
        """
        if nodes is not None:
            a,b = nodes
        else:
            a,b = self.edges[e,:2]

        #-# DBG
        if a>b:
            a,b=b,a
        if self.verbose > 2:
            print("    remove constraint %d->%d"%(a,b))
            self.check()
        #-# /DBG

        # have to figure out the face,index for this edge
        found_edge = 0

        for edge in self.dt_incident_constraints(self.vh[a]): 
            v1,v2 = edge.vertices()
            if self.vh[b] == v1 or self.vh[b] == v2:
                self.dt_remove_constrained_edge(edge)
                return
        raise MissingConstraint("Tried to remove edge %i, but it wasn't in the constrained DT"%e)


    #-#-# API for adding/moving/deleting

    #-# NODES
    def add_node(self,P):
        n = super(LiveDtGridBase,self).add_node(P)

        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[n] = 'add_node'
        else:
            self.vh = array_append(self.vh,None)
            self.dt_insert(n)
            # tricky - a new node may interrupt some existing
            # constraint, but when the node is removed the
            # constraint is not remembered - so check for that
            # explicitly -
            interrupted_edge = []
            for edge in self.dt_incident_constraints(self.vh[n]):
                a,b = edge.vertices()
                if self.vh_info[a] != n:
                    interrupted_edge.append(self.vh_info[a])
                else:
                    interrupted_edge.append(self.vh_info[b])
            if len(interrupted_edge):
                self.push_op(self.uninterrupt_constraint,interrupted_edge)

        return n

    def uninterrupt_constraint(self,ab):
        print("Uninterrupting a constraint. Yes!")
        self.safe_insert_constraint(ab[0],ab[1])

    def unmodify_edge(self, e, old_data):
        """ a bit unsure of this...  I don't know exactly where this
        gets done the first time
        """
        a,b = self.edges[e,:2]
        n = super(LiveDtGridBase,self).unmodify_edge(e,old_data)

        if a!=old_data[0] or b!=old_data[1]:
            print("unmodifying live_dt edge")
            self.safe_insert_constraint(old_data[0],old_data[1])

    def unadd_node(self,old_length):
        if self.freeze:
            pass
        elif self.holding:
            for i in range(old_length,len(self.points)):
                self.holding_nodes[i] = 'unadd'
        else:
            for i in range(old_length,len(self.points)):
                self.dt_remove(i)
            self.vh = self.vh[:old_length]

        super(LiveDtGridBase,self).unadd_node(old_length)

        if not (self.freeze or self.holding):
            print("HEY - this would be a good time to refresh the neighboring constraints")

    def delete_node(self,i,*args,**kwargs):
        # there is a keyword argument, remove_edges
        # does that need to be interpreted here?
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[i] = 'delete_node'

        super(LiveDtGridBase,self).delete_node(i,*args,**kwargs)

        if not self.freeze and not self.holding:
            self.dt_remove( i )

    def undelete_node(self,i,p):
        super(LiveDtGridBase,self).undelete_node(i,p)
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[i] = 'undelete'
        else:
            self.dt_insert(i)

    def unmove_node(self,i,orig_val):
        super(LiveDtGridBase,self).unmove_node(i,orig_val)
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[i] = 'unmove'
        else:
            self.dt_update(i)

    def move_node(self,i,new_pnt,avoid_conflicts=True):
        """ avoid_conflicts: if the new location would cause a
        self-intersection, don't move it so far...

        if the location is modified, return the actual location, otherwise
        return None
        """
        if not self.freeze and not self.holding:
            # pre-emptively remove constraints and the vertex
            # so that there aren't conflicts between the current
            # edges and the probe point.

            # See if the location will be okay -
            to_remove = []
            nbrs = [] # neighbor nodes, based only on constrained edges
            for edge in self.dt_incident_constraints(self.vh[i]):
                v1,v2 = edge.vertices()
                vi1 = self.vh_info[v1]
                vi2 = self.vh_info[v2]

                to_remove.append( (edge, vi1, vi2) )
                if vi1 == i:
                    nbrs.append(vi2)
                else:
                    nbrs.append(vi1)

            if len(to_remove) != len(self.pnt2edges(i)):
                # why is this a warning here, but for unmove_node we bail out?
                # I'm not really sure how this happens in the first place...
                # this was a warning, but investigating...
                # HERE: And now test_sine_sine is failing here.
                pdb.set_trace()
                raise Exception("WARNING: move_node len(DT constraints) != len(pnt2edges(i))")

            for edge,a,b in to_remove:
                self.dt_remove_constrained_edge(edge)

            self.dt_remove(i)

            # With the old edges and vertex out of the way, make sure the new location
            # is safe, and adjust necessary
            new_pnt = self.adjust_move_node(i,new_pnt,nbrs)

        super(LiveDtGridBase,self).move_node(i,new_pnt)

        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[i] = 'move'
        else:
            # put the node back in, and add back any edges that we removed.
            # NB: safer to add only the constraints that were there before, since it
            #  could be that the DT is not perfectly in sync with self.edges[]
            self.dt_insert(i)
            for edge,a,b in to_remove:
                self.safe_insert_constraint(a,b)

        return new_pnt

    def adjust_move_node(self,i,new_pnt,nbrs):
        """ Check if it's okay to move the node i to the given point, and
        if needed, return a different new_pnt location that won't make an
        intersection

        i: node index
        new_pnt: the requested new location of the node
        nbrs: list of neighbor node indices for checking edges
        """

        # HERE -- not compatible with pure python code.
        
        # find existing constrained edges
        # for each constrained edge:
        #   will the updated edge still be valid?
        #   if not, update new_pnt to be halfway between the old and the new,
        #      and loop again.

        for shorten in range(15): # maximum number of shortenings allowed
            all_good = True

            # Create a probe vertex so we can call check_line_is_clear()
            # sort of winging it here for a measure of close things are.
            if abs(self.points[i] - new_pnt).sum() / (1.0+abs(new_pnt).max()) < 1e-8:
                log.warning("adjust_move_node: danger of roundoff issues")
                all_good = False
                break

            all_good=self.check_line_is_clear_batch(p1=new_pnt,n2=nbrs)
            if all_good:
                break
            else:
                new_pnt = 0.5*(self.points[i]+new_pnt)
                log.debug('adjust_move_node: adjusting') 
        if all_good:
            return new_pnt
        else:
            return self.points[i]


    ## EDGES
    def add_edge(self,nodeA,nodeB,*args,**kwargs):
        e = super(LiveDtGridBase,self).add_edge(nodeA,nodeB,*args,**kwargs)
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[ self.edges[e,0] ] ='add_edge'
            self.holding_nodes[ self.edges[e,1] ] ='add_edge'
        else:
            self.dt_add_edge(e)
        return e
    def unadd_edge(self,old_length):
        if self.freeze:
            pass
        elif self.holding:
            for e in range(old_length,len(self.edges)):
                self.holding_nodes[ self.edges[e,0] ] ='unadd_edge'
                self.holding_nodes[ self.edges[e,1] ] ='unadd_edge'
        else:
            for e in range(old_length,len(self.edges)):
                self.dt_remove_edge(e)

        super(LiveDtGridBase,self).unadd_edge(old_length)
    def delete_edge(self,e,*args,**kwargs):
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[ self.edges[e,0] ] = 'delete_edge'
            self.holding_nodes[ self.edges[e,1] ] = 'delete_edge'
        else:
            self.dt_remove_edge(e)

        super(LiveDtGridBase,self).delete_edge(e,*args,**kwargs)

    def undelete_edge(self,e,*args,**kwargs):
        super(LiveDtGridBase,self).undelete_edge(e,*args,**kwargs)
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[ self.edges[e,0] ] = 'undelete_edge'
            self.holding_nodes[ self.edges[e,1] ] = 'undelete_edge'
        else:
            self.dt_add_edge(e)

    def merge_edges(self,e1,e2):
        if self.verbose > 1:
            print("    live_dt: merge edges %d %d"%(e1,e2))
        # the tricky thing here is that we don't know which of
        # these edges will be removed by merge_edges - one
        # of them will get deleted, and then deleted by our
        # delete handler.
        # the other one will get modified, so by the time we get
        # control again after trigrid, we won't know what to update
        # so - save the nodes...
        saved_nodes = self.edges[ [e1,e2],:2]

        remaining = super(LiveDtGridBase,self).merge_edges(e1,e2)

        if self.freeze:
            pass
        elif self.holding:
            for n in saved_nodes.ravel():
                self.holding_nodes[n] = 'merge_edges'
        else:
            if remaining == e1:
                ab = saved_nodes[0]
            else:
                ab = saved_nodes[1]

            # the one that is *not* remaining has already been deleted
            # just update the other one.
            try:
                self.dt_remove_edge(remaining,nodes=ab)
            except MissingConstraint:
                print("    on merge_edges, may have an intervener")
                raise

            self.dt_add_edge(remaining)
        return remaining

    def unmerge_edges(self,e1,e2,*args,**kwargs):
        check_dt_after = False
        if self.freeze:
            pass
        elif self.holding:
            pass
        else:
            # this can be problematic if the middle node is exactly on
            # the line between them, because re-inserting that node
            # will pre-emptively segment the constrained edge.
            try:
                self.dt_remove_edge(e1)
            except MissingConstraint:
                print(" got a missing constraint on merge edges - will verify that it's okay")
                check_dt_after = True
        #print "    after pre-emptive remove_edge"

        super(LiveDtGridBase,self).unmerge_edges(e1,e2,*args,**kwargs)
        #print "    after call to super()"
        if self.freeze:
            pass
        elif self.holding:
            n1,n2 = self.edges[e1,:2]
            n3,n4 = self.edges[e2,:2]
            for n in [n1,n2,n3,n4]:
                self.holding_nodes[ n ] = 'unmerge_edges'
        else:
            if check_dt_after:
                AB = self.edges[e1,:2]
                BC  = self.edges[e2,:2]
                B = np.intersect1d(AB,BC)[0]
                A = np.setdiff1d(AB,B)[0]
                C = np.setdiff1d(BC,B)[0]
                print("while unmerging edges, a constraint was pre-emptively created, but will verify that now %d-%d-%d."%(A,B,C))

                for edge in self.dt_incident_constraints(self.vh[B]):
                    v1,v2 = edge.vertices()
                    if self.vh_info[v1] == A or self.vh_info[v2] == A:
                        A = None
                    elif self.vh_info[v1] == B or self.vh_info[v2] == B:
                        B = None
                    else:
                        print("while unmerging edge, the middle point has another constrained DT neighbor - surprising...")
                if A is not None or B is not None:
                    raise MissingConstraint("Failed to verify that implicit constraint was there")
            else:
                #print "    adding reverted edge e1 and e2"
                self.dt_add_edge(e1)
                # even though trigrid.merge_edges() calls delete_edge()
                # on e2, it doesn't register an undelete_edge() b/c
                # rollback=0.
                self.dt_add_edge(e2)

    # def unsplit_edge(...): # not supported by trigrid

    def split_edge(self,nodeA,nodeB,nodeC):
        """ per trigrid updates, nodeB may be a node index or a tuple (coords, **add_node_opts)
        """
        if self.freeze:
            pass
        elif self.holding:
            self.holding_nodes[nodeA] = 'split_edge'
            if not isinstance(nodeB,Iterable):
                self.holding_nodes[nodeB] = 'split_edge'
            self.holding_nodes[nodeC] = 'split_edge'
        else:
            if self.verbose > 2:
                print("    split_edge: %d %d %d"%(nodeA,nodeB,nodeC))
            e1 = self.find_edge([nodeA,nodeC])
            try:
                self.dt_remove_edge(e1)
            except MissingConstraint:
                if isinstance(nodeB,Iterable):
                    print("    got a missing constraint on split edge, and node has not been created!")
                    raise
                else:
                    print("    got a missing constraint on split edge, but maybe the edge has already been split")
                    self.dt_remove_edge(e1,[nodeA,nodeB])
                    self.dt_remove_edge(e1,[nodeB,nodeC])
                    print("    Excellent.  The middle node had become part of the constraint")

        e2 = super(LiveDtGridBase,self).split_edge(nodeA,nodeB,nodeC)

        if self.freeze:
            pass
        elif self.holding:
            pass
        else:
            self.dt_add_edge(e1)
            self.dt_add_edge(e2)
        return e2

    def delete_node_and_merge(self,n):
        if self.freeze:
            return super(LiveDtGridBase,self).delete_node_and_merge(n)

        if self.holding:
            self.holding_nodes[n] = 'delete_node_and_merge'
        else:
            # remove any constraints going to n -
            self.dt_remove_constraints(self.vh[n])
            self.dt_remove(n)

        # note that this is going to call merge_edges, before it
        # calls delete_node() - and merge_edges will try to add the new
        # constraint, which will fail if the middle node is collinear with
        # the outside nodes.  so freeze LiveDT updates, then here we clean up
        self.freeze = 1
        new_edge = super(LiveDtGridBase,self).delete_node_and_merge(n)
        if self.verbose > 2:
            print("    Got new_edge=%s from trigrid.delete_node_and_merge"%new_edge)
        self.freeze=0

        if self.holding:
            for n in self.edges[new_edge,:2]:
                self.holding_nodes[n] = 'delete_node_and_merge'
        else:
            # while frozen we missed a merge_edges and a delete node.
            # we just want to do them in the opposite order of what trigrid does.
            self.dt_add_edge(new_edge)

        return new_edge

    def renumber(self):
        mappings = super(LiveDtGridBase,self).renumber()

        self.vh = self.vh[ mappings['valid_nodes'] ]

        for i in range(len(self.vh)):
            self.vh_info[self.vh[i]] = i

        return mappings
        
    def dt_interior_cells(self):
        """
        Only valid for a triangulation where all nodes lie on
        the boundary.  there will be some
        cells which fall inside the domain, others outside the
        domain.
        returns cells which are properly inside the domain as 
        triples of nodes 
        """
        log.info("Finding interior cells from full Delaunay Triangulation")
        interior_cells = []

        for a,b,c in self.dt_cell_node_iter():
            # going to be slow...
            # How to test whether this face is internal:
            #  Arbitrarily choose a vertex: a
            #
            # Find an iter for which the face abc lies to the left of the boundary
            internal = 0
            for elt in self.all_iters_for_node(a):
                d = self.points[elt.nxt.data] - self.points[a]
                theta_afwd = np.arctan2(d[1],d[0])
                d = self.points[b] - self.points[a]
                theta_ab   = np.arctan2(d[1],d[0])
                d = self.points[elt.prv.data] - self.points[a]
                theta_aprv = np.arctan2(d[1],d[0])

                dtheta_b = (theta_ab - theta_afwd) % (2*np.pi)
                dtheta_elt = (theta_aprv - theta_afwd) % (2*np.pi)

                # if b==elt.nxt.data, then dtheta_b==0.0 - all good
                if dtheta_b >= 0 and dtheta_b < dtheta_elt:
                    internal = 1
                    break
            if internal:
                interior_cells.append( [a,b,c] )

        cells = np.array(interior_cells)
        return cells


    
    ## DT-based "smoothing"
    # First, make sure the boundary is sufficiently sampled
    def subdivide(self,min_edge_length=1.0,edge_ids=None):
        """ Like medial_axis::subdivide_iterate -
        Add nodes along the boundary as needed to ensure that the boundary
        is well represented in channels

        [ from medial_axis ]
        Find edges that need to be sampled with smaller
        steps and divide them into two edges.
        returns the number of new edges / nodes

        method: calculate voronoi radii
        iterate over edges in boundary
        for each edge, find the voronoi point that they have
        in common.  So this edge should be part of a triangle,
        and we are getting the center of that triangle.

        the voronoi radius with the distance between the voronoi
        point and the edge.  If the edge is too long and needs to
        be subdivided, it will be long (and the voronoi radius large)
        compared to the distance between the edge and the vor. center.
        """

        if edge_ids is None:
            print("Considering all edges for subdividing")
            edge_ids = list(range(self.Nedges()))
        else:
            print("Considering only %d supplied edges for subdividing"%len(edge_ids))

        to_subdivide = []

        # Also keep a list of constrained edges of DT cells for which another edge
        # has been selected for subdivision.
        neighbors_of_subdivide = {}

        print("Choosing edges to subdivide")
        for ni,i in enumerate(edge_ids):   # range(self.Nedges()):
            if ni%500==0:
                log.debug('.')

            if self.edges[i,0] == -37:
                continue # edge has been deleted
            # this only works when one side is unpaved and the other boundary -
            if self.edges[i,3] != trigrid.UNMESHED or self.edges[i,4] != trigrid.BOUNDARY:
                print("Skipping edge %d because it has weird cell ids"%i)
                continue

            a,b = self.edges[i,:2]

            # consult the DT to find who the third node is:
            a_nbrs = self.delaunay_neighbors(a)
            b_nbrs = self.delaunay_neighbors(b)
            abc = np.array([self.points[a],self.points[b],[0,0]])

            c = None
            for nbr in a_nbrs:
                if nbr in b_nbrs:
                    # does it lie to the left of the edge?
                    abc[2,:] = self.points[nbr]
                    if trigrid.is_ccw(abc):
                        c = nbr
                        break
            if c is None:
                print("While looking at edge %d, %s - %s"%(i,self.points[a],self.points[b]))
                raise Exception("Failed to find the third node that makes up an interior triangle")

            pntV = trigrid.circumcenter(abc[0],abc[1],abc[2])

            # compute the point-line distance between
            # this edge and the v center, then compare to
            # the distance from the endpoint to that
            # vcenter
            pntA = self.points[a]
            pntB = self.points[b]

            v_radius = np.sqrt( ((pntA-pntV)**2).sum() )
            # This calculates unsigned distance - with Triangle, that's fine because
            # it takes care of the Steiner points, but with CGAL we do it ourselves.
            # line_clearance = np.sqrt( (( 0.5*(pntA+pntB) - pntV)**2).sum() )
            ab = (pntB - pntA)
            ab = ab / np.sqrt( np.sum(ab**2) )
            pos_clearance_dir = np.array( [-ab[1],ab[0]] )
            av = pntV - pntA
            line_clearance = av[0]*pos_clearance_dir[0] + av[1]*pos_clearance_dir[1]

            # Why do I get some bizarrely short edges?
            ab = np.sqrt( np.sum( (pntA - pntB)**2 ) )

            if v_radius > 1.2*line_clearance and v_radius > min_edge_length and ab>min_edge_length:
                to_subdivide.append(i)
                # Also make note of the other edges of this same DT triangle
                for maybe_nbr in [ [a,c], [b,c] ]:
                    # could be an internal DT edge, or a real edge
                    try:
                        nbr_edge = self.find_edge(maybe_nbr)
                        neighbors_of_subdivide[nbr_edge] = 1
                    except trigrid.NoSuchEdgeError:
                        pass
        print()
        print("Will subdivide %d edges"%(len(to_subdivide)))
        for ni,i in enumerate(to_subdivide):
            if ni%500==0:
                log.debug('.')

            if i in neighbors_of_subdivide:
                del neighbors_of_subdivide[i]

            a,b = self.edges[i,:2]

            elts = self.all_iters_for_node(a)
            if len(elts) != 1:
                raise Exception("How is there not exactly one iter for this node!?")
            scale = 0.5*np.sqrt( np.sum( (self.points[a]-self.points[b])**2 ) )

            # print "Subdividing edge %d with scale %f"%(i,scale)
            new_elt = self.resample_boundary(elts[0],'forward',
                                             local_scale=scale,
                                             new_node_stat=self.node_data[a,0])
            # keep track of any edges that change:
            e1,e2 = self.pnt2edges(new_elt.data)
            neighbors_of_subdivide[e1] = 1
            neighbors_of_subdivide[e2] = 1

        print("done")
        subdivided = np.array( list(neighbors_of_subdivide.keys()) )
        return subdivided

    def subdivide_iterate(self,min_edge_length=1.0):
        modified_edges = None
        while 1:
            # It wasn't enough to just test for no modified edges - rather than
            # trying to be clever about checking exactly edges that may have
            # been affected by a split, have nested iterations, and stop only
            # when globally there are no modified edges
            new_modified_edges = self.subdivide(min_edge_length=min_edge_length,
                                                edge_ids = modified_edges)

            print("Subdivide made %d new nodes"%(len(new_modified_edges)/2) )
            if len(new_modified_edges) == 0:
                if modified_edges is None:
                    # this means we considered all edges, and still found nobody
                    # to split
                    break
                else:
                    # this means we were only considering likely offenders -
                    # step back and consider everyone
                    print("Will reconsider all edges...")
                    modified_edges = None
            else:
                modified_edges = new_modified_edges

    def smoothed_poly(self,density,min_edge_length=1.0):
        """ Returns a polygon for the boundary that has all 'small' concave features
        removed.  Modifies the boundary points, but only by adding new samples evenly
        between originals.
        """
        # Make sure that all edges are sufficiently sampled:
        self.subdivide_iterate(min_edge_length=min_edge_length)

        # The process (same as in smoother.py):

        # For all _interior_ DT cells
        #  calculate circum-radius
        #  mark for deletion any cell with radius < scale/2,
        #  with scale calculated at circumcenter
        # For all non-deleted cells, create an array of all edges
        #  The notes in smoother say that if an edge appears exactly once
        #  then it should be kept. 
        # Edges that appear twice are internal to the domain.
        #  If/when degenerate edges take part in this, they will have to be
        #  handled specially, since they *should* have two adjacent, valid, cells.

        # What is faster?
        #  (a) iterate over known boundary edges, grabbing cells to the left,
        #      and checking against a hash to see that the cell hasn't been included
        #      already
        #  (b) iterate over DT faces, checking to see if it's an internal face or not
        #      by checking ab,bc,ca against the edge hash?
        # probably (b), since we already have this hash built.
        # Actually, (b) isn't sufficient - there are triangles that are internal, but
        #   have no boundary edges.
        # And (a) isn't good either - it would skip over any triangles that are entirely
        #  internal _or_ entirely external (i.e. share no edges with the boundary).

        # Is there a way to do this by tracing edges?  Start at some vertex on a clist.
        # check the next edge forward - is the radius of the DT face to its left big enough?
        # If so, we move forward.
        # If not, detour?
        # That's not quite enough, though.  Really need to be checking every triangle incident
        # to the vertex, not just the ones incident to the edges.

        # So for simplicity, why not use the traversal of the edges to enumerate internal cells,
        # then proceed as before.

        cells = self.dt_interior_cells()
        print("Marking for deletion DT faces that are too small")
        points = self.points[cells]
        vcenters = trigrid.circumcenter(points[:,0],
                                        points[:,1],
                                        points[:,2])
        # Threshold on the radius, squared -
        # 
        r2_min = (density(vcenters)/2.0 * self.scale_ratio_for_cutoff)**2
        # r^2 for each internal DT face
        r2 = np.sum( (vcenters - points[:,0,:])**2,axis=1)
        valid = r2 >= r2_min

        # From here on out it follows smoother.py very closely...

        print("Compiling valid edges")
        # expands cells into edges
        good_cells = cells[valid]
        all_edges = good_cells[:,np.array([[0,1],[1,2],[2,0]])]
        # cells is Nfaces x 3
        # all_edges is then Nfaces x 3 x 2
        # combine the first two dimensions, so we have a regular edges array
        all_edges = all_edges.reshape( (-1,2) )

        print("building hash of edges")
        edge_hash = {}

        for i in range(len(all_edges)):
            k = all_edges[i,:]
            if k[0] > k[1]:
                k=k[::-1]
            k = tuple(k)
            if k not in edge_hash:
                edge_hash[k] = 0
            edge_hash[k] += 1

        print("Selecting boundary edges")
        # good edges are then edges that appear in exactly one face
        good_edges = []

        for k in edge_hash:
            if edge_hash[k] == 1:
                good_edges.append(k)

        good_edges = np.array(good_edges)

        print("Finding polygons from edges")
        tgrid = trigrid.TriGrid(points=self.points,
                                edges =good_edges)
        tgrid.verbose = 2
        polygons = tgrid.edges_to_polygons(None) # none=> use all edges

        self.smooth_all_polygons = polygons # for debugging..

        print("done with smoothing")
        return polygons[0]


    def apollonius_scale(self,r,min_edge_length=1.0,process_islands=True):
        """ Return an apollonius based field giving the scale subject to
        the local feature size of geo and the telescoping rate r
        """
        self.subdivide_iterate(min_edge_length=min_edge_length)

        dt_cells = self.dt_interior_cells()

        points = self.points[dt_cells]
        vcenters = trigrid.circumcenter(points[:,0],
                                        points[:,1],
                                        points[:,2])

        radii = np.sqrt( np.sum( (vcenters - points[:,0,:])**2,axis=1) )
        diam = 2*radii

        if process_islands:
            print("Hang on.  Adding scale points for islands")

            island_centers = []
            island_scales = []

            for int_ring in self.poly.interiors:
                p = int_ring.convex_hull

                points = np.array(p.exterior.coords)
                center = points.mean(axis=0)

                # brute force - find the maximal distance between
                # any two points.  probably a smart way to do this,
                # but no worries...
                max_dsqr = 0
                for i in range(len(points)):
                    pa = points[i]
                    for j in range(i,len(points)):
                        d = ((pa - points[j])**2).sum()
                        max_dsqr = max(d,max_dsqr)

                feature_scale = np.sqrt( max_dsqr )
                print("Ring has scale of ",feature_scale)

                island_centers.append( center )
                # this very roughly says that we want at least 4 edges
                # for representing this thing.
                #   island_scales.append( feature_scale / 2.0)
                # actually it's not too hard to have a skinny island
                # 2 units long that gets reduced to a degenerate pair
                # of edges, so go conservative here:
                island_scales.append( feature_scale / 3.0 )

            island_centers = np.array(island_centers)
            island_scales = np.array(island_scales)

            if len(island_centers) > 0:
                vcenters = np.concatenate( (vcenters,island_centers) )
                diam = np.concatenate( (diam,island_scales) )
            print("Done with islands")

        scale = field.ApolloniusField(vcenters,diam)

        return scale
    
try:
    # If CGAL gives import errors, it may be because gmp is outdated
    # This got past conda because the build of mpfr isn't specific
    # about the version of gmp, just says it depends on gmp.

    # One fix in anaconda land is:
    #   conda update gmp to install 6.1.2

    
    from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
    from CGAL.CGAL_Kernel import Point_2

    class LiveDtCGAL(LiveDtGridBase):
        class Edge(object):
            def __init__(self,**kwargs):
                self.__dict__.update(kwargs)
            def vertices(self):
                return self.f.vertex( (self.v+1)%3 ),self.f.vertex( (self.v+2)%3 )

        def dt_allocate(self):
            """ allocate both the triangulation and the vertex handle
            """ 
            self.DT = Constrained_Delaunay_triangulation_2()
        def dt_insert(self,n):
            """ Given a point that is correctly in self.points, and vh that
            is large enough, do the work of inserting the node and updating
            the vertex handle.
            """
            pnt = Point_2( self.points[n,0], self.points[n,1] )
            self.vh[n] = self.DT.insert(pnt)
            self.vh_info[self.vh[n]] = n
            if self.verbose > 2:
                print("    dt_insert node %d"%n)
                self.check()
        def dt_insert_constraint(self,a,b):
            self.DT.insert_constraint( self.vh[a], self.vh[b] )
        def dt_remove_constraints(self,vh):
            self.DT.remove_incident_constraints(vh)
        def dt_remove(self,n):
            self.DT.remove( self.vh[n] )
            del self.vh_info[self.vh[n]]
            self.vh[n] = 0
            if self.verbose > 2:
                print("    dt_remove node %d"%n)
                self.check()
        def dt_remove_constrained_edge(self,edge):
            self.DT.remove_constrained_edge(edge.f,edge.v)

        def dt_incident_constraints(self,vh):
            constraints = []
            self.DT.incident_constraints(vh,constraints)
            # maybe I should keep a reference to the Edge object, too?
            # that gets through some early crashes.
            return [self.Edge(f=e.first,v=e.second,keepalive=[e]) for e in constraints]


        def dt_cell_node_iter(self):
            """ generator for going over finite cells, returning 
            nodes as triples
            """
            face_it = self.DT.finite_faces()

            for f in face_it:
                yield [self.vh_info[f.vertex(i)] for i in [0,1,2]]
        
        def delaunay_face(self, pnt):
            """ Returns node indices making up the face of the DT in which pnt lies.
            Not explicitly tested, but this should return None for infinite nodes.
            """
            f = self.DT.locate( Point_2(pnt[0],pnt[1]) )
            n = [self.vh_info[f.vertex(i)] for i in [0,1,2]]
            return n
        
        def delaunay_neighbors(self, n):
            """ returns an array of node ids that the DT connects the given node
            to.  Includes existing edges
            """
            nbrs = []

            # how do we stop on a circulator?
            first_v = None
            # somehow it fails HERE, with self.vh[n] being an int, rather
            # than a vertex handle.

            # Note that in some cases, this is not an iterator but a circulator
            # (that's a CGAL thing, not a python thing), and it cannot be used
            # in a regular for loop
            circ=self.DT.incident_vertices(self.vh[n])
            while 1:
                v=circ.next()
                if first_v is None:
                    first_v = v
                elif first_v == v:
                    break

                if self.DT.is_infinite(v):
                    continue

                # print "Looking for vertex at ",v.point()

                # This is going to need something faster, or maybe the store info
                # bits of cgal.
                nbr_i = self.vh_info[v] #  np.where( self.vh == v )[0]
                if nbr_i is None:
                    print("    While looking for vertex at ",v.point())
                    raise Exception("expected vertex handle->node, but instead got %s"%nbr_i)
                nbrs.append( nbr_i )
            return np.array(nbrs)

        def plot_dt(self,clip=None):
            edges = []
            colors = []

            gray = (0.7,0.7,0.7,1.0)
            magenta = (1.0,0.0,1.0,1.0)

            e_iter = self.DT.finite_edges()

            for e in e_iter:
                face,vertex = e

                v1 = face.vertex( (vertex + 1)%3 )
                v2 = face.vertex( (vertex + 2)%3 )

                edges.append( [ [v1.point().x(),v1.point().y()],
                                [v2.point().x(),v2.point().y()] ] )
                if self.DT.is_constrained(e):
                    colors.append(magenta)
                else:
                    colors.append(gray)

            segments = np.array(edges)
            colors = np.array(colors)

            if clip is None:
                clip = self.default_clip
            if clip is not None:
                points_visible = (segments[...,0] >= clip[0]) & (segments[...,0]<=clip[1]) \
                                 & (segments[...,1] >= clip[2]) & (segments[...,1]<=clip[3])
                # so now clip is a bool array of length Nedges
                clip = np.any( points_visible, axis=1)
                segments = segments[clip,...]
                colors = colors[clip,...]

            coll = collections.LineCollection(segments,colors=colors)

            ax = plt.gca()
            ax.add_collection(coll)


        def dt_clearance(self,n):
            """POORLY TESTED
            Returns the diameter of the smallest circumcircle (?) of a face
            incident to the node n.  Currently this doesn't work terribly well
            because sliver triangles will create arbitrarily small clearances
            at obtuse angles.
            """
            diams = []
            f_circ = self.DT.incident_faces( self.vh[n] )
            first_f = next(f_circ)
            f = first_f
            while 1:
                f=f_circ.next()
                if f == first_f:
                    break
                diams.append( self.face_diameter(f) )

            return min(diams)

        # Not 100% sure of these
        def face_nodes(self,face):
            return np.array( [self.vh_info[face.vertex(j)] for j in range(3)] )
        def face_center(self,face):
            points = self.points[self.face_nodes(face)]
            return trigrid.circumcenter(points[0],points[1],points[2])
        def face_diameter(self,face):
            points = self.points[self.face_nodes(face)]
            ccenter = trigrid.circumcenter(points[0],points[1],points[2])
            return 2*norm(points[0] - ccenter)

        #-# Detecting self-intersections
        def face_in_direction(self,vh,vec):
            """ 
            Starting at the vertex handle vh, look in the direction
            of vec to choose a face adjacent to vh.

            Used for the CGAL implementation of line_walk_edges and shoot_ray()
            """
            # vh: vertex handle
            # vec: search direction as array
            theta = np.arctan2(vec[1],vec[0])

            # choose a starting face
            best_f = None
            f_circ = self.DT.incident_faces(vh)
            # python 3 notices that f_circ is not an iterator, and complains
            # about use of next()
            first_f = f_circ.next()
            f = first_f
            while 1:
                # get the vertices of this face:
                vlist=[f.vertex(i) for i in range(3)]
                # rotate to make v1 first:
                vh_index = vlist.index(vh)
                vlist = vlist[vh_index:] + vlist[:vh_index]

                # then check the relative angles of the other two - they are in CCW order
                pnts = np.array( [ [v.point().x(),v.point().y()] for v in vlist] )
                delta01 = pnts[1] - pnts[0]
                delta02 = pnts[2] - pnts[0]
                theta01 = np.arctan2( delta01[1], delta01[0] )
                theta02 = np.arctan2( delta02[1], delta02[0] )

                # 
                d01 = (theta - theta01)%(2*np.pi)
                d02 = (theta02 - theta)%(2*np.pi)

                #print "starting point:",pnts[0]
                #print "Theta01=%f  Theta=%f  Theta02=%f"%(theta01,theta,theta02)

                if (d01 < np.pi) and (d02 < np.pi):
                    best_f = f
                    break

                f = f_circ.next()
                if f == first_f:
                    # this can happen when starting from a vertex and aiming
                    # outside the convex hull
                    return None
            return best_f

        def next_face(self,f,p1,vec):
            """ find the next face from f, along the line through v in the direction vec,
            return the face and the edge that was crossed, where the edge is a face,i tuple

            Used for the CGAL implementation of line_walk_edges() and shoot_ray()
            """
            # First get the vertices that make up this face:

            # look over the edges:
            vlist=[f.vertex(i) for i in range(3)]
            pnts = np.array( [ [v.point().x(),v.point().y()] for v in vlist] )

            # check which side of the line each vertex is on:
            left_vec = np.array( [-vec[1],vec[0]] )
            left_distance = [ (pnts[i,0] - p1[0])*left_vec[0] + (pnts[i,1]-p1[1])*left_vec[1] for i in range(3)]

            # And we want the edge that goes from a negative to positive left_distance.
            # should end with i being the index of the start of the edge that we want
            for i in range(3):
                # This doesn't quite follow the same definitions as in CGAL -
                # because we want to ensure that we get a consecutive list of edges

                # The easy case - the query line exits through an edge that straddles
                # the query line, that's the <
                # the == part comes in where the query line exits through a vertex.
                # in that case, we choose the edge to the left (arbitrary).
                if left_distance[i] <= 0 and left_distance[(i+1)%3] > 0:
                    break
            # so now the new edge is between vertex i,(i+1)%3, so in CGAL parlance
            # that's
            edge = (f,(i-1)%3)
            new_face = f.neighbor( (i-1)%3 )
            return edge,new_face

        # N.B.: the svn (and original git) versions of live_dt included
        # a new set of check_line_is_clear_new, line_walk_edges_new, and
        # various helpers, which made more extensive use of CGAL primitives
        # 

        ## 
        def line_walk_edges(self,n1=None,n2=None,v1=None,v2=None,
                            include_tangent=False,
                            include_coincident=True):
            """ for a line starting at node n1 or vertex handle v1 and
            ending at node n2 or vertex handle v2, return all the edges
            that intersect.

            Used in the CGAL implementation of check_line_is_clear
            """
            # this is a bit dicey in terms of numerical robustness - 
            # face_in_direction is liable to give bad results when multiple faces are
            # indistinguishable (like a colinear set of points with many degenerate faces
            # basically on top of each other).

            # How can this be made more robust?
            # When the query line exactly goes through one or more vertex stuff starts
            # going nuts.
            # So is it possible to handle this more intelligently?
            #   there are 3 possibilities for intersecting edges:
            #    (1) intersect only at an end point, i.e. endpoint lies on query line
            #    (2) intersect in interior of edge - one end point on one side, other endpoint
            #        on the other side of the query line
            #    (3) edge is coincident with query line


            # so for a first cut - make sure that we aren't just directly connected:
            if (n2 is not None) and (n1 is not None) and (n2 in self.delaunay_neighbors(n1)):
                return []

            if v1 is None:
                v1 = self.vh[n1]
            if v2 is None:
                v2 = self.vh[n2]

            # Get the points from the vertices, not self.points, because in some cases
            # (adjust_move_node) we may be probing
            p1 = np.array([ v1.point().x(), v1.point().y()] )
            p2 = np.array([ v2.point().x(), v2.point().y()] )

            # print "Walking the line: ",p1,p2

            vec = p2 - p1
            unit_vec = vec / norm(vec)

            pnt = p1 

            # NB: this can be None - though not sure whether the context can
            # ensure that it never would be.
            f1 = self.face_in_direction(v1,vec)
            f2 = self.face_in_direction(v2,-vec)

            # do the search:
            f_trav = f1
            edges = []
            while 1:
                # print "line_walk_edges: traversing face:"
                # print [f_trav.vertex(i).point() for i in [0,1,2]]

                # Stop condition: we're in a face containing the final vertex
                # check the vertices directly, rather than the face
                still_close = 0
                for i in range(3):
                    if f_trav.vertex(i) == v2:
                        return edges

                    if not still_close:
                        # Check to see if this vertex is beyond the vertex of interest
                        vertex_i_pnt = np.array( [f_trav.vertex(i).point().x(),f_trav.vertex(i).point().y()] )
                        if norm(vec) > np.dot( vertex_i_pnt - p1, unit_vec):
                            still_close = 1

                if not still_close:
                    # We didn't find any vertices of this face that were as close to where we started
                    # as the destination was, so we must have passed it.
                    print("BAILING: n1=%s n2=%s v1=%s v2=%s"%(n1,n2,v1,v2))
                    raise Exception("Yikes - line_walk_edges exposed its numerical issues.  We traversed too far.")
                    return edges

                edge,new_face = self.next_face(f_trav,pnt,vec)

                edges.append(edge)

                f_trav = new_face
            return edges

        def shoot_ray(self,n1,vec,max_dist=None):
            """ Shoot a ray from self.points[n] in the given direction vec
            returns (e_index,pnt), the first edge that it encounters and the location
            of the intersection 

            max_dist: stop checking beyond this distance -- currently doesn't make it faster
              but will return None,None if the point that it finds is too far away
            """

            v1 = self.vh[n1]
            vec = vec / norm(vec) # make sure it's a unit vector
            pnt = self.points[n1]

            f1 = self.face_in_direction(v1,vec)
            if f1 is None:
                return None,None
            
            # do the search:
            f_trav = f1

            while 1:
                edge,new_face = self.next_face(f_trav,pnt,vec)
                # make that into a cgal edge:
                e = edge
                face,i = edge
                va = face.vertex((i+1)%3)
                vb = face.vertex((i-1)%3)

                if max_dist is not None:
                    # Test the distance as we go...
                    pa = va.point()
                    pb = vb.point()

                    d1a = np.array([pa.x()-pnt[0],pa.y() - pnt[1]])

                    # alpha * vec + beta * ab = d1a
                    # | vec[0] ab[0]   | | alpha | = |  d1a[0]  |
                    # | vec[1] ab[1]   | | beta  | = |  d1a[1]  |

                    A = np.array( [[vec[0],  pb.x() - pa.x()],
                                   [vec[1],  pb.y() - pa.y()]] )
                    alpha_beta = solve(A,d1a)

                    dist = alpha_beta[0]
                    if dist > max_dist:
                        return None,None

                if self.DT.is_constrained(e):
                    # print "Found a constrained edge"
                    break
                f_trav = new_face


            na = self.vh_info[va]
            nb = self.vh_info[vb]

            if (na is None) or (nb is None):
                raise Exception("Constrained edge is missing at least one node index")

            if max_dist is None:
                # Compute the point at which they intersect:
                ab = self.points[nb] - self.points[na]
                d1a = self.points[na] - pnt

                # alpha * vec + beta * ab = d1a
                # | vec[0] ab[0]   | | alpha | = |  d1a[0]  |
                # | vec[1] ab[1]   | | beta  | = |  d1a[1]  |

                A = np.array( [[vec[0],ab[0]],[vec[1],ab[1]]] )
                alpha_beta = solve(A,d1a)
            else:
                pass # already calculated alpha_beta

            p_int = pnt + alpha_beta[0]*vec
            edge_id = self.find_edge((na,nb))

            return edge_id,p_int


        ## steppers for line_walk_edges_new
        def next_from_vertex(self, vert, vec):
            # from a vertex, we either go into one of the faces, or along an edge
            qp1,qp2 = vec

            last_left_distance=None
            last_nbr = None

            start = None
            v_circ=self.DT.incident_vertices(vert)
            while 1:
                nbr=v_circ.next()

                if self.DT.is_infinite(nbr):
                    continue
                pnt = np.array( [nbr.point().x(),nbr.point().y()] )

                # fall back to robust_predicates for proper comparison
                # when pnt is left of qp1 => qp2, result should be positive
                left_distance = robust_predicates.orientation(pnt, qp1, qp2)

                # This used to be inside the last_left_distance < 0 block, but it seems to me
                # that if we find a vertex for which left_distance is 0, that's our man.
                # NOPE - having it inside the block caused the code to discard a colinear vertex
                # that was behind us.
                # in the corner case of three colinear points, and we start from the center, both
                # end points will have left_distance==0, and both will be preceeded by the infinite
                # vertex.  So to distinguish colinear points it is necessary to check distance in the
                # desired direction.
                if left_distance==0.0:
                    vert_xy=[vert.point().x(),vert.point().y()]
                    progress=exact_delaunay.rel_ordered(vert_xy,pnt,qp1,qp2)

                    if progress:
                        return ['v',nbr]

                # Note that it's also possible for the infinite vertex to come up.
                # this should be okay when the left_distance==0.0 check is outside the
                # block below.  If it were inside the block, then we would miss the
                # case where we see the infinite vertex (which makes last_left_distance
                # undefined), and then see the exact match.

                if last_left_distance is not None and last_left_distance < 0:
                    # left_distance == 0.0 used to be here.
                    if left_distance > 0:
                        # what is the face between the last one and this one??
                        # it's vertices are vert, nbr, last_nbr
                        f_circ=self.DT.incident_faces(vert)
                        f0=None
                        while 1: # for face in :
                            face=f_circ.next()
                            for j in range(3):
                                if face.vertex(j) == nbr:
                                    for k in range(3):
                                        if face.vertex(k) == last_nbr:
                                            return ['f',face]
                            if f0 is None:
                                f0=face
                            else:
                                assert face!=f0,"Failed to leave circulator loop"
                        raise Exception("Found a good pair of incident vertices, but failed to find the common face.")

                # Sanity check - if we've gone all the way around
                if start is None:
                    start = nbr
                else: # must not be the first time through the loop:
                    if nbr == start:
                        raise Exception("This is bad - we checked all vertices and didn't find a good neighbor")

                last_left_distance = left_distance
                last_nbr = nbr
                if self.DT.is_infinite(nbr):
                    last_left_distance = None

            raise Exception("Fell through!")

        def next_from_edge(self, edge, vec):
            # vec is the tuple of points defining the query line
            qp1,qp2 = vec
            
            # edge is a tuple of face and vertex index
            v1 = edge[0].vertex( (edge[1]+1)%3 )
            v2 = edge[0].vertex( (edge[1]+2)%3 )
            
            # this means the edge was coincident with the query line
            p1 = v1.point()
            p2 = v2.point()

            p1 = np.array( [p1.x(),p1.y()] )
            p2 = np.array( [p2.x(),p2.y()] )

            line12 = p2 - p1

            if np.dot( line12, qp2-qp1 ) > 0:
                return ['v',v2]
            else:
                return ['v',v1]
            
        def next_from_face(self, f, vec):
            qp1,qp2 = vec
            # stepping through a face, along the query line qp1 -> qp2
            # we exit the face either via an edge, or possibly exactly through a
            # vertex.
            # A lot like next_face(), but hopefully more robust handling of
            # exiting the face by way of a vertex.

            # First get the vertices that make up this face:

            # look over the edges:
            vlist=[f.vertex(i) for i in range(3)]
            pnts = np.array( [ [v.point().x(),v.point().y()] for v in vlist] )

            # check which side of the line each vertex is on:

            # HERE is where the numerical issues come up.
            # could possibly do this in terms of the end points of the query line, in order to
            # at least robustly handle the starting and ending points.
            left_distance = [ distance_left_of_line(pnts[i], qp1,qp2 ) for i in range(3)]

            # And we want the edge that goes from a negative to positive left_distance.
            # should end with i being the index of the start of the edge that we want
            for i in range(3):
                # This doesn't quite follow the same definitions as in CGAL -
                # because we want to ensure that we get a consecutive list of edges

                # The easy case - the query line exits through an edge that straddles
                # the query line, that's the <
                # the == part comes in where the query line exits through a vertex.
                # in that case, we choose the edge to the left (arbitrary).
                if left_distance[i] <= 0 and left_distance[(i+1)%3] > 0:
                    break

                # sanity check
                if i==2:
                    raise Exception("Trying to figure out how to get out of a face, and nothing looks good")

            # Two cases - leaving via vertex, or crossing an edge internally.
            if left_distance[i]==0:
                return ['v',vlist[i]]
            else:
                # so now the new edge is between vertex i,(i+1)%3, so in CGAL parlance
                # that's
                new_face = f.neighbor( (i-1)%3 )
                return ['f',new_face]

        
        def line_walk_edges_new(self,n1=None,n2=None,v1=None,v2=None,
                                include_tangent=False,
                                include_coincident=True):
            # Use the CGAL primitives to implement this in a hopefully more
            # robust way.
            # unfortunately we can't use the line_walk() circulator directly
            # because the bindings enumerate the whole list, making it potentially
            # very expensive.

            # ultimately we want to know edges which straddle the query line
            # as well as nodes that fall exactly on the line.
            # is it sufficient to then return a mixed list of edges and vertices
            # that fall on the query line?
            # and any edge that is coincident with the query line will be included
            # in the output.

            # but what is the appropriate traversal cursor?
            # when no vertices fall exactly on the query line, tracking a face
            #  is fine.
            # but when the query line goes through a vertex, it's probably better
            #  to just record the vertex.
            # so for a first cut - make sure that we aren't just directly connected:
            
            if (n2 is not None) and (n1 is not None) and (n2 in self.delaunay_neighbors(n1)):
                return []

            if v1 is None:
                v1 = self.vh[n1]
            if v2 is None:
                v2 = self.vh[n2]

            # Get the points from the vertices, not self.points, because in some cases
            # (adjust_move_node) we may be probing
            p1 = np.array([ v1.point().x(), v1.point().y()] )
            p2 = np.array([ v2.point().x(), v2.point().y()] )

            if self.verbose > 1:
                print("Walking the line: ",p1,p2)
            
            hits = [ ['v',v1] ]

            # do the search:
            # Note that we really need a better equality test here
            # hits[-1][1] != v2 doesn't work beac
            def obj_eq(a,b):
                return type(a)==type(b) and a==b

            while not obj_eq(hits[-1][1], v2):
                # if we just came from a vertex, choose a new face in the given direction
                if hits[-1][0] == 'v':
                    if self.verbose > 1:
                        print("Last hit was the vertex at %s"%(hits[-1][1].point()))

                    # like face_in_direction, but also check for possibility that
                    # an edge is coincident with the query line.

                    next_item = self.next_from_vertex( hits[-1][1],(p1,p2) )

                    if self.verbose > 1:
                        print("Moved from vertex to ",next_item)

                    if next_item[0] == 'v':
                        # Find the edge connecting these two:
                        e0=None
                        e_circ=self.DT.incident_edges( next_item[1] )
                        while 1:
                            e=e_circ.next()
                            f,v_opp = e

                            if f.vertex( (v_opp+1)%3 ) == hits[-1][1] or \
                               f.vertex( (v_opp+2)%3 ) == hits[-1][1]:
                                hits.append( ['e', (f,v_opp)] )
                                break
                            if e0 is None: # sanity check
                                e0=e
                            elif e0==e:
                                raise Exception("Checked all edges, didn't find a hit")

                elif hits[-1][0] == 'f':
                    # either we cross over an edge into another face, or we hit
                    # one of the vertices.

                    next_item = self.next_from_face( hits[-1][1], (p1,p2) )

                    # in case the next item is also a face, go ahead and insert
                    # the intervening edge
                    if next_item[0]=='f':
                        middle_edge = None

                        for v_opp in range(3):
                            if self.verbose > 1:
                                print("Comparing %s to %s looking for the intervening edge"%(hits[-1][1].neighbor(v_opp),
                                                                                             next_item[1]))
                            if hits[-1][1].neighbor(v_opp) == next_item[1]:
                                middle_edge = ['e', (hits[-1][1],v_opp)] 
                                break
                        if middle_edge is not None:
                            hits.append( middle_edge )
                        else:
                            raise Exception("Two faces in a row, but couldn't find the edge between them")

                elif hits[-1][0] == 'e':
                    # This one is easy - just have to check which end of the edge is in the
                    # desired direction
                    next_item = self.next_from_edge( hits[-1][1], (p1,p2) )

                hits.append( next_item )

            if self.verbose > 1:
                print("Got hits: ",hits)

            # but ignore the first and last, since they are the starting/ending points
            hits = hits[1:-1]

            # and since some of those CGAL elements are going to disappear, translate everything
            # into node references
            for i in range(len(hits)):
                if hits[i][0] == 'v':
                    hits[i][1] = [ self.vh_info[ hits[i][1] ] ]
                elif hits[i][0] == 'e':
                    f,v_opp = hits[i][1]

                    hits[i][1] = [ self.vh_info[ f.vertex( (v_opp+1)%3 ) ], self.vh_info[ f.vertex( (v_opp+2)%3 ) ] ]
                elif hits[i][0] == 'f':
                    f = hits[i][1]

                    hits[i][1] = [ self.vh_info[ f.vertex(0) ],
                                   self.vh_info[ f.vertex(1) ],
                                   f.vertex(2) ]

            # have to go back through, and where successive items are faces, we must
            # have crossed cleanly through an edge, and that should be inserted, too
            return hits

        def check_line_is_clear_new(self,n1=None,n2=None,v1=None,v2=None,p1=None,p2=None):
            """ returns a list of vertex tuple for constrained segments that intersect
            the given line.
            in the case of vertices that are intersected, just a tuple of length 1
            (and assumes that all vertices qualify as constrained)
            """

            # if points were given, create some temporary vertices
            if p1 is not None:
                cp1 = Point_2( p1[0], p1[1] )
                v1 = self.DT.insert(cp1) ; self.vh_info[v1] = 'tmp'

            if p2 is not None:
                cp2 = Point_2( p2[0], p2[1] )
                v2 = self.DT.insert(cp2) ; self.vh_info[v2] = 'tmp'

            crossings = self.line_walk_edges_new(n1=n1,n2=n2,v1=v1,v2=v2)

            constrained = []
            for crossing_type,crossing in crossings:
                if crossing_type == 'f':
                    continue
                if crossing_type == 'v':
                    constrained.append( (crossing_type,crossing) )
                    continue
                if crossing_type == 'e':
                    n1,n2 = crossing
                    if self.verbose > 1:
                        print("Got potential conflict with edge",n1,n2)
                    try:
                        self.find_edge( (n1,n2) )
                        constrained.append( ('e',(n1,n2)) )
                    except trigrid.NoSuchEdgeError:
                        pass

            if p1 is not None:
                del self.vh_info[v1]
                self.DT.remove( v1 )
            if p2 is not None:
                del self.vh_info[v2]
                self.DT.remove( v2 )
            return constrained
        
        def check_line_is_clear_batch(self,p1,n2):
            """ 
            When checking multiple nodes against the same point,
            may be faster to insert the point just once.
            p1: [x,y] 
            n2: [ node, node, ... ]
            Return true if segments from p1 to each node in n2 are
            all clear of constrained edges
            """
            pnt = Point_2( p1[0], p1[1] )
            probe = self.DT.insert(pnt)
            self.vh_info[probe] = 'PROBE!'

            try:
                for nbr in n2:
                    crossings = self.check_line_is_clear_new( n1=nbr, v2=probe )
                    if len(crossings) > 0:
                        return False
            finally:
                del self.vh_info[probe]
                self.DT.remove(probe)
                
            return True
        
        def check_line_is_clear(self,n1=None,n2=None,v1=None,v2=None,p1=None,p2=None):
            """ returns a list of vertex tuple for constrained segments that intersect
            the given line
            """

            # if points were given, create some temporary vertices
            if p1 is not None:
                cp1 = Point_2( p1[0], p1[1] )
                v1 = self.DT.insert(cp1) ; self.vh_info[v1] = 'tmp'

            if p2 is not None:
                cp2 = Point_2( p2[0], p2[1] )
                v2 = self.DT.insert(cp2) ; self.vh_info[v2] = 'tmp'

            edges = self.line_walk_edges(n1=n1,n2=n2,v1=v1,v2=v2)
            constrained = []
            for f,i in edges:
                e = (f,i)

                if self.DT.is_constrained(e):
                    vA = f.vertex( (i+1)%3 )
                    vB = f.vertex( (i+2)%3 )
                    print("Conflict info: ",self.vh_info[vA],self.vh_info[vB])
                    constrained.append( (vA,vB) )

            if p1 is not None:
                del self.vh_info[v1]
                self.DT.remove( v1 )
            if p2 is not None:
                del self.vh_info[v2]
                self.DT.remove( v2 )
            return constrained

    
    LiveDtGrid=LiveDtCGAL
        
except ImportError as exc:
    log.warning("CGAL unavailable.")


# Seems like the Edge class is something provided by each
# implementation, and is essentially opaque to LiveDtGridBase.
# it just needs to supply a vertices() method which gives
# the handles for the relevant vertices.

class LiveDtPython(LiveDtGridBase):
    vh_dtype=object

    class Edge(object):
        def __init__(self,g,j):
            self.g=g
            self.j=j
        def vertices(self):
            return self.g.edges['nodes'][self.j]
        
    def dt_allocate(self):
        self.DT=exact_delaunay.Triangulation()
    
    def dt_insert_constraint(self, a, b):
        self.DT.add_constraint(self.vh[a], self.vh[b])
        
    def dt_remove_constraints(self, vh):
        """
        remove all constraints in which node n participates
        """
        # this used to pass the node, but it should be the vertex handle:
        for e in self.dt_incident_constraints(vh):
            a,b = self.DT.edges['nodes'][e.j]
            self.DT.remove_constraint(j=e.j)
            
    def dt_insert(self, n):
        """ Given a point that is correctly in self.points, and vh that
        is large enough, do the work of inserting the node and updating
        the vertex handle.
        """
        # pnt = Point_2( self.points[n,0], self.points[n,1] )
        xy=[self.points[n,0],self.points[n,1]]
        self.vh[n] = self.DT.add_node(x=xy)
        self.vh_info[self.vh[n]] = n
        if self.verbose > 2:
            print("    dt_insert node %d"%n)
            self.check()
            
    def dt_remove(self,n):
        self.DT.delete_node( self.vh[n] )
        del self.vh_info[self.vh[n]]
        self.vh[n] = None # had been 0, but that's a valid index
        if self.verbose > 2:
            print("    dt_remove node %d"%n)
            self.check()
            
    def dt_remove_constrained_edge(self,edge):
        self.DT.remove_constraint(j=edge.j)

    def dt_incident_constraints(self,vh):
        return [self.Edge(g=self.DT,j=e)
                for e in self.DT.node_to_constraints(vh)]

    def dt_cell_node_iter(self):
        """ generator for going over finite cells, returning 
        nodes as triples
        """
        for c in self.DT.valid_cell_iter():
            yield [self.vh_info[n] for n in self.DT.cells['nodes'][c,:3]]
    
    def delaunay_face(self, pnt):
        """ 
        Returns node indices making up the face of the DT in which pnt lies.
        Always returns 3 items, but any number of them could be None.
        In the case that pnt is on an edge or vertex adjacent to a cell, 
        then all three of the cell's nodes are returned, though the specific
        choice of cell is arbitrary.  Not sure if that's the right behavior
        for the current usage of delaunay_face()
        """
        face,loc_type,loc_index = self.DT.locate(pnt)
        if face != self.DT.INF_CELL:
            nodes = [self.vh_info[n] for n in self.DT.cells['nodes'][face]]
        elif loc_type == self.DT.IN_VERTEX:
            nodes = [self.vh_info[loc_index],None,None]
        elif loc_type == self.DT.IN_EDGE:
            e_nodes=self.DT.edges['nodes'][loc_index]
            nodes = [self.vh_info[e_nodes[0]],
                     self.vh_info[e_nodes[1]],
                     None]
        else:
            return [None,None,None]
        return n

    def delaunay_neighbors(self, n):
        """ returns an ndarray of node ids that the DT connects the given node
        to.  Includes existing edges.
        """
        # some callers assume this is an ndarray
        return np.array( [self.vh_info[vh]
                          for vh in self.DT.node_to_nodes(self.vh[n]) ] )

    def plot_dt(self,clip=None):
        self.DT.plot_edges(clip=clip,color='m')
        
    def shoot_ray(self,n1,vec,max_dist=1e6):
        """ Shoot a ray from self.points[n] in the given direction vec
        returns (e_index,pnt), the first edge that it encounters and the location
        of the intersection. 

        max_dist: stop checking beyond this distance -- currently doesn't make it faster
          but will return None,None if the point that it finds is too far away
        """
        # is it just constrained edges? yes -- just an "edge" in self, but a constrained
        # edge in self.DT.
        nA=self.vh[n1] # map to DT node
        # construct target point
        probe=self.DT.nodes['x'][nA] + max_dist*utils.to_unit(vec)
        
        for elt_type,elt_idx in self.DT.gen_intersected_elements(nA=nA,pB=probe):
            if elt_type=='node':
                if elt_idx==nA:
                    continue
                else:
                    # means that we went exactly through some node, and
                    # the caller probably would just want one of the edges of that
                    # node that is facing nA.
                    X=self.DT.nodes['x'][elt_idx]
                    # a bit awkward, as we likely have come in on an unconstrained
                    # edge of the DT, so no cell info to help, and there could be
                    # many constrained edges to choose from, but we want one of the
                    # the two that face p
                    n=self.vh_info[elt_idx] # get back to the grid's node index

                    adj_nodes=self.DT.angle_sort_adjacent_nodes(elt_idx)
                    # probably overkill -
                    # iterate through adjacent DT nodes, checking orientation
                    # relative to nA, stop when a pair of successive edges
                    # brackets nA.
                    orientations=[] #  [(adj_DT_node_index, orientation), ...]
                    for i_adj,n_adj in enumerate(adj_nodes):
                        j=self.DT.nodes_to_edge(elt_idx,n_adj)
                        if not self.DT.edges['constrained'][j]:
                            continue
                        # looking for a transition from >0 to <=0
                        ori=robust_predicates.orientation(self.DT.nodes['x'][nA],
                                                          self.DT.nodes['x'][elt_idx],
                                                          self.DT.nodes['x'][n_adj])
                        if len(orientations):
                            if orientations[-1][1]>0 and ori<=0:
                                my_j = self.find_edge([n,self.vh_info[n_adj]])
                                return (my_j,X)
                        
                        orientations.append( (n_adj,ori) )
                    # node with no constrained edges -- we were called with bad
                    # state?  shouldn't leave nodes hanging around
                    assert len(orientations)>0
                    if len(orientations)==1:
                        sel=0
                    else:
                        # we checked all other pairs - must be this one
                        assert orientations[-1][1]>0 and orientations[0][1]<=0
                        sel=0
                    my_j = self.find_edge([n,self.vh_info[orientations[sel][0]]])
                    return (my_j,X)
                        
            elif elt_type=='cell':
                continue
            elif elt_type=='edge':
                if self.DT.edges['constrained'][elt_idx.j]:
                    he=elt_idx # pretty sure the half-edge is facing nA
                    n_left=self.vh_info[he.node_fwd()]
                    n_right=self.vh_info[he.node_rev()]

                    j=self.find_edge([n_left,n_right])

                    # Construct point of intersection
                    X=ray_intersection(self.points[n1],vec,
                                       self.points[n_left],self.points[n_right])
                    return (j,X)
                else:
                    continue
            else:
                assert False # sanity...
        return None,None # didn't hit any constrained edges or nodes within the alloted distance
        
    def check_line_is_clear(self,n1=None,n2=None,v1=None,v2=None,p1=None,p2=None):
        """ returns a list of vertex tuple for constrained segments that intersect
        the given line.

        n1: specify one end of the segment by a node index in self
        p1: specify one end of the segment by a point, which will be inserted as
           as temporary vertex.
        v1: specify one end of the segment by a vertex which is already in the DT.
        """

        # What are the options for implementing this in exact_delaunay?
        # probably just go with find_intersected_elements, which takes
        # a pair of nodes, and returns nodes, edges, cells which are intersected
        # one quirk is that traversing along an edge is implied by sequential
        # nodes.
        # also edges are returned as half-edges
        
        # from the CGAL implementation:
        # if points were given, create some temporary vertices.
        temp_dt_nodes=[]
        
        if p1 is not None:
            assert n1 is None
            assert v1 is None
            v1 = self.DT.add_node(x=p1)
            self.vh_info[v1] = 'tmp'
            temp_dt_nodes.append(v1)

        if p2 is not None:
            assert n2 is None
            assert v2 is None
            v2 = self.DT.add_node(x=p2)
            self.vh_info[v2] = 'tmp'
            temp_dt_nodes.append(v2)

        if v1 is None:
            assert n1 is not None
            v1=self.vh[n1]
        if v2 is None:
            assert n2 is not None
            v2=self.vh[n2]
        
        crossings = self.DT.find_intersected_elements(v1,v2)

        constr_pairs=[]

        # crossings includes the starting and ending node, which we don't
        # care about
        for ctype,cindex in crossings[1:-1]:
            if ctype=='node':
                n=cindex
                # may have to adjust this to be more in line with older code
                # it's a conflict if a proposed edge goes exactly through an existing
                # vertex, so we return a conflict based on the same vertex repeated
                # this doesn't directly include the case where the query segment
                # is coincident with an edge, but for the purpose of check_line_is_clear,
                # it's enough to avoid vertices and crossing constrained edges.
                constr_pairs.append( [self.vh_info[n],self.vh_info[n]] )
            elif ctype=='edge':
                j=cindex.j
                if self.DT.edges['constrained'][cindex.j]:
                    # yep, crossing a constrained edge
                    pair=[self.vh_info[vh]
                          for vh in self.DT.edges['nodes'][cindex.j]]
                    constr_pairs.append(pair)
            elif ctype=='cell':
                # don't care about passing through cells
                pass
            else:
                assert False
        
        # clean up
        for v in temp_dt_nodes:
            self.DT.delete_node(v)
            del self.vh_info[v]

        return constr_pairs

    def check_line_is_clear_batch(self,p1,n2):
        """ 
        When checking multiple nodes against the same point,
        may be faster to insert the point just once.
        p1: [x,y] 
        n2: [ node, node, ... ]
        Return true if segments from p1 to each node in n2 are
        all clear of constrained edges
        """
        for nbr in n2:
            crossings = self.check_line_is_clear( n1=nbr, p2=p1 )
            if len(crossings) > 0:
                return False

        return True

    def check_line_is_clear_new(self,*a,**k):
        # cruft, but not quite ready to get rid of this in CGAL
        # before more testing
        return self.check_line_is_clear(*a,**k)
    
if LiveDtGrid==LiveDtGridNull:
    LiveDtGrid=LiveDtPython

    
# how many of these are actually used?
# shoot_ray: definitely used in paver.py
# line_walk_edges:
# check_line_is_clear - seems like only this one is used.
# oops - there are actually many uses of check_line_is_clear_new

# face_in_direction and next_face:
#   used in
#     shoot_ray, which is used in paver
#     line_walk_edges, which itself is used in check_line_is_clear

# What is an appropriate level for the abstraction?
#   exact_delaunay has find_intersected_elements, which is basically line_walk_edges.
#   can something be added to support shoot_ray?

# Decide to use check_line_is_clear(), and shoot_ray() as the point of abstraction
# the check_line_is_clear_new and line_walk_edges_new methods are commented but
# temporarily kept around
#

# face_nodes, face_center, face_diameter?
