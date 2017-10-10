"""
ShadowCDT: maintain a constrained delaunay triangulation
which watches operations on a parent unstructured_grid.
"""

# most of the interfacing is done via regular queries to
# the underlying unstructured_grid:
#   node_to_nodes(n)
#   raise cdt.IntersectingConstraints or similar

import numpy as np
import logging
from collections import defaultdict

log=logging.getLogger(__name__)

from . import exact_delaunay

class ShadowCDT(exact_delaunay.Triangulation):
    """ Tracks modifications to an unstructured grid and
    maintains a shadow representation with a constrained Delaunay
    triangulation, which can be used for geometric queries and
    predicates.
    """
    def __init__(self,g,ignore_existing=False):
        super(ShadowCDT,self).__init__(extra_node_fields=[('g_n','i4')])
        self.g=g
        
        self.nodemap_g_to_local={}

        g.subscribe_before('add_node',self.before_add_node)
        g.subscribe_after('add_node',self.after_add_node)
        g.subscribe_before('modify_node',self.before_modify_node)
        g.subscribe_before('delete_node',self.before_delete_node)
        
        g.subscribe_before('add_edge',self.before_add_edge)
        g.subscribe_before('delete_edge',self.before_delete_edge)
        g.subscribe_before('modify_edge',self.before_modify_edge)

        if not ignore_existing and g.Nnodes():
            self.init_from_grid(g)

    def init_from_grid(self,g): # ShadowCDT
        # Nodes:
        n_valid=~g.nodes['deleted']
        points=g.nodes['x'][n_valid]
        self.bulk_init(points)

        pidxs=np.arange(g.Nnodes())[n_valid]
        self.nodes['g_n']=pidxs

        for n in range(self.Nnodes()):
            gn=self.nodes['g_n'][n]
            self.nodemap_g_to_local[gn]=n

        # Edges:
        for ji,j in enumerate(g.valid_edge_iter()):
            if ji%5000==0:
                log.info("Edges: %d/%d"%(ji,g.Nedges()))
            self.before_add_edge(g,'add_edge',nodes=g.edges['nodes'][j])

    def before_add_node(self,g,func_name,**k):
        pass # no checks quite yet
    def after_add_node(self,g,func_name,return_value,**k):
        n=return_value
        my_k={}
        # re: _index
        # as long as there aren't Steiner vertices and the like, then
        # it's safe to force node index here to match the parent
        self.nodemap_g_to_local[n]=self.add_node(x=k['x'],g_n=n,_index=n)
    def before_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            my_n=self.nodemap_g_to_local[n]
            self.modify_node(my_n,x=k['x'])
    def before_delete_node(self,g,func_name,n,**k):
        self.delete_node(self.nodemap_g_to_local[n])
        del self.nodemap_g_to_local[n]
    def before_add_edge(self,g,func_name,**k):
        nodes=k['nodes']
        self.add_constraint(nodes[0],nodes[1])
    def before_modify_edge(self,g,func_name,j,**k):
        if 'nodes' not in k:
            return
        old_nodes=g.edges['nodes'][j]
        new_nodes=k['nodes']
        self.remove_constraint( old_nodes[0],old_nodes[1])
        self.add_constraint( new_nodes[0],new_nodes[1] )
    def before_delete_edge(self,g,func_name,j,**k):
        nodes=g.edges['nodes'][j]
        self.remove_constraint(nodes[0],nodes[1])


        
try:
    from CGAL.CGAL_Triangulation_2 import (Constrained_Delaunay_triangulation_2,
                                           Constrained_Delaunay_triangulation_2_Edge)
    
    from CGAL.CGAL_Kernel import (Segment_2,Point_2)
    has_CGAL=True
except ImportError:
    has_CGAL=False

    
class ShadowCGALCDT(object):
    """ A fast implementation which wraps CGALs constrained delaunay 
    triangulation.
    """
    # Borrow exception types from exact_delaunay
    DuplicateNode=exact_delaunay.DuplicateNode
    IntersectingConstraints=exact_delaunay.IntersectingConstraints
    BadConstraint=exact_delaunay.BadConstraint
    ConstraintCollinearNode=exact_delaunay.ConstraintCollinearNode
    
    def __init__(self,g,ignore_existing=False):
        self.g=g
        self.DT = Constrained_Delaunay_triangulation_2()

        self.instrument_grid(g)
        
        # sometimes CGAL creates vertices automatically, which are detected by
        # having info == None
        self.vh_info = defaultdict(lambda:None)

        if not ignore_existing and g.Nnodes():
            self.init_from_grid(g)

    def instrument_grid(self,g):
        vh=np.zeros( g.Nnodes(), dtype=object )
        g.add_node_field('vh',vh)
        
        g.subscribe_before('add_node',self.before_add_node)
        g.subscribe_after('add_node',self.after_add_node)
        g.subscribe_before('modify_node',self.before_modify_node)
        g.subscribe_before('delete_node',self.before_delete_node)
        
        g.subscribe_before('add_edge',self.before_add_edge)
        g.subscribe_before('delete_edge',self.before_delete_edge)
        g.subscribe_before('modify_edge',self.before_modify_edge)
        
    def uninstrument_grid(self,g):
        g.unsubscribe_before('add_node',self.before_add_node)
        g.unsubscribe_after('add_node',self.after_add_node)
        g.unsubscribe_before('modify_node',self.before_modify_node)
        g.unsubscribe_before('delete_node',self.before_delete_node)
        
        g.unsubscribe_before('add_edge',self.before_add_edge)
        g.unsubscribe_before('delete_edge',self.before_delete_edge)
        g.unsubscribe_before('modify_edge',self.before_modify_edge)
        
        g.delete_node_field('vh')

    def init_from_grid(self,g):

        for n in range(g.Nnodes()):
            if n % 50000==0:
                log.info("init_from_grid: %d/%d"%(n,g.Nnodes()))
            # skip over deleted points:
            if ~g.nodes['delete'][n]:
                self.dt_insert(n)
        
        # Edges:
        for ji,j in enumerate(g.valid_edge_iter()):
            if ji%5000==0:
                log.info("Edges: %d/%d"%(ji,g.Nedges()))

            # something like safe insert constraintr
            self.before_add_edge(g,'add_edge',nodes=g.edges['nodes'][j])

    def add_constraint(self,a,b):
        """ 
        adds a constraint to the DT, but does a few simple checks first
        if it's not safe, raise an Exception
        """
        if a < 0 or b < 0 or a==b:
            raise self.BadConstraint("invalid node indices: %d %d"%(a,b))

        pA=self.g.nodes['x'][a]
        pB=self.g.nodes['x'][b]
        vhA=self.g.nodes['vh'][a]
        vhB=self.g.nodes['vh'][b]
        
        if (pA[0]==pB[0]) and (pA[1]==pB[1]):
            msg="invalid constraint: points[%d]=%s and points[%d]=%s are identical"%(a,pA,
                                                                                     b,pB)
            raise self.BadConstraint(msg)

        log.debug("    Inserting constraint (add_constraint): %d %d  %s %s"%(a,b,vhA,vhB))
        log.debug("      node A=%s   node B=%s"%(pA,pB))
        log.debug("      A.point=%s  B.point=%s"%(vhA.point(), vhB.point()))

        # a priori check on whether the constraint would be valid:
        exc=self.line_is_free(a,b)
        if exc is not None:
            #msg="About to insert a constraint %d-%d (%s - %s), but it's not clear!"%(a,b,
            #                                                                         pA,pB)
            #raise self.IntersectingConstraints(msg)
            raise exc
            
        self.DT.insert_constraint(vhA, vhB)

        # double check to make sure that it's actually in there...
        for edge in self.dt_incident_constraints(vhA):
            v1,v2 = edge.vertices()
            if v1==vhB or v2==vhB:
                break
        else: # did not find it
            # we have a conflict - search from a to b
            msg="Just tried to insert a constraint %d-%d (%s - %s), but it's not there!"%(a,b,pA,pB)
            log.warning(msg)
            # Assume that the problem was an intersection
            raise self.IntersectingConstraints(msg)
        
    def remove_constraint(self,a,b):
        vh_a=self.nodes['vh'][a]
        vh_b=self.nodes['vh'][b]
        
        for edge in self.DT.incident_constraints(vh_a): 
            v1,v2 = edge.vertices()
            if vh_b==v1 or vh_b==v2:
                self.DT.remove_constrained_edge(edge.f,edge.v)
                return
        msg="Tried to remove edge %i-%i, but it wasn't in the constrained DT"%(a,b)
        raise self.MissingConstraint(msg)

    def line_is_free(self,a,b,ax=None):
        """
        Check whether inserting a constraint between nodes a
        and b would intersect any existing constraints.

        This nodes a and b must have vertex handles in g.nodes['vh'].
        The points attached to those vertex handles are used here, and
        may differ from the locations in g.nodes['x'].
        
        ax: if specified and an error arises, will try to plot some details.

        returns an exception if the line is free (does not raise, just
        queues it up for you).
        or None if all is well.
        """
        DT=self.DT
        vh_a=self.g.nodes['vh'][a]
        vh_b=self.g.nodes['vh'][b]
        
        # requires points, not vertex handles
        # N.B. that means it does its own locate, which is O(sqrt(n))
        # also, without a face passed in, it will start at the boundary of
        # the convex hull.
        # so get a real face, which keeps the running time constant
        for init_face in DT.incident_faces(vh_a):
            if not DT.is_infinite(init_face):
                break
        else:
            assert False

        lw=DT.line_walk(vh_a.point(),vh_b.point(),init_face) 
        seg=Segment_2(vh_a.point(), vh_b.point())

        faces=[]
        for face in lw:
            # check to make sure we don't go through a vertex
            for i in [0,1,2]:
                v=face.vertex(i)
                if v==vh_a or v==vh_b:
                    continue
                if seg.has_on(v.point()):
                    n_collide=self.vh_info[v]
                    return self.ConstraintCollinearNode("Hit node %s along %s--%s"%(n_collide,
                                                                                    a,b))
            
            if len(faces):
                # figure out the common edge
                for i in [0,1,2]:
                    if face.neighbor(i)==faces[-1]:
                        break
                else:
                    if ax:
                        tri_1=face_to_tri(face)
                        tri_2=face_to_tri(faces[-1])

                        pcoll=PolyCollection([tri_1,tri_2])
                        ax=ax or plt.gca()
                        ax.add_collection(pcoll)

                        pdb.set_trace()
                    assert False

                edge=Constrained_Delaunay_triangulation_2_Edge(face,i)
                if DT.is_constrained(edge):
                    if ax:
                        tri_1=face_to_tri(face)
                        tri_2=face_to_tri(faces[-1])
                        pcoll=PolyCollection([tri_1,tri_2])
                        ax=ax or plt.gca()
                        ax.add_collection(pcoll)
                    return self.IntersectingConstraints("Intersection along %s--%s"%(a,b))

            faces.append(face)
            if face.has_vertex(vh_b):
                break
        return None
        
    class Edge(object):
        def __init__(self,**kwargs):
            self.__dict__.update(kwargs)
        def vertices(self):
            return self.f.vertex( (self.v+1)%3 ),self.f.vertex( (self.v+2)%3 )
    def dt_incident_constraints(self,vh):
        constraints = []
        self.DT.incident_constraints(vh,constraints)
        # maybe I should keep a reference to the Edge object, too?
        # that gets through some early crashes.
        return [self.Edge(f=e.first,v=e.second,keepalive=[e]) for e in constraints]

    
            
    def before_add_node(self,g,func_name,**k):
        pass # no checks quite yet
    def after_add_node(self,g,func_name,return_value,**k):
        n=return_value
        my_k={}
        # re: _index
        
        xy=k['x']
        pnt = Point_2( k['x'][0], k['x'][1] )
        vh = self.g.nodes['vh'][n] = self.DT.insert(pnt)
        self.vh_info[vh] = n

    def before_modify_node(self,g,func_name,n,**k):
        if 'x' not in k:
            return
        
        # This is a tricky and important one!
        vh=self.nodes['vh'][n]

        # so that there aren't conflicts between the current
        # edges and the probe point.

        # See if the location will be okay -
        to_remove = []
        nbrs = [] # neighbor nodes, based only on constrained edges
        for edge in self.dt_incident_constraints(vh):
            v1,v2 = edge.vertices()
            vi1 = self.vh_info[v1]
            vi2 = self.vh_info[v2]

            to_remove.append( (edge, vi1, vi2) )
            if vi1 == i:
                nbrs.append(vi2)
            else:
                nbrs.append(vi1)

        if len(to_remove) != len(g.node_to_edges(n)):
            # Usually means that there is an inconsistency in the CDT
            log.error("WARNING: modify_node len(DT constraints) != len(pnt2edges(i))")
            pdb.set_trace()

        # remove all of the constraints in one go:
        self.DT.remove_incident_constraints(vh)
            
        self.dt_remove(n)

        #-- testing new location:
        # then proceed as if the node were just inserted:
        # Most efficient to do all of the tests, then insert
        # all constraints (instead of potentially creating new constraints,
        # and having to delete them and roll back)

        self.after_add_node(g,'modify_node',n,**k)

        exc=None
        
        for edge,a,b in to_remove:
            exc=self.line_is_free(a,b)
            if exc:
                # HERE - rollback, then raise exception to stop operation.
                # exc=self.IntersectingConstraints("New location of %s to %s"%(a,b))
                self.dt_remove(n)
                # put it back where it was:
                self.after_add_node(g,'modify_node',n,{'x':self.nodes['x'][n]})
                break

        # By here, we've either verified that all constraints will be okay,
        # or that there is a violation (recorded in exc), and put the vertex back
        # where it was.
        
        for edge,a,b in to_remove:
            # For the moment, this does extra work.  But
            # TODO: once this is more proven, this step could skip the tests
            # and go directly to adding the constraint
            self.add_constraint(a,b)

        if exc:
            log.warning("New location violates some constraints")
            log.exception(exc)
            raise exc

    def dt_remove(self,n):
        vh=self.g.nodes['vh'][n] 
        del self.vh_info[vh]
        self.DT.remove(vh)
        # None would be nice, but it's going to default to 0.
        self.g.nodes['vh'][n] = 0
        
    def before_delete_node(self,g,func_name,n,**k):
        self.dt_remove(n)

    def before_add_edge(self,g,func_name,**k):
        nodes=k['nodes']
        self.add_constraint(nodes[0],nodes[1])

    def before_modify_edge(self,g,func_name,j,**k):
        if 'nodes' not in k:
            return
        old_nodes=g.edges['nodes'][j]
        new_nodes=k['nodes']
        self.remove_constraint( old_nodes[0],old_nodes[1])
        self.add_constraint( new_nodes[0],new_nodes[1] )
        
    def before_delete_edge(self,g,func_name,j,**k):
        nodes=g.edges['nodes'][j]
        self.remove_constraint(nodes[0],nodes[1])
        
