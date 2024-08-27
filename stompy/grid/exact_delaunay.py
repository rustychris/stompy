# A pure-python, exact delaunay triangulation.
# uses robust_predicates for in-circle tests, follows
# the algorithm of CGAL to the extent possible.
import logging
import pdb
logger = logging.getLogger()

import six
import numpy as np
import matplotlib.pyplot as plt

# do these work in py2?
from ..spatial import robust_predicates
from . import unstructured_grid
from ..utils import (circular_pairs, dist, point_segment_distance, set_keywords,
                     segment_segment_intersection)

if six.PY3:
    def cmp(a,b):
        return bool(a>b)-bool(a<b)


try:
    from scipy import spatial
except ImportError:
    spatial=None

class DuplicateNode(Exception):
    pass

class BadConstraint(Exception):
    def __init__(self,*a,**k):
        super(BadConstraint,self).__init__(*a)
        set_keywords(self,k)

class IntersectingConstraints(BadConstraint):
    edge=None

class DuplicateConstraint(BadConstraint):
    nodes=None

class ConstraintCollinearNode(IntersectingConstraints):
    """
    Special case of intersections, when a constraint attempts to 
    run *through* an existing node
    """
    node=None

def ordered(x1,x2,x3):
    """
    given collinear points, return true if they are in order
    along that line
    """
    if x1[0]!=x2[0]:
        i=0
    else:
        i=1
    return (x1[i]<x2[i]) == (x2[i]<x3[i])


def rel_ordered(x1,x2,x3,x4):
    """
    given 4 collinear points, return true if the direction
    from x1->x2 is the same as x3=>x4
    requires x1!=x2, and x3!=x4
    """
    if x1[0]!=x2[0]:
        i=0 # choose a coordinate which is varying
    else:
        i=1
    assert x1[i]!=x2[i]
    assert x3[i]!=x4[i]
    return (x1[i]<x2[i]) == (x3[i]<x4[i])


class Triangulation(unstructured_grid.UnstructuredGrid):
    """ 
    Mimics the Triangulation_2 class of CGAL.
    note that we make some additional assumptions on invariants -
     nodes, cells and edges are ordered in a consistent way:
    
    """ 
    INF_NODE=-666
    INF_CELL=unstructured_grid.UnstructuredGrid.UNMESHED
    max_sides=3

    # local exception types
    DuplicateNode=DuplicateNode
    IntersectingConstraints=IntersectingConstraints
    BadConstraint=BadConstraint
    ConstraintCollinearNode=ConstraintCollinearNode

    post_check=False # enables [expensive] checks after operations
    
    edge_dtype=(unstructured_grid.UnstructuredGrid.edge_dtype +
                [ ('constrained',np.bool_) ] )

    def add_node(self,**kwargs):
        # will eventually need some caching or indexing to make
        # the locate faster.  locate() happens first so that 
        # the mesh complies with invariants and doesn't have a dangling
        # node
        loc=self.locate(kwargs['x'])

        n=super(Triangulation,self).add_node(**kwargs)

        self.tri_insert(n,loc)
        return n

    def modify_node(self,n,_brute_force=False,**kwargs):
        """
        _brute_force: if True, move node by delete/add, rather than trying
          a short cut.
        """
        if 'x' not in kwargs:
            return super(Triangulation,self).modify_node(n,**kwargs)
        old_rec=self.nodes[n]
        # Brute force, removing and re-adding, is no good as the
        # constraints are lost.
        # A slightly more refined, but still brutish, approach, is to save
        # the constraints, delete, add, add constraints.
        # be sped up

        # handle a common case where the node is only moved a small
        # distance, such that we only have to do a small amount of
        # work to fix up the triangulation
        # if the new location is inside a cell adjacent to n, then
        # we can [probably?] move the node
        if self.dim()<2:
            # the short cuts are only written for the 2D case.
            _brute_force=True
            
        if not _brute_force:
            # check whether new node location is on the "right" side
            # of all existing "opposite" edges (the edge of each cell
            # which doesn't contain n.
            shortcut=True
            if shortcut:
                my_cells=self.node_to_cells(n)
                for c in my_cells:
                    c_nodes=self.cells['nodes'][c]
                    c_xy=self.nodes['x'][c_nodes]
                    pnts=[]
                    for i,c_node in enumerate(c_nodes):
                        if c_node==n:
                            pnts.append(kwargs['x'])
                        else:
                            pnts.append(c_xy[i])
                    if robust_predicates.orientation(*pnts) <=0:
                        shortcut=False
            if shortcut:
                # also check for this node being on the convex hull
                # find the pair of edges, if they exist, which have
                # n, and have the infinite cell to the left.

                he_rev=he_fwd=None
                for j in self.node_to_edges(n):
                    if self.edges['cells'][j,1]==self.INF_CELL:
                        he=self.halfedge(j,1)
                    elif self.edges['cells'][j,0]==self.INF_CELL:
                        he=self.halfedge(j,0)
                    else:
                        continue

                    if he.node_fwd()==n:
                        he_rev=he
                    elif he.node_rev()==n:
                        he_fwd=he
                    else:
                        assert False
                # can't have just one.
                assert (he_rev is None) == (he_fwd is None)
                if he_rev is not None:
                    # need to check that the movement of this node does
                    # not invalidate the orientation with respect to
                    # neighboring edges of the convex hull.
                    # get the five consecutive points, where c is the
                    # node being moved.  make sure that a-b-c and c-d-e
                    # are properly oriented
                    cons_idxs=[he_rev.rev().node_rev(),
                               he_rev.node_rev(),
                               n,
                               he_fwd.node_fwd(),
                               he_fwd.fwd().node_fwd()]
                    abcde=self.nodes['x'][cons_idxs]
                    abcde[2]=kwargs['x']

                    if robust_predicates.orientation(*abcde[:3])>0:
                        shortcut=False
                    elif robust_predicates.orientation(*abcde[2:])>0:
                        shortcut=False
                    elif robust_predicates.orientation(*abcde[1:4])>0:
                        shortcut=False

            if shortcut:
                # short cut should work:
                retval=super(Triangulation,self).modify_node(n,**kwargs)
                self.restore_delaunay(n)
                # when refining the above tests, uncomment this to increase
                # the amount of validation
                # if self.check_convex_hull():
                #     pdb.set_trace()
                return retval
                    
        # but adding the constraints back can fail, in which case we should
        # roll back our state, and fire an exception.

        constraints_to_replace=[]
        for j in self.node_to_edges(n):
            if self.edges['constrained'][j]:
                constraints_to_replace.append( self.edges['nodes'][j].copy() )

        old_x=self.nodes['x'][n].copy() # in case of rollback
        
        self.delete_node(n)
        
        for fld in old_rec.dtype.names:
            if fld not in ['x','deleted'] and fld not in kwargs:
                kwargs[fld]=old_rec[fld]
        new_n=self.add_node(_index=n,**kwargs)

        try:
            for n1,n2 in constraints_to_replace:
                self.add_constraint(n1,n2) # This can fail!
        except self.IntersectingConstraints as exc:
            self.log.warning("modify_node: intersecting constraints - rolling back")
            self.delete_node(n)
            kwargs['x']=old_x # move it back to where it started
            new_n=self.add_node(_index=n,**kwargs)
            for n1,n2 in constraints_to_replace:
                self.add_constraint(n1,n2) # This should not fail
            # but signal to the caller that the modify failed
            raise

        assert new_n==n
        
    def add_edge(self,**kw):
        """ add-on: cells default to INF_CELL, not -1.
        """
        j=super(Triangulation,self).add_edge(**kw)
        if 'cells' not in kw:
            self.edges[j]['cells'][:]=self.INF_CELL
        return j

    def choose_start_cell(self,t=None):
        """ choose a starting cell for trying to locate where a new vertex
        should go.  May return INF_CELL if there are no valid cells.
        t: can specify a target point which may be used with a spatial index
        to speed up the query.
        """
        c=0
        try:
            while self.cells['deleted'][c]: 
                c+=1
            return c
        except IndexError:
            return self.INF_CELL
            
    IN_VERTEX=0
    IN_EDGE=2
    IN_FACE=3
    OUTSIDE_CONVEX_HULL=4
    OUTSIDE_AFFINE_HULL=5

    def dim(self):
        if len(self.cells) and not np.all(self.cells['deleted']):
            return 2
        elif len(self.edges) and not np.all(self.edges['deleted']):
            return 1
        elif len(self.nodes) and not np.all(self.nodes['deleted']):
            return 0
        else:
            return -1
    def angle_sort_adjacent_nodes(self,n,ref_nbr=None,topo=True):
        if topo:
            return self.topo_sort_adjacent_nodes(n,ref_nbr)
        else:
            return super(Triangulation,self).angle_sort_adjacent_ndoes(n,ref_nbr=ref_nbr)
        
    def topo_sort_adjacent_nodes(self,n,ref_nbr=None):
        """ like angle_sort_adjacent_nodes, but relying on topology, not geometry.
        """
        nbrs=list(self.node_to_nodes(n))

        if len(nbrs)<3:
            snbrs=nbrs
        else:
            he_nbrs = [ self.nodes_to_halfedge(n,nbr)
                        for nbr in nbrs ]

            map_next={}

            for he in he_nbrs:
                # this doesn't use angle_sort
                c=he.cell_opp()
                map_next[c] = (he.node_fwd(),he.cell())

            trav0=trav=c

            snbrs=[]
            while 1:
                #if len(snbrs)>20: # DBG
                #    pdb.set_trace()
                node,cell = map_next[trav]
                snbrs.append(node)
                trav=cell
                if trav==trav0:
                    break
            
        if ref_nbr is not None: 
            i=list(snbrs).index(ref_nbr)
            snbrs=np.roll(snbrs,-i)

        return snbrs
        
    def locate(self,t,c=None):
        """ t: [x,y] point to locate
        c: starting cell, if known

        return loc=[face,loc_type,loc_index]
        face: INF_CELL if t is not on or inside a finite cell
        loc_type: 
          OUTSIDE_AFFINE_HULL: adding this vertex will increase the dimension of the triangulation.
             empty triangulation: dim=-1
             single vertex: dim=0
             collinear edges: dim=1
             faces: dim=2
             loc_index set to current dimensionality
          OUTSIDE_CONVEX_HULL: dimensionality may still be 1 or 2.
             if the dimension is 1, then loc_index gives the nearest node
             if the dimension is 2, then loc_index gives an adjacent half-edge
          IN_VERTEX: t coincides with existing vertex, 
             if face is finite, then it's a cell containing the vertex, and loc_index
               is the index of that vertex in the cell.
             if face is INF_CELL, implies dimension<2, and loc_index gives existing node
          IN_EDGE: t is collinear with existing edge.  
             if face is finite, it is a cell containing the edge.
             loc_index is the index of the edge itself.
             face may be INF_CELL, which implies dimension<2
          IN_FACE: t is in the interior of a face. face is the containing cell. loc_index
             is not used.
        """
        c=c or self.choose_start_cell(t)

        prev=None # previous face
        # To identify the right orientation of the half-edge, remember
        # the ordering of the nodes -- this is CCW ordering from the 
        # perspective of prev
        last_nodes=None 
        last_edge=None # the edge between c and prev

        # Checks for affine hull -
        # 3rd element gives the current dimensionality of the affine hull
        if self.Nnodes_valid()==0:
            return (self.INF_CELL,self.OUTSIDE_AFFINE_HULL,-1)
        elif self.Nedges_valid()==0:
            return (self.INF_CELL,self.OUTSIDE_AFFINE_HULL,0)
        elif self.Ncells_valid()==0:
            return self.locate_1d(t,c)


        while True:
            if c==self.INF_CELL:
                #       // c must contain t in its interior
                #       lt = OUTSIDE_CONVEX_HULL;
                #       li = c->index(infinite_vertex());
                # Changed to give adjacent edge, rather than 
                # confusing loc_index=4
                #  loc=(self.INF_CELL,self.OUTSIDE_CONVEX_HULL,last_edge)
                # changed again, to give a half-edge
                # flip the order because they were in the order with respect
                # to the prev face, but now we jumped over last_edge
                he=self.nodes_to_halfedge( last_nodes[1],last_nodes[0] )
                loc=(self.INF_CELL,self.OUTSIDE_CONVEX_HULL,he)
                return loc

            p0=self.nodes['x'][self.cells['nodes'][c,0]]
            p1=self.nodes['x'][self.cells['nodes'][c,1]]
            p2=self.nodes['x'][self.cells['nodes'][c,2]]

            prev = c

            # Orientation o0, o1, o2;

            # nodes are stored in CCW order for the cell.
            # 1st edge connects first two nodes 
            # neighboring cells follow the edges

            o0 = robust_predicates.orientation(p0,p1,t)
            if o0 == -1: # CW 
                last_edge=self.cell_to_edges(c)[0]
                last_nodes=self.cells['nodes'][c,[0,1]]
                c=self.cell_to_cells(c)[0] 
                continue

            o1 = robust_predicates.orientation(p1,p2,t)
            if o1 == -1:
                last_edge=self.cell_to_edges(c)[1]
                last_nodes=self.cells['nodes'][c,[1,2]]
                c=self.cell_to_cells(c)[1] 
                continue

            o2 = robust_predicates.orientation(p2,p0,t)
            if o2 == -1:
                last_edge=self.cell_to_edges(c)[2] 
                last_nodes=self.cells['nodes'][c,[2,0]]
                c=self.cell_to_cells(c)[2] 
                continue

            # must be in or on a face --
            break
        # For simplicity, I'm skipping some optimizations which avoid re-checking
        # the previous edge.  see Triangulation_2.h:2616

        # now t is in c or on its boundary
        o_sum=(o0==0)+(o1==0)+(o2==0)

        if o_sum==0:
            loc=(c,self.IN_FACE,4)
        elif o_sum==1:
            if o0==0:
                j=0
            elif o1==0:
                j=1
            else:
                j=2
            # better to consistently return the edge index here, not
            # just its index in the cell
            loc=(c,self.IN_EDGE,self.cells['edges'][c,j])
        elif o_sum==2:
            if o0!=0:
                loc=(c,self.IN_VERTEX,2)
            elif o1!=0:
                loc=(c,self.IN_VERTEX,0)
            else:
                loc=(c,self.IN_VERTEX,1)
        else:
            assert False
        return loc

    def locate_1d(self,t,c):
        # There are some edges, and t may fall within an edge, off the end,
        # or off to the side.
        j=six.next(self.valid_edge_iter())
        
        p0=self.nodes['x'][ self.edges['nodes'][j,0] ]
        p1=self.nodes['x'][ self.edges['nodes'][j,1] ]
        
        o=robust_predicates.orientation(p0,p1,t)
        if o!=0:
            return (self.INF_CELL,self.OUTSIDE_AFFINE_HULL,1)

        # t is collinear - need to find out whether it's in an edge
        # or not

        # choose a coordinate which varies along the line
        if p0[0]!=p1[0]:
            coord=0
        else:
            coord=1

        if (t[coord]<p0[coord]) != (t[coord]<p1[coord]):
            return (self.INF_CELL,self.IN_EDGE,j)

        # do we need to go towards increasing or decreasing coord?
        if (t[coord]<p0[coord]) and (t[coord]<p1[coord]):
            direc=-1
        else:
            direc=1
        
        while True:
            # j indexes the edge we just tested. 
            # p0 and p1 are the endpoints of the edge
            # 1. do we want a neighbor of n0 or n1?
            if direc*cmp(p0[coord],p1[coord]) < 0: # want to go towards p1
                n_adj=self.edges['nodes'][j,1]
            else:
                n_adj=self.edges['nodes'][j,0]
            for jnext in self.node_to_edges(n_adj):
                if jnext!=j:
                    j=jnext
                    break
            else:
                # walked off the end of the line -
                # n_adj is the nearest to us
                return (self.INF_CELL,self.OUTSIDE_CONVEX_HULL,n_adj)

            p0=self.nodes['x'][ self.edges['nodes'][j,0] ]
            p1=self.nodes['x'][ self.edges['nodes'][j,1] ]

            if (t[coord]<p0[coord]) != (t[coord]<p1[coord]):
                return (self.INF_CELL,self.IN_EDGE,j)

    def tri_insert(self,n,loc):
        self.log.info("%s: tri_insert"%self)
        
        # n: index for newly inserted node.
        # note that loc must already be computed -

        # types of inserts:
        #   on an edge, inside a face, outside the convex hull
        #   outside affine hull

        loc_c,loc_type,loc_idx = loc
        if loc_type==self.IN_FACE:
            self.tri_insert_in_face(n,loc)
        elif loc_type==self.IN_EDGE:
            self.tri_insert_in_edge(n,loc)
        elif loc_type==self.IN_VERTEX:
            raise DuplicateNode()
        elif loc_type==self.OUTSIDE_CONVEX_HULL:
            self.tri_insert_outside_convex_hull(n,loc)
        elif loc_type==self.OUTSIDE_AFFINE_HULL:
            self.tri_insert_outside_affine_hull(n,loc)

        # for some of those actions, this could be skipped
        self.restore_delaunay(n)

    def tri_insert_in_face(self,n,loc):
        loc_f,loc_type,_ = loc
        a,b,c=self.cells['nodes'][loc_f]
        self.delete_cell(loc_f)
        self.add_edge(nodes=[n,a])
        self.add_edge(nodes=[n,b])
        self.add_edge(nodes=[n,c])
        self.add_cell(nodes=[n,a,b])
        self.add_cell(nodes=[n,b,c])
        self.add_cell(nodes=[n,c,a])
        
    def tri_insert_in_edge(self,n,loc):
        """ Takes care of splitting the edge and any adjacent cells
        """
        loc_f,loc_type,loc_edge = loc 

        self.log.debug("Loc puts new vertex in edge %s"%loc_edge)
        cells_to_split=[]
        for c in self.edge_to_cells(loc_edge):
            if c<0: continue
            cells_to_split.append(  self.cells[c].copy() )
            self.log.debug("Deleting cell on insert %d"%c)
            self.delete_cell(c)

        # Modify the edge:
        a,c=self.edges['nodes'][loc_edge]
        b=n
        self.delete_edge(loc_edge)
        
        self.add_edge(nodes=[a,b])
        self.add_edge(nodes=[b,c])
        
        for cell_data in cells_to_split:
            common=[n for n in cell_data['nodes']
                    if n!=a and n!=c][0]
            jnew=self.add_edge(nodes=[b,common])
            
            for replace in [a,c]:
                nodes=list(cell_data['nodes'])
                idx=nodes.index(replace)
                nodes[idx]=b
                self.add_cell(nodes=nodes)
        
    def tri_insert_outside_convex_hull(self,n,loc):
        dim=self.dim()
        if dim==2:
            self.tri_insert_outside_convex_hull_2d(n,loc)
        elif dim==1:
            self.tri_insert_outside_convex_hull_1d(n,loc)
        else:
            assert False
    def tri_insert_outside_convex_hull_1d(self,n,loc):
        self.log.debug("tri_insert_outside_convex_hull_1d")
        n_adj=loc[2]
        self.add_edge(nodes=[n,n_adj])
    def tri_insert_outside_convex_hull_2d(self,n,loc):
        # HERE: 
        #   the CGAL code is a little funky because of the use of 
        #   infinite vertices and the like.
        #   the plan here:
        #   a. change 'locate' to return halfedges instead of just an
        #      edge.  otherwise we'd have to redo the orientation check here.
        #   b. traverse the half-edge forwards and backwards, accumulating
        #      lists of adjacent edges which also satisfy the CCW rule.
        #   c. create triangles with n and the given half-edge, as well as the
        #      accumulated adjacent edges
        #   the result then is that the convex hull is built out.
        # Triangulation_2.h:1132
        assert loc[0]==self.INF_CELL # sanity.
        he0=loc[2] # adjacent half-edge

        def check_halfedge(he):
            nodes=[he.node_rev(),he.node_fwd(),n]
            pnts=self.nodes['x'][nodes]
            ccw=robust_predicates.orientation(pnts[0],pnts[1],pnts[2])
            return ccw>0
        assert check_halfedge(he0)

        addl_fwd=[]
        he=he0.fwd()
        while check_halfedge(he):
            addl_fwd.append(he)
            he=he.fwd()
        addl_rev=[]
        he=he0.rev()
        while check_halfedge(he):
            addl_rev.append(he)
            he=he.rev()

        self.add_edge( nodes=[he0.node_rev(),n] )
        self.add_edge( nodes=[he0.node_fwd(),n] )
        self.add_cell( nodes=[he0.node_rev(),he0.node_fwd(),n] )
        for he in addl_fwd:
            self.add_edge( nodes=[he.node_fwd(),n] )
            # the second node *had* been ne0.node_fwd(), but that
            # was probably a typo.
            self.add_cell( nodes=[he.node_rev(),he.node_fwd(),n] )
        for he in addl_rev:
            self.add_edge( nodes=[he.node_rev(),n] )
            # same here.
            self.add_cell( nodes=[he.node_rev(),he.node_fwd(),n] )

        # 1. Check orientation.  Since we get an unoriented edge j_adj,
        #    all we can do is assert that the points are not collinear.
        # 2. loops through faces incident to infinite vertex (?)
        #    gathering a list of external edges which make a CCW triangle
        #    with the vertex to insert.  stop on the first edge which fails this.
        #    This is done first traversing CCW, then again traversing CW
        # 3. Make the new face with the given edge..
        # 

    def tri_insert_outside_affine_hull(self,n,loc):
        self.log.debug("Insert outside affine hull")

        loc_face,loc_type,curr_dim = loc

        if curr_dim==-1:
            self.log.debug("  no nodes, no work")
        elif curr_dim==0:
            self.log.debug("  simply add edge")
            for nbr in self.valid_node_iter():
                if nbr != n:
                    self.add_edge(nodes=[n,nbr])
        elif curr_dim==1:
            self.log.debug("  add edges and cells")
            # the strategy in Triangulation_2.h makes some confusing
            # use of the infinite face - take a less elegant, more explicit
            # approach here
            orig_edges=list(self.valid_edge_iter())
            for nbr in self.valid_node_iter():
                if nbr != n:
                    self.add_edge(nodes=[n,nbr])
            for j in orig_edges:
                n1,n2=self.edges['nodes'][j]
                self.add_cell( nodes=[n,n1,n2] )
        else:
            assert False
    def add_cell(self,_force_invariants=True,**kwargs):
        if _force_invariants:
            nodes=kwargs['nodes']

            # Make sure that topological invariants are maintained:
            # nodes are ordered ccw.
            # edges are populated
            # used to assume/force the edges to be sequenced opposite nodes.
            # but that is a triangulation-specific assumption, while we're using
            # a general unstructured_grid base class.  The base class makes
            # an incompatible assumption, that the first edge connects the first
            # two nodes.  
            pnts=self.nodes['x'][nodes]

            ccw=robust_predicates.orientation(pnts[0],pnts[1],pnts[2]) 
            assert ccw!=0
            if ccw<0:
                nodes=nodes[::-1]
                kwargs['nodes']=nodes
            
            j0=self.nodes_to_edge(nodes[0],nodes[1])
            j1=self.nodes_to_edge(nodes[1],nodes[2])
            j2=self.nodes_to_edge(nodes[2],nodes[0])
            kwargs['edges']=[j0,j1,j2]
        c=super(Triangulation,self).add_cell(**kwargs)

        # update the link from edges back to cells
        for ji,j in enumerate(self.cells['edges'][c]):
            # used to attempt to enforce this:
            #   ji-th edge is the (ji+1)%3,(ji+2)%3 nodes of the cell
            # but that's not compatible with checks in unstructured_grid
            # but need to know if the edge is in that order or the
            # opposite
            if self.edges['nodes'][j,0] == self.cells['nodes'][c,ji]:
                self.edges['cells'][j,0] = c
            else:
                self.edges['cells'][j,1] = c
        return c
                
    def flip_edge(self,j):
        """ 
        rotate the given edge CCW.  requires that triangular cells
        exist on both sides of the edge
        (that's not a hard and fast requirement, just makes it easier
        to implemenet. There *does* have to be a potential cell on either
        side).
        """
        c_left,c_right=self.edges['cells'][j,:]
        self.log.info("Flipping edge %d, with cells %d, %d   nodes %d,%d"%(j,c_left,c_right,
                                                                           self.edges['nodes'][j,0],
                                                                           self.edges['nodes'][j,1]) )
        assert c_left>=0 # could be relaxed, at the cost of some complexity here
        assert c_right>=0
        # could work harder to preserve extra info:
        #c_left_data = self.cells[c_left].copy()
        #c_right_data = self.cells[c_right].copy()


        # This is dangerous! - deleting the cells means that topo_sort is no good,
        # and that breaks half-edge ops.
        # moving to happen a bit later -
        # self.delete_cell(c_left)
        # self.delete_cell(c_right)
        he_left=unstructured_grid.HalfEdge(self,j,0)
        he_right=unstructured_grid.HalfEdge(self,j,1)

        na,nc = self.edges['nodes'][j]
        nd=he_left.fwd().node_fwd()
        nb=he_right.fwd().node_fwd()

        # DBG
        if 0:
            for n,label in zip( [na,nb,nc,nd],
                                "abcd" ):
                plt.text( self.nodes['x'][n,0],
                          self.nodes['x'][n,1],
                          label)
        # keep the time where the cells are deleted to a minimum
        self.delete_cell(c_left)
        self.delete_cell(c_right)

        self.log.info("%s: calling self.modify_edge which is %s"%(self,self.modify_edge))
        self.modify_edge(j,nodes=[nb,nd])
        new_left =self.add_cell(nodes=[na,nb,nd])
        new_right=self.add_cell(nodes=[nc,nd,nb])
        return new_left,new_right

    def delete_node(self,n):
        """ Triangulation version implies cascade, but also 
        patches up the triangulation
        """
        assert n>=0

        N=self.Nnodes_valid()
        if N==1:
            super(Triangulation,self).delete_node(n)
        elif N==2:
            j=self.node_to_edges(n)[0]
            self.delete_edge(j)
            super(Triangulation,self).delete_node(n)
        elif self.dim()==1:
            self.delete_node_1d(n)
        else:
            self.delete_node_2d(n)

    def delete_node_1d(self,n):
        # Triangulation_2.h hands this off to the triangulation data structure
        # That code looks like:
        assert self.dim() == 1
        assert self.Nnodes_valid() > 2


        # Two cases - either n is at the end of a line of nodes,
        # or it's between two nodes.
        nbrs=self.node_to_nodes(n)

        if len(nbrs)==1: # easy, we're at the end
            j=self.nodes_to_edge(n,nbrs[0])
            self.delete_edge(j)
            super(Triangulation,self).delete_node(n)
        else:
            assert len(nbrs)==2
            j1=self.nodes_to_edge(n,nbrs[0])
            j2=self.nodes_to_edge(n,nbrs[1])
            self.delete_edge(j1)
            self.delete_edge(j2)
            super(Triangulation,self).delete_node(n)
            self.add_edge( nodes=nbrs )

    def test_delete_node_dim_down(self,n):
        # see Triangulation_2.h : test_dim_down
        # test the dimensionality of the resulting triangulation
        # upon removing of vertex v
        # it goes down to 1 iff
        #  1) any finite face is incident to v
        #  2) all vertices are collinear
        assert self.dim() == 2
        for c in self.valid_cell_iter():
            if n not in self.cell_to_nodes(c):
                # There is a triangle not involving n
                # deleting n would retain a 2D triangulation
                return False
        pnts=[self.nodes['x'][i]
              for i in self.valid_node_iter()
              if i!=n]
        a,b = pnts[:2]
        for c in pnts[2:]:
            if robust_predicates.orientation(a,b,c) != 0:
                return False
        return True
        
    def delete_node_2d(self,n):
        if self.test_delete_node_dim_down(n):
            # deleting n yields a 1D triangulation - no faces
            for c in self.valid_cell_iter():
                self.delete_cell(c)
            # copy
            for j in list(self.node_to_edges(n)): 
                self.delete_edge(j)
            super(Triangulation,self).delete_node(n)
            return 

        # first, make a hole around n
        deletee=n

        # new way
        nbrs=self.angle_sort_adjacent_nodes(deletee)
        edges_to_delete=[]

        hole_nodes=[]
        for nbrA,nbrB in circular_pairs(nbrs):
            hole_nodes.append(nbrA)
            he=self.nodes_to_halfedge(nbrA,nbrB)
            if (he is None) or (he.cell()<0) or (n not in self.cell_to_nodes(he.cell())):
                hole_nodes.append('inf')
            edges_to_delete.append( self.nodes_to_edge( [deletee,nbrA] ) )

        for j in edges_to_delete:
            self.delete_edge_cascade(j)
        super(Triangulation,self).delete_node(deletee)

        # Use the boundary completion approach described in Devillers 2011
        # it's not terribly slow, and can be done with the existing
        # helpers.
        self.fill_hole(hole_nodes)
    def fill_hole(self,hole_nodes):
        
        # track potentially multiple holes
        # a few place use list-specific semantics - not ndarray
        hole_nodes=list(hole_nodes)
        holes_nodes=[ hole_nodes ]

        while len(holes_nodes):
            hole_nodes=holes_nodes.pop()

            while 'inf' in hole_nodes[:2]:
                hole_nodes = hole_nodes[1:] + hole_nodes[:1]
                
            a,b=hole_nodes[:2]

            self.log.debug("Considering edge %d-%d"%(a,b) )

            # inf nodes:
            # can't test any geometry.  seems like we can only have boundary
            # faces if the hole included an inf node.
            # so drop it from candidates here, but remember that we saw it

            # first, sweep through the candidates to test CCW
            has_inf=False
            c_cand1=hole_nodes[2:]
            c_cand2=[]
            for c in c_cand1:
                if c=='inf':
                    has_inf=True
                elif robust_predicates.orientation( self.nodes['x'][a],
                                                    self.nodes['x'][b],
                                                    self.nodes['x'][c] ) > 0:
                    c_cand2.append(c)

            self.log.debug("After CCW tests, %s are left"%c_cand2)

            while len(c_cand2)>1:
                c=c_cand2[0]
                for d in c_cand2[1:]:
                    tst=robust_predicates.incircle( self.nodes['x'][a],
                                                    self.nodes['x'][b],
                                                    self.nodes['x'][c],
                                                    self.nodes['x'][d] )
                    if tst>0:
                        self.log.debug("%d was inside %d-%d-%d"%(d,a,b,c))
                        c_cand2.pop(0)
                        break
                else:
                    # c passed all the tests
                    c_cand2=[c]
                    break
            # if the hole nodes are already all convex, then they already
            # form the new convex hull - n was on the hull and simply goes
            # away
            if has_inf and not c_cand2:
                c_cand2=['inf']
                c='inf' # was this missing??
            else:
                c=c_cand2[0]

            self.log.debug("Decided on %s-%s-%s"%(a,b,c))

            # n.b. add_cell_and_edges is probably what is responsible
            # for the painless dealing with collinear boundaries.
            if c!='inf':
                self.add_cell_and_edges( nodes=[a,b,c] )

            # what hole to put back on the queue?
            if len(hole_nodes)==3:
                # finished this hole.
                self.log.debug("Hole is finished")
                continue
            elif c==hole_nodes[2]:
                self.log.debug("Hole is trimmed from front")
                hole_nodes[:3] = [a,c]
                holes_nodes.append( hole_nodes )
            elif c==hole_nodes[-1]:
                self.log.debug("Hole is trimmed from back")
                hole_nodes=hole_nodes[1:] # drop a
                self.log.debug("  New hole is %s"%hole_nodes)
                holes_nodes.append( hole_nodes )
            else:
                self.log.debug("Created two new holes")
                idx=hole_nodes.index(c)

                h1=hole_nodes[1:idx+1]
                h2=hole_nodes[idx:] + hole_nodes[:1]
                self.log.debug("  New hole: %s"%h1)
                self.log.debug("  New hole: %s"%h2)

                holes_nodes.append( h1 )
                holes_nodes.append( h2 )


    # Make a check for the delaunay criterion:
    def check_global_delaunay(self):
        bad_checks=[] # [ (cell,node),...]
        for c in self.valid_cell_iter():
            nodes=self.cells['nodes'][c]
            pnts=self.nodes['x'][nodes]

            # brute force - check them all.
            for n in self.valid_node_iter():
                if n in nodes:
                    continue
                t=self.nodes['x'][n]
                check=robust_predicates.incircle(pnts[0],pnts[1],pnts[2],t)
                if check>0:
                    # how do we check for constraints here?
                    # maybe more edge-centric?
                    # tests of a cell on one side of an edge against a node on the
                    # other is reflexive.
                    # 
                    
                    # could go through the edges of c, 
                    msg="Node %d is inside the circumcircle of cell %d (%d,%d,%d)"%(n,c,
                                                                                    nodes[0],nodes[1],nodes[2])
                    self.log.error(msg)
                    bad_checks.append( (c,n) )
        return bad_checks
    
    def check_local_delaunay(self):
        """ Check both sides of each edge - can deal with constrained edges.
        """
        bad_checks=[] # [ (cell,node),...]
        for j in self.valid_edge_iter():
            if self.edges['constrained'][j]:
                continue
            c1,c2 = self.edge_to_cells(j)
            if c1<0 or c2<0:
                continue
            # always check the smaller index -
            # might help with caching later on.
            c=min(c1,c2)
            c_opp=max(c1,c2)
            
            nodes=self.cells['nodes'][c]
            pnts=self.nodes['x'][nodes]

            # brute force - check them all.
            for n in self.cell_to_nodes(c_opp):
                if n in nodes:
                    continue
                t=self.nodes['x'][n]
                check=robust_predicates.incircle(pnts[0],pnts[1],pnts[2],t)
                if check>0:
                    msg="Node %d is inside the circumcircle of cell %d (%d,%d,%d)"%(n,c,
                                                                                    nodes[0],nodes[1],nodes[2])
                    self.log.error(msg)
                    bad_checks.append( (c,n) )
                    raise Exception('fail')
        return bad_checks

    def check_orientations(self):
        """
        Checks all cells for proper CCW orientation,
        return a list of cell indexes of failures.
        """
        bad_cells=[]
        for c in self.valid_cell_iter():
            node_xy=self.nodes['x'][self.cells['nodes'][c]]
            if robust_predicates.orientation(*node_xy) <= 0:
                bad_cells.append(c)
        return bad_cells
    def check_convex_hull(self):
        # find an edge on the convex hull, walk the hull and check
        # all consecutive orientations
        e2c=self.edge_to_cells()
        for j in self.valid_edge_iter():
            if e2c[j,0]==self.INF_CELL:
                he=self.halfedge(j,0)
                break
            elif e2c[j,1]==self.INF_CELL:
                he=self.halfedge(j,1)
                break
        else:
            assert False

        he0=he

        bad_hull=[]
        while 1:
            a=he.node_rev()
            b=he.node_fwd()
            he=he.fwd()
            c=he.node_fwd()
            if robust_predicates.orientation(*self.nodes['x'][[a,b,c]])>0:
                bad_hull.append( [a,b,c])
            if he==he0:
                break
        return bad_hull    
    
    def restore_delaunay(self,n):
        """ n: node that was just inserted and may have adjacent cells
        which do not meet the Delaunay criterion
        """
        self.log.info("%s: call to exact_delaunay.restore_delaunay()"%self)
        # n is node for Vertex_handle v
        if self.dim() <= 1:
            return

        # a vertex is shared by faces, but "stores" only one face.
        # Face_handle f=v->face();

        # This code iterates over the faces adjacent to v
        # in ccw order.

        # Face_handle next;
        # int i;
        # Face_handle start(f);
        # do {
        #   i = f->index(v);
        #   next = f->neighbor(ccw(i));  // turn ccw around v
        #   propagating_flip(f,i);
        #   f=next;
        # } while(next != start);

        # Shaky on the details, but for starters, try marking the CCW sweep
        # based on neighbor nodes.
        nbr_nodes=self.angle_sort_adjacent_nodes(n)
        
        N=len(nbr_nodes)
        for i in range(N):
            trav=nbr_nodes[i]
            trav_next=nbr_nodes[(i+1)%N]
            c=self.nodes_to_cell( [n,trav,trav_next],fail_hard=False)
            if c is not None:    
                for i in [0,1,2]:
                    if self.cells['nodes'][c,i]==n:
                        break
                else:
                    assert False

            if c is not None:
                self.propagating_flip(c,i)
                
        if self.post_check:
            bad=self.check_local_delaunay()
            if bad:
                raise self.GridException("Delaunay criterion violated")

    def propagating_flip(self,c,i):
        # this is taken from non_recursive_propagating_flip
        # c: cell, akin to face_handle
        # i: index of the originating vertex in cell c.

        # track the stack based on the halfedge one place CW
        # from the edge to be flipped.

        edges=[] # std::stack<Edge> edges;
        vp = self.cells['nodes'][c,i]  #  const Vertex_handle& vp = f->vertex(i);
        p=self.nodes['x'][vp] # const Point& p = vp->point();

        # maybe better to use half-edges here.
        # ordering of edges is slightly different than CGAL.
        # if i gives the vertex, 
        # edges.push(Edge(f,i)); # this is the edge *opposite* vp
        # for our ordering, need edge i+1
        edges.append( self.cell_to_halfedge(c,i) )

        while edges: # (! edges.empty()){
            #const Edge& e = edges.top()
            he=edges[-1]

            he_flip=he.fwd()
            # not sure about this part:
            if self.edges['constrained'][he_flip.j]:
                edges.pop()
                continue
            
            nbr=he_flip.cell_opp()

            if nbr>=0:
                # assuming that ON_POSITIVE_SIDE would mean that p (the location of the
                # originating vertex) is *inside* the CCW-defined circle of the neighbor
                # and would thus mean that the delaunay criterion is not satisfied.
                #if ON_POSITIVE_SIDE != side_of_oriented_circle(n,  p, true):
                nbr_points= self.nodes['x'][ self.cells['nodes'][nbr] ]

                p_in_nbr = robust_predicates.incircle(nbr_points[0],
                                                      nbr_points[1],
                                                      nbr_points[2],
                                                      p )
                #if side_of_oriented_circle(n,  p, true) == ON_POSITIVE_SIDE:
                if p_in_nbr > 0: 
                    self.flip_edge(he_flip.j)
                    extra=he.rev().opposite()
                    edges.append(extra)
                    continue
            edges.pop() # drops last item
            continue

    def find_intersected_elements(self,nA,nB):
        """ 
        returns a history of the elements traversed.
        this includes:
          ('node',<node index>)
          ('edge',<half edge>)
          ('cell',<cell index>)

        note that traversing along an edge is not included - but any
        pair of nodes in sequence implies an edge between them.
        """
        assert nA!=nB
        assert not self.nodes['deleted'][nA]
        assert not self.nodes['deleted'][nB]

        # traversal could encounter multiple types of elements
        trav=('node',nA)
        A=self.nodes['x'][nA]
        B=self.nodes['x'][nB]

        history=[trav]

        if self.dim()==1:
            assert trav[0]=='node'
            n_nbrs=self.node_to_nodes(trav[1])
            for n_nbr in n_nbrs:
                if n_nbr==nB:
                    history.append( ('node',nB) )
                    return history
                if ordered( A,
                            self.nodes['x'][n_nbr],
                            B ):
                    trav=('node',n_nbr)
                    history.append( trav )
                    he=self.nodes_to_halfedge(nA,n_nbr)
                    break
            else:
                assert False # should never get here
            
            while trav!=('node',nB):
                he=he.fwd()
                trav=('node',he.node_fwd())
                history.append(trav)
            return history
        else:
            while trav!=('node',nB):
                if len(history)>1 and history[0]==history[1]:
                    #import pdb
                    #pdb.set_trace()
                    self.log.error("find_intersected_elements: history starts with repeated entries %s"%(history[0],))
                    self.log.error(" possible duplicate node near %s"%( self.nodes['x'][nA] ))
                    raise Exception("find_intersected_elements failed, possible duplicate node near %s"%( self.nodes['x'][nA] ))
                    #return history
                    
                if trav[0]=='node':
                    ntrav=trav[1]
                    for c in self.node_to_cells(ntrav):
                        cn=self.cell_to_nodes(c)
                        # print "At node %d, checking cell %d (%s)"%(ntrav,c,cn)
                        ci_trav=list(cn).index(ntrav) # index of ntrav in cell c
                        nD=cn[(ci_trav+1)%3]
                        nE=cn[(ci_trav+2)%3]
                        if nD==nB or nE==nB:
                            trav=('node',nB)
                            # print "Done"
                            break

                        D=self.nodes['x'][nD]
                        oD=robust_predicates.orientation( A,B,D )
                        if oD>0:
                            continue
                        N=self.nodes['x'][ntrav]
                        if oD==0 and ordered(N,D,B):
                            # fell exactly on the A-B segment, and is in the
                            # right direction
                            trav=('node',nD)
                            break

                        E=self.nodes['x'][nE]
                        oE=robust_predicates.orientation( A,B,E )
                        if oE<0:
                            continue
                        if oE==0 and ordered(N,E,B):
                            # direction
                            trav=('node',nE)
                            break
                        j=self.cell_to_edges(c)[ (ci_trav+1)%3 ]
                        j_nbrs=self.edge_to_cells(j)
                        # AB crosses an edge - record the edge, and the side we are
                        # approaching from:
                        history.append( ('cell',c) )
                        if j_nbrs[0]==c:
                            trav=('edge',self.halfedge(j,0))
                            # making sure I got the 0/1 correct
                            assert trav[1].cell()==c
                            break
                        elif j_nbrs[1]==c:
                            trav=('edge',self.halfedge(j,1))
                            # ditto
                            assert trav[1].cell()==c
                            break
                        assert False
                elif trav[0]=='edge':
                    he=trav[1].opposite()
                    #jnodes=self.edges['nodes'][j]
                    # have to choose between the opposite two edges or their common
                    # node:
                    c_next=he.cell()
                    history.append( ('cell',c_next) )

                    nD=he.fwd().node_fwd()
                    # print "Entering cell %d with nodes %s"%(c_next,self.cell_to_nodes(c_next))

                    oD=robust_predicates.orientation( A,B, self.nodes['x'][nD] )
                    if oD==0:
                        trav=('node',nD)
                    elif oD>0:
                        # going to cross
                        trav=('edge',he.fwd())
                    else:
                        trav=('edge',he.rev())
                else:
                    assert False
                history.append(trav)
        return history

    def locate_for_traversal_outside(self,p,p_other,loc_face,loc_type,loc_index):
        """ 
        Helper method for locate_for_traversal()
        handle the case where p is outside the triangulation, so loc_type
        is either OUTSIDE_AFFINE_HULL or OUTSIDE_CONVEX_HULL
        returns 
          ('edge',<half-edge>)
          ('node',<node>)
          (None,None) -- the line between p and p_other doesn't intersect the triangulation
        """
        dim=self.dim()
        if dim<0:
            # there are no nodes, no work to be done
            return (None,None)
        elif dim==0:
            # a single node. either we'll intersect it, or not.
            N=six.next(self.valid_node_iter()) # get the only valid node
            pN=self.nodes['x'][N]
            # p_other could be coincident with N:
            if (pN[0]==p_other[0]) and (pN[1]==p_other[1]):
                return ('node',N)
            # or we have to test for pN falling on the line between p,p_other
            oN=robust_predicates.orientation(p, pN, p_other)
            # either the segment passes through the one node, or doesn't intersect
            # at all:
            if oN==0 and ordered(p, pN, p_other):
                return ('node',N)
            else:
                return (None,None)
        elif dim==1:
            # This could be much smarter, but current use case has this as a rare
            # occasion, so just brute force it.  find a half-edge, make sure it points
            # towards us, and go.
            if loc_type==self.OUTSIDE_AFFINE_HULL:
                # we know that p is not on the line, but p_other could be.
                # get an edge:
                j=six.next(self.valid_edge_iter())
                he=self.halfedge(j,0)

                # get a half-edge facing p:
                oj=robust_predicates.orientation(p,
                                                 self.nodes['x'][he.node_rev()],
                                                 self.nodes['x'][he.node_fwd()])
                assert oj!=0.0 # that would mean we're collinear
                # if the left side of he is facing us, 
                if oj>0:
                    # good - the left side of he, from rev to fwd, is facing p.
                    pass
                else:
                    # flip it.
                    he=he.opposite()

                # first - check against p_other - it could be on the same side
                # of the line, on the line, or on the other side of the line.
                ojo=robust_predicates.orientation(p_other,
                                                  self.nodes['x'][he.node_rev()],
                                                  self.nodes['x'][he.node_fwd()])
                if ojo>0:
                    # p_other is on the same side of the line as p
                    return (None,None)
                elif ojo==0:
                    # still have to figure out whether p_other is in the line or
                    # off the end.
                    o_loc_face,o_loc_type,o_loc_index=self.locate(p_other)
                    # just saw that it was in line, so better not be outside affine hull
                    assert o_loc_type!=self.OUTSIDE_AFFINE_HULL
                    if o_loc_type==self.OUTSIDE_CONVEX_HULL:
                        # a point off the line to a point beyond the ends of the line -
                        # no intersection.
                        return (None,None)
                    else:
                        if o_loc_type==self.IN_VERTEX:
                            return ('node',o_loc_index)
                        elif o_loc_type==self.IN_EDGE:
                            # This had been just returning the index, but we should
                            # be return half-edge.  
                            # Make sure it faces p:
                            he=self.halfedge(o_loc_index,0)
                            oj2=robust_predicates.orientation(p,
                                                              self.nodes['x'][he.node_rev()],
                                                              self.nodes['x'][he.node_fwd()])
                            assert oj2!=0.0 # that would mean we're collinear
                            # if the left side of he is facing us, 
                            if oj2>0:
                                # good - the left side of he, from rev to fwd, is facing p.
                                pass
                            else:
                                # flip it.
                                he=he.opposite()
                            return ('edge',he)
                    # shouldn't be possible
                    assert False
                else: # p_other is on the other side
                    o_rev=robust_predicates.orientation(p,
                                                        self.nodes['x'][he.node_rev()],
                                                        p_other)
                    if o_rev==0.0:
                        return ('node',he.node_rev())
                    if o_rev > 0:
                        # rev is to the right of the p--p_other line,
                        # so walk forward...
                        A=p ; B=p_other
                    else:
                        # flip it around to keep the loop logic the same.
                        # note that this results in one extra loop, since rev
                        # becomes fwd and we already know that rev is not
                        # far enough over.  whatever.
                        A=p_other ; B=p
                        he=he.opposite()
                    while 1:
                        n_fwd=he.node_fwd()
                        o_fwd=robust_predicates.orientation(A,
                                                            self.nodes['x'][n_fwd],
                                                            B)
                        if o_fwd==0.0:
                            return ('node',n_fwd)
                        if o_fwd<0:
                            return ('edge',he) # had been he.j, but we should return half-edge
                        # must go further!
                        he_opp=he.opposite()
                        he=he.fwd()
                        if he == he_opp: # went round the end - no intersection.
                            return (None,None)
            else: # OUTSIDE_CONVEX_HULL
                # points are in a line, and we're on that line but off the end.
                # in this case, loc_index gives a nearby node
                # so either p_other is also on the line, and the answer
                # is ('node',loc_index)
                # or it's not on the line, and the answer is (None,None)
                orient = robust_predicates.orientation(p,
                                                       self.nodes['x'],
                                                       p_other)
                if orient!=0.0:
                    return (None,None)
                if ordered(p,self.nodes['x'][loc_index],p_other):
                    return ('node',loc_index)
                else:
                    return (None,None)

        elif dim==2:
            # use that to get a half-edge facing p...
            # had done this, but loc_index is already a half edge
            # he_original = he = self.halfedge(loc_index,0)
            he_original = he = loc_index

            # make sure we got the one facing out
            if he.cell()>=0:
                he=he.opposite()

            assert he.cell()<0

            # brute force it
            while 1:
                # does this edge, or one of it's nodes, fit the bill?
                n_rev=he.node_rev()
                n_fwd=he.node_fwd()

                o_j=robust_predicates.orientation(p,
                                                  self.nodes['x'][n_rev],
                                                  self.nodes['x'][n_fwd])
                if o_j<0:
                    # this edge is facing away from p - not a candidate.
                    pass
                else:
                    # note that we could be collinear, o_j==0.0.
                    o_rev=robust_predicates.orientation(p,self.nodes['x'][n_rev],p_other)
                    o_fwd=robust_predicates.orientation(p,self.nodes['x'][n_fwd],p_other)
                    if o_rev == 0.0:
                        if o_fwd == 0.0:
                            assert o_j==0.0
                            if ordered(p,self.nodes['x'][n_rev],self.nodes['x'][n_fwd]):
                                return ('node',n_rev)
                            else:
                                return ('node',n_fwd)
                        else:
                            return ('node',n_rev)
                    elif o_rev>0:
                        if o_fwd<0:
                            # found the edge!
                            return ('edge',he) # had been he.j
                        elif o_fwd==0:
                            return ('node',n_fwd)
                        else:
                            # the whole edge is on the wrong side of the segment
                            pass
                    else: # o_rev<0
                        pass
                he=he.fwd()
                if he==he_original:
                    # none satisfied the intersection
                    return (None,None)

    def locate_for_traversal(self,p,p_other):
        """ Given a point [x,y], reformat the result of 
        self.locate() to be compatible with the traversal 
        algorithm below. In cases where p is outside the
        existing cells/edges/nodes, use the combination of p and p_other
        to figure out the first element which would be hit.
        """
        # Here - figure out which cell, edge or node corresponds to pB
        loc_face,loc_type,loc_index=self.locate(p)
        # not ready for ending point far away, outside
        if loc_type in [self.OUTSIDE_AFFINE_HULL,self.OUTSIDE_CONVEX_HULL]:
            return self.locate_for_traversal_outside(p,p_other,loc_face,loc_type,loc_index)
        elif loc_type == self.IN_VERTEX:
            if loc_face == self.INF_CELL:
                feat=('node', loc_index)
            else:
                feat=('node', self.cells['nodes'][loc_face, loc_index])
        elif loc_type == self.IN_EDGE:
            # This should be a half-edge.
            # The half-edge is chosen such that it either faces p_other, or
            # if all four points are collinear, the ordering is rev -- p -- fwd -- p_other
            # or rev -- p -- p_other -- fwd.
            
            he=self.half_edge(loc_index,0) # start with arbitrary orientation
            p_rev,p_fwd = self.nodes['x'][ he.nodes() ]

            o_p_other = robust_predicates.orientation(p_other, p_rev, p_fwd)
            if o_p==0.0:
                # should this use rel_ordered instead?
                if ordered(p_rev,p,p_other):
                    # good - we're looking along, from rev to fwd
                    pass
                else:
                    he=he.opposite()
            elif o_p<0:
                he=he.opposite()
            else:
                pass 
            feat=('edge', he)
        elif loc_type == self.IN_FACE:
            feat=('cell', loc_face)
        else:
            assert False # shouldn't happen
        return feat
    
    def gen_intersected_elements(self,nA=None,nB=None,pA=None,pB=None):
        """ 
        This is a new take on find_intersected_elements, with changes:
        1. Either nodes or arbitrary points can be given
        2. Elements are returned as a generator, rather than compiled into a list
           and returned all at once.
        3. Traversing along an edge was implied in the output of find_intersected_elements,
           but is explicitly included here as a node--half_edge--node sequence.

        returns a history of the elements traversed.
        this includes:
          ('node',<node index>)
          ('edge',<half edge>)
          ('cell',<cell index>)

        Notes:
        The starting and ending features are included. If points were given
        instead of nodes, then the feature here may be a cell, edge or node.
        
        When the point is outside the convex hull or affine hull, then there is not a
        corresponding feature (since otherwise one would assume that the feature
        is truly intersected).  The first feature returned is simply the first feature
        encountered along the path, necessarily an edge or node, not a face.
        """
        # verify that it was called correctly
        if (nA is not None) and (nB is not None):
            assert nA!=nB
        assert (nA is None) or (not self.nodes['deleted'][nA])
        assert (nB is None) or (not self.nodes['deleted'][nB])

        assert (nA is None) != (pA is None)
        assert (nB is None) != (pB is None) 

        dim=self.dim()

        if nA is not None:
            A=self.nodes['x'][nA]
            trav=('node',nA)
        else:
            A=pA # trav set below

        if nB is not None:
            B=self.nodes['x'][nB]
            end=('node',nB)
        else:
            B=pB # trav set below
        
        if nA is None:
            trav=self.locate_for_traversal(A,B)
            if trav[0] is None:
                return # there are not intersections
            
        if nB is None:
            end=self.locate_for_traversal(B,A)
            # but the orientation of an edge has to be flipped
            if end[0]=='edge':
                end=(end[0],end[1].opposite())
                
        # keep tracks of features crossed, including starting/ending
        assert trav[0] is not None
        history=[trav]
        yield trav 

        if trav==end:
            return
        
        if dim==0:
            # already yielded the one possible intersection
            # but this case should be caught by the return just above
            assert False
            return
        elif dim==1:
            # in the case where p -- p_other crosses the 1-dimensional set of
            # nodes, trav==end, and we already returned above.
            # otherwise, we walk along the edges and nodes

            if trav[0]=='node': # get a first half-edge going in the correct direction
                n_nbrs=self.node_to_nodes(trav[1])
                for n_nbr in n_nbrs:
                    if (ordered( A,
                                 self.nodes['x'][n_nbr],
                                 B ) or
                        np.all(B==self.nodes['x'][n_nbr])):
                        he=self.nodes_to_halfedge(nA,n_nbr)
                        break
                else:
                    assert False
                trav=('edge',he)
                history.append(trav)
                yield trav
            else:
                assert trav[0]=='edge'
                he=trav[1]
                
            while trav != end:
                trav=('node',he.node_fwd())
                history.append(trav)
                yield trav

                if trav==end:
                    break
                
                he=he.fwd()
                trav=('edge',he)
                history.append(trav)
                yield trav
            return
        else: # dim==2
            while trav!=end:
                if trav[0]=='node':
                    # Crossing through a node
                    ntrav=trav[1]
                    N=self.nodes['x'][ntrav]
                    
                    for c in self.node_to_cells(ntrav):
                        cn=self.cell_to_nodes(c)
                        # print "At node %d, checking cell %d (%s)"%(ntrav,c,cn)
                        ci_trav=list(cn).index(ntrav) # index of ntrav in cell c
                        # the other two nodes of the cell
                        nD=cn[(ci_trav+1)%3]
                        nE=cn[(ci_trav+2)%3]

                        # maybe this can be folded in below
                        #if end[0]=='node' and (end[1] in [nD,nE]):
                        #    # trav=('node',nB)
                        #    trav=end
                        #    break

                        # Here
                        D=self.nodes['x'][nD]
                        oD=robust_predicates.orientation( A,B,D )
                        if oD>0:
                            # D is to the right of E, and our target, A is to the right
                            # of both, so this cell is not good
                            continue
                        if oD==0 and np.dot(B-A,D-N)>0: # ordered(A,N,D):
                            # used to test for ordered(N,D,B), but B could be on the
                            # edge, at D, or beyond D.  Test with A to know that the 
                            # edge is going in the right direction, then check for where
                            # B might fall.
                            # HERE: This is a problem, though, because it's possible for
                            # A==N.
                            # What I really want is for A-B to be in the same direction
                            # as N-D.
                            # could test a dot product, but that invites some roundoff
                            # in sinister situations.  The differencing is probably not
                            # a big deal - if we can represent the absolute values
                            # distinctly, then we can certainly represent their differences.
                            # the multiplication could lead to numbers which are too small
                            # to represent.  Any of these issues require absurdly small
                            # values/offsets in the input nodes, and we have already
                            # established that these all lie on a line and are distinct.
                            # 
                            # The possible positive orderings
                            #   [A=N] -- D -- B
                            #   A -- N -- D -- B
                            #   [A=N] -- [D==B]
                            #   [A=N] -- B -- D
                            # 
                            
                            # fell exactly on the A-B segment, and is in the
                            # right direction

                            # Announce the edge, which could be the end of the traversal
                            trav=('edge',self.nodes_to_halfedge(ntrav,nD))
                            history.append(trav)
                            yield trav
                            if trav==end:
                                return
                            # And on to the node:
                            trav=('node',nD)
                            break # and we've completed this step

                        E=self.nodes['x'][nE]
                        oE=robust_predicates.orientation( A,B,E )
                        if oE<0:
                            # A is to the left of E
                            continue
                        if oE==0 and np.dot(B-A,E-N): # ordered(A,N,E):
                            # Same as above - establish that it goes in the correct direction.
                            # again, the dot product is mildly dangerous
                            # again - fell exactly on the segment A-B, it's in the right
                            # direction.
                            trav=('edge',self.nodes_to_halfedge(ntrav,nE))
                            history.append(trav)
                            yield trav
                            if trav==end:
                                return
                            trav=('node',nE)
                            break
                        
                        # if we get to here, then A--B passes through the cell, and either
                        # we stop at this cell, or A--B crosses the opposite edge:
                        trav=('cell',c)

                        if trav==end:
                            # don't try to traverse the cell - we're done!
                            # trav will get appended below
                            break
                        else:
                            # announce the cell, and move on to the edge
                            history.append(trav)
                            yield trav
                            trav=None # avoid confusion, clear this out
                            
                            # AB crosses an edge - record the edge, and the side we are
                            # approaching from:
                            
                            j=self.cell_to_edges(c)[ (ci_trav+1)%3 ]
                            j_nbrs=self.edge_to_cells(j)
                            
                            if j_nbrs[0]==c:
                                trav=('edge',self.halfedge(j,0))
                            elif j_nbrs[1]==c:
                                trav=('edge',self.halfedge(j,1))
                            else:
                                assert False
                            # making sure I got the 0/1 correct
                            assert trav[1].cell()==c
                            break
                                
                elif trav[0]=='edge':
                    # trav[1].cell() is the cell we just left
                    # this then is the half-edge facing the cell we're
                    # entering
                    he=trav[1].opposite()
                    
                    c_next=he.cell()
                    trav=('cell',c_next)
                    if trav==end:
                        pass # done!
                    else:
                        # have to choose between the opposite two edges or their common
                        # node.
                        # record the cell we just passed through
                        history.append(trav)
                        yield trav

                        nD=he.fwd().node_fwd()
                        # print "Entering cell %d with nodes %s"%(c_next,self.cell_to_nodes(c_next))

                        oD=robust_predicates.orientation( A,B, self.nodes['x'][nD] )
                        if oD==0:
                            trav=('node',nD)
                        elif oD>0:
                            # going to cross the edge "on the right" (I think)
                            trav=('edge',he.fwd())
                        else:
                            # going to cross the edge "on the left"
                            trav=('edge',he.rev())
                else:
                    assert False
                history.append(trav)
                yield trav
        return
    
    def add_constraint(self,nA,nB):
        if nA==nB:
            # likely a corrupt grid, but we can carry on as if it doesn't matter.
            return
        jAB=self.nodes_to_edge([nA,nB])
        if jAB is not None:
            # no work to do - topology already good.
            if self.edges['constrained'][jAB]:
                raise DuplicateConstraint(nodes=[nA,nB])
            self.edges['constrained'][jAB]=True
            return jAB

        # inserting an edge from 0-5.
        int_elts=self.find_intersected_elements(nA,nB)

        # Now we need to record the two holes bordered the new edge:
        left_nodes=[nA] # will be recorded CW
        right_nodes=[nA] # will be recorded CCW

        # Iterate over the crossed elements, checking that the new
        # edge doesn't encounter any collinear nodes or other constrained
        # edges.  Build up the nodes of the holes at the same time.
        dead_cells=[]
        dead_edges=[]
        for elt in int_elts[1:-1]:
            if elt[0]=='node':
                raise self.ConstraintCollinearNode("Constraint intersects a node",
                                                   node=elt[1])
            if elt[0]=='cell':
                dead_cells.append(elt[1])
            if elt[0]=='edge':
                if self.edges['constrained'][ elt[1].j ]:
                    raise IntersectingConstraints("Constraint intersects a constraint",
                                                  edge=elt[1].j )
                next_left=elt[1].node_fwd()
                if left_nodes[-1]!=next_left:
                    left_nodes.append(next_left)
                next_right= elt[1].node_rev()
                if right_nodes[-1]!=next_right:
                    right_nodes.append(next_right)
                dead_edges.append(elt[1].j)
        left_nodes.append(nB)
        right_nodes.append(nB)
        left_nodes = left_nodes[::-1]

        # tricky business here
        # but the delaunay business is only invoked on node operations - leaving
        # the edge/cell operations free and clear to violate invariants
        for c in dead_cells:
            self.delete_cell(c)
        for j in dead_edges:
            self.delete_edge(j)

        j=self.add_edge(nodes=[nA,nB],constrained=True)
            
        # and then sew up the holes!
        self.fill_hole( left_nodes )
        self.fill_hole( right_nodes )
        return j

    def remove_constraint(self,nA=None,nB=None,j=None):
        """ Assumes that there exists a constraint between nodes
        nA and nB (or that the edge given by j is constrained).
        The constrained flag is removed for the edge, and if
        the Delaunay criterion is no longer satisfied edges are
        flipped as needed.
        """
        if j is None:
            if nA==nB:
                # likely a bad mesh, but for umbra usage it's better
                # to let this pass
                self.log.warning(f"remove_constraint: Ignoring duplicate nodes nA=nB={nA}")
                return 

            j=self.nodes_to_edge([nA,nB])

            if j is None:
                self.log.warning(f"remove_constraint: no edge found for nA={nA} nB={nB}")
                return
            
        assert self.edges['constrained'][j]
        self.edges['constrained'][j]=False

        c1,c2=self.edge_to_cells(j)
        if (c1>=0) and (c2>=0):
            c=c1 # can we just propagate from one side?
            for ni,n in enumerate(self.cell_to_nodes(c1)):
                if n not in self.edges['nodes'][j]:
                    self.propagating_flip(c1,ni)
                    break
        if self.post_check:
            self.check_local_delaunay()

    def node_to_constraints(self,n):
        return [j
                for j in self.node_to_edges(n)
                if self.edges['constrained'][j]]

    def init_from_grid(self,g,node_coordinate='x',set_valid=False,
                       valid_min_area=1e-2,on_intersection='exception'):
        """
        Initialize from the nodes and edges of an existing grid, making
        existing edges constrained
        node_coordinate: supply the name of an alternate coordinate defined
          on the nodes. g.nodes[node_coordinate] should be an [Ncell,2] field.

        set_valid: if True, add a 'valid' field for cells, and set to Tru
          for cells of the triangulation that have finite area and fall 
          within the src grid g.

        on_intersection: 
        'exception': intersecting edges in the input grid raise an error.
        'insert': at intersecting edges construct and insert a new node.
        """
        if set_valid:
            self.add_cell_field('valid',np.zeros(self.Ncells(),np.bool_),
                                on_exists='pass')

        # Seems that the indices will get misaligned if there are
        # deleted nodes.
        # TODO: add node index mapping code here.
        assert np.all( ~g.nodes['deleted'] )
        
        self.bulk_init(g.nodes[node_coordinate][~g.nodes['deleted']])
        all_segs=[ g.edges['nodes'][j]
                   for j in g.valid_edge_iter() ]
        while all_segs:
            nodes=all_segs.pop(0)
            if on_intersection=='exception':
                self.add_constraint( *nodes )
            else:
                self.add_constraint_and_intersections( *nodes )
                
        if set_valid:
            from shapely import geometry
            self.cells['valid']=~self.cells['deleted']
            # Maybe unnecessary.  Had some issues with 0 fill values here.
            self.cells['_area']=np.nan
            self.cells['_center']=np.nan
            areas=self.cells_area()
            self.cells['valid'][areas<=valid_min_area]=False

            poly=g.boundary_polygon()
            centroids=self.cells_centroid()
            for c in np.nonzero(self.cells['valid'])[0]:
                if not poly.contains( geometry.Point(centroids[c]) ):
                    self.cells['valid'][c]=False
                    
    def add_constraint_and_intersections(self,nA,nB,on_exists='exception'):
        """
        Like add_constraint, but in the case of intersections with existing constraints 
        insert new nodes as needed and update existing and new constrained edges.
        """
        all_segs=[ [nA,nB] ]
        result_nodes=[nA]
        result_edges=[]
        
        while all_segs:
            nA,nB=all_segs.pop(0)
            
            try:
                j=self.add_constraint(nA,nB)
            except IntersectingConstraints as exc:
                if isinstance(exc,ConstraintCollinearNode):
                    all_segs.insert(0, [nA,exc.node] )
                    all_segs.insert(1, [exc.node,nB] )
                    continue
                else:
                    j_other=exc.edge
                    assert j_other is not None
                    
                    segA=self.nodes['x'][self.edges['nodes'][j_other]]
                    segB=self.nodes['x'][[nA,nB]]
                    x_int,alphas=segment_segment_intersection(segA,segB)
                    # Getting an error where x_int is one of the endpoints of
                    # segA.  This is while inserting a contour that ends on
                    # the boundary.
                    n_new=self.split_constraint(j=j_other,x=x_int)
                    
                    if nB!=n_new:
                        all_segs.insert(0,[n_new,nB])
                    if nA!=n_new:
                        all_segs.insert(0,[nA,n_new])
                    continue
            except DuplicateConstraint as exc:
                if on_exists=='exception':
                    raise
                elif on_exists=='ignore':
                    j=self.nodes_to_edge(nA,nB)
                elif on_exists=='stop':
                    break
                else:
                    assert False,"Bad value %s for on_exists"%on_exists
            result_nodes.append(nB)
            assert j is not None
            result_edges.append(j)
                
        return result_nodes,result_edges
    
    def split_constraint(self,x,j):
        nodes_other=self.edges['nodes'][j].copy()

        j_data=unstructured_grid.rec_to_dict(self.edges[j].copy())
        
        self.remove_constraint(j=j)

        n_new=self.add_or_find_node(x=x)

        js=[]
        if nodes_other[0]!=n_new:
            js.append( self.add_constraint(nodes_other[0],n_new) )
        if n_new!=nodes_other[1]:
            js.append( self.add_constraint(n_new,nodes_other[1]) )

        for f in j_data:
            if f in ['nodes','cells','deleted']: continue
            self.edges[f][js]=j_data[f]
            
        return n_new
                    
    def add_constrained_linestring(self,coords,
                                   on_intersection='exception',
                                   on_exists='exception',
                                   closed=False):
        """
        Optionally insert new nodes as needed along
        the way.
        on_intersection: when a constraint intersects an existing constraint,
          'exception' => re-raise the exception
          'insert' => insert a constructed node, and divide the new and old constraints.
        on_exists' => when a constraint to be inserted already exists,
          'exception' => re-raise the exception
          'ignore' => keep going
          'stop' => return

        closed: Whether the first and last nodes are also connected

        returns [list of nodes],[list of edges]
        """
        nodes=[self.add_or_find_node(x=x)
               for x in coords]
        result_nodes=[nodes[0]]
        result_edges=[]

        if not closed:
            ab_list=zip(nodes[:-1],nodes[1:])
        else:
            ab_list=zip(nodes,np.roll(nodes,-1))
        for a,b in ab_list:
            if on_intersection=='insert':
                sub_nodes,sub_edges=self.add_constraint_and_intersections(a,b,
                                                                          on_exists=on_exists)
                result_nodes+=sub_nodes[1:]
                result_edges+=sub_edges
                
                if (on_exists=='stop') and (sub_nodes[-1]!=b):
                    print("Stopping early")
                    break
            else:
                try:
                    j=self.add_constraint(a,b)
                except DuplicateConstraint as exc:
                    if on_exists=='exception':
                        raise
                    elif on_exists=='stop':
                        break
                    elif on_exists=='ignore':
                        j=self.nodes_to_edge(a,b)
                result_nodes.append(b)
                result_edges.append(j)
        return result_nodes,result_edges
                    
    def bulk_init_slow(self,points):
        raise Exception("No - it's really slow.  Don't do this.")
    
    def bulk_init(self,points): # ExactDelaunay
        if spatial is None:
            return self.bulk_init_slow(points)

        # looks like centering this affects how many cells Delaunay
        # finds.  That's lame.
        sdt = spatial.Delaunay(points-points.mean(axis=0))

        self.nodes=np.zeros( len(points), self.node_dtype)
        try: # version issues
            vertices=sdt.vertices
        except AttributeError:
            vertices=sdt.simplices
            
        self.cells=np.zeros( vertices.shape[0], self.cell_dtype)

        self.nodes['x']=points
        self.cells['nodes']=vertices

        # looks like it's CGAL style:
        # neighbor[1] shares nodes[0] and nodes[2]
        # vertices are CCW

        for c in range(self.Ncells()):
            for i,(a,b) in enumerate(circular_pairs(self.cells['nodes'][c])):
                # first time - that would be i=0, and the first two nodes.
                # but for neighbors, it's indexed by the opposite node.  so the edge
                # connected the nodes[0]--nodes[1] corresponds with neighbor 2.
                c_nbr=sdt.neighbors[c,(i+2)%3]

                # c_nbr==-1 on convex hull.
                # only worry about cases where c is larger.
                if c<c_nbr:
                    continue

                if c_nbr<0:
                    c_nbr=self.INF_CELL

                j=self.add_edge(nodes=[a,b],
                                cells=[c,c_nbr])
                # and record in the cell, too
                self.cells['edges'][c,i]=j
                if c_nbr!=self.INF_CELL:
                    nbr_nodes=self.cells['nodes'][c_nbr]
                    for i_nbr in [0,1,2]:
                        if nbr_nodes[i_nbr]==b and nbr_nodes[(i_nbr+1)%3]==a:
                            self.cells['edges'][c_nbr,i_nbr]=j
                            break
                    else:
                        assert False
    def constrained_centers(self):
        """
        For cells with no constrained edges, return the circumcenter.
        If return centroid.
        The details may evolve, but the purpose is to get a point which 
        is inside the domain and can be used like a circumcenter (i.e. 
        approximately lies on the medial axis of the continous boundary).
        """
        ccs=self.cells_center(refresh=True) # circumcenters
        centroids=self.cells_centroid()
        e2c=self.edge_to_cells() # recalc=True)
        cell_with_constraint=np.unique( e2c[ self.edges['constrained']] )
        result=ccs.copy()
        result[cell_with_constraint] = centroids[cell_with_constraint]
        return result

    # TODO: def constrained_radii(self):
    #  Calculate the usual circumradius, but for centers which were
    #  adjusted due to a constrained edge also check point-segment
    #  distances.

    def point_clearance(self,x,hint=None):
        """
        Return the distance from point x=[p_x,p_y] to the nearest
        node or constrained segment of the triangulation.

        hint: To speed up consecutive queries with spatial locality, pass
        a dictionary, and a new dictionary will be returned as the second
        item in a tuple. The initial dictionary can be empty, or 'c':int
        to give a starting face of the triangulation.
        """
        if hint is not None:
            loc_face,loc_type,loc_index=self.locate(x,**hint)
        else:
            loc_face,loc_type,loc_index=self.locate(x)

        # I don't think it is strictly necessary to be one of these,
        # but we have to check to know whether loc_face is valid.
        if loc_type==self.OUTSIDE_CONVEX_HULL:
            # loc_index is a half-edge
            min_clearance=min( dist( self.nodes['x'][loc_index.node_fwd()], x ),
                               dist( self.nodes['x'][loc_index.node_rev()], x ) )
            j=loc_index.j
            if self.edges['constrained'][j]:
                j_clearance=point_segment_distance(x, self.nodes['x'][self.edges['nodes'][j]] )
                min_clearance=min(min_clearance,j_clearance)
            # No update to hint
        elif loc_type in (self.IN_VERTEX, self.IN_EDGE, self.IN_FACE):
            face_nodes=self.cells['nodes'][loc_face]

            min_clearance=dist( self.nodes['x'][face_nodes], x ).min()

            for j in self.cell_to_edges(loc_face):
                if self.edges['constrained'][j]:
                    j_clearance=point_segment_distance(x, self.nodes['x'][self.edges['nodes'][j]] )
                    min_clearance=min(min_clearance,j_clearance)
            hint={'c':loc_face}
        else:
            raise Exception("Loc type OUTSIDE_AFFINE_HULL is not implemented for point_clearance")
        
        if hint is not None:
            return min_clearance,hint
        else:
            return min_clearance
    
            
# Issues:
#   Calls like edge_to_cells do not scale well right now.  In particular,
#   it would be better in this code to always specify the edge, so that 
#   a full scan isn't necessary.
