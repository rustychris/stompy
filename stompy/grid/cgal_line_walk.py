"""
Extract line_walk code from live_dt for use in other places.
Provides a more "hands-on" approach to checking an arbitrary
line segment against existing elements of a CGAL constrained
delaunay triangulation
"""
from __future__ import print_function
from stompy.spatial import robust_predicates
from stompy.grid import exact_delaunay
import numpy as np

from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Kernel import Point_2

def distance_left_of_line(pnt, qp1, qp2):
    # return the signed distance for where pnt is located left of
    # of the line qp1->qp2
    #  we don't necessarily get the real distance, but at least something
    #  with the right sign, and monotonicity
    vec = qp2 - qp1
    
    left_vec = np.array( [-vec[1],vec[0]] )
    
    return (pnt[0] - qp1[0])*left_vec[0] + (pnt[1]-qp1[1])*left_vec[1]

def circ_to_gen(circ,repeat_start=False):
    """
    Given a CGAL circulator, loop once through it, yielding elements
    as if it were an iterator.
    
    If repeat_start is True, the first item is returned at the end, too.
    """
    elt0=circ.next()
    yield elt0
    while True:
        eltN=circ.next()
        if eltN==elt0:
            if repeat_start:
                yield eltN
            return
        yield eltN
        
## steppers for line_walk_edges_new
def next_from_vertex(DT, vert, vec):
    # from a vertex, we either go into one of the faces, or along an edge
    qp1,qp2 = vec

    last_left_distance=None
    last_nbr = None

    start = None
    # we need to track the last and next, which is easiest by visiting
    # the first neighbor twice, thanks to repeat_start=True
    for nbr in circ_to_gen(DT.incident_vertices(vert),repeat_start=True):
        if DT.is_infinite(nbr):
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
                for face in circ_to_gen(DT.incident_faces(vert)):
                    for j in range(3):
                        if face.vertex(j) == nbr:
                            for k in range(3):
                                if face.vertex(k) == last_nbr:
                                    return ['f',face]
                raise Exception("Found a good pair of incident vertices, but failed to find the common face.")

        # Sanity check - if we've gone all the way around
        if start is None:
            start = nbr
        else: # must not be the first time through the loop:
            if nbr == start:
                raise Exception("This is bad - we checked all vertices and didn't find a good neighbor")

        last_left_distance = left_distance
        last_nbr = nbr
        if DT.is_infinite(nbr):
            last_left_distance = None

    raise Exception("Fell through!")

def next_from_edge(DT, edge, vec):
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

def next_from_face(DT, f, vec):
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

def delaunay_neighbors(self, n):
    """ returns an array of node ids that the DT connects the given node
    to.  Includes existing edges
    """
    nbrs = []

    # how do we stop on a circulator? it's now handled in circ_to_gen()
    for v in circ_to_gen(DT.incident_vertices(self.vh[n])):
        if DT.is_infinite(v):
            continue

        # This is going to need something faster, or maybe the store info
        # bits of cgal.
        nbr_i = self.vh_info[v] 
        if nbr_i is None:
            print("    While looking for vertex at ",v.point())
            raise Exception("expected vertex handle->node, but instead got %s"%nbr_i)
        nbrs.append( nbr_i )
    return np.array(nbrs)

def line_walk(DT,v1,v2,
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

    # Get the points from the vertices, not self.points, because in some cases
    # (adjust_move_node) we may be probing
    p1 = np.array([ v1.point().x(), v1.point().y()] )
    p2 = np.array([ v2.point().x(), v2.point().y()] )

    # print("Walking the line: ",p1,p2)

    hits = [ ['v',v1] ]

    # do the search:
    # Note that we really need a better equality test here
    # hits[-1][1] != v2 doesn't work (not sure why)
    def obj_eq(a,b):
        return type(a)==type(b) and a==b

    while not obj_eq(hits[-1][1], v2):
        # if we just came from a vertex, choose a new face in the given direction
        if hits[-1][0] == 'v':
            # print("Last hit was the vertex at %s"%(hits[-1][1].point()))

            # like face_in_direction, but also check for possibility that
            # an edge is coincident with the query line.

            next_item = next_from_vertex(DT, hits[-1][1], (p1,p2))

            # print("Moved from vertex to (%s,%s)"%(next_item[0],next_item[1:]))

            if next_item[0] == 'v':
                # Find the edge connecting these two:
                for e in circ_to_gen(DT.incident_edges( next_item[1] )):
                    f,v_opp = e

                    if f.vertex( (v_opp+1)%3 ) == hits[-1][1] or \
                       f.vertex( (v_opp+2)%3 ) == hits[-1][1]:
                        hits.append( ['e', (f,v_opp)] )
                        break

        elif hits[-1][0] == 'f':
            # either we cross over an edge into another face, or we hit
            # one of the vertices.

            next_item = next_from_face(DT, hits[-1][1], (p1,p2))

            # in case the next item is also a face, go ahead and insert
            # the intervening edge
            if next_item[0]=='f':
                middle_edge = None

                for v_opp in range(3):
                    #print("Comparing %s to %s looking for the intervening edge"%(hits[-1][1].neighbor(v_opp),
                    #                                                             next_item[1]))
                    
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
            next_item = next_from_edge(DT, hits[-1][1], (p1,p2))

        hits.append( next_item )

    # print("Got hits: ",hits)

    # but ignore the first and last, since they are the starting/ending points
    hits = hits[1:-1]

    # live_dt code now translates CGAL elements back to triangulation references,
    # include turning (face,idx) edges to pairs of nodes.
    
    return hits

def line_conflicts(DT,v1=None,v2=None,p1=None,p2=None):
    """ returns a list of vertex tuple for constrained segments that intersect
    the given line.
    in the case of vertices that are intersected, just a tuple of length 1
    (and assumes that all vertices qualify as constrained)
    """

    # if points were given, create some temporary vertices
    if p1 is not None:
        cp1 = Point_2( p1[0], p1[1] )
        v1 = DT.insert(cp1) # self.vh_info[v1] = 'tmp'

    if p2 is not None:
        cp2 = Point_2( p2[0], p2[1] )
        v2 = DT.insert(cp2) # self.vh_info[v2] = 'tmp'

    crossings = line_walk(DT,v1=v1,v2=v2)

    constrained = []
    for crossing_type,crossing in crossings:
        if crossing_type == 'f':
            continue
        if crossing_type == 'v':
            constrained.append( (crossing_type,crossing) )
            continue
        if crossing_type == 'e':
            face,vidx = crossing
            # print("Got potential conflict with edge",face,vidx)
            # this probably won't work.
            if DT.is_constrained( (face,vidx) ):
                constrained.append( ('e',"don't worry about it") )

    if p1 is not None:
        DT.remove( v1 )
    if p2 is not None:
        DT.remove( v2 )
    return constrained

# def check_line_is_clear_batch(DT,p1,n2):
#     """ 
#     When checking multiple nodes against the same point,
#     may be faster to insert the point just once.
#     p1: [x,y] 
#     n2: [ node, node, ... ]
#     Return true if segments from p1 to each node in n2 are
#     all clear of constrained edges
#     """
#     pnt = Point_2( p1[0], p1[1] )
#     probe = self.DT.insert(pnt)
#     self.vh_info[probe] = 'PROBE!'
# 
#     try:
#         for nbr in n2:
#             crossings = self.check_line_is_clear_new( n1=nbr, v2=probe )
#             if len(crossings) > 0:
#                 return False
#     finally:
#         del self.vh_info[probe]
#         self.DT.remove(probe)
# 
#     return True
