import pdb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib import cm
from stompy import utils

from CGAL.CGAL_Triangulation_2 import (Constrained_Delaunay_triangulation_2,
                                       Constrained_Delaunay_triangulation_2_Edge)

from CGAL.CGAL_Kernel import (Point_2,Segment_2)

# 1. The CGAL constrained delaunay triangulation wrapped in the bindings
#    appears to allow intersections, constructing an approximate new vertex

##

# Do I get access to the intersection flag?
# probably not, it's a template parameter

DT=Constrained_Delaunay_triangulation_2()

xy=[ [0,0],
     [1,0],
     [1,1],
     [0,-1],
     [2,0],
     [2,1],
     [2,2]]
points= [ Point_2(x,y) for x,y in xy ]
vh=[DT.insert(p) for p in points]

DT.insert_constraint(vh[0],vh[1])
DT.insert_constraint(vh[1],vh[2])
DT.insert_constraint(vh[1],vh[4])
    
# DT.insert_constraint(vh3,vh2) # causes a new vertex to be created

# #

def plot_dt(DT,ax=None):
    segs=[]

    weights=[]

    # this is a proper iterator
    for face,idx in DT.finite_edges():
        if DT.is_constrained( (face,idx) ):
            weight=3
        else:
            weight=1
        weights.append(weight)
        vert1=face.vertex( (idx+1)%3)
        vert2=face.vertex( (idx+2)%3)
        seg= [ [vert1.point().x(),
                vert1.point().y()],
               [vert2.point().x(),
                vert2.point().y()] ]
        segs.append(seg)

    coll=LineCollection(segs)
    colors=cm.jet( np.random.random(len(segs)) )
    coll.set_edgecolors(colors)
    coll.set_linewidths(weights)

    ax=ax or plt.gca()
    ax.add_collection(coll)
    ax.axis('equal')
    
    return coll

def face_to_tri(f):
    pnts=[]
    for i in [0,1,2]:
        pnt=f.vertex(i).point()
        pnts.append( (pnt.x(),pnt.y()) )
    return pnts

def segment_is_free(DT,vh_a,vh_b,ax=None):
    # requires points, not vertex handles
    # N.B. that means it does its own locate, which is O(sqrt(n))
    # also, without a face passed in, it will start at the boundary of
    # the convex hull.
    # so get a real face:
    for init_face in DT.incident_faces(vh_a):
        if not DT.is_infinite(init_face):
            break
    else:
        assert False
        
    lw=DT.line_walk(vh_a.point(),vh_b.point(),init_face) 

    seg=Segment_2( vh_a.point(), vh_b.point() )
    
    faces=[]
    for face in lw:
        # check to make sure we don't go through a vertex
        for i in [0,1,2]:
            v=face.vertex(i)
            if v==vh_a or v==vh_b:
                continue
            if seg.has_on(v.point()):
                return False # collinear!
        if len(faces):
            # figure out the common edge
            for i in [0,1,2]:
                if face.neighbor(i)==faces[-1]:
                    break
            else:
                tri_1=face_to_tri(face)
                tri_2=face_to_tri(faces[-1])

                pcoll=PolyCollection([tri_1,tri_2])
                ax=ax or plt.gca()
                ax.add_collection(pcoll)
                
                #assert False
                pdb.set_trace()

            edge=Constrained_Delaunay_triangulation_2_Edge(face,i)
            if DT.is_constrained(edge):
                if 0:
                    tri_1=face_to_tri(face)
                    tri_2=face_to_tri(faces[-1])
                    pcoll=PolyCollection([tri_1,tri_2])
                    ax=ax or plt.gca()
                    ax.add_collection(pcoll)
                return False
            
        faces.append(face)
        if face.has_vertex(vh_b):
            break
    return True
    
# # 


fig=plt.figure(1)
fig.clf()
ax=fig.gca()
plot_dt(DT,ax=ax)
for idx,p in enumerate(xy):
    ax.text( p[0],p[1],str(idx) )
    


# need some more tests to make sure this does what I think it
# does
# 

# basic tests -
assert segment_is_free(DT,vh[0],vh[1])
assert not segment_is_free(DT,vh[3],vh[2])
assert not segment_is_free(DT,vh[3],vh[6])

assert not segment_is_free(DT,vh[3],vh[5])

##

# And to make it more performant:

def gen_grid(ncols=10,nrows=3):
    vhs=np.zeros( (nrows,ncols), object)

    DT=Constrained_Delaunay_triangulation_2()

    for row in range(nrows):
        for col in range(ncols):
            vhs[row,col] = DT.insert( Point_2(col,row) )

    for row in range(nrows-1):
        for col in range(ncols):
            DT.insert_constraint( vhs[row,col],
                                  vhs[row+1,col] )

    for row in range(nrows):
        for col in range(ncols-1):
            DT.insert_constraint( vhs[row,col],
                                  vhs[row,col+1] )
    return vhs,DT

vhs,DT=gen_grid(10,3)

fig=plt.figure(1)
fig.clf()
ax=fig.gca()
plot_dt(DT,ax=ax)
        
assert segment_is_free(DT,vhs[1,5],vhs[1,6],ax=ax)
assert not segment_is_free(DT,vhs[0,5],vhs[2,6],ax=ax)

##

# checks out
print vhs[1,5].point().x(),vhs[1,5].point().y()
print vhs[1,6].point().x(),vhs[1,6].point().y()

##

# How is the asymptotic performance?
# Good!  This test is slow to run, but does show that by 10k columns,
# the locate() time surpasses line_walk.  Line walk is essentially constant
sizes=np.exp( np.linspace( np.log(10), np.log(100000), 10) ).astype('i4')

##
import time

runs=[]
for size in sizes:
    vhs,DT=gen_grid(size,3)

    t_start=time.time()
    middle=int(size/2)
    
    for _ in range(10):
        assert segment_is_free(DT,vhs[1,5],vhs[1,6],ax=ax)
        assert not segment_is_free(DT,vhs[0,5],vhs[2,6],ax=ax)

        assert segment_is_free(DT,vhs[1,0],vhs[1,1],ax=ax)
        assert not segment_is_free(DT,vhs[0,1],vhs[2,0],ax=ax)

        assert segment_is_free(DT,vhs[1,middle],vhs[1,middle+1],ax=ax)
        assert not segment_is_free(DT,vhs[0,middle],vhs[2,middle+1],ax=ax)
    elapsed=time.time() - t_start

    # And some locate calls
    t_start=time.time()
    for _ in range(20):
        DT.locate( vhs[1,5].point() )
        DT.locate( vhs[1,0].point() )
        DT.locate( vhs[0,middle].point() )
    loc_elapsed=time.time() - t_start
        
    runs.append( (size,elapsed,loc_elapsed) )

##

runs=np.array(runs)

plt.figure(2).clf()

plt.semilogx( runs[:,0],runs[:,1], 'g-o')
plt.semilogx( runs[:,0],runs[:,2], 'k-o')
plt.xlabel('Grid size')
plt.ylabel('Time for 60 evaluations (s)')

##

# would be nice to use unstructured_grid.py and a node
# field to hold vertex handles.
# does that play nice with array_append?

A=np.zeros( vhs.shape[0], dtype=[ ('idx',np.int32),
                                  ('vh',object) ] )
A['vh']=vhs[:,0]

##

rec=np.zeros( (), dtype=A.dtype )
# that works.  the vh is initialized to 0 instead of None, but
# whatever.
A2=utils.array_append(A, rec)

##

# do vertex handles hash appropriately?
face=DT.finite_faces().next()
v_1=face.vertex(0)
v_2=face.vertex(0)

print v_1==v_2 # good

d={v_1:1}
assert v_1 in d # yes
assert v_2 in d # yes

# vertices don't have any sort of set_info method
##

# What about setting info when they are inserted?
# nope. just use hashes.

##

# what about adding a constraint which runs through an existing
# node

from stompy.grid import (unstructured_grid,
                         shadow_cdt)


def test_collinear():
    g=unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)

    xys=[ [0,0],
          [1,0],
          [1,1],
          [0,-1],
          [2,0],
          [2,1] ]

    for xy in xys:
        g.add_node(x=xy)

    g.add_edge(nodes=[0,1])
    g.add_edge(nodes=[0,3])
    g.add_edge(nodes=[1,2])
    g.add_edge(nodes=[1,4])

    # this is problematic -
    #  in one case, it goes through a node, and the successive faces
    #  don't share an edge.  not sure if this is guaranteed behavior.
    #  even if it is, there could be a collinear node, but without
    #  extra adjacent faces, so we wouldn't know...
    # print segment_is_free(DT,vh[3], vh[5])

    # Want to detect that this failed:
    ## 
    fig=plt.figure(1)
    fig.clf()
    g.plot_edges(lw=7,color='0.5')
    g.plot_nodes(labeler=lambda n,rec: str(n))

    plot_dt(cdt.DT,ax=fig.gca())

    ## 

    a=3
    b=5

    # actions in the cdt before allowing the grid to add the edge:

    cdt.DT.insert_constraint(g.nodes['vh'][a],g.nodes['vh'][b])


    ##

    # traverse constrained edges from vh[3], trying to get
    # to vh[5]. Anonymous vertices mean we had intersecting constraints,
    # any other vertices mean collinear nodes.
    # the trick is figuring out how to traverse

    ## 


    # stepping along from vh_trav
    def traverse_fresh_edge(self,a,b):
        """
        invoke after adding a constraint a--b, but before
        it has been added to self.g.

        steps from node a to node b, finding out what constrained
        edges actually were added.
        returns a list of tuples, each tuple is (node, vertex_handle)
        """

        vh_trav=self.g.nodes['vh'][a]
        vh_list=[ (vh_trav,a) ]

        while 1:
            # fetch list of constraints for traversal vertex handle
            constraints=[]
            self.DT.incident_constraints(vh_trav,constraints)

            n_trav=self.vh_info[vh_trav]
            # which neighbors are already constrained
            trav_nbrs_to_ignore = list(self.g.node_to_nodes(n_trav))

            maybe_next=[] # potential vh for next step

            for edge in constraints:
                face=edge[0] # work around missing API
                idx=edge[1]

                v1=face.vertex( (idx+1)%3 )
                v2=face.vertex( (idx+2)%3 )

                if v1==vh_trav:
                    vh_other=v2
                else:
                    vh_other=v1

                n_me = self.vh_info[vh_trav]
                n_other=self.vh_info[vh_other]

                print "Checking constrained edge %s -- %s."%(n_me,n_other)

                # If this is an existing neighbor in some other direction, ignore.

                if n_other!=b and n_other in trav_nbrs_to_ignore:
                    print "   [skip]"
                    continue
                if len(vh_list)>1 and vh_other==vh_list[-2][0]:
                    print "   [backwards - skip]"
                    continue

                maybe_next.append( (vh_other,n_other) )
            assert len(maybe_next)==1

            vh_trav,n_trav=maybe_next[0]
            vh_list.append( maybe_next[0] )

            print "Stepped to node: %s"%( n_trav )

            if n_trav==b:
                print "Done with traversal"
                break
        return vh_list

    # This works okay..
    elts=traverse_fresh_edge(cdt,a,b)

    # what are the options?
    #   a. bring in the big guns like in live_dt.py
    #   b. ignore this, and move on to quad algorithm
    #   c. insert the edge, and if it doesn't come back
    #      looking right, rewind manually
    #      This seems best, assuming we can do that.

##

# def test_intersection():
g=unstructured_grid.UnstructuredGrid()
cdt=shadow_cdt.ShadowCGALCDT(g)

xys=[ [0,0],
      [1,0],
      [1,1],
      [0,-1],
      [2,0],
      [2,2] ]

for xy in xys:
    g.add_node(x=xy)

g.add_edge(nodes=[0,1])
g.add_edge(nodes=[0,3])
g.add_edge(nodes=[1,2])
g.add_edge(nodes=[1,4])

# this is problematic -
#  in one case, it goes through a node, and the successive faces
#  don't share an edge.  not sure if this is guaranteed behavior.
#  even if it is, there could be a collinear node, but without
#  extra adjacent faces, so we wouldn't know...
# print segment_is_free(DT,vh[3], vh[5])

# Want to detect that this failed:
## 

a=3
b=5

# actions in the cdt before allowing the grid to add the edge:

cdt.DT.insert_constraint(g.nodes['vh'][a],g.nodes['vh'][b])


## 
fig=plt.figure(1)
fig.clf()
g.plot_edges(lw=7,color='0.5')
g.plot_nodes(labeler=lambda n,rec: str(n))

plot_dt(cdt.DT,ax=fig.gca())

# traverse constrained edges from vh[3], trying to get
# to vh[5]. Anonymous vertices mean we had intersecting constraints,
# any other vertices mean collinear nodes.
# the trick is figuring out how to traverse

## 


# stepping along from vh_trav
def traverse_fresh_edge(self,a,b):
    """
    invoke after adding a constraint a--b, but before
    it has been added to self.g.

    steps from node a to node b, finding out what constrained
    edges actually were added.
    returns a list of tuples, each tuple is (node, vertex_handle)
    """

    vh_trav=self.g.nodes['vh'][a]
    vh_list=[ (vh_trav,a) ]

    while 1:
        # fetch list of constraints for traversal vertex handle
        constraints=[]
        self.DT.incident_constraints(vh_trav,constraints)

        n_trav=self.vh_info[vh_trav]
        # which neighbors are already constrained
        trav_nbrs_to_ignore = list(self.g.node_to_nodes(n_trav))

        maybe_next=[] # potential vh for next step

        for edge in constraints:
            face=edge[0] # work around missing API
            idx=edge[1]

            v1=face.vertex( (idx+1)%3 )
            v2=face.vertex( (idx+2)%3 )

            if v1==vh_trav:
                vh_other=v2
            else:
                vh_other=v1

            n_me = self.vh_info[vh_trav]
            n_other=self.vh_info[vh_other]

            print "Checking constrained edge %s -- %s."%(n_me,n_other)

            # If this is an existing neighbor in some other direction, ignore.

            if n_other!=b and n_other in trav_nbrs_to_ignore:
                print "   [skip]"
                continue
            if len(vh_list)>1 and vh_other==vh_list[-2][0]:
                print "   [backwards - skip]"
                continue

            maybe_next.append( (vh_other,n_other) )
        assert len(maybe_next)==1

        vh_trav,n_trav=maybe_next[0]
        vh_list.append( maybe_next[0] )

        print "Stepped to node: %s"%( n_trav )

        if n_trav==b:
            print "Done with traversal"
            break
    return vh_list

# This works okay for collinear.
elts=traverse_fresh_edge(cdt,a,b)

# what are the options?
#   a. bring in the big guns like in live_dt.py
#   b. ignore this, and move on to quad algorithm
#   c. insert the edge, and if it doesn't come back
#      looking right, rewind manually.  Hmm - it's harder
#      than one might think to see what happened.


# for intersecting constraints, there are more issues.
#   at an anonymous vertex, none of the edges are in
#   the grid, and we are confronted with the correct
#   edge, and the two "halves" of the severed edge

# what about line_walk, but add the logic to understand going
# through a node?
# that seems like way less machinery than live_dt.
