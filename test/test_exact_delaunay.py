from __future__ import print_function
import numpy as np

from stompy.grid import exact_delaunay
Triangulation=exact_delaunay.Triangulation
from stompy.spatial import robust_predicates

def test_find_intersected_elements():
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5] ]

    nA=dt.add_node( x=pnts[0] ) # This tests insert into empty
    dt.add_node( x=pnts[1] ) # adjacent_vertex
    dt.add_node( x=pnts[2] ) # adjacent_vertex
    dt.add_node( x=pnts[3] ) # adjacent_edge

    dt.add_node( x=[3,0] ) # colinear

    dt.add_node( x=[6,2] ) # into cell interior
    nB=dt.add_node( x=[12,4] ) # collinear cell interior
    
    nodes=list(dt.valid_node_iter())

    for iA in range(len(nodes)):
        for iB in range(iA+1,len(nodes)):
            nA=nodes[iA]
            nB=nodes[iB]
            fwd=dt.find_intersected_elements(nA,nB)
            rev=dt.find_intersected_elements(nB,nA)
            assert len(fwd) == len(rev)



def test_adjacent_nodes_dim1():
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0] ]
    for pnt in pnts:
        dt.add_node( x=pnt )

    assert np.all( dt.topo_sort_adjacent_nodes(1,ref_nbr=0)==[0,2] )
    
def test_find_int_elts_dim1():
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0] ]
    for pnt in pnts:
        dt.add_node( x=pnt )

    assert len(dt.find_intersected_elements(0,1))==2
    assert len(dt.find_intersected_elements(0,2))==3
    assert len(dt.find_intersected_elements(1,2))==2

def test_add_constraint():
    def init():
        # inserting a constraint
        dt = Triangulation()
        pnts = [ [0,0],
                 [5,0],
                 [10,0],
                 [5,5],
                 [3,0],
                 [6,2],
                 [12,4]]
        for pnt in pnts:
            dt.add_node( x=pnt )
        return dt

    dt=init()
    dt.add_constraint(0,5)
    dt.add_constraint(3,2)
    try:
        dt.add_constraint(4,6)
        assert False
    except dt.IntersectingConstraints:
        pass # proper

    # adding a constraint to an edge which already exists
    assert dt.nodes_to_edge([0,4]) is not None
    dt.add_constraint(0,4)

    dt=init()
    try:
        dt.add_constraint(0,6)
        assert False
    except dt.ConstraintCollinearNode:
        pass

    dt.add_constraint(1,3)

    dt=init()
    dt.add_constraint(4,6)
    dt.add_node(x=[7,0.5])


def test_remove_constraint():
    def init():
        # inserting a constraint
        dt = Triangulation()
        pnts = [ [0,0],
                 [5,0],
                 [10,0],
                 [5,5],
                 [3,0],
                 [6,2],
                 [12,4]]
        for pnt in pnts:
            dt.add_node( x=pnt )
        return dt

    dt=init()
    dt.add_constraint(4,6)
    dt.add_node(x=[7,0.5])

    dt.remove_constraint(4,6)
    dt.add_constraint(2,3)
    dt.add_constraint(4,7)
    dt.remove_constraint(2,3)
    dt.remove_constraint(7,4)
    
    dt.check_global_delaunay()

def test_constraints_dim1():
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0] ]
    for pnt in pnts:
        dt.add_node( x=pnt )

    dt.add_constraint(0,1)
    dt.add_constraint(1,2)
    dt.remove_constraint(0,1)
    dt.remove_constraint(1,2)
    try:
        dt.add_constraint(0,2)
        assert False
    except dt.ConstraintCollinearNode:
        pass # 


# # Testing the atomic nature of modify_node()

def test_atomic_move():
    """ Make sure that when a modify_node call tries an
    illegal move of a node with a constraint, the DT state
    is restored to the original state before raising the exception
    """
    def init():
        # inserting a constraint
        dt = Triangulation()
        pnts = [ [0,0],
                 [5,0],
                 [10,0],
                 [5,5],
                 [3,0],
                 [6,2],
                 [12,4]]
        for pnt in pnts:
            dt.add_node( x=pnt )
        return dt

    dt=init()
    dt.add_constraint(0,5)
    dt.add_constraint(3,2)

    assert np.all( dt.nodes['x'][5]==[6,2] )

    try:
        dt.modify_node(5,x=[8,3])
        assert False # it should raise the exception
    except dt.IntersectingConstraints:
        # And the nodes/constraints should be where they started.
        assert np.all( dt.nodes['x'][5]==[6,2] )

    if 0:
        plt.clf()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,values=dt.edges['constrained'],lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)


## 
#-# Building up some basic tests:
def test_basic1():
    plot=False
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5] ]

    dt.add_node( x=pnts[0] ) # This tests insert into empty
    dt.add_node( x=pnts[1] ) # adjacent_vertex
    dt.add_node( x=pnts[2] ) # adjacent_vertex
    dt.add_node( x=pnts[3] ) # adjacent_edge

    dt.add_node( x=[3,0] ) # colinear

    if plot:
        plt.clf()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)
        # This where topo sort is failing
        plt.pause(0.1)
    n=dt.add_node( x=[6,2] ) # into cell interior
    if plot:
        plt.clf()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)


def test_flip1():
    plot=False
    dt = Triangulation()
    pnts = [ [0,0],
             [8,0],
             [10,5],
             [5,5] ]

    dt.add_node( x=pnts[0] ) # This tests insert into empty
    dt.add_node( x=pnts[1] ) # adjacent_vertex
    dt.add_node( x=pnts[2] ) # adjacent_vertex
    dt.add_node( x=pnts[3] ) # adjacent_edge

    dt.add_node( x=[3,0] ) # colinear
    if plot:
        plt.clf()
        dt.plot_nodes()
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)
        

def test_flip2():
    plot=False
    dt = Triangulation()

    dt.add_node( x=[0,0] )
    for i in range(5):
        dt.add_node( x=[10,i] )
    # This one requires a flip:
    n=dt.add_node( x=[5,1] )
    
    if plot:
        plt.cla()
        dt.plot_nodes()
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)


def test_incircle():
    A=[0,0]
    B=[1,0]
    C=[1,1]
    Dout=[2,0]
    Don =[0,1]
    Din =[0.5,0.5]

    assert robust_predicates.incircle(A,B,C,Dout)<0
    assert robust_predicates.incircle(A,B,C,Don) == 0
    assert robust_predicates.incircle(A,B,C,Din) >0

# testing dim_down
def test_test_dim_down():
    dt = Triangulation()

    n=dt.add_node( x=[0,0] )

    for i in range(5):
        other=dt.add_node( x=[10,i] )

    assert dt.test_delete_node_dim_down(n)
    assert not dt.test_delete_node_dim_down(other)

    friend=dt.add_node( x=[0,2])
    assert not dt.test_delete_node_dim_down(friend)
    assert not dt.test_delete_node_dim_down(n)


## 
# developing deletion -

def test_delete1():
    plot=False
    dt = Triangulation() # ExactDelaunay()

    dt.add_node( x=[0,0] )
    for i in range(5):
        dt.add_node( x=[10,i] )
    # This one requires a flip:
    n=dt.add_node( x=[5,1] )
    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nr:str(n))
        dt.plot_edges(alpha=0.5,lw=2)

        bad=(~dt.cells['deleted'])&(dt.cells_area()<=0)
        good=(~dt.cells['deleted'])&(dt.cells_area()>0)
        dt.plot_cells(lw=8,facecolor='#ddddff',edgecolor='w',zorder=-5,mask=good)
        dt.plot_cells(lw=8,facecolor='#ffdddd',edgecolor='r',zorder=-5,mask=bad)

    # deleting n work okay
    dt.delete_node(n)

    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nr:str(n))
        dt.plot_edges(alpha=0.5,lw=2)

        bad=(~dt.cells['deleted'])&(dt.cells_area()<=0)
        good=(~dt.cells['deleted'])&(dt.cells_area()>0)
        dt.plot_cells(lw=8,facecolor='#ddddff',edgecolor='w',zorder=-5,mask=good)
        dt.plot_cells(lw=8,facecolor='#ffdddd',edgecolor='r',zorder=-5,mask=bad)

    # delete a node which defines part of the convex hull:
    dt.delete_node(5) # works, though may need to patch up edges['cells'] ?
    dt.delete_node(2) # collinear portion of the convex hull
    dt.delete_node(3)

    # everything looks okay.  Surprising that we didn't have to
    # add code to deal with this case...

    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nr:str(n))
        dt.plot_edges(alpha=0.5,lw=2)

        bad=(~dt.cells['deleted'])&(dt.cells_area()<=0)
        good=(~dt.cells['deleted'])&(dt.cells_area()>0)
        dt.plot_cells(lw=8,facecolor='#ddddff',edgecolor='w',zorder=-5,mask=good)
        dt.plot_cells(lw=8,facecolor='#ffdddd',edgecolor='r',zorder=-5,mask=bad)

    dt.delete_node(0)

    if plot:
        assert dt.dim() == 1

        plt.cla()
        dt.plot_nodes(labeler=lambda n,nr:str(n))
        dt.plot_edges(alpha=0.5,lw=2)

        bad=(~dt.cells['deleted'])&(dt.cells_area()<=0)
        good=(~dt.cells['deleted'])&(dt.cells_area()>0)
        dt.plot_cells(lw=8,facecolor='#ddddff',edgecolor='w',zorder=-5,mask=good)
        dt.plot_cells(lw=8,facecolor='#ffdddd',edgecolor='r',zorder=-5,mask=bad)
        

##  

# slightly more involved test for deletion
def test_delete2():
    plot=False
    dt = Triangulation() # ExactDelaunay()

    dt.add_node( x=[0,0] )
    for i in range(5):
        if 0: # sparser nodes for testing
            if i>0 and i<4:
                continue
        dt.add_node( x=[10,i] )
    # This one requires a flip:
    n=dt.add_node( x=[5,1] )
    
    far=dt.add_node( x=[12,4] ) 

    dt.delete_node(0) 
    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nr:str(n))
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=6,facecolor='#ddddff',edgecolor='w',zorder=-5)

def test_delete3():
    plot=False
    dt = Triangulation()

    nodes=[ dt.add_node( x=[10,i] )
            for i in range(10) ]

    dt.delete_node( nodes[0])
    dt.delete_node( nodes[4])
    if plot:
        plt.cla()
        dt.plot_nodes()
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)


## 

def test_flip2():
    plot=False
    # testing Delaunay flipping
    dt = Triangulation() # ExactDelaunay()

    dt.add_node( x=[0,0] )
    for i in range(10):
        dt.add_node( x=[10,i] )

    dt.add_node( x=[5,1.0] ) 

    dt.add_node( x=[10,10] ) 
    dt.add_node( x=[5,2] )
    
    # and test a flip when the new vertex is outside the convex hull
    dt.add_node( x=[5,-1])

    dt.delete_node(11)

    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)

# Test the case where the temporary triangulation is not necessarily planar
def test_delete4():
    plot=False
    dt = Triangulation()

    dt.add_node( x=[0,0] )
    dt.add_node( x=[5,0] )
    dt.add_node( x=[10,0] )
    dt.add_node( x=[-2,3] )
    dt.add_node( x=[-2,-3] )

    # and now delete the one in the middle:
    dt.delete_node(1)

    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=7,facecolor='#ddddff',edgecolor='w',zorder=-5)

#  low-dim cases

def test_lowdim1():
    plot=False
    dt = Triangulation()

    dt.add_node( x=[0,0] )
    dt.add_node( x=[5,0] )
    dt.add_node( x=[10,0] )
    dt.add_node( x=[-2,3] )
    dt.add_node( x=[-2,-3] )
    
    # and now delete the one in the middle:
    dt.delete_node(1)
    dt.delete_node(0)
    dt.delete_node(2)
    dt.delete_node(3)
    dt.delete_node(4)
    
    if plot:
        plt.cla()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=7,facecolor='#ddddff',edgecolor='w',zorder=-5)
        


def test_fuzz1():
    plot=False
    # Fuzzing, regular
    x=np.arange(5)
    y=np.arange(5)

    X,Y=np.meshgrid(x,y)
    xys=np.array([X.ravel(),Y.ravel()]).T

    if plot:
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        ax.plot(xys[:,0],xys[:,1],'go',alpha=0.4)
        ax.axis([-1,5,-1,5])

    idxs=np.zeros(len(xys),'i8')-1

    dt = Triangulation()

    # definitely slows down as the number of nodes gets larger.
    # starting off with <1s per 100 operations, later more like 2s
    for repeat in range(1):
        print("Repeat: ",repeat)
        for step in range(1000):
            if step%200==0:
                print("  step: ",step)
            toggle=np.random.randint(len(idxs))
            if idxs[toggle]<0:
                idxs[toggle] = dt.add_node( x=xys[toggle] )
            else:
                dt.delete_node(idxs[toggle])
                idxs[toggle]=-1

            if plot:
                del ax.lines[1:]
                ax.texts=[]
                ax.collections=[]
                dt.plot_nodes(labeler=lambda n,nrec: str(n) )
                dt.plot_edges(alpha=0.5,lw=2)
                dt.plot_cells(lw=7,facecolor='#ddddff',edgecolor='w',zorder=-5)
                plt.draw()


def test_extra1():
    plot=False
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5] ]

    dt.add_node( x=pnts[0] ) # This tests insert into empty
    dt.add_node( x=pnts[1] ) # adjacent_vertex
    dt.add_node( x=pnts[2] ) # adjacent_vertex
    dt.add_node( x=pnts[3] ) # adjacent_edge

    dt.add_node( x=[3,0] ) # colinear

    if plot:
        plt.clf()
        dt.plot_nodes()
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)

    n=dt.add_node( x=[6,2] ) # into cell interior

    if plot:
        plt.clf()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)

def test_move1():
    plot=False
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5] ]

    dt.add_node( x=pnts[0] ) # This tests insert into empty
    dt.add_node( x=pnts[1] ) # adjacent_vertex
    dt.add_node( x=pnts[2] ) # adjacent_vertex
    dt.add_node( x=pnts[3] ) # adjacent_edge

    dt.add_node( x=[3,0] ) # colinear

    if plot:
        plt.clf()
        dt.plot_nodes()
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)

    n=dt.add_node( x=[6,2] ) # into cell interior

    dt.modify_node(4,x=[3.01,0.25])

    dt.modify_node(4,x=[3.01,-0.25])
    if plot:
        plt.clf()
        dt.plot_nodes(labeler=lambda n,nrec: str(n) )
        dt.plot_edges(alpha=0.5,lw=2)
        dt.plot_cells(lw=13,facecolor='#ddddff',edgecolor='w',zorder=-5)

    dt.check_global_delaunay()
    return dt

## 
        
if 0:
    test_find_intersected_elements()
    test_adjacent_nodes_dim1()
    test_find_int_elts_dim1()
    test_add_constraint()    
    test_remove_constraint()
    test_constraints_dim1()
    test_basic1()
    test_flip1()    
    test_flip2()
    test_incircle()
    test_delete1()
    test_delete2()
    test_flip2()        
    test_delete4()
    test_lowdim1()
    test_fuzz1()
    test_extra1()

    
