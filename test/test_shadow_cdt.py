import numpy as np

import nose2

from stompy.grid import exact_delaunay, unstructured_grid, shadow_cdt

reload(shadow_cdt)

## 
def run_node_insertion(cdt_class):
    g = unstructured_grid.UnstructuredGrid()

    cdt=cdt_class(g)
    
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5] ]

    nA=g.add_node( x=pnts[0] ) # This tests insert into empty
    g.add_node( x=pnts[1] ) # adjacent_vertex
    g.add_node( x=pnts[2] ) # adjacent_vertex
    g.add_node( x=pnts[3] ) # adjacent_edge

    g.add_node( x=[3,0] ) # colinear

    g.add_node( x=[6,2] ) # into cell interior
    nB=g.add_node( x=[12,4] ) # collinear cell interior
    return g,cdt # helps with debugging

def test_node_insertion():
    run_node_insertion(shadow_cdt.ShadowCDT)
def test_node_insertion_cgal():
    run_node_insertion(shadow_cdt.ShadowCGALCDT)
    
test_node_insertion_cgal()

##

def run_add_constraint(cdt_class):
    def init():
        # inserting a constraint
        dt = unstructured_grid.UnstructuredGrid()
        cdt=cdt_class(dt)
        pnts = [ [0,0],
                 [5,0],
                 [10,0],
                 [5,5],
                 [3,0],
                 [6,2],
                 [12,4]]
        for pnt in pnts:
            dt.add_node( x=pnt )
        return dt,cdt

    dt,cdt=init()
    dt.add_edge(nodes=[0,5])
    dt.add_edge(nodes=[3,2])
    try:
        dt.add_edge(nodes=[4,6])
        assert False
    except cdt.IntersectingConstraints:
        pass # proper

    dt,cdt=init()
    try:
        dt.add_edge(nodes=[0,6])
        assert False
    except cdt.ConstraintCollinearNode:
        pass

    dt.add_edge(nodes=[1,3])

    dt,cdt=init()
    dt.add_edge(nodes=[4,6])
    dt.add_node(x=[7,0.5])

def test_add_edge():
    run_add_constraint(shadow_cdt.ShadowCDT)
    
def test_add_edge_cgal():
    run_add_constraint(shadow_cdt.ShadowCGALCDT)

test_add_edge_cgal()
##     

def test_remove_constraint():
    def init():
        # inserting a constraint
        dt = exact_delaunay.Triangulation()
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

def test_remove_constraint_cgal():
    g=unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)
    
    # inserting a constraint
    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5],
             [3,0],
             [6,2],
             [12,4]]
    nodes=[g.add_node( x=pnt )
           for pnt in pnts]

    j0=g.add_edge(nodes=[nodes[4],nodes[6]])
    nodes.append( g.add_node(x=[7,0.5]) )

    g.delete_edge(j0)
    j1=g.add_edge(nodes=[nodes[2],nodes[3]])
    j2=g.add_edge(nodes=[nodes[4],nodes[7]])
    g.delete_edge(j1)
    g.delete_edge(j2)
    
test_remove_constraint_cgal()

## 

def test_constraints_dim1():
    dt = exact_delaunay.Triangulation()
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

def test_constraints_dim1_cgal():
    g = unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)
    pnts = [ [0,0],
             [5,0],
             [10,0] ]
    nodes=[g.add_node( x=pnt )
           for pnt in pnts]

    j0=g.add_edge(nodes=[nodes[0],nodes[1]])
    j1=g.add_edge(nodes=[nodes[1],nodes[2]])
    g.delete_edge(j0)
    g.delete_edge(j1)
    try:
        g.add_edge(nodes=[nodes[0],nodes[2]])
        assert False
    except cdt.ConstraintCollinearNode:
        pass # 
    
test_constraints_dim1_cgal()

## 
# # Testing the atomic nature of modify_node()

def test_atomic_move():
    """ Make sure that when a modify_node call tries an
    illegal move of a node with a constraint, the DT state
    is restored to the original state before raising the exception
    """
    def init():
        # inserting a constraint
        dt = exact_delaunay.Triangulation()
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


def test_atomic_move_cgal():
    """ Make sure that when a modify_node call tries an
    illegal move of a node with a constraint, the DT state
    is restored to the original state before raising the exception
    """
    # inserting a constraint
    g = unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)

    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5],
             [3,0],
             [6,2],
             [12,4]]
    nodes=[g.add_node( x=pnt )
           for pnt in pnts]

    j0=g.add_edge(nodes=[nodes[0],nodes[5]])
    j1=g.add_edge(nodes=[nodes[3],nodes[2]])

    assert np.all( g.nodes['x'][5]==[6,2] )

    try:
        g.modify_node(nodes[5],x=[8,3])
        assert False # it should raise the exception
    except cdt.IntersectingConstraints:
        # And the nodes/constraints should be where they started.
        assert np.all( g.nodes['x'][5]==[6,2] )

test_atomic_move_cgal()

## 
#-# Building up some basic tests:
def test_basic1():
    plot=False
    dt = exact_delaunay.Triangulation()
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


def test_basic1_cgal():
    g = unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)

    pnts = [ [0,0],
             [5,0],
             [10,0],
             [5,5],
             [3,0],
             [6,2]]
    nodes=[g.add_node(x=pnt) for pnt in pnts]

test_basic1_cgal()

def test_flip1():
    plot=False
    dt = exact_delaunay.Triangulation()
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

def test_flip1_cgal():
    plot=False
    g = unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)
    
    pnts = [ [0,0],
             [8,0],
             [10,5],
             [5,5],
             [3,0]]

    [g.add_node(x=pnt) for pnt in pnts]

test_flip1_cgal()    

def test_flip2():
    plot=False
    dt = exact_delaunay.Triangulation()

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
    dt = exact_delaunay.Triangulation()

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
    dt = exact_delaunay.Triangulation() # ExactDelaunay()

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
    dt = exact_delaunay.Triangulation() # ExactDelaunay()

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
    dt = exact_delaunay.Triangulation()

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
    dt = exact_delaunay.Triangulation() # ExactDelaunay()

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
    dt = exact_delaunay.Triangulation()

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
    dt = exact_delaunay.Triangulation()

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

    dt = exact_delaunay.Triangulation()

    # definitely slows down as the number of nodes gets larger.
    # starting off with <1s per 100 operations, later more like 2s
    for repeat in range(1):
        print "Repeat: ",repeat
        for step in range(1000):
            if step%200==0:
                print "  step: ",step
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

def test_fuzz1_cgal():
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

    g = unstructured_grid.UnstructuredGrid()
    cdt=shadow_cdt.ShadowCGALCDT(g)

    # definitely slows down as the number of nodes gets larger.
    # starting off with <1s per 100 operations, later more like 2s
    for repeat in range(1):
        print "Repeat: ",repeat
        for step in range(1000):
            if step%200==0:
                print "  step: ",step
            toggle=np.random.randint(len(idxs))
            if idxs[toggle]<0:
                idxs[toggle] = g.add_node( x=xys[toggle] )
            else:
                g.delete_node(idxs[toggle])
                idxs[toggle]=-1

            if plot:
                del ax.lines[1:]
                ax.texts=[]
                ax.collections=[]
                dt.plot_nodes(labeler=lambda n,nrec: str(n) )
                dt.plot_edges(alpha=0.5,lw=2)
                dt.plot_cells(lw=7,facecolor='#ddddff',edgecolor='w',zorder=-5)
                plt.draw()


test_fuzz1_cgal()
                
def test_extra1():
    plot=False
    dt = exact_delaunay.Triangulation()
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
    dt = exact_delaunay.Triangulation()
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

    
