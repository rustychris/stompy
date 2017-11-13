import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb

from scipy import optimize as opt

from stompy.spatial import field
from stompy import utils

from stompy.grid import (unstructured_grid, exact_delaunay, front)

import logging
logging.basicConfig(level=logging.INFO)

from stompy.spatial.linestring_utils import upsample_linearring,resample_linearring
from stompy.spatial import field,constrained_delaunay,wkb2shp

## Curve -

def hex_curve():
    hexagon = np.array( [[0,11],
                         [10,0],
                         [30,0],
                         [40,9],
                         [30,20],
                         [10,20]] )
    return front.Curve(hexagon)

def test_curve_eval():
    crv=hex_curve()
    f=np.linspace(0,2*crv.total_distance(),25)
    crvX=crv(f)
    
    if 0: # skip plots
        plt.clf()
        crv.plot()

        f=np.linspace(0,crv.total_distance(),25)
        crvX=crv(f)
        plt.plot(crvX[:,0],crvX[:,1],'ro')

def test_distance_away():
    crv=hex_curve()

    if 0: # skip plots
        plt.clf()
        crv.plot()
        plt.axis('equal')
        
    rtol=0.05

    for f00,tgt,style in [ (0,10,'g-'),
                           (3.4,20,'r-'),
                           (3.4,-20,'r--') ]:
        for f0 in np.linspace(f00,crv.distances[-1],20):
            x0=crv(f0)
            f,x =crv.distance_away(f0,tgt,rtol=rtol)
            d=utils.dist(x-x0)
            assert np.abs( (d-np.abs(tgt))/tgt) <= rtol
            if 0:
                plt.plot( [x0[0],x[0]],
                          [x0[1],x[1]],style)

    try:
        f,x=crv.distance_away(0.0,50,rtol=0.05)
        raise Exception("That was supposed to fail!")
    except crv.CurveException:
        #print "Okay"
        pass


def test_distance_away2():
    # Towards a smarter Curve::distance_away(), which understands
    # piecewise linear geometry
    island  =np.array([[200,200],[600,200],[200,600]])
    curve=front.Curve(island)

    anchor_f=919.3
    signed_distance=50.0
    res=curve.distance_away(anchor_f,signed_distance)
    assert res[0]>anchor_f
    anchor_pnt=curve(anchor_f)

    rel_err=np.abs( utils.dist(anchor_pnt - res[1]) - abs(signed_distance)) / abs(signed_distance)
    assert np.abs(rel_err)<=0.05

    anchor_f=440
    signed_distance=-50.0
    res=curve.distance_away(anchor_f,signed_distance)

    anchor_pnt=curve(anchor_f)

    rel_err=np.abs( utils.dist(anchor_pnt - res[1]) - abs(signed_distance)) / abs(signed_distance)
    assert res[0]<anchor_f
    assert np.abs(rel_err)<=0.05
    
def test_distance3():
    # Case where the return point is on the same segment as it starts
    curve=front.Curve(np.array([[   0,    0],
                                [1000,    0],
                                [1000, 1000],
                                [   0, 1000]]),closed=True)
    res=curve.distance_away(3308.90,50.0)
    res=curve.distance_away(3308.90,-50.0)
    
def test_is_forward():
    crv=hex_curve()
    assert crv.is_forward(5,6,50)
    assert crv.is_reverse(5,-5,10)


## 
def test_curve_upsample():
    boundary=hex_curve()
    scale=field.ConstantField(3)

    pnts,dists = boundary.upsample(scale,return_sources=True)

    if 0:
        plt.clf()
        line=boundary.plot()
        plt.setp(line,lw=0.5,color='0.5')

        #f=np.linspace(0,crv.total_distance(),25)
        #crvX=crv(f)
        plt.scatter(pnts[:,0],pnts[:,1],30,dists,lw=0)
    
def test_basic_setup():
    boundary=hex_curve()
    af=front.AdvancingTriangles()
    scale=field.ConstantField(3)

    af.add_curve(boundary)
    af.set_edge_scale(scale)

    # create boundary edges based on scale and curves:
    af.initialize_boundaries()

    if 0:
        plt.clf()
        g=af.grid
        g.plot_edges()
        g.plot_nodes()

        # 
        coll=g.plot_halfedges(values=g.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')

    return af


# Going to try more of a half-edge approach, rather than explicitly
# tracking the unpaved rings.
# hoping that a half-edge interface is sufficient for the paver, and
# could be supported by multiple representations internally.

# for starters, don't worry about caching/speed/etc.
# okay to start from scratch each time.

# the product here is a list of the N best internal angles for
# filling with a triangle(s)

def test_halfedge_traverse():
    af=test_basic_setup()
    J,Orient = np.nonzero( (af.grid.edges['cells'][:,:]==af.grid.UNMESHED) )

    # he=he0=HalfEdge(af.grid,J[0],Orient[0])
    he=he0=af.grid.halfedge(J[0],Orient[0])

    for i in range(af.grid.Nedges()*2):
        he=he.fwd()
        if he == he0:
            break
    else:
        assert False
    assert i==31 # that had been 33, but now I'm getting 31.  may need to be smarter.

    he=he0=af.grid.halfedge(J[0],Orient[0])

    for i in range(af.grid.Nedges()*2):
        he=he.rev()
        if he == he0:
            break
    else:
        assert False
    assert i==31 # pretty sure about that number...

    assert he.fwd().rev() == he
    assert he.rev().fwd() == he
    #-# 

def test_free_span():
    r=5
    theta = np.linspace(-np.pi/2,np.pi/2,20)
    cap = r * np.swapaxes( np.array([np.cos(theta), np.sin(theta)]), 0,1)
    box = np.array([ [-3*r,r],
                     [-4*r,-r] ])
    ring = np.concatenate((box,cap))

    density = field.ConstantField(2*r/(np.sqrt(3)/2))
    af=front.AdvancingTriangles()
    af.set_edge_scale(density)

    af.add_curve(ring,interior=False)
    af.initialize_boundaries()

    # N.B. this edge is not given proper cell neighbors
    af.grid.add_edge(nodes=[22,3])

    af.plot_summary()

    he=af.grid.nodes_to_halfedge(4,5)
    span_dist,span_nodes = af.free_span(he,25,1)
    assert span_nodes[-1]!=4

    
def test_merge_edges():
    af=test_basic_setup()

    new_j=af.grid.merge_edges(node=0)
    
    he0=he=af.grid.halfedge(new_j,0)
    c0_left = af.grid.edges['cells'][he.j,he.orient]
    c0_right = af.grid.edges['cells'][he.j,1-he.orient]

    while True:
        he=he.fwd()
        c_left = af.grid.edges['cells'][he.j,he.orient]
        c_right = af.grid.edges['cells'][he.j,1-he.orient]
        assert c_left==c0_left
        assert c_right==c0_right
        
        if he==he0:
            break

    if 0:
        plt.clf()
        af.grid.plot_edges()

        coll=af.grid.plot_halfedges(values=af.grid.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')

# when resample nodes on a sliding boundary, want to calculate the available
# span, and if it's small, start distributing the nodes evenly.
# where small is defined by local_scale * max_span_factor

def test_resample():
    af=test_basic_setup()
    a=0
    b=af.grid.node_to_nodes(a)[0]
    he=af.grid.nodes_to_halfedge(a,b)
    anchor=he.node_rev()
    n=he.node_fwd()
    n2=he.rev().node_rev()
    # Fails here, in grid.modify_node
    af.resample(n=n,anchor=anchor,scale=25,direction=1)
    af.resample(n=n2,anchor=anchor,scale=25,direction=-1)
    
test_resample()


# during modify_node(n=9)
# 9 comes in as node b in call to line_is_free
#   vertex handle gives it as 22.5,0.0, which is the new location
# lw from line_walk is bad.
# after_add_node() just inserts the new point into the DT.
#   - could be related to premature garbage collection of points?
#     nope.
#   - related to init_face? has to be there for proper functioning
#   - or failure to remove the original vertex before creating the new one?
#     no, that seems to be taken care of.

# does a line free call work before modifying the node?
#  nope. So maybe something else in the early part of before_modify_node
#  invalidates the state?
# it's the second time through the loop that fails?
# 10--9 crashes, even when it's the first in the loop
