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
    
def test_basic_setup(klass):
    boundary=hex_curve()
    af=klass() # front.AdvancingTriangles()
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


def test_no_lookahead():
    af=test_basic_setup(front.AdvancingTriangles)
    af.log.setLevel(logging.INFO)
    af.cdt.post_check=False

    af.current=af.root=front.DTChooseSite(af)

    def cb():
        af.plot_summary(label_nodes=False)
        try:
            af.current.site.plot()
        except: # AttributeError:
            pass

    while 1:
        if not af.current.children:
            break # we're done?

        for child_i in range(len(af.current.children)):
            if af.current.try_child(child_i):
                # Accept the first child which returns true
                break
        else:
            assert False # none of the children worked out
    return af

##

# from numba import jit, int32, float64
# @jit(float64[:,:](float64[:,:],float64[:,:],float64[:,:]),nopython=True)
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

##

class TestCC(front.AdvancingTriangles):
    """ Testing other options for node cost
    """
    def cost_function(self,n):
        local_length = self.scale( self.grid.nodes['x'][n] )
        my_cells = self.grid.node_to_cells(n)

        if len(my_cells) == 0:
            return None

        cell_nodes = [self.grid.cell_to_nodes(c)
                      for c in my_cells ]

        # for the moment, can only deal with triangles
        cell_nodes=np.array(cell_nodes)

        # pack our neighbors from the cell list into an edge
        # list that respects the CCW condition that pnt must be on the
        # left of each segment
        for j in range(len(cell_nodes)):
            if cell_nodes[j,0] == n:
                cell_nodes[j,:2] = cell_nodes[j,1:]
            elif cell_nodes[j,1] == n:
                cell_nodes[j,1] = cell_nodes[j,0]
                cell_nodes[j,0] = cell_nodes[j,2] # otherwise, already set

        edges = cell_nodes[:,:2]
        edge_points = self.grid.nodes['x'][edges]

        def cost(x,edge_points=edge_points,local_length=local_length):
            return front.one_point_cost(x,edge_points,target_length=local_length)

        def cost_cc_and_scale(x0):
            tri_cc = circumcenter( edge_points[:,0,:],edge_points[:,1,:],x0[None,:] )

            As=edge_points[:,0,:]
            Bs=edge_points[:,1,:]
            Cs=x0[None,:]

            deltaAB=tri_cc[:,:] - As
            ABs=Bs-As
            magABs=utils.mag(ABs)
            vecAB=ABs/magABs[...,None]
            leftAB=vecAB[:,0]*deltaAB[:,1] - vecAB[:,1]*deltaAB[:,0]

            deltaBC=tri_cc[:,:] - Bs
            BCs=Cs-Bs
            magBCs=utils.mag(BCs)
            vecBC=BCs/magBCs[...,None]
            leftBC=vecBC[:,0]*deltaBC[:,1] - vecBC[:,1]*deltaBC[:,0]

            deltaCA=tri_cc[:,:] - Cs
            CAs=As-Cs
            magCAs=utils.mag(CAs)
            vecCA=CAs/magCAs[...,None]
            leftCA=vecCA[:,0]*deltaCA[:,1] - vecCA[:,1]*deltaCA[:,0]

            EPS=1e-5*local_length
            cc_cost = ( (1./leftAB.clip(EPS,np.inf)).sum() +
                        (1./leftBC.clip(EPS,np.inf)).sum() +
                        (1./leftCA.clip(EPS,np.inf)).sum() )

            scale_cost=(magABs-local_length)**2 + (magBCs-local_length)**2 + (magCAs-local_length)**2
            scale_cost=scale_cost.sum() / local_length*local_length

            return cc_cost+scale_cost
        # return cost
        return cost_cc_and_scale
##

# Get a grid with some candidates for tuning
# af=test_basic_setup(TestCC)
af=test_basic_setup(front.AdvancingTriangles)
t=time.time()
af.loop()
elapsed_base=time.time() - t
# 8.5s for 172 cells.  20.1 cells/s

## 
af2=test_basic_setup(TestCC)
t=time.time()
af2.loop()
elapsed_cc=time.time() - t

# 15.2s for 148 cells.  9.6 cells/s

## 
af.cdt.post_check=False

af.current=af.root=front.DTChooseSite(af)

for _ in range(400):
    if not af.current.children:
        break # we're done?

    for child_i in range(len(af.current.children)):
        if af.current.try_child(child_i):
            # Accept the first child which returns true
            break
    else:
        assert False # none of the children worked out

##

plt.figure(2).clf()
af.grid.plot_edges(lw=1.0)
# af.grid.plot_nodes(labeler=lambda i,r: str(i))

## 
# choose a free node with a few neighbors:
n=37

# def cost_function(self,n):
self=af
if 1:
    local_length = self.scale( self.grid.nodes['x'][n] )
    my_cells = self.grid.node_to_cells(n)

    assert len(my_cells) != 0

    cell_nodes = [self.grid.cell_to_nodes(c)
                  for c in my_cells ]

    # for the moment, can only deal with triangles
    cell_nodes=np.array(cell_nodes)

    # pack our neighbors from the cell list into an edge
    # list that respects the CCW condition that pnt must be on the
    # left of each segment
    for j in range(len(cell_nodes)):
        if cell_nodes[j,0] == n:
            cell_nodes[j,:2] = cell_nodes[j,1:]
        elif cell_nodes[j,1] == n:
            cell_nodes[j,1] = cell_nodes[j,0]
            cell_nodes[j,0] = cell_nodes[j,2] # otherwise, already set

    edges = cell_nodes[:,:2]
    edge_points = self.grid.nodes['x'][edges]

    def cost(x,edge_points=edge_points,local_length=local_length):
        return front.one_point_cost(x,edge_points,target_length=local_length)

## 

cost=af.cost_function(37)

cost(af.grid.nodes['x'][37])

## 
x0=af.grid.nodes['x'][n]

# with numba: 6.3us, without 95us.
cost(x0)

# Other thoughts on optimizing:
#   compute circumcenters, then distance from edges to circumcenters
#   maximize some combination of distances and minimize area?

##

# edge_points.shape # 4 triangles, 2 points each, 2 coords
#tri_points=np.concatenate( ( edge_points,
#                             np.zeros( (4,1,2) ) ), axis=1 )
# tri_points[:,2,:]=x0


# 80us per loop from utils
# 5us if jit'd

# 159us pure python.  113us with jit circumcenter
    
##

from scipy import optimize

x0_cc=optimize.fmin(cost_cc,x0)
x0_cc_s=optimize.fmin(cost_cc_and_scale,x0)
## 
plt.figure(1).clf()
af.grid.plot_edges()
af.grid.plot_nodes(labeler=lambda i,r: str(i))
# plt.plot(tri_cc[:,0],tri_cc[:,1],'go')
plt.plot(x0_cc[None,0],x0_cc[None,1],'ro')
plt.plot(x0_cc_s[None,0],x0_cc_s[None,1],'bo')

##

# Some of this is maybe doable in sympy --
# but it would get involved, and not yet clear that it's worth it.
from sympy.geometry import Point, Triangle, Segment, Line

from sympy import symbols
x, y = symbols('x y')

p1, p2, p3 = Point(0, 0), Point(1, 5), Point(x, y)
t = Triangle(p1, p2, p3)
cc=t.circumcenter

seg12=Line(p1,p2)
seg23=Line(p2,p3)
seg31=Line(p3,p1)

# These fail for segments, but okay with Lines
dist12=seg12.distance(cc)
dist23=seg23.distance(cc)
dist31=seg31.distance(cc)
