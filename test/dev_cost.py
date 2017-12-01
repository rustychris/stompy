import os
import math
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

def circumcenter_py(p1,p2,p3):
    """ no longer vectorized
    """
    ref = p1
    
    p1x = 0
    p1y = 0
    p2x = p2[0] - ref[0]
    p2y = p2[1] - ref[1]
    p3x = p3[0] - ref[0]
    p3y = p3[1] - ref[1]

    # taken from TRANSFORMER_gang.f90
    dd=2.0*((p1x-p2x)*(p1y-p3y) -(p1x-p3x)*(p1y-p2y))
    b_com=p1x*p1x+p1y*p1y
    b1=b_com-p2x*p2x-p2y*p2y
    b2=b_com-p3x*p3x-p3y*p3y 

    return [ (b1*(p1y-p3y)-b2*(p1y-p2y))/dd + ref[0] ,
             (b2*(p1x-p2x)-b1*(p1x-p3x))/dd + ref[1] ]



def gen_circum_partial(p1,p2):
    # return a function which takes a single point which completes
    # all triangles described by the array of vertices in p1 and p2,
    # and returns respective circumcenters.
    # furthermore, p3 is taken as a single point, while p1 and
    # p2 are arrays of points
    ref = p1
    p1x = 0.0
    p1y = 0.0
    p2x = p2[:,0] - ref[:,0]
    p2y = p2[:,1] - ref[:,1]

    p12dx=p1x-p2x
    p12dy=p1y-p2y

    dd0=2*p12dx*p1y - 2*p12dy*p1x 
    dd_f_x=  2*p12dy
    dd_f_y= -2*p12dx
    b1=p1x**2+p1y**2-p2x**2-p2y**2
    b20=p1x**2+p1y**2
    
    def partial(p3):
        p3x = p3[0] - ref[:,0]
        p3y = p3[1] - ref[:,1]

        vc = np.zeros( p1.shape, np.float64)

        # taken from TRANSFORMER_gang.f90
        # original:
        # dd=2.0*(p12dx*(p1y-p3y) - p12dy*(p1x-p3x))
        # distributed:
        # dd=2*p12dx*p1y - 2*p12dx*p3y - 2*p12dy*p1x + 2*p12dy*p3x
        dd=dd0 + dd_f_x*p3x + dd_f_y*p3y
        
        # b2=p1x**2+p1y**2-p3x**2-p3y**2
        b2=b20-p3x*p3x-p3y*p3y
        
        vc[:,0]=(b1*(p1y-p3y)-b2*(p1y-p2y))/dd + ref[:,0]
        vc[:,1]=(b2*(p1x-p2x)-b1*(p1x-p3x))/dd + ref[:,1]

        return vc
    return partial


##

class TestCC(front.AdvancingTriangles):
    """ Testing other options for node cost
    """
    def cost_function(self,n,method='cc_py'):
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

        circum_partial=gen_circum_partial( edge_points[:,0,:],edge_points[:,1,:] )

        Alist=[ [ e[0],e[1] ]
                for e in edge_points[:,0,:] ]
        Blist=[ [ e[0],e[1] ]
                for e in edge_points[:,1,:] ]
        EPS=1e-5*local_length

        def cost_cc_and_scale(x0):
            # tri_cc = circumcenter( edge_points[:,0,:],edge_points[:,1,:],x0[None,:] )
            # tri_cc = circum_partial(x0)
            x0l=list(x0)
            tri_cc=np.array( [ circumcenter_py(A,B,x0l)
                               for A,B in zip(Alist,Blist)] )

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

            cc_cost = ( (1./leftAB.clip(EPS,np.inf)).sum() +
                        (1./leftBC.clip(EPS,np.inf)).sum() +
                        (1./leftCA.clip(EPS,np.inf)).sum() )
            # non-dimensionalize
            cc_cost *= local_length

            scale_cost=(magABs-local_length)**2 + (magBCs-local_length)**2 + (magCAs-local_length)**2
            scale_cost=scale_cost.sum() / (local_length*local_length)

            return 0.1*cc_cost+scale_cost

        def cost_cc_and_scale_py(x0):
            C=list(x0)
            cc_cost=0
            scale_cost=0
            
            for A,B in zip(Alist,Blist):
                tri_cc=circumcenter_py(A,B,C)

                #As=edge_points[:,0,:]
                #Bs=edge_points[:,1,:]
                #Cs=x0[None,:]

                deltaAB=[ tri_cc[0] - A[0],
                          tri_cc[1] - A[1]]
                ABs=[B[0]-A[0],B[1]-A[1]]
                magABs=math.sqrt( ABs[0]*ABs[0] + ABs[1]*ABs[1])
                vecAB=[ABs[0]/magABs, ABs[1]/magABs]
                leftAB=vecAB[0]*deltaAB[1] - vecAB[1]*deltaAB[0] 

                deltaBC=[tri_cc[0] - B[0],
                         tri_cc[1] - B[1]]
                BCs=[C[0]-B[0], C[1]-B[1]]
                magBCs=math.sqrt( BCs[0]*BCs[0] + BCs[1]*BCs[1] )
                vecBC=[BCs[0]/magBCs, BCs[1]/magBCs]
                leftBC=vecBC[0]*deltaBC[1] - vecBC[1]*deltaBC[0]

                deltaCA=[tri_cc[0] - C[0],
                         tri_cc[1] - C[1]]
                CAs=[A[0]-C[0],A[1]-C[1]]
                magCAs=math.sqrt(CAs[0]*CAs[0] + CAs[1]*CAs[1])
                vecCA=[CAs[0]/magCAs, CAs[1]/magCAs]
                leftCA=vecCA[0]*deltaCA[1] - vecCA[1]*deltaCA[0]

                # reciprocal means that we have keep it strictly
                # positive, which in turn means that the optimization
                # can't escape a case where the cc is outside the cell
                # cc_cost += 0.2*local_length*( 1./max(EPS,leftAB) +
                #                           1./max(EPS,leftBC) +
                #                           1./max(EPS,leftCA) )
                # cc_fac=-4. # not bad
                cc_fac=-2.
                cc_cost += ( math.exp(cc_fac*leftAB/local_length) +
                             math.exp(cc_fac*leftBC/local_length) +
                             math.exp(cc_fac*leftCA/local_length) )
                

                scale_cost+=(magABs-local_length)**2 + (magBCs-local_length)**2 + (magCAs-local_length)**2

            scale_cost /= local_length*local_length
            return cc_cost+scale_cost

        if method=='base':
            return cost
        elif method=='cc':
            return cost_cc_and_scale
        elif method=='cc_py':
            return cost_cc_and_scale_py
        else:
            assert False
##

# # Get a grid with some candidates for tuning
# # af=test_basic_setup(TestCC)
# af=test_basic_setup(front.AdvancingTriangles)
# t=time.time()
# af.loop()
# elapsed_base=time.time() - t
# # 8.5s for 172 cells.  20.1 cells/s

af2=test_basic_setup(TestCC)
t=time.time()
af2.loop()
elapsed_cc=time.time() - t

print "%.2f cells/s"%( af2.grid.Ncells() / elapsed_cc)
# 15.2s for 148 cells.  9.6 cells/s
# 14.2s with turning circumcenter into a partial
# Where is the time?
# 15.8s, and 10.3 is in cost_cc_and_scale
# 2.2s in partial(), vs. 3.3 in circumcenter vs 0.9 in cirumcenter_py
# how much faster would it be to use straight python
# about as fast as old code, but has a bug -- actually old code had a bug.
# with more intentional scaling of cc_cost - makes an acceptable grid
# 130 cells (method=cc), 11.4 cells/s
# pure python gets 20.5 cells/s

##

plt.figure(2).clf()
af2.grid.plot_edges(lw=1.0)
# af2.grid.plot_nodes(labeler=lambda i,r: str(i))

## 
# choose a free node with a few neighbors:
n=34

cost=af2.cost_function(n)
print "cc_py: ",cost(af2.grid.nodes['x'][n])

cost=af2.cost_function(n,method='cc')
print "cc: ",cost(af2.grid.nodes['x'][n])

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
