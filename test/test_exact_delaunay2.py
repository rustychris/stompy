from __future__ import print_function

import numpy as np

from stompy.grid import exact_delaunay, unstructured_grid

Triangulation=exact_delaunay.Triangulation
from stompy.spatial import robust_predicates

def test_gen_intersected_elements():
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

    for iA,nA in enumerate(nodes):
        for nB in nodes[iA+1:]:
            print("test_gen_intersected_elements: %s to %s"%(dt.nodes['x'][nA],
                                                             dt.nodes['x'][nB]))
            fwd=list(dt.gen_intersected_elements(nA=nA,nB=nB))
            rev=list(dt.gen_intersected_elements(nA=nB,nB=nA))
            assert len(fwd) == len(rev)
            
def test_gen_int_elts_dim1():
    dt = Triangulation()
    pnts = [ [0,0],
             [5,0],
             [10,0] ]
    for pnt in pnts:
        dt.add_node( x=pnt )

    assert len(list(dt.gen_intersected_elements(0,1)))==3
    assert len(list(dt.gen_intersected_elements(0,2)))==5
    assert len(list(dt.gen_intersected_elements(1,2)))==3

    # and with some points
    assert len(list(dt.gen_intersected_elements(pA=[-1,-1],
                                                pB=[-1,1])))==0
    elts=list(dt.gen_intersected_elements(pA=[0,-1],pB=[0,1]))
    assert len(elts)==1
    assert elts[0][0]=='node'
    
    elts=list(dt.gen_intersected_elements(pA=[0,-1],pB=[1,1]))
    assert len(elts)==1
    assert elts[0][0]=='edge'

def test_gen_int_elts_dim0():
    dt = Triangulation()

    assert len(list(dt.gen_intersected_elements(pA=[-1,0],pB=[1,0])))==0

    dt.add_node(x=[0,0])

    assert len(list(dt.gen_intersected_elements(pA=[-1,0],pB=[1,0])))==1
    assert len(list(dt.gen_intersected_elements(pA=[-1,0],pB=[1,1])))==0
