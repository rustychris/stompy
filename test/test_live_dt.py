import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

from stompy.grid import live_dt, paver, exact_delaunay, trigrid, orthomaker

def run_basic(klass):
    pnts = np.array([ [0,0],
                      [3,0],
                      [5,0],
                      [10,0],
                      [5,5],
                      [6,2],
                      [12,4]], np.float64 )

    g=klass(points=pnts)

    g.add_edge(0,1)
    g.add_edge(1,2)
    g.add_edge(2,3)
    j24=g.add_edge(2,4)
    g.add_edge(0,4)
    j25=g.add_edge(2,5)
    j56=g.add_edge(5,6)
    g.add_edge(3,6)
    g.add_edge(4,5)

    # plt.figure(1).clf()
    # g.plot()
    # g.plot_nodes()

    # point on convex hull away from the triangulation
    assert g.shoot_ray(0,[2,1])[0]==j24
    assert g.shoot_ray(2,[1,1])[0]==j56
    assert g.shoot_ray(3,[-4,2])[0] in (j25,j56) 
    # py code is failing these:
    res=g.shoot_ray(0,[0,1])
    assert res == (None,None) 

def test_basic_cgal():
    run_basic(live_dt.LiveDtCGAL)

def test_basic_python():
    run_basic(live_dt.LiveDtPython)

