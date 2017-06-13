import numpy as np
import nose

from stompy.spatial import gen_spatial_index

reload(gen_spatial_index)

def helper(implementation):
    x=np.linspace(0,1,51)
    y=np.linspace(0,1,51)
    X,Y=np.meshgrid(x,y)

    pnts=np.c_[ X.ravel(), X.ravel(), Y.ravel(), Y.ravel() ]

    klass=gen_spatial_index.point_index_class_factory(implementation=implementation)
    tuples=zip( np.arange(len(pnts)), pnts, [None]*len(pnts) )

    index=klass(tuples,interleaved=False)

    # avoid ambiguous answer - 0.015 better than 0.01
    target=np.array([0.015,0.015,0.02,0.02])

    hits=index.nearest( target, 5)
    hits=list(hits) # in case it's a generator

    brute=np.argmin(utils.dist(pnts[:,[0,2]]-target[[0,2]]))

    assert hits[0]==brute

def test_rtree():
    helper('rtree')

def test_best():
    helper('best')

def test_kdtree():
    helper('kdtree')

def test_qgis():
    # likely to fail if not run from within qgis.
    helper('qgis')

## 

if __name__=='__main__':
    nose.main()

