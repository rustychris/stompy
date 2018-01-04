from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Kernel import Point_2

from stompy.grid import cgal_line_walk

import numpy as np

def test_cgal_link_walk():
    DT=Constrained_Delaunay_triangulation_2()

    xys=np.array( [ [0,0],
                    [1,0],
                    [0,1],
                    [1,2] ],'f8' )
    # in some versions, Point_2 is picky that it gets doubles,
    # not ints.
    pnts=[Point_2(xy[0],xy[1])
          for xy in xys]

    vhs=[DT.insert(p) for p in pnts]

    DT.insert_constraint(vhs[0],vhs[2])

    ##

    res0=cgal_line_walk.line_walk(DT,vhs[0],vhs[1])

    assert not DT.is_constrained(res0[0][1])
    res1=cgal_line_walk.line_walk(DT,vhs[0],vhs[2])

    assert DT.is_constrained(res1[0][1])

    assert len(cgal_line_walk.line_conflicts(DT,p1=[5,5],p2=[5,6]))==0

    assert len(cgal_line_walk.line_conflicts(DT,p1=[0.5,-0.5],p2=[0.5,0.5]))==0

    assert len(cgal_line_walk.line_conflicts(DT,p1=[-0.5,0.5],p2=[0.5,0.5]))>0

    res3=cgal_line_walk.line_conflicts(DT,p1=[0,-1],p2=[2,1])
    assert len(res3)>0
    assert res3[0][0]=='v'


