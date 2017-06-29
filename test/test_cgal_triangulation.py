from CGAL.CGAL_Triangulation_2 import Constrained_Delaunay_triangulation_2
from CGAL.CGAL_Kernel import Point_2

def test_basic():
    def create():
        DT = Constrained_Delaunay_triangulation_2()

        pnts=[Point_2(0.0, 0.0),
              Point_2(1.0, 0.0),
              Point_2(1.0, 1.0)]

        vhs=[ DT.insert(pnt) for pnt in pnts ]

        DT.insert_constraint(vhs[0],vhs[1])
        DT.insert_constraint(vhs[1],vhs[2])
        
        DT.remove_incident_constraints(vhs[1])
        
        return DT

    DTs=[create()
         for _ in range(100)]
    
    
