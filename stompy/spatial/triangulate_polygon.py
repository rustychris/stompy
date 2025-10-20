"""
Given a CCW polygon as ndarray [N,2]
return triples of indices into that point list for the constrained delaunay triangulation
of that polygon.

Uses shapely, which in turn requires GEOS>=3.10.0
"""
import numpy as np
import shapely
from shapely import geometry

##
def constrained_delaunay(poly):
    # iterate over ears (n)
    #   if ear is CW continue
    #   check ear circumcircle against all other nodes (n)
    geom = geometry.Polygon(poly)
    coll = shapely.constrained_delaunay_triangles(geom)
    result = np.zeros((len(coll.geoms),3),np.int32)

    for i,coll_poly in enumerate(coll.geoms):
        for j,vertex in enumerate(coll_poly.exterior.coords):
            for k,p in enumerate(poly):
                if vertex[0]==p[0] and vertex[1]==p[1]:
                    # Flip the order to keep CCW
                    result[i,2-j]=k
                    break
            else:
                assert False,"Failed to find vertex"
            if j==2:
                break # ignore repeated last vertex
        
    return result

if 1: # __name__=='__main__':
    from scipy.spatial import Delaunay
    # one of the cases that crashed
    poly=np.array([[ 7.81615623e-17, -3.09325554e+00],
                   [ 2.67682726e-02, -3.15736539e+00],
                   [ 1.81835496e-01, -3.11693181e+00],
                   [ 1.57940846e-01, -3.00638576e+00],
                   [ 1.30885003e-01, -2.99490569e+00],
                   [ 4.14040662e-02, -2.95984945e+00]])
    
    cdt = constrained_delaunay(poly)
    dt = Delaunay(poly).simplices # N,3 indexes

# CHECK ORDER
# cdt
# array([[0, 5, 4],
#        [4, 3, 2],
#        [2, 1, 0],
#        [0, 4, 2]], dtype=int32)
# 
# 
# dt
# array([[1, 2, 0],
#        [2, 4, 0],
#        [4, 5, 0],
#        [3, 4, 2]], dtype=int32)
# 
    

