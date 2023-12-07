"""
Nearest neighbor interpolation using scipy voronoi

Note that this is not fast enough for interpolation of raster, but instead meant
for a relatively small number of target points, where the same weights can 
be re-used many times.
"""
import numpy as np
from .. import memoize, utils
from scipy.spatial import Voronoi 

def signed_area_py(points):
    # typical voronoi regions don't have that many vertices and
    # using straight pythong is ~3x faster.
    N=points.shape[0]
    area=0.0
    for i in range(N):
        ip1 = (i+1)%N
        area += points[i,0]*points[ip1,1] - points[ip1,0]*points[i,1]
    return 0.5*area

def nn_weights(src_xy,
               dst_xy,
               center=None,
               pad=None):
    """
    src_xy: [N,2] locations of source data points
    dst_xy: [2] location of point to extract
    center: specify a center for the dummy points, defaults to mean
    pd: specify scale for dummy points, default to 100x the larger span.
    """
    if src_xy.shape[0]==0:
        return np.zeros(0)
    if src_xy.shape[0]==1:
        return np.r_[1.0]

    # anything else it should work, even if goofy.
    
    if center is None:
        center=src_xy.mean(axis=0)
    if pad is None:
        pad = 100*(src_xy.max(axis=0) - src_xy.min(axis=0)).max()
        
    assert pad>0,"Degenerate points?"
    
    dummies = np.array( [[center[0]-pad,center[1]-pad],
                         [center[0]+pad,center[1]-pad],
                         [center[0]-pad,center[1]+pad],
                         [center[0]+pad,center[1]+pad]] )
    
    if 1:
        vor1 = Voronoi(np.concatenate([src_xy,dummies]))
        vor2 = Voronoi(np.concatenate([src_xy,dummies,[dst_xy]]))
    
        # how does one deal with the ridge vertices?
        # cheat and put some dummy vertices far away, and ignore
        # area stolen from them.
        
        weights=[]
        for i in range(src_xy.shape[0]):
            reg1=vor1.regions[ vor1.point_region[i] ]
            reg2=vor2.regions[ vor2.point_region[i] ]
            verts1=vor1.vertices[reg1]
            verts2=vor2.vertices[reg2]
            #area1=np.abs(utils.signed_area(verts1))
            #area2=np.abs(utils.signed_area(verts2))
            area1=np.abs(signed_area_py(verts1))
            area2=np.abs(signed_area_py(verts2))
            if area1<area2:
                # a little bit is okay, just some roundoff
                # sketchy calcs, so give it some room relative to
                # 64-bit limits
                if (area2-area1)/area1 < 1e-10:
                    area1=area2
                else:
                    raise Exception("NN interpolation: areas are not good")
            weights.append( area1-area2 )
        weights=np.array(weights)
    else:
        # Tried incremental, but I think there are some limitations on
        # what can be accessed, and it crashed.
        N=src_xy.shape[0]
        vor = Voronoi(np.concatenate([src_xy,dummies]),incremental=True)
        weights=np.zeros(N,np.float64)
        for i in range(N):
            reg=vor.regions[ vor.point_region[i] ]
            verts=vor.vertices[reg]
            weights[i] = abs(signed_area_py(verts))
        vor.add_points([dst_xy])          
        for i in range(N):
            reg=vor.regions[ vor.point_region[i] ]
            verts=vor.vertices[reg]
            area=abs(signed_area_py(verts))
            weights[i] -= area
        weights[weights<0]=0.0

    weights=weights/weights.sum()
    return weights

# memoized version
nn_weights_memo=memoize.memoize(lru=4000)(nn_weights)
