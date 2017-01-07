from __future__ import print_function

# try again to normalize interface to spatial indices.
# no single implementation is perfect, though...

try:
    from rtree.index import Rtree 

    # try to mimic the Rtree interface - starting by just using it...
    PointIndex=Rtree
except ImportError:
    PointIndex=None

if PointIndex is None:
    try:
        from . import qgis_spatialindex
        PointIndex=qgis_spatialindex.RtreeQgis
    except ImportError:
        print("Failed to load qgis spatial index, too")

if PointIndex is None:
    PointIndex="not available"

# other bits of implementation:

# try:
#     from scipy.spatial import KDTree
# except ImportError:
#     print "No spatial indexing available"

## 

# 
# # This isn't working with libspatialindex 1.8.5, but does seem
# # to work with libspatilindex 1.8.0, and rtree 0.8.2 from ioos
# def gen():
#     yield (1,[10,10,20,20],None)
# 
# from rtree import index
# from rtree.index import Rtree
# p = index.Property()
# p.dimension=2
# p.interleaved=False
# 
# idx=Rtree(gen(),properties=p)
# 
# ## 
# 
# # duplicate some test from the website:
# from rtree import index
# from rtree.index import Rtree
# assert int(index.__c_api_version__.split(b'.')[1]) >= 7
