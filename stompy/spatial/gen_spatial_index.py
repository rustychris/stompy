from __future__ import print_function
import numpy as np
import logging

# try again to normalize interface to spatial indices.
# no single implementation is perfect, though...

# This may transition to a factory function pattern, in
# order to allow for more flexible selection of implementation
PointIndex=None

# This is defined outside all of the conditionals as a hedge between
# (a) a factory method which can choose between implementations, and
# (b) having this class defined only once so that it can be subclassed
#     if one desires.

class _PointIndexKDTree(object):
    KDTree=None # This is populated on import in the factory. That's why the
    # class has an underscore, i.e. don't use it directly.
    def __init__(self,tuples,interleaved=False):
        assert self.KDTree is not None

        # stucture of tuples is [(orig_idx, [x, x, y, y], None), ... ]
        cell_centers=np.zeros( (len(tuples),2), 'f8')
        self.idx_to_original=np.zeros(len(cell_centers),'i4')

        # note that orig_idx may not be sequential (due to deleted nodes, cells)
        # so we have to map sequential indices back to original index.
        for i,tup in enumerate(tuples):
            cell_centers[i]= [tup[1][0],tup[1][2]]
            self.idx_to_original[i]=tup[0]

        self.cell_centers=cell_centers
        self.rebuild()

    def rebuild(self):
        if len(self.cell_centers):
            self._kdtree = self.KDTree(self.cell_centers)
        else:
            self._kdtree = None

    def nearest(self,xxyy,count):
        pnt=xxyy[np.array([0,2])]
        if self._kdtree is not None:
            distances,hits=self._kdtree.query(pnt,count)
            if count==1:
                # kdtree returns a single index, no list, when count==1
                hits=np.array([hits])
            return self.idx_to_original[hits]
        else:
            if len(self.cell_centers):
                distances=utils.dist(pnt - self.cell_centers)
                hits=np.argsort(distances)[:count]
                return self.idx_to_original[hits]
            else:
                return []

    def insert(self,idx,point_tuple):
        raise NotImplementedError("KDTree in use by gen_spatial_index; insert not allowed; install rtree...")

    def delete(self,idx,point_tuple):
        raise NotImplementedError("KDTree in use by gen_spatial_index; delete not allowed; install rtree...")


def rect_index_class_factory(implementation='rtree'):
    if implementation in ['rtree','best']:
        try:
            from rtree.index import Rtree 
            return Rtree
        except ImportError:
            if implementation=='rtree':
                raise
            # otherwise fall through to next best
    raise Exception("Failed to load any spatial index based on implementation='%s'"%implementation)
    
def point_index_class_factory(implementation='best'):
    if implementation in ['rtree','best']:
        try:
            from rtree.index import Rtree

            # try to mimic the Rtree interface - starting by just using it...
            return Rtree
        except ImportError:
            if implementation=='rtree':
                raise
            # otherwise fall through to next best

    if implementation in ['qgis','best']:
        try:
            from . import qgis_spatialindex
            return qgis_spatialindex.RtreeQgis
        except ImportError:
            if implementation=='qgis':
                raise
            # otherwise fall through to next

    if implementation in ['kdtree','best']:
        try:
            # keep the try..except a bit tighter around the import
            from scipy.spatial import KDTree 
            _PointIndexKDTree.KDTree=KDTree
            return _PointIndexKDTree
        except ImportError:
            if implementation=='kdtree':
                raise

    raise Exception("Failed to load any spatial index based on implementation='%s'"%implementation)


try:
    PointIndex=point_index_class_factory()
except ImportError:
    logging.info("Spatial index not available")
    PointIndex=None

try:
    RectIndex=rect_index_class_factory()
except ImportError:
    logging.info("Spatial index of rectangles not available")
    RectIndex=None
    
