from __future__ import print_function

# A wrapper around QGIS's builtin spatialindex class (which used to conflict with
# the python rtree / spatialindex implementation), to make it look like
# the rtree api.

import qgis.core as qc

#index = qc.QgsSpatialIndex()

#feat = qc.QgsFeature()
#gPnt = qc.QgsGeometry.fromPoint(qc.QgsPoint(1,1))
#feat.setGeometry(gPnt)

# qc.QgsPoint(25.4, 12.7)
# index.insertFeature(feat)

# currently, trigrid passes interleaved=False to Rtree().
#   the stream is ordered xxyy
#   all calls use xxyy.

class RtreeQgis(object):
    """ wrap qgis internal spatial index to look as much like Rtree class
    as possible
    """
    def __init__(self,stream,interleaved=False):
        """ stream: an iterable, returning tuples of the form (id,[xmin,xmax,ymin,ymax],object)
        for now, requires that xmin==xmax, and ymin==ymax

        For now, only interleaved=False is supported.
        """
        it = iter(stream)
        self.qsi = qc.QgsSpatialIndex()

        if interleaved:
            raise Exception("No support for interleaved index.  You must use xxyy ordering")
        
        for feat_id,rect_xxyy,obj in it:
            self.insert(feat_id,rect=rect_xxyy)
            
    def nearest(self, rect, count):
        try:
            pntxy=qc.QgsPointXY(float(rect[0]),float(rect[2]))
        except TypeError:
            raise Exception("nearest(rect=%s,count=%s"%(rect,count))
        results = self.qsi.nearestNeighbor(pntxy, count)
        return results

    def intersects(self,xxyy):
        """ This should be made compatible with the regular RTree call...
        """
        rect = qc.QgsRectangle(xxyy[0],xxyy[2],xxyy[1],xxyy[3])
        results = self.qsi.intersects(rect)
        return results
    
    def make_feature(self,feat_id,rect):
        feat = qc.QgsFeature(feat_id)
        # feat.setFeatureId(feat_id)
        if rect[0] != rect[1] or rect[2]!=rect[3]:
            print( "WARNING: can only deal with point geometries right now" )
        gPnt = qc.QgsGeometry.fromPointXY(qc.QgsPointXY(float(rect[0]),float(rect[2])))
        feat.setGeometry(gPnt)
        return feat

    def insert(self, feat_id, rect=None ):
        feat = self.make_feature(feat_id,rect=rect)
        self.qsi.insertFeature(feat)

    def delete(self, feat_id, rect ):
        feat = self.make_feature(feat_id,rect = rect)
        self.qsi.deleteFeature(feat)

        
