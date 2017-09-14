# A wrapper around scipy.spatial's kdtree to make it look like 
# the python rtree / spatialindex implementation

from scipy.spatial import KDTree

class RtreeKDTree(object):
    """ wrap scipy KDTree spatial index to look as much like Rtree class
    as possible
    """
    def __init__(self,stream,interleaved=False):
        """ stream: an iterable, returning tuples of the form (id,[xmin,xmax,ymin,ymax],object)
        for now, requires that xmin==xmax, and ymin==ymax

        For now, only interleaved=False is supported.
        """
        if interleaved:
            raise Exception("No support for interleaved index.  You must use xxyy ordering")
        
        it = iter(stream)

        self.points = []
        self.data = []
        for fid,xxyy,obj in it:
            if xxyy[0]!=xxyy[1] or xxyy[2]!=xxyy[3]:
                raise Exception("No support in kdtree for finite sized objects")
            self.points.append([xxyy[0],xxyy[2]])
            self.data.append( (fid,obj) )

        self.refresh_tree()
    def refresh_tree(self):
        self.kdt = KDTree(self.points)
            
    def nearest(self, rect, count):
        xy = [rect[0],rect[2]]
        dists,results = self.kdt.query( xy,k=count )
        if count == 1:
            dists=[dists]
            results=[results]
        
        return [self.data[r][0] for r in results]
    
    def intersects(self,xxyy):
        """ This should be made compatible with the regular RTree call...
        """
        raise Exception("Intersects is not implemented in scipy.KDTree")
        return []
    
    def insert(self, feat_id, rect=None ):
        if rect[0]!=rect[1] or rect[2]!=rect[3]:
            raise Exception("No support in kdtree for finite sized objects")
        if rect is None:
            raise Exception("Not sure what inserting an empty rectangle is supposed to do...")
        self.points.append( [rect[0],rect[2]] )
        self.data.append( [feat_id,None] )

        self.refresh_tree()

    def feat_id_to_index(self,feat_id):
        for i in range(len(self.data)):
            if self.data[i][0] == feat_id:
                return i
        raise Exception("feature id not found")
    
    def delete(self, feat_id, rect ):
        index = self.feat_id_to_index(feat_id)
        del self.data[index]
        del self.points[index]
        self.refresh_tree()

        
