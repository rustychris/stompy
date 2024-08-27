from __future__ import division
from __future__ import print_function

# still tracking down the last few calls missing the np. prefix,
# leftover from 'from numpy import *'
import numpy as np 
import six

import glob,types
import copy

from numpy.random import random
from numpy import ma
from numpy.linalg import norm

import tempfile

from scipy import interpolate
try:
    from scipy.stats import nanmean
except ImportError:
    from numpy import nanmean

from scipy import signal
from scipy import ndimage

from scipy.interpolate import RectBivariateSpline

from functools import wraps

# Lazily loads plt just for plotting functions.
# Gross, but helpful??
def with_plt(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        global plt
        import matplotlib.pyplot as plt
        return f(*args, **kwds)
    return wrapper

try:
    import matplotlib.tri as delaunay
except ImportError:
    # older deprecated module
    from matplotlib import delaunay

from . import wkb2shp
from ..utils import array_append, isnat, circumcenter, dist, set_keywords

try:
    from matplotlib import cm
except ImportError:
    cm = None
    
# load both types of indices, so we can choose per-field
# which one to use
from .gen_spatial_index import PointIndex,RectIndex

# Older code tried to use multiple implementations
# import stree
# from safe_rtree import Rtree
# from rtree.index import Rtree

xxyy = np.array([0,0,1,1])
xyxy = np.array([0,1,0,1])

def as_xxyy(p1p2):
    p1p2=np.asarray(p1p2)
    if p1p2.ndim == 1:
        return p1p2 # presumably it's already xxyy layout
    else:
        return np.array(p1p2[xyxy,xxyy])

from .linestring_utils import upsample_linearring

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
import subprocess,threading
import logging
log=logging.getLogger(__name__)

try:
    from osgeo import gdal,osr,ogr
except ImportError:
    try:
        import gdal,osr,ogr
    except ImportError:
        gdal=osr=ogr=None
        log.warning("GDAL not loaded")

try:
    from shapely import geometry, wkb
    try:
        from shapely.prepared import prep
    except ImportError:
        prep = none
except ImportError:
    log.warning("Shapely not loaded")
    wkb=geometry=None
    
import os.path

if gdal:
    numpy_type_to_gdal = {np.int8:gdal.GDT_Byte, # meh.  not quite real, most likely
                          np.uint8:gdal.GDT_Byte,
                          np.float32:gdal.GDT_Float32,
                          np.float64:gdal.GDT_Float64,
                          np.int16:gdal.GDT_Int16,
                          np.int32:gdal.GDT_Int32,
                          int:gdal.GDT_Int32,
                          np.uint16:gdal.GDT_UInt16,
                          np.uint32:gdal.GDT_UInt32}



#  # try to create an easier way to handle non-uniform meshes.  In particular
#  # it would be nice to be able to something like:
#  
#  foo = field.gdal_source('foo.asc') # grid is lat/lon
#  
#  foo_utm = foo.transform_to('EPSG:26910')
#  
#  bar = field.xyz_source('bar.xyz') # point data
#  
#  # this uses the foo_utm grid, adds values interpolated from bar, and any
#  # points where bar cannot interpolate are assigned nan.
#  foo_bar = foo_utm.add(bar,keep='valid')


class Field(object):
    """ Superclass for spatial fields
    """
    _projection=None
    def __init__(self,projection=None,**kw):
        """
        projection: GDAL/OGR parseable string representation
        """
        self.assign_projection(projection)
        set_keywords(self,kw)

    def assign_projection(self,projection):
        self._projection = projection

    def reproject(self,from_projection=None,to_projection=None):
        """ Reproject to a new coordinate system.
        If the input is structured, this will create a curvilinear
        grid, otherwise it creates an XYZ field.
        """

        xform = self.make_xform(from_projection,to_projection)
        new_field = self.apply_xform(xform)
        new_field._projection = to_projection
        return new_field

    def copy(self):
        return copy.copy(self)

    def make_xform(self,from_projection,to_projection):
        if from_projection is None:
            from_projection = self.projection()
            if from_projection is None:
                raise Exception("No source projection can be determined")

        src_srs = osr.SpatialReference()
        src_srs.SetFromUserInput(from_projection)

        dest_srs = osr.SpatialReference()
        dest_srs.SetFromUserInput(to_projection)

        xform = osr.CoordinateTransformation(src_srs,dest_srs)

        return xform

    def xyz(self):
        raise Exception("Not implemented")
    def crop(self,rect):
        raise Exception("Not implemented")
    def projection(self):
        return self._projection
    def bounds(self):
        raise Exception("Not Implemented")

    def bounds_in_cs(self,cs):
        b = self.bounds()

        xform = self.make_xform(self.projection(),cs)

        corners = [ [b[0],b[2]],
                    [b[0],b[3]],
                    [b[1],b[2]],
                    [b[1],b[3]] ]

        new_corners = np.array( [xform.TransformPoint(c[0],c[1])[:2] for c in corners] )

        xmin = new_corners[:,0].min()
        xmax = new_corners[:,0].max()
        ymin = new_corners[:,1].min()
        ymax = new_corners[:,1].max()

        return [xmin,xmax,ymin,ymax]
    def quantize_space(self,quant):
        self.X = round_(self.X)

    def envelope(self,eps=1e-4):
        """ Return a rectangular shapely geometry the is the bounding box of
        this field.
        """
        b = self.bounds()

        return geometry.Polygon( [ [b[0]-eps,b[2]-eps],
                                   [b[1]+eps,b[2]-eps],
                                   [b[1]+eps,b[3]+eps],
                                   [b[0]-eps,b[3]+eps],
                                   [b[0]-eps,b[2]-eps] ])

    ## Some methods taken from density_field, which Field will soon supplant
    def value(self,X):
        """ in density_field this was called 'scale' - evaluates the field
        at the given point or vector of points.  Some subclasses can be configured
        to interpolate in various ways, but by default should do something reasonable
        """
        raise Exception("not implemented")
        # X = np.array(X)
        # return self.constant * np.ones(X.shape[:-1])

    def value_on_edge(self,e,samples=5,reducer=np.nanmean):
        """ Return the value averaged along an edge - the generic implementation
        just takes 5 samples evenly spaced along the line, using value()
        """
        x=np.linspace(e[0,0],e[1,0],samples)
        y=np.linspace(e[0,1],e[1,1],samples)
        X = np.array([x,y]).transpose()
        return reducer(self.value(X))

    def __call__(self,X):
        return self.value(X)

    def __mul__(self,other):
        return BinopField(self,np.multiply,other)

    def __rmul__(self,other):
        return BinopField(other,np.multiply,self)

    def __add__(self,other):
        return BinopField(self,np.add,other)

    def __sub__(self,other):
        return BinopField(self,np.subtract,other)

    def to_grid(self,nx=None,ny=None,interp='linear',bounds=None,dx=None,dy=None,valuator='value'):
        """ bounds is a 2x2 [[minx,miny],[maxx,maxy]] array, and is *required* for BlenderFields
        bounds can also be a 4-element sequence, [xmin,xmax,ymin,ymax], for compatibility with
        matplotlib axis(), and Paving.default_clip.

        specify *one* of:
          nx,ny: specify number of samples in each dimension
          dx,dy: specify resolution in each dimension

        interp used to default to nn, but that is no longer available in mpl, so now use linear.

        bounds is interpreted as the range of center locations of pixels.  This gets a bit
        gross, but that is how some of the tile functions below work.
        """
        if bounds is None:
            xmin,xmax,ymin,ymax = self.bounds()
        else:
            if len(bounds) == 2:
                xmin,ymin = bounds[0]
                xmax,ymax = bounds[1]
            else:
                xmin,xmax,ymin,ymax = bounds

        if nx is None:
            nx=1+int(np.round((xmax-xmin)/dx))
            ny=1+int(np.round((ymax-ymin)/dy))
        x = np.linspace( xmin,xmax, nx )
        y = np.linspace( ymin,ymax, ny )

        xx,yy = np.meshgrid(x,y)

        X = np.concatenate( (xx[...,None], yy[...,None]), axis=2)

        if valuator=='value':
            newF = self.value(X)
        else:
            valuator == getattr(self,valuator)
            newF = valuator(X)

        return SimpleGrid(extents=[xmin,xmax,ymin,ymax],
                          F=newF,projection=self.projection())


# Different internal representations:
#   SimpleGrid - constant dx, dy, data just stored in array.

class XYZField(Field):
    def __init__(self,X,F,projection=None,from_file=None,**kw):
        """ X: Nx2 array of x,y locations
            F: N   array of values
        """
        Field.__init__(self,projection=projection,**kw)
        self.X = X
        self.F = F
        self.index = None
        self.from_file = from_file

        self.init_listeners()

    @with_plt
    def plot(self,**kwargs):
        # this is going to be slow...
        def_args = {'c':self.F,
                    'antialiased':False,
                    'marker':'s',
                    'lw':0}
        def_args.update(kwargs)
        plt.scatter( self.X[:,0].ravel(),
                     self.X[:,1].ravel(),
                     **def_args)

    def bounds(self):
        if self.X.shape[0] == 0:
            return None
        
        xmin = self.X[:,0].min()
        xmax = self.X[:,0].max()
        ymin = self.X[:,1].min()
        ymax = self.X[:,1].max()

        return (xmin,xmax,ymin,ymax)

    def apply_xform(self,xform):
        new_fld=self.copy()
        
        new_X = self.X.copy()

        if len(self.F)>10000:
            print("Transforming points")
        for i in range(len(self.F)):
            if i>0 and i % 10000 == 0:
                print("%.2f%%"%( (100.0*i) / len(self.F)) )
                
            new_X[i] = xform.TransformPoint(*self.X[i])[:2]
        if len(self.F)>10000:
            print("Done transforming points")

        # projection should get overwritten by the caller
        new_fld.X=new_X
        return new_fld

    # an XYZ Field of our voronoi points
    _tri = None
    def tri(self,aspect=1.0):
        if aspect!=1.0:
            return delaunay.Triangulation(self.X[:,0],
                                          aspect*self.X[:,1])
        if self._tri is None:
            self._tri = delaunay.Triangulation(self.X[:,0],
                                               self.X[:,1])
        return self._tri

    def plot_tri(self,**kwargs):
        import plot_utils
        plot_utils.plot_tri(self.tri(),**kwargs)
    
    _nn_interper = None
    def nn_interper(self,aspect=1.0):
        if aspect!=1.0:
            try:
                return self.tri(aspect=aspect).nn_interpolator(self.F)
            except AttributeError:
                raise Exception("Request for nearest-neighbors, which was discontinued by mpl")
        if self._nn_interper is None:
            try:
                self._nn_interper = self.tri().nn_interpolator(self.F)
            except AttributeError:
                raise Exception("Request for nearest-neighbors, which was discontinued by mpl")
        return self._nn_interper
    _lin_interper = None
    def lin_interper(self,aspect=1.0):
        def get_lin_interp(t,z):
            try:
                return t.linear_interpolator(z)
            except AttributeError: # modern matplotlib separates this out:
                from matplotlib.tri import LinearTriInterpolator
                return LinearTriInterpolator(t,z)
        if aspect!=1.0:
            return get_lin_interp(self.tri(aspect=aspect),self.F)
        if self._lin_interper is None:
            self._lin_interper = get_lin_interp(self.tri(),self.F)
        return self._lin_interper

    #_voronoi = None
    # default_interpolation='naturalneighbor'# phased out by mpl
    default_interpolation='linear'
    # If true, linear interpolation will revert to nearest when queried outside
    # the convex hull
    outside_hull_fallback=True
    def interpolate(self,X,interpolation=None):
        """
        X: [...,2] coordinates at which to interpolate.
        interpolation: should have been called 'method'.
           The type of interpolation.
           'nearest': select nearest source point
           'naturalneighbor': Deprecated (only works with very old MPL)
             Delaunay triangulation-based natural neighbor interpolation.
           'linear': Delaunay-based linear interpolation.
        """
        if interpolation is None:
            interpolation=self.default_interpolation
        # X should be a (N,2) vectors - make it so
        X=np.asanyarray(X).reshape([-1,2])

        newF = np.zeros( X.shape[0], np.float64 )

        if interpolation=='nearest':
            for i in range(len(X)):
                if i % 10000 == 1:
                    print( " %.2f%%"%( (100.0*i)/len(X) ))

                if not self.index:
                    dsqr = ((self.X - X[i])**2).sum(axis=1)
                    j = np.argmin( dsqr )
                else:
                    j = self.nearest(X[i])

                newF[i] = self.F[j]
        elif interpolation=='naturalneighbor':
            newF = self.nn_interper()(X[:,0],X[:,1])
            # print "why aren't you using linear?!"
        elif interpolation=='linear':
            interper = self.lin_interper()
            newF[:] = interper(X[:,0],X[:,1])
            if self.outside_hull_fallback:
                # lin_interper may return masked array instead
                # of nans.
                newF=np.ma.filled(newF,np.nan)
                bad=np.isnan(newF)
                if np.any(bad):
                    # Old approach, use nearest:
                    newF[bad]=self.interpolate(X[bad],'nearest')
        else:
            raise Exception("Bad value for interpolation method %s"%interpolation)
        return newF

    def build_index(self,index_type=None):
        if index_type is not None:
            log.warning("Ignoring request for specific index type")

        self.index_type = 'rtree'
        if self.X.shape[0] > 0:
            # this way we get some feedback
            def gimme():
                i = gimme.i
                if i < self.X.shape[0]:
                    if i %10000 == 0 and i>0:
                        print("building index: %d  -  %.2f%%"%(i, 100.0 * i / self.X.shape[0] ))
                    gimme.i = i+1
                    return (i,self.X[i,xxyy],None)
                else:
                    return None
            gimme.i = 0

            tuples = iter(gimme,None)

            #print "just building Rtree index in memory"
            self.index = PointIndex(tuples,interleaved=False)
        else:
            self.index = PointIndex(interleaved=False)
        #print "Done"

    def within_r(self,p,r):
        if self.index:
            if self.index_type == 'stree':
                subset = self.index.within_ri(p,r)
            else: # rtree
                # first query a rectangle
                rect = np.array( [p[0]-r,p[0]+r,p[1]-r,p[1]+r] )

                subset = self.index.intersection( rect )
                if isinstance(subset, types.GeneratorType):
                    subset = list(subset)
                subset = np.array( subset )

                if len(subset) > 0:
                    dsqr = ((self.X[subset]-p)**2).sum(axis=1)
                    subset = subset[ dsqr<=r**2 ]

            return subset
        else:
            # print "bad - no index"
            dsqr = ((self.X-p)**2).sum(axis=1)
            return np.where(dsqr<=r**2)[0]

    def inv_dist_interp(self,p,
                        min_radius=None,min_n_closest=None,
                        clip_min=-np.inf,clip_max=np.inf,
                        default=None):
        """ inverse-distance weighted interpolation
        This is a bit funky because it tries to be smart about interpolation
        both in dense and sparse areas.

        min_radius: sample from at least this radius around p
        min_n_closest: sample from at least this many points

        """
        if min_radius is None and min_n_closest is None:
            raise Exception("Must specify one of r (radius) or n_closest")
        
        r = min_radius
        
        if r:
            nearby = self.within_r(p,r)
                
            # have we satisfied the criteria?  if a radius was specified
            if min_n_closest is not None and len(nearby) < min_n_closest:
                # fall back to nearest
                nearby = self.nearest(p,min_n_closest)
        else:
            # this is slow when we have no starting radius
            nearby = self.nearest(p,min_n_closest)

        dists = np.sqrt( ((p-self.X[nearby])**2).sum(axis=1) )

        # may have to trim back some of the extras:
        if r is not None and r > min_radius:
            good = np.argsort(dists)[:min_n_closest]
            nearby = nearby[good]
            dists = dists[good]

        if min_radius is None:
            # hrrmph.  arbitrary...
            min_radius = dists.mean()

        dists[ dists < 0.01*min_radius ] = 0.01*min_radius
        
        weights = 1.0/dists

        vals = self.F[nearby]
        vals = np.clip(vals,clip_min,clip_max)

        val = (vals * weights).sum() / weights.sum()
        return val

    def nearest(self,p,count=1):
        # print "  Field::nearest(p=%s,count=%d)"%(p,count)
        
        if self.index:
            if self.index_type=='stree':
                if count == 1:
                    return self.index.closest(p)
                else:
                    return self.index.n_closest(p,count)
            else:  # rtree
                hits = self.index.nearest( p[xxyy], count )
                # deal with API change in RTree
                if isinstance( hits, types.GeneratorType):
                    hits = [next(hits) for i in range(count)]

                if count == 1:
                    return hits[0]
                else:
                    return np.array(hits)
        else:
            # straight up, it takes 50ms per query for a small
            # number of points
            dsqr = ((self.X - p)**2).sum(axis=1)

            if count == 1:
                j = np.argmin( dsqr )
                return j
            else:
                js = np.argsort( dsqr )
                return js[:count]

    def rectify(self,dx=None,dy=None):
        """ Convert XYZ back to SimpleGrid.  Assumes that the data fall on a regular
        grid.  if dx and dy are None, automatically find the grid spacing/extents.
        """
        max_dimension = 10000.

        # Try to figure out a rectilinear grid that fits the data:
        xmin,xmax,ymin,ymax = self.bounds()

        # establish lower bound on delta x:
        if dx is None:
            min_deltax = (xmax - xmin) / max_dimension
            xoffsets = self.X[:,0] - xmin
            dx = xoffsets[xoffsets>min_deltax].min()

        if dy is None:
            min_deltay = (ymax - ymin) / max_dimension
            yoffsets = self.X[:,1] - ymin
            dy = yoffsets[yoffsets>min_deltay].min()

        print("Found dx=%g  dy=%g"%(dx,dy))

        nrows = 1 + int( 0.49 + (ymax - ymin) / dy )
        ncols = 1 + int( 0.49 + (xmax - xmin) / dx )

        # recalculate dx to be accurate over the whole range:
        dx = (xmax - xmin) / (ncols-1)
        dy = (ymax - ymin) / (nrows-1)
        delta = np.array([dx,dy])
        
        newF = np.nan*np.ones( (nrows,ncols), np.float64 )

        new_indices = (self.X - np.array([xmin,ymin])) / delta + 0.49
        new_indices = new_indices.astype(np.int32)
        new_indices = new_indices[:,::-1]

        newF[new_indices[:,0],new_indices[:,1]] = self.F

        return SimpleGrid(extents=[xmin,xmax,ymin,ymax],
                          F=newF,projection=self.projection())

    def to_grid(self,nx=2000,ny=2000,interp='linear',bounds=None,dx=None,dy=None,
                aspect=1.0,max_radius=None):
        """ use the delaunay based griddata() to interpolate this field onto
        a rectilinear grid.  In theory interp='linear' would give bilinear
        interpolation, but it tends to complain about grid spacing, so best to stick
        with the default 'nn' which gives natural neighbor interpolation and is willing
        to accept a wider variety of grids

        Here we use a specialized implementation that passes the extent/stride array
        to interper, since lin_interper requires this.

        interp='qhull': use scipy's delaunay/qhull interface.  this can
        additionally accept a radius which limits the output to triangles
        with a smaller circumradius.
        """
        if bounds is None:
            xmin,xmax,ymin,ymax = self.bounds()
        else:
            if len(bounds) == 4:
                xmin,xmax,ymin,ymax = bounds
            else:
                xmin,ymin = bounds[0]
                xmax,ymax = bounds[1]

        if dx is not None: # Takes precedence of nx/ny
            # This seems a bit heavy handed
            # round xmin/ymin to be an even multiple of dx/dy
            # xmin = xmin - (xmin%dx)
            # ymin = ymin - (ymin%dy)

            # The 1+, -1, stuff feels a bit sketch.  But this is how
            # CompositeField calculates sizes
            nx = 1 + int( (xmax-xmin)/dx )
            ny = 1 + int( (ymax-ymin)/dy )
            xmax = xmin + (nx-1)*dx
            ymax = ymin + (ny-1)*dy

        # hopefully this is more compatible between versions, also exposes more of what's
        # going on
        if interp == 'nn':
            interper = self.nn_interper(aspect=aspect)
        elif interp=='linear':
            interper = self.lin_interper(aspect=aspect)
        elif interp=='qhull':
            interper = self.qhull_interper(max_radius=max_radius)
            
        try:
            griddedF = interper[aspect*ymin:aspect*ymax:ny*1j,xmin:xmax:nx*1j]
        except TypeError: # newer interpolation doesn't have [y,x] notation
            y=np.linspace(aspect*ymin,aspect*ymax,ny)
            x=np.linspace(xmin,xmax,nx)
            # y,x led to the dimensions being swapped
            X,Y=np.meshgrid(x,y)
            # Y,X below led to all values being nan...
            griddedF = interper(X,Y) # not sure about index ordering here...

        return SimpleGrid(extents=[xmin,xmax,ymin,ymax],F=griddedF)

    def qhull_interper(self,max_radius=None):
        from scipy.spatial import Delaunay
        from scipy.interpolate import LinearNDInterpolator
        tri=Delaunay(self.X)
        if max_radius is not None:
            tris=tri.simplices
            ccs=circumcenter( tri.points[tri.simplices[:,0]],
                              tri.points[tri.simplices[:,1]],
                              tri.points[tri.simplices[:,2]] )
            rad=dist(ccs-tri.points[tri.simplices[:,0]])
            bad=rad>max_radius
        else:
            bad=None
        lin_nd=LinearNDInterpolator(tri,self.F)

        def interper(X,Y,lin_nd=lin_nd,bad=bad,tri=tri):
            XY=np.stack((X,Y),axis=-1)
            XYr=XY.reshape([-1,2])
            simps=tri.find_simplex(XYr)
            result=lin_nd(XYr)
            if bad is not None:
                result[(simps<0)|(bad[simps])]=np.nan
            return result.reshape(X.shape)
        return interper
    
    def crop(self,rect):
        if len(rect)==2:
            rect=[rect[0][0],rect[1][0],rect[0][1],rect[1][1]]
        xmin,xmax,ymin,ymax = rect

        good = (self.X[:,0] >= xmin ) & (self.X[:,0] <= xmax ) & (self.X[:,1] >= ymin) & (self.X[:,1]<=ymax)

        newX = self.X[good,:]
        newF = self.F[good]
        
        return XYZField(newX,newF, projection = self.projection() )
    def write_text(self,fname,sep=' '):
        with open(fname,'wt') as fp:
            for i in range(len(self.F)):
                fp.write( "%f%s%f%s%f\n"%(self.X[i,0],sep,
                                          self.X[i,1],sep,
                                          self.F[i] ) )

    def intersect(self,other,op,radius=0.1):
        """ Create new pointset that has points that are in both fields, and combine
        the values with the given operator op(a,b)
        """
        my_points = []
        new_F = []
        
        if not self.index:
            self.build_index()

        for i in range(len(other.F)):
            if i % 10000 == 0:
                print("%.2f%%"%(100.0*i/len(other.F)))
                
            p = self.within_r( other.X[i], radius )
            if len(p) > 0:
                # fudge it and take the first one...
                my_points.append(p[0])
                new_F.append( op(self.F[p[0]],other.F[i] ) )
        my_points = np.array(my_points)
        new_F = np.array(new_F)
        
        new_X = self.X[ my_points ]

        return XYZField( new_X, new_F )
    
    def decimate(self,factor):
        chooser = random( self.F.shape ) < 1.0/factor

        return XYZField( self.X[chooser,:], self.F[chooser], projection = self.projection() )

    def clip_to_polygon(self,poly):
        if not self.index:
            self.build_index()

        if prep:
            chooser = np.zeros(len(self.F),bool8)

            prep_poly = prep(poly)
            for i in range(len(self.F)):
                chooser[i] = prep_poly.contains( geometry.Point(self.X[i]) )
        else:
            # this only works with the stree implementation.
            chooser = self.index.inside_polygon(poly)

        if len(chooser) == 0:
            print("Clip to polygon got no points!")
            print("Returning empty field")
            return XYZField( np.zeros((0,2),np.float64), np.zeros( (0,1), np.float64) )
        else:
            return XYZField( self.X[chooser,:], self.F[chooser] )

    def cs2cs(self,
              src="+proj=utm +zone=10 +datum=NAD27 +nadgrids=conus",
              dst="+proj=utm +zone=10 +datum=NAD83"):
        """  In place modification of coordinate system.  Defaults to UTM NAD27 -> UTM NAD83
        """
        cmd = "cs2cs -f '%%f' %s +to %s"%(src,dst)

        proc = subprocess.Popen(cmd,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE)

        pnts = []
        def reader():
            while 1:
                line = proc.stdout.readline()
                if line == '':
                    break
                pnts.append(list(map(float,line.split()[:2])))

        thr = threading.Thread(target = reader)
        thr.start()

        point_count = len(self.F)
        for i in range(point_count):
            if i % 10000 == 0:
                print("%.2f%%"%( (100.0*i)/point_count ))
            proc.stdin.write("%.2f %.2f\n"%(self.X[i,0], self.X[i,1]) )
        proc.stdin.close()

        print("Finished writing")
        thr.join()

        pnts = np.array(pnts)

        if pnts.shape != self.X.shape:
            raise Exception('Size of converted points is %s, not %s'%( pnts.shape, self.X.shape ) )
        self.X = pnts


    def write(self,fname):
        fp = open(fname,'wb')

        pickle.dump( (self.X,self.F), fp, -1)
        fp.close()

    def to_xyz(self):
        # should this be self, or a copy of self???
        return self


    @staticmethod 
    def read_shp(shp_name,value_field='value'):
        ods = ogr.Open(shp_name)

        X = []
        F = []

        layer = ods.GetLayer(0)

        while 1:
            feat = layer.GetNextFeature()

            if feat is None:
                break

            F.append( feat.GetField(value_field) )

            geo = feat.GetGeometryRef()

            X.append( geo.GetPoint_2D() )
        X = np.array( X )
        F = np.array( F )
        return XYZField(X=X,F=F,from_file=shp_name)

    def write_shp(self,shp_name,value_field='value'):
        drv = ogr.GetDriverByName('ESRI Shapefile')

        ### open the output shapefile
        if os.path.exists(shp_name) and shp_name.find('.shp')>=0:
            print("removing ",shp_name)
            os.unlink(shp_name)

        ods = drv.CreateDataSource(shp_name)
        srs = osr.SpatialReference()
        if self.projection():
            srs.SetFromUserInput(self.projection())
        else:
            srs.SetFromUserInput('EPSG:26910')

        layer_name = os.path.splitext( os.path.basename(shp_name) )[0]
        
        ### Create the layer
        olayer = ods.CreateLayer(layer_name,
                                 srs=srs,
                                 geom_type=ogr.wkbPoint)
        
        olayer.CreateField(ogr.FieldDefn('id',ogr.OFTInteger))
        olayer.CreateField(ogr.FieldDefn(value_field,ogr.OFTReal))
        
        fdef = olayer.GetLayerDefn()
        
        ### Iterate over depth data
        for i in range(len(self.X)):
            x,y = self.X[i]

            wkt = geometry.Point(x,y).wkt

            new_feat_geom = ogr.CreateGeometryFromWkt( wkt )
            feat = ogr.Feature(fdef)
            feat.SetGeometryDirectly(new_feat_geom)
            feat.SetField('id',i)
            feat.SetField(value_field,self.F[i])

            olayer.CreateFeature(feat)

        olayer.SyncToDisk()

        ### Create spatial index:
        ods.ExecuteSQL("create spatial index on %s"%layer_name)

        
    @staticmethod
    def read(fname):
        """
        Read XYZField from a pickle file
        """
        fp = open(fname,'rb')
        X,F = pickle.load( fp )
        fp.close()
        return XYZField(X=X,F=F,from_file=fname)

    @staticmethod
    def merge(all_sources):
        all_X = concatenate( [s.X for s in all_sources] )
        all_F = concatenate( [s.F for s in all_sources] )

        return XYZField(all_X,all_F,projection=all_sources[0].projection())


    ## Editing API for use with GUI editor
    def move_point(self,i,pnt):
        self.X[i] = pnt
        
        if self.index:
            if self.index_type == 'stree':
                self.index = None
            else:
                old_coords = self.X[i,xxyy]
                new_coords = pnt[xxyy]

                self.index.delete(i, old_coords )
                self.index.insert(i, new_coords )
        self.updated_point(i)

    def add_point(self,pnt,value):
        """ Insert a new point into the field, clearing any invalidated data
        and returning the index of the new point
        """
        i = len(self.X)

        self.X = array_append(self.X,pnt)
        self.F = array_append(self.F,value)

        self._tri = None
        self._nn_interper = None
        self._lin_interper = None
        
        if self.index is not None:
            if self.index_type == 'stree':
                print("Stree doesn't know how to add points")
                self.index = None
            else:
                print("Adding new point %d to index at "%i,self.X[i])
                self.index.insert(i, self.X[i,xxyy] )

        self.created_point(i)
        return i

    def delete_point(self,i):
        if self.index is not None:
            if self.index_type == 'stree':
                print("Stree doesn't know how to delete point")
                self.index = None
            else:
                coords = self.X[i,xxyy]
                self.index.delete(i, coords )
            
        self.X[i,0] = np.nan
        self.F[i] = np.nan
        self.deleted_point(i)

    
    # subscriber interface for updates:
    listener_count = 0
    def init_listeners(self):
        self._update_point_listeners = {}
        self._create_point_listeners = {}
        self._delete_point_listeners = {}
    
    def listen(self,event,cb):
        cb_id = self.listener_count
        if event == 'update_point':
            self._update_point_listeners[cb_id] = cb
        elif event == 'create_point':
            self._create_point_listeners[cb_id] = cb
        elif event == 'delete_point':
            self._delete_point_listeners[cb_id] = cb
        else:
            raise Exception("unknown event %s"%event)
            
        self.listener_count += 1
        return cb_id
    def updated_point(self,i):
        for cb in self._update_point_listeners.values():
            cb(i)
    def created_point(self,i):
        for cb in self._create_point_listeners.values():
            cb(i)
    def deleted_point(self,i):
        for cb in self._delete_point_listeners.values():
            cb(i)

    ## Methods taken from XYZDensityField
    def value(self,X):
        """ X must be shaped (...,2)
        """
        X = np.asanyarray(X)
        orig_shape = X.shape

        X = X.reshape((-1,2))

        newF = self.interpolate(X)
        
        newF = newF.reshape(orig_shape[:-1])
        if newF.ndim == 0:
            return float(newF)
        else:
            return newF

    @with_plt
    def plot_on_boundary(self,bdry):
        # bdry is an array of vertices (presumbly on the boundary)
        l = np.zeros( len(bdry), np.float64 )

        ax = plt.gca()
        for i in range(len(bdry)):
            l[i] = self.value( bdry[i] )

            cir = Circle( bdry[i], radius=l[i])
            ax.add_patch(cir)

    # Pickle support -
    def __getstate__(self):
        """ the CGAL.ApolloniusGraph can't be pickled - have to recreate it
            """
        d = self.__dict__.copy()
        d['_lin_interper']=None
        return d

class PyApolloniusField(XYZField):
    """ 
    Takes a set of vertices and the allowed scale at each, and
    extrapolates across the plane based on a uniform telescoping rate
    """

    # But it's okay if redundant factor is None

    def __init__(self,X=None,F=None,r=1.1,redundant_factor=None):
        """r: telescoping rate

        redundant_factor: if a point being inserted has a scale which
        is larger than the redundant_factor times the existing scale
        at its location, then don't insert it.  So typically it would
        be something like 0.95, which says that if the existing scale
        at X is 100, and this point has a scale of 96, then we don't
        insert.
        """
        if X is None:
            assert F is None

        self.r = r
        self.redundant_factor = redundant_factor
        self.offset=np.array([0,0]) # not using an offset for now.
            
        if (X is None) or (redundant_factor is not None):
            super(PyApolloniusField,self).__init__(X=np.zeros( (0,2), np.float64),
                                                   F=np.zeros( 0, np.float64))
        else:
            super(PyApolloniusField,self).__init__(X=X,F=F)
            
        if self.redundant_factor is not None:
            for i in range(F.shape[0]):
                self.insert(X[i],F[i])

    def insert(self,xy,f):
        """ directly insert a point into the Apollonius graph structure
        note that this may be used to incrementally construct the graph,
        if the caller doesn't care about the accounting related to the
        field -
        returns False if redundant checks are enabled and the point was
        deemed redundant.
        """
        
        if (self.X.shape[0]==0) or (self.redundant_factor is None):
            redundant=False
        else:
            existing=self.interpolate(xy)
            redundant=existing*self.redundant_factor < f
        if not redundant:
            self.X=array_append(self.X,xy)
            self.F=array_append(self.F,f)
            return True
        else:
            return False

    def value(self,X):
        return self.interpolate(X)
    
    def interpolate(self,X):
        X=np.asanyarray(X)
        newF = np.zeros( X.shape[:-1], np.float64 )
        
        if len(self.F)==0:
            newF[...]=np.nan
            return newF

        # need to compute all pairs of distances:
        # self.X ~ [N,2]
        # X ~ [L,M,...,2]

        # some manual index wrangling to get an outside-join-multiply
        idx=(slice(None),) + tuple([None]*(X.ndim-1))

        dx=X[None,...,0] - self.X[ idx + (0,)]
        dy=X[None,...,1] - self.X[ idx + (1,)]
        dist = np.sqrt(dx**2 + dy**2)

        f = self.F[idx] + dist*(self.r-1.0)
        
        newF[...] = f.min(axis=0)
        return newF

    def to_grid(self,*a,**k):
        # XYZField implementation is no good to us.
        return Field.to_grid(self,*a,**k)

    @staticmethod 
    def read_shps(shp_names,value_field='value',r=1.1,redundant_factor=None):
        """ Read points or lines from a list of shapefiles, and construct
        an apollonius graph from the combined set of features.  Lines will be
        downsampled at the scale of the line.
        """
        lines=[]
        values=[]

        for shp_name in shp_names:
            print("Reading %s"%shp_name)

            layer=wkb2shp.shp2geom(shp_name,fold_to_lower=True)
            value_field=value_field.lower()
            
            for i in range(len(layer)):
                geo = layer['geom'][i]
                scale=layer[value_field][i]
                if np.isfinite(scale) and scale>0.0:
                    lines.append(np.array(geo.coords))
                    values.append(scale)
        return PyApolloniusField.from_polylines(lines,values,
                                                r=r,redundant_factor=redundant_factor)

    @staticmethod
    def from_polylines(lines,values,r=1.1,redundant_factor=None):
        X = []
        F = []
        edges = []

        for coords,value in zip(lines,values):
            if len(coords) > 1: # it's a line - upsample
                # need to say closed_ring=0 so it doesn't try to interpolate between
                # the very last point back to the first
                coords = upsample_linearring(coords,value,closed_ring=0)
            if all(coords[-1]==coords[0]):
                coords = coords[:-1]

            # remove duplicates:
            mask = np.all(coords[0:-1,:] == coords[1:,:],axis=1)
            mask=np.r_[False,mask]
            if np.sum(mask)>0:
                print("WARNING: removing duplicate points in shapefile")
                print(coords[mask])
                coords = coords[~mask]

            X.append( coords )
            F.append( value*np.ones(len(coords)) )

        X = np.concatenate( X )
        F = np.concatenate( F )
        return PyApolloniusField(X=X,F=F,r=r,redundant_factor=redundant_factor)

has_apollonius=False
try:
    import CGAL
    # And does it have Apollonius graph bindings?
    cgal_bindings = None
    try:
        # from CGAL import Point_2,Site_2
        from CGAL.CGAL_Kernel import Point_2# , Site_2

        import CGAL.CGAL_Apollonius_Graph_2 as Apollonius_Graph_2
        cgal_bindings = 'old'
    except ImportError:
        pass
    if cgal_bindings is None:
        # let it propagate out
        from CGAL.CGAL_Kernel import Point_2
        from CGAL.CGAL_Apollonius_Graph_2 import Apollonius_Graph_2,Site_2
        # print "Has new bindings"
        cgal_bindings = 'new'
    
    has_apollonius=True
    class ApolloniusField(XYZField):
        """ Takes a set of vertices and the allowed scale at each, and
        extrapolates across the plane based on a uniform telescoping rate
        """

        # Trying to optimize some -
        #   it segfault under the conditions:
        #      locality on insert
        #      locality on query
        #      redundant_factor = 0.9
        #      quantize=True/False

        # But it's okay if redundant factor is None

        # These are disabled while debugging the hangs on CGAL 4.2
        # with new bindings
        # enable using the last insert as a clue for the next insert
        locality_on_insert = False # True
        # enable using the last query as a clue for the next query
        locality_on_query = False # True
        quantize=False
        
        def __init__(self,X,F,r=1.1,redundant_factor=None):
            """
            redundant_factor: if a point being inserted has a scale which is larger than the redundant_factor
            times the existing scale at its location, then don't insert it.  So typically it would be something
            like 0.95, which says that if the existing scale at X is 100, and this point has a scale of 96, then
            we don't insert.
            """
            XYZField.__init__(self,X,F)
            self.r = r
            self.redundant_factor = redundant_factor
            self.construct_apollonius_graph()

        # Pickle support -
        def __getstate__(self):
            """ the CGAL.ApolloniusGraph can't be pickled - have to recreate it
            """
            d = self.__dict__.copy()
            d['ag'] = 'recreate'
            d['last_inserted'] = None
            d['last_query_vertex'] = None
            return d
        def __setstate__(self,d):
            self.__dict__.update(d)
            self.construct_apollonius_graph()

        def construct_apollonius_graph(self,quantize=False):
            """
            quantize: coordinates will be truncated to integers.  Not sure why this is relevant -
            might make it faster or more stable??  pretty sure that repeated coordinates will
            keep only the tightest constraint
            """
            self.quantize = quantize
            if len(self.X) > 0:
                self.offset = self.X.mean(axis=0)
            else:
                self.offset = np.zeros(2)

            print("Constructing Apollonius Graph.  quantize=%s"%quantize)
            self.ag = ag = Apollonius_Graph_2()
            self.last_inserted = None

            # if self.redundant_factor is not None:
            self.redundant = np.zeros(len(self.X),bool8)
                
            for i in range(len(self.X)):
                if i % 100 == 0:
                    print(" %8i / %8i"%(i,len(self.X)))
                self.redundant[i] = not self.insert(self.X[i],self.F[i])
            print("Done!")

        def insert(self,xy,f):
            """ directly insert a point into the Apollonius graph structure
            note that this may be used to incrementally construct the graph,
            if the caller doesn't care about the accounting related to the
            field -
            returns False if redundant checks are enabled and the point was
            deemed redundant.
            """
            x,y = xy - self.offset
            # This had been just -self.F[i], but I think that was wrong.
            w = -(f / (self.r-1.0) )
            if self.quantize:
                x = int(x)
                y = int(y)

            pnt = Point_2(x,y)
            ##
            if self.redundant_factor is not None:
                if self.ag.number_of_vertices() > 0:
                    existing_scale = self.value_at_point(pnt)
                    if self.redundant_factor * existing_scale < f:
                        return False
            ## 
            if self.locality_on_insert and self.last_inserted is not None:
                # generally the incoming data have some locality - this should speed things
                # up.
                try:
                    self.last_inserted = self.ag.insert(Site_2( pnt, w),self.last_inserted)
                except Exception: # no direct access to the real type, ArgumentError
                    print("CGAL doesn't have locality aware bindings.  This might be slower")
                    self.locality_on_insert=False
                    self.last_inserted = self.ag.insert(Site_2( pnt, w))
            else:
                s = Site_2(pnt,w)
                # print "AG::insert: %f,%f,%f"%(s.point().x(),s.point().y(),s.weight())
                #self.last_inserted = self.ag.insert(s)
                # try avoiding saving the result
                self.ag.insert(s)
                # retrieve it to see if it really got inserted like we think
                v = self.ag.nearest_neighbor(pnt)
                s = v.site()
                print("            %f,%f,%f"%(s.point().x(),s.point().y(),s.weight()))
            # it seems to crash if queries are allowed to retain this vertex handle -
            # probably the insertion can invalidate it
            self.last_query_vertex = None
            return True

        last_query_vertex = None
        def value_at_point(self,pnt):
            """ Like interpolate, but takes a CGAL point instead.  really just for the
            skip_redundant option, and called inside interpolate()
            """
            if self.ag.number_of_vertices() == 0:
                return np.nan
            
            if self.locality_on_query and self.last_query_vertex is not None:
                # exploit query locality
                try:
                    v = self.ag.nearest_neighbor(pnt,self.last_query_vertex)
                except Exception: # no direct access to the real type, ArgumentError
                    print("CGAL doesn't have locality aware query bindings.  May be slower.")
                    self.locality_on_query = False
                    v = self.ag.nearest_neighbor(pnt)
            else:
                v = self.ag.nearest_neighbor(pnt)
            self.last_query_vertex = v
            site = v.site()
            dist = np.sqrt( (pnt.x() - site.point().x())**2 +
                            (pnt.y() - site.point().y())**2   )
            # before this didn't have the factor dividing site.weight()
            f = -( site.weight() * (self.r-1.0) ) + dist*(self.r-1.0)
            return f

        def interpolate(self,X):
            newF = np.zeros( X.shape[0], np.float64 )

            for i in range(len(X)):
                x,y = X[i] - self.offset
                # remember, the slices are y, x
                p = Point_2(x,y)
                newF[i] = self.value_at_point(p)

            return newF

        def to_grid(self,nx=2000,ny=2000,interp='apollonius',bounds=None):
            if bounds is not None:
                if len(bounds) == 2:
                    extents = [bounds[0],bounds[2],bounds[1],bounds[3]]
                else:
                    extents = bounds
            else:
                extents = self.bounds()

            if interp!='apollonius':
                print("NOTICE: Apollonius graph was asked to_grid using '%s'"%interp)
                return XYZField.to_grid(self,nx,ny,interp)
            else:
                x = np.linspace(extents[0],extents[1],nx)
                y = np.linspace(extents[2],extents[3],ny)

                griddedF = np.zeros( (len(y),len(x)), np.float64 )

                for xi in range(len(x)):
                    for yi in range(len(y)):
                        griddedF[yi,xi] = self( [x[xi],y[yi]] )

                return SimpleGrid(extents,griddedF)

        @staticmethod 
        def read_shps(shp_names,value_field='value',r=1.1,redundant_factor=None):
            """ Read points or lines from a list of shapefiles, and construct
            an apollonius graph from the combined set of features.  Lines will be
            downsampled at the scale of the line.
            """
            lines=[]
            values=[]
            
            for shp_name in shp_names:
                print("Reading %s"%shp_name)

                ods = ogr.Open(shp_name)

                layer = ods.GetLayer(0)

                while 1:
                    feat = layer.GetNextFeature()
                    if feat is None:
                        break

                    geo = wkb.loads(feat.GetGeometryRef().ExportToWkb())

                    lines.append(np.array(geo.coords))
                    values.append(feat.GetField(value_field))
            return ApolloniusField.from_polylines(lines,values,
                                                  r=r,redundant_factor=redundant_factor)

        @staticmethod
        def from_polylines(lines,values,r=1.1,redundant_factor=None):
            X = []
            F = []
            edges = []

            for coords,value in zip(lines,values):
                if len(coords) > 1: # it's a line - upsample
                    # need to say closed_ring=0 so it doesn't try to interpolate between
                    # the very last point back to the first
                    coords = upsample_linearring(coords,value,closed_ring=0)
                if all(coords[-1]==coords[0]):
                    coords = coords[:-1]

                # remove duplicates:
                mask = all(coords[0:-1,:] == coords[1:,:],axis=1)
                if sum(mask)>0:
                    print("WARNING: removing duplicate points in shapefile")
                    print(coords[mask])
                    coords = coords[~mask]

                X.append( coords )
                F.append( value*np.ones(len(coords)) )
                
            X = concatenate( X )
            F = concatenate( F )
            return ApolloniusField(X=X,F=F,r=r,redundant_factor=redundant_factor)
    
except ImportError:
    #print "CGAL unavailable."
    pass
except AttributeError:
    # print("You have CGAL, but no Apollonius Graph bindings - auto-telescoping won't work")
    pass

if not has_apollonius:
    has_apollonius=True
    log.debug("Falling back to slow python implementation of ApolloniusField")
    ApolloniusField=PyApolloniusField

class ConstrainedScaleField(XYZField):
    """ Like XYZField, but when new values are inserted makes sure that
    neighboring nodes are not too large.  If an inserted scale is too large
    it will be made smaller.  If a small scale is inserted, it's neighbors
    will be checked, and made smaller as necessary.  These changes are
    propagated to neighbors of neighbors, etc.

    As points are inserted, if a neighbor is far enough away, this will
    optionally insert new points along the edges connecting with that neighbor
    to limit the extent that the new point affects too large an area
    """
    r=1.1 # allow 10% growth per segment

    def check_all(self):
        t = self.tri()
        edges = t.edge_db

        Ls = np.sqrt( (t.x[edges[:,0]] - t.x[edges[:,1]])**2 +
                      (t.y[edges[:,0]] - t.y[edges[:,1]])**2  )
        dys = self.F[edges[:,0]] - self.F[edges[:,1]]
        
        slopes = abs(dys / Ls)

        if any(slopes > self.r-1.0):
            bad_edges = np.where(slopes > self.r-1.0)[0]
            
            print("Bad edges: ")
            for e in bad_edges:
                a,b = edges[e]
                if self.F[a] > self.F[b]:
                    a,b = b,a
                
                L = np.sqrt( (t.x[a]-t.x[b])**2 + (t.y[a]-t.y[b])**2 )
                allowed = self.F[a] + L*(self.r - 1.0)
                print("%d:%f --[L=%g]-- %d:%f > %f"%(a,self.F[a],
                                                     L,
                                                     b,self.F[b],
                                                     allowed))
                print("  " + str( edges[e] ))
            return False
        return True

    # how much smaller than the 'allowed' value to make nodes
    #  so if the telescope factor says that the node can be 10m,
    #  we'll actually update it to be 8.5m
    safety_factor = 0.85
    
    def add_point(self,pnt,value,allow_larger=False):
        accum = [] # accumulates a list of ( [x,y], scale ) tuples for limiter points
        
        # before adding, see if there is one already in there that's close by
        old_value = self(pnt)

        if old_value < 0:
            print("  count of negative values: ",sum(self.F < 0))
            print("  point in question: ",pnt)
            print("  old_value",old_value)
            fg = self.to_grid(1000,1000)
            fg.plot()
            global bad
            bad = self
            raise Exception("Old value at new point is negative!")

        if not allow_larger and (value > old_value):
            print("Not adding this point, because it is actually larger than existing ones")
            return None

        ## ! Need to be careful about leaning to hard on old_value -
        #    the nearest neighbors interpolation doesn't guarantee the same value
        #    as linear interpolation between nodes ( I think ), so it's possible for
        #    things to look peachy keen from the nn interp but when comparing along edges
        #    it starts looking worse.

        ## STATUS
        #  I think the order of adding intermediate points needs to change.
        #  maybe we add the starting point, using it's old_value
        #  then look at its neighbors... confused...
        
        print("-----------Adding point: %s %g=>%g-----------"%(pnt,old_value,value))
        
        j = self.nearest(pnt)
        dist = np.sqrt( sum((self.X[j] - pnt)**2) )
        if dist < 0.5*value:
            i = j
            print("add_point redirected, b/c a nearby point already exists.")
            # need an extra margin of safety here -
            #   we're updating a point that is dist away, and we need the scale
            #   right here to be value.  
            F_over_there = value - dist*(self.r-1.0)
            if F_over_there < self.F[i]:
                self.F[i] = self.safety_factor * F_over_there
                print("  updating value of %s to %f"%(i,self.F[i]))
                self.check_scale(i,old_value = old_value)
        else:
            i = XYZField.add_point(self,pnt,value)
            print("  inserted %d with value %f"%(i,self.F[i]))
            # these are the edges in which the new node participates
            self.check_scale(i,old_value=old_value)

        return i

    def check_scale(self,i,old_value=None):
        """
        old_value: if specified, on each edge, if the neighbor is far enough away, insert
        a new node along the edge at the scale that it would have been if we hadn't
        adjusted this node
        """
        # print "Check scale of %s"%i
        
        # First, make sure that we are not too large for any neighbors:
        t = self.tri()
        edges = np.where( t.edge_db == i )[0]
        for e in edges:
            a,b = t.edge_db[e]
            # enforce that a is the smaller of the two
            if self.F[a] > self.F[b]:
                a,b = b,a
            # this time around, we only care about places where i is the larger
            if a==i:
                continue
            
            L= np.sqrt( (t.x[a] - t.x[b])**2 + (t.y[a] - t.y[b])**2 )

            A = self.F[a]
            B = self.F[b]

            allowed = A + L*(self.r-1.0)

            if B > allowed:
                # print "Had to adjust down the requested scale of point"
                self.F[b] = self.safety_factor*allowed

        # Now we know that the new point is not too large for anyone - see if any of
        # it's neighbors are too small.
        
        to_visit = [ (i,old_value) ]
        to_add = []

        orig_i = i
        
        # used to be important for this to be breadth-first...
        # also, this whole thing is hack-ish.  
        while len(to_visit) > 0:
            i,old_value = to_visit.pop(0)

            t = self.tri()
            edges = np.where( t.edge_db == i )[0]

            for e in edges:
                a,b = t.edge_db[e]

                # Make b the one that is not i
                if b==i:
                    a,b = b,a

                # print "From a=%d visiting b=%d"%(a,b)

                # So we are checking on point b, having come from a, but
                # ultimately we just care whether b is valid w.r.t orig_i
                
                # print "Checking on edge ",a,b
                L  = np.sqrt( (t.x[orig_i] - t.x[b])**2 + (t.y[orig_i] - t.y[b])**2 )
                La = np.sqrt( (t.x[a] - t.x[b])**2 + (t.y[a] - t.y[b])**2 )
                # print "    Length is ",L

                ORIG = self.F[orig_i]
                A = self.F[a] # 
                B = self.F[b]
                # print "    Scales A(%d)=%g  B(%d)=%g"%(a,A,b,B)

                allowed = min( ORIG + L*(self.r-1.0),
                               A    + La*(self.r-1.0) )
                
                # print "    Allowed from %d or %d is B: %g"%(orig_i,a,allowed)
                
                if B > allowed:
                    self.F[b] = self.safety_factor * allowed # play it safe...
                    # print "  Updating B(%d) to allowed scale %f"%(b,self.F[b])
                    to_visit.append( (b,B) )
                # elif (B < 0.8*allowed) and (old_value is not None) and (A<0.8*old_value) and (L > 5*A):
                elif (B>A) and (old_value is not None) and (A<0.8*old_value) and (L>5*A):
                    # the neighbor was significantly smaller than the max allowed,
                    # so we should limit the influence of this new point.
                    #
                    # used to be a safety_factor*allowed here, now just allowed...
                    alpha = (old_value - A) / (old_value - A + allowed - B)
                    if alpha < 0.65:
                        # if the intersection is close to B, don't bother...
                        new_point = alpha*self.X[b] + (1-alpha)*self.X[a]
                        # another 0.99 just to be safe against rounding
                        # 
                        # New approach: use the distance to original point
                        newL = np.sqrt( (t.x[orig_i] - new_point[0])**2 + (t.y[orig_i] - new_point[1])**2 )

                        # constrained by valid value based on distance from starting point as well as
                        # the old value 
                        new_value = min(ORIG + 0.95*newL*(self.r-1.0), # allowed value 
                                        0.99*(alpha*B + (1-alpha)*old_value) ) # value along the old line

                        # print "INTERMEDIATE:"
                        # print "  old_value at A: %g"%old_value
                        # print "  new value at A: %g"%A
                        # print "  curr value at B: %g"%B
                        # print "  allowed at B: %g"%allowed
                        # print "  alpha from A: %g"%alpha
                        # print "  new value for interpolated point: %g"%new_value
                        # 
                        # print "Will add intermediate point %s = %g"%(new_point,new_value)
                        to_add.append( (new_point, new_value) )

                    
        print("Adding %d intermediate points"%len(to_add))
        for p,v in to_add:
            if v < 0:
                raise Exception("Value of intermediate point is negative")
            i = self.add_point(p+0.01*v,v,allow_larger=1)
            # print "added intermediate point ",i


    def remove_invalid(self):
        """ Remove nodes that are too big for their delaunay neighbors
        """
        while 1:
            t = self.tri()
            edges = t.edge_db

            Ls = np.sqrt( (t.x[edges[:,0]] - t.x[edges[:,1]])**2 +
                          (t.y[edges[:,0]] - t.y[edges[:,1]])**2  )
            dys = self.F[edges[:,0]] - self.F[edges[:,1]]

            slopes = (dys / Ls)

            bad0 = slopes > self.r-1.0
            bad1 = (-slopes) > self.r-1.0

            bad_nodes = union1d( edges[bad0,0], edges[bad1,1] )
            if len(bad_nodes) == 0:
                break
            print("Removing %d of %d"%(len(bad_nodes),len(self.F)))

            to_keep = np.ones(len(self.F),bool)
            to_keep[bad_nodes] = False

            self.F = self.F[to_keep]
            self.X = self.X[to_keep]

            self._tri = None
            self._nn_interper = None
            self._lin_interper = None
            self.index = None

        

            

class XYZText(XYZField):
    def __init__(self,fname,sep=None,projection=None):
        self.filename = fname
        fp = open(fname,'rt')

        data = np.array([list(map(float,line.split(sep))) for line in fp])
        fp.close()

        XYZField.__init__(self,data[:,:2],data[:,2],projection=projection)



## The rest of the density field stuff:
class ConstantField(Field):
    def __init__(self,c):
        self.c = float(c)
        Field.__init__(self)
        
    def value(self,X):
        X=np.asanyarray(X)
        return self.c * np.ones(X.shape[:-1])
        

class BinopField(Field):
    """ Combine arbitrary fields with binary operators """
    def __init__(self,A,op,B):
        Field.__init__(self)
        self.A = A
        self.op = op
        self.B = B

    def __getstate__(self):
        d = self.__dict__.copy()
        d['op'] = self.op2str()
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.op = self.str2op(self.op)

    # cross your fingers...
    def op2str(self):
        return self.op.__name__
    def str2op(self,s):
        return eval(s)
    
        
    def value(self,X):
        try: # if isinstance(self.A,Field):
            a = self.A.value(X)
        except: # FIX - masks errors!
            a = self.A
            
        try: # if isinstance(self.B,Field):
            b = self.B.value(X)
        except: # FIX - masks errors!
            b = self.B
            
        return self.op(a,b)



class Field3D(Field):
    pass

class ZLevelField(Field3D):
    """ One representation of a 3-D field.
    We have a set of XY points and a water column associated with each.
    Extrapolation pulls out the closest water column, and extends the lowest
    cell if necessary.
    """
    def __init__(self,X,Z,F):
        Field3D.__init__(self)

        self.X = X
        self.Z = Z
        self.F = ma.masked_invalid(F)

        # 2-D index:
        self.surf_field = XYZField(self.X,np.arange(len(self.X)))
        self.surf_field.build_index()

    def shift_z(self,delta_z):
        self.Z += delta_z
    
    def distance_along_transect(self):
        d = (diff(self.X,axis=0)**2).sum(axis=1)**0.5
        d = d.cumsum()
        d = concatenate( ([0],d) )
        return d
        
    def plot_transect(self):
        """ Plots the data in 2-D as if self.X is in order as a transect.
        The x axis will be distance between points.  NB: if the data are not
        organized along a curve, this plot will make no sense!
        """
        x = self.distance_along_transect()
        
        meshY,meshX = np.meshgrid(self.Z,x)
        all_x = meshX.ravel()
        all_y = meshY.ravel()
        all_g = transpose(self.F).ravel()

        if any(all_g.mask):
            valid = ~all_g.mask

            all_x = all_x[valid]
            all_y = all_y[valid]
            all_g = all_g[valid]
        scatter(all_x,all_y,60,all_g,linewidth=0)

    def plot_surface(self):
        scatter(self.X[:,0],self.X[:,1],60,self.F[0,:],linewidth=0)

    _cached = None # [(x,y),idxs]
    def extrapolate(self,x,y,z):
        pnt = np.array([x,y])
        if self._cached is not None  and (x,y) == self._cached[0]:
            idxs = self._cached[1]
        else:
            # find the horizontal index:
            count = 4
            idxs = self.surf_field.nearest(pnt,count)
            self._cached = [ (x,y), idxs]
        
        zi = searchsorted( self.Z,z)
        if zi >= len(self.Z):
            zi = len(self.Z) - 1

        vals = self.F[zi,idxs]
        
        weights = 1.0 / ( ((pnt - self.X[idxs] )**2).sum(axis=1)+0.0001)

        val = (vals*weights).sum() / weights.sum()
        return val
    

# from pysqlite2 import dbapi2 as sqlite
# 
# class XYZSpatiaLite(XYZField):
#     """ Use spatialite as a backend for storing an xyz field
#     """
#     def __init__(self,fname,src=None):
#         self.conn = sqlite.connect(fname)
#         self.conn.enable_load_extension(1)
#         self.curs = self.conn.cursor()
#         self.curs.execute("select load_extension('/usr/local/lib/libspatialite.so')")
# 
#         self.ensure_schema()
#         
#         if src:
#             self.load_from_field(src)
# 
#     schema = """
#     create table points (id, geom ..."""
#     def ensure_schema(self):
#         pass
#     
            

    

class QuadrilateralGrid(Field):
    """ Common code for grids that store data in a matrix
    """
    def to_xyz(self):
        xyz = self.xyz()

        good = ~np.isnan(xyz[:,2])

        return XYZField( xyz[good,:2], xyz[good,2], projection = self.projection() )

class CurvilinearGrid(QuadrilateralGrid):
    def __init__(self,X,F,projection=None):
        """ F: 2D matrix of data values
            X: [Frows,Fcols,2] matrix of grid locations [x,y]
            Assumes that the grid is reasonable (i.e. doesn't have intersecting lines
            between neighbors)
        """
        QuadrilateralGrid.__init__(self,projection=projection)
        self.X = X
        self.F = F

    def xyz(self):
        """ unravel to a linear sequence of points
        """
        xyz = np.zeros( (self.F.shape[0] * self.F.shape[1], 3), np.float64 )
        
        xyz[:,:2] = self.X.ravel()
        xyz[:,2] = self.F.ravel()

        return xyz

    @with_plt
    def plot(self,**kwargs):
        # this is going to be slow...
        self.scatter = plt.scatter( self.X[:,:,0].ravel(),
                                    self.X[:,:,1].ravel(),
                                    c=self.F[:,:].ravel(),
                                    antialiased=False,marker='s',lod=True,
                                    lw=0,**kwargs )
        
    def apply_xform(self,xform):
        new_X = self.X.copy()

        print("Transforming points")
        for row in range(new_X.shape[0]):
            print(".")
            for col in range(new_X.shape[1]):
                new_X[row,col,:] = xform.TransformPoint(*self.X[row,col])[:2]
        print("Done transforming points")
                        

        # projection should get overwritten by the caller
        return CurvilinearGrid(new_X,self.F,projection='reprojected')
                
    def bounds(self):
        xmin = self.X[:,:,0].min()
        xmax = self.X[:,:,0].max()
        ymin = self.X[:,:,1].min()
        ymax = self.X[:,:,1].max()

        return (xmin,xmax,ymin,ymax)

    # cross-grid arithmetic.  lots of room for optimization...

    def regrid(self,b,interpolation='nearest'):
        """ returns an F array corresponding to the field B interpolated
        onto our grid
        """

        X = self.X.reshape( (-1,2) )

        newF = b.interpolate(X,interpolation=interpolation)
        
        return newF.reshape( self.F.shape )
        
    def __sub__(self,b):
        if isinstance(b,CurvilinearGrid) and id(b.X) == id(self.X):
            print("Reusing this grid.")
            Fb = self.F - b.F
        else:
            Fb = self.regrid( b )
            Fb = self.F - Fb

        return CurvilinearGrid(X=self.X, F= Fb, projection=self.projection() )

class SimpleGrid(QuadrilateralGrid):
    """
    A spatial field stored as a regular cartesian grid.
    The spatial extent of the field is stored in self.extents 
    (as xmin,xmax,ymin,ymax) and the data in the 2D array self.F
    """
    int_nan = -9999

    # Set to "linear" to have value() calls use linear interpolation
    default_interpolation = "linear"
    dx=None
    dy=None
    def __init__(self,extents,F,projection=None,dx=None,dy=None):
        """ extents: minx, maxx, miny, maxy
            NB: these are node-centered values, so if you're reading in
            pixel-based data where the dimensions are given to pixel edges,
            be sure to add a half pixel.
        """
        self.extents = extents
        self.F = F

        QuadrilateralGrid.__init__(self,projection=projection)

        if dx is not None:
            self.dx=dx
        if dy is not None:
            self.dy=dy

        self.delta() # compute those if unspecified

    @classmethod
    def from_curvilinear(cls, x, y, F):
        all_dx=np.diff(x,axis=1)
        if not np.allclose(all_dx[0], all_dx):
            raise Exception("Not evenly spaced in x")
        all_dy=np.diff(y,axis=0)
        if not np.allclose(all_dy[0], all_dy):
            raise Exception("Not evenly spaced in y")
        skew_x=np.diff(x,axis=0)
        if not np.all(skew_x==0.0):
            raise Exception("Skewed in x")
        skew_y=np.diff(y,axis=1)
        if not np.all(skew_y==0.0):
            raise Exception("Skewed in y")
        return SimpleGrid(extents=[x.min(), x.max(), y.min(), y.max()], F=F)

    @classmethod
    def zeros(cls,extents,dx,dy,dtype=np.float64):
        nx=int( np.ceil((extents[1] - extents[0])/dx) )
        ny=int( np.ceil((extents[3] - extents[2])/dy) )
        
        F=np.zeros((ny,nx),dtype=dtype)
        return cls(extents,F=F)
    
    @property
    def shape(self):
        return self.F.shape
    
    def copy(self):
        return SimpleGrid(extents=list(self.extents),F=self.F.copy(),projection=self.projection())

    def delta(self):
        """
        x and y pixel spacing.  If these are not already set (in self.dx, self.dy)
        compute from extents and F.
        For zero or singleton dimensions the spacing is set to zero.
        """
        if self.dx is None:
            if self.F.shape[1]>1:
                self.dx = (self.extents[1] - self.extents[0]) / (self.F.shape[1]-1.0)
            else:
                self.dx = 0.0
        if self.dy is None:
            assert self.F.shape[0]
            if self.F.shape[0]>1:
                self.dy = (self.extents[3] - self.extents[2]) / (self.F.shape[0]-1.0)
            else:
                self.dy = 0.0

        return self.dx,self.dy

    def trace_contour(self,vmin,vmax,union=True,method='mpl',
                      gdal_contour='gdal_contour'):
        """
        Trace a filled contour between vmin and vmax, returning
        a single shapely geometry (union=True) or a list of
        polygons (union=False).
        Uses matplotlib to do the actual contour construction.

        Note that matplotlib is not infallible here, and complicated
        or large inputs can create erroneous output.  gdal_contour
        might help.

        To use gdal_contour instead, pass method='gdal', and optionally
        specify the path to the gdal_contour executable. This currently
        behaves differently than the mpl approach. Here vmin is traced,
        and vmax is ignored. This should be harmonized at some point. TODO
        """
        if method=='mpl':
            cset=self.contourf([vmin,vmax],ax='hidden')
            segs=cset.allsegs
            geoms=[]
            for seg in segs[0]:
                if len(seg)<3: continue
                geoms.append( geometry.Polygon(seg) )
        elif method=='gdal': 
            import tempfile
            (fd1,fname_tif)=tempfile.mkstemp(suffix=".tif")
            (fd2,fname_shp)=tempfile.mkstemp(suffix=".shp")
            os.unlink(fname_shp)
            os.close(fd1)
            os.close(fd2)
            self.write_gdal(fname_tif,overwrite=True)
            res=subprocess.run([gdal_contour,"-fl",str(vmin),str(vmax),fname_tif,fname_shp],
                                capture_output=True)
            print(res.stdout)
            print(res.stderr)
            geoms=wkb2shp.shp2geom(fname_shp)['geom']
            union=False
        if union:
            poly=geoms[0]
            for geo in geoms[1:]:
                poly=poly.union(geo)
            return poly
        else:
            return geoms

    @with_plt
    def contourf(self,*args,**kwargs):
        X,Y = self.XY()
        ax=kwargs.pop('ax',None)
        if ax=='hidden':
            tmp_ax=True
            fig=plt.figure(999)
            ax=fig.gca()
        else:
            tmp_ax=False
            ax=ax or plt.gca()
        cset=ax.contourf(X,Y,self.F,*args,**kwargs)
        if tmp_ax:
            plt.close(fig)
        return cset
    @with_plt
    def contour(self,*args,**kwargs):
        X,Y = self.XY()
        ax=kwargs.pop('ax',None) or plt.gca()
        return ax.contour(X,Y,self.F,*args,**kwargs)
    @with_plt
    def plot(self,**kwargs):
        F=kwargs.pop('F',self.F)
        func=kwargs.pop('func',lambda x:x)
        F=func(F)
        
        dx,dy = self.delta()

        maskedF = ma.array(self.F,mask=np.isnan(F))

        if 'ax' in kwargs:
            kwargs = dict(kwargs)
            ax = kwargs['ax']
            del kwargs['ax']
            ims = ax.imshow
        else:
            ims = plt.imshow

        if 'offset' in kwargs:
            offset=kwargs.pop('offset')
        else:
            offset=[0,0]

        return ims(maskedF,origin='lower',
                   extent=[self.extents[0]-0.5*dx + offset[0], self.extents[1]+0.5*dx + offset[0],
                           self.extents[2]-0.5*dy + offset[1], self.extents[3]+0.5*dy + offset[1]],
                   **kwargs)

    def xy(self):
        x = np.linspace(self.extents[0],self.extents[1],self.F.shape[1])
        y = np.linspace(self.extents[2],self.extents[3],self.F.shape[0])
        return x,y

    def XY(self):
        X,Y = np.meshgrid(*self.xy())
        return X,Y

    def xyz(self):
        """ unravel to a linear sequence of points
        """
        X,Y = self.XY()
        xyz = np.zeros( (self.F.shape[0] * self.F.shape[1], 3), np.float64 )

        xyz[:,0] = X.ravel()
        xyz[:,1] = Y.ravel()
        xyz[:,2] = self.F.ravel()

        return xyz

    def to_xyz(self):
        """  The simple grid version is a bit smarter about missing values,
        and tries to avoid creating unnecessarily large intermediate arrays
        """
        x,y = self.xy()

        if hasattr(self.F,'mask') and self.F.mask is not False:
            self.F._data[ self.F.mask ] = np.nan
            self.F = self.F._data

        if self.F.dtype in (np.int16,np.int32):
            good = (self.F != self.int_nan)
        else:
            good = ~np.isnan(self.F)

        i,j = np.where(good)

        X = np.zeros( (len(i),2), np.float64 )
        X[:,0] = x[j]
        X[:,1] = y[i]

        return XYZField( X, self.F[good], projection = self.projection() )

    def to_curvilinear(self):
        X,Y = self.XY()
        XY = concatenate( ( X[:,:,None], Y[:,:,None]), axis=2)

        cgrid = CurvilinearGrid(XY,self.F)
        return cgrid

    def to_unstructured(self,node_var='z',trim=np.isnan):
        """
        Generate a rectilinear UnstructuredGrid, with the field value
        saved on nodes[node_var].
        trim: a function applied to the node values where, if true, the node should
          be deleted. Must be vectorized.
        """
        from ..grid import unstructured_grid
        g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                             extra_node_fields=[(node_var,self.F.dtype)])

        X,Y=self.XY()
        # will update coordinates afterwards
        maps=g.add_rectilinear(p0=[0,0],p1=[1,1],nx=X.shape[0],ny=X.shape[1])

        g.nodes['x'][maps['nodes'],0]=X
        g.nodes['x'][maps['nodes'],1]=Y
        g.nodes[node_var][maps['nodes']]=self.F

        # TODO: Possible that cells need to be re-oriented.
        # Should test area for one cell...

        if trim is not None:
            to_trim=trim(g.nodes[node_var])
            for n in np.nonzero(to_trim)[0]:
                g.delete_node_cascade(n)
            g.renumber()

        return g

    def apply_xform(self,xform):
        # assume that the transform is not a simple scaling in x and y,
        # so we have to switch to a curvilinear grid.
        cgrid = self.to_curvilinear()

        return cgrid.apply_xform(xform)

    def xy_to_indexes(self,xy):
        dx,dy = self.delta()
        row = int( np.round( (xy[1] - self.extents[2]) / dy ) )
        col = int( np.round( (xy[0] - self.extents[0]) / dx ) )
        return row,col
        
    def rect_to_indexes(self,xxyy):
        if len(xxyy)==2:
            xxyy=[xxyy[0][0],xxyy[1][0],xxyy[0][1],xxyy[1][1]]
        
        xmin,xmax,ymin,ymax = xxyy

        dx,dy = self.delta()

        min_col = int( max( np.floor( (xmin - self.extents[0]) / dx ), 0) )
        max_col = int( min( np.ceil( (xmax - self.extents[0]) / dx ), self.F.shape[1]-1) )

        min_row = int( max( np.floor( (ymin - self.extents[2]) / dy ), 0) )
        max_row = int( min( np.ceil( (ymax - self.extents[2]) / dy ), self.F.shape[0]-1) )

        return [min_row,max_row,min_col,max_col]

    def crop(self,rect=None,indexes=None):
        if rect is not None:
            indexes=self.rect_to_indexes(rect)

        assert indexes is not None,"Must specify one of rect or indexes"

        min_row,max_row,min_col,max_col = indexes
        newF = self.F[min_row:max_row+1, min_col:max_col+1]
        new_extents = [self.extents[0] + min_col*self.dx,
                       self.extents[0] + max_col*self.dx,
                       self.extents[2] + min_row*self.dy,
                       self.extents[2] + max_row*self.dy ]

        result=SimpleGrid(extents = new_extents,
                          F = newF,
                          projection = self.projection(),
                          dx=self.dx,dy=self.dy)
        return result

    def bounds(self):
        return np.array(self.extents)

    def interpolate(self,X,interpolation=None,fallback=True):
        """ interpolation can be nearest or linear
        """
        X=np.asanyarray(X)
        
        if interpolation is None:
            interpolation = self.default_interpolation

        xmin,xmax,ymin,ymax = self.bounds()
        dx,dy = self.delta()

        if interpolation == 'nearest':
            # 0.49 will give us the nearest cell center.
            # recently changed X[:,1] to X[...,1] - hopefully will accomodate
            # arbitrary shapes for X
            rows = (0.49 + (X[...,1] - ymin) / dy).astype(np.int32)
            cols = (0.49 + (X[...,0] - xmin) / dx).astype(np.int32)
            bad = (rows<0) | (rows>=self.F.shape[0]) | (cols<0) | (cols>=self.F.shape[1])
        elif interpolation == 'linear':
            # for linear, we choose the floor() of both
            row_alpha = ((X[...,1] - ymin) / dy)
            col_alpha = ((X[...,0] - xmin) / dx)

            rows = row_alpha.astype(np.int32)
            cols = col_alpha.astype(np.int32)

            row_alpha -= rows # [0,1]
            col_alpha -= cols # [0,1]

            # and we need one extra on the high end
            bad = (rows<0) | (rows>=self.F.shape[0]-1) | (cols<0) | (cols>=self.F.shape[1]-1)
        else:
            raise Exception("bad interpolation type %s"%interpolation)

        if rows.ndim > 0:
            rows[bad] = 0
            cols[bad] = 0
        elif bad:
            rows = cols = 0

        if interpolation == 'nearest':
            result = self.F[rows,cols]
        else:
            result =   self.F[rows,cols]    *(1.0-row_alpha)*(1.0-col_alpha) \
                     + self.F[rows+1,cols]  *row_alpha      *(1.0-col_alpha) \
                     + self.F[rows,cols+1]  *(1.0-row_alpha)*col_alpha \
                     + self.F[rows+1,cols+1]*row_alpha      *col_alpha

        # It may have been an int field, and now we need to go to float and set some nans:
        if result.dtype in (int,np.int8,np.int16,np.int32,np.int64):
            print("Converting from %s to float"%result.dtype)
            result = result.astype(np.float64)
            result[ result==self.int_nan ] = np.nan
        if result.ndim>0:
            result[bad] = np.nan
        elif bad:
            result = np.nan

        # let linear interpolation fall back to nearest at the borders:
        if interpolation=='linear' and fallback and np.any(bad):
            result[bad] = self.interpolate(X[bad],interpolation='nearest',fallback=False)

        return result

    def value(self,X):
        return self.interpolate(X)

    def value_on_edge(self,e,samples=None,**kw):
        """ Return the value averaged along an edge - the generic implementation
        just takes 5 samples evenly spaced along the line, using value()
        """
        if samples is None:
            res = min(self.dx,self.dy)
            l = norm(e[1]-e[0])
            samples = int(np.ceil(l/res))

        return Field.value_on_edge(self,e,samples=samples,**kw)

    def upsample(self,factor=2):
        x = np.linspace(self.extents[0],self.extents[1],1+factor*(self.F.shape[1]-1))
        y = np.linspace(self.extents[2],self.extents[3],1+factor*(self.F.shape[0]-1))

        new_F = np.zeros( (len(y),len(x)) , np.float64 )

        for row in range(len(y)):
            for col in range(len(x)):
                new_F[row,col] = 0.25 * (self.F[row//2,col//2] +
                                         self.F[(row+1)//2,col//2] +
                                         self.F[row//2,(col+1)//2] +
                                         self.F[(row+1)//2,(col+1)//2])

        return SimpleGrid(self.extents,new_F)
    def downsample(self,factor,method='decimate'):
        """
        method: 'decimate' just takes every nth sample
                'ma_mean' takes the mean of n*n blocks, and is nan
                  and mask aware.
        """
        factor = int(factor)

        # use a really naive downsampling for now:
        if method=='decimate':
            new_F = np.array(self.F[::factor,::factor])
        elif method=='ma_mean':
            # if not isinstance(self.F,np.ma.core.MaskedArray):
            F=self.F
            nr,nc=F.shape
            nr+=(-nr)%factor # pad to even multiple
            nc+=(-nc)%factor # pad...
            F2=np.ma.zeros((nr,nc))
            F2[:]=np.nan
            F2[:F.shape[0],:F.shape[1]]=F
            F2=np.ma.masked_invalid(F2)
            F2=F2.reshape([nr//factor,factor,nc//factor,factor]) 
            F2=F2.transpose([0,2,1,3]).reshape([nr//factor,nc//factor,factor*factor])
            new_F=F2.mean(axis=2)
        else:
            assert False

        x,y = self.xy()

        new_x = x[::factor]
        new_y = y[::factor]

        new_extents = [x[0],x[-1],y[0],y[-1]]

        return SimpleGrid(new_extents,new_F)

    ## Methods to fill in missing data
    def fill_by_griddata(self):
        """ Basically griddata - limits the input points to the borders 
        of areas missing data.
        Fills in everything within the convex hull of the valid input pixels.
        """

        # Find pixels missing one or more neighbors:
        valid = np.isfinite(self.F)
        all_valid_nbrs = np.ones(valid.shape,'bool')
        all_valid_nbrs[:-1,:] &= valid[1:,:] # to the west
        all_valid_nbrs[1:,:] &=  valid[:-1,:] # to east
        all_valid_nbrs[:,:-1] &= valid[:,1:] # to north
        all_valid_nbrs[:,1:] &= valid[:,:-1] # to south

        missing_a_nbr = valid & (~ all_valid_nbrs )

        i,j = nonzero(missing_a_nbr)

        x = np.arange(self.F.shape[0])
        y = np.arange(self.F.shape[1])

        values = self.F[i,j]

        # Try interpolating the whole field - works, but slow...
        # some issue here with transpose.
        # x ~ 1470 - but it's really rows
        # y ~ 1519 - but it's really columns.
        # so griddata takes (xi,yi,zi, x,y)
        # but returns as rows,columns
        # fill_data ~ [1519,1470]
        # old way: fill_data = griddata(i,j,values,x,y)
        fill_data = griddata(j,i,values,y,x)

        self.F[~valid] = fill_data[~valid] # fill_data is wrong orientation

    # Is there a clever way to use convolution here -
    def fill_by_convolution(self,iterations=7,smoothing=0,kernel_size=3):
        """  Better for filling in small seams - repeatedly
        applies a 3x3 average filter.  On each iteration it can grow
        the existing data out by 2 pixels.
        Note that by default there is not 
        a separate smoothing process - each iteration will smooth
        the pixels from previous iterations, but a pixel that is set
        on the last iteration won't get any smoothing.

        Set smoothing >0 to have extra iterations where the regions are not
        grown, but the averaging process is reapplied.

        If iterations is 'adaptive', then iterate until there are no nans.
        """
        kern = np.ones( (kernel_size,kernel_size) )

        valid = np.isfinite(self.F) 

        bin_valid = valid.copy()
        # newF = self.F.copy()
        newF = self.F # just do it in place
        newF[~valid] = 0.0

        if iterations=='adaptive':
            iterations=1
            adaptive=True
        else:
            adaptive=False

        i = 0
        while i < iterations+smoothing:
            #for i in range(iterations + smoothing):

            weights = signal.convolve2d(bin_valid,kern,mode='same',boundary='symm')
            values  = signal.convolve2d(newF,kern,mode='same',boundary='symm')

            # update data_or_zero and bin_valid
            # so anywhere that we now have a nonzero weight, we should get a usable value.

            # for smoothing-only iterations, the valid mask isn't expanded
            if i < iterations:
                bin_valid |= (weights>0)

            to_update = (bin_valid & (~valid)).astype(bool)
            newF[to_update] = values[to_update] / weights[to_update]

            i+=1
            if adaptive and (np.sum(~bin_valid)>0):
                iterations += 1 # keep trying
            else:
                adaptive = False # we're done 

        # and turn the missing values back to nan's
        newF[~bin_valid] = np.nan

    def smooth_by_convolution(self,kernel_size=3,iterations=1):
        """
        Repeatedly apply a 3x3 average filter (or other size: kernel_size).
        Similar to the smoothing step of fill_by_convolution, except that
        the effect is applied everywhere, not just in the newly-filled
        areas.
        """
        kern = np.ones( (kernel_size,kernel_size) )

        valid = np.isfinite(self.F) 

        # avoid nan contamination - set these to zero
        self.F[~valid] = 0.0

        for i in range(iterations):
            weights = signal.convolve2d(valid, kern,mode='same',boundary='symm')
            values  = signal.convolve2d(self.F,kern,mode='same',boundary='symm')

            # update data_or_zero and bin_valid
            # so anywhere that we now have a nonzero weight, we should get a usable value.
            self.F[valid] = values[valid] / weights[valid]

        # and turn the missing values back to nan's
        self.F[~valid] = np.nan

    def xxyy_mask(self,xxyy):
        mask = np.full(self.F.shape, False)
        min_row,max_row,min_col,max_col = self.rect_to_indexes(xxyy)
        mask[min_row:max_row+1, min_col:max_col+1] = True
        return mask

    def polygon_mask(self,poly,crop=True,return_values=False):
        """ similar to mask_outside, but:
        much faster due to outsourcing tests to GDAL
        returns a boolean array same size as self.F, with True for 
        pixels inside the polygon.

        crop: if True, optimize by cropping the source raster first.  should
        provide identical results, but may not be identical due to roundoff.

        return_vales: if True, rather than returning a bitmask the same size
         as self.F, return just the values of F that fall inside poly. This
         can save space and time when just extracting a small set of values 
         from large raster
        """
        # could be made even simpler, by creating OGR features directly from the
        # polygon, rather than create a full-on datasource.
        # likewise, could jump straight to creating a target raster, rather 
        # than creating a SimpleGrid just to get the metadata right.
        if crop:
            xyxy=poly.bounds
            xxyy=[xyxy[0], xyxy[2], xyxy[1], xyxy[3]]
            indexes=self.rect_to_indexes(xxyy)
            cropped=self.crop(indexes=indexes)
            ret=cropped.polygon_mask(poly,crop=False,return_values=return_values)
            if return_values:
                return ret # done!
            else:
                mask_crop=ret
            full_mask=np.zeros(self.F.shape,bool)
            min_row,max_row,min_col,max_col = indexes
            full_mask[min_row:max_row+1,min_col:max_col+1]=mask_crop
            return full_mask

        from . import wkb2shp
        raster_ds=self.write_gdal('Memory')
        poly_ds=wkb2shp.wkb2shp("Memory",[poly])
        target_field=SimpleGrid(F=np.zeros(self.F.shape,np.int32),
                                extents=self.extents)
        target_ds = target_field.write_gdal('Memory')
        # write 1000 into the array where the polygon falls.
        gdal.RasterizeLayer(target_ds,[1],poly_ds.GetLayer(0),None,None,[1000],[])
        new_raster=GdalGrid(target_ds)

        ret=new_raster.F>0
        if return_values:
            return self.F[ret]
        else:
            return ret

    def mask_outside(self,poly,value=np.nan,invert=False,straddle=None):
        """ Set the values that fall outside the given polygon to the
        given value.  Existing nan values are untouched.

        Compared to polygon_mask, this is slow but allows more options on
        exactly how to test each pixel.

        straddle: if None, then only test against the center point
          if True: a pixel intersecting the poly, even if the center is not
          inside, is accepted.
          [future: False: reject straddlers]
        """
        if prep:
            poly = prep(poly)

        X,Y = self.xy()
        rect=np.array([[-self.dx/2.0,-self.dy/2.0],
                    [self.dx/2.0,-self.dy/2.0],
                    [self.dx/2.0,self.dy/2.0],
                    [-self.dx/2.0,self.dy/2.0]])
        for col in range(len(X)):
            # print("%d/%d"%(col,len(X)))
            for row in range(len(Y)):
                if np.isfinite(self.F[row,col]):
                    if straddle is None:
                        p = geometry.Point(X[col],Y[row])
                        if (not poly.contains(p)) ^ invert:# i hope that's the right logic
                            self.F[row,col] = value
                    elif straddle:
                        p = geometry.Polygon( np.array([X[col],Y[row]])[None,:] + rect )
                        if (not poly.intersects(p)) ^ invert:
                            self.F[row,col] = value

    def write(self,fname):
        fp = open(fname,'wb')

        pickle.dump( (self.extents,self.F), fp, -1)
        fp.close()

    def to_rgba(self,cmap='jet',vmin=None,vmax=None):
        """
        map scalar field to pseudocolor rgba.
        """
        if cm is None:
            raise Exception("No matplotlib - can't map to RGB")

        if vmin is None:
            vmin = self.F.min()
        if vmax is None:
            vmax = self.F.max()

        cmap=cm.get_cmap(cmap) # e.g. 'jet' => cm.jet
        
        invalid=np.isnan(self.F)
        fscaled = (self.F-vmin)/(vmax-vmin)
        fscaled[invalid]=0
        frgba = (cmap(fscaled)*255).astype(np.uint8)
        frgba[invalid,:3]=255
        frgba[invalid,3]=0
        
        return SimpleGrid(extents=self.extents,F=frgba,projection=self.projection())
        
    def write_gdal_rgb(self,output_file,**kw):
        if len(self.F.shape)==2:
            # As a convenience convert to RGBA then write
            return self.to_rgba(**kw).write_gdal_rgb(output_file)
            
        # Create gtif
        driver = gdal.GetDriverByName("GTiff")
        dst_ds = driver.Create(output_file, self.F.shape[1], self.F.shape[0], 4, gdal.GDT_Byte,
                               ["COMPRESS=LZW"])
        frgba=self.F
        # assumes that nodata areas are already transparent, or somehow dealt with.

        # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
        # Gdal wants pixel-edge extents, but what we store is pixel center extents...
        dx,dy = self.delta()

        # Some GDAL utilities function better if the output is in image coordinates, so flip back
        # if needed
        if dy > 0:
            # print "Flipping to be in image coordinates"
            dy = -dy
            frgba = frgba[::-1,:,:]

        dst_ds.SetGeoTransform( [ self.extents[0]-0.5*dx, dx,
                                  0, self.extents[3]-0.5*dy, 0, dy ] )

        # set the reference info
        if self.projection() is not None:
            srs = osr.SpatialReference()
            srs.SetWellKnownGeogCS(self.projection())
            dst_ds.SetProjection( srs.ExportToWkt() )

        # write the band
        for band in range(4):
            b1 = dst_ds.GetRasterBand(band+1)
            b1.WriteArray(frgba[:,:,band])
        dst_ds.FlushCache()

    gdalwarp = "gdalwarp" # path to command
    def warp_to_match(self,target):
        """
        Given a separte field trg, warp this one to match pixel for pixel.

        self and target should have meaningful projection().
        """
        # adjust for GDAL wanting to pixel edges, not
        # pixel centers
        halfdx = 0.5*target.dx
        halfdy = 0.5*target.dy
        te = "-te %f %f %f %f "%(target.extents[0]-halfdx,target.extents[2]-halfdy,
                                 target.extents[1]+halfdx,target.extents[3]+halfdy)
        ts = "-ts %d %d"%(target.F.T.shape)
        return self.warp(target.projection(),
                         extra=te + ts)

    def warp(self,t_srs,s_srs=None,fn=None,extra=[]):
        """ interface to gdalwarp
        t_srs: string giving the target projection
        s_srs: override current projection of the dataset, defaults to self._projection
        fn: if set, the result will retained, written to the given file.  Otherwise
          the transformation will use temporary files.        opts: other
        extra: other options to pass to gdalwarp. That used to be specified as
          a string, but now it should be a list of per-separated arguments.
        """
        # 2022-06-03: don't recall the reason for doing it this way.
        # but have to tell write_gdal to overwrite...
        tmp_src = tempfile.NamedTemporaryFile(suffix='.tif',delete=False)
        tmp_src_fn = tmp_src.name ; tmp_src.close()

        if fn is not None:
            tmp_dest_fn = fn
        else:
            tmp_dest  = tempfile.NamedTemporaryFile(suffix='.tif',delete=False)
            tmp_dest_fn = tmp_dest.name
            tmp_dest.close()

        if isinstance(extra,str):
            print("'extra' argument for warp should be a list now")
            extra=extra.split()
            
        s_srs = s_srs or self.projection()
        self.write_gdal(tmp_src_fn,overwrite=True)
        # Seems that windows does not like the quoting if it's all shoved into
        # a single string, with shell=True
        cmd=[self.gdalwarp,
             "-s_srs",s_srs,"-t_srs",t_srs,
             "-dstnodata","nan"] + extra +[tmp_src_fn,tmp_dest_fn]
                                       
        output=subprocess.check_output(cmd)
        self.last_warp_output=output # dirty, but maybe helpful

        result = GdalGrid(tmp_dest_fn)
        os.unlink(tmp_src_fn)
        if fn is None:
            try:
                os.unlink(tmp_dest_fn)
            except PermissionError:
                # appears to be a problem with running on Windows
                # file is in use by another process?
                print("Unable to delete temp file")
                print(tmp_dest_fn)
        return result

    def write_gdal(self,output_file,nodata=None,overwrite=False,options=None):
        """ Write a Geotiff of the field.

        if nodata is specified, nan's are replaced by this value, and try to tell
        gdal about it.

        if output_file is "Memory", will create an in-memory GDAL dataset and return it.
        """
        in_memory= (output_file=='Memory')

        if not in_memory:
            # Create gtif
            driver = gdal.GetDriverByName("GTiff")
            if options is None:
                options=["COMPRESS=LZW"]
                
            if os.path.exists(output_file):
                if overwrite:
                    os.unlink(output_file)
                else:
                    raise Exception("File %s already exists"%output_file)
        else:
            driver = gdal.GetDriverByName("MEM")
            if options is None:
                options=[]

        gtype = numpy_type_to_gdal[self.F.dtype.type]
        dst_ds = driver.Create(output_file, self.F.shape[1], self.F.shape[0], 1, gtype,
                               options)
        raster = self.F

        if nodata is not None:
            raster = raster.copy()
            raster[ np.isnan(raster) ] = nodata

        # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
        # Gdal wants pixel-edge extents, but what we store is pixel center extents...
        dx,dy = self.delta()

        # Some GDAL utilities function better if the output is in image coordinates, so flip back
        # if needed
        if dy > 0:
            # print "Flipping to be in image coordinates"
            dy = -dy
            raster = raster[::-1,:]

        dst_ds.SetGeoTransform( [ self.extents[0]-0.5*dx, dx,
                                  0, self.extents[3]-0.5*dy, 0, dy ] )

        # set the reference info
        if self.projection() not in ('',None):
            srs = osr.SpatialReference()
            if srs.SetFromUserInput(self.projection()) != 0:
                log.warning("Failed to set projection (%s) on GDAL output"%(self.projection()))
            dst_ds.SetProjection( srs.ExportToWkt() )

        # write the band
        b1 = dst_ds.GetRasterBand(1)
        if nodata is not None:
            b1.SetNoDataValue(nodata)
        else:
            # does this work?
            b1.SetNoDataValue(np.nan)
        b1.WriteArray(raster)
        if not in_memory:
            dst_ds.FlushCache()
        else:
            return dst_ds

    def point_to_index(self,X):
        X=np.asarray(X)
        x = (X[...,0]-self.extents[0])/self.dx
        y = (X[...,1]-self.extents[2])/self.dy
        return np.array([y,x]).T

    def extract_tile(self,xxyy=None,res=None,match=None,interpolation='linear',missing=np.nan):
        """ Create the requested tile
        xxyy: a 4-element sequence
        match: another field, assumed to be in the same projection, to match
          pixel for pixel.

        interpolation: 'linear','quadratic','cubic' will pass the corresponding order
           to RectBivariateSpline.
         'bilinear' will instead use simple bilinear interpolation, which has the
         added benefit of preserving nans.

        missing: the value to be assigned to parts of the tile which are not covered 
        by the source data.
        """
        if match is not None:
            xxyy = match.extents
            resx,resy = match.delta()
            x,y = match.xy()
        else:
            if res is None:
                resx = resy = self.dx
            else:
                resx = resy = res
            xxyy=as_xxyy(xxyy)
            x = np.arange(xxyy[0],xxyy[1]+resx,resx)
            y = np.arange(xxyy[2],xxyy[3]+resy,resy)

        myx,myy = self.xy()

        if interpolation == 'bilinear':
            F=self.F
            def interper(y,x):
                # this is taken from a stack overflow answer
                #  "simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python"
                # but altered so that x and y are 1-D arrays, and the result is a
                # 2-D array (x and y as in inputs to meshgrid)

                # scale those to float-valued indices into F
                x = (np.asarray(x)-self.extents[0])/self.dx
                y = (np.asarray(y)-self.extents[2])/self.dy

                x0 = np.floor(x).astype(int)
                x1 = x0 + 1
                y0 = np.floor(y).astype(int)
                y1 = y0 + 1

                x0 = np.clip(x0, 0, F.shape[1]-1)
                x1 = np.clip(x1, 0, F.shape[1]-1)
                y0 = np.clip(y0, 0, F.shape[0]-1)
                y1 = np.clip(y1, 0, F.shape[0]-1)

                Ia = F[ y0,:][:, x0 ]
                Ib = F[ y1,:][:, x0 ]
                Ic = F[ y0,:][:, x1 ]
                Id = F[ y1,:][:, x1 ]

                wa = (x1-x)[None,:] * (y1-y)[:,None]
                wb = (x1-x)[None,:] * (y-y0)[:,None]
                wc = (x-x0)[None,:] * (y1-y)[:,None]
                wd = (x-x0)[None,:] * (y-y0)[:,None]

                result = wa*Ia + wb*Ib + wc*Ic + wd*Id
                result[ y<0,: ] = missing
                result[ y>F.shape[0],: ] = missing
                result[ :, x<0 ] = missing
                result[ :, x>F.shape[1]] = missing

                return result
        else:
            k = ['constant','linear','quadratic','cubic'].index(interpolation)

            if np.any(np.isnan(self.F)):
                F = self.F.copy()
                F[ np.isnan(F) ] = 0.0
            else:
                F = self.F

            # Unfortunately this doesn't respect nan values in F
            interper = RectBivariateSpline(x=myy,y=myx,z=F,kx=k,ky=k)

        # limit to where we actually have data:
        # possible 0.5dx issues here
        xbeg,xend = np.searchsorted(x,self.extents[:2])
        ybeg,yend = np.searchsorted(y,self.extents[2:])
        Ftmp = np.ones( (len(y),len(x)),dtype=self.F.dtype)
        Ftmp[...] = missing
        # This might have some one-off issues
        Ftmp[ybeg:yend,xbeg:xend] = interper(y[ybeg:yend],x[xbeg:xend])
        return SimpleGrid(extents=xxyy,
                          F=Ftmp)

    def gradient(self):
        """ compute 2-D gradient of the field, returning a pair of fields of the
        same size (one-sided differences are used at the boundaries, central elsewhere).
        returns fields: dFdx,dFdy
        """
        # make it the same size, but use one-sided stencils at the boundaries
        dFdx = np.zeros(self.F.shape,np.float64)
        dFdy = np.zeros(self.F.shape,np.float64)

        # central difference in interior:
        dFdx[:,1:-1] = (self.F[:,2:] - self.F[:,:-2]) /(2*self.dx)
        dFdy[1:-1,:] = (self.F[2:,:] - self.F[:-2,:]) /(2*self.dy)

        # one-sided at boundaries:
        dFdx[:,0] = (self.F[:,1] - self.F[:,0])/self.dx
        dFdx[:,-1] = (self.F[:,-1] - self.F[:,-2])/self.dx
        dFdy[0,:] = (self.F[1,:] - self.F[0,:])/self.dy
        dFdy[-1,:] = (self.F[-1,:] - self.F[-2,:])/self.dy

        dx_field = SimpleGrid(extents = self.extents,F = dFdx)
        dy_field = SimpleGrid(extents = self.extents,F = dFdy)
        return dx_field,dy_field

    def hillshade_scalar(self,azimuth_deg=225,zenith_deg=45,z_factor=10):
        dx,dy=self.gradient()
        azimuth_rad=azimuth_deg*np.pi/180
        zenith_rad=zenith_deg*np.pi/180

        slope_rad = np.arctan( z_factor * np.sqrt( dx.F**2 + dy.F**2) )
        aspect_rad = np.arctan2(dy.F,-dx.F)
        hillshade=np.cos(zenith_rad)*np.cos(slope_rad) + \
                   np.sin(zenith_rad)*np.sin(slope_rad)*np.cos(azimuth_rad - aspect_rad)
        return SimpleGrid(F=hillshade,extents=self.extents)
    def hillshade_shader(self,**kwargs):
        hs=self.hillshade_scalar(**kwargs)
        Frgba=np.zeros( hs.F.shape + (4,), 'f4')
        Frgba[...,3] = 1-hs.F.clip(0,1)
        hs.F=Frgba
        return hs

    @with_plt
    def plot_hillshade(self,ax=None,plot_args={},**kwargs):
        shader=self.hillshade_shader(**kwargs)
        ax=ax or plt.gca()
        return shader.plot(ax=ax,**plot_args)

    def overlay_rgba(self,other):
        """
        Composite another field over self.
        Requires that self and other are rgba fields.

        in keeping with matplotlib rgba arrays, values can
        either be [0-1] floating point or [0-255] integer.

        other will be cast as needed to match self.

        other must have matching resolution and extents (this function does not
        currently resample to match self)
        """
        assert np.allclose( self.extents, other.extents)
        assert np.array_equal( self.F.shape, other.F.shape)
        assert self.F.shape[2]==4

        if np.issubdtype(self.F.dtype, np.floating):
            Fother=other.F
            if not np.issubdtype(Fother.dtype, np.floating):
                Fother=(Fother/255).clip(0,1.0)
            alpha=other.F[:,:,3]
            my_alpha=self.F[:,:,3]
            if my_alpha.min()==1.0:
                inv_alpha=1.0
            else:
                new_alpha=alpha + my_alpha*(1-alpha)
                inv_alpha=1./new_alpha
                inv_alpha[ new_alpha==0 ]=0                
        else:
            # integer
            Fother=other.F
            if np.issubdtype(Fother.dtype, np.floating):
                alpha=other.F[:,:,3]
                Fother=(Fother.clip(0,1)*255).astype(np.uint8)
                
            if self.F[:,:,3].min()==255:
                # Special case when background is opaque
                inv_alpha=1.0
            else:
                my_alpha=(self.F[:,:,3]/255.).clip(0,1)
                new_alpha=alpha + my_alpha*(1-alpha)
                inv_alpha=1./new_alpha
                inv_alpha[ new_alpha==0 ]=0
                self.F[:,:,3]=255*new_alpha

        for chan in range(3):
            self.F[:,:,chan] = (self.F[:,:,chan]*(1-alpha) + other.F[:,:,chan]*alpha) * inv_alpha
            
    @staticmethod
    def read(fname):
        fp = open(fname,'rb')

        extents, F = pickle.load( fp )
        fp.close()

        return SimpleGrid(extents=extents,F=F)


class GtxGrid(SimpleGrid):
    def __init__(self,filename,is_vertcon=False,missing=9999,projection='WGS84'):
        """ loads vdatum style binary gtx grids
        is_vertcon: when true, adjusts values from mm to m
        """
        self.filename = filename
        fp=open(self.filename,'rb')

        ll_lat,ll_lon,delta_lat,delta_lon = np.fromstring(fp.read(4*8),'>f8')
        ll_lon = (ll_lon + 180)%360. - 180

        nrows,ncols = np.fromstring(fp.read(2*4),'>i4')

        heights = np.fromstring(fp.read(nrows*ncols*8),'>f4').reshape( (nrows,ncols) )
        heights = heights.byteswap().newbyteorder().astype(np.float64).copy() # does this fix byte order?
        
        heights[ heights == missing ] = np.nan
        if is_vertcon:
            heights *= 0.001 # vertcon heights in mm

        # pretty sure that the corner values from the GTX file are
        # node-centered, so no need here to pass half-pixels around.
        SimpleGrid.__init__(self,
                            extents = [ll_lon,ll_lon+(ncols-1)*delta_lon,ll_lat,ll_lat+(nrows-1)*delta_lat],
                            F = heights,
                            projection=projection) 

class GdalGrid(SimpleGrid):
    """
    A specialization of SimpleGrid that can load single channel and RGB 
    files via the GDAL library.
    Use this for loading GeoTIFFs, some GRIB files, and other formats supported
    by GDAL.
    """
    @staticmethod
    def metadata(filename):
        """ Return the extents and resolution without loading the whole file
        """
        gds = gdal.Open(filename)
        (x0, dx, r1, y0, r2, dy ) = gds.GetGeoTransform()
        nx = gds.RasterXSize
        ny = gds.RasterYSize
        
        # As usual, this may be off by a half pixel...
        x1 = x0 + nx*dx
        y1 = y0 + ny*dy

        xmin = min(x0,x1)
        xmax = max(x0,x1)
        ymin = min(y0,y1)
        ymax = max(y0,y1)

        return [xmin,xmax,ymin,ymax],[dx,dy]

    def __init__(self,filename,bounds=None,geo_bounds=None,target_projection=None,
                 source_projection=None):
        """ Load a raster dataset into memory.
        bounds: [x-index start, x-index end, y-index start, y-index end]
         will load a subset of the raster.

        filename: path to a GDAL-recognize file, or an already opened GDAL dataset.
        geo_bounds: xxyy bounds in geographic coordinates

        target_projection: reproject if needed to given projection, specified as proj.4 
         compatible string. geo_bounds will be interpreted in the target projection. 
        """        
        if isinstance(filename,gdal.Dataset):
            self.gds=filename
        else:
            assert os.path.exists(filename),"GdalGrid: '%s' does not exist"%filename
            self.gds = gdal.Open(filename)
        (x0, dx, r1, y0, r2, dy ) = self.gds.GetGeoTransform()

        tgt_geo_bounds=None

        if source_projection is None:
            source_projection=self.gds.GetProjection()        

        if (target_projection is not None):
            if (source_projection is None) or (source_projection==""):
                raise Exception("Target projection was given, but there is no source projection for %s"%filename)

            from . import proj_utils
            src_srs=proj_utils.to_srs(source_projection)
            tgt_srs=proj_utils.to_srs(target_projection)

            if src_srs.IsSame(tgt_srs):
                print("Source and target reference systems appear identical")
                target_projection=None
            else:
                if geo_bounds is not None:
                    # This part gets more complicated
                    tgt_geo_bounds=geo_bounds # save this away for clipping later on
                    geo_bounds=proj_utils.reproject_bounds(geo_bounds,target_projection,source_projection,mode='outside')

        if geo_bounds is not None:
            # convert that the index bounds:
            ix_start = int( float(geo_bounds[0]-x0)/dx )
            ix_end = int( float(geo_bounds[1]-x0)/dx)+ 1
            # careful about sign of dy
            if dy>0:
                iy_start = int( float(geo_bounds[2]-y0)/dy )
                iy_end   = int( float(geo_bounds[3]-y0)/dy ) + 1
            else:
                iy_start = int( float(geo_bounds[3]-y0)/dy )
                iy_end   = int( float(geo_bounds[2]-y0)/dy ) + 1

            # clip those to valid ranges
            ix_max=self.gds.RasterXSize
            ix_start=max(0,min(ix_start,ix_max-1))
            ix_end=max(0,min(ix_end,ix_max-1))

            iy_max=self.gds.RasterYSize
            iy_start=max(0,min(iy_start,iy_max-1))
            iy_end=max(0,min(iy_end,iy_max-1))

            bounds = [ix_start,ix_end,
                      iy_start,iy_end]
            # print "geo bounds gave bounds",bounds
            self.geo_bounds = geo_bounds
            
        self.subset_bounds = bounds
        
        if bounds:
            A = self.gds.ReadAsArray(xoff = bounds[0],yoff=bounds[2],
                                     xsize = bounds[1] - bounds[0],
                                     ysize = bounds[3] - bounds[2])
            # and doctor up the metadata to reflect this:
            x0 += bounds[0]*dx
            y0 += bounds[2]*dy
        else:
            A = self.gds.ReadAsArray()

        # A is rows/cols !
        # And starts off with multiple channels, if they exist, as the
        # first index.
        if A.ndim == 3:
            print("Putting multiple channels as last index")
            A = A.transpose(1,2,0)

        # often gdal data is in image coordinates, which is just annoying.
        # Funny indexing because sometimes there are multiple channels, and those
        # appear as the first index:
        Nrows = A.shape[0]
        Ncols = A.shape[1]
        
        if dy < 0:
            # make y0 refer to the bottom left corner
            # and dy refer to positive northing
            y0 = y0 + Nrows*dy
            dy = -dy
            # this used to have the extra indices at the start, 
            # but I think that's wrong, as we put extra channels at the end
            A = A[::-1,:,...]

        # and there might be a nodata value, which we want to map to NaN
        b = self.gds.GetRasterBand(1)
        nodata = b.GetNoDataValue()

        if nodata is not None:
            if A.dtype in (np.int16,np.int32):
                A[ A==nodata ] = self.int_nan
            elif A.dtype in (np.uint16,np.uint32):
                A[ A==nodata ] = 0 # not great...
            elif np.issubdtype(A.dtype,np.float32):
                # Oddly, it's possible for nodata to be a float64,
                # and A a float32.
                nodata=np.float32(nodata)
                A[ A==nodata ] = np.nan
            else:
                A[ A==nodata ] = np.nan

        SimpleGrid.__init__(self,
                            extents = [x0+0.5*dx,
                                       x0+0.5*dx + dx*(Ncols-1),
                                       y0+0.5*dy,
                                       y0+0.5*dy + dy*(Nrows-1)],
                            F=A,
                            projection=source_projection )

        # most callers have no need to the GDAL dataset object,
        # and holding a reference here can get in the way of being able
        # to delete files when on windows.
        self.gds=None # effectively close the dataset
        
        if target_projection is not None:
            transformed=self.warp(target_projection)
            
            if tgt_geo_bounds:
                transformed=transformed.crop(tgt_geo_bounds)
            self.extents=transformed.extents
            self.F=transformed.F
            self._projection=transformed.projection()
            self.dx,self.dy = transformed.delta()

def rasterize_grid_cells(g,values,dx=None,dy=None,stretch=True,
                         cell_mask=slice(None),match=None,extra_options=[]):
    """ 
    g: UnstructuredGrid
    values: scalar values for each cell of the grid.  Must be uint16.
    dx,dy: resolution of the resulting raster
    stretch: use the full range of a uint16

    cell_mask: bitmask of cell indices to use. values should still be full
    size.

    match: an existing SimpleGrid field to copy extents/shape from

    returns: SimpleGrid field in memory
    """
    from . import wkb2shp
    dtype=np.uint16
    values=values[cell_mask]
    if stretch:
        vmin=values.min()
        vmax=values.max()
        fac=1./(vmax-vmin) * (np.iinfo(dtype).max-1)
        values=1+( (values-vmin)*fac ).astype(dtype)
    else:
        values=values.astype(np.uint16)
    polys=[g.cell_polygon(c) for c in np.arange(g.Ncells())[cell_mask]]
    poly_ds=wkb2shp.wkb2shp("Memory",polys,
                            fields=dict(VAL=values.astype(np.uint32)))

    if match:
        extents=match.extents
        Ny,Nx=match.F.shape
    else:
        extents=g.bounds()
        Nx=int( 1+ (extents[1]-extents[0])/dx )
        Ny=int( 1+ (extents[3]-extents[2])/dy )
        
    F=np.zeros( (Ny,Nx), np.float64)
    
    target_field=SimpleGrid(F=F,extents=extents)
    target_ds = target_field.write_gdal('Memory')
    
    # write 1000 into the array where the polygon falls.
    gdal.RasterizeLayer(target_ds,[1],poly_ds.GetLayer(0),options=["ATTRIBUTE=VAL"]+extra_options)
    #None,None,[1000],[])
    new_raster=GdalGrid(target_ds)

    if stretch:
        F=new_raster.F
        Fnew=np.zeros(F.shape,np.float64)

        Fnew = (F-1)/fac+vmin
        Fnew[F==0]=np.nan
        new_raster.F=Fnew

    return new_raster
        
if ogr:
    from stompy.spatial import interp_coverage
    class BlenderField(Field):
        """ Delegate to sub-fields, based on polygons in a shapefile, and blending
        where polygons overlap.

        If delegates is specified:
          The shapefile is expected to have a field 'name', which is then used to
          index the dict to get the corresponding field.

        Alternatively, if a factory is given, it should be callable and will take a single argument -
        a dict with the attributse for each source.  The factory should then return the corresponding
        Field.
        """
        def __init__(self,shp_fn=None,delegates=None,factory=None,subset=None,
                     shp_data=None):
            # awkward handling of cobbled together multicall - can pass either shapefile
            # path or pre-parsed shapefile data.
            if shp_fn is not None: # read from shapefile
                self.shp_fn = shp_fn
                self.shp_data=None
                self.ic = interp_coverage.InterpCoverage(shp_fn,subset=subset)
            else:
                assert shp_data is not None
                self.shp_data=shp_data
                self.shp_fn=None
                self.ic = interp_coverage.InterpCoverage(regions_data=shp_data,subset=subset)
                
            Field.__init__(self)

            self.delegates = delegates
            self.factory = factory

            self.delegate_list = [None]*len(self.ic.regions)

        def bounds(self):
            raise Exception("For now, you have to specify the bounds when gridding a BlenderField")

        def load_region(self,i):
            r = self.ic.regions[i]

            if self.delegates is not None:
                d = self.delegates[r.items['name']] 
            else:
                d = self.factory( r.items )

            self.delegate_list[i] = d

        def value(self,X):
            X=np.asanyarray(X)
            print("Calculating weights")
            weights = self.ic.calc_weights(X)
            total_weights = weights.sum(axis=-1)

            vals = np.zeros(X.shape[:-1],np.float64)
            vals[total_weights==0.0] = np.nan

            # iterate over sources:
            for src_i in range(len(self.delegate_list)):
                print("Processing layer ",self.ic.regions[src_i].identifier())

                src_i_weights = weights[...,src_i]

                needed = (src_i_weights != 0.0)
                if needed.sum() > 0:
                    if self.delegate_list[src_i] is None:
                        self.load_region(src_i) # lazy loading
                    src_vals = self.delegate_list[src_i].value( X[needed] )
                    vals[needed] += src_i_weights[needed] * src_vals
            return vals

        def value_on_edge(self,e):
            """ Return the interpolated value for a given line segment"""
            ### UNTESTED
            c = e.mean(axis=0) # Center of edge

            weights = self.ic.calc_weights(c)

            val = 0.0 # np.zeros(X.shape[:-1],np.float64)

            # iterate over sources:
            for src_i in range(len(self.delegate_list)):
                # print "Processing layer ",self.ic.regions[src_i].identifier()

                src_i_weight = weights[src_i]

                if src_i_weight != 0.0:
                    if self.delegate_list[src_i] is None:
                        self.load_region(src_i)
                    src_val = self.delegate_list[src_i].value_on_edge( e )
                    val += src_i_weight * src_val
            return val

        def diff(self,X):
            """ Calculate differences between datasets where they overlap:
            When a point has two datasets, the first is subtracted from the second.
            When there are more, they alternate - so with three, you get A-B+C
            Not very useful, but fast...
            """
            weights = self.ic.calc_weights(X)

            vals = np.zeros(X.shape[:-1],np.float64)

            used = (weights!=0.0)
            n_sources = used.sum(axis=-1)

            # We just care about how the sources differ - if there is only
            # one source then don't even bother calculating it - set all weights
            # to zero.
            weights[(n_sources==1),:] = 0.0 #

            # iterate over sources:
            for src_i in range(len(self.delegates)):
                src_i_weights = weights[...,src_i]

                needed = (src_i_weights != 0.0)
                src_vals = self.delegates[src_i].value( X[needed] )
                vals[needed] = src_vals - vals[needed] 
            return vals

    class MultiBlender(Field):
        """
        A collection of BlenderFields, separated based on a priority
        field in the sources shapefile.
        """
        def __init__(self,shp_fn,factory=None,priority_field='priority',
                     buffer_field=None):
            self.priority_field=priority_field
            self.shp_fn=shp_fn
            self.sources=wkb2shp.shp2geom(shp_fn)

            if buffer_field is not None:
                self.flatten_with_buffer(buffer_field)

            super(MultiBlender,self).__init__()

            # This will sort low to high
            self.priorities=np.unique(self.sources[self.priority_field])

            self.bfs=[]

            for pri in self.priorities:
                subset= np.nonzero( self.sources[self.priority_field]==pri )[0]
                self.bfs.append( BlenderField(shp_data=self.sources,
                                              factory=factory,subset=subset) )

        # def to_grid(self,dx,dy,bounds):
        def value(self,X):
            X=np.asanyarray(X)
            shape_orig=X.shape
            Xlin=X.reshape( [-1,2] )
            V=np.nan*np.ones( len(Xlin), 'f8' )

            # go in reverse order, to grab data from highest priority
            # fields first.
            for bf in self.bfs[::-1]:
                sel=np.isnan(V)
                if np.all(~sel):
                    break
                V[sel] = bf.value(Xlin[sel])
            return V.reshape( shape_orig[:-1] )
            
        def flatten_with_buffer(self,buffer_field='buffer'):
            """ 
            Rather then pure stacking of the rasters by priority,
            automatically create some buffers between the high
            priority fields and lower priority, to get some 
            blending
            """
            sources=self.sources.copy()
            
            priorities=np.unique(self.sources[self.priority_field])

            union_geom=None # track the union of all polygons so far

            from shapely.ops import cascaded_union

            # higher priority layers, after being shrunk by their 
            # respective buffer distances, are subtracted from lower layer polygons.
            # each feature's geometry is updated with the higher priority layers
            # subtracted out, and then contributes its own neg-buffered geometry
            # to the running union

            for pri in priorities[::-1]:
                # if pri<100:
                #     import pdb
                #     pdb.set_trace()
                sel_idxs = np.nonzero( sources['priority'] == pri )[0]

                updated_geoms=[] # just removed higher priority chunks
                slimmed_geoms=[] # to be included in running unionn

                for sel_idx in sel_idxs:
                    sel_geom=sources['geom'][sel_idx]
                    if union_geom is not None:
                        print("Updating %d"%sel_idx)
                        # HERE: this can come up empty
                        vis_sel_geom = sources['geom'][sel_idx] = sel_geom.difference( union_geom )
                    else:
                        vis_sel_geom=sel_geom

                    if vis_sel_geom.area > 0.0:
                        buff=sources[buffer_field][sel_idx]
                        sel_geom_slim=sel_geom.buffer(-buff)
                        # print("Buffering by %f"%(-buff) )
                        slimmed_geoms.append( sel_geom_slim )

                merged=cascaded_union(slimmed_geoms) # polygon or multipolygon
                if union_geom is None:
                    union_geom=merged
                else:
                    union_geom=merged.union(union_geom)
            self.old_sources=self.sources
            sources[self.priority_field]=0.0 # no more need for priorities
            valid=np.array( [ (source['geom'].area > 0.0)
                              for source in sources ] )
            invalid_count=np.sum(~valid)
            if invalid_count:
                print("MultiBlenderField: %d source polygons were totally obscured"%invalid_count)
            self.sources=sources[valid]

class CompositeField(Field):
    """
    In the same vein as BlenderField, but following the model of raster
    editors like Photoshop or the Gimp.

    Individual sources are treated as an ordered "stack" of layers.

    Layers higher on the stack can overwrite the data provided by layers
    lower on the stack.

    A layer is typically defined by a raster data source and a polygon over
    which it is valid.

    Each layer's contribution to the final dataset is both a data value and
    an alpha value.  This allows for blending/feathering between layers.

    The default "data_mode" is simply overlay.  Other data modes like "min" or
    "max" are possible.

    The default "alpha_mode" is "valid()" which is essentially opaque where there's
    valid data, and transparent where there isn't.  A second common option would
    probably be "feather(<distance>)", which would take the valid areas of the layer,
    and feather <distance> in from the edges.

    The sources, data_mode, alpha_mode details are taken from a shapefile.

    Alternatively, if a factory is given, it should be callable and will take a single argument -
    a dict with the attributse for each source.  The factory should then return the corresponding
    Field.

    TODO: there are cases where the alpha is so small that roundoff can cause
    artifacts.  Should push these cases to nan.
    TODO: currently holes must be filled manually or after the fact.  Is there a clean
    way to handle that?  Maybe a fill data mode?

    Guide
    -----

    Create a polygon shapefile, with fields:
     +------------+-----------+
     + priority   | numeric   |
     +------------+-----------+
     + data_mode  | string    |
     +------------+-----------+
     + alpha_mode | string    |
     +------------+-----------+

    These names match the defaults to the constructor.  Note that there is no
    reprojection support -- the code assumes that the shapefile and all source
    data are already in the target projection.  Some code also assumes that it
    is a square projection.

    .. image:: images/composite-shp-table.png

    Each polygon in the shapefile refers to a source dataset and defines where
    that dataset will be used.

    .. image:: images/composite-shp.png
    .. image:: images/composite-shp-zoom.png

    Datasets are processed as layers, building up from the lowest priority
    to the highest priority.  Higher priority sources generally overwrite
    lower priority source, but that can be controlled by specifying
    `data_mode`.  The default is `overlay()`, which simple overwrites
    the lower priority data.  Other common modes are
     * `min()`: use the minimum value between this source and lower
       priority data.  This layer will only *deepen* areas.
     * `max()`: use the maximum value between this source and lower
       priority data.  This layer will only *raise* areas.
     * `fill(dist)`: fill in holes up to `dist` wide in this datasets
       before proceeding.

    Multiple steps can be chained with commas, as in `fill(5.0),min()`, which
    would fill in holes smaller than 5 spatial units (e.g. m), and then take
    the minimum of this dataset and the existing data from previous (lower
    priority) layers.

    Another example:

    .. image:: images/dcc-original.png
    .. image:: images/dcc-dredged.png

    """
    projection=None
    def __init__(self,shp_fn=None,factory=None,
                 priority_field='priority',
                 data_mode='data_mode',
                 alpha_mode='alpha_mode',
                 shp_data=None,
                 shp_query=None,
                 target_date=None):
        self.shp_fn = shp_fn
        if shp_fn is not None: # read from shapefile
            self.sources,self.projection=wkb2shp.shp2geom(shp_fn,return_srs=True,query=shp_query)
        else:
            self.sources=shp_data

        if target_date is not None:
            selA=np.array([ isnat(d) or d<=target_date
                            for d in self.sources['start_date']] )
            selB=np.array([ isnat(d) or d>target_date
                            for d in self.sources['end_date']] )
            orig_count=len(self.sources)
            self.sources=self.sources[selA&selB]
            new_count=len(self.sources)
            log.info("Date filter selected %s of %s sources"%(new_count,orig_count))

        if data_mode is not None:
            self.data_mode=self.sources[data_mode]
        else:
            self.data_mode=['overlay()']*len(self.sources)

        if alpha_mode is not None:
            self.alpha_mode=self.sources[alpha_mode]
        else:
            self.data_mode=['valid()']*len(self.sources)

        # Impose default values on those:
        for i in range(len(self.sources)):
            if self.alpha_mode[i]=='':
                self.alpha_mode[i]='valid()'
            if self.data_mode[i]=='':
                self.data_mode[i]='overlay()'

        super(CompositeField,self).__init__()

        self.factory = factory

        self.delegate_list=[None]*len(self.sources)

        self.src_priority=self.sources[priority_field]
        self.priorities=np.unique(self.src_priority)

    def bounds(self):
        raise Exception("For now, you have to specify the bounds when gridding a BlenderField")

    def load_source(self,i):
        if self.delegate_list[i] is None:
            self.delegate_list[i] = self.factory( self.sources[i] )
        return self.delegate_list[i]

    def to_grid(self,nx=None,ny=None,bounds=None,dx=None,dy=None,
                mask_poly=None,stackup=False):
        """ render the layers to a SimpleGrid tile.
        nx,ny: number of pixels in respective dimensions
        bounds: xxyy bounding rectangle.
        dx,dy: size of pixels in respective dimensions.
        mask_poly: a shapely polygon.  only points inside this polygon
        will be generated.

        stackup: 'return': return a list of the layers involve in compositing
        this tile. 
        'plot': make a figure showing the evolution of the layers as they're
        stacked up.
        """
        # boil the arguments down to dimensions
        if bounds is None:
            xmin,xmax,ymin,ymax = self.bounds()
        else:
            if len(bounds) == 2:
                xmin,ymin = bounds[0]
                xmax,ymax = bounds[1]
            else:
                xmin,xmax,ymin,ymax = bounds
        if nx is None:
            nx=1+int(np.round((xmax-xmin)/dx))
            ny=1+int(np.round((ymax-ymin)/dy))

        if stackup:
            stack=[]

        # in case it came in as 2x2
        bounds=[xmin,xmax,ymin,ymax]
            
        # allocate the blank starting canvas
        result_F =np.ones((ny,nx),'f8')
        result_F[:]=-999 # -999 so we don't get nan contamination
        result_data=SimpleGrid(extents=bounds,F=result_F,projection=self.projection)
        result_alpha=result_data.copy()
        result_alpha.F[:]=0.0

        # Which sources to use, and in what order?
        box=geometry.box(bounds[0],bounds[2],bounds[1],bounds[3])
        if mask_poly is not None:
            box=box.intersection(mask_poly)

        # Which sources are relevant?
        relevant_srcs=np.nonzero( [ box.intersects(geom)
                                    for geom in self.sources['geom'] ])[0]
        # omit negative priorities
        relevant_srcs=relevant_srcs[ self.src_priority[relevant_srcs]>=0 ]

        # Starts with lowest, goes to highest
        order = np.argsort(self.src_priority[relevant_srcs])
        ordered_srcs=relevant_srcs[order]

        # Use to use ndimage.distance_transform_bf.
        # This appears to give equivalent results (at least for binary-valued
        # inputs), and runs about 80x faster on a small-ish input.
        dist_xform=ndimage.distance_transform_edt
        for src_i in ordered_srcs:
            log.info(self.sources['src_name'][src_i])
            log.info("   data mode: %s  alpha mode: %s"%(self.data_mode[src_i],
                                                         self.alpha_mode[src_i]))

            source=self.load_source(src_i) # HERE - need to be smarter about overlapping bounds, and also reproject on the fly

            if isinstance(source,SimpleGrid) and source.F.size==0: # could be other type of field.
                # So the geom overlapped the current tile, but the raster itself came up empty.
                log.info("Source %s came up empty after cropping. Check projection, and whether polygon intersects data"
                         %(self.sources['src_name'][src_i]))
                continue
            # could consider adding a to_grid to simple grid, as this
            # currently goes through the generic interpolate interface.
            src_data = source.to_grid(bounds=bounds,dx=dx,dy=dy)
            src_alpha= SimpleGrid(extents=src_data.extents,
                                  F=np.ones(src_data.F.shape,'f8'))

            src_geom=self.sources['geom'][src_i]
            if mask_poly is not None:
                src_geom=src_geom.intersection(mask_poly)
            mask=src_alpha.polygon_mask(src_geom)
            src_alpha.F[~mask] = 0.0

            # Use nan's to mask data, rather than masked arrays.
            # Convert as necessary here:
            if isinstance(src_data.F,np.ma.masked_array):
                src_data.F=src_data.F.filled(np.nan)

            # create an alpha tile. depending on alpha_mode, this may draw on the lower data,
            # the polygon and/or the data tile.
            # modify the data tile according to the data mode - so if the data mode is 
            # overlay, do nothing.  but if it's max, the resulting data tile is the max
            # of itself and the lower data.
            # composite the data tile, using its alpha to blend with lower data.

            # the various operations
            def min():
                """ new data will only decrease values
                """
                valid=result_alpha.F>0
                src_data.F[valid]=np.minimum( src_data.F[valid],result_data.F[valid] )
            def max():
                """ new data will only increase values
                """
                valid=result_alpha.F>0
                src_data.F[valid]=np.maximum( src_data.F[valid],result_data.F[valid] )
            def fill(dist):
                "fill in small missing areas"
                pixels=int(round(float(dist)/dx))
                # for fill, it may be better to clip this to 1 pixel, rather than
                # bail when pixels==0
                if pixels>0:
                    niters=np.maximum( pixels//3, 2 )
                    src_data.fill_by_convolution(iterations=niters)

            def scale(factor):
                src_data.F[:] *= factor
                
            def blur(dist):
                "smooth data channel with gaussian filter - this allows spreading beyond original poly!"
                pixels=int(round(float(dist)/dx))
                #import pdb
                #pdb.set_trace()
                Fzed=src_data.F.copy()
                valid=np.isfinite(Fzed)
                Fzed[~valid]=0.0
                weights=ndimage.gaussian_filter(1.0*valid,pixels)
                blurred=ndimage.gaussian_filter(Fzed,pixels)
                blurred[weights<0.5]=np.nan
                blurred[weights>=0.5] /= weights[weights>=0.5]
                src_data.F=blurred

            def diffuser():
                self.diffuser(source,src_data,src_geom,result_data)

            def ortho_diffuser(res,aniso=1e-5):
                self.ortho_diffuser(res=res,aniso=aniso,source=source,
                                    src_data=src_data,src_geom=src_geom,result_data=result_data)

            def overlay():
                pass
            # alpha channel operations:
            def valid():
                # updates alpha channel to be zero where source data is missing.
                data_missing=np.isnan(src_data.F)
                src_alpha.F[data_missing]=0.0
            def blur_alpha(dist):
                "smooth alpha channel with gaussian filter - this allows spreading beyond original poly!"
                pixels=int(round(float(dist)/dx))
                if pixels>0:
                    src_alpha.F=ndimage.gaussian_filter(src_alpha.F,pixels)
            def feather_in(dist):
                "linear feathering within original poly"
                pixels=int(round(float(dist)/dx))
                if pixels>0:
                    Fsoft=dist_xform(src_alpha.F)
                    src_alpha.F = (Fsoft/pixels).clip(0,1)
            def buffer(dist):
                "buffer poly outwards (by pixels)"
                # Could do this by erosion/dilation.  but using
                # distance is a bit more compact (maybe slower, tho)
                pixels=int(round(float(dist)/dx))
                if pixels>0:
                    # Like feather_out.
                    # Fsoft gets distance to a 1 pixel
                    Fsoft=dist_xform(1-src_alpha.F)
                    # is this right, or does it need a 1 in there?
                    src_alpha.F = (pixels-Fsoft).clip(0,1)
                elif pixels<0:
                    pixels=-pixels
                    # Fsoft gets the distance to a zero pixel
                    Fsoft=dist_xform(src_alpha.F)
                    src_alpha.F = (Fsoft-pixels).clip(0,1)
                
            feather=feather_in
            def feather_out(dist):
                pixels=int(round(float(dist)/dx))
                if pixels>0:
                    Fsoft=dist_xform(1-src_alpha.F)
                    src_alpha.F = (1-Fsoft/pixels).clip(0,1)
                
            # dangerous! executing code from a shapefile!
            for mode in [self.data_mode[src_i],self.alpha_mode[src_i]]:
                if mode is None or mode.strip() in ['',b'']: continue
                # This is getting a SyntaxError when using python 2.
                # exec(mode) # used to be eval.
                six.exec_(mode)

            data_missing=np.isnan(src_data.F)
            src_alpha.F[data_missing]=0.0
            cleaned=src_data.F.copy()
            cleaned[data_missing]=-999 # avoid nan contamination.

            assert np.allclose( result_data.extents, src_data.extents )
            assert np.all( result_data.F.shape==src_data.F.shape )

            # 2018-12-06: this is how it used to work, but this is problematic
            #  when result_alpha is < 1.
            # result_data.F   = result_data.F *(1-src_alpha.F) + cleaned*src_alpha.F

            # where result_alpha=1.0, then we want to blend with src_alpha and 1-src_alpha.
            # if result_alpha=0.0, then we take src wholesale, and carry its alpha through.
            # 
            total_alpha=result_alpha.F*(1-src_alpha.F) + src_alpha.F
            result_data.F   = result_data.F * result_alpha.F *(1-src_alpha.F) + cleaned*src_alpha.F
            # to avoid contracting data towards zero, have to normalize data by the total alpha.
            valid_alpha=total_alpha>1e-10 # avoid #DIVZERO
            result_data.F[valid_alpha] /= total_alpha[valid_alpha]
            result_alpha.F  = total_alpha

            if stackup:
                stack.append( (self.sources['src_name'][src_i],
                               result_data.copy(),
                               src_alpha.copy() ) )

        # fudge it a bit, and allow semi-transparent data back out, but
        # at least nan out the totally transparent stuff.
        result_data.F[ result_alpha.F==0 ] = np.nan

        if stackup=='return':
            return result_data,stack
        elif stackup=='plot':
            self.plot_stackup(result_data,stack)
        
        return result_data

    def ortho_diffuser(self,res,aniso,source,src_data,src_geom,result_data):
        """
        Strong curvilinear anisotropic interpolation
        """
        from . import interp_orthogonal
        oink=interp_orthogonal.OrthoInterpolator(region=src_geom,
                                                 background_field=result_data,
                                                 anisotropy=aniso,
                                                 nom_res=res)
        fld=oink.field()
        
        rast=fld.to_grid(bounds=result_data.bounds(),
                         dx=result_data.dx,dy=result_data.dy)
        src_data.F[:,:]=rast.F
    
    def diffuser(self,src,src_data,src_geom,result_data):
        """
        src: the source for the layer. Ignored unless it's an XYZField
        in which case the point samples are included.
        src_data: where the diffused field will be saved
        src_geom: polygon to work in
        result_data: the stackup result from previous layers
        """
        from scipy import sparse
        from ..grid import triangulate_hole,quad_laplacian, unstructured_grid
        from . import linestring_utils

        dx=3*src_data.dx # rough guess
        curve=linestring_utils.resample_linearring(np.array(src_geom.exterior),
                                                   dx,closed_ring=1)
        g=unstructured_grid.UnstructuredGrid()
        nodes,edges=g.add_linestring(curve,closed=True)
        g=triangulate_hole.triangulate_hole(g,nodes=nodes,hole_rigidity='all',method='rebay')
        bnodes=g.boundary_cycle()
        bvals=result_data(g.nodes['x'][bnodes])

        nd=quad_laplacian.NodeDiscretization(g)
        dirich={ n:val
                 for n,val in zip(bnodes,bvals) }
        if isinstance(src,XYZField):
            for xy,z in zip(src.X,src.F):
                c=g.select_cells_nearest([x,y],inside=True)
                if c is None: continue
                n=g.select_nodes_nearest([x,y])
                dirich[n]=z

        M,b=nd.construct_matrix(op='laplacian',dirichlet_nodes=dirich)
        diffed=sparse.linalg.spsolve(M.tocsr(),b)

        fld=XYZField(X=g.nodes['x'],F=diffed)
        fld._tri=g.mpl_triangulation()
        rast=fld.to_grid(bounds=result_data.bounds(),
                         dx=result_data.dx,dy=result_data.dy)
        src_data.F[:,:]=rast.F
        return rast

    def plot_stackup(self,result_data,stack,num=None,z_factor=5.,cmap='jet'):
        plt.figure(num=num).clf()
        nrows=ncols=np.sqrt(len(stack))
        nrows=int(np.ceil(nrows))
        ncols=int(np.floor(ncols))
        if nrows*ncols<len(stack): ncols+=1
        
        fig,axs=plt.subplots(nrows,ncols,num=num,squeeze=False)

        for ax,(name,data,alpha) in zip( axs.ravel(), stack ):
            data.plot(ax=ax,vmin=0,vmax=3.5,cmap=cmap)
            data.plot_hillshade(ax=ax,z_factor=z_factor)
            ax.axis('off')
            ax.set_title(name)
        for ax in axs.ravel()[len(stack):]:
            ax.axis('off')
            
        fig.subplots_adjust(left=0,right=1,top=0.95,bottom=0,hspace=0.08)
        # fig.
        return fig


class MultiRasterField(Field):
    """ Given a collection of raster files at various resolutions and with possibly overlapping
    extents, manage a field which picks from the highest resolution raster for any given point.

    Assumes that any point of interest is covered by at least one field (though there may be slower support
    coming for some sort of nearest valid usage).

    There is no blending for point queries!  If two fields cover the same spot, the value taken from the
    higher resolution field will be returned.

    Basic bilinear interpolation will be utilized for point queries.

    Edge queries will resample the edge at the resolution of the highest datasource, and then proceed with
    those point queries

    Cell/region queries will have to wait for another day

    Some effort is made to keep only the most-recently used rasters in memory, since it is not feasible

    to load all rasters at one time. to this end, it is most efficient for successive queries to have some
    spatial locality.
    """

    # If finite, any point sample greater than this value will be clamped to this value
    clip_max = np.inf

    # Values below this will be interpreted is missing data
    min_valid = -np.inf

    order = 1 # interpolation order

    # After clipping, this value will be added to the result.
    # probably shouldn't use this - domain.py takes care of adding in the bathymetry offset
    # and reversing the sign (so everything becomes positive)
    offset = 0.0 # BEWARE!!! read the comment.

    # any: raise an exception if any raster_file_pattern fails to find any
    #  matches
    # all: raise an exception if all patterns come up empty
    # False: silently proceed with no matches.
    error_on_null_input='any' # 'all', or False

    def __init__(self,raster_file_patterns,**kwargs):
        self.__dict__.update(kwargs)
        Field.__init__(self)
        raster_files = []
        for patt in raster_file_patterns:
            if isinstance(patt,tuple):
                patt,pri=patt
            else:
                pri=0 # default priority
            if isinstance(patt,str):
                matches=glob.glob(patt)
                if len(matches)==0 and self.error_on_null_input=='any':
                    raise Exception("Pattern '%s' got no matches"%patt)
                raster_files += [ (m,pri) for m in matches]
            elif isinstance(patt,Field):
                raster_files.append( (patt,pri) )
            else:
                raise Exception("Expected a string regexp or a Field instance. Got %s"%patt)
        if len(raster_files)==0 and self.error_on_null_input=='all':
            raise Exception("No patterns got matches")

        self.raster_files = raster_files

        self.prepare()

    def bounds(self):
        """ Aggregate bounds """
        all_extents=self.sources['extent']

        return [ all_extents[:,0].min(),
                 all_extents[:,1].max(),
                 all_extents[:,2].min(),
                 all_extents[:,3].max() ]

    def prepare(self):
        # find extents and resolution of each dataset:
        sources=np.zeros( len(self.raster_files),
                          dtype=[ ('field','O'),
                                  ('filename','O'),
                                  ('extent','f8',4),
                                  ('resolution','f8'),
                                  ('resx','f8'),
                                  ('resy','f8'),
                                  ('order','f8'),
                                  ('last_used','i4') ] )

        for fi,(f,pri) in enumerate(self.raster_files):
            if isinstance(f,str):
                extent,resolution = GdalGrid.metadata(f)
                sources['extent'][fi] = extent
                sources['resolution'][fi] = max(resolution[0],resolution[1])
                sources['resx'][fi] = resolution[0]
                sources['resy'][fi] = resolution[1]
                # negate so that higher priority sorts to the beginning
                sources['field'][fi]=None
                sources['filename'][fi]=f
            else:
                sources['extent'][fi] = f.extents
                sources['resolution'][fi] = max(f.dx, f.dy)
                sources['resx'][fi] = f.dx
                sources['resy'][fi] = f.dy
                # negate so that higher priority sorts to the beginning
                sources['field'][fi]=f
                sources['filename'][fi]=None
            # common
            sources['last_used'][fi]=-1
            sources['order'][fi] = -pri

        self.sources = sources
        # -1 means the source isn't loaded.  non-negative means it was last used when serial
        # was that value.  overflow danger...

        self.build_index()

    def polygon_mask(self,poly,crop=True,return_values=False):
        """ 
        Mimic SimpleGrid.polygon_mask
        Requires return_values==True, since a bitmask doesn't make
        sense over a stack of layers.

        return_values: must be True, and will 
         return just the values of F that fall inside poly. 
        """
        assert crop==True,"MultiRasterField only implements crop=True behavior"
        assert return_values==True,"MultiRasterField only makes sense for return_values=True"

        xyxy=poly.bounds
        xxyy=[xyxy[0], xyxy[2], xyxy[1], xyxy[3]]
        tile=self.extract_tile(xxyy)
        return tile.polygon_mask(poly,crop=False,return_values=True)
        
    # Thin wrapper to make a multiraster field look like one giant high resolution
    # raster.
    @property
    def dx(self):
        return np.abs(self.sources['resx']).min()
    @property
    def dy(self):
        # many files report negative dy!
        return np.abs(self.sources['resy']).min()

    def crop(self,rect=None):
        return self.to_grid(bounds=rect)
    
    def build_index(self):
        # Build a basic index that will return the overlapping dataset for a given point
        # these are x,x,y,y
        tuples = [(i,extent,None) 
                  for i,extent in enumerate(self.sources['extent'])]

        self.index = RectIndex(tuples,interleaved=False)

    def report(self):
        """ Short text representation of the layers found and their resolutions
        """
        print("Raster sources:")
        print(" Idx  Order  Res  File")
        for fi,rec in enumerate(self.sources):
            print("%4d %4.1f %6.1f: %s"%(fi,
                                         rec['order'],
                                         rec['resolution'],
                                         rec['filename']))

    # TODO:
    #  For large sources rasters, replace them with a list of tiles, so we can load
    #  and cache smaller tiles.
    max_count = 20 
    open_count = 0
    serial = 0
    def source(self,i):
        """ LRU based cache of the datasets
        """
        if self.sources['field'][i] is None:
            if self.open_count >= self.max_count:
                # Have to choose someone to close, ignoring entries with no filename (since 
                # those represent entries supplied as Fields)
                current = np.nonzero( (self.sources['last_used']>=0) & (self.sources['filename']!=None))[0]
                #  - don't close fields that have no filename
                victim = current[ np.argmin( self.sources['last_used'][current] ) ]
                # print "Will evict source %d"%victim
                self.sources['last_used'][victim] = -1
                self.sources['field'][victim] = None
                self.open_count -= 1
            # open the new guy:
            self.sources['field'][i] = src = GdalGrid(self.sources['filename'][i])
            # Need to treat this here, since otherwise those values may propagate
            # in interpolation and then it will be hard to detect them.
            src.F[ src.F < self.min_valid ] = np.nan
            self.open_count += 1

        self.serial += 1
        self.sources['last_used'][i] = self.serial
        return self.sources['field'][i]

    def value_on_point(self,xy):
        hits=self.ordered_hits(xy[xxyy])
        if len(hits) == 0:
            return np.nan

        v = np.nan
        for hit in hits:
            src = self.source(hit)

            # Here we should be asking for some kind of basic interpolation
            v = src.interpolate( np.array([xy]), interpolation='linear' )[0]

            if np.isnan(v):
                continue
            if v > self.clip_max:
                v = self.clip_max
            return v
        # print("Bad sample at point ",xy)
        return v

    def value(self,X):
        """ X must be shaped (...,2)
        """
        X = np.array(X)
        orig_shape = X.shape

        X = X.reshape((-1,2))

        newF = np.zeros( X.shape[0],np.float64 )

        for i in range(X.shape[0]):
            if i > 0 and i % 2000 == 0:
                print("%d/%d"%(i,X.shape[0]))
            newF[i] = self.value_on_point( X[i] )

        newF = newF.reshape(orig_shape[:-1])

        if newF.ndim == 0:
            return float(newF)
        else:
            return newF

    def value_on_edge(self,e,samples=None):
        """
        Subsample the edge, using an interval based on the highest resolution overlapping
        dataset.  Average and return...
        """
        pmin = e.min(axis=0)
        pmax = e.max(axis=0)

        hits=self.ordered_hits( [pmin[0],pmax[0],pmin[1],pmax[1]] )
        if len(hits) == 0:
            return np.nan

        res = self.sources['resolution'][hits].min()

        samples = int( np.ceil( norm(e[0] - e[1])/res) )

        x=np.linspace(e[0,0],e[1,0],samples)
        y=np.linspace(e[0,1],e[1,1],samples)

        X = np.array([x,y]).transpose()

        # old way - about 1.3ms per edge over 100 edges
        # return nanmean(self.value(X))

        # inlining -
        # in order of resolution, query all the points at once from each field.
        edgeF = np.nan*np.ones( X.shape[0],np.float64 )

        for hit in hits:
            missing = np.isnan(edgeF)
            # now redundant with edit of src.F
            #if self.min_valid is not None:
            #    missing = missing | (edgeF<self.min_valid)
            src = self.source(hit)

            # for the moment, keep the nearest interpolation
            edgeF[missing] = src.interpolate( X[missing],interpolation='linear' )
            if np.all(np.isfinite(edgeF)):
                break

        edgeF = np.clip(edgeF,-np.inf,self.clip_max) # ??
        return np.nanmean(edgeF)

    def ordered_hits(self,xxyy):
        hits = self.index.intersection( xxyy )
        if isinstance(hits, types.GeneratorType):
            # so translate that into a list like we used to get.
            hits = list(hits)
        hits = np.array(hits)

        if len(hits) == 0:
            return []

        # include filename here to resolve ties, avoiding the fallback behavior which
        # may error out when comparing None
        hits = hits[ np.argsort( self.sources[hits], order=('order','resolution','filename')) ]
        return hits

    def extract_tile(self,xxyy=None,res=None):
        """ Create the requested tile from merging the sources.  Resolution defaults to
        resolution of the highest resolution source that falls inside the requested region
        """
        return self.to_grid(bounds=xxyy,dx=res,dy=res)

    def to_grid(self,nx=None,ny=None,interp='linear',bounds=None,dx=None,dy=None,valuator='value'):
        """
        Extract data in a grid.  currently only nearest, no linear interpolation.
        """
        # This used to be extract_tile, but the interface of to_grid is broader, so better
        # to have extract_tile be a special case of to_grid.
        xxyy=bounds
        if xxyy is None:
            xxyy=self.bounds()
        xxyy=as_xxyy(xxyy)

        hits = self.ordered_hits(xxyy)
        if not len(hits):
            # this can happen esp. when generating tiles for a larger dataset.
            log.warning("to_grid: no datasets overlap, will return all nan")
            if dx is None or dy is None:
                raise Exception("No hits, and dx/dy not specified so resolution is unknown")

        if dx is None:
            dx=self.sources['resolution'][hits].min()
        if dy is None:
            dy=self.sources['resolution'][hits].min()

        # half-pixel alignment-
        # field.SimpleGrid expects extents which go to centers of pixels.
        # x and y are inclusive of the end pixels (so for exactly abutting rects, there will be 1 pixel
        # of overlap)
        x=np.arange( xxyy[0],xxyy[1]+dx,dx)
        y=np.arange( xxyy[2],xxyy[3]+dy,dy)
        targetF = np.nan*np.zeros( (len(y),len(x)), np.float64)
        pix_extents = [x[0],x[-1], y[0],y[-1] ]
        target = SimpleGrid(extents=pix_extents,F=targetF)

        # iterate over overlapping DEMs until we've filled in all the holes
        # might want some feature where coarse data are only input at their respective
        # cell centers, and then everything is blended,
        # or maybe that as dx increases, we allow a certain amount of blending first
        # the idea being that it there's a 5m hole in some lidar, it's better to blend the
        # lidar than to query a 100m dataset.

        # extend the extents to consider width of pixels (this at least makes the output
        # register with the inputs)

        for hit in hits:
            src = self.source(hit)

            src_x,src_y = src.xy()
            src_dx,src_dy = src.delta()

            # maps cols in the destination to cols in this source
            # map_coordinates wants decimal array indices
            # x has the utm easting for each column to extract
            # x-src_x:   easting relative to start of src tile
            dec_x = (x-src_x[0]) / src_dx
            dec_y = (y-src_y[0]) / src_dy

            if self.order==0:
                dec_x = np.floor( (dec_x+0.5) )
                dec_y = np.floor( (dec_y+0.5) )

            # what range of the target array falls within this tile
            col_range = np.nonzero( (dec_x>=0) & (dec_x <= len(src_x)-1))[0]
            if len(col_range):
                col_range = col_range[ [0,-1]]
            else:
                continue
            row_range = np.nonzero( (dec_y>=0) & (dec_y <= len(src_y)-1))[0]
            if len(row_range):
                row_range=row_range[ [0,-1]]
            else:
                continue

            col_slice = slice(col_range[0],col_range[1]+1)
            row_slice = slice(row_range[0],row_range[1]+1)
            dec_x = dec_x[ col_slice ]
            dec_y = dec_y[ row_slice ]

            C,R = np.meshgrid( dec_x,dec_y )

            newF = ndimage.map_coordinates(src.F, [R,C],order=self.order)

            # only update missing values
            missing = np.isnan(target.F[ row_slice,col_slice ])
            # Now redundant with updating src.F above
            # if self.min_valid is not None:
            #     # Also ignore out-of-bounds values from newF
            #     missing = missing & (newF>=self.min_valid)
            target.F[ row_slice,col_slice ][missing] = newF[missing]
        return target

class FunctionField(Field):
    """ wraps an arbitrary function
    function must take one argument, X, which has
    shape [...,2]
    """
    def __init__(self,func):
        self.func = func
    def value(self,X):
        X=np.asanyarray(X)
        return self.func(X)

# used to be in its own file
class TileMaker(object):
    """ Given a field, create gridded tiles of the field, including some options for blending, filling,
    cropping, etc.
    """
    tx = 1000 # physical size, x, for a tile
    ty = 1000 # physical size, y, for a tile
    dx = 2    # pixel width
    dy = 2    # pixel height
    pad = 50  # physical distance to pad tiles in each of 4 directions
    
    fill_iterations = 10
    smoothing_iterations = 5
    
    force = False # overwrite existing output files
    output_dir = "."

    filename_fmt = "%(left).0f-%(bottom).0f.tif"
    quantize=True # whether to quantize bounds to tx

    # A function(SimpleGrid,**kw) => SimpleGrid
    # If set, this is called after rendering each tile, but before the tile
    # is unpadded.  see code below for the keywords supplied.
    post_render=None
    
    def __init__(self,f,**kwargs):
        """ f: the field to be gridded
        """
        self.f = f
        set_keywords(self,kwargs)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def tile(self,xmin=None,ymin=None,xmax=None,ymax=None):
        self.tile_fns=[]

        if (xmin is None) or (xmax is None) or (ymin is None) or (ymax is None):
            # some fields don't know their bounds, so hold off calling
            # this unless we have to.
            bounds=self.f.bounds()

            if xmin is None: xmin=bounds[0]
            if xmax is None: xmax=bounds[1]
            if ymin is None: ymin=bounds[2]
            if ymax is None: ymax=bounds[3]

        if self.quantize:
            xmin=self.tx*np.floor(xmin/self.tx)
            xmax=self.tx*np.ceil( xmax/self.tx)
            ymin=self.ty*np.floor(ymin/self.ty)
            ymax=self.ty*np.ceil( ymax/self.ty)
            
        nx = int(np.ceil((xmax - xmin)/self.tx))
        ny = int(np.ceil((ymax - ymin)/self.ty))

        print("Tiles: %d x %d"%(nx,ny))

        for xi in range(nx):
            for yi in range(ny):
                ll = [xmin+xi*self.tx,
                      ymin+yi*self.ty]
                ur = [ll[0]+self.tx,
                      ll[1]+self.ty]
                # populate some local variables for giving to the filename format
                left=ll[0]
                right=ll[0]+self.tx
                bottom=ll[1]
                top = ll[1]+self.ty
                dx = self.dx
                dy = self.dy

                bounds = np.array([left,right,bottom,top])
                print("Tile ",bounds)
                
                output_fn = os.path.join(self.output_dir,self.filename_fmt%locals())
                self.tile_fns.append(output_fn)
                
                print("Looking for output file: %s"%output_fn)

                if self.force or not os.path.exists(output_fn):
                    pad_x=self.pad/self.dx
                    pad_y=self.pad/self.dy
                    pad_bounds=np.array([left-pad_x,right+pad_x, bottom-pad_y, top+pad_y])
                    blend = self.f.to_grid(dx=self.dx,dy=self.dy,bounds=pad_bounds)
                    if self.fill_iterations + self.smoothing_iterations > 0:
                        print("Filling and smoothing")
                        blend.fill_by_convolution(self.fill_iterations,self.smoothing_iterations)
                    print("Saving")
                    if self.post_render:
                        blend=self.post_render(blend,output_fn=output_fn,bounds=bounds,pad_bounds=pad_bounds)
                    if self.pad>0:
                        blend=blend.crop(bounds)
                    blend.write_gdal( output_fn )
                    print("Done")
                else:
                    print("Already exists. Skipping")

    def merge(self):
        # and then merge them with something like:
        # if the file exists, its extents will not be updated.
        output_fn=os.path.join(self.output_dir,'merged.tif')
        os.path.exists(output_fn) and os.unlink(output_fn)
        log.info("Merging using gdal_merge.py")
        
        # Try importing gdal_merge directly, which will more reliably
        # find the right library since if we got this far, python already
        # found gdal okay.  Unfortunately it's not super straightforward
        # to get the right way of importing this, since it's intended as
        # a script and not a module.
        try:
            from Scripts import gdal_merge
        except ImportError:
            log.info("Failed to import gdal_merge, will try subprocess") 
            gdal_merge=None

        cmd=["python","gdal_merge.py","-init","nan","-a_nodata","nan",
             "-o",output_fn]+self.tile_fns
        
        log.info(" ".join(cmd))
                
        if gdal_merge:
            gdal_merge.main(argv=cmd[1:])
        else:
            # more likely that gdal_merge.py is on PATH, than the script itself will
            # be seen by python, so drop python, and invoke script directly.
            subprocess.call(" ".join(cmd[1:]),shell=True)



    
if __name__ == '__main__':
    topobathy = "/home/rusty/classes/research/spatialdata/us/ca/suntans/bathymetry/ca-topobathy/85957956/85957956/hdr.adf"
    corrected_fn = "/home/rusty/classes/research/spatialdata/us/ca/suntans/bathymetry/usgs/southbay-corrected.xyz"

    corrected = XYZText(corrected_fn,projection="EPSG:26910")
    corrected2 = corrected.rectify()

    tile = GdalGrid(topobathy)

    zoom_ll = corrected.bounds_in_cs(tile.projection())

    tile_cropped = tile.crop(zoom_ll)

    tile_utm = tile_cropped.reproject(to_projection=corrected.projection())



    # arithmetic interface may change...
    # diff = tile_utm - corrected2
    corr_on_tile = tile_utm.regrid(corrected2)

    corr_cv = CurvilinearGrid(tile_utm.X,corr_on_tile,projection=tile_utm.projection())


    subplot(211)
    corrected.plot(vmin=-10,vmax=2)

    subplot(212)
    tile_utm.plot(vmin=-10,vmax=2)

