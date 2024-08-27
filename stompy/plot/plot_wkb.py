from __future__ import print_function
from matplotlib.collections import PatchCollection,LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy as np
from .. import utils

def plot_linestring(linestring,**kwargs):
    ax=kwargs.pop('ax',plt.gca())
    c = np.array(linestring.coords)
    return ax.plot( c[:,0],c[:,1],**kwargs)[0]

def plot_multilinestring(mls,**kwargs):
    ax=kwargs.pop('ax',plt.gca())
    if mls.geom_type == 'MultiLineString':
        segs = [np.array(ls.coords) for ls in mls.geoms]
        coll = LineCollection(segs,**kwargs)
        ax.add_collection(coll)
        return coll
    else:
        return plot_linestring(mls,**kwargs)
        
########
# New, non-hacked way to plot polygons with holes
# From: http://sgillies.net/blog/1013/painting-punctured-polygons-with-matplotlib/

def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    # unsure of difference between CLOSEPOLY and leaving as is.
    # codes[-1] = Path.CLOSEPOLY # doesn't seem to make a difference
    return codes

def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.

    # 20170707: matplotlib pickier about ordering of internal rings, may have
    # reverse interiors.
    # 20170719: shapely doesn't guarantee one order or the other
    def ensure_orientation(a,ccw=True):
        """
        take an array-like [N,2] set of points defining a polygon,
        return an array which is ordered ccw (or cw is ccw=False)
        """
        a=np.asarray(a) # pre-shapely 2
        area=utils.signed_area(a)
        if ccw == (area<0):
            a=a[::-1]
        return a

    vertices = np.concatenate(
        [ ensure_orientation(polygon.exterior.coords,ccw=True)]
        + [ ensure_orientation(r.coords,ccw=False) for r in polygon.interiors])
    codes = np.concatenate(
        [ring_coding(polygon.exterior)]
        + [ring_coding(r) for r in polygon.interiors])
    return Path(vertices, codes)

def poly_to_patch(polygon,**kwargs):
    return PathPatch(pathify(polygon), **kwargs)

def multipoly_to_patches(multipoly,*args,**kwargs):
    patches = [poly_to_patch(p) for p in multipoly.geoms]
    return PatchCollection(patches,*args,**kwargs)

def plot_polygon(p,*args,**kwargs):
    if 'holes' in kwargs:
        print("dropping obsolete holes keyword argument")
        del kwargs['holes']

    ax = kwargs.pop('ax',plt.gca())
    
    patch = poly_to_patch(p,*args,**kwargs)
    ax.add_patch(patch)
    return patch

def plot_multipolygon(mp,*args,**kwargs):
    if 'holes' in kwargs:
        print("dropping obsolete holes keyword argument")
        del kwargs['holes']
    ax = kwargs.pop('ax',plt.gca())
    coll = multipoly_to_patches(mp,*args,**kwargs) 
    ax.add_collection( coll ) 
    return coll


def plot_wkb(g,*args,**kwargs):
    if g.geom_type == 'MultiPolygon':
        return plot_multipolygon(g,*args,**kwargs)
    elif g.geom_type=='Polygon':
        return plot_polygon(g,*args,**kwargs)
    elif g.geom_type == 'MultiLineString':
        return plot_multilinestring(g,*args,**kwargs)
    elif g.geom_type =='LineString':
        return plot_linestring(g,*args,**kwargs)
    else:
        raise Exception("no match to type")
    
