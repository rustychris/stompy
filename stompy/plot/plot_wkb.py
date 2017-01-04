from __future__ import print_function
from safe_pylab import *
from matplotlib.collections import PatchCollection,LineCollection


def plot_linestring(ls,**kwargs):
    ax=kwargs.pop('ax',gca())
    c = array(ls.coords)
    return ax.plot( c[:,0],c[:,1],**kwargs)[0]

def plot_multilinestring(mls,**kwargs):
    ax=kwargs.pop('ax',gca())
    if mls.type == 'MultiLineString':
        segs = [array(ls.coords) for ls in mls.geoms]
        coll = LineCollection(segs,**kwargs)
        ax.add_collection(coll)
        return coll
    else:
        return plot_linestring(mls,**kwargs)
        

########
# New, non-hacked way to plot polygons with holes

# From: http://sgillies.net/blog/1013/painting-punctured-polygons-with-matplotlib/
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes

def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = concatenate(
                    [asarray(polygon.exterior)]
                    + [asarray(r) for r in polygon.interiors])
    codes = concatenate(
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

    ax = kwargs.pop('ax',gca())
    
    patch = poly_to_patch(p,*args,**kwargs)
    ax.add_patch(patch)
    return patch

def plot_multipolygon(mp,*args,**kwargs):
    if 'holes' in kwargs:
        print("dropping obsolete holes keyword argument")
        del kwargs['holes']
    ax = kwargs.pop('ax',gca())
    coll = multipoly_to_patches(mp,*args,**kwargs) 
    ax.add_collection( coll ) 
    return coll


def plot_wkb(g,*args,**kwargs):
    if g.type == 'MultiPolygon':
        return plot_multipolygon(g,*args,**kwargs)
    elif g.type=='Polygon':
        return plot_polygon(g,*args,**kwargs)
    elif g.type == 'MultiLineString':
        return plot_multilinestring(g,*args,**kwargs)
    elif g.type =='LineString':
        return plot_linestring(g,*args,**kwargs)
    else:
        raise Exception("no match to type")
    
