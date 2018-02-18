"""
Smooth a polygon (typically a shoreline) based on spatially varying length scales
"""
from __future__ import print_function

# Delaunay Method:
#   1. Compute Delaunay triangulation of the domain.  
#   1b Upsample edges where the clearance is smaller than the point
#      spacing along the boundary, and recompute triangulation
#   Remove triangles with a radius smaller than the local scale
#   Reconstruct the new boundary

# For 1b:
#   What about the existing subdivide algorithm?
#     runs, although it does get a few 'whoa's.  probably from steiner points.

# input: a Field object that can provide a local scale length anywhere
#    in the domain
#        a Polygon (possibly with holes) covering the entire domain

# load_shp,
import numpy as np

from ..spatial import field, wkb2shp
from . import trigrid
from shapely import wkb, geometry

from ..spatial import medial_axis as ma

# Issues: seems that triangle doesn't like input where one node is shared between
#  rings.  This comes from when a channel is triangulated and exactly one triangle
#  is removed - it's not good from the simplification point of view, either...
#  

def adjust_scale(geo,scale=None,r=None,min_edge_length=1.0):
    """ The alternative to smooth - creates a new scale that is smaller
    where needed to match the clearances in geo.
    This is crufty and probably only a starting point for something
    functional.

    geo: the shoreline polygon
    scale: the XYZField describing the requested scale
    r: the telescoping rate
    min_edge_length: for computing the medial axis, this is the smallest edge that
       can force insertion of Steiner points
    """
    # global adjust_geo,new_scale,tri,bdry_ma

    if not isinstance(scale,field.XYZField):
        raise ValueError("density must be an XYZField")

    scale = field.ConstrainedScaleField(scale.X, scale.F)

    if r is not None:
        scale.r = r
    
    bdry_ma = ma.Boundary( geo=geo )
    if min_edge_length is not None:
        bdry_ma.min_edge_length = min_edge_length
        
    bdry_ma.subdivide_iterate()

    tri = bdry_ma.triangulation()

    vcenters = tri.vcenters() 
    radii = tri.radii()
    diam = 2*radii

    # The possible new way, starting with too many points and paring down...
    if scale is not None:
        new_X = np.concatenate( (scale.X,vcenters) )
        new_F = np.concatenate( (scale.F,diam) )
    else:
        new_X=vcenters
        new_F=diam

    scale = field.ConstrainedScaleField(X,F)
    scale.remove_invalid()

    return scale


def apollonius_scale(geo,r,min_edge_length=1.0,process_islands=True):
    """ Return an apollonius based field giving the scale subject to
    the local feature size of geo and the telescoping rate r
    """
    bdry_ma = ma.Boundary( geo=geo )
    
    if min_edge_length is not None:
        bdry_ma.min_edge_length = min_edge_length
        
    bdry_ma.subdivide_iterate()

    tri = bdry_ma.triangulation()

    vcenters = tri.vcenters() 
    radii = tri.radii()
    diam = 2*radii

    if process_islands:
        print("Hang on.  Adding scale points for islands")

        island_centers = []
        island_scales = []
        
        for int_ring in geo.interiors:
            p = int_ring.convex_hull

            points = np.array(p.exterior.coords)
            center = points.mean(axis=0)

            # brute force - find the maximal distance between
            # any two points.  probably a smart way to do this,
            # but no worries...
            max_dsqr = 0
            for i in range(len(points)):
                pa = points[i]
                for j in range(i,len(points)):
                    d = ((pa - points[j])**2).sum()
                    max_dsqr = max(d,max_dsqr)

            feature_scale = sqrt( max_dsqr )
            print("Ring has scale of ",feature_scale)

            island_centers.append( center )
            # this very roughly says that we want at least 4 edges
            # for representing this thing.
            #   island_scales.append( feature_scale / 2.0)
            # actually it's not too hard to have a skinny island
            # 2 units long that gets reduced to a degenerate pair
            # of edges, so go conservative here:
            island_scales.append( feature_scale / 3.0 )

        island_centers = np.array(island_centers)
        island_scales = np.array(island_scales)

        if len(island_centers) > 0:
            vcenters = np.concatenate( (vcenters,island_centers) )
            diam = np.concatenate( (diam,island_scales) )
        print("Done with islands")

    # The possible new way, starting with too many points and paring down...
    scale = field.ApolloniusField(vcenters,diam)

    return scale
    

# def smooth(geo,scale):
#     """
#     given a polygon geo, possibly with interior rings, and a scale
#     DensityField or float
#     trim 'concave' features that are smaller than the scale and return
#     a new polygon
#     """
# 
#     if isinstance(scale,number):
#         print "Converting scale to a DensityField"
#         scale = field.ConstantField(scale)
# 
#     bdry_ma = ma.Boundary( geo=geo )
# 
#     bdry_ma.subdivide_iterate()
# 
#     # vor = bdry_ma.vor()
#     tri = bdry_ma.triangulation()
# 
#     # so can we trim based just on the triangles?
#     # radius can be calculated in the same way that voronoi points are calculated
#     # in trigrid
#     vcenters = tri.vcenters() 
#     radii = tri.radii()
# 
#     scales = scale(vcenters)
#     # pretend that scale is a cutoff on channel width
#     scales = scales / 2 
# 
#     elements_to_remove = radii < scales # this is where local scale should come in
# 
#     # To avoid weird cutoffs from removing exactly one triangle in a channel, it
#     # would be better to clump the bad elements, where any neighboring elements that
#     # are close to the cutoff are also removed.  Alternatively, separate those nodes
#     # by some small distance so that the two rings are not tangent
# 
#     good_elements = tri.elements[~elements_to_remove]
# 
#     print "Of %d elements, %d will be kept"%(len(radii),len(good_elements))
# 
#     i = array([[0,1],[1,2],[2,0]])
# 
#     # good edges are then edges that appear in exactly one element
#     all_edges = good_elements[:,i]
#     # good_elements was Nc x 3
#     # and so now I think it will be Nc x 3 x 2
#     # expand the first two dimensions, so we have a regular edges array
#     all_edges_cont = all_edges.reshape( (good_elements.shape[0]*good_elements.shape[1],
#                                          2 ) )
# 
#     print "building hash of edges"
# 
#     # build up a hash of ordered edges
#     edge_hash = {}
# 
#     for i in range(len(all_edges_cont)):
#         k = all_edges_cont[i,:]
#         if k[0] > k[1]:
#             k=k[::-1]
#         # k.sort()
#         k = tuple(k)
#         if not edge_hash.has_key(k):
#             edge_hash[k] = 0
#         edge_hash[k] += 1
# 
#     print "Selecting boundary edges"
#     good_edges = []
# 
#     for k in edge_hash:
#         if edge_hash[k] == 1:
#             good_edges.append(k)
# 
#     good_edges = array(good_edges)
#     tri.edges = good_edges
# 
#     # tri.plot_edges()
#     # bdry_ma.plot()
#     # axis('equal')
# 
#     # then we still have to clean up any figure-eight sorts of issues,
#     # and string it all back together as rings, getting rid of any
#     # orphaned rings
# 
#     # how to deal with figure eights?
#     #   - maybe start traversing rings.  then when more than two edges
#     #     leave a particular node, choose the one most CCW
#     # this code is all part of trigrid
# 
#     print "Finding polygons from edges"
# 
#     tgrid = trigrid.TriGrid(points=tri.nodes,
#                             edges = tri.edges)
#     tgrid.verbose = 2
#     polygons = tgrid.edges_to_polygons(None) # none=> use all edges
# 
#     smooth.all_polygons = polygons # for debugging..
# 
#     print "done with smoother"
#     return polygons[0]


def remove_small_islands(shore_poly,density):
    ext_ring = np.array(shore_poly.exterior.coords)
    int_rings = []

    for int_ring in shore_poly.interiors:
        p = int_ring.convex_hull
        
        points = np.array(p.exterior.coords)
        center = points.mean(axis=0)
        scale = density(center)

        # brute force - find the maximal distance between
        # any two points.  probably a smart way to do this,
        # but no worries...
        max_dsqr = 0
        for i in range(len(points)):
            pa = points[i]
            for j in range(i,len(points)):
                d = ((pa - points[j])**2).sum()
                max_dsqr = max(d,max_dsqr)

        feature_scale = np.sqrt( max_dsqr )
        print("Ring has scale of ",feature_scale)

        # maybe overkill, but with feature_scale ~ scale,
        # that leads to 2 edges for the whole feature, which
        # gets us into trouble.  Even at 2*scale it is still
        # possible for stuff to suck, but give it a try...

        # would be nice to be smarter about this, maybe buffer
        # the ring to 'round up' to a good size, but then we'd
        # have to check for intersections created by the buffering
        if feature_scale > 2*scale:
            int_rings.append(np.array(int_ring.coords))

    new_poly = geometry.Polygon(ext_ring,int_rings)
    return new_poly
    
    
