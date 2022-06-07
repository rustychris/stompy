from __future__ import print_function
from builtins import zip
from builtins import range
from builtins import object
## interp_coverage.py

# The usable interface to this will probably be part of field.py

from functools import reduce

import shapely.geometry

try:
    from . import constrained_delaunay
    import CGAL
except ImportError:
    # print "interp_coverage is not available"
    CGAL = None

# Read a shapefile with overlapping polygons and interpolate in the intersecting regions

# The direct application is merging bathymetry datasets - define a set of polygons that
# cover the domain where each polygon is linked to a dataset.
# For a point that lies in exactly one of these polygons, the corresponding dataset
# determines the value at the point.
# For a point that lies in more than one polygon, it's value is weighted linearly between
# each of the polygons, depending on which border it is closest to.

# Algorithm:
#   Compute a constrained Delaunay triangulation of all edges making up the polygons.
#   Special care must be taken at the intersections
#   Each vertex is given a vector defining its membership in the boundaries of the polygons.
#   Most vertices then have a vector with exactly one nonzero entry.
#   Vertices that define the intersection of polygon boundaries will have nonzero entries
#   for each of the boundaries.

# hmm - may have a problem here - really we want the distance from a given point to each of
#  the boundaries that define the intersection.  But with just a straight DT there's no
#  guarantee that we'll be a in a triangle that touches both polygon boundaries.

# So we need a point-polygon distance.  Does shapely or CGAL help us out?
# This would mean that the DT would tell us which polygons to consult, and the distance
# function would give the weightings.
# Shapely gives a distance function for min. distance between arbitrary geometries.


# Handling an intersection of more than two regions is a little tricky -
# Is it too slow to do an all-combinations intersection?  basically want to enumerate every
#  part of the Venn diagram
# Maybe look at each region in order
# compare it against all subsequent regions
# If there is an intersection, split the region into the non-intersecting portion, and the
# intersecting portion.  Each region, whether original or after being carved up, maintains
# a list of the regions that were intersected to get this region.
#

# Now we have the plane sliced up into pieces, where each polygon defines the intersection
# of N regions.

# Given some point, we find the polygon it lives in - this tells us which regions are going
# to have nonzero weights.

# How to assign weights within a polygon?
#  Definitely require that on a boundary with another region whose src vector has one nonzero
#  we have to be equal to that value.
# For each participating region, find the distance from a point to the edge of that region.
# This is probably done by taking the LinearRing for each regiong and finding the min. distance
# to it.  This gives us a distance for each region.  Then weights can be applied, probably linearly?
 
import numpy as np

import shapely.wkb
import shapely.geometry

from collections import OrderedDict

# from osgeo import ogr
from stompy.spatial import wkb2shp
from stompy.plot import plot_wkb

class Region(object):
    def __init__(self,feat):
        self.items=OrderedDict()

        for fld in feat.dtype.names:
            self.items[fld]=feat[fld]

        self.geom=feat['geom']
        self.boundary=self.geom.boundary
        self.items['boundary'] = self.boundary

    def plot(self):
        plot_wkb.plot_multipolygon(self.geom)
        
    def distance_to_boundary(self,pnt):
        return self.boundary.distance(pnt)

    def identifier(self):
        return ":".join([str(self.items[f])[:20]
                         for f in self.items
                         if f not in ['geom','boundary'] ])
    

class InterpCoverage(object):
    boneyard = []
    
    def __init__(self,regions_shp=None,regions_data=None,subset=None):
        if regions_shp is not None:
            self.regions_shp = regions_shp
            self.regions_data=None
        else:
            assert regions_data is not None
            self.regions_data=regions_data
            self.regions_shp=None

        self.load_shp(subset=subset)
        self.find_intersections()
        
    def load_shp(self,subset=None):
        if self.regions_shp is not None:
            self.shp_data=wkb2shp.shp2geom(self.regions_shp)
        else:
            self.shp_data=self.regions_data

        if subset is not None:
            self.shp_data=self.shp_data[subset]
        
        self.regions = [ Region(rec) for rec in self.shp_data]

    def find_intersections(self):
        geoms = [r.geom for r in self.regions]
        ii = np.arange(len(geoms))
        srcs = [ (ii==i) for i in range(len(geoms))]

        # DBG
        fns = [r.identifier() for r in self.regions]
        
        coverage = list(zip( geoms,srcs ))
        
        # iterate through the geoms, expanding coverage by however
        # each geom cuts it up

        total_coverage = None
        for i in range(len(geoms)):
            #print "processing geom for ",fns[i]
            
            g = geoms[i]

            # at the same time, we compute the union of all regions to speed up
            # out-of-bounds queries later on.
            if total_coverage is None:
                total_coverage = g
            else:
                total_coverage = total_coverage.union(g)
            
            new_coverage = []

            # take a possibly multipart geometry, and the source vector to go with it,
            # and append to new_coverage
            def add_new_geoms(new_geom,new_src):
                # geom_in_g may be multipart:
                if new_geom.type == 'Polygon':
                    new_coverage.append( (new_geom,new_src) )
                elif new_geom.type == 'MultiPolygon' or new_geom.type == 'GeometryCollection':
                    # Avoid memory related seg faults by not freeing these geometries
                    self.boneyard.append(new_geom)
                    for g in new_geom.geoms:
                        if g.type == 'Polygon':
                            new_coverage.append( (g,new_src) )
                        else:
                            print("Skipping sub-geometry of type %s"%g.type)
                            print(g.wkt)
            
            for gcover,src in coverage:
                if src[i]:
                    #print "Comparing %d with src %s"%(i,src)
                    # This coverage is already known to be a subset of g
                    new_coverage.append( (gcover,src) )
                else:
                    geom_in_g = gcover.intersection( g )
                    if geom_in_g.area > 0.0:
                        src_in_g = src.copy()
                        src_in_g[i] = 1

                        add_new_geoms(geom_in_g,src_in_g)
                        
                        # And check for the remainder:
                        geom_not_in_g = gcover.difference( g )
                        if geom_not_in_g.area > 0.0:
                            add_new_geoms( geom_not_in_g,src )
                    else:
                        # they don't overlap at all - coverage doesn't change
                        new_coverage.append( (gcover,src) )
            coverage = new_coverage
            
        self.intersect_srcs  = [src for gcover,src in coverage]
        self.intersect_geoms = [gcover for gcover,src in coverage]
        self.total_coverage = total_coverage.convex_hull

    def plot_regions(self):
        for r in self.regions:
            r.plot()

    def plot_intersections(self):
        for ig in self.intersect_geoms:
            plot_wkb.plot_multipolygon(ig)

    tri_to_geom_src = None
    def prepare_point_to_intersection(self,tol=1e-3):
        """ build up a DT that will speed up queries to the regions
        """
        if CGAL is None:
            print("No CGAL.")
            return
        
        # get the raw vertices from the polygons
        all_rings = []
        for g in self.intersect_geoms:
            all_rings.append( np.array(g.exterior.coords) )
            for r in g.interiors: # probably no interiors, but just to be safe
                all_rings.append( np.array(r.coords) )

        all_vertices = np.concatenate(all_rings)

        # find the unique ones - ConstrainedDelaunay isn't quite ready to do all
        # of this.
        v_hash = {}
        uniq_vertices = []
        for a,b in all_vertices:
            k = (a,b)
            if k not in v_hash:
                # search to see if maybe we are close enough to somebody else.
                v_hash[k] = len(uniq_vertices)
                uniq_vertices.append( k )
        
        vertices = np.array(uniq_vertices)

        # Go over the rings and create unique edges
        e_hash = {}
        for r in all_rings:
            # find indices for each of the points
            ri =[v_hash[ (a,b) ] for a,b in r]
            for i in range(len(ri)):
                c,d = (ri[i],ri[(i+1)%len(ri)])
                if c==d:
                    continue
                elif c > d: # canonicalize edge naming
                    e_hash[(d,c)] = 1
                else:
                    e_hash[(c,d)] = 1

        edges = np.array( list(e_hash.keys()) )

        cdf = constrained_delaunay.ConstrainedXYZField(X=vertices,F=np.zeros(len(vertices),np.int32),
                                                       edges=edges)
        cdf.fix_duplicates(tol=tol)

        # now for each of the triangles figure out which region it goes with.
        t = cdf.tri()

        ## So how do we actually use the triangulation?
        #  ultimately, probably worth it to use this for the distance queries, too.

        #  what can this thing do quickly?
        #  if we feed it to a linear interper, can it interpolate whole vectors? or just
        #  scalars? just scalars.
        #  so we could have an interper for each region
        #  but I'm not sure that we should go straight for the distance queries
        #  for now, just go with the membership.  So each triangle can be associated with exactly one
        #  intersect_geom.  So what we need is a quick way to find the index of the right intersect_geom
        #  for all points in a grid.

        # can we go through the faces and associate an intersect_geom index with each?
        # the DT will give us a quick CGAL way to find a face given a point.  Then we'd have to hash that
        # face
        # we can identify each face by its vertex indices - these should stay the same.

        DT = cdf.DT
        tri_to_geom_src = [None]*DT.number_of_faces() # maps a DT triangle to its (intersect_geom,source)
        face_to_tri_index = {} # maps tuple of vertex indices to a triangle index

        # Find centers of each triangle
        tri_centers = np.mean(cdf.X[ cdf.tri().triangle_nodes ],axis=1)

        # put together a map from (v1,v2,v3) to triangle index - this is required to identify
        # faces coming back from CGAL
        i = 0

        # this is a bit slow - maybe 30s
        print("Associating each triangle with its intersection polygon")
        t = cdf.tri()
        for i in range(len(tri_centers)):
            vvv = tuple(t.triangle_nodes[i])
            face_to_tri_index[ vvv ] = i
            # and go ahead and figure out what intersect_geom it goes to
            if i>0: # try the last hit...
                pnt = shapely.geometry.Point( tri_centers[i] )
                if tri_to_geom_src[i-1][0] and tri_to_geom_src[i-1][0].contains(pnt):
                    tri_to_geom_src[i] = tri_to_geom_src[i-1]
                    continue
            tri_to_geom_src[i] = self.point_to_intersection( tri_centers[i] )

        # And save the results:
        self.tri_to_geom_src = tri_to_geom_src
        self.face_to_tri_index = face_to_tri_index
        self.DT = DT
        self.cdf = cdf
        
    last_face = None
    def point_to_intersection(self,pnt):
        """ Given a shapely point, find the region it should live in """
        # Find the intersecting region that the point lives in -
        # this would be sped up considerably by a DT and keeping queries locally
        # correlated

        if self.tri_to_geom_src is not None:
            p = CGAL.Point_2(pnt[0],pnt[1])
            if self.last_face is not None:
                f = self.DT.locate(p,self.last_face)
            else:
                f = self.DT.locate(p)
            if f is not None:
                self.last_face = f

            # possible that we hit an infinite face of the triangulation -
            # give up in that case.
            if f is None or self.DT.is_infinite(f):
                return None,None
            
            k = (f.vertex(0).info(),f.vertex(1).info(),f.vertex(2).info())
            try:
                i = self.face_to_tri_index[k]
            except KeyError:
                print("ERROR: Failed to find a triangle from the vertices.  This *could* be because there are some colinear edges")
                print("and the code is not smart enough to sort that out")
                print("Check around these vertices:")
                for i in k:
                    if i is not None:
                        print("vertex: %d (%f,%f)"%(i,self.cdf.X[i,0],self.cdf.X[i,1]))
                raise
            return self.tri_to_geom_src[i]
        else:
            pnt = shapely.geometry.Point(pnt)
            for g,src in zip(self.intersect_geoms,self.intersect_srcs):
                if g.contains(pnt):
                    return g,src

        #print "Point wasn't found in any region!"
        return None,None

    last_hit = (None,None)
    def calc_weights(self,parray):
        """ For the nonzero entries in src, compute the min. distance from the point
        to the boundary of that region
        Given a point or points as (...,2) size array, return the vector of weights to apply
        """
        plist = parray.reshape( (-1,2) )
        # Shapely doesn't do well with integer data
        plist = plist.astype(np.float64)
        
        Npnts = len(plist)
        
        pnts = [shapely.geometry.Point(p) for p in plist]
        
        weights = np.zeros( (Npnts,len(self.regions)), np.float64)

        for i in range(Npnts):
            if i>500 and i%5000==0:
                print("calc_weights: %.2f%%"%(100.0*i/Npnts))
            # This is the slowest step, and the one that can be sped up the most by
            # the DT.
            # exploiting spatial correlation may speed it up, though.
            # still, in the test case the majority of points fall outside all regions, and as
            # such we end up testing them against every region.
            # if (self.last_hit[0] is not None) and (self.last_hit[0].contains(pnts[i])):
            #     g,src = self.last_hit
            # elif (self.last_hit[0] is None) and not self.total_coverage.contains(pnts[i]):
            #     # out of bounds queries - tests that the point isn't in anybody.
            #     g,src = None,None
            # else:
            
            g,src = self.point_to_intersection(plist[i])
                
            self.last_hit = g,src

            if src is None:
                continue

            # first calculate the weights:
            for j in np.nonzero(src)[0]:
                weights[i,j] = self.regions[j].distance_to_boundary(pnts[i])
                
            # normalize - for now assume linear.
            weights[i,:] /= weights[i,:].sum()

        new_shape = parray.shape[:-1] + (len(self.regions),)
        return weights.reshape( new_shape )
    


if __name__ == '__main__':
    # Testing:

    ic = InterpCoverage('/home/rusty/classes/research/suntans/bathy_interp/test_input1.shp')

    # Sample plot (depends on there being exactly 3 regions)
    # First find the envelope of everyone:
    total = reduce(lambda x,y: x.union(y), [r.geom.envelope for r in ic.regions])
    minx,miny,maxx,maxy = total.bounds

    x = linspace(minx,maxx,300)
    y = linspace(miny,maxy,300)

    X,Y = meshgrid(x,y)

    pnts = concatenate( (X[...,newaxis],Y[...,newaxis]), axis=2)

    C = ic.calc_weights(pnts)
    
    cla()
    imshow(C,extent=(minx,maxx,miny,maxy),origin='bottom',interpolation='nearest')        

    # it's a little slow, but probably bearable -
    #  a quick profiling run shows that for calc_weights, the time is spent...
    # 27s / 38s in point_to_intersection
