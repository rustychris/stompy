"""
Combine the constrained delaunay algorithm from CGAL with the interpolation methods
available in scikit.delaunay.
"""
from __future__ import print_function

# First - can we manually set the fields in a delaunay object?
# this is deprecated in matplotlib
import collections

# from shapely import wkb
# from osgeo import ogr
from collections import defaultdict
from .. import priority_queue

from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

import numpy as np

from . import field, wkb2shp

from ..grid.trigrid import circumcenter
from ..grid.exact_delaunay import Triangulation as EDT

class ConstrainedXYZField(field.XYZField):
    """ Like an XYZField, but with an additional attribute - a set of edges
    connecting some or all the points.

    These edges are put into a constrained Delaunay Triangulation which can
    then be used for interpolation.

    Edges can be specified directly, or read from a linestring shapefile.
    """
    def __init__(self,X,F,projection=None,from_file=None,edges=None):
        """
        X: [N,2] ndarray of node locations
        F: [N] ndarray of values at nodes
        edges: [M,2] ndarray giving pairs of nodes which form edges
        """
        field.XYZField.__init__(self,X=X,F=F,projection=projection,from_file=from_file)
        if edges is None:
            edges = np.zeros( (0,2),np.int32 )
        self.edges = edges

    # Pickle support, since the DT can't be pickled
    def __getstate__(self):
        d = super(ConstrainedXYZField,self).__getstate__()
        d['DT'] = None
        d['_tri'] = None
        d['vh_info'] = None
        return d

    def __setstate__(self,d):
        self.__dict__.update(d)

    # nearest neighbor doesn't play nice with the constrained triangulation.
    default_interpolation='linear'
    @staticmethod 
    def read_shps(shp_names,value_field='value'):
        X = []
        F = []
        edges = []

        value_field=value_field.lower()
        
        for shp_name in shp_names:
            print("Reading %s"%shp_name)

            layer=wkb2shp.shp2geom(shp_name,fold_to_lower=True)
            
            for row in layer:
                value = row[value_field]
                if not (np.isfinite(value) and value>0):
                    continue
                
                geo = row['geom']

                coords = np.array(geo.coords)
                mask = np.all(coords[0:-1,:] == coords[1:,:],axis=1)
                mask=np.r_[False,mask]
                if sum(mask)>0:
                    print("WARNING: found duplicate points in shapefile")
                    print(coords[mask])
                    coords = coords[~mask]

                i0 = len(X) # offset for our coordinate indices

                # Add the first point 'manually':
                F.append( value )
                X.append( coords[0] )

                if len(coords) > 1: # handle linestrings
                    N = len(coords)
                    
                    for i in range(1,N-1):
                        F.append( value )
                        X.append( coords[i] )
                        edges.append( [i0+i-1,i0+i] )

                    # handle last point manually:
                    if np.all(coords[-1] == coords[0]):
                        # so we don't add coords[N-1] - instead connect the
                        # previous one to the first one
                        edges.append( [i0+N-2,i0] )
                    else:
                        # or if the line just stops -
                        F.append( value )
                        X.append( coords[N-1] )
                        edges.append( [i0+N-2,i0+N-1] )
                    
        X = np.array( X )
        F = np.array( F )
        edges = np.array( edges )
        return ConstrainedXYZField(X=X, F=F, from_file=shp_names[0], edges=edges)

    @staticmethod 
    def from_polylines(lines,values):
        X = []
        F = []
        edges = []

        for coords,line_values in zip(lines,values):
            i0 = len(X) # offset for our coordinate indices

            # check here whether values is a scalar and should be repeated, or
            # a vector to be applied along the line...
            line_values=np.atleast_1d(line_values)
            if len(line_values)==1:
                line_values=line_values[0]*np.ones(len(coords))
            
            # Add the first point 'manually':
            F.append( line_values[0] )
            X.append( coords[0] )

            if len(coords) > 1: # handle linestrings
                N = len(coords)

                for i in range(1,N-1):
                    F.append( line_values[i] )
                    X.append( coords[i] )
                    edges.append( [i0+i-1,i0+i] )

                # handle last point manually:
                if np.all(coords[-1] == coords[0]):
                    # so we don't add coords[N-1] - instead connect the
                    # previous one to the first one
                    edges.append( [i0+N-2,i0] )
                else:
                    # or if the line just stops -
                    F.append( line_values[N-1] )
                    X.append( coords[N-1] )
                    edges.append( [i0+N-2,i0+N-1] )

        X = np.array( X )
        F = np.array( F )
        edges = np.array( edges )
        return ConstrainedXYZField(X=X,F=F,edges = edges)

    # And overload how the triangulation is computed
    def tri(self):
        """
        Return a matplotlib triangulation corresponding to the
        nodes and edges
        """
        if len(self.edges) == 0:
            return super(ConstrainedXYZField,self).tri()
        
        if self._tri is None:
            # the original code used CGAL, but since these are typically not very
            # large inputs, use exact_delaunay for ease.  The original code
            # also went through a lot of extra work to construct the triangulation,
            # but recent matplotlib handles this for us, I think.
            DT=EDT()
            DT.bulk_init(points=self.X)
            
            print("Adding constraints")
            # Insert edge constraints
            for a,b in self.edges:
                DT.add_constraint(a, b)

            self._tri = DT.mpl_triangulation()

        return self._tri

    def to_grid(self,*args,**kwargs):
        if 'interp' in kwargs:
            if kwargs['interp']!='linear' and len(self.edges) > 0:
                print("Warning - constrained triangulation doesn't like natural neighbors")
        else:
            kwargs['interp']='linear'
            
        return field.XYZField.to_grid(self,*args,**kwargs)

    def fix_duplicates(self,tol=0.0):
        """ if tol > 0, also check for points that differ by less than that
        distance, and treat them as duplicates
        """
        points = {} # hash of (x,y) to valid, unique index

        # index by old index, get the new index
        # starts as 1:1
        mapping = np.arange(len(self.F))

        self.build_index()
        
        for i in range(len(self.F)):
            x,y = self.X[i]
            k = (x,y)
            if k in points:
                # print "duplicate ",k
                mapping[i] = points[k]
                continue

            # I think this part is buggy.
            if tol > 0.0:
                # choose the smallest index here as the canonical point
                nbrs = self.within_r(self.X[i],tol)
                if len(nbrs) > 0:
                    canon = nbrs.min()
                    if canon < i:
                        # print "Remapping nearby point"
                        # and canon may itself be remapped
                        mapping[i] = mapping[canon]
                        points[k] = mapping[i]
                        continue
                    # else: # There are duplicates, but we have the lowest index, so
                    # we stay.  fall through.
                # else: # no duplicates.  fall through
            points[k] = i

        # old indices of the good points
        valid = np.unique( mapping )

        # so mapping takes an old, non-unique index and maps
        # to the unique indices.

        # renumber takes old unique indices, and maps to the
        # new indices
        renumber = -10000*np.ones(len(mapping),np.int32) # start with nonsense
        # only the valid ones get renumbered
        renumber[valid] = np.arange(len(valid))

        # and combine these, to map all old indices to unique new indices
        mapping = renumber[mapping]

        if len(self.edges) > 0:
            # translate all of the vertices
            self.edges = mapping[self.edges]

            # And the possibility of degenerate edges:
            bad_edges = (self.edges[:,0] == self.edges[:,1])
            self.edges = self.edges[~bad_edges,:]

        self.index = None
        self.F = self.F[valid]
        self.X = self.X[valid,:]
        self._edge_map = None
        self._neighbor_map = None
        self._tri = None

    _edge_map = None
    _neighbor_map = None
    
    def find_edge(self,a,b):
        """ Returns the index of an edge if it was specified as a constrained,
        or None if the given pair of indices do not describe an edge.
        Note that this is *not* a test of Delaunay edges, just constraints.
        """
        if self._edge_map is None:
            self.build_edge_data()

        if a>b:
            a,b=b,a
        k = (a,b)
        if k in self._edge_map:
            return self._edge_map[k]
        else:
            return None
    def node_neighbors(self,j):
        if self._neighbor_map is None:
            self.build_edge_data()
        return self._neighbor_map[j]
    
    def build_edge_data(self):
        # Build both the edge map and the neighbor map
        self._edge_map = {}
        self._neighbor_map = collections.defaultdict(list)

        for j in range(len(self.edges)):
            v1,v2 = self.edges[j]

            if v1 > v2:
                v1,v2 = v2,v1
            self._edge_map[ (v1,v2) ] = j
            self._neighbor_map[v1].append(v2)
            self._neighbor_map[v2].append(v1)

    comps = None
    def calc_connected_components(self):
        comps = dict( [[i,set([i])] for i in range(len(self.F))] )

        # maybe not that fastest - can't remember if this is N^2 or NlogN
        for a,b in self.edges:
            if comps[a] == comps[b]:
                continue # must not be acyclic
            
            if len(comps[a]) > len(comps[b]): # Small one is a
                a,b = b,a
                
            comps[b].update(comps[a])
            for i in comps[a]:
                comps[i] = comps[b]

        self.comps=comps
        
    def unique_connected_components(self):
        if self.comps is None:
            self.calc_connected_components()
            
        visited = {}
        unique_comps = []
        for c in self.comps.values():
            if id(c) in visited:
                continue
            else:
                visited[ id(c) ] = 1
                unique_comps.append(c)
        return unique_comps

    def fill_by_graph_search(self):
        """ Fill in nan values by interpolation along the edges
        """
        # accumulates F/dist
        accum = self.F.copy()

        # accumulates 1/dist
        weights = np.array( np.isfinite(accum), np.float64 )
        accum[ isnan(accum) ] = 0.0

        # For each good point, DFS to update nearby nan values

    def plot(self,**kwargs):
        if len(self.edges)>0:
            edges = self.X[self.edges]
            coll = LineCollection( edges )
            plt.gca().add_collection(coll)
        field.XYZField.plot(self,**kwargs)
        
    def linesearch(self,a,b):
        """
        returns a nodes,nexts,
        where nodes is a list of nodes, starting with a and going in the
        direction of b, and nexts is other lines leaving the end of this
        one in the case of a junction

        in the case of a ring, it will return one segment where the first and
        last nodes are identical
        """

        nodes = [a,b]

        while 1:
            if nodes[-1] == nodes[0]:
                print("linesearch got a ring")
                nexts = []
                break

            nbrs = [n for n in self.node_neighbors(nodes[-1])
                    if n != nodes[-2]]
            if len(nbrs) == 1:
                nodes.append(nbrs[0])
                continue

            if len(nbrs) > 1:
                #print "Hit a junction"
                nexts = [ (nodes[-1],nbr) for nbr in nbrs ]
            else:
                #print "End of line"
                nexts = []
            break
        return nodes,nexts


    def all_segments(self,c):
        """
        given a connected component c, i.e. list of nodes that are all connected,
        return a list of segments [ [n0,n1,n2,..],...] *and*
        how those lists are in turn connected 

        junctions are noted by a list of tuples [ (seg_index,end_mark), ... ]
         denoting that the each of the segments share a point, and end_mark= 0
         means it's the first node of that segment, or -1 means it's the last
         node of that segment.
        """

        # Choose a random node:
        n = c[0]

        n_nbrs = self.node_neighbors(n)

        all_segs = []
        to_trace = []

        # for junctions, endpoints.
        #   - node indices go in here once all traces leaving that node have been
        #     either handled or queued in to_trace.
        # before adding something to to_trace, then, should make sure that it's
        #     starting node isn't already in visited_ends
        visited_ends = {}

        # First one is special, because we may have started in the middle of
        # it.
        if len(n_nbrs) == 1:
            # Chose an endpoint
            visited_ends[n] = 1
            seg,nexts = self.linesearch(n,n_nbrs[0])
            to_trace += nexts
        elif len(n_nbrs) == 0:
            # there is only one node in the whole thing
            seg = [n]
        elif len(n_nbrs) == 2:
            # hit the middle of it:
            segA,nextsA = self.linesearch(n,n_nbrs[0])
            segB,nextsB = self.linesearch(n,n_nbrs[-1])
            seg = np.concatenate( (segA[::-1],segB) )
            to_trace += nextsA
            to_trace += nextsB
            visited_ends[seg[0]] = 1
            visited_ends[seg[-1]] = 1
        else:
            # we're starting at a junction
            # trace one, and queue the rest
            seg,nexts = self.linesearch(n,n_nbrs[0])
            to_trace += nexts + [ (n,nbr) for nbr in n_nbrs[1:]]
            visited_ends[n] = 1

        all_segs.append(seg)

        while len(to_trace)>0:
            a,b = to_trace.pop(0)
            seg,nexts = self.linesearch(a,b)
            all_segs.append(seg)

            if seg[-1] in visited_ends:
                print("won't queue more traces starting at end of that segment")
            else:
                to_trace += nexts
                visited_ends[seg[-1]] = 1

        return all_segs

    def fill_by_graph_search(self):
        """ Interpolate along edges to fill in NaN values where possible.
        Will not extrapolate, though - to update a value it must have valid
        neighbors in more than one direction.
        """
        accum = self.F.copy()
        weights = np.array(np.isfinite(accum), np.float64)
        counts = np.zeros(len(self.F), np.int32)
        counts[ np.isfinite(accum) ] = 1000
        accum[ np.isnan(accum) ] = 0.0

        for i in np.nonzero(np.isfinite(self.F))[0]:
            # Shortest path search starting from i
            # print i

            visited = {}
            heap = priority_queue.priorityDictionary()

            heap[i] = 0.0

            while len(heap)>0:
                j = heap.smallest()
                j_dist = heap[j]
                del heap[j]
                visited[j]=1

                #print "Visiting %d - dist is %g"%(j,j_dist)

                if j_dist > 0:
                    # update costs here
                    accum[j]   += self.F[i] / j_dist
                    weights[j] += 1.0 / j_dist
                    counts[j]  += 1

                # Queue neighbors:
                for nbr in self.node_neighbors(j):
                    if (nbr not in visited) and np.isnan(self.F[nbr]):
                        nbr_dist = j_dist + np.norm(self.X[nbr]-self.X[j])
                        if (nbr not in heap) or (heap[nbr]>nbr_dist):
                            heap[nbr] = nbr_dist

        missing = np.isnan(self.F) & (counts>1)

        self.F[missing] = accum[missing] / weights[missing]
    
    def resample(self,spacing):
        """ Returns a field without edges, but with each of our edges
        resample at the given spacing.

        Any edges that have one or both endpoints with NaN are skipped
        """
        valid = np.isfinite(self.F)
        
        X=[self.X[valid]]
        F=[self.F[valid]]

        for a,b in self.edges:
            if np.isnan(self.F[a]) or np.isnan(self.F[b]):
                continue
            
            length = np.norm( self.X[b] - self.X[a] )
            steps = int(np.ceil(length/spacing))

            alpha = np.arange(1,steps) / float(steps)

            X.append( (1-alpha[:,None])*self.X[a] + alpha[:,None]*self.X[b] )
            F.append( (1-alpha)*self.F[a] + alpha*self.F[b] )

        X = np.concatenate( X )
        F = np.concatenate( F )

        return field.XYZField(X=X,F=F)

