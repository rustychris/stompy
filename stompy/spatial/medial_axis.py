from __future__ import print_function

import os
import subprocess,os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

from shapely import geometry

from . import field

from ..utils import point_in_polygon

def plot_geo(geo):
    def plot_ring(r):
        points = np.array(r.coords)
        plt.plot( points[:,0],points[:,1],'k' )
    plot_ring(geo.exterior)
    for r in geo.interiors:
        plot_ring(r)

def geo2poly(geo_poly,poly_filename):
    """
    given a polygon geometry, write a triangle compatible poly
    file
    """
    print("Writing poly file ", poly_filename)
    
    # and then only the exterior ring:
    point_list = np.array(geo_poly.exterior.coords)
    # which at least sometimes has a duplicate node at the end that
    # we don't want
    if np.all(point_list[0]==point_list[-1]):
        point_list = point_list[:-1]

    npoints = point_list.shape[0]

    # triangle wants the basic planar-straight-line-graph
    poly_fp = open(poly_filename,'wt')

    # first line is
    # First line: <# of vertices> <dimension (must be 2)>  <# of attributes> <# of boundary markers (0 or 1)>
    poly_fp.write("%i 2 0 0\n"%(npoints))
    # Write out vertices
    for i in range(npoints):
        # <vertex #> <x> <y>  [attributes] [boundary marker] 
        poly_fp.write("%i %f %f\n"%(i,point_list[i,0],point_list[i,1]))
    # Write out segments
    # <# of segments> <# of boundary markers (0 or 1)>
    poly_fp.write("%i 0\n"%(npoints))
    for i in range(npoints):
        # <segment #> <endpoint> <endpoint>  [boundary marker]
        poly_fp.write("%i %i %i\n"%(i,i,(i+1)%npoints))
    # number of holes, which for the moment we ignore:
    poly_fp.write("0\n")
    poly_fp.close()

def load_triangle_nodes(node_filename):
    """
    load nodes as output by triangle
    """
    fp = open(node_filename,'rt')
    n_nodes, dim, nattrs, has_boundary_markers = map(int,fp.readline().split())
    
    nodes = np.zeros( (n_nodes,dim), np.float64)

    for i in range(n_nodes):
        idx,nodes[i,0],nodes[i,1] = map(float,fp.readline().split()[:3])
    fp.close()
    return nodes

def load_triangle_edges(edge_filename):
    """ load finite edges from output from triangle
    """
    fp = open(edge_filename,'rt')
    n_edges,n_markers = map(int,fp.readline().split())
    # indexed into corresponding node file:
    edges = []
    
    for i in range(n_edges):
        vals=map(int,fp.readline().split()[:3])
        if vals[2] == -1:
            continue # it's a ray
        edges.append( vals[1:3] )
    fp.close()

    return np.array(edges)

def plot_voronoi(poly_filename):
    vor_node = poly_file.replace('.poly','.1.v.node')
    vor_edge = poly_file.replace('.poly','.1.v.edge')

    # load the vor nodes and show them:
    vor_nodes = load_triangle_nodes(vor_node)
    plt.plot(vor_nodes[:,0],vor_nodes[:,1],'r+')

    vor_edges = load_triangle_edges(vor_edge)

    # plot the finite edges:
    # build up the list of lines:
    all_lines = vor_nodes[vor_edges]
    coll = LineCollection(all_lines)
    ax = plt.gca()
    ax.add_collection(coll)


def load_triangle_elements(ele_file):
    fp = open(ele_file,'rt')
    n_elts, nodes_per_elt, n_attrs = map(int,fp.readline().split())

    tris = np.zeros( (n_elts,3), np.int32)

    for i in range(n_elts):
        dummy, tris[i,0],tris[i,1],tris[i,2] = map(int,fp.readline().split()[:4])
    return tris

def plot_elements(tris,nodes):
    edges = set()
    for t in range(tris.shape[0]):
        t_verts = np.sorted(tris[t])
        edges.add( (t_verts[0],t_verts[1]) )
        edges.add( (t_verts[0],t_verts[2]) )
        edges.add( (t_verts[1],t_verts[2]) )

    edges = np.array(list(edges))

    all_lines = nodes[edges]
    coll = LineCollection(all_lines)
    ax = plt.gca()
    ax.add_collection(coll)


# that writes out these files:
# node_file = poly_file.replace('.poly','.1.node')
# element_file = poly_file.replace('.poly','.1.ele')


# tris = load_triangle_elements(element_file)
# nodes = load_triangle_nodes(node_file)
# plot_elements(tris,nodes)
    
# Look into how to compute the local radius based on the voronoi
# diagram:

# Find the radius at each voronoi center:
#  1. load the voronoi nodes:


# some sort of issue loading the tri information - might be worth
# trying it w/o any islands, but first taking a look...
#  nodes: 2-D, 0 attributes, 1 boundary marker
#  8690 nodes (compare to 8003 nodes in input)


class Graph(object):
    def __init__(self,basename):
        node_file = basename + '.node'
        if os.path.exists(node_file):
            self.nodes = load_triangle_nodes(node_file)
        else:
            self.nodes = None

        edge_file = basename + '.edge'
        if os.path.exists(edge_file):
            self.edges = load_triangle_edges(edge_file)
        else:
            self.edges = None

        element_file = basename + '.ele'
        if os.path.exists(element_file):
            self.elements = load_triangle_elements(element_file)
        else:
            self.elements = None
            
    def plot(self,colors=None):
        if self.edges is not None:
            self.plot_edges(colors=colors)
        else:
            self.plot_elements(colors=colors)
            
    def plot_edges(self,colors=None):
        all_lines = self.nodes[self.edges]
        coll = LineCollection(all_lines)
        if colors is not None:
            coll.set_array(colors)
        ax = plt.gca()
        ax.add_collection(coll)
        plt.draw()
        
    def plot_elements(self,colors=None):
        i = np.array([0,1,2,0])
        
        all_lines = self.nodes[self.elements[:,i]]
        coll = LineCollection(all_lines)
        if colors is not None:
            coll.set_array(colors)
        ax = plt.gca()
        ax.add_collection(coll)
        plt.draw()

    _vcenters = None
    def vcenters(self):
        if self.elements is None:
            raise Exception("vcenters() called but elements is None")
        
        if self._vcenters is None:
            # just copied from trigrid
            self._vcenters = np.zeros(( len(self.elements),2 ), np.float64)

            p1x = self.nodes[self.elements[:,0]][:,0]
            p1y = self.nodes[self.elements[:,0]][:,1]
            p2x = self.nodes[self.elements[:,1]][:,0]
            p2y = self.nodes[self.elements[:,1]][:,1]
            p3x = self.nodes[self.elements[:,2]][:,0]
            p3y = self.nodes[self.elements[:,2]][:,1]

            # taken from TRANSFORMER_gang.f90
            dd=2.0*((p1x-p2x)*(p1y-p3y) -(p1x-p3x)*(p1y-p2y))
            b1=p1x**2+p1y**2-p2x**2-p2y**2
            b2=p1x**2+p1y**2-p3x**2-p3y**2 
            xc=(b1*(p1y-p3y)-b2*(p1y-p2y))/dd
            yc=(b2*(p1x-p2x)-b1*(p1x-p3x))/dd

            self._vcenters[:,0] = xc
            self._vcenters[:,1] = yc
        return self._vcenters
    _radii = None
    def radii(self):
        if self._radii is None:
            vcenters = self.vcenters()
            vcorners = self.nodes[self.elements[:,0]]
            self._radii = np.sqrt( ((vcenters - vcorners)**2).sum(axis=1) )

        return self._radii

    _nodes2elements = None
    def nodes2elements(self,n1,n2):
        if self._nodes2elements is None:
            e2e = {}
            print("building hash of edges to elements")
            for c in range(len(self.elements)):
                for i in range(3):
                    a = self.elements[c,i]
                    b = self.elements[c,(i+1)%3]
                    if a > b:
                        a,b = b,a

                    k = (a,b)

                    if not e2e.has_key(k):
                        e2e[k] = []

                    e2e[ k ].append(c)
                    
            self._nodes2elements = e2e
            
            print("done")

        if n1 > n2:
            n1,n2 = n2,n1
        return self._nodes2elements[(n1,n2)]
        

class Boundary(object):
    n_cleaned = 0 # bean-counter for remove_repeated
    
    def __init__(self,geo=None,nodes=None,clean_geo=True):
        """ 
        geo: a Shapely polygon (with holes, ok)
        nodes: an array of points, taken to be the exterior ring of a polygon
        clean_geo: if true, traverse the rings and removed repeated nodes
        """
        
        if geo:
            all_nodes = []
            all_edges = []
            holes = []
            start_n = 0

            rings = [geo.exterior] + list(geo.interiors)
            for ring in rings:
                orig_nodes = np.array(ring.coords)
                if clean_geo:
                    orig_nodes = self.remove_repeated(orig_nodes)
                    
                # remove repeated last coordinate
                these_nodes = orig_nodes[:-1] 
                
                n_nodes = these_nodes.shape[0]
                n = np.arange(n_nodes)
                these_edges = start_n + np.transpose( np.array([n,(n+1)%n_nodes]) )
                
                all_nodes.append(these_nodes)
                all_edges.append(these_edges)
                start_n += n_nodes

                ring_poly = geometry.Polygon( these_nodes )
                point_inside = point_in_polygon(ring_poly)
                holes.append(point_inside)
                
            self.nodes = np.concatenate( all_nodes ) # array(geo.exterior.coords)[:-1,:]
            self.edges = np.concatenate( all_edges )
            self.holes = np.array(holes[1:])
            self.geo = geo

            if clean_geo:
                print("Removed %i repeated nodes"%self.n_cleaned)
        else:
            self.nodes = nodes
            
            n_nodes = self.nodes.shape[0]
            # construct an edge array that just matches consecutive
            # nodes
            n = np.arange(n_nodes)
            self.edges = np.transpose(np.array([n,(n+1)%n_nodes]))
            self.holes = np.zeros((0,2))

        # automatically find a basic lower-bound length scale
        min_dist_sqr = (((self.nodes[1:] - self.nodes[:-1])**2).sum(axis=1)).min()
        self.min_edge_length = np.sqrt(min_dist_sqr)
        #print("Minimum edge length in boundary inputs is ",self.min_edge_length)

        self._vor = None
        self._tri = None

    _nodes2edge = None
    def nodes2edge(self,a,b):
        # if a,b is boundary edge, return the edge id, otherwise return None
        if self._nodes2edge is None:
            self._nodes2edge = {}

            for e in range(len(self.edges)):
                c,d = self.edges[e]
                if c > d:
                    d,c = c,d

                self._nodes2edge[ (c,d) ] = e
        if a>b:
            b,a = a,b
        k = (a,b)
        if self._nodes2edge.has_key(k):
            return self._nodes2edge[k]
        else:
            return None
        
    def remove_repeated(self,ring):
        """Remove repeated nodes from an array.
        """
        mask = np.zeros( len(ring),np.bool_ )

        mask[:-1] = np.all(ring[:-1]==ring[1:],axis=1)
        
        # for i in range(len(ring)-1):
        #     if all(ring[i+1]==ring[i]):
        #         mask[i] = True
        self.n_cleaned += mask.sum()
        
        return ring[~mask,:]
        
    def vor(self):
        if self._vor is None:
            self.triangulate()
        return self._vor
    def triangulation(self):
        if self._tri is None:
            self.triangulate()
        return self._tri

    def plot(self,colors=None):
        all_lines = self.nodes[self.edges]
        coll = LineCollection(all_lines)
        if colors is not None:
            coll.set_array(colors)
        ax = plt.gca()
        ax.add_collection(coll)

        # if len(self.holes) > 0:
        #     plot(self.holes[:,0],self.holes[:,1],'ro')
        
        plt.draw()
    def plot_lines(self):
        plt.plot(self.nodes[:,0], self.nodes[:,1], 'k')

    def split_edges(self,edge_indexes):
        new_nodes = np.nan * np.ones((len(edge_indexes),2), np.float64)
        new_edges = -1 * np.ones((len(edge_indexes),2), np.int32)

        # remember what the next free edge and node are
        next_edge = self.edges.shape[0]
        next_node = self.nodes.shape[0]

        # extend nodes and edges:
        self.nodes = np.concatenate( (self.nodes,new_nodes), axis=0 )
        self.edges = np.concatenate( (self.edges,new_edges), axis=0 )

        ordering = np.arange(self.nodes.shape[0],dtype=np.float64)
        ordering[next_node:] = -1

        for i in range(len(edge_indexes)):
            # node indices to the old endpoints
            pntA,pntC = self.edges[edge_indexes[i]]
            pntB = next_node+i
            
            self.nodes[pntB] = 0.5*(self.nodes[pntA] + self.nodes[pntC])

            self.edges[edge_indexes[i],1] = pntB
            self.edges[next_edge+i] = [pntB,pntC]
            ordering[pntB] = 0.5*(ordering[pntA]+ordering[pntC])
            
        new_order = np.argsort(ordering)
        # so j = new_order[i] means that old node j will get mapped
        # to new node i
        self.nodes = self.nodes[new_order]

        # the "inverse" of new_order
        mapping = np.argsort(new_order)

        # not sure about this.  too late to prove it to myself that
        # it works short of just testing it
        self.edges = mapping[self.edges]
        self._nodes2edge = None

    def write_poly(self,poly_filename):
        """ write a triangle compatible poly file
        """
        # and then only the exterior ring:
        point_list = self.nodes
        
        # probably unnecessary
        if np.all(point_list[0]==point_list[-1]):
            raise Exception("Boundary should have already stripped any repeated endpoints")

        npoints = point_list.shape[0]

        # triangle wants the basic planar-straight-line-graph
        poly_fp = open(poly_filename,'wt')

        # first line is
        # First line: <# of vertices> <dimension (must be 2)>  <# of attributes> <# of boundary markers (0 or 1)>
        poly_fp.write("%i 2 0 0\n"%(npoints))
        # Write out vertices
        for i in range(npoints):
            # <vertex #> <x> <y>  [attributes] [boundary marker] 
            poly_fp.write("%i %f %f\n"%(i,point_list[i,0],point_list[i,1]))
        # Write out segments
        # <# of segments> <# of boundary markers (0 or 1)>
        poly_fp.write("%i 0\n"%(npoints))
        for i in range(len(self.edges)):
            # <segment #> <endpoint> <endpoint>  [boundary marker]
            poly_fp.write("%i %i %i\n"%(i,self.edges[i,0],self.edges[i,1]))
        # number of holes
        poly_fp.write( "%d\n"%self.holes.shape[0] )
        for i in range(self.holes.shape[0]):
            poly_fp.write("%d %f %f\n"%(i, self.holes[i,0], self.holes[i,1]) )
        poly_fp.close()
        
    # def triangulate(self):
    #     ### Run some triangle stuff:
    #     poly_file = "test2.poly"
    #     self.write_poly(poly_file)
    #     
    #     cmd = "%s -e -D -p -v %s"%(triangle_path,poly_file)
    #     subprocess.call(cmd,shell=True) # ,stdout=file('/dev/null','w') )
    # 
    #     # probably we should get the real geometry that was used, otherwise
    #     # things will get confusing
    #     self.read_poly('test2.1.poly')
    #     
    #     self._tri = Graph('test2.1')
    #     self._vor = VoronoiDiagram('test2.1.v')

    def read_poly(self,poly_file):
        """ After triangulating, there may have been Steiner points
        added, and they will exist in the output .poly file.
        This reads that file and replaces self.nodes and self.edges
        with the information in the given polyfile.  Holes will be
        kept the same (although it would be valid to re-read holes, too.

        """
        poly_fp = open(poly_file,'rt')
        new_edges = []
        new_nodes = []

        n_nodes,dim,n_attrs,n_markers = map(int,poly_fp.readline().split())
        if n_nodes == 0:
            # print("Reading nodes from separate file")
            new_nodes = load_triangle_nodes(poly_file.replace('.poly','.node'))
        else:
            raise Exception("Not ready for reading inline nodes")

        n_segments,n_markers = map(int,poly_fp.readline().split())
        new_edges = np.zeros((n_segments,dim), np.int32)

        for i in range(n_segments):
            vals = map(int,poly_fp.readline().split())
            new_edges[i] = vals[1:3]

        # install the new data:
        self.edges = new_edges
        self.nodes = new_nodes
        self.geo = None
        self.src = poly_file
        
    def subdivide(self):
        """ Find edges that need to be sampled with smaller
        steps and divide them into two edges.
        returns the number of new edges / nodes

        method: calculate voronoi radii
           iterate over edges in boundary
           for each edge, find the voronoi point that they have
             in common.  So this edge should be part of a triangle,
             and we are getting the center of that triangle.

           the voronoi radius with the distance between the voronoi
             point and the edge.  If the edge is too long and needs to
             be subdivided, it will be long (and the voronoi radius large)
             compared to the distance between the edge and the vor. center.

             Can this be done without the vor. radii?
               need 
        """

        # the old way calculated voronoi radii and searched for nodes
        # on those circumcircles.  For subdividing, we just need to match
        # each edge with the one voronoi point it belongs to.
        
        # vor = self.vor()
        # vor.calc_radii(self.nodes)

        # the new way - calculated voronoi points directly from the triangles
        # in the delaunay triangulation, then match with edges with a hash
        # on edge [a,b] node pairs
        triangulation = self.triangulation()
        vcenters = triangulation.vcenters()

        n_edges = self.edges.shape[0]
        to_subdivide = np.zeros(n_edges, np.float64)


        # the only way this works is for the boundary nodes to be exactly
        # the same, so we go boundary edge -> nodes -> delaunay element
        if np.any( self.nodes != triangulation.nodes ):
            raise Exception("Triangulation and boundary use different nodes.")
        
        print("Choosing edges to subdivide")
        for i in range(n_edges): # over boundary edges
            a,b = self.edges[i]
            elements = triangulation.nodes2elements(a,b)

            if len(elements) != 1:
                print("Edge %d,%d mapped to elements %s"%(a,b,elements))
                raise Exception("Boundary edges should map to exactly one element")
            
            element = elements[0]

            # compute the point-line distance between
            # this edge and the v center, then compare to
            # the distance from the endpoint to that
            # vcenter
            pntV = vcenters[element]
            pntA = self.nodes[a]
            pntB = self.nodes[b]

            v_radius = np.sqrt( ((pntA-pntV)**2).sum() )
            line_clearance = np.sqrt( (( 0.5*(pntA+pntB) - pntV)**2).sum() )

            if v_radius > 1.2*line_clearance and v_radius > self.min_edge_length:
                # second check - make sure that neither AC nor BC are also on the
                # boundary
                p1,p2,p3 = triangulation.elements[element]
                count = 0
                if self.nodes2edge(p1,p2) is not None:
                    count += 1
                if self.nodes2edge(p2,p3) is not None:
                    count += 1
                if self.nodes2edge(p3,p1) is not None:
                    count += 1

                if count == 1:
                    to_subdivide[i] = 3
                elif count == 0:
                    global bad_boundary
                    bad_boundary = self
                    print("While looking at edge %d=(%d,%d)"%(i,a,b))
                    raise Exception("We should have found at least 1 boundary edge")
                elif count == 3:
                    print("WARNING: Unexpected count of boundary edges in one element: ",count)
                # if 2, then it's a corner and we probably don't want to subdivide

        self.to_subdivide = to_subdivide
        bad_edges = where(to_subdivide)[0]
        self.split_edges( bad_edges )

        # invalidate these:
        self._vor = None
        self._tri = None
        return len(bad_edges)

    def subdivide_iterate(self):
        while 1:
            n_new = self.subdivide()
            print("Subdivide made %d new nodes"%n_new)
            if n_new == 0:
                break

class VoronoiDiagram(Graph):
    radii = None
    dual_nodes = None
    dual_lookup = {}
    
    def calc_radii(self,del_nodes):
        """  for each of the voronoi points, find it's radius and
        which delaunay points are responsible for it.
        """
        n_nodes = self.nodes.shape[0]
        
        self.radii = np.zeros( n_nodes, np.float64)
        self.dual_nodes = [None]*n_nodes
        self.dual_lookup = {} # map dual node index to list of vcenters

        # this is where all the time goes!
        # so make a field for the delaunay nodes that will speed up finding them
        I = np.arange(len(del_nodes))
        
        del_field = field.XYZField(del_nodes, 'nope')
        del_field.build_index()
        
        for i in range(n_nodes):
            if i % 1000 == 0:
                print(i)

            # find the nearest one...
            nearest = del_field.nearest(self.nodes[i])
            min_radius = np.sqrt( ((del_nodes[nearest] - self.nodes[i])**2).sum() )
            all_near = del_field.within_r(self.nodes[i], 1.00000001*min_radius)
            
            # dists_sqr = ((del_nodes - self.nodes[i,:])**2).sum(axis=1)
            # rad_sqr = dists_sqr.min()
            # self.dual_nodes[i] = find( dists_sqr <= 1.00001*rad_sqr )
            self.dual_nodes[i] = np.array(all_near)

            for dual_node_idx in self.dual_nodes[i]:
                if not self.dual_lookup.has_key(dual_node_idx):
                    self.dual_lookup[dual_node_idx] = []
                self.dual_lookup[dual_node_idx].append(i)
                
            self.radii[i] = min_radius # sqrt(rad_sqr)


    def merge_points(self,tol):
        """ After a call to calc_radii(), this can be called to coalesce voronio points
        that are close to each other
        """

        while len(self.nodes) > 1:
            # look for short edges:
            edge_ends = self.nodes[ self.edges ]

            edge_centers = edge_ends.mean(axis=1)
            edge_tols = tol(edge_centers)

            edge_lengths = np.sqrt( ((edge_ends[:,1,:] - edge_ends[:,0,:])**2).sum(axis=1) )
            rel_edge_lengths = edge_lengths / edge_tols

            to_merge = np.argmin(rel_edge_lengths)

            if rel_edge_lengths[ to_merge ] < 1.0:
                # print(" got an edge to merge.")
                self.merge_edge( to_merge )
            else:
                break
            
    def merge_edge(self,e):
        a,b = self.edges[e]
        # print("merging voronoi edge ",a,b)

        self.edges = np.concatenate( (self.edges[:e], self.edges[e+1:]) )

        # map old node indices to new ones:
        node_mapping = np.arange(len(self.nodes))
        # b has become a
        node_mapping[b] = a
        # and everybody greater than b is shifted down
        node_mapping[ node_mapping > b] -= 1

        if self.radii is not None:
            self.radii = np.concatenate( (self.radii[:b], self.radii[b+1:]) )

            # combine their dual nodes:
            self.dual_nodes[a] = np.unique( np.concatenate( (self.dual_nodes[a],self.dual_nodes[b]) ) )

            # then remove b from the list
            self.dual_nodes = self.dual_nodes[:b] + self.dual_nodes[b+1:]
            
            for k in self.dual_lookup.keys():
                l = self.dual_lookup[k]
                # k is an index to the boundary points
                # l is a list of indices to voronoi centers
                
                if b in l:
                    l.remove( b )
                    if not a in l:
                        l.append(a)

                # keep it as a list for now.
                self.dual_lookup[k] = node_mapping[ np.array(l) ].tolist()

        # new node is between the old two nodes:
        self.nodes[a] = 0.5*(self.nodes[a] + self.nodes[b])

        self.edges = node_mapping[ self.edges ]

        self.nodes = np.concatenate( (self.nodes[:b], self.nodes[b+1:] ) )

    def centers_for_dual_node(self,dual_node):
        if self.dual_lookup.has_key(dual_node):
            return self.dual_lookup[dual_node]
        else:
            return []
            
    def plot_radii(self):
        a = gca()

        for i in range(self.nodes.shape[0]):
            cir = Circle( self.nodes[i], radius=self.radii[i])
            a.add_patch(cir)

    def plot_vor_points(self):
        try:
            colors = self.radii
            print("Got colors from radii")
            plt.scatter(self.nodes[:,0],self.nodes[:,1],50,colors,
                        lw=0,vmin=200,vmax=250)
        except:
            plt.plot(self.nodes[:,0],self.nodes[:,1],'r+')
        
    def plot(self,show_vor_points=True):
        if show_vor_points:
            self.plot_vor_points()

        # plot the finite edges:
        # build up the list of lines:
        all_lines = self.nodes[self.edges]
        coll = LineCollection(all_lines)
        coll.set_color('m')
        ax = plt.gca()
        ax.add_collection(coll)

        plt.draw()


# since the triangulation didn't add any nodes, just
# use the boundaries nodes instead of tri.nodes


# ### Check radius against edge / voronoi center
# if __name__ == '__main__':
#     ### Load the data
#     # boundary = load_shp.Boundary('/home/rusty/classes/research/meshing/dumbarton.shp')
# 
#     # this is full bay, already filtered at 50m
#     boundary = load_shp.Boundary('/home/rusty/classes/research/spatialdata/us/ca/suntans/shoreline/noaa-medres/sfbay-100km-arc/sfbay-100km-arc-50_20.shp')
#     
#     geo = boundary.geo
# 
#     # points = array( geo.exterior.coords )
#     # points = points[:-1]
#     
#     # from paver import upsample_linearring
#     # points = upsample_linearring(points,50)
#     
#     bdry_ma = Boundary( geo=geo )
#     print("subdividing...")
#     bdry_ma.subdivide_iterate()
#     print("done")
# 
#     vor = bdry_ma.vor()
#     #tri = bdry_ma.tri()
#     #tri.plot()
# 
#     print("Calculating radii")
#     vor.calc_radii(bdry_ma.nodes)
#     print("done")
# 
#     bdry_ma.plot()
#     bdry_ma.vor().plot_vor_points()
#     plt.axis('equal')
#     plt.draw()

