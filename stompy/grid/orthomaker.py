"""
Grid generation and modification related methods.  paver.py is the main
user of these methods, and they may be folded into paver.py in the future.
"""
from __future__ import print_function

import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    from pylab import *
    from matplotlib import cm
    from matplotlib.collections import *
except ImportError:
    print("Missing matplotlib library: plotting will not be available")

try:
    from numpy import *
    from numpy.linalg import norm
except ImportError:
    print("Missing numpy library.  Please install it")
    raise

try:
    from shapely import geometry
except:
    print("Missing shapely library.  Please install it")

from . import trigrid
from .trigrid import rot
from ..utils import point_in_polygon

### Some geometric utility functions ###


def intersect_geoms(glist):
    g = glist[0]
    for gb in glist[1:]:
        g = g.intersection(gb)
        if g.area == 0.0:
            return g
    return g


def free_node_bounds(points,max_angle=85*pi/180. ):
    """ assumed that the first point can be moved and the
    other two are fixed

    returns an array of vertices that bound the legal region
    for the free node

    currently this uses four vertices to approximate the shape
    of the region
    """
    
    # try to construct a tight-ish bound on the legal locations for
    # one node of a triangle where the other two nodes are constrained
    if len(points) == 3:
        orig_pntC = points[0]

        pntA = points[1]  # should be oriented such that with A,B,C is CCW
        pntB = points[2]

        # make sure the orientation is correct:
        if dot( rot(pi/2,(pntB-pntA)),orig_pntC - pntA) < 0:
            pntA,pntB = pntB,pntA
    else:
        # we got only the fixed points, which are assumed to be CCW, with
        # free point to the left of AB
        pntA,pntB = points
        
    # point at the tip of the triangle:
    pntC1 = 0.5*(pntA+pntB) + rot(pi/2,pntB-pntA)*tan(max_angle)/2

    # closest legal point to pntA:
    pntC2 = pntB + rot(-(pi-2*max_angle), pntA-pntB)
    # closest legal point to pntB:
    pntC3 = pntA + rot(pi-2*max_angle, pntB-pntA)

    # and find the closest point in the middle
    min_isosc_angle = (pi - max_angle)/2
    pntC4 = 0.5*(pntA+pntB) + rot(pi/2,(pntB-pntA))*tan(min_isosc_angle)/2

    if 0:
        C_edges = array([pntC2,pntC1,pntC3,pntC4,pntC2])
        plot( C_edges[:,0],C_edges[:,1],'c')

    return array( [pntC2,pntC1,pntC3,pntC4] )


def free_node_bounds_conservative(points,max_angle=85*pi/180.):
    """ Given the two points, with the intended third point lying
    to the left, return points describing a polygon the ensures a
    relatively nice triangle.

    here nice means that we take the closest point that the new
    vertex can be, and force the new point to be at least that
    far away, so avoiding narrow isosceles triangles.
    """
    pntA,pntB = points
        
    # point at the tip of the triangle:
    pntC1 = 0.5*(pntA+pntB) + rot(pi/2,pntB-pntA)*tan(max_angle)/2

    l_ab = norm(pntB - pntA)
    
    l_leg = l_ab*0.5 / (tan(max_angle/2.) * sin(max_angle) )

    pntC2 = pntA + l_leg * (pntC1-pntA) / norm(pntC1-pntA)
    pntC3 = pntB + l_leg * (pntC1-pntB) / norm(pntC1-pntB)
    
    arc_points = [pntC1,pntC2,pntC3]
    
    return array( arc_points )
    

def free_node_bounds_fine(points,max_angle=85*pi/180.,region_steps=3 ):
    """ assumed that the first point can be moved and the
    other two are fixed

    returns an array of vertices that bound the legal region
    for the free node

    this version discretizes the curved boundary with variable number of
    nodes
    """
    
    # try to construct a tight-ish bound on the legal locations for
    # one node of a triangle where the other two nodes are constrained

    if len(points) == 3:
        orig_pntC = points[0]

        pntA = points[1]  # should be oriented such that with A,B,C is CCW
        pntB = points[2]

        # make sure the orientation is correct:
        if dot( rot(pi/2,(pntB-pntA)),orig_pntC - pntA) < 0:
            pntA,pntB = pntB,pntA
    else:
        # we got only the fixed points, which are assumed to be CCW, with
        # free point to the left of AB
        pntA,pntB = points
        
    # point at the tip of the triangle:
    pntC1 = 0.5*(pntA+pntB) + rot(pi/2,pntB-pntA)*tan(max_angle)/2

    # # closest legal point to pntA:
    # pntC2 = pntB + rot(-(pi-2*max_angle), pntA-pntB)
    # # closest legal point to pntB:
    # pntC3 = pntA + rot(pi-2*max_angle, pntB-pntA)

    min_angle_A = pi - 2*max_angle
    max_angle_A = max_angle

    arc_points = [pntC1]
    
    for angle_A in linspace(min_angle_A,max_angle_A,region_steps):
        # too lazy to write robust predicate here - just dish it out
        # to shapely
        angle_B = pi-max_angle-angle_A
        # 10 is overkill - I think this can easily be bounded by
        # the dimensions of the equilateral
        Aray = pntA + rot(angle_A,10*(pntB-pntA))
        Bray = pntB + rot(-angle_B,10*(pntA-pntB))
        Aline = geometry.LineString([pntA,Aray])
        Bline = geometry.LineString([pntB,Bray])
        
        crossing = Aline.intersection(Bline)
        try:
            new_point = array(crossing.coords[0])
        except:
            print("Couldn't find intersection of",crossing)
            print(Aline)
            print(Bline)
            raise

        arc_points.append(new_point)
        
    arc_points = array(arc_points)
        
    return array( arc_points )

class OrthoMaker(trigrid.TriGrid):
    """  attempts at auto-orthogonalizing, starting with SMS output
    """
    max_angle = 85 * pi / 180

    boundary_cells_merged = 0
    quads_merged = 0
    nodes_nudged = 0

    max_depth = 8

    
    def __init__(self,*args,**kwargs):
        if 'max_angle' in kwargs:
            self.max_angle = kwargs['max_angle']
            del kwargs['max_angle']
            
        super(OrthoMaker,self).__init__(*args,**kwargs)
        self.node_stack = []


    def tri(self,cell_i):
        """ returns 3,2 array of points defining the given cell
        """
        # untested:
        return self.points[self.cells[cell_i,:]]

    def tri_angles(self,cells=None):
        """ returns an array of size Ncells,3 where each
        element is the angle (in radians) for that node in
        each triangle.
        """

        if cells is not None:
            if len(cells) == 0:
                # seems that this used to work, but
                # between new numpy and CGAL, doesn't.
                return zeros((0,3),float64)
            p1x = self.points[self.cells[cells,0]][:,0]
            p1y = self.points[self.cells[cells,0]][:,1]
            p2x = self.points[self.cells[cells,1]][:,0]
            p2y = self.points[self.cells[cells,1]][:,1]
            p3x = self.points[self.cells[cells,2]][:,0]
            p3y = self.points[self.cells[cells,2]][:,1]
        else:
            p1x = self.points[self.cells[:,0]][:,0]
            p1y = self.points[self.cells[:,0]][:,1]
            p2x = self.points[self.cells[:,1]][:,0]
            p2y = self.points[self.cells[:,1]][:,1]
            p3x = self.points[self.cells[:,2]][:,0]
            p3y = self.points[self.cells[:,2]][:,1]
            
        dx12 = p2x - p1x
        dx23 = p3x - p2x
        dx31 = p1x - p3x

        dy12 = p2y - p1y
        dy23 = p3y - p2y
        dy31 = p1y - p3y

        # find angles of each segment
        ang12 = arctan2(dy12,dx12)
        ang23 = arctan2(dy23,dx23)
        ang31 = arctan2(dy31,dx31)

        # pi - (difference in angles) to get the interior angle
        # the other pi is so that we can later subtract pi and get
        # an angle of +/- 180
        ang1 = abs( (2*pi - (ang31 - ang12)) % (2*pi) - pi )
        ang2 = abs( (2*pi - (ang12 - ang23)) % (2*pi) - pi )
        ang3 = abs( (2*pi - (ang23 - ang31)) % (2*pi) - pi )

        angles = transpose(array( [ang1,ang2,ang3] ))

        if cells is None:
            del_cells = self.cells[:,0] < 0
            angles[del_cells,:] = -1
        return angles
    
    def bad_cells(self,subset=None):
        if len(self.cells) > 0:
            angles = self.tri_angles(subset)
            if subset is not None:
                return subset[find( angles.max(axis=1) > self.max_angle )]
            else:
                return find( angles.max(axis=1) > self.max_angle )
        else:
            return array([],int32)
    
    def stats(self):
        """ prints some stats on how good/bad the mesh currently is """
        print("Number of elements: ",self.Ncells())
        print("Counting triangles with angles >= 90")

        angles = self.tri_angles() 
        n_bad_angles = (angles > (pi/2)).sum()
        n_med_angles = (angles > self.max_angle).sum()
        print("angle > 90 degrees: %d  %g%%"%(n_bad_angles, 100.0*n_bad_angles/self.Ncells()))
        print("angle > %g degrees: %d  %g%%"%(self.max_angle*180/pi,n_med_angles,
                                              100.0*n_med_angles/self.Ncells()))
        print("Boundary cells merged: ",self.boundary_cells_merged)
        print("Quads merged: ",self.quads_merged)
        print("Good nudges: ",self.nodes_nudged)
        


    def pass_one(self,verbose=False):
        print("Pass 1")
        

        # First look for untenable cell structures
        skipped_count = 1

        while skipped_count > 0:
            bad_cells = self.bad_cells()
            print("Count of bad cells: ",len(bad_cells))
            skipped_count = 0
            self.changed_cells = []
            
            for bad_cell in bad_cells:
                if bad_cell in self.changed_cells:
                    skipped_count += 1
                    continue

                if verbose:
                    print("--- bad cell = %i ---"%bad_cell)

                result = self.try_to_fix_cell_structure(bad_cell,verbose)

                if result:
                    sys.stdout.write('.')
                    sys.stdout.flush()

        print()
        print("Renumbering after structural changes")
        self.renumber()
        
    def pass_two(self):
        print("Pass 2")
        
        # And then look for ways to move vertices around
        bad_i = 0
        bad_cells = self.bad_cells()
        starting_count = len(bad_cells)
        region_steps = 3 # do a pass of coarse valid regions first

        while 1:
            print("Count of bad cells: ",len(bad_cells))
            
            for bad_cell in bad_cells:
                if self.try_to_fix_by_nudging(bad_cell,region_steps):
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    self.nodes_nudged += 1
                # this may have fixed/broken other cells, but forget
                # about recomputing bad_cells for the moment

            old_bad_count = len(bad_cells)
            bad_cells = self.bad_cells()
            print("End of loop: ",len(bad_cells))
            # only bail if we've tried with the high-resolution
            if region_steps ==100 and len(bad_cells) >= old_bad_count:
                print("Not getting any better.  Bail")
                break
            
            # for subsequent runs, use slower, high resolution regions
            region_steps = 100

    def pass_three(self,depths=None):
        if depths is None:
            depths = [1,2,3,4,5,6]
        for depth in depths:
            self.max_depth = depth+1
            bc = self.bad_cells()
            print("Start of cell-nudge with search depth %d, bad_cell count %d "%(depth,len(bc)))

            for bad_cell in bc:
                self.nudge_cell_search(bad_cell,depth)
    
    def full_go(self):
        self.stats()
        self.pass_one()
        self.stats()
        self.pass_two()
        self.stats()
        
    def try_to_fix_cell_structure(self,bad_cell,verbose=False):
        """
        attempt to fix the angle constraints for this cell.
        return true if we changed anything (such that bad cells
        should be recomputed)
        """
        # try to figure out what's going on with bad_cell:
        verts = self.points[self.cells[bad_cell],:2]
        angles = self.tri_angles( [bad_cell] )[0]

        # optionally plot what's going on
        if 0:
            verts4 = concatenate( (verts,verts[:1]) )
            plot(verts4[:,0],verts4[:,1],'r-o')
        
            for i in range(3):
                annotate( "%g"%(angles[i]*180/pi) , verts[i] )
            axis('equal')

        ###  Test 1: is it on the boundary?
        ##     note: doesn't handle a triangle that is a corner of the boundary
        bdry_points = [] # indices to points that are on the boundary:
        for edge_i in self.cell2edges(bad_cell):
            nbr = self.edges[edge_i,4]
            if nbr == -1:
                bdry_points.append( self.edges[edge_i,0] )
                bdry_points.append( self.edges[edge_i,1] )

        bdry_points = unique(bdry_points)
        if verbose:
            print("boundary points: ",bdry_points)

        if len(bdry_points) == 2:
            if verbose:
                print("2 boundary points - dispatch to fix_boundary_cell")
            return self.try_to_fix_boundary_cell(bad_cell,bdry_points,verbose)


        ### Test 2: is the node with the bad angle an internal node with
        ##    4 cells?
        bad_node = self.cells[bad_cell,find(angles > self.max_angle)[0] ]
        
        bad_nodes_cells = self.pnt2cells(bad_node)
        total_angle = self.boundary_angle(bad_node)
        
        if total_angle == 0.0 and len(bad_nodes_cells) == 4:
            return self.try_to_fix_quad(bad_node,verbose)
        
        return False # didn't fix anything

    def try_to_fix_boundary_cell(self,bad_cell,bdry_points,verbose=False):
        """ look for ways to fix a boundary cell with a bad angle.
        """

        # right now we only know how to look for a boundary point on a straight
        # edge that can then be removed
        
        for bdry_point in bdry_points:
            n_cells = len( self.pnt2cells(bdry_point) )
            # compute the boundary angle at this point

            angle = self.boundary_angle(bdry_point)

            if verbose:
                print("Angle at boundary point %i is %g"%(bdry_point,angle))
                
            # is it a problem?
            # if the interior angle is divided evenly amongst
            # the current cells, how well do we do?
            if (2*pi - angle) / n_cells > self.max_angle:
                if verbose:
                    print("Yep, there are %d cells, interior angle %g, violates max angle"%(n_cells,360 - angle*180/pi))


                # common case - the node is along a straight or nearly straight boundary
                if abs(pi - angle) < 10*pi/180. and n_cells == 2:
                    # make sure we're not creating a 4-way cross
                    # first find the interior point common to our two cells
                    my_cells = list( self.pnt2cells(bdry_point) )
                    cellA = my_cells[0]
                    cellB = my_cells[1]

                    int_point = None
                    for p in self.cells[cellA]:
                        if p != bdry_point and p in self.cells[cellB]:
                            int_point = p
                            break
                    if int_point is None:
                        raise "Why can't I find the interior point common to these cells?"


                    n_cells_at_interior = len( self.pnt2cells(int_point) )
                    angle_at_interior = 2*pi - self.boundary_angle(int_point)
                    if angle_at_interior / (n_cells_at_interior-1) > self.max_angle:
                        if verbose:
                            print("Hmm - can't remove %d because it would leave too few cells for %d"%(bdry_point,int_point))
                    else:
                        if verbose:
                            print("Great - we should merge %d and %d, and remove node %d"%(cellA,cellB,bdry_point))
                        self.merge_cells(cellA,cellB,bdry_point)
                        self.boundary_cells_merged += 1
                        return True
                else:
                    pass
                    # print "Boundary angle is bad, but it's not close enough to a straight line"
            else:
                # print "The boundary angle is not the problem"
                pass
        return False
                
                
    def merge_cells(self,cellA,cellB,dead_node):
        """ merge two cells that are on the border
        """
        
        # the structures that have to be updated:
        # cells, edges
        # and invalidate these:
        # _pnt2cells, _vcenters  [ pnts2edge no longer ] , _pnt2edges

        new_points = setdiff1d(ravel( self.cells[ [cellA,cellB] ] ), [dead_node])
        
        if 0:
            figure(1)
            cla()
            self.plot_cells([cellA,cellB])

        # if there were information to copy over about the
        # cells, this would be where, but I think all of the interesting
        # stuff is in the edges.

        cellA_edges = find( any(self.edges[:,3:]==cellA,axis=1) )
        cellB_edges = find( any(self.edges[:,3:]==cellB,axis=1) )

        common_edge = intersect1d(cellA_edges,cellB_edges)
        dead_node_edges = self.pnt2edges(dead_node)

        # rather than deleting the cells outright, maybe
        # it's smarter to record how to fix their edges
        # for each cell, one edge stays the same (but gets its
        # cell neighbor updated to the new_cell_id).
        # one edge is removed entirely, and one edge gets
        # merged with the other triangle extra edge, to create
        # a new edge.  here it should take the 

        new_cell_id = cellA #
        self.cells[new_cell_id,:] = new_points

        # edges that used to point to cellB now point to cellA
        self.edges[ self.edges[:,3]==cellB, 3 ] = new_cell_id
        self.edges[ self.edges[:,4]==cellB, 4 ] = new_cell_id

        # this should create one edge that now has new_cell_id on
        # both sides, which is the edge we want to delete outright
        dead_edge = find( all(self.edges[:,3:]==new_cell_id,axis=1) )[0]

        edges_to_merge = setdiff1d( dead_node_edges, [dead_edge] )
        self.edges[dead_edge,:] = -1


        points_on_merge_edge = setdiff1d(ravel( self.edges[edges_to_merge,:2] ),[dead_node])
        merged_edge = edges_to_merge[0]

        raise Exception("This code needs to be updated - fix _pnt2edges!")
        # good time to fix pnts2edge:
        # remove the mappings for the two short edges:
        for other_point in points_on_merge_edge:
            nodes = (dead_node,other_point)
            if nodes[0] > nodes[1]:
                nodes = (other_point,dead_node)
            del self.pnts2edge[nodes]

        # make sure it's endpoints are set right, and
        self.edges[merged_edge,:2] = points_on_merge_edge
        # if self.edges[edges_to_merge[0],2] != self.edges[edges_to_merge[1],2]:
        #     print "Marker is different on these two edges... just guessing"
        if points_on_merge_edge[0] > points_on_merge_edge[1]:
            new_node_pair = (points_on_merge_edge[1],points_on_merge_edge[0])
        else:
            new_node_pair = (points_on_merge_edge[0],points_on_merge_edge[1])
        self.pnts2edge[new_node_pair] = merged_edge    


        # mark the other edge as deleted
        self.edges[edges_to_merge[1],:] = -1

        # mark the other cell deleted:
        self.delete_cell(cellB) # self.cells[cellB,:] = -1

        # so what all has been invalidated at this point?
        # _pnt2cells, _vcenters, pnts2edge

        if self._pnt2cells:
            del self._pnt2cells[dead_node]
            for new_point in new_points:
                set_of_cells = self._pnt2cells[new_point]
                if cellB in set_of_cells:
                    set_of_cells.remove(cellB)
                set_of_cells.add(cellA)

        self._vcenters = None # lazy
        
        # record that we changed stuff:
        self.changed_cells.append(cellA)
        self.changed_cells.append(cellB)
        

        
    def plot_cells(self,c_list,nbr_count=0,label_nodes=True,label_cells=True,label_edges=False):
        """ plot cell ids, vertex ids and edges for
        the given cells.
        if nbr_count is > 0, include cells that are up to
        nbr_count cells away
        """
        c_list = array(c_list)
        
        while nbr_count > 0:
            new_c_list = c_list
            
            for cell in c_list:
                new_c_list = concatenate([new_c_list,self.cell_neighbors(cell)])
            c_list = unique(new_c_list)
            nbr_count -= 1
                
        ax = gca()
        
        points = unique(ravel( self.cells[ c_list ] ))
        plot(self.points[points,0],self.points[points,1],'ro')
        if label_nodes:
            for p in points:
                annotate(str(p),self.points[p,:2])

        # annotate centers of cells
        vc = self.vcenters()
        for c in c_list:
            if 0: # enable if there are possible edge/cell discrepancies
                nodes = self.cells[c,[0,1,2,0]]
                plot(self.points[nodes,0],
                     self.points[nodes,1],'b-')
            #ctr = mean(self.points[self.cells[c],:2],axis=0)
            # use the voronoi center:
            ctr = vc[c]
            if label_cells:
                annotate('c%i'%c,ctr)

        # plot edges:
        edge_list = array([],int32)
        for c in c_list:
            edge_list = concatenate( (edge_list,find(any(self.edges[:,3:]==c,axis=1))) )

        for e in unique(edge_list):
            if self.edges[e,4] < 0:
                color = 'r'
                lw = 2
            else:
                color = 'b'
                lw = 1
            nodes = self.points[ self.edges[e,:2],:2 ]
            plot( nodes[:,0],nodes[:,1],color,lw=lw )

            if label_edges:
                annotate("%i"%e,mean(nodes,axis=0)[:2])
            
        axis('equal')
        
    def try_to_fix_quad(self,node,verbose=False):
        """ group of four triangles around one point
        merge pairs.
        """
        # print "fix_quad"
        
        # find the four outside vertices, in order (starting point
        # doesn't matter)

        # compute interior quad angle at each

        # the pair (0,2) or (1,3) with smaller average angle
        # define the endpoints (along with node) of the
        # edges to be removed.

        # get with it...
        cells = list(self.pnt2cells(node))

        if 0:
            subplot(211)
            cla()
            self.plot_cells(cells)

        cell_points_not_node = setdiff1d(ravel(self.cells[cells,:]),[node])

        # print cell_points_not_node

        deltas = self.points[cell_points_not_node,:2] - self.points[node,:2]
        angles = arctan2(deltas[:,1],deltas[:,0])

        quad_verts = cell_points_not_node[ argsort(angles) ]
        quad_points = self.points[quad_verts,:2]


        # now we have the outer four vertices in CCW order.
        quad_angles = zeros( (4,), float64)
        # plot( quad_points[ [0,1,2,3,0],0],
        #       quad_points[ [0,1,2,3,0],1] )

        for i in range(4):
            im1 = (i-1)%4
            ip1 = (i+1)%4

            delta_prev = quad_points[i] - quad_points[im1]
            delta_next = quad_points[ip1] - quad_points[i]

            angle_prev = arctan2(delta_prev[1],delta_prev[0])
            angle_next = arctan2(delta_next[1],delta_next[0])

            quad_angles[i] = (angle_prev+pi - angle_next) % (2*pi)
            # print quad_points[i]
            # annotate( "%g"%(quad_angles[i]*180/pi), quad_points[i] )

        # now decide which way to merge:
        # switch the order of quad_verts so that 0 and 2 are the points that
        # define the merge axis
        if (quad_angles[1]+quad_angles[3]) < (quad_angles[0] + quad_angles[2]):
            quad_verts = quad_verts[ [1,2,3,0] ]


        # print "Edges to be removed: %i-%i  and %i-%i"%(node,quad_verts[0],
        #                                                node,quad_verts[2])

        # first figure out the indices for everyone:
        # the edges that get removed entirely:
        # names are oriented with the first node of the pointy end up
        dead_edge_top = self.find_edge( [node,quad_verts[0]] )
        dead_edge_bot = self.find_edge( [node,quad_verts[2]] )

        merge_edge_left = self.find_edge( [node,quad_verts[1]] )
        merge_edge_right = self.find_edge( [node,quad_verts[3]] )

        # order cells such that cell i is node,quad_verts[i],quad_verts[i+1]
        ordered_cells = -1*ones((4,),int32)
        for i in range(4):
            for j in range(4):
                # is cells[j] the ordered cell i?
                if all( sort(self.cells[ cells[j] ]) == sort([node,quad_verts[i],quad_verts[(i+1)%4]])):
                    ordered_cells[i] = cells[j]
                    break
        if any(ordered_cells) < 0:
            raise "Failed to reorder the cells CCW"

        # rename for better visuals...
        cell_nw,cell_sw,cell_se,cell_ne = ordered_cells

        # record that we're changing stuff:
        self.changed_cells.append(cell_nw)
        self.changed_cells.append(cell_ne)
        self.changed_cells.append(cell_se)
        self.changed_cells.append(cell_sw)

        # start combining cells:

        ### Combine northwest and northeast:
        north_points = setdiff1d(concatenate( (self.cells[cell_nw],self.cells[cell_ne]) ),[node])
        south_points = setdiff1d(concatenate( (self.cells[cell_sw],self.cells[cell_se]) ),[node])
        cell_n = cell_nw # new cell takes on northwest's index
        self.cells[cell_n,:] = north_points
        cell_s = cell_sw
        self.cells[cell_s,:] = south_points

        # mark the other cells as deleted
        self.cells[cell_ne,:] = -1
        self.cells[cell_se,:] = -1

        ## Edges
        self.edges[dead_edge_top] = -1
        self.edges[dead_edge_bot] = -1

        self.edges[merge_edge_right] = -1

        # and the one that gets rewritten to span across the quad
        self.edges[merge_edge_left,0] = quad_verts[1]
        self.edges[merge_edge_left,1] = quad_verts[3]
        # leave the marker as is
        self.edges[merge_edge_left,3] = cell_n
        self.edges[merge_edge_left,4] = cell_s

        # and surrounding edges - the diagonals northeast and southeast
        # have to have one of their cells rewritten:
        northeast_edge = self.find_edge( [quad_verts[3],quad_verts[0]] )
        southeast_edge = self.find_edge( [quad_verts[2],quad_verts[3]] )
        if self.edges[northeast_edge,3] == cell_ne:
            self.edges[northeast_edge,3] = cell_n
        if self.edges[northeast_edge,4] == cell_ne:
            self.edges[northeast_edge,4] = cell_n
        if self.edges[southeast_edge,3] == cell_se:
            self.edges[southeast_edge,3] = cell_s
        if self.edges[southeast_edge,4] == cell_se:
            self.edges[southeast_edge,4] = cell_s
            

        # and now fixup any stuff that we destroyed along the way:

        if self._pnt2cells:
            del self._pnt2cells[node] # this node is totally disowned
            for new_point in quad_verts:
                set_of_cells = self._pnt2cells[new_point]
                if cell_ne in set_of_cells:
                    set_of_cells.remove(cell_ne)
                    set_of_cells.add(cell_n)
                if cell_se in set_of_cells:
                    set_of_cells.remove(cell_se)
                    set_of_cells.add(cell_s)
        if self._vcenters:
            self._vcenters = None # lazy

        # fix pnts2edge:
        for other_point in quad_verts:
            nodes = (node,other_point)
            if nodes[0] > nodes[1]:
                nodes = (other_point,node)
            del self.pnts2edge[nodes]
        # and insert the new edge:
        nodes = (quad_verts[1],quad_verts[3])
        if nodes[0] > nodes[1]:
            nodes = (quad_verts[3],quad_verts[1])
        self.pnts2edge[nodes] = merge_edge_left
        # end fixing pnts2edge

        if verbose:
            #subplot(212)
            #cla()
            print("fix_quad: node %d, create cells %d %d"%(node,cell_n,cell_s))
            # self.plot_cells([cell_n,cell_s])

        self.quads_merged += 1
        return True

    ### Mesh quality analyses:
    def plot_clearance_hist(self,**plotargs):
        if 'bins' not in plotargs:
            plotargs['bins'] = 200
        if 'log' not in plotargs:
            plotargs['log'] = True
        
        clearances = self.vor_clearances()
        clearances = clip(clearances,-100,200)
        hist(clearances,**plotargs)
        lims = list(axis())
        lims[2] = 0.1
        axis(lims)
        
    def vor_clearances(self):
        # return, for each cell, the minimum distance
        # between the cell's voronoi center and an edge.
        # negative means it does not satisfy the ortho.
        # condition
        return self.fast_vor_clearances()

        # old, iterative code
        vcenters = self.vcenters()
        
        clearances = nan*ones( (self.Ncells()), float64 )
        for cell_i in range(self.Ncells()):
            if cell_i%300==0:
                print("%g%%"%( (100.0*cell_i)/self.Ncells() ))
                
            vor = vcenters[cell_i]

            min_dist = inf
            
            for edge_i in range(3):
                pntA = self.points[ self.cells[cell_i,edge_i], :2 ]
                pntB = self.points[ self.cells[cell_i,(edge_i+1)%3] , :2]
                pntC = self.points[ self.cells[cell_i,(edge_i+2)%3] , :2]

                # get unit vector along this edge:
                AV = vor - pntA
                AC = pntC - pntA # to figure out which side of the edge we're on
                AB_unit = (pntB - pntA) / norm(pntB-pntA)

                # project VA, CA onto AB, subtract to get perpendicular component
                AV_par =  AB_unit*dot(AV,AB_unit)
                
                AV_perp = AV - AB_unit*dot(AV,AB_unit)
                AC_perp = AC - AB_unit*dot(AC,AB_unit)

                dist = norm(AV_perp)
                if dot(AV_perp,AC_perp) < 0: # lie on opposite sides of AB
                    dist = -dist
                if dist < min_dist:
                    min_dist = dist

                if 0:
                    # debugging:
                    self.plot_cells([cell_i],label_nodes=False)
                    # interesting points:
                    pnts = array([vor,pntA,pntB,pntC])
                    plot(pnts[:,0],pnts[:,1],'ro')
                    annotate('vor',vor)
                    annotate('A',pntA)
                    annotate('B',pntB)
                    annotate('C',pntC)

                    # this one is okay...
                    vec = array([pntA,pntA+AV])
                    plot(vec[:,0],vec[:,1],'r--')

                    # the parallel projection:
                    vec = array([pntA,pntA+AV_par])
                    plot(vec[:,0],vec[:,1],'g--')

                    # should be the perpendicular:
                    vec = array([vor,vor-AV_perp])
                    plot(vec[:,0],vec[:,1],'r--')


                    print("Vor clearance: ",dist)
                    raise StopIteration
            clearances[cell_i] = min_dist

        return clearances

    def fast_vor_clearances(self):
        # return, for each cell, the minimum distance
        # between the cell's voronoi center and an edge.
        # negative means it does not satisfy the ortho.
        # condition

        vor = self.vcenters()
        
        clearances = inf*ones( (self.Ncells()), float64 )
        
        def rowdot(x,y):
            return sum(x*y,axis=1)
        def rownorm(v):
            return sqrt( v[:,0]**2 + v[:,1]**2)

        for edge_i in range(3):
            pntA = self.points[ self.cells[:,edge_i] ]
            pntB = self.points[ self.cells[:,(edge_i+1)%3]]
            pntC = self.points[ self.cells[:,(edge_i+2)%3]]

            # get unit vector along this edge:
            AV = vor - pntA
            AC = pntC - pntA # to figure out which side of the edge we're on

            AB = pntB - pntA
            
            AB_unit = AB / (rownorm(AB)[:,newaxis] )

            # project VA, CA onto AB, subtract to get perpendicular component
            # is dot going to do the right thing??
            # AB_unit is [n,2], AV is [n,2]
            #    want to dot each pair of rows to get a column vector
            #   dot gets too weird - just write out the row-dot-product
            AV_par =  AB_unit*(rowdot(AV,AB_unit)[:,newaxis])

            AV_perp = AV - AB_unit*(rowdot(AV,AB_unit)[:,newaxis])
            AC_perp = AC - AB_unit*(rowdot(AC,AB_unit)[:,newaxis])

            dist = rownorm(AV_perp)

            # negate distances that are on the opposite side from the
            # third node:
            dist = where( rowdot(AV_perp,AC_perp) < 0,
                          -dist,
                          dist)

            # store min distances
            clearances = where(dist < clearances,dist,clearances)


            if 0:
                # debugging:
                self.plot_cells([cell_i],label_nodes=False)
                # interesting points:
                pnts = array([vor,pntA,pntB,pntC])
                plot(pnts[:,0],pnts[:,1],'ro')
                annotate('vor',vor)
                annotate('A',pntA)
                annotate('B',pntB)
                annotate('C',pntC)

                # this one is okay...
                vec = array([pntA,pntA+AV])
                plot(vec[:,0],vec[:,1],'r--')

                # the parallel projection:
                vec = array([pntA,pntA+AV_par])
                plot(vec[:,0],vec[:,1],'g--')

                # should be the perpendicular:
                vec = array([vor,vor-AV_perp])
                plot(vec[:,0],vec[:,1],'r--')

                print("Vor clearance: ",dist)
                raise StopIteration

        return clearances


    def edge_clearances(self,include_boundary=True):
        """ for each edge return the signed distance between their
        voronoi centers
        """

        vor = self.vcenters()
        
        clearances = zeros( (self.edges.shape[0]), float64 )
        
        def rowdot(x,y):
            return sum(x*y,axis=1)
        def rownorm(v):
            return sqrt( v[:,0]**2 + v[:,1]**2)

        for edge_i in range(3):
            pntA = self.points[ self.cells[:,edge_i], :2 ]
            pntB = self.points[ self.cells[:,(edge_i+1)%3] , :2]
            pntC = self.points[ self.cells[:,(edge_i+2)%3] , :2]

            # get unit vector along this edge:
            AV = vor - pntA
            AC = pntC - pntA # to figure out which side of the edge we're on

            AB = pntB - pntA
            
            AB_unit = AB / (rownorm(AB)[:,newaxis] )

            # project VA, CA onto AB, subtract to get perpendicular component
            # is dot going to do the right thing??
            # AB_unit is [n,2], AV is [n,2]
            #    want to dot each pair of rows to get a column vector
            #   dot gets too weird - just write out the row-dot-product
            AV_par =  AB_unit*(rowdot(AV,AB_unit)[:,newaxis])

            AV_perp = AV - AB_unit*(rowdot(AV,AB_unit)[:,newaxis])
            AC_perp = AC - AB_unit*(rowdot(AC,AB_unit)[:,newaxis])

            dist = rownorm(AV_perp)

            # negate distances that are on the opposite side from the
            # third node:
            dist = where( rowdot(AV_perp,AC_perp) < 0,
                          -dist,
                          dist)

            for cell_i in range(self.cells.shape[0]):
                nodeA = self.cells[cell_i,edge_i]
                nodeB = self.cells[cell_i,(edge_i+1)%3]
                ABedge = self.find_edge( (nodeA,nodeB) )
                
                clearances[ABedge] += dist[cell_i]
                
        if not include_boundary:
            internal = self.edges[:,4] >= 0
            clearances = clearances[internal]

        return clearances

    ### Fix cells by moving nodes around ###
    def can_node_be_repositioned(self,free_node,region_steps=3):
        """ Returns a polygon enclosing valid locations for the
        node if it exists, otherwise return False
        region_steps: how finely to discretize the valid regions
        """
        is_internal = (self.boundary_angle(free_node) == 0.0)

        if is_internal:
            # keeps the node in a region where it doesn't change
            # the boundary shape too much
            intersection = None
        else:
            intersection = self.boundary_envelope(free_node)

        for nbr_cell in self.pnt2cells(free_node):
            # rotate the node list so the free node is first
            nodes = self.cells[nbr_cell]
            bring_to_front = find(nodes==free_node)[0]

            nodes = nodes[ (arange(3)+bring_to_front)%3 ]
            # now nodes[0] is our free node...

            points = self.points[ nodes, :2]

            free_bounds = free_node_bounds_fine(points,
                                                max_angle=self.max_angle,
                                                region_steps=region_steps)
            geom = geometry.Polygon( free_bounds )
            # geoms.append(geom)
            if intersection is None:
                intersection = geom
            else:
                intersection = intersection.intersection(geom)
            if intersection.area == 0.0:
                return False
        return intersection


    def try_to_fix_by_nudging(self,bad_cell,region_steps=3):
        """  Look for ways to fix a cell by nudging it's
        vertices.
        """
        for node in self.cells[bad_cell]:
            # only move internal nodes for now...
            repos = self.can_node_be_repositioned(node,region_steps=region_steps)
            if repos:
                new_point = point_in_polygon( repos )
                self.points[node,:2] = new_point
                self._vcenters = None
                # print "nudged: ",bad_cell
                return True
        return False


    def boundary_envelope(self,node):
        """ return a polygon that approximates the region that
        a boundary node can move in without disrupting the
        boundary too much
        """
        # find it's boundary neighbors:
        edges = self.edges[self.pnt2edges( node )]

        boundary_nodes = unique( edges[ edges[:,4]==-1,:2 ] )
        if len(boundary_nodes) != 3:
            print("How can node %i not have some friends on the boundary?"%node)
            print(boundary_nodes)
            
        nodeA,nodeC = setdiff1d(boundary_nodes,[node])

        offset = 0.03 # nodes can move 0.1 of the segment length

        pntA = self.points[nodeA,:2]
        pntB = self.points[node,:2]
        pntC = self.points[nodeC,:2]

        tot_length = norm(pntA-pntB) + norm(pntB-pntC)

        # first, the rectangle based on AB
        AB_unit = (pntB-pntA)/norm(pntB-pntA)
        perp_vec = offset*tot_length*rot(pi/2,AB_unit)

        pntNW = pntA+perp_vec
        pntSW = pntA-perp_vec

        pntNE = pntA + AB_unit*tot_length + perp_vec
        pntSE = pntA + AB_unit*tot_length - perp_vec

        AB_ring = array([pntNW,pntSW,pntSE,pntNE])

        AB_geom = geometry.Polygon( AB_ring )

        # and the rectangle based on BC
        CB_unit = (pntB-pntC)/norm(pntB-pntC)
        perp_vec = offset*tot_length*rot(pi/2,CB_unit)

        pntNW = pntC+perp_vec
        pntSW = pntC-perp_vec

        pntNE = pntC + CB_unit*tot_length + perp_vec
        pntSE = pntC + CB_unit*tot_length - perp_vec

        CB_ring = array([pntNW,pntSW,pntSE,pntNE])

        CB_geom = geometry.Polygon( CB_ring )

        return CB_geom.intersection(AB_geom)



    def nudge_cell_search(self,bad_cell,depth=1,verbose=False):
        print("--- nudge_cell_search %d ---"%bad_cell)

        for node in self.cells[bad_cell]:
            print("nudge_cell_search: trying node %d of cell %d"%(node,bad_cell))

            points_checkpoint = self.checkpoint()

            # trying this out - is this going to limit the fixes to only
            # moving one vertex of the bad_cell?
            frozen = self.cells[bad_cell]

            if self.nudge_node_search(node,frozen=frozen,depth=depth,verbose=verbose):
                print("SUCCESS")
                break # commits those changes to the nodes:
            else:
                # print "couldn't find a way out - reverting"
                self.revert(points_checkpoint)

    def nudge_node_search(self,free_node,frozen,depth,verbose=False,history=[]):
        """ try to nudge the given node, and if depth > 0,
        allow for screwing up one other triangle, then try
        to fix that triangle, without moving any nodes in frozen
        """
        indent = "   "*(self.max_depth - depth)
        history = history + [free_node]
        print(indent,"Nudging: ",history)

        if (self.boundary_angle(free_node) == 0.0): # Internal
            constraints = None
        else:
            constraints = self.boundary_envelope(free_node)

        nbr_cells = list(self.pnt2cells(free_node))

        bounds_per_neighbor = [None]*len(nbr_cells)

        for i in range(len(nbr_cells)):
            # rotate the node list so the free node is first
            nbr_cell = nbr_cells[i]
            nodes = self.cells[nbr_cell]
            bring_to_front = find(nodes==free_node)[0]
            nodes = nodes[ (arange(3)+bring_to_front)%3 ]
            # now nodes[0] is our free node...

            points = self.points[ nodes, :2]

            free_bounds = free_node_bounds_fine(points,
                                                max_angle = self.max_angle,
                                                region_steps=100)
            geom = geometry.Polygon( free_bounds )
            bounds_per_neighbor[i] = geom

        # build list with any extra constraints at the end
        if constraints:
            all_geoms = bounds_per_neighbor + [constraints]
        else:
            all_geoms = bounds_per_neighbor


        # first try to satisfy everyone:
        all_intersection = intersect_geoms( all_geoms )

        if all_intersection.area > 0.0:
            # success: choose a point in here and return it
            pnt = point_in_polygon(all_intersection)
            self.push_node(free_node,pnt)

            print(indent,"Found a good intersection, nudging node %i without branching"%free_node)
            return True

        if depth > 0:
            print(indent,"looking for branches")
            # sub-calls cannot change the node that we are repositioning
            sub_frozen = concatenate( (frozen,[free_node]) )

            for omit in range(len(nbr_cells)):
                omitted_cell = nbr_cells[omit]

                omitted_nodes = self.cells[omitted_cell]

                # if we can't move any of the omitted nodes, then don't even consider it
                if len( setdiff1d(omitted_nodes,sub_frozen) ) < 1:
                    # print indent,"Can't omit %d"%omitted_cell
                    continue

                # print indent,"Trying to omit cell %d"%omitted_cell

                partial_intersection = intersect_geoms( all_geoms[:omit] + all_geoms[omit+1:] )

                if partial_intersection.area > 0.0:
                    print(indent,"partial intersection by omitting %d"%omitted_cell)
                    pnt = point_in_polygon(partial_intersection)
                    checkpoint_this_node = self.checkpoint()

                    self.push_node(free_node,pnt)

                    # try each of the nodes in the cell that we just wrinkled
                    for nbr_node in self.cells[omitted_cell]:
                        checkpoint_branch = self.checkpoint()

                        if nbr_node not in sub_frozen:
                            res = self.nudge_node_search(nbr_node,sub_frozen,depth-1,history=history)
                            if res:
                                print(indent,"Success from the branch!")
                                return True
                            else:
                                self.revert(checkpoint_branch)
                    self.revert(checkpoint_this_node)

        return False

    def plot_spacing(self,max_clearance=20,include_boundary=False):
        ec = self.edge_clearances()

        edge_mask = ec < max_clearance

        if not include_boundary:
            edge_mask = edge_mask & (self.edges[:,4]>=0)

        self.plot(all_cells=False,line_collection_args={'cmap':cm.gray},
                  vmax=1000,vmin=-1000)
        # save this in case somebody wants to muck with it after the fact
        self.shore_collection = self.edge_collection

        self.plot(edge_mask=edge_mask, edge_values=ec[edge_mask],line_collection_args={'linewidth':3})
        ecoll = self.edge_collection
        ecoll.norm.vmin = 0
        ecoll.norm.vmax = max_clearance
        colorbar(ecoll)
    



 
# possible next steps:
# To fix narrow channels full of bad cells, maybe just clear out the triangles
# and get Triangle to fill it in with the constraint that voronoi points fall
# within the domain.
# Recognize more degenerate cases and at least skip them
# Possibility of flipping edges in quads
# Incremental increasing of the max_angle so that even valid triangles are improved
# where possible


if __name__ == '__main__':
    import getopt

    def usage():
        print("python orthomaker.py -s <input suntans datadir>")
        print("                     -t <input tecplot file>")
        print("                     -g <input SMS ADCIRC .grd file>")
        print("                     -S <output suntans datadir>")
        print("                     -G <output SMS ADCIRC .grd file>")
        print("                     -a <angle in degrees>  # max allowed angle")

    try:
        opts,rest = getopt.getopt(sys.argv[1:],'g:t:s:G:S:h')
    except getopt.GetoptError as exc:
        print(exc)
        usage()
        sys.exit(1)

    om_args = {}

    suntans_output = None
    sms_output = None

    n_inputs = 0
    n_outputs = 0
    
    for opt,val in opts: 
        if opt=='-g':
            print("Open %s as an SMS ADCIRC file"%val)
            om_args['sms_fname'] = val
            n_inputs += 1
        elif opt=='-t':
            print("Open %s as a tecplot file"%val)
            om_args['tec_file'] = val
            n_inputs += 1
        elif opt=='-s':
            print("Open suntans files in directory %s"%val)
            om_args['suntans_path'] = val
            n_inputs += 1 
        elif opt=='-h':
            usage() 
            sys.exit(1)
        elif opt == '-S':
            suntans_output = val
            print("Will write results to Suntans format in directory %s"%val)
            n_outputs += 1
        elif opt == '-G':
            sms_output = val
            print("Will write results to SMS ADCIRC format in directory %s"%val)
            n_outputs += 1
        elif opt == '-a':
            a = 180.0 * float(val) / pi
            om_args['max_angle'] = a
        else:
            print("Unknown option %s"%opt)
            usage()
            sys.exit(1)

    if n_inputs != 1:
        print("Must specify exactly one input")
        usage()
        sys.exit(1)

    if n_outputs == 0:
        print("No output specified!  Results will be discarded!")

    grid = OrthoMaker(**om_args)
    
    def save_results():
        if suntans_output:
            grid.write_suntans(suntans_output)
        if sms_output:
            grid.write_sms(sms_output)

    print("Input grid:")
    grid.stats()

    print("First pass: merge cells (internal nodes with 4 cells & straight")
    print("   boundary nodes with 2 cells)")
    grid.pass_one()

    print("After first pass:")
    grid.stats()
    save_results()

    print("Second pass: nudge nodes to satisfy local angle criteria")
    grid.pass_two()
    print("After second pass:")
    grid.stats()
    save_results()

    print("Third pass: localized search to nudge multiple nodes")
    grid.pass_three()
    print("After third pass:")
    grid.stats()
    save_results()

    
    

    
