"""
Pure python implementation of Rebay frontal delaunay method
"""

import heapq
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from . import unstructured_grid, front, exact_delaunay
from ..spatial import field
from .. import utils

DELETED=-1
UNSET=0
EXT=1
WAITING=2
ACTIVE=3
DONE=4

class RebayAdvancingDelaunay(front.AdvancingFront):
    """ 
    Implementation of Rebay, 1993 for fast triangular mesh generation.

    The approach is quite different from AdvancingFront, so while this is 
    a subclass there is relatively little shared code.
    """
    # nominal edge length
    scale=None
    
    # radius scale, for an equilateral triangle follows this:
    # (1/1.73) = 0.578
    # Generally don't have equilateral, so derate that some.
    # With apollonius scale, 1/1.5 seems to make a layer of too-long
    # edges just inside the boundary.  But decreasing to 0.625 just
    # pushes that one row in when used with an apollonius scale.
    rad_scale_factor=0.667

    # queue negative radius so heap tracks max radius
    # should expose this as an option, though.
    # in the initial test, it will run either way, and
    # mostly controls where the smoothest part of the
    # grid emerges.  a max heap (-r) fills a regular grid
    # in the interior, and the boundaries are less regular.
    # a min heap (r) prioritizes a regular grid along the
    # boundary, 
    heap_sign=-1

    # If true, follow the original paper and recover boundary
    # edges by subdivision. In tight spots (stings) this is not
    # robust, but relying on constraints may have issues when
    # cell centers fall outside the domain.
    # The real solution might be to retain constraints, but
    # subdivide those edges when a circumcenter is outside the
    # valid domain.
    recover_constraints=False
    
    def __init__(self,grid=None,**kw):
        """
        grid: edges will be copied into a constrained delaunay triangulation,
        and cells will be used to define what is 'inside'
        """
        self.grid=exact_delaunay.Triangulation()
        utils.set_keywords(self,kw)

        if grid is not None:
            self.init_from_grid(grid)
        self.xm=None
        self.new_x=None

    def execute(self):
        if self.recover_constraints:
            self.recover_boundary()
        self.instrument_grid()
        self.init_rebay()
        count=0
        while 1:
            new_n=self.step()
            if new_n is None:
                break
            count+=1

    def rad_scale(self,X):
        return self.rad_scale_factor * self.scale(X)
        
    def init_from_grid(self,g):
        self.grid.init_from_grid(g,set_valid=True,valid_min_area=1e-2)

    def instrument_grid(self):
        self.grid.add_cell_field( 'stat', np.zeros( self.grid.Ncells(),np.int32 ) )
        self.grid.add_cell_field( 'radius', np.zeros( self.grid.Ncells(),np.float64 ) )

    def recover_boundary(self):
        """
        subdivide edges as needed so that all boundary edges
        appear in the unconstrained DT.  Assumes that boundary
        edges start off as constrained
        """
        node_pairs=[]
        g=self.grid
        
        for j in np.nonzero(g.edges['constrained'])[0]:
            node_pairs.append( g.edges['nodes'][j] )
            g.remove_constraint(j=j)

        # once an edge exists, will subdividing another edge potentially
        # break it?  for now, be conservative and check for that
        while 1:
            new_node_pairs=[]
            for a,b in node_pairs:
                j=g.nodes_to_edge(a,b)
                if j is None:
                    # subdivide:
                    # 
                    raise Exception("need logic to update cells['valid']")
                    mid=0.5*(g.nodes['x'][a]+g.nodes['x'][b])
                    n_mid=g.add_node(x=mid)
                    new_node_pairs.append( [a,n_mid] )
                    new_node_pairs.append( [n_mid,b] )
                else:
                    new_node_pairs.append( [a,b] )
            assert len(new_node_pairs)>=len(node_pairs)
            if len(new_node_pairs)==len(node_pairs):
                break
            node_pairs=new_node_pairs
            
    def on_delete_cell(self,g,func,cell,*a,**k):
        self.cell_log.append( [func,cell,k] )
    def on_modify_cell(self,g,func,cell,**k):
        self.cell_log.append( [func,cell,k] )
    def on_add_cell(self,g,func,**k):
        cell=k['return_value']
        self.cell_log.append( [func,cell,k] )

    def init_rebay(self):
        """ 
        Initialize status of cells, radii
        """
        g=self.grid

        g.cells['stat'][ ~g.cells['valid'] ] = EXT

        # What do we have to listen for?
        # We're tracking cells, and need
        self.cell_log=[] # [ ['action', cell id], ...]
        
        g.subscribe_after( 'delete_cell', self.on_delete_cell )
        g.subscribe_after( 'modify_cell', self.on_modify_cell )
        g.subscribe_after( 'add_cell', self.on_add_cell )

        cc=g.cells_center()
        centroids=g.cells_centroid()

        g.cells['radius']=utils.dist( cc - g.nodes['x'][ g.cells['nodes'][:,0] ] )
        target_radii=self.rad_scale(centroids)
        alpha=g.cells['radius']/target_radii

        valid=g.cells['valid']

        g.cells['stat'][ valid & (alpha<1.0) ] = DONE
        g.cells['stat'][ valid & (alpha>=1.0) ] = WAITING

        # And ACTIVE:
        e2c=g.edge_to_cells(recalc=True)
        assert e2c[:,0].min()>=0,"Was hoping that this invariant is honored"

        good_ext=(e2c<0) | (g.cells['stat'][e2c]==DONE) | (g.cells['stat'][e2c]==EXT)
        good_int_cells = (g.cells['stat']==WAITING) | (g.cells['stat']==ACTIVE)
        good_int = good_int_cells[ e2c ]

        active_c1=good_ext[:,0] & good_int[:,1]
        active_c0=good_ext[:,1] & good_int[:,0]

        g.cells['stat'][ e2c[active_c0,0] ] = ACTIVE
        g.cells['stat'][ e2c[active_c1,1] ] = ACTIVE
        
        # priority queue for active cells
        self.active_heap=[] # elements are [radius, cell, valid]
        self.active_hash={} # cell=>entry in active_heap
        
        for c in np.nonzero( g.cells['stat']==ACTIVE )[0]:
            self.push_cell_radius( c, g.cells['radius'][c] )

    # Algorithm:
    # Choose the active cell with the largest circumradius.

    # Will heapq do what I need?
    #  it's just methods on a list. there's not a way to easily
    #  update an element's cost 

    def push_cell_radius(self,c,r):
        if c in self.active_hash:
            self.remove_active(c)
        entry=[self.heap_sign*r,c,True]
        heapq.heappush(self.active_heap, entry)
        self.active_hash[c]=entry

    def remove_active(self,cell):
        entry = self.active_hash.pop(cell)
        entry[-1] = False

    def pop_active(self):
        while self.active_heap:
            radius, cell, valid = heapq.heappop(self.active_heap)
            if valid:
                del self.active_hash[cell]
                return cell
        return None

    def select_next_edge(self):
        c_target=self.pop_active()
        if c_target is None:
            return None,None

        # Select the edge of c_target:
        j_target=None
        j_target_L=np.inf
        
        g=self.grid

        # had 12k calls to edge_to_cells, and it was over half the
        # time.  try not updating?
        e2c=g.edge_to_cells() # recalc=True)
        # assert e2c[:,0].min()>=0,"Was hoping that this invariant is honored"

        for j in g.cell_to_edges(c_target):
            c0,c1=e2c[j]
            if c1==c_target:
                c_nbr=c0
            elif c0==c_target:
                c_nbr=c1
            else:
                assert False
            if (c_nbr<0) or (g.cells['stat'][c_nbr] in [EXT,DONE]):
                # It's a candidate -- is the shortest candidate?
                # Rebay notes this is a good, but not necessarily
                # optimal choice
                L=g.edges_length(j)
                if L<j_target_L:
                    j_target=j
                    j_target_L=L
        #j=j_target  # Cruft
        
        return c_target,j_target

    def add_next_node(self,c_target,j):
        """
        calculate position of node, add it and return index.
        """
        g=self.grid
        xm=0.5*(g.nodes['x'][g.edges['nodes'][j,0]] +
                g.nodes['x'][g.edges['nodes'][j,1]])
        self.xm=xm
        C_A=g.cells_center(refresh=[c_target])[c_target]

        L=g.edges_length(j)

        rho_m=self.rad_scale(xm)
        p=0.5*L
        q=utils.dist( C_A, xm)
        rho_hat_m = min( max(rho_m,p), (p**2+q**2)/(2*q))

        d=rho_hat_m + np.sqrt( rho_hat_m**2 - p**2)
        assert np.isfinite(d) # sanity
        e_vec=utils.to_unit( C_A - xm)
        new_x=xm+d*e_vec # Can get a point outside the valid region... ???
        # for now, at least fail when this goes off the rails.
        c_new_x = g.select_cells_nearest(new_x,inside=True)
        if c_new_x is None or g.cells['stat'][c_new_x]==EXT:
            raise Exception("Rebay algorithm stepped out of bounds. Buggy code.")
        self.new_x=new_x
        new_n=g.add_node(x=new_x)
        return new_n
    
    def step(self):
        """
        Choose an edge, add a node, update state and return node index.
        Return None if complete.
        """
        print(".",end="")
        c_target,j = self.select_next_edge()
        if c_target is None:
            return None
        new_n=self.add_next_node(c_target,j)
        self.update_state(new_n)
        return new_n

    def update_state(self,new_n):
        changed_cells=set( [ entry[1] for entry in self.cell_log ] )
        del self.cell_log[:]

        g=self.grid
        Nc=g.Ncells()
        live_cells=[]

        for c in changed_cells:
            if c in self.active_hash:
                self.remove_active(c)

            if c>=Nc: # deleted and cell array truncated
                continue

            if g.cells['deleted'][c]:
                g.cells['stat'][c]=DELETED
                continue

            # Either modified or added.  update radius,
            # status, and potentially requeue
            cc=g.cells_center(refresh=[c])[c]
            rad=utils.dist( cc - g.nodes['x'][ g.cells['nodes'][c,0] ] )
            g.cells['radius'][c]=rad
            g.cells['valid'][c]=True # Should be an invariant...
            live_cells.append(c)

        live_cells=np.array(live_cells)

        centers=g.cells_centroid(live_cells)
        target_radii=self.rad_scale(centers)

        new_nbrs=list(g.node_to_cells(new_n))

        for ci,c in enumerate(live_cells):
            if c in new_nbrs:
                # per Rebay, cells adjacent to the new node are accepted
                # with a ratio of 1.5.
                thresh=1.5
            else:
                print("new non-adjacent cell") # curious if this happens
                thresh=1.1 # add 0.1 for some slop.
            if g.cells['radius'][c]/target_radii[ci] < thresh:
                g.cells['stat'][c]=DONE

        def set_active(c):
            g.cells['stat'][c]=ACTIVE
            self.push_cell_radius(c, g.cells['radius'][c])

        done=g.cells['stat'][live_cells]==DONE
        for c in live_cells[~done]:
            # ACTIVE or WAITING?
            nbrs=g.cell_to_cells(c,ordered=True) # ordered doesn't really matter I guess
            for nbr in nbrs:
                if (nbr<0) or (g.cells['stat'][nbr] in [DONE,EXT]):
                    set_active(c)
                    break
            else:
                g.cells['stat'][c]=WAITING

        # And loop through the neighbors of cells adjacent to live_cells[done]
        # and set them to be active.
        for c in live_cells[done]:
            for nbr in g.cell_to_cells(c,ordered=True):
                if nbr>=0 and g.cells['stat'][nbr]==WAITING:
                    set_active(nbr)

    def plot_progress(self):
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)
        g=self.grid
        g.plot_edges(color='k',lw=0.5)
        if self.xm is not None:
            ax.plot( [self.xm[0]], [self.xm[1]], 'bo')
        if self.new_x is not None:
            ax.plot( [self.new_x[0]], [self.new_x[1]], 'go')
        ccoll=g.plot_cells(values=g.cells['stat'],cmap='rainbow',ax=ax)
        ccoll.set_clim([0,4])

        ax.axis('equal')

    def extract_result(self):
        """ 
        Extract a clean grid from the DT
        (i.e. remove external cells)
        """
        g=self.grid.copy()

        for c in np.nonzero(g.cells['stat']==EXT)[0]:
            g.delete_cell(c)
        g.delete_orphan_edges()
        g.delete_orphan_nodes()
        g.renumber()
        return g
