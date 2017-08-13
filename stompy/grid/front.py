"""

An advancing front grid generator for use with unstructured_grid

Largely a port of paver.py.

"""
import pdb
import unstructured_grid
import exact_delaunay
import numpy as np
import time
from scipy import optimize as opt

from .. import utils

import logging
log=logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    log.warning("Plotting not available - no matplotlib")
    plt=None

# from numba import jit, int32, float64

# copied from paver verbatim, with edits to reference
# numpy identifiers via np._
# @jit(nopython=True)
# @jit
# @jit(float64(float64[:],float64[:,:,:],float64),nopython=True)
def one_point_cost(pnt,edges,target_length=5.0):
    # pnt is intended to complete a triangle with each
    # pair of points in edges, and should be to the left
    # of each edge
    penalty = 0
    
    max_angle = 85.0*np.pi/180.

    # all_edges[triangle_i,{ab,bc,ca},{x,y}]
    all_edges = np.zeros( (edges.shape[0], 3 ,2), np.float64 )
    
    # get the edges:
    all_edges[:,0,:] = edges[:,0] - pnt  # ab
    all_edges[:,1,:] = edges[:,1] - edges[:,0] # bc
    all_edges[:,2,:] = pnt - edges[:,1] # ca

    i = np.arange(3)
    im1 = (i-1)%3
    
    #--# cost based on angle:
    abs_angles = np.arctan2( all_edges[:,:,1], all_edges[:,:,0] )
    all_angles = (np.pi - (abs_angles[:,i] - abs_angles[:,im1]) % (2*np.pi)) % (2*np.pi)
        
    # a_angles = (pi - (ab_angles - ca_angles) % (2*pi)) % (2*pi)
    # b_angles = (pi - (bc_angles - ab_angles) % (2*pi)) % (2*pi)
    # c_angles = (pi - (ca_angles - bc_angles) % (2*pi)) % (2*pi)

    if 1:
        # 60 is what it's been for a while, but I think in one situation
        # this put too much weight on small angles.
        # tried considering just large angles, but that quickly blew up.
        # even just changing this to 50 still blows up.
        #  how about a small tweak - s/60/58/ ??
        worst_angle = np.abs(all_angles - 60*np.pi/180.).max() 
        alpha = worst_angle /(max_angle - 60*np.pi/180.0)

        # 10**alpha: edges got very short...
        # 5**alpha - 1: closer, but overall still short edges.
        # alpha**5: angles look kind of bad
        angle_penalty = 10*alpha**5

        # Seems like it doesn't try hard enough to get rid of almost bad angles.
        # in one case, there is a small angle of 33.86 degrees, and another angle
        # of 84.26 degrees. so the cost function only cares about the small angle
        # because it is slightly more deviant from 60deg, but we may be in a cell
        # where the only freedom we have is to change the larger angles.

        # so add this in:
        if 1:
            # extra exponential penalty for nearly bad triangles:
            # These values mean that 3 degrees before the triangle is invalid
            # the exponential cuts in and will add a factor of e by the time the
            # triangles is invalid.

            scale_rad = 3.0*np.pi/180. # radians - e-folding scale of the cost
            # max_angle - 2.0*scale_rad works..
            thresh = max_angle - 1.0*scale_rad # angle at which the exponential 'cuts in'
            big_angle_penalty = np.exp( (all_angles.max() - thresh) / scale_rad)
    else:
        alphas = (all_angles - 60*np.pi/180.) / (max_angle - 60*np.pi/180.)
        alphas = 10*alphas**4
        angle_penalty = alphas.sum()
    
    penalty += angle_penalty + big_angle_penalty

    #--# Length penalties:
    if 1:
        ab_lens = (all_edges[:,0,:]**2).sum(axis=1)
        ca_lens = (all_edges[:,2,:]**2).sum(axis=1)
        min_ab=min(ab_lens)
        min_ca=min(ca_lens)
    else:
        # maybe better for numba?
        min_ab=np.inf
        min_ca=np.inf
        for idx in range(edges.shape[0]):
            l_ab=(all_edges[idx,0,:]**2).sum()
            l_ca=(all_edges[idx,2,:]**2).sum()
            if l_ab<min_ab:
                min_ab=l_ab
            if l_ca<min_ca:
                min_ca=l_ca

    # had been using ab_lens.min(), but numba didn't like that.
    # okay - the problem is that numba doesn't understand the sum
    # above, and thinks that ab_lens is a scalar.

    min_len = min( min_ab,min_ca )
    max_len = max( min_ab,min_ca )

    undershoot = target_length**2 / min_len
    overshoot  = max_len / target_length**2

    length_penalty = 0

    length_factor = 2
    length_penalty += length_factor*(max(undershoot,1) - 1)
    length_penalty += length_factor*(max(overshoot,1) - 1)

    # paver had two other approachs, effectively commented out
    penalty += length_penalty

    return penalty
    

class Curve(object):
    """
    Boundaries which can be open or closed, indexable
    by a floating point value (including modulo arithmetic).
    By default, indexes by distance along each segment.
    """
    class CurveException(Exception):
        pass
    
    def __init__(self,points,closed=True,ccw=None):
        """
        points: [N,2] 
        closed: if True, treat this as a closed ring
        ccw: if True, make sure the order is ccw,
        False - make sure cw
        None - leave as is.
        """
        if ccw is not None:
            area=utils.signed_area(points)
            if (area>0) != bool(ccw):
                points=points[::-1,:]
                
        self.points=np.asarray(points)
        self.closed=bool(closed)
        if self.closed:
            if np.all(self.points[0]==self.points[-1]):
                pass # already duplicated
            else:
                self.points = np.concatenate( (self.points,
                                               self.points[:1,:] ) )
        else:
            assert not np.all(self.points[0]==self.points[-1])
            
        self.distances=utils.dist_along(self.points)
    def __call__(self,f,metric='distance'):
        if metric=='distance':
            if self.closed:
                # wraps around
                # double mod in case f==-eps
                f=(f % self.distances[-1]) % self.distances[-1]
            # side='right' ensures that f=0 works
            # it's unfortunately possible to get f=-eps, which rounds in
            # a way such that (f % distances[-1]) == distances[-1]
            # the double mod above might solve that
            idxs=np.searchsorted(self.distances,f,side='right') - 1
            
            alphas = (f - self.distances[idxs]) / (self.distances[idxs+1]-self.distances[idxs])
            if not np.isscalar(alphas):
                alphas = alphas[:,None]
            return (1-alphas)*self.points[idxs] + alphas*self.points[idxs+1]
        else:
            assert False
    def total_distance(self):
        return self.distances[-1]

    def upsample(self,scale,return_sources=False):
        """
        return_sources: return a second array having the distance values for each
          return point, if this is true.
        """
        # def upsample_linearring(points,density,closed_ring=1,return_sources=False):
        new_segments = []
        sources = []

        for i,(A,B) in enumerate(zip( self.points[:-1,:],
                                      self.points[1:,:] ) ):
            l = utils.dist(B-A)
            local_scale = scale( 0.5*(A+B) )

            npoints = max(1,round( l/local_scale ))
            alphas = np.arange(npoints) / float(npoints)
            alphas=alphas[:,None]
            
            new_segment = (1.0-alphas)*A + alphas*B
            new_segments.append(new_segment)
            if return_sources:
                sources.append(self.distances[i] + alphas*l)

        new_points = np.concatenate( new_segments )

        if return_sources:
            sources = np.concatenate(sources)
            return new_points,sources
        else:
            return new_points

    def OLD_distance_away(self,anchor_f,signed_distance,rtol=0.05):
        """  Find a point on the curve signed_distance away from the
        point corresponding to anchor_f, within the given relative tolerance.
        returns new_f,new_x.

        If a point could not be found within the requested tolerance, raises
        a self.CurveException.

        Starting implementation is weak - it ignores any knowledge of the piecewise
        linear geometry.  Will need to be amended to take that into account, since 
        in its current state it will succumb to local minima/maxima.
        """
        anchor_x = self(anchor_f)
        offset=signed_distance
        direc=np.sign(signed_distance)
        target_d=np.abs(signed_distance)

        last_offset=0.0
        last_d=0.0

        # first loop to bracket the distance:
        for step in range(10): # ad-hoc limit to avoid bad juju.
            new_x=self(anchor_f + offset)
            d=utils.dist(anchor_x-new_x)
            rel_err=(d - target_d)/target_d
            if -rtol < rel_err < rtol:
                return anchor_f + offset,new_x
            if rel_err<0:
                if d<last_d:
                    # this could easily be a local minimum - as this becomes important,
                    # then it would be better to include some bounds checking, and only
                    # fail when we can prove that no solution exists.
                    raise self.CurveException("Distance got smaller - need to be smarter")
                last_offset=offset
                last_d=d
                offset*=1.5
                continue
            else:
                break # can binary search
        # binary search
        low_offset=last_offset
        high_offset=offset
        for step in range(10):
            mid_offset = 0.5*(low_offset + high_offset)
            mid_x = self(anchor_f + mid_offset)
            mid_d = utils.dist(anchor_x - mid_x)
            rel_err=(mid_d - target_d)/target_d
            if -rtol<rel_err<rtol:
                return anchor_f+mid_offset,mid_x
            elif mid_d < target_d:
                low_offset=mid_offset
            else:
                high_offset=mid_offset
        else:
            raise self.CurveException("Binary search failed")

    def distance_away(self,anchor_f,signed_distance,rtol=0.05):
        """  Find a point on the curve signed_distance away from the
        point corresponding to anchor_f, within the given relative tolerance.
        returns new_f,new_x.

        If a point could not be found within the requested tolerance, raises
        a self.CurveException.
        """
        sign=int(np.sign(signed_distance))
        abs_dist=np.abs(signed_distance)

        anchor_pnt=self(anchor_f)
        anchor_idx_a=np.searchsorted(self.distances,anchor_f,side='right') - 1
        anchor_idx_b=(anchor_idx_a+1)%(len(self.points)-1)

        if sign<0:
            anchor_idx_a,anchor_idx_b=anchor_idx_b,anchor_idx_a

        # How many segment of the curve are we willing to examine? all of them,
        # but no more.
        Npnts=len(self.points)-1 # duplicate for closed ring
        max_segs=Npnts
        for segi in range(max_segs):
            idxa=anchor_idx_a+sign*segi
            idxb=idxa+sign # +-1
            idxa=idxa%Npnts
            idxb=idxb%Npnts
            if segi==0:
                # only care about the portion of the first segment
                # "ahead" of anchor (TODO: handle sign<0)
                pnta=anchor_pnt
            else:
                pnta=self.points[idxa]
            pntb=self.points[idxb]
            dista=utils.dist(pnta - anchor_pnt)
            distb=utils.dist(pntb - anchor_pnt)
            if (dista<(1-rtol)*abs_dist) and (distb<(1-rtol)*abs_dist):
                # No way this segment is good.
                continue
            else:
                break
        else:
            # i.e. checked everybody, could never get far enough
            # away
            raise self.CurveException("Could not get far enough away")
        assert dista<distb
        assert dista<(1+rtol)*abs_dist
        assert distb>(1-rtol)*abs_dist

        if segi==0:
            close_f=anchor_f
        else:
            close_f=self.distances[idxa]
        far_f=self.distances[idxb]
        if sign*far_f<sign*close_f:
            far_f+=sign*self.distances[-1]

        for maxit in range(10):
            mid_f=0.5*(close_f+far_f)
            pnt_mid=self(mid_f)
            dist_mid=utils.dist(pnt_mid - anchor_pnt)
            rel_err = (dist_mid-abs_dist)/abs_dist
            if rel_err < -rtol:
                close_f=mid_f
            elif rel_err > rtol:
                far_f=mid_f
            else:
                result=mid_f,pnt_mid
                break
        else:
            assert False
        return result

        
    def is_forward(self,fa,fb,fc):
        """ return true if fa,fb and fc are distinct and
        ordered CCW around the curve
        """
        if fa==fb or fb==fc or fc==fa:
            return False
        d=self.total_distance()
        return ((fb-fa) % d) < ((fc-fa)%d)
    def is_reverse(self,fa,fb,fc):
        return self.is_forward(fc,fb,fa)

    def signed_area(self):
        assert self.closed
        return utils.signed_area(self.points)
    def reverse(self):
        return Curve(points=self.points[::-1,:],
                     closed=self.closed)
    def plot(self,ax=None,**kw):
        ax=ax or plt.gca()
        return ax.plot(self.points[:,0],self.points[:,1],**kw)[0]
        

def internal_angle(A,B,C):
    BA=A-B
    BC=C-B
    theta_BA = np.arctan2( BA[1], BA[0] )
    theta_BC = np.arctan2( BC[1], BC[0] )
    return (theta_BA - theta_BC) % (2*np.pi)

class StrategyFailed(Exception):
    pass

class Strategy(object):
    def metric(self,site,scale_factor):
        assert False
    def execute(self,site):
        """
        Apply this strategy to the given Site.
        Returns a dict with nodes,cells which were modified 
        """
        assert False

class WallStrategy(Strategy):
    """ 
    Add two edges and a new triangle to the forward side of the
    site.
    """
    def __str__(self):
        return "<Wall>"
    def metric(self,site):
        # rough translation from paver
        theta=site.internal_angle * 180/np.pi
        scale_factor = site.edge_length / site.local_length

        # Wall can be applied in a wide variety of situations
        # angles greater than 90, Wall may be the only option
        # angles less than 60, and we can't do a wall.
        
        # np.clip( (120 - theta) / 30, 0,np.inf)
        
        # at 90, we can try, but a bisect would be better.
        # at 180, this is the only option.
        return (180-theta) / 180

    def execute(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        b,c = grid.nodes['x'][ [nb,nc] ]
        bc=c-b
        new_x = b + utils.rot(np.pi/3,bc)
        nd=grid.add_node(x=new_x,fixed=site.af.FREE)

        # new_c=grid.add_cell_and_edges( [nb,nc,nd] )
        j0=grid.nodes_to_edge(nb,nc)
        unmesh2=[grid.UNMESHED,grid.UNMESHED]
        # the correct unmeshed will get overwritten in
        # add cell.
        j1=grid.add_edge(nodes=[nc,nd],cells=unmesh2)
        j2=grid.add_edge(nodes=[nb,nd],cells=unmesh2)
        new_c=grid.add_cell(nodes=[nb,nc,nd],
                            edges=[j0,j1,j2])

        return {'nodes': [nd],
                'cells': [new_c] }

class BisectStrategy(Strategy):
    """ 
    Add three edges and two new triangles.  
    """
    def __str__(self):
        return "<Bisect>"
    def metric(self,site):
        # rough translation from paver
        theta=site.internal_angle * 180/np.pi
        scale_factor = site.edge_length / site.local_length

        # Ideal is 120 degrees for a bisect
        # Can't bisect when it's nearing 180.
        if theta> 2*89:
            return np.inf # not allowed
        else:
            ideal=120 + (1-scale_factor)*30
            return np.abs( (theta-ideal)/ 50 ).clip(0,1)

    def execute(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        b,c = grid.nodes['x'][ [nb,nc] ]
        bc=c-b
        new_x = b + utils.rot(np.pi/3,bc)
        nd=grid.add_node(x=new_x,fixed=site.af.FREE)

        # new_c=grid.add_cell_and_edges( [nb,nc,nd] )
        j_ab=grid.nodes_to_edge(na,nb)
        j_bc=grid.nodes_to_edge(nb,nc)
        
        unmesh2=[grid.UNMESHED,grid.UNMESHED]
        # the correct unmeshed will get overwritten in
        # add cell.
        j_cd=grid.add_edge(nodes=[nc,nd],cells=unmesh2)
        j_bd=grid.add_edge(nodes=[nb,nd],cells=unmesh2)
        j_ad=grid.add_edge(nodes=[na,nd],cells=unmesh2)
        new_c1=grid.add_cell(nodes=[nb,nc,nd],
                             edges=[j_bc,j_cd,j_bd])
        new_c2=grid.add_cell(nodes=[na,nb,nd],
                             edges=[j_ab,j_bd,j_ad])

        return {'nodes': [nd],
                'cells': [new_c1,new_c2],
                'edges': [j_cd,j_bd,j_ad] }
    
    
class CutoffStrategy(Strategy):
    def __str__(self):
        return "<Cutoff>"
    def metric(self,site):
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Cutoff wants a small-ish internal angle
        # If the sites edges are long, scale_factor > 1
        # and we'd like to be making smaller edges, so ideal angle gets smaller
        # 
        if theta> 89*np.pi/180:
            return np.inf # not allowed
        else:
            ideal=60 + (1-scale_factor)*30
            return np.abs(theta - ideal*np.pi/180.)
    def execute(self,site):
        grid=site.grid
        na,nb,nc=site.abc
        j0=grid.nodes_to_edge(na,nb)
        j1=grid.nodes_to_edge(nb,nc)
        j2=grid.nodes_to_edge(nc,na)
        if j2 is None:
            # typical, but if we're finishing off the last triangle, this edge
            # exists.
            j2=grid.add_edge(nodes=[nc,na],cells=[grid.UNMESHED,grid.UNMESHED])
        
        c=site.grid.add_cell(nodes=site.abc,
                             edges=[j0,j1,j2])
        
        return {'cells':[c] }

class JoinStrategy(Strategy):
    """ 
    Given an inside angle, merge the two edges
    """
    def __str__(self):
        return "<Join>"
    
    def metric(self,site):
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Cutoff wants a small-ish internal angle
        # If the sites edges are long, scale_factor > 1
        # and we'd like to be making smaller edges, so ideal angle gets smaller
        # 
        if theta> 89*np.pi/180:
            return np.inf # not allowed
        else:
            # as theta goes to 0, a Join has no effect on scale.
            # 
            # at larger theta, a join effectively coarsens
            # so if edges are too small, we want to coarsen, scale_factor
            # will be < 1
            # adding the factor of 2: it was choosing join too often.
            return 2*scale_factor * theta
    def execute(self,site):
        grid=site.grid
        na,nb,nc=site.abc
        # special case, when na and nc share a second common neighbor,
        # forming a quad, that neighbor will be kept in nd
        nd=None
        
        # choose the node to move -
        mover=None

        j_ac=grid.nodes_to_edge(na,nc)

        if j_ac is not None:
            # special case: nodes are already joined, but there is no
            # cell.
            # this *could* be extended to allow the deletion of thin cells,
            # but I don't want to get into that yet (since it's modification,
            # not creation)
            if (grid.edges['cells'][j_ac,0] >=0) or (grid.edges['cells'][j_ac,1]>=0):
                raise StrategyFailed("Edge already has real cells")
            grid.delete_edge(j_ac)
            j_ac=None
            
        # a previous version only checked fixed against HINT and SLIDE
        # when the edge j_ac existed.  Why not allow this comparison
        # even when j_ac doesn't exist?
        # need to be more careful than that, though.  The only time it's okay
        # for a SLIDE or HINT to be the mover is if anchor is on the same ring,
        # and the path between them is clear, which means b cannot be on that
        # ring.
        
        if grid.nodes['fixed'][na]==site.af.FREE:
            mover=na
            anchor=nc
        elif grid.nodes['fixed'][nc]==site.af.FREE:
            mover=nc
            anchor=na
        elif grid.nodes['oring'][na]>0 and grid.nodes['oring'][nc]>0:
            # *might* be legal but requires more checks:
            ring=grid.nodes['oring'][na]
            if ring!=grid.nodes['oring'][nc]:
                raise StrategyFailed("Cannot join across rings")
            if grid.nodes['oring'][nb]==ring:
                # this is a problem if nb falls in between them.
                fa,fb,fc=grid.nodes['ring_f'][ [na,nb,nc] ]
                curve=site.af.curves[ring-1]
                # simple orientation test, such as
                # (curve.is_forward(fa,fb,fc) or
                #  curve.is_forward(fc,fb,fa))
                # is not sufficient, since fb is between them for one
                # direction or the other.
                # so choose the shorter distance around between a and
                # c, and check that b isn't in the middle of that.
                # not sure that this will be correct in bizarrely small
                # corner cases
                tdist=curve.total_distance()
                if (fa-fc) % tdist < tdist/2:
                    if curve.is_forward(fc,fb,fa):
                        raise StrategyFailed("Cannot join across middle node")
                else:
                    if curve.is_forward(fa,fb,fc):
                        raise StrategyFailed("Cannot join across middle node")
            # probably okay, not sure if there are more checks to attempt
            if grid.nodes['fixed'][na]==site.af.HINT:
                mover,anchor=na,nc
            else:
                mover,anchor=nc,na
        else:
            raise StrategyFailed("Neither node can be moved")

        he_ab=grid.nodes_to_halfedge(na,nb)
        he_da=he_ab.rev()
        pre_a=he_da.node_rev()
        he_bc=he_ab.fwd()
        he_cd=he_bc.fwd()
        post_c=he_cd.node_fwd()
        
        if pre_a==post_c:
            log.info("Found a quad - proceeding carefully with nd")
            nd=pre_a

        # figure out external cell markers before the half-edges are invalidated.
        # note the cell index on the outside of mover, and record half-edges
        # for the anchor side
        if mover==na:
            cell_opp_mover=he_ab.cell_opp()
            cell_opp_dmover=he_da.cell_opp()
            he_anchor=he_bc
            he_danchor=he_cd
        else:
            cell_opp_mover=he_bc.cell_opp()
            cell_opp_dmover=he_cd.cell_opp()
            he_anchor=he_ab
            he_danchor=he_da
        
        edits={'cells':[],'edges':[] }

        cells_to_replace=[]
        def archive_cell(c):
            cells_to_replace.append( (c,grid.cells[c].copy()) )
            grid.delete_cell(c)

        edges_to_replace=[]
        def archive_edge(j):
            for c in grid.edges['cells'][j]:
                if c>=0:
                    archive_cell(c)

            edges_to_replace.append( (j,grid.edges[j].copy()) )
            grid.delete_edge(j)

        for j in list(grid.node_to_edges(mover)):
            archive_edge(j)
        grid.delete_node(mover)

        for j,data in edges_to_replace:
            nodes=data['nodes']

            for i in [0,1]:
                if nodes[i]==mover:
                    if (nodes[1-i]==nb) or (nodes[1-i]==nd):
                        nodes=None # signal that we don't add it
                    else:
                        nodes[i]=anchor
                    break
            if nodes is not None:
                # need to remember boundary, but any real
                # cells get added in the next step, so can
                # be -2 here.
                cells=data['cells']
                if cells[0]>=0:
                    cells[0]=-2
                if cells[1]>=0:
                    cells[1]=-2

                # This can raise Collinear exceptions
                # also, it's possible that one of these edges will be a dupe,
                # in the case of a quad
                try:
                    jnew=grid.add_edge( nodes=nodes, cells=cells )
                except exact_delaunay.ConstraintCollinearNode:
                    raise StrategyFailed("Edge was collinear with existing nodes")
                edits['edges'].append(jnew)


        for c,data in cells_to_replace:
            nodes=data['nodes']
            for ni,n in enumerate(nodes):
                if n==mover:
                    nodes[ni]=anchor
            cnew=grid.add_cell(nodes=nodes)
            edits['cells'].append(cnew)

        if cell_opp_mover<0: # need to update boundary markers
            j_cells=grid.edges['cells'][he_anchor.j,:].copy()
            j_cells[he_anchor.orient]=cell_opp_mover
            grid.modify_edge(he_anchor.j,cells=j_cells)
            
        if nd is not None and cell_opp_dmover<0:
            j_cells=grid.edges['cells'][he_danchor.j,:].copy()
            j_cells[he_danchor.orient]=cell_opp_dmover
            grid.modify_edge(he_danchor.j,cells=j_cells)
            
        # This check could also go in unstructured_grid, maybe optionally?
        areas=grid.cells_area()
        if np.any( areas[edits['cells']]<=0.0 ):
            raise StrategyFailed("Join created non-positive area cells")

        return edits

class NonLocalStrategy(Strategy):
    """ 
    Add an edge to a nearby, but not locally connected, element.
    Currently, this is not very strong in identifying whether a
    nearby node.
    """
    def __str__(self):
        return "<Nonlocal>"

    def nonlocal_pair(self,site):
        """
        Nonlocal nodes for a site 
        """
        af=site.af
        best_pair=None,None
        best_dist=np.inf

        # skip over neighbors of any of the sites nodes

        # take any neighbors in the DT.
        each_dt_nbrs=[af.cdt.node_to_nodes(n) for n in site.abc]
        if 1:
            # filter out neighbors which are not within the 'sector'
            # defined by the site.
            apnt,bpnt,cpnt=af.grid.nodes['x'][site.abc]

            ba_angle=np.arctan2(apnt[1] - bpnt[1],
                                apnt[0] - bpnt[0])
            bc_angle=np.arctan2(cpnt[1] - bpnt[1],
                                cpnt[0] - bpnt[0])

            old_each_dt_nbrs=each_dt_nbrs
            each_dt_nbrs=[]
            for nbrs in old_each_dt_nbrs:
                nbrs_pnts=af.grid.nodes['x'][nbrs]
                diffs=nbrs_pnts - bpnt
                angles=np.arctan2(diffs[:,1], diffs[:,0])
                # want to make sure that the angles from b to a,nbr,c
                # are consecutive
                angle_sum = (angles-bc_angle)%(2*np.pi) + (ba_angle-angles)%(2*np.pi)
                valid=(angle_sum < 2*np.pi)
                each_dt_nbrs.append(nbrs[valid])
            
        each_nbrs=[af.grid.node_to_nodes(n) for n in site.abc]

        # flat list of grid neighbors.  note that since a-b-c are connected,
        # this will include a,b,c, too.
        if 0:
            all_nbrs=[n for l in each_nbrs for n in l]
        else:
            all_nbrs=list(site.abc) # the way it's written, only c will be
            # picked up by the loops below.
            
            # HERE - this needs to go back to something similar to the old
            # code, where the neighbors to avoid are defined by being connected
            # along local edges within the given straight-line distance.
            he0=af.grid.nodes_to_halfedge(site.abc[0],site.abc[1])

            for incr,node,ref_pnt in [ (lambda x: x.rev(),
                                        lambda x: x.node_rev(),
                                        apnt), # walk along b->a
                                       (lambda x: x.fwd(),
                                        lambda x: x.node_fwd(),
                                        cpnt)]: # walk along b->c
                trav=incr(he0)

                while trav!=he0: # in case of small loops
                    ntrav=node(trav)
                    # some decision here about whether to calculate straight line
                    # distance from a or b, and whether the threshold is
                    # local_length or some factor thereof
                    straight_dist=utils.dist(af.grid.nodes['x'][ntrav] - ref_pnt)
                    if straight_dist > 1.0*site.local_length:
                        break
                    all_nbrs.append(ntrav)
                    trav=incr(trav)
            
        for n,dt_nbrs in zip(site.abc,each_dt_nbrs):
            # DBG: maybe only DT neighbors of 'b' can be considered?
            # when considering 'a' and 'c', too many possibilities
            # of extraneous connections, which in the past were ruled
            # out based on looking only at 'b', and by more explicitly
            # enumerating local connections
            if n!=site.abc[1]:
                continue # TESTING
            
            # most of those we are already connected to, weed them out.
            good_nbrs=[nbr
                       for nbr in dt_nbrs
                       if nbr not in all_nbrs]
            if not good_nbrs:
                continue
            
            dists=[utils.dist(af.grid.nodes['x'][n] - af.grid.nodes['x'][nbr])
                   for nbr in good_nbrs]
            idx=np.argmin(dists)
            if dists[idx]<best_dist:
                best_dist=dists[idx]
                best_pair=(n,good_nbrs[idx])
        # is the best nonlocal node connection good enough?
        # not worrying about angles, just proximity
        return best_pair[0],best_pair[1],best_dist

    def metric(self,site):
        # something high if it's bad.
        # 0.0 if it looks good
        site_node,nonlocal_node,dist = self.nonlocal_pair(site)

        scale=site.local_length
        if site_node is not None:
            # score it such that if the nonlocal connection is
            # less than or equal to the target scale away, then
            # it gets the highest score, and linearly increasing
            # based on being longer than that.
            # This may reach too far in some cases, and will need to be
            # scaled or have a nonlinear term.
            return max(0.0, (dist - scale)/scale)
        else:
            return np.inf
    
    def execute(self,site):
        site_node,nonlocal_node,dist = self.nonlocal_pair(site)
        
        if site_node is None:
            raise StrategyFailed()
        
        grid=site.grid
        
        j=grid.add_edge(nodes=[site_node,nonlocal_node],
                        cells=[grid.UNMESHED,grid.UNMESHED])

        return {'nodes': [],
                'cells': [],
                'edges': [j] }


    
Wall=WallStrategy()
Cutoff=CutoffStrategy()
Join=JoinStrategy()
Bisect=BisectStrategy()
NonLocal=NonLocalStrategy()

class Site(object):
    """
    represents a potential location for advancing the front.
    """
    def __init__(self):
        pass
    def metric(self):
        """ Smaller number means more likely to be chosen.
        """
        assert False
    def actions(self):
        return []

class FrontSite(object):
    resample_status=None
    def metric(self):
        assert False
    def plot(self,ax=None):
        assert False
    def actions(self):
        assert False

class TriangleSite(FrontSite):
    """ 
    When adding triangles, the heuristic is to choose
    tight locations.
    """
    def __init__(self,af,nodes):
        self.af=af
        self.grid=af.grid
        assert len(nodes)==3
        self.abc = nodes
    def metric(self):
        return self.internal_angle
    def points(self):
        return self.grid.nodes['x'][ self.abc ]
    
    @property
    def internal_angle(self):
        A,B,C = self.points() 
        return internal_angle(A,B,C)
    @property
    def edge_length(self):
        return utils.dist( np.diff(self.points(),axis=0) ).mean()
    
    @property
    def local_length(self):
        scale = self.af.scale
        return scale( self.points().mean(axis=0) )

    def plot(self,ax=None):
        ax=ax or plt.gca()
        points=self.grid.nodes['x'][self.abc]
        return ax.plot( points[:,0],points[:,1],'r-o' )[0]
    def actions(self):
        theta=self.internal_angle
        return [Wall,Cutoff,Join,Bisect,NonLocal]

    def resample_neighbors(self):
        """ may update site! used to be part of AdvancingFront, but
        probably better here, as part of the site.
        """
        a,b,c = self.abc
        local_length = self.af.scale( self.points().mean(axis=0) )
        
        grid=self.af.grid
        self.resample_status=True

        if self.grid.nodes['fixed'][b] == self.af.HINT:
            self.grid.modify_node(b,fixed=self.af.SLIDE)

        for n,direction in [ (a,-1),
                             (c,1) ]:
            # used to check for SLIDE and degree
            # not sure whether we should let SLIDE through...
            if grid.nodes['fixed'][n] in [self.af.HINT,self.af.SLIDE]:
                try:
                    n_res=self.af.resample(n=n,anchor=b,scale=local_length,direction=direction)
                except Curve.CurveException as exc:
                    self.resample_status=False
                    continue
                
                # is this the right time to change the fixed status?
                if grid.nodes['fixed'][n] == self.af.HINT:
                    grid.modify_node(n,fixed=self.af.SLIDE)
                
                if n!=n_res:
                    log.info("resample_neighbors changed a node")
                    if n==a:
                        self.abc[0]=n_res
                    else:
                        self.abc[2]=n_res
        return self.resample_status

# without a richer way of specifying the scales, have to start
# with marked edges
class QuadCutoffStrategy(Strategy):
    def metric(self,site):
        # how to get scale here?
        # FIX
        return 1.0 # ?
    def execute(self,site):
        """
        Apply this strategy to the given Site.
        Returns a dict with nodes,cells which were modified 
        """
        nodes=[site.abcd[0],site.abcd[3]]
        j=site.grid.nodes_to_edge(nodes)
        if j is None: # typ. case
            # Set cells to unmeshed, and one will be overwritten by add_cell.
            j=site.grid.add_edge(nodes=nodes,
                                 para=site.grid.edges['para'][site.js[1]],
                                 cells=[site.grid.UNMESHED,site.grid.UNMESHED])
        else:
            log.info("Cutoff found edge %d already exists"%j)
            
        cnew=site.grid.add_cell(nodes=site.abcd)
        
        return {'edges': [j],
                'cells': [cnew] }
QuadCutoff=QuadCutoffStrategy()

class QuadSite(FrontSite):
    def __init__(self,af,nodes):
        self.af=af
        self.grid=af.grid
        assert len(nodes)==4
        self.abcd = nodes

        self.js=[ self.grid.nodes_to_edge(nodes[:2]),
                  self.grid.nodes_to_edge(nodes[1:3]),
                  self.grid.nodes_to_edge(nodes[2:])]
        
    def metric(self):
        return 1.0 # ?
    def points(self):
        return self.grid.nodes['x'][ self.abcd ]
    
    # def internal_angle(self): ...
    # def edge_length(self): ...
    # def local_length(self): ...

    def plot(self,ax=None):
        ax=ax or plt.gca()
        points=self.grid.nodes['x'][self.abcd]
        return ax.plot( points[:,0],points[:,1],'r-o' )[0]
    
    def actions(self):
        return [QuadCutoff] # ,FloatLeft,FloatRight,FloatBoth,NonLocal?]

    def resample_neighbors(self):
        """ may update site! 
        if resampling failed, returns False. It's possible that some 
        nodes have been updated, but no guarantee that they are as far
        away as requested.

        this is where HINT nodes which part of the site are set to SLIDE nodes.
        """
        a,b,c,d = self.abcd
        # could extend to something more dynamic, like triangle does
        local_para=self.af.para_scale
        local_perp=self.af.perp_scale

        g=self.af.grid

        if g.edges['para'][self.js[1]] == self.af.PARA:
            scale=local_perp
        else:
            scale=local_para

        for n in [b,c]:
            if self.grid.nodes['fixed'][n] == self.af.HINT:
                self.grid.modify_node(n,fixed=self.af.SLIDE)
                
        self.resample_status=True
        for n,anchor,direction in [ (a,b,-1),
                                    (d,c,1) ]:
            # this used to check SLIDE and degree
            # not sure if we should let SLIDE through now...
            if self.grid.nodes['fixed'][n] in [self.af.HINT,self.af.SLIDE]:
                try:
                    n_res=self.af.resample(n=n,anchor=anchor,scale=scale,direction=direction)
                except Curve.CurveException as exc:
                    log.warning("Unable to resample neighbors")
                    self.resample_status=False
                    continue
                
                # is this the right time to change the fixed status?
                if self.grid.nodes['fixed'][n] == self.af.HINT:
                    self.grid.modify_node(n,fixed=self.af.SLIDE)
                
                if n!=n_res:
                    log.info("resample_neighbors changed a node")
                    if n==a:
                        self.abcd[0]=n_res
                    else:
                        self.abcd[3]=n_res
        return self.resample_status

class ShadowCDT(exact_delaunay.Triangulation):
    """ Tracks modifications to an unstructured grid and
    maintains a shadow representation with a constrained Delaunay
    triangulation, which can be used for geometric queries and
    predicates.
    """
    def __init__(self,g,ignore_existing=False):
        super(ShadowCDT,self).__init__(extra_node_fields=[('g_n','i4')])
        self.g=g
        
        self.nodemap_g_to_local={}

        g.subscribe_before('add_node',self.before_add_node)
        g.subscribe_after('add_node',self.after_add_node)
        g.subscribe_before('modify_node',self.before_modify_node)
        g.subscribe_before('delete_node',self.before_delete_node)
        
        g.subscribe_before('add_edge',self.before_add_edge)
        g.subscribe_before('delete_edge',self.before_delete_edge)
        g.subscribe_before('modify_edge',self.before_modify_edge)

        if not ignore_existing and g.Nnodes():
            self.init_from_grid(g)

    def init_from_grid(self,g): # ShadowCDT
        # Nodes:
        n_valid=~g.nodes['deleted']
        points=g.nodes['x'][n_valid]
        self.bulk_init(points)

        pidxs=np.arange(g.Nnodes())[n_valid]
        self.nodes['g_n']=pidxs

        for n in range(self.Nnodes()):
            gn=self.nodes['g_n'][n]
            self.nodemap_g_to_local[gn]=n

        # Edges:
        for ji,j in enumerate(g.valid_edge_iter()):
            if ji%5000==0:
                log.info("Edges: %d/%d"%(ji,g.Nedges()))
            self.before_add_edge(g,'add_edge',nodes=g.edges['nodes'][j])

    def before_add_node(self,g,func_name,**k):
        pass # no checks quite yet
    def after_add_node(self,g,func_name,return_value,**k):
        n=return_value
        my_k={}
        # re: _index
        # as long as there aren't Steiner vertices and the like, then
        # it's safe to force node index here to match the parent
        self.nodemap_g_to_local[n]=self.add_node(x=k['x'],g_n=n,_index=n)
    def before_modify_node(self,g,func_name,n,**k):
        if 'x' in k:
            my_n=self.nodemap_g_to_local[n]
            self.modify_node(my_n,x=k['x'])
    def before_delete_node(self,g,func_name,n,**k):
        self.delete_node(self.nodemap_g_to_local[n])
        del self.nodemap_g_to_local[n]
    def before_add_edge(self,g,func_name,**k):
        nodes=k['nodes']
        self.add_constraint(nodes[0],nodes[1])
    def before_modify_edge(self,g,func_name,j,**k):
        if 'nodes' not in k:
            return
        old_nodes=g.edges['nodes'][j]
        new_nodes=k['nodes']
        self.remove_constraint( old_nodes[0],old_nodes[1])
        self.add_constraint( new_nodes[0],new_nodes[1] )
    def before_delete_edge(self,g,func_name,j,**k):
        nodes=g.edges['nodes'][j]
        self.remove_constraint(nodes[0],nodes[1])

        
class AdvancingFront(object):
    """
    Implementation of advancing front
    """
    grid=None
    cdt=None

    # 'fixed' flags:
    #  in order of increasing degrees of freedom in its location.
    # don't use 0 here, so that it's easier to detect uninitialized values
    RIGID=1 # should not be moved at all
    SLIDE=2 # able to slide along a ring.  
    FREE=3  # not constrained
    HINT=4  # slidable and can be removed.

    StrategyFailed=StrategyFailed
    
    def __init__(self,grid=None):
        """
        """
        self.log = logging.getLogger("AdvancingFront")

        if grid is None:
            grid=unstructured_grid.UnstructuredGrid()
        self.grid = self.instrument_grid(grid)

        self.curves=[]
        
    def add_curve(self,curve,interior=None):
        if not isinstance(curve,Curve):
            if interior is not None:
                ccw=not interior
            else:
                ccw=None
            curve=Curve(curve,ccw=ccw,closed=True)
        elif interior is not None:
            assert curve.closed
            a=curve.signed_area()
            if a>0 and interior:
                curve=curve.reverse()
            
        self.curves.append( curve )
        return len(self.curves)-1

    def instrument_grid(self,g):
        """
        Add fields to the given grid to support advancing front
        algorithm.  Modifies grid in place, and returns it.

        Also creates a Triangulation which follows modifications to 
        the grid, keeping a constrained Delaunay triangulation around.
        """
        # oring is stored 1-based, so that the default 0 value is
        # the nan value.
        g.add_node_field('oring',np.zeros(g.Nnodes(),'i4'),on_exists='pass')
        g.add_node_field('fixed',np.zeros(g.Nnodes(),'i4'),on_exists='pass')
        g.add_node_field('ring_f',-1*np.ones(g.Nnodes(),'f8'),on_exists='pass')

        # Subscribe to operations *before* they happen, so that the constrained
        # DT can signal that an invariant would be broken
        self.cdt=ShadowCDT(g)
                          
        return g
    
    def initialize_boundaries(self):
        for curve_i,curve in enumerate(self.curves):
            curve_points,srcs=curve.upsample(self.scale,return_sources=True)

            # add the nodes in:
            # used to initialize as SLIDE
            nodes=[self.grid.add_node(x=curve_points[j],
                                      oring=curve_i+1,
                                      ring_f=srcs[j],
                                      fixed=self.HINT)
                   for j in range(len(curve_points))]

            if curve.closed:
                Ne=len(curve_points)
            else:
                Ne=len(curve_points) - 1

            pairs=zip( np.arange(Ne),
                       (np.arange(Ne)+1)%Ne)
            for na,nb in pairs:
                self.grid.add_edge( nodes=[nodes[na],nodes[nb]],
                                    cells=[self.grid.UNMESHED,
                                           self.grid.UNDEFINED] )

    def enumerate_sites(self):
        raise Exception("Implement in subclass")

    def choose_site(self):
        sites=self.enumerate_sites()
        if len(sites):
            scores=[ site.metric()
                     for site in sites ]
            best=np.argmin( scores ) 
            return sites[best]
        else:
            return None
        
    def free_span(self,he,max_span,direction):
        """
        returns the distance, and the nodes making up the 
        span, starting from anchor (the rev node of he),
        and going until either max_span distance is found, 
        it wraps around, or encounters a non-SLIDE-able node.

        the reason this works with halfedges is that we only
        move along nodes which are simply connected (degree 2)
        """
        span=0.0
        if direction==1:
            trav=he.node_fwd()
            last=anchor=he.node_rev()
        else:
            trav=he.node_rev()
            last=anchor=he.node_fwd()

        nodes=[last] # anchor is included

        def pred(n):
            # used to check for SLIDE and degree
            return self.grid.nodes['fixed'][n]== self.HINT

        while pred(trav) and (trav != anchor) and (span<max_span):
            span += utils.dist( self.grid.nodes['x'][last] -
                                self.grid.nodes['x'][trav] )
            nodes.append(trav)
            if direction==1:
                he=he.fwd()
                last,trav = trav,he.node_fwd()
            elif direction==-1:
                he=he.rev()
                last,trav = trav,he.node_rev()
            else:
                assert False
        # could use some loop retrofitting..
        span += utils.dist( self.grid.nodes['x'][last] -
                            self.grid.nodes['x'][trav] )
        nodes.append(trav)
        return span,nodes
    
    max_span_factor=4     
    def resample(self,n,anchor,scale,direction):
        """
        move/replace n, such that from anchor to n/new_n the edge
        length is close to scale.

        assumes that n is SLIDE or HINT.
        If n has more than 2 neighbors, does nothing and returns n as is.

        normally, a SLIDE node cannot be deleted.  in some cases resample will
        create a new node for n, and it will be a SLIDE node.  in that case, should
        n retain SLIDE, too? is it the responsibility of resample(), or the caller?
        can we at least guarantee that no other nodes need to be changing status?

        as it stands, if this code creates a new node, it is given fixed=SLIDE.
        it is up to the caller to reset the fixed of the original
        node to HINT, if that is what is desired.  

        Returns the resampled node index -- often same as n, but may be a different
        node.
        """
        self.log.debug("resample %d to be %g away from %d in the %s direction"%(n,scale,anchor,
                                                                                direction) )
        n_deg=self.grid.node_degree(n)
        if n_deg!=2:
            self.log.info("Will not resample node %d because degree=%d, not 2"%(n,n_deg))
            return n
        
        if direction==1: # anchor to n is t
            he=self.grid.nodes_to_halfedge(anchor,n)
        elif direction==-1:
            he=self.grid.nodes_to_halfedge(n,anchor)
        else:
            assert False

        span_length,span_nodes = self.free_span(he,self.max_span_factor*scale,direction)
        # anchor-n distance should be in there, already.
        
        self.log.debug("free span from the anchor is %g"%span_length)

        if span_length < self.max_span_factor*scale:
            n_segments = max(1,round(span_length / scale))

            if n_segments==1:
                # in tight situations, need to make sure
                # that for a site a--b--c we're not trying
                # move c all the way on top of a.
                # it is not sufficient to just force two
                # segments, as that just pushes the issue into
                # the next iteration, but in an even worse state.
                if direction==-1:
                    he_other=he.fwd()
                    opposite_node=he_other.node_fwd()
                else:
                    he_other=he.rev()
                    opposite_node=he_other.node_rev()
                if opposite_node==span_nodes[-1]:
                    self.log.info("n_segment=1, but that would be an implicit join")
                    
                    # rather than force two segments, force it
                    # to remove all but the last edge
                    span_nodes=span_nodes[:-1]
            target_span = span_length / n_segments
            if n_segments==1:
                self.log.debug("Only one space for 1 segment")
                for d in span_nodes[1:-1]:
                    self.grid.merge_edges(node=d)
                return span_nodes[-1]
        else:
            target_span=scale

        # first, find a point on the original ring which satisfies the target_span
        anchor_oring=self.grid.nodes['oring'][anchor]-1
        oring=self.grid.nodes['oring'][n]-1
        if anchor_oring != oring:
            self.log.warning('resample: anchor and n on different rings.  Will not resample')
            # could try to get clever and resample n in the "correct" direction.
            return n
        curve = self.curves[oring]
        anchor_f = self.grid.nodes['ring_f'][anchor]
        try:
            new_f,new_x = curve.distance_away(anchor_f,direction*target_span)
        except Curve.CurveException as exc:
            raise

        # check to see if there are other nodes in the way, and remove them.
        nodes_to_delete=[]
        trav=he
        while True:
            if direction==1:
                trav=trav.fwd()
            else:
                trav=trav.rev()
            # we have anchor, n, and
            if trav==he:
                self.log.error("Made it all the way around!")
                raise Exception("This is probably bad")

            if direction==1:
                n_trav=trav.node_fwd()
                f_trav=self.grid.nodes['ring_f'][n_trav]
                if curve.is_forward( anchor_f, new_f, f_trav ):
                    break
            else:
                n_trav=trav.node_rev()
                f_trav=self.grid.nodes['ring_f'][n_trav]
                if curve.is_reverse( anchor_f, new_f, f_trav ):
                    break

            nodes_to_delete.append(n_trav)

        for d in nodes_to_delete:
            self.grid.merge_edges(node=d)

        # on the other hand, it may be that the next node is too far away, and it
        # would be better to divide the edge than to shift a node from far away.
        # also possible that our neighbor was RIGID and can't be shifted
        method='slide'
        if (self.grid.nodes['fixed'][n] == self.RIGID):
            method='split'
        else:
            dist_orig = utils.dist( self.grid.nodes['x'][anchor] - self.grid.nodes['x'][n] )
            # tunable parameter here - how do we decide between shifting a neighbor and
            # dividing the edge.  Larger threshold means shifting nodes from potentially far
            # away, which distorts later steps.  smaller threshold means subdividing, but then
            # there could be the potential to bump into that node during optimization (which
            # is probably okay - we clear out interfering nodes like that).
            if dist_orig / scale > 1.5: 
                method='split'
        if method=='slide':
            self.grid.modify_node(n,x=new_x,ring_f=new_f)
            return n
        else: # 'split'
            j=self.grid.nodes_to_edge([anchor,n])
            jnew,nnew = self.grid.split_edge(j,x=new_x,ring_f=new_f,oring=oring+1,
                                             fixed=self.SLIDE)
            return nnew

    def resample_neighbors(self,site):
        return site.resample_neighbors()

    def cost_function(self,n):
        raise Exception("Implement in subclass")

    def eval_cost(self,n):
        fn=self.cost_function(n)
        return fn and fn(self.grid.nodes['x'][n])

    def optimize_nodes(self,nodes,max_levels=3,cost_thresh=2):
        max_cost=0

        for level in range(max_levels):
            for n in nodes:
                max_cost=max(max_cost,self.relax_node(n))
            if max_cost <= cost_thresh:
                break
            if level==0:
                # just try re-optimizing once
                pass
            else:
                pass
                # expand list of nodes one level

    def optimize_edits(self,edits,**kw):
        """
        Given a set of elements (which presumably have been modified
        and need tuning), jostle nodes around to improve the cost function
        """
        nodes = edits.get('nodes',[])
        for c in edits.get('cells',[]):
            for n in self.grid.cell_to_nodes(c):
                if n not in nodes:
                    nodes.append(n)
        return self.optimize_nodes(nodes,**kw)

    def relax_node(self,n):
        """ Move node n, subject to its constraints, to minimize
        the cost function.  Return the final value of the cost function
        """
        self.log.debug("Relaxing node %d"%n)
        if self.grid.nodes['fixed'][n] == self.FREE:
            return self.relax_free_node(n)
        elif self.grid.nodes['fixed'][n] == self.SLIDE:
            return self.relax_slide_node(n)
        else:
            raise Exception("relax_node with fixed=%s"%self.grid.nodes['fixed'][n])

    def relax_free_node(self,n):
        cost=self.cost_function(n)
        if cost is None:
            return None
        x0=self.grid.nodes['x'][n]
        local_length=self.scale( x0 )
        new_x = opt.fmin(cost,
                         x0,
                         xtol=local_length*1e-4,
                         disp=0)
        dx=utils.dist( new_x - x0 )
        self.log.debug('Relaxation moved node %f'%dx)
        if dx !=0.0:
            self.grid.modify_node(n,x=new_x)
        return cost(new_x)

    def relax_slide_node(self,n):
        cost_free=self.cost_function(n)
        if cost_free is None:
            return 
        x0=self.grid.nodes['x'][n]
        f0=self.grid.nodes['ring_f'][n]
        ring=self.grid.nodes['oring'][n]-1

        assert np.isfinite(f0)
        assert ring>=0

        # used to just be f, but I think it's more appropriate to
        # be f[0]
        cost_slide=lambda f: cost_free( self.curves[ring](f[0]) )

        local_length=self.scale( x0 )

        slide_limits=self.find_slide_limits(n,3*local_length)
        
        new_f = opt.fmin(cost_slide,
                         [f0],
                         xtol=local_length*1e-4,
                         disp=0)

        if not self.curves[ring].is_forward(slide_limits[0],
                                            new_f,
                                            slide_limits[1]):
            self.log.info("Slide went outside limits")
            return cost_free(x0)
        
        if new_f[0]!=f0:
            self.slide_node(n,new_f[0]-f0)
        return cost_slide(new_f)

    def find_slide_limits(self,n,cutoff=None):
        """ Returns the range of allowable ring_f for n.
        limits are exclusive
        cutoff: a distance along the curve beyond which we don't
        care. note that this is not as the crow flies, but tracing
        the segments.  So a point which is cutoff away may be much
        closer as the crow flies.
        """
        n_ring=self.grid.nodes['oring'][n]-1
        n_f=self.grid.nodes['ring_f'][n]
        curve=self.curves[n_ring]
        L=curve.total_distance()

        # find our two neighbors on the ring:check forward:
        nbrs=[]
        for nbr in self.grid.node_to_nodes(n):
            if self.grid.nodes['oring'][nbr]-1!=n_ring:
                continue
            nbrs.append(nbr)
        if len(nbrs)>2:
            # annoying, but happens.  one or more edges are internal,
            # and two are along the curve.
            nbrs.append(n)
            # sort them along the ring
            all_f=(self.grid.nodes['ring_f'][nbrs]-n_f) % L
            order=np.argsort(all_f)
            nbrs=[ nbrs[order[-1]], nbrs[order[1]] ]
        assert len(nbrs)==2
        
        if curve.is_forward(self.grid.nodes['ring_f'][nbrs[0]],
                            n_f,
                            self.grid.nodes['ring_f'][nbrs[1]] ):
            pass # already in nice order
        else:
            nbrs=[nbrs[1],nbrs[0]]
        
        # Backward then forward
        stops=[]
        for sgn,nbr in zip( [-1,1], nbrs ):
            trav=[n,nbr]
            while 1:
                # beyond cutoff?
                if ( (cutoff is not None) and
                     (sgn*(self.grid.nodes['ring_f'][trav[1]] - n_f) )%L > cutoff ):
                    break
                # is trav[1] something which limits the sliding of n?
                trav_nbrs=self.grid.node_to_nodes(trav[1])
                # if len(trav_nbrs)>2:
                #     break
                # if self.grid.nodes['fixed'][trav[1]] != self.SLIDE:
                #     break

                # the transition to HINT
                if self.grid.nodes['fixed'][trav[1]] != self.HINT:
                    break
                
                for nxt in trav_nbrs:
                    if nxt not in trav:
                        break
                # before updating, check to see if this edge has
                # a cell on it.  If it does, then even if the node is degree
                # 2, we can't slide through it.
                j=self.grid.nodes_to_edge( [trav[1],nxt] )
                j_c=self.grid.edges['cells'][j]
                if j_c[0]>=0 or j_c[1]>=0:
                    # adjacent cells, can't slide through here.
                    break

                trav=[trav[1],nxt]
            stops.append(trav[1])
            
        return self.grid.nodes['ring_f'][ stops ]
    
    def find_slide_conflicts(self,n,delta_f):
        n_ring=self.grid.nodes['oring'][n]-1
        n_f=self.grid.nodes['ring_f'][n]
        new_f=n_f + delta_f
        curve=self.curves[n_ring]
        # Want to find edges in the direction of travel
        # it's a little funny to use half-edges, since what
        # really care about is what it's facing
        # would like to use half-edges here, but it's not entirely
        # well-defined, so rather than introduce some future pitfalls,
        # do things a bit more manually.
        to_delete=[]
        for nbr in self.grid.node_to_nodes(n):
            if self.grid.nodes['oring'][nbr]-1!=n_ring:
                continue

            nbr_f=self.grid.nodes['ring_f'][nbr]
            if self.grid.node_degree(nbr)!=2:
                continue

            if delta_f>0:
                # either the nbr is outside our slide area, or could
                # be in the opposite direction along the ring
                if curve.is_forward(n_f,n_f+delta_f,nbr_f):
                    continue
                to_delete.append(nbr)
                he=self.grid.nodes_to_halfedge(n,nbr)
                while 1:
                    he=he.fwd()
                    nbr=he.node_fwd()
                    nbr_f=self.grid.nodes['ring_f'][nbr]
                    if curve.is_forward(n_f,n_f+delta_f,nbr_f):
                        break
                    to_delete.append(nbr)
                break
            else:
                if curve.is_reverse(n_f,n_f+delta_f,nbr_f):
                    continue
                to_delete.append(nbr)
                he=self.grid.nodes_to_halfedge(nbr,n)
                while 1:
                    he=he.rev()
                    nbr=he.node_rev()
                    nbr_f=self.grid.nodes['ring_f'][nbr]
                    if curve.is_reverse(n_f,n_f+delta_f,nbr_f):
                        break
                    to_delete.append(nbr)
                break
        # sanity checks:
        for nbr in to_delete:
            assert n_ring==self.grid.nodes['oring'][nbr]-1
            # OLD COMMENT:
            # For now, depart a bit from paver, and rather than
            # having HINT nodes, HINT and SLIDE are both fixed=SLIDE,
            # but differentiate based on node degree.
            # NEW COMMENT:
            # actually, that was a bad idea.  better to stick with
            # how it was in paver
            assert self.grid.nodes['fixed'][nbr]==self.HINT # SLIDE
            assert self.grid.node_degree(nbr)==2
        return to_delete
    
    def slide_node(self,n,delta_f):
        conflicts=self.find_slide_conflicts(n,delta_f)
        for nbr in conflicts:
            self.grid.merge_edges(node=nbr)

        n_ring=self.grid.nodes['oring'][n]-1
        n_f=self.grid.nodes['ring_f'][n]
        new_f=n_f + delta_f
        curve=self.curves[n_ring]

        self.grid.modify_node(n,x=curve(new_f),ring_f=new_f)

    def loop(self,count=0):
        while 1:
            site=self.choose_site()
            if site is None:
                break
            self.advance_at_site(site)
            count-=1
            if count==0:
                break
            
    def advance_at_site(self,site):
        # This can modify site! May also fail.
        resampled_success = self.resample_neighbors(site)
        
        actions=site.actions()
        metrics=[a.metric(site) for a in actions]
        bests=np.argsort(metrics)
        for best in bests:
            try:
                cp=self.grid.checkpoint()
                self.log.info("Chose strategy %s"%( actions[best] ) )
                edits=actions[best].execute(site)
                self.optimize_edits(edits)
                # could commit?
            except self.cdt.IntersectingConstraints as exc:
                self.log.error("Intersecting constraints - rolling back")
                self.grid.revert(cp)
                continue
            break
        else:
            self.log.error("Exhausted the actions!")
            return False
        return True
        
    zoom=None
    def plot_summary(self,ax=None,
                     label_nodes=True,
                     clip=None):
        ax=ax or plt.gca()
        ax.cla()

        for curve in self.curves:
            curve.plot(ax=ax,color='0.5',zorder=-5)

        self.grid.plot_edges(ax=ax,clip=clip)
        if label_nodes:
            labeler=lambda ni,nr: str(ni)
        else:
            labeler=None
        self.grid.plot_nodes(ax=ax,labeler=labeler,clip=clip)
        ax.axis('equal')
        if self.zoom:
            ax.axis(self.zoom)


class AdvancingTriangles(AdvancingFront):
    """ 
    Specialization which roughly mimics tom, creating only triangles
    """
    scale=None
    def __init__(self,grid=None,scale=None):
        super(AdvancingTriangles,self).__init__(grid=grid)
        if scale is not None:
            self.set_edge_scale(scale)
    def set_edge_scale(self,scale):
        self.scale=scale

    def enumerate_sites(self):
        sites=[]
        # FIX: This doesn't scale!
        valid=(self.grid.edges['cells'][:,:]==self.grid.UNMESHED) 
        J,Orient = np.nonzero(valid)

        for j,orient in zip(J,Orient):
            if self.grid.edges['deleted'][j]:
                continue
            he=self.grid.halfedge(j,orient)
            he_nxt=he.fwd()
            a=he.node_rev()
            b=he.node_fwd()
            bb=he_nxt.node_rev()
            c=he_nxt.node_fwd()
            assert b==bb

            sites.append( TriangleSite(self,nodes=[a,b,c]) )
        return sites

    def cost_function(self,n):
        local_length = self.scale( self.grid.nodes['x'][n] )
        my_cells = self.grid.node_to_cells(n)

        if len(my_cells) == 0:
            return None

        cell_nodes = [self.grid.cell_to_nodes(c)
                      for c in my_cells ]

        # for the moment, can only deal with triangles
        cell_nodes=np.array(cell_nodes)

        # pack our neighbors from the cell list into an edge
        # list that respects the CCW condition that pnt must be on the
        # left of each segment
        for j in range(len(cell_nodes)):
            if cell_nodes[j,0] == n:
                cell_nodes[j,:2] = cell_nodes[j,1:]
            elif cell_nodes[j,1] == n:
                cell_nodes[j,1] = cell_nodes[j,0]
                cell_nodes[j,0] = cell_nodes[j,2] # otherwise, already set

        edges = cell_nodes[:,:2]
        edge_points = self.grid.nodes['x'][edges]

        def cost(x,edge_points=edge_points,local_length=local_length):
            return one_point_cost(x,edge_points,target_length=local_length)

        return cost


#### 

def one_point_quad_cost(x,edge_scales,quads,para_scale,perp_scale):
    # orthogonality cost:
    ortho_cost=0.0

    base_scale=np.sqrt( para_scale**2 + perp_scale**2 )
    
    quads[:,0,:] = x # update the first point of each quad

    for quad in quads:
        cc=utils.poly_circumcenter(quad)
        dists=utils.mag(quad-cc)
        err=np.std(dists) / base_scale

        ortho_cost += 10*err # ad hoc hoc hoc

    # length cost:
    scale_cost=0.0

    dists=utils.mag(x - edge_scales[:,:2])
    errs=(dists - edge_scales[:,2]) / edge_scales[:,2]
    scale_cost = (2*errs**2).sum()

    return ortho_cost+scale_cost

class AdvancingQuads(AdvancingFront):
    PARA=1
    PERP=2
    
    para_scale=None
    perp_scale=None

    def __init__(self,grid=None,scale=None,perp_scale=None):
        super(AdvancingQuads,self).__init__(grid=grid)
        
        if scale is not None:
            if perp_scale is None:
                self.set_edge_scales(scale,scale)
            else:
                self.set_edge_scales(scale,perp_scale)

    def instrument_grid(self,g):
        super(AdvancingQuads,self).instrument_grid(g)

        # 0 for unknown, 1 for parallel, 2 for perpendicular
        g.add_edge_field('para',np.zeros(g.Nedges(),'i4'),on_exists='pass')
        
        return g
    def set_edge_scales(self,para_scale,perp_scale):
        self.para_scale=para_scale
        self.perp_scale=perp_scale

    def add_existing_curve_surrounding(self,x):
        # Get the nodes:
        pc=self.grid.enclosing_nodestring(x,self.grid.Nnodes())
        if pc is None:
            raise Exception("No ring around this rosey")

        curve_idx=self.add_curve( Curve(self.grid.nodes['x'][pc],closed=True) )
        curve=self.curves[curve_idx]

        # update those nodes to reflect their relationship to this curve.
        # don't forget it's 1-based!
        self.grid.nodes['oring'][pc]=1+curve_idx
        self.grid.nodes['ring_f'][pc]=curve.distances[:-1] 

        for n in pc:
            degree=self.grid.node_degree(n)
            assert degree >= 2
            if degree==2:
                self.grid.nodes['fixed']=self.HINT # self.SLIDE
            else:
                self.grid.nodes['fixed']=self.RIGID

        # and mark the internal edges as unmeshed:
        for na,nb in utils.circular_pairs(pc):
            j=self.grid.nodes_to_edge([na,nb])
            if self.grid.edges['nodes'][j,0]==na:
                side=0
            else:
                side=1
            self.grid.edges['cells'][j,side]=self.grid.UNMESHED
            # and for later sanity checks, mark the other side as outside (-1)
            # if it's -99.
            if self.grid.edges['cells'][j,1-side]==self.grid.UNKNOWN:
                self.grid.edges['cells'][j,1-side]=self.grid.UNDEFINED
            
    def orient_quad_edge(self,j,orient):
        self.grid.edges['para'][j]=orient

    def enumerate_sites(self):
        sites=[]
        # FIX: This doesn't scale!
        valid=(self.grid.edges['cells'][:,:]==self.grid.UNMESHED)& (self.grid.edges['para']!=0)[:,None]
        J,Orient = np.nonzero(valid)

        for j,orient in zip(J,Orient):
            if self.grid.edges['deleted'][j]:
                continue
            he=self.grid.halfedge(j,orient)
            he_nxt=he.fwd()
            a=he.rev().node_rev()
            b=he.node_rev()
            c=he.node_fwd()
            d=he.fwd().node_fwd()

            sites.append( QuadSite(self,nodes=[a,b,c,d]) )
        return sites

    def cost_function(self,n):
        local_para = self.para_scale
        local_perp = self.perp_scale

        my_cells = self.grid.node_to_cells(n)

        if len(my_cells) == 0:
            return None

        if 0:
            # HERE: needs to handle mix of n-gons
            cell_nodes = [self.grid.cell_to_nodes(c)
                          for c in my_cells ]

            cell_nodes=np.array(cell_nodes) # may contain undef nodes

            # make sure all quads:
            assert np.all( cell_nodes[:,:4]>=0 )
            assert np.all( cell_nodes[:,4:]<0 ) # does that work?
        else:
            # more general -
            cell_nodes=self.grid.cells['nodes'][my_cells]
            # except that for the moment I'm only going to worry about the
            # quads:
            sel_quad=(cell_nodes[:,3]>=0)
            if self.grid.max_sides>4:
                sel_quad &=(cell_nodes[:,4]<0)
            cell_nodes=cell_nodes[sel_quad]

        # For each quad, rotate our node to be at the front of the list:
        quad_nodes=[np.roll(quad,-list(quad).index(n))
                    for quad in cell_nodes[:,:4]]
        quad_nodes=np.array(quad_nodes)
        quads=self.grid.nodes['x'][quad_nodes]

        # for the moment, don't worry about reestablishing scale, just
        # focus on orthogonality

        edge_scales=np.zeros( [0,3], 'f8') # np.array( [ [x,y,target_distance], ... ] )

        def cost(x,edge_scales=edge_scales,quads=quads,
                 local_para=local_para,local_perp=local_perp):
            return one_point_quad_cost(x,edge_scales,quads,local_para,local_perp)

        return cost

    def scale(self, x0):
        # temporary hack - needed for relax_slide_node
        return 0.5*(self.para_scale+self.perp_scale)

# Classes related to the decision tree
class DTNode(object):
    parent=None 
    af=None # AdvancingTriangles object
    cp=None # checkpoint
    ops_parent=None # chunk of op_stack to get from parent to here.
    options=None # node-specific list of data for child options
    
    children=None # filled in by subclass [DTNode, ... ]
    child_prior=None # est. cost for child
    child_post =None # actual cost for child
    
    def __init__(self,af,parent=None):
        self.af=af
        self.parent=parent
        # in cases where init of the node makes some changes,
        # this should be updated
        self.cp=af.grid.checkpoint() 
        self.active_child=None
    def set_options(self,options,priors):
        self.options=options
        self.child_prior=priors
        
        N=len(options)
        self.children=[None] * N
        self.child_post =[None]*N
        self.child_order=np.argsort(self.child_prior) 
        
    def revert_to_parent(self):
        if self.parent is None:
            return False
        return self.parent.revert_to_here()
    def revert_to_here(self):
        self.af.grid.revert(self.cp)
        self.af.current=self
        self.active_child=None

    def try_child(self,i):
        assert False # implemented by subclass
        
    def best_child(self,count=0,cb=None):
        """
        Try all, (or up to count) children, 
        use the best one based on post scores.
        If no children succeeded, return False, otherwise True
        """
        if count:
            count=min(count,len(self.options))
        else:
            count=len(self.options)

        best=None
        for i in range(count):
            print("best_child: trying %d / %d"%(i,count))
            
            if self.try_child(i):
                if cb: cb()
                if best is None:
                    best=i
                elif self.child_post[i] < self.child_post[best]:
                    best=i
                if i<count-1: 
                    self.revert_to_here()
            else:
                print("best_child: option %d did not succeed"%i)
        if best is None:
            # no children worked out -
            print("best_child: no children worked")
            return False
        
        # wait to see if the best was the last, in which case
        # can save an undo/redo
        if best!=count-1:
            self.revert_to_here()
            self.try_child(best)
        return True

class DTChooseSite(DTNode):
    def __init__(self,af,parent=None):
        super(DTChooseSite,self).__init__(af=af,parent=parent)
        sites=af.enumerate_sites()
        
        priors=[ site.metric()
                 for site in sites ]
        
        self.set_options(sites,priors)
        
    def try_child(self,i):
        """ 
        Assumes that af state is currently at this node,
        try the decision of the ith child, create the new DTNode
        for that, and shift af state to be that child.

        Returns true if successful.  On failure (topological violation?)
        return false, and state should be unchanged.
        """
        assert self.af.current==self
        
        site=self.options[self.child_order[i]]

        self.children[i] = DTChooseStrategy(af=self.af,parent=self,site=site)
        # nothing to update for posterior
        self.child_post[i] = self.child_prior[i]
        
        self.af.current=self.children[i]
        self.active_child=i
        return True
    
    def best_child(self,count=0,cb=None):
        """
        For choosing a site, prior is same as posterior
        """
        if count:
            count=min(count,len(self.options))
        else:
            count=len(self.options)

        best=None
        for i in range(count):
            print("best_child: trying %d / %d"%(i,count))
            if self.try_child(i):
                if cb: cb()
                # no need to go further
                return True
        return False
        
class DTChooseStrategy(DTNode):
    def __init__(self,af,parent,site):
        super(DTChooseStrategy,self).__init__(af=af,parent=parent)
        self.site=site

        self.af.resample_neighbors(site)
        self.cp=af.grid.checkpoint() 

        actions=site.actions()
        priors=[a.metric(site)
                for a in actions]
        self.set_options(actions,priors)

    def try_child(self,i):
        try:
            edits=self.options[self.child_order[i]].execute(self.site)
            self.af.optimize_edits(edits)
            # could commit?
        except self.af.cdt.IntersectingConstraints as exc:
            self.af.log.error("Intersecting constraints - rolling back")
            self.af.grid.revert(self.cp)
            return False
        except self.af.StrategyFailed as exc:
            self.af.log.error("Strategy failed - rolling back")
            self.af.grid.revert(self.cp)
            return False
        
        self.children[i] = DTChooseSite(af=self.af,parent=self)
        self.active_edits=edits # not sure this is the right place to store this
        self.af.current=self.children[i]

        nodes=[]
        for c in edits['cells']:
            nodes += list(self.af.grid.cell_to_nodes(c))
        for n in edits.get('nodes',[]):
            nodes.append(n)
        for j in edits.get('edges',[]):
            # needed in particular for nonlocal, where nothing
            # changes except the creation of an edge
            nodes += list(self.af.grid.edges['nodes'][j])
        nodes=list(set(nodes))
        assert len(nodes) # something had to change, right?
        cost = np.max( [self.af.eval_cost(n)
                        for n in nodes] )
        self.child_post[i]=cost
        return True
