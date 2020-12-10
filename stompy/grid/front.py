"""
An advancing front grid generator for use with unstructured_grid

Largely a port of paver.py.

"""
from __future__ import print_function
import math
import numpy as np
from collections import defaultdict

import time
from scipy import optimize as opt
import pdb
import logging
log=logging.getLogger(__name__)

from shapely import geometry

from . import (unstructured_grid,
               exact_delaunay,
               shadow_cdt)

from .. import utils


try:
    import matplotlib.pyplot as plt
except ImportError:
    log.warning("Plotting not available - no matplotlib")
    plt=None

def circumcenter_py(p1,p2,p3):
    """ Compute circumcenter of a single triangle using pure python.
    For small input sizes, this is much faster than using the vectorized
    numpy version in utils.
    """
    ref = p1
    
    p1x = 0
    p1y = 0
    p2x = p2[0] - ref[0]
    p2y = p2[1] - ref[1]
    p3x = p3[0] - ref[0]
    p3y = p3[1] - ref[1]

    # taken from TRANSFORMER_gang.f90
    dd=2.0*((p1x-p2x)*(p1y-p3y) -(p1x-p3x)*(p1y-p2y))
    b_com=p1x*p1x+p1y*p1y
    b1=b_com-p2x*p2x-p2y*p2y
    b2=b_com-p3x*p3x-p3y*p3y 

    # avoid division by zero is the points are collinear
    dd=max(dd,1e-40)
    return [ (b1*(p1y-p3y)-b2*(p1y-p2y))/dd + ref[0] ,
             (b2*(p1x-p2x)-b1*(p1x-p3x))/dd + ref[1] ]


# from numba import jit, int32, float64
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
        min_ab=ab_lens.min() # min(ab_lens)
        min_ca=ca_lens.min() # min(ca_lens)
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

    tl2=target_length**2
    # min_len can be 0.0, so soften undershoot
    undershoot = tl2 / (min_len + 0.01*tl2) 
    overshoot  = max_len / tl2

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
            
            assert not np.any( f>self.distances[-1] ),"Curve: Range or round off problem"
            idxs=idxs.clip(0,len(self.distances)-2) # to be sure equality doesn't push us off the end

            alphas = (f - self.distances[idxs]) / (self.distances[idxs+1]-self.distances[idxs])
            if not np.isscalar(alphas):
                alphas = alphas[:,None]
            return (1-alphas)*self.points[idxs] + alphas*self.points[idxs+1]
        else:
            assert False
    def tangent(self,f,metric='distance'):
        assert metric=='distance'
        if self.closed:
            # wraps around
            # double mod in case f==-eps
            f=(f % self.distances[-1]) % self.distances[-1]
        # side='right' ensures that f=0 works
        # it's unfortunately possible to get f=-eps, which rounds in
        # a way such that (f % distances[-1]) == distances[-1]
        # the double mod above might solve that
        idxs=np.searchsorted(self.distances,f,side='right') - 1
        assert not np.any( f>self.distances[-1] ),"Curve: Range or round off problem"
        idxs=idxs.clip(0,len(self.distances)-2) # to be sure equality doesn't push us off the end

        tng=utils.to_unit( self.points[idxs+1] - self.points[idxs] )
        return tng
    
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
            # as written, this may bail out of the iteration with an
            # inferior solution (i.e. stop when the error is 5%, rather
            # than go to the next segment where we could get an exact
            # answer).  It's not too bad though.
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

        # explicitly check the far end point
        if abs(distb-abs_dist) / abs_dist < rtol:
            # good enough
            result=far_f,self(far_f)
        else:
            # if there are large disparities in adjacent edge lengths
            # it's possible that it takes many iterations here.
            for maxit in range(20):
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

    def point_to_f(self,x,f_start=0,direction=1,rel_tol=1e-4):
        """
        Return the ring_f which yields a point close to x.
        This scans the points in the curve, starting with f_start
        and proceeding in the given direction.

        if direction is 0, both directions will be attempted and
        the first valid result returned.

        rel_tol: stop when a point is found within rel_tol*len(segment)
        of a segment.

        This is intended for finding f for a point that is already
        approximately on the curve. So it's a greedy approach.

        To project a point onto the curve, specify rel_tol='best'
        """
        # Walk along the curve, looking for a segment which approximately
        # contains x.

        if rel_tol=='best':
            # Do a full sweep, check all segments.
            # Could be smarter, but for now this isn't performance critical
            segs=np.stack( [self.points[:-1,:],
                            self.points[1:,:]], axis=1)
            dists,alphas = utils.point_segments_distance(x,segs,return_alpha=True)
            best=np.argmin(dists)
            seg_len=utils.dist( segs[best,0], segs[best,1] )
            new_f=self.distances[best] + alphas[best]*seg_len
            return new_f
        else:
            # Have to be careful about exact matches.  distances[i] should always
            # yield idx_start=i.
            # But anything in between depends on the direction
            if direction==1:
                idx_start=np.searchsorted(self.distances,f_start,side='right') - 1
            elif direction==-1:
                idx_start=np.searchsorted(self.distances,f_start,side='left')
            elif direction==0:
                # try either, accept any hit.
                try: 
                    return self.point_to_f(x,f_start=f_start,direction=1,rel_tol=rel_tol)
                except self.CurveException:
                    return self.point_to_f(x,f_start=f_start,direction=-1,rel_tol=rel_tol)
            else:
                raise Exception("direction must be +-1")

        # Start traversing the segments:
        seg_idx_a=idx_start

        best=None

        # closed loops have a point duplicated, and open strings
        # have one less segment than points
        # Either way, -1.
        Nseg=len(self.points)-1
            
        for i in range(Nseg): # max possible traversal
            if self.closed:
                seg_idx_b=(seg_idx_a + direction)%Nseg
            else:
                seg_idx_b=seg_idx_a+direction
                # Wrapping
                if seg_idx_b<0:
                    break
                if seg_idx_b>Nseg: # same as >=len(self.points)
                    break

            seg=self.points[ [seg_idx_a,seg_idx_b] ]
            seg_len=utils.dist(seg[0],seg[1])
            dist,alpha = utils.point_segment_distance(x,seg,return_alpha=True)

            if rel_tol=='best':
                if (best is None) or (dist<best[0]):
                    new_f=self.distances[seg_idx_a] + direction*alpha*seg_len
                    best=[dist,new_f,seg_idx_a,seg_idx_b]
            else:
                if dist/seg_len < rel_tol: 
                    # How to get to an f from this?
                    new_f=self.distances[seg_idx_a] + direction*alpha*seg_len
                    if not self.closed:
                        new_f=max(0,min(new_f,self.distances[-1]))
                    return new_f

            seg_idx_a=seg_idx_b

        if rel_tol=='best':
            return best[1]

        raise self.CurveException("Failed to find a point within tolerance")

    def is_forward(self,fa,fb,fc):
        """ return true if fa,fb and fc are distinct and
        ordered CCW around the curve
        """
        if fa==fb or fb==fc or fc==fa:
            return False
        if self.closed:
            d=self.total_distance()
            return ((fb-fa) % d) < ((fc-fa)%d)
        else:
            return fa<fb<fc
            # return ( (fb-fa) < (fc-fa) )
    def is_reverse(self,fa,fb,fc):
        # for closed curves, is_reverse=not is_forward, but
        # for open curves, that's not necessarily true.
        # when degenerate situations are included, then they
        # are not opposites even for closed curves.
        if fa==fb or fb==fc or fc==fa:
            return False
        if self.closed:
            d=self.total_distance()
            return ((fb-fc) % d) < ((fa-fc)%d)
        else:
            # return (fa-fb) < (fa-fc)
            return fc<fb<fa

    def is_ordered(self,fa,fb,fc):
        """
        Non-robust check for fb falling between fc.  For a closed
        curve, this resorts to the heuristic of whether fb falls
        between fa and fc on the shorter way around.
        """
        if self.closed:
            tdist=self.total_distance()
            if (fa-fc) % tdist < tdist/2:
                if self.is_forward(fc,fb,fa):
                    return True
            else:
                if self.is_forward(fa,fb,fc):
                    return True
            return False
        else:
            return (fa<fb<fc) or (fa>fb>fc)
        
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
    def new_point(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        b,c = grid.nodes['x'][ [nb,nc] ]
        bc=c-b
        return b + utils.rot(np.pi/3,bc)
    def execute(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        
        new_x = self.new_point(site)
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

class WallCloseStrategy(WallStrategy):
    """
    Wall, but with a very close-in initial guess point
    """
    def metric(self,site):
        # always try regular Wall first.
        return 0.5+super(WallCloseStrategy,self).metric(site)
        
    def new_point(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        b,c = grid.nodes['x'][ [nb,nc] ]
        bc=c-b
        usual_x=b + utils.rot(np.pi/3,bc)
        midpoint=0.5*(b+c)
        alpha=0.95
        return alpha*midpoint + (1-alpha)*usual_x
    
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
    
class ResampleStrategy(Strategy):
    """ TESTING: resample one step beyond.
    """
    def __str__(self):
        return "<Resample>"
    def nodes_beyond(self,site):
        he=site.grid.nodes_to_halfedge(site.abc[0],site.abc[1])
        pre_a=he.rev().node_rev()
        post_c=he.fwd().fwd().node_fwd()
        return pre_a,post_c

    def distances(self,site):
        "return pair of distances from the site to next node"
        
        pre_a,post_c = self.nodes_beyond(site)

        p_pa,p_a,p_c,p_pc=site.grid.nodes['x'][ [pre_a,
                                                 site.abc[0],
                                                 site.abc[2],
                                                 post_c] ]
        dists=[utils.dist( p_pa - p_a ),
               utils.dist( p_c - p_pc )]
        return dists
        
    def metric(self,site):
        dists=self.distances(site)
        # return a good low score when those distances are short relative
        # scale
        scale=site.local_length

        return min( dists[0]/scale,dists[1]/scale )

    def execute(self,site):
        grid=site.grid
        scale=site.local_length

        metric0=self.metric(site)
        
        def maybe_resample(n,anchor,direction):
            if n in site.abc:
                # went too far around!  Bad!
                return n

            # Is this overly restrictive?  What if the edge is nice
            # and long, and just wants a node in the middle?
            # That should be allowed, until there is some way of annotating
            # edges as rigid.
            # But at the moment that breaks things.
            # it shouldn't though.  And the checks here duplicate checks in
            # af.resample().  So skip the test, and go for it.
            
            # if grid.nodes['fixed'][n] in [site.af.HINT,site.af.SLIDE]:
            try:
                n=site.af.resample(n=n,anchor=anchor,scale=scale,
                                   direction=direction)
            except Curve.CurveException as exc:
                pass
            return n
                
        # execute one side at a time, since it's possible for a
        # resample on one side to reach into the other side.
        he=site.grid.nodes_to_halfedge(site.abc[0],site.abc[1])

        pre_a=he.rev().node_rev()
        new_pre_a=maybe_resample(pre_a,site.abc[0],-1)
        post_c=he.fwd().fwd().node_fwd()
        new_post_c=maybe_resample(post_c,site.abc[2],1)

        metric=self.metric(site)
        
        if metric>metric0:
            # while other nodes may have been modified, these are
            # the ones still remaining, and even these are probably of
            # no use for optimization.  may change this to report no
            # optimizable items
            return {'nodes':[new_pre_a,new_post_c]}
        else:
            log.warning("Resample made no improvement (%f => %f)"%(metric0,metric))
            raise StrategyFailed("Resample made no improvement")
            
class CutoffStrategy(Strategy):
    def __str__(self):
        return "<Cutoff>"
    def metric(self,site):
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Cutoff wants a small-ish internal angle
        # If the sites edges are long, scale_factor > 1
        # and we'd like to be making smaller edges, so ideal angle gets smaller

        # this used to be a comparison to 89, but that is too strict.
        # there could be an obtuse angle that we'd like to Cutoff and then
        # optimize back to acute.
        if theta>179*np.pi/180:
            return np.inf # not allowed
        else:
            ideal=60 + (1-scale_factor)*30
            return np.abs(theta - ideal*np.pi/180.)
    def execute(self,site):
        grid=site.grid
        na,nb,nc=site.abc

        he_ab=grid.nodes_to_halfedge(na,nb)
        he_bc=grid.nodes_to_halfedge(nb,nc)
        # Special case detect final quad
        he_da=he_ab.rev()
        he_cd=he_bc.fwd()
        j_ab=he_ab.j
        j_bc=he_bc.j

        ret={'cells':[]}
        
        if he_da.node_rev()==he_cd.node_fwd():
            # Quad handling:
            nd=he_cd.node_fwd()
            abcd=[na,nb,nc,nd]
            x=grid.nodes['x'][abcd]
            delta_x=np.roll(x,-1,axis=0) - x
            seg_theta=np.arctan2(delta_x[:,1],delta_x[:,0])
            internal_angles=((np.pi - ((np.roll(seg_theta,-1) - seg_theta))) % (2*np.pi))
            # first of these is internal angle of abc, and should be the smallest (based on
            # how sites are chosen).
            cutoff_bd=internal_angles[0]+internal_angles[2]
            cutoff_ac=internal_angles[1]+internal_angles[3]

            if cutoff_bd>cutoff_ac:
                # angles at b and d are larger, so should add the edge b--d
                j_bd=grid.add_edge(nodes=[nb,nd],cells=[grid.UNMESHED,grid.UNMESHED])
                c_abd=site.grid.add_cell(nodes=[na,nb,nd],
                                         edges=[j_ab,j_bd,he_da.j])
                c_bcd=site.grid.add_cell(nodes=[nb,nc,nd],
                                         edges=[j_bc,he_cd.j,j_bd])
                ret['cells'].extend( [c_abd,c_bcd] )
            else:
                j_ca=grid.add_edge(nodes=[nc,na],cells=[grid.UNMESHED,grid.UNMESHED])

                c_abc=site.grid.add_cell(nodes=site.abc,
                                         edges=[j_ab,j_bc,j_ca])
                c_cda=site.grid.add_cell(nodes=[nc,nd,na],
                                         edges=[he_cd.j,he_da.j,j_ca])
                ret['cells'].extend([c_abc,c_cda])
        else:
            # non-quad handling:
            nd=None
        
            j_ca=grid.nodes_to_edge(nc,na)
            if j_ca is None:
                # typical, but if we're finishing off the last triangle, this edge
                # exists.
                j_ca=grid.add_edge(nodes=[nc,na],cells=[grid.UNMESHED,grid.UNMESHED])

            c=site.grid.add_cell(nodes=site.abc,
                                 edges=[j_ab,j_bc,j_ca])
            ret['cells'].append(c)
        return ret

class SplitQuadStrategy(Strategy):
    """
    When the remaining node string has 4 nodes, often splitting this
    into two triangles and calling it done is the thing to do.
    """
    def __str__(self):
        return "<SplitQuad>"
    def metric(self,site):
        he_ab=site.grid.nodes_to_halfedge(site.abc[0],site.abc[1])
        he_da=he_ab.rev()
        he_cd=he_da.rev()
        c_maybe=he_cd.node_rev()
        d=he_cd.node_fwd()
        
        if c_maybe!=site.abc[2]:
            return np.inf # not a quad
        # Otherwise see if the scale is close:
        L=utils.dist_along( site.grid.nodes['x'][ site.abc + [d] + [site.abc[0]]] )[-1]
        scale_factor = L / (4*site.local_length)
        # if scale_factor<1, definitely want to try this.
        # if scale_factor>2, probably not.
        return 0.05 * 500**(scale_factor-1)


class JoinStrategy(Strategy):
    """ 
    Given an inside angle, merge the two edges.
    """
    def __str__(self):
        return "<Join>"
    
    def metric(self,site):
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Cutoff wants a small-ish internal angle
        # If the sites edges are long, scale_factor > 1
        # and we'd like to be making smaller edges, so ideal angle gets smaller
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
        j_ac_oring=0
        
        if j_ac is not None:
            # special case: nodes are already joined, but there is no
            # cell.
            # this *could* be extended to allow the deletion of thin cells,
            # but I don't want to get into that yet (since it's modification,
            # not creation)
            if (grid.edges['cells'][j_ac,0] >=0) or (grid.edges['cells'][j_ac,1]>=0):
                raise StrategyFailed("Edge already has real cells")
            # remember for tests below:
            j_ac_oring=grid.edges['oring'][j_ac]
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
            if ring!=grid.nodes['oring'][nc]: # this can maybe get relaxed to join onto a fixed node on multiple rings
                raise StrategyFailed("Cannot join across rings")
            if grid.nodes['oring'][nb]==ring:
                # This original check is too lenient.  in a narrow
                # channel, it's possible to have the three nodes
                # on the same ring, straddling the channel, and this
                # may allow for a join across the channel.
                
                #   # this is a problem if nb falls in between them.
                #   fa,fb,fc=grid.nodes['ring_f'][ [na,nb,nc] ]
                #   curve=site.af.curves[ring-1]
                #   
                #   if curve.is_ordered(fa,fb,fc):
                #       raise StrategyFailed("Cannot join across middle node")
                
                # instead, check for an edge between a and c.
                if j_ac_oring!=ring:
                    raise StrategyFailed("Cannot join non-adjacent along ring")
                
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
                    # fairly sure there are tests above which prevent
                    # this from having to populate additional fields, but
                    # not positive. 2018-02-26: need to think about oring.
                    jnew=grid.add_edge( nodes=nodes, cells=cells,
                                        oring=data['oring'],ring_sign=data['ring_sign'],
                                        fixed=data['fixed'] )
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
        each_dt_nbrs=[af.cdt.delaunay_neighbors(n) for n in site.abc]
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
        # as much as it would be nice to blindly execute these
        # things, the current state of the cost functions means
        # that a really bad nonlocal may not show up in the cost
        # function, and that means that best_child() will get tricked
        # So until there is a better cost function, this needs to
        # be more careful about which edges it will attempt
        if self.metric(site) > 0.75:
            raise StrategyFailed("NonLocal: too far away")
        
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
WallClose=WallCloseStrategy()
Cutoff=CutoffStrategy()
Join=JoinStrategy()
Bisect=BisectStrategy()
NonLocal=NonLocalStrategy()
Resample=ResampleStrategy()

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
        return [Wall,WallClose,Cutoff,Join,Bisect,NonLocal,Resample]

    def resample_neighbors(self):
        """ may update site! used to be part of AdvancingFront, but
        probably better here, as part of the site.
        """
        a,b,c = self.abc
        # local_length = self.af.scale( self.points().mean(axis=0) )
        # Possible that the site has very long edges ab or bc.
        # averaging the position can give a point far from the actual
        # site of the action which is b.
        # This is safer:
        local_length = self.af.scale( self.points()[1] )
        
        grid=self.af.grid
        self.resample_status=True

        if self.grid.nodes['fixed'][b] == self.af.HINT:
            self.grid.modify_node(b,fixed=self.af.SLIDE)

        for n,direction in [ (a,-1),
                             (c,1) ]:
            # used to check for SLIDE and degree
            # not sure whether we should let SLIDE through...
            # probably want to relax this to allow for subdividing
            # long edges if the edge itself is not RIGID.  But
            # we still avoid FREE nodes, since they are not on the boundary
            # and cannot be resampled
            if grid.nodes['fixed'][n] in [self.af.HINT,self.af.SLIDE,self.af.RIGID]:
                try:
                    n_res=self.af.resample(n=n,anchor=b,scale=local_length,direction=direction)
                except Curve.CurveException as exc:
                    self.resample_status=False
                    n_res=n

                if n!=n_res:
                    log.info("resample_neighbors changed a node")
                    if n==a:
                        self.abc[0]=n_res
                    else:
                        self.abc[2]=n_res
                    n=n_res # so that modify_node below operates on the right one.
                
                # is this the right time to change the fixed status?
                if grid.nodes['fixed'][n] == self.af.HINT:
                    grid.modify_node(n,fixed=self.af.SLIDE)
                
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
        print("call to QuadSite: resample_neighbors, %d %d %d %d"%(a,b,c,d))
        
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
                if self.grid.nodes['fixed'][n_res] == self.af.HINT:
                    self.grid.modify_node(n_res,fixed=self.af.SLIDE)
                
                if n!=n_res:
                    log.info("resample_neighbors changed a node")
                    if n==a:
                        self.abcd[0]=n_res
                    else:
                        self.abcd[3]=n_res
        return self.resample_status

        
class AdvancingFront(object):
    """
    Implementation of advancing front
    """
    grid=None
    cdt=None

    # 'fixed' flags:
    #  in order of increasing degrees of freedom in its location.
    # don't use 0 here, so that it's easier to detect uninitialized values
    UNSET=0
    RIGID=1 # should not be moved at all
    SLIDE=2 # able to slide along a ring.  
    FREE=3  # not constrained
    HINT=4  # slidable and can be removed.

    StrategyFailed=StrategyFailed
    
    def __init__(self,grid=None,**kw):
        """
        """
        self.log = logging.getLogger("AdvancingFront")
        utils.set_keywords(self,kw)
        
        if grid is None:
            grid=unstructured_grid.UnstructuredGrid()
        self.grid = self.instrument_grid(grid)

        self.curves=[]

    def add_curve(self,curve=None,interior=None,nodes=None,closed=True):
        """
        Add a Curve, upon which nodes can be slid.

        curve: [N,2] array of point locations, or a Curve instance.
        interior: true to force this curve to be an island.

        nodes: use existing nodes, given by the indices here.

        Any node which is already part of another ring will be set to RIGID,
        but will retain its original oring.

        The nodes must have existing edges connecting them, and those edges
        will be assigned to this ring via edges['oring'] and ['ring_sign']
        """
        if nodes is not None:
            nodes=np.asarray(nodes)
            curve=self.grid.nodes['x'][nodes]

        if not isinstance(curve,Curve):
            if interior is not None:
                ccw=not interior
            else:
                ccw=None
            curve=Curve(curve,ccw=ccw,closed=closed)
        elif interior is not None:
            assert curve.closed
            a=curve.signed_area()
            if a>0 and interior:
                curve=curve.reverse()

        self.curves.append( curve )
        oring=len(self.curves) # 1-based
        if nodes is not None:
            # Update nodes to be on this curve:
            on_a_ring=self.grid.nodes['oring'][nodes]>0

            self.grid.nodes['oring'][nodes[~on_a_ring]]=oring
            # curve.distances has an extra entry when a closed loop
            self.grid.nodes['ring_f'][nodes[~on_a_ring]]=curve.distances[:len(nodes)][~on_a_ring]
            self.grid.nodes['fixed'][nodes[~on_a_ring]]=self.HINT
            self.grid.nodes['fixed'][nodes[on_a_ring]]=self.RIGID

            # And update the edges, too:
            if closed:
                pairs=utils.circular_pairs(nodes)
            else:
                pairs=zip(nodes[:-1],nodes[1:])
            for a,b in pairs:
                j=self.grid.nodes_to_edge([a,b])
                self.grid.edges['oring'][j]=oring
                if self.grid.edges['nodes'][j,0]==a:
                    self.grid.edges['ring_sign'][j]=1
                elif self.grid.edges['nodes'][j,0]==b: # little sanity check
                    self.grid.edges['ring_sign'][j]=-1
                else:
                    assert False,"Failed invariant"

        return oring-1

    def instrument_grid(self,g):
        """
        Add fields to the given grid to support advancing front
        algorithm.  Modifies grid in place, and returns it.

        Also creates a Triangulation which follows modifications to 
        the grid, keeping a constrained Delaunay triangulation around.
        """
        # oring is stored 1-based, so that the default 0 value is
        # indicates no data / missing.
        g.add_node_field('oring',np.zeros(g.Nnodes(),'i4'),on_exists='pass')
        g.add_node_field('fixed',np.zeros(g.Nnodes(),'i1'),on_exists='pass')
        g.add_node_field('ring_f',-1*np.ones(g.Nnodes(),'f8'),on_exists='pass')

        # track a fixed field on edges, too, as it is not always sufficient
        # to tag nodes as fixed, since a long edge between two fixed nodes may
        # or may not be subdividable.  Note that for edges, we are talking about
        # topology, not the locations, since locations are part of nodes.
        # for starters, support RIGID (cannot subdivide) and 0, meaning no
        # additional information beyond existing node and topological constraints.
        g.add_edge_field('fixed',np.zeros(g.Nedges(),'i1'),on_exists='pass')
        # if nonzero, which curve this edge follows
        g.add_edge_field('oring',np.zeros(g.Nedges(),'i4'),on_exists='pass')
        # if oring nonzero, then +1 if n1=>n2 is forward on the curve, -1 
        # otherwise
        g.add_edge_field('ring_sign',np.zeros(g.Nedges(),'i1'),on_exists='pass')

        # Subscribe to operations *before* they happen, so that the constrained
        # DT can signal that an invariant would be broken
        self.cdt=self.shadow_cdt_factory(g)

        return g

    def shadow_cdt_factory(self,g):
        """ 
        Create a shadow CDT for the given grid.
        This extra level of indirection is to facilitate
        testing of one method vs the other in subclasses.
        """
        return shadow_cdt.shadow_cdt_factory(g)

    def initialize_boundaries(self,upsample=True):
        """
        Add nodes and edges to the the grid from curves.
        if upsample is True, resample curves at scale.
        """
        for curve_i,curve in enumerate(self.curves):
            # this is problematic when the goal is to have an
            # entirely rigid set of nodes.
            if upsample:
                curve_points,srcs=curve.upsample(self.scale,return_sources=True)
            else:
                if curve.closed:
                    # avoid repeated point
                    curve_points=curve.points[:-1]
                else:
                    curve_points=curve.points
                srcs=curve.distances[:len(curve_points)]

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
                                           self.grid.UNDEFINED],
                                    oring=curve_i+1,
                                    ring_sign=1 )

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

        TODO: this reports along edge distances, but it's
        used (exclusively?) in walking along boundaries which
        might be resampled.  It would be better to look at
        the distance in discrete jumps.

        """
        span=0.0
        if direction==1:
            trav0=he.node_fwd()
            anchor=he.node_rev()
        else:
            trav0=he.node_rev()
            anchor=he.node_fwd()
        last=anchor
        trav=trav0

        nodes=[last] # anchor is included

        def pred(n):
            # N.B. possible for trav0 to be SLIDE
            degree=self.grid.node_degree(n)

            # 2020-11-28: there used to be a blanket exception for trav0,
            # but it's only in the case that trav0 is SLIDE that we want
            # to return True for it.
            if degree>2:
                return False
            if n==trav0 and self.grid.nodes['fixed'][n]==self.SLIDE:
                return True
            if self.grid.nodes['fixed'][n]==self.HINT:
                return True
            return False

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

        If n has more than 2 neighbors, does nothing and returns n as is.
        Used to assume that n was SLIDE or HINT.  Now checks for either
        nodes['fixed'][n] in (SLIDE,HINT), or that the edge can be subdivided.

        normally, a SLIDE node cannot be deleted.  in some cases resample will
        create a new node for n, and it will be a SLIDE node.  in that case, should
        n retain SLIDE, too? is it the responsibility of resample(), or the caller?
        can we at least guarantee that no other nodes need to be changing status?

        in the past, new nodes created here were given fixed=SLIDE.  This is
        probably better set to HINT, as the SLIDE nodes can get in the way if
        they aren't used immediately for a cell.

        Returns the resampled node index -- often same as n, but may be a different
        node.
        """
        #self.log.debug("resample %d to be %g away from %d in the %s direction"%(n,scale,anchor,
        #                                                                        direction) )
        if direction==1: # anchor to n is t
            he=self.grid.nodes_to_halfedge(anchor,n)
        elif direction==-1:
            he=self.grid.nodes_to_halfedge(n,anchor)
        else:
            assert False

        n_deg=self.grid.node_degree(n)

        if self.grid.nodes['oring'][n]==0:
            self.log.debug("Node is not on a ring, no resampling possible")
            return n
        # must be able to either muck with n, or split the anchor-n edge
        # in the past we assumed that this sort of check was already done
        j=he.j
        edge_resamplable=( (self.grid.edges['fixed'][j]!=self.RIGID)
                           and (self.grid.edges['cells'][j,0]<0)
                           and (self.grid.edges['cells'][j,1]<0) )

        # node_resamplable=(n_deg==2) and (self.grid.nodes['fixed'][n] in [self.HINT,self.SLIDE])

        # it's possible to have a node that, based on the above test, is resamplable,
        # but the edge is not (because the edge test includes the possibility of
        # a cell on the opposite side).  
        #if not (node_resamplable or edge_resamplable):
        if not edge_resamplable:
            self.log.debug("Edge and node are RIGID/deg!=2, no resampling possible")
            return n

        span_length,span_nodes = self.free_span(he,self.max_span_factor*scale,direction)
        # anchor-n distance should be in there, already.

        # self.log.debug("free span from the anchor is %g"%span_length)

        if span_length < self.max_span_factor*scale:
            n_segments = max(1,round(span_length / scale))
            target_span = span_length / n_segments
        else:
            target_span=scale
            n_segments = None

        def handle_one_segment():
            # this is a function because there are two times
            # (one proactive, one reactive) it might get used below.
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
                # self.log.info("n_segment=1, but that would be an implicit join")

                # rather than force two segments, force it
                # to remove all but the last edge.
                del span_nodes[-1]

            # self.log.debug("Only space for 1 segment")
            for d in span_nodes[1:-1]:
                cp=self.grid.checkpoint()
                try:
                    self.grid.merge_edges(node=d)
                except self.cdt.IntersectingConstraints as exc:
                    self.log.info("handle_one_segment: cut short by exception")
                    self.grid.revert(cp)
                    # only got that far..
                    return d
            return span_nodes[-1]

        if n_segments==1:
            return handle_one_segment()

        # first, find a point on the original ring which satisfies the target_span
        anchor_oring=self.grid.nodes['oring'][anchor]-1
        n_oring=self.grid.nodes['oring'][n]-1
        oring=self.grid.edges['oring'][j]-1 

        # Default, may be overwritten below
        anchor_f = self.grid.nodes['ring_f'][anchor]
        n_f = self.grid.nodes['ring_f'][n]
        
        if anchor_oring != oring:
            self.log.warning('resample: anchor on different rings.  Cautiously resample')
            if n_oring==oring:
                f_start=n_f # can use n to speed up point_to_f
            else:
                f_start=0.0 # not valid, so full search in point_to_f

            anchor_f = self.curves[oring].point_to_f(self.grid.nodes['x'][anchor],
                                                     n_f,
                                                     direction=0)

        if n_oring != oring:
            # anchor_f is valid regardless of its original oring
            n_f = self.curves[oring].point_to_f(self.grid.nodes['x'][n],
                                                anchor_f,
                                                direction=0) 

        # Easing into use of explicit edge orings
        assert oring==self.grid.edges['oring'][j]-1
        curve = self.curves[oring]

        # at any point might encounter a node from a different ring, but want
        # to know it's ring_f for this ring.
        def node_f(m):
            # first two cases are partially to be sure that equality comparisons will
            # work.
            if m==n:
                return n_f
            elif m==anchor:
                return anchor_f
            elif self.grid.nodes['oring'][m]==oring+1:
                return self.grid.nodes['ring_f'][m]
            else:
                return curve.point_to_f(self.grid.nodes['x'][m],
                                        n_f,direction=0)

        if 0: # delete this once the new stanza below is trusted
            # explicitly record whether the curve has the opposite orientation
            # of the edge.  Hoping to retire this way.
            # This is actually dangerous, as the mid_point does not generally
            # fall on the line, and so we have to give it a generous rel_tol.
            mid_point = 0.5*(self.grid.nodes['x'][n] + self.grid.nodes['x'][anchor])
            mid_f=self.curves[oring].point_to_f(mid_point)

            if curve.is_forward(anchor_f,mid_f,n_f):
                curve_direction=1
            else:
                curve_direction=-1
        if 1: # "new" way
            # logic is confusing
            edge_ring_sign=self.grid.edges['ring_sign'][he.j]
            curve_direction=(1-2*he.orient)*direction*edge_ring_sign
            #assert new_curve_direction==curve_direction
            
            assert edge_ring_sign!=0,"Edge %d has sign %d, should be set"%(he.j,edge_ring_sign)

        # a curve forward that bakes in curve_direction
        if curve_direction==1:
            rel_curve_fwd=lambda a,b,c: curve.is_forward(a,b,c)
        else:
            rel_curve_fwd=lambda a,b,c: curve.is_reverse(a,b,c)

        try:
            new_f,new_x = curve.distance_away(anchor_f,curve_direction*target_span)
        except Curve.CurveException as exc:
            raise

        # it's possible that even though the free_span distance yielded
        # n_segments>1, distance_away() went too far since it cuts out some
        # curvature in the along-curve distance.
        # this leads to a liability that new_f is beyond span_nodes[-1], and
        # we should follow the same treatment as above for n_segments==1
        end_span_f=node_f(span_nodes[-1])

        # 2018-02-13: hoping this also changes to curve_direction
        if ( rel_curve_fwd(anchor_f,end_span_f,new_f)
             and end_span_f!=anchor_f):
            self.log.warning("n_segments=%s, but distance_away blew past it"%n_segments)
            return handle_one_segment()
            
        # check to see if there are other nodes in the way, and remove them.
        # in the past, this started with the node after n, deleting things up
        # to, and *including* a node at the location where we want n to be.
        # in simple cases, it would be better to delete n, and only move the
        # last node.  But there is a chance that n cannot be deleted, more likely
        # that n cannot be deleted than later nodes.  However... free_span
        # would not allow those edges, so we can assume anything goes here.
        eps=0.001*target_span

        nodes_to_delete=[]
        trav=he
        while True:
            # start with the half-edge from anchor to n
            # want to loop until trav.node_fwd() (for direction=1)
            # is at or beyond our target, and all nodes from n
            # until trav.node_rev() are in the list nodes_to_delete.
            
            if direction==1:
                n_trav=trav.node_fwd()
            else:
                n_trav=trav.node_rev()
            f_trav=node_f(n_trav)

            # EPS needs some TLC here.  The corner cases have not been
            # sufficiently take care of, i.e. new_f==f_trav, etc.
            if rel_curve_fwd(anchor_f, new_f+curve_direction*eps, f_trav ):
                break

            # that half-edge wasn't far enough
            nodes_to_delete.append(n_trav)
                
            if direction==1:
                trav=trav.fwd()
            else:
                trav=trav.rev()
                
            # sanity check.
            if trav==he:
                self.log.error("Made it all the way around!")
                raise Exception("This is probably bad")

        # either n was already far enough, in which case we should split
        # this edge, or there are some nodes in nodes_to_delete.
        # the last of those nodes will be saved, and become the new n
        if len(nodes_to_delete):
            nnew=nodes_to_delete.pop()
            # slide, because it needs to move farther out
            method='slide'
        else:
            # because n is already too far
            method='split'
            nnew=n
            
        # Maybe better to fix the new node with any sliding necessary,
        # and then delete these, but that would require more checks to
        # see if it's safe to reposition the node?
        for d in nodes_to_delete:
            cp=self.grid.checkpoint()
            try:
                self.grid.merge_edges(node=d)
            except self.cdt.IntersectingConstraints as exc:
                self.log.info("resample: had to stop short due to intersection")
                self.grid.revert(cp)
                return d

        # on the other hand, it may be that the next node is too far away, and it
        # would be better to divide the edge than to shift a node from far away.
        # also possible that our neighbor was RIGID and can't be shifted

        cp=self.grid.checkpoint()
        
        try:
            if method=='slide':
                self.grid.modify_node(nnew,x=new_x,ring_f=new_f)
                assert self.grid.nodes['oring'][nnew]==oring+1
            else: # 'split'
                j=self.grid.nodes_to_edge([anchor,nnew])

                # get a newer nnew
                # This used to set fixed=SLIDE, but since there is no additional
                # topology attached to nnew, it probably makes more sense for it
                # to be HINT. changed 2018-02-26
                jnew,nnew,j_next = self.grid.split_edge(j,x=new_x,ring_f=new_f,oring=oring+1,
                                                        fixed=self.HINT)
        except self.cdt.IntersectingConstraints as exc:
            self.log.info("resample - slide() failed. will return node at original loc")
            self.grid.revert(cp)
            
        return nnew

    def resample_neighbors(self,site):
        return site.resample_neighbors()

    def resample_cycles(self):
        """
        Resample all edges along cycles. Useful when the boundary has
        a mix of rigid and non-rigid, with coarse spacing that needs 
        to be resampled.
        """
        cycs=self.grid.find_cycles(max_cycle_len=self.grid.Nnodes())

        for cyc in cycs:
            n0=cyc[0]
            he=self.grid.nodes_to_halfedge(cyc[0],cyc[1])

            while 1:
                a=he.node_rev()
                b=he.node_fwd()

                res=self.resample(b,a,
                                  scale=self.scale(self.grid.nodes['x'][a]),
                                  direction=1)
                he=self.grid.nodes_to_halfedge(a,res).fwd()
                if he.node_rev()==n0:
                    break # full circle.

    def cost_function(self,n):
        raise Exception("Implement in subclass")

    def eval_cost(self,n):
        if self.grid.nodes['fixed'][n]==self.RIGID:
            return 0.0
        fn=self.cost_function(n)
        if fn:
            return fn(self.grid.nodes['x'][n])
        else:
            return 0.0

    cost_thresh_default=0.22
    def optimize_nodes(self,nodes,max_levels=4,cost_thresh=None):
        """
        iterate over the given set of nodes, optimizing each location,
        and possibly expanding the set of nodes in order to optimize
        a larger area.

        2019-03-12: max_levels used to default to 3, but there were
         cases where it needed a little more perseverance.
         cost_thresh defaults to 0.22, following the tuning of paver.py
        """
        if cost_thresh is None:
            cost_thresh=self.cost_thresh_default
            
        for level in range(max_levels):
            # following paver, maybe will decrease number of calls
            # didn't help.
            nodes.sort(reverse=True)
            
            max_cost=0
            for n in nodes:
                # relax_node can return 0 if there was no cost
                # function to optimize

                # this node may already be good enough
                initial_cost=self.eval_cost(n)
                if initial_cost<cost_thresh: continue
                new_cost=self.relax_node(n) or 0.0

                max_cost=max(max_cost,new_cost)
            if max_cost <= cost_thresh:
                break
            # as in paver -- if everybody is valid, good enough
            failures=self.check_edits(dict(nodes=nodes))
            if len(failures['cells'])==0:
                break
                
            if level==0:
                # just try re-optimizing once
                pass
            else:
                # expand list of nodes one level
                new_nodes=set(nodes)
                for n in nodes:
                    new_nodes.update(self.grid.node_to_nodes(n))
                nodes=list(new_nodes)

    def optimize_edits(self,edits,**kw):
        """
        Given a set of elements (which presumably have been modified
        and need tuning), jostle nodes around to improve the cost function

        Returns an updated edits with any additional changes.  No promise
        that it's the same object or a copy.
        """
        if 'nodes' not in edits:
            edits['nodes']=[]

        nodes = list(edits.get('nodes',[]))

        for c in edits.get('cells',[]):
            for n in self.grid.cell_to_nodes(c):
                if n not in nodes:
                    nodes.append(n)

        def track_node_edits(g,func_name,n,**k):
            if n not in edits['nodes']:
                edits['nodes'].append(n)

        self.grid.subscribe_after('modify_node',track_node_edits)
        self.optimize_nodes(nodes,**kw)
        self.grid.unsubscribe_after('modify_node',track_node_edits)
        return edits

    def relax_node(self,n):
        """ Move node n, subject to its constraints, to minimize
        the cost function.  Return the final value of the cost function
        """
        # self.log.debug("Relaxing node %d"%n)
        if self.grid.nodes['fixed'][n] == self.FREE:
            return self.relax_free_node(n)
        elif self.grid.nodes['fixed'][n] == self.SLIDE:
            return self.relax_slide_node(n)
        else:
            # Changed to silent pass because ResampleStrategy currently
            # tells the truth about nodes it moves, even though they
            # are HINT nodes.  
            # raise Exception("relax_node with fixed=%s"%self.grid.nodes['fixed'][n])
            return 0.0

    def relax_free_node(self,n):
        cost=self.cost_function(n)
        if cost is None:
            return None
        x0=self.grid.nodes['x'][n]
        local_length=self.scale( x0 )
        init_cost=cost(x0)
        new_x = opt.fmin(cost,
                         x0,
                         xtol=local_length*1e-4,
                         disp=0)
        opt_cost=cost(new_x)
        dx=utils.dist( new_x - x0 )

        if (dx != 0.0) and opt_cost<init_cost:
            # self.log.debug('Relaxation moved node %f'%dx)
            cp=self.grid.checkpoint()
            try:
                self.grid.modify_node(n,x=new_x)
                return opt_cost
            except self.cdt.IntersectingConstraints as exc:
                self.grid.revert(cp)
                self.log.info("Relaxation caused intersection, reverting")
        return init_cost
        
    def relax_slide_node(self,n):
        cost_free=self.cost_function(n)
        if cost_free is None:
            return 
        x0=self.grid.nodes['x'][n]
        f0=self.grid.nodes['ring_f'][n]
        ring=self.grid.nodes['oring'][n]-1

        assert np.isfinite(f0)
        assert ring>=0

        local_length=self.scale( x0 )

        slide_limits=self.find_slide_limits(n,3*local_length)

        # used to just be f, but I think it's more appropriate to
        # be f[0]
        def cost_slide(f):
            # lazy bounded optimization
            f=f[0]
            fclip=np.clip(f,*slide_limits)
            err=(f-fclip)**2
            return err+cost_free( self.curves[ring](fclip) )

        base_cost=cost_free(x0)

        new_f = opt.fmin(cost_slide,
                         [f0],
                         xtol=local_length*1e-4,
                         disp=0)

        if not self.curves[ring].is_forward(slide_limits[0],
                                            new_f,
                                            slide_limits[1]):
            # Would be better to just optimize within bounds.
            # still, can check the two bounds, and if the 
            # cost is lower, return one of them.
            self.log.warning("Slide went outside limits")

            slide_length=(slide_limits[1] - slide_limits[0])
            lower_f=0.95*slide_limits[0]+0.05*slide_limits[1]
            upper_f=0.05*slide_limits[0]+0.95*slide_limits[1]
            lower_cost=cost_slide([lower_f])
            upper_cost=cost_slide([upper_f])

            if lower_cost<upper_cost and lower_cost<base_cost:
                self.log.warning("Truncate slide on lower end")
                new_f=[lower_f]
            elif upper_cost<base_cost:
                new_f=[upper_f]
                self.log.warning("Truncate slide on upper end")
            else:
                self.log.warning("Couldn't truncate slide.")
                return base_cost
        new_cost=cost_slide(new_f)

        if new_cost<base_cost:
            cp=self.grid.checkpoint()
            try:
                self.slide_node(n,new_f[0]-f0)
                return new_cost
            except self.cdt.IntersectingConstraints as exc:
                self.grid.revert(cp)
                self.log.info("Relaxation caused intersection, reverting")
        return base_cost

    def node_ring_f(self,n,ring0):
        """
        return effective ring_f for node n in terms of ring0.
        if that's the native ring for n, just return ring_f,
        otherwise calculates where n would fall on ring0
        """
        if self.grid.nodes['oring'][n]-1==ring0:
            return self.grid.nodes['ring_f'][n]
        else:
            return self.curves[ring0].point_to_f(self.grid.nodes['x'][n])
        
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
            j=self.grid.nodes_to_edge([n,nbr])
            j_ring=self.grid.edges['oring'][j]
            if j_ring==0:
                continue
            assert j_ring-1==n_ring

            # The test below is not robust with intersecting curves,
            # and is why edges have to track their own ring.
            
            #if self.grid.nodes['oring'][nbr]-1!=n_ring:
            #    continue
            nbrs.append(nbr)
        # With the above check on edge oring, this should not be necessary.
        # if len(nbrs)>2:
        #     # annoying, but happens.  one or more edges are internal,
        #     # and two are along the curve.
        #     nbrs.append(n)
        #     # sort them along the ring - HERE this logic is likely not robust for open curves
        #     all_f=(self.grid.nodes['ring_f'][nbrs]-n_f) % L 
        #     order=np.argsort(all_f)
        #     nbrs=[ nbrs[order[-1]], nbrs[order[1]] ]
        assert len(nbrs)==2

        if curve.is_forward(self.node_ring_f(nbrs[0],n_ring),
                            n_f,
                            self.node_ring_f(nbrs[1],n_ring) ):
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
                     (sgn*(self.node_ring_f(trav[1],n_ring) - n_f) )%L > cutoff ):
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
            
        limits=[self.node_ring_f(m,n_ring)
                for m in stops]
        # make sure limits are monotonic increasing.  for circular,
        # this may require some modulo
        if curve.closed and (limits[0]>limits[1]):
            if limits[1] < n_f:
                limits[1] += curve.total_distance()
            elif limits[0] > n_f:
                limits[0] -= curve.total_distance()
            else:
                assert False,"Not sure how to get the range to enclose n"
                
        assert limits[0] < limits[1]
        return limits
    
    def find_slide_conflicts(self,n,delta_f):
        """ Find nodes in the way of sliding node n
        to a new ring_f=old_oring_f + delta_f.
        N.B. this does not appear to catch situations 
        where n falls exactly on an existing node, though
        it should (i.e. it's a bug)
        """
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

            nbr_f=self.node_ring_f(nbr,n_ring)
            if self.grid.node_degree(nbr)!=2:
                continue

            if delta_f>0:
                # either the nbr is outside our slide area, or could
                # be in the opposite direction along the ring
                
                if not curve.is_forward(n_f,nbr_f,n_f+delta_f):
                    continue

                to_delete.append(nbr)
                he=self.grid.nodes_to_halfedge(n,nbr)
                while 1:
                    he=he.fwd()
                    nbr=he.node_fwd()
                    nbr_f=self.node_ring_f(nbr,n_ring)
                    if curve.is_forward(n_f,n_f+delta_f,nbr_f):
                        break
                    to_delete.append(nbr)
                break
            else:
                if not curve.is_reverse(n_f,nbr_f,n_f+delta_f):
                    continue
                to_delete.append(nbr)
                he=self.grid.nodes_to_halfedge(nbr,n)
                while 1:
                    he=he.rev()
                    nbr=he.node_rev()
                    nbr_f=self.node_ring_f(nbr,n_ring)
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

    loop_count=0
    def loop(self,count=0):
        while 1:
            site=self.choose_site()
            if site is None:
                break
            if not self.advance_at_site(site):
                self.log.error("Failed to advance. Exiting loop early")
                return False
            count-=1
            self.loop_count+=1
            if count==0:
                break
        return True

    def advance_at_site(self,site):
        # This can modify site! May also fail.
        resampled_success = self.resample_neighbors(site)

        actions=site.actions()
        metrics=[a.metric(site) for a in actions]
        bests=np.argsort(metrics)
        for best in bests:
            try:
                cp=self.grid.checkpoint()
                self.log.debug("Chose strategy %s"%( actions[best] ) )
                edits=actions[best].execute(site)
                opt_edits=self.optimize_edits(edits)

                failures=self.check_edits(opt_edits)
                if len(failures['cells'])>0:
                    self.log.info("Some cells failed")
                    raise StrategyFailed("Cell geometry violation")
                # could commit?
            except self.cdt.IntersectingConstraints as exc:
                # arguably, this should be caught lower down, and rethrown
                # as a StrategyFailed.
                self.log.error("Intersecting constraints - rolling back")
                self.grid.revert(cp)
                continue
            except StrategyFailed as exc:
                self.log.error("Strategy failed - rolling back")
                self.grid.revert(cp)
                continue
            break
        else:
            self.log.error("Exhausted the actions!")
            return False
        return True

    def check_edits(self,edits):
        return defaultdict(list)
        
    zoom=None
    def plot_summary(self,ax=None,
                     label_nodes=True,
                     clip=None):
        ax=ax or plt.gca()
        ax.cla()

        for curve in self.curves:
            curve.plot(ax=ax,color='0.5',lw=0.4,zorder=-5)

        self.grid.plot_edges(ax=ax,clip=clip,lw=1)
        if label_nodes:
            labeler=lambda ni,nr: str(ni)
        else:
            labeler=None
        self.grid.plot_nodes(ax=ax,labeler=labeler,clip=clip,sizes=10)
        ax.axis('equal')
        if self.zoom:
            ax.axis(self.zoom)


class AdvancingTriangles(AdvancingFront):
    """ 
    Specialization which roughly mimics tom, creating only triangles
    """
    scale=None
    def __init__(self,grid=None,scale=None,**kw):
        super(AdvancingTriangles,self).__init__(grid=grid,**kw)
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

    # reject edit that puts a cell circumcenter outside the cell
    reject_cc_outside_cell=True
    # If a numeric value, check distance between adjacent circumcenters
    # reject if signed distance below this value, normalize by sqrt(cell area)
    reject_cc_distance_factor=None
    def check_edits(self,edits):
        """
        edits: {'nodes':[n1,n2,...],
                'cells': ...,
                'edges': ... }
        Checks for any elements which fail geometric checks, such
        as orthogonality.
        """
        failures=defaultdict(list)
        cells=set( edits.get('cells',[]) )

        for n in edits.get('nodes',[]):
            cells.update( self.grid.node_to_cells(n) )

        for c in list(cells):
            pnts=self.grid.nodes['x'][self.grid.cell_to_nodes(c)]
            cc=circumcenter_py(pnts[0],pnts[1],pnts[2])
            if self.reject_cc_outside_cell:
                if not self.grid.cell_polygon(c).contains(geometry.Point(cc)):
                    failures['cells'].append(c)
            if self.reject_cc_distance_factor is not None:
                # More expensive but closer to what really matters
                for j in self.grid.cell_to_edges(c):
                    ec=self.grid.edges['cells'][j,:]
                    n=self.grid.edges_normals(j)
                    if ec[0]==c:
                        nbr=ec[1]
                    elif ec[1]==c:
                        nbr=ec[0]
                        n=-n
                    else: assert False
                    if nbr<0: continue

                    pnts=self.grid.nodes['x'][self.grid.cell_to_nodes(nbr)]
                    nbr_cc=circumcenter_py(pnts[0],pnts[1],pnts[2])

                    l_perp=(np.array(nbr_cc)-np.array(cc)).dot(n)
                    L=np.sqrt( self.grid.cells_area(sel=[c,nbr]).sum() )
                    if l_perp < self.reject_cc_distance_factor*L:
                        failures['cells'].append(c)
                        break
                    
        return failures

    # cc_py is more elegant and crappier
    cost_method='base'
    cost_thresh_default=0.22
    def cost_function(self,n):
        """
        Return a function which takes an x,y pair, and evaluates
        a geometric cost function for node n based on the shape and
        scale of triangle cells containing n
        """
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

        Alist=[ [ e[0],e[1] ]
                for e in edge_points[:,0,:] ]
        Blist=[ [ e[0],e[1] ]
                for e in edge_points[:,1,:] ]
        EPS=1e-5*local_length

        def cost_cc_and_scale_py(x0):
            C=list(x0)
            cc_cost=0
            scale_cost=0

            for A,B in zip(Alist,Blist):
                tri_cc=circumcenter_py(A,B,C)

                deltaAB=[ tri_cc[0] - A[0],
                          tri_cc[1] - A[1]]
                ABs=[B[0]-A[0],B[1]-A[1]]
                magABs=math.sqrt( ABs[0]*ABs[0] + ABs[1]*ABs[1])
                vecAB=[ABs[0]/magABs, ABs[1]/magABs]
                leftAB=vecAB[0]*deltaAB[1] - vecAB[1]*deltaAB[0] 

                deltaBC=[tri_cc[0] - B[0],
                         tri_cc[1] - B[1]]
                BCs=[C[0]-B[0], C[1]-B[1]]
                magBCs=math.sqrt( BCs[0]*BCs[0] + BCs[1]*BCs[1] )
                vecBC=[BCs[0]/magBCs, BCs[1]/magBCs]
                leftBC=vecBC[0]*deltaBC[1] - vecBC[1]*deltaBC[0]

                deltaCA=[tri_cc[0] - C[0],
                         tri_cc[1] - C[1]]
                CAs=[A[0]-C[0],A[1]-C[1]]
                magCAs=math.sqrt(CAs[0]*CAs[0] + CAs[1]*CAs[1])
                vecCA=[CAs[0]/magCAs, CAs[1]/magCAs]
                leftCA=vecCA[0]*deltaCA[1] - vecCA[1]*deltaCA[0]

                cc_fac=-4. # not bad
                # cc_fac=-2. # a little nicer shape
                # clip to 100, to avoid overflow in math.exp
                if 0:
                    # this can favor isosceles too much
                    this_cc_cost = ( math.exp(min(100,cc_fac*leftAB/local_length)) +
                                     math.exp(min(100,cc_fac*leftBC/local_length)) +
                                     math.exp(min(100,cc_fac*leftCA/local_length)) )
                else:
                    # maybe?
                    this_cc_cost = ( math.exp(min(100,cc_fac*leftAB/magABs)) +
                                     math.exp(min(100,cc_fac*leftBC/magBCs)) +
                                     math.exp(min(100,cc_fac*leftCA/magCAs)) )

                # mixture
                # 0.3: let's the scale vary too much between the cells
                #      adjacent to n
                alpha=1.0
                avg_length=alpha*local_length + (1-alpha)*(magABs+magBCs+magCAs)/3
                this_scale_cost=( (magABs-avg_length)**2 
                                  + (magBCs-avg_length)**2 
                                  + (magCAs-avg_length)**2 )
                this_scale_cost/=avg_length*avg_length

                cc_cost+=this_cc_cost
                scale_cost+=this_scale_cost

            # With even weighting between these, some edges are pushed long rather than
            # having nice angles.
            # 3 is a shot in the dark.
            # 50 is more effective at avoiding a non-orthogonal cell
            return 50*cc_cost+scale_cost

        if self.cost_method=='base':
            return cost
        elif self.cost_method=='cc_py':
            return cost_cc_and_scale_py
        else:
            assert False

##


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
                self.grid.nodes['fixed'][n]=self.HINT # self.SLIDE
            else:
                self.grid.nodes['fixed'][n]=self.RIGID

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

            # infer the fixed nature of the edge
            if self.grid.edges['cells'][j,1-side]>=0:
                self.grid.edges['fixed'][j]=self.RIGID
            # Add in the edge data to link it to this curve
            if self.grid.edges['oring'][j]==0:
                # only give it a ring if it is not already on a ring.
                # There may be reason to override this in the future, since the ring
                # information may be stale from an existing grid, and now we want
                # to regenerate it.
                self.grid.edges['oring'][j]=1+curve_idx
                # side=0 when the edge is going the same direction as the
                # ring, which in turn should be ring_sign=1.
                self.grid.edges['ring_sign'][j]=1-2*side 
            
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
        self.active_child=None # we don't manipulate this, but just signal that it's fresh
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
        """
        rewind to the state when we first encountered this node 
        """
        self.af.grid.revert(self.cp)
        self.af.current=self

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
        for c in edits.get('cells',[]):
            nodes += list(self.af.grid.cell_to_nodes(c))
        for n in edits.get('nodes',[]):
            nodes.append(n)
        for j in edits.get('edges',[]):
            # needed in particular for nonlocal, where nothing
            # changes except the creation of an edge
            nodes += list(self.af.grid.edges['nodes'][j])
        nodes=list(set(nodes))
        assert len(nodes) # something had to change, right?
        cost = np.max( [ (self.af.eval_cost(n) or 0.0)
                        for n in nodes] )
        self.child_post[i]=cost

        return True
