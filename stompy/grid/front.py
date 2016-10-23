"""

An advancing front grid generator for use with unstructured_grid

Largely a port of paver.py.

"""
import unstructured_grid
import exact_delaunay
import numpy as np
import utils
import logging
log=logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError:
    log.warning("Plotting not available - no matplotlib")
    plt=None


class Curve(object):
    """
    Boundaries which can be open or closed, indexable
    by a floating point value (including modulo arithmetic).
    By default, indexes by distance along each segment.
    """
    class CurveException(Exception):
        pass
    
    def __init__(self,points,closed=True):
        self.points=np.asarray(points)
        self.closed=closed
        if self.closed:
            self.points = np.concatenate( (self.points,
                                           self.points[:1,:] ) )
        
        self.distances=utils.dist_along(self.points)
    def __call__(self,f,metric='distance'):
        if metric=='distance':
            if self.closed:
                # wraps around
                f=f % self.distances[-1]
            # side='right' ensures that f=0 works
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

    def distance_away(self,anchor_f,signed_distance,rtol=0.05):
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

    def is_forward(self,fa,fb,fc):
        d=self.total_distance()
        return ((fb-fa) % d) < ((fc-fa)%d)
    def is_reverse(self,fa,fb,fc):
        return self.is_forward(fc,fb,fa)
    
    def plot(self,ax=None):
        ax=ax or plt.gca()
        return ax.plot(self.points[:,0],self.points[:,1])[0]
        

def internal_angle(A,B,C):
    BA=A-B
    BC=C-B
    theta_BA = np.arctan2( BA[1], BA[0] )
    theta_BC = np.arctan2( BC[1], BC[0] )
    return (theta_BA - theta_BC) % (2*np.pi)


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
        
class TriangleSite(object):
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
        theta=self.internal_angle()
        HERE


class ShadowCDT(exact_delaunay.Triangulation):
    """ Tracks modifications to an unstructured grid and
    maintains a shadow representation with a constrained Delaunay
    triangulation, which can be used for geometric queries and
    predicates.
    """
    def __init__(self,g):
        super(ShadowCDT,self).__init__(extra_node_fields=[('g_n','i4')])
        self.g=g
        
        self.nodemap_g_to_local={}

        g.subscribe_before('add_node',self.before_add_node)
        g.subscribe_after('add_node',self.after_add_node)
        g.subscribe_before('modify_node',self.before_modify_node)
        g.subscribe_before('delete_node',self.before_delete_node)
        
        g.subscribe_before('add_edge',self.before_add_edge)
        g.subscribe_before('delete_edge',self.before_delete_edge)
    def before_add_node(self,g,func_name,**k):
        pass # no checks quite yet
    def after_add_node(self,g,func_name,return_value,**k):
        n=return_value
        self.nodemap_g_to_local[n]=self.add_node(x=k['x'],g_n=n)
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
    def before_delete_edge(self,g,func_name,j,**k):
        nodes=g.edges['nodes'][j]
        self.remove_constraint(nodes[0],nodes[1])

        
class AdvancingFront(object):
    """
    Implementation of advancing front
    """
    scale=None
    grid=None
    cdt=None

    # 'fixed' flags:
    #  in order of increasing degrees of freedom in its location.
    # don't use 0 here, so that it's easier to detect uninitialized values
    RIGID=1 # should not be moved at all
    SLIDE=2 # able to slide along a ring
    FREE=3  # not constrained 
    
    def __init__(self,grid=None,scale=None):
        """
        """
        self.log = logging.getLogger("AdvancingFront")

        if grid is None:
            grid=unstructured_grid.UnstructuredGrid()
        self.grid = self.instrument_grid(grid)

        self.curves=[]
        
    def add_curve(self,curve):
        self.curves.append( curve )
        return len(self.curves)-1

    def set_edge_scale(self,scale):
        self.scale=scale
        
    def instrument_grid(self,g):
        """
        Add fields to the given grid to support advancing front
        algorithm.  Modifies grid in place, and returns it.

        Also creates a Triangulation which follows modifications to 
        the grid, keeping a constrained Delaunay triangulation around.
        """
        g.add_node_field('oring',-1*np.ones(g.Nnodes(),'i4'),on_exists='pass')
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
            nodes=[self.grid.add_node(x=curve_points[j],
                                      oring=curve_i,
                                      ring_f=srcs[j],
                                      fixed=self.SLIDE)
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

    def choose_site(self):
        sites=[]
        J,Orient = np.nonzero( (self.grid.edges['cells'][:,:]==self.grid.UNMESHED) )

        for j,orient in zip(J,Orient):
            he=self.grid.halfedge(j,orient)
            he_nxt=he.fwd()
            a=he.node_rev()
            b=he.node_fwd()
            bb=he_nxt.node_rev()
            c=he_nxt.node_fwd()
            assert b==bb

            sites.append( TriangleSite(self,nodes=[a,b,c]) )
        if len(sites):
            scores=[ site.metric()
                     for site in sites ]
            best=np.argmin( scores ) 
            return sites[best]
        else:
            return None
        
    def free_span(self,he,max_span,direction):
        span=0.0
        if direction==1:
            trav=he.node_fwd()
            last=anchor=he.node_rev()
        else:
            trav=he.node_rev()
            last=anchor=he.node_fwd()

        def pred(n):
            return ( (self.grid.nodes['fixed'][n]== self.SLIDE) and
                     len(self.grid.node_to_edges(n))<=2 )

        while pred(trav) and (trav != anchor) and (span<max_span):
            span += utils.dist( self.grid.nodes['x'][last] -
                                 self.grid.nodes['x'][trav] )
            if direction==1:
                he=he.fwd()
                last,trav = trav,he.node_fwd()
            elif direction==-1:
                he=he.rev()
                last,trav = trav,he.node_rev()
            else:
                assert False
        return span
    
    max_span_factor=4     
    def resample(self,n,anchor,scale,direction):
        self.log.debug("resample %d to be  %g away from %d in the %s direction"%(n,scale,anchor,
                                                                                 direction) )
        if direction==1: # anchor to n is t
            he=self.grid.nodes_to_halfedge(anchor,n)
        elif direction==-1:
            he=self.grid.nodes_to_halfedge(n,anchor)
        else:
            assert False

        span_length = self.free_span(he,self.max_span_factor*scale,direction)
        self.log.debug("free span from the anchor is %g"%span_length)

        if span_length < self.max_span_factor*scale:
            n_segments = max(1,round(span_length / scale))
            target_span = span_length / n_segments
            if n_segments==1:
                self.log.debug("Only one space for 1 segment")
                return
        else:
            target_span=scale

        # first, find a point on the original ring which satisfies the target_span
        oring=self.grid.nodes['oring'][anchor]
        curve = self.curves[oring]
        anchor_f = self.grid.nodes['ring_f'][anchor]
        try:
            new_f,new_x = curve.distance_away(anchor_f,direction*target_span)
        except curve.CurveException as exc:
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

        self.grid.modify_node(n,x=new_x,ring_f=new_f)

    def resample_neighbors(self,site):
        a,b,c = site.abc
        local_length = self.scale( site.points().mean(axis=0) )

        for n,direction in [ (a,-1),
                             (c,1) ]:
            if ( (self.grid.nodes['fixed'][n] == self.SLIDE) and
                 len(self.grid.node_to_edges(n))<=2 ):
                self.resample(n=n,anchor=b,scale=local_length,direction=direction)
