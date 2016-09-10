"""

An advancing front grid generator for use with unstructured_grid

Largely a port of paver.py.

"""
import unstructured_grid
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
    def __init__(self,grid,nodes):
        self.grid=grid
        assert len(nodes)==3
        self.abc = nodes
    def metric(self):
        return self.internal_angle()
    def points(self):
        return self.grid.nodes['x'][ self.abc ]
    def internal_angle(self):
        A,B,C = self.points() 
        return internal_angle(A,B,C)
    def plot(self,ax=None):
        ax=ax or plt.gca()
        points=self.grid.nodes['x'][self.abc]
        return ax.plot( points[:,0],points[:,1],'r-o' )[0]
    def actions(self):
        theta=self.internal_angle()
        HERE


class AdvancingFront(object):
    """
    Implementation of advancing front
    """
    scale=None
    grid=None

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
        """
        g.add_node_field('oring',-1*np.ones(g.Nnodes(),'i4'),on_exists='pass')
        g.add_node_field('fixed',np.zeros(g.Nnodes(),'i4'),on_exists='pass')
        g.add_node_field('ring_f',-1*np.ones(g.Nnodes(),'f8'),on_exists='pass')
        
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

            sites.append( TriangleSite(self.grid,nodes=[a,b,c]) )
        if len(sites):
            scores=[ site.metric()
                     for site in sites ]
            best=np.argmin( scores ) 
            return sites[best]
        else:
            return None
