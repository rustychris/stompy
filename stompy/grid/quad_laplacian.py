"""
Create a nearly orthogonal quad mesh by solving for stream function
and velocity potential inside a given boundary.

Still trying to improve the formulation of the Laplacian.  Psi
(stream function) and phi (velocity potential) are solved
simultaneously. There are other options, and each field 
uniquely constraints the other field to within a constant offset.

A block matrix is constructed which solves the Laplacian for
each of psi and phi. The boundaries of the domain are restricted
to be contours of either phi or psi, and these edges meet at right
angles.

For a grid with N nodes there are 2N unknowns.
Each interior node implies 2 constraints via the Laplacian on psi and phi.

For boundary nodes not at corner, one of psi/phi implies a no-flux boundary
and d phi/dn=0 or d psi/dn=0.

The question is how to constrain the remaining boundary nodes. These boundaries
are contours of the respective field.  For a boundary segment of
s nodes, inclusive of corners, that yields s-1 constraints.


TODO: 
 - enforce monotonic sliding of nodes on the same segment.  Currently it's possible
   for nodes to slide over each other resulting in an invalid (but sometimes salvageable)
   intermediate grid.
 - Allow ragged edges.
 - Allow multiple patches that can be solved simultaneously.  
 - Depending on how patches go, may allow for overlaps in phi/psi space if they are distinct in geographic space.
 - allow patch connections that are rotated (psi on one side matches phi on the other, or a 
   full inversion


"""

import numpy as np
from collections import defaultdict
from shapely import geometry, ops
from scipy import sparse

import matplotlib.pyplot as plt
from matplotlib import colors
import itertools

from . import unstructured_grid, exact_delaunay,orthogonalize, triangulate_hole
from .. import utils, filters
from ..spatial import field
from . import front

import six

##

# borrow codes as in front.py
RIGID=front.AdvancingFront.RIGID

class NodeDiscretization(object):
    def __init__(self,g):
        self.g=g
    def construct_matrix(self,op='laplacian',dirichlet_nodes={},
                         zero_tangential_nodes=[],
                         gradient_nodes={}):
        """
        Construct a matrix and rhs for the given operation.
        dirichlet_nodes: boundary node id => value
        zero_tangential_nodes: list of lists.  each list gives a set of
          nodes which should be equal to each other, allowing specifying
          a zero tangential gradient BC.  
        gradient_nodes: boundary node id => gradient unit vector [dx,dy]
        """
        g=self.g
        # Adjust tangential node data structure for easier use
        # in matrix construction
        tangential_nodes={} 
        for grp in zero_tangential_nodes:
            leader=grp[0]
            for member in grp:
                # NB: This includes leader=>leader
                assert member not in tangential_nodes
                tangential_nodes[member]=leader

        if 0:
            # previously nodes constrained by a BC were omitted from the
            # regular equations, so the resulting matrix is always square,
            # but can have some zero rows.
            B=np.zeros(g.Nnodes(),np.float64)
            M=sparse.dok_matrix( (g.Nnodes(),g.Nnodes()),np.float64)
            multiple=False
        else:
            # Now I want to allow multiple BCs to constrain the same node.
            # How many rows will I end up with?

            # First count up the nodes that will get a regular laplacian
            # row.  This includes boundary nodes that have a no-flux BC.
            # (because that's the behavior of the discretization on a
            # boundary)
            nlaplace_rows=0
            laplace_nodes={}
            for n in range(g.Nnodes()):
                if n in dirichlet_nodes: continue
                if n in gradient_nodes: continue
                if n in tangential_nodes: continue
                laplace_nodes[n]=True
                nlaplace_rows+=1

            ndirichlet_nodes=len(dirichlet_nodes)
            # Each group of tangential gradient nodes provides len-1 constraints
            ntangential_nodes=len(tangential_nodes) - len(zero_tangential_nodes)
            ngradient_nodes=len(gradient_nodes)

            nrows=nlaplace_rows + ndirichlet_nodes + ntangential_nodes + ngradient_nodes
            
            B=np.zeros(nrows,np.float64)
            M=sparse.dok_matrix( (nrows,g.Nnodes()),np.float64)
            multiple=True

        if not multiple:
            for n in range(g.Nnodes()):
                if n in dirichlet_nodes:
                    nodes=[n]
                    alphas=[1]
                    rhs=dirichlet_nodes[n]
                elif n in gradient_nodes:
                    vec=gradient_nodes[n] # The direction of the gradient
                    normal=[vec[1],-vec[0]] # direction of zero gradient
                    dx_nodes,dx_alphas,_=self.node_discretization(n,op='dx')
                    dy_nodes,dy_alphas,_=self.node_discretization(n,op='dy')
                    assert np.all(dx_nodes==dy_nodes),"Have to be cleverer"
                    nodes=dx_nodes
                    # So if vec = [1,0], then normal=[0,-1]
                    # and I want dx*norma[0]+dy*normal[1] = 0
                    alphas=np.array(dx_alphas)*normal[0] + np.array(dy_alphas)*normal[1]
                    rhs=0
                elif n in tangential_nodes:
                    leader=tangential_nodes[n]
                    if n==leader:
                        # Really should drop the row
                        rhs=0.0
                        nodes=[n]
                        alphas=[0]
                    else:
                        rhs=0.0
                        nodes=[n,leader]
                        alphas=[1,-1]
                else:
                    nodes,alphas,rhs=self.node_discretization(n,op=op)
                    # could add to rhs here
                B[n]=rhs
                for node,alpha in zip(nodes,alphas):
                    M[n,node]=alpha
        else:
            # Very similar code, but messy to refactor so write a new loop.
            ndirichlet_nodes=len(dirichlet_nodes)
            # Each group of tangential gradient nodes provides len-1 constraints
            ntangential_nodes=len(tangential_nodes) - len(zero_tangential_nodes)
            ngradient_nodes=len(gradient_nodes)

            nrows=nlaplace_rows + ndirichlet_nodes + ntangential_nodes + ngradient_nodes
            
            B=np.zeros(nrows,np.float64)
            M=sparse.dok_matrix( (nrows,g.Nnodes()),np.float64)
            multiple=True

            row=0
            for n in laplace_nodes:
                nodes,alphas,rhs=self.node_discretization(n,op=op)
                B[row]=rhs
                for node,alpha in zip(nodes,alphas):
                    M[row,node]=alpha
                row+=1
                
            for n in dirichlet_nodes:
                B[row]=dirichlet_nodes[n]
                M[row,n]=1
                row+=1

            for n in gradient_nodes:
                vec=gradient_nodes[n] # The direction of the gradient
                normal=[vec[1],-vec[0]] # direction of zero gradient
                dx_nodes,dx_alphas,_=self.node_discretization(n,op='dx')
                dy_nodes,dy_alphas,_=self.node_discretization(n,op='dy')
                assert np.all(dx_nodes==dy_nodes),"Have to be cleverer"
                nodes=dx_nodes
                # So if vec = [1,0], then normal=[0,-1]
                # and I want dx*norma[0]+dy*normal[1] = 0
                alphas=np.array(dx_alphas)*normal[0] + np.array(dy_alphas)*normal[1]
                B[row]=0
                for node,alpha in zip(nodes,alphas):
                    M[row,node]=alpha
                row+=1
                    
            for n in tangential_nodes:
                leader=tangential_nodes[n]
                if n==leader:
                    print("skip leader")
                    continue
                M[row,n]=1
                M[row,leader]=-1
                B[row]=0.0
                row+=1
            assert row==nrows
            
        return M,B
    
    def node_laplacian(self,n0):
        return self.node_discretization(n0,'laplacian')

    def node_dx(self,n0):
        return self.node_discretization(n0,'dx')

    def node_dy(self,n0):
        return self.node_discretization(n0,'dy')

    def node_discretization(self,n0,op='laplacian'):
        def beta(c):
            return 1.0
        
        N=self.g.angle_sort_adjacent_nodes(n0)
        P=len(N)
        is_boundary=int(self.g.is_boundary_node(n0))
        M=len(N) - is_boundary

        if is_boundary:
            # roll N to start and end on boundary nodes:
            nbr_boundary=[self.g.is_boundary_node(n)
                          for n in N]
            while not (nbr_boundary[0] and nbr_boundary[-1]):
                N=np.roll(N,1)
                nbr_boundary=np.roll(nbr_boundary,1)
        
        # area of the triangles
        A=[] 
        for m in range(M):
            tri=[n0,N[m],N[(m+1)%P]]
            Am=utils.signed_area( self.g.nodes['x'][tri] )
            A.append(Am)
        AT=np.sum(A)

        alphas=[]
        x=self.g.nodes['x'][N,0]
        y=self.g.nodes['x'][N,1]
        x0,y0=self.g.nodes['x'][n0]
        
        for n in range(P):
            n_m_e=(n-1)%M
            n_m=(n-1)%P
            n_p=(n+1)%P
            a=0
            if op=='laplacian':
                if n>0 or P==M: # nm<M
                    a+=-beta(n_m_e)/(4*A[n_m_e]) * ( (y[n_m]-y[n])*(y0-y[n_m]) + (x[n] -x[n_m])*(x[n_m]-x0))
                if n<M:
                    a+= -beta(n)/(4*A[n])  * ( (y[n]-y[n_p])*(y[n_p]-y0) + (x[n_p]-x[n ])*(x0 - x[n_p]))
            elif op=='dx':
                if n>0 or P==M: # nm<M
                    a+= beta(n_m_e)/(2*AT) * (y0-y[n_m])
                if n<M:
                    a+= beta(n)/(2*AT) * (y[n_p]-y0)
            elif op=='dy':
                if n>0 or P==M: # nm<M
                    a+= beta(n_m_e)/(2*AT) * (x[n_m]-x0)
                if n<M:
                    a+= beta(n)/(2*AT) * (x0 - x[n_p])
            else:
                raise Exception('bad op')
                
            alphas.append(a)

        alpha0=0
        for e in range(M):
            ep=(e+1)%P
            if op=='laplacian':
                alpha0+= - beta(e)/(4*A[e]) * ( (y[e]-y[ep])**2 + (x[ep]-x[e])**2 )
            elif op=='dx':
                alpha0+= beta(e)/(2*AT)*(y[e]-y[ep])
            elif op=='dy':
                alpha0+= beta(e)/(2*AT)*(x[ep]-x[e])
            else:
                raise Exception('bad op')
                
        if op=='laplacian' and P>M:
            norm_grad=0 # no flux bc
            L01=np.sqrt( (x[0]-x0)**2 + (y0-y[0])**2 )
            L0P=np.sqrt( (x[0]-x[-1])**2 + (y0-y[-1])**2 )

            gamma=3/AT * ( beta(0) * norm_grad * L01/2
                           + beta(P-1) * norm_grad * L0P/2 )
        else:
            gamma=0
        assert np.isfinite(alpha0)
        return ([n0]+list(N),
                [alpha0]+list(alphas),
                -gamma)

class QuadsGen(object):
    """
    Handle a single generating grid with multiple quad patches.
    Mostly dispatches subgrids to QuadGen
    """
    def __init__(self,gen,seeds=None,cells=None,**kwargs):
        self.qgs=[]
        
        if seeds is not None:
            raise Exception('Seeds would be nice but not yet implemented')
        if cells is None:
            cells=range(gen.Ncells())

        self.g_final=None
        for cell in cells:
            loc_gen=gen.copy()
            for c in range(gen.Ncells()):
                if c!=cell:
                    loc_gen.delete_cell(c)
            loc_gen.delete_orphan_edges()
            loc_gen.delete_orphan_nodes()
            loc_gen.renumber(reorient_edges=False)

            qg=QuadGen(gen=loc_gen,**kwargs)
            self.qgs.append(qg)
            try:
                loc_g_final=qg.g_final
            except AttributeError:
                # Probably didn't execute
                continue 
            if self.g_final is None:
                self.g_final=loc_g_final
            else:
                self.g_final.add_grid(loc_g_final,merge_nodes='auto')
                
    def plot_result(self,num=5):
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)
        self.g_final.plot_edges(color='k',lw=0.5,ax=ax)
        self.g_final.plot_cells(color='0.85',lw=0,zorder=-2,ax=ax)
        ax.set_position([0,0,1,1])
        ax.axis('off')
        ax.axis('tight')
        ax.axis('equal')
        return fig,ax
    
class QuadGen(object):
    
    # default behavior computes a nominal, isotropic grid for the calculation
    #  of the orthogonal mapping, and then a separate anisotropic grid for the
    #  final result. If anisotropic is False, the isotropic grid is kept, 
    #  and its ij indices will be updated to reflect the anisotropic inputs.
    final='anisotropic' # 'isotropic', 'triangle'

    patchwise=True # use patch-by-patch method for constructing quad grid from psi/phi field.
    
    # The cell spacing in geographic coordinates for the nominal, isotropic grid
    nom_res=4.0

    scales=None
    
    # Minimum number of edges along a boundary segment in the nominal isotropic grid
    min_steps=2

    # How many iterations to execute during the smoothing of the solution grid
    # (smooth_interior_quads)
    smooth_iterations=3

    # When assigning boundary nodes to edges of the generating grid, how tolerant
    # can we be?  This is a distance in ij space.  For cartesian boundaries can
    # be small, and is hardcoded at 0.1. For ragged boundaries, compare against
    # this value.  Seems like the max ought to be a little over 1. This is
    # probably too generous
    max_ragged_node_offset=2.0

    # The additional constraints that link psi and phi create an over-determined
    # system.  Still trying to understand when this is a problem.  The solution
    # can in some cases become invalid when the gradient terms are either too
    # strong (maybe) or too loose (true).
    # 'scaled' will scale the gradient terms according to the number of extra dofs.
    # Possibe that the scaled code was a weak attempt to fix something that was
    # really a bug elsewhere, and that 1.0 is the best choice
    gradient_scale=1.0

    # 'tri' or 'quad' -- whether the intermediate grid is a quad grid or triangle
    # grid.
    intermediate='tri' # 'quad'

    # 'rebay' or 'front'.  When intermediate is 'tri', this chooses the method for
    # generating the intermediate triangular grid
    triangle_method='rebay'

    # How internal angles are specified.  'node' is only valid when a single cell
    # is selected in gen.
    angle_source='halfedge' # 'halfedge' or 'node'
    
    def __init__(self,gen,execute=True,cells=None,**kw):
        """
        gen: the design grid. cells of this grid will be filled in with 
        quads.  
        nodes should have separate i and j fields.

        i,j are interpeted as x,y indices in the reference frame of the quad grid.

        execute: if True, run the full generation process.  Otherwise preprocess
          inputs but do not solve.
        """
        utils.set_keywords(self,kw)
        gen=gen.copy()

        if self.scales is None:
            self.scales=[field.ConstantField(self.nom_res),
                         field.ConstantField(self.nom_res)]
        
        if cells is not None:
            for c in range(gen.Ncells()):
                if (c not in cells) and (not gen.cells['deleted'][c]):
                    gen.delete_cell(c)
        gen.delete_orphan_edges()
        gen.delete_orphan_nodes()
        gen.renumber(reorient_edges=False)
        
        self.gen=gen
        # [ [node_a,node_b,angle], ...]
        # node indices reference gen, which provide
        # additional groupings of nodes.
        self.internal_edges=[]

        if execute:
            self.execute()

    def add_internal_edge(self,nodes,angle=None):
        self.internal_edges.append( [nodes[0], nodes[1],angle] )
        
    def execute(self):
        self.prepare_angles()
        self.add_bezier(self.gen)
        self.g_int=self.create_intermediate_grid_tri()
        self.calc_psi_phi()
        self.g_final=self.create_final_by_patches()

    def node_ij_to_edge(self,g,dest='ij'):
        dij=(g.nodes[dest][g.edges['nodes'][:,1]]
             - g.nodes[dest][g.edges['nodes'][:,0]])
        g.add_edge_field('d'+dest,dij,on_exists='overwrite')

    def coalesce_ij_nominal(self,gen,dest='IJ',nom_res=None,min_steps=None,
                            max_cycle_len=1000):
        """ 
        Similar to coalesce_ij(), but infers a scale for I and J
        from the geographic distances.
        nom_res: TODO -- figure out good default.  This is the nominal
         spacing for i and j in geographic units.
        min_steps: edges should not be shorter than this in IJ space.

        max_cycle_len: only change for large problems.  Purpose here
          is to abort on bad inputs/bugs instead of getting into an
          infinite loop
        """
        if nom_res is None:
            nom_res=self.nom_res
        if min_steps is None:
            min_steps=self.min_steps
            
        IJ=np.zeros( (gen.Nnodes(),2), np.float64)
        IJ[...]=np.nan

        # Very similar to fill_ij_interp, but we go straight to
        # assigning dIJ
        cycles=gen.find_cycles(max_cycle_len=1000)

        assert len(cycles)==1,"For now, cannot handle multiple cycles"

        ij_in=np.c_[ gen.nodes['i'], gen.nodes['j'] ]
        ij_fixed=np.isfinite(ij_in)

        # Collect the steps so that we can close the sum at the end
        for idx in [0,1]: # i, j
            steps=[] # [ [node a, node b, delta], ... ]
            for s in cycles:
                # it's a cycle, so we can roll
                is_fixed=np.nonzero( ij_fixed[s,idx] )[0]
                assert len(is_fixed),"There are no nodes with fixed i,j!"
                
                s=np.roll(s,-is_fixed[0])
                s=np.r_[s,s[0]] # repeat first node at the end
                # Get the new indices for fixed nodes
                is_fixed=np.nonzero( ij_fixed[s,idx] )[0]

                dists=utils.dist_along( gen.nodes['x'][s] )

                for a,b in zip( is_fixed[:-1],is_fixed[1:] ):
                    d_ab=dists[b]-dists[a]
                    dij_ab=ij_in[s[b],idx] - ij_in[s[a],idx]
                    if dij_ab==0:
                        steps.append( [s[a],s[b],0] )
                    else:
                        n_steps=max(min_steps,d_ab/nom_res)
                        dIJ_ab=int( np.sign(dij_ab) * n_steps )
                        steps.append( [s[a],s[b],dIJ_ab] )
                steps=np.array(steps)
                err=steps[:,2].sum()

                stepsizes=np.abs(steps[:,2])
                err_dist=np.round(err*np.cumsum(np.r_[0,stepsizes])/stepsizes.sum())
                err_per_step = np.diff(err_dist)
                steps[:,2] -= err_per_step.astype(np.int32)

            # Now steps properly sum to 0.
            IJ[steps[0,0],idx]=0 # arbitrary starting point
            IJ[steps[:-1,1],idx]=np.cumsum(steps[:-1,2])

        gen.add_node_field(dest,IJ,on_exists='overwrite')
        gen.add_node_field(dest+'_fixed',ij_fixed,on_exists='overwrite')

    def coalesce_ij(self,gen,dest='ij'):
        """
        Copy incoming 'i' and 'j' node fields to 'ij', and note which
        values were finite in 'ij_fixed'
        """
        ij_in=np.c_[ gen.nodes['i'], gen.nodes['j'] ]
        gen.add_node_field(dest,ij_in,on_exists='overwrite')
        gen.add_node_field(dest+'_fixed',np.isfinite(ij_in))
        
    def fill_ij_interp(self,gen,dest='ij'):
        """
        Interpolate the values in gen.nodes[dest] evenly between
        the existing fixed values
        """
        # the rest are filled by linear interpolation
        strings=gen.extract_linear_strings()

        for idx in [0,1]:
            node_vals=gen.nodes[dest][:,idx] 
            for s in strings:
                # This has some 'weak' support for multiple cycles. untested.
                if s[0]==s[-1]:
                    # cycle, so we can roll
                    has_val=np.nonzero( np.isfinite(node_vals[s]) )[0]
                    if len(has_val):
                        s=np.roll(s[:-1],-has_val[0])
                        s=np.r_[s,s[0]]
                s_vals=node_vals[s]
                dists=utils.dist_along( gen.nodes['x'][s] )
                valid=np.isfinite(s_vals)
                fill_vals=np.interp( dists[~valid],
                                     dists[valid], s_vals[valid] )
                node_vals[s[~valid]]=fill_vals

    def create_intermediate_grid_tri_boundary(self,scale=None):
        """
        Create the boundaries for the intermediate grid, upsampling the bezier edges
        and assigning 'ij' along the way for fixed nodes.
        """
        if scale is None:
            scale=field.ConstantField(self.nom_res)
            
        g=unstructured_grid.UnstructuredGrid(max_sides=3,
                                             extra_edge_fields=[ ('gen_j',np.int32) ],
                                             extra_node_fields=[ ('gen_n',np.int32) ])
        g.edges['gen_j']=-1
        g.nodes['gen_n']=-1
        g.edge_defaults['gen_j']=-1
        g.node_defaults['gen_n']=-1

        gen=self.gen
        
        for j in gen.valid_edge_iter():
            # Just to get the length
            points=self.gen_bezier_linestring(j=j,samples_per_edge=10,span_fixed=False)
            dist=utils.dist_along(points)[-1]
            local_res=scale(points).min(axis=0) # min=>conservative
            N=max( self.min_steps, int(dist/local_res))
            points=self.gen_bezier_linestring(j=j,samples_per_edge=N,span_fixed=False)
            nodes=[g.add_or_find_node(x=p,tolerance=0.1)
                   for p in points]
            g.nodes['gen_n'][nodes[0]] =gen.edges['nodes'][j,0]
            g.nodes['gen_n'][nodes[-1]]=gen.edges['nodes'][j,1]

            for a,b in zip(nodes[:-1],nodes[1:]):
                g.add_edge(nodes=[a,b],gen_j=j)
        return g
    
    def create_intermediate_grid_tri(self):
        """
        Create a triangular grid for solving psi/phi.

        src: base variable name for the ij indices to use.
          i.e. gen.nodes['ij'], gen.nodes['ij_fixed'],
             and gen.edges['dij']

          the resulting grid will use 'ij' regardless, this just for the
          generating grid.

        this text needs to be updated after adapting the code below
        --
        coordinates: 
         'xy' will interpolate the gen xy coordinates to get
          node coordinates for the result.
         'ij' will leave 'ij' coordinate values in both x and 'ij'
        """

        g=self.create_intermediate_grid_tri_boundary()
        nodes=g.find_cycles(max_cycle_len=5000)[0]
        gnew=triangulate_hole.triangulate_hole(g,nodes=nodes,hole_rigidity='all',
                                               method=self.triangle_method)
        gnew.add_node_field('rigid',
                            (gnew.nodes['gen_n']>=0) & (self.gen.nodes['fixed'][gnew.nodes['gen_n']]))
        # Really it should be sufficient to have edge_defaults give -1 for gen_j, but that's
        # getting lost.  easiest to just fix non-boundary edges:
        internal=np.all( gnew.edge_to_cells()>=0, axis=1)
        gnew.edges['gen_j'][internal]=-1
        return gnew
    
    def plot_intermediate(self,num=1):
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)
        self.gen.plot_edges(lw=1.5,color='b',ax=ax)
        self.g_int.plot_edges(lw=0.5,color='k',ax=ax)

        self.g_int.plot_nodes(mask=self.g_int.nodes['rigid']>0)
        ax.axis('tight')
        ax.axis('equal')

    def add_bezier(self,gen):
        """
        Generate bezier control points for each edge.  Uses ij in
        the generating grid to calculate angles at each vertex, and then
        choose bezier control points to achieve that angle.
        """
        # Need to force the corners to be 90deg angles, otherwise
        # there's no hope of getting orthogonal cells in the interior.
        
        order=3 # cubic bezier curves
        bez=np.zeros( (gen.Nedges(),order+1,2) )
        bez[:,0,:] = gen.nodes['x'][gen.edges['nodes'][:,0]]
        bez[:,order,:] = gen.nodes['x'][gen.edges['nodes'][:,1]]

        gen.add_edge_field('bez', bez, on_exists='overwrite')

        cycles=gen.find_cycles(max_cycle_len=1000)
        assert len(cycles)==1
        cycle=cycles[0]
        
        for a,b,c in zip( np.roll(cycle,1),
                          cycle,
                          np.roll(cycle,-1) ):
            ab=gen.nodes['x'][b] - gen.nodes['x'][a]
            bc=gen.nodes['x'][c] - gen.nodes['x'][b]
            
            j_ab=gen.nodes_to_edge(a,b)
            j_bc=gen.nodes_to_edge(b,c)

            # This makes use of angle being defined relative to a CCW
            # cycle, not the order of edge['nodes']
            dtheta_ij=(gen.edges['angle'][j_bc] - gen.edges['angle'][j_ab])*np.pi/180.
            dtheta_ij=(dtheta_ij+np.pi)%(2*np.pi) - np.pi
            
            # Angle of A->B
            theta0=np.arctan2(ab[1],ab[0])
            theta1=np.arctan2(bc[1],bc[0])
            dtheta=(theta1 - theta0 + np.pi) % (2*np.pi) - np.pi

            theta_err=dtheta-dtheta_ij
            # Make sure we're calculating error in the shorter direction
            theta_err=(theta_err+np.pi)%(2*np.pi) - np.pi
            
            cp0 = gen.nodes['x'][b] + utils.rot(  theta_err/2, 1./3 * -ab )
            if gen.edges['nodes'][j_ab,0]==b:
                cp_i=1
            else:
                cp_i=2
            gen.edges['bez'][j_ab,cp_i] = cp0
            
            cp1 = gen.nodes['x'][b] + utils.rot( -theta_err/2, 1./3 * bc )
            if gen.edges['nodes'][j_bc,0]==b:
                cp_i=1
            else:
                cp_i=2
            gen.edges['bez'][j_bc,cp_i] = cp1

    def plot_gen_bezier(self,num=10):
        gen=self.gen
        fig=plt.figure(num)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
        gen.plot_edges(lw=0.3,color='k',alpha=0.5,ax=ax)
        gen.plot_nodes(alpha=0.5,ax=ax,zorder=3,color='orange')
        
        for j in self.gen.valid_edge_iter():
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            t=np.linspace(0,1,21)

            B0=(1-t)**3
            B1=3*(1-t)**2 * t
            B2=3*(1-t)*t**2
            B3=t**3
            points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]

            ax.plot(points[:,0],points[:,1],'b-',zorder=2,lw=1.5)
            ax.plot(bez[:,0],bez[:,1],'r-o',zorder=1,alpha=0.5,lw=1.5)

        for n12 in self.internal_edges:
            ax.plot( gen.nodes['x'][n12[:2],0],
                     gen.nodes['x'][n12[:2],1], 'g-')
            i_angle=self.internal_edge_angle(n12)
            mid=gen.nodes['x'][n12[:2],:].mean(axis=0)
            ax.text( mid[0],mid[1], str(i_angle),color='g')

    def gen_bezier_curve(self,j=None,samples_per_edge=10,span_fixed=True):
        """
        j: make a curve for gen.edges[j], instead of the full boundary cycle.
        samples_per_edge: how many samples to use to approximate each bezier
         segment
        span_fixed: if j is specified, create a curve that includes j and
         adjacent edges until a fixed node is encountered
        """
        points=self.gen_bezier_linestring(j=j,samples_per_edge=samples_per_edge,
                                          span_fixed=span_fixed)
        if j is None:
            return front.Curve(points,closed=True)
        else:
            return front.Curve(points,closed=False)
        
    def gen_bezier_linestring(self,j=None,samples_per_edge=10,span_fixed=True):
        """
        Calculate an up-sampled linestring for the bezier boundary of self.gen
        
        j: limit the curve to a single generating edge if given.
        span_fixed: see gen_bezier_curve()
        """
        gen=self.gen

        # need to know which ij coordinates are used in order to know what is
        # fixed. So far fixed is the same whether IJ or ij, so not making this
        # a parameter yet.
        src='IJ'
        
        if j is None:
            node_pairs=zip(bound_nodes,np.roll(bound_nodes,-1))
            bound_nodes=self.gen.boundary_cycle() # probably eating a lot of time.
        else:
            if not span_fixed:
                node_pairs=[ self.gen.edges['nodes'][j] ]
            else:
                nodes=[]

                # Which coord is changing on j? I.e. which fixed should
                # we consult?
                # A little shaky here.  Haven't tested this with nodes
                # that are fixed in only coordinate.
                j_coords=self.gen.nodes[src][ self.gen.edges['nodes'][j] ]
                if j_coords[0,0] == j_coords[1,0]:
                    coord=1
                elif j_coords[0,1]==j_coords[1,1]:
                    coord=0
                else:
                    raise Exception("Neither coordinate is constant on this edge??")
                
                trav=self.gen.halfedge(j,0)
                while 1: # FWD
                    n=trav.node_fwd()
                    nodes.append(n)
                    if self.gen.nodes[src+'_fixed'][n,coord]:
                        break
                    trav=trav.fwd()
                nodes=nodes[::-1]
                trav=self.gen.halfedge(j,0)
                while 1: # REV
                    n=trav.node_rev()
                    nodes.append(n)
                    if self.gen.nodes[src+'_fixed'][n,coord]:
                        break
                    trav=trav.rev()
                node_pairs=zip( nodes[:-1], nodes[1:])
            
        points=[]
        for a,b in node_pairs:
            j=gen.nodes_to_edge(a,b)
            
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            t=np.linspace(0,1,1+samples_per_edge)
            if n0==b: # have to flip order
                t=t[::-1]

            B0=(1-t)**3
            B1=3*(1-t)**2 * t
            B2=3*(1-t)*t**2
            B3=t**3
            edge_points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]

            points.append(edge_points[:-1])
        if j is not None:
            # When the curve isn't closed, then be inclusive of both
            # ends
            points.append(edge_points[-1:])
            
        return np.concatenate(points,axis=0)

    def adjust_intermediate_bounds(self):
        """
        Adjust exterior of intermediate grid with bezier
        curves
        """
        gen=self.gen
        g=self.g_int.copy()

        # This one gets tricky with the floating-point ij values.
        # gen.nodes['ij'] may be float valued.
        # The original code iterates over gen edges, assumes that
        # Each gen edge divides to an exact number of nodes, then
        # we know the exact ij of those nodes,
        # pre-evaluate the spline and then just find the corresponding
        # nodes.

        # With float-valued gen.nodes['ij'], though, we still have
        # a bezier curve, but it's ends may not be on integer values.
        # The main hurdle is that we need a different way of associating
        # nodes in self.g to a generating edge
        
        for j in gen.valid_edge_iter():
            n0=gen.edges['nodes'][j,0]
            nN=gen.edges['nodes'][j,1]
            bez=gen.edges['bez'][j]
            
            g_nodes=np.nonzero( g.nodes['gen_j']==j )[0]

            p0=gen.nodes['x'][n0]
            pN=gen.nodes['x'][nN]

            T=utils.dist(pN-p0)
            t=utils.dist( g.nodes['x'][g_nodes] - p0 ) / T

            too_low=(t<0)
            too_high=(t>1)
            if np.any(too_low):
                print("Some low")
            if np.any(too_high):
                print("Some high")
                
            t=t.clip(0,1)

            if 1: # the intended bezier way:
                B0=(1-t)**3
                B1=3*(1-t)**2 * t
                B2=3*(1-t)*t**2
                B3=t**3
                points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]
            else: # debugging linear way
                print("Debugging - no bezier boundary")
                points=(1-t)[:,None]*p0 + t[:,None]*pN

            for n,point in zip(g_nodes,points):
                g.modify_node(n,x=point)

    def smooth_interior_quads(self,g,iterations=None):
        """
        Smooth quad grid by allowing boundary nodes to slide, and
        imparting a normal constraint at the boundary.
        """
        if iterations is None:
            iterations=self.smooth_iterations
        # So the anisotropic smoothing has a weakness where the spacing
        # of boundary nodes warps the interior.
        # Currently I smooth x and y independently, using the same matrix.

        # But is there a way to locally linearize where slidable boundary nodes
        # can fall, forcing their internal edge to be perpendicular to the boundary?

        # For a sliding boundary node [xb,yb] , it has to fall on a line, so
        # c1*xb + c2*yb = c3
        # where [c1,c2] is a normal vector of the line

        # And I want the edge to its interior neighbor (xi,yi) perpendicular to that line.
        # (xb-xi)*c1 + (yb-yi)*c2 = 0

        # Old approach: one curve for the whole region:
        # curve=self.gen_bezier_curve()

        # New approach: One curve per straight line, and constrain nodes to
        # that curve. This should keep node bound to intervals between rigid
        # neighboring nodes.  It does not keep them from all landing on top of each
        # other, and does not distinguish rigid in i vs rigid in j
        n_to_curve={}
        for n in g.valid_node_iter():
            if g.nodes['gen_j'][n]>=0:
                n_to_curve[n]=self.gen_bezier_curve(j=g.nodes['gen_j'][n],span_fixed=True)
        
        N=g.Nnodes()

        for slide_it in utils.progress(range(iterations)):
            M=sparse.dok_matrix( (2*N,2*N), np.float64)

            rhs=np.zeros(2*N,np.float64)

            for n in g.valid_node_iter():
                if g.is_boundary_node(n):
                    dirichlet=g.nodes['rigid'][n]==RIGID
                    if dirichlet:
                        M[n,n]=1
                        rhs[n]=g.nodes['x'][n,0]
                        M[N+n,N+n]=1
                        rhs[N+n]=g.nodes['x'][n,1]
                    else:
                        boundary_nbrs=[]
                        interior_nbr=[]
                        for nbr in g.node_to_nodes(n):
                            if g.nodes['gen_j'][nbr]>=0:
                                boundary_nbrs.append(nbr)
                            else:
                                interior_nbr.append(nbr)
                        assert len(boundary_nbrs)==2
                        assert len(interior_nbr)==1

                        if 0: # figure out the normal from neighbors.
                            vec=np.diff( g.nodes['x'][boundary_nbrs], axis=0)[0]
                            nrm=utils.to_unit( np.array([vec[1],-vec[0]]) )
                            tng=utils.to_unit( np.array(vec) )
                        else: # figure out normal from bezier curve
                            curve=n_to_curve[n]
                            f=curve.point_to_f(g.nodes['x'][n],rel_tol='best')
                            tng=curve.tangent(f)
                            nrm=np.array([-tng[1],tng[0]])

                        c3=np.dot(nrm,g.nodes['x'][n])
                        # n-equation puts it on the line
                        M[n,n]=nrm[0]
                        M[n,N+n]=nrm[1]
                        rhs[n]=c3
                        # N+n equation set the normal
                        # the edge to interior neighbor (xi,yi) perpendicular to that line.
                        # (xb-xi)*c1 + (yb-yi)*c2 = 0
                        # c1*xb - c1*xi + c2*yb - c2*yi = 0
                        inbr=interior_nbr[0]
                        M[N+n,n]=tng[0]
                        M[N+n,inbr]=-tng[0]
                        M[N+n,N+n]=tng[1]
                        M[N+n,N+inbr]=-tng[1]
                        rhs[N+n]=0.0
                else:
                    nbrs=g.node_to_nodes(n)
                    if 0: # isotropic
                        M[n,n]=-len(nbrs)
                        M[N+n,N+n]=-len(nbrs)
                        for nbr in nbrs:
                            M[n,nbr]=1
                            M[N+n,N+nbr]=1
                    else:
                        # In the weighting, want to normalize by distances
                        i_length=0
                        j_length=0
                        dists=utils.dist(g.nodes['x'][n],g.nodes['x'][nbrs])
                        ij_deltas=np.abs(g.nodes['ij'][n] - g.nodes['ij'][nbrs])
                        # length scales for i and j
                        ij_scales=1./( (ij_deltas*dists[:,None]).sum(axis=0) )

                        assert np.all( np.isfinite(ij_scales) )

                        for nbr,ij_delta in zip(nbrs,ij_deltas):
                            fac=(ij_delta*ij_scales).sum()
                            M[n,nbr]=fac
                            M[n,n]-=fac
                            M[N+n,N+nbr]=fac
                            M[N+n,N+n]-=fac

            new_xy=sparse.linalg.spsolve(M.tocsr(),rhs)

            g.nodes['x'][:,0]=new_xy[:N]
            g.nodes['x'][:,1]=new_xy[N:]

            # And nudge the boundary nodes back onto the boundary
            for n in g.valid_node_iter():
                if (g.nodes['gen_j'][n]>=0) and (g.nodes['rigid'][n]!=RIGID):
                    curve=n_to_curve[n]
                    new_f=curve.point_to_f(g.nodes['x'][n],rel_tol='best')
                    g.nodes['x'][n] = curve(new_f)

        return g

    def bezier_boundary_polygon(self):
        """
        For trimming nodes that got shifted outside the proper boundary
        """
        # This would be more efficient if unstructured_grid just provided
        # some linestring methods that accepted a node mask
        
        g_tri=self.g_int.copy()
        internal_nodes=g_tri.nodes['gen_j']<0
        for n in np.nonzero(internal_nodes)[0]:
            g_tri.delete_node_cascade(n)
        boundary_linestring = g_tri.extract_linear_strings()[0]
        boundary=g_tri.nodes['x'][boundary_linestring]
        return geometry.Polygon(boundary)

    def calc_bc_gradients(self,gtri):
        """
        Calculate gradient vectors for psi and phi along
        the boundary.
        """
        bcycle=gtri.boundary_cycle()

        # First calculate psi gradient per edge:
        j_grad_psi=np.zeros( (len(bcycle),2), np.float64)
        j_angles=self.gen.edges['angle'][ gtri.edges['gen_j'] ] * np.pi/180.
        # trial and error correction
        j_angles-=np.pi/2

        for ji,(n1,n2) in enumerate( zip(bcycle[:-1],bcycle[1:]) ):
            j=gtri.nodes_to_edge(n1,n2)
            tang_xy=utils.to_unit( gtri.nodes['x'][n2] - gtri.nodes['x'][n1] )

            tang_ij=np.r_[ np.cos(j_angles[j]), np.sin(j_angles[j])]

            # Construct a rotation R such that R.dot(tang_ij)=[1,0],
            # then apply to tang_xy
            Rpsi=np.array([[tang_ij[0], tang_ij[1]],
                           [-tang_ij[1], tang_ij[0]] ] )
            j_grad_psi[ji,:]=Rpsi.dot(tang_xy)

        # Interpolate to nodes
        bc_grad_psi=np.zeros( (len(bcycle),2), np.float64)

        N=len(bcycle)
        for ni in range(N):
            bc_grad_psi[ni,:]=0.5*( j_grad_psi[ni,:] +
                                    j_grad_psi[(ni-1)%N,:] )

        bc_grad_phi=np.zeros( (len(bcycle),2), np.float64)

        # 90 CW from psi
        bc_grad_phi[:,0]=bc_grad_psi[:,1]
        bc_grad_phi[:,1]=-bc_grad_psi[:,0]

        # Convert to dicts:
        grad_psi={}
        grad_phi={}
        for ni,n in enumerate(bcycle):
            grad_psi[n]=bc_grad_psi[ni,:]
            grad_phi[n]=bc_grad_phi[ni,:]
        return grad_psi,grad_phi

    def prepare_angles(self):
        if self.angle_source=='node':
            self.prepare_angles_nodes()
        else:
            self.prepare_angles_halfedge()

    def prepare_angles_halfedge(self):
        """
        Move turn angles from half edges to absolute angles of edges.
        Internal edges will get an angle, then be removed from
        gen and recorded instead in self.internal_edges
        """
        # Might get smarter in the future, but for now we save some internal
        # edge info, sum turns to nodes, have prepare_angles_nodes() do its
        # thing, then return to complete the internal edge info

        gen=self.gen
        e2c=gen.edge_to_cells()

        internals=[]

        gen.add_node_field('turn',np.nan*np.zeros(gen.Nnodes(),np.float32), on_exists='overwrite')

        valid_fwd=(e2c[:,0]>=0) & np.isfinite(gen.edges['turn_fwd']) & (gen.edges['turn_fwd']!=0)
        valid_rev=(e2c[:,1]>=0) & np.isfinite(gen.edges['turn_rev']) & (gen.edges['turn_rev']!=0)

        # iterate over nodes, so that edges can use default values 
        fixed_nodes=np.unique( np.concatenate( (gen.edges['nodes'][valid_fwd,1],
                                                gen.edges['nodes'][valid_rev,0]) ) )

        j_int={}
        for n in fixed_nodes:
            turn=0
            nbrs=gen.node_to_nodes(n)
            he=he0=gen.nodes_to_halfedge(nbrs[0],n)
            # Start on the CCW-most external edge:
            while he.cell_opp()>=0:
                he=he.fwd().opposite()
                assert he!=he0
            he0=he

            while 1:
                if he.cell()>=0:
                    if he.orient==0:
                        sub_turn=gen.edges['turn_fwd'][he.j]
                    else:
                        sub_turn=gen.edges['turn_rev'][he.j]
                    if np.isnan(sub_turn): sub_turn=180
                    elif sub_turn==0.0: sub_turn=180
                    turn=turn+sub_turn
                else:
                    sub_turn=np.nan

                he=he.fwd().opposite()

                if (e2c[he.j,0]>=0) and (e2c[he.j,1]>=0):
                    if he.j not in j_int:
                        j_int[he.j]=1
                        print(f"Adding j={he.j} as an internal edge")
                        internals.append( dict(j=he.j,
                                               nodes=[he.node_fwd(),he.node_rev()],
                                               turn=turn,
                                               j0=he0.j) )


                if he==he0: break
            gen.nodes['turn'][n]=turn

        # Come back for handling of internal edges
        for internal in internals:
            print("Internal edge: ",internal['nodes'])
            gen.merge_cells(j=internal['j'])

        self.prepare_angles_nodes()

        for internal in internals:
            self.add_internal_edge(internal['nodes'],
                                   gen.edges['angle'][internal['j0']]+internal['turn'])
        self.internals=internals # just for debugging

            
    def prepare_angles_nodes(self):
        """
        Move angles from turns at node to
        absolute orientations of edges.
        """
        # Allow missing angles to either be 0 or nan
        gen=self.gen

        missing=np.isnan(gen.nodes['turn'])
        gen.nodes['turn'][missing]=0.0
        no_turn=gen.nodes['turn']==0.0
        gen.nodes['turn'][no_turn]=180.0
        gen.add_node_field('fixed',~no_turn,on_exists='pass')

        # Do the angles add up okay?
        net_turn=(180-gen.nodes['turn']).sum()
        assert np.abs(net_turn-360.0)<1e-10,"Net turn %.2f!=0"%net_turn

        gen.add_edge_field('angle',np.nan*np.zeros(gen.Nedges()),
                           on_exists='overwrite')

        # relative to the orientation of the first edge
        # that's encountered, and relative to a CCW traversal
        # of the cell (so not necessarily the orientation of
        # the individual edges)
        orientation=0 

        cycles=gen.find_cycles(max_cycle_len=1000)
        assert len(cycles)==1,"For now, cannot handle multiple cycles"
        cycle=cycles[0]

        for a,b in zip( cycle, np.roll(cycle,-1) ):
            j=gen.nodes_to_edge(a,b)
            assert j is not None
            gen.edges['angle'][j]=orientation

            orientation=(orientation + (180-gen.nodes['turn'][b])) % 360.0
    
    def internal_edge_angle(self,gen_edge):
        """
        not exact, but try a heuristic.
        use adjacent edges to estimate the +i tangent vector,
        Then compare to the angle of gen_edge.
        Returns 0 or 90 (0 vs 180 is not unique)
        """
        # Use the specified angle if it's set:
        if gen_edge[2] is not None:
            return gen_edge[2]
        
        gen=self.gen
        e2c=gen.edge_to_cells()
        i_tan_vecs=[]
        for n in gen_edge[:2]:
            for j in gen.node_to_edges(n):
                angle=gen.edges['angle'][j]
                tan_vec=np.diff(gen.nodes['x'][ gen.edges['nodes'][j] ],axis=0)[0]
                tan_vec=utils.to_unit(tan_vec)
                if e2c[j,0]<0:
                    # This edge goes opposite the cycle direction
                    tan_vec*=-1 
                i_tan_vec=utils.rot(-angle*np.pi/180.,tan_vec)
                i_tan_vecs.append(i_tan_vec)
        i_tan=utils.to_unit( np.mean(i_tan_vecs,axis=0) )
        j_tan=np.array( [i_tan[1],-i_tan[0]] ) # sign may be off, no worries

        d_gen_edge= np.diff(gen.nodes['x'][gen_edge[:2]],axis=0)[0]

        j_score=np.dot(j_tan,d_gen_edge)
        i_score=np.dot(i_tan,d_gen_edge)

        if np.abs(j_score)>np.abs(i_score):
            return 90
        else:
            return 0
    
    def calc_psi_phi(self):
        gtri=self.g_int
        self.nd=nd=NodeDiscretization(gtri)

        e2c=gtri.edge_to_cells()

        # check boundaries and determine where Laplacian BCs go
        boundary=e2c.min(axis=1)<0
        i_dirichlet_nodes={} # for psi
        j_dirichlet_nodes={} # for phi

        # Block of nodes with a zero-tangential-gradient BC
        i_tan_groups=[]
        j_tan_groups=[]
        # i_tan_groups_i=[] # the input i value
        # j_tan_groups_j=[] # the input j value

        # Try zero-tangential-gradient nodes.  Current code will be under-determined
        # without the derivative constraints.
        bcycle=gtri.boundary_cycle()
        n1=bcycle[-1]
        i_grp=None
        j_grp=None

        psi_gradients,phi_gradients=self.calc_bc_gradients(gtri)
        psi_gradient_nodes={} # node => unit vector of gradient direction
        phi_gradient_nodes={} # node => unit vector of gradient direction

        j_angles=self.gen.edges['angle'][ gtri.edges['gen_j'] ]

        for n2 in bcycle:
            j=gtri.nodes_to_edge(n1,n2)

            imatch=j_angles[j] % 180==0
            jmatch=j_angles[j] % 180==90

            if imatch: 
                if i_grp is None:
                    i_grp=[n1]
                    i_tan_groups.append(i_grp)
                    # i_tan_groups_i.append(i1)
                i_grp.append(n2)
            else:
                i_grp=None

            if jmatch:
                if j_grp is None:
                    j_grp=[n1]
                    j_tan_groups.append(j_grp)
                    # j_tan_groups_j.append(j1)
                j_grp.append(n2)
            else:
                j_grp=None

            if not (imatch or jmatch):
                # Register gradient BC for n1
                psi_gradient_nodes[n1]=psi_gradients[n1]
                psi_gradient_nodes[n2]=psi_gradients[n2]
                phi_gradient_nodes[n1]=phi_gradients[n1]
                phi_gradient_nodes[n2]=phi_gradients[n2]
            n1=n2

        # bcycle likely starts in the middle of either a j_tan_group or i_tan_group.
        # see if first and last need to be merged
        if i_tan_groups[0][0]==i_tan_groups[-1][-1]:
            i_tan_groups[0].extend( i_tan_groups.pop()[:-1] )
        if j_tan_groups[0][0]==j_tan_groups[-1][-1]:
            j_tan_groups[0].extend( j_tan_groups.pop()[:-1] )

        # Extra degrees of freedom:
        # Each tangent group leaves an extra dof (a zero row)
        # and the above BCs constrain 3 of those
        dofs=len(i_tan_groups) + len(j_tan_groups) - 3
        assert dofs>0

        # Use the internal_edges to combine tangential groups
        def join_groups(groups,nA,nB):
            grp_result=[]
            grpA=grpB=None
            for grp in groups:
                if nA in grp:
                    assert grpA is None
                    grpA=grp
                elif nB in grp:
                    assert grpB is None
                    grpB=grp
                else:
                    grp_result.append(grp)
            assert grpA is not None
            assert grpB is not None
            grp_result.append( list(grpA) + list(grpB) )
            return grp_result

        for gen_edge in self.internal_edges:
            internal_angle=self.internal_edge_angle(gen_edge)
            edge=[gtri.select_nodes_nearest(x)
                  for x in self.gen.nodes['x'][gen_edge[:2]]]
            if internal_angle%180==0: # join on i
                print("Joining two i_tan_groups")
                i_tan_groups=join_groups(i_tan_groups,edge[0],edge[1])
            elif internal_angle%180==90: # join on j
                print("Joining two j_tan_groups")
                j_tan_groups=join_groups(j_tan_groups,edge[0],edge[1])
            else:
                import pdb
                pdb.set_trace()
                print("Internal edge doesn't appear to join same-valued contours")

        # find longest consecutive stretch of angle=90 edges 
        longest=(0,None,None)
        # start at a nice corner
        cycle=np.roll( bcycle,-np.nonzero( gtri.nodes['rigid'][bcycle])[0][0] )
        n_start=cycle[0]
        dist=0.0
        for na,nb in zip(cycle[:-1],cycle[1:]):
            j=gtri.nodes_to_edge(na,nb)
            assert j is not None
            angle=self.gen.edges['angle'][gtri.edges['gen_j'][j]]
            if angle==90:
                dist+=gtri.edges_length(j)
            else:
                if dist>longest[0]:
                    longest=(dist,n_start,na)
                n_start=nb
                dist=0.0
        if dist>longest[0]:
            longest=(dist,n_start,nb)

        assert longest[1] is not None
        assert longest[2] is not None
        # Can I really decide the sign here?
        i_dirichlet_nodes[longest[1]]=-1
        i_dirichlet_nodes[longest[2]]=1
        j_dirichlet_nodes[j_tan_groups[1][0]]=1

        self.i_dirichlet_nodes=i_dirichlet_nodes
        self.i_tan_groups=i_tan_groups
        self.i_grad_nodes=psi_gradient_nodes
        self.j_dirichlet_nodes=j_dirichlet_nodes
        self.j_tan_groups=j_tan_groups
        self.j_grad_nodes=phi_gradient_nodes

        Mblocks=[]
        Bblocks=[]
        if 1: # PSI
            M_psi_Lap,B_psi_Lap=nd.construct_matrix(op='laplacian',
                                                    dirichlet_nodes=i_dirichlet_nodes,
                                                    zero_tangential_nodes=i_tan_groups,
                                                    gradient_nodes=psi_gradient_nodes)
            Mblocks.append( [M_psi_Lap,None] )
            Bblocks.append( B_psi_Lap )
        if 1: # PHI
            # including phi_gradient_nodes, and the derivative links below
            # is redundant but balanced.
            M_phi_Lap,B_phi_Lap=nd.construct_matrix(op='laplacian',
                                                    dirichlet_nodes=j_dirichlet_nodes,
                                                    zero_tangential_nodes=j_tan_groups,
                                                    gradient_nodes=phi_gradient_nodes)
            Mblocks.append( [None,M_phi_Lap] )
            Bblocks.append( B_phi_Lap )
        if 1:
            # Not sure what the "right" value is here.
            # When the grid is coarse and irregular, the
            # error in these blocks can overwhelm the BCs
            # above.  This scaling decreases the weight of
            # these blocks.
            # 0.1 was okay
            # Try normalizing based on degrees of freedom.
            # how many dofs are we short?
            # This assumes that the scale of the rows above is of
            # the same order as the scale of a derivative row below.

            # each of those rows constrains 1 dof, and I want the
            # set of derivative rows to constrain dofs. And there
            # are 2*Nnodes() rows.
            # Hmmm.  Had a case where it needed to be bigger (lagoon)
            # Not sure why.
            if self.gradient_scale=='scaled':
                gradient_scale = dofs / (2*gtri.Nnodes())
            else:
                gradient_scale=self.gradient_scale

            # PHI-PSI relationship
            # When full dirichlet is used, this doesn't help, but if
            # just zero-tangential-gradient is used, this is necessary.
            Mdx,Bdx=nd.construct_matrix(op='dx')
            Mdy,Bdy=nd.construct_matrix(op='dy')
            if gradient_scale!=1.0:
                Mdx *= gradient_scale
                Mdy *= gradient_scale
                Bdx *= gradient_scale
                Bdy *= gradient_scale
            Mblocks.append( [Mdy,-Mdx] )
            Mblocks.append( [Mdx, Mdy] )
            Bblocks.append( np.zeros(Mdx.shape[1]) )
            Bblocks.append( np.zeros(Mdx.shape[1]) )

        self.Mblocks=Mblocks
        self.Bblocks=Bblocks

        bigM=sparse.bmat( Mblocks )
        rhs=np.concatenate( Bblocks )

        psi_phi,*rest=sparse.linalg.lsqr(bigM,rhs)
        self.psi_phi=psi_phi
        self.psi=psi_phi[:gtri.Nnodes()]
        self.phi=psi_phi[gtri.Nnodes():]

        # Using the tan_groups, set the values to be exact
        for i_grp in i_tan_groups:
            self.psi[i_grp]=self.psi[i_grp].mean()
        for j_grp in j_tan_groups:
            self.phi[j_grp]=self.phi[j_grp].mean()
            
    def plot_psi_phi_setup(self,num=11):
        """
        Plot the BCs that went into the psi_phi calculation:
        """
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)
        self.g_int.plot_edges(color='k',lw=0.5,ax=ax,alpha=0.2)
        ax.axis('off')

        for i_d in self.i_dirichlet_nodes:
            ax.annotate( f"$\psi$={self.i_dirichlet_nodes[i_d]}",
                         self.g_int.nodes['x'][i_d], va='top',
                         arrowprops=dict(arrowstyle='simple',alpha=0.4))
        for j_d in self.j_dirichlet_nodes:
            ax.annotate( f"$\phi$={self.j_dirichlet_nodes[j_d]}",
                         self.g_int.nodes['x'][i_d], va='bottom' ,
                         arrowprops=dict(arrowstyle='simple',alpha=0.4))

        from matplotlib import cm
        from itertools import cycle
        group_colors=cycle( list(colors.TABLEAU_COLORS.values()) )

        for i_grp in self.i_tan_groups:
            ax.plot( self.g_int.nodes['x'][i_grp,0],self.g_int.nodes['x'][i_grp,1],
                     '.',color=next(group_colors))

        for j_grp in self.j_tan_groups:
            ax.plot( self.g_int.nodes['x'][j_grp,0],self.g_int.nodes['x'][j_grp,1],
                     '+',color=next(group_colors))

        i_quivs=np.array( [ [self.g_int.nodes['x'][n], self.i_grad_nodes[n] ]
                            for n in self.i_grad_nodes] )
        j_quivs=np.array( [ [self.g_int.nodes['x'][n], self.j_grad_nodes[n] ]
                            for n in self.j_grad_nodes] )

        ax.quiver(i_quivs[:,0,0], i_quivs[:,0,1],
                  i_quivs[:,1,0], i_quivs[:,1,1],
                  color='k')
        ax.quiver(j_quivs[:,0,0], j_quivs[:,0,1],
                  j_quivs[:,1,0], j_quivs[:,1,1],
                  color='r')

        ax.set_position([0,0,1,1])

        return fig,ax
    
    def plot_psi_phi(self,num=4,thinning=2,ax=None):
        if ax is None:
            plt.figure(num).clf()
            fig,ax=plt.subplots(num=num)

        #di,dj=np.nanmax(self.gen.nodes['ij'],axis=0) - np.nanmin(self.gen.nodes['ij'],axis=0)
        di=self.psi.max() - self.psi.min()
        dj=self.phi.max() - self.phi.min()
        delta=max(di,dj)/30 # 30 contours in the larger dimension
        di/=delta
        dj/=delta
        
        self.g_int.plot_edges(color='k',lw=0.5,alpha=0.2)
        cset_psi=self.g_int.contour_node_values(self.psi,int(di/thinning),
                                                linewidths=1.5,linestyles='solid',colors='orange',
                                                ax=ax)
        cset_phi=self.g_int.contour_node_values(self.phi,int(dj/thinning),
                                                linewidths=1.5,linestyles='solid',colors='blue',
                                                ax=ax)
        ax.axis('tight')
        ax.axis('equal')

        ax.clabel(cset_psi, fmt="$\psi$=%g", fontsize=10, inline=False, use_clabeltext=True)
        ax.clabel(cset_phi, fmt="$\phi$=%g", fontsize=10, inline=False, use_clabeltext=True)

    def plot_result(self,num=5):
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)
        self.g_final.plot_edges(color='k',lw=0.5,ax=ax)
        self.g_final.plot_cells(color='0.85',lw=0,zorder=-2,ax=ax)
        ax.set_position([0,0,1,1])
        ax.axis('off')
        ax.axis('tight')
        ax.axis('equal')
        return fig,ax

    def psiphi_to_ij(self,gen,g_int,src='ij',inverse=False):
        """
        Return a mapping of psi=>i and phi=>j
        This is built from fixed nodes of gen, and self.psi,self.phi defined 
        on all of the nodes of g_int.
        src defines what field is taken from gen.
        Nodes are matched by nearest node search.

        For now, this assumes the mapping is independent for the two coordinates.
        For more complicated domains this mapping will have to become a 
        function [psi x phi] => [i x j].
        Currently it's psi=>i, phi=>j.

        Returns a function that takes [N,2] in psi/phi space, and returns [N,2]
        in ij space (or the inverse of that if inverse is True)
        """
        for coord in [0,1]: # i,j
            gen_valid=(~gen.nodes['deleted'])&(gen.nodes[src+'_fixed'][:,coord])
            # subset of gtri nodes that map to fixed gen nodes
            gen_to_int_nodes=[g_int.select_nodes_nearest(x)
                              for x in gen.nodes['x'][gen_valid]]

            # i or j coord:
            all_coord=gen.nodes[src][gen_valid,coord]
            if coord==0:
                all_field=self.psi[gen_to_int_nodes]
            else:
                all_field=self.phi[gen_to_int_nodes]

            # Build the 1-D mapping of i/j to psi/phi
            # [ {i or j value}, {mean of psi or phi at that i/j value} ]
            coord_to_field=np.array( [ [k,np.mean(all_field[elts])]
                                       for k,elts in utils.enumerate_groups(all_coord)] )
            if coord==0:
                i_psi=coord_to_field
            else:
                j_phi=coord_to_field

        # the mapping isn't necessarily monotonic at this point, but it
        # needs to be..  so force it.
        # enumerate_groups will put k in order, but not the field values
        # Note that phi is sorted decreasing
        i_psi[:,1] = np.sort(i_psi[:,1])
        j_phi[:,1] = np.sort(j_phi[:,1])[::-1]

        def mapper(psiphi,i_psi=i_psi,j_phi=j_phi):
            ij=np.zeros_like(psiphi)
            ij[:,0]=np.interp(psiphi[:,0],i_psi[:,1],i_psi[:,0])
            ij[:,1]=np.interp(psiphi[:,1],j_phi[::-1,1],j_phi[::-1,0])
            return ij
        def mapper_inv(ij,i_psi=i_psi,j_phi=j_phi):
            psiphi=np.zeros_like(ij)
            psiphi[:,0]=np.interp(ij[:,0],i_psi[:,0],i_psi[:,1])
            psiphi[:,1]=np.interp(ij[:,1],j_phi[:,0],j_phi[:,1])
            return psiphi

        if inverse:
            return mapper_inv
        else:
            return mapper
        
    def remap_ij(self,g,src='ij'):
        """
        g: grid with a nodes['ij'] field
        src: a differently scaled 'ij' field on self.gen

        returns an array like g.node['ij'], but mapped to self.gen.nodes[src].

        In particular, this is useful for calculating what generating ij values
        would be on a nominal resolution grid (i.e. where the grid nodes and edges
        are uniform in IJ space).
        """
        # The nodes of g are defined on IJ, and I want
        # to map those IJ to ij in a local way. Local in the sense that 
        # I may map to different i in different parts of the domain.

        IJ_in=g.nodes['ij'] # g may be generated in IJ space, but the field is still 'ij'

        # Make a hash to ease navigation
        IJ_to_n={ tuple(IJ_in[n]):n 
                  for n in g.valid_node_iter() }
        ij_out=np.zeros_like(IJ_in)*np.nan

        for coord in [0,1]: # psi/i,  phi/j
            fixed=np.nonzero( self.gen.nodes[src+'_fixed'][:,coord] )[0]
            for gen_n in fixed:
                val=self.gen.nodes[src][gen_n,coord]
                # match that with a node in g
                n=g.select_nodes_nearest( self.gen.nodes['x'][gen_n] )
                # Should be a very good match.  Could also search
                # based on IJ, and get an exact match
                assert np.allclose( g.nodes['x'][n], self.gen.nodes['x'][gen_n] ), "did not find a good match g~gen, based on x"
                ij_out[n,coord]=val

                # Traverse in IJ space (i.e. along g grid lines)
                for incr in [1,-1]:
                    IJ_trav=IJ_in[n].copy()
                    while True:
                        # 1-coord, as we want to move along the constant contour of coord.
                        IJ_trav[1-coord]+=incr
                        if tuple(IJ_trav) in IJ_to_n:
                            n_trav=IJ_to_n[tuple(IJ_trav)]
                            if np.isfinite( ij_out[n_trav,coord] ):
                                assert ij_out[n_trav,coord]==val,"Encountered incompatible IJ"
                            else:
                                ij_out[n_trav,coord]=val
                        else:
                            break

            # just one coordinate at a time
            valid=np.isfinite( ij_out[:,coord] )
            interp_IJ_to_ij=utils.LinearNDExtrapolator(IJ_in[valid,:], ij_out[valid,coord])
            ij_out[~valid,coord] = interp_IJ_to_ij(IJ_in[~valid,:])
        return ij_out

    # --- Patch Construction ---
    def map_fixed_int_to_gen(self,g_int,gen):
        """
        Return a dictionary mapping nodes of self.g_int to fixed nodes of 
        self.gen
        """
        # This code assumes that either ij are both fixed, or neither fixed.
        fixed_int_to_gen={}
        for n in g_int.valid_node_iter():
            g_n=g_int.nodes['gen_n'][n]
            if (g_n>=0) and (gen.nodes['fixed'][g_n]):
                fixed_int_to_gen[n]=g_n
        return fixed_int_to_gen

    def create_final_by_patches(self):
        fixed_int_to_gen = self.map_fixed_int_to_gen(self.g_int,self.gen)
        n_fixed=list(fixed_int_to_gen.keys())

        g_int=self.g_int
        angles=np.zeros(g_int.Nedges(),np.float32)
        angles=np.where( g_int.edges['gen_j']>=0,
                         self.gen.edges['angle'][g_int.edges['gen_j']],
                         np.nan )
        g_int.add_edge_field('angle',angles,on_exists='overwrite')

        # misnomer.  Not final.  Just for finding exact intersections
        g_final=exact_delaunay.Triangulation(extra_edge_fields=[
            #('dij',np.float64,2),
            #('ij',np.float64,2),
            ('angle',np.float64),
            ('psiphi',np.float64,2)])

        # g_final.edge_defaults['dij']=np.nan
        # Not great - when edges get split, this will at least leave the fields as nan
        # instead of 0.
        # g_final.edge_defaults['ij']=np.nan
        g_final.edge_defaults['psiphi']=np.nan
        g_final.edge_defaults['angle']=np.nan

        def trace_contour(b,angle):
            """
            angle: 0 is constant psi, with psi increasing to left
            """
            if angle==90:
                # trace constant phi
                node_field=self.phi # the field to trace a contour of
                cval_pos='right' # guess and check
            elif angle==270:
                node_field=self.phi # the field to trace a contour of
                cval_pos='left'
            elif angle==0:
                node_field=self.psi
                cval_pos='left' # guess and check
            elif angle==180:
                node_field=self.psi
                cval_pos='right'
            else:
                raise Exception("what?")
            cval=node_field[b]
            return g_int.trace_node_contour(n0=b,cval=cval,
                                            node_field=node_field,
                                            pos_side=cval_pos,
                                            return_full=True)

        # g_final node index =>
        # list of [
        #   ( dij, from the perspective of leaving the node,
        #     'internal' or 'boundary' )
        node_exits=defaultdict(list)

        def insert_contour(trace_items,angle=None,
                           psiphi0=[np.nan,np.nan]):
            assert np.isfinite(psiphi0[0]) or np.isfinite(psiphi0[1])

            trace_points=np.array( [pnt
                                    for typ,idx,pnt in trace_items
                                    if pnt is not None])

            # Check whether the ends need to be forced into the boundary
            # but here we preemptively doctor up the ends
            for i in [0,-1]:
                if trace_items[i][0]=='edge':
                    # When it hits a non-cartesian edge this will fail (which is okay)
                    # Feels a bit fragile:
                    j_int=trace_items[i][1].j # it's a halfedge
                    j_gen=g_int.edges['gen_j'][j_int] # from this original edge
                    angle_gen=self.gen.edges['angle'][j_gen]
                    if angle_gen%90 != 0:
                        print("Not worrying about contour hitting diagonal")
                        continue

                    # Force that point into an existing constrained edge of g_final
                    pnt=trace_points[i]
                    best=[None,np.inf]
                    for j in np.nonzero(g_final.edges['constrained'] & (~g_final.edges['deleted']))[0]:
                        d=utils.point_segment_distance( pnt,
                                                        g_final.nodes['x'][g_final.edges['nodes'][j]] )
                        if d<best[1]:
                            best=[j,d]
                    j,d=best
                    # Typ. 1e-10 when using UTM coordinates
                    assert d<1e-5
                    if d>0.0:
                        n_new=g_final.split_constraint(x=pnt,j=j)

            trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert',
                                                                       on_exists='stop')
            if angle is not None:
                g_final.edges['angle'][trace_edges]=angle
            #if ij0 is not None:
            #    g_final.edges['ij'][trace_edges]=ij0
            if psiphi0 is not None:
                g_final.edges['psiphi'][trace_edges]=psiphi0

            # Update node_exits:
            exit_angle=angle
            for a in trace_nodes[:-1]:
                node_exits[a].append( (exit_angle,'internal') )
            if angle is not None:
                angle=(angle+180)%360
            for b in trace_nodes[1:]:
                node_exits[b].append( (exit_angle,'internal') )

        def trace_and_insert_contour(b,angle):
            # does dij_angle fall between the angles formed by the boundary, including
            # a little slop.
            print(f"{angle} looks good")
            gn=fixed_int_to_gen[b] # below we already check to see that b is in there.

            # ij0=self.gen.nodes['ij'][gn].copy()
            # only pass the one constant along the contour
            if angle%180==0:
                psiphi0=[self.psi[b],np.nan]
            elif angle%180==90:
                psiphi0=[np.nan,self.phi[b]]

            trace_items=trace_contour(b,angle=angle)
            return insert_contour(trace_items,angle=angle,
                                  psiphi0=psiphi0)

        def trace_and_insert_boundaries(cycle):
            for a,b in utils.progress( zip( cycle, np.roll(cycle,-1) )):
                j=g_int.nodes_to_edge(a,b)
                angle=g_int.edges['angle'][j] # angle from a to b
                if angle%90!=0: continue # ragged edge

                trace_points=g_int.nodes['x'][[a,b]]
                trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
                g_final.edges['angle'][trace_edges]=angle

                # Update node_exits, which are referenced by nodes in g_final
                for a_fin in trace_nodes[:-1]:
                    node_exits[a_fin].append( (angle,'boundary') )
                opp_angle=(angle+180)%360
                for b_fin in trace_nodes[1:]:
                    node_exits[b_fin].append( (opp_angle,'boundary') )

                # This used to also fill in ij, but we don't have that now.
                # need to update psiphi for these edges, too.
                if angle%180==0: # psi constant
                    psiphi=[self.psi[a],np.nan]
                elif angle%180==90:
                    psiphi=[np.nan, self.phi[a]]
                else:
                    assert False

                g_final.edges['psiphi'][trace_edges]=psiphi

        # Add boundaries when they coincide with contours
        cycle=g_int.boundary_cycle() # could be multiple eventually...

        print("Tracing boundaries...",end="")
        trace_and_insert_boundaries(cycle)
        print("done")

        # Need to get all of the boundary contours in first, then
        # return with internal.
        for a,b,c in zip(cycle,
                         np.roll(cycle,-1),
                         np.roll(cycle,-2)):
            # if b==290: # side-channel
            #     g_int.plot_nodes(mask=g_int.nodes['rigid'],labeler='id')
            #     g_int.plot_nodes(mask=[a,c], labeler='id')
            #     g_int.plot_edges(mask=[j_ab,j_bc],labeler='angle')
            #     import pdb
            #     pdb.set_trace()

            if b not in fixed_int_to_gen: continue

            j_ab=g_int.nodes_to_edge(a,b)
            j_bc=g_int.nodes_to_edge(b,c)
            # flip to be the exit angle
            angle_ba = (180+g_int.edges['angle'][j_ab])%360
            angle_bc = g_int.edges['angle'][j_bc]

            for angle in [0,90,180,270]:
                # is angle into the domain?
                trace=None

                # if angle is left of j_ab and right of j_bc,
                # then it should be into the domain and can be traced
                # careful with sting angles
                # a,b,c are ordered CCW on the cycle, domain is to the
                # left.
                # so I want bc - angle - ba to be ordered CCW
                if ( ((angle_bc==angle_ba) and (angle!=angle_bc))
                     or (angle-angle_bc)%360 < ((angle_ba-angle_bc)%360) ):
                    b_final=g_final.select_nodes_nearest(g_int.nodes['x'][b],max_dist=0.0)
                    dupe=False
                    if b_final is not None:
                        for exit_angle,exit_type in node_exits[b_final]:
                            if exit_angle==angle:
                                dupe=True
                                print("Duplicate exit for internal trace from %d. Skip"%b)
                                break
                    if not dupe:
                        trace_and_insert_contour(b,angle)

        def tri_to_grid(g_final):
            g_final2=g_final.copy()

            for c in g_final2.valid_cell_iter():
                g_final2.delete_cell(c)

            for j in np.nonzero( (~g_final2.edges['deleted']) & (~g_final2.edges['constrained']))[0]:
                g_final2.delete_edge(j)

            g_final2.modify_max_sides(2000)
            g_final2.make_cells_from_edges()
            return g_final2

        g_final2=tri_to_grid(g_final)

        if 1: 
            # Add any diagonal edges here, so that ragged edges
            # get cells in g_final2, too.
            ragged=np.isfinite(g_int.edges['angle']) & (g_int.edges['angle']%90!=0.0)
            j_ints=np.nonzero( ragged )[0]
            for j_int in j_ints:
                nodes=[g_final2.add_or_find_node(g_int.nodes['x'][n])
                       for n in g_int.edges['nodes'][j_int]]
                j_fin2=g_final2.nodes_to_edge(nodes)
                angle=g_int.edges['angle'][j_int]
                if j_fin2 is None:
                    j_fin2=g_final2.add_edge(nodes=nodes,constrained=True,
                                             angle=angle,
                                             psiphi=[np.nan,np.nan])

            g_final2.make_cells_from_edges()

        # Patch grid g_final2 completed.
        # fixed the missing the ragged edge.

        # plt.clf()
        # g_final2.plot_edges()
        # plt.draw()

        #import pdb
        #pdb.set_trace()

        # --- Compile Swaths ---
        e2c=g_final2.edge_to_cells(recalc=True)

        i_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)
        j_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)

        for j in g_final2.valid_edge_iter():
            c1,c2=e2c[j,:]
            if c1<0 or c2<0: continue

            # if the di of dij is 0, the edge joins cell in i_adj
            # I think angle==0 is the same as dij=[1,0]

            #if g_final2.edges['dij'][j,0]==0:
            if g_final2.edges['angle'][j] % 180==0: # guess failed.
                i_adj[c1,c2]=i_adj[c2,c1]=True
            elif g_final2.edges['angle'][j] % 180==90:
                j_adj[c1,c2]=j_adj[c2,c1]=True
            else:
                print("What? Ragged edge okay, but it shouldn't have both cell neighbors")

        n_comp_i,labels_i=sparse.csgraph.connected_components(i_adj.astype(np.int32),directed=False)
        n_comp_j,labels_j=sparse.csgraph.connected_components(j_adj,directed=False)

        # preprocessing for contour placement
        nd=NodeDiscretization(g_int)
        Mdx,Bdx=nd.construct_matrix(op='dx')
        Mdy,Bdy=nd.construct_matrix(op='dy')
        psi_dx=Mdx.dot(self.psi)
        psi_dy=Mdy.dot(self.psi)
        phi_dx=Mdx.dot(self.phi)
        phi_dy=Mdy.dot(self.phi)

        # These should be about the same.  And they are, but
        # keep them separate in case the psi_phi solution procedure
        # evolves.
        psi_grad=np.sqrt( psi_dx**2 + psi_dy**2)
        phi_grad=np.sqrt( phi_dx**2 + phi_dy**2)

        pp_grad=[psi_grad,phi_grad]

        # Just figures out the contour values and sets them on the patches.
        patch_to_contour=[{},{}] # coord, cell index=>array of contour values

        def add_swath_contours_new(comp_cells,node_field,coord,scale):
            # Check all of the nodes to find the range ij
            comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
            comp_nodes=np.unique( np.concatenate(comp_nodes) )
            comp_ijs=[] # Certainly could have kept this info along while building...

            field_values=[]

            # plt.figure(2).clf()
            # g_final2.plot_edges(color='k',lw=0.5)
            # g_final2.plot_cells(mask=comp_cells)
            # # g_final2.plot_nodes(mask=comp_nodes)
            # plt.draw()

            # comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
            comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])

            # it's actually the other coordinate that we want to consider.
            field_min=np.nanmin( comp_pp[:,1-coord] )
            field_max=np.nanmax( comp_pp[:,1-coord] )

            # coord_min=np.nanmin( comp_ij[:,1-coord] )
            # coord_max=np.nanmax( comp_ij[:,1-coord] )

            # Could do this more directly from topology if it mattered..
            swath_poly=ops.cascaded_union( [g_final2.cell_polygon(c) for c in comp_cells] )
            swath_nodes=g_int.select_nodes_intersecting(swath_poly)
            swath_vals=node_field[swath_nodes]
            swath_grad=pp_grad[1-coord][swath_nodes] # right?
            order=np.argsort(swath_vals)
            o_vals=swath_vals[order]
            o_dval_ds=swath_grad[order]
            o_ds_dval=1./o_dval_ds

            # trapezoid rule integration
            d_vals=np.diff(o_vals)
            # Particularly near the ends there are a lot of
            # duplicate swath_vals.
            # Try a bit of lowpass to even things out.
            if 1:
                winsize=int(len(o_vals)/5)
                if winsize>1:
                    o_ds_dval=filters.lowpass_fir(o_ds_dval,winsize)

            s=np.cumsum(d_vals*0.5*(o_ds_dval[:-1]+o_ds_dval[1:]))
            s=np.r_[0,s]

            # HERE -- calculate this from resolution
            # might have i/j swapped.  range of s is 77m, and field
            # is 1. to 1.08.  better now..
            local_scale=scale( g_int.nodes['x'][swath_nodes] ).mean(axis=0)
            n_swath_cells=int(np.round( (s.max() - s.min())/local_scale))
            n_swath_cells=max(1,n_swath_cells)

            s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
            adj_contours=np.interp( s_contours,
                                    s,o_vals)
            adj_contours[0]=field_min
            adj_contours[-1]=field_max

            for c in comp_cells:
                patch_to_contour[coord][c]=adj_contours

        if 1: # Swath processing
            for coord in [0,1]: # i/j
                print("Coord: ",coord)
                if coord==0:
                    labels=labels_i
                    n_comp=n_comp_i
                    node_field=self.phi # feels backwards.. it's right, just misnamed 
                else:
                    labels=labels_j
                    n_comp=n_comp_j
                    node_field=self.psi

                for comp in range(n_comp):
                    print("Swath: ",comp)
                    comp_cells=np.nonzero(labels==comp)[0]
                    add_swath_contours_new(comp_cells,node_field,coord,self.scales[coord])

        # Direct grid gen from contour specifications:
        patch_grids=[]

        g_int.edge_to_cells()

        for c in utils.progress(g_final2.valid_cell_iter()):
            psi_cvals=patch_to_contour[1][c]
            phi_cvals=patch_to_contour[0][c]

            g_patch=unstructured_grid.UnstructuredGrid(max_sides=4)
            g_patch.add_rectilinear( [0,0], [len(psi_cvals)-1,len(phi_cvals)-1],
                                     len(psi_cvals),len(phi_cvals))
            g_patch.add_node_field( 'ij', g_patch.nodes['x'].astype(np.int32))
            pp=np.c_[ psi_cvals[g_patch.nodes['ij'][:,0]],
                      phi_cvals[g_patch.nodes['ij'][:,1]] ]
            g_patch.add_node_field( 'pp', pp)

            x0=g_final2.cells_centroid([c])[0]
            for n in g_patch.valid_node_iter():
                x=g_int.fields_to_xy(g_patch.nodes['pp'][n],
                                     node_fields=[self.psi,self.phi],
                                     x0=x0)
                if np.isnan(x[0]):
                    # If it's a ragged cell, probably okay.
                    edge_angles=g_final2.edges['angle'][ g_final2.cell_to_edges(c) ]
                    ragged_js=(edge_angles%90!=0.0)
                    if np.any(ragged_js):
                        print("fields_to_xy() failed, but cell is ragged.")
                        g_patch.delete_node_cascade(n)
                    else:
                        print("ERROR: fields_to_xy() failed. Cell not ragged.")
                g_patch.nodes['x'][n]=x
                # Hmm -
                # When it works, this probably reduces the search time considerably,
                # but there is the possibility, particularly at corners, that
                # this x will be a corner, that corner will lead to the cell *around*
                # the corner, and then we get stuck.
                # Even the centroid isn't great since it might not even fall inside
                # the cell.
                # x0=x 
            patch_grids.append(g_patch)

        g=patch_grids[0]
        for g_next in patch_grids[1:]:
            g.add_grid(g_next,merge_nodes='auto',tol=1e-6)

        return g

