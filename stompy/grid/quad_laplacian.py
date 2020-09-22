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
from shapely import geometry
from scipy import sparse

import matplotlib.pyplot as plt

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
        self.g_final.plot_edges()
        plt.axis('tight')
        plt.axis('equal')
    
class QuadGen(object):
    
    # default behavior computes a nominal, isotropic grid for the calculation
    #  of the orthogonal mapping, and then a separate anisotropic grid for the
    #  final result. If anisotropic is False, the isotropic grid is kept, 
    #  and its ij indices will be updated to reflect the anisotropic inputs.
    final='anisotropic' # 'isotropic', 'triangle'

    patchwise=True # use patch-by-patch method for constructing quad grid from psi/phi field.
    
    # The cell spacing in geographic coordinates for the nominal, isotropic grid
    nom_res=4.0
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
    
    def __init__(self,gen,execute=True,cell=None,**kw):
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

        if cell is not None:
            for c in range(gen.Ncells()):
                if c!=cell:
                    gen.delete_cell(c)
            gen.delete_orphan_edges()
            gen.delete_orphan_nodes()
            gen.renumber(reorient_edges=False)
        
        self.gen=gen
        # list of node pairs, referencing nodes in gen, which provide
        # additional groupings of nodes.
        self.internal_edges=[]

        # Prep the target resolution grid information
        self.coalesce_ij(self.gen)
        self.fill_ij_interp(self.gen)
        self.node_ij_to_edge(self.gen)

        # Prep the nominal resolution grid information
        self.coalesce_ij_nominal(self.gen,dest='IJ')
        self.fill_ij_interp(self.gen,dest='IJ')
        self.node_ij_to_edge(self.gen,dest='IJ')

        if execute:
            self.execute()

    def add_internal_edge(self,nodes):
        self.internal_edges.append(nodes)
        
    def execute(self):
        self.add_bezier(self.gen)
        if self.intermediate=='quad':
            self.g_int=self.create_intermediate_grid_quad(src='IJ')
            # This now happens as a side effect of smooth_interior_quads
            # self.adjust_intermediate_bounds()
            self.smooth_interior_quads(self.g_int)
        elif self.intermediate=='tri':
            self.g_int=self.create_intermediate_grid_tri(src='IJ')
        self.calc_psi_phi()
        if self.final=='anisotropic':
            if self.patchwise:
                self.g_final=self.create_final_by_patches()
            else:
                self.g_final=self.create_intermediate_grid_quad(src='ij',coordinates='ij')
                self.adjust_by_psi_phi(self.g_final, src='ij')
        elif self.final=='isotropic':
            if self.intermediate=='tri':
                self.g_final=self.create_intermediate_grid_quad(src='IJ',coordinates='ij')
            else:
                self.g_final=self.g_int.copy()
            self.adjust_by_psi_phi(self.g_final, src='IJ')
            ij=self.remap_ij(self.g_final,src='ij')
            self.g_final.nodes['ij']=ij
        elif self.final=='triangle':
            # Assume that nobody wants to build a quad grid for the intermediate calcs,
            # but then map it onto a triangle grid. I suppose it could be done, but more
            # likely it's an error.
            assert self.intermediate=='tri'
            self.g_final=self.g_int.copy()
            map_pp_to_ij=self.psiphi_to_ij(self.gen,self.g_int)
            ij=map_pp_to_ij( np.c_[self.psi,self.phi])
            self.g_final.nodes['ij']=ij

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

    def create_intermediate_grid_quad(self,src='ij',coordinates='xy'):
        """
        src: base variable name for the ij indices to use.
          i.e. gen.nodes['ij'], gen.nodes['ij_fixed'],
             and gen.edges['dij']

          the resulting grid will use 'ij' regardless, this just for the
          generating grid.

        coordinates: 
         'xy' will interpolate the gen xy coordinates to get
          node coordinates for the result.
         'ij' will leave 'ij' coordinate values in both x and 'ij'
        
        """
        # target grid
        g=unstructured_grid.UnstructuredGrid(max_sides=4,
                                             extra_node_fields=[('ij',np.float64,2),
                                                                ('gen_j',np.int32),
                                                                ('rigid',np.int32)])
        gen=self.gen
        for c in gen.valid_cell_iter():
            local_edges=gen.cell_to_edges(c,ordered=True)
            flip=(gen.edges['cells'][local_edges,0]!=c)
            
            edge_nodes=gen.edges['nodes'][local_edges]
            edge_nodes[flip,:] = edge_nodes[flip,::-1]

            dijs=gen.edges['d'+src][local_edges] * ((-1)**flip)[:,None]
            xys=gen.nodes['x'][edge_nodes[:,0]]
            ij0=gen.nodes[src][edge_nodes[0,0]]
            ijs=np.cumsum(np.vstack([ij0,dijs]),axis=0)

            # Sanity check to be sure that all the dijs close the loop.
            assert np.allclose( ijs[0],ijs[-1] )

            ijs=np.array(ijs[:-1])
            # Actually don't, so that g['ij'] and gen['ij'] match up.
            # ijs-=ijs.min(axis=0) # force to have ll corner at (0,0)
            ij0=ijs.min(axis=0)
            ijN=ijs.max(axis=0)
            ij_size=ijN-ij0

            # Create in ij space
            patch=g.add_rectilinear(p0=ij0,
                                    p1=ijN,
                                    nx=int(1+ij_size[0]),
                                    ny=int(1+ij_size[1]))
            pnodes=patch['nodes'].ravel()

            g.nodes['gen_j'][pnodes]=-1

            # Copy xy to ij, then optionally remap xy
            g.nodes['ij'][pnodes] = g.nodes['x'][pnodes]

            if coordinates=='xy':
                Extrap=utils.LinearNDExtrapolator

                # There is a danger that a triangulation of ij
                # is not valid in xy space.
                # Would it work to build the triangulation in xy space,
                # but then use ij coordinates
                int_x=Extrap(ijs,xys[:,0])
                node_x=int_x(g.nodes['x'][pnodes,:])

                int_y=Extrap(ijs,xys[:,1])
                node_y=int_y(g.nodes['x'][pnodes,:])

                g.nodes['x'][pnodes]=np.c_[node_x,node_y]

            ij_poly=geometry.Polygon(ijs)
            # This should be fairly robust since these are mostly
            # axis-aligned, with integer coordinates.  But exact
            # comparisons make me nervous, so give it 0.001.
            ij_poly=ij_poly.buffer(0.001)

            if 1:
                # Previously trimmed only based on cell center (ish)
                # but it's the node locations which have to be
                # calculated, so better to trim based on node location
                # too.  Do both, since we could have a narrow crannie
                # that splits a cell but has all valid nodes.  Not likely,
                # but still..
                for n in patch['nodes'].ravel():
                    if not ij_poly.contains(geometry.Point(g.nodes['ij'][n])):
                        g.delete_node_cascade(n)
            if 1:
                for cc in patch['cells'].ravel():
                    if g.cells['deleted'][cc]: continue
                    cn=g.cell_to_nodes(cc)
                    c_ij=np.mean(g.nodes['ij'][cn],axis=0)
                    if not ij_poly.contains(geometry.Point(c_ij)):
                        g.delete_cell(cc)
                g.delete_orphan_edges()
                g.delete_orphan_nodes()

            # Mark nodes as rigid if they match a point in the generator
            for n in g.valid_node_iter():
                match0=gen.nodes[src][:,0]==g.nodes['ij'][n,0]
                match1=gen.nodes[src][:,1]==g.nodes['ij'][n,1]
                match=np.nonzero(match0&match1)[0]
                if len(match):
                    # Something is amiss, but this part looks okay in pdb
                    # import pdb
                    # pdb.set_trace()
                    g.nodes['rigid'][n]=RIGID

            # Fill in generating edges for boundary nodes
            boundary_nodes=g.boundary_cycle()
            # hmm -
            # each boundary node in g sits at either a node or
            # edge of gen.
            # For any non-rigid node in g, it should sit on
            # an edge of gen.  ties can go either way, doesn't
            # matter (for bezier purposes)
            # Can int_x/y help here?
            # or just brute force it
            
            local_edge_ijs=np.array( [ ijs, np.roll(ijs,-1,axis=0)] )
            lower_ij=local_edge_ijs.min(axis=0)
            upper_ij=local_edge_ijs.max(axis=0)
            
            for n in boundary_nodes:
                n_ij=g.nodes['ij'][n]

                # [ {nA,nB}, n_local_edges, {i,j}]
                candidates=np.all( (n_ij>=lower_ij) & (n_ij<=upper_ij),
                                   axis=1)
                best_lj=None
                best_offset=np.inf
                
                for lj in np.nonzero(candidates)[0]:
                    # is n_ij approximately on the line
                    # local_edge_ijs[lj] ?
                    offset=utils.point_line_distance(n_ij,local_edge_ijs[:,lj,:])
                    if offset<best_offset:
                        best_lj=lj
                        best_offset=offset
                if best_lj is None:
                    raise Exception("Failed to match up a boundary node: no candidates")
                dij = np.diff( local_edge_ijs[:,lj,:], axis=0)
                if np.all(dij!=0): # ragged edge
                    allowable=self.max_ragged_node_offset
                else:
                    allowable=0.1
                if best_offset>allowable:
                    raise Exception("Failed to match up a boundary node: offset %.2f > allowable %.2f"%
                                    (best_offset,allowable))
                g.nodes['gen_j'][n]=local_edges[lj]
                
        g.renumber()
        return g

    def create_intermediate_grid_tri_boundary(self,src='ij',coordinates='xy',scale=None):
        """
        Create the boundaries for the intermediate grid, upsampling the bezier edges
        and assigning 'ij' along the way for fixed nodes.
        """
        if scale is None:
            scale=field.ConstantField(self.nom_res)
            
        g=unstructured_grid.UnstructuredGrid(max_sides=3,
                                             extra_edge_fields=[ ('gen_j',np.int32) ],
                                             extra_node_fields=[ ('ij',np.float64,2) ])
        g.nodes['ij']=np.nan
        g.node_defaults['ij']=np.nan

        gen=self.gen
        
        for j in gen.valid_edge_iter():
            # Just to get the length
            points=self.gen_bezier_linestring(j=j,samples_per_edge=10,span_fixed=False)
            dist=utils.dist_along(points)[-1]
            local_res=scale(points).min(axis=0) # min=>conservative
            N=max( self.min_steps, int(dist/local_res))
            points=self.gen_bezier_linestring(j=j,samples_per_edge=N,span_fixed=False)

            # Figure out what IJ to assign:
            ij0=gen.nodes[src][gen.edges['nodes'][j,0]]
            ijN=gen.nodes[src][gen.edges['nodes'][j,1]]

            nodes=[]
            for p_i,p in enumerate(points):
                n=g.add_or_find_node(x=p,tolerance=0.1)
                alpha=p_i/(len(points)-1.0)
                assert alpha>=0
                assert alpha<=1
                ij=(1-alpha)*ij0 + alpha*ijN
                g.nodes['ij'][n]=ij
                nodes.append(n)

            for a,b in zip(nodes[:-1],nodes[1:]):
                g.add_edge(nodes=[a,b],gen_j=j)
        return g
    
    def create_intermediate_grid_tri(self,src='ij',coordinates='xy'):
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

        g=self.create_intermediate_grid_tri_boundary(src=src,coordinates=coordinates)

        # seed=gen.cells_centroid()[0]
        # This is more robust
        nodes=g.find_cycles(max_cycle_len=5000)[0]
        
        # This will suffice for now.  Probably can use something
        # less intense.
        gnew=triangulate_hole.triangulate_hole(g,nodes=nodes,hole_rigidity='all',
                                               method=self.triangle_method)
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

        for n in gen.valid_node_iter():
            js=gen.node_to_edges(n)
            assert len(js)==2
            # orient the edges
            njs=[]
            deltas=[]
            dijs=[]
            flips=[]
            for j in js:
                nj=gen.edges['nodes'][j]
                dij=gen.edges['dij'][j]
                flip=0
                if nj[0]!=n:
                    nj=nj[::-1]
                    dij=-dij
                    flip=1
                assert nj[0]==n
                njs.append(nj)
                dijs.append(dij)
                flips.append(flip)
                deltas.append( gen.nodes['x'][nj[1]] - gen.nodes['x'][nj[0]] )
            # now node n's two edges are in njs, as node pairs, with the first
            # in each pair being n
            # dij is the ij delta along that edge
            # flip records whether it was necessary to flip the edge
            # and deltas records the geometry delta
            
            # the angle in ij space tells us what it *should* be
            # these are angles going away from n
            # How does this work out when it's a straight line in ij space?
            theta0_ij=np.arctan2( -dijs[0][1], -dijs[0][0])
            theta1_ij=np.arctan2(dijs[1][1],dijs[1][0]) 
            dtheta_ij=(theta1_ij - theta0_ij + np.pi) % (2*np.pi) - np.pi

            theta0=np.arctan2(-deltas[0][1],-deltas[0][0])
            theta1=np.arctan2(deltas[1][1],deltas[1][0])
            dtheta=(theta1 - theta0 + np.pi) % (2*np.pi) - np.pi

            theta_err=dtheta-dtheta_ij
            # Make sure we're calculating error in the shorter direction
            theta_err=(theta_err+np.pi)%(2*np.pi) - np.pi
            
            cp0 = gen.nodes['x'][n] + utils.rot( theta_err/2, 1./3 * deltas[0] )
            cp1 = gen.nodes['x'][n] + utils.rot( -theta_err/2, 1./3 * deltas[1] )

            # save to the edge
            gen.edges['bez'][js[0],1+flips[0]] = cp0
            gen.edges['bez'][js[1],1+flips[1]] = cp1

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
        for ji,(n1,n2) in enumerate( zip(bcycle[:-1],bcycle[1:]) ):
            tang_xy=utils.to_unit( gtri.nodes['x'][n2] - gtri.nodes['x'][n1] )
            tang_ij=utils.to_unit( gtri.nodes['ij'][n2] - gtri.nodes['ij'][n1] )

            # Construct a rotation R such that R.dot(tang_ij)=[1,0],
            # then apply to tang_ij
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
    
    def calc_psi_phi(self,gtri=None):
        if gtri is None:
            gtri=self.g_int
        self.nd=nd=NodeDiscretization(gtri)

        e2c=gtri.edge_to_cells()

        # psi and phi are both computed by solving the Laplacian
        # on the intermediate grid. Input values of i,j in the input
        # are used to identify strings of boundary nodes with the same
        # value (zero tangential gradient), and this constraint is
        # encoded in the matrix. This leaves the system under-determined,
        # 2*nedges too few constraints.  Three additional constraints
        # come from setting the scale and location of psi and the location
        # of phi.
        # It is still a bit unclear what the remaining degrees of freedom
        # are, but they can, in practice, be eliminated by additionally
        # the coupling terms d psi /dy ~ d phi/dx, and vice versa.

        # One approach would be to split the problem into constraints and
        # costs.  Then the known BCs and Laplacians can be constraints,
        # and the remaining DOFs can be solved as a least-squares problem.
        # This is "equality-constrained linear least squares"
        # For dense matrices:
        # With Cx=d constraint, minimize Ax=b
        #  from scipy.linalg import lapack
        #  # Define the matrices as usual, then
        #  x = lapack.dgglse(A, C, b, d)[3]
        # And there is ostensibly a way to do this by solving an augmented
        # system
        
        # check boundaries and determine where Laplacian BCs go
        boundary=e2c.min(axis=1)<0
        i_dirichlet_nodes={} # for psi
        j_dirichlet_nodes={} # for phi

        # Block of nodes with a zero-tangential-gradient BC
        i_tan_groups=[]
        j_tan_groups=[]
        i_tan_groups_i=[] # the input i value
        j_tan_groups_j=[] # the input j value

        # Try zero-tangential-gradient nodes.  Current code will be under-determined
        # without the derivative constraints.
        bcycle=gtri.boundary_cycle()
        n1=bcycle[-1]
        i_grp=None
        j_grp=None

        psi_gradients,phi_gradients=self.calc_bc_gradients(gtri)
        psi_gradient_nodes={} # node => unit vector of gradient direction
        phi_gradient_nodes={} # node => unit vector of gradient direction

        for n2 in bcycle:
            i1=gtri.nodes['ij'][n1,0]
            i2=gtri.nodes['ij'][n2,0]
            j1=gtri.nodes['ij'][n1,1]
            j2=gtri.nodes['ij'][n2,1]
            imatch=np.allclose(i1,i2) # too lazy to track down how i'm getting a 2e-12 offset
            jmatch=np.allclose(j1,j2)
            
            if imatch: 
                if i_grp is None:
                    i_grp=[n1]
                    i_tan_groups.append(i_grp)
                    i_tan_groups_i.append(i1)
                i_grp.append(n2)
            else:
                i_grp=None

            if jmatch:
                if j_grp is None:
                    j_grp=[n1]
                    j_tan_groups.append(j_grp)
                    j_tan_groups_j.append(j1)
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
            
        # Set the range of psi to [-1,1], and pin some j to 1.0
        low_i=np.argmin(i_tan_groups_i)
        high_i=np.argmax(i_tan_groups_i)

        i_dirichlet_nodes[i_tan_groups[low_i][0]]=-1
        i_dirichlet_nodes[i_tan_groups[high_i][0]]=1
        j_dirichlet_nodes[j_tan_groups[1][0]]=1

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
            edge=[self.g_int.select_nodes_nearest(x)
                  for x in self.gen.nodes['x'][gen_edge]]
            edge_ij=self.gen.nodes['ij'][gen_edge]
            dij=np.abs( edge_ij[1] - edge_ij[0] )
            
            if dij[0]<1e-10: # join on i
                print("Joining two i_tan_groups")
                i_tan_groups=join_groups(i_tan_groups,edge[0],edge[1])
            elif dij[1]<1e-10: # join on j
                print("Joining two j_tan_groups")
                j_tan_groups=join_groups(j_tan_groups,edge[0],edge[1])
            else:
                import pdb
                pdb.set_trace()
                print("Internal edge doesn't appear to join same-valued contours")

        if 0: # DBG
            print("i_dirichlet_nodes:",i_dirichlet_nodes)
            print("i_tan_groups:",i_tan_groups)
            print("j_dirichlet_nodes:",j_dirichlet_nodes)
            print("j_tan_groups:",j_tan_groups)

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

    def plot_psi_phi(self,num=4,thinning=2,ax=None):
        if ax is None:
            plt.figure(num).clf()
            fig,ax=plt.subplots(num=num)

        di,dj=np.nanmax(self.gen.nodes['ij'],axis=0) - np.nanmin(self.gen.nodes['ij'],axis=0)

        self.g_int.plot_edges(color='k',lw=0.5,alpha=0.2)
        cset_psi=self.g_int.contour_node_values(self.psi,int(di/thinning),
                                                linewidths=1.5,linestyles='solid',colors='orange',
                                                ax=ax)
        cset_phi=self.g_int.contour_node_values(self.phi,int(dj/thinning),
                                                linewidths=1.5,linestyles='solid',colors='blue',
                                                ax=ax)
        ax.axis('tight')
        ax.axis('equal')

        ax.clabel(cset_psi, fmt="i=%g", fontsize=10, inline=False, use_clabeltext=True)
        ax.clabel(cset_phi, fmt="j=%g", fontsize=10, inline=False, use_clabeltext=True)
        
    def adjust_by_psi_phi(self,g,update=True,src='ij'):
        """
        Move internal nodes of g according to phi and psi fields

        update: if True, actually update g, otherwise return the new values

        g: The grid to be adjusted. Must have nodes['ij'] filled in fully.

        src: the ij coordinate field in self.gen to use.  Note that this needs to be
          compatible with the ij coordinate field used to create g.
        """
        # Check to be sure that src and g['ij'] are approximately compatible.
        assert np.allclose( g.nodes['ij'].min(), self.gen.nodes[src].min() )
        assert np.allclose( g.nodes['ij'].max(), self.gen.nodes[src].max() )

        map_ij_to_pp = self.psiphi_to_ij(self.gen,self.g_int,inverse=True,src=src)

        # Calculate the psi/phi values on the nodes of the target grid
        # (which happens to be the same grid as where the psi/phi fields were
        #  calculated)
        g_psiphi=map_ij_to_pp( g.nodes['ij'] )
        # g_psi=np.interp( g.nodes['ij'][:,0],
        #                  psi_i[:,0],psi_i[:,1])
        # g_phi=np.interp( g.nodes['ij'][:,1],
        #                  phi_j[:,0], phi_j[:,1])

        # Use gtri to go from phi/psi to x,y
        # I think this is where it goes askew.
        # This maps {psi,phi} space onto {x,y} space.
        # But psi,phi is close to rectilinear, and defined on a rectilinear
        # grid.  Whenever some g_psi or g_phi is close to the boundary,
        # the Delaunay triangulation is going to make things difficult.
        interp_xy=utils.LinearNDExtrapolator( np.c_[self.psi,self.phi],
                                              self.g_int.nodes['x'],
                                              eps=None)
        # Save all the pieces for debugging:
        self.interp_xy=interp_xy
        self.interp_domain=np.c_[self.psi,self.phi]
        self.interp_image=self.g_int.nodes['x']
        self.interp_tgt=g_psiphi
        
        new_xy=interp_xy( g_psiphi )

        if update:
            g.nodes['x']=new_xy
            g.refresh_metadata()
        else:
            return new_xy

    def plot_result(self,num=5):
        plt.figure(num).clf()
        self.g_final.plot_edges()
        plt.axis('equal')

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
            val=g_int.nodes['ij'][n,:]
            if np.isnan(val[0] + val[1]): continue
            # does it appear in gen?
            x=g_int.nodes['x'][n]
            gn=gen.select_nodes_nearest(x)
            gx=gen.nodes['x'][gn]
            delta=utils.dist( x-gx)
            if delta>0.01: continue
            if not np.any(gen.nodes['ij_fixed'][gn]): continue
            fixed_int_to_gen[n]=gn
        return fixed_int_to_gen
    
    def create_final_by_patches(self):
        fixed_int_to_gen = self.map_fixed_int_to_gen(self.g_int,self.gen)
        n_fixed=list(fixed_int_to_gen.keys())

        g_int=self.g_int

        # misnomer.  Not final.  Just for finding exact intersections
        g_final=exact_delaunay.Triangulation(extra_edge_fields=[('dij',np.float64,2),
                                                                ('ij',np.float64,2),
                                                                ('psiphi',np.float64,2)])
        g_final.edge_defaults['dij']=np.nan
        # Not great - when edges get split, this will at least leave the fields as nan
        # instead of 0.
        g_final.edge_defaults['ij']=np.nan
        g_final.edge_defaults['psiphi']=np.nan

        final_traces=[]

        def trace_contour(b,dij):
            if dij[0]!=0:
                # trace constant phi
                cval=self.phi[b] # The contour to trace
                node_field=self.phi # the field to trace a contour of
                if dij[0]<0:
                    cval_pos='left'
                else:
                    cval_pos='right' # guess and check
            elif dij[1]!=0:
                cval=self.psi[b]
                node_field=self.psi
                if dij[1]<0:
                    cval_pos='left' # guess and check
                else:
                    cval_pos='right' # guess and check
            else:
                raise Exception("what?")
            return g_int.trace_node_contour(n0=b,cval=cval,
                                            node_field=node_field,
                                            pos_side=cval_pos,
                                            return_full=True)

        # g_final node index =>
        # list of [
        #   ( dij, from the perspective of leaving the node,
        #     'internal' or 'boundary' )
        node_exits=defaultdict(list)

        def insert_contour(trace_items,dij=None,
                           psiphi0=[np.nan,np.nan],ij0=[np.nan,np.nan]):
            assert np.isfinite(ij0[0]) or np.isfinite(ij0[1])
            assert np.isfinite(psiphi0[0]) or np.isfinite(psiphi0[1])

            if dij is not None:
                dij=np.asarray(dij)

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
                    dij_gen=self.gen.edges['dij'][j_gen]
                    if (dij_gen!=0).sum()==2:
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
            if dij is not None:
                g_final.edges['dij'][trace_edges]=dij
            if ij0 is not None:
                g_final.edges['ij'][trace_edges]=ij0
            if psiphi0 is not None:
                g_final.edges['psiphi'][trace_edges]=psiphi0

            trace_data=dict(fin_nodes=trace_nodes,
                            fin_edges=trace_edges,
                            items=trace_items,
                            dij=dij,
                            psiphi0=psiphi0)
            final_traces.append(trace_data)

            # Update node_exits:
            exit_dij=dij
            for a in trace_nodes[:-1]:
                node_exits[a].append( (exit_dij,'internal') )
            if dij is not None:
                exit_dij=-dij
            for b in trace_nodes[1:]:
                node_exits[b].append( (exit_dij,'internal') )

        def trace_and_insert_contour(b,dij):
            # does dij_angle fall between the angles formed by the boundary, including
            # a little slop.
            print(f"{dij} looks good")
            gn=fixed_int_to_gen[b] # below we already check to see that b is in there.

            ij0=self.gen.nodes['ij'][gn].copy()
            # only pass the one constant along the contour
            if dij[0]==0:
                ij0[1]=np.nan
                psiphi0=[self.psi[b],np.nan]
            else:
                ij0[0]=np.nan
                psiphi0=[np.nan,self.phi[b]]

            trace_items=trace_contour(b,dij)
            return insert_contour(trace_items,dij=dij,
                                  psiphi0=psiphi0,
                                  ij0=ij0)

        def trace_boundary(b,dij):
            nodes=[b]
            while 1:
                nbrs=g_int.node_to_nodes(nodes[-1])
                last_ij=g_int.nodes['ij'][nodes[-1]]
                for n in nbrs:
                    nbr_dij=utils.to_unit( g_int.nodes['ij'][n] - last_ij )
                    if (dij*nbr_dij).sum() > 0.99:
                        nodes.append(n)
                        break
                else:
                    # no good neighbor -- end of while loop
                    break
            return nodes

        def trace_and_insert_boundary(b,dij):
            if dij is not None:
                dij=np.asarray(dij)

            trace_int_nodes=trace_boundary(b,dij)
            trace_int_edges=[g_int.nodes_to_edge(a,b)
                             for a,b in zip(trace_int_nodes[:-1],trace_int_nodes[1:])]
            trace_int_cells=[]

            trace_points=g_int.nodes['x'][trace_int_nodes]
            trace_nodes,trace_edges=g_final.add_constrained_linestring(trace_points,on_intersection='insert')
            g_final.edges['dij'][trace_edges]=dij
            # need to update ij,psiphi for these edges, too.
            gn=fixed_int_to_gen[b]
            if dij[0]==0:
                ij=[self.gen.nodes['ij'][gn,0],np.nan]
                psiphi=[self.psi[b],np.nan]
            elif dij[1]==0:
                ij=[np.nan, self.gen.nodes['ij'][gn,1]]
                psiphi=[np.nan, self.phi[b]]
            else:
                assert False
            g_final.edges['ij'][trace_edges]=ij
            g_final.edges['psiphi'][trace_edges]=psiphi

            trace_data=dict(int_nodes=trace_int_nodes,
                            int_cells=trace_int_cells,
                            int_edges=trace_int_edges,
                            fin_nodes=trace_nodes,
                            fin_edges=trace_edges,
                            dij=dij,
                            ij0=g_int.nodes['ij'][b])
            final_traces.append(trace_data)

            # Update node_exits:
            for a in trace_nodes[:-1]:
                node_exits[a].append( (dij,'boundary') )
            for b in trace_nodes[1:]:
                node_exits[b].append( (-dij,'boundary') )

        # Add boundaries when they coincide with contours
        cycle=g_int.boundary_cycle() # could be multiple eventually...

        # Need to get all of the boundary contours in first, then
        # return with internal.
        for mode in ['boundary','internal']:
            for a,b,c in zip(cycle,
                             np.roll(cycle,-1),
                             np.roll(cycle,-2)):
                if b not in fixed_int_to_gen: continue

                # First, should the edge a--b be included as a boundary edge coincident
                # with a contour?
                ij_a=g_int.nodes['ij'][a]
                ij_b=g_int.nodes['ij'][b]
                ij_c=g_int.nodes['ij'][c]

                ij_angle_ab=np.arctan2( ij_a[1] - ij_b[1],
                                        ij_a[0] - ij_b[0] )
                ij_angle_cb=np.arctan2( ij_c[1] - ij_b[1],
                                        ij_c[0] - ij_b[0] )

                for dij in [ [-1,0], [1,0], [0,-1],[0,1]]:
                    # is dij into the domain?
                    dij_angle=np.arctan2( dij[1],dij[0] )
                    trace=None
                    eps=1e-5

                    # If dij coincides with a boundary, trace it.
                    # Only check ij_angle_cb, since each boundary edge
                    # should have a fixed node at each end.
                    if np.abs( (dij_angle-ij_angle_cb+np.pi)%(2*np.pi)-np.pi)<eps:
                        if mode!='boundary': continue
                        print("Trace boundary")
                        trace_and_insert_boundary(b,dij)
                    elif ( ( (dij_angle-(ij_angle_cb+eps)) % (2*np.pi) )
                           < ( (ij_angle_ab-eps-ij_angle_cb)%(2*np.pi))):
                        if mode!='internal': continue
                        b_final=g_final.select_nodes_nearest(g_int.nodes['x'][b],max_dist=0.0)
                        dupe=False
                        if b_final is not None:
                            for exit_dij,exit_type in node_exits[b_final]:
                                if np.all( exit_dij==dij ):
                                    dupe=True
                                    print("Duplicate exit for internal trace from %d. Skip"%b)
                                    break
                        if not dupe:
                            trace_and_insert_contour(b,dij)

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
            ragged=np.nonzero( (self.gen.edges['dij']!=0.0).sum(axis=1)==2 )[0]
            for gen_j in ragged:
                j_ints=np.nonzero( g_int.edges['gen_j']==gen_j )[0]
                for j_int in j_ints:
                    nodes=[g_final2.add_or_find_node(g_int.nodes['x'][n])
                           for n in g_int.edges['nodes'][j_int]]
                    j_fin2=g_final2.nodes_to_edge(nodes)
                    if j_fin2 is None:
                        j_fin2=g_final2.add_edge(nodes=nodes,constrained=True,
                                                 # May need other sentinel values here to simplify
                                                 # code below
                                                 dij=[1,1],
                                                 ij=[np.nan,np.nan],
                                                 psiphi=[np.nan,np.nan])

            g_final2.make_cells_from_edges()

        # Patch grid g_final2 completed.

        # --- Compile Swaths ---
        e2c=g_final2.edge_to_cells(recalc=True)

        i_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)
        j_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool8)

        for j in g_final2.valid_edge_iter():
            c1,c2=e2c[j,:]
            if c1<0 or c2<0: continue

            if g_final2.edges['dij'][j,0]==0:
                i_adj[c1,c2]=i_adj[c2,c1]=True
            elif g_final2.edges['dij'][j,1]==0:
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

        def add_swath_contours_new(comp_cells,node_field,coord):
            # Check all of the nodes to find the range ij
            comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
            comp_nodes=np.unique( np.concatenate(comp_nodes) )
            comp_ijs=[] # Certainly could have kept this info along while building...

            field_values=[]

            comp_ij=np.array(g_final2.edges['ij'][ g_final2.cell_to_edges(comp_cells[0]) ])
            comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cells[0]) ])

            # it's actually the other coordinate that we want to consider.
            field_min=np.nanmin( comp_pp[:,1-coord] )
            field_max=np.nanmax( comp_pp[:,1-coord] )

            coord_min=np.nanmin( comp_ij[:,1-coord] )
            coord_max=np.nanmax( comp_ij[:,1-coord] )

            n_swath_cells=int(np.round(coord_max-coord_min))

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
                    add_swath_contours_new(comp_cells,node_field,coord)

        # Direct grid gen from contour specifications:

        @utils.add_to(g_int)
        def fields_to_xy(self,target,node_fields,x0):
            """
            target: values of node_fields to locate
            x0: starting point

            NB: edges['cells'] must be up to date before calling
            """
            c=self.select_cells_nearest(x0)

            while 1:
                c_nodes=self.cell_to_nodes(c)
                M=np.array( [ node_fields[0][c_nodes],
                              node_fields[1][c_nodes],
                              [1,1,1] ] )
                b=[target[0],target[1],1.0]

                weights=np.linalg.solve(M,b)
                if min(weights)<0: # not there yet.
                    min_w=np.argmin(weights)
                    c_edges=self.cell_to_edges(c,ordered=True)# nodes 0--1 is edge 0, ...
                    sel_j=c_edges[ (min_w+1)%(len(c_edges)) ]
                    edges=self.edges['cells'][sel_j]
                    if edges[0]==c:
                        next_c=edges[1]
                    elif edges[1]==c:
                        next_c=edges[0]
                    else:
                        raise Exception("Fail.")
                    if next_c<0:
                        if weights.min()<-1e-5:
                            print("Left triangulation (min weight: %f)"%weights.min())
                            # Either the starting cell didn't allow a simple path
                            # to the target, or the target doesn't fall inside the
                            # grid (e.g. ragged edge)
                            return [np.nan,np.nan]
                        # Clip the answer to be within this cell (will be on an edge
                        # or node).
                        weights=weights.clip(0)
                        weights=weights/weights.sum()
                        break
                    c=next_c
                    continue
                else:
                    break
            x=(self.nodes['x'][c_nodes]*weights[:,None]).sum(axis=0)
            return x

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
                    edge_dijs=g_final2.edges['dij'][ g_final2.cell_to_edges(c) ]
                    ragged_js=(edge_dijs!=0.0).sum(axis=1)
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
