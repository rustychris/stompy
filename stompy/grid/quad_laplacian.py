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
from scipy import sparse, signal

import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator,TriFinder,TrapezoidMapTriFinder

from matplotlib import colors
import itertools

from . import unstructured_grid, exact_delaunay,orthogonalize, triangulate_hole
from .. import utils, filters
from ..spatial import field, linestring_utils
from . import front

import logging

log=logging.getLogger('quad_laplacian')

import six

##

# A hack for linear interpolation on g_int. Nodes outside the triangulation
# take their value from the nearest cell.
class PermissiveFinder(TrapezoidMapTriFinder):
    def __init__(self,grid):
        self.grid=grid
        self.mp_tri=grid.mpl_triangulation()
        super(PermissiveFinder,self).__init__(self.mp_tri)
    def __call__(self, x, y):
        base=super(PermissiveFinder,self).__call__(x,y)
        missing=np.nonzero(base==-1)[0]
        for i in missing:
            base[i]=self.grid.select_cells_nearest( [x[i],y[i]] )
        return base


# borrow codes as in front.py
RIGID=front.AdvancingFront.RIGID

class NodeDiscretization(object):
    # original code set gradients by forcing the normal of the gradient
    # to zero. This doesn't force a sign or magnitude on the gradient.
    # If this is False, gradients are directly enforced.
    gradient_by_normal=True
    def __init__(self,g,**kw):
        self.g=g
        utils.set_keywords(self,kw)
        
    def construct_matrix(self,op='laplacian',dirichlet_nodes={},
                         zero_tangential_nodes=[],
                         gradient_nodes={},
                         skip_dirichlet=True):
        """
        Construct a matrix and rhs for the given operation.
        dirichlet_nodes: boundary node id => value
        zero_tangential_nodes: list of lists.  each list gives a set of
          nodes which should be equal to each other, allowing specifying
          a zero tangential gradient BC.  
        gradient_nodes: boundary node id => gradient unit vector [dx,dy]

        skip_dirichlet: should dirichlet nodes be omitted from other BCs?
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

        # Now I want to allow multiple BCs to constrain the same node.
        # How many rows will I end up with?

        # First count up the nodes that will get a regular laplacian
        # row.  This includes boundary nodes that have a no-flux BC.
        # (because that's the behavior of the discretization on a
        # boundary)
        nlaplace_rows=0
        laplace_nodes={}
        for n in range(g.Nnodes()):
            if skip_dirichlet and (n in dirichlet_nodes): continue
            if n in gradient_nodes: continue
            if n in tangential_nodes: continue
            laplace_nodes[n]=True
            nlaplace_rows+=1

        ndirichlet_nodes=len(dirichlet_nodes)
        # Each group of tangential gradient nodes provides len-1 constraints
        ntangential_nodes=len(tangential_nodes) - len(zero_tangential_nodes)
        ngradient_nodes=len(gradient_nodes)

        nrows=nlaplace_rows + ndirichlet_nodes + ntangential_nodes + ngradient_nodes

        log.info(f"row breakdown:  Lap: {nlaplace_rows}  "
                 f"Dir: {ndirichlet_nodes}  Tan: {ntangential_nodes} "
                 f"({len(zero_tangential_nodes)} grps)  Grad: {ngradient_nodes}")
        log.info(f"nrows={nrows} N={g.Nnodes()}")

        B=np.zeros(nrows,np.float64)
        M=sparse.dok_matrix( (nrows,g.Nnodes()),np.float64)

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
            if self.gradient_by_normal:
                # So if vec = [1,0], then normal=[0,-1]
                # and I want dx*norma[0]+dy*normal[1] = 0
                alphas=np.array(dx_alphas)*normal[0] + np.array(dy_alphas)*normal[1]
                B[row]=0
            else:
                # So if vec = [1,0]
                # mag= sqrt(vec[0]**2 + vec[1]**2)
                # I want dx*vec[0]+dy*vec[1] = vec[0]**2 + vec[1]**2
                alphas=np.array(dx_alphas)*vec[0] + np.array(dy_alphas)*vec[1]
                B[row]=vec[0]**2 + vec[1]**2
                
            for node,alpha in zip(nodes,alphas):
                M[row,node]=alpha
            row+=1

        for n in tangential_nodes:
            leader=tangential_nodes[n]
            if n==leader:
                # print("skip leader")
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
            # 
            #nbr_boundary=[self.g.is_boundary_node(n)
            #              for n in N]
            nbr_boundary=[self.g.is_boundary_edge(self.g.nodes_to_edge(n0,n))
                          for n in N]
            while not (nbr_boundary[0] and nbr_boundary[-1]):
                N=np.roll(N,1)
                nbr_boundary=np.roll(nbr_boundary,1)
        
        # area of the triangles
        A=[] 
        for m in range(M):
            tri=[n0,N[m],N[(m+1)%P]]
            Am=utils.signed_area( self.g.nodes['x'][tri] )
            assert Am!=0.0
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

def classify_nodes(g,gen):
    """
    Find fixed vs. free nodes in g by matching up with gen.nodes['fixed']
    """
    n_fixed=[]
    for gen_n in np.nonzero( gen.nodes['fixed'] )[0]:
        n=g.select_nodes_nearest( gen.nodes['x'][gen_n], max_dist=0.001)
        if n is not None:
            n_fixed.append(n)

    n_free=[n for n in g.valid_node_iter() if n not in n_fixed]
    return n_fixed, n_free


def snap_angles(gen):
    """
    gen: unstructured grid
    will populate turn_fwd/turn_rev fields on the edges, by iterating over
    cells and setting the smallest 4 internal angles in each cell to 90, and the
    rest to 180.  Internal angles not associated with cells are set to 0
    """
    gen.add_edge_field('turn_fwd',np.zeros(gen.Nedges(),np.float64),on_exists='overwrite')
    gen.add_edge_field('turn_rev',np.zeros(gen.Nedges(),np.float64),on_exists='overwrite')
                       
    def he_angle(he):
        # Calculate absolute angle of the half edge
        xy=he.grid.nodes['x']
        seg=xy[ [he.node_rev(), he.node_fwd()] ]
        dxy=seg[1] - seg[0]
        return np.arctan2(dxy[1],dxy[0])
    def set_angle(he,angle):
        # set turn_fwd/rev for the half edge
        if he.orient==0:
            gen.edges['turn_fwd'][he.j]=angle
        else:
            gen.edges['turn_rev'][he.j]=angle
    
    for c in gen.valid_cell_iter():
        angle_and_he=[]
        he0=he=gen.cell_to_halfedge(c,0)
        while 1:
            he_nxt=he.fwd()

            angle0=he_angle(he)
            angle1=he_angle(he_nxt)
            turn=(angle1-angle0)*180/np.pi
            turn_d=(turn + 180) % 360 - 180
            # Now turn is 0 for straight ahead, 90 for a left
            angle=180-turn_d
            angle_and_he.append( [angle,he] )
            if he_nxt==he0:
                break
            else:
                he=he_nxt
                
        order=np.argsort( [angle for angle,he in angle_and_he] )
        for idx in order[:4]:
            set_angle(angle_and_he[idx][1],90)
        for idx in order[4:]:
            set_angle(angle_and_he[idx][1],180)

def prepare_angles_halfedge(gen):
    """
    Move turn angles from half edges to absolute angles of edges.
    This used to be done later in the game, and remove the internal edges
    at the same time.
    """
    # at this stage, angles are absolute, and reflect the natural direction
    # of the edge
    # breakpoint()
    edge_angles=np.nan*np.zeros(gen.Nedges(),np.float32)
    def he_angle(he,val=None):
        if val is not None:
            edge_angles[he.j] = (val + 180*he.orient) % 360
        if not np.isfinite(edge_angles[he.j]):
            return np.nan # avoid math warnings
        angle=(edge_angles[he.j] + 180 * he.orient) % 360
        return angle

    j_turns=np.c_[ gen.edges['turn_fwd'],
                   gen.edges['turn_rev'] ]
    j_turns[ j_turns==0.0 ] = 180
    j_turns[ np.isnan(j_turns) ]=180
    # And convert all to delta angle, not internal angle
    j_turns=180-j_turns

    # Have to loop over all cells in case there are disconnected areas.
    for c0 in gen.valid_cell_iter():
        # Graph traversal to set edge angles:
        he0=gen.cell_to_halfedge(c0,0)
        if np.isfinite(he_angle(he0)):
            continue
        stack=[ (he0,0.0) ]

        while stack:
            he,angle=stack.pop()
            if he.cell()<0: continue

            existing_angle=he_angle(he)
            if np.isfinite(existing_angle):
                if existing_angle!=angle:
                    plt.figure(2).clf()
                    gen.plot_edges(labeler=lambda j,r: ["",edge_angles[j]][int(np.isfinite(edge_angles[j]))])
                    gen.plot_nodes(labeler='id')
                    edge_tans=utils.to_unit( np.diff(gen.nodes['x'][gen.edges['nodes']],axis=1)[:,0,:] )
                    ecs=gen.edges_center()
                    plt.quiver(ecs[:,0],ecs[:,1],edge_tans[:,0],edge_tans[:,1],
                               color='red',scale=20,width=0.01)
                    plt.axis('tight')
                    plt.axis('equal')
                    gen.plot_edges(mask=[he.j],color='r',lw=3)
                    raise Exception("Angle mismatch")
                continue
            else:
                # Set it
                he_angle(he,angle)

            he_fwd=he.fwd()
            angle_fwd=(angle+j_turns[he.j,he.orient])%360
            # print(f"  fwd: he={he_fwd}  angle_fwd={angle_fwd}")
            stack.append( (he_fwd,angle_fwd) )

            he_opp=he.opposite()
            if he_opp.cell()<0: continue
            he_rev=he_opp.fwd()
            angle_rev=(angle+180+j_turns[he.j,1-he.orient])%360
            # print(f"  rev: he={he_rev}  angle_fwd={angle_rev}")

            stack.append( (he_rev,angle_rev) )

    gen.add_edge_field('angle',edge_angles,on_exists='overwrite')

def adjusted_scale(gen, scale, nom_res=None):
    """
    Replace positive scale values on shared edges with negative counts
    """
    scale=scale.copy()
    e2c=gen.edge_to_cells()
    for j in np.nonzero( (e2c.min(axis=1)>=0) )[0]:
        if scale[j]<0: continue # already good
        L=gen.edges_length(j)
        if scale[j]==0.0:
            if nom_res is not None:
                res=nom_res
            else:
                continue
        else:
            res=scale[j]
        scale[j]=-max(1,np.round(L/res))
        print("Mapped edge scale %g to %g"%(res,scale[j]))
    return scale

def linear_scales(gen,method='adhoc'):
    """
    Infer scales from the edges of the generating grid
    Create a scale field for the two canonical coordinates.

    method: 'adhoc' reverts to a simple extrapolation approach, rather 
    than solving a proper laplacian.  The Laplacian code turned out be
    brittle on the very coarse triangulation of gen.
    """
    scales=gen.edges['scale']
    scales=np.where( np.isfinite(scales), scales, 0.0)

    i_edges=np.nonzero( (scales!=0) & (gen.edges['angle']%180== 0) )[0]
    j_edges=np.nonzero( (scales!=0) & (gen.edges['angle']%180==90) )[0]

    gen_tri=exact_delaunay.Triangulation()
    gen_tmp=gen.copy()
    gen_tmp.renumber_edges()
    gen_tmp.renumber_nodes()

    gen_tri.init_from_grid(gen_tmp,set_valid=True)
    gen_tri=gen_tri.copy()
    for c in np.nonzero(~gen_tri.cells['valid'])[0]:
        if not gen_tri.cells['deleted'][c]:
            gen_tri.delete_cell(c)
    gen_tri.delete_orphan_edges()
    gen_tri.delete_orphan_nodes()
    gen_tri.renumber()

    # First, the i scale:
    extraps=[]
    el=gen.edges_length()
    for edge_list in [i_edges,j_edges]:
        dirich={} # nodes of gen => dirichlet BC
        for j in edge_list:
            scale=scales[j]
            if scale<0:
                scale=el[j]/(-scale)
                
            for n in gen.edges['nodes'][j]:
                if n in dirich:
                    dirich[n] = 0.5*(scale+dirich[n])
                else:
                    dirich[n]=scale

        mapped_dirich={}
        for n in dirich:
            n_tri=gen_tri.select_nodes_nearest(gen.nodes['x'][n],max_dist=0.0)
            assert n_tri is not None
            mapped_dirich[n_tri]=dirich[n]

        if method=='adhoc':
            # This really shouldn't be necessary, but I'm having issues with
            # negative results coming out of NodeDiscretization.
            # This is at least positive definite (right?)
            N=gen_tri.Nnodes()
            M=sparse.dok_matrix((N,N))
            b=np.zeros( gen_tri.Nnodes())
            for n in range(gen_tri.Nnodes()):
                if n in mapped_dirich:
                    M[n,n]=1
                    b[n]=mapped_dirich[n]
                else:
                    nbrs=gen_tri.node_to_nodes(n)
                    f=1./len(nbrs)
                    M[n,n]=1
                    for nbr in nbrs:
                        M[n,nbr]=-f
            soln=sparse.linalg.spsolve(M.tocsr(),b)
        else:
            nd=NodeDiscretization(gen_tri)
            M,b=nd.construct_matrix(op='laplacian',dirichlet_nodes=mapped_dirich)
            soln=sparse.linalg.spsolve(M.tocsr(),b)
        extraps.append(soln)

    mp_tri=gen_tri.mpl_triangulation()
    i_field=field.XYZField(X=gen_tri.nodes['x'],F=extraps[0])
    i_field._tri = mp_tri
    j_field=field.XYZField(X=gen_tri.nodes['x'],F=extraps[1])
    j_field._tri = mp_tri

    # As it stands, it's possible for the input grid to have disconnected
    # sets of cells, such that the node discretization yields 0 in the
    # disconnected areas (unless they have their own scale data).
    bad=(extraps[0]<=0) | (extraps[1]<=0)
    if np.any(bad):
        idxs=np.nonzero(bad)[0]
        log.warning("Bad nodes: %s"%str(gen_tri.nodes['x'][idxs]))
        # push the error to later -- possible that the nodes we care about
        # have scale.
        #raise Exception("Probably disconnected cells with no scale")
    return i_field, j_field


def add_bezier(gen):
    """
    This version only needs turn_rev/turn_fwd, and does not need
    the absolute angles to be globally consistent.

    Note that the quad generation still relies on absolute angles.

    Generate bezier control points for each edge.  Uses angles (in ij
    space) in the generating grid to calculate angles at each vertex, and then
    choose bezier control points to achieve that angle.

    This version is standalone, and can handle nodes with degree>2.
    """
    # Need to force the internal angles at nodes to match xy space
    # angles and ij space angles.

    order=3 # cubic bezier curves
    bez=np.nan*np.zeros( (gen.Nedges(),order+1,2) )
    valid_edges=~gen.edges['deleted']
    bez[valid_edges,0    ,:] = gen.nodes['x'][gen.edges['nodes'][valid_edges,0]]
    bez[valid_edges,order,:] = gen.nodes['x'][gen.edges['nodes'][valid_edges,1]]

    gen.add_edge_field('bez', bez, on_exists='overwrite')

    for n in gen.valid_node_iter():
        hes=gen.node_to_halfedges(n)

        # Filter out non-quad grid edges in there, which will have angle=nan
        hes=[he for he in hes 
             if gen.edges['turn_fwd'][he.j]>0 or gen.edges['turn_rev'][he.j]>0]

        if len(hes)==0:
            continue
        
        # when a node is part of multple cells, but those
        # cells have no edges in common, treat it as two separate
        # groups of angles

        # label nodes:
        # These are already angle sorted, but they won't necessarily
        # start with the most CW of the group.
        node_groups=np.arange(len(hes))
        group_start=0 # index into hes giving the first element of a group.
        for a in range(len(hes)):
            b=(a+1)%len(hes)
            he_a=hes[a]
            he_b=hes[b]
            if he_a.cell()>=0:
                # in the old code this was an additional condition.
                # But this should always be true.
                assert he_a.cell()==he_b.cell_opp()
                node_groups[ node_groups==node_groups[b] ] = node_groups[a]
            else:
                group_start=b

        # rotate so we start at the beginning of a group.
        hes=np.roll(hes,-group_start)
        node_groups=np.roll(node_groups,-group_start)
        
        # Don't actually need absolute angles, just need to be able to assign
        # locally consistent ij angles.
        
        # Each of those half-edges has an angle in ij space, relative
        # to natural edge direction.
        def he_xy_angle(he):
            A=gen.nodes['x'][he.node_rev()]
            B=gen.nodes['x'][he.node_fwd()]
            delta=B-A
            return 180/np.pi * np.arctan2(delta[1],delta[0])        
        xy_angles=np.array( [he_xy_angle(he) for he in hes] )

        xy_tgts=0*xy_angles
        
        for grp in np.arange(len(hes)):
            sel=node_groups==grp
            if np.all(~sel): continue
            grp_ij=[0] # by fiat, start of group is ij angle 0
            
            # turn information will be on the incoming half-edge
            # but each of he are outgoing half-edges.
            for he in hes[sel][1:]:
                if he.orient==0:
                    turn = gen.edges['turn_rev'][he.j]
                else:
                    turn = gen.edges['turn_fwd'][he.j]
                assert turn>0
                grp_ij.append(grp_ij[-1] + turn)
                
            grp_xy = xy_angles[sel] 
            grp_ij = np.array(grp_ij)
            # overall rotation -
            # 0 degress in xy space is rot degrees from 0 degrees in
            # ij space.
            xy_to_ij=np.pi/180 * (grp_ij-grp_xy)
            
            # angular average:            
            rot=180/np.pi * np.arctan2( np.sin(xy_to_ij).mean(),
                                        np.cos(xy_to_ij).mean())
            xy_tgts[sel]=grp_ij[sel] - rot
            
        xy_errs=xy_angles - xy_tgts

        for he,xy_err in zip(hes,xy_errs):
            vec=gen.nodes['x'][he.node_fwd()] - gen.nodes['x'][he.node_rev()]
            cp=gen.nodes['x'][n] + utils.rot(-xy_err*np.pi/180., 1./3 * vec)
            gen.edges['bez'][he.j,1+he.orient]=cp




def add_bezier_from_angle(gen):
    """
    This version requires absolute angle set on each edge.
 
    Generate bezier control points for each edge.  Uses angles (in ij
    space) in the generating grid to calculate angles at each vertex, and then
    choose bezier control points to achieve that angle.

    This version is standalone, and can handle nodes with degree>2.
    """
    # Need to force the internal angles at nodes to match xy space
    # angles and ij space angles.

    order=3 # cubic bezier curves
    bez=np.nan*np.zeros( (gen.Nedges(),order+1,2) )
    valid_edges=~gen.edges['deleted']
    bez[valid_edges,0    ,:] = gen.nodes['x'][gen.edges['nodes'][valid_edges,0]]
    bez[valid_edges,order,:] = gen.nodes['x'][gen.edges['nodes'][valid_edges,1]]

    gen.add_edge_field('bez', bez, on_exists='overwrite')

    for n in gen.valid_node_iter():
        nbrs=gen.angle_sort_adjacent_nodes(n)
        if len(nbrs)==0:
            continue
        hes=[gen.nodes_to_halfedge(n,nbr) for nbr in nbrs]

        # Filter out non-quad grid edges in there, which will have angle=nan
        hes=[he for he in hes if np.isfinite(gen.edges['angle'][he.j])]

        if len(hes)==0:
            continue
        
        # Each of those half-edges has an angle in ij space, relative
        # to natural edge direction.
        def he_ij_angle(he):
            return (gen.edges['angle'][he.j]+(he.orient*180.))%360.
        def he_xy_angle(he):
            A=gen.nodes['x'][he.node_rev()]
            B=gen.nodes['x'][he.node_fwd()]
            delta=B-A
            return 180/np.pi * np.arctan2(delta[1],delta[0])

        ij_angles=np.array( [he_ij_angle(he) for he in hes] )
        xy_angles=np.array( [he_xy_angle(he) for he in hes] )
        
        # overall rotation -
        # 0 degress in xy space is rot degrees from 0 degrees in
        # ij space.
        xy_to_ij=np.pi/180 * (ij_angles-xy_angles)

        # when a node is part of multple cells, but those
        # cells have no edges in common, treat it as two separate
        # groups of angles
        
        # label nodes:
        node_groups=np.arange(len(hes))
        for a in range(len(hes)):
            b=(a+1)%len(hes)
            he_a=hes[a]
            he_b=hes[b]
            if he_a.cell()>=0 and he_a.cell()==he_b.cell_opp():
                node_groups[ node_groups==node_groups[b] ] = node_groups[a]

        xy_tgts=0*ij_angles
        for grp in np.arange(len(hes)):
            sel=node_groups==grp
            if np.all(~sel): continue
            
            rot=180/np.pi * np.arctan2( np.sin(xy_to_ij[sel]).mean(),
                                        np.cos(xy_to_ij[sel]).mean())
            xy_tgts[sel]=ij_angles[sel] - rot
            
        xy_errs=xy_angles - xy_tgts

        for he,xy_err in zip(hes,xy_errs):
            vec=gen.nodes['x'][he.node_fwd()] - gen.nodes['x'][he.node_rev()]
            cp=gen.nodes['x'][n] + utils.rot(-xy_err*np.pi/180., 1./3 * vec)
            gen.edges['bez'][he.j,1+he.orient]=cp


def plot_gen_bezier(gen,num=10,ax=None,connect_handles=True,
                    handle_kw=dict(color='r',zorder=1,alpha=0.5,lw=1.5),
                    smooth_kw=dict(color='b',zorder=2,lw=1.5)):
    """
    gen: grid with 'bez' on edges
    ax: optional axes to plot into
    connect_handles: each bezier segment has two ends and two internal
     handles. controls whether a line is drawn connecting the handles
    """
    if ax is None:
        fig=plt.figure(num)
        fig.clf()
        ax=fig.add_subplot(1,1,1)
    else:
        fig=None
        
    gen.plot_edges(lw=0.3,color='k',alpha=0.5,ax=ax)
    gen.plot_nodes(alpha=0.5,ax=ax,zorder=3,color='orange')

    for j in gen.valid_edge_iter():
        n0=gen.edges['nodes'][j,0]
        nN=gen.edges['nodes'][j,1]
        bez=gen.edges['bez'][j]

        t=np.linspace(0,1,21)

        B0=(1-t)**3
        B1=3*(1-t)**2 * t
        B2=3*(1-t)*t**2
        B3=t**3
        points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]

        ax.plot(points[:,0],points[:,1],**smooth_kw)
        if connect_handles:
            ax.plot(bez[:,0],bez[:,1],**handle_kw)
        else:
            ax.plot(bez[:2,0],bez[:2,1],**handle_kw)
            ax.plot(bez[2:,0],bez[2:,1],**handle_kw)
            
    return fig,ax

def discretize_bezier_halfedge(gen,he,samples_per_edge=10):
    j=he.j # gen.nodes_to_edge(a,b)

    n0=gen.edges['nodes'][j,0]
    nN=gen.edges['nodes'][j,1]
    bez=gen.edges['bez'][j]

    t=np.linspace(0,1,1+samples_per_edge)

    if he.orient==1: # have to flip order
        t=t[::-1]        

    B0=(1-t)**3
    B1=3*(1-t)**2 * t
    B2=3*(1-t)*t**2
    B3=t**3
    edge_points = B0[:,None]*bez[0] + B1[:,None]*bez[1] + B2[:,None]*bez[2] + B3[:,None]*bez[3]
    return edge_points

class QuadGen(object):
    # The cell spacing in geographic coordinates for the nominal, isotropic grid
    nom_res=4.0
    gmsh_path="gmsh"
    
    scales=None
    
    # Minimum number of edges along a boundary segment in the nominal isotropic grid
    min_steps=2

    # The additional constraints that link psi and phi create an over-determined
    # system.  Still trying to understand when this is a problem.  The solution
    # can in some cases become invalid when the gradient terms are either too
    # strong (maybe) or too loose (true).
    # 'scaled' will scale the gradient terms according to the number of extra dofs.
    # Possibe that the scaled code was a weak attempt to fix something that was
    # really a bug elsewhere, and that 1.0 is the best choice
    gradient_scale=1.0

    # 'rebay', 'front', 'gmsh'.  When intermediate is 'tri', this chooses the method for
    # generating the intermediate triangular grid
    # note that rebay is buggy! gmsh is great if installed.
    triangle_method='rebay'

    # How internal angles are specified.  'node' is only valid when a single cell
    # is selected in gen.
    angle_source='halfedge' # 'halfedge' or 'node'

    # Whether to smooth and relax the resulting grid
    smooth=True
    
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
        gen.modify_max_sides( gen.Nnodes() )

        # Process angles on the whole quad grid, so we can also get
        # scales
        if self.angle_source=='halfedge':
            snap_angles(gen)
            prepare_angles_halfedge(gen)
        elif self.angle_source=='existing':
            pass
        else:
            raise Exception("bad angle source: %s"%self.angle_source)
        add_bezier(gen)
        
        if self.scales is None:
            if 'scale' in gen.edges.dtype.names:
                self.scales=linear_scales(gen)
            else:
                self.scales=[field.ConstantField(self.nom_res),
                             field.ConstantField(self.nom_res)]
        
        if cells is not None:
            for c in range(gen.Ncells()):
                if (c not in cells) and (not gen.cells['deleted'][c]):
                    gen.delete_cell(c)
        gen.delete_orphan_edges()
        gen.delete_orphan_nodes()
        gen.renumber(reorient_edges=False)

        self.gen_orig=gen.copy() # before merging
        self.gen=gen
        # [ [node_a,node_b,angle], ...]
        # node indices reference gen, which provide
        # additional groupings of nodes.
        self.internal_edges=[]
        self.breaklines=[]

        if execute:
            self.execute()

    def add_internal_edge(self,nodes,angle=None,colinear=True):
        """
        nodes: the two nodes making up the edge
        angle: if known, the orientation of the edge.
        colinear: if true, marks the edge as being colinear with
        an external edge.
        """
        if colinear:
            self.internal_edges.append( [nodes[0], nodes[1], angle] )
        else:
            # towards aligning internal rows
            self.breaklines.append( [nodes[0],nodes[1],angle] )
            
    def execute(self):
        self.process_internal_edges(self.gen)
        self.g_int=self.create_intermediate_grid_tri()
        self.calc_psi_phi()
        self.g_final=self.create_final_by_patches()
        if self.smooth:
            self.smooth_to_scale(self.g_final)
        return self.g_final
    
    def set_scales_diffusion(self):
        # Probably not what I'll end up with, but try a diffusion approach
        i_scale_dir={}
        j_scale_dir={}

        for j in np.nonzero( self.g_int.edges['gen_j']>= 0)[0]:
            gen_j=self.g_int.edges['gen_j'][j]
            scale=self.gen.edges['scale'][gen_j]
            if scale in [0,np.nan]: continue
            orient=self.gen.edges['angle'][gen_j] % 180
            if orient==0:
                # Add to i scale (?)
                for n in self.g_int.edges['nodes'][j]:
                    i_scale_dir[n]=scale
            elif orient==90:
                # Add to j scale (?)
                for n in self.g_int.edges['nodes'][j]:
                    j_scale_dir[n]=scale

        nd=self.nd
        M,B=nd.construct_matrix(op='laplacian',
                                dirichlet_nodes=i_scale_dir,
                                skip_dirichlet=True)
        i_scale=sparse.linalg.spsolve(M.tocsr(),B)
        M,B=nd.construct_matrix(op='laplacian',
                                dirichlet_nodes=j_scale_dir,
                                skip_dirichlet=True)
        j_scale=sparse.linalg.spsolve(M.tocsr(),B)

        self.i_scale=field.XYZField(X=self.g_int.nodes['x'],F=i_scale)
        self.j_scale=field.XYZField(X=self.g_int.nodes['x'],F=j_scale)
        self.scales=[self.i_scale,self.j_scale]
        
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
            assert local_res>0
            assert np.isfinite(dist)
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

        if self.triangle_method=='gmsh':
            fn='tmp.geo'
            g.write_gmsh_geo(fn)
            import subprocess
            subprocess.run([self.gmsh_path,fn,'-2'])
            try:
                g_gmsh=unstructured_grid.UnstructuredGrid.read_gmsh('tmp.msh')
            except:
                log.error("Intermediate grid generation failed. Check for self-intersection in bezier grid")
                raise
            g.add_grid(g_gmsh,merge_nodes='auto',tol=1e-3)
            gnew=g
        else:    
            nodes=g.find_cycles(max_cycle_len=g.Nnodes()+1)[0]
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

    def plot_gen_bezier(self,num=10,ax=None):
        fig,ax=plot_gen_bezier(self.gen,ax=ax)
        
        for n12 in self.internal_edges:
            ax.plot( self.gen.nodes['x'][n12[:2],0],
                     self.gen.nodes['x'][n12[:2],1], 'g-')
            i_angle=self.internal_edge_angle(n12)
            mid=self.gen.nodes['x'][n12[:2],:].mean(axis=0)
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
        
    def gen_bezier_linestring(self,j=None,nodestring=None,samples_per_edge=10,span_fixed=True,
                              gen=None):
        """
        Calculate an up-sampled linestring for the bezier boundary of self.gen
        
        j: limit the curve to a single generating edge if given.
        span_fixed: see gen_bezier_curve()
        """
        if gen is None:
            gen=self.gen

        # need to know which ij coordinates are used in order to know what is
        # fixed. So far fixed is the same whether IJ or ij, so not making this
        # a parameter yet.
        src='IJ'

        if nodestring is None:
            if j is None:
                bound_nodes=self.gen.boundary_cycle() # probably eating a lot of time.
                nodestring=np.r_[bound_nodes, bound_nodes[:1]]
                # node_pairs=zip(bound_nodes,np.roll(bound_nodes,-1))
            else:
                if not span_fixed:
                    nodestring=self.gen.edges['nodes'][j]
                else:
                    nodestring=[]

                    # Which coord is changing on j? I.e. which fixed should
                    # we consult?
                    # A little shaky here.  Haven't tested this with nodes
                    # that are fixed in only one coordinate.
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
                        nodestring.append(n)
                        if self.gen.nodes[src+'_fixed'][n,coord]:
                            break
                        trav=trav.fwd()
                    nodestring=nodestring[::-1]
                    trav=self.gen.halfedge(j,0)
                    while 1: # REV
                        n=trav.node_rev()
                        nodestring.append(n)
                        if self.gen.nodes[src+'_fixed'][n,coord]:
                            break
                        trav=trav.rev()
                    
        node_pairs=zip( nodestring[:-1], nodestring[1:])
            
        points=[]
        for a,b in node_pairs:
            he=gen.nodes_to_halfedge(a,b)
            edge_points=discretize_bezier_halfedge(gen,he,samples_per_edge=samples_per_edge)

            points.append(edge_points[:-1])
        if nodestring[0]!=nodestring[-1]:
            # When the curve isn't closed, then be inclusive of both
            # ends
            points.append(edge_points[-1:])
            
        return np.concatenate(points,axis=0)

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

    def process_internal_edges(self,gen):
        """
        Remove and save internal edges.
        self.internal_edges will end up with internal
        edges that are colinear with external edges (i.e. inside corners)

        Flip angle for remaining edge to reflect orientation
        along boundary cycle.
        Add 'fixed' and 'turn' field to gen.nodes
        """
        e2c=gen.edge_to_cells()
        internal=(e2c.min(axis=1)>=0)

        for j in np.nonzero(internal)[0]:
            if gen.edges['deleted'][j]: continue
            
            # Only when there are parallel edges on both
            # sides do we actually record the internal
            # edge
            angle=gen.edges['angle'][j]

            parallel_count=0
            for nbr in gen.edges['nodes'][j]:
                for j_nbr in gen.node_to_edges(nbr):
                    if internal[j_nbr]: # j or another internal edge
                        continue
                    if (gen.edges['angle'][j_nbr] - angle)%180==0:
                        parallel_count+=1
                        break # parallel. good
            self.add_internal_edge(gen.edges['nodes'][j],
                                   gen.edges['angle'][j],
                                   colinear=parallel_count>=2) # probably just ==2
            # DBG
            # Tricky: if there are multiple segments, the first succeeds,
            # and the second is degenerate (spike into the cell interior)
            # this should be handled now down in delete_edge_cascade
            # but it's not robust. I think it can still fail when some of the interior
            # edges become disconnected -- then they are left with -2,-2 neighbors
            if np.all(gen.edge_to_cells(j)==-2):
                gen.delete_edge(j)
            else:
                gen.merge_cells(j=j)
        # A bit of insurance? Hopefully doesn't break things
        gen.delete_orphan_edges()
            
        cycles=gen.find_cycles(max_cycle_len=1000)
        assert len(cycles)==1,"For now, cannot handle multiple cycles"
        cycle=cycles[0]

        last_angle=None

        gen.add_node_field('fixed',np.zeros(gen.Nnodes(),np.bool_))
        
        for a,b in zip( cycle, np.roll(cycle,-1) ):
            j=gen.nodes_to_edge(a,b)
            assert j is not None
            if np.all(gen.edges['nodes'][j]==[a,b]):
                pass
            else:
                assert np.all(gen.edges['nodes'][j]==[b,a])
                gen.edges['angle'][j]=(gen.edges['angle'][j]+180)%360

        gen.add_node_field('turn',np.zeros(gen.Nnodes(),np.float64), on_exists='overwrite')
        
        for prv,n,nxt in zip( cycle, np.roll(cycle,-1),np.roll(cycle,-2) ):
            jprv=gen.nodes_to_edge(prv,n)
            jnxt=gen.nodes_to_edge(n,nxt)
            
            assert jprv is not None
            assert jnxt is not None

            angle_prv=gen.edges['angle'][jprv]
            angle_nxt=gen.edges['angle'][jnxt]
            
            gen.nodes['fixed'][n] = angle_prv!=angle_nxt
            # These are internal angles
            # funky 1 is for internal angle of 360
            gen.nodes['turn'][n] = (180 - (angle_nxt-angle_prv) -1 ) % 360 + 1
    
    def internal_edge_angle(self,gen_edge):
        """
        not exact, but try a heuristic.
        use adjacent edges to estimate the +i tangent vector,
        Then compare to the angle of gen_edge.
        Returns 0 or 90 (0 vs 180 is not unique)
        """
        # Use the specified angle if it's set:
        assert gen_edge[2] is not None
        return gen_edge[2]
        
        # gen=self.gen
        # e2c=gen.edge_to_cells()
        # i_tan_vecs=[]
        # for n in gen_edge[:2]:
        #     for j in gen.node_to_edges(n):
        #         angle=gen.edges['angle'][j]
        #         tan_vec=np.diff(gen.nodes['x'][ gen.edges['nodes'][j] ],axis=0)[0]
        #         tan_vec=utils.to_unit(tan_vec)
        #         if e2c[j,0]<0:
        #             # This edge goes opposite the cycle direction
        #             tan_vec*=-1 
        #         i_tan_vec=utils.rot(-angle*np.pi/180.,tan_vec)
        #         i_tan_vecs.append(i_tan_vec)
        # i_tan=utils.to_unit( np.mean(i_tan_vecs,axis=0) )
        # j_tan=np.array( [i_tan[1],-i_tan[0]] ) # sign may be off, no worries
        # 
        # d_gen_edge= np.diff(gen.nodes['x'][gen_edge[:2]],axis=0)[0]
        # 
        # j_score=np.dot(j_tan,d_gen_edge)
        # i_score=np.dot(i_tan,d_gen_edge)
        # 
        # if np.abs(j_score)>np.abs(i_score):
        #     return 90
        # else:
        #     return 0
    
    def calc_psi_phi(self):
        if False:
            self.psi_phi_setup(n_j_dirichlet=1)
            self.psi_phi_solve_coupled()
        else:
            self.psi_phi_setup(n_j_dirichlet=2)
            self.psi_phi_solve_separate()

    i_nf_cells=None
    j_nf_cells=None
    def psi_phi_solve_separate(self):
        """
        Solve psi and phi fields separately, each fully determined.
        Assumes that psi_phi_setup() has already been called, and with
        n_j_dirichlet=2 specified (so that the phi system is fully
        determined)
        """
            
        for coord in [0,1]:
            # signify we're working on psi vs. phi
            nf_cells=[]
            
            if coord==0:
                grad_nodes=dict(self.i_grad_nodes)
                dirichlet_nodes=dict(self.i_dirichlet_nodes)
                tan_groups=self.i_tan_groups
            else:
                grad_nodes=dict(self.j_grad_nodes)
                dirichlet_nodes=dict(self.j_dirichlet_nodes)
                tan_groups=self.j_tan_groups

            # Find these automatically.
            # For ragged edges: not sure, but punt by dropping the
            # the gradient BC on the acute end (node 520)
            noflux_tris=[]
            for n in np.nonzero(self.g_int.nodes['rigid'])[0]:
                gen_n=self.g_int.nodes['gen_n'][n]
                assert gen_n>=0
                gen_turn=self.gen.nodes['turn'][gen_n]
                # For now, ignore non-cartesian, and 90
                # degree doesn't count
                if (gen_turn>90) and (gen_turn<180):
                    # A ragged edge -- try out removing the gradient BC
                    # here
                    if n in grad_nodes:
                        # This is maybe causing a problem with phi in cell 1.
                        log.info(f"n {n}: angle={gen_turn} Dropping gradient BC")
                        del grad_nodes[n]
                    continue

                if gen_turn not in [270,360]: continue
                if gen_turn==270:
                    log.info(f"n {n}: turn=270")
                elif gen_turn==360:
                    log.info(f"n {n}: turn=360")

                js=self.g_int.node_to_edges(n)
                e2c=self.g_int.edge_to_cells()

                for j in js:
                    if (e2c[j,0]>=0) and (e2c[j,1]>=0): continue
                    gen_j=self.g_int.edges['gen_j'][j]
                    angle=self.gen.edges['angle'][gen_j]
                    if self.g_int.edges['nodes'][j,0]==n:
                        nbr=self.g_int.edges['nodes'][j,1]
                    else:
                        nbr=self.g_int.edges['nodes'][j,0]
                    log.info(f"j={j}  {n} -- {nbr}  angle={angle} coord={coord}")
                    # Does the angle 
                    if (angle + 90*coord)%180. == 90.:
                        log.info("  YES")
                        c=e2c[j,:].max()
                        tri=self.g_int.cells['nodes'][c]
                        nf_cells.append(c)
                        while tri[2] in [n,nbr]:
                            tri=np.roll(tri,1)
                        noflux_tris.append( (n,tri) )
                    else:
                        log.info("  NO")

            if coord==0:
                self.i_nf_cells=nf_cells
                joins=self.i_tan_joins
            else:
                self.j_nf_cells=nf_cells
                joins=self.j_tan_joins

            # Drop an nf_cell constraint for every internal edge
            log.info(f"About to process joins, starting with {len(noflux_tris)} nf tris")
            for join in joins:
                found=False
                slim_noflux_tris=[]
                log.info(f"Looking for an nf_tri to drop for join {join[0]}--{join[1]}")
                for idx,(n,tri) in enumerate(noflux_tris):
                    if (not found) and (n in join):
                        log.info(f"  will drop {n}")
                        # Skip this, and copy the rest
                        found=True
                    else:
                        slim_noflux_tris.append( (n,tri) )
                if not found:
                    log.warning(f"  Uh-oh! couldn't find a no-flux tri to drop for this internal edge")
                noflux_tris=slim_noflux_tris
                
            nf_block=sparse.dok_matrix( (len(noflux_tris),self.g_int.Nnodes()), np.float64)
            nf_rhs=np.zeros( len(noflux_tris) )
            node_xy=self.g_int.nodes['x'][:,:]

            for idx,(n,tri) in enumerate(noflux_tris):
                target_dof=idx # just controls where the row is written
                d01=node_xy[tri[1],:] - node_xy[tri[0],:]
                d02=node_xy[tri[2],:] - node_xy[tri[0],:]
                # Derivation in sympy below
                nf_block[target_dof,:]=0 # clear old
                nf_block[target_dof,tri[0]]= -d01[0]**2 + d01[0]*d02[0] - d01[1]**2 + d01[1]*d02[1]
                nf_block[target_dof,tri[1]]= -d01[0]*d02[0] - d01[1]*d02[1]
                nf_block[target_dof,tri[2]]= d01[0]**2 + d01[1]**2
                nf_rhs[target_dof]=0
                        

            M_Lap,B_Lap=self.nd.construct_matrix(op='laplacian',
                                                 dirichlet_nodes=dirichlet_nodes,
                                                 skip_dirichlet=False,
                                                 zero_tangential_nodes=tan_groups,
                                                 gradient_nodes=grad_nodes)

            M=sparse.bmat( [ [M_Lap],[nf_block]] )
            B=np.concatenate( [B_Lap,nf_rhs] )

            if M.shape[0] != M.shape[1]:
                log.error(f"M.shape: {M.shape}")
                self.M_Lap=M_Lap
                self.B_Lap=B_Lap
                self.nf_block=nf_block
                self.nf_rhs=nf_rhs
                raise Exception("Matrix did not end up square!")

            # Direct solve is reasonably fast and gave better results.
            soln=sparse.linalg.spsolve(M.tocsr(),B)
            assert np.all(np.isfinite(soln))

            for grp in tan_groups:
                # Making the tangent groups exact helps in contour tracing later
                soln[grp]=soln[grp].mean()

            if coord==0:
                self.psi=soln
            else:
                self.phi=soln

    def psi_phi_setup(self,n_j_dirichlet=1):
        """
        Build the lists of BCs for solving psi/phi.

        n_j_dirichlet: whether to include just a location BC or both location and scale
        for the phi/j field.
        """
        # Record internal edges that actually led to a join.
        self.i_tan_joins=[]
        self.j_tan_joins=[]
        
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

        # a turn=360 node should induce a tan_group (to drop it's no-flux BC)
        for n in np.nonzero(self.g_int.nodes['rigid'])[0]:
            gen_n=self.g_int.nodes['gen_n'][n]
            assert gen_n>=0
            gen_turn=self.gen.nodes['turn'][gen_n]
            if gen_turn!=360: continue
            idx=np.nonzero(bcycle==n)[0][0]
            N=len(bcycle)
            prv=bcycle[(idx-1)%N]
            nxt=bcycle[(idx+1)%N]
            jprv=gtri.nodes_to_edge(prv,n)
            jnxt=gtri.nodes_to_edge(n,nxt)
            assert (jprv is not None) and (jnxt is not None)
            if (j_angles[jprv]%180==90) and (j_angles[jnxt]%180==90):
                i_tan_groups.append([n])
                phi_gradient_nodes[n]=phi_gradients[n]
            elif (j_angles[jprv]%180==0) and (j_angles[jnxt]%180==0):
                j_tan_groups.append([n])
                psi_gradient_nodes[n]=psi_gradients[n]
            else:
                log.error('Yikes - turn is 360, but not axis-aligned')
            
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

            if np.any( self.gen.nodes['turn'][gen_edge[:2]]==180.0):
                # This may still be too lenient.  Might be better to
                # check whether the angle of the internal edge is parallel
                # to edges for the two nodes. Or rather than joining groups
                # create a new group. The code as is assumes that both
                # nodes of gen_edge[:2] have adjacent edges parallel to
                # the internal edge, such that they both have existing groups
                # that can be joined.
                log.info("Internal edge connects straight boundaries. No join")
                continue
                
            if internal_angle%180==0: # join on i
                log.info("Joining two i_tan_groups")
                i_tan_groups=join_groups(i_tan_groups,edge[0],edge[1])
                self.i_tan_joins.append( edge )
            elif internal_angle%180==90: # join on j
                log.info("Joining two j_tan_groups")
                j_tan_groups=join_groups(j_tan_groups,edge[0],edge[1])
                self.j_tan_joins.append( edge )
            else:
                #import pdb
                #pdb.set_trace()
                raise Exception("Internal edge doesn't appear to join same-valued contours")

        # find longest consecutive stretch of the target_angle,
        # bounded by edges that are perpendicular to the target_angle
        # Possible that a domain could fail to have this!  If it had lots of
        # weird ragged edges
        el=gtri.edges_length()

        def longest_straight(target_angle):
            longest=(0,None,None)
            # start at a nice corner
            cycle=np.roll( bcycle,-np.nonzero( gtri.nodes['rigid'][bcycle])[0][0] )
            j_cycle=[gtri.nodes_to_edge(na,nb) for na,nb in zip(cycle, np.roll(cycle,-1))]
            j_angles=self.gen.edges['angle'][gtri.edges['gen_j'][j_cycle]]

            N=len(j_angles)
            breaks=np.r_[0, 1+np.nonzero(np.diff(j_angles))[0], N]

            for run_start,run_stop in zip(breaks[:-1],breaks[1:]):
                run_angle=j_angles[run_start]
                prv_angle=j_angles[(run_start-1)%N]
                nxt_angle=j_angles[(run_stop+1)%N]
                # Look for runs aligned with target_angle
                if run_angle != target_angle:
                    continue
                # And the ends must be perpendicular to target
                # specifically they need to be part of a tan_group, I think
                # Actually that's too restrictive.  They just need to *not*
                # be laplacian nodes.  But any node with a turn is not laplacian,
                # we're set.
                if False:
                    if (prv_angle%180) != ( (target_angle+90)%180):
                        continue
                    if (nxt_angle%180) != ( (target_angle+90)%180):
                        continue
                dist=el[j_cycle[run_start:run_stop]].sum()
                if dist>longest[0]:
                    # nth edge joins the nth node and n+1th node
                    n_start=cycle[run_start]
                    n_stop =cycle[run_stop%N] # maybe?
                    longest=(dist,n_start,n_stop)

            assert longest[1] is not None
            assert longest[2] is not None
            return longest

        # Can I really decide the sign here?  As long as they are consistent with each
        # other. The longest stretches need to be oriented, not just parallel
        i_longest=longest_straight(90)
        j_longest=longest_straight(0)
        
        i_dirichlet_nodes[i_longest[1]]=-1
        i_dirichlet_nodes[i_longest[2]]=1
        j_dirichlet_nodes[j_longest[1]]=-1
        if n_j_dirichlet==2:
            # When the signs were opposite this, tracing failed on phi
            j_dirichlet_nodes[j_longest[2]]=1

        self.i_dirichlet_nodes=i_dirichlet_nodes
        self.i_tan_groups=i_tan_groups
        self.i_grad_nodes=psi_gradient_nodes
        self.j_dirichlet_nodes=j_dirichlet_nodes
        self.j_tan_groups=j_tan_groups
        self.j_grad_nodes=phi_gradient_nodes

    def psi_phi_solve(self):
        gtri=self.g_int
        
        Mblocks=[]
        Bblocks=[]
        if 1: # PSI
            M_psi_Lap,B_psi_Lap=nd.construct_matrix(op='laplacian',
                                                    dirichlet_nodes=self.i_dirichlet_nodes,
                                                    zero_tangential_nodes=self.i_tan_groups,
                                                    gradient_nodes=self.i_grad_nodes)
            Mblocks.append( [M_psi_Lap,None] )
            Bblocks.append( B_psi_Lap )
        if 1: # PHI
            # including phi_gradient_nodes, and the derivative links below
            # is redundant but balanced.
            M_phi_Lap,B_phi_Lap=nd.construct_matrix(op='laplacian',
                                                    dirichlet_nodes=self.j_dirichlet_nodes,
                                                    zero_tangential_nodes=self.j_tan_groups,
                                                    gradient_nodes=self.j_grad_nodes)
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

            # Extra degrees of freedom:
            # Each tangent group leaves an extra dof (a zero row)
            # and the above BCs constrain 3 of those
            dofs=len(i_tan_groups) + len(j_tan_groups) - 3
            assert dofs>0
            
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
                         self.g_int.nodes['x'][j_d], va='bottom' ,
                         arrowprops=dict(arrowstyle='simple',alpha=0.4))

        from matplotlib import cm
        from itertools import cycle
        group_colors=cycle( list(colors.TABLEAU_COLORS.values()) )

        for idx,i_grp in enumerate(self.i_tan_groups):
            ax.plot( self.g_int.nodes['x'][i_grp,0],self.g_int.nodes['x'][i_grp,1],
                     '.',color=next(group_colors),label=f"i grp {idx}")

        for idx,j_grp in enumerate(self.j_tan_groups):
            ax.plot( self.g_int.nodes['x'][j_grp,0],self.g_int.nodes['x'][j_grp,1],
                     '+',color=next(group_colors),label=f"j grp {idx}")

        ax.legend(loc='upper right')

        i_quivs=np.array( [ [self.g_int.nodes['x'][n], self.i_grad_nodes[n] ]
                            for n in self.i_grad_nodes] )
        j_quivs=np.array( [ [self.g_int.nodes['x'][n], self.j_grad_nodes[n] ]
                            for n in self.j_grad_nodes] )

        if len(i_quivs):
            i_qkey=ax.quiver(i_quivs[:,0,0], i_quivs[:,0,1],
                             i_quivs[:,1,0], i_quivs[:,1,1],
                             color='k')
            ax.quiverkey(i_qkey,0.15,0.95,1.0,"I gradient",coordinates='figure')
        if len(j_quivs):
            j_qkey=ax.quiver(j_quivs[:,0,0], j_quivs[:,0,1],
                             j_quivs[:,1,0], j_quivs[:,1,1],
                             color='r')
            ax.quiverkey(j_qkey,0.3,0.95,1.0,"J gradient",coordinates='figure')

        if self.i_nf_cells:
            self.g_int.plot_cells(mask=self.i_nf_cells,color='r',alpha=0.4)
        if self.j_nf_cells:
            self.g_int.plot_cells(mask=self.j_nf_cells,color='0.6',alpha=0.4)
            
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

        This is more specific than just looking at gen's nodes['fixed'].
        Omit nodes that have an angle of 180
        """
        # This code assumes that either ij are both fixed, or neither fixed.
        fixed_int_to_gen={}
        for n in g_int.valid_node_iter():
            g_n=g_int.nodes['gen_n'][n]
            #if (g_n>=0) and (gen.nodes['fixed'][g_n]):
            if (g_n>=0) and (gen.nodes['turn'][g_n]!=180):
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
                        log.info("Not worrying about contour hitting diagonal")
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
            assert angle is not None,"Pretty sure this should always be supplied"
            exit_angle=angle
            for a in trace_nodes[:-1]:
                node_exits[a].append( (exit_angle,'internal') )
            exit_angle=(exit_angle+180)%360
            for b in trace_nodes[1:]:
                node_exits[b].append( (exit_angle,'internal') )

        def trace_and_insert_contour(b,angle):
            # does dij_angle fall between the angles formed by the boundary, including
            # a little slop.
            log.info(f"{angle} looks good")
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

        log.info("Bulk init with boundaries")
        # Bulk init with the points, then come back to fix metadata
        # Don't add the constraints, since that would mean adding
        # ragged edges potentially too early. This saves a huge amount
        # of time in building the DT, and the constraint handling is
        # relatively fast.
        g_final.bulk_init(g_int.nodes['x'][cycle])
        log.info("Tracing boundaries...")
        trace_and_insert_boundaries(cycle)
        log.info("done")

        # Add internal contours
        # return with internal.
        for a,b,c in zip(cycle,
                         np.roll(cycle,-1),
                         np.roll(cycle,-2)):
            # if b in [331,434]: # side-channel
            #     plt.figure(1).clf()
            #     g_int.plot_edges()
            #     g_int.plot_nodes(mask=g_int.nodes['rigid'],labeler='id')
            #     g_int.plot_nodes(mask=[a,c], labeler='id')
            #     g_int.plot_edges(mask=[j_ab,j_bc],labeler='angle')
            #     zoom=(552573.3257994705, 552606.492118541, 4124415.575118965, 4124440.2893760786)
            #     plt.axis(zoom)
            #     plt.draw()
            #     import pdb
            #     pdb.set_trace()

            if b not in fixed_int_to_gen: continue

            j_ab=g_int.nodes_to_edge(a,b)
            j_bc=g_int.nodes_to_edge(b,c)
            # flip to be the exit angle
            angle_ba = (180+g_int.edges['angle'][j_ab])%360
            angle_bc = g_int.edges['angle'][j_bc]

            b_final=None # lazy lookup
            for angle in [0,90,180,270]:
                # is angle into the domain?
                # cruft trace=None
                # if angle is left of j_ab and right of j_bc,
                # then it should be into the domain and can be traced
                # careful with sting angles
                # a,b,c are ordered CCW on the cycle, domain is to the
                # left.
                # so I want bc - angle - ba to be ordered CCW
                if ( ((angle_bc==angle_ba) and (angle!=angle_bc))
                     or (angle-angle_bc)%360 < ((angle_ba-angle_bc)%360) ):
                    if b_final is None:
                        b_final=g_final.select_nodes_nearest(g_int.nodes['x'][b],max_dist=0.0)
                    dupe=False
                    assert b_final is not None # should be in there from trace_boundaries
                    for exit_angle,exit_type in node_exits[b_final]:
                        if exit_angle==angle:
                            dupe=True
                            log.info("Duplicate exit for internal trace from %d. Skip"%b)
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

        # DBG
        self.g_not_final=g_final
        self.g_final2=g_final2
        
        # --- Compile Swaths ---
        e2c=g_final2.edge_to_cells(recalc=True)

        i_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool_)
        j_adj=np.zeros( (g_final2.Ncells(), g_final2.Ncells()), np.bool_)

        # tag ragged cells
        j_ragged=g_final2.edges['angle']%90 != 0.0
        c_ragged=np.unique( g_final2.edge_to_cells()[j_ragged,:] )
        c_ragged=c_ragged[c_ragged>=0]

        for j in g_final2.valid_edge_iter():
            c1,c2=e2c[j,:]
            if c1<0 or c2<0: continue
            # if the di of dij is 0, the edge joins cell in i_adj
            # I think angle==0 is the same as dij=[1,0]
            
            # Need to ignore ragged edges here -- if a contour intersects
            # a ragged edge, the join is not guaranteed at the ragged
            # boundary, and that can create a ragged cell that erroneously
            # joins two parallel swaths.  Not sure how this will affect
            # downstream handling of ragged cells, but try omitting them
            # entirely here.
            if (c1 in c_ragged) or (c2 in c_ragged):
                continue

            if g_final2.edges['angle'][j] % 180==0: # guess failed.
                i_adj[c1,c2]=i_adj[c2,c1]=True
            elif g_final2.edges['angle'][j] % 180==90:
                j_adj[c1,c2]=j_adj[c2,c1]=True
            else:
                log.warning("What? Ragged edge okay, but it shouldn't have both cell neighbors")

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
        self.patch_to_contour=patch_to_contour

        def add_swath_contours_new(comp_cells,node_field,coord,scale):
            if len(comp_cells)==1 and comp_cells[0] in c_ragged:
                # Ragged cells are handled afterwards by compiling
                # contours from neighboring cells
                return
            # Check all of the nodes to find the range ij
            comp_nodes=[ g_final2.cell_to_nodes(c) for c in comp_cells ]
            comp_nodes=np.unique( np.concatenate(comp_nodes) )
            comp_ijs=[] # Certainly could have kept this info along while building...

            field_values=[]

            # To do this, need to do it over all cells, not just picking comp_cells[0]
            field_min=np.inf
            field_max=-np.inf
            for comp_cell in comp_cells:
                comp_pp=np.array(g_final2.edges['psiphi'][ g_final2.cell_to_edges(comp_cell) ])
                # it's actually the other coordinate that we want to consider.
                field_min=min(field_min,np.nanmin( comp_pp[:,1-coord] ))
                field_max=max(field_max,np.nanmax( comp_pp[:,1-coord] ))

            # Could do this more directly from topology if it mattered..
            swath_poly=ops.cascaded_union( [g_final2.cell_polygon(c) for c in comp_cells] )
            swath_nodes=g_int.select_nodes_intersecting(swath_poly)
            swath_vals=node_field[swath_nodes]
            swath_grad=pp_grad[1-coord][swath_nodes] # right?
            order=np.argsort(swath_vals)
            o_vals=swath_vals[order]
            o_dval_ds=swath_grad[order]
            o_ds_dval=1./o_dval_ds

            uniform_scale=False

            if not uniform_scale:
                # Rather than using s as a distance in geographic space, which
                # scaled by a single value of local_scale, here normalize
                # ds by variable scale in o_scales, and consider s a distance in grid space
                o_scales=scale( g_int.nodes['x'][swath_nodes])[order]
                o_ds_dval /= o_scales

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

            n_swath_cells=None
            # def prescribed_cell_count():
            # want to get the edges in g_final2 for the two relevant ends of the
            # swath. Pull all edges from g_final2 that are on the boundary of the
            # swath
            swath_boundaries=[]
            e2c=g_final2.edge_to_cells()
            for comp_cell in comp_cells:
                swath_boundaries.append( np.nonzero( (e2c.min(axis=1)<0) & (e2c.max(axis=1)==comp_cell))[0] )
            swath_boundaries=np.concatenate(swath_boundaries)
            angle_to_nodes={} # angle => list of nodes on that border
            for angle in [0,90,180,270]:
                angle_to_edges=swath_boundaries[ g_final2.edges['angle'][swath_boundaries]==angle]
                angle_to_nodes[angle] = set( g_final2.edges['nodes'][angle_to_edges].ravel())

             # hoping these can be mapped back to gen...
            for angle in [0+90*coord,180+90*coord]:
                cornerA=(angle_to_nodes[angle] & angle_to_nodes[(angle-90)%360]).pop()
                cornerB=(angle_to_nodes[angle] & angle_to_nodes[(angle+90)%360]).pop()
                genA = self.gen.select_nodes_nearest( g_final2.nodes['x'][cornerA] )
                genB = self.gen.select_nodes_nearest( g_final2.nodes['x'][cornerB] )
                gen_edges = self.gen.shortest_path(genA,genB,
                                                   edge_selector=lambda j,direc: self.gen.edges['angle'][j]==angle,
                                                   return_type='edges')
                scales=self.gen.edges['scale'][gen_edges]
                if np.all(scales<0):
                    count=-scales.sum()
                    if n_swath_cells is None:
                        n_swath_cells=int(count)
                        print("Swath cell prescribed as ",n_swath_cells)
                    elif n_swath_cells!=count:
                        raise Exception("Both ends of this swath have set counts, but the counts don't match")
            
            if n_swath_cells is None:
                # This is just calculating a mean scale over the whole swath
                if uniform_scale:
                    local_scale=scale( g_int.nodes['x'][swath_nodes] ).mean(axis=0)
                else:
                    local_scale=1.0

                # s gives the average geographic distance along the swath.
                # this takes the average geographic length of the swath,
                # and divides by local scale to get the number of cells
                # It may be worthwhile to shift towards something more like SimpleSingleQuadGen --
                # advantage is that information from the user is more directly incorporated, including
                # distribution of edges. Downside is that it becomes a local determination and
                # ignores how the internal parts of the domain are turning/expanding/contracting.
                n_swath_cells=int(np.round( (s.max() - s.min())/local_scale))
                n_swath_cells=max(1,n_swath_cells)


            # Evenly divide up geographic space into the desired number of cells
            s_contours=np.linspace(s[0],s[-1],1+n_swath_cells)
            # Then map back to the node field values.
            adj_contours=np.interp( s_contours,
                                    s,o_vals)
            adj_contours[0]=field_min
            adj_contours[-1]=field_max

            assert np.all(np.diff(adj_contours)>0),"should be monotonic, right?"
            for c in comp_cells:
                patch_to_contour[coord][c]=adj_contours

        if 1: # Swath processing
            for coord in [0,1]: # i/j
                log.info("Coord: %s"%coord)
                if coord==0:
                    labels=labels_i
                    n_comp=n_comp_i
                    node_field=self.phi # feels backwards.. it's right, just misnamed 
                else:
                    labels=labels_j
                    n_comp=n_comp_j
                    node_field=self.psi

                    
                for comp in range(n_comp):
                    log.info("Swath: %s"%comp)
                    comp_cells=np.nonzero(labels==comp)[0]
                    add_swath_contours_new(comp_cells,node_field,coord,self.scales[coord])

            # come back to handle ragged cells
            for c in c_ragged:
                c_contours=[[],[]]

                for j in g_final2.cell_to_edges(c):
                    c1,c2 = g_final2.edges['cells'][j]
                    if min(c1,c2)<0: continue # only internal edges matter
                    if c1==c:
                        c_nbr=c2
                    elif c2==c:
                        c_nbr=c1
                    else:
                        raise Exception("Sanity lost")

                    if c_nbr in c_ragged:
                        log.info("Brave! Two ragged cells adjacent to each other")
                        continue
                    # similar logic as above
                    orient=g_final2.edges['angle'][j] % 180

                    if orient==0:
                        c_contours[0].append( patch_to_contour[0][c_nbr] )
                    elif orient==90:
                        c_contours[1].append( patch_to_contour[1][c_nbr] )
                # import pdb
                # pdb.set_trace()
                patch_to_contour[0][c]=np.unique(np.concatenate(c_contours[0]))
                patch_to_contour[1][c]=np.unique(np.concatenate(c_contours[1]))
                    
        # Direct grid gen from contour specifications:
        self.patch_grids=patch_grids=[]

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
                        log.info("fields_to_xy() failed, but cell is ragged.")
                        g_patch.delete_node_cascade(n)
                        continue 
                    else:
                        log.error("ERROR: fields_to_xy() failed. Cell not ragged.")
                g_patch.nodes['x'][n]=x
                # Hmm -
                # When it works, this probably reduces the search time considerably,
                # but there is the possibility, particularly at corners, that
                # this x will be a corner, that corner will lead to the cell *around*
                # the corner, and then we get stuck.
                # Even the centroid isn't great since it might not even fall inside
                # the cell.
                # x0=x

            # Cells are coming out CW, probably because phi has the opposite sign
            # relative j.
            # Maybe a better fix is to transpose the grid above.
            g_patch.orient_cells()
            patch_grids.append(g_patch)

        g=patch_grids[0].copy()
        for g_next in patch_grids[1:]:
            g.add_grid(g_next,merge_nodes='auto',tol=1e-6)

        return g

    def label_edge_orientation_and_scale(self,g):
        """
        Expects a 'pp' node field on g, uses that to 
        add an 'orient' field {0,90} to edges.  Then
        add target_l to each edge based on orientation and
        self.scales
        """
        j_orients=np.zeros( g.Nedges(), np.int32) - 1
        jns=g.edges['nodes']

        psi_match=(g.nodes['pp'][jns[:,0],0]==g.nodes['pp'][jns[:,1],0])
        phi_match=(g.nodes['pp'][jns[:,0],1]==g.nodes['pp'][jns[:,1],1])
        assert np.all( psi_match | phi_match )

        j_orients[psi_match]=0
        j_orients[phi_match]=90

        g.add_edge_field('orient',j_orients,on_exists='overwrite')

        ec=g.edges_center()
        target_scales=np.zeros(g.Nedges(),np.float64)
        target_scales[psi_match]=self.scales[0]( ec[psi_match] )
        target_scales[phi_match]=self.scales[1]( ec[phi_match] )
        assert np.all(target_scales>0)
        
        g.add_edge_field('target_l',target_scales,on_exists='overwrite')
        return target_scales

    def smooth_to_scale(self,g=None,smooth_iters=3,nudge_iters=2):
        """
        Wrapper for orthogonalize.Tweaker.smooth_to_scale
        """
        from . import orthogonalize
        if g is None:
            g=self.g_final
        target_scales=self.label_edge_orientation_and_scale(g)
        n_fixed,n_free=classify_nodes(g,self.gen)
        tweaker=orthogonalize.Tweaker(g)
        tweaker.smooth_to_scale(n_free,target_scales,
                                smooth_iters=smooth_iters,
                                nudge_iters=nudge_iters)

    def nudge_to_internal(self):
        """
        For internal breaklines nudge nodes onto the breakline
        """
        nudge_lines=self.find_nudge_lines()
        self.do_nudging(nudge_lines)
    def do_nudging(self, nudge_lines):
        # Do the actual nudging. For now, only nudge the specific nodes that fall on the
        # contour, but record all nudges
        g_final=self.g_final
        nudges=np.zeros( (g_final.Nnodes(),2), np.float64)

        for rec in nudge_lines:
            geom=geometry.LineString(rec['linestring'])
            orient=rec['orient']
            for n in g_final.valid_node_iter():
                if ( g_final.nodes['ij'][n,orient]==rec['contour']
                     and g_final.nodes['ij'][n,1-orient]>=rec['perp_range'][0]
                     and g_final.nodes['ij'][n,1-orient]<=rec['perp_range'][1]):
                    x_orig=g_final.nodes['x'][n]
                    pt,_ = ops.nearest_points(geom,geometry.Point(x_orig))
                    x_nudged=np.array([pt.x,pt.y])
                    nudges[n] += x_nudged - x_orig
                    g_final.nodes['x'][n] = x_nudged
        
    def find_nudge_lines(self):
        gen_orig=self.gen_orig # get gen before internal edges were removed.
        gen=self.gen
        g_final=self.g_final
        e2c=gen_orig.edge_to_cells()

        nudge_lines=[] # [ ( {0 for i, 1 for j}, contour value, nodestring from gen_orig, linestring), ]
        internal=(e2c.min(axis=1)>=0)

        internal_linestrings=[]
        for orient in [0,90]:
            # Careful with internal linestrings that intersect. Search separately in
            # each orientation, and override the default behavior that stops on node
            # degree>2
            edge_sel=internal & (gen_orig.edges['angle']%180==orient)
            orient_ls=gen_orig.extract_linear_strings(edge_select=edge_sel,
                                                      end_func=lambda n,js: len(js)<2)
            internal_linestrings.extend(orient_ls)


        node_to_ij={} # map node index of gen_orig to ij value in g_final, based on exact matches.
        for n in gen_orig.valid_node_iter():
            n_final = g_final.select_nodes_nearest(gen_orig.nodes['x'][n])
            dist = utils.dist( gen_orig.nodes['x'][n], g_final.nodes['x'][n_final])
            if dist<1e-5:
                node_to_ij[n] = g_final.nodes['ij'][n_final]
        
        for internal_ls in internal_linestrings:
            # HERE:
            # Rather than all of this craziness -- can it be more of a graph search question?
            # From each end of the line string, search gen_orig, accumulating the scales along
            # the way if they are of the right orientation, until reaching a node that has
            # i,j values in g_final.
            # possible that we need to update gen to include that information?
            # the main caveat is that angles in gen_orig are ambiguous % 180.
            # but the shortest path I think will take care of that.
            # the remaining issue is that it won't know if the coordinate should be
            # increasing or decreasing.
            # Really just need a way to set di/dj on all edges of gen_orig.

            # start a search from each end
            # Trying to get to any node for which we can identify i,j
            # g_final has ij

            end_nodes=[internal_ls[0],internal_ls[-1]]

            end_ijs=[]
            for end_node in end_nodes:
                node_paths=gen_orig.shortest_path(n1=end_node,
                                                  n2=list(node_to_ij.keys()),
                                                  return_type='nodes')
                ij_recon=[np.nan,np.nan]
                for (n_target,node_path) in node_paths:
                    dists={0:0.0, 90:0.0, 180:0.0, 270:0.0 }
                    for a,b in zip(node_path[:-1],node_path[1:]):
                        he=gen_orig.nodes_to_halfedge(a,b)
                        scale=gen_orig.edges['scale'][he.j]
                        if scale<0:
                            steps = -scale
                        else:
                            # only count up cases where the steps are exact.
                            # otherwise will revert to nearest node check
                            steps=np.nan 
                        angle=(gen_orig.edges['angle'][he.j] + he.orient*180)%360
                        dists[angle]+=steps
                    dist90=dists[90] - dists[270]
                    dist0=dists[0] - dists[180] # this one has not been tested.

                    one_ij_recon=node_to_ij[n_target] - np.r_[dist90,dist0]
                    for ii in [0,1]:
                        if np.isnan(one_ij_recon[ii]): continue
                        if np.isnan(ij_recon[ii]):
                            ij_recon[ii]=one_ij_recon[ii]
                        else:
                            # hopefully these match
                            print(f"reconstructured coord: {ij_recon[ii]} vs {one_ij_recon[ii]}")
                for ii in [0,1]:
                    if np.isfinite(ij_recon[ii]): continue
                    # This isn't great -- ideally we'd get a fractional value here.
                    nodes_final=g_final.select_nodes_nearest(gen_orig.nodes['x'][end_node],count=4)
                    values=g_final.nodes['ij'][nodes_final,ii]
                    # linear interp or inv-distance weighted would be better, but this is probably
                    # sufficient -- just need to know if a g_final node is inside or outside the
                    # range.
                    ij_recon[ii]=values.mean()
                end_ijs.append(ij_recon)
            # sign and 0/90 bugs to test!
            # seems to have worked..

            # And what is the orientation for the line?
            parallel_angle=gen_orig.edges['angle'][ gen_orig.nodes_to_edge(internal_ls[:2]) ]
            if parallel_angle%180==0:
                orient=0 # TEST
            else:
                orient=1
            
            # package up the line:
            # generally should be the same on the two ends, but just in case:
            contour_value=int( np.round(0.5*(end_ijs[0][orient]+end_ijs[1][orient])))
            # ideally also make sure contour_value is not on the boundary!
            min_value=g_final.nodes['ij'][:,orient].min()
            max_value=g_final.nodes['ij'][:,orient].max()
            # This is not sufficient for non-rectangular cases...
            if contour_value==min_value: contour_value+=1
            if contour_value==max_value: contour_value-=1
            #gen.plot_nodes(mask=internal_ls,labeler=lambda i,r: contour_value)
            #g_final.plot_nodes( labeler=lambda i,r: r['ij'][orient])

            perp_values=[end_ijs[0][1-orient], end_ijs[1][1-orient]]

            if 0:
                linestring=gen_orig.nodes['x'][internal_ls]
            else:
                linestring=self.gen_bezier_linestring(nodestring=internal_ls,gen=gen_orig)

            nudge_lines.append( dict(orient=orient,
                                     contour=contour_value,
                                     perp_range=[min(perp_values),max(perp_values)],
                                     nodestring=internal_ls,
                                     linestring=linestring) )

        return nudge_lines

        
        
class SimpleQuadGen(object):
    """
    A streamline quad generator that is more usable.
    - each cell must map to a rectangle, and the smallest 4 internal angles
      will be automatically labeled as 90-degree turns, the rest 180.
    - all edges shared by two cells must have a specific count of nodes, given
      by a negative scale value. (though by default this is enforced in a
      preprocessing step with adjust_scale=True)
    """
    nom_res=3.5 # needs to be adaptive..
    triangle_method='gmsh'
    gmsh_path='gmsh'
    merge_tol=0.01
    
    def __init__(self,gen,cells=None,execute=True,adjust_scale=True,**kw):
        """
        adjust_scale: make a copy of the edge array and update scale to be sure that 
        shared edges always have a given edge count
        """
        utils.set_keywords(self,kw)
        self.gen=gen
        self.cells=cells
        if adjust_scale:
            self.gen.edges=self.gen.edges.copy()
            self.gen.edges['scale']=adjusted_scale(self.gen,self.gen.edges['scale'],self.nom_res)
        if execute:
            self.execute()
            
    def execute(self):
        snap_angles(self.gen)
        prepare_angles_halfedge(self.gen)
        add_bezier(self.gen)

        self.grids=[]
        self.qgs=[]
        
        for c in self.cells:
            qg=self.process_one_cell(c)
            self.grids.append(qg.g_final)

        if self.grids:
            self.g_final=self.merge_grids()
            return self.g_final
    
    def merge_grids(self):
        g=unstructured_grid.UnstructuredGrid(max_sides=4)
        for sub_g in self.grids:
            g.add_grid(sub_g,merge_nodes='auto',tol=self.merge_tol)
        return g
            
    def process_one_cell(self,c,**kw):
        kwargs=dict(cell=c,
                    triangle_method=self.triangle_method,
                    gmsh_path=self.gmsh_path,
                    nom_res=self.nom_res)
        kwargs.update(kw)
        qg=SimpleSingleQuadGen(self.gen,**kwargs)
        self.qgs.append(qg)
        qg.execute()
        return qg

def patch_contours(g_int,node_field,scale,count=None,Mdx=None,Mdy=None,
                   lowpass_ds_dval=True):
    """
    Given g_int, a node field (psi/phi) defined on g_int, a scale field, and 
    a count of edges, return the contour values of the node field which
    would best approximate the requested scale.
    
    g_int: UnstructuredGrid
    node_field: a psi or phi field defined on the nodes of g_int
    scale: length scale Field with domain include g_int
    count: if specified, the number of nodes in the resulting discretization
    
    Mdx,Mdy: matrix operators to calculate derivatives of the node field. 
      by default create from scratch

    returns the contour values (one more than the number of edges)
    """
    field_min=node_field.min()
    field_max=node_field.max()

    # original swath code had to pull out a subset of node in the node
    # field, but now we can assume that g_int is congruent to the target
    # patch.
    swath_nodes=np.arange(g_int.Nnodes())
    swath_vals=node_field[swath_nodes]

    # preprocessing for contour placement
    nd=NodeDiscretization(g_int)
    if Mdx is None:
        Mdx,_=nd.construct_matrix(op='dx') 
    if Mdy is None:
        Mdy,_=nd.construct_matrix(op='dy') 
        
    field_dx=Mdx.dot(node_field)
    field_dy=Mdy.dot(node_field)
    field_grad=np.sqrt( field_dx**2 + field_dy**2 )
    swath_grad=field_grad

    order=np.argsort(swath_vals)
    o_vals=swath_vals[order]
    o_dval_ds=swath_grad[order] # s: coordinate perpendicular to contours
    local_scale=scale( g_int.nodes['x'][swath_nodes[order]] )

    # local_scale is ds/di or ds/dj, so now s is approx. grid index
    o_ds_dval=1./(o_dval_ds*local_scale)

    # trapezoid rule integration
    d_vals=np.diff(o_vals)
    # Particularly near the ends there are a lot of
    # duplicate swath_vals.
    # Try a bit of lowpass to even things out.
    if lowpass_ds_dval:
        # HERE this needs to be scale-aware!!
        # could use count, but we may not have it, and
        # it ends up being evenly spread out, which isn't
        # ideal.
        # How bad is it to just drop this? Does have an effect,
        # particularly in the lateral
        if count is None:
            # There is probably something more clever to do using the actual
            # swath vals.
            winsize=int(len(o_vals)/10)
        else:
            winsize=int(len(o_vals)/count)
        if winsize>1:
            o_ds_dval=filters.lowpass_fir(o_ds_dval,winsize)
    else:
        print("No lowpass on ds_dval")

    s=np.cumsum(d_vals*0.5*(o_ds_dval[:-1]+o_ds_dval[1:]))
    s=np.r_[0,s]

    # calculate this from resolution
    # might have i/j swapped.  range of s is 77m, and field
    # is 1. to 1.08.  better now..
    if count is None:
        count=max(2,int(np.round(s.max())))

    s_contours=np.linspace(s[0],s[-1],count)
    adj_contours=np.interp( s_contours,
                            s,o_vals)
    
    adj_contours[0]=field_min
    adj_contours[-1]=field_max

    assert np.all(np.diff(adj_contours)>0),"should be monotonic, right?"

    return adj_contours


class SimpleSingleQuadGen(QuadGen):
    """
    Rewrite of portions of QuadGen to handle the local single-cell portion.
    """
    triangle_method='gmsh'
    angle_source='existing'
    # The lowpass seems like it would defeat some of the scale handling.
    lowpass_ds_dval=True
    smooth_patch=True
    
    def __init__(self,gen,cell,**kw):
        super(SimpleSingleQuadGen,self).__init__(gen,execute=False,cells=[cell],
                                                 **kw)

    def discretize_perimeter(self):
        """
        up-sample the bezier curves of the generating grid, 
        populating 
         self.perimeter: [N,{x,y}]
         self.node_to_idx: node index of self.gen mapped to index into perimeter
         self.angle_to_segments: map each of the 4 cardinal angles to a list of segments,
           proceeding CCW around the cell. each segment is a start node end node, and a count
             if count was specified in the input (i.e. gen.edges['scale'] has negative value)
         self.perimeter_scale: [N] - linearly interpolated scale along the perimeter nodes
        """
        def he_angle(he):
            # return (he.grid.edges['angle'][he.j] + 180*he.orient)%360.0
            # Since this is being used after the internal edges handling,
            # angles are oriented to the cycle of the cell, not the natural
            # edge orientation.
            return he.grid.edges['angle'][he.j]

        # Start at a corner
        he=self.gen.cell_to_halfedge(0,0)
        while 1:
            he_fwd=he.fwd()
            corner= he_angle(he) != he_angle(he_fwd)
            he=he_fwd
            if corner:
                break

        he0=he
        idx=0 # current location into list of perimeter samples
        perimeter=[]
        perimeter_scales=[] # [N,2] scale interpolated along boundary
        
        node_to_idx={}
        angle_to_segments={0:[],
                           90:[],
                           180:[],
                           270:[]}

        last_fixed_node=he.node_rev()

        # scale here is just for edges with exact number of
        # cells (negative).
        if 'scale' in self.gen.edges.dtype.names:
            edge_scale=self.gen.edges['scale']
        else:
            edge_scale=np.zeros(self.gen.Nedges())
        
        while 1:
            pnts=self.gen_bezier_linestring(j=he.j,span_fixed=False)
            
            if he.orient:
                pnts=pnts[::-1]
            perimeter.append(pnts[:-1])

            nA=he.node_rev()
            nB=he.node_fwd()

            # It's possible to trace the nodes here back to nodes of the original
            # gen, and then query the scale values directly.  But note that
            # these are not the same node indices, since gen has been renumbered
            # after selecting a single cell. For now, just interpolate:
            scales0=np.linspace( self.scales[0](pnts[0]), self.scales[0](pnts[-1]), len(pnts))
            scales1=np.linspace( self.scales[1](pnts[0]), self.scales[1](pnts[-1]), len(pnts))
            
            scales=np.c_[scales0,scales1]
            perimeter_scales.append(scales[:-1])
            
            node_to_idx[nA]=idx
            idx+=len(pnts)-1
            he_fwd=he.fwd()
            angle=he_angle(he)
            angle_fwd=he_angle(he_fwd)
                
            if  ( (angle!=angle_fwd) # a corner
                  or (edge_scale[he.j]<0)
                  or (edge_scale[he_fwd.j]<0) ):
                if edge_scale[he.j]<0:
                    count=-int( edge_scale[he.j] )
                else:
                    count=0
                angle_to_segments[angle].append( [last_fixed_node,he.node_fwd(),count] )
                last_fixed_node=he.node_fwd()

            he=he_fwd
            if he==he0:
                break
        self.perimeter=np.concatenate(perimeter)
        self.perimeter_scales=np.concatenate(perimeter_scales)
        self.angle_to_segments=angle_to_segments
        self.node_to_idx=node_to_idx

    def discretize_string(self,string,density):
        """
        string: a node string with counts, 
           [ (start node, end node, count), ... ]
           where a count of 0 means use the provided density
        density: NEW: density evaluated at perimeter points. (old: a density field)
        returns: (N,2) discretized linestring and (N,) bool array of
         rigid-ness.
        """
        result=[]
        rigids=[]

        for a,b,count in string:
            if count==0:
                idx_a=self.node_to_idx[a]
                idx_b=self.node_to_idx[b]
                if idx_a<idx_b:
                    pnts=self.perimeter[idx_a:idx_b+1]
                    dens_pc=density[idx_a:idx_b+1]
                else:
                    pnts=np.concatenate( [self.perimeter[idx_a:],
                                          self.perimeter[:idx_b+1]] )
                    dens_pc=np.concatenate( [density[idx_a:],
                                             density[:idx_b+1]] )
                assert len(pnts)>0
                # To defend against discontinuity in the density field,
                # use a precalculated density continuous along the linestring,
                # dens_pc=density(pnts) # old usage
                seg=linestring_utils.resample_linearring(pnts,dens_pc,closed_ring=0)
                rigid=np.zeros(len(seg),np.bool_)
                rigid[0]=rigid[-1]=True
            else:
                pnt_a=self.gen.nodes['x'][a]
                pnt_b=self.gen.nodes['x'][b]
                j=self.gen.nodes_to_edge(a,b)
                
                if j is None:
                    log.warning("Rigid segment doesn't match a single edge in gen?")
                    seg=np.c_[ np.linspace(pnt_a[0], pnt_b[0], count+1),
                               np.linspace(pnt_a[1], pnt_b[1], count+1) ]
                else:
                    # Try using bezier:
                    # need to be sure that the result doesn't have any dependence on which of the
                    # two cells we're processing, so that each patch will get the same
                    # nodes here (bit-equal not necessary since the grids will be merged
                    # with some small tolerance, but we should be talking machine-precision)
                    # Should be okay.
                    # count is correct, shouldn't have +1 here.
                    seg=self.gen_bezier_linestring(j=j,samples_per_edge=count,span_fixed=False)
                    if self.gen.edges['nodes'][j,1]==a:
                        seg=seg[::-1]

                rigid=np.ones(len(seg),np.bool_)
            result.append(seg[:-1])
            rigids.append(rigid[:-1])
        result.append( seg[-1:] )
        rigids.append( rigid[-1:] )
        result=np.concatenate(result)
        rigids=np.concatenate(rigids)
        return result,rigids

    def calculate_coord_count(self,left,right,density):
        """
        Shift node densities to get the count on opposite sides to match up.
        left,right: lists of segments as in self.angle_to_segments (node start, stop, count)
        density: the coordinate-specific density field, already evaluated at perimeter points.

        returns: (points along left, rigid markers for left), (points along right, rigid markers for right)
        NB: the assumption is that left and right have opposite orientations (i.e. both are CCW, so anti-parallel)
        the right sequences are reversed to make them parallel.
        """
        af_low=-5
        af_high=5
        while 1:
            af=(af_low+af_high)/2
            assert abs(af)<4.9,"af=%s"%af
            if not af_high-af_low>1e-8:
                raise Exception("Calculate coord count failed to converge")
                #import pdb
                #pdb.set_trace()
            pnts0,rigid0=self.discretize_string(left,(2**af)*density)
            pnts1,rigid1=self.discretize_string(right,(0.5**af)*density)
            c0=len(pnts0)
            c1=len(pnts1)
            if c0==c1:
                break
            # 0,180: positive af makes c1 larger
            if c0>c1: #  af should be larger
                af_low=af
                continue
            if c0<c1:
                af_high=af
                continue
        return (pnts0,rigid0),(pnts1[::-1],rigid1[::-1])

    def select_node_counts(self):
        """
        Solve for the count of nodes
        First get the full perimeter at 10 point per bezier segment
        Group edges by angle.
        Within each group, consecutive edges with non-negative scale
        are treated as a unit.
        af is an asymmetry factor.
        It starts at 0.  We go through the grouped edges, count up
        the number of nodes, and see if opposite edges agree.  If they
        don't agree, af is adjusted.

        Sets (left_i,left_i_rigid),(right_i,right_i_rigid) and 
        (left_j,left_j_rigid),(right_j,right_j_rigid)
        """
        self.discretize_perimeter()

        # Precalculate density fields along perimeter:

        (left_i,left_i_rigid),(right_i,right_i_rigid)=self.calculate_coord_count(self.angle_to_segments[0],
                                                                                 self.angle_to_segments[180],
                                                                                 self.perimeter_scales[:,0])
        (left_j,left_j_rigid),(right_j,right_j_rigid)=self.calculate_coord_count(self.angle_to_segments[90],
                                                                                 self.angle_to_segments[270],
                                                                                 self.perimeter_scales[:,1])

        # Necessary to have order match grid order below
        left_i=left_i[::-1]
        left_i_rigid=left_i_rigid[::-1]
        right_i=right_i[::-1]
        right_i_rigid=right_i_rigid[::-1]

        self.left_i=left_i
        self.left_i_rigid=left_i_rigid
        self.left_j=left_j
        self.left_j_rigid=left_j_rigid
        
        self.right_i=right_i
        self.right_i_rigid=right_i_rigid
        self.right_j=right_j
        self.right_j_rigid=right_j_rigid

    def create_patch(self):
        """
        Using the count information in self.left_{i,j}, create the rectilinear patch,
        and return the cell/node map, too.
        """
        patch=unstructured_grid.UnstructuredGrid(max_sides=4,
                                                 extra_node_fields=[('rigid',np.bool_),
                                                                    ('pp',np.float64,2)],
                                                 extra_edge_fields=[('orient',np.float32)])
        elts=patch.add_rectilinear( [0,0],
                                    [len(self.left_i)-1, len(self.left_j)-1],
                                    len(self.left_i), len(self.left_j),
                                    reverse_cells=True)
        
        # Fill in orientation
        segs=patch.nodes['x'][ patch.edges['nodes'] ]
        deltas=segs[:,1,:] - segs[:,0,:]

        patch.edges['orient'][deltas[:,0]==0]=90 # should be consistent with gen.edges['angle']
        patch.edges['orient'][deltas[:,1]==0]=0

        return patch,elts

    def field_interpolators(self):
        """
        Return two interpolating function that map X=>psi/phi.
        Some care is taken to return reasonable values when X is slightly outside
        self.g_int.
        """
        finder=PermissiveFinder(self.g_int)
        psi_interp=LinearTriInterpolator(finder.mp_tri,self.psi,finder)
        phi_interp=LinearTriInterpolator(finder.mp_tri,self.phi,finder)
        psi_field=lambda x: psi_interp(x[...,0],x[...,1]).filled(np.nan)
        phi_field=lambda x: phi_interp(x[...,0],x[...,1]).filled(np.nan)
        return psi_field,phi_field
    
    def execute(self):
        # Note that self.gen is *not* the same grid as gen that is passed in.
        # self.gen has a single cell, while gen is the original input with potentially
        # many cells
        self.process_internal_edges(self.gen) # N.B. this flips angles
        self.g_int=self.create_intermediate_grid_tri()
        self.calc_psi_phi()

        # got a nice psi/phi field.
        # I have some edges with negative scale.  Other edges
        # will get just the ambient scale.

        self.select_node_counts()

        # The target contour values, if there weren't any fixed nodes aside from corners:
        nd=NodeDiscretization(self.g_int)
        Mdx,_=nd.construct_matrix(op='dx') # discard rhs, it's all 0.
        Mdy,_=nd.construct_matrix(op='dy') # discard rhs
        self.i_contours=patch_contours(self.g_int,self.phi,self.scales[0], len(self.left_i),
                                       Mdx=Mdx,Mdy=Mdy,lowpass_ds_dval=self.lowpass_ds_dval)
        self.j_contours=patch_contours(self.g_int,self.psi,self.scales[1], len(self.left_j),
                                       Mdx=Mdx,Mdy=Mdy,lowpass_ds_dval=self.lowpass_ds_dval)

        # Now I have the target psi/phi contours
        # I know which nodes should be rigid, and their locations.

        self.patch,self.patch_elts=self.create_patch()
        self.g_final=self.patch

        self.psi_field,self.phi_field=self.field_interpolators()

        self.position_patch_nodes()
        self.patch.cells['_center']=np.nan # otherwise all 0...
        
        if self.smooth_patch:
            self.smooth_patch_psiphi_implicit()
        
        self.patch.cells['_area']=np.nan # force recalculation
        # heavy handed -- could be done more gracefully during construction
        # in fact, this shouldn't be necessary at all anymore.
        self.patch.orient_cells() 
        return self.patch

    def position_patch_nodes(self):
        """
        Use i_contours,j_contours to set psi/phi for non-rigid nodes.
        Use left_i/right_i/left_j/right_j to set x for rigid nodes.

        Fill in nodes['pp'] and ['x']. Non-rigid nodes get a target pp, from which
        we calculate x.  Rigid nodes get a prescribed x, from which we calculate pp.
        """

        for i in range(len(self.left_i)):
            for j in range(len(self.left_j)):
                n=self.patch_elts['nodes'][i,j]
                rigid=True

                if i==0 and self.left_j_rigid[j]:
                    x=self.left_j[j]
                elif i+1==len(self.left_i) and self.right_j_rigid[j]:
                    x=self.right_j[j]
                elif j==0 and self.left_i_rigid[i]:
                    x=self.left_i[i]
                elif j+1==len(self.left_j) and self.right_i_rigid[i]:
                    x=self.right_i[i]
                else:
                    rigid=False
                    pp=[self.j_contours[j],
                        self.i_contours[-i-1]] # I think I have to reverse i

                    x=self.g_int.fields_to_xy(pp,[self.psi,self.phi],x)

                if rigid:
                    pp=[min(1,max(-1,self.psi_field(x))),
                        min(1,max(-1,self.phi_field(x)))]

                self.patch.nodes['x'][n]=x
                self.patch.nodes['pp'][n]=pp
                self.patch.nodes['rigid'][n]=rigid

    def smooth_patch_psiphi_implicit(self,aniso=0.02,monotonify=True):
        """
        Anisotropic, implicit smoothing of psi/phi values.
        Using the target contours in self.{i,j}_contours
        and the existing values in patch.nodes['pp'], adjust
        patch.nodes['pp'] to smoothly transition between the
        rigid and non-rigid areas. 
        aniso controls how much smaller the off-axis smoothing
        is.  "On-axis" means how much each grid line follows a
        contour, and "off-axis" means how much the spacing between
        grid lines is evened out.

        monotonify: if True, then check for monotonicity of coordinates
        and iteratively decrease anisotropy until monotonic. 
        This is a hack fallback for boundary spacing that's not even,
        when the default anisotropy would cause self-intersections.
        There might be a better solution where the requested resolution
        information could be incorporated into the psiphi setup. The
        goal of this hack is to make sure the generation step completes.

        """
        patch=self.patch
        patch_nodes=self.patch_elts['nodes']

        target_pp=np.zeros( (patch.Nnodes(),2),np.float64)
        target_pp[patch_nodes,0]=self.j_contours[None,:]
        target_pp[patch_nodes,1]=self.i_contours[::-1,None]
        dpp=patch.nodes['pp']-target_pp

        rigid_r=patch.nodes['rigid'][patch_nodes]
        rigid_r0=rigid_r.copy() # coordinate 0
        rigid_r1=rigid_r.copy()
        rigid_r0[:,0]=True
        rigid_r0[:,-1]=True
        rigid_r1[0,:]=True
        rigid_r1[-1,:]=True

        N=patch.Nnodes()

        dpp_smooths=[None,None]
        nrows,ncols=patch_nodes.shape

        # dc/dt= d/dx Kx dc/dx + d/dy Ky dc/dy
        # steady state, uniform K, dx=dy=1
        #  0 = Kx c[-1,0] + Kx c[1,0] + Ky c[0,-1] + Ky c[0,1] - (2Kx+2Ky) c[0,0]
        for coord in [0,1]:
            adj_aniso=aniso
            while 1: # monotonify loop

                M=sparse.dok_matrix( (N,N), np.float64)
                b=np.zeros(N,np.float64)
                K=[1,1]
                K[1-coord]*=adj_aniso
                rigid_this_coord=[rigid_r0,rigid_r1][coord]
                dirich=dpp[:,coord]

                for row in range(nrows):
                    for col in range(ncols):
                        n=patch_nodes[row,col]
                        if rigid_this_coord[row,col]:
                            M[n,n]=1
                            b[n]=dirich[n]
                        else:
                            b[n]=0.0
                            M[n,n]=-2*(K[0]+K[1])
                            if row==0:
                                M[n,patch_nodes[row+1,col]]=2*K[0]
                            elif row==nrows-1:
                                M[n,patch_nodes[row-1,col]]=2*K[0]
                            else:
                                M[n,patch_nodes[row-1,col]]=K[0]
                                M[n,patch_nodes[row+1,col]]=K[0]
                            if col==0:
                                M[n,patch_nodes[row,col+1]]=2*K[1]
                            elif col==ncols-1:
                                M[n,patch_nodes[row,col-1]]=2*K[1]
                            else:
                                M[n,patch_nodes[row,col-1]]=K[1]
                                M[n,patch_nodes[row,col+1]]=K[1]

                dpp_smooths[coord]=sparse.linalg.spsolve(M.tocsr(),b)
                if monotonify:
                    # Check for monotonicity
                    result=target_pp[...,coord] + dpp_smooths[coord]
                    result_r=result[patch_nodes]
                    if coord==0:
                        if np.all(np.diff(result_r,axis=1-coord)>0):
                            break
                    else: # coord==1
                        if np.all(np.diff(result_r,axis=1-coord)<0):
                            break
                    adj_aniso*=2
                    if adj_aniso>=1:
                        log.warning("Could not adjust anisotropy enough to regain monotonicity")
                        break
                    else:
                        log.warning("Will smooth with decreased anisotropy (%g)"%adj_aniso)
                else:
                    break
                
        dpp_smooth=np.c_[dpp_smooths[0],dpp_smooths[1]]
        
        # Copy back to pp
        sel=~patch.nodes['rigid']
        patch.nodes['pp'][sel] = target_pp[sel] + dpp_smooth[sel]

        # And remap those nodes:
        for n in np.nonzero(~patch.nodes['rigid'])[0]:
            x_orig=patch.nodes['x'][n]
            patch.nodes['x'][n]=self.g_int.fields_to_xy(patch.nodes['pp'][n],
                                                        [self.psi,self.phi],
                                                        x_orig)

    def nudge_boundaries_monotonic(self):
        """
        Check that boundary nodes have monotonic psi/phi, and 
        nudge any violating nodes to be linearly interpolated between
        the okay nodes.

        Updates pp and x for offending nodes.
        This was a halfway solution for non-monotonic nodes before
        the smoothing code above was adapted to adjust for non-monotonicity.
        Currently nothing uses this method.
        """
        nodes_r=self.patch_elts['nodes']

        for nlist,coord,sign in [ (nodes_r[:,0],1,-1),
                                  (nodes_r[:,-1],1,-1),
                                  (nodes_r[0,:],0,1),
                                  (nodes_r[-1,:],0,1)]:
            # nlist: node indices into patch
            # coord: which coordinate of pp to adjust
            # sign: +1 for increasing, -1 for decreasing
            vals=self.patch.nodes['pp'][nlist,coord]
            if np.all(sign*np.diff(vals)>0): continue

            rigid=self.patch.nodes['rigid'][nlist]
            # At least the rigid ones better be monotonic.
            assert np.all(sign*np.diff(vals[rigid]))>0

            i=np.arange(len(vals))
            # each entry is the index of the next rigid node, self included.
            i_next=i[rigid][ np.searchsorted(i[rigid],i) ]
            # each entry is the index of the previous rigid node, self included
            i_prev=i[rigid][::-1][ np.searchsorted(-i[rigid][::-1],-i) ]
            assert np.all( i_prev[ i[rigid] ] == i[rigid] )
            assert np.all( i_next[ i[rigid] ] == i[rigid] )

            bad= (~rigid) & ( (sign*vals[i_prev] >= sign*vals) | (sign*vals[i_next]<=sign*vals))
            print(bad.sum())
            vals[bad] = np.interp(i[bad], i[~bad], vals[~bad] )
            self.patch.nodes['pp'][nlist,coord]=vals
            for n in nlist[bad]:
                x=self.g_int.fields_to_xy(self.patch.nodes['pp'][n],[self.psi,self.phi],
                                          self.patch.nodes['x'][n])
                self.patch.nodes['x'][n]=x
                
    def smooth_patch_psiphi(self,n_iter=10):
        """
        Smooth out the deviations from target i_contour and j_contour, blending
        from rigid nodes to the smooth interior
        """
        elts=self.patch_elts
        patch=self.patch
        
        target_pp=np.zeros( (patch.Nnodes(),2),np.float64)
        target_pp[elts['nodes'],0]=self.j_contours[None,:]
        target_pp[elts['nodes'],1]=self.i_contours[::-1,None]
        dpp=patch.nodes['pp']-target_pp

        # Move to 2D array to streamline the smoothing
        dpp_r=dpp[elts['nodes']].copy()
        rigid_r=patch.nodes['rigid'][elts['nodes']]
        rigid_r0=rigid_r.copy() # coordinate 0
        rigid_r1=rigid_r.copy()
        rigid_r0[:,0]=True # I think this is correct.
        rigid_r0[:,-1]=True
        rigid_r1[0,:]=True
        rigid_r1[-1,:]=True

        # could also use a larger window...
        #win=np.array([0.5,0,0.5])
        win=np.ones(3)/3.
        
        for it in range(n_iter):
            smooth=0*dpp_r
            # Smoothing is only along the respective coordinate. I.e. phi
            # anomalies are smoothed along contours of phi, and psi anomalies
            # are smoothed along contours of psi.
            smooth[...,0]=signal.fftconvolve(dpp_r[...,0],win[:,None],mode='same')
            smooth[...,1]=signal.fftconvolve(dpp_r[...,1],win[None,:],mode='same')
            if 1: # well, in some cases might need this...
                smooth[...,0]=signal.fftconvolve(smooth[...,0],win[None,:],mode='same')
                smooth[...,1]=signal.fftconvolve(smooth[...,1],win[:,None],mode='same')

            # Just update the non-rigid nodes:
            dpp_r[~rigid_r0,0]=smooth[~rigid_r0,0]
            dpp_r[~rigid_r1,1]=smooth[~rigid_r1,1]

        dpp=0*patch.nodes['pp']
        dpp[elts['nodes']] = dpp_r

        # Copy back to pp
        sel=~patch.nodes['rigid']
        patch.nodes['pp'][sel] = target_pp[sel] + dpp[sel]

        # And remap those nodes:
        for n in np.nonzero(~patch.nodes['rigid'])[0]:
            x_orig=patch.nodes['x'][n]
            patch.nodes['x'][n]=self.g_int.fields_to_xy(patch.nodes['pp'][n],
                                                        [self.psi,self.phi],
                                                        x_orig)
            
        # Invalid circumcenters.  May need to invalidate other geometry, too.
        patch.cells['_center']=np.nan
##

# # HERE: try some relaxation approaches.
# #   First, pin the known fixed nodes, and don't worry about
# #   trying to keep everybody on the bezier boundary.
# 
# # rigid-ness is carried through from the discretized nodestrings,
# # with nodes with negative scale and corner nodes set as rigid
# 
# from stompy.grid import orthogonalize
# tweaker=orthogonalize.Tweaker(patch)
# 
# # First, just nudge everybody towards orthogonal:
# # BAD.  Too far out of orthogonal.
# for n in patch.valid_node_iter():
#     if patch.nodes['rigid'][n]: continue
#     tweaker.nudge_node_orthogonal(n)
# 
# plt.figure(1)
# plt.cla()
# patch.plot_nodes(mask=patch.nodes['rigid'],color='r',sizes=30)
# patch.plot_edges()
# #plt.axis( (552066.9646997608, 552207.1805374735, 4124548.347825134, 4124660.504092434) )
# plt.axis( (552447.0573990112, 552507.7547532236, 4124244.839335523, 4124293.3901183335) )
# 
# ## 
# from stompy.grid import orthogonalize
# tweaker=orthogonalize.Tweaker(patch)
# 
# n_free=np.nonzero(~patch.nodes['rigid'])[0]
# edge_scales=np.zeros(patch.Nedges(),np.float64)
# ec=patch.edges_center()
# 
# for orient,scale in zip( [0,90], qg.scales):
#     sel=patch.edges['orient']==orient
#     edge_scales[sel] = scale(ec[sel])
# 
# ##
# 
# # This produces okay results, but it's going to be super slow
# # to converge.
# tweaker.smooth_to_scale( n_free, edge_scales,
#                          smooth_iters=1,nudge_iters=1)
#     
# plt.figure(1)
# plt.cla()
# patch.plot_nodes(mask=patch.nodes['rigid'],color='r',sizes=30)
# patch.plot_edges()
# plt.axis( (552066.9646997608, 552207.1805374735, 4124548.347825134, 4124660.504092434) )
# 
# ##

            

