from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse import linalg
import logging
log=logging.getLogger(__name__)

from ..grid import unstructured_grid
from .. import utils

class Diffuser(object):
    dt = 0.5
    alpha=0 # first order decay
    
    def __init__(self,grid,edge_depth=None,cell_depth=None):
        """
        edge_depth: [Nedges] array of strictly positive flux-face heights
        for edges.

        cell_depth: [Ncells] array of strictly positive cell thicknesses
        """
        self.dirichlet_bcs = []
        self.neumann_bcs = []

        self.forced_cells = set()
        self.grid = grid
        self.init_grid_geometry()

        if edge_depth is None:
            self.dzf = np.ones(self.grid.Nedges())
        else:
            assert np.all( edge_depth>0 )
            self.dzf=edge_depth

        if cell_depth is None:
            self.dzc = np.ones(self.grid.Ncells())
        else:
            assert np.all( cell_depth>0 )
            self.dzc = cell_depth

    class PointOutsideDomain(Exception):
        pass
    
    def set_dirichlet(self,value,cell=None,xy=None,on_duplicate='error'):
        if cell is None:
            cell = self.grid.point_to_cell(xy)
        if xy is None:
            xy = self.grid.cells_center()[cell]

        if cell is None:
            raise self.PointOutsideDomain("no cell found for dirichlet BC (cell=%s  xy=%s)"%(cell,xy))
        if cell in self.forced_cells:
            if on_duplicate=='error':
                raise Exception("Cell forced by multiple BCs")
            elif on_duplicate=='skip':
                return
            else:
                raise ValueError("on_duplicate: %s"%on_duplicate)

        self.forced_cells.add(cell)
        self.dirichlet_bcs.append([cell,value,xy])
    def set_flux(self,value,cell=None,xy=None,on_duplicate='add'):
        """ value: mass/time, ish?
        """
        if cell is None:
            cell = self.grid.point_to_cell(xy)
        if xy is None:
            xy = self.grid.cells_center()[cell]
            
        assert on_duplicate=='add'
        self.neumann_bcs.append([cell,value,xy])

    # janet grids may have cell centers coincident with edge -
    # this will clip those distances to avoid zero-div
    d_j_min=1
    def init_grid_geometry(self):
        """ pre-calculate geometric quantities related to the grid
        for use in the discretization
        """
        self.vc = self.grid.cells_center() # circumcenters
        self.ec = self.grid.edges_center()
        
        self.c1 = self.grid.edges['cells'][:,0]
        self.c2 = self.grid.edges['cells'][:,1]

        # distance between cell centers
        self.d_j = utils.mag( self.vc[self.c1] - self.vc[self.c2] )
        bdry=self.c2<0
        # grid has a few places where vc is coincident with outer boundary, thanks
        # to janet
        self.d_j[bdry] = 2*utils.mag( self.vc[self.c1[bdry]] - self.ec[bdry] ).clip(self.d_j_min,np.inf)
        self.l_j = self.grid.edges_length()

        self.normal_j = self.grid.edges_normals()
        self.area_c = self.grid.cells_area()

        self.K_j = 100*np.ones(self.grid.Nedges())

        j_valid=~self.grid.edges['deleted']

        print("Checking finite geometry")
        assert np.all( np.isfinite(self.d_j[j_valid]))
        assert np.all( np.isfinite(self.l_j[j_valid]))
        assert np.all( np.isfinite(self.area_c))
        assert np.all( np.isfinite(self.normal_j[j_valid]))
        assert np.all( self.d_j[j_valid] > 0 )
        assert np.all( self.l_j[j_valid] > 0 )
        assert np.all( self.area_c > 0 )


    def vector_edge_to_cell(self,F_edge_normal):
        """ given signed values on edges, relative to the natural normal
        for the edge, interpolate to two-vectors at cell centers
        """
        F_cell = np.zeros((self.grid.Ncells(),2),np.float64)
        for c in range(self.grid.Ncells()):
            js = self.grid.cell_to_edges(c)
            dist_edge_face = utils.mag( self.vc[c,None,:] - self.ec[js] )
            # missing edge_normals - what does that comment mean?
            # Actually I don't think this is the right interpolation - although
            # maybe the one from suntans doesn't apply since there are quads here.
            F_cell[c] = (self.normal_j[js]*F_edge_normal[js,None]*dist_edge_face[:,None]).sum(axis=0) / self.area_c[c]
        return F_cell

    def local_K_factor(self,xxyy,factor):
        dammed =  np.nonzero( (self.ec[:,0]>xxyy[0])&\
                              (self.ec[:,0]<xxyy[1])&\
                              (self.ec[:,1]>xxyy[2])&\
                              (self.ec[:,1]<xxyy[3])   )[0]
        self.K_j[dammed] *= factor

        
    def set_decay_rate(self,alpha):
        """ Set a first order decay coefficient
        """
        if not isinstance(alpha,np.ndarray):
            alpha = alpha*np.ones(self.grid.Ncells())
        self.alpha=alpha
                         
    def construct_linear_system(self):
        """
        construct the sparse matrix
        more algorithms are available if the matrix is symmetric, which
        requires evaluating the Dirichlet BCs rather than putting them in the
        matrix
        """
        N=self.grid.Ncells()
        Nbc = len(self.dirichlet_bcs)
        self.Ncalc=Ncalc = N - Nbc

        # map cells to forced values
        dirichlet = dict( [ (c,v) for c,v,xy in self.dirichlet_bcs])

        self.is_calc_c = is_calc_c = np.ones(N,np.bool_)
        for c,v,xy in self.dirichlet_bcs:
            is_calc_c[c] = False

        # is_calc_c[self.c_mask] = False

        # c_map is indexed by real cell indices, and returns the matrix index
        c_map = self.c_map = np.zeros(N,np.int32)
        self.c_map[is_calc_c] = np.arange(Ncalc)

        dzc=self.dzc
        dzf=self.dzf
        area_c=self.area_c

        meth='coo' # 'dok'
        if meth == 'dok':
            A=sparse.dok_matrix((Ncalc,Ncalc),np.float64)
        else:
            # construct the matrix from a sequence of indices and values
            ij=[]
            values=[] # successive value for the same i.j will be summed
        
        b = np.zeros(Ncalc,np.float64)
        flux_per_gradient_j = -self.K_j * self.l_j * dzf / self.d_j *  self.dt

        self.grid.edge_to_cells() # makes sure that edges['cells'] exists.
        
        for j in range(self.grid.Nedges()):
            e = self.grid.edges[j]
            ic1,ic2 = e['cells']
            
            if ic1<0 or ic2<0 or e['deleted']:
                continue # boundary edge, or deleted edge
                
            flux_per_gradient=flux_per_gradient_j[j]
            
            # this is the desired operation:
            #  Cdiff[ic1] -= flux_per_gradient / (An[ic1]*dzc) * (C[ic2] - C[ic1])
            #  Cdiff[ic2] += flux_per_gradient / (An[ic2]*dzc) * (C[ic2] - C[ic1])
            # Where Cdiff is row, C is col

            if is_calc_c[ic1] and is_calc_c[ic2]:
                mic2 = c_map[ic2]
                mic1 = c_map[ic1]
                v1=flux_per_gradient / (area_c[ic1]*dzc[ic1])
                v2=flux_per_gradient / (area_c[ic2]*dzc[ic2])
                
                if meth == 'dok':
                    A[mic1,mic2] -= v1
                    A[mic1,mic1] += v1
                    A[mic2,mic2] += v2
                    A[mic2,mic1] -= v2
                else:
                    ij.append( (mic1,mic2) ) ; values.append(-v1)
                    ij.append( (mic1,mic1) ) ; values.append(v1)
                    ij.append( (mic2,mic2) ) ; values.append(v1)
                    ij.append( (mic2,mic1) ) ; values.append(-v1)
                    
            elif not ( is_calc_c[ic1] or is_calc_c[ic2] ):
                # both are dirichlet, so nothing to do
                pass
            elif not is_calc_c[ic2]:
                mic1 = c_map[ic1]
                v=flux_per_gradient / (self.area_c[ic1]*dzc[ic1])
                if meth == 'dok':
                    A[mic1,mic1] += v
                else:
                    ij.append(  (mic1,mic1) )
                    values.append(v)

                # roughly
                # A[1,1]*x[1] + A[1,2]*x[2] + ... = b[1]
                # but we already know x[2],
                # A[1,1]*x[1] + ... = b[1] - A[1,2]*x[2]
                # so flip the sign, multiply by known dirichlet value, and
                # add to the RHS
                b[mic1] += flux_per_gradient / (area_c[ic1]*dzc[ic1]) * dirichlet[ic2]
            else: # not is_calc_c[c1]
                mic2 = c_map[ic2]
                # A[mic2,mic2] += flux_per_gradient / (area_c[ic2]*dzc[ic2])
                # A[mic2,mic1] -= flux_per_gradient / (area_c[ic2]*dzc[ic2])

                # A[mic2,mic2]*x[2] + A[mic2,mic1]*x[1] = b[2]
                # ...
                # A[mic2,mic2]*x[2] - flux_per_gradient / (area_c[ic2]*dzc[ic2])*x[1] = b[2]
                # ...
                # A[mic2,mic2]*x[2]  = b[2] + flux_per_gradient / (area_c[ic2]*dzc[ic2])*x[1]
                v=flux_per_gradient / (area_c[ic2]*dzc[ic2])
                if meth == 'dok':
                    A[mic2,mic2] += v
                else:
                    ij.append( (mic2,mic2) )
                    values.append(v)
                b[mic2] += flux_per_gradient / (area_c[ic2]*dzc[ic2]) * dirichlet[ic1]

        # Used to test 'is not 0:' but modern python complains
        if isinstance(self.alpha,np.ndarray): 
            for c in range(N):
                if self.is_calc_c[c]:
                    mic=self.c_map[c]
                    v=self.alpha[c]*self.dt
                    if meth == 'dok':
                        A[mic,mic] -= v
                    else:
                        ij.append( (mic,mic) )
                        values.append(-v)

        # Flux boundary conditions:
        for ic,value,xy in self.neumann_bcs:
            mic=c_map[ic]
            # make mass/time into concentration/step
            # arrived at minus sign by trial and error.
            # 2023-08-04: there was a bug here that used ic2 instead of ic.
            b[mic] -= value/(area_c[ic]*dzc[ic]) * self.dt

        if meth == 'dok':
            self.A = sparse.coo_matrix(A)
        else:
            ijs=np.array(ij,dtype=np.int32)
            data=np.array(values,dtype=np.float64)
            A=sparse.coo_matrix( (data, (ijs[:,0],ijs[:,1]) ), shape=(Ncalc,Ncalc) )
            self.A=A
            
        # report scale to get a sense of whether dt is too large
        Ascale = A.diagonal().min()
        log.debug("Ascale is %s"%Ascale)

        self.b = b

    def expand(self,v):
        vv = np.zeros(self.grid.Ncells(),np.float64)
        vv[self.is_calc_c] = v
        return vv

    def initial_guess(self):
        #return np.random.random(self.Ncalc)
        return (np.arange(self.Ncalc) % 10.0) / 10.0

    def compute(self):
        self.construct_linear_system()
        self.solve_linear_system()
        return self.C_solved
        
    solve_method='direct'
    solve_tol=1e-6
    def solve_linear_system(self,animate=False):
        x0=self.initial_guess()

        if animate:
            plt.figure(1)
            plt.clf()
            for c,v,xy in self.dirichlet_bcs:
                plt.annotate( str(v), xy )
            coll = self.grid.plot_cells(values=self.expand(x0))
            coll.set_lw(0)
            coll.set_clim([0,1])
            plt.axis('equal')
            plt.pause(0.01)

        ctr=itertools.count()
        def plot_progress(xk):
            count=next(ctr)
            log.debug("Count: %d"%count)
            if animate and count%1000==0:
                coll.set_array(self.expand(xk))
                plt.title(str(count))
                plt.pause(0.01)

        # I think that cgs means the matrix doesn't have to be
        # symmetric, which makes boundary conditions easier
        # with only showing progress every 100 steps,
        # this takes maybe a minute on a 28k cell grid.
        # But cgs seems to have more convergence problems with
        # pure diffusion.

        if animate:
            coll.set_clim([0,1])

        maxiter=int(1.5*self.grid.Ncells())
        code = -1
        if 1:
            C_solved=linalg.spsolve(self.A.tocsr(),self.b)
            code=0
        elif 1:
            C_solved,code = linalg.cgs(self.A,self.b,x0=x0,
                                       callback=plot_progress,
                                       tol=self.solve_tol,
                                       maxiter=maxiter)
        elif 0:
            C_solved,code = linalg.cg(self.A,self.b,x0=x0,
                                      callback=plot_progress,
                                      tol=self.tol,
                                      maxiter=maxiter)
        elif 1:
            C_solved,code = linalg.bicgstab(self.A,self.b,x0=x0,
                                            callback=plot_progress,
                                            tol=self.solve_tol,
                                            maxiter=maxiter)
        elif 1:
            log.debug("Time integration")
            x = x0
            for i in range(maxiter):
                x = A.dot(x)
                plot_progress(x)
        else:
            def print_progress(rk):
                count=next(ctr)
                if count%1000==0:
                    log.debug("count=%d rk=%s"%(count,rk))
            C_solved,code = linalg.gmres(self.A,self.b,x0=x0,tol=self.solve_tol,callback=print_progress)

        self.C_solved=self.expand(C_solved)
        for c,v,xy in self.dirichlet_bcs:
            self.C_solved[c]=v

        self.code = code

        if animate:
            evenly_spaced=np.zeros(len(C_solved))
            evenly_spaced[np.argsort(C_solved)] = np.arange(len(C_solved))
            coll.set_array(evenly_spaced)
            coll.set_clim([0,self.grid.Ncells()])
            plt.draw()

    def calc_fluxes(self):
        # vector calculation of all fluxes

        # positive towards c2:
        # mass per time, per area:
        # v04: include edge height here
        self.Fperp = -self.K_j*(self.C_solved[self.c2] - self.C_solved[self.c1])/self.d_j
        # simple no flux boundaries (discarded the repellent boundaries)
        self.Fperp[self.c2<0]=0
        
        # mass per time, per face
        # this has worked well with dzf**2, but would be nice if that wasn't necessary
        # self.flux_j = self.l_j*self.dzf**2 * self.Fperp # was fluxes
        self.flux_j = self.l_j*self.dzf*self.Fperp # was fluxes

    def calc_flux_vectors_and_grad(self):
        self.flux_vector_c = self.vector_edge_to_cell(self.flux_j)
        self.flux_mag_c = utils.mag(self.flux_vector_c)
        # rather than a zero-gradient at boundaries, take it as if a ghost cell
        # has zero flux magnitude.
        self.grad_flux_j = (self.flux_mag_c[self.c2]*(self.c2>=0) - self.flux_mag_c[self.c1])/self.d_j

    def flux_vectors(self):
        return self.flux_j[:,None]*self.normal_j
        
    def vector_edge_to_node(self,c,v_normal_j):
        """ for a specific cell c, with edge-normal vector values
        (with sign relative natural edge normal), return 2-vector
        values for the points ug.cells['nodes'][c]

        note that v_edge_normal is for *all* edges, the code
        here will pull out the relevant values
        """
        nodes=self.grid.cell_to_nodes(c)
        Nsides=len(nodes)
        modn=lambda ni: nodes[ni%Nsides]
        node_vectors=np.zeros((Nsides,2),np.float64)
        for ns in range(Nsides):
            j_prev = self.grid.nodes_to_edge(modn(ns-1),modn(ns))
            j_next = self.grid.nodes_to_edge(modn(ns),modn(ns+1))
            # Looking for the vector which satisfies
            # dot(vec,n_prev) = flux_prev
            # dot(vec,n_next) = flux_next
            mat = np.array([self.normal_j[j_prev],
                            self.normal_j[j_next]])
            projected = np.array([v_normal_j[j_prev],v_normal_j[j_next]])
            node_vectors[ns,:] = np.linalg.solve(mat,projected)
        return node_vectors

