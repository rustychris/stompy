from __future__ import print_function

import numpy as np

from shapely import geometry

import scipy.integrate
import pdb

from . import basic
from stompy import utils

class ParticlesKGS(basic.UgridParticles):
    """
    Extension of basic particle tracking to (1) use the method
    of Ketefian, Gross and Stelling for interpolation within cells, 
    and (2) apply some awkward corrections and back-calculations on
    suntans cell-centered outputs to get a good approximation of edge-centered
    fluxes.
    """
    dt_s=None
    dz_edge_eps=0.001 # kludge to avoid singular matrix
    dz_cell_eps=0.001 # and division by zero

    def load_grid(self,grid=None):
        super(ParticlesKGS,self).load_grid(grid=grid)

        self.log.info('Top of ParticlesKGS:load_grid')
        # The method and this code can only deal with triangular grids
        # for now.
        assert self.g.max_sides==3

        # additional static geometry, beyond self.edge_norm
        # static geometry:
        self.Ac=self.g.cells_area()
        self.cell_center=self.g.cells_center()
        self.edge_len =self.g.edges_length()
        self.edge_center=self.g.edges_center()

        self.edepths=self.ncs[0].edgedepth.values
        self.cdepths=self.ncs[0].depth.values

        self.bndry=self.g.edges['cells'][:,1]<0

        self.signs=np.zeros( (self.g.Ncells(),self.g.max_sides), 'i4') 
        self.cell_edges=-1 + np.zeros( (self.g.Ncells(),self.g.max_sides), 'i4')

        for c in self.g.valid_cell_iter():
            c2e=self.g.cell_to_edges(c)
            self.cell_edges[c,:] = c2e
            for ji,j in enumerate(self.cell_edges[c,:]):
                # not entirely sure here, but I got smaller residuals
                # for a boundary edge this way, and reasonable
                # comparison at the end to d_eta_dt.
                # the sign convention seems to be the sign*u_edge gives
                # flow INTO a cell when positive.
                if self.g.edges['cells'][j,0] == c:
                    self.signs[c,ji] = -1 
                else:
                    self.signs[c,ji] = 1

        # time invariant part of the KSG interpolation
        self.set_invM14()

        # factors for the forward interpolation - used during
        # the translation from suntans cell-centered to edge-centered
        # must be updated by calls to set_dz_edge - time variant!

        #self.mAs = [None]*self.g.Ncells()
        self.mAs=np.zeros( (self.g.Ncells(),3,3), 'f8' )

        # setup the constant parts of mA
        def alpha_beta(c,j): # coefficient on u[j] with respect to cell center velocity of c
            return ( (1./self.Ac[c]) * self.edge_norm[j] 
                     * utils.dist( self.edge_center[j] - self.cell_center[c])
                     * self.edge_len[j] )
        for c in self.g.valid_cell_iter():
            if c%5000==0:
                print("%d / %d"%( c, self.g.Ncells() ))

            c_edges=self.cell_edges[c,:] 

            alpha_betas = [alpha_beta(c,j) for j in c_edges]

            self.mAs[c,0,:]= [ alpha_betas[0][0], alpha_betas[1][0], alpha_betas[2][0]]
            self.mAs[c,1,:]= [ alpha_betas[0][1], alpha_betas[1][1], alpha_betas[2][1]]
            self.mAs[c,2,:]= np.nan # Time varying

        self.log.info('Finish ParticlesKGS:load_grid')

    def update_particle_velocity_for_new_step(self):
        """ this gets called within update_velocity, when 
        the correct netcdf and time step within that netcdf has been
        selected.
        """
        if 1: # code from Basic, mostly
            # face, layer, time.
            # assumes 2D here.
            u=self.current_nc.cell_east_velocity.values[:,0,self.nc_time_i]
            v=self.current_nc.cell_north_velocity.values[:,0,self.nc_time_i]

            self.U=np.array( [u,v] ).T # again assume 2D
            # but skip the part where it updates the per-particle velocity

        if 1: # And now we do some extra work
            # basically same as update_time(self,ti)
            ti=self.nc_time_i
            if self.dt_s is None:
                # disgusting!
                self.dt_s=np.diff(self.current_nc.time[:2])[0].astype('timedelta64[s]').item().seconds 

            eta=self.current_nc.surface.isel(time=ti).values

            # finite differences at the end of the time series
            if ti+1<len(self.current_nc.time):
                eta_p=self.current_nc.surface.isel(time=ti+1).values
                d_eta_dt = (eta_p-eta)/self.dt_s
            else:
                eta_m=self.nc.surface.isel(time=ti-1).values
                d_eta_dt = (eta-eta_m)/self.dt_s

            self.d_eta_dt=d_eta_dt

            edge_etas=eta[ self.g.edges['cells'] ]
            edge_etas[ self.bndry, 1 ] = edge_etas[ self.bndry, 0] 
            # not sure if the code takes max eta or upwind eta.
            edge_eta=edge_etas.max(axis=1)

            # only works for 2-D simulations!
            # plus this check makes an assumption of the variable naming
            # which is true probably only for boffinator inputs.
            assert len(self.current_nc.nMeshGlobal_layers)==1

            new_dz_edge=(edge_eta+self.edepths).clip(self.dz_edge_eps,np.inf)
            # enforce zero flux face area for boundary edges
            # note that this is incorrect for flow boundaries.
            new_dz_edge[self.bndry]=0.0
            self.set_dz_edge(new_dz_edge)

            dz_cell=(eta+self.cdepths).clip(self.dz_cell_eps,np.inf)
            self.dz_cell=dz_cell

            # SLOW :-(
            self.update_fluxes()
            self.set_coeffs()

    def set_dz_edge(self,dz_edge):
        self.dz_edge=dz_edge

        if 0: # old, non-vector approach
            for c in self.g.valid_cell_iter():
                if c%5000==0:
                    print("%d / %d"%( c, self.g.Ncells() ))

                c_edges=self.cell_edges[c,:] 

                for ji,j in enumerate(c_edges):
                    # ripe for one more vectorization
                    self.mAs[c,2,ji] = ( (1e5*self.bndry[j]) 
                                         + (1./self.Ac[c]) * dz_edge[j] * self.edge_len[j] * self.signs[c,ji] )
        else:
            # even most of this could be kept in a constant, and we just have to multiply by dz_edge
            # and add the bndry part.
            self.mAs[:,2,: ] = ( (1e5*self.bndry[self.cell_edges]) 
                                 + (1./self.Ac[:,None]) 
                                 * dz_edge[self.cell_edges] * self.edge_len[self.cell_edges] * self.signs )

    def u_cell_to_u_edge(self,Uc):
        """
        Given cell-centered velocities, as computed by suntans *without*
        edge-depth consideration, invert to get edge-centered velocities.

        Uc: [Ncells,2] cell centered velocities

        dz_edge only works for the velocity field computed without edge-depth
        considerations, and the output of this method only makes sense in the
        same way.  When edge-depths are accounted for, the flux at an edge is
        well-defined, but the height and the velocity are discontinuous.
        """
        u_norms=np.zeros( self.g.Nedges(), 'f8')
        u_norm_count=np.zeros(self.g.Nedges(),'i4')

        if 0: # old, nonvectorized way
            B=np.zeros(3,'f8')

            #for c in self.g.valid_cell_iter():
            for c in range(self.g.Ncells()):
                if c%10000==0:
                    print("%d / %d"%( c, self.g.Ncells() ))

                B[:2]=Uc[c]
                B[2]=self.d_eta_dt[c]

                c_edges=self.cell_edges[c,:]

                # if mA.shape[0]==3:
                #     solve=np.linalg.solve
                # else:
                #     solve=lambda A,B: np.linalg.lstsq(A,B)[0]
                #     while len(B)<mA.shape[0]:
                #         B.append(0)
                #
                # B=np.array(B)

                u_j = np.linalg.solve(self.mAs[c],B)

                # old way:
                # for j,u in zip(c_edges,u_j):
                #     # additional control on boundary flows:
                #     u_norms[j,u_norm_count[j]]= (~self.bndry[j])*u
                # new way - slightly better?
                u_norms[c_edges] += (~self.bndry[c_edges])*u_j
                u_norm_count[c_edges]+=1
        else: # new vectorized way...
            Bs=np.zeros((self.g.Ncells(),3),'f8')
            Bs[:,:2]=Uc
            Bs[:,2]=self.d_eta_dt
            # mAs: (251760, 3, 3)
            # Bs: (251760, 3) That's failing at 754, where the
            # 3rd row and entry of mA/B are zero.
            # this is maybe solved by clipping dz_edge
            u_js=np.linalg.solve(self.mAs,Bs)

            js=self.cell_edges.ravel()
            u_norms=np.bincount(js,weights=u_js.ravel())
            u_norm_count=np.bincount(js)

        # possible that adjacent cells have slightly different opinions
        # on the edge velocity
        # np.nanmean(u_norms,axis=1) # old way
        u_norms /= u_norm_count 
        return u_norms

    def set_invM14(self):
        """ set up the time invariant part of the interpolation
        """
        self.invM14=np.zeros( [self.g.Ncells(),6,6],'f8' )
        self.M14=np.zeros( [self.g.Ncells(),6,6],'f8' ) # just for debugging

        M14=np.zeros( (6,6), 'f8')

        #cc=self.g.cells_center()
        cc=self.cell_center

        for c in range(self.g.Ncells()):
            #c_edge=self.g.cell_to_edges(c) 
            c_edge=self.cell_edges[c,:]

            M14[:,:]=np.nan # DBG.  unnecessary

            # assemble 6x6 matrix.  from equation 14 in Ketefian et al

            # n_ij is the outward normal
            # xctr_ij is the midpoint of the edge
            # L_ij is the length of the edge

            nxi123= self.edge_norm[c_edge,0]*(-1)*self.signs[c]
            nyi123= self.edge_norm[c_edge,1]*(-1)*self.signs[c]

            ec=self.edge_center[c_edge]
            xctri123=ec[:,0] - cc[c,0]
            yctri123=ec[:,1] - cc[c,1]

            M14[:3,0] = nxi123 * xctri123
            M14[:3,1] = nxi123 * yctri123
            M14[:3,2] = nyi123 * xctri123
            M14[:3,3] = nyi123 * yctri123
            M14[:3,4] = nxi123 
            M14[:3,5] = nyi123

            M14[3:6,0] = -nxi123 * nyi123 # had been missing y here
            M14[3:6,1] = nxi123**2
            M14[3:6,2] = -nyi123**2
            M14[3:6,3] = nxi123*nyi123
            M14[3:6,4:6] = 0.0 

            self.M14[c,:,:] = M14
            self.invM14[c,:,:] =np.linalg.inv(M14)

        self.coeffs=np.zeros( [self.g.Ncells(),3,2], 'f8')

    def update_fluxes(self):
        """ Return m3/s flow rates per edge at currently selected nc and time
        """
        # halfway thinking about the waq_scenario.Hydro interface, but
        # not really trying
        # self.U got set way back in update_particle_velocity_for_new_step()
        self.u_edge=self.u_cell_to_u_edge(self.U)
        self.edge_flux=self.u_edge * self.dz_edge * self.edge_len
        return self.edge_flux

    def set_coeffs(self):
        """
        Use invM14, along with edge normal velocity, to get the per-cell
        velocity interpolation coefficients.
        This is the time varying part of the interpolation
        """
        if 0: # old iterative way:
            B14=np.zeros(6,'f8')

            for c in range(self.g.Ncells()):
                c_edge=self.cell_edges[c,:]
                u_norm=self.edge_flux[c_edge] / (self.dz_cell[c] * self.edge_len[c_edge])
                u_edge_into_cell = self.signs[c,:] * u_norm 

                # the first 3 entries are Qij/ (Lij * hi)
                # that's just the in-cell outward normal velocity
                B14[:3] = -u_edge_into_cell
                # last 3 are gradients along the edge, which we can assume are 0
                B14[3:] = 0.0

                # coeffs=np.linalg.solve(M14,B14)
                coeffs=np.dot(self.invM14[c,:,:],B14)
                # coeffs here is 
                # [ a b ]
                # [ c d ]
                # [bxy1 bxy2]
                self.coeffs[c,:,:] = coeffs.reshape([3,2] )
        if 1: # vectorized
            B14s=np.zeros( (self.g.Ncells(),6), 'f8')
            u_norms=self.edge_flux[self.cell_edges] / (self.dz_cell[:,None] * self.edge_len[self.cell_edges])
            B14s[:,:3]=-self.signs[:,:] * u_norms 
            B14s[:,3:]=0.0 # no gradients along edges, please
            # soln=np.dot(self.invM14[:,:,:],B14s)
            # Dicey getting the transposts just right, but this checks
            # out as identical to the above iterative calc
            soln=np.matmul(self.invM14,B14s[:,:,None])[:,:,0]
            soln=soln.reshape( [self.g.Ncells(),3,2] )
            self.coeffs[:,:,:]=soln

    def vel_interp(self,c,x):
        #cc=self.g.cells_center()
        cc=self.cell_center
        A=self.coeffs[c,:2,:]
        bxy=self.coeffs[c,2,:]
        return np.dot(A,x-cc[c]) + bxy

    def move_particles_ode(self,stop_t):
        """
        Don't use this one!
        Advance each particle to the correct state at stop_t.
        Assumes that no input (updating velocities) or output
        is needed between self.t_unix and stop_t.

        Caller is responsible for updating self.t_unix
        """
        g=self.g

        last_c=[None,None]

        for i,p in enumerate(self.P):
            # advance each particle to the correct state at stop_t
            part_t=self.t_unix

            # start with a slow implementation
            stuck=np.zeros(2)
            def vel(t,x):
                # assumes that the velocity field doesn't
                # change during the integration time step, so
                # ignore t.
                c,poly=last_c

                if poly is not None and poly.contains(geometry.Point(x[0],x[1])):
                    pass # great!
                else:
                    c=self.g.select_cells_nearest(x,inside=True)
                    if c is not None:
                        poly=self.g.cell_polygon(c)
                        last_c[:]=[c,poly]

                if c is None:
                    # annoying, but ode doesn't know where the boundaries
                    # are...
                    print("!")
                    return stuck
                else:
                    return self.vel_interp(c,x)

            # the first method is much faster, but smears out the steps
            # between cells.  Second method is significantly slower and safer.
            new_c=None
            for meth in [dict(name='vode',method='bdf', order=3, nsteps=3000),
                         dict(name='dopri5',nsteps=3000)]:
                # omit dense output, and reverse tracking
                r = scipy.integrate.ode(vel)

                # This is the magic
                # r.set_integrator('vode', method='bdf', order=3, nsteps=3000)
                r.set_integrator(**meth)
                r.set_initial_value(self.P['x'][i],part_t)

                r.integrate(stop_t)
                if not r.successful():
                    print("X1")
                    continue # try with slow method
                new_c=self.g.select_cells_nearest(r.y,inside=True)                
                if new_c is None:
                    print("X2")
                    continue
                break
            else:
                # Bad news - neither was great.
                print("Couldn't salvage step.")
                pdb.set_trace()

            self.P['x'][i] = r.y
            if new_c is not None:
                self.P['c'][i] = new_c
            # assert self.g.select_cells_nearest(r.y,inside=True) is not None

    def move_particles(self,stop_t):
        """
        Advance each particle to the correct state at stop_t.
        Assumes that no input (updating velocities) or output
        is needed between self.t_unix and stop_t.

        Caller is responsible for updating self.t_unix
        """
        g=self.g

        for i,p in enumerate(self.P):
            # advance each particle to the correct state at stop_t
            start_t=self.t_unix
            part_xy=self.P['x'][i]
            # cell=g.select_cells_nearest( part_xy, inside=True ) 
            cell=self.P['c'][i] # is somebody else taking care of initing this?
            last_edge=None # *could* be saved, but maybe not necessary?

            for it in range(2000):
                if start_t>=stop_t:
                    break
                (part_xy,start_t,cell,last_edge)=self.move_particle_in_cell(part_xy,start_t,stop_t,cell,last_edge)
                # traj.append(part_xy)
            else:
                # this might be okay, though.
                raise Exception("Too many iterations?")

            self.P['x'][i] = part_xy
            self.P['c'][i] = cell
            # assert self.g.select_cells_nearest(r.y,inside=True) is not None


    def plot_cell_velocity(self,c,ax):
        """ partial implementation to show interpolated velocity fields
        """
        cpoly=self.g.cell_polygon(c)
        xyxy=cpoly.bounds

        xs=np.linspace(xyxy[0],xyxy[2],30)
        ys=np.linspace(xyxy[1],xyxy[3],30)
        X,Y=np.meshgrid(xs,ys)
        XY=np.array( [X.ravel(),Y.ravel()] ).T
        UV=np.array( [ self.vel_interp(c,xy)
                       for xy in XY ] )

        ax.quiver(XY[:,0],XY[:,1],
                  UV[:,0],UV[:,1])


    def move_particle_in_cell(self,part_xy,t_start,t_stop,cell,last_edge):
        poly=self.g.cell_polygon(cell)

        # Lots of edge-cases, so no point in trying to test this
        # assert( poly.contains( geometry.Point(part_xy) ) )

        xy=self.exact_integrator(cell)

        xy_max=xy(part_xy,t_stop - t_start)

        if poly.contains( geometry.Point(xy_max) ):
            return xy_max,t_stop, cell,last_edge

        # subdivide time intervals until we find the moment
        # the trajectory leaves this cell.
        x_delta=1e-5 * np.sqrt(self.Ac[cell])

        low=(t_start,part_xy)
        high=(t_stop,xy_max)

        # something newton-like would be much faster.
        # 15 iterations..
        for it in range(500):
            if utils.dist( low[1] - high[1] ) < x_delta:
                break
            t_mid = (low[0] + high[0])/2.0
            xy_mid = xy(part_xy,t_mid-t_start)
            if poly.contains( geometry.Point(xy_mid) ):
                low=(t_mid,xy_mid)
            else:
                high=(t_mid,xy_mid)
        else:
            print("Exhausted max iterations trying to find moment of particle exit")

        # which one of this cell's edges is crossed first within this period?
        trav_xy = high[1] - low[1] # where is the trajectory going

        # a bit of slop here.  any valid crossing should have remaining
        # of <= 1.0
        best_remaining=1.001

        for ji,j in enumerate(self.cell_edges[cell,:]):
            if j==last_edge:
                # no back sliding
                continue 
            n1=self.g.edges['nodes'][j,0]
            p1=self.g.nodes['x'][n1]
            # I want this to be the outward normal, so according
            # to ketefian.py comments, need to negate
            outward=-self.signs[cell,ji] * self.edge_norm[j]
            dist_normal = np.dot( p1 - low[1], outward )
            closing=np.dot(trav_xy,outward)
            if closing>0:
                remaining=dist_normal / closing
                if remaining < best_remaining:
                    best_remaining=remaining
                    best_j=j
        assert best_remaining <= 1.0 # will probably have to relax a little            
        next_j=best_j
        next_xy=low[1] + trav_xy * best_remaining
        next_t=low[0] + (high[0] - low[0])*best_remaining

        cells=self.g.edges['cells'][next_j]
        if cells[0]==cell:
            next_cell=cells[1]
        elif cells[1]==cell:
            next_cell=cells[0]
        else:
            raise Exception("Lost track of cells!")

        return next_xy,next_t,next_cell,next_j

    def exact_integrator(self,cell):
        """ Returns a function which takes a starting position and time elapsed since
        then, returning the integrated position.
        """
        cc=self.cell_center[cell]

        A=self.coeffs[cell,:2,:]
        bx,by=self.coeffs[cell,2,:]

        # Following appendix A:
        a=A[0,0] ; b=A[0,1] ; c=A[1,0] ; d=A[1,1]

        lamb_avg=0.5*(a+d)
        eps=0.5*np.sqrt( (a-d)**2 + 4*b*c + 0j)
        det=(lamb_avg+eps)*(lamb_avg-eps)

        small=1e-14

        if det != 0: # and np.isreal(eps): 
            # First case - A is invertible, with real and distinct eigenvalues
            # this lumps in the second case with repeated eigenvalues.
            # and in fact, since we're working off complex values anyway,
            # the same code works for complex eigenvalues.
            # eq A11:

            def xy(part_xy,t,
                   eps=eps,lamb_avg=lamb_avg,a=a,b=b,c=c,d=d,bx=bx,by=by):
                x0,y0=part_xy - cc

                epst=eps*t
                sexp=(np.exp(epst)+np.exp(-epst))

                # handle some of the limits manually
                # this handles both t->0 and eps->0.
                # not really sure what the proper value of small is.
                if np.abs(epst)<small:
                    expexp=1
                else:
                    expexp=(np.exp(epst)-np.exp(-epst))/(2*epst)

                lamb_t=lamb_avg*t

                #denom=lamb_t**2 + epst**2
                #if t<small:
                # isn't always just this - why have the extraneous t**2 ?
                t2_frac=1./(lamb_avg**2 - eps**2)
                #else:
                #    t2_frac=t**2/denom

                xterm1 = (a*x0 + b*y0 + bx)*t*expexp * np.exp(lamb_t)
                xterm2 = x0*(0.5*sexp-lamb_t*expexp)*np.exp(lamb_t)
                xterm3 = -( (d*bx-b*by)* t2_frac * 
                           ( 1-(0.5*sexp-lamb_t*expexp)*np.exp(lamb_t)) )
                x=np.real(xterm1+xterm2+xterm3)

                yterm1=(c*x0+d*y0+by)*t*expexp*np.exp(lamb_t)
                yterm2=y0*(0.5*sexp-lamb_t*expexp)*np.exp(lamb_t)
                yterm3= -( (-c*bx+a*by)* t2_frac *
                          ( 1-(0.5*sexp -lamb_t*expexp)*np.exp(lamb_t) ) )
                y=np.real(yterm1+yterm2+yterm3)
                return cc+np.array([x,y])
        else:
            raise Exception("Not ready for other matrix types")
        return xy



# For complex eigenvalues, 
# very similar, but expexp becomes sin(eps_tilde t)/(eps_tilde t)
#  sexp becomes cos(eps_tilde t)
#  and t2_frac gets a - in front of eps_tilde.
# seems like if the values are already complex, then the expressions don't
# have to change at all.
