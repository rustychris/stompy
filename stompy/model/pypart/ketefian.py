import numpy as np

import scipy.integrate

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
    
    def load_grid(self,grid=None):
        super(ParticlesKGS,self).load_grid(grid=grid)

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
        for c in self.g.valid_cell_iter():
            for ji,j in enumerate(self.g.cell_to_edges(c)):
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
        self.mAs = [None]*self.g.Ncells()

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

            new_dz_edge=(edge_eta+self.edepths).clip(0,np.inf)
            # enforce zero flux face area for boundary edges
            # note that this is incorrect for flow boundaries.
            new_dz_edge[self.bndry]=0.0
            self.set_dz_edge(new_dz_edge)

            dz_cell=eta+self.cdepths
            self.dz_cell=dz_cell

            # SLOW :-(
            self.update_fluxes()
            self.set_coeffs()
    
    def set_dz_edge(self,dz_edge):
        self.dz_edge=dz_edge

        # mA cannot be set statically, so here we set it with updated 
        # values of dz_edge..
        def alpha_beta(c,j): # coefficient on u[j] with respect to cell center velocity of c
            return ( (1./self.Ac[c]) * self.edge_norm[j] 
                     * utils.dist( self.edge_center[j] - self.cell_center[c])
                     * self.edge_len[j] )
        def gamma(c,ji,j): # coefficient on u[j] with respect to d_eta / d_t
            return (1./self.Ac[c]) * dz_edge[j] * self.edge_len[j] * self.signs[c,ji]

        for c in self.g.valid_cell_iter():
            if c%5000==0:
                print "%d / %d"%( c, self.g.Ncells() )

            c_edges=self.g.cell_to_edges(c) 

            alpha_betas = [alpha_beta(c,j) for j in c_edges]

            rows=[ [ alpha_betas[0][0], alpha_betas[1][0], alpha_betas[2][0]],
                   [ alpha_betas[0][1], alpha_betas[1][1], alpha_betas[2][1]],
                   [ gamma(c,ji,j) for ji,j in enumerate(c_edges)] ]

            for ji,j in enumerate(c_edges):
                if self.bndry[j]:
                    row=[0,0,0]
                    row[ji]=1
                    rows.append(row)

            mA=np.array(rows)

            self.mAs[c]=mA

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
        u_norms=np.nan*np.zeros( (self.g.Nedges(),2), 'f8')
        u_norm_count=np.zeros(self.g.Nedges(),'i4')

        for c in self.g.valid_cell_iter():
            if c%10000==0:
                print "%d / %d"%( c, self.g.Ncells() )

            B=[Uc[c,0],
               Uc[c,1],
               self.d_eta_dt[c] ] 

            c_edges=self.g.cell_to_edges(c) 
            
            # assumes that mAs has already been set
            mA=self.mAs[c]

            if mA.shape[0]==3:
                solve=np.linalg.solve
            else:
                solve=lambda A,B: np.linalg.lstsq(A,B)[0]
                while len(B)<mA.shape[0]:
                    B.append(0)

            B=np.array(B)

            u_j = solve(mA,B)

            for j,u in zip(c_edges,u_j):
                u_norms[j,u_norm_count[j]]=u
            u_norm_count[c_edges]+=1

        # possible that adjacent cells have slightly different opinions
        # on the edge velocity
        u_norm=np.nanmean(u_norms,axis=1)
        return u_norm

    def set_invM14(self):
        """ set up the time invariant part of the interpolation
        """
        self.invM14=np.zeros( [self.g.Ncells(),6,6],'f8' )
        self.M14=np.zeros( [self.g.Ncells(),6,6],'f8' ) # just for debugging

        M14=np.zeros( (6,6), 'f8')

        cc=self.g.cells_center()

        for c in range(self.g.Ncells()):
            c_edge=self.g.cell_to_edges(c) 

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
        B14=np.zeros(6,'f8')

        for c in range(self.g.Ncells()):
            c_edge=self.g.cell_to_edges(c) 
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

    def vel_interp(self,c,x):
        cc=self.g.cells_center()
        A=self.coeffs[c,:2,:]
        bxy=self.coeffs[c,2,:]
        return np.dot(A,x-cc[c]) + bxy

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
            part_t=self.t_unix

            # start with a slow implementation
            def vel(t,x):
                # assumes that the velocity field doesn't
                # change during the integration time step, so
                # ignore t.
                c=self.g.select_cells_nearest(x,inside=True)
                return self.vel_interp(c,x)

            # omit dense output, and reverse tracking
            r = scipy.integrate.ode(vel)

            # This is the magic
            r.set_integrator('vode', method='bdf', order=3, nsteps=3000)
            r.set_initial_value(self.P['x'][i],part_t)

            r.integrate(stop_t)
            if not r.successful():
                print "!"
            self.P['x'][i] = r.y
            self.P['c'][i] = self.g.select_cells_nearest(r.y,inside=True)
