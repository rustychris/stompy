"""
Water column 'model'
"""

from scipy import sparse, linalg
import numpy as np
from .. import utils
import matplotlib.pyplot as plt

class WaterColumn:
    """
    Simple water column model.
    constant depth
    spatially variable verticle velocity, but assumed w >= 0 (upward)
    no flux BCs at bed and surface.
    parabolic eddy viscosity.
    Advection is explicit with TVD.
    Diffusion is implicit.
    Constant time step.
    """
    H=8 # depth of water column.
    # t_spin=np.timedelta64(10,"D") # first 10 days will be repeated

    Cd=0.001
    ws=10.0 # swimming, m/d

    tvd='vanLeer'

    diffusion="implicit"

    t0=np.datetime64("2000-01-01 00:00")
    
    dz=0.2
    # For explicit stuck around 1s.
    # For implicit can go up to 1500s or so
    dt=np.timedelta64(900,'s')
    dt_spinup=np.timedelta64(2,"D") # call .spin() to get this spinup
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.init_state()

    def init_state(self):
        self.z_int=np.linspace(-self.H, 0.0, 1+int(self.H/self.dz))
        self.z_ctr=0.5*(self.z_int[:-1] + self.z_int[1:])
        self.C = np.ones_like(self.z_ctr)
        self.nu_t = 1e-4 * np.ones_like(self.z_int)
        z_bed=-self.H
        z_surf=0.0
        self.turb_L = (self.z_int - z_bed) * (z_surf - self.z_int)/self.H
        self.t = self.t0

        # times in decimal seconds
        self.t_s = 0.0
        self.dt_s = self.dt / np.timedelta64(1,'s')

        # Initialize constant swimming
        self.w = np.full(self.z_int.shape, self.ws/86400.0)
        self.w[0]=0.0 # no flux at bed
        self.w[-1]=0.0 # no flux at surface.
        
    def plot(self,ax=None,**kw):
        ax=ax or plt.gca()
        ax.plot(self.C,self.z_ctr,**kw)
        #ax.plot(self.nu_t,self.z_int)

    def step_for(self,interval):
        """ interval: timedelta64 """
        self.step_until(self.t + interval)
        
    def step_until(self,t_stop):
        """ Step until a given time
        t_stop: datetime64
        Stops on or immediately after t_stop
        """
        while self.t < t_stop:
            self.step()

    def spin(self):
        """ Repeat steps at the current time """
        t=self.t
        spin_steps = int(round(self.dt_spinup / self.dt ))
        
        for spin_step in range(spin_steps):
            self.step()
            self.t = t
            
    def step(self):
        """ take a single step """
        self.t_s = (self.t - self.t0)/np.timedelta64(1,'s')
        self.step_viscosity()
        self.step_scalar()
        self.t += self.dt

    def step_viscosity(self):
        self.nu_t = 0.4 * self.u_star() * self.turb_L

    def u_star(self):
        return self.u_mag() * np.sqrt(self.Cd)

    def u_mag(self):
        return 1.0 # placeholder. Override!

    def step_scalar(self):
        F=np.zeros( self.C.shape[0]-1, np.float64)

        self.add_advective_flux(F)
        if self.diffusion=="explicit":
            self.add_diffusive_flux(F)

        J = F * self.dt_s / self.dz
        self.C[1:] += J
        self.C[:-1] -= J

        if self.diffusion=="implicit":
            self.step_implicit_diffusion()
        
    def add_diffusive_flux(self,F):
        Cp=self.C[1:]
        Cm=self.C[:-1]
        # Are we exceeding a diffusive time step limit?
        dt_max = self.dz**2 / self.nu_t[1:-1].max().clip(1e-12) 
        if dt_max<self.dt_s:
            raise Exception(f"Timestep {self.dt_s} is longer than diffusive limit {dt_max}")
        
        F[:] += -self.nu_t[1:-1] * (Cp - Cm) / self.dz

    def add_advective_flux(self,F):
        # Advection:
        w_int=self.w[1:-1] # only for interfaces.
        # flux at interior interfaces. Upwind, assume we know sign of w_s.
        #CFL = w_int*self.dt_s / self.dz
        #assert np.all(CFL>=0.0)
        #assert np.all(CFL<=1.0)

        F[:] += w_int*self.C[:-1] # 1st order upwind           

        if self.tvd: # Use TVD flux where possible.
            # the subset of cells with a complete stencil. This assumes
            # all velocities are positive.
            # Note that F[0] is really F[0.5], the flux between C[0] and C[1]
            # ignore the first interface, since we can't do TVD there.
            i=np.arange(self.C.shape[0])[1:-1]
            i05 = i # just to make the code clearer. maybe?
            
            # This is just a shorthand way to write upwind.
            # Already have this in F, so skip.
            # Fi05_uw=0.5*w_int*(C[i] + C[i+1]) - np.abs(w_int)*0.5*(Cp-C)

            # The flux limiter part.
            # The first interface, i.e. the first entry in F, there is not enough
            # "room" for TVD. This section operates on the remaining interfaces.
            
            # I=i-sgn(w_int) , which for us is i-1. The "1" suffix denotes values that
            # are offset from F by 1 index.
            I=i-1
            #   r=(C[I+1] - C[I])/(C[i+1] - C[i])
            #   r=(C[i] - C[i-1])/(C[i+1] - C[i])
            # "upwind" gradient / local gradient 
            num=(self.C[I+1] - self.C[I]) 
            den=(self.C[i+1] - self.C[i])
            r=np.ones(len(i))
            valid=np.abs(den)>1e-10
            r[valid] = num[valid] / den[valid]
            if self.tvd=='minmod':
                phi = r.clip(0,1) # minmod limiter
            elif self.tvd=='vanLeer':
                phi = (r+np.abs(r))/(1+np.abs(r))

            F[i05] += phi*(1-w_int[i05] * self.dt_s / self.dz)*w_int[i05]*0.5*(self.C[i+1]-self.C[i])

    def step_implicit_diffusion(self):
        Nx=len(self.C)

        # Each row is then 
        #  C_i - dt/dz**2 * nu_t[i-1/2] * (C_i - C_{i-1}) + dt/dz**2 * nu_t[i+1/2] * (C_{i+1}-C_i)
        #d_sub   = np.zeros(Nx-1,np.float64)
        #d_super = np.zeros(Nx-1,np.float64)
        J = -self.dt_s / self.dz**2 * self.nu_t[1:-1] # defined just on interior interfaces.

        if 0:
            # more verbose, and use sparse matrix solver.
            # low side diffusive flux
            d_main = np.ones(Nx,np.float64)
            d_main[1:] -= J 
            d_sub = J
            # high side diffusive flux
            d_main[:-1] -= J
            d_super = J
            self.M = sparse.diags([d_sub,d_main,d_super],[-1,0,1],(Nx,Nx),format='csc')
            result = sparse.linalg.spsolve(self.M,self.C)
        else: 
            banded=np.zeros([2,Nx],np.float64)
            banded[0,1:] = J
            banded[1,:] = 1
            banded[1,1:] -= J
            banded[1,:-1] -= J
            result = linalg.solveh_banded(banded,self.C, lower=False, overwrite_ab=True, overwrite_b=True)
            

        self.C[:] = result
        #print(result)


class RouseColumn(WaterColumn):
    # Simplified WaterColumn to confirm convergence on Rouse profile
    def u_star(self):
        return 0.01
    
    def plot_rouse(self,ax):
        b=self.ws/86400. / (0.4*self.u_star())
        z=-self.z_ctr
        Crouse = (z/(self.H-z))**(-b)
        Crouse = Crouse * self.C.sum() / Crouse.sum()
        ax.plot(Crouse,self.z_ctr,label="Rouse")


if __name__=='__main__':
    # May not work. Just stuffed some dev/test stuff in here.
    wcs=[
        #WaterColumn(df=df,tvd=None),
        #WaterColumn(df=df,tvd='minmod'),
        #WaterColumn(df=df,tvd='vanLeer'),
        RouseColumn(df=df,tvd=None,diffusion="implicit"),
        #RouseColumn(df=df,tvd='vanLeer',diffusion="implicit"),
        ]

    import time
    t_start = time.time()
    for _ in range(500):
        for wc in wcs:
            wc.step()
    t_elapse = time.time() - t_start

    print(f"{t_elapse:.02f}s for {wc.t_s / t_elapse:.1f}x realtime")

    fig,ax=plt.subplots()
    for wc in wcs:
        wc.plot(ax=ax)
        print(wc.C.sum())

    wc.plot_rouse(ax=ax)

    ax.lines[1].set_ls('--')
    ax.legend()
