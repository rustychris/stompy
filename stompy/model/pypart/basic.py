"""
Encapsulating some of the 2D particle tracking from
currents-figures-netcdf-animation-v03.ipynb
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import matplotlib.dates as mdates
from shapely import geometry

# from stompy.model.suntans import sunreader

import xarray as xr
from stompy import utils
from stompy.plot import (plot_utils, cmap, plot_wkb)
from stompy.grid import unstructured_grid
from stompy.spatial import (wkb2shp,field)

## 


# ---
fac=86400 # adjustment for time base of all_u,all_v to t in days
t_steps=fac*dn_steps
# Running state with initial conditions
xy=np.array(p0).copy()
c=g.select_cells_nearest(p0,inside=True)
t=t_steps[0]
output=[xy.copy()] # [xy, ...]
dense=[ (xy[0],xy[1],t) ]
all_t=fac*all_dn
ti=np.searchsorted(all_t,t)
u = all_u[c_mapper[c],ti]
v = all_v[c_mapper[c],ti]
last_j=None
edge_norm=g.edges_normals()

assert c is not None # should at least start in the domain!
assert (t>=all_t[0]) and (t<all_t[-1])

# Loop over intervals:
for stepi in range(len(t_steps)-1):
    t0,tN=t_steps[ [stepi,stepi+1] ]
    while t<tN: # integrate within this interval:
        
        # the max time step we can take is the minimum of
        # (i) time to next output interval
        # (ii) time to next update of input velocities
        # (iii) time to cross into a new cell
        dt_max_out=tN-t
        dt_max_in =all_t[ti+1] - t

        dt_max_edge=np.inf
        j_cross=None
        for j in g.cell_to_edges(c):
            if j==last_j:
                continue # don't cross back
            normal=edge_norm[j]
            if g.edges['cells'][j,1]==c: # ~checked
                normal=-normal
            # vector from xy to a point on the edge
            d_xy_n = g.nodes['x'][g.edges['nodes'][j,0]] - xy 
            # perpendicular distance 
            dp_xy_n=d_xy_n[0] * normal[0] + d_xy_n[1]*normal[1]
            assert dp_xy_n>=0 #otherwise sgn probably wrong above
            closing=u*normal[0] + v*normal[1]
            dt_j=dp_xy_n/closing
            if dt_j>0 and dt_j<dt_max_edge:
                j_cross=j
                dt_max_edge=dt_j
        dt_min=min(dt_max_edge,dt_max_out,dt_max_in)

        # Take the step
        xy[0] += u*dt_min
        xy[1] += v*dt_min
        t+=dt_min

        if dt_max_edge==dt_min:
            # cross edge j, update time.  careful that j isn't boundary
            # or start sliding on boundary.
            print "Cross edge"
            cells=g.edges['cells'][j_cross]
            if cells[0]==c:
                new_c=cells[1]
            elif cells[1]==c:
                new_c=cells[0]
            else:
                assert False
            if new_c<0:
                print "Whoa - hit the boundary"
                assert False
            else:
                c=new_c
                last_j=j_cross
                u=all_u[c_mapper[c],ti]
                v=all_v[c_mapper[c],ti]
        if dt_max_out==dt_min:
            t=tN # in case of roundoff in t+=dt_min
            output.append(xy.copy())
        if dt_max_in==dt_min:
            ti+=1
            u = all_u[c_mapper[c],ti]
            v = all_v[c_mapper[c],ti]
            last_j=None # okay to cross back if the velocities changed.
        dense.append( (xy[0],xy[1],t) )
                
## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,clip=wide_zoom)

path=np.array(dense)

ax.plot( path[:,0], path[:,1],'k-')
ax.scatter(path[:,0],path[:,1],30,path[:,2],lw=0)

# c and xy have gotten out of sync
tiny_zoom=(568033.41600910132, 568713.36700664461, 4179074.5373536395, 4179602.5038718581)

# g.plot_cells(clip=tiny_zoom,labeler=lambda c,r: str(c))
ax.axis(tiny_zoom)

## 
# Failed attempt


reload(unstructured_grid)


## 
cdt=exact_delaunay.Triangulation()
cdt.post_check=False

import time

def load_it():
    t_last=time.time()
    n_last=0
    for n in range(g.Nnodes()):
        if n-n_last>=500:
            elapsed=time.time() - t_last
            print "%d / %d  %gs per %d"%(n,g.Nnodes(),elapsed,n-n_last)
            t_last=time.time()
            n_last=n
        cdt.add_node(x=g.nodes['x'][n],
                     _index=n)

# for 500 nodes:
# starting point is 23.7s
# after adding code to avoid full scans on edge_to_cells,
# and skip delaunay checks, down to 13s for 500 nodes.
# if we were lucky enough for this to be linear,
# then this takes about 30 minutes to load the full grid
# it does speed up a bit after a few thousad, and at 8k,
# it's running about 5s per 500
load_it()

## 
class UgridParticles(object):
    """
    time: all timestamps are stored as 64-bit double, in seconds 
    since the epoch.

    This is to hedge between the various desires on time - data is
    rarely at a resolution finer than 1s.  A unix timestamp in 64-bit
    double is exact to the second for over 280 million years.
    """
    # x: horizontal location
    # c: cell containing particle
    # j_last: last edge the particle crossed
    part_dtype = [ ('x',(np.float64,2)),
                   ('c',np.int32),
                   ('j_last',np.int32) ]

    def __init__(self,ncs,grid=None):
        self.ncs=ncs
        self.scan_ncs()

        self.load_grid(grid=grid)
        
        self.init_particles()
        self.t_unix=None
        self.current_nc_idx=-1

        self.U=None

    def scan_ncs(self):
        self.nc_catalog=np.zeros( len(self.ncs),
                                  dtype=[('start_t','f8'),
                                         ('end_t','f8')] )
        for nc_i,nc in enumerate(self.ncs):
            self.nc_catalog['start_t'][nc_i] = utils.to_unix(nc.time[0])
            self.nc_catalog['end_t'][nc_i]   = utils.to_unix(nc.time[-1])

    def set_time(self,t):
        self.t_unix=t
        self.update_velocity()

    def load_grid(self,grid=None):
        if grid is None:
            self.g=unstructured_grid.UnstructuredGrid.from_ugrid(self.ncs[0])
        else:
            self.g=grid
        self.g.edge_to_cells()

        self.edge_norm=self.g.edges_normals()

    def set_current_nc(self,nc_i):
        self.current_nc_idx=nc_i
        self.current_nc=self.ncs[self.current_nc_idx]
        self.nc_t_unix=utils.to_unix(self.current_nc.time.values)
        self.nc_time_i=-999

    def update_velocity(self):
        changed=False
        nc_i = np.searchsorted(self.nc_catalog['start_t'],self.t_unix,side='right') - 1

        if nc_i != self.current_nc_idx:
            self.set_current_nc(nc_i)
            changed=True

        new_time_i=np.searchsorted(self.nc_t_unix, self.t_unix,side='right')-1
        if new_time_i!=self.nc_time_i:
            changed=True
        self.nc_time_i=new_time_i

        # find out how long this velocity is good for
        next_step=np.inf
        if self.nc_time_i+1 < len(self.nc_t_unix):
            next_step=self.nc_t_unix[self.nc_time_i+1]
        if self.current_nc_idx+1<len(self.ncs):
            next_step=min(next_step, 
                          self.nc_catalog['start_t'][self.current_nc_idx+1] )

        assert np.isfinite(next_step) # did we integrate off the end of the avail. data?

        self.velocity_valid_time=[self.nc_t_unix[self.nc_time_i],
                                  next_step ]
        # face, layer, time.
        # assumes 2D here.
        u=self.current_nc.cell_east_velocity.values[:,0,self.nc_time_i]
        v=self.current_nc.cell_north_velocity.values[:,0,self.nc_time_i]

        self.U=np.array( [u,v] ).T # again assume 2D

    def init_particles(self):
        self.P=np.zeros( 0, self.part_dtype )

    def add_particles(self,**kw):
        fields=list(kw.keys())

        Nold=len(self.P)
        Nnew=np.atleast_1d( kw[fields[0]] ).shape[0]

        recs=np.zeros( Nnew, dtype=self.part_dtype)

        slc=slice(Nold,Nold+Nnew)

        self.P=utils.array_concatenate( [self.P,recs] )

        for k,v in six.iteritems(kw):
            self.P[k][slc]=v

        # figure out which cell they are in
        for i in range(Nold,Nold+Nnew):
            c=self.g.select_cells_nearest(self.P['x'][i],inside=True)
            self.P['c'][i]=c
            self.P['j_last'][i]=-999

    def integrate(self,output_times_unix):
        next_out_idx=0
        next_out_time=output_times_unix[next_out_idx]
        
        self.output=[]
        self.dense=[]
        self.append_state(self.output)
        self.append_state(self.dense)

        next_vel_time=self.velocity_valid_time[1]

        assert self.t_unix>=self.velocity_valid_time[0]
        assert self.t_unix<=self.velocity_valid_time[1]
        assert next_out_time>=self.t_unix

        while next_out_time is not None: # main loop
            # the max time step we can take is the minimum of
            # (i) time to next output interval
            # (ii) time to next update of input velocities
            # (iii) time to cross into a new cell
            #[(iv) eventually, time until update behavior]

            t_next=min(next_out_time,next_vel_time)

            self.move_particles(t_next)
            
            self.t_unix=t_next
            if t_next==next_out_time:
                self.append_state(self.output)
                next_out_idx+=1
                if next_out_idx<len(output_times_unix):
                    next_out_time=output_times_unix[next_out_idx]
                else:
                    next_out_time=None
            if t_next==next_vel_time:
                self.update_velocity()
                next_vel_time=self.velocity_valid_time[1]
                self.P['j_last']=-999 # okay to cross back if the velocities changed.


    def append_state(self,A):
        A.append( (self.P['x'].copy(), self.t_unix ))

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

            while part_t<stop_t:
                #if len(self.dense)==47:
                #    pdb.set_trace()

                dt_max_edge=np.inf
                j_cross=None

                for j in g.cell_to_edges(p['c']):
                    if j==p['j_last']:
                        continue # don't cross back
                    normal=self.edge_norm[j]
                    if g.edges['cells'][j,1]==p['c']: # ~checked
                        normal=-normal
                    # vector from xy to a point on the edge
                    d_xy_n = g.nodes['x'][g.edges['nodes'][j,0]] - p['x']
                    # perpendicular distance 
                    dp_xy_n=d_xy_n[0] * normal[0] + d_xy_n[1]*normal[1]
                    assert dp_xy_n>=0 #otherwise sgn probably wrong above
                    u,v=self.U[p['c']]
                    closing=u*normal[0] + v*normal[1]
                    dt_j=dp_xy_n/closing
                    if dt_j>0 and dt_j<dt_max_edge:
                        j_cross=j
                        dt_max_edge=dt_j


                t_max_edge=part_t+dt_max_edge
                if t_max_edge>stop_t:
                    # don't make it to the edge
                    dt=stop_t-part_t
                    part_t=stop_t
                else:
                    dt=dt_max_edge
                    part_t=t_max_edge

                    # cross edge j, update time.  careful that j isn't boundary
                    # or start sliding on boundary.
                    print "Cross edge"
                    cells=g.edges['cells'][j_cross]
                    if cells[0]==p['c']:
                        new_c=cells[1]
                    elif cells[1]==p['c']:
                        new_c=cells[0]
                    else:
                        assert False
                    if new_c<0:
                        print "Whoa - hit the boundary"
                        assert False

                    self.P['c'][i]=new_c
                    self.P['j_last'][i]=j_cross

                # Take the step
                self.P['x'][i,0] += u*dt
                self.P['x'][i,1] += v*dt

                self.append_state(self.dense)


## 

fig,ax=plt.subplots(1,1,num=1)

# coll=g.plot_cells(values=depth_min,ax=ax,edgecolors='face',mask=cell_sel)
# coll.set_clim([0,0.2])
# coll.set_cmap('summer')
ecoll=g.plot_edges(clip=wide_zoom,ax=ax)
ecoll.set_lw(0.5)
ecoll.set_color('0.5')

for p in paths:
    ax.plot(p[:,0],p[:,1],'b-')
for p in rpaths:
    ax.plot(p[:,0],p[:,1],'m-')
    
ax.plot(paths[:,0,0],paths[:,0,1],'ko',label='Start')
ax.plot(paths[:,-1,0],paths[:,-1,1],'ks',label='End')

ax.xaxis.set_visible(0)
ax.yaxis.set_visible(0)
ax.legend(loc='lower right')

ax.axis('equal')
ax.axis(narrow_zoom)


## 

xyxy=poly.bounds
# had been 10x10
x=np.linspace(xyxy[0],xyxy[2],40)
y=np.linspace(xyxy[1],xyxy[3],40)

X,Y=np.meshgrid(x,y)
XY=np.array([X.ravel(),Y.ravel()]).T

# instead of just being inside the model domain, make sure that it's not 
# right on the edge, as those particles are easily stuck
#sel=np.array( [ g.select_cells_nearest(xy,inside=True) is not None
#                for xy in XY] )
interior_poly=boundary.buffer(-20) # shrink by 20m
sel=np.array( [interior_poly.contains( geometry.Point(xy))
               for xy in XY])

XY=XY[sel]
starts=XY

# Forward tracks:
# one ebb-flood cycle as shown in the time series
dn_fsteps=np.arange(ebb_flood_dns1[0],
                    ebb_flood_dns1[1],
                    5*60/86400.)

fpaths=[]
for xyi,xy in enumerate(starts):
    if xyi%10==0:
        print "%d / %d"%( xyi,len(starts) )
    else:
        print ".",
    path= my_odeint(vel,xy,dn_fsteps)
    fpaths.append(path)
fpaths=np.array(fpaths)


# In[40]:

# Reverse tracks:
rdn_steps=np.arange(ebb_flood_dns0[0],
                    ebb_flood_dns0[1],
                    5*60/86400.)

rpaths=[]
for xyi,xy in enumerate(starts):
    if xyi%10==0:
        print "%d / %d"%( xyi,len(starts) )
    else:
        print ".",
    path=my_odeint(vel,xy,dn_steps,reverse=True)
    rpaths.append(path)
rpaths=np.array(rpaths)



p0=np.array([  568274.16871831,  4178143.03463385])

# Forward tracks:
# one ebb-flood cycle as shown in the time series
dn_fsteps=np.arange(ebb_flood_dns1[0],
                    ebb_flood_dns1[1],
                    5*60/86400.)



## 


# Testing:
runs=glob.glob("/home/rusty/data/boffinator/data*")
runs.sort()
# the ones I was using on boffinator:
rsel=['/home/rusty/data/boffinator/data736060',
      '/home/rusty/data/boffinator/data736062'] 
runs=[r for r in runs if r>=rsel[0] and r<=rsel[1]]
print "Found %d days/runs to use"%len(runs)

ncs=[xr.open_dataset(os.path.join(p,'transcribed-global-proj01.nc')) for p in runs]
wide_zoom=utils.expand_xxyy([561841, 569835, 4175233, 4181423 ],2.)


ebb_flood_dts1=[numpy.datetime64('2016-04-07T06:42:00'),
                numpy.datetime64('2016-04-07T19:23:00') ]
# ebb_flood_dns0=[736060.7650459058, 736061.279]

fwd_steps=np.arange(utils.to_unix(ebb_flood_dts1[0]),
                    utils.to_unix(ebb_flood_dts1[1]),
                    10*60) 

#rdn_steps=np.arange(ebb_flood_dns0[0],
#                    ebb_flood_dns0[1],
#                    10*60/86400.)

starts=np.array( [
    [  568274.16871831,  4178143.03463385] 
])


g=unstructured_grid.UnstructuredGrid.from_ugrid(ncs[0])

## 

ptm=UgridParticles(ncs=ncs,grid=g) 

ptm.set_time(fwd_steps[0])
ptm.add_particles( x=starts )

# integrate starting from this location:

ptm.integrate(fwd_steps)

## 

clip=(565918.24332594627,
      569900.63026706409,
      4176104.4380515865,
      4183193.9995333287)
zoom=(568119.50640395319,
      568467.89101941977,
      4178006.8156055976,
      4178277.0946379215)

plt.figure(1).clf() 
g=ptm.g
fig,ax=plt.subplots(num=1)

dens_xy=np.array( [r[0] for r in ptm.dense] )
dens_t=np.array( [r[1] for r in ptm.dense] )

ax.plot(dens_xy[:,0,0],dens_xy[:,0,1],'k-')

ax.scatter(dens_xy[:,0,0],dens_xy[:,0,1],40,dens_t,lw=0)
g.plot_edges(clip=clip,ax=ax) # ,labeler=lambda e,r: str(e) )
ccoll=g.plot_cells(clip=clip) #,labeler=lambda c,r:str(c))
ccoll.set_color('w')
ccoll.set_zorder(-5)
ax.axis(zoom)

# runs to 48 steps, then jumps too far

