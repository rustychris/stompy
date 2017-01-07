import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

## 

station_xy=np.loadtxt('windsuv.stations')

# this has 999 in most places, and a handful of signed values
query=np.loadtxt('windsuv.query') # 246,108
u_query=query[:nrows ]
v_query=query[nrows:]

# this has a nice smooth field interpolated between stations
output=np.loadtxt('windsuv.out') # 246,108
u_output=output[:nrows]
v_output=output[nrows:]

# has similar pattern as query, but different values.
dat=np.loadtxt('windsuv.dat')
u_dat=dat[:nrows]
v_dat=dat[nrows:]

# basic DEM
terrain=np.loadtxt('terrain.dat')

## 
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

ax.plot(station_xy[:,0],
        station_xy[:,1],'ko')

# not sure that came out right
nrows=query.shape[0]//2

# mostly 999, but with seemingly independent, signed values, in the
# range -100, 176.  dm/s ?
# clumping of signs supports this being wind vectors
# ax.imshow(u_query[::-1,:],cmap='jet',origin='lower')

# ax.imshow(u_dat[::-1,:],cmap='jet',origin='lower')
# ax.imshow(v_output[::-1,:],cmap='jet',origin='lower')

# ax.imshow( (u_dat-u_query)[::-1,:], cmap='jet',origin='lower')


ax.imshow(terrain[::-1,:],cmap='jet',origin='lower')

##

ds=xr.open_dataset('raob_soundings25653.cdf',decode_times=0)

##


# quiver plot:
x=np.arange(u_output.shape[1])
y=np.arange(u_output.shape[0])[::-1]

X,Y=np.meshgrid(x,y)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
#ax.imshow(v_output[::-1,:],origin='lower')
ax.imshow(terrain[::-1,:],origin='lower',interpolation='nearest')

slc=slice(None,None,5)
ax.quiver(X[slc,slc],Y[slc,slc],
          u_output[slc,slc],v_output[slc,slc])
