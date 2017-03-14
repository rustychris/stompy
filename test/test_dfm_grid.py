from stompy.model.delft import dfm_grid
import matplotlib.pyplot as plt
import xarray as xr

## 

# g=dfm_grid.DFMGrid("/home/rusty/models/grids/mick_alviso_v4_net.nc/mick_alviso_v4_net.nc")
nc_fn="/home/rusty/models/delft/dfm/alviso/20151214/Alviso_input/alviso2012_net.nc"
g=dfm_grid.DFMGrid(nc_fn)
ds=xr.open_dataset(nc_fn)

## 

fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot(1,1,1)

g.plot_edges(ax=ax,color='k')
