"""
Example of reading DWAQ output, plotting map data, history data, and
the grid.
"""

import os
import matplotlib.pyplot as plt

from stompy.grid import unstructured_grid
import stompy.model.delft.io as dio
from stompy.model.delft import dfm_grid

## 

# This run is too large to include in the repository.  It's a 64 core
# DFM hydro dataset, with nutrient datasources run for water year 2011.
run_dir="/media/hpc/opt/data/dwaq/cascade/suisun_nutrient_cycling/cascade_nutrients_v01_004/"
map_fn=os.path.join(run_dir,'wy2011.map')
hyd_fn=os.path.join(run_dir,'com-wy2011.hyd')
his_fn=os.path.join(run_dir,'wy2011.his')

## Read binary map output

ds=dio.read_map(map_fn,hyd_fn)

# Extract the grid from the dataset:
g=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

# Multicore runs of DFM end up creating a grid file with duplicate edges.
# This makes plots that include edges look bad, so use this method to clean
# up those edges.
g=dfm_grid.cleanup_multidomains(g)

## Plot map output
fig=plt.figure(1)
fig.clf()
ax=fig.add_subplot()

g.plot_edges(ax=ax,color='k',lw=0.4,alpha=0.2)
nh4=ds.NH4.isel(time=-1,layer=0).values
nh4=nh4.copy() # data from read_map may be read-only
nh4[nh4<0]=np.nan # otherwise they are -999.
ccoll=g.plot_cells(values=nh4)
ccoll.set_clim([0,0.4])
ccoll.set_edgecolors('face')

fig.savefig('demo_map.png')

## History / monitoring output

# Two formats for the returned data:
# pandas:
hist_df=dio.mon_his_file_dataframe(his_fn)
# This returns a dataframe where the rows are time,
# and columns are hierarchical with location and substance
# Time is returned as seconds since the reference time of
# the simulation.
# This particular run has only a single monitoring location
# named 'dummy', and about 10 substances

# xarray:
hist_ds=dio.his_file_xarray(his_fn)

## 


plt.figure(2).clf()
fig,axs=plt.subplots(2,num=2)

axs[0].plot(hist_df.index.values/86400.,
            hist_df.loc[:,('dummy','LocalDepth')] )

axs[1].plot(hist_ds.time,
            hist_ds.bal.sel(region='dummy',field='NH4'))

fig.savefig('demo_history.png')
