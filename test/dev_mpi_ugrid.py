"""
Use dask.Array to lazily combine subdomain outputs
"""
import os,glob
import copy
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt


## 

out_dir="/home/rusty/src/csc/dflowfm/runs/v03regroup_20190115/DFM_OUTPUT_flowfm"
map_files=glob.glob(os.path.join(out_dir,'*0???_map.nc'))

##

from stompy.grid import multi_ugrid

# Slow, all the time is in add_grid().
mu=multi_ugrid.MultiUgrid(map_files,cleanup_dfm=True)
    
ucmag=mu['mesh2d_ucmag'].isel(time=-1).values

plt.figure(1).clf()

ccoll=mu.grid.plot_cells(values=ucmag,cmap='jet')

plt.axis('tight')
plt.axis('equal')


