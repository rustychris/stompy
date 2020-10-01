"""
Example of scripting DFM and DWAQ

This script runs a simple DFM domain (square cartesian grid, with
oscillating flow boundary in one corner.), and then a DWAQ dye 
release simulation.

sets up a tracer run with a spatially-variable initial
condition and runs it for the duration of the hydro.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from stompy.grid import unstructured_grid
import stompy.plot.cmap as scmap

import stompy.model.hydro_model as hm
import stompy.model.delft.waq_scenario as dwaq
import stompy.model.delft.dflow_model as dfm

# DEV
import six
six.moves.reload_module(hm)
six.moves.reload_module(dfm)
six.moves.reload_module(dwaq)

##

# put machine specific paths and settings in local_config.py
import local_config
local_config.install()

## 

# DFM 
def base_model(force=True,num_procs=1,run_dir='dfm_run'):
    """
    Create or load a simple DFM hydro run for testing purposes.
    If the run already exists and ran to completion, simply load the existing model.

    force: if True, force a re-run even if the run exists and completed.
    num_procs: if >1, attempt MPI run.
    run_dir: path to location of the run.
    """
    if (not force) and dfm.DFlowModel.run_completed(run_dir):
        model=dfm.DFlowModel.load(run_dir)
        return model

    # construct a very basic hydro run
    g=unstructured_grid.UnstructuredGrid(max_sides=4)

    # rectilinear grid
    L=500
    ret=g.add_rectilinear([0,0],[L,L],50,50)

    # sloping N-S from -10 at y=0 to -5 at y=500
    # node_z_bed will be used by the DFlowModel script to set bathymetry.
    # This is a positive-up quantity
    g.add_node_field('node_z_bed',-10 + g.nodes['x'][:,1] * 5.0/L)

    model=dfm.DFlowModel()
    # Load defaults from a template:
    model.load_template('dflow-template.mdu')
    model.set_grid(g)
    # 3D, 5 layers. Defaults to sigma, evenly spaced
    model.mdu['geometry','Kmx']=5

    model.num_procs=num_procs

    # pristine means clean existing files that might get in the way
    model.set_run_dir(run_dir, mode='pristine')
    model.run_start=np.datetime64("2018-01-01 00:00")
    model.run_stop =np.datetime64("2018-01-03 00:00")
    dt=np.timedelta64(300,'s')
    t=np.arange(model.run_start-20*dt,
                model.run_stop +20*dt,
                dt)
    # Add a periodic flow boundary condition. 4h period
    periodic_Q=10*np.sin((t-t[0])/np.timedelta64(1,'s') * 2*np.pi/(4*3600.))
    Q=xr.DataArray(periodic_Q,dims=['time'],coords={'time':t})
    # enters the domain over 100m along one edge
    # the name is important here -- it can be used when setting up the
    # DWAQ run (where it will become 'inflow_flow')
    inflow=hm.FlowBC(name='inflow',
                     geom=np.array([ [0,0],[0,100]]),
                     flow=Q)
    # just a little salt, to get some baroclinicity but nothing crazy
    inflow_salt=hm.ScalarBC(parent=inflow,scalar='salinity',value=2.0)
    model.add_bcs([inflow,inflow_salt])

    # Also add a steady source BC with a temperature signature
    point_src=hm.SourceSinkBC(name='pnt_source',
                              geom=np.array([300,300] ),
                              flow=10)
    point_src_temp=hm.ScalarBC(parent=point_src,scalar='temperature',value=10.0)
    model.add_bcs([point_src,point_src_temp])
    
    model.projection='EPSG:26910' # some steps want a projection, though in this case it doesn't really matter.
    model.mdu['geometry','WaterLevIni']=0.0
    # turn on DWAQ output at half-hour steps
    model.mdu['output','WaqInterval']=1800
    # and map output at the same interval
    model.mdu['output','MapInterval']=1800

    # Write out the model setup
    model.write()

    # Some preprocessing (this is necessary even if it's not an MPI run)
    model.partition()
    # Do it
    output=model.run_model()
    # Check to see that it actually ran.
    if not model.is_completed():
        print(output.decode())
        raise Exception('Model run failed')
    
    return model

# Run/load small hydro run:
model=base_model(force=True)

##

# 'model' represents the whole DFM model.
# dwaq.Hydro, and subclasses like dwaq.HydroFiles, represent
# the hydro information used by DWAQ.
base_hydro=dwaq.HydroFiles(model.hyd_output())

##
                           
# Design a tracer release
#  this is a simple gaussian blob.
def release_conc_fn(X):
    X=X[...,:2] # drop z coordinate if it's there
    X0=np.array([250,250]) # center of gaussian
    L=50
    c=np.exp( -((X-X0)**2).sum(axis=-1)/L**2 )
    c=c/c.max() # make max value 1
    return c

# Get the grid that DWAQ will use:
grid=base_hydro.grid()
# and evaluate the gaussian at the centers of its cells.
C=release_conc_fn(grid.cells_center())

# Could plot that like this:for check on sanity:
# fig=plt.figure(1)
# fig.clf()
# ax=fig.add_subplot(1,1,1)
# grid.plot_cells(values=C,cmap='jet',ax=ax)
# ax.axis('equal')

##

# Set up the DWAQ run, pointing it to the hydro instance:
wm=dwaq.WaqModel(overwrite=True,
                 base_path='dwaq_run',
                 hydro=base_hydro)
# Model will default to running for the full period of the hydro.
# Adjust start time to get a short spinup...
wm.start_time += np.timedelta64(2*3600,'s')

# Create the dye tracer with initial condition.
# Note that C was calculated as a 2D tracer above, but here
# it is used as a 3D per-segment tracer.  For 2D you could get
# away with that, but safer to have the Hydro instance convert
# 2D (element) to 3D (segment)
C_3d=base_hydro.extrude_element_to_segment(C)
# boundary condition will default to 0.0
wm.substances['dye1']=dwaq.Substance(initial=C_3d)
# uniform tracer:
wm.substances['unity']=dwaq.Substance(initial=1.0)
# and a tracer set on the boundary flows. Initial defaults to 0.0
wm.substances['boundary_dye']=dwaq.Substance()
wm.add_bc(['inflow'],'boundary_dye',1.0)
wm.map_output += ('salinity','temp')

wm.cmd_write_hydro()
wm.cmd_write_inp()
wm.cmd_delwaq1()
wm.cmd_delwaq2()
wm.write_binary_map_nc()

##

# Open the map output
ds=xr.open_dataset(os.path.join(wm.base_path, 'dwaq_map.nc'))
# Extract the grid from the output
grid_ds=unstructured_grid.UnstructuredGrid.from_ugrid(ds)

## 
# Plot that up:

tracers=['dye1','unity','boundary_dye','salinity','temp']

fig=plt.figure(1)
fig.clf()
fig.set_size_inches([10,4],forward=True)

fig,axs=plt.subplots(1,len(tracers),num=1)

cmap=scmap.load_gradient('turbo.cpt')

for ax,scal in zip(axs,tracers):
    ax.text(0.05,0.95,scal,transform=ax.transAxes,va='top')

# Drop the last time step -- the DFM tracers are not valid
# then (not sure why)

for ti in range(len(ds.time)-1):
    for ax,scal in zip(axs,tracers):
        ax.collections=[]
        clim=dict(salinity=[0,2],temp=[5,12]).get(scal,[0,1])
        ccoll=grid_ds.plot_cells(values=ds[scal].isel(time=ti,layer=0),ax=ax,cmap=cmap,
                                 clim=clim)
        ax.axis('equal')
        # plt.colorbar(ccoll,ax=ax,orientation='horizontal')

    plt.draw()
    plt.pause(0.025)

ds.close() # keeping this open can interfere with deleting or overwriting the netcdf file.
