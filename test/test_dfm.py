"""
Test setup for DFM model run configuration
"""

import subprocess
import six
import os
import xarray as xr
from stompy.grid import unstructured_grid
from stompy import utils
import numpy as np
import stompy.model.delft.dflow_model as dfm
import stompy.model.hydro_model as hm

##

# machine-specific paths --
# for tests, assume that PATH and LD_LIBRARY_PATH are set appropriately.

def base_model(run_dir='run-dfm-test00'):
    g=unstructured_grid.UnstructuredGrid(max_sides=4)

    ret=g.add_rectilinear([0,0],[500,500],50,50)
    # sloping N-S from -3 at y=0 to +2 at y=500
    g.add_node_field('node_z_bed',-3 + g.nodes['x'][:,1]/100)

    model=dfm.DFlowModel()
    model.load_template('dflow-template.mdu')
    model.set_grid(g)

    model.num_procs=0

    model.set_run_dir(run_dir, mode='pristine')
    model.run_start=np.datetime64("2018-01-01 00:00")
    model.run_stop =np.datetime64("2018-01-03 00:00")
    dt=np.timedelta64(300,'s')
    t=np.arange(model.run_start-20*dt,
                model.run_stop +20*dt,
                dt)
    # 4h period
    periodic_Q=5*np.sin((t-t[0])/np.timedelta64(1,'s') * 2*np.pi/(4*3600.))
    Q=xr.DataArray(periodic_Q,dims=['time'],coords={'time':t})
    inflow=hm.FlowBC(name='inflow',
                     geom=np.array([ [0,0],[0,500]]),
                     flow=Q)

    stage=hm.HarmonicStageBC(name='stage',
                             msl=1.0,S2=(0.5,0),
                             geom=np.array([ [500,0],[500,500]]))

    source=hm.SourceSinkBC(name='source', flow=20.0,
                           geom=np.array([100,100]))
    
    model.add_bcs([inflow,stage,source])

    model.projection='EPSG:26910'
    model.mdu['geometry','WaterLevIni']=0.0
    model.mdu['output','WaqInterval']=1800
    model.mdu['output','MapInterval']=1800
    model.write()

    return model

##
def test_bcs():
    # Get a well-behaved small hydro run:
    model=base_model()
    model.partition()
    model.run_model()

