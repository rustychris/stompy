"""
Test setup for DFM/DWAQ output manipulations
"""

import subprocess
import six
import os
import xarray as xr
from stompy.grid import unstructured_grid
from stompy import utils
from stompy.spatial import wkb2shp
import numpy as np
from stompy.plot import plot_wkb
from stompy.model.delft import dflow_model, dfm_to_ptm
from shapely import ops
from scipy.cluster import vq

from stompy.model.delft import waq_hydro_editor, waq_scenario


##
# For the nosetests version (not the dev script), assumes that dflowfm and mpiexec are
# on PATH

dfm_bin_dir=""
dflow_model.DFlowModel.dfm_bin_dir=dfm_bin_dir
dflow_model.DFlowModel.mpi_bin_dir=dfm_bin_dir

## 
def base_model(force=True,num_procs=1,run_dir='run-dfm-test'):
    if not force and dflow_model.DFlowModel.run_completed(run_dir):
        model=dflow_model.DFlowModel.load(run_dir)
        return model
            
    g=unstructured_grid.UnstructuredGrid(max_sides=4)

    ret=g.add_rectilinear([0,0],[500,500],50,50)

    # sloping N-S from -3 at y=0 to +2 at y=500
    g.add_node_field('depth',-3 + g.nodes['x'][:,1]/100)

    model=dflow_model.DFlowModel()
    model.load_template('dflow-template.mdu')
    model.set_grid(g)

    model.num_procs=num_procs

    model.set_run_dir(run_dir, mode='pristine')
    model.run_start=np.datetime64("2018-01-01 00:00")
    model.run_stop =np.datetime64("2018-01-03 00:00")
    dt=np.timedelta64(300,'s')
    t=np.arange(model.run_start-20*dt,
                model.run_stop +20*dt,
                dt)
    # 4h period
    periodic_Q=50*np.sin((t-t[0])/np.timedelta64(1,'s') * 2*np.pi/(4*3600.))
    Q=xr.DataArray(periodic_Q,dims=['time'],coords={'time':t})
    inflow=dflow_model.FlowBC(name='inflow',
                              geom=np.array([ [0,0],[0,500]]),
                              Q=Q)
    model.add_bcs(inflow)

    model.projection='EPSG:26910'
    model.mdu['geometry','WaterLevIni']=0.0
    model.mdu['output','WaqInterval']=1800
    model.mdu['output','MapInterval']=1800
    model.write()

    if not model.is_completed():
        model.partition()
        model.run_model()
    
    return model

##

def gen_aggregation_shp(model):
    pnts=model.grid.cells_centroid()

    # make this deterministic
    np.random.seed(37)

    centroids,labels=vq.kmeans2(pnts,k=20,iter=5,minit='points')
    permute=np.argsort(np.random.random(labels.max()+1))
    # Make a shapefile out of that
    polys=[]
    for k,grp in utils.enumerate_groups(labels):
        grp_poly = ops.cascaded_union([model.grid.cell_polygon(i) for i in grp])
        assert grp_poly.type=='Polygon',"Hmm - add code to deal with multipolygons"
        polys.append(grp_poly)

    agg_shp_fn="dwaq_aggregation.shp"
    wkb2shp.wkb2shp(agg_shp_fn,polys,overwrite=True)
    return agg_shp_fn

##

def test_agg():
    """
    Test aggregation of a small run,
    also continuity checks.
    and detiding
    """
    # Get a well-behaved small hydro run:
    model=base_model(force=False)
    agg_shp_fn=gen_aggregation_shp(model)

    hyd_path=os.path.join(model.run_dir,"DFM_DELWAQ_%s"%model.mdu.name,"%s.hyd"%model.mdu.name)
    assert os.path.exists(hyd_path)

    waq_hydro_editor.main(args=["-i",hyd_path,"-a",agg_shp_fn,"-o","output_agg/output"])

    # aggregated accumulates some additional error - 1e-6 is not alarming.
    waq_hydro_editor.main(args=["-i","output_agg/com-output.hyd","-c"]) # aggregated run

    # With data stored as float32, 1e-8 relative error is machine precision.
    waq_hydro_editor.main(args=["-i",hyd_path,"-c"]) # the original run

    # De-tide the original
    waq_hydro_editor.main(args=["-i",hyd_path,"-l","-o","output_lp/output"])

    # Accumulates a bit more error than the original, but still 1e-6 or so
    waq_hydro_editor.main(args=["-i","output_lp/com-output.hyd","-c"]) # lowpass run

    # De-tide the aggregated
    waq_hydro_editor.main(args=["-i","output_agg/com-output.hyd","-l","-o","output_agg_lp/output"])

    # still about 1e-6 errors.
    waq_hydro_editor.main(args=["-i","output_agg_lp/com-output.hyd","-c"]) 

def test_mpi():
    # splicing of mpi runs:
    mpi_model=base_model(force=False,run_dir='run-dfm-test-mpi',num_procs=4)

    waq_hydro_editor.main(args=["-m","run-dfm-test-mpi/flowfm.mdu","-s","-o","output_splice/output"])

    # this does trip a warning -- seems the relative errors are sometimes 1e-8, sometimes as bad
    # as 1e-4.
    waq_hydro_editor.main(args=["-i","output_splice/com-output.hyd","-c"]) 



