"""
Test setup for DFM/DWAQ => PTM conversion with wetting and drying
in 2D
"""
import os
import xarray as xr
from stompy.grid import unstructured_grid
import numpy as np
from stompy.model.delft import dflow_model, dfm_to_ptm

# machine-specific paths
dfm_bin_dir=os.path.join(os.environ['HOME'],
                         "src/dfm/r53925-opt/bin")
dflow_model.DFlowModel.dfm_bin_dir=dfm_bin_dir
dflow_model.DFlowModel.mpi_bin_dir=dfm_bin_dir

## 
def base_model():
    g=unstructured_grid.UnstructuredGrid(max_sides=4)

    ret=g.add_rectilinear([0,0],[500,500],50,50)

    # sloping N-S from -3 at y=0 to +2 at y=500
    g.add_node_field('depth',-3 + g.nodes['x'][:,1]/100)

    model=dflow_model.DFlowModel()
    model.load_template('dflow-template.mdu')
    model.set_grid(g)

    model.num_procs=1

    model.set_run_dir('run-dfm_to_ptm', mode='pristine')
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

    return model

model=base_model()

share_bin_dir=dfm_bin_dir

model.partition()
model.run_model()

##

ptm_out_dir='ptm_out'
os.path.exists(ptm_out_dir) or os.makedirs(ptm_out_dir)

hydro_out_fn=os.path.join(ptm_out_dir,'test_hydro.nc')
grd_fn=os.path.join(ptm_out_dir,'test_sub.grd')
converter=dfm_to_ptm.DFlowToPTMHydro(model.mdu.filename,hydro_out_fn,grd_fn=grd_fn,
                                     overwrite=True)

##

# Continuity checks:

hydro_nc=xr.open_dataset(hydro_out_fn)
grd=unstructured_grid.PtmGrid(grd_fn)

## 
# reconstruct continuity check per notes,
# test all cells.

def check_continuity(hydro_nc,c,k):
    # Instantaneous volumes
    k=0
    cell_vol=hydro_nc['Mesh2_face_water_volume'].isel(nMesh2_face=c, nMesh2_layer_3d=k).values

    t=hydro_nc['Mesh2_data_time'].values
    t_sec=(t-t[0])/np.timedelta64(1,'s')
    dt_sec=np.median(np.diff(t_sec))

    Qout_sum=np.zeros(len(t_sec),np.float64)

    for j in hydro_nc['Mesh2_face_edges'].isel(nMesh2_face=c).values:
        # pretty sure this is zero-based, and assuming that non-existent
        # edges have j<0
        if j<0: continue

        j_kbot=hydro_nc['Mesh2_edge_bottom_layer'].isel(nMesh2_edge=j)
        j_ktop=hydro_nc['Mesh2_edge_top_layer'].isel(nMesh2_edge=j) 

        Qhor=hydro_nc['h_flow_avg'].isel(nMesh2_edge=j, nMesh2_layer_3d=k).values
        # limit to wet exchanges in order to confirm ktop/kbot.
        Qhor=np.where(j_ktop>=j_kbot,Qhor,0.0)

        # What is the sign convention?
        j_cells=hydro_nc['Mesh2_edge_faces'].isel(nMesh2_edge=j).values
        if j_cells[0]==c:
            Qhor_out=Qhor
        elif j_cells[1]==c:
            Qhor_out=-Qhor
        else:
            raise Exception("Face->edge->face didn't work")
        Qout_sum[:] += Qhor_out

    Qout_int=Qout_sum*dt_sec

    # The invariant is:
    # cell_vol[i] - cell_vol[i-1] == Qout_int[i]
    for i in range(1,len(t_sec)):
        Vpred=cell_vol[i-1] - dt_sec*Qout_sum[i]
        Vreal=cell_vol[i]

        # 1e-10: allow for slop when V approaches and equals 0
        # 1e-5: WAQ uses 32-bit floats (~7 digits), and we're potentially
        # doing a lot of differencing, so be generous and allow 1e-5.
        atol=1e-10 + 1e-05 * max(cell_vol[i-1],cell_vol[i])
        assert np.abs(Vpred-Vreal)<atol


# Start with something relatively easy:
c=grd.select_cells_nearest([208., 55.])
check_continuity(hydro_nc,c,k=0)

for c in range(grd.Ncells()):
    check_continuity(hydro_nc,c,k=0)

##


