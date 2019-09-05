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

##
# machine-specific paths
# This is an older version, just because don't have the newest on
# my dev machine
dfm_bin_dir=os.path.join(os.environ['HOME'],
                         "src/dfm/r53925-opt/bin")
dflow_model.DFlowModel.dfm_bin_dir=dfm_bin_dir
dflow_model.DFlowModel.mpi_bin_dir=dfm_bin_dir

## 
def base_model(force=True):
    run_dir='run-dfm_to_ptm'
    
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

# Get a well-behaved small hydro run:
model=base_model(force=False)
if not model.is_completed():
    model.partition()
    model.run_model()

##

# Fabricate a really basic aggregation
from scipy.cluster import vq

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
##

import matplotlib.pyplot as plt
plt.figure(1).clf()
ax=plt.gca()
model.grid.plot_cells(values=permute[labels],ax=ax)
ax.axis('equal')

for poly in polys:
    plot_wkb.plot_wkb(poly,ax=ax,fc='none',lw=3)

##

hyd_path=os.path.join(model.run_dir,"DFM_DELWAQ_%s"%model.mdu.name,"%s.hyd"%model.mdu.name)
assert os.path.exists(hyd_path)

##

# Build up the command to test:
if 0:
    cmd="python -m stompy.model.delft.waq_hydro_editor -i %s -a %s"%(hyd_path,agg_shp_fn)
    subprocess.run(cmd,shell=True)

##

from stompy.model.delft import waq_hydro_editor, waq_scenario

six.moves.reload_module(waq_scenario)
six.moves.reload_module(waq_hydro_editor)

waq_hydro_editor.main(args=["-i",hyd_path,"-a",agg_shp_fn])

# gets a ways, but then it's not quite finding the boundary elements?
# sub_geom just has 0 for all of the boundary elements -- (waq_scenario:4723)
# but we're looking for bc_elt_pos+1 == 2402
# ah - something is causing it not to find those bc elements.
# okay - I think part of the confusion is from whether I'm reading the map
# file or the flowgeom file.

##

# So now running into an issue that we don't have FlowLink.
# already figured out that the number of flow links is
# the edge type 1 and 2 edges.
# but how are they ordered?
fg1=xr.open_dataset('run-dfm_to_ptm-map3/DFM_DELWAQ_flowfm/flowfm_waqgeom.nc')
map1=xr.open_dataset('run-dfm_to_ptm-map3/DFM_OUTPUT_flowfm/flowfm_map.nc')

fg4=xr.open_dataset('run-dfm_to_ptm/DFM_DELWAQ_flowfm/flowfm_waqgeom.nc')
map4=xr.open_dataset('run-dfm_to_ptm/DFM_OUTPUT_flowfm/flowfm_map.nc')

##

# How does the ordering of edges vary?
map4.mesh2d_edge_faces
fg4.mesh2d_edge_faces

for label,ds in [('Map output, format=4',map4),
                 ('flowgeom, format=4',fg4)]:
    print(label)
    print(f"   nan? {np.isnan(ds.mesh2d_edge_faces.values).sum(axis=0)}")
    print(f"   0?   {(ds.mesh2d_edge_faces.values==0).sum(axis=0)}")

for label,ds in [('Map output, format=1',map1),
                 ('flowgeom, format=1',fg1)]:
    print(label)
    print(f"   nan? {np.isnan(ds.FlowLink.values).sum(axis=0)}")
    print(f"   0?   {(ds.FlowLink.values==0).sum(axis=0)}")

## 
# Are face indices the same between these? yes
assert np.allclose(fg1.FlowElem_xcc.values,fg4.mesh2d_face_x.values)
assert np.allclose(fg1.FlowElem_ycc.values,fg4.mesh2d_face_y.values)


# pull out the face
edge_type=fg4.mesh2d_edge_type.values
flow_edges=(edge_type==1) | (edge_type==2)

FlowLink_xu=fg4.mesh2d_edge_x.values[flow_edges]
FlowLink_yu=fg4.mesh2d_edge_y.values[flow_edges]

# these succeed
assert np.allclose( fg1.FlowLink_xu.values, FlowLink_xu )
assert np.allclose( fg1.FlowLink_yu.values, FlowLink_yu )

flowlink_to_edge=np.nonzero( (edge_type==1)|(edge_type==2) )[0]

# fabricating FlowLink
FlowLink=fg4.mesh2d_edge_faces.values[flowlink_to_edge,:]
# 
FlowLink[ np.isnan(FlowLink) ] = -1
FlowLink=FlowLink.astype(np.int32)
bc_links=(FlowLink[:,1]<0)
assert np.all(FlowLink[:,0]>=0) # not sure if that 0- or 1-based
# Boundary links get flipped to have the outside fake element first.
FlowLink[bc_links]=FlowLink[bc_links,::-1]
FlowLink[bc_links,0]=fg4.dims['nmesh2d_face'] + 1 + np.arange(bc_links.sum())

assert np.all(fg1.FlowLink.values==FlowLink)
FlowLinkType=2*np.ones(len(flowlink_to_edge))

assert np.all( fg1.FlowLinkType.values==FlowLinkType)
# don't care about lon/lat.
#     FlowLink_lonu                (nFlowLink) float64 ...
#     FlowLink_latu                (nFlowLink) float64 ...

