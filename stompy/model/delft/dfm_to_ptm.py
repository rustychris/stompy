"""
Augment DFM/DWAQ output so that it can be used as hydro input
for FISH-PTM.

in FISH_PTM.inp, set subgrid bathy to false.

Requirements on the run:
  DFM and DWAQ output must be synchronized, with the same start/stop/interval
  settings.  In most cases this just means that MapInterval and WaqInterval
  have the same value in the MDU file.

MapFormat: in theory this can be 1,3 or 4, meaning old-style netcdf (1,3) or
  UGRID-ish netcdf (4).  Experience with subversion dflowfm rev 52184 and 53925
  shows a bug when combining MPI, DWAQ output, and UGRID.

  To get to a working setup as quickly as possible, development is focusing on
  converting single-core UGRID w/ DWAQ output.


Variables required by PTM at each output interval include (those listed
in get_netcdf_hydro_record):
  h_flow_avg
  v_flow_avg
  Mesh2_edge_wet_area
  Mesh2_face_wet_area
  Mesh2_face_water_volume
  Mesh2_salinity_3d
  Mesh2_vertical_diffusivity_3D
  Mesh2_sea_surface_elevation
  Mesh2_edge_bottom_layer
  Mesh2_edge_top_layer
  Mesh2_face_bottom_layer
  Mesh2_face_bottom_layer

An example metadata description:
double h_flow_avg(nMesh2_edge=76593, nMesh2_layer_3d=54, nMesh2_data_time=483);
  :standard_name = "ocean_volume_transport_across_line";
  :long_name = "horizontal volume flux average over integration interval";
  :coordinates = "Mesh2_edge_x Mesh2_edge_y Mesh2_edge_lon Mesh2_edge_lat Mesh2_edge_z_3d";
  :mesh = "Mesh2";
  :grid_mapping = "Mesh2_crs";
  :location = "edge";
  :units = "m3 s-1";
  :_ChunkSizes = 11949, 18, 1; // int

This is the average volume flux at an edge (each j,k) over the preceeding time interval.
"""
##
import six
import os
import glob
import netCDF4
import numpy as np
import xarray as xr

from stompy import utils
import stompy.model.delft.io as dio
from stompy.model.delft import dfm_grid
from stompy.grid import unstructured_grid
import stompy.model.delft.waq_scenario as waq

#mdu_path="/home/rusty/src/csc/dflowfm/runs/20180807_grid97_04_ptm/flowfm.mdu"
mdu_path=("/home/rusty/mirrors/ucd-X/mwtract/TASK2_Modeling/"
          "Hydrodynamic_Model_Files/DELFT3D/Model Run Files/"
          "Feb11_Jun06_2017_08082018-rusty/FlowFM.mdu")


mdu=dio.MDUFile(mdu_path)

# check the naming of DFM output files
dfm_out_dir=mdu.output_dir()

map_file_serial=os.path.join( dfm_out_dir,
                              mdu.name+"_map.nc")
if os.path.exists(map_file_serial):
    nprocs=1
    map_file=map_file_serial
else:
    raise Exception("Not ready for MPI runs")

##

map_ds=xr.open_dataset(map_file)
g=unstructured_grid.UnstructuredGrid.from_ugrid(map_ds)

out_ds=g.write_to_xarray(mesh_name="Mesh2",
                         node_coordinates="Mesh2_node_x Mesh2_node_y",
                         face_node_connectivity='Mesh2_face_nodes',
                         edge_node_connectivity='Mesh2_edge_nodes',
                         face_dimension='nMesh2_face',
                         edge_dimension='nMesh2_edge',
                         node_dimension='nMesh2_node')
out_ds=out_ds.rename({
    'maxnode_per_face':'nMaxMesh2_face_nodes',
    'node_per_edge':'Two'
})

##
mod_map_ds=map_ds.rename({'time':'nMesh2_data_time',
                          'nmesh2d_face':'nMesh2_face',
                          'nmesh2d_edge':'nMesh2_edge',
                          'nmesh2d_node':'nMesh2_node',
                          'max_nmesh2d_face_nodes':'nMaxMesh2_face_nodes',
                          'mesh2d_face_x':'Mesh2_face_x',
                          'mesh2d_face_y':'Mesh2_face_y',
                          'mesh2d_edge_x':'Mesh2_edge_x',
                          'mesh2d_edge_y':'Mesh2_edge_y'
})

# Additional grid information:
# xarray wants the dimension made explicit here -- don't know why.
out_ds['Mesh2_face_x']=('nMesh2_face',),mod_map_ds['Mesh2_face_x']
out_ds['Mesh2_face_y']=('nMesh2_face',),mod_map_ds['Mesh2_face_y']

out_ds['Mesh2_edge_x']=('nMesh2_edge',),mod_map_ds['Mesh2_edge_x']
out_ds['Mesh2_edge_y']=('nMesh2_edge',),mod_map_ds['Mesh2_edge_y']

e2c=g.edge_to_cells()
out_ds['Mesh2_edge_faces']=('nMesh2_edge','Two'),e2c

face_edges=np.array([g.cell_to_edges(c,pad=True)
                     for c in range(g.Ncells())] )

##
out_ds['Mesh2_face_edges']=('nMesh2_face','nMaxMesh2_face_nodes'),face_edges
out_ds['Mesh2_face_depth']=('nMesh2_face',),-mod_map_ds['mesh2d_flowelem_bl'].values
out_ds['Mesh2_face_depth'].attrs.update({'positive':'down',
                                         'unit':'m',
                                         'standard_name':'sea_floor_depth_below_geoid',
                                         'mesh':'Mesh2',
                                         'long_name':'Mean elevation of bed in face'})

# recreate edge bed level based on a constant bedlevtype
bedlevtype=int(mdu['geometry','BedLevType'])
if bedlevtype==3:
    edge_z=map_ds.mesh2d_node_z.values[g.edges['nodes']].mean(axis=1)
elif bedlevtype==4:
    edge_z=map_ds.mesh2d_node_z.values[g.edges['nodes']].min(axis=1)
else:
    raise Exception("Only know how to deal with bed level type 3,4 not %d"%bedlevtype)
out_ds['Mesh2_edge_depth']=('nMesh2_edge',),edge_z

out_ds['Mesh2_data_time']=mod_map_ds.nMesh2_data_time

out_ds['Mesh2_sea_surface_elevation']=('nMesh2_data_time','nMesh2_face'),mod_map_ds.mesh2d_s1

##

# from DFM: 0=> internal closed, 1=>internal, 2=>flow or stage bc, 3=>closed
# map to -1 (error), 0=>internal, 1=>closed, 2=>flow.  no easy way to
# distinguish flow from stage bc right here.
translator=np.array([-1,0,2,1])

edge_marks=translator[ mod_map_ds.mesh2d_edge_type.values.astype(np.int32) ]
assert not np.any(edge_marks<0),"Need to implement internal closed edges"
# this gets us to 0: internal, 1:boundary, 2: boundary_closed
# this looks like 1 for stage or flow BC, 2 for land, 0 for internal.
out_ds['Mesh2_edge_bc']=('nMesh2_edge',),edge_marks
##

# 'facemark':'Mesh2_face_bc'
cell_marks=np.zeros(g.Ncells(),np.int8)
# punt, and call any cell adjacent to a marked edge BC a stage-bc cell
bc_edges=np.nonzero(edge_marks>1)[0]
bc_cells=g.edge_to_cells(bc_edges).max(axis=1) # drop the negative neighbors

cell_marks[bc_cells]=1
out_ds['Mesh2_face_bc']=('nMesh2_face',),cell_marks


##

ucx=mod_map_ds['mesh2d_ucx']
if ucx.ndim==2:
    nkmax=1
    map_2d=True
else:
    nkmax=ucx.shape[-1] # is this safe?
    map_2d=False

out_ds['nMesh2_layer_3d']=('nMesh2_layer_3d',),np.arange(nkmax)

##

# based on sample output, the last of these is not used,
# so this would be interfaces, starting with the top of the lowest layer.
# fabricate something in sigma coordinates for now.
sigma_layers=np.linspace(-1,0,nkmax+1)[1:]
out_ds['Mesh2_layer_3d']=('nMesh2_layer_3d',),sigma_layers
attrs=dict(standard_name="ocean_sigma_coordinate",
           dz_min=0.001, # not real, but ptm tries to read this.
           long_name="sigma layer coordinate at flow element top",
           units="",
           positive="up", # kind of baked into the sigma definition
           formula_terms="sigma: Mesh2_layer_3d eta: Mesh2_sea_surface_elevation bedlevel: Mesh2_face_depth"
)
out_ds['Mesh2_layer_3d'].attrs.update(attrs)

# from http://cfconventions.org/Data/cf-conventions/cf-conventions-1.0/build/apd.html
#    z(n,k,j,i) = eta(n,j,i) + sigma(k)*(depth(j,i)+eta(n,j,i))
# sigma coordinate definition has z=positive:up baked in, likewise depth is
# positive down, and eta positive up, with sigma ranging from -1 (bed) to 0 (surface)


##
# for writing the output, use xarray to initialize the file, but
# the big data part is best handled directly by netCDF4 so we can
# control how much data is in RAM at a time.

out_fn="dfm_ptm_hydro.nc"
os.path.exists(out_fn) and os.unlink(out_fn)

out_ds.to_netcdf(out_fn)

out_nc=netCDF4.Dataset(out_fn,mode="a")

##

# Create the variables:

cell_3d_data_dims=('nMesh2_data_time','nMesh2_face','nMesh2_layer_3d')
edge_3d_data_dims=('nMesh2_data_time','nMesh2_edge','nMesh2_layer_3d')
cell_2d_data_dims=('nMesh2_data_time','nMesh2_face')
edge_2d_data_dims=('nMesh2_data_time','nMesh2_edge')

if 'mesh2d_sa1' in mod_map_ds:
    salt_var=out_nc.createVariable('Mesh2_salinity_3d',
                                   np.float64,cell_3d_data_dims)
else:
    salt_var=None

nut_var=out_nc.createVariable('Mesh2_vertical_diffusivity_3d',
                              np.float64,cell_3d_data_dims)

edge_k_bot_var=out_nc.createVariable('Mesh2_edge_bottom_layer',
                                     np.int32,edge_2d_data_dims)
edge_k_top_var=out_nc.createVariable('Mesh2_edge_top_layer',
                                     np.int32,edge_2d_data_dims)
cell_k_bot_var=out_nc.createVariable('Mesh2_face_bottom_layer',
                                     np.int32,cell_2d_data_dims)
cell_k_top_var=out_nc.createVariable('Mesh2_face_top_layer',
                                     np.int32,cell_2d_data_dims)

if 1: # sigma layers - can assign all of the top/bottom right here
    # have to write this out in the untrim sense - 0 would be the bed
    # cell, Nkmax-1 is the top (for a full watercolumn).
    # afaict, layer counting starts at 0, and the top index is exclusive.

    # suntans code: bottom layer:
    # e.g. Nkmax=10, and some cell has 8 layers, so Nk=8
    #  0 at the surface, 7 is the lowest cell.
    #  then the untrim value is:
    #         tmp2d = sun.Nkmax-sun.Nk # one based
    # 10-8=2. untrim k=0 is the bed, Nkmax-1 is the top
    #
    # nc.variables[vname][:,ii]=tmp2d
    cell_k_bot_var[:]=0
    cell_k_top_var[:]=nkmax
    # do these have to be adjusted for boundaries?
    edge_k_bot_var[:]=0
    edge_k_top_var[:]=nkmax

h_flow_var=out_nc.createVariable('h_flow_avg',np.float64,edge_3d_data_dims)
# what are the expectations for surface/bed vertical velocity?
v_flow_var=out_nc.createVariable('v_flow_avg',np.float64,cell_3d_data_dims)


vol_var=out_nc.createVariable('Mesh2_face_water_volume',np.float64,cell_3d_data_dims)
A_edge_var=out_nc.createVariable('Mesh2_edge_wet_area',np.float64,edge_3d_data_dims)
A_face_var=out_nc.createVariable('Mesh2_face_wet_area',np.float64,cell_3d_data_dims)

##

hyd_fn=os.path.join(mdu.base_path,
                    "DFM_DELWAQ_%s"%mdu.name,
                    "%s.hyd"%mdu.name)

hyd=waq.HydroFiles(hyd_fn)

hyd.infer_2d_links()
poi0=hyd.pointers-1

##

# establish mapping from hydro links to grid edges.
link_to_edge_sign=[] # an edge index in g.edges, and a +-1 sign for whether the link is aligned the same.
mapped_edges={} # make sure we don't map multiple links onto the same edge

for link_idx,(l_from,l_to) in enumerate(hyd.links):
    if l_from>=0 and l_to>=0:
        j=g.cells_to_edge(l_from,l_to)
        j_cells=g.edge_to_cells(j)
        if j_cells[0]==l_from and j_cells[1]==l_to:
            sign=1
        elif j_cells[1]==l_from and j_cells[0]==l_to:
            sign=-1
        else:
            assert False,"We have lost our way"
        link_to_edge_sign.append( (j,sign) )
        assert j not in mapped_edges
        mapped_edges[j]=link_idx
    else:
        assert l_to>=0,"Was only expecting 'from' for the link to be negative"
        nbr_cells=np.array(g.cell_to_cells(l_to))
        nbr_edges=np.array(g.cell_to_edges(l_to))
        potential_edges=nbr_edges[nbr_cells<0]
        if len(potential_edges)==1:
            j=potential_edges[0]
        elif len(potential_edges)==0:
            print("No boundary edge for link %d->%d to an edge"%(l_from,l_to))
            link_to_edge_sign.append( (9999999,0) ) # may be able to relax this
            continue
        else:
            print("Link %d->%d could map to %d edges - will choose first unclaimed"
                  %(l_from,l_to,len(potential_edges)))
            # may not have enough information to know which boundary
            for j in potential_edges:
                if j in mapped_edges:
                    continue
                break
            else:
                raise Exception("Couldn't find an edge for link %d->%d"%(l_from,l_to))
        mapped_edges[j]=link_idx
        j_cells=g.edge_to_cells(j)
        if j_cells[0]==l_to:
            link_to_edge_sign.append( (j,-1) )
        elif j_cells[1]==l_to:
            link_to_edge_sign.append( (j,1) )
        else:
            assert False,"whoa there"


##

times=mod_map_ds.nMesh2_data_time.values
for ti,t in enumerate(times):
    if True: # ti%24==0:
        print("%d/%d t=%s"%(ti,len(times),t))

    def copy_3d_cell(src,dst):
        # Copies single time step of cell-centered 3D data.
        # src: string name of variable in mod_map_ds
        # dst: netCDF variable to assign to.
        src_data=mod_map_ds[src].isel(nMesh2_data_time=ti).values
        if map_2d:
            dst[ti,:,0]=src_data
        else:
            dst[ti,:,:]=src_data

    if salt_var is not None:
        copy_3d_cell('mesh2d_sa1',salt_var)
    if nkmax>1:
        copy_3d_cell('mesh2d_viw',nut_var)
    else:
        # punt.
        nut_var[ti,:,:]=0.0

    # h_flow gets interesting as we have to read dwaq output
    # dwaq uses seconds from reference time
    hyd_t_sec=(t-utils.to_dt64(hyd.time0))/np.timedelta64(1,'s')
    # flows is all horizontal flows, layer by layer, surface to bed,
    # and then vertical flows.
    # only flow edges get flows, though.
    flows=hyd.flows(hyd_t_sec)
    areas=hyd.areas(hyd_t_sec)

    # compose exch_to_2d_link,link_to_edge_sign, weed out unmapped
    # links, and copy into h_flow_avg.  start with naive loops
    h_flow_avg=np.zeros((g.Nedges(),nkmax),np.float64)
    h_area_avg=np.zeros_like(h_flow_avg)
    for exch in range(hyd.n_exch_x+hyd.n_exch_y):
        Q=flows[exch]
        # for horizontal exchanges in a sigma grid, this is a safe way
        # to get layer:
        k=hyd.seg_k[poi0[exch][1]]
        link=hyd.exch_to_2d_link['link'][exch]
        Q=Q*hyd.exch_to_2d_link['sgn'][exch]

        j,j_sgn=link_to_edge_sign[link]
        if j<0:
            continue
        Q=Q*j_sgn
        # so far this is k in the DWAQ world, surface to bed.
        # but now we assign in the untrim sense, bed to surface.
        h_flow_avg[j,nkmax-k-1]=Q
        h_area_avg[j,nkmax-k-1]=areas[exch]
    h_flow_var[ti,:,:]=h_flow_avg
    A_edge_var[ti,:,:]=h_area_avg

    v_flow_avg=np.zeros((g.Ncells(),nkmax), np.float64)
    v_area_avg=np.zeros_like(v_flow_avg)
    for exch in range(hyd.n_exch_x+hyd.n_exch_y,hyd.n_exch):
        # negate, because dwaq records this relative to the exchange,
        # which is top-segment to next segment down.
        Q=-flows[exch]
        assert poi0[exch][0] >=0,"Wasn't expecting BC exchanges in the vertical"
        seg_up,seg_down=poi0[exch][0,:2]
        k_upper=hyd.seg_k[seg_up]
        k_lower=hyd.seg_k[seg_down]
        elt=hyd.seg_to_2d_element[seg_up]
        assert k_upper+1==k_lower,"Thought this was a given"
        assert elt==hyd.seg_to_2d_element[seg_down],"Maybe this wasn't a vertical exchange"
        # based on looking at the untrim output, this should be recorded to
        # the k of the lower layer, but also flipped to be bed->surface
        # ordered
        # assumes that dwaq cells are numbered the same as dfm cells.
        v_flow_avg[elt,nkmax-k_lower-1]=Q
        if k_upper==0: # repeat top flux
            v_flow_avg[elt,nkmax-k_upper-1]=Q
    v_flow_var[ti,:,:]=v_flow_avg
    A_face_var[ti,:,:]=v_area_avg

    #   Mesh2_face_water_volume
    vols=hyd.volumes(hyd_t_sec)
    # assume again that cells are numbered the same.
    # they come to us ordered by first all the top layer, then the second
    # layer, on down to the bed.  convert to 3D, and reorder the layers
    vols=vols.reshape( (nkmax,g.Ncells()) )[::-1,:]
    vol_var[ti,:,:] = vols.T


# for vertical flows, there are nkmax layers, but nkmax-1
# internal flux faces, or nkmax+1 total faces.  how is the
# staggering expected to be handled? best guess:
#  v_flow_avg[ k=10 ] is the volume flux between volume[k=10]
# and volume[k=11], i.e. transport with the volume above
# this one.  it looks like, at least in the untrim output, that
# the last flux is repeated.  one confusing point is that in the
# case of a dry surface layer, we'd expect to see something
# like [Qa, Qb, Qc, Qc, 0], but instead I've seen
# [Qa, Qb, Qc, Qd, Qd].  I don't understand what that's about.
# look at cell 18, around time index 200, 201


##

## Matt Rayson's code:
#
## Dimensions with hard-wired values
#other_dims = {\
#        'Three':3,\
#        'Two':2,\
#        'nsp':1,\
#        'date_string_length':19,\
#        'nMesh2_time':1\
#        }
#
#            'h_flow_avg',\
#            'v_flow_avg',\
#            'Mesh2_edge_wet_area',\
#            'Mesh2_face_wet_area',\
#            'Mesh2_edge_bottom_layer',\
#            'Mesh2_edge_top_layer',\
#            'Mesh2_face_bottom_layer',\
#            'Mesh2_face_top_layer',\
#            'Mesh2_face_water_volume',\
#
#
#FILLVALUE=-9999
#
#def suntans2untrim(ncfile,outfile,tstart,tend,grdfile=None):
#    """
#    Converts a suntans averages netcdf file into untrim format
#    for use in particle tracking
#    """
#    ####
#    # Step 1: Load the suntans data object
#    ####
#    print "ncfile",ncfile
#    sun = Spatial(ncfile,klayer=[-99])
#
#    # Calculate some other variables
#    sun.de = sun.get_edgevar(sun.dv,method='min')
#    sun.mark[sun.mark==5]=0
#    sun.mark[sun.mark==3]=2
#    sun.facemark = np.zeros((sun.Nc,),dtype=np.int)
#
#    # Update the grad variable from the ascii grid file if supplied
#    if not grdfile == None:
#        print 'Updating grid with ascii values...'
#        grd = Grid(grdfile)
#        sun.grad = grd.grad[:,::-1]
#
#    ###
#    # Step 2: Write the grid variables to a netcdf file
#    ###
#    nc = Dataset(outfile,'w',format='NETCDF4_CLASSIC')
#
#    # Global variable
#    nc.Description = 'UnTRIM history file converted from SUNTANS output'
#
#    # Write the dimensions
#    for dd in untrim_griddims.keys():
#        if dd == 'time':
#            nc.createDimension(untrim_griddims[dd],0)
#        elif dd =='numsides':
#            nc.createDimension(untrim_griddims[dd],sun.maxfaces)
#        else:
#            nc.createDimension(untrim_griddims[dd],sun[dd])
#
#
#    for dd in other_dims:
#        nc.createDimension(dd,other_dims[dd])
#
#    ###
#    # Step 3: Initialize all of the grid variables
#    ###
#    def create_nc_var(name, dimensions, attdict,data=None, \
#        dtype='f8',zlib=False,complevel=0,fill_value=999999.0):
#            
#        tmp=nc.createVariable(name, dtype, dimensions,\
#            zlib=zlib,complevel=complevel,fill_value=fill_value)
#
#        for aa in attdict.keys():
#            tmp.setncattr(aa,attdict[aa])
#        
#        if not data==None:
#            nc.variables[name][:] = data
#     
#    # Make sure the masked cells have a value of -1
#    mask = sun['cells'].mask.copy()
#    sun['cells'][mask]=FILLVALUE
#    sun['face'][mask]=FILLVALUE
#
#    for vv in untrim_gridvars.keys():
#        vname = untrim_gridvars[vv]
#        print 'Writing grid variable %s (%s)...'%(vname,vv)
#
#        if vv=='time':
#            continue
#
#        # add dz_min attribute to z_r variable
#        if vv == 'z_r':
#            ugrid[vname]['attributes'].update({'dz_min':1e-5})
#            #sun[vv][:]=sun[vv][::-1]
#            sun[vv][:]=sun['z_w'][0:-1][::-1]
#
#        # Reverse the order of grad(???)
#        if vv=='grad':
#            sun[vv][:]=sun[vv][:,::-1]
#
#        ## Fix one-based indexing
#        #if vv in ['cells','edges','grad']:
#        #    mask = sun[vv][:]==-1
#        #    tmp = sun[vv][:]+1
#        #    tmp[mask]=-1
#        #    #sun[vv][:]=sun[vv][:]+1
#        #    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
#        #        data=tmp,dtype=ugrid[vname]['dtype'])
#
#        create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
#            data=sun[vv],dtype=ugrid[vname]['dtype'])
#
#            
#    # Initialize the two time variables
#    vname=untrim_gridvars['time']
#    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
#            dtype=ugrid[vname]['dtype'])
#    vname = 'Mesh2_data_time_string'
#    create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
#            dtype=ugrid[vname]['dtype'])
#
#    ###
#    # Step 4: Initialize all of the time-varying variables (but don't write) 
#    ###
#    for vname  in varnames:
#        print 'Creating variable %s...'%(vname)
#
#        create_nc_var(vname,ugrid[vname]['dimensions'],ugrid[vname]['attributes'],\
#            dtype=ugrid[vname]['dtype'],zlib=True,complevel=1,fill_value=999999.)
#
#    ###
#    # Step 5: Loop through all of the time steps and write the variables
#    ###
#    tsteps = sun.getTstep(tstart,tend)
#    tdays = othertime.DaysSince(sun.time,basetime=datetime(1899,12,31))
#    for ii, tt in enumerate(tsteps):
#        # Convert the time to the untrim formats
#        timestr = datetime.strftime(sun.time[tt],'%Y-%m-%d %H:%M:%S')
#
#        print 'Writing data at time %s (%d of %d)...'%(timestr,tt,tsteps[-1]) 
#
#        #Write the time variables
#        nc.variables['Mesh2_data_time'][ii]=tdays[ii]
#        nc.variables['Mesh2_data_time_string'][:,ii]=timestr
#
#        # Load each variable or calculate it and convert it to the untrim format
#        sun.tstep=[tt]
#
#        ###
#        # Compute a few terms first
#        eta = sun.loadData(variable='eta' )  
#        U = sun.loadData(variable='U_F' )  
#        #U = sun.loadData(variable='U' )  
#        dzz = sun.getdzz(eta)
#        dzf = sun.getdzf(eta)
#
#        
#        vname='Mesh2_sea_surface_elevation'
#        #print '\tVariable: %s...'%vname
#        nc.variables[vname][:,ii]=eta
#
#        vname = 'Mesh2_salinity_3d'
#        #print '\tVariable: %s...'%vname
#        tmp3d = sun.loadData(variable='salt' )  
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        vname = 'Mesh2_vertical_diffusivity_3d'
#        #print '\tVariable: %s...'%vname
#        tmp3d = sun.loadData(variable='nu_v' )  
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        vname = 'h_flow_avg'
#        #print '\tVariable: %s...'%vname
#        nc.variables[vname][:,:,ii]=U.swapaxes(0,1)[:,::-1]
#
#
#        vname = 'v_flow_avg'
#        #print '\tVariable: %s...'%vname
#        tmp3d = sun.loadData(variable='w' ) * sun.Ac # m^3/s
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        # Need to calculate a few terms for the other variables
#
#        vname = 'Mesh2_edge_wet_area'
#        #print '\tVariable: %s...'%vname
#        #dzf = sun.loadData(variable='dzf')
#        tmp3d = dzf*sun.df
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        vname = 'Mesh2_face_water_volume'
#        #print '\tVariable: %s...'%vname
#        #dzz = sun.loadData(variable='dzz')
#        tmp3d = dzz*sun.Ac
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        vname = 'Mesh2_face_wet_area'
#        #print '\tVariable: %s...'%vname
#        tmp3d = np.repeat(sun.Ac[np.newaxis,...],sun.Nkmax,axis=0)
#        nc.variables[vname][:,:,ii]=tmp3d.swapaxes(0,1)[:,::-1]
#
#        # UnTRIM references from bottom to top i.e.
#        # k = 0 @ bed ; k = Nkmax-1 @ top
#
#        vname = 'Mesh2_edge_bottom_layer'
#        #print '\tVariable: %s...'%vname
#        #tmp2d = sun.Nkmax-sun.Nke # zero based
#        tmp2d = sun.Nkmax-sun.Nke+1 # one based
#        nc.variables[vname][:,ii]=tmp2d
#
#        vname = 'Mesh2_edge_top_layer'
#        #print '\tVariable: %s...'%vname
#        etop = sun.loadData(variable='etop')
#        #tmp2d = sun.Nkmax-etop-1 # zero based
#        tmp2d = sun.Nkmax-etop # one based
#        nc.variables[vname][:,ii]=tmp2d
#
#        vname = 'Mesh2_face_bottom_layer'
#        #print '\tVariable: %s...'%vname
#        #tmp2d = sun.Nkmax-sun.Nk + 1 # zero based
#        tmp2d = sun.Nkmax-sun.Nk # one based
#        nc.variables[vname][:,ii]=tmp2d
#
#        vname = 'Mesh2_face_top_layer'
#        #print '\tVariable: %s...'%vname
#        ctop = sun.loadData(variable='ctop')
#        #tmp2d = sun.Nkmax-ctop-1 # zero based
#        tmp2d = sun.Nkmax-ctop # one based
#        nc.variables[vname][:,ii]=tmp2d
#
#    print 72*'#'
#    print '\t Finished SUNTANS->UnTRIM conversion'
#    print 72*'#'
#
#
#
#
#
#
#    # close the file
#    nc.close()

