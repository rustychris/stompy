"""
Automate parts of setting up a DFlow hydro model.

TODO:
  allow for setting grid bathy from the model instance
  consider a procedural approach to BCs:
    add_bc_shp(...)
    add_bc_match(FlowBC(...), name="SJ_upstream")
"""
import os,shutil,glob,inspect
import six
import logging as log

import numpy as np
import xarray as xr
from shapely import geometry

import stompy.model.delft.io as dio
from stompy import xr_utils
from stompy.io.local import noaa_coops
from stompy import utils, filters, memoize
from stompy.spatial import wkb2shp, proj_utils
from stompy.model.delft import dfm_grid

from . import io as dio

class BC(object):
    name=None
    geom=None # will get populated if this came from a shapefile
    def __init__(self,model,**kw):
        """
        Create boundary condition object.  Note that no work should be done
        here, as the rest of the model data is not yet in place, and this
        instance does not even have access yet to its geometry or other
        shapefile attributes.
        """
        self.model=model # make sure we got a model instance
        self.__dict__.update(kw)

    def write(self):
        log.info("Writing feature: %s"%self.name)

        self.write_pli()
        self.write_config()
        self.write_data()

    def write_config(self):
        log.warning("Boundary condition '%s' has no write_config method"%self.name)
    def write_data(self):
        log.warning("Boundary condition '%s' has no write_data method"%self.name)

    def filename_base(self):
        """
        filename base (no extension, relative to model run_dir) used to construct
        other filenames.
        """
        return self.name

    def pli_filename(self):
        """
        Name of polyline file, relative to model run_dir
        """
        return self.filename_base() + '.pli'

    def write_pli(self):
        if self.geom is not None:
            assert self.geom.type=='LineString'
            pli_data=[ (self.name, np.array(self.geom.coords)) ]
            pli_fn=os.path.join(self.model.run_dir,self.pli_filename())
            dio.write_pli(pli_fn,pli_data)

    def default_tim_fn(self):
        """
        full path for a time file matched to the first node of the pli.
        This is only used as a default tim output path when none is
        specified.
        """
        return os.path.join(self.model.run_dir,self.filename_base() + "_0001.tim")

    def default_t3d_fn(self):
        """
        same as above, but for t3d
        """
        return os.path.join(self.model.run_dir,self.filename_base() + "_0001.t3d")

    def write_tim(self,da,fn=None):
        """
        Write a DFM tim file based on the timeseries in the DataArray.
        da must have a time dimension.  No support yet for vector-values here.
        """
        ref_date,start,stop = self.model.mdu.time_range()

        elapsed_minutes=(da.time.values - ref_date)/np.timedelta64(60,'s')

        data=np.c_[elapsed_minutes,da.values]
        if fn is None:
            fn=self.default_tim_fn()

        np.savetxt(fn,data)

    def write_t3d(self,da,z_bed,fn=None):
        """
        Write a 3D boundary condition for a feature from a vertical profile (likely
           ROMS or HYCOM data)
         - most of the time writing boundaries is here
         - DFM details for rev52184:
             the LAYERS line is silently truncated to 100 characters.
             LAYER_TYPE=z assumes a coordinate of 0 at the bed, positive up

        we assume that the incoming data has no nan, has a positive-up
        z coordinate with 0 being model datum (i.e. NAVD88)
        """
        ref_date,t_start,t_stop = self.model.mdu.time_range()

        # not going to worry about 3D yet.  see ocean_dfm.py
        # for some hints.
        assert da.ndim==2

        # new code gets an xr dataset coming in with z coordinate.
        # old code did some cleaning on ROMS data.  no more.

        # Do sort the vertical
        dz=np.diff(da.z.values)
        if np.all(dz>0):
            log.debug("Vertical order ok")
        elif np.all(dz<0):
            log.debug("3D velo flip ertical order")
            da=da.isel(z=slice(None,None,-1))

        if np.median(da.z.values) > 0:
            log.warning("Weak sign check suggests t3d input data has wrong sign on z")

        max_line_length=100 # limitation in DFM on the whole LAYERS line
        # 7 is '_2.4567'
        # -1 for minor bit of safety
        max_layers=(max_line_length-len("LAYERS=")) // 7 - 1

        # This should be the right numbers, but reverse order
        # that's probably not right now...
        sigma = (z_bed - da.z.values) / z_bed

        # Force it to span the full water column
        sigma[0]=min(0.0,sigma[0])
        sigma[-1]=max(1.0,sigma[-1])

        assert len(sigma)<=max_layers

        #     remapper=lambda y: np.interp(np.linspace(0,1,max_layers),
        #                                  np.linspace(0,1,len(sigma)),y)
        #     # Just because the use of remapper below is not compatible
        #     # with vector quantities at this time.
        #     assert da_sub.ndim-1 == 1

        sigma_str=" ".join(["%.4f"%s for s in sigma])

        # This line is truncated at 100 characters in DFM r52184.
        layer_line="LAYERS=%s"%sigma_str
        assert len(layer_line)<max_line_length

        elapsed_minutes=(da_sub.time.values - ref_date)/np.timedelta64(60,'s')

        ref_date_str=utils.to_datetime(ref_date).strftime('%Y-%m-%d %H:%M:%S')

        if fn is None:
            fn=self.default_t3d_fn()

        assert da.dims[0]=='time' # for speed-up of direct indexing

        # Can copy this to other node filenames if necessary
        with open(fn,'wt') as fp:
            fp.write("\n".join([
                "LAYER_TYPE=sigma",
                layer_line,
                "VECTORMAX=%d"%(da.ndim-1), # default, but be explicit
                "quant=velocity",
                "quantity1=velocity", # why is this here?
                "# start of data",
                ""]))
            for ti,t in enumerate(elapsed_minutes):
                fp.write("TIME=%g minutes since %s\n"%(t,ref_date_str))
                # Faster direct indexing:
                # The ravel will interleave components - unclear if that's correct.
                data=" ".join( ["%.3f"%v for v in da.values[ti,:].ravel()] )
                fp.write(data)
                fp.write("\n")


class StageBC(BC):
    # If other than None, can compare to make sure it's the same as the model
    # datum.
    datum=None

    def __init__(self,model,z,**kw):
        super(StageBC,self).__init__(model,**kw)
        self.z=z

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=waterlevelbnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   ""]
            fp.write("\n".join(lines))

    def filename_base(self):
        """
        Make it clear in the filenames what is being forced
        """
        return super(StageBC,self).filename_base()+"_ssh"

    def write_data(self):
        ref_date,start,stop=self.model.mdu.time_range()

        pad=2*np.timedelta64(24,'h')

        ds=xr.Dataset()
        ds['time']=('time',),np.array( [start-pad,stop+pad] )
        ds['water_level']=('time',),np.array([self.z,self.z])

        # just write a single node
        self.write_tim(ds['water_level'])

class FlowBC(BC):
    dredge_depth=-1.0
    Q=None

    def __init__(self,model,Q,**kw):
        super(FlowBC,self).__init__(model,**kw)
        self.Q=Q

    def filename_base(self):
        return super(FlowBC,self).filename_base()+"_Q"

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=dischargebnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   ""]
            fp.write("\n".join(lines))

    def write_pli(self):
        super(FlowBC,self).write_pli()

        # Additionally modify the grid to make sure there is a place for inflow to
        # come in.
        log.info("Dredging grid for flow BC %s"%self.name)
        dfm_grid.dredge_boundary(self.model.grid,
                                 np.array(self.geom.coords),
                                 self.dredge_depth)

    def write_data(self):
        ref_date,run_start,run_stop=self.model.mdu.time_range()

        pad=np.timedelta64(24,'h')

        ds=xr.Dataset()
        ds['time']=('time',),np.array( [run_start-pad,run_stop+pad] )
        ds['flow']=('time',),np.array( [self.Q,self.Q] )

        self.write_tim(ds['flow'])

#class ScalarBC(BC):
#    def __init__(self,name,value):
#        """
#        name: 'salinity','temperature', other
#        value: floating point
#        """
#        self.name=name
#        self.value=value
#    def write(self,*a,**kw):
#        # Base implementation does nothing
#        pass


#class NoaaTides(BC):
#    datum=None
#    def __init__(self,station,datum=None,z_offset=0.0):
#        self.station=station
#        self.datum=datum
#        self.z_offset=z_offset
#    def write(self,mdu,feature,grid):
#        print("Writing feature: %s"%(feature['name']))
#
#        name=feature['name']
#        old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )
#
#        for var_name in self.var_names:
#            if feature['geom'].type=='LineString':
#                pli_data=[ (name, np.array(feature['geom'].coords)) ]
#                base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,var_name))
#                pli_fn=base_fn+'.pli'
#                dio.write_pli(pli_fn,pli_data)
#
#                if var_name=='ssh':
#                    quant='waterlevelbnd'
#                else:
#                    assert False
#
#                with open(old_bc_fn,'at') as fp:
#                    lines=["QUANTITY=%s"%quant,
#                           "FILENAME=%s_%s.pli"%(name,var_name),
#                           "FILETYPE=9",
#                           "METHOD=3",
#                           "OPERAND=O",
#                           ""]
#                    fp.write("\n".join(lines))
#
#                self.write_data(mdu,feature,var_name,base_fn)
#            else:
#                assert False
#
#    def write_data(self,mdu,feature,var_name,base_fn):
#        tides=noaa_coops.coops_dataset_product(self.station,'water_level',
#                                               mdu.time_range()[1],mdu.time_range()[2],
#                                               days_per_request='M',cache_dir=cache_dir)
#        tide=tides.isel(station=0)
#        water_level=utils.fill_tidal_data(tide.water_level) + self.z_offset
#        # IIR butterworth.  Nicer than FIR, with minor artifacts at ends
#        # 3 hours, defaults to 4th order.
#        water_level[:] = filters.lowpass(water_level[:].values,
#                                         utils.to_dnum(water_level.time),
#                                         cutoff=3./24)
#
#        ref_date=mdu.time_range()[0]
#        elapsed_minutes=(tide.time.values - ref_date)/np.timedelta64(60,'s')
#
#        # just write a single node
#        tim_fn=base_fn + "_0001.tim"
#        data=np.c_[elapsed_minutes,water_level]
#        np.savetxt(tim_fn,data)
#
#class DischargeBC(FlowBC):
#    """
#    Similar to Storm, but implement as mass source, not a flow BC
#    """
#    def __init__(self,*a,**kw):
#        self.salinity=kw.pop('salinity',None)
#        self.temperature=kw.pop('temperature',None)
#
#        super(Discharge,self).__init__(*a,**kw)
#
#    def write(self,mdu,feature,grid):
#        # obvious copy and paste from above.
#        # not quite ready to abstract, though
#        print("Writing feature: %s"%(feature['name']))
#
#        name=feature['name']
#        old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )
#
#        assert feature['geom'].type=='LineString'
#
#        pli_data=[ (name, np.array(feature['geom'].coords)) ]
#        base_fn=os.path.join(mdu.base_path,"%s"%(name))
#        pli_fn=base_fn+'.pli'
#        dio.write_pli(pli_fn,pli_data)
#
#        with open(old_bc_fn,'at') as fp:
#            lines=["QUANTITY=discharge_salinity_temperature_sorsin",
#                   "FILENAME=%s"%os.path.basename(pli_fn),
#                   "FILETYPE=9",
#                   "METHOD=1",
#                   "OPERAND=O",
#                   "AREA=0 # no momentum",
#                   ""]
#            fp.write("\n".join(lines))
#
#        self.write_data(mdu,feature,base_fn)
#
#        # Really just need to dredge the first and last nodes
#        dfm_grid.dredge_discharge(grid,pli_data[0][1],self.dredge_depth)
#
#    def write_data(self,mdu,feature,base_fn):
#        ref_date,run_start,run_stop=mdu.time_range()
#
#        def h_to_td64(h):
#            # allows for decimal hours
#            return int(h*3600) * np.timedelta64(1,'s')
#
#        # trapezoid hydrograph
#        times=np.array( [run_start,
#                         run_start+h_to_td64(self.storm_start_h-1),
#                         run_start+h_to_td64(self.storm_start_h),
#                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h),
#                         run_start+h_to_td64(self.storm_start_h+self.storm_duration_h+1),
#                         run_stop+np.timedelta64(1,'D')] )
#        flows=np.array( [0.0,0.0, 
#                         self.storm_flow,self.storm_flow,0.0,0.0] )
#
#        elapsed_minutes=(times - ref_date)/np.timedelta64(60,'s')
#        items=[elapsed_minutes,flows]
#
#        if self.salinity is not None:
#            items.append(self.salinity * np.ones(len(times)))
#
#        if self.temperature is not None:
#            items.append(self.temperature * np.ones(len(times)))
#
#        # just write a single node
#        tim_fn=base_fn + ".tim"
#        data=np.c_[tuple(items)]
#        np.savetxt(tim_fn,data)

class VerticalCoord(object):
    """
    A placeholder for now, but potentially a place to describe the
    vertical coordinate structure
    """
    pass

class SigmaCoord(VerticalCoord):
    sigma_growth_factor=1


class DFlowModel(object):
    # If these are the empty string, then assumes that the executables are
    # found in existing $PATH
    dfm_bin_dir="" # .../bin  giving directory containing dflowfm
    mpi_bin_dir=None # same, but for mpiexec.  None means use dfm_bin_dir
    num_procs=1
    run_dir="." # working directory when running dflowfm
    cache_dir=None

    run_start=None
    run_stop=None

    mdu_basename='flowfm.mdu'

    mdu=None
    grid=None

    projection=None
    z_datum=None

    def __init__(self):
        self.log=log
        self.bcs=[]

    def create_with_mode(self,path,mode='create'):
        """
        path: absolute, or relative to pwd
        mode: 'create'  create the folder if it doesn't exist
         'pristine' create, and clear anything already in there
         'noclobber' create, and fail if it already exists.
        """
        if mode=='create':
            if not os.path.exists(path):
                os.makedirs(path)
        elif mode=='pristine':
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
        elif mode=='noclobber':
            assert not os.path.exists(path)

    def set_run_dir(self,path,mode='create'):
        """
        Set the working directory for the simulation.
        See create_with_mode for details on 'mode' parameter.
        set_run_dir() supports an additional mode "clean",
        which removes files known to be created during the
        script process, as opposed to 'pristine' which deletes
        everything.
        """
        self.run_dir=path
        if mode=="clean":
            self.create_with_mode(path,"create")
            self.clean_run_dir()
        else:
            self.create_with_mode(path,mode)

    def clean_run_dir(self):
        """
        Clean out most of the run dir, deleting files known to be
        created by DFlowModel
        """
        patts=['*.pli','*.tim','*.t3d','*.mdu','FlowFM.ext','*_net.nc','DFM_*', '*.dia',
               '*.xy*','initial_conditions*','dflowfm-*.log']
        for patt in patts:
            matches=glob.glob(os.path.join(self.run_dir,patt))
            for m in matches:
                if os.path.isfile(m):
                    os.unlink(m)
                elif os.path.isdir(m):
                    shutil.rmtree(m)
                else:
                    raise Exception("What is %s ?"%m)

    def set_cache_dir(self,path,mode='create'):
        """
        Set the cache directory, mainly for BC data.
        See create_with_mode for details on 'mode' parameter
        """
        self.create_with_mode(path,mode)

    def set_grid(self,grid):
        if isinstance(grid,six.string_types):
            grid=dfm_grid.DFMGrid(grid)
        self.grid=grid

    default_grid_target_filename='grid_net.nc'
    def grid_target_filename(self):
        """
        The filename, relative to self.run_dir, of the grid.  Not guaranteed
        to exist, and if no grid has been set, or the grid has no filename information,
        this will default to self.default_grid_target_filename
        """
        if self.grid is None or self.grid.filename is None:
            return self.default_grid_target_filename
        else:
            return os.path.basename(self.grid.filename)

    def load_mdu(self,fn):
        self.mdu=dio.MDUFile(fn)

    def update_mdu(self):
        """
        Update fields in the mdu object with data from self.
        """
        if self.mdu is None:
            self.mdu=dio.MDUFile()

        self.mdu.set_time_range(start=self.run_start,stop=self.run_stop)
        self.mdu.set_filename(os.path.join(self.run_dir,self.mdu_basename))

        self.mdu['geometry','NetFile'] = self.grid_target_filename()

    def write(self):
        # Make sure instance data has been pushed to the MDUFile, this
        # is used by write_forcing() and write_grid()
        self.update_mdu()
        log.info("Writing MDU to %s"%self.mdu.filename)
        self.mdu.write()
        self.write_forcing()
        # Must come after write_forcing() to allow BCs to modify grid
        self.write_grid()

    def write_grid(self):
        """
        Write self.grid to the run directory.
        Must be called after MDU is updated.  Should also be called
        after write_forcing(), since some types of BCs can update
        the grid (dredging boundaries)
        """
        dest=os.path.join(self.run_dir, self.mdu['geometry','NetFile'])
        dfm_grid.write_dfm(self.grid,dest,overwrite=True)

    def ext_force_file(self):
        return os.path.join(self.run_dir,self.mdu['external forcing','ExtForceFile'])

    def write_forcing(self,overwrite=True):
        bc_fn=self.ext_force_file()

        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)

        for bc in self.bcs:
            bc.write()

    def partition(self):
        if self.num_procs<=1:
            return

        cmd="--partition:ndomains=%d %s"%(self.num_procs,self.mdu['geometry','NetFile'])
        self.run_dflowfm(cmd)

        # similar, but for the mdu:
        gen_parallel=os.path.join(self.dfm_bin_dir,"generate_parallel_mdu.sh")
        cmd="%s %s %d 6"%(gen_parallel,os.path.basename(self.mdu.filename),self.num_procs)
        utils.call_with_path(cmd,self.run_dir)

    def run_dflowfm(self,cmd):
        # Names of the executables
        dflowfm=os.path.join(self.dfm_bin_dir,"dflowfm")

        if self.num_procs>1:
            mpi_bin_dir=self.mpi_bin_dir or self.dfm_bin_dir
            mpiexec=os.path.join(mpi_bin_dir,"mpiexec")
            real_cmd="%s -n %d %s %s"%(mpiexec,self.num_procs,dflowfm,cmd)
        else:
            real_cmd="%s %s"%(dflowfm,cmd)

        self.log.info("Running command: %s"%real_cmd)
        utils.call_with_path(real_cmd,self.run_dir)

    def run_model(self):
        self.run_dflowfm(cmd="--autostartstop %s"%os.path.basename(self.mdu.filename))

    def add_bcs_from_shp(self,forcing_shp):
        shp_data=wkb2shp.shp2geom(forcing_shp)
        self.add_bcs_from_features(shp_data)
    def add_bcs_from_features(self,shp_data):
        for feat in shp_data:
            params=dict( [ (k,feat[k]) for k in feat.dtype.names])
            bcs=self.bc_factory(params)
            self.add_bcs(bcs)

    # Default attribute for code to evaluate to generate BCs from shapefile
    # or general dictionary.
    bc_code_field='bcs'
    def bc_factory(self,params):
        """
        Create a list of BC objects based on the contents of the dictionary
        params.
        The default implementation here looks for a 'code' field, which
        will be evaluated with methods of the model available in the local
        namespace.
        keys in params will be assigned to each BC object.
        """
        assert self.bc_code_field in params,"Default implementation requires a %s attribute"%self.bc_code_field
        namespace=dict(params)
        namespace['self']=self

        for name in self.__dir__():
            obj=getattr(self,name)
            if inspect.ismethod(obj) or inspect.isfunction(obj) or isinstance(obj,BC):
                namespace[name]=obj

        # Get it to always return a list:
        code='[' + params[self.bc_code_field] + ']'
        try:
            bcs=eval(code,namespace)
        except Exception as exc:
            print("Error occurred while evaluated BC feature %s"%params)
            raise
        for bc in bcs:
            for k in params:
                setattr(bc,k,params[k])
        return bcs

    def flow_bc(self,Q,**kw):
        return FlowBC(model=self,Q=Q,**kw)
    def stage_bc(self,z,**kw):
        return StageBC(model=self,z=z,**kw)

    def add_bcs(self,bcs):
        """
        Add BC objects to this models definition.

        bcs: None (do nothing), one BC instance, or a list of BC instances
        """
        if bcs is None:
            return
        if isinstance(bcs,BC):
            self.bcs.append(bcs)
        else:
            self.bcs.extend(bcs)

    @property
    @memoize.member_thunk
    def ll_to_native(self):
        """
        Project array of longitude/latitude [...,2] to
        model-native (e.g. UTM meters)
        """
        return proj_utils.mapper('WGS84',self.projection)

    @property
    @memoize.member_thunk
    def native_to_ll(self):
        """
        Project array of x/y [...,2] coordinates in model-native
        project (e.g. UTM meters) to longitude/latitude
        """
        return proj_utils.mapper(self.projection,'WGS84')

    # Some BC methods need to know more about the domain, so DFlowModel
    # provides these accessors
    def edge_depth(self,j,datum=None):
        """
        Return the bed elevation for edge j, in meters, positive=up.
        """
        z=self.grid.nodes['depth'][ self.grid.edges['nodes'][j] ].min()
        if z>0:
            log.warning("Edge %d has positive depth %.2f"%(j,z))

        if datum is not None:
            if datum=='eta0':
                z+=float(self.mdu['geometry','WaterLevIni'])
        return z

# Functions for manipulating DFM input/output

def extract_transect(ds,line,grid=None,dx=None,cell_dim='nFlowElem',
                     include=None,rename=True,add_z=True,name=None):
    """
    Extract a transect from map output.

    ds: xarray Dataset
    line: [N,2] polyline
    grid: UnstructuredGrid instance, defaults to loading from ds, although this
      is typically much slower as the spatial index cannot be reused
    dx: sample spacing along line
    cell_dim: name of the dimension
    include: limit output to these data variables
    rename: if True, follow naming conventions in xr_transect
    """
    missing=np.nan
    assert dx is not None,"Not ready for adaptively choosing dx"
    if grid is None:
        grid=dfm_grid.DFMGrid(ds)

    from stompy.spatial import linestring_utils
    line_sampled=linestring_utils.resample_linearring(line,dx,closed_ring=False)
    N_sample=len(line_sampled)

    # Get the mapping from sample index to cell, or None if
    # the point misses the grid.
    cell_map=[ grid.select_cells_nearest( line_sampled[samp_i], inside=True)
               for samp_i in range(N_sample)]
    # to make indexing more streamlined, replace missing cells with 0, but record
    # who is missing and nan out later.  Note that this need to be None=>0, to avoid
    # changing index of 0 to something else.
    cell_mask=[ c is None for c in cell_map]
    cell_map_safe=[ c or 0 for c in cell_map]

    if include is not None:
        exclude=[ v for v in ds.data_vars if v not in include]
        ds_orig=ds
        ds=ds_orig.drop(exclude)

    new_ds=ds.isel(**{cell_dim:cell_map_safe})

    # Record the intended sampling location:
    new_ds['x_sample']=(cell_dim,),line_sampled[:,0]
    new_ds['y_sample']=(cell_dim,),line_sampled[:,1]
    distance=utils.dist_along(line_sampled)
    new_ds['d_sample']=(cell_dim,),distance
    # And some additional spatial data:
    dx_sample=utils.center_to_interval(distance)

    new_ds['dx_sample']=(cell_dim,),dx_sample
    new_ds['d_sample_bnd']=(cell_dim,'two'), np.array( [distance-dx_sample/2,
                                                        distance+dx_sample/2]).T
    new_ds=new_ds.rename({cell_dim:'sample'})

    if add_z:
        new_ds.update( xr_utils.z_from_sigma(new_ds,'ucx',interfaces=True,dz=True) )

    # need to drop variables with dimensions like nFlowLink
    to_keep_dims=set(['wdim','laydim','two','three','time','sample'])
    to_drop=[]
    for v in new_ds.variables:
        if (set(new_ds[v].dims) - to_keep_dims):
            to_drop.append(v)

    new_ds=new_ds.drop(to_drop)

    xr_utils.bundle_components(new_ds,'U',['ucx','ucy'],'xy',['N','E'])
    xr_utils.bundle_components(new_ds,'U_avg',['ucxa','ucya'],'xy',['N','E'])

    if rename:
        new_ds=new_ds.rename( {'ucx':'Ve',
                               'ucy':'Vn',
                               'ucz':'Vu',
                               'ucxa':'Ve_avg',
                               'ucya':'Vn_avg',
                               's1':'z_surf',
                               'FlowElem_bl':'z_bed',
                               'laydim':'layer'} )

    # Add metadata if missing:
    if (name is None) and ('name' not in new_ds.attrs):
        new_ds.attrs['name']='DFM Transect'
    elif name is not None:
        new_ds.attrs['name']=name
    if 'filename' not in new_ds.attrs:
        new_ds.attrs['filename']=new_ds.attrs['name']
    if 'source' not in new_ds.attrs:
        new_ds.attrs['source']=new_ds.attrs['source']

    return new_ds

class OTPSStageBC(StageBC):
    def __init__(self,model,otps_model,**kw):
        super(OTPSStageBC,self).__init__(model,z=None,**kw)
        self.otps_model=otps_model # something like OhS

    # write_config same as superclass
    # filename_base same as superclass

    def write_data(self):
        from stompy.model.otps import read_otps
        ref_date,start,stop=self.model.mdu.time_range()

        pad=2*np.timedelta64(24,'h')

        ds=xr.Dataset()

        times=np.arange( start-pad,stop+pad, 15*np.timedelta64(60,'s') )
        log.debug("Will generate tidal prediction for %d time steps"%len(times))

        ds['time']=('time',),times

        modfile=read_otps.model_path(self.otps_model)

        xy=np.array(self.geom.coords)
        ll=self.model.native_to_ll(xy)

        pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=ll[:,0],lat=ll[:,1],
                                                 time=times)

        # Here - we'll query OTPS for this
        ds['water_level']=('time',),pred_h[:,0]

        self.write_tim(ds['water_level'])

class VelocityBC(BC):
    """
    BC setting edge-normal velocity (velocitybnd), uniform in the vertical.
    positive is into the domain.
    """
    # write a velocitybnd BC
    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=velocitybnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   ""]
            fp.write("\n".join(lines))
    def filename_base(self):
        """
        Make it clear in the filenames what is being forced
        """
        return super(VelocityBC,self).filename_base()+"_vel"
    def write_data(self):
        raise Exception("Implement write_data() in subclass")
    def get_inward_normal(self):
        """
        Query the grid based on self.grid_edge to find the unit
        normal vector for this velocity BC, positive pointing into
        the domain.
        """
        assert self.grid_edge is not None
        norm=self.model.grid.edges_normals(self.grid_edge,force_inward=True)
        return norm
    def get_depth(self):
        """
        Estimate the water column depth associated with this BC.
        This is currently limited to a constant value, calculated for
        self.grid_edge.
        For the purposes here, this is a strictly positive quantity.
        """
        assert self.grid_edge is not None
        # This feels like it should be somewhere else, maybe in DFlowModel?
        h=-self.model.edge_depth(self.grid_edge,datum='eta0')
        if h<=0:
            log.warning("Depth for velocity BC is %f, should be >0"%h)
        return h

class OTPSVelocityBC(VelocityBC):
    """
    Force 2D transport based on depth-integrated transport from OTPS.
    """
    # water columns shallower than this will have a velocity calculated
    # based on this water column depth rather than their actual value.
    min_h=5.0
    def __init__(self,model,otps_model,**kw):
        super(OTPSVelocityBC,self).__init__(model,**kw)
        self.otps_model=otps_model # something like OhS or wc

    def transport_ds(self):
        from stompy.model.otps import read_otps
        ref_date,start,stop=self.model.mdu.time_range()
        pad=2*np.timedelta64(24,'h')
        ds=xr.Dataset()

        times=np.arange( start-pad,stop+pad, 15*np.timedelta64(60,'s') )
        log.debug("Will generate tidal prediction for %d time steps"%len(times))

        ds['time']=('time',),times
        modfile=read_otps.model_path(self.otps_model)
        xy=np.array(self.geom.coords)
        ll=self.model.native_to_ll(xy)
        # read_otps returns velocities in m/s
        pred_h,pred_U,pred_V=read_otps.tide_pred(modfile,lon=ll[:,0],lat=ll[:,1],
                                                 time=times,z=1)
        # mean() is goofy - it's because xy is actually an array of points,
        ds['h']=('time',),pred_h.mean(axis=1)
        ds['U']=('time',),pred_U.mean(axis=1)
        ds['V']=('time',),pred_V.mean(axis=1)
        return ds

    def velocity_ds(self):
        """
        Return time series of normal velocity in 'unorm'
        variable in a xr.dataset.
        """
        ds=self.transport_ds()
        UV=np.c_[ ds.U.values, ds.V.values ]
        assert self.grid_edge is not None,"Normal velocity BC from OTPS requires grid_edge"

        norm=self.get_inward_normal()
        h=self.get_depth() # not the water surface elevation ds.h, but surf-bed depth

        # clip h here to avoid anything too crazy
        unorm=(UV*norm).sum(axis=1) / max(h,self.min_h)
        ds['unorm']=('time',),unorm
        return ds

    def write_data(self):
        ds=self.velocity_ds()
        da=ds['unorm']
        if 'z' in da.dims:
            self.write_t3d(da,z_bed=self.model.edge_depth(self.grid_edge))
        else:
            self.write_tim(da)

class OTPSVelocity3DBC(OTPSVelocityBC):
    """
    Force 3D transport based on depth-integrated transport from OTPS.
    This is a temporary shim to test setting a 3D velocity BC.
    """
    def velocity_ds(self):
        ds=super(OTPSVelocity3DBC,self).velocity_ds()

        # so there is a 'unorm'
        z_bed=self.model.edge_depth(self.grid_edge)
        z_surf=1.0

        assert z_bed<0

        # pad out a bit above/below
        ds['z']=('z',), np.array([z_bed-10,z_surf+10])

        new_unorm,_=xr.broadcast(ds.unorm,ds.z)
        ds['unorm']=new_unorm

        import pdb
        pdb.set_trace()

        return ds


class MultiBC(BC):
    """
    Break up a boundary condition spec into per-edge boundary conditions.
    Hoping that this can be done in a mostly opaque way, without exposing to
    the caller that one BC is being broken up into many.
    """
    def __init__(self,model,cls,**kw):
        self.saved_kw=kw
        super(MultiBC,self).__init__(model,**kw)
        self.cls=cls
        self.sub_bcs="not yet!" # not populated until self.write()

    def filename_base(self):
        assert False,'This should never be called, right?'

    def write(self):
        # delay enumeration until now, so we have the most up-to-date
        # information about the model, grid, etc.
        self.enumerate_sub_bcs()

        for sub_bc in self.sub_bcs:
            sub_bc.write()

    def enumerate_sub_bcs(self):
        # dredge_grid already has some of the machinery
        grid=self.model.grid

        edges=dfm_grid.polyline_to_boundary_edges(grid,np.array(self.geom.coords))

        self.model.log.info("MultiBC will be applied over %d edges"%len(edges))

        self.sub_bcs=[]

        for j in edges:
            seg=grid.nodes['x'][ grid.edges['nodes'][j] ]
            sub_geom=geometry.LineString(seg)
            # This slightly breaks the abstraction -- in theory, the caller
            # can edit all of self's values up until write() is called, yet
            # here we are grabbing the values at time of instantiation of self.
            # hopefully it doesn't matter, especially since geom and model
            # are handled explicitly.
            sub_kw=dict(self.saved_kw) # copy original
            sub_kw['geom']=sub_geom
            sub_kw['name']="%s%04d"%(self.name,j)
            sub_kw['grid_edge']=j

            sub_bc=self.cls(model=self.model,**sub_kw)
            self.sub_bcs.append(sub_bc)



