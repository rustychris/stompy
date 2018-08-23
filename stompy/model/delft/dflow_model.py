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
import logging
log=logging.getLogger('DFlowModel')

import copy

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
    _geom=None
    # not sure if I'll keep these -- may be better to query at time of use
    grid_edge=None
    grid_cell=None

    def __init__(self,name,model=None,**kw):
        """
        Create boundary condition object.  Note that no work should be done
        here, as the rest of the model data is not yet in place, and this
        instance does not even have access yet to its geometry or other
        shapefile attributes.  model should either be passed in, or assigned
        immediately by caller, since most later steps rely on access to a model
        object.
        """
        self.model=model # may be None!
        self.name=name
        if 'geom' in kw:
            kw['_geom']=kw['geom']

        for k in kw:
            try:
                getattr(self,k)
            except AttributeError:
                raise Exception("Setting attribute %s failed because it doesn't exist on %s"%(k,self))
            self.__dict__[k]=kw[k]

    # A little goofy - the goal is to make geometry lazily
    # fetched against the model gazetteer, but it makes
    # get/set operations awkward
    @property
    def geom(self):
        if (self._geom is None):
            self._geom=self.model.get_geometry(name=self.name)
        return self._geom
    @geom.setter
    def set_geom(self,g):
        self._geom=g


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
        dt=np.timedelta64(60,'s') # always minutes
        # self.model.mdu.t_unit_td64()
        elapsed_time=(da.time.values - ref_date)/dt 

        data=np.c_[elapsed_time,da.values]
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
        # used to allow it to go slightly beyond, but
        # in trying to diagnose a 3D profile run in 52184, limit
        # to exactly 0,1
        # well, maybe that's not necessary -- before trying to do any resampling
        # here, maybe go ahead and let it span too far
        bed_samples=np.nonzero(sigma<=0)[0]
        surf_samples=np.nonzero(sigma>=1.0)[0]
        slc=slice(bed_samples[-1],surf_samples[0]+1)
        da=da.isel(z=slc)
        sigma=sigma[slc]
        sigma[0]=0.0 # min(0.0,sigma[0])
        sigma[-1]=1.0 # max(1.0,sigma[-1])

        assert np.all(np.diff(sigma)>0),"Need more sophisticated treatment of sigma in t3d file"
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

        # NB: this is independent of the TUnit setting in the MDU, because
        # it is written out in the file (see below).
        elapsed_minutes=(da.time.values - ref_date)/np.timedelta64(60,'s')

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

    def as_data_array(self,data,quantity='value'):
        """
        Convert several types into a consistent DataArray ready to be written
        data:
        dataarray => no change
        dataset => pull just the data variable, either based on quantity, or if there
          is a single data variable that is not a coordinate, use that.
        constant => create a two-point timeseries
        """
        if isinstance(data,xr.DataArray):
            return data
        elif isinstance(data,xr.Dataset):
            if len(data.data_vars)==1:
                return data[data.data_vars[0]]
            else:
                raise Exception("Dataset has multiple data variables -- not sure which to use: %s"%( str(data.data_vars) ))
        elif isinstance(data,(np.integer,np.floating,int,float)):
            # handles expanding a constant to the length of the run
            ds=xr.Dataset()
            pad=np.timedelta64(24,'h')
            ds['time']=('time',),np.array( [self.model.run_start-pad,self.model.run_stop+pad] )
            ds[quantity]=('time',),np.array( [data,data] )
            return ds[quantity]
        else:
            raise Exception("Not sure how to cast %s to be a DataArray"%data)

class RoughnessBC(BC):
    shapefile=None
    def __init__(self,shapefile=None,**kw):
        if 'name' not in kw:
            kw['name']='roughness'

        super(RoughnessBC,self).__init__(**kw)
        self.shapefile=shapefile
    def write_config(self):
        with open(self.model.ext_force_file(),'at') as fp:
            lines=["QUANTITY=frictioncoefficient",
                   "FILENAME=%s"%self.xyz_filename(),
                   "FILETYPE=7",
                   "METHOD=4",
                   "OPERAND=O",
                   "\n"
                   ]
            fp.write("\n".join(lines))

    def xyz_filename(self):
        return self.filename_base()+".xyz"

    def data(self):
        assert self.shapefile is not None,"Currently only support shapefile input for roughness"
        shp_data=wkb2shp.shp2geom(self.shapefile)
        coords=np.array( [np.array(pnt) for pnt in shp_data['geom'] ] )
        n=shp_data['n']
        xyz=np.c_[coords,n]
        return xyz

    def write_data(self):
        data_fn=os.path.join(self.model.run_dir,self.xyz_filename())
        xyz=self.data()
        np.savetxt(data_fn,xyz)

class StageBC(BC):
    # If other than None, can compare to make sure it's the same as the model
    # datum.
    datum=None

    def __init__(self,z=None,**kw):
        super(StageBC,self).__init__(**kw)
        self.z=z

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=waterlevelbnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

    def filename_base(self):
        """
        Make it clear in the filenames what is being forced
        """
        return super(StageBC,self).filename_base()+"_ssh"

    def dataarray(self):
        return self.as_data_array(self.z)

    def write_data(self):
        # just write a single node
        self.write_tim(self.dataarray())

class FlowBC(BC):
    dredge_depth=-1.0
    Q=None

    def __init__(self,Q=None,**kw):
        super(FlowBC,self).__init__(**kw)
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
                   "\n"]
            fp.write("\n".join(lines))

    def write_pli(self):
        super(FlowBC,self).write_pli()

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for flow BC %s"%self.name)
            dfm_grid.dredge_boundary(self.model.grid,
                                     np.array(self.geom.coords),
                                     self.dredge_depth)
        else:
            log.info("Dredging disabled")

    def dataarray(self):
        # probably need some refactoring here...
        return self.as_data_array(self.Q)

    def write_data(self):
        self.write_tim(self.dataarray())



class SourceSinkBC(BC):
    # The grid, at the entry point, will be taken down to this elevation
    # to ensure that prescribed flows are not prevented due to a dry cell.
    dredge_depth=-1.0
    def __init__(self,Q=None,**kw):
        """
        Q: one of:
          a constant value in m3/s
          an xarray DataArray with a time index.
        """
        super(SourceSinkBC,self).__init__(**kw)
        self.Q=Q

    def filename_base(self):
        return super(SourceSinkBC,self).filename_base()+"_Q"

    def write_config(self):
        assert self.Q is not None

        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=discharge_salinity_temperature_sorsin",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=1", # how is this different than method=3?
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

    def write_pli(self):
        super(SourceSinkBC,self).write_pli()

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for flow BC %s"%self.name)
            dfm_grid.dredge_discharge(self.model.grid,
                                      np.array(self.geom.coords),
                                      self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_data(self):
        assert self.Q is not None

        da=self.as_data_array(self.Q)
        self.write_tim(da)
    def dataarray(self):
        return self.as_data_array(self.Q)

class WindBC(BC):
    wind=None
    def __init__(self,**kw):
        if 'name' not in kw:
            # commonly applied globally, so may not have a geographic name
            kw['name']='wind'
        super(WindBC,self).__init__(**kw)
    def write_pli(self):
        assert self.geom is None,"Spatially limited wind not yet supported"
        return # nothing to do

    def default_tim_fn(self):
        # different than super class because typically no nodes
        return os.path.join(self.model.run_dir,self.filename_base() + ".tim")

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=windxy",
                   "FILENAME=%s.tim"%self.filename_base(),
                   "FILETYPE=2",
                   "METHOD=1",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))
    def write_data(self):
        assert self.wind is not None
        da=self.as_data_array(self.wind)
        self.write_tim(da)
    def dataarray(self):
        return self.as_data_array(self.wind)

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
#        elapsed_minutes=(tide.time.values - ref_date)/BADnp.timedelta64(60,'s')BAD
#
#        # just write a single node
#        tim_fn=base_fn + "_0001.tim"
#        data=np.c_[elapsed_minutes,water_level]
#        np.savetxt(tim_fn,data)

class VerticalCoord(object):
    """
    A placeholder for now, but potentially a place to describe the
    vertical coordinate structure
    """
    pass

class SigmaCoord(VerticalCoord):
    sigma_growth_factor=1


class HydroModel(object):
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
        self.extra_files=[]
        self.gazetteers=[]

    def add_extra_file(self,path,copy=True):
        self.extra_files.append( (path,copy) )

    def write_extra_files(self):
        for f in self.extra_files:
            path,copy = f
            if copy:
                tgt=os.path.join( self.run_dir, os.path.basename(path))
                if not (os.path.exists(tgt) and os.path.samefile(tgt,path)):
                    shutil.copyfile(path,tgt)
                else:
                    log.info("Extra file %s points to the target.  No-op"%path)

    def copy(self,deep=True):
        """
        Make a copy of this model instance.
        """
        # Starting point is just python deepcopy, but can customize
        # as needed.
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['log']: # shallow for some object
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

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

    def write(self):
        # Make sure instance data has been pushed to the MDUFile, this
        # is used by write_forcing() and write_grid()
        self.update_config()
        log.info("Writing MDU to %s"%self.mdu.filename)
        self.write_config()
        self.write_extra_files()
        self.write_forcing()
        # Must come after write_forcing() to allow BCs to modify grid
        self.write_grid()

    def write_grid(self):
        raise Exception("Implement in subclass")
    def write_forcing(self):
        for bc in self.bcs:
            self.write_bc(bc)

    def write_bc(self,bc):
        if isinstance(bc,MultiBC):
            bc.enumerate_sub_bcs()
            for sub_bc in bc.sub_bcs:
                self.write_bc(sub_bc)
        else:
            raise Exception("BC type %s not handled by class %s"%(bc.__class__,self.__class__))

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

    def add_gazetteer(self,shp_fn):
        """
        Register a shapefile for resolving feature locations
        """
        self.gazetteers.append(wkb2shp.shp2geom(shp_fn))
    def get_geometry(self,**kws):
        """
        The gazetteer interface for BC geometry.  given criteria as keyword arguments,
        i.e. name='Old_River', return the matching geometry from the gazetteer as
        a shapely geometry.
        if no match, return None.  Error if more than one match
        """
        hits=self.match_gazetteer(**kws)
        if hits:
            assert len(hits)==1
            return hits[0]['geom']
        else:
            return None
    def match_gazetteer(self,**kws):
        """
        search all gazetteers with criteria specified in keyword arguments,
        returning a list of shapefile records (note that this is a python
        list of numpy records, not a numpy array, since shapefiles may not
        have the same fields).
        return empty list if not hits
        """
        hits=[]
        for gaz in self.gazetteers:
            for idx in range(len(gaz)):
                if self.match_feature(kws,gaz[idx]):
                    hits.append( gaz[idx] )
        return hits
    def match_feature(self,kws,feat):
        """
        check the critera in dict kws against feat, a numpy record as
        returned by shp2geom.
        """
        for k in kws:
            try:
                if feat[k] == kws[k]:
                    continue
                else:
                    return False
            except KeyError:
                return False
        return True

    # some read/write methods which may have to refer to model state to properly
    # parse inputs.
    def read_bc(self,fn):
        """
        Read a new-style BC file into an xarray dataset
        """
        return dio.read_dfm_bc(fn)

    def read_tim(self,fn,time_unit=None,columns=['val1','val2','val3']):
        """
        Parse a tim file to xarray Dataset.  This needs to be a model method so
        that we know the units, and reference date.  Currently, this immediately
        reads the file, which may have to change in the future for performance
        or ease-of-use reasons.

        time_unit: 'S' for seconds, 'M' for minutes.  Relative to model reference
        time.

        returns Dataset with 'time' dimension, and data columns labeled according
        to columns.
        """
        if time_unit is None:
            # time_unit=self.mdu['time','Tunit']
            # always minutes, unless overridden by caller
            time_unit='M'

        ref_time,_,_ = self.mdu.time_range()
        return dio.read_dfm_tim(fn,time_unit=time_unit,
                                ref_time=ref_time,
                                columns=columns)

    def add_FlowBC(self,**kw):
        self.add_bcs(FlowBC(model=self,**kw))
    def add_SourceSinkBC(self,*a,**kw):
        self.add_bcs(SourceSinkBC(*a,model=self,**kw))
    def add_StageBC(self,**kw):
        self.add_bcs(StageBC(model=self,**kw))
    def add_WindBC(self,**kw):
        self.add_bcs(WindBC(model=self,**kw))
    def add_RoughnessBC(self,**kw):
        self.add_bcs(RoughnessBC(model=self,**kw))

    def add_bcs(self,bcs):
        """
        Add BC objects to this models definition.

        bcs: None (do nothing), one BC instance, or a list of BC instances
        """
        if bcs is None:
            return
        if isinstance(bcs,BC):
            bcs=[bcs]
        for bc in bcs:
            assert (bc.model is None) or (bc.model==self),"Not expecting to share BC objects"
            bc.model=self
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
    def __init__(self,otps_model,**kw):
        super(OTPSStageBC,self).__init__(**kw)
        self.otps_model=otps_model # something like OhS

    # write_config same as superclass
    # filename_base same as superclass
    def dataset(self):
        from stompy.model.otps import read_otps

        pad=2*np.timedelta64(24,'h')
        ds=xr.Dataset()
        times=np.arange( self.model.run_start-pad,
                         self.model.run_stop+pad,
                         15*np.timedelta64(60,'s') )
        log.debug("Will generate tidal prediction for %d time steps"%len(times))
        ds['time']=('time',),times
        modfile=read_otps.model_path(self.otps_model)

        xy=np.array(self.geom.coords)
        ll=self.model.native_to_ll(xy)

        pred_h,pred_u,pred_v=read_otps.tide_pred(modfile,lon=ll[:,0],lat=ll[:,1],
                                                 time=times)
        # Here - we'll query OTPS for this
        ds['water_level']=('time',),pred_h[:,0]
        return ds
    def dataarray(self):
        return self.dataset()['water_level']

    def write_data(self): # DFM IMPLEMENTATION!
        self.write_tim(self.dataarray())

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
                   "\n"]
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
        # and try populating more levels, in case things are getting chopped off
        N=10
        z_pad=10.0
        ds['z']=('z',), np.linspace(z_bed-z_pad,z_surf+z_pad,N)
        sig=np.linspace(-1,1,N)

        new_unorm,_=xr.broadcast(ds.unorm,ds.z)
        ds['unorm']=new_unorm

        # Add some vertical structure to test 3D nature of the BC
        delta=xr.DataArray(0.02*sig,dims=['z'])
        ds['unorm'] = ds.unorm + delta

        return ds


class MultiBC(BC):
    """
    Break up a boundary condition spec into per-edge boundary conditions.
    Hoping that this can be done in a mostly opaque way, without exposing to
    the caller that one BC is being broken up into many.
    """
    def __init__(self,cls,**kw):
        self.saved_kw=dict(kw) # copy
        # These are all passed on to the subclass, but only the
        # known parameters are kept for MultiBC.
        # if at some we need to pass parameters only to MultiBC, but
        # not to the subclass, this would have to check both ways.
        keys=list(kw.keys())
        for k in keys:
            try:
                getattr(self,k)
            except AttributeError:
                del kw[k]

        super(MultiBC,self).__init__(**kw)
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
            j_cells=grid.edge_to_cells(j)
            assert j_cells.min()<0
            assert j_cells.max()>=0
            sub_kw['grid_cell']=j_cells.max()

            assert self.model is not None,"Why would that be?"
            assert sub_geom is not None,"Huh?"

            sub_bc=self.cls(model=self.model,**sub_kw)
            self.sub_bcs.append(sub_bc)


class DFlowModel(HydroModel):
    # flow and source/sink BCs will get the adjacent nodes dredged
    # down to this depth in order to ensure the impose flow doesn't
    # get blocked by a dry edge. Set to None to disable.
    dredge_depth=-1.0


    def write_forcing(self,overwrite=True):
        bc_fn=self.ext_force_file()
        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)
        super(DFlowModel,self).write_forcing()

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

    def load_mdu(self,fn):
        self.mdu=dio.MDUFile(fn)

    def update_config(self):
        """
        Update fields in the mdu object with data from self.
        """
        if self.mdu is None:
            self.mdu=dio.MDUFile()

        self.mdu.set_time_range(start=self.run_start,stop=self.run_stop)
        self.mdu.set_filename(os.path.join(self.run_dir,self.mdu_basename))

        self.mdu['geometry','NetFile'] = self.grid_target_filename()

    def write_config(self):
        # Assumes update_config() already called
        self.mdu.write()

    def write_bc(self,bc):
        if isinstance(bc,StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,SourceSinkBC):
            self.write_source_bc(bc)
        elif isinstance(bc,WindBC):
            self.write_wind_bc(bc)
        elif isinstance(bc,RoughnessBC):
            self.write_roughness_bc(bc)
        else:
            super(DFlowModel,self).write_bc(bc)

    def write_tim(self,da,file_path):
        """
        Write a DFM tim file based on the timeseries in the DataArray.
        da must have a time dimension.  No support yet for vector-values here.
        file_path is relative to the working directory of the script, not
        the run_dir.
        """
        if len(da.dims)==0:
            raise Exception("Not implemented for constant waterlevel...")
        ref_date,start,stop = self.mdu.time_range()
        dt=np.timedelta64(60,'s') # always minutes
        elapsed_time=(da.time.values - ref_date)/dt 
        data=np.c_[elapsed_time,da.values]

        np.savetxt(file_path,data)

    def write_stage_bc(self,bc):
        self.write_gen_bc(bc,quantity='stage')

    def write_flow_bc(self,bc):
        self.write_gen_bc(bc,quantity='flow')

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            dfm_grid.dredge_boundary(self.grid,
                                     np.array(bc.geom.coords),
                                     self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_source_bc(self,bc):
        self.write_gen_bc(bc,quantity='source')

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            dfm_grid.dredge_discharge(self.grid,
                                      np.array(bc.geom.coords),
                                      self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_gen_bc(self,bc,quantity):
        """
        handle the actual work of writing flow and stage BCs.
        quantity: 'stage','flow','source'
        """
        bc_id=bc.name+"_" + quantity

        #self.write_pli()
        assert bc.geom.type=='LineString'
        pli_data=[ (bc_id, np.array(bc.geom.coords)) ]
        pli_fn=bc_id+'.pli'
        dio.write_pli(os.path.join(self.run_dir,pli_fn),pli_data)

        #self.write_config()
        with open(self.ext_force_file(),'at') as fp:
            lines=[]
            method=3 # default
            if quantity=='stage':
                lines.append("QUANTITY=waterlevelbnd")
            elif quantity=='flow':
                lines.append("QUANTITY=dischargebnd")
            elif quantity=='source':
                lines.append("QUANTITY=discharge_salinity_temperature_sorsin")
                method=1 # not sure how this is different
            else:
                assert False
            lines+=["FILENAME=%s"%pli_fn,
                    "FILETYPE=9",
                    "METHOD=%d"%method,
                    "OPERAND=O",
                    ""]
            fp.write("\n".join(lines))

        #self.write_data()
        da=bc.dataarray()
        assert len(da.dims)<=1,"Only ready for dimensions of time or none"
        tim_path=os.path.join(self.run_dir,bc_id+"_0001.tim")
        self.write_tim(da,tim_path)

    def write_wind_bc(self,bc):
        assert bc.geom is None,"Spatially limited wind not yet supported"

        tim_fn=bc.name+".tim"
        tim_path=os.path.join(self.run_dir,tim_fn)

        # write_config()
        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=windxy",
                   "FILENAME=%s"%tim_fn,
                   "FILETYPE=2",
                   "METHOD=1",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

        # write_data()
        self.write_tim(bc.dataarray(),tim_path)

    def write_roughness_bc(self,bc):
        # write_config()
        xyz_fn=bc.name+".xyz"
        xyz_path=os.path.join(self.run_dir,xyz_fn)

        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=frictioncoefficient",
                   "FILENAME=%s"%xyz_fn,
                   "FILETYPE=7",
                   "METHOD=4",
                   "OPERAND=O",
                   "\n"
                   ]
            fp.write("\n".join(lines))

        # write_data()
        np.savetxt(xyz_path,bc.data())

