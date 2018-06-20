import os
import inspect
import six

import numpy as np
import xarray as xr

import stompy.model.delft.io as dio

import logging as log

from stompy.io.local import noaa_coops
from stompy import utils, filters
from stompy.spatial import wkb2shp
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


class DFlowModel(object):
    dfm_bin_dir=None # .../bin  giving directory containing dflowfm
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
        See create_with_mode for details on 'mode' parameter
        """
        self.run_dir=path
        self.create_with_mode(path,mode)

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
        if self.nprocs<=1:
            return
        self.mdu.partition(self.nprocs,dfm_bin_dir=self.dfm_bin_dir)

    def add_bcs_from_shp(self,forcing_shp):
        shp_data=wkb2shp.shp2geom(forcing_shp)

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

# class Scalar(gen_bc.Scalar):
#     def write_(self,model,feature,grid):
#         print("Writing feature: %s"%(feature['name']))
# 
#         name=feature['name']
#         old_bc_fn=mdu.filepath( ['external forcing','ExtForceFile'] )
# 
#         assert feature['geom'].type=='LineString'
#         pli_data=[ (name, np.array(feature['geom'].coords)) ]
#         base_fn=os.path.join(mdu.base_path,"%s_%s"%(name,self.var_name))
#         pli_fn=base_fn+'.pli'
#         dio.write_pli(pli_fn,pli_data)
# 
#         if self.var_name=='salinity':
#             quant='salinitybnd'
#         elif self.var_name=='temperature':
#             quant='temperaturebnd'
#         else:
#             assert False
# 
#         with open(old_bc_fn,'at') as fp:
#             lines=["QUANTITY=%s"%quant,
#                    "FILENAME=%s_%s.pli"%(name,self.var_name),
#                    "FILETYPE=9",
#                    "METHOD=3",
#                    "OPERAND=O",
#                    ""]
#             fp.write("\n".join(lines))
# 
#         self.write_data(mdu,feature,self.var_name,base_fn)
# 
#     def write_data(self,mdu,feature,var_name,base_fn):
#         ref_date,start_date,end_date = mdu.time_range()
#         period=np.array([start_date,end_date])
#         elapsed_minutes=(period - ref_date)/np.timedelta64(60,'s')

#
#         # just write a single node
#         tim_fn=base_fn + "_0001.tim"
#         with open(tim_fn,'wt') as fp:
#             for t in elapsed_minutes:
#                 fp.write("%g %g\n"%(t,self.value))
#     
