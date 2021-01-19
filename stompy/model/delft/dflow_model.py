"""
Automate parts of setting up a DFlow hydro model.

TODO:
  allow for setting grid bathy from the model instance
"""
import os,shutil,glob,inspect
import six
import logging
log=logging.getLogger('DFlowModel')

import copy

import numpy as np
import xarray as xr
import pandas as pd
from shapely import geometry

import stompy.model.delft.io as dio
from stompy import xr_utils
from stompy.io.local import noaa_coops, hycom
from stompy import utils, filters, memoize
from stompy.spatial import wkb2shp, proj_utils
from stompy.model.delft import dfm_grid
import stompy.grid.unstructured_grid as ugrid

from . import io as dio
from .. import hydro_model as hm

class DFlowModel(hm.HydroModel,hm.MpiModel):
    # If these are the empty string, then assumes that the executables are
    # found in existing $PATH
    dfm_bin_dir="" # .../bin  giving directory containing dflowfm
    dfm_bin_exe='dflowfm'
    
    mdu_basename='flowfm.mdu'
    
    ref_date=None
    mdu=None
    # If set, a DFlowModel instance which will be continued
    restart_from=None
    
    # flow and source/sink BCs will get the adjacent nodes dredged
    # down to this depth in order to ensure the impose flow doesn't
    # get blocked by a dry edge. Set to None to disable.
    # This has moved to just the BC objects, and removed here to avoid
    # confusion.
    # dredge_depth=-1.0

    def __init__(self,*a,**kw):
        super(DFlowModel,self).__init__(*a,**kw)
        self.structures=[]
        self.load_default_mdu()

        if self.restart_from is not None:
            self.set_restart_from(self.restart_from)
        
    def load_default_mdu(self):
        """
        Load a default set of config values from data/defaults-r53925.mdu
        """
        # This is copied straight from the source distribution
        fn=os.path.join(os.path.dirname(__file__),"data","defaults-r53925.mdu")
        self.load_mdu(fn)
        
        # And some extra settings to make it compatible with this script
        self.mdu['external forcing','ExtForceFile']='FlowFM.ext'

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
        
    def write_forcing(self,overwrite=True):
        bc_fn=self.ext_force_file()
        assert bc_fn,"DFM script requires old-style BC file.  Set [external forcing] ExtForceFile"
        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)
        utils.touch(bc_fn)
        super(DFlowModel,self).write_forcing()

    def set_grid(self,grid):
        super(DFlowModel,self).set_grid(grid)

        # Specific to d-flow -- see if it's necessary to copy node-based depth
        # to node_z_bed.
        # Used to be that 'depth' was used as a node field, and it was implicitly
        # positive-up.  trying to shift away from 'depth' being a positive-up
        # quantity, and instead use 'z_bed' and specifically 'node_z_bed'
        # for a node-centered, positive-up bathymetry value.
        node_fields=self.grid.nodes.dtype.names
        
        if 'node_z_bed' not in node_fields:
            if 'z_bed' in node_fields:
                self.grid.add_node_field('node_z_bed',self.grid.nodes['z_bed'])
                self.log.info("Duplicating z_bed to node_z_bed for less ambiguous naming")
            elif 'depth' in node_fields:
                self.grid.add_node_field('node_z_bed',self.grid.nodes['depth'])
                self.log.info("Duplicating depth to node_z_bed for less ambiguous naming, and assuming it was already positive-up")
        
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
            grid_fn=self.grid.filename
            if not grid_fn.endswith('_net.nc'):
                if grid_fn.endswith('.nc'):
                    grid_fn=grid_fn.replace('.nc','_net.nc')
                else:
                    grid_fn=grid_fn+"_net.nc"
            return os.path.basename(grid_fn)
        
    def dredge_boundary(self,linestring,dredge_depth):
        super(DFlowModel,self).dredge_boundary(linestring,dredge_depth,node_field='node_z_bed',
                                               edge_field=None,cell_field=None)
        
    def dredge_discharge(self,point,dredge_depth):
        super(DFlowModel,self).dredge_discharge(point,dredge_depth,node_field='node_z_bed',
                                                edge_field=None,cell_field=None)
        
    def write_grid(self):
        """
        Write self.grid to the run directory.
        Must be called after MDU is updated.  Should also be called
        after write_forcing(), since some types of BCs can update
        the grid (dredging boundaries)
        """
        dest=os.path.join(self.run_dir, self.mdu['geometry','NetFile'])
        self.grid.write_dfm(dest,overwrite=True,)

    def subdomain_grid_filename(self,proc):
        base_grid_name=self.mdu.filepath(('geometry','NetFile'))
        proc_grid_name=base_grid_name.replace('_net.nc','_%04d_net.nc'%proc)
        return proc_grid_name
    def subdomain_grid(self,proc):
        """
        For a run that has been partitioned, load the grid for a specific
        subdomain.
        """
        g=ugrid.UnstructuredGrid.read_dfm(self.subdomain_grid_filename(proc))
        return g
        
    def ext_force_file(self):
        return self.mdu.filepath(('external forcing','ExtForceFile'))

    def load_template(self,fn):
        """ more generic name for load_mdu """
        return self.load_mdu(fn) 
    def load_mdu(self,fn):
        self.mdu=dio.MDUFile(fn)

    @classmethod
    def load(cls,fn):
        """
        Populate Model instance from an existing run
        """
        fn=cls.to_mdu_fn(fn) # in case fn was a directory
        if fn is None:
            # no mdu was found
            return None
        model=DFlowModel()
        model.load_mdu(fn)
        try:
            model.grid = ugrid.UnstructuredGrid.read_dfm(model.mdu.filepath( ('geometry','NetFile') ))
        except FileNotFoundError:
            log.warning("Loading model from %s, no grid could be loaded"%fn)
            model.grid=None
        d=os.path.dirname(fn) or "."
        model.set_run_dir(d,mode='existing')
        # infer number of processors based on mdu files
        # Not terribly robust if there are other files around..
        sub_mdu=glob.glob( fn.replace('.mdu','_[0-9][0-9][0-9][0-9].mdu') )
        if len(sub_mdu)>0:
            model.num_procs=len(sub_mdu)
        else:
            # probably better to test whether it has even been processed
            model.num_procs=1

        ref,start,stop=model.mdu.time_range()
        model.ref_date=ref
        model.run_start=start
        model.run_stop=stop

        model.load_gazetteer_from_run()
        return model

    def load_gazetteer_from_run(self):
        """
        Populate gazetteers with geometry read in from an existing run.
        So far only gets stations.  Will have to come back to handle
        transects, regions, etc. and maybe even read back in BC locations,
        or query output history files.
        """
        fn=self.mdu.filepath(['output','ObsFile'])
        if os.path.exists(fn):
            stations=pd.read_csv(self.mdu.filepath(['output','ObsFile']),
                                 sep=' ',names=['x','y','name'],quotechar="'")
            stations['geom']=[geometry.Point(x,y) for x,y in stations.loc[ :, ['x','y']].values ]
            self.gazetteers.append(stations.to_records())

    @classmethod
    def to_mdu_fn(cls,path):
        """
        coerce path that is possibly a directory to a best guess
        of the MDU path.  file paths are left unchanged. returns None
        if path is a directory but no mdu files is there.
        """
        # all mdu files, regardless of case
        if not os.path.isdir(path):
            return path
        fns=[os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith('.mdu')]
        # assume shortest is the one that hasn't been partitioned
        if len(fns)==0:
            return None

        unpartitioned=np.argmin([len(f) for f in fns])
        return fns[unpartitioned]

    def close(self):
        """
        Close open file handles -- this can help on windows where
        having a file open prevents it from being deleted.
        """
        # nothing right now
        pass

    def partition(self,partition_grid=None):
        if self.num_procs<=1:
            return
        # precompiled 1.5.2 linux binaries are able to partition the mdu okay,
        # so switch to always using dflowfm to partition grid and mdu.
        # unfortunately there does not appear to be an option to only partition
        # the mdu.

        if partition_grid is None:
            partition_grid=not self.restart

        if partition_grid:
            # oddly, even on windows, dflowfm requires only forward
            # slashes in the path to the mdu (ver 1.4.4)
            # since run_dflowfm uses run_dir as the working directory
            # here we strip to the basename
            cmd=["--partition:ndomains=%d:icgsolver=6"%self.num_procs,
                 os.path.basename(self.mdu.filename)]
            self.run_dflowfm(cmd,mpi=False)
        else:
            # Copy the partitioned network files:
            for proc in range(self.num_procs):
                old_grid_fn=self.restart_from.subdomain_grid_filename(proc)
                new_grid_fn=self.subdomain_grid_filename(proc)
                print("Copying pre-partitioned grid files: %s => %s"%(old_grid_fn,new_grid_fn))
                shutil.copyfile(old_grid_fn,new_grid_fn)
                
            # not a cross platform solution!
            gen_parallel=os.path.join(self.dfm_bin_dir,"generate_parallel_mdu.sh")
            cmd=[gen_parallel,os.path.basename(self.mdu.filename),"%d"%self.num_procs,'6']
            return utils.call_with_path(cmd,self.run_dir)

    _dflowfm_exe=None
    @property
    def dflowfm_exe(self):
        if self._dflowfm_exe is None:
            p=os.path.join(self.dfm_bin_dir,self.dfm_bin_exe)
            if os.path.sep!="/":
                p=p.replace("/",os.path.sep)
            return p
        else:
            return self._dflowfm_exe
    @dflowfm_exe.setter
    def dflowfm_exe(self,v):
        self._dflowfm_exe=v

    def run_dflowfm(self,cmd,mpi='auto',wait=True):
        """
        Invoke the dflowfm executable with the list of
        arguments given in cmd=[arg1,arg2, ...]
        mpi: generally if self.num_procs>1, mpi will be used. this
          can be set to False or 0, in which case mpi will not be used
          even when num_procs is >1. This is useful for partition which
          runs single-core.

        wait: True: do not return until the command finishes.
          False: return immediately.
          For now, the backend can only support one or the other, depending
          on platform. See hydro_model.py:MpiModel for details.
        """
        if mpi=='auto':
            num_procs=self.num_procs
        else:
            num_procs=1

        if num_procs>1:
            real_cmd=( [self.dflowfm_exe] + cmd )
            return self.mpirun(real_cmd,working_dir=self.run_dir,wait=wait)
        else:
            real_cmd=[self.dflowfm_exe]+cmd

            self.log.info("Running command: %s"%(" ".join(real_cmd)))
            return utils.call_with_path(real_cmd,self.run_dir)
    
    def run_simulation(self,extra_args=[]):
        self.run_dflowfm(cmd=["-t","1","--autostartstop",
                              os.path.basename(self.mdu.filename)]
                         + extra_args)

    
    @classmethod
    def run_completed(cls,fn):
        """
        fn: path to mdu file.  will attempt to guess the right mdu if a directory
        is provided, but no guarantees.

        returns: True if the file exists and the folder contains a run which
          ran to completion. Otherwise False.
        """
        if not os.path.exists(fn):
            return False
        model=cls.load(fn)
        if model is not None:
            result=model.is_completed()
            model.close()
        else:
            result=False
        return result
    
    def is_completed(self):
        """
        return true if the model has been run.
        this can be tricky to define -- here completed is based on
        a report in a diagnostic that the run finished.
        this doesn't mean that all output files are present.
        """
        root_fn=self.mdu.filename[:-4] # drop .mdu suffix
        # Look in multiple locations for diagnostic file.
        # In older DFM, MPI runs placed it next to mdu, while
        # serial and newer DFM (>=1.6.2?) place it in
        # output folder
        dia_fns=[]
        dia_fn_base=os.path.basename(root_fn)
        if self.num_procs>1:
            dia_fn_base+='_0000.dia'
        else:
            dia_fn_base+=".dia"
            
        dia_fns.append(os.path.join(self.run_dir,dia_fn_base))
        dia_fns.append(os.path.join(self.run_dir,
                                    "DFM_OUTPUT_%s"%self.mdu.name,
                                    dia_fn_base))

        for dia_fn in dia_fns:
            assert dia_fn!=self.mdu.filename,"Probably case issues with %s"%dia_fn

            if os.path.exists(dia_fn):
                break
        else:
            return False
        
        # Read the last 1000 bytes
        with open(dia_fn,'rb') as fp:
            fp.seek(0,os.SEEK_END)
            tail_size=min(fp.tell(),1000)
            fp.seek(-tail_size,os.SEEK_CUR)
            # This may not be py2 compatible!
            tail=fp.read().decode(errors='ignore')
        return "Computation finished" in tail

    def update_config(self):
        """
        Update fields in the mdu object with data from self.
        """
        if self.mdu is None:
            self.mdu=dio.MDUFile()

        self.mdu.set_time_range(start=self.run_start,stop=self.run_stop,
                                ref_date=self.ref_date)
        self.mdu.set_filename(os.path.join(self.run_dir,self.mdu_basename))

        self.mdu['geometry','NetFile'] = self.grid_target_filename()

        # Try to allow for the caller handling observation and cross-section
        # files externally or through the interface -- to that end, don't
        # overwrite ObsFile or CrsFile, but if internally there are point/
        # line observations set, make sure that there is a filename there.
        if len(self.mon_points)>0 and not self.mdu['output','ObsFile']:
            self.mdu['output','ObsFile']="obs_points.xyn"
        if len(self.mon_sections)>0 and not self.mdu['output','CrsFile']:
            self.mdu['output','CrsFile']="obs_sections.pli"

    def write_config(self):
        # Assumes update_config() already called
        self.write_structures() # updates mdu
        self.write_monitors()
        log.info("Writing MDU to %s"%self.mdu.filename)
        self.mdu.write()

    def write_monitors(self):
        self.write_monitor_points()
        self.write_monitor_sections()

    def write_monitor_points(self):
        fn=self.mdu.filepath( ('output','ObsFile') )
        if fn is None: return
        with open(fn,'at') as fp:
            for i,mon_feat in enumerate(self.mon_points):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_pnt_%03d"%i
                xy=np.array(mon_feat['geom'])
                fp.write("%.3f %.3f '%s'\n"%(xy[0],xy[1],name))
    def write_monitor_sections(self):
        fn=self.mdu.filepath( ('output','CrsFile') )
        if fn is None: return
        with open(fn,'at') as fp:
            for i,mon_feat in enumerate(self.mon_sections):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_sec_%03d"%i
                xy=np.array(mon_feat['geom'])
                dio.write_pli(fp,[ (name,xy) ])

    def add_Structure(self,**kw):
        self.structures.append(kw)

    def write_structures(self):
        structure_file='structures.ini'
        if len(self.structures)==0:
            return

        self.mdu['geometry','StructureFile']=structure_file

        with open( self.mdu.filepath(('geometry','StructureFile')),'wt') as fp:
            for s in self.structures:
                lines=[
                    "[structure]",
                    "type         = %s"%s['type'],
                    "id           = %s"%s['name'],
                    "polylinefile = %s.pli"%s['name'],
                    "door_height  = %.3f"%s['door_height'],
                    "lower_edge_level = %.3f"%s['lower_edge_level'],
                    "opening_width = %.3f"%s['opening_width'],
                    "sill_level     = %.3f"%s['sill_level'],
                    "horizontal_opening_direction = %s"%s['horizontal_opening_direction'],
                    "\n"
                ]
                fp.write("\n".join(lines))
                pli_fn=os.path.join(self.run_dir,s['name']+'.pli')
                if 'geom' in s:
                    geom=s['geom']
                    if isinstance(geom,np.ndarray):
                        geom=geometry.LineString(geom)
                else:
                    geom=self.get_geometry(name=s['name'])
                    
                assert geom.type=='LineString'
                pli_data=[ (s['name'], np.array(geom.coords)) ]
                dio.write_pli(pli_fn,pli_data)
                
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
                
    def write_bc(self,bc):
        if isinstance(bc,hm.StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,hm.FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,hm.SourceSinkBC):
            self.write_source_bc(bc)
        elif isinstance(bc,hm.WindBC):
            self.write_wind_bc(bc)
        elif isinstance(bc,hm.RoughnessBC):
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
        ref_date,start,stop = self.mdu.time_range()
        dt=np.timedelta64(60,'s') # always minutes

        if len(da.dims)==0:
            # raise Exception("Not implemented for constant waterlevel...")
            pad=np.timedelta64(86400,'s')
            times=np.array([start-pad,stop+pad])
            values=np.array([da.values.item(),da.values.item()])
        else:
            times=da.time.values
            values=da.values
        elapsed_time=(times - ref_date)/dt
        data=np.c_[elapsed_time,values]

        np.savetxt(file_path,data)

    def write_stage_bc(self,bc):
        self.write_gen_bc(bc,quantity='stage')

    def write_flow_bc(self,bc):
        self.write_gen_bc(bc,quantity='flow')

        if (bc.dredge_depth is not None) and (self.restart_from is None):
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            self.dredge_boundary(np.array(bc.geom.coords),bc.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_source_bc(self,bc):
        self.write_gen_bc(bc,quantity='source')

        if (bc.dredge_depth is not None) and (self.restart_from is None):
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            # These are now class methods using a generic implementation in HydroModel
            # may need some tlc
            self.dredge_discharge(np.array(bc.geom.coords),bc.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_gen_bc(self,bc,quantity):
        """
        handle the actual work of writing flow and stage BCs.
        quantity: 'stage','flow','source'
        """
        # 2019-09-09 RH: the automatic suffix is a bit annoying. it is necessary
        # when adding scalars, but for any one BC, only one of stage, flow or source
        # would be present.  Try dropping the suffix here.
        bc_id=bc.name # +"_" + quantity

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
        da=bc.data()
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

        self.write_tim(bc.data(),tim_path)

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
        da=bc.data()
        xyz=np.c_[ da.x.values,
                   da.y.values,
                   da.values ]
        np.savetxt(xyz_path,xyz)

    def initial_water_level(self):
        """
        some BC methods which want a depth need an estimate of the water surface
        elevation, and the initial water level is as good a guess as any.
        """
        return float(self.mdu['geometry','WaterLevIni'])

    def map_outputs(self):
        """
        return a list of map output files
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_map.nc'))
        fns.sort()
        return fns
    
    def his_output(self):
        """
        return path to history file output
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_his.nc'))
        assert len(fns)==1
        return fns[0]

    def hyd_output(self):
        """ Path to DWAQ-format hyd file """
        return os.path.join( self.run_dir,
                             "DFM_DELWAQ_%s"%self.mdu.name,
                             "%s.hyd"%self.mdu.name )

    def restartable_time(self):
        """
        Based on restart files, what is the latest time that restart
        data exists for continuing this run?
        Returns None of no restart data was found
        """
        fns=glob.glob(os.path.join(self.mdu.output_dir(),'*_rst.nc'))
        fns.sort() # sorts both processors and restart times
        if len(fns)==0:
            return None
        
        last_rst=xr.open_dataset(fns[-1])
        rst_time=last_rst.time.values[0]
        last_rst.close()
        return rst_time
    
    def create_restart(self):
        new_model=self.__class__() # in case of subclassing, rather than DFlowModel()
        new_model.set_restart_from(self)
        return new_model

    def set_restart_from(self,model):
        """
        Pull the restart-related settings from model into the current instance.
        This is going to need tweaking. Previously it would re-use the original
        run directory, since outputs would go into a new sub-directory.  but
        that's not flexible enough for general use of restarts.
        """
        self.mdu=model.mdu.copy()
        name=model.mdu.name
        self.mdu_basename=os.path.basename( model.mdu_basename )
        self.mdu.set_filename( os.path.join(self.run_dir, self.mdu_basename) )
        self.restart=True
        self.restart_model=model
        self.ref_date=model.ref_date
        self.run_start=model.restartable_time()
        assert self.run_start is not None,"Trying to restart run that has no restart data"
        
        self.num_procs=model.num_procs
        self.grid=model.grid
        
        # Too soon to test this?
        assert self.run_dir != model.run_dir
        
        rst_base=os.path.join(model.mdu.output_dir(),
                              (model.mdu.name
                               +'_'+utils.to_datetime(self.run_start).strftime('%Y%m%d_%H%M%S')
                               +'_rst.nc'))
        # That gets rst_base relative to the cwd, but we need it relative
        # to the new runs run_dir
        self.mdu['restart','RestartFile']=os.path.relpath(rst_base,start=self.run_dir)

    def restart_inputs(self):
        """
        Return a list of paths to restart data that will be used as the 
        initial condition for this run. Assumes nonmerged style of restart data.
        """
        rst_base=self.mdu['restart','RestartFile']
        path=os.path.dirname(rst_base)
        base=os.path.basename(rst_base)
        # Assume that it has the standard naming
        suffix=base[-23:] # just the date-time portion
        rsts=[ (rst_base[:-23] + '_%04d'%p + rst_base[-23:])
               for p in range(self.num_procs)]
        return rsts
    
    def modify_restart_data(self,modify_ic):
        """
        Apply the given function to restart data, and copy the restart
        files at the same time.
        Updates self.mdu['restart','RestartFile'] to point to the new
        location, which will be the output folder for this run.

        modify_ic: fn(xr.Dataset, **kw) => None or xr.Dataset

        it should take **kw, to flexibly allow more information to be passed in
         in the future.
        """
        for proc,rst in enumerate(self.restart_inputs()):
            old_dir=os.path.dirname(rst)
            new_rst=os.path.join(self.mdu.output_dir(),os.path.basename(rst))
            assert rst!=new_rst
            ds=xr.open_dataset(rst)
            new_ds=modify_ic(ds,proc=proc,model=self)
            if new_ds is None:
                new_ds=ds # assume modified in place

            dest_dir=os.path.dirname(new_rst)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            new_ds.to_netcdf(new_rst)
        old_rst_base=self.mdu['restart','RestartFile']
        new_rst_base=os.path.join( self.mdu.output_dir(), os.path.basename(old_rst_base))
        self.mdu['restart','RestartFile']=new_rst_base
        
    def extract_section(self,name=None,chain_count=1,refresh=False,
                        xy=None,ll=None,data_vars=None):
        """
        Return xr.Dataset for monitored cross section.
        currently only supports selection by name.  may allow for 
        xy, ll in the future.

        refresh: force a close/open on the netcdf.
        """
        assert name is not None,"Currently sections can only be pulled by name"
        
        his=xr.open_dataset(self.his_output())
        if refresh:
            his.close()
            his=xr.open_dataset(self.his_output())
            
        names=his.cross_section_name.values
        try:
            names=[n.decode() for n in names]
        except AttributeError:
            pass

        if name not in names:
            print("section %s not found.  Options are:"%name)
            print(", ".join(names))
            return None

        idx=names.index(name)
        # this has a bunch of extra cruft -- some other time remove
        # the parts that are not relevant to the cross section.
        ds=his.isel(cross_section=idx)
        return self.translate_vars(ds,requested_vars=data_vars)

    def translate_vars(self,ds,requested_vars=[]):
        """
        Not sure if this is the right place to handle this sort of thing.
        Trying to deal with the fact that we'd like to request 'water_level'
        from a model, but it may be named 'eta', 'waterlevel', 'sea_surface_height',
        's1', and so on.

        The interface is going to evolve here...

        For now:
        ds: xr.Dataset, presumably from model output.
        requested_vars: if present, a list of variable names that the caller 
        wants.

        Updates ds, try to find candidates for the requested variables.
        """
        lookup={'flow':'cross_section_discharge',
                'water_level':'waterlevel'}
        for v in requested_vars:
            if v in ds: continue
            if (v in lookup) and (lookup[v] in ds):
                ds[v]=ds[ lookup[v] ]
                ds[v].attrs['history']='Copied from %s'%lookup[v]
        return ds
    
    def extract_station(self,xy=None,ll=None,name=None,refresh=False,
                        data_vars=None):
        his=xr.open_dataset(self.his_output())
        
        if refresh:
            his.close()
            his=xr.open_dataset(self.his_output())
        
        if name is not None:
            names=his.station_name.values
            try:
                names=[n.decode() for n in names]
            except AttributeError:
                pass

            if name not in names:
                return None
            idx=names.index(name)
        else:
            raise Exception("Only picking by name has been implemented for DFM output")
        
        # this has a bunch of extra cruft -- some other time remove
        # the parts that are not relevant to the station
        ds=his.isel(stations=idx)
        # When runs are underway, some time values beyond the current point in the
        # run are set to t0.  Remove those.
        non_increasing=(ds.time.values[1:] <= ds.time.values[:-1])
        if np.any(non_increasing):
            # e.g. time[1]==time[0]
            # then diff(time)[0]==0
            # nonzero gives us 0, and the correct slice is [:1]
            stop=np.nonzero(non_increasing)[0][0]
            ds=ds.isel(time=slice(None,stop+1))

        return self.translate_vars(ds,requested_vars=data_vars)

import sys
if sys.platform=='win32':
    cls=DFlowModel
    cls.dfm_bin_exe="dflowfm-cli.exe"
    cls.mpi_bin_exe="mpiexec.exe"
