"""
Automate parts of setting up a DFlow hydro model.

TODO:
  allow for setting grid bathy from the model instance
"""
import os,shutil,glob,inspect
import six
import pdb

import sys

import logging
log=logging.getLogger('DFlowModel')

import copy
import numpy as np
import xarray as xr
import pandas as pd
from shapely import geometry
from collections import defaultdict

import stompy.model.delft.io as dio
from stompy import xr_utils
from stompy.io.local import noaa_coops, hycom
from stompy import utils, filters, memoize
from stompy.spatial import wkb2shp, proj_utils
from stompy.model.delft import dfm_grid
import stompy.grid.unstructured_grid as ugrid

from scipy import sparse

from . import io as dio
from . import waq_scenario
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

    fixed_weirs=None

    # If True, initialize to WaqOnlineModel instance.
    dwaq=False
    # Specify location of proc_def.def file:
    waq_proc_def=None
    
    # flow and source/sink BCs will get the adjacent nodes dredged
    # down to this depth in order to ensure the impose flow doesn't
    # get blocked by a dry edge. Set to None to disable.
    # This has moved to just the BC objects, and removed here to avoid
    # confusion.
    # dredge_depth=-1.0

    def __init__(self,*a,**kw):
        # Still working out the ordering.
        # currently HydroModel calls configure(), and configure
        # might assume that self.mdu exists.
        # I don't think there is a downside to setting up a default
        # mdu right here.

        # non-DWAQ tracers. WIP. Setting tracers to empty here is slightly
        # problematic when a subclass defines self.tracers at the class
        # level. Try having this in configure instead.
        #self.tracers=[] 
        
        self.load_default_mdu()

        super().__init__(*a,**kw)

    def __repr__(self):
        return '<DFlowModel: %s>'%self.run_dir

    def configure(self):
        self.tracers=[] # non-DWAQ tracers, init moved here from __init__()
        
        super(DFlowModel,self).configure()

        # This is questionable -- new code in create_restart does this
        # explicitly and pass configure=False
        if self.restart_from is not None:
            self.set_restart_from(self.restart_from)

        if self.dwaq is True:
            self.dwaq=waq_scenario.WaqOnlineModel(model=self)

    def load_default_mdu(self):
        """
        Load a default set of config values from data/defaults-r53925.mdu
        """
        # Updated defaults-r53925.mdu by removing settings that 2021.03
        # complains about.
        fn=os.path.join(os.path.dirname(__file__),"data","defaults-2021.03.mdu")
        # Made some updates to this, but not yet tested:
        # fn=os.path.join(os.path.dirname(__file__),"data","defaults-2023.02.mdu")
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

    def write(self):
        if not bool(self.restart):
            super().write()
        else:
            # Do some of the typical work of restart, but then copy as much
            # as possible from the parent run.
            if self.restart_deep:
                self.set_run_dir(self.run_dir,mode='create')
                self.update_config()
                # 2024-07-08: At least some parts of write_config() would be
                # better handled after copy_files_for_restart(). There is the
                # danger that write_config() will write a file and copy_files_for
                # restart will overwrite. This happened with sources.sub.
                self.write_config()
                self.copy_files_for_restart()
                                
                # If/when this gets smarter, say overriding BCs, it will have to become
                # more granular here. One option would be to create BC instances that know
                # how to copy over the original files and stanzas verbatim.
            else:
                self.log.warning("Shallow restart logic in DFlowModel.write() is sketchy")
                # less sure about these.
                self.update_config()
                self.mdu.write()

        if self.dwaq:
            self.dwaq.write_waq_forcing()
                
    def copy_files_for_restart(self):
        """
        Do the real work of setting up a restart by copying files
        from parent run.
        Implied that this is a deep restart
        """

        # The restart equivalent of these steps in write():
        #   self.write_extra_files()
        #   self.write_forcing()
        #   self.write_grid()
        
        # hard to know what all files we might want.
        # include any tim, pli, pliz, ext, xyz, ini
        # restart version of partition I think handles the grid?
        # also include any file that appears in FlowFM.ext
        # (because some forcing input like wind can have weird suffixes)
        # and include the original grid (check name in mdu)
        
        # skip any .steps, .cache
        # skip any mdu
        # probably skip anything without an extension
        with open(self.restart_from.mdu.filepath(('external forcing','ExtForceFile'))) as fp:
            flowfm_ext=fp.read()
        with open(self.mdu.filename) as fp:
            flowfm_mdu=fp.read()

        for fn in os.listdir(self.restart_from.run_dir):
            _,suffix = os.path.splitext(fn)
            do_copy = ( (suffix in ['.tim','.pli','.pliz','.ext','.xyz','.ini','.xyn'])
                        or (fn in flowfm_ext)
                        # sources.sub is explicitly handled in write_config
                        or (fn in flowfm_mdu and fn !='sources.sub')
                        or (fn==self.mdu['geometry','NetFile']) )
            # a bit kludgey. restart paths often include DFM_OUTPUT_flowfm, but definitely
            # don't want to copy that.
            fn_path=os.path.join(self.restart_from.run_dir,fn)
            if fn.startswith('DFM_OUTPUT') or os.path.isdir(fn_path):
                do_copy=False
                
            if do_copy:
                shutil.copyfile(fn_path, os.path.join(self.run_dir,fn))
                
    def write_forcing(self,overwrite=True):
        # Prep external forcing file
        bc_fn=self.ext_force_file()
        assert bc_fn,"DFM script requires old-style BC file.  Set [external forcing] ExtForceFile"
        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)
        utils.touch(bc_fn)

        # Could compile names of any generic tracers. For now, require user to
        # add names to self.tracers.
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
        """
        Reads an mdu into self.mdu.  Does not update mdu_basename,
        such that self.write() will still use self.mdu_basename.
        """
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
        # use cls(), so that custom subclasses can be used
        model=cls(configure=False)
        model.load_from_mdu(fn)
        return model

    def load_from_mdu(self,fn):
        self.load_mdu(fn)
        self.mdu_basename=os.path.basename(fn)

        try:
            self.grid = ugrid.UnstructuredGrid.read_dfm(self.mdu.filepath( ('geometry','NetFile') ))
        except FileNotFoundError:
            log.warning("Loading model from %s, no grid could be loaded"%fn)
            self.grid=None
        d=os.path.dirname(fn) or "."
        self.set_run_dir(d,mode='existing')
        # infer number of processors based on mdu files
        # Not terribly robust if there are other files around..
        sub_mdu=glob.glob( fn.replace('.mdu','_[0-9][0-9][0-9][0-9].mdu') )
        if len(sub_mdu)>0:
            self.num_procs=len(sub_mdu)
        else:
            # probably better to test whether it has even been processed
            self.num_procs=1

        ref,start,stop=self.mdu.time_range()
        self.ref_date=ref
        self.run_start=start
        self.run_stop=stop

        self.load_gazetteer_from_run()
        self.load_structures_from_run()

    def load_structures_from_run(self):
        struct_file=self.mdu.filepath( ('geometry','structurefile'))
        if struct_file is None or not os.path.exists(struct_file):
            return
        structures=[]
        for sec in dio.SectionedConfig(struct_file).section_dicts():
            # sec will have _section(sec)
            del sec['_section']
            # Replace .tim entries with loaded timeseries
            for k in sec:
                if sec[k].endswith('.tim'):
                    tim_fn=os.path.join(self.mdu.base_path,sec[k])
                    if os.path.exists(tim_fn):
                        sec[k]=self.read_tim(tim_fn)
                else:
                    # Try parsing as float
                    try:
                        sec[k]=float(sec[k])
                    except ValueError:
                        pass
            # Try to populate geometry from the pli
            if 'polylinefile' in sec:
                pli_fn=os.path.join(self.mdu.base_path,sec['polylinefile'])
                pli=dio.read_pli(pli_fn)
                # This probably doesn't round-trip correctly -- should it be
                # an array? shapely.LineString?
                sec['geom']=pli[0][1] # grab first polyline, and just the coordinates

            structures.append(sec)
        self.structures=structures

    def load_fixed_weirs_from_run(self):
        fw_file=self.mdu.filepath( ('geometry','FixedWeirFile') )
        if fw_file is None or not os.path.exists(fw_file):
            return

        self.fixed_weirs=dio.read_pli(fw_file)
    

    def load_gazetteer_from_run(self):
        """
        Populate gazetteers with geometry read in from an existing run.
        So far only gets stations.  Will have to come back to handle
        transects, regions, etc. and maybe even read back in BC locations,
        or query output history files.
        """
        fn=self.mdu.filepath(['output','ObsFile'])
        if fn and os.path.exists(fn):
            stations=pd.read_csv(self.mdu.filepath(['output','ObsFile']),
                                 sep=r'\s+',names=['x','y','name'],quotechar="'")
            # crude workaround to silence warning. numpy and pandas will attempt
            # to use the array interface to streamline storage of Points.
            # that angers shapely. probably once that interface disappears this
            # will silently work just fine. but I'm tired of seeing the warnings.
            pnts=np.zeros(len(stations),dtype=object)
            for i,(x,y) in enumerate(stations.loc[ :, ['x','y']].values):
                pnts[i]=geometry.Point(x,y)
            stations['geom']=pnts
            self.gazetteers.append(stations.to_records())

    def parse_old_bc(self,fn):
        """
        Parse syntax of old-style BC files into a list of dictionaries.
        Keys are forced upper case.
        """
        def key_value(s):
            k,v=s.strip().split('=',1)
            k,v=k.strip().upper(),v.strip()
            return k,v

        rec=None
        recs=[]

        with open(fn,'rt') as fp:
            stanza=[]
            while 1:
                raw_line=fp.readline()
                if raw_line=="": break
                
                stanza.append(raw_line.strip())

                line=raw_line.split('#')[0].strip()
                if not line: # blank line or comment
                    continue 
                k,v=key_value(line)

                # each tracer starts with a quantity line
                if k=='QUANTITY':
                    rec={k:v}
                    recs.append(rec)
                    stanza=[stanza.pop()] # Shift this line to a new stanza
                    rec['stanza']=stanza
                else:
                    rec[k]=v
        return recs

    def load_bcs(self,load_data=True,ext_fn=None,keep_anonymous=True):
        """
        Woefully inadequate parsing of external forcing file.
        Ostensibly boundary conditions, but external forcing file has all sorts of
        stuff: initial conditions, WAQ mass balance areas, friction coefficients. etc.

        For now, returns a list of dictionaries. Some items have a 'bc' entry pointing
        to a BC instance.

        If load_data is False avoid potentially expensive file reading operations.
        
        Handle other BCs like at least flow.
        Breaks discharge_temperature_sorsin BCs into a SourceSink and child
        ScalarBCs. If keep_anonymous=False, then child scalar bcs aside from
        salinity and temperature are discarded. Otherwise they are kept but
        will get names like 'tmp_val5'
        """
        if ext_fn is None:
            ext_fn=self.mdu.filepath(['external forcing','ExtForceFile'])
        ext_new_fn=self.mdu.filepath(['external forcing','ExtForceFileNew'])

        recs=self.parse_old_bc(ext_fn)
        # translation of source sink BCs can create additional scalar BCs
        # these are collected in this list and appended to recs on the way
        # out
        extra_recs=[]

        # warn of various known shortcomings, but just once.
        warn_anonymous=True
        warn_mass_balance_not_parsed=True
        warn_tracer_no_parent=True

        unhandled=defaultdict(lambda: 0)

        for rec in recs:
            if 'FILENAME' in rec:
                # The ext file doesn't have a notion of name.
                # punt via the filename
                rec['name'],ext=os.path.splitext(os.path.basename(rec['FILENAME']))
            else:
                rec['name']=rec['QUANTITY'].upper()
                ext=None

            assert 'data' not in rec,"How is there already a data item?"
            rec['data']=None # will get updated if there is data and load_data=True
                
            if ext=='.pli':
                pli_fn=os.path.join(os.path.dirname(ext_fn),
                                    rec['FILENAME'])
                pli=dio.read_pli(pli_fn)
                rec['pli']=pli

                rec['coordinates']=rec['pli'][0][1]
                if rec['coordinates'].ndim==2 and rec['coordinates'].shape[0]>1:
                    # Could be a polygon, too. Not worrying about that rn
                    geom=geometry.LineString(rec['coordinates'])
                else:
                    # probably a point source
                    geom=geometry.Point(np.squeeze(rec['coordinates']))
                rec['geom']=geom

                data=None
                tim_fn=None
                if load_data:
                    # timeseries at one or more points along boundary:
                    # DIFFERENT logic if it's a source/sink.
                    if rec['QUANTITY'].upper()=="DISCHARGE_SALINITY_TEMPERATURE_SORSIN":
                        tim_fn=pli_fn.replace('.pli','.tim')
                        if os.path.exists(tim_fn):
                            t_ref,t_start,t_stop=self.mdu.time_range()
                            # These have a column for flow and each tracer
                            # let read_dfm_time figure out the number of columns.
                            # trying to assign names to the column is too fraught. The order
                            # depends on the defined tracers, and the order in which they
                            # are mentioned in boundary conditions, initial conditions and
                            # the substance file. Punt the complexity to the caller.
                            data=dio.read_dfm_tim(tim_fn,t_ref)
                    else:
                        tims=[]
                        for node_i,node_xy in enumerate(rec['coordinates']):
                            tim_fn=pli_fn.replace('.pli','_%04d.tim'%(node_i+1))
                            if os.path.exists(tim_fn):
                                t_ref,t_start,t_stop=self.mdu.time_range()
                                tim_ds=dio.read_dfm_tim(tim_fn,t_ref,columns=['stage'])
                                tim_ds['x']=(),node_xy[0]
                                tim_ds['y']=(),node_xy[1]

                                tims.append(tim_ds)
                        if len(tims):
                            data=xr.concat(tims,dim='node')
                rec['data']=data
            elif ext=='.xyz':
                if load_data:
                    xyz_fn=os.path.join(os.path.dirname(ext_fn),
                                        rec['FILENAME'])
                    df=pd.read_csv(xyz_fn,sep=r'\s+',names=['x','y','z'])
                    ds=xr.Dataset()
                    ds['x']=('sample',),df['x']
                    ds['y']=('sample',),df['y']
                    ds['z']=('sample',),df['z']
                    ds=ds.set_coords(['x','y'])
                    rec['data']=ds.z
            else:
                pli=geom=pli_fn=None # avoid pollution
                
            if rec['QUANTITY'].upper()=='WATERLEVELBND':
                bc=hm.StageBC(name=rec['name'],geom=rec['geom'],water_level=rec['data'])
                rec['bc']=bc
            elif rec['QUANTITY'].upper()=='DISCHARGEBND':
                if rec.get('data',None) is not None:
                    # Single flow value, no sense of multiple time series
                    rec['data']=rec['data'].isel(node=0).rename({'stage':'flow'})
                    bc=hm.FlowBC(name=rec['name'],geom=geom,flow=rec['data'])
                else:
                    if load_data:
                        print("Reading discharge boundary, did not find data (%s)"%tim_fn)
                    bc=hm.FlowBC(name=rec['name'],geom=geom)

                rec['bc']=bc
            elif rec['QUANTITY'].upper()=='FRICTIONCOEFFICIENT':
                rec['bc']=hm.RoughnessBC(name=rec['name'],data_array=rec['data'])
            elif rec['QUANTITY'].upper()  in ['SALINITYBND','TEMPERATUREBND']:
                if warn_tracer_no_parent:
                    self.log.warning("Parsing external forcing: tracers not yet connected to parents")
                    warn_tracer_no_parent=False
                tracer=rec['QUANTITY'].replace('bnd','').lower()
                rec['bc']=hm.ScalarBC(name=rec['name'],geom=rec['geom'],
                                      scalar=tracer,data=rec['data'])
                
            elif rec['QUANTITY'].upper()=='DISCHARGE_SALINITY_TEMPERATURE_SORSIN':
                # since we didn't specify names for the tim file, it's val1, val2, val3,
                # etc.
                if load_data:
                    flow=rec['data'].val1.rename('flow')
                else:
                    flow=None
                    
                rec['bc']=hm.SourceSinkBC(name=rec['name'], geom=rec['geom'], flow=flow)

                if load_data:
                    var_names=list(rec['data'].data_vars)
                else:
                    var_names=[]
                    
                def add_scalar_bc(tracer,da):
                    bc=hm.ScalarBC(parent=rec['bc'],
                                   scalar=tracer,value=da)
                    extra_recs.append(dict(bc=bc,name=rec['name'],geom=rec['geom'],
                                           child=True))
                tracer_idx=2
                
                if float(self.mdu['physics','Salinity'])!=0.0:
                    vname='val%d'%tracer_idx
                    if load_data:
                        salt=rec['data'][vname].rename('salinity')
                    else:
                        salt=None
                    add_scalar_bc('salinity',salt)
                    tracer_idx+=1
                if float(self.mdu['physics','Temperature'])!=0.0:
                    vname='val%d'%tracer_idx
                    if load_data:
                        temp=rec['data'][vname].rename('temperature')
                    else:
                        temp=None
                    add_scalar_bc('temperature',temp)
                    tracer_idx+=1
                    
                # The tricky part here is that the number and names of tracers attached to the
                # BC depends on other parts of the model configuration. Those might
                # have been changed, or haven't been set yet.
                if keep_anonymous:
                    while True:
                        vname='val%d'%tracer_idx
                        if vname not in var_names: break
                        if warn_anonymous:
                            self.log.warning("Tracers for source/sink BC beyond temp and salinity will be anonymous")
                            warn_anonymous=False
                        add_scalar_bc('tmp_'+vname,rec['data'][vname])
                        tracer_idx+=1
            elif rec['QUANTITY'].upper().startswith('WAQMASSBALANCEAREA'):
                if warn_mass_balance_not_parsed:
                    self.log.warning("Parsing external forcing. WAQ mass balance areas are not parsed")
                    warn_mass_balance_not_parsed=False
            else:
                unhandled[rec['QUANTITY']]+=1
                #print("Not implemented: reading BC quantity=%s"%rec['QUANTITY'])
        for k in unhandled:
            print("Encountered %d BCs with quantity=%s that weren't fully parsed"%
                  (unhandled[k],k))
        return recs + extra_recs

    # def tracer_list(self):
    #     """
    #     For source/sink BCs DFM requires a single time series file with the full suite
    #     of tracer concentrations. Note that this relies on the current state of the model.
    #     May abandon this because it is likely to be fragile. 
    #     """
    #     columns=['flow']
    #     if float(self.mdu['physics','Salinity'])!=0.0:
    #         columns.append('salinity')
    #     if float(self.mdu['physics','Temperature'])!=0.0:
    #         columns.append('temperature')
    # 
    #     # 
    #         
    #     data=dio.read_dfm_tim(tim_fn,t_ref,columns=self.tracer_list())

    @property
    def n_layers(self):
        """
        Returns 0 for 2D, 1 for 3D with a single layer.
        """
        return int(self.mdu['geometry','Kmx'])
    
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
        print(f"Top of partition: num_procs={self.num_procs}")
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
            print(f"About to call {cmd}")
            output=self.run_dflowfm(cmd,mpi=False)
            print("-"*80)
            print(output)
            print("-"*80)
        else:
            # Copy the partitioned network files:
            if self.restart_deep:
                for proc in range(self.num_procs):
                    old_grid_fn=self.restart_from.subdomain_grid_filename(proc)
                    new_grid_fn=self.subdomain_grid_filename(proc)
                    self.log.info("Copying pre-partitioned grid files: %s => %s"%(old_grid_fn,new_grid_fn))
                    shutil.copyfile(old_grid_fn,new_grid_fn)
            else:
                self.log.info("Shallow restart, don't copy partitioned grid")
                
            # not a cross platform solution!
            gen_parallel=os.path.join(self.dfm_bin_dir,"generate_parallel_mdu.sh")
            cmd=[gen_parallel,os.path.basename(self.mdu.filename),"%d"%self.num_procs,'6']
            print(f"About to call {cmd}")
            return utils.call_with_path(cmd,self.run_dir)

    def chain_restarts(self):
        """
        Attempt to chain back restarts, returning a list of
        MDU objects in chronological order ending with self.mdu
        """
        # eventually support criteria on how far back
        # returns a list of MDU objects (with .filename set),
        # including self.mdu
        mdus=[self.mdu]

        mdu=self.mdu

        while 1:
            if not mdu['restart','RestartFile']:
                break
            restart=mdu['restart','RestartFile']
            # '../data_2016long_3d_asbuilt_impaired_scen2_l100-v007/DFM_OUTPUT_flowfm/flowfm_20160711_000000_rst.nc'
            # For now, this only works if the paths are 'normal'
            restart_mdu=os.path.dirname(restart).replace('DFM_OUTPUT_','')+".mdu"
            restart_mdu=os.path.normpath( os.path.join(os.path.dirname(mdu.filename),restart_mdu) )
            if not os.path.exists(restart_mdu):
                self.log.warning("Expected preceding restart at %s, but not there"%restart_mdu)
                break
            mdu=dio.MDUFile(restart_mdu)
            mdus.insert(0,mdu)
        return mdus
        
    @property
    def restart_deep(self):
        """
        True if this is a restart and we are doing a deep copy
        """
        return bool(self.restart) and (self.run_dir!=self.restart_from.run_dir)
        
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
    
    def run_simulation(self,threads=1,extra_args=[]):
        """
        Start simulation. 
          threads: if specified, pass on desired number of openmp threads to dfm.
          extra_args: additional list of other commandline arguments. Note that
          arguments must be split up into a list (e.g. ["--option","value"] as
          opposed to "--option value").
        """
        cmd=[]
        if threads is not None:
            cmd += ["-t","%d"%threads]
        cmd += ["--autostartstop",os.path.basename(self.mdu.filename)]
        
        if self.dwaq:
            cmd=self.dwaq.update_command(cmd)
            
        cmd += extra_args
        return self.run_dflowfm(cmd=cmd)
    
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

    def dia_fn(self):
        """
        Return path to diagnostic filename (rank 0 if mpi run).
        returns None if file cannot be found.
        """
        root_fn=self.mdu.filename[:-4] # drop .mdu suffix
        # Look in multiple locations for diagnostic file.
        # In older DFM, MPI runs placed it next to mdu, while
        # serial and newer DFM (>=1.6.2?) place it in
        # output folder
        dia_fns=[]
        dia_fn_base=os.path.basename(root_fn)

        # use self.num_procs as a hint, but it's not always reliable so check
        # for both MPI and serial output.
        dia_fn_bases=[dia_fn_base+'_0000.dia',
                      dia_fn_base+'.dia']

        if self.num_procs<=1:
            # check for serial first.
            dia_fn_bases = dia_fn_bases[::-1]

        dia_paths=[self.run_dir,
                   os.path.join(self.run_dir, "DFM_OUTPUT_%s"%self.mdu.name)]

        for dia_fn_base in dia_fn_bases:
            for dia_path in dia_paths:
                dia_fn=os.path.join(dia_path,dia_fn_base)
                assert dia_fn!=self.mdu.filename,"Probably case issues with %s"%dia_fn

                if os.path.exists(dia_fn):
                    return dia_fn
        return None
        
    def is_completed(self):
        """
        return true if the model has been run.
        this can be tricky to define -- here completed is based on
        a report in a diagnostic that the run finished.
        this doesn't mean that all output files are present.
        """

        dia_fn=self.dia_fn()
        if dia_fn is None:
            return False
        
        # Read the last 1000 bytes
        with open(dia_fn,'rb') as fp:
            fp.seek(0,os.SEEK_END)
            tail_size=min(fp.tell(),10000)
            fp.seek(-tail_size,os.SEEK_CUR)
            # This may not be py2 compatible!
            tail=fp.read().decode(errors='ignore')
        return "Computation finished" in tail

    def timing_stats(self):
        dia_fn=self.dia_fn()
        if dia_fn is None:
            return None

        # Read the last 1000 bytes                                                                                                             
        with open(dia_fn,'rb') as fp:                                                                                                          
            fp.seek(0,os.SEEK_END)                                                                                                             
            tail_size=min(fp.tell(),10000)                                                                                                     
            fp.seek(-tail_size,os.SEEK_CUR)                                                                                                    
            # This may not be py2 compatible!                                                                                                  
            tail=fp.read().decode(errors='ignore')

            def parse_time(days,hours):
                days=np.timedelta64(int(days.strip('d')),'D')
                seconds=sum( [mult*int(val) for mult,val in zip([3600,60,1],hours.split(':'))])
                return days+np.timedelta64(seconds,'s')

            result={}
            for line in tail.split("\n"):
                parts=line.split()
                if len(parts)!=14: continue

                result['sim_time_completed']=parse_time(parts[3],parts[4])
                result['sim_time_remaining']=parse_time(parts[5],parts[6])
                result['wall_time_completed']=parse_time(parts[7],parts[8])
                result['wall_time_remaining']=parse_time(parts[9],parts[10])
                result['percent_complete']=float(parts[12].strip('%'))
                result['mean_dt']=float(parts[13])

                # Looking for progress lines like
                # ** INFO   :          12d  0:00:00          0d  0:00:00          1d 15:28:44          0d  0:00:00          0   100.0%     6.12245
        return result
    
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

        self.update_initial_water_level()

        if self.fixed_weirs is not None:
            self.mdu['geometry','FixedWeirFile']='fixed_weirs.pliz'
            dio.write_pli(self.mdu.filepath(('geometry','FixedWeirFile')),self.fixed_weirs)
            
        if self.dwaq:
            # This updates a few things in self.mdu
            # Also actually writes some output, though that could be
            # folded into a later part of the process if it turns out
            # the dwaq config depends on reading some of the DFM
            # details.
            self.dwaq.write_waq()

        if self.restart_from is not None:
            # This really breaks things if we've messed with modified ICs.
            self.log.warning("SKIPPING self.set_restart_file()")
            #self.set_restart_file()
        
    def write_config(self):
        # Assumes update_config() already called
        self.write_structures() # updates mdu
        self.write_monitors()
        log.info("Writing MDU to %s"%self.mdu.filename)
        self.mdu.write()

    def write_monitors(self):
        # start with empty
        if not self.mdu['output','ObsFile']:
            self.mdu['output','ObsFile']='obs.xyn'
            
        open(self.mdu.filepath( ('output','ObsFile') ),'wt').close()
        self.write_monitor_points()
        self.write_monitor_sections()

    def write_monitor_points(self):
        fn=self.mdu.filepath( ('output','ObsFile') )
        if fn is None: return
        if not fn.startswith(self.run_dir):
            # Assume we should just use pre-existing file.
            if not os.path.exists(fn):
                self.log.warning("Monitor points file '%s' is outside run directory but does not exist"%fn)
            return
        with open(fn,'at') as fp:
            for i,mon_feat in enumerate(self.mon_points):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_pnt_%03d"%i
                xy=np.array(mon_feat['geom'].coords[0]) # shapely api update
                fp.write("%.3f %.3f '%s'\n"%(xy[0],xy[1],name))
    def write_monitor_sections(self,append=True):
        fn=self.mdu.filepath( ('output','CrsFile') )
        if fn is None: return
        if not fn.startswith(self.run_dir):
            # Assume we should just use pre-existing file.
            if not os.path.exists(fn):
                self.log.warning("Monitor sections file '%s' is outside run directory but does not exist"%fn)
            return
        
        if append:
            mode='at'
        else:
            mode='wt'
            
        with open(fn,mode) as fp:
            for i,mon_feat in enumerate(self.mon_sections):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_sec_%03d"%i
                xy=np.array(mon_feat['geom'].coords) # shapely api update
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
                    "polylinefile = %s.pli"%s['name']
                    ]
                for k in s:
                    if k in ['type','name','geom']: continue
                    if isinstance(s[k],xr.DataArray):
                        # Note that only a few of the parameters can be time series
                        # iirc, crest level, gate opening, gate height
                        log.debug(f"{k} appears to be data")
                        tim_base=f"{s['name']}_{k}.tim"
                        tim_fn=os.path.join(self.run_dir,tim_base)
                        self.write_tim(s[k],tim_fn)
                        lines.append( "%s = %s"%(k,tim_base) )
                    else:
                        lines.append( "%s = %s"%(k,s[k]) )
                lines.append("\n")
                # "door_height  = %.3f"%s['door_height'],
                # "lower_edge_level = %.3f"%s['lower_edge_level'],
                # "opening_width = %.3f"%s['opening_width'],
                # "sill_level     = %.3f"%s['sill_level'],
                # "horizontal_opening_direction = %s"%s['horizontal_opening_direction'],
                # "\n"

                fp.write("\n".join(lines))
                pli_fn=os.path.join(self.run_dir,s['name']+'.pli')
                if 'geom' in s:
                    geom=s['geom']
                    if isinstance(geom,np.ndarray):
                        geom=geometry.LineString(geom)
                else:
                    geom=self.get_geometry(name=s['name'])

                if geom.geom_type=='MultiLineString' and len(geom.geoms)==1:
                    geom=geom.geoms[0] # geojson I think does this.
                assert geom.geom_type=='LineString'
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
        elif isinstance(bc,hm.SourceSinkBC):
            self.write_source_bc(bc)
        elif isinstance(bc,hm.FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,hm.WindBC):
            self.write_wind_bc(bc)
        elif isinstance(bc,hm.RainfallRateBC):
            self.write_rainfall_rate_bc(bc)
        elif isinstance(bc,hm.RoughnessBC):
            self.write_roughness_bc(bc)
        elif isinstance(bc,hm.ScalarBC):
            self.write_scalar_bc(bc)
        else:
            super(DFlowModel,self).write_bc(bc)

    # If True, timesteps in the forcing data beyond the run
    # will be trimmed out.
    bc_trim_time=True
    
    def write_tim(self,da,file_path,trim_time=None):
        """
        Write a DFM tim file based on the timeseries in the DataArray.
        da must have a time dimension.  No support yet for vector-values here.
        file_path is relative to the working directory of the script, not
        the run_dir.
        """
        if trim_time is None:
            trim_time=self.bc_trim_time
            
        ref_date,start,stop = self.mdu.time_range()
        dt=np.timedelta64(60,'s') # always minutes

        if 'time' not in da.dims:
            pad=np.timedelta64(86400,'s')
            times=np.array([start-pad,stop+pad])
            values=np.array([da.values,da.values])
        else:
            # Be sure time is the first dimension
            dim_order=['time'] + [d for d in da.dims if d!='time']
            da=da.transpose(*dim_order)

            times=da.time.values
            values=da.values

            if trim_time:
                # Check for no original data within the time span
                if times[-1] < start:
                    log.warning(f'{file_path}: data ends ({times.max()}) before simulation period: {start} - {stop}.')
                    times = np.array([start, stop])
                    values=np.r_[values[-1],values[-1]]
                elif times[0] > stop:
                    log.warning(f'{file_path}: data starts ({times.min()}) after simulation period: {start} - {stop}.')
                    times = np.array([start, stop])
                    values=np.r_[values[0],values[0]]
                else:
                    # common case -- trim and pad out 1 sample
                    i_start,i_stop=np.searchsorted(da.time,[start,stop])
                    # i_start will come back pointing to the first element in da.time
                    # >=start. Either way, step back 1 just to be sure
                    # i_stop will come back pointing to the first element ...
                    # >=stop. Add 2: 1 because stop is exclusive, and 1 so we get one
                    # entry beyond
                    i_start=max(0,i_start-1)
                    i_stop=min(len(da.time),i_stop+2)

                    times=times[i_start:i_stop]
                    values=values[i_start:i_stop]
            
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
            log.info("Dredging grid for flow BC %s"%bc.name)
            self.dredge_boundary(np.array(bc.geom.coords),bc.dredge_depth)
        else:
            log.info("dredging disabled")

    def default_source_sink_forcing(self,single_ended):
        """
        Source Sink BCs in DFM have to include all of the scalars in one go.
        Build a list of scalar names and default BCs. This will probably
        have to get smarter w.r.t to tracer order, at least allowing
        the order of waq tracers to be overridden. It should also
        get some caching, but that's for another day.
        """
        scalar_names=[] # forced to lower case
        scalar_das=[]

        if int(self.mdu['physics','Salinity']):
            scalar_names.append('salinity')
            if single_ended:
                default=self.mdu['physics','InitialSalinity']
                if default is None: default=0.0
                else: default=float(default)
            else:
                default=0.0
            scalar_das.append(xr.DataArray(default,name='salinity'))
        if int(self.mdu['physics','Temperature']):
            scalar_names.append('temperature')
            if single_ended:
                default=self.mdu['physics','InitialTemperature']
                if default is None: default=0.0
                else: default=float(default)
            else:
                default=0.0
            scalar_das.append(xr.DataArray(default,name='temp'))
        for tracer in self.tracers:
            # TODO: track more information in a small class or dict
            scalar_names.append(tracer)
            scalar_das.append(xr.DataArray(0.0,name=tracer))
        if self.dwaq:
            for sub_name in self.dwaq.substances:
                sub=self.dwaq.substances[sub_name]
                if not sub.active: continue
                scalar_names.append(sub_name.lower())
                if single_ended:
                    default=sub.initial.default
                else:
                    default=0.0
                scalar_das.append(xr.DataArray(default,name=sub_name))

        return scalar_names, scalar_das
            
    def write_source_bc(self,bc,**write_opts):
        """
        Write a source/sink BC.
        Start with default then scan for any specified
        scalar BCs to use instead of defaults.
        write_opts: e.g. write_data,write_geom,write_ext
        """
        # DFM source/sinks have salinity and temperature attached
        # the same data file.
        # the pli file can have a single entry, and include a z coordinate,
        # based on lsb setup
        
        # In the case of a single-ended source/sink, these
        # should pull default value from the model config, instead of
        # assuming 0.0
        single_ended=bc.geom.geom_type=='Point'
        
        scalar_names,scalar_das = self.default_source_sink_forcing(single_ended=single_ended)

        salt_bc=None
        temp_bc=None
        for scalar_bc in self.bcs:
            if isinstance(scalar_bc, hm.ScalarBC) and scalar_bc.parent==bc:
                scalar=scalar_bc.scalar.lower()
                try:
                    idx=scalar_names.index( scalar )
                except ValueError:
                    raise Exception("Scalar %s not in known list %s"%(scalar,scalar_names))
                scalar_das[idx]=scalar_bc.data()

        # Source/sink bcs in DFM include salinity and temperature, as well as any tracers
        # from dwaq
        das=[bc.data()] + scalar_das

        # merge data arrays including time
        # write_tim has been updated to transpose time to be the first dimension
        # as needed, so this should be okay
        # But we do need to broadcast before they can be concatenated.
        das=xr.broadcast(*das)
        # 'minimal' here avoids a crash if one of the dataarrays has an
        # extra coordinate that isn't actually used (like a singleton coordinate
        # from an isel() )
        da_combined=xr.concat(das,dim='component',coords='minimal')

        self.write_gen_bc(bc,quantity='source',da=da_combined,**write_opts)

        if write_opts.get('write_geom',True):
            if (bc.dredge_depth is not None) and (self.restart_from is None):
                # Additionally modify the grid to make sure there is a place for inflow to
                # come in.
                log.info("Dredging grid for source/sink BC %s"%bc.name)
                # These are now class methods using a generic implementation in HydroModel
                # may need some tlc
                self.dredge_discharge(np.array(bc.geom.coords),bc.dredge_depth)
            else:
                log.info("dredging disabled")

    def write_gen_bc(self,bc,quantity,da=None,write_ext=True,write_data=True,write_geom=True):
        """
        handle the actual work of writing flow and stage BCs.
        quantity: 'stage','flow','source'
        da: override value for bc.data()
        write_ext, write_data: optionally disable updating the external forcing
          file and/or the separate data file.
        """
        # 2019-09-09 RH: the automatic suffix is a bit annoying. it is necessary
        # when adding scalars, but for any one BC, only one of stage, flow or source
        # would be present.  Try dropping the suffix here.
        bc_id=bc.name # +"_" + quantity

        if write_geom:
            assert isinstance(bc.geom_type,list),"Didn't fully refactor, looks like"
            if (bc.geom is None) and (None not in bc.geom_type):
                raise Exception("BC %s, name=%s has no geometry. Maybe missing from shapefiles?"%(bc,bc.name))
            assert bc.geom.type in bc.geom_type

            coords=np.array(bc.geom.coords)
            ndim=coords.shape[1] # 2D or 3D geometry

            # Special handling when it's a source/sink, with z/z_src specified
            if quantity=='source':
                if ndim==2 and bc.z is not None:
                    # construct z
                    missing=-9999.
                    z_coords=missing*np.ones(coords.shape[0],np.float64)
                    for z_val,idx in [ (bc.z,-1),
                                       (bc.z_src,0) ]:
                        if z_val is None: continue
                        if z_val=='bed':
                            z_val=-10000
                        elif z_val=='surface':
                            z_val=10000
                        z_coords[idx]=z_val
                    if z_coords[0]==missing:
                        z_coords[0]=z_coords[-1]
                    # middle coordinates, if any, don't matter
                    coords=np.c_[ coords, z_coords ]
                    ndim=3

            pli_data=[ (bc_id, coords) ]

            if ndim==2:
                pli_fn=bc_id+'.pli'
            else:
                pli_fn=bc_id+'.pliz'

            dio.write_pli(os.path.join(self.run_dir,pli_fn),pli_data)

        lines=[]
        method=3 # default
        if quantity=='stage':
            lines.append("QUANTITY=waterlevelbnd")
            tim_path=os.path.join(self.run_dir,bc_id+"_0001.tim")
        elif quantity=='flow':
            lines.append("QUANTITY=dischargebnd")
            tim_path=os.path.join(self.run_dir,bc_id+"_0001.tim")
        elif quantity=='source':
            lines.append("QUANTITY=discharge_salinity_temperature_sorsin")
            method=1 # not sure how this is different
            tim_path=os.path.join(self.run_dir,bc_id+".tim")
        else:
            assert False
            
        if write_ext:
            with open(self.ext_force_file(),'at') as fp:
                lines+=["FILENAME=%s"%pli_fn,
                        "FILETYPE=9",
                        "METHOD=%d"%method,
                        "OPERAND=O",
                        ""]
                fp.write("\n".join(lines))

        if write_data:
            if da is None:
                da=bc.data()
            # assert len(da.dims)<=1,"Only ready for dimensions of time or none"
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

    def write_rainfall_rate_bc(self,bc):
        assert bc.geom is None,"Spatially rain not yet supported"

        tim_fn=bc.name+".tim"
        tim_path=os.path.join(self.run_dir,tim_fn)

        # write_config()
        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=rainfall_rate",
                   "FILENAME=%s"%tim_fn,
                   "FILETYPE=1", # uniform scalar
                   "METHOD=1",   # copying from wind above
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

        self.write_tim(bc.data(),tim_path)

    def write_scalar_bc(self,bc):
        bc_id=bc.name+"_"+bc.scalar

        parent_bc=bc.parent
        if isinstance(parent_bc,hm.SourceSinkBC):
            log.debug("BC %s should be handled by SourceSink"%bc_id)
            return
        
        assert isinstance(parent_bc, (hm.StageBC,hm.FlowBC)),"Haven't implemented point-source scalar yet"
        assert parent_bc.geom.geom_type=='LineString'
        
        pli_data=[ (bc_id, np.array(parent_bc.geom.coords)) ]
        pli_fn=bc_id+'.pli'
        dio.write_pli(os.path.join(self.run_dir,pli_fn),pli_data)

        is_dwaq=bool(self.dwaq) and (bc.scalar in self.dwaq.substances)
        
        if is_dwaq: # DelwaqScalarBC is defunct
            quant=f'tracerbnd{bc.scalar}'
        elif bc.scalar=='salinity':
            quant='salinitybnd'
        elif bc.scalar=='temperature':
            quant='temperaturebnd'
        else:
            self.log.info("scalar '%s' will be passed to DFM as generic tracer"%bc.scalar)
            quant=f'tracerbnd{bc.scalar}'

        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=%s"%quant,
                   "FILENAME=%s"%pli_fn,
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"
                   ]
            fp.write("\n".join(lines))

        da=bc.data()
        # Write tim
        assert len(da.dims)<=1,"Only ready for dimensions of time or none"
        tim_path=os.path.join(self.run_dir,bc_id+"_0001.tim")
        self.write_tim(da,tim_path)
        
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

    def update_initial_water_level(self):
        wl=self.infer_initial_water_level()
        if wl is not None:
            self.mdu['geometry','WaterLevIni']=wl
            self.log.info("Pulling initial water level from BC: %.3f"%wl)
        
    def map_outputs(self):
        """
        return a list of map output files
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_map.nc'))
        fns.sort()
        return fns

    _mu=None
    def map_dataset(self,force_multi=False,grid=None,chain=False,xr_kwargs={},
                    clone_from=None):
        """
        Return map dataset. For MPI runs, this will emulate a single, merged
        global dataset via multi_ugrid. For serial runs it directly opens
        an xarray dataset.

        grid: if given, the subdomains will be mapped to the given grid, instead
        of constructing a grid.

        xr_kwargs: options to pass to xarray, whether multi or single.

        chain: if True, attempt to chain in time. Experimental!

        clone_from: MultiUgrid instance from a previous call to map_dataset,
          presumably for a different restart but otherwise matching run.
          Will pull grid and local<-->global mappings from the provided MultiUgrid
          which should speedup the loading process. Currently only handled when
          chain is False.
        """
        if not chain:
            if self.num_procs<=1 and not force_multi:
                # xarray caches this.
                ds=xr.open_dataset(self.map_outputs()[0],**xr_kwargs)
                # put the grid in as an attribute so that the non-multiugrid
                # and multiugrid return values can be used the same way.
                ds.attrs['grid'] = ugrid.UnstructuredGrid.read_ugrid(ds)
                return ds
            else:
                from ...grid import multi_ugrid
                # This is slow so cache the result
                if self._mu is None:
                    self._mu=multi_ugrid.MultiUgrid(self.map_outputs(),grid=grid,xr_kwargs=xr_kwargs,
                                                    clone_from=clone_from)
                return self._mu
        else:
            # as with his_dataset(), caching is not aware of options like chain.
            if self._mu is None:
                mdus=self.chain_restarts()
                # This gets complicated since we are chaining in time and dealing
                # with potentially multiple subdomains.
                # Since MultiUgrid is not a proper Dataset, have to chain in time
                # at the processor level.

                # Scan for filenames across restarts
                map_fns=np.zeros( (len(mdus),self.num_procs), object)

                for i_restart,mdu in enumerate(mdus):
                    output_dir=mdu.output_dir()
                    fns=glob.glob(os.path.join(output_dir,'*_map.nc'))
                    fns.sort()
                    assert len(fns) == self.num_procs
                    map_fns[i_restart,:]=fns

                # Create chained datasets per processor:
                all_proc_dss=[] # chained dataset for each processor
                for proc in range(self.num_procs):
                    one_proc_dss=[xr.open_dataset(fn,**xr_kwargs)
                                  for fn in map_fns[:,proc]]
                    proc_dasks=[]
                    for ds in one_proc_dss[::-1]:
                        if len(proc_dasks)>0:
                            cutoff=proc_dasks[0].time.values[0]
                            tidx=np.searchsorted(ds.time.values, cutoff)
                            if tidx==0:
                                continue
                            ds=ds.isel(time=slice(0,tidx))
                        dask_ds=ds.chunk()
                        proc_dasks.insert(0,dask_ds)
                    # data_vars='minimal' is necessary otherwise things like
                    # mesh topology will also get concatenated in time.
                    # 'different' would probably also work.
                    proc_ds=xr.concat(proc_dasks,dim='time',data_vars='minimal')
                    all_proc_dss.append(proc_ds)
                if len(all_proc_dss)==1 and not force_multi:
                    self._mu=all_proc_dss[0]
                else:
                    from ...grid import multi_ugrid
                    # multi_ugrid will take a list of datasets
                    # and create a merged dataset
                    self._mu=multi_ugrid.MultiUgrid(all_proc_dss,grid=grid,xr_kwargs=xr_kwargs)
            return self._mu
                    
    def his_output(self):
        """
        return path to history file output
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_his.nc'))
        # Turns out [sometimes] DFM writes a history file from each processor at the
        # very end. The rank 0 file has all time steps, others just have a single
        # time step.
        # assert len(fns)==1
        fns.sort()
        return fns[0]

    @classmethod
    def clean_his_dataset(cls, fn, decode_geometry=True,set_coordinates=True,refresh=False,
                          **xr_kwargs):
        """
        The work of his_dataset. 
        """
        if refresh:
            # might be able to make this call faster by turning off all decoding?
            his_ds=xr.open_dataset(fn,**xr_kwargs) # is there a way to know if it's cached?
            his_ds.close()

        his_ds=xr.open_dataset(fn,**xr_kwargs)
            
        if set_coordinates:
            # Doctor up the dimensions
            # Misconfigured runs may have duplicates here.

            for coord,names in [ ('cross_section','cross_section_name'),
                                 ('weirgens','weirgen_id'),
                                 ('source_sink','source_sink_name'),
                                 ('stations','station_name'),
                                 ('general_structures','general_structure_id'),
                                 ('gategens','gategen_name')]:
                if names not in his_ds: continue
                coord_vals=[s.decode().strip() for s in his_ds[names].values]
                
                if len(coord_vals)>len(np.unique(coord_vals)):
                    print('Yuck - duplicate %s names'%coord)
                    mask=[val not in coord_vals[:i]
                          for i,val in enumerate(coord_vals)]
                    mask=np.array(mask, np.bool_ )
                    his_ds=his_ds.isel(**{coord:mask})
                    coord_vals=np.array(coord_vals)[mask]
                    
                his_ds[coord]=(coord,),coord_vals

        if decode_geometry:
            if 'cross_section_geom' in his_ds:
                xr_utils.decode_geometry(his_ds,'cross_section_geom',replace=True)
            elif 'cross_section_x_coordinate' in his_ds:
                from shapely import geometry
                geoms=np.zeros(his_ds.dims['cross_section'],dtype=object)
                # old-school.
                for sec_idx in range(his_ds.dims['cross_section']):
                    x=his_ds['cross_section_x_coordinate'].isel(cross_section=sec_idx)
                    y=his_ds['cross_section_y_coordinate'].isel(cross_section=sec_idx)
                    valid=x<1e35 # fill values are 9.9e36
                    x=x[valid]
                    y=y[valid]
                    geoms[sec_idx]=geometry.LineString(np.c_[x,y])
                his_ds['cross_section_geom']=('cross_section',),geoms
        return his_ds

    _his_ds=None
    _his_ds_chain=None
    def his_dataset(self,decode_geometry=True,set_coordinates=True,refresh=False,
                    chain=False,prechain=None,**xr_kwargs):
        """
        Return history dataset, with some minor additions to make
        it friendly.
        If chain is True, chain back through restarts, concatenate (slicing out
        duplicate time stamps from earlier runs), and return. Note that this will
        result in a xr.Dataset with dask array entries. That may result in poor
        performance, in which case ds=ds.compute() might help (but only after the
        dataset has been subsetted enough to fit in memory).

        prechain: when chaining, the list of un-merged datasets will be passed
        to this function before attempting to concatenate, and will proceed with
        whatever datasets are returned from this function.
        """
        if not refresh:
            if chain:
                if self._his_ds_chain is not None:
                    return self._his_ds_chain
            else:
                if self._his_ds is not None:
                    return self._his_ds

        clean_kwargs=dict(decode_geometry=decode_geometry,
                          set_coordinates=set_coordinates,refresh=refresh)
        clean_kwargs.update(xr_kwargs)
        
        # for chaining: aside from the path what is needed from self?
        # basically nothing.
        if chain:
            mdus=self.chain_restarts()
            his_fns=[]
            for mdu in mdus:
                output_dir=mdu.output_dir()
                fns=glob.glob(os.path.join(output_dir,'*_his.nc'))
                fns.sort()
                his_fns.append(fns[0])
            his_dss=[self.clean_his_dataset(fn,**clean_kwargs)
                     for fn in his_fns]
            his_dasks=[]
            for ds in his_dss[::-1]:
                if len(his_dasks)>0:
                    cutoff=his_dasks[0].time.values[0]
                    tidx=np.searchsorted( ds.time.values, cutoff)
                    if tidx==0:
                        continue
                    ds=ds.isel(time=slice(0,tidx))
                dask_ds=ds.chunk()
                his_dasks.insert(0,dask_ds)
            print(f"{len(his_dasks)} chained datasets")
            if prechain: # can i do this here?
                his_dasks=prechain(his_dasks)
            # data_vars='minimal' should avoid adding time dimension to static things like
            # geometry. 'different' also worth trying, though currently there are kludges
            # in client code for flipped sections, and they would have to be more complete
            # to avoid issues with using 'different'
            his_ds=xr.concat(his_dasks,dim='time',data_vars='minimal')
        else:
            his_ds=self.clean_his_dataset(self.his_output(),**clean_kwargs)

        if chain:
            self._his_ds_chain=his_ds
        else:
            self._his_ds=his_ds
            
        return his_ds

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
    
    def create_restart(self,deep=True,mdu_suffix="",**kwargs):
        # Consider skipping configure as we want to preserve as much of the original
        # run as possible.
        # 2022-08-10: trying that, fingers crossed
        new_model=self.__class__(configure=False,**kwargs) # in case of subclassing, rather than DFlowModel()
        new_model.set_restart_from(self,deep=deep,mdu_suffix=mdu_suffix)
        return new_model

    def set_restart_from(self,model,deep=True,mdu_suffix=""):
        """
        Pull the restart-related settings from model into the current instance.
        This is going to need tweaking. Previously it would re-use the original
        run directory, since outputs would go into a new sub-directory. But
        that's not flexible enough for general use of restarts.
        The default is a 'deep' restart, with a separate run dir.
        If deep is false, then mdu_suffix must be nonempty, a new mdu will be
        written alongside the existing one.
        """
        if not deep:
            assert mdu_suffix,"Shallow restart must provide suffix for new mdu file"
            self.run_dir=model.run_dir
            
        self.mdu=model.mdu.copy()
        self.mdu_basename=os.path.basename( model.mdu_basename.replace('.mdu',mdu_suffix+".mdu") )
        self.mdu.set_filename( os.path.join(self.run_dir, self.mdu_basename) )

        # recent DFM errors if Tlfsmo is set for a restart.
        self.mdu['numerics','Tlfsmo'] = 0.0
        # And at least in tag 140737, restarts from a restart file fail if renumbering is enabled.
        # 2023-12-08: restart is failing for a case with initial run had RenumberFlowNodes=1, and
        #   restart has RenumberFlowNodes=1.
        # Maybe we should just leave it??
        # self.mdu['geometry','RenumberFlowNodes']=0

        self.restart=True
        self.restart_from=model # used to be restart_model, but I don't think makes sense
        self.ref_date=model.ref_date
        self.run_start=model.restartable_time()
        assert self.run_start is not None,"Trying to restart run that has no restart data"
        
        self.num_procs=model.num_procs
        self.grid=model.grid

        # For now, these are synonymous
        if deep:
            assert self.run_dir != model.run_dir
        else:
            assert self.run_dir == model.run_dir

        self.set_restart_file()
    def set_restart_file(self):
        """ 
        Update mdu['restart','RestartFile'] based on run_start and self.restart_from
        Should be called when dealing with a restart and self.run_start is modified.

        This is currently fragile and error prone. The entry in the mdu depends on the
        restart file *and* run_dir. And the common usage is to just specify
        a DFlowModel instance in self.restart_from, somewhere along the way set the
        start time of this run to an output restart file time of the restart_from,
        and this method will take care of getting the path correct.

        The problem is when one or more of these inputs change.
        """
        self.log.info("set_restart_file: Setting RestartFile based on self.restart_from")
        rst_base=os.path.join(self.restart_from.mdu.output_dir(),
                              (self.restart_from.mdu.name
                               +'_'+utils.to_datetime(self.run_start).strftime('%Y%m%d_%H%M%S')
                               +'_rst.nc'))
        # That gets rst_base relative to the cwd, but we need it relative
        # to the new runs run_dir
        
        # raise Exception('HERE - this is effectively including an extra basepath I think??')
        self.mdu['restart','RestartFile']=os.path.relpath(rst_base,start=self.run_dir)
        # Fragile! if run_dir is later updated, this path will be wrong
        
    def restart_inputs(self):
        """
        Return a list of paths to restart data that will be used as the 
        initial condition for this run. Assumes nonmerged style of restart data.
        Paths are relative to run_dir. For MPI runs, expands paths to reflect
        all subdomains. For serial runs RestartFile is returned in a list, no
        modifications
        """
        rst_base=self.mdu['restart','RestartFile']
        path=os.path.dirname(rst_base)
        base=os.path.basename(rst_base)
        # Assume that it has the standard naming
        suffix=base[-23:] # just the date-time portion
        if self.num_procs>1:
            rsts=[ (rst_base[:-23] + '_%04d'%p + rst_base[-23:])
                   for p in range(self.num_procs)]
        else:
            self.log.warning("Handling restart data with serial run is not tested")
            rsts=[rst_base]
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

        This updates self.mdu. Should be called after self.write(), since
        update_config() alters RestartFile. It will make
        the new model run directory as necessary.
        """
        for proc,rst in enumerate(self.restart_inputs()):
            old_dir=os.path.dirname(rst)  # '../run_dye_test-p06a/DFM_OUTPUT_flowfm'
            # new_rst=os.path.join(self.mdu.output_dir(),os.path.basename(rst)) # 'run_dye_test-p06b/DFM_OUTPUT_flowfm/flowfm_0000_20161210_060000_rst.nc'
            new_rst=os.path.join(os.path.basename(rst)) # 'run_dye_test-p06b/DFM_OUTPUT_flowfm/flowfm_0000_20161210_060000_rst.nc'
            # previously this kept restart data in output_dir. Cleaner to have restart
            # data in the run_dir.
            # Now new_rst is relative to run_dir, and should be different from rst regardless
            # of deep/shallow restart
            assert rst!=new_rst
            # Note: rst and new_rst both relative to self.run_dir, but cwd may be something
            # else
            rst_abs=os.path.join(self.run_dir,rst)
            try:
                ds=xr.open_dataset(rst_abs)
            except:
                pdb.set_trace()
            new_ds=modify_ic(ds,proc=proc,model=self)
            if new_ds is None:
                new_ds=ds # assume modified in place

            new_rst_abs=os.path.join(self.run_dir,new_rst)
            dest_dir=os.path.dirname(new_rst_abs)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            new_ds.to_netcdf(new_rst_abs)
        # Update mdu to reflect new files
        old_rst_base=self.mdu['restart','RestartFile']
        new_rst_base=os.path.basename(old_rst_base) # relative to run_dir
        self.mdu['restart','RestartFile']=new_rst_base
        self.log.info(f"Updating RestartFile to {new_rst_base}")
        
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
            return

        idx=names.index(name)
        # this has a bunch of extra cruft -- some other time remove
        # the parts that are not relevant to the cross section.
        ds=his.isel(cross_section=idx)
        return self.translate_vars(ds,requested_vars=data_vars)

    def translate_vars(self,ds,requested_vars=None):
        """
        Not sure if this is the right place to handle this sort of thing.
        Trying to deal with the fact that we'd like to request 'water_level'
        from a model, but it may be named 'eta', 'waterlevel', 'sea_surface_height',
        's1', and so on.

        The interface is going to evolve here...

        For now:
        ds: xr.Dataset, presumably from model output.
        requested_vars: if present, a list of variable names that the caller 
        wants. Otherwise all data variables.

        Updates ds, try to find candidates for the requested variables.
        """
        lookup={'flow':'cross_section_discharge',
                'water_level':'waterlevel'}
        if requested_vars is None:
            requested_vars=ds.data_vars
            
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

    
# Used to be distinct from ScalarBC, but that creates problems. In particular
# we may not know whether a tracer is specifically for DWAQ or just a DFM tracer
# at the time of instantiation.
DelwaqScalarBC=hm.ScalarBC

if sys.platform=='win32':
    cls=DFlowModel
    cls.dfm_bin_exe="dflowfm-cli.exe"
    cls.mpi_bin_exe="mpiexec.exe"

if __name__=='__main__':
    import argparse, sys

    parser=argparse.ArgumentParser(description="Command line manipulation of DFM runs")

    parser.add_argument('--restart', action="store_true", help='restart a run')
    parser.add_argument('--mdu', metavar="file.mdu", default=None, 
                        help='existing MDU file')
    #parser.add_argument('--output', metavar="path", default=None, nargs=1, 
    #                    help='new output directory')
    args=parser.parse_args()

    if args.restart:
        mdu_fn=args.mdu
        if mdu_fn is None:
            mdus=glob.glob("*.mdu")
            mdus.sort()
            mdu_fn=mdus[0]
        print("Will use mdu_fn '%s' for input"%mdu_fn)
        # Super simple approach for the moment
        model=DFlowModel.load(mdu_fn)
        
        # Update MDU
        t_restart=model.restartable_time()
        if t_restart is None:
            print("Didn't find a restartable time")
            sys.exit(1)
        print("Restartable time is ",t_restart)
        # Should make this configurable.
        # For now, default to a 'shallow' restart, same run dir, same inputs,
        # same run_stop, only changing the start time, and specifying restart file.
        # Also need to add MPI support. 
        restart=model.create_restart(deep=False,mdu_suffix="r")
        restart.run_stop=model.run_stop
        restart.update_config()

        assert restart.mdu.filename != mdu_fn
        restart.mdu.write()
        print("Shallow restart: %s to %s, mdu=%s"%(restart.run_start,
                                                   restart.run_stop,
                                                   restart.mdu.filename))


def extract_transect_his(his_ds,pattern):
    """
    Helper method to create a single xr.Dataset compatible with xr_transect
    out of a group of history output locations. 
    his_ds: xr.Dataset for history output of a run.
    pattern: regular expression for the station names. For example, if the
    stations are tranA_0000, tranA_0001, ..., tranA_0099
    then pattern='tranA_00..' or just 'tranA.*'
    Station names are assumed to be sorted along the transect. Sorting is by
    python default ordering, so tranA_01 and tranA_1 are not the same.
    
    TODO: include projected velocities
    """
    import re
    # Gather station indexes for matching names
    names={}
    for i,name in enumerate(his_ds.station_name.values):
        if name in names: continue # on the off chance that names are repeated.
        if re.match(pattern,name.decode()):
            names[name]=i

    # sort names
    roster=list(names.keys())
    if len(roster)==0:
        return None
    order=np.argsort(roster)
    idxs=[ names[roster[i]] for i in order]

    extra_dims=['cross_section','gategens','general_structures','nFlowLink',
                'nNetLink','nFlowElemWithBnd','station_geom_nNodes']
    extra_dims=[d for d in extra_dims if d in his_ds.dims]
    ds=his_ds.drop_dims(extra_dims).isel(stations=idxs)

    # Make it look like an xr_transect
    dsxr=ds.rename(stations='sample',station_x_coordinate='x_sample',station_y_coordinate='y_sample')
    z_renames=dict(laydim='layer',laydimw='interface',zcoordinate_c='z_ctr',zcoordinate_w='z_int')
    # zcoordinate_c is not a dim, though it's a coordinate.
    z_renames={k:z_renames[k] for k in z_renames if (k in dsxr) or (k in dsxr.dims)}
    
    dsxr=dsxr.rename(**z_renames)
    
    # add distance?
    xy=np.c_[ dsxr.x_sample.values,
              dsxr.y_sample.values ]
    dsxr['d_sample']=('sample',),utils.dist_along(xy)

    return dsxr

# Utilities for setting grid bathymetry
def dem_to_cell_bathy(dem,g,fill_iters=20):
    """
    dem: field.SimpleGrid
    g: UnstructuredGrid
    fill_iters: how hard to try filling in missing data

    returns cell-mean values, shape=[g.Ncells()]
    """
    cell_means=np.zeros(g.Ncells(),np.float64)
    for c in utils.progress(range(g.Ncells()),msg="dem_to_cell_bathy: %s"):
        #msk=dem.polygon_mask(g.cell_polygon(c))
        #cell_means[c]=np.nanmean(dem.F[msk])
        cell_means[c]=np.nanmean(dem.polygon_mask(g.cell_polygon(c),return_values=True))
    
    for _ in range(fill_iters):
        missing=np.nonzero(np.isnan(cell_means))[0]
        if len(missing)==0:
            break
        new_depths=[]
        print("filling %d missing cell depths"%len(missing))
        for c in missing:
            new_depths.append( np.nanmean(cell_means[g.cell_to_cells(c)]) )
        cell_means[missing]=new_depths
    else:
        print("Filling still left %d nan cell elevations"%len(missing))
    return cell_means
    
def dem_to_cell_node_bathy(dem,g,cell_z=None):
    """
    dem: field.SimpleGrid
    g: UnstructuredGrid
    cell_z: optional precomputed cell values

    Extract cell-mean values from dem, and map to nodes.
    The mapping inverts the node->cell conversion that can happen
    in DFM (depending on bedlevtype).
    For example, bedlevtype 6 averages node values to get cell values, then
    averages adjacent cells to get edge values. This code will find node values
    that approximate the dem-based cell value.

    This is not bulletproof!  In small-ish domains it can help with maintaining
    conveyance in small channels. In some cases there is a tough tradeoff between
    the damping (which favors small node movements, but tends to spread errors
    out) and no damping (favors small errors, and spreads large node movements out
    over many nodes).
    """
    if cell_z is None:
        cell_z=dem_to_cell_bathy(dem,g)
    
    V=[]
    I=[]
    J=[]
    for c in utils.progress(range(g.Ncells())):
        nodes=g.cell_to_nodes(c)
        val=1./len(nodes)
        V.append( [val]*len(nodes) )
        I.append( [c]*len(nodes))
        J.append( nodes )
        # Equivalent to this for a dok_matrix:
        # node_z_to_cell_z[c,nodes]=val
        # But 10x faster
    V=np.concatenate(V)
    I=np.concatenate(I)
    J=np.concatenate(J)
    node_z_to_cell_z=sparse.coo_matrix( (V,(I,J)), shape=(g.Ncells(), g.Nnodes()))

    # A x = b
    # A: node_z_to_cell_z
    #  x: node_z
    #    b: cell_z
    # to better allow regularization, change this to a node elevation update.
    # A ( node_z0 + node_delta ) = cell_z
    # A*node_delta = cell_z - A*node_z0 
    
    node_z0=dem(g.nodes['x'])
    bad_nodes=np.isnan(node_z0)
    node_z0[bad_nodes]=0.0 # could come up with something better..
    if np.any(bad_nodes):
        print("%d bad node elevations"%bad_nodes.sum())
    b=cell_z - node_z_to_cell_z.dot(node_z0)

    # damp tries to keep the adjustments to O(2m)
    res=sparse.linalg.lsqr(node_z_to_cell_z.tocsr(),b,damp=0.05)
    node_delta, istop, itn, r1norm  = res[:4]
    print("Adjustments to node elevations are %.2f to %.2f"%(node_delta.min(),
                                                             node_delta.max()))
    final=node_z0+node_delta
    if np.any(np.isnan(final)):
        print("Bad news")
        import pdb
        pdb.set_trace()
    return final
    
    

def rst_mappers(rst,g,signed=True):
    """
    Create a sparse matrix that maps a per-flowlink, signed quantity (i.e. flow)
    or unsigned (salinity?)
    ported from waq_scenario.py
    rst: an xr.Dataset from a DFM restart file
    """
    M=sparse.dok_matrix( (g.Nedges(),rst.dims['nFlowLink']), np.float64)
    Melem=sparse.dok_matrix( (g.Ncells(),rst.dims['nFlowElem']), np.float64)
    e2c=g.edge_to_cells(recalc=True)
    cc=g.cells_center()
    elem_xy=np.c_[ rst.FlowElem_xzw.values,
                   rst.FlowElem_yzw.values ]

    def elt_to_cell(elt):
        # in general elts are preserved as the same cell index,
        # and this is actually more robust than the geometry
        # check because of some non-orthogonal cells that have
        # a circumcenter outside the corresponding cell.
        if utils.dist(elem_xy[elt] - cc[elt])<2.0:
            Melem[elt,elt]=1
            return elt
        # in a few cases the circumcenter is not inside the cell,
        # so better to select the nearest circumcenter than the
        # cell containing it.
        c=g.select_cells_nearest(elem_xy[elt],inside=False)
        assert c is not None
        Melem[c,elt]=1
        return c

    flow_links0=rst.FlowLink.values-1
    for link,(eltA,eltB) in utils.progress(enumerate(flow_links0)):
        assert eltB>=0
        cB=elt_to_cell(eltB)

        if (eltA<0) or (eltA>=rst.dims['nFlowElem']): # it's a boundary.
            # so find a boundary edge for that cell
            for j in g.cell_to_edges(cB):
                if e2c[j,0]<0:
                    sgn=1
                    break
                elif e2c[j,1]<0:
                    sgn=-1
                    break
            else:
                print("Link %d -- %d does not map to a grid boundary, likely a discharge, and will be ignored."%(eltA,eltB))
                # This is probably a discharge. Ignore it.
                continue
        else:
            cA=elt_to_cell(eltA)
            j=g.cells_to_edge(cA,cB)
            if j is None:
                raise Exception("%d to %d was not an edge in the grid"%(eltA,eltB))
            if (e2c[j,0]==cA) and (e2c[j,1]==cB):
                # positive DWAQ flow is A->B
                # positive edge normal for grid is the same
                sgn=1
            elif (e2c[j,1]==cA) and (e2c[j,0]==cB):
                sgn=-1
            else:
                raise Exception("Bad match on link->edge")
        if not signed: 
            sgn=1
        M[j,link]=sgn
    return M,Melem


def source_sink_add_tracers(model,bc,new_values=[],orig_num_values=3):
    """
    model: DFlowModel instance, or the run_dir
    bc: dictionary for the source/sink entry as returned from load_bcs.
    
    Add additional columns to a source/sink data file. Makes a backup of the
    original tim file. If a backup is already present, it will not be
    overwritten, and will be read to get the original data. 
    
    So if the new run will include two dwaq tracers, pass new_values=[0,1]
    (which would tag sources with 0 for the first and 1.0 for the second)
    orig_num_values: 3 for run with salinity and temperature. I think
    less than that if temperature and/or salinity are disabled. 
    """
    if isinstance(model,str):
        run_dir=model
    else:
        run_dir=model.run_dir
        
    pli_fn=os.path.join(model.run_dir,bc['FILENAME'])
    assert pli_fn.lower().endswith('.pli')
    fn=pli_fn[:-4] + ".tim"
    assert os.path.exists(fn)
    fn_orig=fn+".orig"
    if not os.path.exists(fn_orig):
        shutil.copyfile(fn,fn_orig)
    data_orig=np.loadtxt(fn_orig)
    # drop previous forcing for new tracers. leaving time column and the original Q,S,T values
    columns=[data_orig[:,:1+orig_num_values]] 
    for new_val in new_values:
        columns.append( np.full(data_orig.shape[0],new_val))
    data=np.column_stack(columns)
    np.savetxt(fn,data,fmt="%.6g")
