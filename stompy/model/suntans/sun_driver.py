import os
import glob
import copy
import subprocess

import six
from collections import defaultdict
import re
import xarray as xr
import numpy as np
import datetime
from matplotlib.dates import date2num, num2date

from ... import utils, memoize
#from ..delft import dflow_model as dfm
from .. import hydro_model as hm
from ..delft import dfm_grid
from ...grid import unstructured_grid
from ...spatial import linestring_utils

from . import store_file

import logging as log

try:
    import pytz
    utc = pytz.timezone('utc')
except ImportError:
    log.warning("Couldn't load utc timezone")
    utc = None

datenum_precision_per_s = 100 # 10ms  - should be evenly divisible into 1e6

def dt_round(dt):
    """ Given a datetime or timedelta object, round it to datenum_precision
    """
    if isinstance(dt,datetime.timedelta):
        td = dt
        # days are probably fine
        dec_seconds = td.seconds + 1e-6 * td.microseconds
        # the correct number of time quanta
        quanta = int(round(dec_seconds * datenum_precision_per_s))

        # how to get that back to an exact number of seconds?
        new_seconds = quanta // datenum_precision_per_s
        # careful to keep it integer arithmetic
        us_per_quanta = 1000000 // datenum_precision_per_s
        new_microseconds = (quanta % datenum_precision_per_s) * us_per_quanta

        return datetime.timedelta( days=td.days,
                                   seconds = new_seconds,
                                   microseconds = new_microseconds )
    else:
        # same deal, but the fields have slightly different names
        # And the integer arithmetic cannot be used to count absolute seconds -
        # that will overflow 32-bit ints (okay with 64, but better not
        # to assume 64-bit ints are available)
        dec_seconds = dt.second + 1e-6 * dt.microsecond
        quanta = int(round(dec_seconds * datenum_precision_per_s))

        # how to get that back to an exact number of seconds?
        new_seconds = quanta // datenum_precision_per_s
        # careful to keep it integer arithmetic
        us_per_quanta = 1000000// datenum_precision_per_s
        new_microseconds = (quanta % datenum_precision_per_s) * us_per_quanta

        # to handle the carries between microseconds, seconds, days,
        # construct an exact timedelta object - also avoids having to do
        # int arithmetic with seconds over many days, which could overflow.
        td = datetime.timedelta(seconds = new_seconds - dt.second,
                                microseconds = new_microseconds - dt.microsecond)

        return dt + td

# certainly there is a better way to do this...
MultiBC=hm.MultiBC
StageBC=hm.StageBC
FlowBC=hm.FlowBC
VelocityBC=hm.VelocityBC
ScalarBC=hm.ScalarBC
SourceSinkBC=hm.SourceSinkBC
OTPSStageBC=hm.OTPSStageBC
OTPSFlowBC=hm.OTPSFlowBC
OTPSVelocityBC=hm.OTPSVelocityBC
HycomMultiVelocityBC=hm.HycomMultiVelocityBC
HycomMultiScalarBC=hm.HycomMultiScalarBC
NOAAStageBC=hm.NOAAStageBC
NwisFlowBC=hm.NwisFlowBC
NwisStageBC=hm.NwisStageBC
CdecFlowBC=hm.CdecFlowBC
CdecStageBC=hm.CdecStageBC

class GenericConfig(object):
    """ Handles reading and writing of suntans.dat formatted files.
    Older code I think was case-insensitive, but seems that it is
    now case-sensitive.
    """
    keys_are_case_sensitive=True

    def __init__(self,filename=None,text=None):
        """ filename: path to file to open and parse
            text: a string containing the entire file to parse
        """
        self.filename = filename

        if filename:
            fp = open(filename,'rt')
        else:
            fp = [s+"\n" for s in text.split("\n")]

        self.entries = {}
        self.originals = []

        for line in fp:
            # save original text so we can write out a new suntans.dat with
            # only minor changes
            self.originals.append(line)
            i = len(self.originals)-1

            m = re.match(r"^\s*((\S+)\s+(\S+))?\s*.*",line)
            if m and m.group(1):
                key = m.group(2)
                if not self.keys_are_case_sensitive:
                    key=key.lower()
                val = m.group(3)
                self.entries[key] = [val,i]
        if filename:
            fp.close()

    def copy(self):
        # punt copy semantics and handling to copy module
        return copy.deepcopy(self)

    def conf_float(self,key):
        return self.conf_str(key,float)
    def conf_int(self,key,default=None):
        x=self.conf_str(key,int)
        if x is None:
            return default
        return x
    def conf_str(self,key,caster=lambda x:x):
        if not self.keys_are_case_sensitive:
            key = key.lower()

        if key in self.entries:
            return caster(self.entries[key][0])
        else:
            return None

    def __setitem__(self,key,value):
        self.set_value(key,value)
    def __getitem__(self,key):
        return self.conf_str(key)
    def __delitem__(self,key):
        # if the line already exists, it will be written out commented, otherwise
        # it won't be written at all.
        self.set_value(key,None)
    def __contains__(self,key):
        return self[key] is not None
    def get(self,key,default=None):
        if key in self:
            return self[key]
        else:
            return default
    def __eq__(self,other):
        return self.is_equal(other)
    def is_equal(self,other,limit_to_keys=None):
        # key by key equality comparison:
        log.debug("Comparing two configs")
        for k in self.entries.keys():
            if limit_to_keys and k not in limit_to_keys:
                continue
            if k not in other.entries:
                log.debug("Other is missing key %s"%k)
                return False
            elif self.val_to_str(other.entries[k][0]) != self.val_to_str(self.entries[k][0]):
                log.debug("Different values key %s => %s, %s"%(k,self.entries[k][0],other.entries[k][0]))
                return False
        for k in other.entries.keys():
            if limit_to_keys and k not in limit_to_keys:
                continue
            if k not in self.entries:
                log.debug("other has extra key %s"%k)
                return False
        return True

    def disable_value(self,key):
        if not self.keys_are_case_sensitive:
            key = key.lower()
        if key not in self.entries:
            return
        old_val,i = self.entries[key]

        self.originals[i] = "# %s"%(self.originals[i])
        self.entries[key][0] = None

    def val_to_str(self,value):
        # make sure that floats are formatted with plenty of digits:
        # and handle annoyance of standard Python types vs. numpy types
        # But None stays None, as it gets handled specially elsewhere
        if value is None:
            return None
        if isinstance(value,float) or isinstance(value,np.floating):
            value = "%.12g"%value
        else:
            value = str(value)
        return value

    def set_value(self,key,value):
        """ Update a value in the configuration.  Setting an item to None will
        comment out the line if it already exists, and omit the line if it does
        not yet exist.
        """
        if not self.keys_are_case_sensitive:
            key = key.lower()
        else:
            if (key not in self.entries):
                for other in self.entries:
                    if key.lower()==other.lower():
                        raise Exception("Probably a case-sensitive error: %s vs %s"%(key,other))

        if key not in self.entries:
            if value is None:
                return
            self.originals.append("# blank #")
            i = len(self.originals) - 1
            self.entries[key] = [None,i]

        old_val,i = self.entries[key]

        value = self.val_to_str(value)

        if value is not None:
            self.originals[i] = "%s   %s # from sunreader code\n"%(key,value)
        else:
            self.originals[i] = "# " + self.originals[i]

        self.entries[key][0] = value

    def write_config(self,filename=None,check_changed=True,backup=True):
        """
        Write this config out to a text file
        filename: defaults to self.filename
        check_changed: if True, and the file already exists and is not materially different,
          then do nothing.  Good for avoiding unnecessary changes to mtimes.
        backup: if true, copy any existing file to <filename>.bak
        """
        filename = filename or self.filename
        if filename is None:
            raise Exception("No clue about the filename for writing config file")

        if check_changed:
            if os.path.exists(filename):
                existing_conf = self.__class__(filename)
                if existing_conf == self:
                    log.debug("No change in config")
                    return

        if os.path.exists(filename) and backup:
            filename_bak = filename + ".bak"
            os.rename(filename,filename_bak)

        fp = open(filename,'wt')
        for line in self.originals:
            fp.write(line)
        fp.close()

class SunConfig(GenericConfig):
    def time_zero(self):
        """ return python datetime for the when t=0 is"""

        # try the old way, where these are separate fields:
        start_year = self.conf_int('start_year')
        start_day  = self.conf_float('start_day')
        if start_year is not None:
            # Note: we're dealing with 0-based start days here.
            start_datetime = datetime.datetime(start_year,1,1,tzinfo=utc) + dt_round(datetime.timedelta(start_day))
            return start_datetime

        # That failed, so try the other way
        log.debug("Trying the new way of specifying t0")
        s = self.conf_str('TimeZero') # 1999-01-01-00:00
        start_datetime = datetime.datetime.strptime(s,'%Y-%m-%d-%H:%M')
        start_datetime = start_datetime.replace(tzinfo=utc)
        return start_datetime

    def simulation_seconds(self):
        return self.conf_float('dt') * self.conf_int('nsteps')

    def timestep(self):
        """ Return a timedelta object for the timestep - should be safe from roundoff.
        """
        return dt_round( datetime.timedelta(seconds=self.conf_float('dt')) )

    def simulation_period(self):
        """ This is more naive than the SunReader simulation_period(), in that
        it does *not* look at any restart information, just start_year, start_day,
        dt, and nsteps

        WARNING: this used to add an extra dt to start_date - maybe trying to make it
        the time of the first profile output??  this seems like a bad idea.  As of
        Nov 18, 2012, it does not do that (and at the same time, moves to datetime
        arithmetic)

        return a  pair of python datetime objects for the start and end of the simulation.
        """
        t0 = self.time_zero()

        # why did it add dt here???
        # start_date = t0 + datetime.timedelta( self.conf_float('dt') / (24.*3600) )
        # simulation_days = self.simulation_seconds() / (24.*3600)
        # end_date   = start_date + datetime.timedelta(simulation_days)

        start_date = t0
        end_date = start_date + self.conf_int('nsteps')*self.timestep()

        return start_date,end_date

    def copy_t0(self,other):
        self.set_value('start_year',other.conf_int('start_year'))
        self.set_value('start_day',other.conf_float('start_day'))

    # def set_simulation_period(self,start_date,end_date):
    #     """ Based on the two python datetime instances given, sets
    #     start_day, start_year and nsteps
    #     """
    #     self.set_value('start_year',start_date.year)
    #     t0 = datetime.datetime( start_date.year,1,1,tzinfo=utc )
    #     self.set_value('start_day',date2num(start_date) - date2num(t0))
    #
    #     # roundoff dangers here -
    #     # self.set_simulation_duration_days( date2num(end_date) - date2num(start_date))
    #     self.set_simulation_duration(delta=(end_date - start_date))
    #
    # def set_simulation_duration_days(self,days):
    #     self.set_simulation_duration(days=days)
    # def set_simulation_duration(self,
    #                             days=None,
    #                             delta=None,
    #                             seconds = None):
    #     """ Set the number of steps for the simulation - exactly one of the parameters should
    #     be specified:
    #     days: decimal number of days - DANGER - it's very easy to get some round-off issues here
    #     delta: a datetime.timedelta object.
    #       hopefully safe, as long as any differencing between dates was done with UTC dates
    #       (or local dates with no daylight savings transitions)
    #     seconds: total number of seconds - this should be safe, though there are some possibilities for
    #       roundoff.
    #
    #     """
    #     print("Setting simulation duration:")
    #     print("  days=",days)
    #     print("  delta=",delta)
    #     print("  seconds=",seconds)
    #
    #     # convert everything to a timedelta -
    #     if (days is not None) + (delta is not None) + (seconds is not None) != 1:
    #         raise Exception("Exactly one of days, delta, or seconds must be specified")
    #     if days is not None:
    #         delta = datetime.timedelta(days=days)
    #     elif seconds is not None:
    #         delta = datetime.timedelta(seconds=seconds)
    #
    #     # assuming that dt is also a multiple of the precision (currently 10ms), this is
    #     # safe
    #     delta = dt_round(delta)
    #     print("  rounded delta = ",delta)
    #     timestep = dt_round(datetime.timedelta(seconds=self.conf_float('dt')))
    #     print("  rounded timestep =",timestep)
    #
    #     # now we have a hopefully exact simulation duration in integer days, seconds, microseconds
    #     # and a similarly exact timestep
    #     # would like to do this:
    #     #   nsteps = delta / timestep
    #     # but that's not supported until python 3.3 or so
    #     def to_quanta(td):
    #         """ return integer number of time quanta in the time delta object
    #         """
    #         us_per_quanta = 1000000 // datenum_precision_per_s
    #         return (td.days*86400 + td.seconds)*datenum_precision_per_s + \
    #                int( round( td.microseconds/us_per_quanta) )
    #     quanta_timestep = to_quanta(timestep)
    #     quanta_delta = to_quanta(delta)
    #
    #     print("  quanta_timestep=",quanta_timestep)
    #     print("  quanta_delta=",quanta_delta)
    #     nsteps = quanta_delta // quanta_timestep
    #
    #     print("  nsteps = ",nsteps)
    #     # double-check, going back to timedelta objects:
    #     err = nsteps * timestep - delta
    #     self.set_value('nsteps',int(nsteps))
    #     print("Simulation duration requires %i steps (rounding error=%s)"%(self.conf_int('nsteps'),err))

    def is_grid_compatible(self,other):
        """ Compare two config's, and return False if any parameters which would
        affect grid partitioning/celldata/edgedata/etc. are different.
        Note that differences in other input files can also cause two grids to be different,
        esp. vertspace.dat
        """
        # keep all lowercase
        keys = ['Nkmax',
                'stairstep',
                'rstretch',
                'CorrectVoronoi',
                'VoronoiRatio',
                'vertgridcorrect',
                'IntDepth',
                'pslg',
                'points',
                'edges',
                'cells',
                'depth',
                # 'vertspace.dat.in' if rstretch==0
                'topology.dat',
                'edgedata',
                'celldata',
                'vertspace.dat']
        return self.is_equal(other,limit_to_keys=keys)

class SuntansModel(hm.HydroModel):
    # Annoying, but suntans doesn't like signed elevations
    # this offset will be applied to grid depths and freesurface boundary conditions.
    # This is error prone, though, and makes it difficult to "round-trip"
    # grid information.  In particular, if a new run is created by loading an old
    # run, there will be an issue where the grid may get z_offset applied twice.
    # This should be reimplemented as a z_datum.  So no behind-the-scenes offsets,
    # just have a standardized place for saying that my model's z=0 is z_offset
    # from z_datum, e.g. z_datum='NAVD88' and z_offset.
    # maybe the appropriate thing is a dictionary, mapping datum names to offsets.
    # like z_datum['NAVD88']=-5.
    z_offset=0.0
    ic_ds=None
    bc_ds=None
    met_ds=None
    
    # None: not a restart, or
    # path to suntans.dat for the run being restarted, or True if this is
    # a restart but we don't we have a separate directory for the restart,
    # just StartFiles
    restart=None
    restart_model=None # model instance being restarted
    restart_symlink=True # default to symlinking restarts

    # for partition, run, etc.
    sun_bin_dir=None
    mpi_bin_dir=None

    # 'auto': the grid and projection information will be used to
    # update the coriolis parameter.
    # None: leave whatever value is in the template
    # <float>: use that as the coriolis parameter
    coriolis_f='auto'

    # experimental -- not yet working.
    # the suntans code does not yet remap edge data from the original
    # order to the -g ordering (which is different, even when running
    # single-core). 
    use_edge_depths=False # write depth data per-edge in a separate file.

    def __init__(self,*a,**kw):
        self.load_template(os.path.join(os.path.dirname(__file__),"data","suntans.dat"))
        super(SuntansModel,self).__init__(*a,**kw)

    def configure(self):
        super(SuntansModel,self).configure()
        
        if self.restart_model is not None:
            self.set_restart_from(self.restart_model)
        
    @property
    def time0(self):
        self.config['starttime']
        dt=datetime.datetime.strptime(self.config['starttime'],
                                      "%Y%m%d.%H%M%S")
        return utils.to_dt64(dt)

    def create_restart(self,symlink=True):
        new_model=self.__class__() # in case of subclassing
        # SuntansModel()
        new_model.config=self.config.copy()
        # things that have to match up, but are not part of the config:
        new_model.num_procs=self.num_procs
        new_model.restart=self.config_filename
        new_model.restart_model=self
        new_model.restart_symlink=symlink
        # There is some extra machinery in load_grid(...) to get the right cell and
        # edge depths -- this call would lose those
        # new_model.set_grid(unstructured_grid.UnstructuredGrid.read_suntans(self.run_dir))
        # So copy the grid we already have.
        # UnstructuredGrid.copy() is naive and doesn't get all the depth fields, so
        # here just pass self.grid, even though it may get mutated.
        new_model.set_grid(self.grid)
        new_model.run_start=self.restartable_time()
        return new_model

    @classmethod
    def run_completed(cls,fn):
        """
        fn: path to either folder containing suntans.dat, or path
        to suntans.dat itself.

        returns: True if the file exists and the folder contains a run which
          ran to completion. Otherwise False.
        """
        if not os.path.exists(fn):
            return False
        if os.path.isdir(fn):
            fn=os.path.join(fn,"suntans.dat")
            if not os.path.exists(fn):
                return False

        model=cls.load(fn)
        if model is None:
            return False
        return model.is_completed()
    def is_completed(self):
        step_fn=os.path.join(self.run_dir,self.config['ProgressFile'])
        if not os.path.exists(step_fn):
            return False
        with open(step_fn,'rt') as fp:
            progress=fp.read()
        return "100% Complete" in progress

    def set_grid(self,grid):
        """
        read/load grid, check for depth data and edge marks.
        This does not apply the z_offset -- that is only
        applied during writing out the rundata.
        """
        if isinstance(grid,six.string_types):
            # step in and load as suntans, rather than generic
            grid=unstructured_grid.SuntansGrid(grid)

        # depending on the source of the grid, it may need edges flipped
        # to be consistent with suntans expectations that nc1 is always into
        # the domain, and nc2 may be external
        grid.orient_edges()
        super(SuntansModel,self).set_grid(grid)

        # 2019-05-29: trying to transition to using z for elevation, since
        # 'depth' has a positive-down connotation
        
        # make sure we have the fields expected by suntans
        if 'z_bed' not in grid.cells.dtype.names:
            if 'depth' in grid.cells.dtype.names:
                self.log.warning("For now, assuming that cells['depth'] is positive up")
                cell_z_bed=grid.cells['depth']
            elif 'z_bed' in grid.nodes.dtype.names:
                cell_z_bed=grid.interp_node_to_cell(grid.nodes['z_bed'])
                # and avoid overlapping names
                grid.delete_node_field('z_bed')
            elif 'depth' in grid.nodes.dtype.names:
                cell_z_bed=grid.interp_node_to_cell(grid.nodes['depth'])
                self.log.warning("For now, assuming that nodes['depth'] is positive up")
            else:
                self.log.warning("No depth information in grid nodes or cells.  Creating zero-depth")
                cell_z_bed=np.zeros(grid.Ncells(),np.float64)
            grid.add_cell_field('z_bed',cell_z_bed)

        # with the current suntans version, depths are on cells, but model driver
        # code in places wants an edge depth.  so copy those here.
        e2c=grid.edge_to_cells() # this is assumed in other parts of the code that do not recalculate it.
        nc1=e2c[:,0].copy()   ; nc2=e2c[:,1].copy()
        nc1[nc1<0]=nc2[nc1<0] ; nc2[nc2<0]=nc1[nc2<0]
        # edge depth is shallower of neighboring cells
        # these depths are still positive up, though.
        edge_z_bed=np.maximum(grid.cells['z_bed'][nc1],grid.cells['z_bed'][nc2])

        if 'edge_z_bed' in grid.edges.dtype.names:
            deep_edges=(grid.edges['edge_z_bed']<edge_z_bed)
            if np.any(deep_edges):
                self.log.info("%d edges had a specified depth deeper than neighboring cells.  Replaced them"%
                              deep_edges.sum())
            grid.edges['edge_z_bed'][deep_edges]=edge_z_bed[deep_edges]
        else:        
            grid.add_edge_field('edge_z_bed',edge_z_bed)
        
        if 'mark' not in grid.edges.dtype.names:
            mark=np.zeros( grid.Nedges(), np.int32)
            grid.add_edge_field('mark',mark)
        self.grid=grid
        self.set_default_edge_marks()

    def set_default_edge_marks(self):
        # update marks to a reasonable starting point
        e2c=self.grid.edge_to_cells()
        bc_edge=e2c.min(axis=1)<0
        mark=self.grid.edges['mark']
        mark[mark<0] = 0
        mark[ (mark==0) & bc_edge ] = 1
        # allow other marks to stay
        self.grid.edges['mark'][:]=mark

    def edge_depth(self,j,datum=None):
        """
        Return the bed elevation for edge j, in meters, positive=up.
        Suntans implementation relies on set_grid() having set edge depths
        to be the min. of neighboring cells
        """
        z=self.grid.edges['edge_z_bed'][j]

        if datum is not None:
            if datum=='eta0':
                z+=self.initial_water_level()
        return z

    @classmethod
    def load(cls,fn,load_grid=True,load_met=False,load_ic=False,load_bc=False):
        """
        Open an existing model setup, from path to its suntans.dat
        return None if run could not be loaded.

        load_met: if true, load an existing Met netcdf file to self.met_ds
        load_ic: likewise for initial conditions
        load_bc: likewise for boundary conditions
        """
        model=cls()
        if os.path.isdir(fn):
            fn=os.path.join(fn,'suntans.dat')
        if not os.path.exists(fn):
            return None
        
        model.load_template(fn)
        model.set_run_dir(os.path.dirname(fn),mode='existing')
        # infer number of processors based on celldata files
        # for restarts, this is overridden in infer_restart() by looking
        # at the number of restart files, since in some scripts those
        # are created earlier, while the celldata files aren't created until
        # partition is called.
        sub_cells=glob.glob( os.path.join(model.run_dir,'celldata.dat.*') )
        if len(sub_cells)>0:
            model.num_procs=len(sub_cells)
        else:
            # probably better to test whether it has even been processed
            model.num_procs=1
        model.infer_restart()
        model.set_times_from_config()
        # This will need some tweaking to fail gracefully
        if load_grid:
            try:
                model.load_grid()
            except OSError:
                # this may be too strict -- a multiproc run could be fine but not
                # necessarily have the global grid.
                return None

        if load_met:
            model.load_met_ds()
        if load_ic:
            model.load_ic_ds()
        if load_bc:
            model.load_bc_ds()

        return model
    def load_grid(self):
        """
        Set self.grid from existing suntans-format grid in self.run_dir.
        """
        g=unstructured_grid.UnstructuredGrid.read_suntans(self.run_dir)

        # hacked in support to read cell depths
        cell_depth_fn=self.file_path('depth')+"-voro"
        if ( ('z_bed' not in g.cells.dtype.names)
             and
             (os.path.exists(cell_depth_fn)) ):
            self.log.debug("Will read cell depths, too")
            cell_xyz=np.loadtxt(cell_depth_fn)
            assert cell_xyz.shape[0]==g.Ncells(),"%s didn't have the right number of cells (%d vs %d)"%(cell_depth_fn,
                                                                                                        cell_xyz.shape[0],
                                                                                                        g.Ncells())
            # cell centers can be a bit lenient in case there are centroid vs. circumcenter vs nudged
            # differences.
            if not np.allclose(cell_xyz[:,:2], g.cells_center()):
                self.log.warning("%s cell locations don't match grid"%cell_depth_fn)
                self.log.warning("Will forge ahead nevertheless")
            # on disk these are positive down, but model driver convention is positive up
            # (despite being called depth...)
            g.add_cell_field('z_bed',-cell_xyz[:,2]) 
            g.add_cell_field('depth',-cell_xyz[:,2]) # will be phased out
            
        # hacked in support to read depth on edges
        edge_depth_fn=self.file_path('depth')+"-edge"
        if ( ('edge_z_bed' not in g.edges.dtype.names)
             and
             (os.path.exists(edge_depth_fn)) ):
            self.log.debug("Will read edge depths, too")
            edge_xyz=np.loadtxt(edge_depth_fn)
            assert edge_xyz.shape[0]==g.Nedges(),"%s didn't have the right number of edges (%d vs %d)"%(edge_depth_fn,
                                                                                                        edge_xyz.shape[0],
                                                                                                        g.Nedges())
            assert np.allclose(edge_xyz[:,:2], g.edges_center()),"%s edge locations don't match"%edge_depth_fn
            # on disk these are positive down, but model driver convention is positive up
            # (despite being called depth...)  -- in the process of using edge_z_bed in the driver r
            # script to make the sign convention more apparent.
            # g.add_edge_field('edge_depth',-edge_xyz[:,2]) # being phased out
            g.add_edge_field('edge_z_bed',-edge_xyz[:,2])
        
        self.set_grid(g)
            
        return g
    
    def infer_restart(self):
        """
        See if this run is a restart.
        Sets self.restart to:
          None: not a restart
          True: is a restart, but insufficient information to find the parent run
          string: path to suntans.dat for the parent run
        """
        if self.config['StartFile'] is None:
            # Possibly not a valid config file
            self.restart=None
            return
        
        start_path=os.path.join(self.run_dir,self.config['StartFile']+".0")
        if os.path.exists(start_path):
            log.debug("Looks like a restart")
            self.restart=True

            # Get num_procs from the number of restart files.
            for proc in range(1024):
                fn=os.path.join(self.run_dir,self.config['StartFile']+".%d"%proc)
                if not os.path.exists(fn):
                    break
            self.num_procs=proc
            log.debug("Restart appears to have %d subdomains"%self.num_procs)

            if os.path.islink(start_path):
                start_path=os.path.realpath(start_path)
                parent_dir=os.path.dirname(start_path)
                assert not os.path.samefile(parent_dir,self.run_dir)
                parent_sun=os.path.join(parent_dir,"suntans.dat")
                if os.path.exists(parent_sun):
                    log.debug("And the previous suntans.dat: %s"%parent_sun)
                    self.restart=parent_sun
                else:
                    log.info("Checked for %s but no luck"%parent_sun)
            else:
                log.info("Restart file %s is not a link"%start_path)
        else:
            log.debug("Does not look like a restart based on %s"%start_path)
            self.restart=None

    def chain_restarts(self,count=None,load_grid=False):
        """
        return a list of up to count (None: unlimited) Model instances
        in forward chronological order of consecutive restarts.
        load_grid: defaults to *not* loading the grid of the earlier runs.
        The last item is always self.

        count: either the count of how many runs to return, or a np.datetime64
          such that we'll go back to a run covering that date if possible.
          if this is a tuple of datetimes, only return the runs covering that time
          range.
        """
        runs=[self]
        run=self
        while 1:
            if isinstance(count,np.datetime64):
                if runs[0].run_start <=count:
                    break
            elif isinstance(count,tuple):
                if runs[0].run_start < count[0]:
                    break
            elif count and len(runs)>=count:
                break
            run.infer_restart()
            if run.restart and run.restart is not True:
                run=SuntansModel.load(run.restart,load_grid=load_grid)
                runs.insert(0,run)
            else:
                break

        if isinstance(count,tuple):
            # Trim runs coming after the requested period
            runs=[run for run in runs if run.run_start<count[1]]
            if len(runs)==0:
                log.warning("chain_restarts wound up with zero runs for count=%s"%str(count))
        return runs

    def chain_start(self,count=None):
        """
        Analog of run_start, but across chained restarts.
        count is passed to chain_restarts().
        """
        runs=self.chain_restarts(count=count)
        return runs[0].run_start
    def chain_stop(self,count=None):
        """
        Analog of run_stop, but across chained restarts.
        Included for completeness, but this is always the same
        as self.run_stop (since we don't chain forward in time).
        """
        return self.run_stop

    def load_template(self,fn):
        self.template_fn=fn
        self.config=SunConfig(fn)

    def set_run_dir(self,path,mode='create'):
        assert mode!='clean',"Suntans driver doesn't know what clean is"
        return super(SuntansModel,self).set_run_dir(path,mode)

    def file_path(self,key,proc=None):
        fn=os.path.join(self.run_dir,self.config[key])
        if proc is not None:
            fn+=".%d"%proc
        return fn

    @property
    def config_filename(self):
        return os.path.join(self.run_dir,"suntans.dat")

    def write_config(self):
        log.info("Writing config to %s"%self.config_filename)
        self.config.write_config(self.config_filename)

    def write_monitor(self):
        if not self.mon_points: return

        xys=[ np.array(feat['geom']) for feat in self.mon_points]
        valid_xys=[xy
                   for xy in xys
                   if self.grid.select_cells_nearest(xy,inside=True) is not None]
        np.savetxt( os.path.join(self.run_dir,self.config['DataLocations']),
                    np.array(valid_xys) )

    def write(self):
        self.update_config()
        self.write_config()
        self.write_monitor()
        self.write_extra_files()
        self.write_forcing()
        # Must come after write_forcing() to allow BCs to modify grid
        self.write_grid()

        # Must come after write_forcing(), to get proper grid and to
        # have access to freesurface BCs
        if self.restart:
            self.log.info("Even though this is a restart, write IC")
        # There are times that it is useful to be able to read the IC
        # back in, e.g. to set a boundary condition equal to its initial
        # condition.  For a restart, this would ideally be the same state
        # as in the StartFiles.  That's going to take some work for
        # relatively little gain.  So just create the IC as if this was
        # not a restart.
        self.write_ic()

        if self.restart:
            self.write_startfiles()

    def initialize_initial_condition(self):
        """
        Populate self.ic_ds with a baseline initial condition.
        This should be called after all boundary conditions are in place.
        """
        self.ic_ds=self.zero_initial_condition()
        self.set_initial_h_from_bc()

    def write_ic(self):
        """
        Will have to think about how best to order this -- really need
        to set this as a zero earlier on, and then have some known time
        for the script to modify it, before finally writing it out here.
        """
        # Creating an initial condition netcdf file:
        if self.ic_ds is None:
            self.initialize_initial_condition()
        self.write_ic_ds()

    def write_startfiles(self):
        src_base=os.path.join(os.path.dirname(self.restart),
                              self.restart_model.config['StoreFile'])
        dst_base=os.path.join(self.run_dir,self.config['StartFile'])

        for proc in range(self.num_procs):
            src=src_base+".%d"%proc
            dst=dst_base+".%d"%proc
            self.restart_copier(src,dst)

    def copy_ic_to_bc(self,ic_var,bc_var):
        """
        Copy IC values to the boundary conditions

        Copies data for the given IC variable (e.g. 'salt'), to
        open and flow boundaries for bc_var (e.g. 'S').
        for flow boundaries, 'boundary_' is prepended to bc_var.

        The initial condition is copied into bc_ds for all time steps,
        and all layers.
        """

        # Open boundaries
        for ci,c in enumerate(utils.progress(self.bc_ds.cellp.values,msg="IC=>Open BCs")):
            ic_values = self.ic_ds[ic_var].values[0,:,c]
            self.bc_ds[bc_var].isel(Ntype3=ci).values[:,:]=ic_values[None,:]

        # Flow boundaries
        for ei,e in enumerate(utils.progress(self.bc_ds.edgep.values,msg="IC=>Flow BCs")):
            c=self.grid.edges['cells'][e,0]
            assert c>=0,"Is this edge flipped"
            ic_values=self.ic_ds[ic_var].values[0,:,c]
            self.bc_ds["boundary_"+bc_var].isel(Ntype2=ei).values[:,:]=ic_values[None,:]

    def write_ic_ds(self):
        self.ic_ds.to_netcdf( os.path.join(self.run_dir,self.config['initialNCfile']) )
    def load_ic_ds(self):
        fn=os.path.join(self.run_dir,self.config['initialNCfile'])
        if not os.path.exists(fn): return False
        self.ic_ds=xr.open_dataset(fn)
        
    def set_initial_h_from_bc(self):
        """
        prereq: self.bc_ds has been set.
        """
        if self.bc_ds is not None:
            if len(self.bc_ds.Ntype3)==0:
                log.warning("Cannot set initial h from self.bc_ds because there are no type 3 edges")
                return
            log.info("Will pull initial h from self.bc_ds")

            time_i=np.searchsorted(self.bc_ds.time.values,self.run_start)

            # both bc_ds and ic_ds should already incorporate the depth offset, so
            # no further adjustment here.
            h=self.bc_ds.h.isel(Nt=time_i).mean().values
        else:
            log.info("Will pull initial h from self.bcs")
            for bc in self.bcs:
                if isinstance(bc,hm.StageBC):
                    data=bc.data()
                    if 'time' in data.dims:
                        data=data.sel(time=self.run_start,method='nearest')
                    h=float(data.values)
                    break
            else:
                log.warning("Cannot set initial h from self.bcs because there are no Stage BCs")
                return
                
        self.ic_ds.eta.values[...]=h

        log.info("Setting initial eta from BCs, value=max(z_bed,%.4f) (including z_offset of %.2f)"%(h,self.z_offset))

    def write_forcing(self,overwrite=True):
        # these map to lists of BCs, in case there are BC with mode='add'
        # map edge to BC data
        self.bc_type2=defaultdict(lambda: defaultdict(list)) # [<edge index>][<variable>]=>[DataArray,...]
        # map cell to BC data
        self.bc_type3=defaultdict(lambda: defaultdict(list)) # [<cell index>][<variable>]=>[DataArray,...]
        # Flow BCs are handled specially since they apply across a group of edges
        # Each participating edge should have an entry in bc_type2,
        # [<edge index>]["Q"]=>"segment_name"
        # and a corresponding entry in here:
        self.bc_type2_segments=defaultdict(lambda: defaultdict(list)) # [<segment name>][<variable>]=>[DataArray,...]

        # point sources.
        # indexed by a tuple of (cell,k)
        # [(cell,k][<variable>] => [DataArray]
        self.bc_point_sources=defaultdict(lambda: defaultdict(list))

        super(SuntansModel,self).write_forcing()

        # Get a time series that's the superset of all given timeseries
        all_times=[]
        # edge, cells, groups of edges
        for bc_typ in [self.bc_type2,self.bc_type3,self.bc_type2_segments]:
            for bc in bc_typ.values(): # each edge idx/cell idx/segment name
                for vlist in bc.values(): # each variable on that edge/cell/segment
                    for v in vlist: #list of BCs for this variable on this element
                        if isinstance(v,six.string_types):
                            # type2 edges which reference a segment have no
                            # time series of their own.
                            continue
                        if 'time' in v.dims:
                            all_times.append( v['time'].values )
        if all_times:
            common_time=np.unique(np.concatenate(all_times))
        else:
            # no boundary conditions have times, so fabricate.
            common_time=np.array( [self.run_start,self.run_stop] )
        # Make sure that brackets the run:
        pad=np.timedelta64(1,'D')
        if common_time[0]>=self.run_start:
            common_time=np.concatenate(( [self.run_start-pad],
                                         common_time ))
        # make sure there are *two* times beyond the end for quadratic
        # interpolation
        while len(common_time)<3 or common_time[-2]<=self.run_stop:
            if common_time[-1]<self.run_stop+pad:
                new_time=self.run_stop+pad
            else:
                new_time=common_time[-1]+pad
            common_time=np.concatenate((common_time,[new_time]))
        # SUNTANS applies quadratic interpolation in time, so it requires at least
        # 3 time values - seems that it wants one time before and two times after
        # the current time.
        assert len(common_time)>2

        self.bc_time=common_time
        self.bc_ds=self.compile_bcs()

        self.write_bc_ds()
        if self.met_ds is None:
            self.met_ds=self.zero_met()
        self.write_met_ds()

    def ds_time_units(self):
        """
        setting for how to write time to netcdf
        specifically as suntans expects.  pass as
        ...
        encoding=dict(time={'units':self.ds_time_units()}),
        ...
        in xarray dataset to_netcdf(..)
        """
        basetime=self.config['basetime']
        assert len(basetime)==15 # YYYYMMDD.hhmmss
        time_units="seconds since %s-%s-%s %s:%s:%s"%(basetime[0:4],
                                                      basetime[4:6],
                                                      basetime[6:8],
                                                      basetime[9:11],
                                                      basetime[11:13],
                                                      basetime[13:15])
        return time_units
    def write_bc_ds(self):
        self.bc_ds.to_netcdf( os.path.join(self.run_dir,
                                           self.config['netcdfBdyFile']),
                              encoding=dict(time={'units':self.ds_time_units()}))
    def load_bc_ds(self):
        fn=os.path.join(self.run_dir,
                        self.config['netcdfBdyFile'])
        if not os.path.exists(fn): return False
        self.bc_ds=xr.open_dataset(fn)
        return self.bc_ds

    def write_met_ds(self):
        fn=os.path.join(self.run_dir,
                        self.config['metfile'])
        if os.path.exists(fn):
            log.info("Will replace %s"%fn)
            os.unlink(fn)
        else:
            log.debug("Writing met ds to %s"%fn)
        log.debug(str(self.met_ds))
        self.met_ds.to_netcdf( fn,
                               encoding=dict(nt={'units':self.ds_time_units()},
                                             Time={'units':self.ds_time_units()}) )

    def load_met_ds(self):
        fn=os.path.join(self.run_dir,
                        self.config['metfile'])
        if not os.path.exists(fn): return False
        self.met_ds=xr.open_dataset(fn)
        
    def layer_data(self,with_offset=False,edge_index=None,cell_index=None,z_bed=None):
        """
        Returns layer data without z_offset applied, and
        positive up.

        with no additional arguments, returns global information.  edge_index or
        cell_index will use a z_bed based on that element.  z_bed is used to clip
        z layers.  z_bed should be a positive-up quantity.  A specified z_bed
        takes precendece over edge_index or cell_index.

        Returns a xr.Dataset
        with z_min, z_max, Nk, z_interface, z_mid.

        z_interface and z_mid are ordered surface to bed.

        if with_offset is True, the z_offset is included, which yields
        more accurate (i.e. similar to suntans) layers when there is stretching
        """
        if z_bed is None:
            if edge_index is not None:
                z_bed=self.grid.edge_depths()[edge_index]
            elif cell_index is not None:
                z_bed=self.grid.cell_depths()[cell_index]

        Nk=int(self.config['Nkmax'])
        z_min=self.grid.cells['z_bed'].min() # bed
        z_max=self.grid.cells['z_bed'].max() # surface

        r=float(self.config['rstretch'])

        if with_offset:
            z_min-=self.z_offset
            z_max=0

        depth=-z_min # positive:down
        dzs=np.zeros(Nk, np.float64)
        if r>1.0:
            dzs[0]=depth*(r-1)/(r**Nk-1)
            for k in range(1,Nk):
                dzs[k]=r*dzs[k-1]
        else:
            dzs[:]=depth/float(Nk)
        z_interface=np.concatenate( ( [z_max],
                                      z_max-np.cumsum(dzs) ) )
        
        z_mid=0.5*(z_interface[:-1]+z_interface[1:])

        ds=xr.Dataset()
        ds['z_min']=(),z_min
        ds['z_max']=(),z_max
        ds['z_interface']=('Nkp1',),z_interface
        ds['z_mid']=('Nk',),z_mid
        for v in ['z_min','z_max','z_interface','z_mid']:
            ds[v].attrs['positive']='up'
        return ds

    def compile_bcs(self):
        """
        Postprocess the information from write_forcing()
        to create the BC netcdf dataset.

        Note that bc_ds includes the z_offset.
        """
        ds=xr.Dataset()
        layers=self.layer_data()

        Nk=layers.dims['Nk']
        ds['z']=('Nk',),-(layers.z_mid.values + self.z_offset)

        # suntans assumes that this dimension is Nt, not time
        Nt=len(self.bc_time)
        ds['time']=('Nt',),self.bc_time

        # Scalars will introduce type3 and type2 because they may not know
        # what type of flow forcing is there.  Here we skim out scalars that
        # do not have an associated h (type3) or flow (type2) boundary

        # the list(...keys()) part is to make a copy, so the del's
        # don't upset the iteration
        for cell in list(self.bc_type3.keys()):
            if 'h' not in self.bc_type3[cell]:
                del self.bc_type3[cell]
        # 'u' 'v' and 'Q' for type2
        for edge in list(self.bc_type2.keys()):
            if not ( ('u' in self.bc_type2[edge]) or
                     ('v' in self.bc_type2[edge]) or
                     ('Q' in self.bc_type2[edge])):
                del self.bc_type2[edge]

        Ntype3=len(self.bc_type3)
        ds['cellp']=('Ntype3',),np.zeros(Ntype3,np.int32)-1
        ds['xv']=('Ntype3',),np.zeros(Ntype3,np.float64)
        ds['yv']=('Ntype3',),np.zeros(Ntype3,np.float64)

        # the actual data variables for type 3:
        ds['uc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['vc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['wc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['T']=('Nt','Nk','Ntype3',),20*np.ones((Nt,Nk,Ntype3),np.float64)
        ds['S']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['h']=('Nt','Ntype3'),np.zeros( (Nt, Ntype3), np.float64 )

        def interp_time(da):
            if 'time' not in da.dims: # constant value
                # this should  do the right thing for both scalar and vector
                # values
                return da.values * np.ones( (Nt,)+da.values.shape )
                
            if da.ndim==2:
                assert da.dims[0]=='time'
                # recursively call per-layer, which is assumed to be the second
                # dimension
                profiles=[ interp_time(da[:,i]) for i in range(da.shape[1]) ]
                return np.vstack(profiles).T
            return np.interp( utils.to_dnum(ds.time.values),
                              utils.to_dnum(da.time.values), da.values )
        import time
        elapsed=[0.0]

        def combine_items(values,bc_items,offset=0.0):
            base_item=None
            # include the last mode='overwrite' bc, and sum the mode='add'
            # bcs.
            values[:]=offset

            # aside from Q and h, other variables are 3D, which means
            # that if the data comes back 2D, pad out the layer dimension

            def pad_dims(data):
                if values.ndim==2 and data.ndim==1:
                    return data[:,None] # broadcastable vertical dimension
                else:
                    return data

            for bc_item in bc_items:
                if bc_item.mode=='add':
                    t0=time.time()
                    values[:] += pad_dims(interp_time(bc_item))
                    elapsed[0]+=time.time()-t0
                else:
                    base_item=bc_item
            if base_item is None:
                self.log.warning("BC for cell %d has no overwrite items"%type3_cell)
            else:
                t0=time.time()
                values[:] += pad_dims(interp_time(base_item))
                elapsed[0]+=time.time()-t0

        cc=self.grid.cells_center()

        for type3_i,type3_cell in enumerate(self.bc_type3): # each edge/cell
            ds['cellp'].values[type3_i]=type3_cell
            ds['xv'].values[type3_i]=cc[type3_cell,0]
            ds['yv'].values[type3_i]=cc[type3_cell,1]

            bc=self.bc_type3[type3_cell]
            for v in bc.keys(): # each variable on that edge/cell
                if v=='h':
                    offset=self.z_offset
                else:
                    offset=0
                # will set bc values in place
                combine_items(ds[v].isel(Ntype3=type3_i).values,
                              bc[v],
                              offset=offset)

        Ntype2=len(self.bc_type2)
        Nseg=len(self.bc_type2_segments)
        ds['edgep']=('Ntype2',),np.zeros(Ntype2,np.int32)-1
        ds['xe']=('Ntype2',),np.zeros(Ntype2,np.float64)
        ds['ye']=('Ntype2',),np.zeros(Ntype2,np.float64)
        ds['boundary_h']=('Nt','Ntype2'),np.zeros( (Nt, Ntype2), np.float64) + self.z_offset
        ds['boundary_u']=('Nt','Nk','Ntype2'),np.zeros( (Nt, Nk, Ntype2), np.float64)
        ds['boundary_v']=('Nt','Nk','Ntype2'),np.zeros( (Nt, Nk, Ntype2), np.float64)
        ds['boundary_w']=('Nt','Nk','Ntype2'),np.zeros( (Nt, Nk, Ntype2), np.float64)
        ds['boundary_T']=('Nt','Nk','Ntype2'),np.zeros( (Nt, Nk, Ntype2), np.float64)
        ds['boundary_S']=('Nt','Nk','Ntype2'),np.zeros( (Nt, Nk, Ntype2), np.float64)
        ds['boundary_Q']=('Nt','Nseg'),np.zeros( (Nt, Nseg), np.float64)

        # Iterate over segments first, so that edges below can grab the correct
        # index.
        segment_names=list(self.bc_type2_segments.keys()) # this establishes the order of the segments
        # make this distinct from 0 or 1 to aid debugging
        segment_ids=100 + np.arange(len(segment_names))
        ds['seg_name']=('Nseg',),segment_names # not read by suntans, but maybe helps debugging
        ds['segedgep']=('Ntype2',),np.zeros(Ntype2,np.int32)-1
        ds['segp']=('Nseg',),segment_ids # np.arange(Nseg,dtype=np.int32)

        for seg_i,seg_name in enumerate(segment_names):
            bc=self.bc_type2_segments[seg_name]
            for v in bc.keys(): # only Q, but stick to the same pattern
                combine_items(ds['boundary_'+v].isel(Nseg=seg_i).values,
                              bc[v])

        ec=self.grid.edges_center()
        for type2_i,type2_edge in enumerate(self.bc_type2): # each edge
            ds['edgep'].values[type2_i]=type2_edge
            ds['xe'].values[type2_i]=ec[type2_edge,0]
            ds['ye'].values[type2_i]=ec[type2_edge,1]

            bc=self.bc_type2[type2_edge]
            for v in bc.keys(): # each variable on that edge/cell
                if v=='h':
                    offset=self.z_offset
                else:
                    offset=0.0

                if v!='Q':
                    combine_items(ds['boundary_'+v].isel(Ntype2=type2_i).values,
                                  bc[v],offset)
                else:
                    seg_name=bc[v]
                    # too lazy to work through the right way to deal with combined
                    # bcs for Q right now, so just warn the user that it may be
                    # a problem.
                    if len(seg_name)!=1:
                        log.warning("Only tested with a single value, but got %s"%str(seg_name))
                    seg_name=seg_name[0]
                    seg_idx=segment_ids[segment_names.index(seg_name)]
                    ds['segedgep'].values[type2_i] = seg_idx

        # -- Set grid marks --
        for c in ds.cellp.values:
            assert c>=0
            for j in self.grid.cell_to_edges(c):
                j_cells=self.grid.edge_to_cells(j)
                if j_cells.min()<0:# boundary
                    self.grid.edges['mark'][j]=3 # set to type 3

        for j in ds.edgep.values:
            assert j>=0,"Some edge pointers did not get set"
            self.grid.edges['mark'][j]=2

        # --- Point source code ---
        Npoint=len(self.bc_point_sources)
        ds['point_cell']=('Npoint',), np.zeros(Npoint,np.int32) # point_cell
        ds['point_layer']=('Npoint',), np.zeros(Npoint,np.int32) # point_layer
        ds['point_Q']=('Nt','Npoint'), np.zeros( (Nt,Npoint), np.float64) # np.stack(point_Q,axis=-1)
        ds['point_S']=('Nt','Npoint'), np.zeros( (Nt,Npoint), np.float64) # np.stack(point_S,axis=-1)
        ds['point_T']=('Nt','Npoint'), np.zeros( (Nt,Npoint), np.float64) # np.stack(point_T,axis=-1)

        for pnt_idx,key in enumerate(self.bc_point_sources.keys()):
            (c,k)=key
            log.info("Point source for cell=%d, k=%d"%(c,k))
            assert 'Q' in self.bc_point_sources[key]

            combine_items(ds['point_Q'].isel(Npoint=pnt_idx).values,
                          self.bc_point_sources[key]['Q'])

            ds['point_cell'].values[pnt_idx]=c
            ds['point_layer'].values[pnt_idx]=k

            # really shaky ground here..
            if 'T' in self.bc_point_sources[key]:
                combine_items( ds['point_T'].isel(Npoint=pnt_idx).values,
                               self.bc_point_sources[key]['T'] )
            if 'S' in self.bc_point_sources[key]:
                combine_items( ds['point_S'].isel(Npoint=pnt_idx).values,
                               self.bc_point_sources[key]['S'] )
                               
        # End new point source code

        log.info("Total time in interp_time: %.3fs"%elapsed[0])
        return ds

    def write_bc(self,bc):
        if isinstance(bc,hm.StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,hm.SourceSinkBC):
            self.write_source_sink_bc(bc)
        elif isinstance(bc,hm.FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,hm.VelocityBC):
            self.write_velocity_bc(bc)
        elif isinstance(bc,hm.ScalarBC):
            self.write_scalar_bc(bc)
        else:
            super(SuntansModel,self).write_bc(bc)

    def write_stage_bc(self,bc):
        water_level=bc.data()
        assert len(water_level.dims)<=1,"Water level must have dims either time, or none"

        cells=bc.grid_cells or self.bc_geom_to_cells(bc.geom)
        
        for cell in cells:
            self.bc_type3[cell]['h'].append(water_level)

    def write_velocity_bc(self,bc):
        # interface isn't exactly nailed down with the BC
        # classes.  whether the model wants vector velocity
        # or normal velocity varies by model.  could
        # standardize on vector velocity, and project to normal
        # here?
        ds=bc.dataset()
        edges=bc.grid_edges or self.bc_geom_to_edges(bc.geom)

        for j in edges:
            for comp in ['u','v']:
                da=ds[comp]
                da.attrs['mode']=bc.mode
                self.bc_type2[j][comp].append(da)

    def write_scalar_bc(self,bc):
        da=bc.data()

        scalar_name=bc.scalar
        # canonicalize scalar names for suntans BC files
        if scalar_name.lower() in ['salinity','salt','s']:
            scalar_name='S'
        elif scalar_name.lower() in ['temp','t','temperature']:
            scalar_name='T'
        else:
            self.log.warning("Scalar %s is not S or T or similar"%scalar_name)

        # scalars could be set on edges or cells, or points in cells
        # this should be expanded to make more use of the information in bc.parent
        # if that is set
        if bc.geom.type=='Point':
            self.log.info("Assuming that Point geometry on a scalar bc implies point source")
            ck=self.bc_to_interior_cell_layer(bc) # (cell,layer) tuple
            self.bc_point_sources[ck][scalar_name].append(da)
        else:
            # info is duplicated on type2 (flow) and type3 (stage) BCs, which
            # is sorted out later.
            for j in (bc.grid_edges or self.bc_geom_to_edges(bc.geom)):
                self.bc_type2[j][scalar_name].append(da)
            for cell in (bc.grid_cells or self.bc_geom_to_cells(bc.geom)):
                self.bc_type3[cell][scalar_name].append(da)

    def dredge_boundary(self,linestring,dredge_depth):
        # Restarts appear to be making dredge calls.  Not sure why.
        print("Call to dredge_boundary, restart is",self.restart)

        return super(SuntansModel,self).dredge_boundary(linestring,dredge_depth,
                                                        edge_field='edge_z_bed',
                                                        cell_field='z_bed')
    def dredge_discharge(self,point,dredge_depth):
        print("Call to dredge discharge, restart is",self.restart)
        return super(SuntansModel,self).dredge_discharge(point,dredge_depth,
                                                         edge_field='edge_z_bed',
                                                         cell_field='z_bed')
        
    def write_flow_bc(self,bc):
        da=bc.data()
        self.bc_type2_segments[bc.name]['Q'].append(da)

        assert len(da.dims)<=1,"Flow must have dims either time, or none"

        if (bc.dredge_depth is not None) and (self.restart is None):
            log.info("Dredging grid for flow boundary %s"%bc.name)
            self.dredge_boundary(np.array(bc.geom.coords),
                                 bc.dredge_depth)

        edges=(bc.grid_edges or self.bc_geom_to_edges(bc.geom))
        for j in edges:
            self.bc_type2[j]['Q'].append(bc.name)

    def write_source_sink_bc(self,bc):
        da=bc.data()

        assert bc.geom.type=='Point',"Suntans driver does not support src/sink pair"
        
        if (bc.dredge_depth is not None) and (self.restart is None):
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            self.dredge_discharge(np.array(bc.geom.coords),
                                  bc.dredge_depth)
        ck=self.bc_to_interior_cell_layer(bc) # (cell,layer) tuple

        self.bc_point_sources[ck]['Q'].append(da)
        
        assert len(da.dims)<=1,"Flow must have dims either time, or none"
            
    def bc_geom_to_cells(self,geom):
        """ geom: a LineString geometry. Return the list of cells interior
        to the linestring
        """
        cells=[]
        for j in self.bc_geom_to_edges(geom):
            j_cells=self.grid.edge_to_cells(j)
            assert j_cells.min()<0
            assert j_cells.max()>=0
            cells.append(j_cells.max())
        return cells

    def bc_to_interior_cell_layer(self,bc):
        """
        Determine the cell and layer for a source/sink BC.
        """
        # TODO: use bc.z, which is either an elevation or a 'bed'
        # to choose the layer
        c=self.bc_geom_to_interior_cell(bc.geom)
        self.log.warning("Assuming source/sink is at bed")
        k=int(self.config['Nkmax'])-1
        return (c,k)
    
    def bc_geom_to_interior_cell(self,geom):
        """ geom: a Point or LineString geometry. In the case of a LineString,
        only the first point is used.
        return the index of the cell that the point or linestring node fall in
        """
        coords=np.array(geom)
        if coords.ndim==2:
            coords=coords[0]
        c=self.grid.select_cells_nearest(coords,inside=True)
        assert c is not None,"%s did not match any cells. LineString may be reversed?"%str(coords)
        return c
    
    def bc_geom_to_edges(self,geom):
        """
        geom: LineString geometry
        return list of boundary edges adjacent to geom.
        """
        return self.grid.select_edges_by_polyline(geom,update_e2c=False)

    def set_times_from_config(self):
        """
        Pull run_start,run_stop from a loaded config file.
        """
        if self.restart:
            start_files=self.start_inputs()
            if start_files:
                start=store_file.StoreFile(model=self,proc=0,filename=start_files[0])
                self.run_start=start.time()
            else:
                # maybe we're constructing a restart?  sequencing of this stuff,
                # and the exact state of the model is quirky and under-designed
                self.run_start=self.restart_model.restartable_time()
            log.debug("Inferred start time of restart to be %s"%self.run_start)
        else:
            start_dt=datetime.datetime.strptime(self.config['starttime'],'%Y%m%d.%H%M%S')
            self.run_start=utils.to_dt64(start_dt)

        nsteps=int(self.config['nsteps'])
        dt=np.timedelta64(1,'us') * int(1e6*float(self.config['dt']))
        self.run_stop=self.run_start + nsteps*dt

    def update_config(self):
        assert self.config is not None,"Only support starting from template"

        # Have to be careful about the difference between starttime,
        # which reflects time0, and the start of the initial run,
        # vs. run_start, which is when this simulation will begin
        # (possibly restarting a prior simulation)
        start_dt=utils.to_datetime(self.run_start)
        end_dt=utils.to_datetime(self.run_stop)

        # In the case of restarts, this needs to reflect the
        # start of the first simulation, not a later restart.
        if not self.restart:
            self.config['starttime']=start_dt.strftime('%Y%m%d.%H%M%S')
        else:
            log.info("starttime pulled from previous run: %s"%self.config['starttime'])
            restart_time=self.restart_model.restartable_time()
            assert self.run_start==restart_time,"Configured sim start and restart timestamp don't match"

        dt=np.timedelta64(1,'us') * int(1e6*float(self.config['dt']))
        nsteps=(self.run_stop-self.run_start)/dt
        log.info("Number of steps in this simulation: %d"%nsteps)
        self.config['nsteps']=nsteps

        max_faces=self.grid.max_sides
        if int(self.config['maxFaces']) < max_faces:
            log.debug("Increasing maxFaces to %d"%max_faces)
            self.config['maxFaces']=max_faces

        if self.coriolis_f=='auto':
            if self.projection is None:
                log.warning("No projection and coriolis_f is 'auto'.  No coriolis!")
                self.config['Coriolis_f']=0.0
            else:
                xy_ctr=self.grid.nodes['x'].mean(axis=0)
                ll_ctr=self.native_to_ll(xy_ctr)
                lat=ll_ctr[1]
                # f=2*Omega*sin(phi)
                Omega=7.2921e-5 # rad/s
                f=2*Omega*np.sin(lat*np.pi/180.)
                self.config['Coriolis_f']="%.5e"%f
                log.debug("Using %.2f as latitude for Coriolis => f=%s"%(lat,self.config['Coriolis_f']))
        elif self.coriolis_f is not None:
            self.config['Coriolis_f']=self.coriolis_f

        if len(self.mon_points):
            self.config['numInterpPoints']=1
            self.config['DataLocations']='profile_locs.dat'
            self.config['NkmaxProfs']=0 # all layers
            self.config['ProfileDataFile']="profdata.dat"
            # could also make sure that config['ProfileVariables'] has a default like 'hu'
            #   and ntoutProfs has a reasonable value.

    def restart_copier(self,src,dst,on_exists='replace_link'):
        """
        src: source file for copy, relative to present working dir
        dst: destination.
        will either symlink or copy src to dst, based on self.restart_symlink
        setting
        In order to avoid a limit on chained symlinks, symlinks will point to
        the original file.
        """
        if os.path.lexists(dst):
            # Allow replacing symlinks, but if dst is a real file, bail out
            # to avoid anything sketchy
            if on_exists=='replace_link' and os.path.islink(dst):
                os.unlink(dst)
            elif on_exists=='replace':
                os.unlink(dst)
            elif on_exists=='fail':
                raise Exception("Restart copier %s=>%s found destination already exists. on_exists='fail'"%(src,dst))
            else:
                raise Exception("Unknown option for on_exists: %s"%on_exists)
                
        if self.restart_symlink:
            # this ensures that we don't build up long chains of
            # symlinks
            src=os.path.realpath(src)
            src_rel=os.path.relpath(src,self.run_dir)

            os.symlink(src_rel,dst)
        else:
            shutil.copyfile(src,dst)

    def write_grid(self):
        if not self.restart:
            # Write a grid that suntans will read:
            self.grid.write_suntans_hybrid(self.run_dir,overwrite=True,z_offset=self.z_offset)
            self.write_grid_bathy()
        else:
            parent_base=os.path.dirname(self.restart)
            for fn in ['cells.dat','edges.dat',
                       'points.dat','depths.dat-voro']:
                self.restart_copier(os.path.join(parent_base,fn),
                                    os.path.join(self.run_dir,fn))
    def write_grid_bathy(self):
        # And write cell bathymetry separately
        # This filename is hardcoded into suntans, not part of
        # the settings in suntans.dat (maybe it can be overridden?)
        cell_depth_fn=os.path.join(self.run_dir,"depths.dat-voro")

        cell_xy=self.grid.cells_center()

        # make depth positive down
        z=-(self.grid.cells['z_bed'] + self.z_offset)

        min_depth=0+float(self.config['minimum_depth'])
        shallow=z<min_depth
        if np.any(shallow):
            log.warning("%d of %d cell depths extend above z=0 even with offset of %.2f"%(np.sum(shallow),
                                                                                          len(shallow),
                                                                                          self.z_offset))
            z=z.clip(min_depth,np.inf)

        cell_xyz=np.c_[cell_xy,z]
        np.savetxt(cell_depth_fn,cell_xyz) # space separated

        if self.use_edge_depths:
            # And edges, preparing for edge-based bathy
            edge_depth_fn=os.path.join(self.run_dir,"depths.dat-edge")

            edge_xy=self.grid.edges_center()

            # make depth positive down
            edge_depth=-(self.grid.edges['edge_z_bed'] + self.z_offset)
            edge_xyz=np.c_[edge_xy,edge_depth]
            np.savetxt(edge_depth_fn,edge_xyz) # space separated

    def grid_as_dataset(self):
        """
        Return the grid and vertical geometry in a xr.Dataset
        following the naming of suntans/ugrid.
        Note that this does not yet set all attributes -- TODO!
        This method does apply z_offset to the grid.
        """
        ds=self.grid.write_to_xarray()

        ds=ds.rename({'face':'Nc',
                      'edge':'Ne',
                      'node':'Np',
                      'node_per_edge':'two',
                      'maxnode_per_face':'numsides'})
        layers=self.layer_data()

        z_min=layers.z_min.values
        z_max=layers.z_max.values
        Nk=layers.dims['Nk']

        cc=self.grid.cells_center()
        ds['xv']=('Nc',),cc[:,0]
        ds['yv']=('Nc',),cc[:,1]

        ds['z_r']=('Nk',),layers.z_mid.values + self.z_offset
        ds['z_r'].attrs['positive']='down'

        # not right for 3D..
        ds['Nk']=('Nc',),Nk*np.ones(self.grid.Ncells(),np.int32)

        # don't take any chances on ugrid assumptions -- exactly mimic
        # the example:
        ds['suntans_mesh']=(),0
        ds.suntans_mesh.attrs.update( dict(cf_role='mesh_topology',
                                           long_name='Topology data of 2D unstructured mesh',
                                           topology_dimension=2,
                                           node_coordinates="xp yp",
                                           face_node_connectivity="cells",
                                           edge_node_connectivity="edges",
                                           face_coordinates="xv yv",
                                           edge_coordinates="xe ye",
                                           face_edge_connectivity="face",
                                           edge_face_connectivity="grad") )

        ds['cells']=('Nc','numsides'),self.grid.cells['nodes']
        ds['nfaces']=('Nc',), [self.grid.cell_Nsides(c) for c in range(self.grid.Ncells())]
        ds['edges']=('Ne','two'),self.grid.edges['nodes']
        ds['neigh']=('Nc','numsides'), [self.grid.cell_to_cells(c,pad=True)
                                        for c in range(self.grid.Ncells())]

        ds['grad']=('Ne','two'),self.grid.edge_to_cells()
        ds['xp']=('Np',),self.grid.nodes['x'][:,0]
        ds['yp']=('Np',),self.grid.nodes['x'][:,1]

        depth=-(self.grid.cells['z_bed'] + self.z_offset)
        ds['dv']=('Nc',),depth.clip(float(self.config['minimum_depth']),np.inf)

        # really ought to set attrs for everybody, but sign of depth is
        # particular, so go ahead and do it here.
        ds.dv.attrs.update( dict( standard_name='sea_floor_depth_below_geoid',
                                  long_name='seafloor depth',
                                  comment='Has offset of %.3f applied'%(-self.z_offset),
                                  units='m',
                                  mesh='suntans_mesh',
                                  location='face',
                                  positive='down') )
        ds['dz']=('Nk',),-np.diff(layers.z_interface.values)
        ds['mark']=('Ne',),self.grid.edges['mark']
        return ds

    def zero_initial_condition(self):
        """
        Return a xr.Dataset for initial conditions, with all values
        initialized to nominal zero values.

        This dataset has z_offset applied.
        """
        ds_ic=self.grid_as_dataset()
        ds_ic['time']=('time',),[self.run_start]

        for name,dims in [ ('eta',('time','Nc')),
                           ('uc', ('time','Nk','Nc')),
                           ('vc', ('time','Nk','Nc')),
                           ('salt',('time','Nk','Nc')),
                           ('temp',('time','Nk','Nc')),
                           ('agec',('time','Nk','Nc')),
                           ('agesource',('Nk','Nc')) ]:
            shape=tuple( [ds_ic.dims[d] for d in dims] )
            if name=='agealpha':
                dtype=np.timedelta64
            else:
                dtype=np.float64
            vals=np.zeros(shape,dtype)
            if name=='eta':
                vals += self.z_offset
            ds_ic[name]=dims,vals

        return ds_ic

    met_pad=np.timedelta64(1,'D')
    def zero_met(self,times=None):
        """
        Create an empty (zero valued, and T=20degC) dataset for met
        forcing.
        times: defaults to 4 time steps bracketing the run, pass in
        other ndarray(datetime64) to override
        """
        ds_met=xr.Dataset()

        # this is nt in the sample, but maybe time is okay??
        # nope -- needs to be nt.
        # quadratic interpolation is used, so we need to pad out before/after
        # the simulation
        if times is None:
            times=[self.run_start-self.met_pad,
                   self.run_start,
                   self.run_stop,
                   self.run_stop+self.met_pad]
        ds_met['nt']=('nt',),times
        ds_met['Time']=('nt',),ds_met.nt.values

        xxyy=self.grid.bounds()
        xy0=[ 0.5*(xxyy[0]+xxyy[1]), 0.5*(xxyy[2]+xxyy[3])]
        ll0=self.native_to_ll(xy0)

        for name in ['Uwind','Vwind','Tair','Pair','RH','rain','cloud']:
            ds_met["x_"+name]=("N"+name,),[ll0[0]]
            ds_met["y_"+name]=("N"+name,),[ll0[1]]
            ds_met["z_"+name]=("N"+name,),[10]

        def const(dims,val):
            shape=tuple( [ds_met.dims[d] for d in dims] )
            return dims,val*np.ones(shape)

        ds_met['Uwind']=const(('nt','NUwind'), 0.0)

        ds_met['Vwind']=const(('nt','NVwind'), 0.0)
        ds_met['Tair'] =const(('nt','NTair'), 20.0)
        ds_met['Pair'] =const(('nt','NPair'), 1000.) # units?
        ds_met['RH']=const(('nt','NRH'), 80.)
        ds_met['rain']=const(('nt','Nrain'), 0.)
        ds_met['cloud']=const(('nt','Ncloud'), 0.5)

        return ds_met

    def partition(self):
        if self.restart:
            # multiprocessor files except the .<proc> suffix, to be symlinked
            # or copied
            parent_base=os.path.dirname(self.restart)
            multi_proc_files=['celldata.dat','cells.dat',
                              'edgedata.dat','edges.dat',
                              'nodes.dat','topology.dat']
            if os.path.exists(os.path.join(parent_base,'depths.dat-edge.0')):
                multi_proc_files.append('depths.dat-edge')
            for fn_base in multi_proc_files:
                for proc in range(self.num_procs):
                    fn=fn_base+".%d"%proc
                    self.restart_copier(os.path.join(parent_base,fn),
                                        os.path.join(self.run_dir,fn))
            # single files
            single_files=['vertspace.dat']
            if 'DataLocations' in self.config:
                # UNTESTED
                single_files.append(self.config['DataLocations'])
            if 'ProfileDataFile' in self.config:
                # UNTESTED
                single_files.append(self.config['ProfileDataFile'])
            for fn in single_files:
                self.restart_copier(os.path.join(parent_base,fn),
                                    os.path.join(self.run_dir,fn))
        else:
            self.run_mpi(["-g",self.sun_verbose_flag,"--datadir=%s"%self.run_dir])
    sun_verbose_flag="-vv" 
    def run_simulation(self):
        args=['-s']
        if self.restart:
            args.append("-r")
        args+=[self.sun_verbose_flag,"--datadir=%s"%self.run_dir]
        self.run_mpi(args)
    def run_mpi(self,sun_args):
        sun="sun"
        if self.sun_bin_dir is not None:
            sun=os.path.join(self.sun_bin_dir,sun)
        cmd=[sun] + sun_args
        if self.num_procs>1:
            mpiexec="mpiexec"
            if self.mpi_bin_dir is not None:
                mpiexec=os.path.join(self.mpi_bin_dir,mpiexec)
            cmd=[mpiexec,"-n","%d"%self.num_procs] + cmd
        subprocess.call(cmd)

    # Methods related to using model output
    def restartable_time(self):
        """
        If store output is enabled, and this run has already been
        executed, return the datetime64 of the restart files.
        Otherwise None
        """
        store_files=self.store_outputs()
        if not store_files:
            return None

        store=store_file.StoreFile(model=self,proc=0,filename=store_files[0])
        return store.time()

    def store_outputs(self):
        store_fn=os.path.join(self.run_dir,self.config['StoreFile'])
        fns=glob.glob( store_fn+"*" )
        fns.sort()
        return fns

    def start_inputs(self):
        start_fn=os.path.join(self.run_dir,self.config['StartFile'])
        fns=glob.glob( start_fn+"*" )
        fns.sort()
        return fns

    def avg_outputs(self):
        """
        with mergeArrays=1, these get sequenced with nstepsperncfile
        with mergeArrays=0, each processor gets a file.
        currently this function does not expose the difference
        """
        if int(self.config['calcaverage']):
            fns=glob.glob(os.path.join(self.run_dir,self.config['averageNetcdfFile']+"*"))
            fns.sort()
            return fns
        else:
            return []
        
    def map_outputs(self):
        """
        return a list of map output files -- if netcdf output is enabled,
        that is what will be returned.
        Guaranteed to be in the order of subdomain numbering if mergeArrays=0,
        and in chronological order if mergeArrays=1.

        Currently you can't distinguish which is which just from the output
        of this method.
        """
        if int(self.config['outputNetcdf']):
            if self.config['mergeArrays'] is None or int(self.config['mergeArrays']):
                # in this case the outputs are chunked in time
                # with names like Estuary_SUNTANS.nc_0000.nc
                #  i.e. <outputNetcdfFile>_<seqN>.nc
                fns=glob.glob(os.path.join(self.run_dir,self.config['outputNetcdfFile']+"_*.nc"))
                fns.sort()
                return fns
            else:
                # convoluted, but allow for some of the odd name construction for
                # per-domain files, relying only on the assumption that the
                # suffix is the processor number.
                fns=glob.glob(os.path.join(self.run_dir,self.config['outputNetcdfFile']+"*"))
                procs=[int(fn.split('.')[-1]) for fn in fns]
                order=np.argsort(procs)
                fns=[fns[i] for i in order]
                return fns
        else:
            raise Exception("Need to implement map output filenames for non-netcdf")

    @classmethod
    def parse_profdata(cls,fn):
        """
        Parse the profdata.dat file associated with a run.
        fn: path to file to parse.
        This is a classmethod to allow external usage but keep it bundled with the 
        SunDriver class.
        Returns an xarray dataset 

        NOTE: if this uses caching at some point in the future, monitor_output should
        be adapted to make a copy since it mutates the dataset.

        data format:
         (4 byte int)numTotalDataPoints: Number of data points found on all processors.  Note that
             that this could be different from the number specified since some may lie outside the domain.
         (4 byte int)numInterpPoints: Number of nearest neighbors to each point used for interpolation.
         (4 byte int)NkmaxProfs: Number of vertical levels output in the profiles.
         (4 byte int)nsteps: Total number of time steps in the simulation.
         (4 byte int)ntoutProfs: Frequency of profile output.  This implies a total of nsteps/ntoutProfs are output.
         (8 byte double)dt: Time step size
         (8 byte double array X NkmaxProfs)dz: Contains the vertical grid spacings.
         (4 byte int array X numTotalDataPoints)allIndices: Contains the indices of each point that determines its
             original location in the data file.  This is mostly for debugging since the output data is resorted
             so that it is in the same order as it appeared in the data file.
         (4 byte int array X 2*numTotalDataPoints)dataXY: Contains the original data points at (or near) which profiles
             are output.
         (8 byte double array X numTotalDataPoints*numInterpPoints)xv: Array containing the x-locations of the nearest
             neighbors to the dataXY points.  If numInterpPoints=3, then the 3 closest neighbors to the point
             (dataXY[2*i],dataXY[2*i+1]) are (xv[3*i],yv[3*i]), (xv[3*i+1],yv[3*i+1]), (xv[3*i+2],yv[3*i+2]).
         (8 byte double array X numTotalDataPoints*numInterpPoints)yv: Array containing the y-locations of the nearest
             neighbors to the dataXY points (see xv above).
        """
        pdata=xr.Dataset()
        with open(fn,'rb') as fp:
            hdr_ints = np.fromfile(fp,np.int32,count=5)
            pdata['num_total_data_points']=(),hdr_ints[0]
            pdata['num_interp_points'] =(), hdr_ints[1]
            pdata['nkmax_profs'] =(), hdr_ints[2]
            pdata['nsteps'] =(), hdr_ints[3]
            pdata['ntout_profs'] =(), hdr_ints[4]

            pdata['dt'] =(), np.fromfile(fp,np.float64,1)[0]
            pdata['dzz'] = ('layer',),np.fromfile(fp,np.float64,pdata['nkmax_profs'].item() )
            pdata['all_indices'] = np.fromfile(fp,np.int32,pdata['num_total_data_points'].item())
            dataxy = np.fromfile(fp,np.float64,2*pdata['num_total_data_points'].item())
            pdata['request_xy'] =('request','xy'), dataxy.reshape( (-1,2) )
            pdata['request_xy'].attrs['description']="Coordinates of the requested profiles"
            
            xvyv = np.fromfile(fp,np.float64,2*(pdata['num_total_data_points']*pdata['num_interp_points']).item())
            pdata['prof_xy'] =('profile','xy'), xvyv.reshape( (2,-1) ).transpose()
            pdata['prof_xy'].attrs['description']="Coordinates of the output profiles"
        return pdata

    def read_profile_data_raw(self,scalar,pdata=None,memmap=True):
        """
        scalar is one of HorizontalVelocityFile,
        FreeSurfaceFile, etc

        pdata: a previously parsed ProfData file, from parse_profdata. Can be passed
           in to avoid re-parsing this file.
        memmap: by default the file is memory mapped, which can be a huge performance
        savings for large files.  In some cases and platforms it is less stable,
        though.
        """
        if pdata is None:
            pdata=self.parse_profdata(self.file_path('ProfileDataFile'))
        prof_pnts = pdata.prof_xy
        prof_len = prof_pnts.shape[0]

        prof_fname = self.file_path(scalar) + ".prof"

        if not os.path.exists(prof_fname):
            log.debug("Request for profile for %s, but %s does not exist"%(scalar,prof_fname))
            return None

        # Figure out the shape of the output:
        #  I'm assuming that profile data gets spat out in the same
        #  ordering of dimensions as regular grid-based data

        shape_per_step = []

        # profiles.c writes u first then v, then w, each with a
        # separate call to Write3DData()
        if scalar == 'HorizontalVelocityFile':
            shape_per_step.append(3)

        # the outer loop is over profile points
        shape_per_step.append(prof_len)

        # And does it have z-levels? if so, that is the inner-most
        #  loop, so the last dimension of the array
        if scalar != 'FreeSurfaceFile':
            nkmax_profs = pdata['nkmax_profs'].item() 
            shape_per_step.append(nkmax_profs)

        # better to use the size of the specific file we're opening:
        prof_dat_size=os.stat(prof_fname).st_size
        REALSIZE=8
        bytes_per_step = REALSIZE * np.prod( np.array(shape_per_step) )
        n_steps_in_file=int(prof_dat_size//bytes_per_step )

        final_shape = tuple([n_steps_in_file] + shape_per_step)

        if memmap: # BRAVE!
            # print "Trying to memory map the data.."
            data = np.memmap(prof_fname, dtype=np.float64, mode='r', shape=final_shape)
        else:
            data = np.fromfile(prof_fname,float64)
            data = data.reshape(*final_shape)

        # no caching at this point..
        return data

    monitor_nodata=999999
    monitor_dv=None # caches dv_from_map results
    def monitor_output(self,nan_nodata=False,dv_from_map=False):
        """
        Return xarray Dataset including the monitor output
        """
        if 'DataLocations' not in self.config: return None

        pdata=self.parse_profdata(self.file_path('ProfileDataFile'))

        file_to_var={'FreeSurfaceFile':'eta',
                     'HorizontalVelocityFile':'u',
                     'TemperatureFile':'temp',
                     'SalinityFile':'salt',
                     'EddyViscosityFile':'nut',
                     'VerticalVelocityFile':'w',
                     'ScalarDiffusivityFile':'kappa'}
        # Try to figure out which variables have been output in profiles
        # Just scan what's there, to avoid trying to figure out defaults.
        for scalar in list(file_to_var.keys()):
            raw_data=self.read_profile_data_raw(scalar,pdata=pdata)
            if raw_data is not None:
                if scalar=='FreeSurfaceFile':
                    dims=('time','profile')
                elif scalar=='HorizontalVelocityFile':
                    dims=('time','xyz','profile','layer')
                else:
                    dims=('time','profile','layer')
                # May need to use a different layer dimension for w...
                # print("%s:  raw data shape: %s  dims: %s"%(scalar,str(raw_data.shape),dims))
                if nan_nodata and np.any(raw_data==self.monitor_nodata):
                    # this can significantly slow down the process if ultimately we're
                    # only going to use a small slice of the data
                    raw_data=np.where(raw_data==self.monitor_nodata,
                                      np.nan,raw_data)
                pdata[file_to_var[scalar]]=dims,raw_data

        # This may need some tweaking, but it's a start.
        # use microseconds to get some reasonable precision for fraction dt
        # but note that this isn't necessarily exact.
        dt_prof=np.timedelta64( int( pdata['ntout_profs']*pdata['dt']*1e6),'us')
        pdata['time']=('time',),(self.run_start + dt_prof*np.arange(pdata.dims['time']))

        if dv_from_map:
            if self.monitor_dv is None:
                if 0: # read from map file, but that may not be valid until end of run
                    print("Loading dv for monitor data - should happen once!")
                    self.monitor_dv=self.extract_station_map(xy=pdata.prof_xy.values[:,:],data_vars='dv')
                else: # read from subdomain grids.
                    mon_dv=np.zeros(pdata.dims['profile'],np.float64)
                    mon_dv[:]=np.nan

                    for proc in range(self.num_procs):
                        gsub=self.subdomain_grid(proc)
                        for i,xy in enumerate(pdata.prof_xy.values):
                            c=gsub.select_cells_nearest(xy,inside=True)
                            if c is not None:
                                mon_dv[i]=gsub.cells[c]['dv']
                    assert np.all(np.isfinite(mon_dv)),"Failed to get depths for all profile locatins"
                    self.monitor_dv=xr.Dataset()
                    self.monitor_dv['dv']=('profile',),mon_dv
                                
            pdata['dv']=('profile',),self.monitor_dv['dv'].values
                
        # Total hack for convenience -- add a closest_to([x,y]) method to extract a single
        # profile.
        @utils.add_to(pdata)
        def closest_to(self,target):
            dists=utils.dist(target,self['prof_xy'].values)
            idx=np.argmin(dists)
            return self.isel(profile=idx)
                
        return pdata
    
    _subdomain_grids=None
    def subdomain_grid(self,p):
        if self._subdomain_grids is None:
            self._subdomain_grids={}

        if p not in self._subdomain_grids:
            sub_g=unstructured_grid.UnstructuredGrid.read_suntans_hybrid(path=self.run_dir,
                                                                         points='points.dat',
                                                                         edges='edges.dat.%d'%p,
                                                                         cells='cells.dat.%d'%p)
            # edge depth is an ad-hoc extension, not "standard" enough to be in
            # read_suntans_hybrid, so add it in here:
            edge_depth_fn=self.file_path('depth')+"-edge.%d"%p
            if os.path.exists(edge_depth_fn):
                edge_xyz=np.loadtxt(edge_depth_fn)
                # 2019-05-29: this did not have a negation.  probably that was wrong.
                # transition away from edge_depth, anyway.
                # sub_g.add_edge_field('edge_depth',edge_xyz[:,2])
                sub_g.add_edge_field('edge_z_bed',-edge_xyz[:,2])
            if ('dv' in sub_g.cells.dtype.names) and ('z_bed' not in sub_g.cells.dtype.names):
                sub_g.add_cell_field('z_bed',-sub_g.cells['dv'])

            self._subdomain_grids[p]=sub_g
        return self._subdomain_grids[p]
    
    @memoize.imemoize(lru=64)
    def extract_transect_monitor(self,xy=None,ll=None,time=None,
                                 time_mode='inner',dv_from_map=False,
                                 dzmin_surface=None):
        """
        In progress alternate approach for transects.
        xy: [N,2] location of vertical profiles making up the transect
        ll: like xy, but lon/lat to be converted via self.ll_to_native
        time: can be used to pull a specific time for each xy (with time_mode='inner').
        
        time_mode: for now, only 'inner'.  May be expanded to control whether
         time is used orthogonal to xy, or parallel (i.e. for each xy, do we pull
         one corresponding time from time, or pull all of the time for each).

        if time is not covered by the output, or the run has no monitor output,
        will return None.
        """
        if xy is None:
            xy=self.ll_to_native(ll)

        if time_mode=='outer':
            assert time.ndim==0,"Not ready for array-valued time with time_mode='outer'"
        
        def xyt():
            if time_mode=='inner':
                for loc,t in zip(xy,time):
                    yield loc,t
            else:
                for loc in xy:
                    yield loc,time

        stns=[]
        for loc,t in xyt():
            if time_mode=='inner':
                # then each location has a single time associated with it
                # we can narrow extract_station in that case.
                t_slice=(t,t)
            else:
                # potentially a range of times
                # this should also a work when time is a scalar datetime64.
                t_slice=(t.min(),t.max())

            stn=self.extract_station_monitor(xy=loc,chain_count=t_slice,
                                             dv_from_map=dv_from_map)
            if stn is None:
                log.warning('Found no monitor data for %s. Skip transect'%str(t_slice))
                return None
            if np.isscalar(t):
                if (t<stn.time.values[0]) or (t>stn.time.values[-1]):
                    log.info("Requested time %s is outside the range of the model output"%t)
                    return None
                ti=utils.nearest(stn.time.values,t)
                stn=stn.isel(time=ti)
            stns.append(stn)

        tran=xr.concat(stns,dim='time')
        
        # now cleanup nan/nodata
        for v in tran.data_vars:
            if not np.issubdtype(tran[v].dtype,np.floating): continue
            missing=tran[v].values==self.monitor_nodata
            tran[v].values[missing]=np.nan

        xy=np.c_[ tran.station_x,tran.station_y ]
        tran=tran.rename(time='sample')
        tran['d_sample']=('sample',),utils.dist_along(xy)

        if 'dzz' in tran:
            assert 'eta' in tran,"Not ready for transect processing without eta"
            dzz_2d,eta_2d=xr.broadcast(tran.dzz,tran.eta)                
            z_max=eta_2d
            
            #Not ready for this.
            if 'dv' in tran:
                _,dv_2d=xr.broadcast(eta_2d,tran.dv)
                z_min=-dv_2d
            else:
                z_min=-np.inf
                
            tran['z_bot']=-dzz_2d.cumsum(dim='layer')
            tran['z_top']=tran.z_bot+dzz_2d
            tran['z_bot']=tran.z_bot.clip(z_min,z_max)
            tran['z_top']=tran.z_top.clip(z_min,z_max)
            tran['z_ctr']=0.5*(tran.z_bot+tran.z_top)
            for fld in ['z_bot','z_top','z_ctr']:
                tran[fld].attrs['positive']='up'
            # to be consistent with xr_transect, and np.diff(z_ctr),
            # z_dz is _negative_
            tran['z_dz'] =(tran.z_bot-tran.z_top)
        if dzmin_surface is not None:
            self.adjust_transect_for_dzmin_surface(tran,dzmin_surf=dzmin_surface)
        return tran

    def adjust_transect_for_dzmin_surface(self,tran,update_vars=['salt','temp'],dzmin_surf=0.25):
        """
        janky - it is not always clear in the output which layers are valid, versus when a layer
        was really thin and was coalesced with the next layer down.  This method
        takes an xr_transect style transect, finds thin surface layers and copies the values from
        lower down up to the surface cells.
        This currently probably doesn't work for velocity, just scalar.

        extract_transect_monitor will call this automatically if dzmin_surface is specified.
        """
        from ... import xr_transect
        z_dz=xr_transect.get_z_dz(tran) 
        for samp_i in range(tran.dims['sample']):
            eta=tran.eta.isel(sample=samp_i)
            k_update=[]
            for k in range(tran.dims['layer']):
                if z_dz[samp_i,k]==0.0: 
                    continue # truly dry
                elif tran.eta[samp_i] - tran.z_bot[samp_i,k] < dzmin_surf:
                    log.debug("[sample %s,k %s] too thin"%(samp_i,k))
                    k_update.append(k)
                else:
                    # valid layer
                    for ku in k_update:
                        for v in update_vars:
                            tran[v].values[samp_i,ku] = tran[v].values[samp_i,k]
                    break
    
    def extract_transect(self,xy=None,ll=None,time=slice(None),dx=None,
                         vars=['uc','vc','Ac','dv','dzz','eta','w'],
                         datasets=None,grids=None):
        """
        xy: [N,2] coordinates defining the line of the transect
        time: if an integer or slice of integers, interpret as index
          into time dimension. otherwise try to convert to datetime64, 
          and then index into time coordinate.
        dx: omit to use xy as is, or a length scale for resampling xy

        returns xr.Dataset, unless xy does not intersect the grid at all,
        in which case None is returned.

        Simple chaining is allowed, but if time spans two runs, the later
        run will be used.

        datasets,grids: if supplied, a list of datasets to use instead of chaining
         and/or merging subdomains. datasets can be either a path to netcdf file
         or xr.Dataset. if grids are not supplied, grids will be extracted from
         respective Datasets.
        """
        if xy is None:
            xy=self.ll_to_native(ll)
        if dx is not None:
            xy=linestring_utils.upsample_linearring(xy,dx,closed_ring=False)

        # check for chaining
        if np.issubdtype(type(time),np.integer):
            # time came in as an index, so no chaining.
            pass
        else:
            # asarray() helps avoid xarray annoyance
            dt=np.max(utils.to_dt64(np.asarray(time)))
            if dt<self.run_start:
                log.info("extract_transect: chain back")
                run=self.chain_restarts(count=dt)[0]
                if run is not self: # avoid inf. recursion
                    return run.extract_transect(xy=xy,ll=ll,time=time,dx=dx,
                                                vars=vars)
                else:
                    log.info("extract_transect: chain back just returned self.")
            
        proc_point_cell=np.zeros( [self.num_procs,len(xy)], np.int32)-1
        point_datasets=[None]*len(xy)

        good_vars=None  # set on-demand below

        merged=int(self.config['mergeArrays'])>0
        def gen_sources(): # iterator over proc,sub_g,map_fn
            if datasets is not None:
                for proc,ds in enumerate(datasets):
                    if grids is None:
                        g=unstructured_grid.UnstructredGrid.from_ugrid(ds)
                    else:
                        g=grids[proc]
                    yield [proc,g,ds]
            elif merged:
                map_fn=self.map_outputs()[0]
                g=unstructured_grid.UnstructuredGrid.from_ugrid(map_fn)
                yield [0,g,map_fn]
            else:
                for proc in range(self.num_procs):
                    yield proc,self.subdomain_grid(proc),self.map_outputs()[proc]

        def time_to_isel(ds,times,mode='nearest'):
            """
            return an argument suitable for isel, to pull one or more time steps
            from ds. 
            ds: dataset with time dimension
            times: integer, datetime64, or slice thereof.
            mode: 'nearest' map a time to the nearest matching time
                  'before' map a time to the matching or preceding time step
                  'after' map a timem to the following time step.
            """
            if isinstance(times,slice):
                return slice(time_to_isel(ds,times.start,mode='before'),
                             time_to_isel(ds,times.stop,mode='after'))
            else:
                if np.issubdtype(type(times),np.integer):
                    # already an index
                    return times
                else:
                    dns=utils.to_dnum(ds.time.values)
                    dn=utils.to_dnum(times)
                    if mode=='nearest':
                        return utils.nearest(dns,dn)
                    elif mode=='before':
                        return np.searchsorted(dns,dn)
                    elif mode=='after':
                        return np.searchsorted(dns,dn,side='right')
                    else:
                        raise Exception("Bad mode: %s"%mode)
                    
        for proc,sub_g,map_fn in gen_sources():
            ds=None
            for pnti,pnt in enumerate(xy):
                if point_datasets[pnti] is not None:
                    continue
                c=sub_g.select_cells_nearest(pnt,inside=True)
                if c is not None:
                    proc_point_cell[proc,pnti]=c
                    if ds is None:
                        if isinstance(map_fn,str):
                            ds=xr.open_dataset(map_fn)
                        else:
                            ds=map_fn.copy()
                        # doctor up the Nk dimensions
                        ds['Nkf']=ds['Nk'] # copy the variable
                        del ds['Nk'] # delete old variable, leaving Nk as just a dimension
                        if good_vars is None:
                            # drop any variables that don't appear in the output
                            good_vars=[v for v in vars if v in ds]
                    time_idx=time_to_isel(ds,time)
                    point_ds=ds[good_vars].isel(time=time_idx,Nc=c)
                    point_ds['x_sample']=pnt[0]
                    point_ds['y_sample']=pnt[1]
                    point_datasets[pnti]=point_ds
        # drop xy points that didn't hit a cell
        point_datasets=[p for p in point_datasets if p is not None]
        if len(point_datasets)==0: # transect doesn't intersect grid at all.
            log.debug("Transect points do not intersect model")
            return None
        transect=xr.concat(point_datasets,dim='sample')
        renames=dict(Nk='layer',Nkw='interface',
                     uc='Ve',vc='Vn',w='Vu_int')
        renames={x:renames[x] for x in renames if (x in transect) or (x in transect.dims)}
        transect=transect.rename(**renames)
        transect['U']=('sample','layer','xy'),np.concatenate( [transect.Ve.values[...,None],
                                                               transect.Vn.values[...,None]],
                                                              axis=-1)
        if 'Vu_int' in transect:
            Vu_int=transect.Vu_int.values.copy()
            Vu_int[np.isnan(Vu_int)]=0.0
            transect['Vu']=('sample','layer'), 0.5*(Vu_int[:,1:] + Vu_int[:,:-1])

        # construct layer-center depths
        if 'dzz' not in transect:
            # fabricate a dzz
            eta_2d,dv_2d,z_w_2d=xr.broadcast( transect['eta'], transect['dv'], -ds['z_w'])
            z_w_2d=z_w_2d.clip(-dv_2d,eta_2d)
            z_bot=z_w_2d.isel(Nkw=slice(1,None)).values
            z_top=z_w_2d.isel(Nkw=slice(None,-1)).values
            # must use values to avoid xarray getting smart with aligning axes.
            dzz=z_top-z_bot
            z_ctr=0.5*(z_bot+z_top)
            z_ctr[dzz==0.0]=np.nan
        else:
            dzz=transect.dzz.values.copy() # sample, Nk
            z_bot=transect['eta'].values[:,None] - dzz.cumsum(axis=1)
            z_top=z_bot+dzz
            z_ctr=0.5*(z_top+z_bot)
            z_ctr[dzz==0.0]=np.nan # indicate no data

        transect['z_ctr']=('sample','layer'), z_ctr
        transect['z_top']=('sample','layer'), z_top
        transect['z_bot']=('sample','layer'), z_bot

        # first, the interior interfaces
        def choose_valid(a,b):
            return np.where(np.isfinite(a),a,b)
        z_int=choose_valid(z_top[:,1:],z_bot[:,:-1])
        # and no choice of where the first and last rows come from
        z_int=np.concatenate( [z_top[:,:1],
                               z_int,
                               z_bot[:,-1:]],
                              axis=1)
        transect['z_int']=('sample','interface'),z_int
        # we've got dzz, so go ahead and use it, but honor xr_transect
        # sign convention that z_dz ~ diff(z_int)
        transect['z_dz']=('sample','layer'),-dzz

        # helps with plotting
        transect.attrs['source']=self.run_dir
        return transect

    warn_initial_water_level=0
    def initial_water_level(self):
        """
        some BC methods which want a depth need an estimate of the water surface
        elevation, and the initial water level is as good a guess as any.
        """
        if self.ic_ds is not None:
            return float(self.ic_ds.eta.mean())
        else:
            if self.warn_initial_water_level==0:
                log.warning("Request for initial water level, but no IC is set yet")
            self.warn_initial_water_level+=1
            return 0.0

    def extract_station_monitor(self,xy=None,ll=None,chain_count=1,
                                dv_from_map=False,data_vars=None):
        """
        Return a dataset for a single point in the model
        xy: native model coordinates, [Nstation,2]
        ll: lon/lat coordinates, [Nstation,2]
        chain_count: max number of restarts to go back.
          1=>no chaining just this model.  None or 0:
          chain all runs possible.  Otherwise, go back max
          number of runs up to chain_count
          if chain_count is a np.datetime64, go back enough restarts to 
          get to that date (see chain_restarts())
          if chain_count is a tuple of datetime64, only consider restarts covering
          that period.

        This version pulls output from history files

        if dv_from_map is True, additionally pulls dv from map output.

        if no data matches the time range of chain_count, or profile output     
        wasn't enable, returns None.
        """
        if xy is None:
            xy=self.ll_to_native(ll)

        if chain_count!=1:
            restarts=self.chain_restarts(count=chain_count,load_grid=False)
            # dv should be constant, so only load it on self.
            dss=[mod.extract_station_monitor(xy=xy,ll=ll,chain_count=1,
                                             data_vars=data_vars,dv_from_map=False)
                 for mod in restarts]
            if len(dss)==0:
                return None
            chained=xr.concat(dss,dim='time',data_vars='minimal')
            if dv_from_map:
                # just to get dv...
                me=self.extract_station_monitor(xy=xy,ll=ll,chain_count=1,
                                                dv_from_map=True)
                chained['dv']=me.dv
            return chained

        mon=self.monitor_output(dv_from_map=dv_from_map)
        if mon is None:
            return None # maybe profile output wasn't enabled.

        xy=np.asarray(xy)
        orig_ndim=xy.ndim
        if orig_ndim==1:
            xy=xy[None,:]
        elif orig_ndim>2:
            raise Exception("Can only handle single coordinates or an list of coordinates")
            
        num_stations=len(xy)

        stations=[]

        for stn in range(num_stations):
            dists=utils.dist(xy[stn,:],mon.prof_xy.values)
            best=np.argmin(dists)
            station=mon.isel(profile=best)
            if data_vars is not None:
                for v in list(station.data_vars):
                    if v not in data_vars:
                        del station[v]
            station['distance_from_target']=(),dists[best]
            station['profile_index']=best
            station['source']='monitor'
            stations.append(station)

        if orig_ndim==1:
            # This used to be done after the fact -- just isel(station=0)
            # but concatenation in xarray is super slow
            combined_ds=stations[0]
            combined_ds['station_x']=(), xy[0,0]
            combined_ds['station_y']=(), xy[0,1]
        else:
            combined_ds=xr.concat(stations,dim='station')
            combined_ds['station_x']=('station',), xy[...,0]
            combined_ds['station_y']=('station',), xy[...,1]

        return combined_ds
        
    def extract_station(self,xy=None,ll=None,chain_count=1,source='auto',dv_from_map=False,
                        data_vars=None):
        """
        See extract_station_map, extract_station_monitor for details.
        Will try monitor output if it exists, otherwise map output.
        source: 'auto' (default), 'map' or 'monitor' to force a choice.

        If a specific source is chosen and doesn't exist, returns None
        """
        if source in ['auto','monitor']:
            ds=self.extract_station_monitor(xy=xy,ll=ll,chain_count=chain_count,
                                            dv_from_map=dv_from_map,data_vars=data_vars)
            if (ds is not None) or (source=='monitor'):
                return ds
        if source in ['auto','map']:
            return self.extract_station_map(xy=xy,ll=ll,chain_count=chain_count,
                                            data_vars=data_vars)
        assert False,"How did we get here"
        
    def extract_station_map(self,xy=None,ll=None,chain_count=1,data_vars=None):
        """
        Return a dataset for a single point in the model
        xy: native model coordinates, [Nstation,2]
        ll: lon/lat coordinates, [Nstation,2]
        chain_count: max number of restarts auto go back.
          1=>no chaining just this model.  None or 0:
          chain all runs possible.  Otherwise, go back max
          number of runs up to chain_count
        data_vars: list of variables to include, otherwise all.

        This version pulls output from map files
        """
        if xy is None:
            xy=self.ll_to_native(ll)
            
        map_fns=self.map_outputs()

        # First, map request locations to processor and cell
        
        xy=np.asarray(xy)
        orig_ndim=xy.ndim
        if orig_ndim==1:
            xy=xy[None,:]
        elif orig_ndim>2:
            raise Exception("Can only handle single coordinates or an list of coordinates")
            
        num_stations=len(xy)

        # allocate, [proc,cell,distance] per point
        matches=[[None,None,np.inf] for i in range(num_stations)]

        # outer loop on proc
        for proc,map_fn in enumerate(map_fns):
            map_ds=xr.open_dataset(map_fn)
            g=unstructured_grid.UnstructuredGrid.from_ugrid(map_ds)
            cc=g.cells_center()
            # inner loop on station
            for station in range(num_stations):
                c=g.select_cells_nearest(xy[station],inside=False)
                d=utils.dist(cc[c],xy[station])
                if d<matches[station][2]:
                    matches[station]=[proc,c,d]
        # Now we know exactly which procs are useful, and can close
        # the others
        hot_procs={} # dictionary tracking which processors are useful
        for station,(proc,c,d) in enumerate(matches):
            hot_procs[proc]=(station,c,d)
        for proc,map_fn in enumerate(map_fns):
            if proc not in hot_procs:
                xr.open_dataset(map_fn).close()
            # otherwise close later
            
        if chain_count==1:
            runs=[self]
        else:
            runs=self.chain_restarts(count=chain_count)

        dss=[] # per-restart datasets

        # workaround for cases where numsides was not held constant
        max_numsides=0
        min_numsides=1000000

        for run in runs:
            model_out=None
            for proc,map_fn in enumerate(run.map_outputs()):
                if proc not in hot_procs: continue # doesn't have any hits
                
                map_ds=xr.open_dataset(run.map_outputs()[proc])

                # Work around bad naming of dimensions
                map_ds['Nk_c']=map_ds['Nk']
                del map_ds['Nk']

                # wait until we've loaded one to initialize the dataset for this run
                if model_out is None:
                    model_out=xr.Dataset() # not middle out
                    model_out['time']=map_ds.time

                    # allocate output variables:
                    for d in map_ds.data_vars:
                        if data_vars and d not in data_vars:
                            log.debug('Skipping variable %s'%d)
                            continue
                        if 'Nc' in map_ds[d].dims:
                            # put station first
                            new_dims=['station']
                            new_shape=[num_stations]
                            for d_dim in map_ds[d].dims:
                                if d_dim=='Nc':
                                    continue # replaced by station above
                                else:
                                    new_dims.append(d_dim)
                                    new_shape.append(map_ds.dims[d_dim])
                            model_out[d]=tuple(new_dims), np.zeros(new_shape, map_ds[d].dtype )

                # For vectorized indexing, pulls the stations we want want from this
                # processor, but only gets as far as ordering them densely on this
                # proc
                Nc_indexer=xr.DataArray( [m[1] for m in matches if m[0]==proc ],
                                         dims=['proc_station'] )
                assert len(Nc_indexer),"Somehow this proc has no hits"
                
                # and the station indexes in model_out to assign to.
                # this can't use vectorized indexing because you can't assign to the
                # result of vectorized indexing.
                proc_stations=np.array( [i for i,m in enumerate(matches) if m[0]==proc] )
                                        
                for d in map_ds.data_vars:
                    if d not in model_out: continue

                    # potentially gets one isel out of the tight loop
                    # this appears to work.
                    extracted=map_ds[d].isel(Nc=Nc_indexer)
                    # extracted will have 'station' in the wrong place. transpose
                    dims=['proc_station'] + [d for d in extracted.dims if d!='proc_station']
                    extractedT=extracted.transpose(*dims)
                    # this line is 90% of the time:
                    ext_vals=extractedT.values
                    model_out[d].values[proc_stations,...] = ext_vals
                    
                    #for station in lin_idx,arr_idx in enumerate(np.ndindex(stn_shape)):
                    #    if matches[lin_idx][0]!=proc: continue
                    #    
                    #    extracted=map_ds[d].isel(Nc=matches[lin_idx][1])
                    #    # seems like voodoo -- construct an index into the output,
                    #    # which will let us point to the desired station.
                    #    sel=dict(zip(stn_dims,arr_idx))
                    #    model_out[d].isel(sel).values[...]=extracted
            if dss:
                # limit to non-overlapping
                time_sel=model_out.time.values>dss[-1].time.values[-1]
                model_out=model_out.isel(time=time_sel)
            if 'numsides' in model_out.dims:
                max_numsides=max(max_numsides,model_out.dims['numsides'])
                min_numsides=min(min_numsides,model_out.dims['numsides'])
            dss.append(model_out)

        if 'numsides' in model_out.dims:
            if max_numsides!=min_numsides:
                log.warning("numsides varies %d to %d over restarts.  Kludge around"%
                            (min_numsides,max_numsides))
                dss=[ ds.isel(numsides=slice(0,min_numsides)) for ds in dss]
                
        combined_ds=xr.concat(dss,dim='time',data_vars='minimal',coords='minimal')

        # copy from matches
        combined_ds['distance_from_target']=('station',), np.zeros(num_stations, np.float64)
        combined_ds['subdomain']=('station',), np.zeros(num_stations,np.int32)
        combined_ds['station_cell']=('station',), np.zeros(num_stations,np.int32)
        combined_ds['station_cell'].attrs['description']="Cell index in subdomain grid"
        combined_ds['station_x']=('station',), xy[...,0]
        combined_ds['station_y']=('station',), xy[...,1]
        combined_ds['source']=('station',), ["map"]*num_stations
        
        for station in range(num_stations):
            # here we know the order and can go straight to values
            combined_ds['distance_from_target'].values[station]=matches[station][2]
            combined_ds['subdomain'].values[station]=matches[station][0]

        if orig_ndim==1:
            combined_ds=combined_ds.isel(station=0)
            
        return combined_ds
