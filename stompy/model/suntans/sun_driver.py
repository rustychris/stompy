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

from ... import utils
from ..delft import dflow_model as dfm
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
MultiBC=dfm.MultiBC
StageBC=dfm.StageBC
FlowBC=dfm.FlowBC
VelocityBC=dfm.VelocityBC
ScalarBC=dfm.ScalarBC
SourceSinkBC=dfm.SourceSinkBC
OTPSStageBC=dfm.OTPSStageBC
OTPSFlowBC=dfm.OTPSFlowBC
OTPSVelocityBC=dfm.OTPSVelocityBC
HycomMultiVelocityBC=dfm.HycomMultiVelocityBC
HycomMultiScalarBC=dfm.HycomMultiScalarBC
NOAAStageBC=dfm.NOAAStageBC
NwisFlowBC=dfm.NwisFlowBC
NwisStageBC=dfm.NwisStageBC
CdecFlowBC=dfm.CdecFlowBC
CdecStageBC=dfm.CdecStageBC

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

            m = re.match("^\s*((\S+)\s+(\S+))?\s*.*",line)
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

    def __eq__(self,other):
        return self.is_equal(other)
    def is_equal(self,other,limit_to_keys=None):
        # key by key equality comparison:
        print("Comparing two configs")
        for k in self.entries.keys():
            if limit_to_keys and k not in limit_to_keys:
                continue
            if k not in other.entries:
                print("Other is missing key %s"%k)
                return False
            elif self.val_to_str(other.entries[k][0]) != self.val_to_str(self.entries[k][0]):
                print("Different values key %s => %s, %s"%(k,self.entries[k][0],other.entries[k][0]))
                return False
        for k in other.entries.keys():
            if limit_to_keys and k not in limit_to_keys:
                continue
            if k not in self.entries:
                print("other has extra key %s"%k)
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
                    print("No change in config")
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
        print("Trying the new way of specifying t0")
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

class SuntansModel(dfm.HydroModel):
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

    # None: not a restart, or
    # path to suntans.dat for the run being restarted, or True if this is
    # a restart but we don't we have a separate directory for the restart,
    # just StartFiles
    restart=None
    restart_model=None # model instance being restarted

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

    @property
    def time0(self):
        self.config['starttime']
        dt=datetime.datetime.strptime(self.config['starttime'],
                                      "%Y%m%d.%H%M%S")
        return utils.to_dt64(dt)

    def create_restart(self,symlink=True):
        new_model=SuntansModel()
        new_model.config=self.config.copy()
        # things that have to match up, but are not part of the config:
        new_model.num_procs=self.num_procs
        new_model.restart=self.config_filename
        new_model.restart_model=self
        new_model.restart_symlink=symlink
        new_model.set_grid(unstructured_grid.UnstructuredGrid.read_suntans(self.run_dir))
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

        # make sure we have the fields expected by suntans
        if 'depth' not in grid.cells.dtype.names:
            if 'depth' in grid.nodes.dtype.names:
                cell_depth=grid.interp_node_to_cell(grid.nodes['depth'])
                # and avoid overlapping names
                grid.delete_node_field('depth')
            elif 'depth' in grid.edges.dtype.names:
                raise Exception("Not implemented interpolating edge to cell bathy")
            else:
                self.log.warning("No depth information in grid.  Creating zero-depth")
                cell_depth=np.zeros(grid.Ncells(),np.float64)
            grid.add_cell_field('depth',cell_depth)

            
        # with the current suntans version, depths are on cells, but model driver
        # code in places wants an edge depth.  so copy those here.
        e2c=grid.edge_to_cells()
        nc1=e2c[:,0].copy()   ; nc2=e2c[:,1].copy()
        nc1[nc1<0]=nc2[nc1<0] ; nc2[nc2<0]=nc1[nc2<0]
        # edge depth is shallower of neighboring cells
        edge_depth=np.minimum(grid.cells['depth'][nc1],grid.cells['depth'][nc2])

        if 'edge_depth' in grid.edges.dtype.names:
            deep_edges=(grid.edges['edge_depth']<edge_depth)
            print("%d edges had a specified depth deeper than neighboring cells.  Replaced them"%
                  deep_edges.sum())
            grid.edges['edge_depth'][deep_edges]=edge_depth[deep_edges]
        else:        
            grid.add_edge_field('edge_depth',edge_depth)
        
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
        z=self.grid.edges['edge_depth'][j]

        if datum is not None:
            if datum=='eta0':
                z+=self.initial_water_level()
        return z

        
    @staticmethod
    def load(fn,load_grid=True):
        """
        Open an existing model setup, from path to its suntans.dat
        """
        model=SuntansModel()
        if os.path.isdir(fn):
            fn=os.path.join(fn,'suntans.dat')
        model.load_template(fn)
        model.set_run_dir(os.path.dirname(fn),mode='existing')
        # infer number of processors based on celldata files
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
            model.load_grid()
        return model
    def load_grid(self):
        """
        Set self.grid from existing suntans-format grid in self.run_dir.
        """
        g=unstructured_grid.UnstructuredGrid.read_suntans(self.run_dir)
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
        start_path=os.path.join(self.run_dir,self.config['StartFile']+".0")
        if os.path.exists(start_path):
            log.debug("Looks like a restart")
            self.restart=True

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
        """
        runs=[self]
        run=self
        while 1:
            if count and len(runs)>=count:
                break
            run.infer_restart()
            if run.restart and run.restart is not True:
                run=SuntansModel.load(run.restart,load_grid=load_grid)
                runs.insert(0,run)
            else:
                break
        return runs

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

    def write(self):
        self.update_config()
        self.write_config()
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

    def set_initial_h_from_bc(self):
        """
        prereq: self.bc_ds has been set.
        """
        if len(self.bc_ds.Ntype3)==0:
            log.warning("Cannot set initial h from BC because there are no type 3 edges")
            return

        time_i=np.searchsorted(self.bc_ds.time.values,self.run_start)

        # both bc_ds and ic_ds should already incorporate the depth offset, so
        # no further adjustment here.
        h=self.bc_ds.h.isel(Nt=time_i).mean().values

        # this is positive down, already shifted, clipped.
        #cell_depths=self.ic_ds['dv'].values

        # This led to drying issues in 3D, and ultimately was not the fix
        # for issues in 2D
        #self.ic_ds.eta.values[:]=np.maximum(h,-cell_depths)

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

    def write_met_ds(self):
        self.met_ds.to_netcdf( os.path.join(self.run_dir,
                                            self.config['metfile']),
                               encoding=dict(nt={'units':self.ds_time_units()},
                                             Time={'units':self.ds_time_units()}) )

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
        z_min=self.grid.cells['depth'].min() # bed
        z_max=self.grid.cells['depth'].max() # surface

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
                    values[:] += pad_dims(interp_time(bc_item))
                else:
                    base_item=bc_item
            if base_item is None:
                self.log.warning("BC for cell %d has no overwrite items"%type3_cell)
            else:
                values[:] += pad_dims(interp_time(bc_item))

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
            print("Point source for cell=%d, k=%d"%(c,k))
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
        
        return ds

    def write_bc(self,bc):
        if isinstance(bc,dfm.StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,dfm.FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,dfm.VelocityBC):
            self.write_velocity_bc(bc)
        elif isinstance(bc,dfm.ScalarBC):
            self.write_scalar_bc(bc)
        elif isinstance(bc,dfm.SourceSinkBC):
            self.write_source_sink_bc(bc)
        else:
            super(SuntansModel,self).write_bc(bc)

    def write_stage_bc(self,bc):
        water_level=bc.data()
        assert len(water_level.dims)<=1,"Water level must have dims either time, or none"

        cells=self.bc_geom_to_cells(bc.geom)
        for cell in cells:
            self.bc_type3[cell]['h'].append(water_level)

    def write_velocity_bc(self,bc):
        # interface isn't exactly nailed down with the BC
        # classes.  whether the model wants vector velocity
        # or normal velocity varies by model.  could
        # standardize on vector velocity, and project to normal
        # here?
        ds=bc.dataset()
        edges=self.bc_geom_to_edges(bc.geom)

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
            for j in self.bc_geom_to_edges(bc.geom):
                self.bc_type2[j][scalar_name].append(da)
            for cell in self.bc_geom_to_cells(bc.geom):
                self.bc_type3[cell][scalar_name].append(da)

    def dredge_boundary(self,linestring,dredge_depth):
        super(SuntansModel,self).dredge_boundary(linestring,dredge_depth,
                                                 edge_field='edge_depth',
                                                 cell_field='depth')
    def dredge_discharge(self,point,dredge_depth):
        super(SuntansModel,self).dredge_discharge(point,dredge_depth,
                                                  edge_field='edge_depth',
                                                  cell_field='depth')
        
    def write_flow_bc(self,bc):
        da=bc.data()
        self.bc_type2_segments[bc.name]['Q'].append(da)

        assert len(da.dims)<=1,"Flow must have dims either time, or none"

        if bc.dredge_depth is not None:
            log.info("Dredging grid for flow boundary %s"%bc.name)
            self.dredge_boundary(np.array(bc.geom.coords),
                                 bc.dredge_depth)

        edges=self.bc_geom_to_edges(bc.geom)
        for j in edges:
            self.bc_type2[j]['Q'].append(bc.name)


    def write_source_sink_bc(self,bc):
        da=bc.data()

        if bc.dredge_depth is not None:
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
        # return dfm_grid.polyline_to_boundary_edges(self.grid,np.array(geom.coords))
        # now part of UnstructuredGrid
        return self.grid.select_edges_by_polyline(geom)

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

    def restart_copier(self,src,dst):
        """
        src: source file for copy, relative to present working dir
        dst: destination.
        will either symlink or copy src to dst, based on self.restart_symlink
        setting
        In order to avoid a limit on chained symlinks, symlinks will point to
        the original file.
        """
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
        z=-(self.grid.cells['depth'] + self.z_offset)

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
            z=-(self.grid.edges['edge_depth'] + self.z_offset)
            edge_xyz=np.c_[edge_xy,z]
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


        depth=-(self.grid.cells['depth'] + self.z_offset)
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
    def zero_met(self):
        ds_met=xr.Dataset()

        # this is nt in the sample, but maybe time is okay??
        # nope -- needs to be nt.
        # quadratic interpolation is used, so we need to pad out before/after
        # the simulation
        ds_met['nt']=('nt',),[self.run_start-self.met_pad,
                              self.run_start,
                              self.run_stop,
                              self.run_stop+self.met_pad]
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
            for fn_base in ['celldata.dat','cells.dat',
                            'edgedata.dat','edges.dat',
                            'nodes.dat','topology.dat']:
                for proc in range(self.num_procs):
                    fn=fn_base+".%d"%proc
                    self.restart_copier(os.path.join(parent_base,fn),
                                        os.path.join(self.run_dir,fn))
            # single files
            for fn in ['vertspace.dat']:
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
        """
        if int(self.config['outputNetcdf']):
            if int(self.config['mergeArrays']):
                # should just be 1
                return [os.path.join(self.run_dir,self.config['outputNetcdfFile'])]
            else:
                fns=glob.glob(os.path.join(self.run_dir,self.config['outputNetcdfFile']+"*"))
                fns.sort()
                return fns
        else:
            raise Exception("Need to implement map output filenames for non-netcdf")

    _subdomain_grids=None
    def subdomain_grid(self,p):
        if self._subdomain_grids is None:
            self._subdomain_grids={}

        if p not in self._subdomain_grids:
            edges_fn=os.path.join(self.run_dir,'edges.dat.%d')
            cells_fn=os.path.join(self.run_dir,'cells.dat.%d')
            points_fn=os.path.join(self.run_dir,'points.dat')
            sub_g=unstructured_grid.UnstructuredGrid.read_suntans_hybrid(path=self.run_dir,
                                                                         points='points.dat',
                                                                         edges='edges.dat.%d'%p,
                                                                         cells='cells.dat.%d'%p)
            self._subdomain_grids[p]=sub_g
        return self._subdomain_grids[p]

    def extract_transect(self,xy,time=slice(None),dx=None,
                         vars=['uc','vc','Ac','dv','dzz','eta','w']):
        """
        xy: [N,2] coordinates defining the line of the transect
        time: time index or slice to extract
        dx: omit to use xy as is, or a length scale for resampling xy

        returns xr.Dataset, unless xy does not intersect the grid at all,
        in which case None is returned.
        """
        if dx is not None:
            xy=linestring_utils.upsample_linearring(xy,dx,closed_ring=False)
        proc_point_cell=np.zeros( [self.num_procs,len(xy)], np.int32)-1
        point_datasets=[None]*len(xy)
        
        for proc in range(self.num_procs):
            sub_g=self.subdomain_grid(proc)
            ds=None
            for pnti,pnt in enumerate(xy):
                if point_datasets[pnti] is not None:
                    continue
                c=sub_g.select_cells_nearest(pnt,inside=True)
                if c is not None:
                    proc_point_cell[proc,pnti]=c
                    if ds is None:
                        ds=xr.open_dataset(self.map_outputs()[proc])
                        # doctor up the Nk dimensions
                        ds['Nkf']=ds['Nk'] # copy the variable
                        del ds['Nk'] # delete old variable, leaving Nk as just a dimension
                    point_ds=ds[vars].isel(time=time,Nc=c)
                    point_ds['x_sample']=pnt[0]
                    point_ds['y_sample']=pnt[1]
                    point_datasets[pnti]=point_ds
        # drop xy points that didn't hit a cell
        point_datasets=[p for p in point_datasets if p is not None]
        if len(point_datasets)==0: # transect doesn't intersect grid at all.
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
        dzz=transect.dzz.values.copy() # sample, Nk
        z_bot=transect['eta'].values[:,None] - dzz.cumsum(axis=1)
        z_top=z_bot+dzz
        z_ctr=0.5*(z_top+z_bot)
        z_ctr[dzz==0.0]=np.nan # indicate no data
        transect['z_ctr']=('sample','layer'), z_ctr

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

    def extract_station(self,xy=None,ll=None,chain_count=1):
        """
        Return a dataset for a single point in the model
        xy: native model coordinates, [Nstation,2]
        ll: lon/lat coordinates, [Nstation,2]
        chain_count: max number of restarts to go back.
          1=>no chaining just this model.  None or 0:
          chain all runs possible.  Otherwise, go back max
          number of runs up to chain_count
        """
        # For the moment, just use map output
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
                    model_out[d].values[proc_stations,...] = extractedT.values
                    
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
            max_numsides=max(max_numsides,model_out.dims['numsides'])
            min_numsides=min(min_numsides,model_out.dims['numsides'])
            dss.append(model_out)
            
        if max_numsides!=min_numsides:
            log.warning("numsides varies %d to %d over restarts.  Kludge around"%(min_numsides,max_numsides))
            dss=[ ds.isel(numsides=slice(0,min_numsides)) for ds in dss]
        combined_ds=xr.concat(dss,dim='time',data_vars='minimal',coords='minimal')

        # copy from matches
        combined_ds['distance_from_target']=('station',), np.zeros(num_stations, np.float64)
        combined_ds['subdomain']=('station',), np.zeros(num_stations,np.int32)
        combined_ds['station_cell']=('station',), np.zeros(num_stations,np.int32)
        combined_ds['station_cell'].attrs['description']="Cell index in subdomain grid"
        combined_ds['station_x']=('station',), xy[...,0]
        combined_ds['station_y']=('station',), xy[...,1]
        
        for station in range(num_stations):
            # here we know the order and can go straight to values
            combined_ds['distance_from_target'].values[station]=matches[station][2]
            combined_ds['subdomain'].values[station]=matches[station][0]

        if orig_ndim==1:
            combined_ds=combined_ds.isel(station=0)
            
        return combined_ds
