import os
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
OTPSStageBC=dfm.OTPSStageBC
StageBC=dfm.StageBC
FlowBC=dfm.FlowBC

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

    def set_simulation_period(self,start_date,end_date):
        """ Based on the two python datetime instances given, sets
        start_day, start_year and nsteps
        """
        self.set_value('start_year',start_date.year)
        t0 = datetime.datetime( start_date.year,1,1,tzinfo=utc )
        self.set_value('start_day',date2num(start_date) - date2num(t0))

        # roundoff dangers here -
        # self.set_simulation_duration_days( date2num(end_date) - date2num(start_date))
        self.set_simulation_duration(delta=(end_date - start_date))

    def set_simulation_duration_days(self,days):
        self.set_simulation_duration(days=days)
    def set_simulation_duration(self,
                                days=None,
                                delta=None,
                                seconds = None):
        """ Set the number of steps for the simulation - exactly one of the parameters should
        be specified:
        days: decimal number of days - DANGER - it's very easy to get some round-off issues here
        delta: a datetime.timedelta object.
          hopefully safe, as long as any differencing between dates was done with UTC dates
          (or local dates with no daylight savings transitions)
        seconds: total number of seconds - this should be safe, though there are some possibilities for
          roundoff.

        """
        print("Setting simulation duration:")
        print("  days=",days)
        print("  delta=",delta)
        print("  seconds=",seconds)

        # convert everything to a timedelta -
        if (days is not None) + (delta is not None) + (seconds is not None) != 1:
            raise Exception("Exactly one of days, delta, or seconds must be specified")
        if days is not None:
            delta = datetime.timedelta(days=days)
        elif seconds is not None:
            delta = datetime.timedelta(seconds=seconds)

        # assuming that dt is also a multiple of the precision (currently 10ms), this is
        # safe
        delta = dt_round(delta)
        print("  rounded delta = ",delta)
        timestep = dt_round(datetime.timedelta(seconds=self.conf_float('dt')))
        print("  rounded timestep =",timestep)

        # now we have a hopefully exact simulation duration in integer days, seconds, microseconds
        # and a similarly exact timestep
        # would like to do this:
        #   nsteps = delta / timestep
        # but that's not supported until python 3.3 or so
        def to_quanta(td):
            """ return integer number of time quanta in the time delta object
            """
            us_per_quanta = 1000000 // datenum_precision_per_s
            return (td.days*86400 + td.seconds)*datenum_precision_per_s + \
                   int( round( td.microseconds/us_per_quanta) )
        quanta_timestep = to_quanta(timestep)
        quanta_delta = to_quanta(delta)

        print("  quanta_timestep=",quanta_timestep)
        print("  quanta_delta=",quanta_delta)
        nsteps = quanta_delta // quanta_timestep

        print("  nsteps = ",nsteps)
        # double-check, going back to timedelta objects:
        err = nsteps * timestep - delta
        self.set_value('nsteps',int(nsteps))
        print("Simulation duration requires %i steps (rounding error=%s)"%(self.conf_int('nsteps'),err))

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
    z_offset=0.0
    ic_ds=None

    # for partition, run, etc.
    sun_bin_dir=None
    mpi_bin_dir=None

    # 'auto': the grid and projection information will be used to
    # update the coriolis parameter.
    # None: leave whatever value is in the template
    # <float>: use that as the coriolis parameter
    coriolis_f='auto'

    def set_grid(self,grid):
        """
        read/load grid, check for depth data and edge marks.
        This does not apply the z_offset -- that is only
        applied during writing out the rundata.
        """
        if isinstance(grid,six.string_types):
            # step in and load as suntans, rather than generic
            grid=unstructured_grid.SuntansGrid(grid)
        super(SuntansModel,self).set_grid(grid)

        # make sure we have the fields expected by suntans
        if 'depth' not in grid.cells.dtype.names:
            if 'depth' in grid.nodes.dtype.names:
                cell_depth=grid.interp_node_to_cell(grid.nodes['depth'])
            elif 'depth' in grid.edges.dtype.names:
                raise Exception("Not implemented interpolating edge to cell bathy")
            else:
                cell_depth=np.zeros(grid.Ncells(),np.float64)
            grid.add_cell_field('depth',cell_depth)
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

    def load_template(self,fn):
        self.template_fn=fn
        self.config=SunConfig(fn)

    def set_run_dir(self,path,mode='create'):
        assert mode!='clean',"Suntans driver doesn't know what clean is"
        return super(SuntansModel,self).set_run_dir(path,mode)

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
        self.write_ic()

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
        # map edge to BC data
        self.bc_type2=defaultdict(dict) # [<edge index>][<variable>]=>DataArray
        # map cell to BC data
        self.bc_type3=defaultdict(dict) # [<cell index>][<variable>]=>DataArray
        # Flow BCs are handled specially since they apply across a group of edges
        # Each participating edge should have an entry in bc_type2,
        # [<edge index>]["Q"]=>"segment_name"
        # and a corresponding entry in here:
        self.bc_type2_segments=defaultdict(dict) # [<segment name>][<variable>]=>DataArray

        super(SuntansModel,self).write_forcing()

        # Get a time series that's the superset of all given timeseries
        all_times=[]
        # edge, cells, groups of edges
        for bc_typ in [self.bc_type2,self.bc_type3,self.bc_type2_segments]:
            for bc in bc_typ.values(): # each edge idx/cell idx/segment name
                for v in bc.values(): # each variable on that edge/cell/segment
                    if isinstance(v,six.string_types):
                        # type2 edges which reference a segment have no
                        # time series of their own.
                        continue
                    if 'time' in v.dims:
                        all_times.append( v['time'].values )
        common_time=np.unique(np.concatenate(all_times))
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

        # with the above loop, I don't think is needed anymore
        # if len(common_time)==2:
        #     # cast to seconds to be sure it doesn't get rounded to 0 days.
        #     delta=(common_time[1]-common_time[0]).astype('<m8[s]')/2
        #     common_time=np.array( [ common_time[0],common_time[0]+delta,
        #                             common_time[1]] )

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
                               encoding=dict(time={'units':self.ds_time_units()}) )

    def layer_data(self):
        """
        Returns layer data without z_offset applied, and
        positive up.

        Returns a xr.Dataset
        with z_min, z_max, Nk, z_interface, z_mid.

        z_interface and z_mid are ordered surface to bed.
        """
        Nk=int(self.config['Nkmax'])
        z_min=self.grid.cells['depth'].min() # bed
        z_max=self.grid.cells['depth'].max() # surface
        log.warning("Layers not fully implemented -- assuming evenly spaced Nk=%d"%Nk)
        z_interface=-np.linspace(-z_max,-z_min,Nk+1) # evenly spaced...
        z_mid=0.5*(z_interface[:-1]+z_interface[1:])

        ds=xr.Dataset()
        ds['z_min']=(),z_min
        ds['z_max']=(),z_max
        ds['z_interface']=('Nkp1',),z_interface
        ds['z_mid']=('Nk',),z_mid
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

        Ntype3=len(self.bc_type3)
        ds['cellp']=('Ntype3',),np.zeros(Ntype3,np.int32)-1
        ds['xv']=('Ntype3',),np.zeros(Ntype3,np.float64)
        ds['yv']=('Ntype3',),np.zeros(Ntype3,np.float64)

        # the actual data variables for typr 3:
        ds['uc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['vc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['wc']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['T']=('Nt','Nk','Ntype3',),20*np.ones((Nt,Nk,Ntype3),np.float64)
        ds['S']=('Nt','Nk','Ntype3',),np.zeros((Nt,Nk,Ntype3),np.float64)
        ds['h']=('Nt','Ntype3'),np.zeros( (Nt, Ntype3), np.float64 )

        def interp_time(da):
            return np.interp( utils.to_dnum(ds.time.values),
                              utils.to_dnum(da.time.values), da.values )

        cc=self.grid.cells_center()

        for type3_i,type3_cell in enumerate(self.bc_type3): # each edge/cell
            ds['cellp'].values[type3_i]=type3_cell
            ds['xv'].values[type3_i]=cc[type3_cell,0]
            ds['yv'].values[type3_i]=cc[type3_cell,1]

            bc=self.bc_type3[type3_cell]
            for v in bc.keys(): # each variable on that edge/cell
                # hmm - how to assign bc[v].values to ds[v] ?
                if v=='h':
                    offset=self.z_offset
                else:
                    offset=0
                ds[v].isel(Ntype3=type3_i).values[:] = interp_time(bc[v]) + offset

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
                ds['boundary_'+v].isel(Nseg=seg_i).values[:] = interp_time(bc[v])

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
                    ds['boundary_'+v].isel(Ntype2=type2_i).values[:] = interp_time(bc[v]) + offset
                else:
                    seg_name=bc[v]
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

        return ds

    def write_bc(self,bc):
        if isinstance(bc,dfm.StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,dfm.FlowBC):
            self.write_flow_bc(bc)
        else:
            super(SuntansModel,self).write_bc(bc)

    def write_stage_bc(self,bc):
        water_level=bc.dataarray()
        assert len(water_level.dims)<=1,"Water level must have dims either time, or none"

        cells=self.bc_geom_to_cells(bc.geom)
        for cell in cells:
            self.bc_type3[cell]['h']=water_level

    def write_flow_bc(self,bc):
        da=bc.dataarray()
        self.bc_type2_segments[bc.name]['Q']=da

        assert len(da.dims)<=1,"Flow must have dims either time, or none"

        edges=self.bc_geom_to_edges(bc.geom)
        for j in edges:
            self.bc_type2[j]['Q']=bc.name

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

    def bc_geom_to_edges(self,geom):
        """
        geom: LineString geometry
        return list of boundary edges adjacent to geom.
        """
        return dfm_grid.polyline_to_boundary_edges(self.grid,np.array(geom.coords))

    def update_config(self):
        assert self.config is not None,"Only support starting from template"

        # This is old, for my old version of suntans
        start_dt=utils.to_datetime(self.run_start)
        end_dt=utils.to_datetime(self.run_stop)
        self.config.set_simulation_period(start_date=start_dt,end_date=end_dt)

        # and this is a newer approach:
        self.config['starttime']=start_dt.strftime('%Y%m%d.%H%M%S')

        if self.coriolis_f=='auto':
            xy_ctr=self.grid.nodes['x'].mean(axis=0)
            ll_ctr=self.native_to_ll(xy_ctr)
            lat=ll_ctr[1]
            # f=2*Omega*sin(phi)
            Omega=7.2921e-5 # rad/s
            f=2*Omega*np.sin(lat*np.pi/180.)
            self.config['Coriolis_f']="%.5e"%f
            log.info("Using %.2f as latitude for Coriolis => f=%s"%(lat,self.config['Coriolis_f']))
        elif self.coriolis_f is not None:
            self.config['Coriolis_f']=self.coriolis_f

    def write_grid(self):
        # Write a grid that suntans will read:
        self.grid.write_suntans_hybrid(self.run_dir,overwrite=True,z_offset=self.z_offset)
        self.write_grid_bathy()
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

    def zero_met(self):
        ds_met=xr.Dataset()

        # this is nt in the sample, but maybe time is okay??
        ds_met['time']=('time',),[self.run_start,self.run_stop]

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

        ds_met['Uwind']=const(('time','NUwind'), 0.0)


        ds_met['Vwind']=const(('time','NVwind'), 0.0)
        ds_met['Tair'] =const(('time','NTair'), 20.0)
        ds_met['Pair'] =const(('time','NPair'), 1000.) # units?
        ds_met['RH']=const(('time','NRH'), 80.)
        ds_met['rain']=const(('time','Nrain'), 0.)
        ds_met['cloud']=const(('time','Ncloud'), 0.5)

        return ds_met

    def partition(self):
        self.run_mpi(["-g","-vv","--datadir=%s"%self.run_dir])
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

