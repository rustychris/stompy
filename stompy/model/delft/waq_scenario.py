"""
Classes for hydrodynamic inputs (Hydro) and D-WAQ setup
(Scenario) for using Delft Water Quality.
"""

from __future__ import print_function

import os
import re
import textwrap
import shlex

import glob
import sys
import subprocess
import shutil
import datetime
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy import sparse
from ... import filters
from ... import utils
forwardTo = utils.forwardTo
import logging

import time
import pandas as pd
# 2023-05-23: avoid directly using MPL conversions. Go through
# stompy.utils, to maintain a single point of control for the definition
# of dnums.
#from matplotlib.dates import num2date, date2num

import matplotlib.pyplot as plt
from itertools import count
from six import iteritems
import six # next

from . import io as dio

try:
    from ...spatial import wkb2shp
except ImportError:
    print("wkb2shp not found - not loading/saving of shapefiles")

from shapely import geometry
try:
    from shapely.ops import cascaded_union
except ImportError:
    cascaded_union = None

from collections import defaultdict, OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import scipy.spatial

from  ... import scriptable
from ...io import qnc
import xarray as xr
from ...grid import unstructured_grid
from ...grid import ugrid
import threading

from . import nefis
from . import nefis_nc
from . import waq_process
from . import dfm_grid
from . import process_diagram

DEFAULT='_DEFAULT_'

def normalize_to_str(item):
    if isinstance(item,str): pass
    elif isinstance(item,bytes): item=item.decode()
    return item

def waq_timestep_to_timedelta(s):
    """ parse a delwaq-style timestep (as string or integer) into a python timedelta object.
    """
    s="%09d"%int(s)
    d, h, m, secs = [int(x) for x in [s[:-6], s[-6:-4], s[-4:-2], s[-2:]]]
    return datetime.timedelta(days=d, hours=h, minutes=m, seconds=secs)

def timedelta_to_waq_timestep(td):
    """
    Convert a python timedelta into a DWAQ style timestep string as in
    DDDHHMMSS
    """
    total_seconds = td.total_seconds()
    assert td.microseconds==0
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = (total_seconds // 3600) % 24
    days = (total_seconds // 86400)
    
    #                                       seconds
    #                                     minutes
    #                                   hours
    #                                days
    # hydrodynamic-timestep    '00000000 00 3000'
    
    return "%08d%02d%02d%02d"%(days, hours, mins, secs)

def rel_symlink(src, dst, overwrite=False):
    """ Create a symlink, adjusting for a src path relative
    to cwd rather than the directory of dst. 
    """
    # given two paths that are either absolute or relative to
    # pwd, create a symlink that includes the right number of
    # ../..'s
    if os.path.lexists(dst):
        assert not os.path.samefile(src,dst),"Attempt to symlink file to itself"
        if not overwrite:
            raise Exception("%s already exists, and overwrite is False"%dst)
        else:
            os.unlink(dst)
            
    if os.path.isabs(src): # no worries
        os.symlink(src, dst)
    else:
        pre = os.path.relpath(os.path.dirname(src), os.path.dirname(dst))
        os.symlink(os.path.join(pre, os.path.basename(src)), dst)


# Classes used in defining a water quality model scenario

CLOSED=0
BOUNDARY='boundary'

def tokenize(fp, comment=';'):
    """ 
    tokenize waq inputs, handling comments, possibly include.
    no casting.
    """
    for line in fp:
        items = line.split(comment)[0].strip().split()
        for tok in items:
            yield tok

            
class MonTail(object):
    """
    Helper class to watch output on a log file.  Used to track progress
    while model is running.
    """
    def __init__(self, mon_fn, log=None, sim_time_seconds=None):
        """ 
        mon_fn: path to delwaq2 monitor file
        log: logging object to which messages are sent via .info()
        sim_time_seconds: length of simulation, for calculation of relative speed
        """
        self.signal_stop=False
        self.sim_time_seconds=sim_time_seconds
        self.log=log
        if 1:
            self.thr=threading.Thread(target=self.tail, args=[mon_fn])
            self.thr.daemon=True
            self.thr.start()
        else:
            self.tail(mon_fn)
    def stop(self):
        self.signal_stop=True
        self.thr.join()

    def msg(self, s):
        if self.log == None:
            print(s)
        else:
            # can be annoying to get info to print, but less alarming than constantly
            # seeing warnings
            self.log.info(s)

    def tail(self, mon_fn):
        # We may have to wait for the file to exist...
        if not os.path.exists(mon_fn):
            self.msg("Waiting for %s to be created"%mon_fn)
            while not os.path.exists(mon_fn):
                if self.signal_stop:
                    self.msg("Got the signal to stop, but never saw file")
                    return
                time.sleep(0.1)
            self.msg("Okay - %s exists now"%mon_fn)

        # There is still the danger that the file changes size..
        sample_pcts=[]
        sample_secs=[]
        with open(mon_fn) as fp:
            # First, get up to the last line:
            last_line=""
            while not self.signal_stop:
                next_line=fp.readline()
                if next_line=='':
                    break
                last_line=next_line

            # and begin to tail:
            while not self.signal_stop:
                next_line=fp.readline()
                if next_line=='':
                    # a punt on cases where the file has been truncated.
                    fp.seek(0, 2) # seek to end
                    time.sleep(0.1)
                else:
                    last_line=next_line
                    if 'Completed' in last_line:
                        self.msg(last_line.strip())
                        try:
                            pct_complete=float(re.match(r'([0-9\.]+).*', last_line.strip()).group(1))
                            sample_pcts.append(pct_complete)
                            sample_secs.append(time.time())
                            if len(sample_pcts)>1:
                                # pct/sec
                                mb=np.polyfit(sample_secs, sample_pcts, 1)
                                if mb[0] != 0:
                                    time_remaining=(100 - sample_pcts[-1]) / mb[0]
                                    # mb[0] has units pct/sec
                                    etf=datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)
                                    if self.sim_time_seconds is not None:
                                        speed=self.sim_time_seconds*mb[0]/100.0
                                        speed="%.2fx realtime"%speed
                                    else:
                                        speed="n/a"
                                    self.msg("Time remaining: %.3fh (%s) %s"%(time_remaining/3600.,
                                                                              etf.strftime('%c'),
                                                                              speed))
                        except Exception as exc:
                            # please, just don't let this stupid thing stop the process
                            print(exc)


class WaqException(Exception):
    """
    Super class for exceptions specific to waq_scenario
    """
    pass

class Hydro(object):
    """
    Read/write hydrodynamics data for use as input to Delwaq
    """
    time0 = None # a datetime instance, for the *reference* time.
    # note that a hydro object may start at some nonzero offset 
    # from this reference time.

    # set by subclasses
    n_exch_x = None
    n_exch_y = None
    n_exch_z = None

    # this gets confusing - some input data includes exchanges to
    # inactive segments.  These cannot be omitted from pointers or
    # any exchange-centered arrays, since the sizes must be kept
    # consistent.  This flag signals that infer_2d_elements() should
    # not map inactive exchanges.  This is needed in some cases when
    # reading per-subdomain hydro from z-layer DFM.
    # There shouldn't be cases where this is a problem, but while it
    # tested, leave this flag in place so it's easier to find where
    # the change is
    omit_inactive_exchanges=True

    # if True, allow symlinking to original files where possible.
    # not directly relevant for some Hydro instances, but this will be
    # used as the default for parameters, too.
    enable_write_symlink=True

    # Checks are only present in a few places, but generally, write*()
    # methods should fail if files already exist and overwrite is not
    # true.
    overwrite=True

    @property
    def reference_originals(self):
        # HydroFiles can override, but other Hydro classes cannot
        # implement this, so force to False.
        return False

    @property
    def fn_base(self): # base filename for output. typically com-<scenario name>
        return 'com-{}'.format(self.scenario.name)

    t_secs=None # timesteps in seconds from time0 as 'i4'

    scenario=None

    # constants:
    CLOSED=CLOSED
    BOUNDARY=BOUNDARY

    def __init__(self, **kws):
        self.log=logging.getLogger(self.__class__.__name__)
        utils.set_keywords(self,kws)

    @property
    def t_dn(self):
        """ convert self.time0 and self.t_secs to datenums
        """
        from matplotlib.dates import num2date, date2num
        #return date2num(self.time0) + self.t_secs/86400.
        return utils.to_dnum(self.time0) + self.t_secs/86400.

    @property
    def time_step(self):
        """ Return an integer in DelWAQ format for the time step.
        i.e. ddhhmmss.  Assumes that scu is 1s.
        """
        dt_secs=np.diff(self.t_secs)
        dt_sec=dt_secs[0]
        assert np.all( dt_sec==dt_secs )

        rest, seconds=divmod(dt_sec, 60)
        rest, minutes=divmod(rest, 60)
        days, hours=divmod(rest, 24)
        return ((days*100 + hours)*100 + minutes)*100 + seconds

    # num_exch => use n_exch
    @property
    def n_exch(self):
        # unstructured would have num_exch_y==0
        return self.n_exch_x + self.n_exch_y + self.n_exch_z

    # num_seg => use n_seg
    n_seg=None # overridden in subclass or explicitly set.
    
    def areas(self, t):
        """ returns ~ np.zeros(self.n_exch,'f4'), for the timestep given by time t
        specified in seconds from time0.  areas in m2.
        """
        raise WaqException("Implement in subclass")

    def flows(self, t):
        """ returns flow rates ~ np.zeros(self.n_exch,'f4'), for given timestep.
        flows in m3/s.
        """
        raise WaqException("Implement in subclass")

    @property
    def scen_t_secs(self):
        """
        the subset of self.t_secs needed for the scenario's timeline
        this is the subset of times typically used when self.write() is called
        """
        hydro_datetimes=self.t_secs*self.scenario.scu + self.time0 
        start_i, stop_i=np.searchsorted(hydro_datetimes,
                                        [self.scenario.start_time,
                                         self.scenario.stop_time])
        if start_i>0:
            start_i-=1
        if stop_i <= len(self.t_secs):
            # careful to check <= so we don't drop the last time
            # step.
            stop_i+=1
        return self.t_secs[start_i:stop_i]

    @property 
    def are_filename(self):
        return os.path.join(self.scenario.base_path, self.fn_base+".are")
        
    def write_are(self):
        """
        Write are file
        """
        with open(self.are_filename, 'wb') as fp:
            for t_sec in utils.progress(self.scen_t_secs.astype('i4'),
                                        msg="writing area: %s"):
                fp.write(t_sec.tobytes()) # write timestamp
                fp.write(self.areas(t_sec).astype('f4').tobytes())

    @property
    def flo_filename(self):
        return os.path.join(self.scenario.base_path, self.fn_base+".flo")

    def write_flo(self):
        """
        Write flo file
        """
        with open(self.flo_filename, 'wb') as fp:
            for t_sec in utils.progress(self.scen_t_secs.astype('i4'),
                                        msg="writing flo: %s"):
                fp.write(t_sec.tobytes()) # write timestamp
                fp.write(self.flows(t_sec).astype('f4').tobytes())

    def seg_attrs(self, number):
        """ 
        1: active/inactive
          defaults to all active
        2: top/mid/bottom
          inferred from results of infer_2d_elements()
        """
        if number==1:
            # default, all active. may need to change this?
            return np.ones(self.n_seg, 'i4')
        if number==2:
            self.infer_2d_elements()

            # 0: single layer, 1: surface, 2: mid-water column, 3: bed
            attrs=np.zeros(self.n_seg, 'i4')

            for elt_i, sel in utils.enumerate_groups(self.seg_to_2d_element):
                if elt_i<0:
                    continue # inactive segments

                # need a different code if it's a single layer
                if len(sel)>1:
                    attrs[sel[0]]=1
                    attrs[sel[1:-1]]=2
                    attrs[sel[-1]]=3
                else:
                    attrs[sel[0]]=0 # top and bottom
            return attrs

    def text_atr(self):
        """ This used to return just the single number prefix (1 for all segs, no defaults)
        and the per-seg values.  Now it returns the entire attribute section, including 
        constant and time-varying.  No support yet for time-varying attributes, though.
        """

        # grab the values to find out which need to be written out.
        attrs1=self.seg_attrs(number=1)
        attrs2=self.seg_attrs(number=2)

        lines=[]
        count=0
        if np.any(attrs1!=1): # departs from default
            count+=1
            lines+=["1 1 1 ; num items, feature, input here",
                    "    1 ; all segs, without defaults",
                    "\n".join([str(a) for a in attrs1])]
        if np.any(attrs2!=0): # departs from default
            count+=1
            lines+=["1 2 1 ; num items, feature, input here",
                    "    1 ; all segs, without defaults",
                    "\n".join([str(a) for a in attrs2])]
        lines[:0]=["%d ; count of time-independent contributions"%count]
        lines.append(" 0    ; no time-dependent contributions")

        return "\n".join(lines)
                
    def write_atr(self):
        """
        write atr file
        """
        # might need to change with z-level aggregation
        with open(os.path.join(self.scenario.base_path, self.fn_base+".atr"), 'wt') as fp:
            fp.write(self.text_atr())

    # lengths from src segment to face, face to destination segment. 
    # in order of directions - x first, y second, z third.
    # [n_exch,2]*'f4' 
    exchange_lengths=None
    def write_len(self):
        """
        write len file
        """
        with open(os.path.join(self.scenario.base_path, self.fn_base+".len"),'wb') as fp:
            fp.write( np.array(self.n_exch, 'i4').tobytes() )
            fp.write(self.exchange_lengths.astype('f4').tobytes())

    # like np.zeros( (n_exch,4),'i4')
    # 2nd dimension is upwind, downwind, up-upwind, down-downwind
    # N.B. this format is likely specific to structured hydro
    pointers=None 

    def write_poi(self):
        """
        write poi file
        """
        with open(os.path.join(self.scenario.base_path, self.fn_base+".poi"), 'wb') as fp:
            fp.write(self.pointers.astype('i4').tobytes())

    def volumes(self, t):
        """ segment volumes in m3, [n_seg]*'f4'
        """
        raise WaqException("Implement in subclass")

    @property
    def vol_filename(self):
        return os.path.join(self.scenario.base_path, self.fn_base+".vol")

    def write_vol(self):
        """ write vol file
        """
        with open(self.vol_filename, 'wb') as fp:
            for t_sec in utils.progress(self.scen_t_secs.astype('i4'),
                                        msg="writing vol: %s"):
                fp.write(t_sec.tobytes()) # write timestamp
                fp.write(self.volumes(t_sec).astype('f4').tobytes())

    def vert_diffs(self, t):
        """ returns [n_segs]*'f4' vertical diffusivities in m2/s
        """
        raise WaqException("Implement in subclass")

    @property
    def flowgeom_filename(self):
        # This used to return just the basename, but that makes it different
        # than all other xxx_filename properties.
        return os.path.join(self.scenario.base_path,'flowgeom.nc')
    
    def write_geom(self):
        ds=self.get_geom()
        if ds is None:
            self.log.debug("This Hydro class does not support writing geometry")
            
        #dest=os.path.join(self.scenario.base_path,
        #                  self.flowgeom_filename)
        ds.to_netcdf(self.flowgeom_filename)
    def get_geom(self):
        # Return the geometry as an xarray / ugrid-ish Dataset.
        return None

    # How is the vertical handled in the grid?
    # affects outputting ZMODEL NOLAY in the inp file
    VERT_UNKNOWN=0
    ZLAYER=1
    SIGMA=2
    SINGLE=3
    _vertical=None
    @property
    def vertical(self):
        if self._vertical is not None:
            return self._vertical
        
        geom=self.get_geom()
        if geom is None:
            return self.VERT_UNKNOWN
        for v in geom.variables:
            standard_name=geom[v].attrs.get('standard_name', None)
            if standard_name == 'ocean_sigma_coordinate':
                return self.SIGMA
            if standard_name == 'ocean_zlevel_coordinate':
                return self.ZLAYER
        return self.VERT_UNKNOWN
    @vertical.setter
    def vertical(self, v):
        self._vertical = v
    
    def grid(self):
        """ if possible, return an UnstructuredGrid instance for the 2D 
        layout.  returns None if the information is not available.
        """
        return None

    _params=None
    # force had been false, but it doesn't play well with running multiple
    # scenarios with the same hydro.
    def parameters(self, force=True):
        if force or (self._params is None):
            hyd=NamedObjects(scenario=self.scenario,cast_value=cast_to_parameter)
            self._params = self.add_parameters(hyd)
        return self._params
        
    def add_parameters(self, hyd):
        """ Moved from waq_scenario init_hydro_parameters
        """
        self.log.debug("Adding planform areas parameter")
        hyd['SURF']=self.planform_areas()
        
        try:
            self.log.debug("Adding bottom depths parameter")
            hyd['bottomdept']=self.bottom_depths()
        except NotImplementedError:
            self.log.info("Bottom depths will be inferred")

        try:
            # Hmm - clunky, but not a good existing way to deterine whether
            # there is vertical diffusion data or not
            self.vert_diffs(0)
            has_vert_diffs=True
        except Exception as exc:
            self.log.info("No vertical dispersion (%s)"%str(exc))
            has_vert_diffs=False

        if has_vert_diffs:
            self.log.debug("Adding VertDisper parameter")
            hyd['VertDisper']=ParameterSpatioTemporal(func_t=self.vert_diffs,
                                                      times=self.t_secs,
                                                      hydro=self)

        self.log.debug("Adding depths parameter")
        try:
            hyd['DEPTH']=self.depths()
        except NotImplementedError:
            self.log.info("Segment depth will be inferred")
        return hyd

    def write(self):
        """
        Write hydro data out or link to existing files.
        overwrite: if true, will overwrite existing files.  Not fully implemented yet,
        no guarantees.
        """
        self.log.debug('Writing 2d links')
        self.write_2d_links()
        self.log.debug('Writing boundary links')
        self.write_boundary_links()
        self.log.debug('Writing attributes')
        self.write_atr()
        self.log.info('Writing hyd file')
        self.write_hyd()
        self.log.info('Writing srf file')
        self.write_srf()
        self.log.info('Writing hydro parameters')
        self.write_parameters()
        self.log.debug('Writing geom')
        self.write_geom()
        self.log.debug('Writing areas')
        self.write_are()
        self.log.debug('Writing flows')
        self.write_flo()
        self.log.debug('Writing lengths')
        self.write_len()
        self.log.debug('Writing pointers')
        self.write_poi()
        self.log.debug('Writing volumes')
        self.write_vol()

    def write_srf(self):
        if 0: # old Hydro behavior:
            self.log.info("No srf to write")
        else:
            try:
                plan_areas=self.planform_areas()
            except WaqException as exc:
                self.log.warning("No planform areas to write")
                return
            
            self.infer_2d_elements()
            nelt=self.n_2d_elements
            
            # painful breaking of abstraction.
            if isinstance(plan_areas, ParameterSpatioTemporal):
                surfaces=plan_areas.evaluate(t=0).data
            elif isinstance(plan_areas, ParameterSpatial):
                surfaces=plan_areas.data
            elif isinstance(plan_areas, ParameterConstant):
                surfaces=plan_areas.value * np.ones(nelt, 'f4')
            elif isinstance(plan_areas, ParameterTemporal):
                surfaces=plan_areas.values[0] * np.ones(nelt, 'f4')
            else:
                raise Exception("plan areas is %s - unhandled"%(str(plan_areas)))

            # this needs to be in sync with what write_hyd writes, and
            # the supporting_file statement in the hydro_parameters
            fn=os.path.join(self.scenario.base_path, self.surf_filename)
            
            with open(fn, 'wb') as fp:
                # shape, shape, count, x,x,x according to waqfil.m
                hdr=np.zeros(6, 'i4')
                hdr[0]=hdr[2]=hdr[3]=hdr[4]=nelt
                hdr[1]=1
                hdr[5]=0
                fp.write(hdr.tobytes())
                fp.write(surfaces.astype('f4'))
        
    def write_parameters(self):
        # parameters are updated with force=True on Scenario instantiation,
        # don't need to do it here.
        for param in self.parameters(force=False).values():
            # don't care about the textual description
            _=param.text(write_supporting=True)
    def planform_areas(self):
        """
        return Parameter, typically ParameterSpatial( Nsegs * 'f4' )
        """
        raise WaqException("Implement in subclass")

    def depths(self):
        raise NotImplementedError("This class does not directly provide depth")

    def bottom_depths(self):
        """ 
        return Parameter, typically ParameterSpatial( Nsegs * 'f4' )
        """
        raise NotImplementedError("Implement in subclass")

    def element_depth(self,t_secs=0):
        """
        Returns array of total water column depth per 2d element. 
        """
        return self.seg_z_range(t_secs)[1][-self.n_2d_elements:]
    
    def seg_active(self):
        # this is now just a thin wrapper on seg_attrs
        return self.seg_attrs(number=1).astype('b1')

    seg_to_2d_element=None
    seg_k=None
    n_2d_elements=0
    def infer_2d_elements(self):
        """
        populates seg_to_2d_element: [n_seg] 0-based indices,
        mapping each segment to its 0-based 2d element
        Also populates self.seg_k as record of vertical layers of segments.
        inactive segments are not assigned any of these.
        """
        if self.seg_to_2d_element is None:
            n_2d_elements=0
            seg_to_2d=np.zeros(self.n_seg, 'i4')-1 # 0-based segment => 0-based 2d element.
            # 0-based layer, k=0 is surface
            # accuracy seg_k depends on prismatic topology of cells
            seg_k=np.zeros(self.n_seg, 'i4')-1 

            poi=self.pointers
            #poi_vert=poi[-self.n_exch_z:] # unsafe with 2D!
            poi_vert=poi[self.n_exch_x + self.n_exch_y:]

            # don't make any assumptions about layout -
            # but by enumerating 2D segments in the same order as the
            # first segments, should preserve ordering from the top layer.

            # really this should use self.seg_to_exchs, so that we don't
            # duplicate preprocessing.  Another day.
            # preprocess neighbor queries:
            nbr_up=defaultdict(list) 
            nbr_down=defaultdict(list)

            self.log.debug("Inferring 2D elements, preprocess adjacency")

            seg_active=self.seg_active()

            # all 0-based
            for seg_from, seg_to in (poi_vert[:,:2] - 1):
                # N.B. this includes some exchanges which may go to
                # inactive segments, but that is checked below
                nbr_up[seg_to].append(seg_from)
                nbr_down[seg_from].append(seg_to)

            for seg in range(self.n_seg): # 0-based segment
                if seg%50000==0:
                    self.log.info("Inferring 2D elements, %d / %d 3-D segments"%(seg,self.n_seg))

                if not seg_active[seg]:
                    continue

                def trav(seg,elt,k):
                    # mark this segment as being from 2d element elt,
                    # and mark any unmarked segments vertically adjacent
                    # with the same element.
                    # returns the number segments marked
                    if seg_to_2d[seg]>=0:
                        return 0 # already marked
                    if self.omit_inactive_exchanges and not seg_active[seg]:
                        return 0 # don't mark inactive segments even if they have an exchange
                    
                    seg_to_2d[seg]=elt
                    seg_k[seg]=k
                    count=1

                    # would like to check departures from the standard
                    # format where (i) vertical exchanges are upper->lower,
                    # and all of the top-layer segments come first.

                    v_nbrs1=nbr_down[seg]
                    v_nbrs2=nbr_up[seg]
                    v_nbrs=v_nbrs1+v_nbrs2

                    # extra check on conventions
                    for nbr in v_nbrs2: # a segment 'above' us
                        if nbr>0 and seg_to_2d[nbr]<0:
                            # not a boundary, and not visited
                            self.log.warning("infer_2d_elements: spurious segments on top.  segment ordering may be off")

                    for nbr in v_nbrs1:
                        if nbr>=0: # not a boundary
                            count+=trav(nbr,elt,k+1)
                    for nbr in v_nbrs2: 
                        # really shouldn't hit any segments this way
                        if nbr>=0:
                            count+=trav(nbr,elt,k-1)
                    return count

                if trav(seg,n_2d_elements,k=0):
                    # print("traverse from seg=%d incrementing n_2d_elements from %d"%(seg,n_2d_elements))
                    n_2d_elements+=1
            self.n_2d_elements=n_2d_elements
            self.seg_to_2d_element=seg_to_2d
            self.seg_k=seg_k
        return self.seg_to_2d_element

    def segment_select(self,element,k):
        """
        Identify segment IDs by element and layer. This was referenced
        in some Load code, but somehow did not exist in hydro. May be expanded in the
        future to allow other means of identifying segments.
        returns an array of segment ids
        """
        self.infer_2d_elements()
        idxs=np.nonzero( (self.seg_to_2d_element==element) & (self.seg_k==k) )[0]
        return idxs
    
    def extrude_element_to_segment(self,V):
        """ V: [n_2d_elements] array
        returns [n_seg] array
        """
        self.infer_2d_elements()
        return V[self.seg_to_2d_element]

    def seg_z_range(self,t_secs):
        """
        Calculate seg_ztop, seg_zbot as depth from water surface to the
        top and bottom of each segment.
        """
        vols=self.volumes(t_secs)
        areas=self.planform_areas().data # a top-down area for each segment.
        dzs=vols/areas

        # depth below water surface of the bottom of the segment prism
        self.infer_2d_elements()
        n_layer = self.n_seg // self.n_2d_elements
        dzs_2d=dzs.reshape((n_layer,-1))
        seg_zbot=np.cumsum(dzs_2d,axis=0).ravel()
        seg_ztop=seg_zbot-dzs
        return seg_ztop,seg_zbot

    # hash of segment id (0-based) to list of exchanges
    # order by horizontal, decreasing z, then vertical, decreasing z.
    _seg_to_exchs=None 
    def seg_to_exchs(self,seg):
        if self._seg_to_exchs is None:
            self._seg_to_exchs=ste=defaultdict(list)
            for exch,(s_from,s_to,dumb,dumber) in enumerate(self.pointers):
                ste[s_from-1].append(exch)
                ste[s_to-1].append(exch)
        return self._seg_to_exchs[seg]
            
    def seg_to_exch_z(self,preference='lower'):
        """ Map 3D segments to an associated vertical exchange
        (i.e. for getting areas)
        preference=='lower': will give preference to an exchange lower down in the watercolum
        NB: if a water column has only one layer, the corresponding 
        exch index will be set to -1.
        """
        nz=self.n_exch-self.n_exch_z # first index of vert. exchanges
        
        seg_z_exch=np.zeros(self.n_seg,'i4')
        pointers=self.pointers
        
        warned=False

        for seg in range(self.n_seg):
            # used to have a filter expression here, but that got weird in Py3k.
            vert_exchs=[s for s in self.seg_to_exchs(seg) if s>=nz]

            if len(vert_exchs)==0:
                # dicey! some callers may not expect this.
                if not warned:
                    self.log.warning("seg %d has 1 layer - no z exch"%seg)
                    self.log.warning("further warnings suppressed")
                    warned=True
                vert_exch=-1
            elif preference=='lower':
                vert_exch=vert_exchs[-1] 
            elif preference=='upper':
                vert_exch=vert_exchs[0]
            else:
                raise ValueError("Bad preference value: %s"%preference)
            seg_z_exch[seg]=vert_exch
        return seg_z_exch

    def check_volume_conservation_nonincr(self):
        """
        Compare time series of segment volumes to the integration of 
        fluxes.  This version loads basically everything into RAM,
        so should only be used with very simple models.
        """
        flows=[] 
        volumes=[]

        # go through the time variation, see if we can show that volume
        # is conserved
        print("Loading full period, aggregated flows and volumes")
        for ti,t in enumerate(self.t_secs):
            sys.stdout.write('.') ; sys.stdout.flush()
            if (ti+1)%50==0:
                print()

            flows.append( self.flows(t) )
            volumes.append( self.volumes(t) )
        print()
        flows=np.array(flows)
        volumes=np.array(volumes)

        print("Relative error in volume conservation.  Expect 1e-6 with 32-bit floats")
        for seg in range(self.n_agg_segments):
            seg_weight=np.zeros( self.n_exch )
            seg_weight[ self.agg_exch['from']==seg ] = -1
            seg_weight[ self.agg_exch['to']==seg ] = 1

            seg_Q=np.dot(seg_weight,flows.T)
            dt=np.diff(self.t_secs)
            seg_dV=seg_Q*np.median(dt)

            pred_V=volumes[0,seg]+np.cumsum(seg_dV[:-1])
            err=volumes[1:,seg] - pred_V
            rel_err=err / volumes[:,seg].mean()
            rmse=np.sqrt( np.mean( rel_err**2 ) )
            print(rmse)

    _QtodV=None
    _QtodVabs=None
    def mats_QtodV(self):
        # refactored out of check_volume_conservation_incr()
        if self._QtodV is None:
            # build a sparse matrix for mapping exchange flux to segments
            # QtodV.dot(Q): rows of QtodV correspond to segment
            # columns correspond to exchanges
            rows=[]
            cols=[]
            vals=[]

            for exch_i,(seg_from,seg_to) in enumerate(self.pointers[:,:2]):
                if seg_from>0:
                    rows.append(seg_from-1)
                    cols.append(exch_i)
                    vals.append(-1.0)
                if seg_to>0:
                    rows.append(seg_to-1)
                    cols.append(exch_i)
                    vals.append(1.0)

            QtodV=sparse.coo_matrix( (vals, (rows,cols)),
                                     (self.n_seg,self.n_exch) )
            QtodVabs=sparse.coo_matrix( (np.abs(vals), (rows,cols)),
                                        (self.n_seg,self.n_exch) )
            self._QtodV = QtodV
            self._QtodVabs=QtodVabs
        return self._QtodV,self._QtodVabs

    def check_volume_conservation_incr(self,seg_select=slice(None),
                                       tidx_select=slice(None),
                                       err_callback=None,
                                       verbose=True):
        """
        Compare time series of segment volumes to the integration of 
        fluxes.  This version loads just two timesteps at a time,
        and also includes some more generous checks on how well the
        fluxes close.

        seg_select: an slice or bitmask to select which segments are
        included in error calculations.  Use this to omit ghost segments, 
        for instance.

        err_callback(time_index,error_summary): called for each interval
         checked, with the time index and summary of errors.  The first time
         step (or first of subset when tidx_select is used) is not returned,
         since the calculation requires two timesteps.
         
        see code below for fields of error_summary.  Callback used to get
        just relative error, but now it's a struct array.

        verbose: print detailed information about the errors, and the worst error.

        returns the last summary (i.e. same thing handed to err_callback)
        """
        assert (tidx_select.step is None) or (tidx_select.step==1),"Times must be consecutive"
        t_secs=self.t_secs[tidx_select]
        t_idxs=np.arange(len(self.t_secs))[tidx_select]

        QtodV,QtodVabs = self.mats_QtodV()

        try:
            plan_areas=self.planform_areas()
        except OSError: # FileNotFoundError only exists in >=3.3
            plan_areas=None

        # Supply more info to the callback
        # rel_err: abs(err)/(V+abs(dV))
        # vol_err: error in volume, as Vnow - Vpred
        # Q_err: vol_err/dt
        segs=np.arange(self.n_seg)[seg_select]
        summary=np.zeros(len(segs),[('seg','i4'),('rel_err','f8'),('vol_err','f8'),('Q_err','f8')])
        summary['seg']=segs
            
        # Iterate - i is the count within the steps we're analyzing,
        # ti is the index into all timesteps, with corresponding timestamps
        # t_secs[i]
        for i,ti in enumerate(t_idxs):
            t_sec=t_secs[i]
            Vnow=self.volumes(t_sec)

            if plan_areas is not None:
                seg_plan_areas=plan_areas.evaluate(t=t_sec).data
            else:
                seg_plan_areas=None

            if i>0:
                dt=t_secs[i]-t_secs[i-1]
                dVmag=QtodVabs.dot(np.abs(Qlast)*dt)
                Vpred=Vlast + QtodV.dot(Qlast)*dt

                err=Vnow - Vpred
                valid=(Vnow+dVmag)!=0.0
                # rel_err=np.abs(err) / (Vnow+dVmag)
                rel_err=np.zeros(len(err),'f8')
                rel_err[valid]=np.abs(err[valid])/(Vnow+dVmag)[valid]
                rel_err[~valid] = np.abs(err[~valid])

                # Supply more info to the callback
                # seg: 0-based segment index
                # rel_err: abs(err)/(V+abs(dV))
                # vol_err: error in volume, as Vnow - Vpred
                # Q_err: vol_err/dt
                summary['rel_err'][:]=rel_err[seg_select]
                summary['vol_err'][:]=err[seg_select]
                summary['Q_err'][:]=summary['vol_err']/dt
                                           
                if err_callback:
                    err_callback(ti,summary)

                rmse=np.sqrt( np.mean( rel_err[seg_select]**2 ) )
                if verbose and (rel_err[seg_select].max() > 1e-4):
                    self.log.warning("****************BAD Volume Conservation*************")
                    self.log.warning("  t=%10d   RMSE: %e    Max rel. err: %e"%(t_sec,rmse,rel_err[seg_select].max()))
                    self.log.warning("  %d segments above rel err tolerance"%np.sum( rel_err[seg_select]>1e-4 ))
                    self.log.info("Bad segments: %s"%( np.nonzero( rel_err[seg_select]>1e-4 )[0] ) )

                    bad_seg=np.arange(len(rel_err))[seg_select][np.argmax(rel_err[seg_select])]
                    self.log.warning("  Worst segment is index %d"%bad_seg)
                    self.log.warning("  Vlast=%f  Vpred=%f  Vnow=%f"%(Vlast[bad_seg],
                                                         Vpred[bad_seg],
                                                         Vnow[bad_seg]))
                    if seg_plan_areas is not None:
                        self.log.warning("  z error=%f m"%( err[bad_seg] / seg_plan_areas[bad_seg] ))
                    Vin =dt*Qlast[ self.pointers[:,1] == bad_seg+1 ]
                    Vout=dt*Qlast[ self.pointers[:,0] == bad_seg+1 ]
                    bad_Q=np.concatenate( [Vin,-Vout] )
                    Qmag=np.max(np.abs(bad_Q))
                    self.log.warning("  Condition of dV: Qmag=%f Qnet=%f mag/net=%f"%(Qmag,bad_Q.sum(),Qmag/bad_Q.sum()))

            Qlast=self.flows(t_sec)
            Vlast=Vnow
        return summary

    # Boundary handling
    # this representation follows the naming in the input file
    boundary_dtype=[('id','S20'),
                    ('name','S20'),
                    ('type','S20')]
    @property
    def n_boundaries(self):
        return -self.pointers[:,:2].min()

    # for a while, was using 'grouped', but that relies on boundary segments,
    # which didn't transfer well to aggregated domains which usually have many
    # unaggregated links going into the same aggregated element.
    boundary_scheme='lgrouped'
    _boundary_defs=None
    def boundary_defs(self):
        """ Generic boundary defs - types default to ids
        """
        if self._boundary_defs is None:
            Nbdry=self.n_boundaries

            bdefs=np.zeros(Nbdry, self.boundary_dtype)
            for i in range(Nbdry):
                bdefs['id'][i]="boundary %d"%(i+1)
            bdefs['name']=bdefs['id']
            if self.boundary_scheme=='id':
                bdefs['type']=bdefs['id']
            elif self.boundary_scheme in ['element','grouped']:
                self.log.warning("This is outdated, and may be incorrect for DFM output with out-of-order exchanges")
                bc_segs=self.bc_segs() 
                self.infer_2d_elements()
                if self.boundary_scheme=='element':
                    bdefs['type']=["element %d"%( self.seg_to_2d_element[seg] )
                                   for seg in bc_segs]
                elif self.boundary_scheme=='grouped':
                    bc_groups=self.group_boundary_elements()
                    bdefs['type']=[ bc_groups['name'][self.seg_to_2d_element[seg]] 
                                    for seg in bc_segs]
            elif self.boundary_scheme == 'lgrouped':
                bc_lgroups=self.group_boundary_links()
                if 0:
                    # old way, assume the order of things we return here should
                    # follow the order of they appearin pointers
                    bc_exchs=np.nonzero(self.pointers[:,0]<0)[0]
                    self.infer_2d_links()
                    bdefs['type']=[ bc_lgroups['name'][self.exch_to_2d_link['link'][exch]] 
                                    for exch in bc_exchs]
                else:
                    #self.log.info("Slowly setting boundary info")
                    # Not so slow anymore
                    boundary_exchs=np.nonzero(self.pointers[:,0]<0)[0] # these are the
                    boundary_segs = -self.pointers[boundary_exchs,0]

                    for bdry0 in range(Nbdry):
                        # This scans the whole pointer table
                        # exchs=np.nonzero(-self.pointers[:,0] == bdry0+1)[0]
                        # This still scans (meh), but only the boundary exchanges
                        exchs=boundary_exchs[boundary_segs==bdry0+1]

                        assert len(exchs)==1 # may be relaxable.
                        exch=exchs[0]
                        # 2018-11-29: getting some '0' types.
                        # this is because exch_link is mapping to a non-boundary
                        # link.
                        exch_link=self.exch_to_2d_link['link'][exch]
                        bdefs['type'][bdry0] = bc_lgroups['name'][exch_link]
                    self.log.info("Done setting boundary info")
            else:
                raise ValueError("Boundary scheme is bad: %s"%self.boundary_scheme)
            self._boundary_defs=bdefs
        return self._boundary_defs

    def bc_segs(self):
        # Return an array of segments (0-based) corresponding to the receiving side of
        # the boundary exchanges
        poi=self.pointers
        # need to associate an internal segment with each bc exchange
        bc_exch=(poi[:,0]<0)
        bc_external=-1-poi[bc_exch,0]
        assert bc_external[0]==0
        assert np.all( np.diff(bc_external)==1)
        return poi[bc_exch,1]-1

    def datetime_to_index(self,dt):
        """ takes a scalar argument convertible to a datetime via utils, 
        and return the timestep index for this Hydro data.
        """
        t_sec=(utils.to_datetime(dt) - self.time0).total_seconds()
        return self.t_sec_to_index(t_sec)
    
    def t_sec_to_index(self,t):
        """ 
        This used to be called time_to_index, but since it accepts time as an integer
        number of seconds, better to make that clear in the name
        """
        return np.searchsorted(self.t_secs,t).clip(0,len(self.t_secs)-1)

    # not really that universal, but moving towards a common
    # data structure which includes names for elements, in 
    # which case this maps element names to indexes
    def coerce_to_element_index(self,x,return_boundary_name=True): 
        if isinstance(x,str):
            try:
                x=np.nonzero( self.elements['name']==x )[0][0]
            except IndexError:
                if return_boundary_name:
                    # probably got a string that's the name of a boundary
                    return x
                else:
                    raise
        return x

    def timeline_scen(self):
        """
        This used to be timeline_data, but the implementation was not consistent 
        between classes.
        For clarity, it is now timeline_scen to reflect that it limits the time 
        frame to that of the scenario.
        """
        scu=self.scenario.scu
        time_start = (self.time0+self.scen_t_secs[0] *scu)
        time_stop  = (self.time0+self.scen_t_secs[-1]*scu)
        timedelta  = (self.t_secs[1] - self.t_secs[0])*scu
        return time_start,time_stop,timedelta

    def timeline_data(self):
        """
        Hydro::timeline_data() used to used scen_t_secs, but HydroFiles
        contined to use t_secs.  This is now explicitly reflecting the
        period of the data, and timeline_scen() is the time frame for
        the scenario's subset of the data.
        """
        scu=self.scenario.scu
        time_start = (self.time0+self.t_secs[0] *scu)
        time_stop  = (self.time0+self.t_secs[-1]*scu)
        timedelta  = (self.t_secs[1] - self.t_secs[0])*scu
        return time_start,time_stop,timedelta
    
    def write_hyd(self,fn=None):
        """ Write an approximation to the hyd file output by D-Flow FM
        for consumption by delwaq

        DwaqAggregator has a good implementation, but with some
        specialization which would need to be factored out for here.

        That implementation has been copied here, and is in the process
        of being fixed to more general usage.

        Write an approximation to the hyd file output by D-Flow FM
        for consumption by delwaq or HydroFiles
        respects scen_t_secs
        """
        # currently the segment names here are out of sync with 
        # the names used by write_parameters.
        #  this is relevant for salinity-file,  vert-diffusion-file
        #  maybe surfaces-file, depths-file.
        # for example, surfaces file is written as tbd-SURF.seg
        # but below we call it com-tbd.srf
        # maybe easiest to just change the code below since it's
        # already arbitrary
        fn=fn or os.path.join( self.scenario.base_path,
                               self.fn_base+".hyd")
        if os.path.exists(fn):
            if self.overwrite:
                os.unlink(fn)
            else:
                self.log.warning("hyd file %s already exists.  Not overwriting!"%fn)
                return
        
        name=self.scenario.name

        dfmt="%Y%m%d%H%M%S"

        scu=self.scenario.scu

        # If symlinking, we want to report the full time period.
        if self.enable_write_symlink:
            time_start,time_stop,timedelta=self.timeline_data()
        else:
            time_start,time_stop,timedelta = self.timeline_scen()
            
        timestep = timedelta_to_waq_timestep(timedelta)

        self.infer_2d_elements()
        n_layers=1+self.seg_k.max()

        # New code - maybe not right at all.
        # This code is also duplicated across several of the classes in this file.
        # crying out for refactoring.
        if 'temp' in self.parameters():
            temp_file="'%s-temp.seg'"%name
        else:
            temp_file='none'
            
        if 'tau' in self.parameters():
            tau_file="'%s-tau.seg'"%name
        else:
            tau_file='none'
            
        lines=[
            "file-created-by  SFEI, waq_scenario.py",
            "file-creation-date  %s"%( datetime.datetime.utcnow().strftime('%H:%M:%S, %d-%m-%Y') ),
            "task      full-coupling",
            "geometry  unstructured",
            "horizontal-aggregation no",
            "reference-time           '%s'"%( self.time0.strftime(dfmt) ),
            "hydrodynamic-start-time  '%s'"%( time_start.strftime(dfmt) ),
            "hydrodynamic-stop-time   '%s'"%( time_stop.strftime(dfmt)  ),
            "hydrodynamic-timestep    '%s'"%timestep, 
            "conversion-ref-time      '%s'"%( self.time0.strftime(dfmt) ),
            "conversion-start-time    '%s'"%( time_start.strftime(dfmt) ),
            "conversion-stop-time     '%s'"%( time_stop.strftime(dfmt)  ),
            "conversion-timestep      '%s'"%timestep, 
            "grid-cells-first-direction       %d"%self.n_2d_elements,
            "grid-cells-second-direction          0",
            "number-hydrodynamic-layers          %s"%( n_layers ),
            "number-horizontal-exchanges      %d"%( self.n_exch_x ),
            "number-vertical-exchanges        %d"%( self.n_exch_z ),
            # little white lie.  this is the number in the top layer.
            # and no support for water-quality being different than hydrodynamic
            "number-water-quality-segments-per-layer       %d"%( self.n_2d_elements),
            "number-water-quality-layers          %s"%( n_layers ),
            "hydrodynamic-file        '%s'"%self.fn_base,
            "aggregation-file         none",
            # filename handling not as elegant as it could be..
            # e.g. self.vol_filename should probably be self.vol_filepath, then
            # here we could reference the filename relative to the hyd file
            "grid-indices-file     '%s.bnd'"%self.fn_base,# lies, damn lies
            "boundaries-file       '%s.bnd'"%self.fn_base, # this one might be true.
            "grid-coordinates-file '%s'"%os.path.basename(self.flowgeom_filename),
            "attributes-file       '%s.atr'"%self.fn_base,
            "volumes-file          '%s.vol'"%self.fn_base,
            "areas-file            '%s.are'"%self.fn_base,
            "flows-file            '%s.flo'"%self.fn_base,
            "pointers-file         '%s.poi'"%self.fn_base,
            "lengths-file          '%s.len'"%self.fn_base,
            "salinity-file         '%s-salinity.seg'"%name,
            "temperature-file      %s"%temp_file,
            "vert-diffusion-file   '%s-vertdisper.seg'"%name,
            # not a segment function!
            "surfaces-file         '%s'"%self.surf_filename,
            "shear-stresses-file   %s"%tau_file,
            "hydrodynamic-layers",
            "\n".join( ["%.5f"%(1./n_layers)] * n_layers ),
            "end-hydrodynamic-layers",
            "water-quality-layers   ",
            "\n".join( ["1.000"] * n_layers ),
            "end-water-quality-layers"]
        txt="\n".join(lines)
        with open(fn,'wt') as fp:
            fp.write(txt)

    @property
    def surf_filename(self):
        return self.fn_base+".srf"

    n_2d_links=None
    exch_to_2d_link=None
    links=None
    def infer_2d_links(self):
        """
        populate self.n_2d_links, self.exch_to_2d_link, self.links 
        note: compared to the incoming grid, this may include internal
        boundary exchanges.
        exchanges are identified based on unique from/to pairs of 2d elements.

        """
        if self.exch_to_2d_link is None:
            self.infer_2d_elements() 
            poi0=self.pointers-1

            #  map 0-based exchange index to 0-based link index
            exch_to_2d_link=np.zeros(self.n_exch_x+self.n_exch_y,[('link','i4'),
                                                                  ('sgn','i4')])
            exch_to_2d_link['link']=-1

            #  track some info about links
            links=[] # elt_from,elt_to
            mapped=dict() # (src_2d, dest_2d) => link idx

            # hmm - if there are multiple boundary exchanges coming into the
            # same segment, how can those be differentiated?  probably it's just
            # up to the sub-implementations to make the distinction.
            # so here they will get lumped together, but the datastructure should
            # allow for them to be distinct.

            seg_bc_count=defaultdict(int)
            
            for exch_i,(a,b,_,_) in enumerate(poi0[:self.n_exch_x+self.n_exch_y]):
                if a<0 and b<0:
                    # probably DFM writing dense exchanges information for segment
                    # which don't exist.  I think it's safest to just not include these
                    # at all.
                    continue
                
                if a>=0:
                    a2d=self.seg_to_2d_element[a]
                else:
                    # this is a source of the problem mentioned above, since
                    # we throw away an unique identity of distinct boundary
                    # exchanges here.
                    # a2d=-1 # ??
                    # instead, count up how many bc exchanges hit this segment
                    # first one will be -1, but we can see distinct additional
                    # exchanges as -2, -3, etc.
                    # the 'b' here is because we are labeling boundary a values
                    # with the count of bcs entering b.
                    seg_bc_count[b]+=1
                    a2d=-seg_bc_count[b]
                if b>=0:
                    b2d=self.seg_to_2d_element[b]
                else:
                    #b2d=-1 # ??
                    # see above comments for a
                    seg_bc_count[a]+=1
                    b2d=-seg_bc_count[a]

                # cruft -- just a reminder, now tested with a,b above.
                #if a2d<0 and b2d<0:
                #    # probably DFM writing dense exchanges information for segment
                #    # which don't exist.  I think it's safest to just not include these
                #    # at all.
                #    continue
                
                if (b2d,a2d) in mapped:
                    exch_to_2d_link['link'][exch_i] = mapped[(b2d,a2d)]
                    exch_to_2d_link['sgn'][exch_i]=-1
                else:
                    k=(a2d,b2d)
                    if k not in mapped:
                        mapped[k]=len(links)
                        links.append( [a2d,b2d] )
                    exch_to_2d_link['link'][exch_i] = mapped[k]
                    exch_to_2d_link['sgn'][exch_i]=1

            links=np.array(links)
            n_2d_links=len(links)

            # Bit of a sanity warning on multiple boundary exchanges involving the
            # same segment - this would indicate that there should be multiple 2D
            # links into that segment, but this generic code doesn't have a robust
            # way to deal with that.
            if 1:
                # indexes of which links are boundary
                bc_links=np.nonzero( links[:,0] < 0 )[0]

                for bc_link in bc_links:
                    # index of which exchanges map to this link
                    exchs=np.nonzero( exch_to_2d_link['link']==bc_link )[0]
                    # link id, sgn for each of those exchanges
                    ab=exch_to_2d_link[exchs]
                    # find the internal segments for each of those exchanges
                    segs=np.zeros(len(ab),'i4')
                    sel0=exch_to_2d_link['sgn'][exchs]>0 # regular order
                    segs[sel0]=poi0[exchs[sel0],1]
                    if np.any(~sel0):
                        # including checking for weirdness
                        self.log.warning("Some exchanges had to be flipped when flattening to 2D links")
                        segs[~sel0]=poi0[exchs[~sel0],0]
                    # And finally, are there any duplicates into the same segment? i.e. a segment
                    # which has multiple boundary exchanges which we have failed to distinguish (since
                    # in this generic implementation we have little info for distinguishing them).
                    # note that in the case of suntans output, this is possible, but if it has been
                    # mapped from multiple domains to a global domain, those exchanges have probably
                    # already been combined.
                    if len(np.unique(segs)) < len(segs):
                        self.log.warning("In flattening exchanges to links, link %d has ambiguous multiple exchanges for the same segment"%bc_link)

            self.exch_to_2d_link=exch_to_2d_link
            self.links=links
            self.n_2d_links=n_2d_links
    def write_2d_links(self):
        """
        Write the results of infer_2d_links to two text files - directly mirror the
        structure of exch_to_2d_link and links.
        """
        self.infer_2d_links()
        path = self.scenario.base_path
        np.savetxt(os.path.join(path,'links.csv'),
                   self.links,fmt='%d')
        np.savetxt(os.path.join(path,'exch_to_2d_link.csv'),
                   self.exch_to_2d_link,fmt='%d')

    def flowlink_to_edge(self,g):
        """
        Create a sparse matrix that maps a per-flowlink, signed quantity (i.e. flow)
        to the edges of g. BC flows entering at the boundary are handled, but
        internal outfalls are ignored.
        """
        from scipy import sparse
        self.infer_2d_links()
        M=sparse.dok_matrix( (g.Nedges(),self.n_2d_links), np.float64)
        e2c=g.edge_to_cells()
        geom=self.get_geom()

        cc=g.cells_center()
        elem_xy=np.c_[ geom.FlowElem_xcc.values,
                       geom.FlowElem_ycc.values ]

        def elt_to_cell(elt):
            # in general elts are preserved as the same cell index,
            # and this is actually more robust then the geometry
            # check because of some non-orthogonal cells that have
            # a circumcenter outside the corresponding cell.
            if utils.dist(elem_xy[elt] - cc[elt])<2.0:
                return elt
            # in a few cases the circumcenter is not inside the cell,
            # so better to select the nearest circumcenter than the
            # cell containing it.
            c=g.select_cells_nearest(elem_xy[elt],inside=False)
            assert c is not None
            return c
        
        for link,(eltA,eltB) in utils.progress(enumerate(self.links)):
            assert eltB>=0
            cB=elt_to_cell(eltB)
            
            if eltA<0: # eltA<0 means it's a boundary.
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
            M[j,link]=sgn
        return M
                
    def path_to_transect_exchanges(self,xy,on_boundary='warn_and_skip',on_edge=False):
        """
        xy: [N,2] points.
        Each point is mapped to a node of the grid, and grid edges
        are identified to link up successive nodes.
        for each grid edge, find the exchanges which are part of that 2d link.
        return a list of these exchanges, but ONE BASED, where exchanges
        with their 'from' segment left of the path are positive, otherwise
        negated.

        on_boundary:
         'warn_and_skip': any of the edges which are closed edges in the original
            grid (unless a flow boundary), are mentioned, but omitted.
        on_edge: xy are already on the edges of the polygons [from Zhenlin Zhang].
        """
        # align the input nodes along nodes of the grid
        g=self.grid()
        input_nodes=[g.select_nodes_nearest(p)
                     for p in xy]
        if on_edge:
            # RH 2020-01-31: In theory this shouldn't be needed --
            # not sure if ZZ was working around a bug, or if the original
            # code was slow. 
            legs=input_nodes
        else:
            legs=[ input_nodes[0] ] 
            for a,b in zip(input_nodes[:-1],input_nodes[1:]):
                if a==b:
                    continue
                path=g.shortest_path(a, b)
                legs+=list(path[1:])

        self.infer_2d_links()

        # RH: I think the crux of the changes below from ZZ
        # was dealing with multidomain grids
        # from MPI DFM that had not been cleaned.
        # those grids have duplicate nodes and edges, so ZZ added code to
        # choose node pairs geographically and test all pairs.
        # Given that this may still be a possibility, at least include the
        # check here and give a marginally useful message.
        for ncheck in legs:
            count=0
            for nbr in g.select_nodes_nearest(g.nodes['x'][ncheck],count=10):
                if utils.dist( g.nodes['x'][ncheck] - g.nodes['x'][nbr] ) < 1e-3:
                    count+=1
            if count>1:
                raise Exception("Encountered duplicate nodes. May need to clean MPI output grid, or revert to ZZ edge search")
            elif count==0:
                if on_edge:
                    raise Exception("Node search failed and on_edge is set, but supplied points do not line up.")
                else:
                    raise Exception("Node search failed but on_edge is not set.  Something very wrong")

        link_and_signs=[] # (link idx, sign to make from->to same as left->right
        for a,b in zip(legs[:-1],legs[1:]):
            j=g.nodes_to_edge(a,b)
            
            if j is None:
                # this happens when the line cuts across an island, and edges were
                # specified directly. ignore the exchange.
                if on_edge:
                    continue
                else:
                    # if legs came from shortest_path() above, it really shouldn't
                    # miss any edges, so signal bad news
                    raise Exception("edge couldn't be found, but it came from the grid.")
            
            # possible to have missing cells with other marks (as in
            # marking an ocean or flow boundary), but boundary links are
            # just -1:
            c1_c2=g.edge_to_cells(j).clip(-1,g.Ncells())

            # ZZ changes were right here.
            
            leg_to_edge_sign=1
            if g.edges['nodes'][j,0] == b:
                leg_to_edge_sign=-1

            # assumes that hydro elements and grid cells have the same numbering
            # make sure that any missing cell is just labeled -1
            fwd_hit= np.nonzero( np.all( self.links[:,:]==c1_c2, axis=1 ) )[0]
            rev_hit= np.nonzero( np.all( self.links[:,:]==c1_c2[::-1], axis=1 )) [0]
            nhits=len(fwd_hit)+len(rev_hit)
            if nhits==0:
                if np.any(c1_c2<0):
                    self.log.warning("Discarding boundary edge in path_to_transect_exchanges")
                    continue
                else:
                    raise Exception("Failed to match edge to link")
            elif nhits>1:
                raise Exception("Somehow got two matches.  Bad stuff.")

            if len(fwd_hit):
                link_and_sign = [fwd_hit[0],leg_to_edge_sign] 
            else:
                link_and_sign =[rev_hit[0],-leg_to_edge_sign]
            if link_and_signs and link_and_signs[-1][0]==link_and_sign[0]:
                self.log.warning("Discarding repeated link")
            else:
                link_and_signs.append(link_and_sign)

        link_to_exchs=defaultdict(list)
        for exch,(link,sgn) in enumerate(self.exch_to_2d_link):
            link_to_exchs[link].append( (exch,sgn) )

        transect_exchs=[]

        for link,sign in link_and_signs:
            for exch,exch_sign in link_to_exchs[link]:
                # here is where we switch to 1-based.
                transect_exchs.append( sign*exch_sign*(1+exch) )

        return transect_exchs
            
    link_group_dtype=[('id','i4'),
                      ('name','O'),
                      ('attrs','O')]
    def group_boundary_links(self):
        """ 
        a [hopeful] improvement over group_boundary_elements, since boundaries
        are properties of links.
        follows the same representation as group_boundary_elements, but enumerating
        2D links instead of 2D elements.
        
        maps all link ids (0-based) to either -1 (not a boundary)
        or a nonnegative id corresponding to contiguous boundary links which can
        be treated as a whole.

        This generic implementation isn't very smart, though.  We have so little
        geometry on boundary exchanges - so there's not a way to group boundary links
        which wouldn't risk grouping things which are really distinct.

        We could maybe conservatively do some grouping based on marked edges in the 
        original grid, but we don't necessarily have those marks in this code (but
        see Suntans subclass where more info is available).

        Return an array mapping of 2D links (i.e. elements of self.links)
        to a struct array with elements  ('id',<index into boundary groups>), 
        ('name', <string identifier>), ('attrs',<hash of additional info>).
        """
        self.infer_2d_links()

        bc_lgroups=np.zeros(self.n_2d_links,self.link_group_dtype)
        bc_lgroups['id']=-1 # most links are internal and not part of a boundary group
        for lg in bc_lgroups:
            lg['attrs']={} # we have no add'l information for the groups.
        sel_bc=np.nonzero( (self.links[:,0]<0) )[0]
        bc_lgroups['id'][sel_bc]=np.arange(len(sel_bc))
        bc_lgroups['name'][sel_bc]=['group %d'%i for i in bc_lgroups['id'][sel_bc]]
        return bc_lgroups

    def write_boundary_links(self):
        """ calls group_boundary_links, and writes the result out to a csv file,
        first few columns always the same: index (0-based, of the boundary link), 
        link0 (0-based index of the link, i.e. including all links), and a string-valued name.
        The rest of the fields are whatever group_boundary_links returned.  Some may
        have embedded commas and will be double-quote-escaped.  Results are written
        to boundary-links.csv in the base_path directory.
        """
        rows=[]
        gbl=self.group_boundary_links()
        
        for link_idx in range(len(gbl)):
            rec=gbl[link_idx]
            if rec['id']<0:
                continue
            row=OrderedDict()
            row['index']=rec['id']
            row['link0']=link_idx
            row['name']=rec['name']
            for k,v in iteritems(rec['attrs']):
                if k=='geom':
                    try:
                        v=v.wkt
                    except AttributeError:
                        # might not actually be a geometry object
                        pass
                if k not in row:
                    # careful not to allow incoming attributes to overwrite
                    # index or link0 from above
                    row[k]=v
            rows.append(row)

        df=pd.DataFrame(rows)
        # reorder those a bit..
        cols0=df.columns.tolist()
        cols=['index','link0','name']
        cols+=[c for c in cols0 if c not in cols]
        df=df[cols].set_index('index')
        if 0:
            # this is too strict. assumes both that these are sorted
            # and that every group has exactly one link.
            assert np.all( np.diff(df.index.values)==1 )
        if 1:
            # more lenient - just make sure that the id's present have 
            # no gaps
            unique_ids=np.unique(df.index.values) # sorted, too
            assert np.all(unique_ids == np.arange(len(unique_ids)))
        df.to_csv( os.path.join(self.scenario.base_path,"boundary-links.csv") )
    
    group_dtype=[('id','i4'), # 0-based id of this elements group, -1 for unset.
                 ('name','O'), # had been S40, but that get confusing with bytes vs. str
                 ('attrs','O')] # a place for add'l key-value pairs
    def group_boundary_elements(self):
        """ map all element ids (0-based) to either -1 (not a boundary)
        or a nonnegative id corresponding to contiguous boundary elements.
        
        Only works if a grid is available.
        """
        self.infer_2d_elements()

        g=self.grid()
        if g is None:
            # This code is wrong!
            # bc_groups should only be set for elements with a boundary link.
            assert False
            self.log.warning("No grid for grouping boundary elements")
            bc_groups=np.zeros(self.n_2d_elements,self.group_dtype)
            bc_groups['id']=np.arange(self.n_2d_elements)
            bc_groups['name']=['group %d'%i for i in self._bc_groups['id']]
            return bc_groups

        poi=self.pointers
        bc_sel = (poi[:,0]<0)
        bc_elts = np.unique(self.seg_to_2d_element[ poi[bc_sel,1]-1 ])

        def adjacent_cells(g,c,candidates):
            """ g: unstructured grid
            c: element/cell index
            candidates: subset of cells in the grid

            returns a list of cell ids which are adjacent to c and in candidates,
            based on two adjacency checks:
              shares an edge
              has boundary edges which share a node.
            """
            a=list(g.cell_to_adjacent_boundary_cells(c))
            b=list(g.cell_to_cells(c))
            nbrs=filter(lambda cc: cc in candidates,a+b)
            return np.unique(nbrs)

        groups=np.zeros(self.n_2d_elements,self.group_dtype)
        groups['id'] -= 1

        def trav(c,mark):
            groups['id'][c]=mark
            groups['name'][c]="group %d"%mark 
            for nbr in adjacent_cells(g,c,bc_elts):
                if groups['id'][nbr]<0:
                    trav(nbr,mark)

        ngroups=0
        for bc_elt in bc_elts:
            if groups['id'][bc_elt]<0:
                trav(bc_elt,ngroups)
                ngroups+=1
        return groups

    def extract_transect_flow(self,transect,func=False,time_range=None):
        """
        Extract time series of discharge through a transect as a function of time.
        transect is expected to have the same structure as it is configured on
        a Scenario or WaqModel instance: (name, [+-exchs...])
        Exchanges are numbered from 1, with sign indicating how they are added to the total.

        if time_range is None, then extract over the full hydro time range.
        if given, evaluate for all hydro steps within the range and return a xr.DataArray.
        time_range: [np.datetime64,np.datetime64]

        if func is True, instead return a function that takes a datetime64 and returns flow.
        """
        exchs=transect[1]

        def fn(t,exchs=exchs,hydro=self):
            t_secs = (t - np.datetime64(hydro.time0))/np.timedelta64(1,'s')
            flo = hydro.flows(t_secs)
            signs=np.sign(exchs)
            exch0=np.abs(exchs)-1
            return np.sum(flo[exch0]*signs)
        if func:
            return fn

        if time_range is None:
            tidx_start=0
            tidx_stop = len(self.t_secs)
        else:
            tidx_start,tidx_stop = [ np.searchsorted( self.t_secs,
                                                      (t-np.datetime64(self.time0))/np.timedelta64(1,'s'))
                                     for t in time_range ]

        t_secs=self.t_secs[tidx_start:tidx_stop]
        times=np.datetime64(self.time0)+t_secs*np.timedelta64(1,'s')
        
        Q=np.zeros( len(t_secs), np.float64)
        for i,t in enumerate(times):
            Q[i] = fn(t)

        da= xr.DataArray(data=Q, dims=['time'], 
                         coords=dict( time=times,
                                      time_seconds=("time",t_secs) ),
                         attrs=dict(transect=transect[0], units='m3 s-1'))
        da.name="discharge"
        return da
    
    # Data formats on disk
    def flo_dtype(self):
        return np.dtype([ ('tstamp','<i4'),
                          ('flow','<f4',self.n_exch) ])
    def are_dtype(self):
        return np.dtype([ ('tstamp','<i4'),
                          ('area','<f4',self.n_exch) ])
    def vol_dtype(self):
        return np.dtype([ ('tstamp','<i4'),
                          ('volume','<f4',self.n_seg) ])

    
def parse_datetime(s):
    """ 
    parse YYYYMMDDHHMMSS style dates.
    strips single quotes in case it came from a hyd file
    """
    return datetime.datetime.strptime(s.strip("'"),'%Y%m%d%H%M%S')

class HydroFiles(Hydro):
    """
    DWAQ hydro data read from existing files, by parsing
    .hyd file.
    """
    # When loading a DFM grid (ala waqgeom), MPI output includes
    # ghost nodes and edges (but cells are fine).  This flag is
    # passed to UnstructuredGrid.read_dfm(), and when true it
    # will clean out those nodes and edges.  This is a bit slower,
    # and potentially renumbers nodes and edges, so it's not always
    # the right thing to do.
    clean_mpi_dfm_grid=True

    # If True, override vol_filename and friends to point to the original
    # file.
    reference_originals=False
    
    def __init__(self,hyd_path,**kw):
        self.hyd_path=hyd_path
        self.parse_hyd()

        super(HydroFiles,self).__init__(**kw)
        if sys.platform=='win32' and self.enable_write_symlink:
            self.log.warning("Symlinks disabled on windows")
            self.enable_write_symlink=False
            
    def parse_hyd(self):
        self.hyd_toks={}

        with open(self.hyd_path,'rt') as fp:
            while 1:
                line=fp.readline().strip()
                if line=='':
                    break

                try:
                    tok,rest=line.split(None,1)
                except ValueError:
                    tok=line ; rest=None

                if tok in ['hydrodynamic-layers','water-quality-layers']:
                    layers=[]
                    while 1:
                        line=fp.readline().strip()
                        if line=='' or line=='end-'+tok:
                            break
                        layers.append(float(line))
                    self.hyd_toks[tok]=layers
                elif tok == 'sink-sources':
                    sink_sources=[]
                    while 1:
                        line=fp.readline().strip()
                        if line=='' or line=='end-'+tok:
                            break
                        index1,link_id,elt_id,from_x,from_y,to_x,to_y,name = line.split()
                        sink_sources.append(dict(index1=int(index1),
                                                 link_id=int(link_id),
                                                 elt_id=int(elt_id),
                                                 from_x=float(from_x), from_y=float(from_y),
                                                 to_x=float(to_x), to_y=float(to_y),
                                                 name=name))
                    self.hyd_toks[tok]=sink_sources
                else:
                    self.hyd_toks[tok]=rest

    _t_secs=None
    @property
    def t_secs(self):
        if self._t_secs is None:
            conv_start=parse_datetime(self.hyd_toks['conversion-start-time'])
            conv_stop =parse_datetime(self.hyd_toks['conversion-stop-time'])
            conv_step = waq_timestep_to_timedelta(self.hyd_toks['conversion-timestep'].strip("'"))

            # important to keep all of these integers
            step_secs=conv_step.total_seconds() # seconds in a step
            n_steps=1+(conv_stop - conv_start).total_seconds() / step_secs
            n_steps=int(round(n_steps))
            start=(conv_start-self.time0).total_seconds()
            start=int(round(start))

            if abs(step_secs - round(step_secs)) > 1e-5:
                print("WARNING: total seconds in step was not an integer: %s"%step_secs)
            step_secs=int(round(step_secs))

            self._t_secs=(start+np.arange(n_steps)*step_secs).astype('i4')
        return self._t_secs

    @property
    def time0(self):
        return parse_datetime(self.hyd_toks['conversion-ref-time'])


    def __getitem__(self,k):
        val=self.hyd_toks[k]
        if k in ['grid-cells-first-direction',
                 'grid-cells-second-direction',
                 'number-hydrodynamic-layers',
                 'number-horizontal-exchanges',
                 'number-vertical-exchanges',
                 'number-water-quality-segments-per-layer',
                 'number-water-quality-layers']:
            return int(val)
        elif k in ['water-quality-layers',
                   'hydrodynamic-layers']:
            return val
        else:
            return val.strip("'")

    @property
    def vol_filename(self):
        if self.reference_originals:
            return self.get_path('volumes-file')
        else:
            # os.path.join(self.scenario.base_path, self.fn_base+".vol")
            return super().vol_filename

        
    def get_dir(self):
        return os.path.dirname(self.hyd_path)
    
    def get_path(self,k,check=False):
        """ Return full pathname for a file referenced by its 
        key in .hyd.
        May throw KeyError.  

        check: if True, check that file exists, and throw KeyError otherwise
        """
        p=os.path.join( self.get_dir(),self[k] )
        if check and not os.path.exists(p):
            raise KeyError(p)
        return p

    # be tolerant of mismatch in file sizes up to this many steps
    nstep_mismatch_threshold=3
    _n_seg = None
    @property
    def n_seg(self):
        if self._n_seg is None:
            # assumes that every element has the same number of layers
            # in some processes dwaq assumes this, too!
            n_seg_dense=self['number-water-quality-layers'] * self['number-water-quality-segments-per-layer']

            # try to support partial water columns:

            nx=self['grid-cells-first-direction']
            ny=max(1,self['grid-cells-second-direction'])
            n_elts=nx*ny
            # bit of sanity check:
            if os.path.exists(self.get_path('grid-coordinates-file')):
                # g=dfm_grid.DFMGrid(self.get_path('grid-coordinates-file'))# deprecated
                g=unstructured_grid.UnstructuredGrid.read_dfm(self.get_path('grid-coordinates-file'))
                n_elts_nc=g.Ncells() 
                assert n_elts_nc==n_elts

            # assumes dense exchanges
            n_seg=self['number-vertical-exchanges'] + n_elts

            if 1: # allow for sparse exchanges and dense segments
                # more work, and requires that we have area and volume files
                nsteps=len(self.t_secs)

                # sanity check on areas:
                are_size=os.stat(self.get_path('areas-file')).st_size
                #pred_n_exch = (are_size/float(nsteps) - 4) / 4.
                #pred_n_exch2= (are_size/float(nsteps-1) - 4) / 4.
                #assert (pred_n_exch==self.n_exch) or (pred_n_exch2==self.n_exch)
                
                # kludge - suntans writer (as of 2016-07-13)
                # creates one fewer time-steps of exchange-related data
                # than volume-related data.
                # each step has 4 bytes per exchange, plus a 4 byte time stamp.
                # 2017-07-20 RH: bigger problems, as the output number o
                pred_n_steps = are_size/4./(self.n_exch+1)

                if pred_n_steps==nsteps:
                    pass # great
                elif abs(pred_n_steps-nsteps)<=self.nstep_mismatch_threshold:
                    self.log.info("Area file has %s steps vs. %s expected - proceed with caution"%(pred_n_steps,
                                                                                                   nsteps))
                else:
                    # Sometimes this is intentional, so don't bail out, just hope the warning is visible.
                    #raise Exception("nsteps %s too different from size of area file (~ %s steps)"%(nsteps,
                    #                                                                               pred_n_steps))
                    self.log.warning("DWAQ run may be incomplete")

                vol_size=os.stat(self.get_path('volumes-file')).st_size
                # kludgY.  Ideally have the same number of volume and area output timesteps, but commonly
                # one off.
                for step_error in range(1+self.nstep_mismatch_threshold):
                    for vol_n_steps in pred_n_steps - step_error,pred_n_steps+step_error:
                        n_seg = (vol_size/float(vol_n_steps) -4)/ 4.
                        if n_seg%1.0 != 0.0:
                            n_seg=-1 # continue
                        else:
                            n_seg=int(n_seg)
                            break
                    if n_seg>0:
                        break
                else:
                    raise Exception("Volume file steps not in [%d,%d]"%(pred_n_steps-self.nstep_mismatch_threshold,
                                                                        pred_n_steps+self.nstep_mismatch_threshold))

                if n_seg==n_seg_dense:
                    self.log.debug("Discovered that hydro is dense")

            # make sure we're consistent with pointers --
            assert n_seg>=self.pointers.max()
            self._n_seg=n_seg
        return self._n_seg

    @property
    def n_exch_x(self):
        return self['number-horizontal-exchanges']
    @property
    def n_exch_y(self):
        return 0
    @property
    def n_exch_z(self):
        return self['number-vertical-exchanges']

    @property
    def exchange_lengths(self):
        with open(self.get_path('lengths-file'),'rb') as fp:
            n_exch=np.fromfile(fp,'i4',1)[0]
            if n_exch==0:
                # at least in the output of ddcouplefm, it seems not to bother
                # setting the number of exchanges, just writing 0 instead.
                self.log.warning("Exchange length file lazily reports 0 exchanges")
                n_exch=self.n_exch
            else:
                assert n_exch == self.n_exch
            return np.fromfile(fp,'f4',2*self.n_exch).reshape( (self.n_exch,2) )

    def write_are(self):
        if self.reference_originals:
            pass
        elif self.enable_write_symlink:
            rel_symlink(self.get_path('areas-file'),
                        self.are_filename,overwrite=self.overwrite)
        else:
            return super(HydroFiles,self).write_are()

    @property
    def are_filename(self):
        if self.reference_originals:
            return self.get_path('areas-file')
        else:
            return super().are_filename
        

    _areas_mmap=None
    def areas(self,t,memmap=False):
        ti_req=ti=self.t_sec_to_index(t)

        stride=4+self.n_exch*4
        area_fn=self.get_path('areas-file')

        if memmap:
            if self._areas_mmap is None:
                self._areas_mmap=np.memmap(area_fn,self.are_dtype())
                if (ti>=self._areas_mmap.shape[0]) or ( self._areas_mmap['tstamp'][ti] !=t ):
                    self.log.warning("area memmap failed -- too short or mismatched timestamp")
                    self._areas_mmap=False
            if self._areas_mmap is not False:
                return self._areas_mmap['area'][ti]
        
        with open(area_fn,'rb') as fp:
            while 1: # in case we have to scan backwards to find real data
                fp.seek(stride*ti)

                tstamp_data=fp.read(4)
                n_raw_bytes=self.n_exch*4
                raw=fp.read(n_raw_bytes)
                
                if len(tstamp_data)<4 or len(raw)!=n_raw_bytes:
                    # Incomplete data.  Scan back one step
                    ti-=1
                    if ti>0:
                        continue
                    else:
                        raise Exception("No complete frames in areas data")
                break # Found a good frame
            
        if ti<ti_req:
            self.log.warning("Area data ends early by %d steps. Use previous"%(ti_req-ti))

        tstamp=np.frombuffer(tstamp_data,np.int32)[0]
        if (ti==ti_req) and (tstamp!=t):
            self.log.warning("WARNING: time stamp mismatch: %d [file] != %d [requested]"%(tstamp,t))
        return np.frombuffer(raw,np.float32)

    def write_vol(self):
        if self.reference_originals:
            pass
        elif self.enable_write_symlink:
            rel_symlink(self.get_path('volumes-file'),
                        self.vol_filename,
                        overwrite=self.overwrite)
        else:
            return super(HydroFiles,self).write_vol()


    def volumes(self,t,**kw):
        return self.seg_func(t,label='volumes-file',**kw)

    _seg_mmap=None # dict of fn => memmap'd data
    
    def seg_func(self,t_sec=None,fn=None,label=None,memmap=False):
        """ 
        Get segment function data at a given timestamp (must match a timestamp
        - no interpolation).
        t: time in seconds, or a datetime instance
        fn: full path to data file
        label: key in the hydr file (e.g. "volumes-file")
        
        if t_sec is not specified, returns a callable which takes t_sec
        """
        def f(t_sec,closest=False,memmap=memmap):
            if isinstance(t_sec,datetime.datetime):
                t_sec = int( (t_sec - self.time0).total_seconds() )
            
            filename=fn or self.get_path(label)
            # Optimistically assume that the seg function has the same time steps
            # as the hydro:
            ti=self.t_sec_to_index(t_sec) 

            stride=4+self.n_seg*4

            if memmap:
                if self._seg_mmap is None: self._seg_mmap={} # jit init
                if self._seg_mmap.get(filename,None) is False:
                    memmap=False # tried and failed, revert to file access
                else:
                    if filename not in self._seg_mmap:
                        # TODO: catch errors, e.g. network filesystem
                        self._seg_mmap[filename] = np.memmap(filename, self.vol_dtype(),mode='r')
                    data=self._seg_mmap[filename]
                    # confirm time stamp
                    if (ti>=data.shape[0]) or (data['tstamp'][ti]!=t_sec):
                        # could add scanning to the memmap code path, or factor it out.
                        # another day.  for now fall through to file-based.
                        self.log.warning("Tried to memmap but time stamps don't align")
                        self._seg_mmap[filename].close()
                        self._seg_mmap[filename]=False
                        memmap=False
                    else:
                        return data['volume'][ti]
            
            with open(filename,'rb') as fp:
                fp.seek(stride*ti)
                tstamp=np.fromfile(fp,'i4',1)
                
                if len(tstamp)==0 or tstamp[0]!=t_sec:
                    if 0:# old behavior, no scanning:
                        if len(tstamp)==0:
                            print("WARNING: no timestamp read for seg function")
                        else: 
                            print("WARNING: time stamp mismatch: %s != %d should scan but won't"%(tstamp[0],t_sec))
                    else: # new behavior to accomodate hydro parameters with variable time steps
                        # assumes at least two time steps, and that all steps are the same size
                        fp.seek(0)
                        tstamp0=np.fromfile(fp,'i4',1)[0]
                        fp.seek(stride*1)
                        tstamp1=np.fromfile(fp,'i4',1)[0]
                        dt=tstamp1 - tstamp0
                        ti,err=divmod( t_sec-tstamp0, dt )
                        if err!=0:
                            print("WARNING: time stamp mismatch after inferring nonstandard time step")
                        # also check for bounds:
                        warning=None
                        if ti<0:
                            if t_sec>=0:
                                warning="WARNING: inferred time index %d is negative in %s!"%(ti,filename)
                            else:
                                # kludgey - the problem is that something like the temperature field
                                # can have a different time line, and to be sure that it has data
                                # t=0, an extra step at t<0 is included.  But then there isn't any
                                # volume data to be used, and that comes through here, too.
                                # so downgrade it to a less dire message
                                warning="INFO: inferred time index %d is negative, ignoring as t=%d"%(ti,t_sec)
                            ti=0
                        max_ti=os.stat(filename).st_size // stride
                        if ti>=max_ti:
                            warning="WARNING: inferred time index %d is beyond the end of the file!"%ti
                            ti=max_ti-1
                        # try that again:
                        fp.seek(stride*ti)
                        tstamp=np.fromfile(fp,'i4',1)
                        if warning is None and tstamp[0]!=t_sec:
                            warning="WARNING: Segment function appears to have unequal steps"
                        if warning:
                            print(warning)

                return np.fromfile(fp,'f4',self.n_seg)
        if t_sec is None:
            return f
        else:
            return f(t_sec)

    def vert_diffs(self,t_sec):
        return self.seg_func(t_sec,label='vert-diffusion-file')

    def write_flo(self):
        if self.reference_originals:
            pass
        elif  self.enable_write_symlink:
            rel_symlink(self.get_path('flows-file'),
                        self.flo_filename,
                        overwrite=self.overwrite)
        else:
            return super(HydroFiles,self).write_flo()
        
    @property
    def flo_filename(self):
        if self.reference_originals:
            return self.get_path('flows-file')
        else:
            return super().flo_filename
        

    _flows_mmap=None
    def flows(self,t,memmap=False):
        """ returns flow rates ~ np.zeros(self.n_exch,'f4'), for given timestep.
        flows in m3/s.  Sometimes there is no flow data for the last timestep,
        since flow is integrated over [t,t+dt].  Checks file size and may return
        zero flow

        memmap: if True, attempt to memory map the file for faster access.  If
        successful, the memory mapped data is referenced by self._flows_mmap.
        When _flows_mmap is False, as opposed to None, then the attempt failed.
        """
        ti=self.t_sec_to_index(t)
        
        stride=4+self.n_exch*4
        flo_fn=self.get_path('flows-file')

        if memmap and self._flows_mmap is not False:
            if self._flows_mmap is None:
                self._flows_mmap=np.memmap(flo_fn, self.flo_dtype(),
                                           mode='r')
            if ti>=len(self._flows_mmap):
                self.log.warning("Flow data ends early by %d steps"%(len(self.t_secs)-1-ti))
                return np.zeros(self.n_exch,'f4')
            
            tstamp=self._flows_mmap['tstamp'][ti]
            if tstamp!=t:
                self.log.warning("flows: time stamp mismatch: %d != %d"%(tstamp,t))
            return self._flows_mmap['flow'][ti]
        
        with open(flo_fn,'rb') as fp:
            fp.seek(stride*ti)
            tstamp_data=fp.read(4)
            if len(tstamp_data)<4:
                if ti==len(self.t_secs)-1:
                    self.log.info("Short read on last frame of flow data - fabricate zero flows")
                else:
                    self.log.warning("Flow data ends early by %d steps"%(len(self.t_secs)-1-ti))
                return np.zeros(self.n_exch,'f4')
            else:
                tstamp=np.frombuffer(tstamp_data,'i4')[0]
                if tstamp!=t:
                    self.log.warning("flows: time stamp mismatch: %d != %d"%(tstamp,t))
                data=np.fromfile(fp, np.float32, self.n_exch)
                if len(data)!=self.n_exch:
                    self.log.warning("flow: incomplete frame, %d items < %d exchanges"%(len(data),self.n_exch))
                    return np.zeros(self.n_exch,'f4')
                return data # all's well.

    def update_flows(self,t,new_flows):
        """ the 'reverse' of flows(), this will overwrite flow data in the existing
        flo file.

        This does not currently allow for extending the length of the file-- only
        existing time steps can be overwritten.

        return True on success, False otherwise
        """
        ti=self.t_sec_to_index(t)
        
        stride=4+self.n_exch*4
        flo_fn=self.get_path('flows-file')
        with open(flo_fn,'rb+') as fp:
            fp.seek(stride*ti)
            tstamp_data=fp.read(4)
            if len(tstamp_data)<4:
                self.log.info("update_flows: File is too short")
                return False
            else:
                tstamp=np.frombuffer(tstamp_data,'i4')[0]
                if tstamp!=t:
                    self.log.warning("update_flows: time stamp mismatch: %d != %d"%(tstamp,t))
                fp.write(new_flows.astype('f4'))
                return True

    def adjust_boundaries_for_conservation(self,tidx_select=slice(None)):
        """
        Adjust for missing boundary exchange fluxes in hydro.
        Updates flow data in place for boundary exchanges, by checking
        for mass conservation in segments adjacent to a boundary.
        Specify tidx_select as a slice to only update a subset of time steps.
        Note that due to how conservation is defined in DWAQ, tidx_select
        specifies the "now" timestep, which is fixed by changing "now"-1
        flows.  This effectively means that the first step in the slice is 
        ignored.
        So tidx_select=slice(10,20) would select times t_secs[10]..t_secs[19]
        for which conservation would be calculated at t_secs[11]..t_secs[19],
        which would be used to update flows(t_secs[10])..flows(t_secs[18]).
        Clear?
        """
        # Find all segments which have a boundary exchange
        bc_exchs=np.nonzero( self.pointers[:,0]<0 )[0]
        bc_adj_segs=self.pointers[bc_exchs,1]-1 # make 0-based

        def cb(ti,summary):
            print("ti=%5d: RMS Q_err: %.5f"%(ti,utils.rms(summary['Q_err'])))
            flo=self.flows(self.t_secs[ti-1])
            flo[bc_exchs] += summary['Q_err']
            self.update_flows(self.t_secs[ti-1],flo)

        self.check_volume_conservation_incr(seg_select=bc_adj_segs,
                                            tidx_select=tidx_select,
                                            verbose=False,
                                            err_callback=cb)

    @property
    def pointers(self):
        poi_fn=self.get_path('pointers-file')
        with open(poi_fn,'rb') as fp:
            #return np.fromstring( fp.read(), 'i4').reshape( (self.n_exch,4) )
            return np.fromfile(fp, np.int32).reshape( (self.n_exch,4) )

    def bottom_depths_2d(self):
        """ 
        Return per-element bottom depths.  You may prefer bottom_depths(),
        which wraps this up into a ParameterSpatial.
        """
        try:
            fn=self.get_path('depths-file')
        except KeyError:
            return None

        with open(fn,'rb') as fp:
            sizes=np.fromfile(fp,'i4',count=6)
            _,count1,count2,count3,count4,_ = sizes
            depths=np.fromfile(fp,'f4')
            assert len(depths)==count1
        return depths

    def bottom_depths(self):
        """
        Returns a parameter object (most likely ParameterSpatial) for 
        the bottom depths.  Raises NotImplementedError if that information is 
        not available in the source data.
        """
        elt_depths=self.bottom_depths_2d()
        if elt_depths is None:
            raise NotImplementedError("Bottom depths not available")
        self.infer_2d_elements()
        assert self.n_2d_elements==len(elt_depths)
        return ParameterSpatial(elt_depths[self.seg_to_2d_element],hydro=self)
        
    def planform_areas(self):
        # any chance we have this info written out to file?
        # seems like there are two competing ideas of what is in surfaces-file
        # DwaqAggregator might have written this out as if it were a segment
        # function
        # but here it's expected to be constant in time, and have some header info
        # okay - see delwaq.c or waqfil.m or details on the format.
        if 'surfaces-file' in self.hyd_toks:
            # actually would be pretty easy, but not implemented yet.
            srf_fn=self.get_path('surfaces-file')
            with open(srf_fn,'rb') as fp:
                hdr=np.fromfile(fp,np.int32,6)
                # following waqfil.m
                elt_areas=np.fromfile(fp,np.float32,hdr[2])
            self.infer_2d_elements()
            assert self.n_2d_elements==len(elt_areas)
            return ParameterSpatial(elt_areas[self.seg_to_2d_element],hydro=self)
        else:
            # cobble together areas from the exchange areas
            seg_z_exch=self.seg_to_exch_z(preference='upper')

            # then pull exchange area for each time step
            A=np.zeros( (len(self.t_secs),self.n_seg) )
            # some segments have no vertical exchanges - they'll just get
            # A=1 (below) for lack of a better guess.
            sel=seg_z_exch>=0
            for ti,t_sec in enumerate(self.t_secs):
                areas=self.areas(t_sec)
                A[ti,sel] = areas[seg_z_exch[sel]]

            # without this, but with zero area exchanges, and monotonicize
            # enabled, it was crashing, complaining that DDEPTH ran into
            # zero SURF.
            # enabling this lets it run, though depths are pretty wacky.
            A[ A<1.0 ] = 1.0
            return ParameterSpatioTemporal(times=self.t_secs,values=A,hydro=self)

    # segment attributes - namely surface/middle/bed
    _read_seg_attrs=None # seg attributes read from file. shaped [nseg,2]
    def seg_attrs(self,number):
        """ corresponds to the 'number 2' set of properties, unless number is specified.
        number is 1-based!
        """
        if self._read_seg_attrs is None and 'attributes-file' in self.hyd_toks:
            self.log.debug("Reading segment attributes from file")
    
            seg_attrs=np.zeros( (self.n_seg,2), 'i4')
            seg_attrs[:,0] = 1 # default for active segments
            seg_attrs[:,1] = 0 # default, depth-averaged segments
            
            with open(self.get_path('attributes-file'),'rt') as fp:
                # this should all be integers
                toker=lambda t=tokenize(fp): int(next(t))
                
                n_const_blocks=toker()
                for const_contrib in range(n_const_blocks):
                    nitems=toker()
                    feat_numbers=[toker() for item in range(nitems)]
                    assert nitems==1 # not implemented for multiple..
                    assert toker()==1 # input is in this file, nothing else implemented.
                    assert toker()==1 # all segments written, no defaults.
                    for seg in range(self.n_seg):
                        seg_attrs[seg,feat_numbers[0]-1] = toker()
                n_variable_blocks=toker()
                assert n_variable_blocks==0 # not implemented
            self._read_seg_attrs=seg_attrs
        if self._read_seg_attrs is not None:
            assert number>0
            return self._read_seg_attrs[:,number-1]
        else:
            return super(HydroFiles,self).seg_attrs(number)

    def write_geom(self):
        if self.reference_originals:
            return
        
        # just copy existing grid geometry
        try:
            orig=self.get_path('grid-coordinates-file',check=True)
        except KeyError:
            return

        dest=self.flowgeom_filename
        if os.path.exists(dest) or os.path.lexists(dest):
            assert not os.path.samefile(dest,orig)
            if self.overwrite:
                self.log.warning("Removing old geom file %s"%dest)
                os.unlink(dest)
            else:
                raise Exception("Overwrite is not true, and geometry file %s already exists"%dest)
                
        if self.enable_write_symlink:
            rel_symlink(orig,dest)
        else:
            shutil.copyfile(orig,dest)

    @property
    def flowgeom_filename(self):
        if self.reference_originals:
            return self.get_path('grid-coordinates-file')
        else:
            # WIP changing this from a basename to a path.
            return super().flowgeom_filename

            
    def get_geom(self):
        try:
            return xr.open_dataset( self.get_path('grid-coordinates-file',check=True) )
        except KeyError:
            return

    _grid=None
    def grid(self,force=False):
        if force or self._grid is None:
            try:
                orig=self.get_path('grid-coordinates-file',check=True)
            except KeyError:
                return None
            try:
                self._grid=unstructured_grid.UnstructuredGrid.from_ugrid(orig)
            except (IndexError,AssertionError,unstructured_grid.GridException):
                self.log.warning("Grid wouldn't load as ugrid, trying dfm grid")
                dg=unstructured_grid.UnstructuredGrid.read_dfm(orig,cleanup=self.clean_mpi_dfm_grid)
                self._grid=dg
        return self._grid

    def add_parameters(self,hyd):
        super(HydroFiles,self).add_parameters(hyd)

        # can probably add bottomdept,depth,salinity.
        self.log.debug("Incoming parameters are %s"%list(hyd.keys()))
        
        # do NOT include surfaces-files here - it's a different format.
        for var,key in [('vertdisper','vert-diffusion-file'),
                        ('tau','shear-stresses-file'),  # play 'em if you got 'em.
                        ('temp','temperature-file'),
                        ('salinity','salinity-file')]:
            fn=self.get_path(key)
            if os.path.exists(fn):
                self.log.debug("%s does exist, so will add it to the parameters"%fn)
                hyd[var]=ParameterSpatioTemporal(seg_func_file=fn,
                                                 hydro=self)
        return hyd

    def read_2d_links(self):
        """
        Read the files written by self.write_2d_links, set attributes, and return true.
        If the files don't exist, return False, log an info message.
        """
        path = self.get_dir()
        links_fn=os.path.join(path,'links.csv')
        if os.path.exists(links_fn):
            links=np.loadtxt(links_fn,dtype='i4')
            exch_to_2d_link = np.loadtxt(os.path.join(path,'exch_to_2d_link.csv'),
                                         dtype=[('link','i4'),('sgn','i4')])
        else:
            return False
        self.links=links
        self.exch_to_2d_link=exch_to_2d_link
        self.n_2d_links=len(links)
        return True

    def infer_2d_links(self,force=False):
        if self.exch_to_2d_link is not None and not force:
            return

        if not self.read_2d_links():
            # self.log.warning("Couldn't read 2D link info in HydroFiles - will compute")
            super(HydroFiles,self).infer_2d_links()

    @property
    def bnd_filename(self):
        return os.path.join(self.scenario.base_path, self.fn_base+".bnd")
        
    def write_boundary_links(self):
        """
        On the way to following the .bnd format, at least in cases where we already
        have a .bnd file, sneak in and copy that over
        """
        try:
            src_fn=self.get_path('boundaries-file')
        except KeyError:
            src_fn='no file found'
        dest_fn=self.bnd_filename
        
        if os.path.exists(src_fn):
            self.log.info("Using .bnd file, not writing out kludgey boundary-links.csv")
            if src_fn!=dest_fn: # a bit of paranoia with name check
                shutil.copyfile(src_fn,dest_fn)
        else:
            super(HydroFiles,self).write_boundary_links()
            
    def group_boundary_links(self):
        """
        Use either a .bnd file (written by DFM) or a boundaries.csv (kludge
        format from suntans runs) to return an array annotating each 2D link with
        name, index, and other boundary condition metadata.
        """
        bnds=self.read_bnd()
        if bnds is not None:
            return self.group_boundary_links_bnd(bnds)

        gbl=self.read_group_boundary_links()
        if gbl is not None:
            return gbl
            
        self.log.info("Couldn't find file with group_boundary_links data")
        return super(HydroFiles,self).group_boundary_links()

    def group_boundary_links_bnd(self,bnds):
        poi=self.pointers
        self.infer_2d_elements()
        self.infer_2d_links()

        gbl=np.zeros( self.n_2d_links,self.link_group_dtype )
        gbl['id']=-1 # initialize to non-boundary

        # each bnd entry has one or more surface links
        # not comfortable assuming that their link numbers are
        # the same as mine, but appears that theirs are consistent
        # with surface layer exchange ids.
        for rec_i,bnd in enumerate(bnds):
            # bc_elt is a negative boundary element index, which
            # can be matched to a "from" segment in poi
            # except note that these are links, not exchanges!
            for bc_elt in bnd[1]['link']:
                # This line makes a bit of a leap to match the 2D element
                # to a pointer record which refers to segments.
                exch_i=np.nonzero(poi[:,0]==bc_elt)[0]
                assert len(exch_i)==1
                exch_i=exch_i[0]
                # seems that these come in only as horizontal exchanges.
                # a vertical exchange couldn't be a link, so play it safe:
                assert exch_i < self.n_exch_x + self.n_exch_y

                # This is a 0-based link index
                link0=self.exch_to_2d_link['link'][exch_i]
                if gbl['id'][link0]>=0:
                    print("WARNING: multiple id values map to the same link (%s)"%bnd[0])
                gbl['id'][link0]=rec_i
                gbl['name'][link0]=bnd[0]

                other={}

                other['wkt']=geometry.LineString(bnd[1]['x'].reshape([-1,2]) ).wkt
                other['geom']=other['wkt'] # follow previous interface, leaving geom
                # as text wkt.
                gbl['attrs'][link0]=other
        # 2018-12-12: losing some boundary links on re-reading aggregated
        # output.  at least warn if boundary links did not get labeled
        n_bc_links=(self.links[:,0]<0).sum()
        n_grouped_links=(gbl['id']>=0).sum()
        if n_bc_links!=n_grouped_links:
            self.log.warning("labeling boundary links: %d expected bc links != %d that got names from bnd"%
                             (n_bc_links,n_grouped_links))
        return gbl
    
    def read_bnd(self):
        """
        parse the .bnd file present in some DFM runs.
        Returns a list [ ['boundary_name',array([ boundary_link_idx,[[x0,y0],[x1,y1]] ])], ...]

        This used to call the link index 'exch', but that is misleading since it's 
        a 2D item.
        """
        try:
            fn=self.get_path('boundaries-file',check=True)
        except KeyError:
            return None

        return dio.read_bnd(fn)
    

    def read_group_boundary_links(self):
        """ Attempt to read grouped boundary links from file.  Return None if
        file doesn't exist.
        """
        gbl_fn=os.path.join(self.get_dir(),'boundary-links.csv')
        if not os.path.exists(gbl_fn):
            return None
        
        df=pd.read_csv(gbl_fn)

        self.infer_2d_links()
        gbl=np.zeros( self.n_2d_links,self.link_group_dtype )
        gbl['id']=-1 # initialize to non-boundary
        for reci,rec in df.iterrows():
            link0=rec['link0']
            gbl['id'][link0]=reci # not necessarily unique!
            gbl['name'][link0]=rec['name']
            other={}
            for k,v in rec.iteritems():
                # name is repeated here to make downstream code simpler
                if k not in ['id']:
                    # could be cleverish and convert 'geom' WKT back to geometry.
                    # asking a little much, I'd say.
                    other[k]=v
            gbl['attrs'][link0]=other

        return gbl
        

REINDEX=-9999
class DwaqAggregator(Hydro):
    """
    Aggregate hydrodynamics from source data potentially spread across
    many subdomains, and possibly only a spatial or temporal subset of
    that data.
    """

    # a sentinel used to indicate that data has changed an index is no longer
    # valid
    REINDEX=REINDEX

    # whether or not to force all layers to have the same number of
    # segments.
    sparse_layers=True

    # Force links between unaggregated elements to be consistent with links
    # in the aggregation polygon by nudging elements between aggregated
    # elements.  See init_elt_mapping
    nudge_elements=True

    # if True, boundary exchanges are combined so that any given segment has
    # at most one boundary exchange
    # if False, these exchanges are kept distinct, leading to easier addressing
    # of boundary conditions but pressing one's luck in terms of how many exchanges
    # can go to a single segment.
    agg_boundaries=True

    # # Not yet implemented:
    # # 'geometric': use element geometry to calculate areas and length of overlap
    # #   then divide to get a length scale#
    # # 'circumcenters': use element circumcenters
    # # 'centroids': use element centroids
    horizontal_length_scales='subdomain,geometric'

    # Rather than loading DWAQ flowgeom netcdf, try to load DFM map output, which
    # can have more information.
    flowgeom_use_dfm_map=True

    # When creating aggregated grid from a shapefile use this tolerance for matching
    # nodes.
    agg_shp_tolerance=0.0

    # how many times to loop through trying to adjust the mapping of input cells
    # to aggregated cells, with the goal of avoiding any new aggregated faces.
    # This isn't always possible even with a perfect method, and the algorithm here
    # is not particularly clever. Alternatively, nudging can be disabled via
    # max_nudge_iterations=0.
    max_nudge_iterations=5
    
    # how many of these can be factor out into above classes?
    # agg_shp: can specify, but if not specified, will try to generate a 1:1 
    #   mapping
    # run_prefix: 
    # path:
    # nprocs: 
    def __init__(self,agg_shp=None,nprocs=None,skip_load_basic=False,sparse_layers=None,
                 merge_only=False,
                 **kwargs):
        super(DwaqAggregator,self).__init__(**kwargs)
        # where/how to auto-create agg_shp??
        # where is it first needed? load_basic -> init_elt_mapping -> init_agg_elements_2d
        # what is the desired behavior?
        #   what about several possible types for agg_shp?
        #    - filename -> load as shapefile, extract name if it exists
        #    - an unstructured grid -> load those cells, use cell id for name.
        if sparse_layers is not None:
            self.sparse_layers=sparse_layers

        # some steps can be streamline when we know that there is no aggregation, just
        # merging multiprocessor to single domain, and no change of ordering for elements
        self.merge_only=merge_only

        self.agg_shp=agg_shp
        if nprocs is None:
            nprocs=self.infer_nprocs()
        self.nprocs=nprocs

        if not skip_load_basic:
            self.load_basic()

    def load_basic(self):
        """ 
        populate general, time-invariant info
        """
        self.find_maxima()

        self.init_elt_mapping()
        self.init_seg_mapping()
        self.init_exch_mapping()
        self.reindex()
        self.update_agg_to_local()
        self.add_exchange_data()
                    
        self.init_exch_matrices()
        self.init_seg_matrices()
        self.init_boundary_matrices()

    def open_hyd(self,p,force=False):
        raise Exception("Must be overloaded")

    _flowgeoms=None
    def open_flowgeom(self,p):
        """
        p: processor number.
        Returns the flowgeom netcdf file, opened as a QNC dataset.
        This will be cached in self._flowgeoms
        """
        # hopefully this works for both subclasses...
        if self._flowgeoms is None:
            self._flowgeoms={}
        if p not in self._flowgeoms:
            hyd=self.open_hyd(p)
            fg=qnc.QDataset(hyd.get_path('grid-coordinates-file'))
            self._flowgeoms[p] = fg
        return self._flowgeoms[p]

    def dfm_map_file(self,p):
        """
        Try to infer the name of the dfm map output for processor p, and
        if it exists, return that path.
        """
        assert p==0,"For HydroAggregator, only serial runs are supported."
        map_fn=self.hydro_in.hyd_path.replace("DFM_DELWAQ_","DFM_OUTPUT_").replace('.hyd','_map.nc')
        if os.path.exists(map_fn):
            return map_fn
        else:
            self.log.warning("Tried to find DFM map output at %s, but it wasn't there"%map_fn)
            return None
    
    _flowgeoms_xr=None
    def open_flowgeom_ds(self,p):
        """ Same as open_flowgeom(), but transitioning to xarray dataset instead of qnc.

        This also adds in some fields which can be inferred in the case the original data
        is missing them, namely FlowElemDomain, FlowElemGlobalNr.

        Also adds a FlowLinkSS field and dimensions that adds sink-source links back in
        from the hyd file.
        """
        if self._flowgeoms_xr is None:
            self._flowgeoms_xr={}

        if p not in self._flowgeoms_xr:
            hyd=self.open_hyd(p)
            flowgeom_fn=hyd.get_path('grid-coordinates-file')
            # some information is here, but not in the map file,
            # so always try to open this, and optionally try to open the map, too.
            fg_ds=ds=xr.open_dataset(flowgeom_fn)
            map_ds=None
            
            if self.flowgeom_use_dfm_map:
                if p==0:
                    msg=self.log.info
                else:
                    msg=self.log.debug
                msg("[proc=%d] Checking to see if a DFM map file can be used instead of flowgeom"%p)
                map_fn=self.dfm_map_file(p)
                if map_fn:
                    msg("Yes - will load %s"%map_fn)
                    map_ds=xr.open_dataset(map_fn)
                    
                    # when map is available, use it, but remember fg_ds is also still around for
                    # things like global element number
                    ds=map_ds
            
            # if this is ugrid output, include copies of some variables to their pre-ugrid
            # names
            for old,new in [ ('NetNode_x','mesh2d_node_x'),
                             ('NetNode_y','mesh2d_node_y'),
                             ('NetNode_z','mesh2d_node_z'),
                             ('NetElemNode','mesh2d_face_nodes'),
                             # some duplicates because newer DFM output varies
                             ('FlowElemGlobalNr','mesh2d_face_global_number'),
                             ('FlowElemGlobalNr','mesh2d_flowelem_globalnr'),
                             ('FlowElemDomain','mesh2d_face_domain_number'),
                             ('FlowElemDomain','mesh2d_flowelem_domain'),
                             ('FlowElem_bl','mesh2d_flowelem_bl'),
                             ('FlowElem_bac','mesh2d_flowelem_ba')
                             ]:
                if old in ds: continue
                
                if new in ds:
                    data=ds[new] # should just be a shallow copy.  cheap.
                elif old in fg_ds:
                    # waqgeom / flowgeom might have it?
                    data=fg_ds[old].copy()
                elif new in fg_ds:
                    # waqgeom / flowgeom might have the new version.
                    data=fg_ds[new].copy()
                else:
                    # still no source for this. move on.
                    continue 
                    
                # also some of those index fields are saved as float(?!)
                if old in ['FlowElemDomain','FlowElemGlobalNr','NetElemNode']:
                    data.values[ np.isnan(data.values) ]=-999999
                    data=data.astype(np.int32)

                ds[old]=data

            # special handling for other dimensions
            if 'nFlowElem' not in ds.dims and 'nmesh2d_face' in ds.dims:
                # helps down the line to make this its own dimension, not just reuse
                # the dimension from nmesh2d_face.
                ds['nFlowElem']=('nFlowElem',),ds['nmesh2d_face'].values
            
            if ('NetNode_x' in ds) and ('NetElemNode' in ds['NetNode_x'].dims):
                self.log.info("Trying to triage bad dimensions in NetCDF (probably ddcouplefm output)")
                netnode_x=ds.NetNode_x.values
                netnode_y=ds.NetNode_y.values
                netnode_z=ds.NetNode_z.values

                del ds['NetNode_x']
                del ds['NetNode_y']
                del ds['NetNode_z']

                netelemnode = ds.NetElemNode.values
                del ds['NetElemNode']

                ds['NetNode_x']=('nNetNode',),netnode_x
                ds['NetNode_y']=('nNetNode',),netnode_y
                ds['NetNode_z']=('nNetNode',),netnode_z
                ds['NetElemNode']=('nNetElem','nNetElemMaxNode'),netelemnode
                    
            if self.nprocs==1:
                elem_dim='nFlowElem'
                n_elem=len(ds[elem_dim])
                if 'FlowElemGlobalNr' not in ds:
                    self.log.info("Synthesizing multi domain data for single domain run")
                    ds['FlowElemGlobalNr']=(elem_dim,),1+np.arange(n_elem,dtype=np.int32)
                if 'FlowElemDomain' not in ds:
                    ds['FlowElemDomain']  =(elem_dim,),np.zeros(n_elem,np.int32)
                
            # This had been just for nprocs==1.  But it's needed for MPI, too.
            if 'FlowLink' not in ds:
                link_dim='nFlowLink'

                # when FlowLink is not present, go ahead and create it, by
                # looking at edge_type==1 or 2, which should encompass all edges
                # with flow.
                edge_type=ds.mesh2d_edge_type.values
                
                # array that is indexed by flowlink, and returns the
                # edge index.
                # generally it could be more complicated, but 
                # based on a single comparison between two runs, exactly the same
                # except for MapFormat, the FlowLinks are in the same order
                # as the edges, it's just that the edges have the extra closed_boundary
                # edges.
                self.flowlink_to_edge=np.nonzero( (edge_type==1)|(edge_type==2) )[0]
                n_flowlinks=len(self.flowlink_to_edge)

                # Fill in FlowLink-related fields --
                ds['FlowLink_xu']=(link_dim,), ds.mesh2d_edge_x.values[self.flowlink_to_edge]
                ds['FlowLink_yu']=(link_dim,), ds.mesh2d_edge_y.values[self.flowlink_to_edge]

                # fabricating FlowLink.  This is more complicated because of how boundaries
                # are handled.  Tested to some degree, but this is getting into DFM details
                # that may not be documented or guaranteed to stay the same.
                # Starting point - just the subset of edges that are flow links
                FlowLink=ds.mesh2d_edge_faces.values[self.flowlink_to_edge,:]
                # swap out nan for -1 and switch to integers
                FlowLink[ np.isnan(FlowLink) ] = -1
                # somehow in this case the boundary elements are coming in as 0, while in
                # testing external to this code they were nan.
                # try to hedge by assuming any index less than start_index is a boundary,
                # and default to start_index of 1 since that is what DFM has generally used
                # here.  The general UGRID/CF default is 0, but there is a better chance
                # of somebody dropping the attribute, than of DFM silently switching to
                # the default value.
                start_index=ds.mesh2d_edge_faces.attrs.get('start_index',1)
                FlowLink[ FlowLink<start_index ] = -1

                FlowLink=FlowLink.astype(np.int32)
                # Which links are boundary links?
                # The ordering of elements for links gets screwy. Format=1 this is
                # moot, since it already includes FlowLinks.  Format=4, it seems that
                # flowgeom files get nan-valued boundary elements placed in the second
                # column.  but a map file from the same run doesn't constrain the ordering
                # and the 0-valued boundary elements can appear in either column.
                # so this code tries to accommodate both, while still including some
                # sanity checks
                bc_links=(FlowLink.min(axis=1)<0)
                bc_link_flip=FlowLink[:,1]<0
                # map files may not need all links flipped.
                FlowLink[bc_link_flip]=FlowLink[bc_link_flip,::-1]
                # minor sanity check
                assert np.all(FlowLink[:,1]>=start_index),"Bad assumption on boundary links"

                # and boundary elements get numbered beyond the number of regular elements.
                face_dim=ds['mesh2d'].attrs.get('face_dimension','nmesh2d_face')
                FlowLink[bc_links,0]=ds.dims[face_dim] + 1 + np.arange(bc_links.sum())

                assert np.all(FlowLink[:,0]>=start_index),"Missed some boundary links??"

                ds['FlowLink'] = (link_dim,'nFlowLinkPts'), FlowLink

                # type 2 is flow link between 2D elements. Will have to change if/when we get into
                # mixed 1D/2D/3D grids.
                ds['FlowLinkType']=(link_dim,), 2*np.ones(n_flowlinks,np.int32)
                # ignore FlowLink_lonu, FlowLink_latu

                print("FlowLink near end of open_flowgeom_ds(%d), id(ds)=%s"%(p,id(ds)))
                print(ds.FlowLink)

            if 'FlowLinkDomain' not in ds:
                # 2024-04-04: move this out one level -- appears it is possible for FlowLink
                # to exist, but not FlowLinkDomain
                # use the element domains, grabbing the internal/2nd flow element from
                # each link.
                # remember that FlowLink has 1-based numbering (can't assume that mesh2d_edge_faces
                # exists, and no obvious metadata for start_index, but it's always 1 with DFM stuff)
                start_index=1 
                FlowLink =ds['FlowLink'].values
                ds['FlowLinkDomain'] = ds['FlowLink'].dims[:1],ds['FlowElemDomain'].values[FlowLink[:,1]-start_index]

            if True: # Fabricate FlowLink that includes sink-source links
                if 'sink-sources' in hyd.hyd_toks:
                    sink_srcs= hyd.hyd_toks['sink-sources']
                    n_sink_src = len(sink_srcs)
                    extra_FlowLink = np.zeros( (n_sink_src,2), np.int32)
                    for ss_idx,sink_src in enumerate(sink_srcs):
                        extra_FlowLink[ss_idx,0]=ds.dims['nFlowElem'] + -sink_src['link_id']
                        extra_FlowLink[ss_idx,1]=sink_src['elt_id']
                    FlowLinkSS = np.concatenate( (ds.FlowLink.values, extra_FlowLink), axis=0)
                    FlowLinkSSDomain = np.concatenate( (ds.FlowLinkDomain.values, np.full(n_sink_src,p)) )
                    FlowLinkSSType = np.concatenate( (ds.FlowLinkType.values, np.full(n_sink_src,2)) )
                    # ignore the latu/lonu fields
                else:
                    FlowLinkSS = ds.FlowLink.values
                    FlowLinkSSDomain = ds.FlowLinkDomain.values
                    FlowLinkSSType = ds.FlowLinkType.values
                ds['FlowLinkSS'] = ('nFlowLinkSS',ds.FlowLink.dims[1]),FlowLinkSS
                ds['FlowLinkSSDomain'] = ('nFlowLinkSS',),FlowLinkSSDomain
                ds['FlowLinkSSType'] = ('nFlowLinkSS',),FlowLinkSSType

                # Make sure we're not double-counting. 
                sub_bnds = hyd.read_bnd()
                min_bc_linkid = min( [0] +[sub_bnd[1]['link'].min() for sub_bnd in sub_bnds] )
                # I think this is fair, but it might be too stringent.
                assert FlowLinkSS.max() == ds.dims['nFlowElem'] - min_bc_linkid
                
            self._flowgeoms_xr[p] = ds
            
        return self._flowgeoms_xr[p]

    seg_local_dtype=[('agg','i4'), # aggregated segment it maps to, even if ghost
                     ('ghost','i1'), # whether this is a ghost segment
                     ]
    exch_local_dtype=[('agg','i4'), # the aggregated exchange it maps to
                      ('sgn','i1'), # whether it maps forward or reversed
                      ]
    
    agg_elt_2d_dtype=[('plan_area','f4'),('name','S100'),('zcc','f4'),('poly',object)]
    agg_seg_dtype=[('k','i4'),('elt','i4'),('active','b1')]
    agg_exch_dtype=[('from_2d','i4'),('from','i4'),
                    ('to_2d','i4'),('to','i4'),
                    ('from_len','f4'),('to_len','f4'),
                    ('direc','S1'),
                    ('k','f4') # float: vertical exchanges have fractional layer.
    ]

    @property
    def n_seg(self):
        return self.n_agg_segments

    def seg_active(self):
        # return boolean array of whether each segment is active
        return self.agg_seg['active']

    def seg_attrs(self,number):
        if number==1:
            return self.seg_active().astype('i4')
        elif number==2:
            return super(DwaqAggregator,self).seg_attrs(number=number)

    def init_agg_elements_2d(self):
        """
        load the aggregation polygons, setup the corresponding 2D 
        data for those elements.

        populates self.elements, self.n_agg_elements_2d
        """
        if isinstance(self.agg_shp,str):
            box_defs=wkb2shp.shp2geom(self.agg_shp)
            box_polys=box_defs['geom']
            try:
                box_names=box_defs['name']
            except ValueError:
                box_names=["%d"%i for i in range(len(box_polys))]
        else:
            agg_shp=self.agg_shp
            if agg_shp is None and self.nprocs==1:
                agg_shp=self.open_hyd(0).grid()

            if isinstance(agg_shp,unstructured_grid.UnstructuredGrid):
                g=agg_shp
                box_polys=[g.cell_polygon(i) for i in g.valid_cell_iter()]
                box_names=["%d"%i for i in range(len(box_polys))]
            else:
                raise Exception("Need some guidance on agg_shp")

        self.n_agg_elements_2d=len(box_polys)

        agg_elts_2d=[]
        # used to work, but appears to be broken with newer python
        #    for agg_i in range(self.n_agg_elements_2d):
        #        elem=np.zeros((),dtype=self.agg_elt_2d_dtype)
        #        elem['name']=box_names[agg_i]
        #        elem['poly']=box_polys[agg_i]
        #        agg_elts_2d.append(elem)
        #    # per http://stackoverflow.com/questions/15673155/keep-getting-valueerror-with-numpy-while-trying-to-create-array
        #    self.elements=rfn.stack_arrays(agg_elts_2d)
        # 2018-10-01 RH: try different approach.  should be faster, better anyway.
        self.elements=np.zeros(self.n_agg_elements_2d,dtype=self.agg_elt_2d_dtype)
        for agg_i in range(self.n_agg_elements_2d):
            self.elements['name'][agg_i]=box_names[agg_i]
            self.elements['poly'][agg_i]=box_polys[agg_i]

    def find_maxima(self):
        """
        find max global flow element id to preallocate mapping table
        and the links
        """
        max_gid=0
        max_elts_2d_per_proc=0
        max_lnks_2d_per_proc=0

        for p in range(self.nprocs):
            nc=self.open_flowgeom_ds(p)
            if 'FlowElemGlobalNr' in nc:
                max_gid=max(max_gid, nc.FlowElemGlobalNr.values.max() )
            #elif 'mesh2d_face_global_number' in nc:
            #    max_gid=max(max_gid, nc.mesh2d_face_global_number.values.max() )
            elif self.nprocs==1:
                max_gid=len(nc['nFlowElem'])
            else:
                raise Exception("Need global element numbers for multi-processor run")
            if 'mesh2d' in nc:
                ncell=nc.mesh2d.attrs['face_dimension']
            elif 'nFlowElem' in nc.dims:
                ncell='nFlowElem'
            elif 'nmesh2d_face' in nc.dims:
                ncell='nmesh2d_face'
            else:
                raise Exception("Could't find face dimension")
            max_elts_2d_per_proc=max(max_elts_2d_per_proc,len(nc[ncell]))
            # If this works, FlowLinkSS should always get populated.
            assert 'nFlowLinkSS' in nc.dims,"Seems we're not using the new FlowLinkSS code??"
            if 'nFlowLinkSS' in nc.dims:
                nlinks_here=len(nc['nFlowLinkSS'])
            elif 'nFlowLink' in nc.dims:
                nlinks_here=len(nc['nFlowLink'])
            elif 'mesh2d_face_links' in nc:
                nlinks_here=nc['mesh2d_face_links'].max()
            elif 'nmesh2d_edge' in nc.dims:
                # missing link info may be a problem later, but at the moment we just need
                # an upper bound
                self.log.warning("No Link information in waqgeom, so will estimate max by number of edges")
                nlinks_here=nc.dims['nmesh2d_edge']
            else:
                raise Exception("Couldn't find link dimension")
            # Since the above is referencing nFlowLinkSS, no need to add them in this way.
            # hyd = self.open_hyd(p)
            # if 'sink-sources' in hyd.hyd_toks:
            #     # 2024-01-26: source/sink pairs might contribute, but it appears
            #     # that the netcdf information no longer includes them, and instead
            #     # lists them in the hyd file:
            #     nlinks_here += len(hyd.hyd_toks['sink-sources'])
                                   
            max_lnks_2d_per_proc=max(max_lnks_2d_per_proc,nlinks_here)
            # no nc.close(), as it is now cached

        n_global_elements=int(max_gid) # ids should be 1-based, so this is also the count

        # For exchanges, have to be get a bit more involved - in fact, just consult
        # the hyd file, since we don't know whether closed exchanges are included or not.
        max_hor_exch_per_proc=0
        max_ver_exch_per_proc=0
        max_exch_per_proc=0
        max_bc_exch_per_proc=0
        self.n_src_layers=0
        
        for p in range(self.nprocs):
            hyd=self.open_hyd(p)
            n_hor=hyd['number-horizontal-exchanges']
            n_ver=hyd['number-vertical-exchanges']
            max_hor_exch_per_proc=max(max_hor_exch_per_proc,n_hor)
            max_ver_exch_per_proc=max(max_ver_exch_per_proc,n_ver)
            max_exch_per_proc=max(max_exch_per_proc,n_hor+n_ver)
            
            poi=hyd.pointers
            n_bc=np.sum( poi[:,:2] < 0 )
            max_bc_exch_per_proc=max(max_bc_exch_per_proc,n_bc)

            # more generally the *max* number of layers
            # For DFM multiprocessor output with z layers, it's possible
            # for this to vary among domains
            self.n_src_layers=max(self.n_src_layers,hyd['number-water-quality-layers'])

        # could be overridden, but take max number of aggregated layers to be
        # same as max number of unaggregated layers
        self.n_agg_layers=self.n_src_layers

        max_segs_per_proc=self.n_src_layers * max_elts_2d_per_proc

        self.log.debug("Max global flow element id: %d"%max_gid )
        self.log.debug("Max 2D elements per processor: %d"%max_elts_2d_per_proc )
        self.log.debug("Max 3D segments per processor: %d"%max_segs_per_proc)
        self.log.debug("Max 3D exchanges per processor: %d (h: %d,v: %d)"%(max_exch_per_proc,
                                                                  max_hor_exch_per_proc,
                                                                  max_ver_exch_per_proc))
        self.log.debug("Max 3D boundary exchanges per processor: %d"%(max_bc_exch_per_proc))

        self.n_global_elements=n_global_elements
        self.max_segs_per_proc=max_segs_per_proc
        self.max_gid=max_gid
        self.max_exch_per_proc=max_exch_per_proc
        self.max_hor_exch_per_proc=max_hor_exch_per_proc
        self.max_ver_exch_per_proc=max_ver_exch_per_proc
        self.max_bc_exch_per_proc=max_bc_exch_per_proc

    
    # fast-path for matching local elements to aggregation polys.
    agg_query_size=5 
    def init_elt_mapping(self):
        """
        Map global, 2d element ids to 2d boxes (i.e. flowgeom)
        """
        self.init_agg_elements_2d()

        # initialize to -1, signifying that unaggregated elts are by default
        # not mapped to an aggregated element.
        self.elt_global_to_agg_2d=np.zeros(self.n_global_elements,'i4') - 1

        self.elements['plan_area']=0.0
        self.elements['zcc']=0.0

        # best way to speed this up?
        # right now, 3 loops
        # processors
        #  cells on processor
        #    polys in agg_shp.

        # a kd-tree of the agg_shp centroids?
        # this is still really bad in the case where there are many local cells,
        # many aggregation polys, but the latter do not cover the former. 
        # For that, RTree would be helpful since it can handle real overlaps.
        if not self.merge_only:
            agg_centers=np.array( [p.centroid.coords[0] for p in self.elements['poly']] )
            kdt=scipy.spatial.KDTree(agg_centers)
            total_poly=[None] # box it for py2/py3 compatibility
        else:
            kdt=total_poly=agg_centers="na"

        def match_center_to_agg_poly(x,y):
            pnt=geometry.Point(x,y)
            
            dists,idxs=kdt.query([x,y],self.agg_query_size)
            
            for poly_i in idxs:
                if self.elements['poly'][poly_i].contains(pnt):
                    return poly_i
            else:
                self.log.debug("Falling back on exhaustive search")

            if total_poly[0] is None and cascaded_union is not None:
                self.log.info("Calculating union of all aggregation polys")
                total_poly[0] = cascaded_union(self.elements['poly'])
            if total_poly[0] is not None and not total_poly[0].contains(pnt):
                return None

            for poly_i,poly in enumerate(self.elements['poly']):
                if poly.contains(pnt):
                    return poly_i
            else:
                return None

        for p in range(self.nprocs):
            self.log.info("init_elt_mapping: proc=%d"%p)
            # used to use circumcenters, but that can be problematic
            nc=self.open_flowgeom_ds(p)
            if 'FlowElem_zcc' in nc:
                # typically positive down
                ccz=nc.FlowElem_zcc[:].values
            elif 'FlowElem_bl' in nc:
                # typically positive up, negate to be consistent with FlowElem_zcc
                ccz=-nc.FlowElem_bl[:].values
            else:
                raise Exception("flowgeom does not have cell depths.  Maybe there is DFM map output that we didn't find?")
                
            ncg=unstructured_grid.UnstructuredGrid.read_dfm(nc) # used to be dfm_grid
            g_centroids=ncg.cells_centroid()
            ccx=g_centroids[:,0]
            ccy=g_centroids[:,1]

            dom_id=nc.FlowElemDomain.values.astype(np.int32)
            global_ids=(nc.FlowElemGlobalNr.values - 1).astype(np.int32)  # make 0-based

            try:
                areas=nc.FlowElem_bac.values
            except AttributeError:
                self.log.info("Cell area missing in netcdf, will be computed from grid")
                areas=ncg.cells_area()

            # The datum doesn't matter, volumes are just calculated to get a proper
            # average bed depth.
            vols=areas*ccz # volume to nominal MSL

            n_local_elt=len(ccx)
            hits=0
            # as written, quadratic in the number of cells.
            # Looks like that is no longer true
            for local_i in range(n_local_elt):
                if dom_id[local_i]!=p:
                    continue # only check elements in their native subdomain
                if self.merge_only:
                    poly_i=global_ids[local_i]
                else:
                    poly_i = match_center_to_agg_poly(ccx[local_i],
                                                      ccy[local_i] )
                    if poly_i is None:
                        continue

                if hits%2000==0:
                    self.log.info('2D element within aggregation polygon: %d'%hits)
                hits+=1

                self.elt_global_to_agg_2d[global_ids[local_i]]=poly_i
                self.elements[poly_i]['plan_area'] = self.elements[poly_i]['plan_area'] + areas[local_i]
                # weighted by area, to be normalized below.
                self.elements[poly_i]['zcc'] = self.elements[poly_i]['zcc'] + vols[local_i]
                                
            msg="Processor %4d: %6d 2D elements within an aggregation poly"%(p,hits)
            if hits:
                self.log.info(msg)
            else:
                self.log.debug(msg)


        if (not self.merge_only) and self.nudge_elements:
            global_remapped={} # global id => nudged element.
            if isinstance(self.agg_shp,unstructured_grid.UnstructuredGrid):
                agg_g=self.agg_shp
            else:
                agg_g=unstructured_grid.UnstructuredGrid.from_shp(self.agg_shp,tolerance=self.agg_shp_tolerance)
            
            for nudge_iter in range(self.max_nudge_iterations):
                nudge_count=0
                # First, identify problem edges
                for proc in range(self.nprocs):
                    self.log.info("Checking proc %d for inconsistent links"%proc)
                    nc=self.open_flowgeom_ds(proc)
                    proc_global_ids=nc.FlowElemGlobalNr.values - 1  # make 0-based
                    ncg=unstructured_grid.UnstructuredGrid.read_dfm(nc,cleanup=True) 

                    e2c=ncg.edge_to_cells()
                    for j in np.nonzero(e2c.min(axis=1)>=0)[0]: # only internal edges
                        loc_c1,loc_c2=e2c[j]
                        # which aggregation polygons do these map to?
                        glb_c1,glb_c2=proc_global_ids[ e2c[j] ]
                        agg_1,agg_2=self.elt_global_to_agg_2d[ [glb_c1,glb_c2] ]
                        if (agg_1<0) or (agg_2<0):
                            continue # not concerned about edges leaving the domain

                        agg_j=agg_g.cells_to_edge(agg_1,agg_2)
                        if agg_j is None:
                            self.log.info("proc %d, j=%d is a problem edge"%(proc,j))

                            # First attempt: assume that loc_c1 can be nudged to a different
                            # aggregated element.  This resulted in some cases where it was
                            # nudged multiple times.

                            # Default to nudging c1, but if it has already been nudged due
                            # to some other edge, try flipping around and nudging c2
                            if glb_c1 in global_remapped:
                                assert glb_c2 not in global_remapped,"Both sides of this edge have been remapped!"
                                self.log.info("   Problem edge will be flipped")
                                loc_c1,loc_c2=loc_c2,loc_c1
                                glb_c1,glb_c2=glb_c2,glb_c1
                                agg_1,agg_2=agg_2,agg_1

                            # Try to find a new home for loc_c1:
                            # First get the list of allowable new elements for c1:
                            agg_c1_nbrs=np.unique( agg_g.cell_to_cells(agg_1) )
                            agg_c1_nbrs=agg_c1_nbrs[ agg_c1_nbrs>=0 ]
                            agg_c2_nbrs=np.unique( agg_g.cell_to_cells(agg_2) )
                            agg_c2_nbrs=agg_c2_nbrs[ agg_c2_nbrs>=0 ]

                            # tempting to just slide c1 over into the same element as c2, but
                            # that would just make the problem worse since c1 will almost certainly
                            # have neighbors in agg_c1, but now they are farther from the real edge
                            # potential_agg_c1s=np.concatenate( (agg_c2_nbrs,[agg_2]) )

                            # potential_agg_c1s=agg_c2_nbrs # close, but still leaves some bad edges
                            # This may be too strict -- there is an assumption that c1 has additional
                            # unaggregated edges with other cells in agg_c1.  In that case, we can only
                            # move c1 to a new element X such that agg_c1-X and X-agg_c2 are both
                            # allowable.
                            potential_agg_c1s=np.intersect1d( agg_c1_nbrs, agg_c2_nbrs)
                            assert len(potential_agg_c1s), "May have been too strict with intersection"

                            new_agg_1=None
                            c1_poly=ncg.cell_polygon(loc_c1)
                            geometric_matches=[]
                            for potential_agg_c1 in potential_agg_c1s:
                                nbr_poly=agg_g.cell_polygon(potential_agg_c1)
                                if nbr_poly.intersects(c1_poly):
                                    new_agg_1=potential_agg_c1
                                    # for testing - does it match geometrically to multiple choices?
                                    geometric_matches.append(potential_agg_c1)
                            if new_agg_1 is None:
                                self.log.warning("No matches with geometry, will use topology for proc=%d loc_c1=%d"%(proc,loc_c1))
                                new_agg_1=potential_agg_c1s[0] # punt. Could try computing distances?
                            # assert new_agg_1 is not None

                            if len(geometric_matches)>1:
                                self.log.warning(" Multiple choices of where to map this problem cell")

                            # Record that it has been nudged, to avoid un-nudging later
                            global_remapped[glb_c1]=new_agg_1
                            self.elt_global_to_agg_2d[glb_c1]=new_agg_1
                            nudge_count+=1
                if nudge_count==0:
                    break
                else:
                    self.log.info('Had to nudge %d elements - will loop again'%nudge_count)
                
        # and normalize those depths
        self.elements['zcc'] = self.elements['zcc'] / self.elements['plan_area']

        print("-"*40)
        n_elt_to_print=10
        if len(self.elements)>n_elt_to_print:
            print("Only showing first %d elements"%n_elt_to_print)

        for elt2d in self.elements[:n_elt_to_print]:
            self.log.info("{:<20}   area: {:6.2f} km2  mean"
                          " depth to 0 datum: {:5.2f} m".format(elt2d['name'],
                                                                elt2d['plan_area']/1e6,
                                                                elt2d['zcc']))

    def init_seg_mapping(self):
        """
        And use that to also build up map of (proc,local_seg_index) => agg_segment for 3D
        populates 
        self.seg_local[nprocs,max_seg_per_proc]
         - maps processor-local segment indexes to aggregated segment, all 0-based

        segments are added to agg_seg, but not necessarily in final order.
        """
        # maybe hold off, get a real number?
        # self.n_agg_segments=self.agg_linear_map.max()+1

        self.agg_seg=[]
        self.agg_seg_hash={} # k,elt => index into agg_seg
        
        self.seg_local=np.zeros( (self.nprocs,self.max_segs_per_proc),self.seg_local_dtype)
        self.seg_local['agg'][...] = -1

        if not self.sparse_layers:
            # pre-allocate segments to set them to inactive -
            # then below the code make them active as it goes.
            # inactive_segs then is set based on agg_seg[]['active']
            
            # set all agg_segs to inactive, and make them active only when somebody
            # maps to them.
            for agg_elt in range(self.n_agg_elements_2d):
                for k in range(self.n_agg_layers):
                    seg=self.get_agg_segment(agg_k=k,agg_elt=agg_elt)
                    self.agg_seg[seg]['active']=False
                    
        for p in range(self.nprocs):
            nc=self.open_flowgeom_ds(p) 
            seg_to_2d=None # lazy loaded

            dom_id=nc.FlowElemDomain.values
            n_local_elt=len(dom_id)
            global_ids=nc.FlowElemGlobalNr.values-1 # make 0-based

            for local_i in range(n_local_elt):
                global_id=global_ids[local_i]
                agg_elt=self.elt_global_to_agg_2d[global_id]
                if agg_elt<0:
                    continue # not in an aggreg. polygon

                # and 3D mapping:
                if seg_to_2d is None:
                    hyd=self.open_hyd(p)
                    hyd.infer_2d_elements()
                    seg_to_2d=hyd.seg_to_2d_element

                # Okay to test for ghost segments here, but they are still mapped
                # so that exchange mapping can follow them.
                local_elt_is_ghost=(dom_id[local_i]!=p)

                # this is going to be slow...
                segs=np.nonzero(seg_to_2d==local_i)[0]
                
                for k,local_3d in enumerate(segs):
                    # assumption: the local segments
                    # start with the surface, and match with the top subset
                    # of aggregated segments (i.e. hydro layers same as
                    # aggregated layers, except hydro may be missing deep
                    # layers).  that is what lets us use k (being the index
                    # of local segments in this local elt) as an equivalent
                    # to k_agg
                    agg_k=self.get_agg_k(proc=p,k=k,seg=local_3d)
                    one_agg_seg=self.get_agg_segment(agg_k=agg_k,agg_elt=agg_elt)
                    self.seg_local['agg'][p,local_3d]=one_agg_seg
                    self.seg_local['ghost'][p,local_3d]=local_elt_is_ghost
                    self.agg_seg[one_agg_seg]['active']=True

        # make a separate list of inactive segments
        self.inactive_segs=[]
        for agg_segi,agg_seg in enumerate(self.agg_seg):
            if not agg_seg['active']:
                self.inactive_segs.append(agg_segi)

    def get_agg_k(self,proc,k,seg):
        return k # no vertical aggregation

    link_ownership="FlowLinkDomain" # trust subdomain FlowLinkDomain to resolve ownership
    # "owner_of_min_elem" # link ownership goes to owner of the element with min. global nr.
    def init_exch_mapping(self):
        """
        Populates self.exch_local, including sign.  Ghost segments are currently
        skipped, as opposed to included but marked ghost.

        Old: 
        populate self.exch_local_to_agg[ nproc,max_exch_per_proc ]
         maps to aggregated exchange indexes.
         (-1: not mapped, otherwise 0-based index of exchange)
         and exch_local_to_agg_sgn, same size, but -1,1 depending on
         how the sign should map, or 0 if the exchange is not mapped.

        bc_local_to_agg used to be here, but became crufty.
        """
        # boundaries:
        # how are boundaries dealt with in existing poi?
        # see check_boundary_assumptions() for some details and verification

        # boundaries are always listed with the outside (negative)
        # index first (the from segment).
        # each segment with a boundary face gets its own boundary face index.
        # decreasing negative indices in poi correspond to the increasing positive
        # indices in flowgeom (-1=>nelem+1, -2=>nelem+2,...)
        
        # so far, all boundary exchanges are completely local - the internal segment
        # is always local, and the exchange is always local.  this code makes
        # a less stringent assumption - that the exchange is local iff the internal
        # segment is local.

        self.agg_exch=[] # entries add via reg_agg_exch=>get_agg_exch
        self.agg_exch_hash={} # (agg_from,agg_to) => index into agg_exch

        self.exch_local=np.zeros( (self.nprocs,self.max_exch_per_proc),self.exch_local_dtype) 
        self.exch_local['agg'][...] = -1
        self.exch_local['sgn'][...] = 0  # sign is separate - should be 1 or -1 (or 0 if no mapping)

        n_layers=self.n_agg_layers

        for p in range(self.nprocs):
            # this loop is a little slow.
            # skip a processor if it has no segments within
            # our aggregation regions:
            if np.all(self.seg_local['agg'][p,:]<0):
                self.log.debug("Processor %d - skipped"%p)
                continue

            self.log.debug("Processor %d"%p)

            hyd=self.open_hyd(p)
            pointers=hyd.pointers

            seg_active=hyd.seg_active()
            
            nc=self.open_flowgeom_ds(p) # for subdomain ownership

            elem_dom_id=nc.FlowElemDomain.values # domains are numbered from 0

            if self.link_ownership=="FlowLinkDomain":
                if p==0 and 'FlowLinkSSDomain' not in nc:
                    self.log.warning("FlowLinkSSDomain not found, so link ownership will use min element")
                    self.link_ownership="owner_of_min_elem"
                else:
                    link_dom_id=nc.FlowLinkSSDomain.values
            else:
                # otherwise, don't even load it.  probably means we can't trust it.
                pass 

            n_hor=hyd['number-horizontal-exchanges']
            n_ver=hyd['number-vertical-exchanges']

            if 'mesh2d' in nc:
                face_dim=nc.mesh2d.attrs.get('face_dimension','nFlowElem')
            else:
                face_dim='nFlowElem'
            nFlowElem=nc.dims[face_dim]

            pointers2d=pointers[:,:2].copy()
            sel=(pointers2d>0) # only map non-boundary segments
            hyd.infer_2d_elements() # all 0-based..
            pointers2d[sel] = hyd.seg_to_2d_element[ pointers2d[sel]-1 ] + 1

            # But really this code is problematic when dealing with z-level
            # or arbitrary grids.  Is it possible to map to 2D elements based
            # on order, but only for the top layer?
            # should be...

            # hmm - with ugrid output do not necessarily have link variable.
            if 'FlowLinkSS' in nc:
                links=nc.FlowLinkSS.values # 1-based
            else:
                raise Exception("Really should have FlowLinkSS")
                self.log.warning("Danger: faking missing link data with edges")
                # These are 1-based, with closed, non-computational neighbors
                # ==0
                links=nc.mesh2d_edge_faces.values.astype(np.int32)
            
            def local_elts_to_link(from_2d,to_2d):
                # slow - have to lookup the link
                sela=(links[:,0]==from_2d)&(links[:,1]==to_2d)
                selb=(links[:,0]==to_2d)  &(links[:,1]==from_2d)
                idxs=np.nonzero(sela|selb)[0]
                assert(len(idxs)==1),"Probably need to include source-sinks in links"
                return idxs[0] # 0-based index

            hits=0
            for local_i in range(len(pointers)):
                # these are *all* 1-based indices
                local_from,local_to=pointers[local_i,:2]
                from_2d,to_2d=pointers2d[local_i,:2] # Probably have to update pointers2d code.

                if local_i<n_hor:
                    direc='x' # structured grids not supported, no 'y'
                else:
                    direc='z'

                # Is the exchange real and local
                #
                if local_from==self.CLOSED or local_to==self.CLOSED:
                    continue # just omit CLOSED exchanges.
                elif local_to<0:
                    raise Exception("Failed assumption that boundary exchange is always first")
                elif local_from<0:
                    # it's a true boundary in the original grid
                    assert direc=='x' # really hope that it's horizontal

                    # is it a ghost?  Check based on locality of the internal segment:
                    internal_is_local=(elem_dom_id[to_2d-1]==p)

                    # should match locality of the link
                    # this makes too many assumptions about constant number of 
                    # exchanges per layer - in regular grids this was never violated

                    if not internal_is_local:
                        continue
                    # it's a true boundary, and not a ghost - proceed
                elif not seg_active[local_from-1] or not seg_active[local_to-1]:
                    # it's a bogus exchange on the local domain -- the exchange
                    # exists, but it involves an inactive segment.  Some DFM output
                    # with z-layers includes all exchanges, even to inactive segments,
                    # and here we weed those out, in addition to them being skipped
                    # in HydroFiles:infer_2d_elements()
                    continue
                else:
                    # it's a real exchange, but might be a ghost.
                    if direc=='x': # horizontal - could be a ghost
                        assert from_2d!=to_2d
                        # quick check - if both elements are local, then the
                        # link is local (true for the files I have)
                        # convert to 0-based for indexing
                        if elem_dom_id[from_2d-1]==p and elem_dom_id[to_2d-1]==p:
                            # it's local
                            pass
                        else:
                            # Ownership is not clear
                            if self.link_ownership=="FlowLinkDomain":
                                # trust subdomain FlowLinkDomain to resolve ownership
                                link_idx=local_elts_to_link(from_2d,to_2d)
                                if link_dom_id[link_idx]!=p:
                                    # print("ghost horizontal - skip")
                                    continue
                                else:
                                    # looked up the link, and it is local.
                                    # for sanity - local links have at least one local
                                    # element
                                    if elem_dom_id[from_2d-1]!=p and elem_dom_id[to_2d-1]!=p:
                                        print( "[proc=%d] from_2d=%d  to_2d=%d"%(p,from_2d,to_2d) )
                                        print( "          from_3d=%d  to_3d=%d"%(local_from,local_to) )
                                        print( "              dom=%d    dom=%d"%(elem_dom_id[from_2d],
                                                                                elem_dom_id[to_2d]) )
                                        print( "Consider setting link_ownership='owner_of_min_elem'")
                                        assert False
                            elif self.link_ownership=="owner_of_min_elem":
                                # link ownership goes to the lower numbered element owner.

                                # Was going to choose the min global elements, then go with
                                # that element's owner, but that means grabbing the globalnr,
                                
                                # Use this when FlowLinkDomain cannot be trusted.
                                # There is still an assumption that everyone agrees on element
                                # ownership.
                                if p==min(elem_dom_id[from_2d-1],
                                          elem_dom_id[to_2d-1]):
                                    # We won - continue with work below
                                    pass
                                else:
                                    # We lost - ghost link, move on
                                    continue
                    elif direc=='z':
                        # it's vertical - check if the elt is a ghost
                        if elem_dom_id[from_2d-1]!=p:
                            #print "ghost vertical - skip"
                            continue

                # so it's real (not closed), and local (not ghost)
                # might be an unaggregated boundary, in which case local_from<0

                if local_from<0:
                    # boundary
                    agg_to=self.seg_local['agg'][p,local_to-1]
                    agg_from=BOUNDARY
                    if agg_to==-1:
                        # unaggregated boundary exchange going to segment we aren't
                        # tracking
                        continue
                    else:
                        # a real unaggregated boundary which we have to track.
                        self.reg_agg_exch(direc=direc,
                                          agg_to=agg_to,
                                          agg_from=agg_from,
                                          proc=p,
                                          local_exch=local_i,
                                          local_from=local_from,
                                          local_to=local_to)
                else:
                    # do we care about either of these, and do they map
                    # to different aggregated volumes?
                    agg_from,agg_to=self.seg_local['agg'][p,pointers[local_i,:2]-1]

                    # -1 => not in an aggregation segment

                    if agg_from==-1 and agg_to==-1:
                        # between two segments we aren't tracking
                        continue 
                    elif agg_from==agg_to:
                        # within the same segment - fluxes cancel
                        continue
                    elif agg_from==-1 or agg_to==-1:
                        # agg boundary - have to manufacture boundary here
                        # currently, not explicitly recording the boundary segment
                        # info here, so when aggregated bcs are created, that info
                        # will not be packaged in the same way as bringing unaggregated
                        # boundary data into aggregated boundary data.
                        self.reg_agg_exch(direc=direc,
                                          agg_to=agg_to,
                                          agg_from=agg_from,
                                          proc=p,
                                          local_exch=local_i,
                                          local_from=local_from,
                                          local_to=local_to)
                    else:
                        if 1: # new style
                            self.reg_agg_exch(direc=direc,
                                              agg_from=agg_from,
                                              agg_to=agg_to,
                                              proc=p,
                                              local_exch=local_i,
                                              local_from=local_from,
                                              local_to=local_to)
                        else:
                            # this is a linkage which crosses between aggregated segments.
                            # the simplest real case
                            sel_fwd=(self.agg_exch['from']==agg_from)&(self.agg_exch['to']==agg_to)
                            sel_rev=(self.agg_exch['from']==agg_to)  &(self.agg_exch['to']==agg_from)
                            idxs_fwd=np.nonzero(sel_fwd)[0]
                            idxs_rev=np.nonzero(sel_rev)[0]

                            assert(len(idxs_fwd)+len(idxs_rev) == 1 )
                            if len(idxs_fwd):
                                self.exch_local['agg'][p,local_i]=idxs_fwd[0]
                                self.exch_local['sgn'][p,local_i]=1
                            else:
                                self.exch_local['agg'][p,local_i]=idxs_rev[0]
                                self.exch_local['sgn'][p,local_i]=-1

    def reg_agg_exch(self,direc,agg_from,agg_to,proc,local_exch,local_from,local_to):
        if agg_from is BOUNDARY: # was a boundary exchange in the unaggregated grid
            # the difference between this and the cases below is that here we want
            # to remember the index of the local bc, and associate it with
            # the aggregated BC.

            # if self.agg_boundaries is True, then            
            # all boundary exchanges for this agg_to segment map to a single
            # aggregated exchange.
            # if self.agg_boundaries is False, then repeated calls to get_agg_exchange
            # with the same agg_to create and return multiple, distinct boundary exchanges.
            
            # the -999 isn't really used beyond checking the sign, but there
            # to aid debugging.
            # 
            agg_exch_idx,sgn=self.get_agg_exchange(-999,agg_to)

            # dropped bc_local_to_agg cruft from here.
        else:
            # get_agg_exchange handles whether agg_from or agg_to are negative
            # indicating a boundary
            agg_exch_idx,sgn=self.get_agg_exchange(agg_from,agg_to)

        self.exch_local['agg'][proc,local_exch]=agg_exch_idx
        self.exch_local['sgn'][proc,local_exch]=sgn
        
    def reindex(self):
        """
        take the list versions of agg_exch and agg_seg, sort 
        in a reasonable way, and update indices accordingly.

        updates:
        agg_seg
        agg_exch
        bc_...
        agg_seg_hash
        agg_exch_hash
        ...
        """
        # keep anybody from using these - may have to 
        # repopulate depending...
        self.agg_seg_hash=None
        self.agg_exch_hash=None

        # presumably agg_seg and agg_exch start as lists - normalize
        # to arrays for reindexing
        agg_seg=np.asarray(self.agg_seg,dtype=self.agg_seg_dtype)
        agg_exch=np.asarray(self.agg_exch,dtype=self.agg_exch_dtype)

        # seg_order[0] is the index of the original segments which
        # comes first in the new order.
        seg_order=np.lexsort( (agg_seg['elt'], agg_seg['k']) )
        agg_seg=agg_seg[seg_order]
        # seg_mapping[0] gives the new index of what used to be index 0
        seg_mapping=utils.invert_permutation(seg_order)
        self.agg_seg=agg_seg

        # update seg_local_to_agg
        sel=self.seg_local['agg']>=0
        self.seg_local['agg'][sel]=seg_mapping[ self.seg_local['agg'][sel] ]

        # update from/to indices in exchanges:
        sel=agg_exch['from']>=0
        agg_exch['from'][sel] = seg_mapping[ agg_exch['from'][sel] ]
        # this used to be >0.  Is it possible that was the source of strife?
        sel=agg_exch['to']>=0
        agg_exch['to'][sel] =   seg_mapping[ agg_exch['to'][sel] ]
        self.n_agg_segments=len(self.agg_seg)

        # lexsort handling of boundary segments: 
        # should be okay - they will be sorted to the beginning of each layer

        exch_order=np.lexsort( (agg_exch['from'],agg_exch['to'],agg_exch['k'],agg_exch['direc']) )
        agg_exch=agg_exch[exch_order]
        exch_mapping=utils.invert_permutation(exch_order) 

        sel=self.exch_local['agg']>=0
        self.exch_local['agg'][sel]=exch_mapping[self.exch_local['agg'][sel]]

        # bc_local_to_agg - excised

        self.n_exch_x=np.sum( agg_exch['direc']==b'x' )
        self.n_exch_y=np.sum( agg_exch['direc']==b'y' )
        self.n_exch_z=np.sum( agg_exch['direc']==b'z' )

        # used to populate agg_{x,y,z}_exch, too.
        self.agg_exch=agg_exch # used to be self.agg_exch

        # with all the exchanges in place and ordered correctly, assign boundary segment
        # indices to boundary exchanges
        n_bdry_exch=0
        for exch in self.agg_exch:
            if exch['from']>=0:
                continue # skip internal
            assert(exch['from']==REINDEX) # sanity, make sure nothing is getting mixed up.
            n_bdry_exch+=1
            # these have to start from -1, going negative
            exch['from']=-n_bdry_exch
       
        self.log.info("Aggregated output will have" )
        self.log.info(" %5d segments"%(self.n_agg_segments))
        self.log.info(" %5d exchanges (%d,%d,%d)"%(len(self.agg_exch),
                                                   self.n_exch_x,self.n_exch_y,self.n_exch_z) )
        
        self.log.info(" %5d boundary exchanges"%n_bdry_exch)

        
    lookup=None
    def elt_to_elt_length(self,elt_a,elt_b):
        if self.lookup is None:
            self.lookup={}

        key=(elt_a,elt_b) 
        if key not in self.lookup:
            if elt_a<0:
                a_len=b_len=np.sqrt(self.elements['poly'][elt_b].area)/2.
            elif elt_b<0:
                a_len=b_len=np.sqrt(self.elements['poly'][elt_a].area)/2.
            else:
                apoly=self.elements['poly'][elt_a]
                bpoly=self.elements['poly'][elt_b]

                buff=1.0 # geos => sloppy intersection test
                iface=apoly.buffer(buff).intersection(bpoly.buffer(buff))
                iface_length=iface.area/(2*buff) # verified

                # rough scaling of distance from centers to interface - seems reasonable.
                a_len=apoly.area / iface_length / 2.
                b_len=bpoly.area / iface_length / 2.
            self.lookup[key]=(a_len,b_len)
        return self.lookup[key]

    # A faster add_exchange_data:
    def update_agg_to_local(self):
        """
        Generate mapping of aggregated exchanges to [(proc,local_exch,sgn),...]
        and aggregated segments to [(proc,local_seg),...]
        saved to self.agg_to_local_exch, agg_to_local_seg
        Uses self.exch_local_to_agg, and self.seg_local_to_agg
        """
        self.exch_agg_to_local=[None]*self.n_exch
        self.seg_agg_to_local=[None]*self.n_seg

        # hmm - this might be pulling in duplicate mappings where grids overlap
        for p in range(self.nprocs):
            for j_local in np.nonzero(self.exch_local['agg'][p,:]>=0)[0]:
                j_agg=self.exch_local['agg'][p,j_local]
                if self.exch_agg_to_local[j_agg] is None:
                    self.exch_agg_to_local[j_agg]=[]
                self.exch_agg_to_local[j_agg].append( (p,j_local,self.exch_local['sgn'][p,j_local]) )

            seg_sel=(self.seg_local['agg'][p,:]>=0) & (self.seg_local['ghost'][p,:]==0)
            for i_local in np.nonzero(seg_sel)[0]:
                i_agg=self.seg_local['agg'][p,i_local]
                if self.seg_agg_to_local[i_agg] is None:
                    self.seg_agg_to_local[i_agg]=[]
                self.seg_agg_to_local[i_agg].append( (p,i_local) )

    def add_exchange_data(self):
        """ Fill in extra details about exchanges, e.g. length scales
        """
        # Where possible (exactly one unaggregated exchange for an aggregated exchange)
        # pull existing length scale data
        # Signal missing:
        self.agg_exch['from_len']=np.nan 
        self.agg_exch['to_len']=np.nan
        count=np.zeros(len(self.agg_exch))

        for p in range(self.nprocs):
            hyd=self.open_hyd(p)
            lengths=hyd.exchange_lengths

            for j in np.nonzero(self.exch_local['agg'][p,:]>=0)[0]:
                agg_j=self.exch_local['agg'][p,j]

                agg_from=self.agg_exch['from'][agg_j]
                agg_to=self.agg_exch['to'][agg_j]

                if agg_from>=0:
                    n_local_from = len(self.seg_agg_to_local[agg_from])
                else:
                    # aggregated boundary
                    n_local_from = 1

                n_local_to = len(self.seg_agg_to_local[agg_to])
                if n_local_from>1:
                    # 2018-04-05: this caution was probably left over from a merge-only
                    # run.  This can and should happen on aggregation runs.
                    #print("What?")
                    #import pdb
                    #pdb.set_trace()
                    count[agg_j]+=999 # signal that we can't use unaggregated data here.
                    continue
                if n_local_to>1:
                    # print("Really?")
                    count[agg_j]+=999 # signal that we can't use unaggregated data here.
                    continue

                count[agg_j]+=1
                if self.exch_local['sgn'][p,j] > 0: # forward
                    self.agg_exch['from_len'][agg_j] = lengths[j,0]
                    self.agg_exch['to_len'][agg_j] =lengths[j,1]
                else: # reversed
                    self.agg_exch['from_len'][agg_j] = lengths[j,1]
                    self.agg_exch['to_len'][agg_j] =lengths[j,0]
        duped=count>1
        nduped=duped.sum()
        if nduped>0:
            # Not a problem, just means that we actually are aggregated
            self.log.info("%d aggregated exchanges come from multiple unaggregated exchanges"%nduped)
        self.agg_exch['from_len'][duped]=np.nan
        self.agg_exch['to_len'][duped]=np.nan

        self.duped_exchanges=duped

        for exch in self.agg_exch:
            if np.isfinite(exch['from_len']):
                continue # filled in from above
            if exch['direc']==b'x':
                flen,tlen=self.elt_to_elt_length(exch['from_2d'],exch['to_2d'])
                exch['from_len']=flen
                exch['to_len']  =tlen
            elif exch['direc']==b'z':
                # not entirely sure about this...
                exch['from_len']=0.5
                exch['to_len']=0.5

    def get_agg_segment(self,agg_k,agg_elt):
        """ used to be agg_linear_map.
        on-demand allocation/temporary indexing for segments.
        
        for aggregated segments, map [layer,element] indices to a linear
        index.  This will only be dense if all elements have the maximum
        number of segments - otherwise some sorting/subsetting afterwards is
        likely necessary.
        """
        # outer loop is vertical, fastest varying index is horizontal
        # map [agg_horizontal_cell,k_from_top] to [agg_element_id]
        # n_agg_layers is the *max* number of layers in an aggregated element

        #if not self.sparse_layers:
        # #old static approach:
        # return agg_elt+agg_k*self.n_agg_elements_2d

        # even with dense layers, use this so that the implementation doesn't
        # fracture too much
        key=(agg_k,agg_elt)
        if key not in self.agg_seg_hash:
            idx=len(self.agg_seg)
            seg=np.zeros( (), dtype=self.agg_seg_dtype)
            seg['k']=agg_k
            seg['elt']=agg_elt
            seg['active']=True # default, but caller can overwrite
            self.agg_seg.append(seg)
            self.agg_seg_hash[key]=idx
        else:
            idx=self.agg_seg_hash[key]
        return idx

    def get_agg_exchange(self,agg_from,agg_to,direc=None):
        """
        return a [temporary] linear index and sign for the requested exchange.
        if there is already an exchange going the opposite direction,
        returns index,-1 to indicate that the sign is reversed, otherwise,
        index,1.
        either agg_from or agg_to can be negative - the value will be replaced with
        REINDEX
        (had tried preserving...
        but does not affect lookups (i.e. f(-1,2) and f(-2,2) will return the 
        same exchange, which will reflect the value of agg_from in the first call).
        )

        direc is typically inferred based on whether agg_from and agg_to are in the
        same element (direc<-'z').

        2016-07-18: behavior is modified by self.agg_boundaries.  if False, then
        unique exchanges are returned for multiple calls with the same agg_from<0 and
        agg_to.
        """
        def create_exch(a_from,a_to):
            """ 
            given the aggregated from/to segments, fill in some other useful
            exchange fields, notably from_2d/to_2d, direc, k, and set lengths to nan
            """
            exch=np.zeros( (), dtype=self.agg_exch_dtype)
            k=None
            if a_from>=0:
                exch['from_2d']=self.agg_seg[a_from]['elt']
                k=self.agg_seg[a_from]['k']
            else:
                exch['from_2d']=-1

            if a_to>=0:
                exch['to_2d']=self.agg_seg[a_to]['elt']
                if k is None:
                    k=self.agg_seg[a_to]['k']
            else:
                exch['to_2d']=-1

            if exch['from_2d']==exch['to_2d']:
                exch['direc']  = b'z'
            else:
                exch['direc'] = b'x' 

            exch['from']=a_from
            exch['to']=a_to
            exch['k']=k

            # filled in later:
            exch['from_len']=np.nan
            exch['to_len']=np.nan

            self.agg_exch.append(exch)
            idx=len(self.agg_exch)-1
            return idx
            
        # special handling for agg_from<0, indicating a boundary exchange
        # indexed only by the internal, agg_to index.
        if agg_from<0:
            agg_from=REINDEX
            if self.agg_boundaries:
                if agg_to not in self.agg_exch_hash:
                    idx=create_exch(agg_from,agg_to)
                    self.agg_exch_hash[agg_to]=idx
                else:
                    idx=self.agg_exch_hash[agg_to]
            else:
                # when aggregating these, can just index it by agg_to.
                # it's important here that there aren't extraneous calls
                # to get_agg_exchange or reg_exchange.

                # so we always create an exchange -
                idx=create_exch(agg_from,agg_to)

                # and scan for increasingly negative numbers for it's
                # hash.  HERE: what are the expectations on self.agg_exch_hash
                # outside of this function?

                # when *not* aggregating, index non-aggregated boundary exchanges
                # by a negative count - which starts -10000 as a bit of a hint
                # when things go south
                count=-10000
                while (count,agg_to) in self.agg_exch_hash:
                    count-=1
                self.agg_exch_hash[(count,agg_to)]=idx
            sgn=1
        elif agg_to<0: # does this happen?
            # It can happen if the aggregation polygons don't cover the entire input grid 
            self.log.warning("get_exchange with a negative/boundary for the *to* segment - likely incomplete coverage")
            assert self.agg_boundaries # if this is a problem, port the above stanza to here.
            agg_to=REINDEX
            if agg_from not in self.agg_exch_hash:
                idx=create_exch(agg_to,agg_from)
                self.agg_exch_hash[agg_from]=idx
            sgn=-1
            idx=self.agg_exch_hash[agg_from]
        elif (agg_to,agg_from) in self.agg_exch_hash:
            # regular exchange, but reversed from how we already have it.
            sgn=-1
            idx=self.agg_exch_hash[ (agg_to,agg_from) ]
        else:
            if (agg_from,agg_to) not in self.agg_exch_hash:
                # create new exchange
                idx=create_exch(agg_from,agg_to)
                self.agg_exch_hash[(agg_from,agg_to)]=idx
            else:
                # this exact exchange already exists
                idx=self.agg_exch_hash[ (agg_from,agg_to) ]
            sgn=1

        return idx,sgn

    _pointers=None
    @property
    def pointers(self):
        if self._pointers is None:
            pointers=np.zeros( (len(self.agg_exch),4),'i4')
            bc=(self.agg_exch['from']<0)
            pointers[:,0]=self.agg_exch['from'] 
            pointers[~bc,0] += 1 # internal exchanges should use 1-based index
            # pointers[bc,0] - reindex() already has the right numbering for these
            pointers[:,1]=self.agg_exch['to']   + 1
            pointers[:,2:]=self.CLOSED # no support for higher order advection
            self._pointers=pointers

        return self._pointers

    @property
    def exchange_lengths(self):
        exchange_lengths=np.zeros( (len(self.agg_exch),2),'f4')
        exchange_lengths[:,0]=self.agg_exch['from_len']
        exchange_lengths[:,1]=self.agg_exch['to_len']
        return exchange_lengths

    @property
    def time0(self):
        return self.open_hyd(0).time0

    @property
    def t_secs(self):
        return self.open_hyd(0).t_secs

    def init_seg_matrices(self):
        """ initialize dict:
        self.seg_matrix
        which maps processor ids to sparse matrices, which can be
        left multiplied with a per-processor vector:
          E.dot(local_seg_value) => agg_seg_value
        as a sum of local_seg_value
        """
        self.seg_matrix={}

        for p in range(self.nprocs):
            # be sure to limit sel to segments local to p (non-ghost)
            hyd=self.open_hyd(p)
            
            # new way, using pre-determined ghost-ness of segments
            sel=(self.seg_local['agg'][p,:]>=0) & (self.seg_local['ghost'][p,:]==0)

            rows=self.seg_local['agg'][p,sel]
            cols=np.nonzero(sel)[0]
            vals=np.ones_like(cols)

            S=sparse.coo_matrix( (vals, (rows,cols)),
                                 (self.n_seg,hyd.n_seg),dtype='f4')
            self.seg_matrix[p]=S.tocsr()
            
    def init_exch_matrices(self):
        """ initialize dicts:
        self.flow_matrix, self.area_matrix
        which maps processor ids to sparse matrices, which can be
        left multiplied with a per-processor vector:
          E.dot(local_flow) => agg_flow
        or 
          E.dot(local_area) => agg_area
        The difference between flow and area being that flow is signed
        while area is unsigned.
        """
        self.flow_matrix={}
        self.area_matrix={}

        for p in range(self.nprocs):
            hyd=self.open_hyd(p)
            n_exch_local=hyd['number-horizontal-exchanges']+hyd['number-vertical-exchanges']

            exch_local_to_agg=self.exch_local['agg'][p,:]
            idxs=np.nonzero( exch_local_to_agg>=0 )[0]

            if len(idxs)==0:
                continue
            assert( idxs.max() < n_exch_local )
            rows=exch_local_to_agg[idxs]
            cols=idxs
            values=self.exch_local['sgn'][p,idxs]

            Eflow=sparse.coo_matrix( (values, (rows,cols)),
                                     (self.n_exch,n_exch_local),dtype='f4')
            self.flow_matrix[p]=Eflow.tocsr()

            # areas always sum
            Earea=sparse.coo_matrix( (np.abs(values), (rows,cols)),
                                     (self.n_exch,n_exch_local),dtype='f4')
            self.area_matrix[p]=Earea.tocsr()

    def init_boundary_matrices(self):
        """ populates:
          self.bc_local_segs={} # bc_segs[proc] => [0-based seg indices] 
          self.bc_local_exchs={}
          self.bc_exch_local_to_agg={}

        local_seg_scalar[bc_local_segs[p]] gives the subset of segment
        concentrations on proc p which appear at aggregated boundaries.
        
        local_exch_area[bc_local_exchs[p]] gives the corresponding area
        of exchanges which span aggregated boundaries.
        
        bc_exch_local_to_agg is a matrix, E.dot(bc_values) aggregates local
        values from above to aggregated boundaries.
        """

        # E.dot(seg_values) => sum of segment values adjacent to aggregated boundary
        # not quite right for BCs, since we really want to be weighing by exchange
        # area.

        self.bc_seg_matrix=bc_seg_matrix={} 

        self.bc_local_segs={} # bc_segs[proc] => [0-based seg indices] 
        self.bc_local_exchs={}
        self.bc_exch_local_to_agg={}

        warned_internal_boundary_seg=False # control one-off warning below

        for p in range(self.nprocs): # following exch and area matrix code
            # set defaults, so if no exchanges map to this processor, just
            # bail on the loop
            self.bc_seg_matrix[p]=None
            self.bc_local_segs[p]=None
            self.bc_local_exchs[p]=None
            self.bc_exch_local_to_agg[p]=None

            hyd=self.open_hyd(p)
            n_seg_local=hyd.n_seg

            exch_local_to_agg=self.exch_local['agg'][p,:] # 0-based
            idxs=np.nonzero( exch_local_to_agg>=0 )[0]

            if len(idxs)==0: 
                continue

            local_bc_segs=[] # indices
            local_bc_exchs=[] # indices into local exchanges
            # row is aggregated bc exch, col is local_bc_exch

            # count number of local bc exchanges a priori
            n_local_bc_exch = np.sum(  self.pointers[exch_local_to_agg[idxs],0]<0 )
            #                                                         ^ narrow to boundary exchs
            #                                        ^ index of agg. exch. for each local exch
            #                         # ^ get the aggregated 'from' segment 
            #                                                                  ^ test for it being an agg bc

            local_bc_to_agg=sparse.dok_matrix( (self.n_boundaries,n_local_bc_exch) )

            rows=[] # index of aggregated boundary exchanges - 0-based
            cols=[] # index of local segment - 0-based
            vals=[] # just 1 or 0

            # indices of aggregated exchanges which are part of the boundary
            agg_bc_exchanges=np.nonzero(self.pointers[:,0]<0)[0] # 0-based

            # the ordering in pointers (-1,-2,-3,...) should match the ordering
            # of boundaries in the input file
            local_pointers=hyd.pointers
            for j,agg_exch_idx in enumerate(agg_bc_exchanges):
                # j: 0-based index of aggregated boundary exchanges, i.e. BCs.
                # agg_exch_idx: 0-based index of that exchange

                # find local exchanges which map to this aggregated exchange:
                local_exch_match_idxs=np.nonzero( exch_local_to_agg==agg_exch_idx )[0]

                for local_exch_idx in local_exch_match_idxs:
                    seg1,seg2=local_pointers[local_exch_idx,:2] # 1-based!
                    assert(seg1!=0) # paranoid
                    assert(seg2!=0) # paranoid

                    # if one is negative, it's a boundary and it's either on a different
                    # processor or was a boundary in the unaggregated domain

                    # we want a way to find out the aggregated boundary condition,
                    # but if one segment is negative, we don't have the data, and
                    # have to settle for grabbing data from the internal segment
                    # as being representative of the boundary condition.

                    # in cases where the exchange represents an unaggregated boundary
                    # exchange, then there is the opportunity to assign boundary conditions
                    # based on original forcing data.  not sure if this really ought to
                    # be a warning

                    if seg1<0:
                        if not warned_internal_boundary_seg:
                            self.log.info("had to choose internal segment for agg boundary")
                            warned_internal_boundary_seg=True
                        local_seg=seg2
                    elif seg2<0:
                        if not warned_internal_boundary_seg:
                            self.log.info("had to choose internal segment for agg boundary")
                            warned_internal_boundary_seg=True
                        local_seg=seg1
                    elif self.seg_local['agg'][p,seg1-1]>=0:
                        # try to get the segment which is exterior to the aggregated segment
                        assert self.seg_local['agg'][p,seg2-1]<0
                        local_seg=seg2
                    elif self.seg_local['agg'][p,seg2-1]>=0:
                        assert self.seg_local['agg'][p,seg1-1]<0
                        local_seg=seg1
                    else:
                        self.log.error("Boundary exchange had local exch where neither local segment is internal" )
                        assert False

                    # record these for the somewhat defunct bc_seg_matrix
                    rows.append(j) # which aggregated boundary exchanges is involved
                    cols.append(local_seg-1)
                    vals.append(1)

                    # record this for the more complex but correct local_bc_exch 
                    local_bc_segs.append(local_seg)
                    local_bc_exchs.append(local_exch_idx)
                    local_bc_to_agg[j,len(local_bc_exchs)-1]=1

            if len(rows):
                Eboundary=sparse.coo_matrix( (vals, (rows,cols)),
                                             (self.n_boundaries,n_seg_local),dtype='f4')
                self.bc_seg_matrix[p]=Eboundary.tocsr()

                self.bc_local_segs[p]       =np.array(local_bc_segs)
                self.bc_local_exchs[p]      =np.array(local_bc_exchs)
                self.bc_exch_local_to_agg[p]=local_bc_to_agg.tocsr()

                assert(n_local_bc_exch)
            else:
                if n_local_bc_exch:
                    print( "WARNING: a priori test showed that there should be local bc exchanges!")
                    print( "  Processor: %d"%p)

    def boundary_values(self,t_sec,label):
        """ Aggregate boundary condition data - segment data are pulled
        from each processor by reading the given label from the hyd files.
        """
        # follow logic similar to aggregated areas calculation
        areas=np.zeros(self.n_boundaries,'f4')
        aC_products=np.zeros(self.n_boundaries,'f4')

        for p,Eboundary in iteritems(self.bc_exch_local_to_agg):
            if Eboundary is None:
                continue
            hyd=self.open_hyd(p)
            p_bc_area=hyd.areas(t_sec)[self.bc_local_exchs[p]]
            p_bc_conc=hyd.seg_func(t_sec,label=label)[self.bc_local_segs[p]]

            areas       += Eboundary.dot(p_bc_area)
            aC_products += Eboundary.dot(p_bc_area*p_bc_conc)
        sel=(areas!=0.0)
        aC_products[sel] = aC_products[sel]/areas[sel]
        aC_products[~sel] = -999
        return aC_products

    def volumes(self,t_sec,explicit=False):
        if not explicit: # much faster on dense outputs
            return self.segment_aggregator(t_sec=t_sec,
                                           seg_fn=lambda _: 1.0,
                                           normalize=False)
        else:
            # original, explicit version.  Slow!
            # retained for debugging, eventually remove.
            agg_vols=np.zeros(self.n_agg_segments,'f4')

            # loop on processor:
            for p in range(self.nprocs):
                hydp=self.open_hyd(p)

                vols=None

                # inner loop on target segment
                # this is the slow part!
                for agg_seg in range(self.n_agg_segments):
                    sel_3d=self.seg_local['agg'][p,:]==agg_seg
                    if np.any(sel_3d):
                        if vols is None:
                            vols=hydp.volumes(t_sec)
                        # print "Found %d 3D segments which map to aggregated segment"%np.sum(sel_3d)
                        # sel_3d is "ragged", but stored rectangular.  trim it to avoid numpy 
                        # warnings
                        sel_3d=sel_3d[:len(vols)]
                        agg_vols[agg_seg]+= np.sum(vols[sel_3d])
            return agg_vols

    # delwaq doesn't tolerate any zero area exchanges - at least I think not.
    # a little unsure of how zero areas in the unfiltered data might have
    # worked.  
    exch_area_min=1.0
    exch_z_area_constant=True # if true, force all segment in a column to have same plan area.

    warned_forcing_constant_area=False
    def areas(self,t):
        areas=np.zeros(self.n_exch,'f4')
        for p,Earea in iteritems(self.area_matrix):
            hyd=self.open_hyd(p)
            p_area=hyd.areas(t)
            areas += Earea.dot(p_area)
        # try re-introducing this line... had coincided with this setup breaking
        # okay - that ran okay..  but it ran okay without this line, so
        # maybe nix it?
        # areas[areas<self.exch_area_min]=self.exch_area_min
        # trying to reintroduce this line... seemed okay

        # no longer trying to do wacky things with area, maybe okay to drop this
        # self.monotonicize_areas(areas)

        # fast forward to 2016-07-22: pretty sure we need the planform area to be
        # constant through the water column.  I thought that was already in place, but
        # the output shows that's not the case.
        # if we are only merging, then assume that the data coming in already
        # has constant areas in the vertical (i.e. it's original hydro cells which
        # don't have any partial areas
        if self.exch_z_area_constant:
            if not self.warned_forcing_constant_area:
                self.warned_forcing_constant_area=True
                self.log.warning('Forcing constant area within water column')
            self.monotonicize_areas(areas)
            self.monotonicize_areas(areas,top_down=True)
        return areas

    def monotonicize_areas(self,areas,top_down=False):
        """ areas: n_exch * 'f4'
        Modify areas so that vertical exchange areas are monotonically 
        decreasing.
        by default, this means starting at the bottom of the water column
        and make sure that areas are non-decreasing as we go up.  but it can
        also be called with top_down=True, to do the opposite.  This is mostly
        just useful to make the area constant in the entire water column
        """
        # self.log.info("Call to monotonicize areas!")
        # this looks very slow.
        seg_A=np.zeros(self.n_seg)
        pointers=self.pointers
        js=np.arange(self.n_exch-self.n_exch_z,self.n_exch)
        if not top_down:
            js=js[::-1]
        for j in js:
            top,bot = pointers[j,:2] - 1
            if not top_down:
                if bot>=0: # update exchange from segment below
                    areas[j]=max(areas[j],seg_A[bot])
                if top>=0: # update segment above
                    seg_A[top]=max(seg_A[top],areas[j])
            else:
                if top>=0: # update exchange from the segment above
                    areas[j]=max(areas[j],seg_A[top])
                if bot>=0: # update segment below
                    seg_A[bot]=max(seg_A[bot],areas[j])
    
    def flows(self,t):
        """ 
        returns flow rates ~ np.zeros(self.n_exch,'f4'), for given timestep.
        flows in m3/s.
        """
        flows=np.zeros(self.n_exch,'f4')
        for p,Eflow in iteritems(self.flow_matrix):
            hyd=self.open_hyd(p)
            p_flow=hyd.flows(t)
            flows += Eflow.dot(p_flow)
        return flows

    def segment_aggregator(self,t_sec,seg_fn,normalize=True,min_volume=0.00001,
                           nan_method='pass'):
        """ 
        Generic segment scalar aggregation
        t_sec: simulation time, integer seconds
        seg_fn: lambda proc => scalar for each unaggregated segment.  If this is a single
        processor run, it's okay to pass the array directly.

        normalize: if True, divide by aggregated volume, otherwise just sum
        min_volume: if normalizing by volume, this volume is added, so that zero-volume
        inputs with valid scalars will produce valid output.  Note that this included 
        for all unaggregated segments - in cases where there are large numbers of 
        empty segments aggregated with a few small non-empty segments, then there will
        be some error.  but min_volume can be very small

        nan_method: 'pass' does no special handling of nan in scalar. 'ignore' will
          zero out the scalar and volume for segments with a nan scalar.
        """
        # volume-weighted averaging
        agg_scalars=np.zeros(self.n_seg,'f4')
        agg_volumes=np.zeros(self.n_seg,'f4')

        # loop on processor:
        for p in range(self.nprocs):
            if np.all(self.seg_local['agg'][p,:]<0):
                continue

            hydp=self.open_hyd(p)
            vols=hydp.volumes(t_sec)
            if min_volume>0:
                vols=vols.clip(min_volume,np.inf)

            if self.nprocs==1 and not callable(seg_fn):
                scals=seg_fn
            else:
                scals=seg_fn(p)
            scals=scals * np.ones_like(vols) # mult in case seg_fn returns a scalar

            # inner loop on target segment
            S=self.seg_matrix[p]
            if nan_method=='ignore':
                invalid=np.isnan(scals)
                vols=np.where( invalid, 0, vols)
                scals=np.where( invalid, 0, scals)
            agg_scalars += S.dot( vols*scals )
            agg_volumes += S.dot( vols )

        if normalize:
            valid=agg_volumes>0
            agg_scalars[valid] /= agg_volumes[valid]
            agg_scalars[~valid] = 0.0
        return agg_scalars

    def seg_func(self,t_sec=None,label=None,param_name=None):
        """ return a callable which implements a segment function using data
        from unaggregated files, either with the given label mentioned in the
        hyd file (i.e. label='salinity-file'), or by grabbing a parameter 
        of a given name.

        if t_sec is given, evaluate at that time and return the result
        """
        def f_label(t,label=label):
            return self.segment_aggregator(t,
                                           lambda proc: self.open_hyd(proc).seg_func(t,label=label),
                                           normalize=True)
        def f_param(t,param_name=param_name):
            per_proc=lambda proc: self.open_hyd(proc).parameters(force=False)[param_name].evaluate(t=t).data
            return self.segment_aggregator(t,per_proc,normalize=True)
        if param_name:
            f=f_param
        elif label:
            f=f_label
        else:
            raise Exception("One of label or param_name must be supplied")
            
        if t_sec is not None:
            return f(t_sec)
        else:
            return f

    def vert_diffs(self,t_sec):
        # returns [n_segs]*'f4' vertical diffusivities in m2/s
        # based on the output from Rose, the top layer is missing
        # vertical diffusivity entirely.
        diffs=self.segment_aggregator(t_sec,
                                      lambda proc: self.open_hyd(proc).vert_diffs(t_sec),
                                      normalize=True)
        # kludge - get a nonzero diffusivity in the top layer
        n2d=self.n_2d_elements
        if len(diffs)>n2d:
            diffs[:n2d] = diffs[n2d:2*n2d]
        else:
            pass # probably a 2D run.
        return diffs

    def planform_areas(self):
        """ 
        Return a Parameter object encapsulating variability of planform 
        area.  Typically this is a per-segment, constant-in-time 
        parameter, but it could be globally constant or spatially and 
        temporally variable.
        Old interface returned Nsegs * 'f4', which can be recovered 
        in simple cases by accessing the .data attribute of the
        returned parameter

        HERE: when bringing in z-layer data, this needs some extra 
        attention.  In particular, planform area needs to be (i) the same 
        for all layers, and (ii) should be chosen to preserve average depth,
        presumably average depth of the wet part of the domain?
        """
        # prior to 4/27/16 this was set to lazy.  but with 1:1 mapping that
        # was leading to bad continuity results.  overall, seems like we should
        # stick with constant.
        # mode='lazy'
        mode='constant'
        # mode='explicit'

        min_planform_area=1.0

        if mode=='constant':
            #switching to the code below coincided with this setup breaking
            # This is the old code - just maps maximum area from the grid
            map2d3d=self.infer_2d_elements() # agg_seg_to_agg_elt_2d()
            data=(self.elements['plan_area'][map2d3d]).astype('f4')
            return ParameterSpatial(data,hydro=self)
        else: # new code, copied from FilteredBC
            # pull areas from exchange areas of vertical exchanges

            seg_z_exch=self.seg_to_exch_z(preference='upper')

            missing= (seg_z_exch<0)
            if np.any(missing):
                self.log.warning("Some segments have no vertical exchanges - will revert to element area")
                map2d3d=self.infer_2d_elements() 
                constant_areas=(self.elements['plan_area'][map2d3d]).astype('f4')
            else:
                constant_areas=None

            if mode=='lazy':
                def planform_area_func(t_sec):
                    A=np.zeros(self.n_seg,'f4')
                    areas=self.areas(t_sec)
                    if constant_areas is None:
                        A[:]=areas[seg_z_exch]
                    else:
                        A[~missing]=areas[seg_z_exch[~missing]]
                        A[missing] =constant_areas[seg_z_exch[missing]]
                    A[ A<min_planform_area ] = min_planform_area
                    return A
                return ParameterSpatioTemporal(func_t=planform_area_func,
                                               times=self.t_secs,hydro=self)

            else: # 'explicit'
                # then pull exchange area for each time step
                A=np.zeros( (len(self.t_secs),self.n_seg) )
                for ti,t_sec in enumerate(self.t_secs):
                    areas=self.areas(t_sec)
                    A[ti,~missing] = areas[seg_z_exch[~missing]]
                    A[ti,missing]=constant_areas[missing]

                # without this, but with zero area exchanges, and monotonicize
                # enabled, it was crashing, complaining that DDEPTH ran into
                # zero SURF.
                # enabling this lets it run, though depths are pretty wacky.
                A[ A<min_planform_area ] = min_planform_area

                return ParameterSpatioTemporal(times=self.t_secs,values=A,hydro=self)

    def depths(self):
        """ Temporarily copied from FilteredBC
        Compute time-varying segment thicknesses.  With z-levels, this is
        a little more nuanced than the standard calc. in delwaq.

        It uses a combination of planform area and vertical exchange area
        to get depths.  
        """
        mode='lazy'
        # mode='explicit'

        min_depth=0.001

        # use upper, since some bed segment would have a zero area for the
        # lower exchange
        self.log.debug("Call to WaqAggregator::depth()")

        # used to duplicate some of the code in planform_areas, grabbing
        # exchange areas and mapping the vertical exchanges to segments
        # should be fine to delegate that
        plan_areas=self.planform_areas()

        #seg_z_exch=self.seg_to_exch_z(preference='upper')
        #assert np.all(seg_z_exch>=0) # could be a problem if an element has 1 layer
        def clean_depth(data):
            """ fix up nan and zero depth values in place.
            """
            sel=(~np.isfinite(data))
            if np.any(sel):
                self.log.warning("Depths: %d steps*segments with invalid depth"%( np.sum(sel) ))
            data[sel]=0

            # seems that this is necessary.  ran okay with 0.01m
            data[ data<min_depth ] = min_depth

        if mode=='lazy':
            def depth_func(t_sec):
                # a little unsure on the .data part
                D=self.volumes(t_sec) / plan_areas.evaluate(t=t_sec).data
                clean_depth(D)
                return D.astype('f4')
            return ParameterSpatioTemporal(times=self.t_secs,func_t=depth_func,hydro=self)
                
        if mode=='explicit':
            D=np.zeros( (len(self.t_secs),self.n_seg) )
            for ti,t_sec in enumerate(self.t_secs):
                areas=self.areas(t_sec)
                volumes=self.volumes(t_sec)
                D[ti,:] = volumes / plan_areas.evaluate(t=t_sec).data

            clean_depth(D)
            return ParameterSpatioTemporal(times=self.t_secs,values=D,hydro=self)
        assert False

    def bottom_depths(self):
        """ 
        Like planform_areas, but for bottom depth.
        old interface: return Nsegs * 'f4' 
        """
        map2d3d=self.infer_2d_elements() # agg_seg_to_agg_elt_2d()
        data=(self.elements['zcc'][map2d3d]).astype('f4')
        return ParameterSpatial(data,hydro=self)

    def check_boundary_assumptions(self):
        """
        checks that boundary segments and exchanges obey some assumed
        invariants:
         - the boundary segment always appears first in exchanges
         - pointers show only horizontal exchanges having boundary segments
         - flowgeom shows that for boundary exchanges, the local/nonlocal status
           of the internal segment determines the local/nonlocal status of the
           exchange.
        """
        # This may need some updating with FlowLinkSS. 
        for p in range(self.nprocs):
            #print "------",p,"------"
            nc=self.open_flowgeom_ds(p)
            hyd=self.open_hyd(p)
            poi=hyd.pointers
            n_layers=hyd['number-water-quality-layers']

            for ab in [0,1]:
                poi_bc_segs=poi[:,ab]<0
                idxs=np.nonzero(poi_bc_segs)[0]
                assert( np.all(idxs<hyd['number-horizontal-exchanges']) )
                idxs=idxs[:(len(idxs)/n_layers)]
                #print "poi[%d]: "%ab
                #print np.array( [idxs,poi[idxs,ab]] ).T
                if ab==1:
                    assert(len(idxs)==0)

            link=nc.FlowLink.values # Should this be FlowLinkSS?
            link_domain=nc.FlowLinkDomain.values
            elem_domain=nc.FlowElemDomain.values
            nelems=len(elem_domain)

            for ab in [0,1]:
                nc_bc_segs=link[:,ab]>nelems
                idxs=np.nonzero(nc_bc_segs)[0]

                other_elem_is_local=elem_domain[link[idxs,1-ab]-1]==p

                link_is_local=link_domain[idxs]==p

                if ab==1:
                    assert(len(idxs)==0)

                # print " nc[%d]:"%ab
                # print np.array( [idxs,
                #                  link[idxs,ab],
                #                  other_elem_is_local,
                #                  link_is_local] ).T
                assert( np.all(other_elem_is_local==link_is_local) )

    def ghost_segs(self,p):
        hyd=self.open_hyd(p)
        nc=self.open_flowgeom(p)
        sel_2d=(nc.FlowElemDomain[:]!=p)
        return np.tile(sel_2d,self.n_src_layers)

    # def agg_seg_to_agg_elt_2d(self):
    #     """ Array of indices mapping 0-based aggregated segments
    #     to 0-based aggregated 2D segments. 
    #     """
    #     self.log.warning('agg_seg_to_agg_elt_2d is deprecated in favor of infer_2d_elements')
    #     # old implementation which assumed constant segments/layer
    #     # return np.tile(np.arange(len(self.elements)),self.n_agg_layers)
    #     return self.infer_2d_elements()

    def add_parameters(self,hparams):
        hparams=super(DwaqAggregator,self).add_parameters(hparams)
        hyd0=self.open_hyd(0)

        # new approach - use unaggregated parameter objects.
        hyd0_params=hyd0.parameters(force=False)

        for pname in hyd0_params:
            if pname.lower() in ['surf','bottomdept']:
                self.log.info("Original hydro has parameter %s, but it will not be copied to aggregated"%pname)
                continue
            
            hyd0_param=hyd0_params[pname]
            # in the past, used the label to grab this from each unaggregated
            # source.
            # now we use the parameter name
            hparams[pname]=ParameterSpatioTemporal(func_t=self.seg_func(param_name=pname),
                                                   times=hyd0_param.times,
                                                   hydro=self)

        return hparams

    def write_hyd(self,fn=None):
        """ Write an approximation to the hyd file output by D-Flow FM
        for consumption by delwaq or HydroFiles
        respects scen_t_secs
        """
        # currently the segment names here are out of sync with 
        # the names used by write_parameters.
        #  this is relevant for salinity-file,  vert-diffusion-file
        #  maybe surfaces-file, depths-file.
        # for example, surfaces file is written as tbd-SURF.seg
        # but below we call it com-tbd.srf
        # maybe easiest to just change the code below since it's
        # already arbitrary
        fn=fn or os.path.join( self.scenario.base_path,
                               self.fn_base+".hyd")

        name=self.scenario.name

        dfmt="%Y%m%d%H%M%S"
        time_start = (self.time0+self.scen_t_secs[0]*self.scenario.scu)
        time_stop  = (self.time0+self.scen_t_secs[-1]*self.scenario.scu)
        timedelta = (self.t_secs[1] - self.t_secs[0])*self.scenario.scu
        timestep = timedelta_to_waq_timestep(timedelta)

        # some values just copied from the first subdomain
        # new code uses max across all domains
        # No support yet for differing source and aggregated layers here.
        n_layers=self.n_agg_layers

        # New code - maybe not right at all - same as Hydro.write_hyd
        if 'temp' in self.parameters():
            temp_file="'%s-temp.seg'"%name
        else:
            temp_file='none'

        if 'tau' in self.parameters():
            tau_file="'%s-tau.seg'"%name
        else:
            tau_file='none'

        lines=[
            "file-created-by  SFEI, waq_scenario.py",
            "file-creation-date  %s"%( datetime.datetime.utcnow().strftime('%H:%M:%S, %d-%m-%Y') ),
            "task      full-coupling",
            "geometry  unstructured",
            "horizontal-aggregation no",
            "reference-time           '%s'"%( self.time0.strftime(dfmt) ),
            "hydrodynamic-start-time  '%s'"%( time_start.strftime(dfmt) ),
            "hydrodynamic-stop-time   '%s'"%( time_stop.strftime(dfmt)  ),
            "hydrodynamic-timestep    '%s'"%timestep, 
            "conversion-ref-time      '%s'"%( self.time0.strftime(dfmt) ),
            "conversion-start-time    '%s'"%( time_start.strftime(dfmt) ),
            "conversion-stop-time     '%s'"%( time_stop.strftime(dfmt)  ),
            "conversion-timestep      '%s'"%timestep, 
            "grid-cells-first-direction       %d"%self.n_2d_elements,
            "grid-cells-second-direction          0",
            "number-hydrodynamic-layers          %s"%( n_layers ),
            "number-horizontal-exchanges      %d"%( self.n_exch_x ),
            "number-vertical-exchanges        %d"%( self.n_exch_z ),
            # little white lie.  this is the number in the top layer.
            # and no support for water-quality being different than hydrodynamic
            "number-water-quality-segments-per-layer       %d"%( self.n_2d_elements),
            "number-water-quality-layers          %s"%( n_layers ),
            "hydrodynamic-file        '%s'"%self.fn_base,
            "aggregation-file         none",
            "boundaries-file          '%s.bnd'"%self.fn_base,
            # filename handling not as elegant as it could be..
            # e.g. self.vol_filename should probably be self.vol_filepath, then
            # here we could reference the filename relative to the hyd file
            "grid-indices-file     '%s.bnd'"%self.fn_base,# lies, damn lies
            "grid-coordinates-file '%s'"%os.path.basename(self.flowgeom_filename), # hyd files are always relative...
            "attributes-file       '%s.atr'"%self.fn_base,
            "volumes-file          '%s.vol'"%self.fn_base,
            "areas-file            '%s.are'"%self.fn_base,
            "flows-file            '%s.flo'"%self.fn_base,
            "pointers-file         '%s.poi'"%self.fn_base,
            "lengths-file          '%s.len'"%self.fn_base,
            "salinity-file         '%s-salinity.seg'"%name,
            "temperature-file      %s"%temp_file,
            "vert-diffusion-file   '%s-vertdisper.seg'"%name,
            # not a segment function!
            "surfaces-file         '%s'"%self.surf_filename,
            "shear-stresses-file   %s"%tau_file,
            "hydrodynamic-layers",
            "\n".join( ["%.5f"%(1./n_layers)] * n_layers ),
            "end-hydrodynamic-layers",
            "water-quality-layers   ",
            "\n".join( ["1.000"] * n_layers ),
            "end-water-quality-layers"]
        txt="\n".join(lines)
        with open(fn,'wt') as fp:
            fp.write(txt)

    @property
    def surf_filename(self):
        return self.fn_base+".srf"
    
    def write_srf(self):
        surfaces=self.elements['plan_area']
        # this needs to be in sync with what write_hyd writes, and
        # the supporting_file statement in the hydro_parameters
        fn=os.path.join(self.scenario.base_path,self.surf_filename)

        nelt=self.n_2d_elements
        with open(fn,'wb') as fp:
            # shape, shape, count, x,x,x according to waqfil.m
            hdr=np.zeros(6,'i4')
            hdr[0]=hdr[2]=hdr[3]=hdr[4]=nelt
            hdr[1]=1
            hdr[5]=0
            fp.write(hdr.tobytes())
            fp.write(surfaces.astype('f4'))

    def get_geom(self):
        ds=xr.Dataset()

        xycc = np.array( [poly.centroid.coords[0] for poly in self.elements['poly']] )

        ds['FlowElem_xcc']=xr.DataArray(xycc[:,0],dims=['nFlowElem'],
                                        attrs=dict(units='m',
                                                   standard_name = "projection_x_coordinate",
                                                   long_name = "Flow element centroid x",
                                                   bounds = "FlowElemContour_x",
                                                   grid_mapping = "projected_coordinate_system"))
        ds['FlowElem_ycc']=xr.DataArray(xycc[:,1],dims=['nFlowElem'],
                                  attrs=dict(units='m',
                                             standard_name = "projection_y_coordinate",
                                             long_name = "Flow element centroid y",
                                             bounds = "FlowElemContour_y",
                                             grid_mapping = "projected_coordinate_system"))

        ds['FlowElem_zcc']=xr.DataArray(self.elements['zcc'],dims=['nFlowElem'],
                                        attrs=dict(long_name = ("Flow element average"
                                                                " bottom level (average of all corners)"),
                                                   positive = "down",
                                                   mesh = "FlowMesh",
                                                   location = "face"))

        ds['FlowElem_bac']=xr.DataArray(self.elements['plan_area'],
                                        dims=['nFlowElem'],
                                        attrs=dict(long_name = "Flow element area",
                                                   units = "m2",
                                                   standard_name = "cell_area",
                                                   mesh = "FlowMesh",
                                        location = "face" ) )

        # make a ragged list first
        # but shapely repeats the first point, so shave that off
        poly_points = [np.array(p.exterior.coords)[:-1]
                       for p in self.elements['poly']]
        # also shapely may give the order CW
        for i in range(len(poly_points)):
            if utils.signed_area(poly_points[i])<0:
                poly_points[i] = poly_points[i][::-1]

        max_points=np.max([len(pnts) for pnts in poly_points])

        packed=np.zeros( (len(poly_points),max_points,2), 'f8')
        packed[:]=np.nan
        for pi,poly in enumerate(poly_points):
            packed[pi,:len(poly),:] = poly

        ds['FlowElemContour_x']=xr.DataArray(packed[...,0],
                                             dims=['nFlowElem','nFlowElemContourPts'],
                                             attrs=dict(units = "m",
                                                        standard_name = "projection_x_coordinate" ,
                                                        long_name = "List of x-points forming flow element" ,
                                                        grid_mapping = "projected_coordinate_system"))
        ds['FlowElemContour_y']=xr.DataArray(packed[...,1],
                                             dims=['nFlowElem','nFlowElemContourPts'],
                                             attrs=dict(units="m",
                                                        standard_name="projection_y_coordinate",
                                                        long_name="List of y-points forming flow element",
                                                        grid_mapping="projected_coordinate_system"))

        ds['FlowElem_bl']=xr.DataArray(-self.elements['zcc'],dims=['nFlowElem'],
                                       attrs=dict(units="m",
                                                  positive = "up" ,
                                                  standard_name = "sea_floor_depth" ,
                                                  long_name = "Bottom level at flow element\'s circumcenter." ,
                                                  grid_mapping = "projected_coordinate_system" ,
                                                  mesh = "FlowMesh",
                                                  location = "face"))

        sel = (self.agg_exch['direc']==b'x') & (self.agg_exch['k']==0)

        # use the seg from, not from_2d, because they have the real
        # numbering for the boundary exchanges (from_2d just has -1)
        links=np.array( [ self.agg_exch['from'][sel],
                          self.agg_exch['to'][sel] ] ).T
        bc=(links<0)
        self.infer_2d_elements()
        links[bc]=self.n_2d_elements - links[bc] - 1
        bclinks=np.any(bc,axis=1)

        # 1-based
        ds['FlowLink']=xr.DataArray(links+1,dims=['nFlowLink','nFlowLinkPts'],
                                    attrs=dict(long_name="link/interface between two flow elements"))

        ds['FlowLinkType']=xr.DataArray(2*np.ones(len(links)),dims=['nFlowLink'],
                                        attrs=dict(long_name="type of flowlink",
                                                   valid_range=[1,2], # "1,2", this causes problems
                                                   flag_values=[1,2], # "1,2", CF conventions
                                                   flag_meanings=("link_between_1D_flow_elements "
                                                                  "link_between_2D_flow_elements" )))

        xyu=np.zeros((len(links),2),'f8')
        xyu[~bclinks]=xycc[links[~bclinks]].mean(axis=1) # average centroids
        xyu[bclinks]=xycc[links[bclinks].min(axis=1)] # centroid of real element

        ds['FlowLink_xu']=xr.DataArray(xyu[:,0],dims=['nFlowLink'],
                                       attrs=dict(units="m",
                                                  standard_name = "projection_x_coordinate" ,
                                                  long_name = "Center coordinate of net link (velocity point)." ,
                                                  grid_mapping = "projected_coordinate_system"))
        ds['FlowLink_yu']=xr.DataArray(xyu[:,1],dims=['nFlowLink'],
                                       attrs=dict(units="m",
                                                  standard_name="projection_y_coordinate" ,
                                                  long_name="Center coordinate of net link (velocity point)." ,
                                                  grid_mapping="projected_coordinate_system"))

        ds['FlowElemDomain']=xr.DataArray(np.zeros(len(self.elements),'i2'),dims=['nFlowElem'],
                                          attrs=dict(long_name="Domain number of flow element"))

        ds['FlowLinkDomain']=xr.DataArray(np.zeros(len(links),'i2'),dims=['nFlowLink'],
                                          attrs=dict(long_name="Domain number of flow link"))
        ds['FlowElemGlobalNr']=xr.DataArray(1+np.arange(len(self.elements)),
                                            dims=['nFlowElem'],
                                            attrs=dict(long_name="Global flow element numbering"))

        # node stuff - more of a pain....
        # awkward python2/3 compat.
        xy_to_node=defaultdict(lambda c=count(): next(c) ) # tuple of (x,y) to node
        nodes=np.zeros( ds.FlowElemContour_x.shape, 'i4')
        for c in range(nodes.shape[0]):
            for cc in range(nodes.shape[1]):
                if np.isfinite(packed[c,cc,0]):
                    nodes[c,cc] = xy_to_node[ (packed[c,cc,0],packed[c,cc,1]) ]
                else:
                    nodes[c,cc]=-1
        Nnodes=1+nodes.max()
        node_xy=np.zeros( (Nnodes,2), 'f8')
        for k,v in iteritems(xy_to_node):
            node_xy[v,:]=k

        ds['Node_x']=xr.DataArray(node_xy[:,0],dims=['nNode'])
        ds['Node_y']=xr.DataArray(node_xy[:,1],dims=['nNode'])

        ds['FlowElemContour_node']=xr.DataArray(nodes,dims=['nFlowElem','nFlowElemContourPts'],
                                                attrs=dict(cf_role="face_node_connectivity",
                                                           long_name="Maps faces to constituent vertices/nodes",
                                                           start_index=0))
        # Edges
        points=np.array( [ds.Node_x.values,
                          ds.Node_y.values] ).T
        cells=np.array(ds.FlowElemContour_node.values)
        ug=unstructured_grid.UnstructuredGrid(points=points,
                                              cells=cells)
        ug.make_edges_from_cells()
        
        # Note that this isn't going to follow any particular ordering
        ds['FlowEdge_node']=xr.DataArray(ug.edges['nodes'],dims=['nFlowEdge','nEdgePts'],
                                         attrs=dict(cf_role="edge_node_connectivity",
                                                    long_name = "Maps edge to constituent vertices" ,
                                                    start_index=0))

        # from sundwaq - for now assume that we have a 
        sub=self.open_flowgeom_ds(0)

        layers='nFlowMesh_layers'
        layer_bnds='nFlowMesh_layers_bnds'
        
        if layers in sub:
            # This code is brittle, not set up to handle arbritrary grid inputs
            # But flowgeom doesn't really need vertical information
            attrs=dict(standard_name="ocean_zlevel_coordinate",
                       long_name="elevation at layer midpoints" ,
                       positive="up" ,
                       units="m")
            if layer_bnds in sub:
                attrs['bounds']=layer_bnds
                
            ds[layers]=xr.DataArray(sub[layers].values,
                                    dims=[layers],
                                    attrs=attrs)
            # this syntax works better:
            if layer_bnds in sub:
                ds[layer_bnds]=( [layers,'d2'],
                                 sub[layer_bnds].values.copy() )

        ds['FlowMesh']=xr.DataArray(1,
                                    attrs=dict(cf_role = "mesh_topology" ,
                                               long_name = "Topology data of 2D unstructured mesh" ,
                                               dimension = 2 ,
                                               node_coordinates = "Node_x Node_y" ,
                                               face_node_connectivity = "FlowElemContour_node" ,
                                               edge_node_connectivity = "FlowEdge_node" ,
                                               face_coordinates = "FlowElem_xcc FlowElem_ycc" ,
                                               face_face_connectivity = "FlowLink"))

        # global attrs
        ds.attrs['institution'] = "San Francisco Estuary Institute"
        ds.attrs['references'] = "http://www.deltares.nl" 
        ds.attrs['source'] = "Python/Delft tools, rustyh@sfei.org" 
        ds.attrs['history'] = "Generated by stompy" 
        ds.attrs['Conventions'] = "CF-1.5:Deltares-0.1" 
        return ds

    def create_bnd(self):
        """
        populate self.bnd, same format as read_bnd()
        but based on pulling the names from the subdomains.
        Note that this was originally developed in MultiAggregator, but is now
        being adapted for more general use.  The original MultiAggregator
        implementation is included in that class
        Note that if BCs are *not* aggregated, the data returned here can have
        multiple boundary entries for the same x coordinate / coordinates
        """
        # Subdomains do not list sources in FlowLink, but the aggregated grid does
        # at least for now.
        bc_names={}
        # Collect subdomain geometry as it maps to aggregated domain
        bc_x=defaultdict(list)

        # finally, get back to a
        geom=self.get_geom()

        flow_link_inside=geom.FlowLink.values.min(axis=1)
        flow_link_outside=geom.FlowLink.values.max(axis=1)
        # outside is the larger of the 1-based element indices
        # len(nFlowElem) is the largest 1-based value of an inside element
        # set those to -1
        # What I want here is the negative bc id
        nFlowElem=len(geom.nFlowElem)
        bc_links= flow_link_outside>nFlowElem
        flow_link_outside[bc_links] = nFlowElem - flow_link_outside[bc_links]

        self.infer_2d_links()
        
        # iterate over bnds from each subdomain
        fail=False
        for proc in range(self.nprocs):
            sub_hyd=self.open_hyd(proc)
            sub_hyd.infer_2d_elements()

            sub_geom=self.open_flowgeom_ds(proc)
            if 'mesh2d' in sub_geom:
                face_dim=sub_geom.mesh2d.attrs.get('face_dimension','nFlowElem')
            else:
                face_dim='nFlowElem'
            sub_nFlowElem=sub_geom.dims[face_dim]
            sub_bnds=sub_hyd.read_bnd()
            sub_g=unstructured_grid.UnstructuredGrid.read_dfm(sub_geom)

            sub_hyd.infer_2d_links()

            if 1: # debugging
                print(f"[proc={proc}] n_2d_elements={sub_hyd.n_2d_elements} g.Ncells={sub_g.Ncells()}")
                print(f"              FlowLink min={sub_geom.FlowLink.values.min()} max={sub_geom.FlowLink.values.max()}")
                min_bc_linkid = min( [0] +[sub_bnd[1]['link'].min() for sub_bnd in sub_bnds] )
                # Is this 1-off? No, I think it's correct.
                print(f"              min_bc_link_id={min_bc_linkid} expected max FlowLink {sub_hyd.n_2d_elements-min_bc_linkid}")

            for sub_bnd in sub_bnds:
                name,segs=sub_bnd
                for seg in segs:
                    # This link_id is negative
                    # This "link_id" is really the negative index to the boundary
                    # element
                    link_id=seg['link']
                    # Get its positive counterpart: this is a 0-based index to
                    # a boundary element
                    bc_elt_pos=sub_hyd.n_2d_elements - link_id - 1
                    x=seg['x']

                    # Can we now get back to the local link this belongs to?
                    # Then go from that local link to an aggregated link
                    if np.all( x[0] == x[1] ):
                        # Would like figure out which entry in FlowLink to point to
                        # the aggregated output includes these source links in FlowLink
                        # but the source domains do not
                        # Can it be found based on coordinate?
                        # Not an exact match.
                        x0=x[0]
                        elt_inside=sub_g.select_cells_nearest(x[0],inside=True)
                        elt_outside=None # subdomains don't have these in FlowLink

                        # RH 2019-09-10: this stanza is new. previously, there was a bug
                        # here as this if clause did not set link1, which was then used
                        # below. when reading spliced hydro for a second aggregation step,
                        # it was failing, but it appeared that sub_geom.FlowLink 
                        # actually had all the data, and could have been matched to bc_elt_pos+1.
                        # so try that, but it's possible it will break again when splicing.
                        # RH 2024-01-26: operate on FlowLinkSS, which puts sink-source links back
                        #  in.
                        link1,fromto=np.nonzero( sub_geom.FlowLinkSS.values==bc_elt_pos+1)
                        if len(link1)!=1:
                            print("Trouble finding the FlowLink which goes with this [src] bc element")
                            print("  link1: %s"%link1)
                            fail=True
                            continue # Not sure if this will work...
                        else:
                            link1=link1[0]
                            fromto=fromto[0]
                            # if all is well, elt_inside from above should match with what's in FlowLink.
                            # I don't think elt_outside really matters
                            assert sub_geom.FlowLinkSS.values[link1,1-fromto]-1 == elt_inside,"Sanity comparison on source element failed"
                    else:
                        # print("Horizontal bnd entry")
                        # for regular horizontal bnd entries, not necessary to go to FlowLinkSS
                        # but it shouldn't hurt and hopefully reduces confusion.
                        link1,fromto=np.nonzero( sub_geom.FlowLinkSS.values==bc_elt_pos+1)
                        if len(link1)!=1:
                            print("Trouble finding the FlowLink which goes with this bc element")
                            print("  link1: %s"%link1)
                        link1=link1[0]
                        fromto=fromto[0]

                        # this link in terms of local elements, as 0-based
                        elt_inside=sub_geom.FlowLinkSS.values[link1,1-fromto] - 1 
                        elt_outside=sub_geom.FlowLinkSS.values[link1,fromto]  - 1

                    # this comes as 1-based, based on waq_scenario.py code.
                    elt_inside_global=sub_geom.FlowElemGlobalNr.values[elt_inside] - 1

                    # Is it local to this proc?
                    elt_is_local=sub_geom.FlowElemDomain.values[elt_inside]==proc

                    elt_agg=self.elt_global_to_agg_2d[elt_inside_global]

                    if (not elt_is_local) or (elt_agg<0):
                        # only consider inside elements which are in the aggregation
                        # only get data from the owner of the inside element
                        continue 

                    # This is the right processor to get data from, and
                    # this element is within the aggregation

                    # elt_agg is 0-based
                    # This is probably a source of problems -- we match too broadly here, ignoring
                    # the value of flow_link_outside, and lumping together all links that are
                    # boundary and match on interior element -- and then below we disregard
                    # additional matches.
                    bc_flow_links=np.nonzero( (flow_link_inside==elt_agg+1) & (flow_link_outside<0) )[0]

                    # to solve this multiple mapping problem, go through the exchanges
                    if 1: # trying
                        # This is a sanity check
                        assert sub_hyd.links[link1,1-fromto]==elt_inside,"links in flowgeom may not be same as sub_hyd.links"
                        assert not self.agg_boundaries,"This code only makes sense with unaggregated boundaries"
                        # which exchanges are involved in the unaggregated grid?
                        sub_link_exchs=np.nonzero( sub_hyd.exch_to_2d_link['link']==link1 )[0]
                        # just trace the first one:
                        sub_link_exch=sub_link_exchs[0] 
                        # what aggregated exchange does it map to?
                        agg_link_exch=self.exch_local['agg'][proc,sub_link_exch]
                        agg_link=self.exch_to_2d_link[agg_link_exch]
                        assert agg_link['sgn']==1,"Expecting no sign flips for bc links"
                        bc_flow_link=agg_link['link']
                        assert bc_flow_link in bc_flow_links,"Validation against old approach failed"
                        bc_flow_links=np.array([bc_flow_link])
                    
                    # Used to assert this was a unique match
                    if 1: # this is probably better -- at least when not aggregating boundaries
                        assert len(bc_flow_links)==1
                        bc_id=flow_link_outside[bc_flow_links[0]]
                        bc_names[bc_id]=name
                        bc_x[bc_id].append(x)
                    else:
                        if len(bc_flow_links)!=1:
                            #import pdb
                            #pdb.set_trace()
                            print("Matching bc links expected 1 match, but elt_agg=%d, name=%s matched %s"%
                                  (elt_agg,name,bc_flow_links))

                        # Now allow multiple -- may cause problems with writing them out, though.
                        for bc_flow_link in bc_flow_links:
                            bc_id=flow_link_outside[bc_flow_links[0]]
                            if bc_id in bc_names:
                                # 2018-12-12: increase verbosity during test
                                print("bc_id %d already mapped to %s, would have mapped to %s"%
                                      (bc_id,bc_names[bc_id],name))
                                #import pdb
                                #pdb.set_trace()
                                continue
                            bc_names[bc_id]=name
                            bc_x[bc_id].append(x) # Will have to deal with this x not being unique!
                            # break # 2018-12-12 makes me nervous
                            # this seems wrong -- if we match multiple flow links, that probably means
                            # that the matching code above is too broad.

        if fail:
            raise Exception("Delayed fail")
        
        # Reverse that mapping
        bc_ids_for_name=defaultdict(list)

        for k in bc_names:
            name=bc_names[k]
            bc_ids_for_name[name].append(k)

        bnds=[]
        for name in bc_ids_for_name:
            bc_ids=bc_ids_for_name[name]
            links=np.zeros( len(bc_ids), [('link','i4'),('x','f8',(2,2))] )
            links['link']=bc_ids

            for i,bc_id in enumerate(bc_ids):
                if len(bc_x[bc_id])==1:
                    links['x'][i]=bc_x[bc_id][0]
                else:
                    self.log.warning("Don't know how to combine multiple segment")
                    # This happens when multiple BCs enter a single element
                    # 2018-04-05: is that correct?  I think this is when one name maps
                    # to multiple BC links
                    links['x'][i]=bc_x[bc_id][0]
            bnds.append( [name,links])

        self.bnd = bnds
        return self.bnd

    @property
    def bnd_filename(self):
        # Maybe this should be in Hydro, instead of copied to HydroFiles and
        # MultiAggregator?
        return os.path.join(self.scenario.base_path, self.fn_base+".bnd")
        
    def write_boundary_links(self):
        """
        On the way to following the .bnd format
        """
        dest_fn=self.bnd_filename

        bnd=self.create_bnd()
        
        dio.write_bnd(bnd,dest_fn)

    
class HydroAggregator(DwaqAggregator):
    """ Aggregate hydro, where the source hydro is already in one hydro
    object.
    """
    def __init__(self,hydro_in,**kwargs):
        self.hydro_in=hydro_in
        super(HydroAggregator,self).__init__(nprocs=1,
                                             **kwargs)

    def open_hyd(self,p,force=False):
        assert p==0
        return self.hydro_in

    def grid(self):
        if self.agg_shp is not None:
            if isinstance(self.agg_shp,unstructured_grid.UnstructuredGrid):
                return self.agg_shp
            else:
                g=unstructured_grid.UnstructuredGrid.from_shp(self.agg_shp)
                self.log.info("Inferring grid from aggregation shapefile")
                #NB: the edges here do *not* line up with links of the hydro.
                # at some point it may be possible to adjust the aggregation
                # shapefile to make these line up.
                g.make_edges_from_cells()
                return g
        else:
            return self.hydro_in.grid()
    
    def infer_nprocs(self):
        # return 1 # should never get called.
        assert False

    def group_boundary_elements(self):
        # in the simple case with 1:1 mapping, we can just delegate
        # to the hydro_in.
        if self.agg_shp is None: # tantamount to 1:1 mapping
            return self.hydro_in.group_boundary_elements()
        else:
            assert False # try using group_boundary_links() instead!

    def group_boundary_links(self):
        self.hydro_in.infer_2d_links()
        self.infer_2d_links()

        unagg_lgroups = self.hydro_in.group_boundary_links()

        # initialize 
        bc_lgroups=np.zeros(self.n_2d_links,self.link_group_dtype)
        bc_lgroups['id']=-1 # most links are internal and not part of a boundary group
        for lg in bc_lgroups:
            lg['attrs']={} # we have no add'l information for the groups.

        sel_bc=np.nonzero( (self.links[:,0]<0) )[0]

        for group_id,bci in enumerate(sel_bc):
            unagg_matches=np.nonzero(self.link_global_to_agg==bci)[0]
            m_groups=unagg_lgroups[unagg_matches]
            for extra in m_groups[1:]:
                # this means that multiple unaggregated link groups map to the
                # same aggregated link.  So we need some application-specific way
                # of combining them.
                # absent that, the first match will be carried through
                if extra['name'] != m_groups[0]['name']:
                    self.log.warning('Not ready for aggregating boundary link groups - skipping %s'%extra['name'])
            bc_lgroups['id'][bci] = group_id
            if len(m_groups):
                bc_lgroups['name'][bci] = m_groups['name'][0]
                bc_lgroups['attrs'][bci] = m_groups['attrs'][0]
            else:
                self.log.warning("Nobody matched to this aggregated boundary link group bci=%d"%bci)
                bc_lgroups['name'][bci] = "group_%d"%group_id
                bc_lgroups['attrs'][bci] = {}
                break
        return bc_lgroups

    n_2d_links=None
    exch_to_2d_link=None
    links=None
    def infer_2d_links(self): # DwaqAggregator version
        """
        populate self.n_2d_links, self.exch_to_2d_link, self.links 
        note: compared to the incoming _grid_, this may include internal
        boundary exchanges.
        exchanges are identified based on unique from/to pairs of 2d elements.
        in the aggregated case, can additionally distinguish based on the
        collection of unaggregated exchanges which map to these.
        """

        if self.exch_to_2d_link is None:
            self.infer_2d_elements() 
            poi0=self.pointers-1

            # map 0-based exchange index to 0-based link index, limited
            # to horizontal exchangse
            exch_to_2d_link=np.zeros(self.n_exch_x+self.n_exch_y,[('link','i4'),
                                                                  ('sgn','i4')])
            exch_to_2d_link['link']=-1

            #  track some info about links
            links=[] # elt_from,elt_to
            mapped=dict() # (src_2d, dest_2d) => link idx

            # two boundary exchanges, can't be distinguished based on the internal segment.
            # but we can see which unaggregated exchanges/links map to them. 
            # at this point, is there ever a time that we don't want to keep these separate?
            # I think we always want to keep them separate, the crux is how to keep track of
            # who is who between layers.  And that is where we can use the mapping from the
            # unaggregated hydro, where the external id, instead of setting it to -1 and
            # distinguishing only on aggregated internal segment, we can now refine that
            # and label it based on ... maybe the smallest internal segment for boundary
            # exchanges which map to this aggregated exchange?

            self.hydro_in.infer_2d_links()
            # some of the code below can't deal with multiple subdomains
            assert self.exch_local.shape[0]==1

            # and special to aggregated code, also build up a mapping of unaggregated
            # links to aggregated links.  And since we're forcing this code to deal with
            # only a single, global unaggregated domain, this mapping is just global to agg.
            # Maybe this should move out to a more general purpose location??
            link_global_to_agg=np.zeros(self.hydro_in.n_2d_links,'i4')-1

            # build hash table to accelerate the lookup below
            agg_exch_to_locals=defaultdict(list)
            # iterate over the local exchanges, here just proc 0
            for unagg_exch,agg in enumerate(self.exch_local['agg'][0]):
                agg_exch_to_locals[ agg ].append(unagg_exch)
                
            for exch_i,(a,b,_,_) in enumerate(poi0[:self.n_exch_x+self.n_exch_y]):
                # probably have to speed this up with some hashing
                # my_unagg_exchs=np.nonzero(self.exch_local['agg'][0,:]==exch_i)[0]
                my_unagg_exchs=np.array(agg_exch_to_locals[exch_i])
                
                # this is [ (link, sgn), ... ]
                my_unagg_links=self.hydro_in.exch_to_2d_link[my_unagg_exchs]
                if a>=0:
                    a2d=self.seg_to_2d_element[a]
                else:
                    # assuming this only works for global domains
                    # we *could* have multiple unaggregated boundary exchanges mapping
                    # onto this single aggregated boundary exchange.  or not.
                    # what do we know about how the collection of unagg links will be
                    # consistent across layers? ... hmmmph
                    # unsure.. but will use the smallest unaggregated link as a label
                    # to make this aggregated link distinction
                    a2d=-1 - my_unagg_links['link'].min()

                assert b>=0 # too lazy, and this shouldn't happen. 
                b2d=self.seg_to_2d_element[b]

                k='not yet set'
                if (b2d,a2d) in mapped:
                    k=(b2d,a2d) 
                    exch_to_2d_link['link'][exch_i] = mapped[k]
                    exch_to_2d_link['sgn'][exch_i]=-1
                else:
                    k=(a2d,b2d)
                    if k not in mapped:
                        mapped[k]=len(links)
                        # does anyone use the values in links[:,0] ??
                        links.append( [a2d,b2d] )

                    exch_to_2d_link['link'][exch_i] = mapped[k]
                    exch_to_2d_link['sgn'][exch_i]=1
                # record this mapping for later use.  There is some duplicated
                # effort here, since in most cases we'll get the same answer for each
                # of the exchanges in this one link.  But it's possible that some
                # exchanges exist at only certain elevations, or something?  for now
                # duplicate effort in exchange for being sure that all of the links
                # get set.
                # actually, getting some cases where this gets overwritten with
                # different values.  Shouldn't happen!
                prev_values=link_global_to_agg[my_unagg_links['link']]
                # expect that these are either already set, or uninitialized.  but if
                # set to a different link, then we have problems.
                prev_is_okay= (prev_values==mapped[k]) | (prev_values==-1)
                assert np.all(prev_is_okay)
                link_global_to_agg[my_unagg_links['link']]=mapped[k]

            self.link_global_to_agg=link_global_to_agg
            links=np.array(links)
            n_2d_links=len(links)

            ##

            # Bit of a sanity warning on multiple boundary exchanges involving the
            # same segment - this would indicate that there should be multiple 2D
            # links into that segment, but this generic code doesn't have a robust
            # way to deal with that.
            if 1:
                # get 172 of these now.  sounds roughly correct.
                # ~50 in the ocean, 113 or 117 sources, and a handful of
                # others (false_*) which take up multiple links for
                # a single source.

                # indexes of which links are boundary
                bc_links=np.nonzero( links[:,0] < 0 )[0]

                for bc_link in bc_links:
                    # index of which exchanges map to this link
                    exchs=np.nonzero( exch_to_2d_link['link']==bc_link )[0]
                    # link id, sgn for each of those exchanges
                    ab=exch_to_2d_link[exchs]
                    # find the internal segments for each of those exchanges
                    segs=np.zeros(len(ab),'i4')
                    sel0=exch_to_2d_link['sgn'][exchs]>0 # regular order
                    segs[sel0]=poi0[exchs,1]
                    if np.any(~sel0):
                        # including checking for weirdness
                        self.log.warning("Some exchanges had to be flipped when flattening to 2D links")
                        segs[~sel0]=poi0[exchs,0]
                    # And finally, are there any duplicates into the same segment? i.e. a segment
                    # which has multiple boundary exchanges which we have failed to distinguish (since
                    # in this generic implementation we have little info for distinguishing them).
                    # note that in the case of suntans output, this is possible, but if it has been
                    # mapped from multiple domains to a global domain, those exchanges have probably
                    # already been combined.
                    if len(np.unique(segs)) < len(segs):
                        self.log.warning("In flattening exchanges to links, link %d has ambiguous multiple exchanges for the same segment"%bc_link)

            ##
            self.exch_to_2d_link=exch_to_2d_link
            self.links=links
            self.n_2d_links=n_2d_links

    def plot_aggregation(self,ax=None):
        """ 
        schematic of the original grid, aggregated grid, links
        """
        gagg=self.grid()
        gun=self.hydro_in.grid()

        if ax is None:
            ax=plt.gca()

        coll_agg=gagg.plot_cells(ax=ax)
        coll_agg.set_facecolor('none')

        coll_gun=gun.plot_edges(ax=ax,lw=0.3)
        ax.axis('equal')

        centers=np.array( [np.array(gagg.cell_polygon(c).centroid)
                           for c in range(gagg.Ncells()) ] )


        ax.plot(centers[:,0],centers[:,1],'go')

        #for elt in range(gagg.Ncells()):
        #    ax.text(centers[elt,0],centers[elt,1],"cell %d"%elt,size=7,color='red')

        for li,(a,b) in enumerate(self.links):
            # find a point representative of the unaggregated links making up this
            # boundary.
            unagg_links=np.nonzero(self.link_global_to_agg==li)
            unagg_links_xs=[]
            for ab in self.hydro_in.links[unagg_links]: # from elt,to elt
                ab=ab[ab>=0]
                unagg_links_xs.append( np.mean(gun.cells_center()[ab],axis=0) )
            edge_x=np.mean(unagg_links_xs,axis=0) 

            pnts=[]
            if a>=0:
                pnts.append( centers[a] )
            pnts.append(edge_x)
            if b>=0:
                pnts.append( centers[b])

            pnts=np.array(pnts)

            ax.plot( pnts[:,0],pnts[:,1],'g-')
            # ec=centers[[a,b]].mean(axis=0)
            # ax.text(ec[0],ec[1],"link %d"%li,size=7)

        
class HydroMultiAggregator(DwaqAggregator):
    """ Aggregate hydro runs with multiple inputs (i.e. mpi hydro run)
    """
    def __init__(self,run_prefix,path,agg_shp=None,nprocs=None,skip_load_basic=False,
                 **kwargs):
        """ 
        run_prefix: the run name which dfm/sun uses in naming the per-processor directories.
        path: path to the directory containing the per-processor directories.

        E.g. <path>/DFM_DELWAQ_<run_prefix>_0000  should be the folder containing WAQ-formatted
        output for the first processor
        """
        self.run_prefix=run_prefix
        self.path=path
        super(HydroMultiAggregator,self).__init__(agg_shp=agg_shp,nprocs=nprocs,
                                                  skip_load_basic=skip_load_basic,
                                                  **kwargs)

    def sub_dir(self,p):
        return os.path.join(self.path,"DFM_DELWAQ_%s_%04d"%(self.run_prefix,p))

    def sub_hyd(self,p,separate_dir="auto"):
        """
        Path to hyd file for the given subdomain.
        separate_dir:
          True assumes each processor output is in a separate folder
          False: all output in one folder.
          "auto": try True, but if the path does not exist, then try False, then return None
        """
        if separate_dir==True:
            return os.path.join(self.sub_dir(p), "%s_%04d.hyd"%(self.run_prefix,p))
        elif separate_dir==False:
            return os.path.join(self.path,"DFM_DELWAQ_%s"%self.run_prefix,"%s_%04d.hyd"%(self.run_prefix,p))
        elif separate_dir=="auto":
            for sep in [True,False]:
                fn=self.sub_hyd(p,sep)
                if os.path.exists(fn): return fn
            return None

    _hyds=None
    def open_hyd(self,p,force=False):
        if self._hyds is None:
            self._hyds={}
        if force or (p not in self._hyds):
            fn=self.sub_hyd(p)
            if fn is None: raise Exception("Failed to find hyd file for subdomain")
            self._hyds[p]=HydroFiles(fn)
        return self._hyds[p]
    
    def dfm_map_file(self,p):
        """
        Try to infer the name of the dfm map output for processor p, and
        if it exists, return that path
        """
        # map_fn=os.path.join(self.path,"DFM_OUTPUT_%s"%self.run_prefix,"%s_%04d_map.nc"%(self.run_prefix,p))
        # if os.path.exists(map_fn):
        #     return map_fn
        # else:
        #     return None

        # In some cases there is also a timestamp in the name, between the processor part and _map.nc suffix.
        # Allow either of these:
        # short_summer2016_0628_dwaq_0000_map.nc        
        # short_summer2016_0628_dwaq_0000_20160628_140000_map.nc        
        map_patt=os.path.join(self.path,"DFM_OUTPUT_%s"%self.run_prefix,"%s_%04d*_map.nc"%(self.run_prefix,p))
        map_fns=glob.glob(map_patt)
        map_fns.sort()
        if len(map_fns)>1:
            print("CAREFUL! Multiple DFM map files. Might cause problems")
        if map_fns:
            return map_fns[0]
        else:
            return None
        
    def infer_nprocs(self):
        max_nprocs=1024
        for p in range(1+max_nprocs):
            if self.sub_hyd(p) is None: # not os.path.exists(self.sub_dir(p)):
                if p==0:
                    raise WaqException("Failed to find any subdomains -- may be a serial run")
                return p
        else:
            raise Exception("Really - there are more than %d subdomains?"%max_nprocs)

class HydroStructured(Hydro):
    """
    INCOMPLETE.
    Create a rectilinear, structured hydrodynamic input set.
    """
    n_x=n_y=n_z=None # number of cells in three directions

    def __init__(self,**kws):
        """ 
        expects self.n_{x,y,z} to be defined
        """
        super(HydroStructured,self).__init__(**kws)

        # map 3D index to segment index.  1-based
        linear=1+np.arange( self.n_x*self.n_y*self.n_z )
        self.seg_ids=linear.reshape( [self.n_x,self.n_y,self.n_z] )
        
    @property
    def n_seg(self):
        """ active segments """
        return np.sum(self.seg_ids>0)

    @property
    def n_exch_x(self):
        return (self.n_x-1)*self.n_y*self.n_z 
    @property
    def n_exch_y(self):
        return (self.n_y-1)*self.n_x*self.n_z
    @property
    def n_exch_z(self):
        return (self.n_z-1)*self.n_x*self.n_y

    # assumes fully dense grid, and more than 1 z level.
    # @property
    # def n_top(self): # number of surface cells - come first, I think
    #     return self.n_x * self.n_y
    # @property
    # def n_middle(self): # number of mid-watercolumn cells.
    #     return self.n_x * self.n_y*(self.n_z-2)
    # @property
    # def n_bottom(self): # number of bottom cells
    #     return self.n_x * self.n_y

    @property
    def pointers(self):
        pointers=np.zeros( (self.n_exch,4),'i4')
        pointers[...] = self.CLOSED

        # with 3D structured, this will be expanded, and from
        # the seg_ids array it can be auto-generated for sparse
        # grids, too.

        ei=0 # exchange index

        xi=0 ; yi=0 # common index
        for zi in np.arange(self.n_z-1):

            s_up=self.seg_ids[xi,yi,zi]
            s_down=self.seg_ids[xi,yi,zi+1]

            if zi==0: # surface cell - 
                s_upup=self.CLOSED
            else:
                s_upup=self.seg_ids[xi,yi,zi-1]

            if zi==self.n_z-2:
                s_downdown=self.CLOSED
            else:
                s_downdown=self.seg_ids[xi,yi,zi+2]

            pointers[ei,:]=[s_up,s_down,s_upup,s_downdown]
            ei+=1
        return pointers

class FilterHydroBC(Hydro):
    """ 
    Subclass of Hydro which shifts tidal fluxes into changing volumes, with
    only subtidal fluxes.
    """
    lp_secs=86400*36./24
    selection='boundary'

    # for large inputs, use memmap.
    # apply_filter() will create temporary files, and put the names
    # here.  then write*() can just move these files.
    use_memmap=True
    tmp_flo_fn=None
    tmp_vol_fn=None
    tmp_are_fn=None
    # the corresponding mmap objects
    new_flo_mmap=None
    new_vol_mmap=None
    new_are_mmap=None
    
    def __init__(self,original,**kws):
        """
        selection: which exchanges will be filtered.  
           'boundary': only open boundaries
           'all': all exchanges
           bool array: length Nexchanges, with True meaning it will get filtered.
           int array:  indices of exchanges to be filtered.
        """
        self.orig=original
        super(FilterHydroBC,self).__init__(**kws)
        self.lp_secs=float(self.lp_secs) # just to be sure
        self.apply_filter()

    # awkward handling of scenario - it gets set by the scenario, so we have
    # relay the setattr on to the original hydro
    @property
    def scenario(self):
        return self.orig.scenario
    @scenario.setter
    def scenario(self,value):
        self.orig.scenario=value

    # somewhat manual forwarding of attributes and methods
    n_exch_x=forwardTo('orig','n_exch_x')
    n_exch_y=forwardTo('orig','n_exch_y')
    n_exch_z=forwardTo('orig','n_exch_z')
    pointers  =forwardTo('orig','pointers')
    time0     =forwardTo('orig','time0')
    t_secs    =forwardTo('orig','t_secs')
    seg_attrs =forwardTo('orig','seg_attrs')
    
    boundary_values=forwardTo('orig','boundary_values')
    seg_func       =forwardTo('orig','seg_func')
    bottom_depths  =forwardTo('orig','bottom_depths')
    vert_diffs     =forwardTo('orig','vert_diffs')
    n_seg        =forwardTo('orig','n_seg')
    boundary_defs  =forwardTo('orig','boundary_defs')
    exchange_lengths=forwardTo('orig','exchange_lengths')
    elements       =forwardTo('orig','elements')
    write_geom = forwardTo('orig','write_geom')
    grid = forwardTo('orig','grid')

    group_boundary_links = forwardTo('orig','group_boundary_links')
    group_boundary_element = forwardTo('orig','group_boundary_elements')

    seg_active = forwardTo('orig','seg_active')

    bnd_filename = forwardTo('orig','bnd_filename')
    write_boundary_links = forwardTo('orig','write_boundary_links')

    # butter: Butterworth filter.  Good frequency response, but can have some overshoots
    #  when cells dry up, leading to negative volume.
    filter_type='butter' # or 'fir'

    _pad=None
    @property
    def pad(self):
        if self._pad is None:
            npad=int(5*self.lp_secs / self.dt)
            self._pad=np.zeros(npad)
        return self._pad
    
    _dt=None
    @property
    def dt(self):
        if self._dt is None:
            self._dt=np.median(np.diff(self.t_secs))
        return self._dt
            
    def lowpass(self,data):
        if self.filter_type=='butter':
            # For butterworth pad out the ends
            pad =self.pad
            npad=len(pad)

            flow_padded=np.concatenate( ( pad, 
                                          data,
                                          pad) )
            lp_flows=filters.lowpass(flow_padded,
                                     cutoff=self.lp_secs,dt=self.dt)
            lp_flows=lp_flows[npad:-npad] # trim the pad
        elif self.filter_type=='fir':
            # try no padding here
            lp_flows=filters.lowpass_fir(data,winsize=int(self.lp_secs/self.dt))
        else:
            raise Exception('Bad filter type: %s'%self.filter_type)
        return lp_flows

    def apply_filter(self):
        """
        Filters all of the hydro information.  Note that this is not smart about
        loading only a small part of the data, and thus won't work when
        applied to a large dataset.  Better to aggregate some and
        then apply the lowpass.
        """
        if self.use_memmap:
            # for this to work, the source data needs to be sitting on disk where
            # it can be memory-mapped.
            # would be better to make this part of HydroFiles
            assert isinstance(self.orig,HydroFiles),"Memmap only works with HydroFiles"
            flo_fn=self.orig.get_path('flows-file')
            flo_mmap=np.memmap(flo_fn, self.orig.flo_dtype(), mode='r')
            vol_fn=self.orig.get_path('volumes-file')
            vol_mmap=np.memmap(vol_fn, self.orig.vol_dtype(), mode='r')
            are_fn=self.orig.get_path('areas-file')
            are_mmap=np.memmap(are_fn, self.orig.are_dtype(), mode='r')

            self.orig_volumes=vol_mmap['volume']
            self.orig_flows  =flo_mmap['flow']
            self.orig_areas  =are_mmap['area']

            # To modify these, have to create new file
            self.tmp_flo_fn="tmp.flo"
            self.tmp_vol_fn="tmp.vol"
            self.tmp_are_fn="tmp.are"

            self.log.info("Copying flo file")
            shutil.copyfile(flo_fn,self.tmp_flo_fn)
            self.log.info("Copying vol file")
            shutil.copyfile(vol_fn,self.tmp_vol_fn)
            self.log.info("Copying are file") # eventually gets modified
            shutil.copyfile(are_fn,self.tmp_are_fn)

            self.new_flo_mmap=np.memmap(self.tmp_flo_fn, self.flo_dtype(), mode='r+')
            self.filt_flows=self.new_flo_mmap['flow']

            self.new_vol_mmap=np.memmap(self.tmp_vol_fn, self.vol_dtype(), mode='r+')
            self.filt_volumes=self.new_vol_mmap['volume']

            self.new_are_mmap=np.memmap(self.tmp_are_fn, self.are_dtype(), mode='r+')
            self.filt_areas =self.new_are_mmap['area']
        else:
            # old way - uses tons of RAM.  all of the RAM.
            self.filt_volumes=np.array( [self.orig.volumes(t) for t in self.orig.t_secs] )
            self.filt_flows  =np.array( [self.orig.flows(t)   for t in self.orig.t_secs] )
            self.filt_areas  =np.array( [self.orig.areas(t)   for t in self.orig.t_secs] )
            self.orig_volumes=self.filt_volumes.copy()
            self.orig_flows  =self.filt_flows.copy()

        dt=np.median(np.diff(self.t_secs))
        pointers=self.pointers

        # 4th order butterworth gives better rejection of tidal
        # signal than FIR filter.
        # but there can be some transients at the beginning, so pad the flows
        # out with 0s:
        # npad=int(5*self.lp_secs / dt)
        
        for j in utils.progress(self.exchanges_to_filter()):
            # j: index into self.pointers.  
            segA,segB=pointers[j,:2]

            lp_flows=self.lowpass(self.filt_flows[:,j])
            
            # separate into tidal and subtidal constituents
            tidal_flows=self.filt_flows[:,j]-lp_flows
            self.filt_flows[:,j]=lp_flows 

            tidal_volumes= np.cumsum(tidal_flows[:-1]*np.diff(self.t_secs))
            tidal_volumes= np.concatenate ( ( [0],
                                              tidal_volumes ) )
            # a positive flow is *out* of segA, and *in* to segB
            # positive volumes represent water which is now part of the cell
            if segA>0:
                self.filt_volumes[:,segA-1] += tidal_volumes
                #if np.any( self.filt_volumes[:,segA-1]<0 ):
                #    self.log.warning("while filtering fluxes had negative volume (may be temporary)")
            if segB>0:
                self.filt_volumes[:,segB-1] -= tidal_volumes
                #if np.any( self.filt_volumes[:,segB-1]<0 ):
                #    self.log.warning("while filtering fluxes had negative volume (may be temporary)")

        self.adjust_negative_volumes()

        # it's possible to have some transient negative volumes that work themselves out
        # when other fluxes are included.  but in the end, can't have any negatives.
        assert( np.all(self.filt_volumes>=0) )

        if np.any(self.filt_volumes<self.min_volume):
            self.log.warning("All volumes non-negative, but some below threshold of %f"%self.min_volume)

        self.adjust_plan_areas()

    min_volume=0.0 # 
    min_area = 1.0 # this is very important, I thought
    # actually, well, it may be that planform_areas must be positive, but exchange
    # areas can be zero.  Since those are closely linked, it's easiest and doesn't 
    # seem to break anything to enforce a min_area here.

    # If true, volumes which go negative and have no boundary exchange to adjust
    # will just get their volume increased by whatever amount necessary to be
    # non-negative over time.
    force_min_volume=True
    def adjust_negative_volumes(self):
        has_negative=np.nonzero( np.any(self.filt_volumes<self.min_volume,axis=0 ) )[0]
        dt=np.median(np.diff(self.t_secs))

        for seg in has_negative:
            self.log.info("Attempting to undo negative volumes in seg %d"%seg)
            # Find a BC segment
            bc_exchs=np.nonzero( (self.pointers[:,0] < 0) & (self.pointers[:,1]==seg+1))[0]
            if len(bc_exchs)==0:
                # will lead to a failure 
                self.log.warning("Segment with negative volume has no boundary exchanges")
                if self.force_min_volume:
                    V_add=self.min_volume-self.filt_volumes[:,seg].min()
                    self.log.warning("Forcing minimum volume. Adding %.3e  Original volumes max: %.3e, mean %.3e"%
                                     (V_add, self.orig_volumes[:,seg].max(),self.orig_volumes[:,seg].mean()))
                    self.filt_volumes[:,seg]+=V_add
                
                continue
            elif len(bc_exchs)>1:
                self.log.info("Segment with negative volume has multiple BC exchanges.  Choosing the first")
            bc_exch=bc_exchs[0]

            orig_vol=self.orig_volumes[:,seg]# here
            orig_flow=self.orig_flows[:,bc_exch]
            filt_vol=self.filt_volumes[:,seg]
            filt_flow=self.filt_flows[:,bc_exch]

            # correct up to min_volume
            err_vol=filt_vol.clip(-np.inf,self.min_volume) - self.min_volume
            corr_flow=-np.diff(err_vol) / np.diff(self.t_secs)
            # last flow entry isn't used, and we don't have the next volume to know 
            # what it should be anyway.
            corr_flow=np.concatenate( ( corr_flow, [0]) )

            orig_rms=utils.rms(orig_flow)
            filt_rms=utils.rms(filt_flow)

            self.filt_volumes[:,seg]   -=err_vol
            self.filt_flows[:,bc_exch]+=corr_flow

            # report change in rms flow:
            upd_rms=utils.rms(self.filt_flows[:,bc_exch])

            print("    Original flow rms: ",orig_rms)
            print("    Filtered flow rms: ",filt_rms)
            print("    Updated flow rms:  ",upd_rms)

    def adjust_plan_areas(self):
        """ 
        Modifying the volume of segments should be reflected in a change
        in at least some exchange area.  Most appropriate is to change
        the planform area.  
        It's not clear what invariants are required, expected, or most common.
        Assume that it's best to keep planform area constant within a water
        column (i.e. a prismatic grid).  Adjusted area is then Aorig*Vfilter/Vorig.
        """

        # for the vertical integration, have to figure out the structure of the
        # water columns.  This populates seg_to_2d_element:
        self.orig.infer_2d_elements()

        # Then sum volume in each water column - ratio of new volume to old volume
        # gives the factor by which plan-areas should be increased.
        # group in the sense of SQL group by
        groups=self.orig.seg_to_2d_element

        volumes     =np.array( [self.volumes(t)      for t in self.t_secs] )
        orig_volumes=np.array( [self.orig.volumes(t) for t in self.t_secs] )

        # sum volume in each water column
        # might have dense output of z-levels, for which there segments which don't
        # belong to a water column - bincount doesn't like those negative values.
        valid=groups>=0
        
        Vratio_filt_to_orig_2d=[ np.bincount(groups[valid],volumes[ti,valid])  / \
                                 np.bincount(groups[valid],orig_volumes[ti,valid])                                 
                                 for ti in range(len(self.t_secs)) ]
        Afactor_per_2d_element=np.array( Vratio_filt_to_orig_2d )

        # loop over vertical exchanges, updating areas
        exchs=self.pointers
        for j in range(self.n_exch_x+self.n_exch_y,self.n_exch):
            segA,segB=exchs[j,:2] - 1 # seg now 0-based
            if segA<0: # boundary
                group=groups[segB]
            elif segB<0: # boundary
                group=groups[segA]
            else:
                assert(groups[segA]==groups[segB])
                group=groups[segA]
            # update this exchanges area, for all time steps
            self.filt_areas[:,j] *= Afactor_per_2d_element[:,group]

        # clean up a slightly different issue while we're at it.
        # since upper layers can dry out, it's possible that we'll
        # add some lowpass flow, but the area will be zero.
        # there is also the very likely case that unused exchanges
        # have zero flow and zero area, but maybe that's not a big deal.
        for exch in range(self.n_exch):
            sel=(self.filt_areas[:,exch]==0) & (self.filt_flows[:,exch]!=0)
            if np.any(sel):
                self.log.warning("Cleaning up zero area exchange %d"%exch)
                if np.all( self.filt_areas[:,exch]==0  ):
                    raise Exception("An exchange has some flow, but never has any area")
                self.filt_areas[sel,exch] = np.nan
                self.filt_areas[:,exch] = utils.fill_invalid(self.filt_areas[:,exch])
                
        # and finally, delwaq2 doesn't like to have any zero-area exchanges, even if
        # they never have any flow.  so they all get unit area.
        
        self.filt_areas[ self.filt_areas<self.min_area ] = self.min_area


    def exchanges_to_filter(self):
        """
        return indices into self.pointers which should get the filtering
        not restricted to boundary exchanges
        """
        # defaults to boundary exchanges
        pointers=self.pointers
        selection=self.selection
        if isinstance(selection,str):
            if selection=='boundary':
                sel=np.nonzero(pointers[:,0]<0)[0]
            elif selection=='all':
                sel=np.arange(len(pointers))
            else:
                assert False
        else:
            selection=np.asarray(selection)
            if selection.dtype==np.bool_:
                sel=np.nonzero(selection)
            else:
                sel=selection
        return sel

    def volumes(self,t):
        ti=self.t_sec_to_index(t)
        return self.filt_volumes[ti,:]
    def flows(self,t):
        ti=self.t_sec_to_index(t)
        return self.filt_flows[ti,:]
    def areas(self,t):
        ti=self.t_sec_to_index(t)
        return self.filt_areas[ti,:]

    def planform_areas(self):
        """ Here have to take into account the time-variability of
        planform area.
        """
        # pull areas from exchange areas of vertical exchanges
    
        seg_z_exch=self.seg_to_exch_z(preference='upper')
    
        # then pull exchange area for each time step
        A=np.zeros( (len(self.t_secs),self.n_seg) )
        for ti,t_sec in enumerate(self.t_secs):
            areas=self.areas(t_sec)
            A[ti,:] = areas[seg_z_exch]

        A[ A<1.0 ] = 1.0 # just to be safe, in case area_min above is removed.
        return ParameterSpatioTemporal(times=self.t_secs,values=A,hydro=self)
    
    def depths(self):
        """ Compute time-varying segment thicknesses.  With z-levels, this is
        a little more nuanced than the standard calc. in delwaq.
        """
        if 1: 
            # just defer depth to the unfiltered data - since we'd actually like
            # to be replicating depth variation, and the filtering is just
            # to reduce numerical diffusion.
            return self.orig.depths()
        else:
            # reconstruct segment thickness from area/volume.
            # use upper, since some bed segment would have a zero area for the
            # lower exchange
            print("Call to depths parameter!")

            seg_z_exch=self.seg_to_exch_z(preference='upper')
            D=np.zeros( (len(self.t_secs),self.n_seg) )
            for ti,t_sec in enumerate(self.t_secs):
                areas=self.areas(t_sec)
                volumes=self.volumes(t_sec)
                D[ti,:] = volumes / areas[seg_z_exch]

            # following new code from WaqAggregator above.
            sel=(~np.isfinite(D))
            D[sel]=0.0

            return ParameterSpatioTemporal(times=self.t_secs,values=D,hydro=self)

    # These cannot be forwarded, b/c other code assumes that after calling,
    # additional state is set on self
    def infer_2d_links(self):
        self.orig.infer_2d_links()
        self.n_2d_links=self.orig.n_2d_links
        self.exch_to_2d_link=self.orig.exch_to_2d_link
        self.links=self.orig.links

    # overload these in the case of memory mapped files
    def write_flo(self):
        self.copy_mmap_or_delegate("tmp_flo_fn","new_flo_mmap",
                                   super(FilterHydroBC,self).write_flo,
                                   self.flo_filename)
    def write_vol(self):
        self.copy_mmap_or_delegate("tmp_vol_fn","new_vol_mmap",
                                   super(FilterHydroBC,self).write_vol,
                                   self.vol_filename)
    def write_are(self):
        self.copy_mmap_or_delegate("tmp_are_fn","new_are_mmap",
                                   super(FilterHydroBC,self).write_are,
                                   self.are_filename)

    def copy_mmap_or_delegate(self,fn_attr,mmap_attr,deleg,new_fn):
        tmp_fn=getattr(self,fn_attr)
        if self.use_memmap and tmp_fn is not None:
            assert os.path.exists(tmp_fn),"Hmm - should have memory mapped but %s is not there"%tmp_fn
            delattr(self,mmap_attr)
            setattr(self,mmap_attr,None)
            os.rename(tmp_fn,new_fn)
            return
        self.log.warning("Filter Hydro could not simply rename mapped file")
        # Make it easier to catch this condition while developing
        raise Exception("During testing this shouldn't happen")
        deleg()
        
class FilterAll(FilterHydroBC):
    """ Minor specialization when you want filter everything - i.e. turn a tidal
    run into a subtidal run.
    """
    # In the past parameter were always filtered (opposite of what I thought).
    # that can be disabled here.
    filter_parameters=True
    
    def __init__(self,original,**kw):
        super(FilterAll,self).__init__(original,selection='all',**kw)

    def adjust_plan_areas(self):
        """ 
        The original FilterHydroBC code adjust plan areas, 
        meant to shift tidal signals into horizontally expanding/contracting
        segments.
        But when filtering all exchanges, probably better to just remove 
        tidal variation from horizontal exchanges.  
        """
        dt=np.median(np.diff(self.t_secs))

        npad=int(5*self.lp_secs / dt)
        pad =np.zeros(npad)
        
        # loop over horizontal exchanges, updating areas
        poi0=self.pointers-1
        for j in range(self.n_exch_x+self.n_exch_y):
            padded=np.concatenate( ( pad, 
                                     self.filt_areas[:,j],
                                     pad) )
            lp_areas=filters.lowpass(padded,cutoff=self.lp_secs,dt=dt)
            lp_areas=lp_areas[npad:-npad] # trim the pad
            
            self.filt_areas[:,j] = lp_areas

        # FilterHydroBC does some extra work right here, but I'm hoping that's
        # not necessary??
                
        # and finally, delwaq2 doesn't like to have any zero-area exchanges, even if
        # they never have any flow.  so they all get unit area.
        self.filt_areas[ self.filt_areas<self.min_area ] = self.min_area

    def planform_areas(self):
        """ Skip FilterHydroBC's filtering of planform areas, use the original hydro
        instead.
        """
        return self.orig.planform_areas()
    
    def depths(self):
        """ Compute time-varying segment thicknesses.  With z-levels, this is
        a little more nuanced than the standard calc. in delwaq.
        """
        # reconstruct segment thickness from area/volume.
        # use upper, since some bed segment would have a zero area for the
        # lower exchange
        print("Call to depths parameter!")

        # this also gets simpler due to the constant area in each water
        # column.
        seg_z_exch=self.seg_to_exch_z(preference='upper')
        
        D=np.zeros( (len(self.t_secs),self.n_seg) )
        for ti,t_sec in enumerate(self.t_secs):
            areas=self.areas(t_sec)
            volumes=self.volumes(t_sec)
            D[ti,:] = volumes / areas[seg_z_exch]

        # following new code from WaqAggregator above.
        sel=(~np.isfinite(D))
        D[sel]=0.0

        return ParameterSpatioTemporal(times=self.t_secs,values=D,hydro=self)

    def add_parameters(self,hyd):
        hyd=super(FilterAll,self).add_parameters(hyd)

        for key,param in iteritems(self.orig.parameters()):
            self.log.info('Original -> filtered parameter %s'%key)
            # overwrite vertdisper
            if key in hyd and key not in ['vertdisper']:
                self.log.info('  parameter already set')
                continue
            elif isinstance(param,ParameterSpatioTemporal):
                if self.filter_parameters:
                    self.log.info("  original parameter is spatiotemporal - let's FILTER")
                    hyd[key]=param.lowpass(self.lp_secs)
                    if key in ['vertdisper','tau','salinity']: # force non-negative
                        # a bit dangerous, since there is no guarantee that lowpass made a copy.
                        # but if it didn't make a copy, and all of the source data were
                        # valid, then there should be nothing to clip.
                        hyd[key].values = hyd[key].values.clip(0,np.inf)
                    hyd[key].hydro=self
                    self.log.info("  FILTERED.")
                else:
                    self.log.info("  original parameter is spatiotemporal - but filter_parameters=%s.  no filter"%self.filter_parameters)
                    hyd[key]=param
            elif isinstance(param,ParameterTemporal):
                self.log.warning("  original parameter is temporal - should filter")
                hyd[key]=param # FIX - copy and set hydro, maybe filter, too.
            else:
                self.log.info("  original parameter is not temporal - just copy")
                hyd[key]=param # ideally copy and set hydro
                
        return hyd


# utility for Sigmified
def rediscretize(src_dx,src_y,n_sigma,frac_samples=None,intensive=True):
    """
    Redistribute src_y values from bins of size src_dx to n_sigma
    bins of equal size.  
    Promises that sum(src_y*src_dx) == sum(dest_y*dest_dx), (or without
    the *_dx part if intensive is False).
    """
    
    #seg_sel_z=np.nonzero( self.hydro_z.seg_to_2d_element==elt )[0]
    #seg_v_z=vol_z[seg_sel_z] # that's src_dx
    # seg_scal_z=scalar_z[seg_sel_z] # that's src_y
    src_dx_sum=src_dx.sum()
    if src_dx_sum==0:
        assert np.all(src_y==0.0)
        return np.zeros(n_sigma)
    
    src_dx = src_dx / src_dx_sum # normalize to 1.0 volume for ease
    src_xsum=np.cumsum(src_dx)

    # would like to integrate that, finding s_i = 10 * (Int i/10,(i+1)/10 s df)
    # instead, use cumsum to handle the discrete integral then interp to pull
    # out the individual values

    if intensive:
        src_y_ext = src_y * src_dx
    else:
        src_y_ext = src_y
        
    cumul_mass =np.concatenate( ( [0],
                                  np.cumsum(src_y_ext) ) )
    frac_sum=np.concatenate( ( [0], src_xsum ) )
    if frac_samples is None:
        frac_samples=np.linspace(0,1,n_sigma+1)
        
    dest_y = np.diff(np.interp(frac_samples,
                               frac_sum,cumul_mass) )
    if intensive:
        dest_y *= n_sigma # assumes evenly spread out layers 
    return dest_y

class Sigmified(Hydro):
    def __init__(self,hydro_z,n_sigma=10,**kw):
        super(Sigmified,self).__init__(**kw)
        self.hydro_z=hydro_z
        self.n_sigma=n_sigma
        self.init_exchanges()
        self.init_lengths()
        self.init_2_to_3_maps()

    def init_2_to_3_maps(self):
        self.infer_2d_elements()
        self.hydro_z.infer_2d_elements()

        elt_to_seg_z=[ [] for _ in range(self.n_2d_elements) ]
        
        #for elt in range(self.n_2d_elements):
        #    self.elt_to_seg_z[elt]=np.nonzero( self.hydro_z.seg_to_2d_element==elt )[0]

        # this should be faster
        for seg,elt in enumerate(self.hydro_z.seg_to_2d_element):
            elt_to_seg_z[elt].append(seg)
        self.elt_to_seg_z=[ np.array(segs) for segs in elt_to_seg_z]

        # similar, but for links, and used for both z and sig
        link_to_exch_z=  [ [] for _ in range(self.n_2d_links)]
        link_to_exch_sig=[ [] for _ in range(self.n_2d_links)]

        for exch,link in enumerate(self.hydro_z.exch_to_2d_link['link']):
            link_to_exch_z[link].append(exch)
        for exch,link in enumerate(self.exch_to_2d_link['link']):
            link_to_exch_sig[link].append(exch)
        self.link_to_exch_z=[ np.array(exchs) for exchs in link_to_exch_z ]
        self.link_to_exch_sig=[ np.array(exchs) for exchs in link_to_exch_sig ]

    @property
    def n_seg(self):
        return self.n_sigma * self.hydro_z.n_2d_elements
    
    time0     =forwardTo('hydro_z','time0')
    t_secs    =forwardTo('hydro_z','t_secs')
    group_boundary_links = forwardTo('hydro_z','group_boundary_links')
    group_boundary_element = forwardTo('hydro_z','group_boundary_elements')

    def init_exchanges(self):
        """
        populates pointers, n_2d_links, n_exch_{x,y,z}
        """
        self.hydro_z.infer_2d_links()

        # links are the same for the z-layer and sigma
        self.links=self.hydro_z.links.copy()
        self.n_2d_links = self.hydro_z.n_2d_links

        poi0_z = self.hydro_z.pointers - 1
        # start with all of the internal exchanges, then the top-layer boundary
        # exchanges, then to lower layers.
        # write it first without any notion of boundary fluxes, then fix it up

        poi0_sig=[]
        exch_to_2d_link=[] # build this up as we go
        n_exch_x=0
        n_exch_z=0

        self.infer_2d_elements()

        n_bc=0 # counter for boundary exchanges
        # horizontal exchanges:
        for sig in range(self.n_sigma):
            for link_i,(link_from,link_to) in enumerate(self.links):
                # doesn't distinguish between boundary conditions and internal
                # possible that boundary conditionsa are only in certain layers.
                # In our specific setup, even boundary conditions which are in the hydro
                # only at the surface probably *ought* to be spread across the
                # water column.
                # But - do need to be smarter about how the outside segments are
                # numbered.
                # In particular, link_from is sometimes negative, and probably we're
                # not supposed to just multiply by sig
                if link_from<0:
                    # first one gets a 0-based index of -2, so -1 in real pointers
                    seg_from=-2 - n_bc
                    n_bc+=1
                else:
                    seg_from=link_from + sig*self.n_2d_elements
                seg_to=link_to + sig*self.n_2d_elements
                exchi=len(poi0_sig)
                poi0_sig.append( [seg_from,seg_to,-1,-1] )
                # all forward:
                exch_to_2d_link.append( [link_i,1] )
        self.n_exch_x=len(poi0_sig)
        
        # vertical exchangse
        for sig in range(self.n_sigma-1):
            for elt in range(self.n_2d_elements):
                seg_from=elt + sig*self.n_2d_elements
                seg_to  =elt + (sig+1)*self.n_2d_elements
                exchi=len(poi0_sig)
                poi0_sig.append( [seg_from,seg_to,-1,-1] )

        # not quite ready for vertical boundary exchanges, so make sure they
        # don't exist in the input
        assert self.hydro_z.pointers[(self.hydro_z.n_exch_x+self.hydro_z.n_exch_y):,:2].min()>0
        
        self.n_exch_y=0
        self.n_exch_z=len(poi0_sig) - self.n_exch_x - self.n_exch_y
        self.pointers=np.array(poi0_sig)+1
        exch_to_2d_link=np.array( exch_to_2d_link )
        self.exch_to_2d_link=np.zeros(self.n_exch_x+self.n_exch_y,
                                      [('link','i4'),('sgn','i4')])
        self.exch_to_2d_link['link']=exch_to_2d_link[:,0]
        self.exch_to_2d_link['sgn']=exch_to_2d_link[:,1]

    def init_lengths(self):
        lengths = np.zeros( (self.n_exch,2), 'f4' )
        lengths[:self.n_exch_x+self.n_exch_y,:] = np.tile(self.hydro_z.exchange_lengths[:self.n_2d_links,:],(self.n_sigma,1))
        lengths[self.n_exch_x+self.n_exch_y:,:] = 1./self.n_sigma
        self.exchange_lengths=lengths
        
    def areas(self,t):
        poi0_sig = self.pointers - 1

        Af_sig=np.zeros(len(poi0_sig),'f4')

        # calculate flux-face areas
        Af_z = self.hydro_z.areas(t)
        Af_x_z=Af_z[:self.hydro_z.n_exch_x + self.hydro_z.n_exch_y]

        # horizontal first:
        # sum per 2d link:
        Af_per_link_z = np.bincount( self.hydro_z.exch_to_2d_link['link'],
                                     weights=Af_x_z )
        # then evenly divide by n_sigma
        Af_sig[:self.n_exch_x+self.n_exch_y] = Af_per_link_z[ self.exch_to_2d_link['link'] ] / self.n_sigma

        # vertical: here there could be z-layer water columns with no exchanges, so no area here, but
        # sigma grid will have areas
        # instead, we go to planform_areas()
        plan_areas=self.planform_areas().evaluate(t=t).data # [Nseg]

        from_seg=poi0_sig[self.n_exch_x+self.n_exch_y:,0]
        assert np.all( from_seg>= 0 )
        Af_sig[self.n_exch_x+self.n_exch_y:] = plan_areas[from_seg]
        return Af_sig
        
    def volumes(self,t_sec):
        self.hydro_z.infer_2d_elements()

        v_z = self.hydro_z.volumes(t_sec)

        seg_to_elt = self.hydro_z.seg_to_2d_element.copy() # negative for below-bed segments
        assert np.all( v_z[ seg_to_elt<0 ] ==0 ) # sanity.

        seg_to_elt[ seg_to_elt<0 ] = 0 # to allow easy summation.

        elt_v_z = np.bincount(seg_to_elt,weights=v_z)

        # divide volume evenly across layers, tile out to make dense linear matrix.
        seg_v_sig=np.tile( elt_v_z/self.n_sigma, self.n_sigma)
        return seg_v_sig
    
    dz_deficit_threshold=-0.001
    def flows(self,t):
        Q_z = self.hydro_z.flows(t)
        Q_sig=np.zeros( self.n_exch, 'f8' )
        A_z  =self.hydro_z.areas(t)
        A_sig=self.areas(t)

        frac_samples=np.linspace(0,1,self.n_sigma+1)
        
        # start with just the horizontal flows:
        for link_i,link in enumerate(self.links):
            exch_sel_z  =self.link_to_exch_z[link_i]   # np.nonzero( self.hydro_z.exch_to_2d_link['link']==link_i )[0]
            exch_sel_sig=self.link_to_exch_sig[link_i] # np.nonzero( self.exch_to_2d_link['link']==link_i )[0]

            areas_z=A_z[exch_sel_z]
            areas_sig=A_sig[exch_sel_sig]

            # This hasn't been a problem, and it's slow - so skip it.
            # assert np.allclose( np.sum(areas_z), np.sum(areas_sig) ) # sanity

            # aggregated z-level grid doesn't necessarily have the same sign for all of the
            # exchanges in a column, but sigma does.  Go ahead and flip signs as needed
            # trouble with sgn - exch_sel_z goes up to 4727, while exch_to_2d_link 
            q_z = Q_z[exch_sel_z] * self.hydro_z.exch_to_2d_link['sgn'][exch_sel_z]

            Q_sig[exch_sel_sig] = rediscretize(areas_z,q_z,self.n_sigma,
                                               frac_samples=frac_samples,
                                               intensive=False)

        # Vertical fluxes:
        # starting with volume now, apply the horizontal fluxes to get predicted volumes
        # for the next step.  Step surface to bed, any discrepancy between predicted and
        # the reported Vnext should be vertical flux.  

        poi0=self.pointers-1
        Vnow=self.volumes(t)
        dt=self.t_secs[1] - self.t_secs[0]
        Vnext=self.volumes(t+dt)

        Vpred=Vnow.copy()

        if 0: # reference implementation
            for exch in range(self.n_exch_x):
                seg_from,seg_to = poi0[exch,:2]
                if seg_from>=0:
                    Vpred[seg_from] -= Q_sig[exch]*dt
                    Vpred[seg_to] += Q_sig[exch]*dt
        else: # vectorized implementation of that:
            seg_from=poi0[:self.n_exch_x,0]
            seg_to  =poi0[:self.n_exch_x,1]
            seg_from_valid=(seg_from>=0)

            # This fails because duplicates in seg_from/to overwrite each other.
            # simple vectorization fails due to duplicate indices in seg_from/to
            Vpred -= dt*np.bincount(seg_from[seg_from_valid],
                                    weights=Q_sig[:self.n_exch_x][seg_from_valid],
                                    minlength=len(Vpred))

            Vpred += dt*np.bincount(seg_to,
                                    weights=Q_sig[:self.n_exch_x],
                                    minlength=len(Vpred))

        for exch in range(self.n_exch_x+self.n_exch_y,self.n_exch)[::-1]:
            seg_from,seg_to = poi0[exch,:2]
            if seg_from<0:
                # shouldn't happen, as we're not yet considering vertical boundary exchanges,
                # and besides there shouldn't be adjustments to boundary fluxes, anyway.
                continue
            Vsurplus=Vpred[seg_to] - Vnext[seg_to]
            Q_sig[exch] = -Vsurplus / dt
            Vpred[seg_from] += Vsurplus
            Vpred[seg_to]   -= Vsurplus

        # if 0: # vectorization more complicated here - may return...
        #     exchs=np.arange(self.n_exch_x+self.n_exch_y,self.n_exch)[::-1]
        #     segs_from,segs_to = poi0[exchs,:2]
        #     assert np.all(segs_from>=0)
        #     
        #     Vsurplus=Vpred0[segs_to] - Vnext[segs_to]
        #     Q_sig0[exchs] = -Vsurplus / dt
        # 
        #     for exch in :
        #         Vpred[seg_from] += Vsurplus
        #         Vpred[seg_to]   -= Vsurplus
            
        rel_err = (Vpred - Vnext) / (1+Vnext)
        rel_err = np.abs(rel_err)
        rel_err_thresh=0.5
        if rel_err.max() >= rel_err_thresh:
            self.log.warning("Vertical fluxes still had relative errors up to %.2f%%"%( 100*rel_err.max() ) )
            self.log.warning("  at t=%s  (ti=%d)"%(t, self.t_sec_to_index(t)))

            # It's possible that precip or mass limits from the hydro code yield Vpred which go negative.
            # this would be bad news for dwaq scalar transport.  Find any water columns which have a
            # segment that goes negative, and fully redistribute the verical fluxes to have the new volumes
            # equal throughout the water column.

            # find the segments with a violation:
            bads = np.nonzero( rel_err>= rel_err_thresh )[0]
            bad_elts = np.unique( self.seg_to_2d_element[bads] )
            self.log.warning(" Bad 2d elements: %s"%str(bad_elts))
            for bad_elt in bad_elts:
                segs=np.nonzero( self.seg_to_2d_element==bad_elt )[0]
                # assumption of evenly spaced, no vertical boundaries, etc.
                exchs=self.n_exch_x + self.n_exch_y + bad_elt + np.arange(self.n_sigma-1)*self.n_2d_elements
                assert np.all( self.seg_to_2d_element[poi0[exchs,:2]] == bad_elt ) # sanity check

                # Q_sig as it stands leads to volumes Vpred[segs].
                # we'd like for it to lead to volumes Vpred[segs].mean()
                # 
                netQ_correction = (Vpred[segs] - Vpred[segs].mean()) / dt

                # This seems like the right approach, but leaves Q_sig[exchs]==[0,...]
                # which is suspicious
                # but - this is a single layer in hydro_z, so all horizontal fluxes will be
                # evenly divided across the sigma layers, thus there is no gradient in transport
                # to drive vertical velocities.
                # I think it's correct

                # drop the last one - it's interpretation is a flux out of the bed cell,
                # to some cell below it.  It had better be close to zero...
                Q_sig[exchs] += np.cumsum(netQ_correction)[:-1]

                Vpred[segs] = Vpred[segs].mean()

        Vpred_min=Vpred.min()
        if Vpred_min < 0.0:
            # normalize by area to see just how bad these are:
            plan_areas=self.planform_areas().evaluate(t=t).data # [Nseg]
            dz_pred=Vpred / plan_areas

            # compare to the errors already in hydro_z:
            z_errors=self.check_hydro_z_conservation(ti=self.t_sec_to_index(t))
            
            self.log.warning("Some predicted volumes are negative, for min(dz)=%f at seg %d"%(dz_pred.min(),
                                                                                              np.argmin(dz_pred)))
            # assert dz_pred.min()>self.dz_deficit_threshold # -0.001
            # no need to sum over water column in the dz_pred figures, since it's evenly distributed
            # just multiply by n_sigma.
            # make sure we're not more negative than the size of the errors in hydro_z. Note that
            # this is not quite apples-to-apples - z_errors will be worse since it is the error between
            # predicted thickness and prescribed thickness, while dz_pred is a deficit below 0 volume.
            assert self.n_sigma*dz_pred.min() > z_errors.min()
            self.log.warning(" Apparently the erros in the z-layer model are at least as bad")
        
        return Q_sig

    def check_hydro_z_conservation(self,ti):
        """ For reality checks on flows above, reach back to the z-layer 
        hydro and see if there were already continuity errors.  

        Given a time index into t_secs, perform continuity check on ti => ti+1
        and return the error in the "prediction", per-element in terms of thickness.
        So if the fluxes suggest that a water column went from 0.10m to -0.05m thick,
        but the volume data suggests it just went from 0.10m to 0.01m, then that element
        would get an error of -0.06m.
        """
        hyd=self.hydro_z
        QtodV,QtodVabs=hyd.mats_QtodV()

        t_last=hyd.t_secs[ti]
        t_next=hyd.t_secs[ti+1]
        Qlast=hyd.flows( t_last )
        Vlast=hyd.volumes( t_last )
        Vnow=hyd.volumes( t_next )

        plan_areas=hyd.planform_areas()
        seg_plan_areas=plan_areas.evaluate(t=t_last).data

        dt=t_next - t_last
        dVmag=QtodVabs.dot(np.abs(Qlast)*dt)
        Vpred=Vlast + QtodV.dot(Qlast)*dt

        err=Vpred - Vnow 
        valid=(Vnow+dVmag)!=0.0

        # for the purposes of the sigmify code, want this in water columns:
        hyd.infer_2d_elements() 

        sel=hyd.seg_to_2d_element>=0
        elt_err=np.bincount(hyd.seg_to_2d_element[sel],
                            weights=err[sel]/seg_plan_areas[sel],
                            minlength=hyd.n_2d_elements)
        return elt_err
    
    def segment_interpolator(self,t_sec,scalar_z):
        """ 
        Generic segment scalar aggregation
        t_sec: simulation time, integer seconds
        scalar_z: values from the z grid
        """
        orig_t_sec=t_sec
        t_sec=utils.nearest_val(self.t_secs,orig_t_sec)
        dt=self.t_secs[1] - self.t_secs[0]
        if abs(orig_t_sec-t_sec) > 1.5*dt:
            self.log.warning("segment_interpolator: requested time and my time off by %.2f steps"%( (orig_t_sec-t_sec)/dt ))
        
        self.infer_2d_elements()
        interp_scalars=np.zeros(self.n_seg,'f4')

        vol_z=self.hydro_z.volumes(t_sec)

        elt_to_seg_z=self.elt_to_seg_z
        
        # Start with super-slow approach - looping through elements
        # yes, indeed, it is super slow.
        frac_samples=np.linspace(0,1,self.n_sigma+1)
        for elt in range(self.n_2d_elements):
            # this is going to hurt:
            seg_sel_z=elt_to_seg_z[elt] # np.nonzero( self.hydro_z.seg_to_2d_element==elt )[0]
            seg_v_z=vol_z[seg_sel_z]
            seg_scal_z=scalar_z[seg_sel_z]
            
            seg_scal_sig = rediscretize(seg_v_z,seg_scal_z,self.n_sigma,
                                        intensive=True,
                                        frac_samples=frac_samples)
            
            # stripe it out across the nice evenly spaced layers:
            interp_scalars[elt::self.n_2d_elements]=seg_scal_sig

            #plt.figure(3).clf()
            #plt.plot(frac_sum,s_mass,'k-o')
            #plt.figure(2).clf()
            #plt.bar(left=seg_vsum_z-seg_v_z,width=seg_v_z,height=seg_scal_z)
            #plt.bar(left=frac_samples[:-1],width=1./self.n_sigma,height=seg_scal_sig,
            #        color='g',alpha=0.5)

        return interp_scalars
    def add_parameters(self,hyd):
        for p,param in iteritems(self.hydro_z.parameters(force=False)):
            if p=='surf':
                # copy top value down to others
                hyd[p] = self.param_z_copy_from_surface(param)
            else:
                self.log.info("Adding hydro parameter with z interpolation %s"%p)
                hyd[p] = self.param_z_interpolate(param)
        self.log.info("Done with Hydro::add_parameters()")
        return hyd
    def param_z_interpolate(self,param_z):
        if isinstance(param_z,ParameterSpatioTemporal):
            def interped(t_sec,self=self,param_z=param_z):
                return self.segment_interpolator(t_sec=t_sec,
                                                 scalar_z=param_z.evaluate(t=t_sec).data)
            # this had been using self.t_secs, but for temperature, and probably in general,
            # we should respect the original times
            return ParameterSpatioTemporal(func_t=interped,
                                           times=param_z.times,
                                           hydro=self)
        elif isinstance(param_z,ParameterSpatial):
            # interpolate based on the initial volumes
            seg_sig=self.segment_interpolator(t_sec=self.t_secs[0],
                                              scalar_z=param_z.data)
            return ParameterSpatial(per_segment=seg_sig,hydro=self)
        else:
            return param_z # constant or only time-varying
    def param_z_copy_from_surface(self,param_z):
        # all we know how to deal with so far:
        assert isinstance(param_z,ParameterSpatial)
        per_seg_z=param_z.data
        self.infer_2d_elements()
        per_seg_sig = np.tile( per_seg_z[:self.n_2d_elements], self.n_sigma )
        return ParameterSpatial(per_segment=per_seg_sig,hydro=self)
        
    def planform_areas(self):
        return self.param_z_copy_from_surface(self.hydro_z.planform_areas())
    def infer_2d_elements(self):
        # easy!
        if self.seg_to_2d_element is None:
            self.hydro_z.infer_2d_elements()
            self.n_2d_elements=self.hydro_z.n_2d_elements
            self.seg_to_2d_element = np.tile( np.arange(self.n_2d_elements),
                                              self.n_sigma )
            self.seg_k = np.repeat( np.arange(self.n_sigma), self.n_2d_elements )
        return self.seg_to_2d_element
    def infer_2d_links(self):
        # this is pre-computed in init_exchanges / delegated to
        # hydro_z
        return

    def grid(self):
        return self.hydro_z.grid()

    def get_geom(self):
        # copy all of the 2D info from the z-layer grid, and just slide
        # in appropriate sigma layer info for the vertical
        ds=self.hydro_z.get_geom()

        bounds = np.linspace(0,-1,1+self.n_sigma)
        centers= 0.5*(bounds[:-1] + bounds[1:])

        # remove these first, and xarray forgets about the old size for
        # this dimension allowing it to be redefined
        del ds['nFlowMesh_layers_bnds']
        del ds['nFlowMesh_layers']

        ds['nFlowMesh_layers']=xr.DataArray( centers,
                                             dims=['nFlowMesh_layers'],
                                             attrs=dict(standard_name="ocean_sigma_coordinate",
                                                        long_name="elevation at layer midpoints",
                                                        formula_terms="sigma: nFlowMesh_layers eta: eta depth: FlowElem_bl",
                                                        positive="up" ,
                                                        units="m" ,
                                                        bounds="nFlowMesh_layers_bnds"))
        
        # order correct?
        bounds_d2=np.array( [bounds[:-1], 
                             bounds[1:]] ).T
        # trying this without introducing the duplicate dimensions
        # this syntax avoids issues with trying interpolate between coordinates
        ds['nFlowMesh_layers_bnds']=( ('nFlowMesh_layers','d2'), bounds_d2 )

        if 'nFlowMesh_layers2' in ds:
            ds=ds.drop('nFlowMesh_layers2')

        return ds
    
    #  depths and bottom_depths: I think generic implementations will be sufficient


    
class Substance(object):
    _scenario=None # set by scenario
    active=True
    initial=None

    @property
    def scenario(self):
        return self._scenario
    @scenario.setter
    def scenario(self,s):
        self._scenario=s
        self.initial.scenario=s

    def __init__(self,initial=None,name=None,scenario=None,active=True):
        self.name = name or "unnamed"
        if initial is None:
            self.initial=Initial(default=0.0)
        else:
            self.initial=Initial.from_user(initial)
        self.scenario=scenario
        self.active=active
        self.initial.scenario=scenario

    def lookup(self):
        if self.scenario:
            return self.scenario.lookup_item(self.name)
        else:
            return None

    def copy(self):
        # assumes that name and scenario will be set by the caller separately
        return Substance(initial=self.initial,active=self.active)

class Initial(object):
    """ 
    descriptions of initial conditions.
    Initial() # default to 0
    Initial(10) # default to 10
    Initial(seg_values=[Nseg values]) # specify spatially varying directly
    ic=Initial()
    ic[1] = 12 # default to 0, segment 12 (0-based) gets 12.

    for this last syntax, 1 and 12 just have to work for numpy array assignment.
    """
    scenario=None # Substance will set this.
    def __init__(self,default=0.0,seg_values=None):
        assert np.isscalar(default)
        self.default=default
        self.seg_values=seg_values

    def __setitem__(self,k,v):
        # print "Setting initial condition for segments",k,v
        if self.seg_values is None:
            self.seg_values = np.zeros(self.scenario.hydro.n_seg,'f4')
            self.seg_values[:] = self.default

        self.seg_values[k]=v

    # def eval_for_segment(self,seg_idx):
    #     if self.segment is not None:
    #         return self.segment[seg_idx]
    #     else:
    #         return self.d

    # just needs to have seg_values populated before the output is generated.
    # what is the soonest that we know about hydro geometry?
    # the substances know the scenario, but that's before 

    @staticmethod
    def from_user(v):
        """
        When an initial condition is passed to a substance, this method takes 
        care of turning that into an instance of Initial.
        """
        if isinstance(v,Initial):
            return v
        elif isinstance(v,np.ndarray):
            return Initial(seg_values=v)
        elif np.isscalar(v):
            return Initial(default=v)
        else:
            raise Exception("Not sure how to interpret initial value '%s'"%v)

        
class ModelForcing(object):
    """ 
    holds some common code between BoundaryCondition and 
    Load.
    """
    class_serial=0 # to generate names for data files
    
    scenario=None # set by the scenario

    # Control whether data is written inline in the input file or
    # into a separate data file which is then included in the main input
    # file.
    separate_data_file=True
    # if not None and separate_data_file is True, then create forcing
    # files in the named subdirectory (created on demand).
    data_file_subdirectory='forcing'
    
    def __init__(self,items,substances,data):
        if isinstance(substances,str):
            substances=[substances]
        # items have a tendency to come in as bytes.
        # ideally support, in both python 2 and 3,

        if isinstance(items,six.string_types+(bytes,)):
            items=[items]
        items=[normalize_to_str(item) for item in items]

        self.items=items
        self.substances=substances
        self.data=data
        self.my_serial=ModelForcing.next_serial()

    @classmethod
    def next_serial(cls):
        cls.class_serial+=1
        return cls.class_serial
        
    @property
    def safe_name(self):
        """ identifying name which can be used in filenames.
        """
        # There isn't a great choice at this level - 
        return "forcing-%03d"%self.my_serial
        
    def text_item(self):
        lines=['ITEM']

        for item in self.items:
            item=self.fmt_item(item)
            lines.append("  '%s'"%item )
        return "\n".join(lines)

    def text_substances(self):
        lines=['CONCENTRATION'] # used to be plural - should be okay this way
        lines.append("   " + "  ".join(["'%s'"%s for s in self.substances]))
        return "\n".join(lines)

    @property
    def supporting_file(self):
        """ base name of the supporting binary file (dir name will come from scenario,
        and possibly self.data_file_subdirectory
        """
        return self.scenario.name + "-" + self.safe_name + ".dat"
    
    def text_data(self,write_supporting=True):
        data=self.render_data_to_text()

        if len(data)>100 and self.separate_data_file:
            if write_supporting:
                if self.data_file_subdirectory is not None:
                    dir_name=os.path.join(self.scenario.base_path,
                                          self.data_file_subdirectory)
                    rel_path=os.path.join(self.data_file_subdirectory,
                                          self.supporting_file)
                else:
                    dir_name=self.scenario.base_path
                    rel_path=self.supporting_file

                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                    
                fn=os.path.join(dir_name,
                                self.supporting_file)
                
                with open(fn,'wt') as fp:
                    fp.write(data)
            return "INCLUDE '%s' ; external time series data"%rel_path
        else:
            return data
        
    def render_data_to_text(self):
        lines=[]

        # FIX: somewhere we should limit the output to the simulation period plus
        # some buffer.
        data=self.data

        if isinstance(data,pd.Series):
            # coerce to tuple with datenums
            data=(utils.to_dnum(data.index.values),
                  data.values)

        if isinstance(data,tuple):
            data_t,data_values=data
            lines.append('TIME {}'.format(self.time_interpolation))
        else:
            data_t=[None]
            data_values=np.asarray(data)

        lines.append('DATA')

        for ti,t in enumerate(data_t):
            if len(data_t)>1: # time varying
                step_data=data_values[ti,...]
                lines.append(self.scenario.fmt_datetime(t)) # like 1990/08/05-12:30:00
            else:
                step_data=data_values

            for item_i,item in enumerate(self.items):
                if step_data.ndim==2:
                    item_data=step_data[item_i,:]
                else:
                    item_data=step_data

                line=[]
                for sub_i,substance in enumerate(self.substances):
                    if item_data.ndim==1:
                        item_sub_data=item_data[sub_i]
                    else:
                        item_sub_data=item_data

                    line.append("%g"%item_sub_data) 
                line.append("; %s"%self.fmt_item(item))
                lines.append(" ".join(line))
        return "\n".join(lines)

    def fmt_item(self,item):
        return str(item) # probably not what you want...

    def text(self,write_supporting=True):
        lines=[self.text_item(),
               self.text_substances(),
               self.text_data(write_supporting=write_supporting)]
        return "\n".join(lines)

class BoundaryCondition(ModelForcing):
    """ descriptions of boundary conditions """
    
    # only used if time varying - can also be blank or 'BLOCK'
    time_interpolation='LINEAR' 

    def __init__(self,boundaries,substances,data):
        """
        boundaries: list of string id's of individual boundary exchanges, 
           types of boundaries as strings,
           index (negative from -1) of boundary exchanges
        Strings should not be pre-quoted

        substances: list of string names of substances.

        data: depends on type -
          constant: apply the same value for all items, all substances
          1d array: apply the same value for all items, but different values
            for different substances
          2d array: data[i,j] is for item item i, substance j

          tuple (datetime 1d array,values 2d or 3d array): time series.  t_secs
          gives the time of each set of values.  first dimension of values
          is time, and must match length of t_secs.
           2nd dimension is item  (had been switched with substance)
           3rd dimension is substance

          pandas Series with DatetimeIndex - only works for scalar timeseries.

        datetimes are specified as in Scenario.as_datetime - DateTime instance, integer seconds
         or float datenum.
        """
        super(BoundaryCondition,self).__init__(items=boundaries,substances=substances,data=data)

    bdefs=None
    def fmt_item(self,bdry):
        if self.bdefs is None:
            self.bdefs=self.scenario.hydro.boundary_defs()
        if isinstance(bdry,int):
            bdry=self.bdefs[-1-bdry]['id']
        return bdry
        
class Discharge(object):
    """ 
    Simple naming for load/withdrawal location.  Most of the
    work is in Load.

    No support yet for things like SURFACE, BANK or BED.
    """
    def __init__(self,
                 seg_id=None, # directly specify a segment, 0-based
                 element=None,k=0, # segment from element,k combination
                 load_id=None, # defaults to using seg_id
                 load_name=None, # defaults to load_id,
                 load_type=None, # defaults to load_id,
                 option=None): # defaults to 'MASS' substances
        self.scenario=None # will be set by Scenario
        self.element=element
        self.seg_id=seg_id
        self.k=k
        self.load_id=load_id
        self.load_name=load_name
        self.load_type=load_type
        self.option=option
    def update_fields(self):
        """ since some mappings are not available (like segment name => id_
        until we have a scenario, calling this will update relevant fields
        when a scenario is available.
        """
        if self.seg_id is None:
            assert self.element is not None
            self.seg_id=self.scenario.hydro.segment_select(element=self.element,k=self.k)[0]

        self.load_id=self.load_id or "seg-%d"%self.seg_id
        self.load_name=self.load_name or self.load_id
        self.load_type=self.load_type or self.load_id
        self.option=self.option or "MASS"

    def text(self):
        self.update_fields()
        fmt=" {seg} {self.option} '{self.load_id}' '{self.load_name}' '{self.load_type}' "
        return fmt.format(seg=self.seg_id+1, # to 1-based
                          self=self)

class Load(ModelForcing):
    """
    descriptions of mass sources/sinks (loads, withdrawals).

    When setting the load on a 'MASS' discharge (the default), d-waq expects
    data with units of g/s.
    """
    
    # only used if time varying - can also be blank or 'BLOCK'
    time_interpolation='LINEAR' 

    def __init__(self,discharges, 
                 substances,
                 data):
        """
        see ModelForcing or BoundaryCondition docstring
        """
        super(Load,self).__init__(items=discharges,
                                  substances=substances,
                                  data=data)

    def fmt_item(self,disch):
        if isinstance(disch,int):
            disch=self.scenario.discharges[disch].load_id
        if isinstance(disch,Discharge):
            disch=disch.load_id
        assert(isinstance(disch,str))
        return disch

    # text_substances() from parent class
    # text_data() from parent class
    # text() from parent class


class Parameter(object):
    scenario=None # to be set by the scenario
    def __init__(self,scenario=None,name=None,hydro=None):
        self.name = name or "unnamed"  # may be set later.
        self.scenario=scenario # may be set later
        self._hydro = hydro
        
    @property
    def safe_name(self):
        """ reformatted self.name which can be used in filenames 
        """
        return self.name.replace(' ','_').lower()

    @property
    def hydro(self):
        if self._hydro is not None:
            return self._hydro
        
        try:
            return self.scenario.hydro
        except AttributeError:
            return None
    @hydro.setter
    def hydro(self,value):
        self._hydro=value
    
    def text(self,write_supporting=True):
        """
        write_supporting=True will create any relevant binary files
        at the same time.
        """
        raise WaqException("To be implemented in subclasses")
    def evaluate(self,**kws):
        """ interface is evolving, but roughly, subclasses can
        interpret elements of kws as they wish, returning a presumably
        narrower parameter object.  Example usage would be to take
        a ParameterSpatioTemporal, and evaluate with t=<some time>,
        returning a ParameterSpatial.
        """ 
        return self

class ParameterConstant(Parameter):
    """
    A constant in time, constant in space parameter
    """
    def __init__(self,value,scenario=None,name=None,hydro=None):
        super(ParameterConstant,self).__init__(name=name,scenario=scenario,hydro=hydro)
        self.data=self.value=value

    def text(self,write_supporting=True):
        return "CONSTANTS  '{}'  DATA {:.5e}".format(self.name,self.value)


class ParameterSpatial(Parameter):
    """ Process parameter which varies only in space - same 
    as DWAQ's 'PARAMETERS'
    """
    # To support runs with a layered bed or streamlined specification of parameters
    # If None, should probably specify data for all segments, water and bed.
    # if '__default__', will use 'water-grid' if n_bottom_layers>0, otherwise the
    #  same as None.
    # any other value is used as the name of the input grid directly.
    grid_name=DEFAULT
    inline_data=False
    def __init__(self,per_segment=None,par_file=None,scenario=None,name=None,hydro=None,
                 grid_name=DEFAULT,inline_data=False):
        super(ParameterSpatial,self).__init__(name=name,scenario=scenario,hydro=hydro)
        if par_file is not None:
            self.par_file=par_file
            with open(par_file,'rb') as fp:
                fp.read(4) # toss zero timestamp
                # no checks for proper size....living on the edge
                # note that with multiple grids, the correct number of elements is not
                # simply scenario.n_seg
                per_segment=np.fromfile(fp,'f4')
        self.data=per_segment
        self.grid_name = grid_name
        self.inline_data = inline_data

    @property
    def supporting_file(self):
        """ base name of the supporting binary file (dir name will come from scenario)
        """
        return self.scenario.name + "-" + self.safe_name + ".par"
    def text(self,write_supporting=True):
        if write_supporting and not self.inline_data:
            self.write_supporting()
            
        if self.grid_name==DEFAULT:
            self.grid_name = self.scenario.water_grid

        if self.grid_name is None:
            grid_text='ALL'
        else:
            grid_text="INPUTGRID '{self.grid_name}'"

        if self.inline_data:
            data_text="DATA " + "\n".join( ["%g"%f for f in self.data])
        else:
            data_text="BINARY_FILE '{self.supporting_file}'"
        # if there is a bottom grid, I think this will expect data for both water segments and
        # sediment segments unless a specific INPUTGRID is given. Even then, having trouble
        # using a non-water grid here.
        return ("PARAMETERS '{self.name}' " + grid_text + " " + data_text).format(self=self)
        
    def write_supporting(self):
        with open(os.path.join(self.scenario.base_path,self.supporting_file),'wb') as fp:
            # leading 'i4' with value 0.
            # I didn't see this in the docs, but it's true of the 
            # .par files written by the GUI. probably this is to make the format
            # the same as a segment function with a single time step.
            fp.write(np.array(0,dtype='i4').tobytes())
            fp.write(np.asarray(self.data,dtype=np.float32).tobytes())

    def evaluate(self,**kws):
        if 'seg' in kws:
            return ParameterConstant( self.data[kws.pop('seg')] )
        else:
            return self

class ParameterTemporal(Parameter):
    """ Process parameter which varies only in time
    aka DWAQ's FUNCTION
    """
    # Control whether data is written inline in the input file or
    # into a separate data file which is then included in the main input
    # file.
    separate_data_file=True
    
    def __init__(self,times,values,scenario=None,name=None,hydro=None):
        """
        times: [N] sized array, 'i4', giving times as seconds after time0
        values: [N] sized array, 'f4', giving function values.
        """
        super(ParameterTemporal,self).__init__(name=name,scenario=scenario,hydro=hydro)
        self.times=times
        self.values=values
    def text(self,write_supporting=False):
        data=self.text_data()
        if self.separate_data_file:
            if write_supporting:
                with open(os.path.join(self.scenario.base_path,self.supporting_file),'wt') as fp:
                    fp.write(data)
            return "INCLUDE '%s' ; external time series data"%self.supporting_file
        else:
            return data
        
    def text_data(self):
        lines=["FUNCTIONS '{}' BLOCK DATA".format(self.name),
               ""]
        for t,v in zip(self.times,self.values):
            lines.append("{}  {:e}".format(self.scenario.fmt_datetime(t),v) )
        return "\n".join(lines)
        

    
    @property
    def supporting_file(self):
        """ base name of the supporting binary file, (no dir. name) """
        return self.scenario.name + "-" + self.safe_name + ".ts"
    
    def evaluate(self,**kws):
        if 't' in kws:
            t=kws.pop('t')
            tidx=np.searchsorted(self.times,t)
            return ParameterConstant( self.values[tidx] )
        else:
            return self


class ParameterSpatioTemporal(Parameter):
    """ Process parameter which varies in time and space - aka DWAQ 
    SEG_FUNCTIONS
    """
    interpolation='LINEAR' # or 'BLOCK'
    warned_2d_to_3d=False # track one-time warning when data is supplied as 2D
    # in case of multiple grids. Note that the input file manual does not explicitly describe
    # this syntax, but it is used in the sediment manual.
    grid_name=DEFAULT

    reference_originals=False
    
    def __init__(self,times=None,values=None,func_t=None,scenario=None,name=None,
                 seg_func_file=None,enable_write_symlink=None,n_seg=None,
                 hydro=None,grid_name=DEFAULT,reference_originals=None):
        """
        times: [N] sized array, 'i4', giving times in system clock units
          (typically seconds after time0)
        values: [N,n_seg] array, 'f4', giving function values

        or func_t, which takes a time as 'i4', and returns the values for
        that moment

        or seg_func_file, a path to an existing file.  if enable_write_symlink
          is True, then write_supporting() will symlink to this file.  otherwise
          it is copied.

        note that on write(), a subset of the times may be used based on 
        start/stop times of the associated scenario.  Still, on creation, should
        pass the full complement of times for which data exists (of course consistent
        with the shape of data when explicit data is passed)

        reference_originals: override hydro setting. True means that instead of symlinking
          to an existing file use the full path to it.
        """
        if seg_func_file is None:
            assert(times is not None)
            assert(values is not None or func_t is not None)
        super(ParameterSpatioTemporal,self).__init__(name=name,scenario=scenario,hydro=hydro)
        self.func_t=func_t
        self._times=times
        self.values=values
        self.seg_func_file=seg_func_file
        # generally follow hydro on whether to use symlinks, but allow
        # an explicit option, too.
        if enable_write_symlink is None:
            if hydro is not None:
                self.enable_write_symlink=hydro.enable_write_symlink
            else:
                self.enable_write_symlink=False
        else:
            self.enable_write_symlink=enable_write_symlink

        if reference_originals is None:
            if hydro is not None:
                self.reference_originals = hydro.reference_originals
            else:
                pass # leave as default from class
        else:
            self.reference_originals = reference_originals
            
        self._n_seg=n_seg # only needed for evaluate() when scenario isn't set
        self.grid_name=grid_name

    def copy(self):
        return ParameterSpatioTemporal(times=self._times,
                                       values=self.values,
                                       func_t=self.func_t,
                                       scenario=self.scenario,
                                       name=self.name,
                                       seg_func_file=self.seg_func_file,
                                       enable_write_symlink=self.enable_write_symlink,
                                       reference_originals=self.reference_originals,
                                       n_seg = self._n_seg,
                                       hydro=self.hydro)

    # goofy helpers when n_seg or times can only be inferred after instantiation
    @property
    def n_seg(self):
        if (self._n_seg is None):
            try:
                # awkward reference to hydro.
                self._n_seg = self.hydro.n_seg
            except AttributeError:
                pass
        return self._n_seg

    @property
    def times(self):
        if (self._times is None) and (self.seg_func_file is not None):
            self.load_from_segment_file()
        return self._times

    @property
    def supporting_file(self):
        raise Exception("Should be using supporting_path()")
        if self.reference_originals:
            HERE
        else:
            return self.scenario.name + "-" + self.safe_name + ".seg"
    
    @property
    def supporting_path(self):
        if self.reference_originals and self.seg_func_file is not None:
            return self.seg_func_file
        else:
            # Update this so we know whether or not the returned path
            # is the original file or the place to write a new file.
            self.reference_originals=False
            basename=self.scenario.name + "-" + self.safe_name + ".seg"
            return os.path.join(self.scenario.base_path,basename)
    
    def text(self,write_supporting=True):
        if write_supporting:
            self.write_supporting()
            
        if self.grid_name==DEFAULT:
            self.grid_name=self.scenario.water_grid

        # if we're writing a new file, it will be in scen.base_path, and
        # we just get back to the basename here.
        # But if we're referencing a file elsewhere, this gives a relative
        # path for it.
        supporting_file = os.path.relpath(self.supporting_path, self.scenario.base_path)

        print("Writing paramater spatiotemporal: supporting_file is %s"%supporting_file)
        
        if self.grid_name is None:
            return ("SEG_FUNCTIONS '{self.name}' {self.interpolation}"
                    " ALL BINARY_FILE '{supporting_file}'").format(self=self,supporting_file=supporting_file)
        else:
            return ("SEG_FUNCTIONS '{self.name}' {self.interpolation}"
                    " INPUTGRID '{self.grid_name}' BINARY_FILE '{supporting_file}'").format(self=self,supporting_file=supporting_file)
            
    def write_supporting_try_symlink(self):
        if self.seg_func_file is not None:
            dst=self.supporting_path
            # With the updated logic in supporting_path(), reference_originals is only
            # true when the returned path exists, making this assertion redundant.
            if self.reference_originals:
                assert os.path.exists(dst),"Expected to use existing file, but %s does not exist"%dst
                return True
            
            if os.path.lexists(dst):
                if self.scenario.overwrite:
                    os.unlink(dst)
                else:
                    raise Exception("%s exists, and scenario.overwrite is False"%dst)
                
            if self.enable_write_symlink:
                rel_symlink(self.seg_func_file,dst)
            else:
                shutil.copyfile(self.seg_func_file,dst)
            return True
        else:
            return False
        
    def write_supporting(self):
        if self.write_supporting_try_symlink():
            return
        
        target=self.supporting_path

        assert not self.reference_originals,"Bailing before writing %s as it might be original input"%target
        
        # limit to the time span of the scenario
        tidxs=np.arange(len(self.times))
        datetimes=self.times*self.scenario.scu + self.scenario.time0
        start_i,stop_i = np.searchsorted(datetimes,
                                         [self.scenario.start_time,
                                          self.scenario.stop_time])
        start_i=max(0,start_i-1)
        stop_i =min(stop_i+1,len(tidxs))
        tidxs=tidxs[start_i:stop_i]
        msg="write_supporting: writing %d of %d timesteps."%(len(tidxs),
                                                             len(self.times))
        self.scenario.log.info(msg)

        # This is split out so that the parallel implementation can jump in just at this
        # point
        if os.path.lexists(target):
            if self.scenario.overwrite:
                os.unlink(target)
            else:
                raise Exception("%s exists, and scenario.overwrite is False"%target)

        with open(target,'wb') as fp:
            self.write_supporting_loop(tidxs,fp,name=target)

    def write_supporting_loop(self,tidxs,fp,name='n/a'):
        t_secs=self.times.astype('i4')
        
        for tidx in utils.progress(tidxs,msg=name+": %s"):
            t=t_secs[tidx]
            fp.write(t_secs[tidx].tobytes())
            if self.values is not None:
                values=self.values[tidx,:]
                if (len(values)==self.hydro.n_2d_elements) and (len(values)!=self.hydro.n_seg):
                    if not self.warned_2d_to_3d:
                        self.scenario.log.warning("Padding parameter %s from 2D to 3D"%self.safe_name)
                        self.warned_2d_to_3d=True
                    values=values[self.hydro.seg_to_2d_element]
            else:
                values=self.func_t(t)
            fp.write(values.astype('f4').tobytes())

    def load_from_segment_file(self,convert_time=False):
        """
        Set self.values and self._times from the segment function file.
        This uses memmap, so it should be fairly efficient and safe to
        do even when you only want a fraction of the data.

        convert_time: convert times to dt64
        """
        self._mmap_data=np.memmap(self.seg_func_file,
                                  dtype=[('t',np.int32),
                                         ('value',np.float32,self.n_seg)],
                                  mode='r')
        self._times=self._mmap_data['t']
        if convert_time:
            self._times = self._times*np.timedelta64(1,'s') + utils.to_dt64(self.hydro.time0)
            
        self.values=self._mmap_data['value']

    def evaluate(self,**kws):
        # This implementation is pretty rough - 
        # this class is really a mix of
        # ParameterSpatial and ParameterTemporal, yet it duplicates
        # the code from both of those here.

        if (self.seg_func_file is not None) and (self.values is None):
            self.load_from_segment_file()
                
        param=self

        if 't' in kws:
            t=kws.pop('t')
            if self.values is not None:
                tidx=np.searchsorted(self.times,t)
                # copy to avoid keeping extra data around too long
                param=ParameterSpatial( self.values[tidx,:].copy() )
            elif self.func_t is not None:
                param=ParameterSpatial( self.func_t(t) )
        elif 'seg' in kws:
            seg=kws.pop('seg')
            # Copy, since these are much smaller than the original data and
            # better not to force the original array to stay around.
            param=ParameterTemporal(times=self.times.copy(),values=self.values[:,seg].copy())
        if param is not self:
            # allow other subclasses to do fancier things
            return param.evaluate(**kws)
        else:
            return self
        
    def lowpass(self,lp_secs,volume_threshold=1.0,pad_mode='constant'):
        """
        segments with a volume less than the given threshold are removed 
        from the filtering, replaced by linear interpolation.
        pad_mode: 'constant' pads the time series with the first/last values
                  'zero' pads with zeros.
        """
        dt=np.median(np.diff(self.times))
        if dt>lp_secs:
            return self

        # brute force - load all of the data at once.
        # for a year of 30 minute data over 4k segments,
        # loading the data takes 10s, filtering takes 10s.
        values=[]

        for ti,t in enumerate(self.times):
            if ti%5000==0:
                print("%d / %d"%(ti,len(self.times)))
            spatial=self.evaluate(t=t).data
            if volume_threshold>0:
                volumes=self.hydro.volumes(t)
                mask=(volumes<volume_threshold)
                spatial=spatial.copy() # in case evaluate gave us a reference/view
                spatial[mask] = np.nan
            values.append(spatial)

        values=np.array(values)

        npad=int(5*lp_secs / dt)
        pad =np.ones(npad)

        for seg in range(self.n_seg):
            if pad_mode=='constant':
                prepad=values[0,seg] * pad
                postpad=values[-1,seg] * pad
            elif pad_mode=='zero':
                prepad=postpad=0*pad
            else:
                raise Exception("Bad pad_mode: %s"%pad_mode)
            padded=np.concatenate( ( prepad, 
                                     values[:,seg],
                                     postpad) )
            if volume_threshold>0:
                utils.fill_invalid(padded)
            # possible, especially with a dense-output z-level model
            # where some segments are below the bed and thus always nan
            # that there are still nans hanging out.  so explicitly call
            # them -999
            if np.isnan(padded[0]):
                values[:,seg]=-999
            else:
                lp_values=filters.lowpass(padded,
                                          cutoff=lp_secs,dt=dt)
                values[:,seg]=lp_values[npad:-npad] # trim the pad
        return ParameterSpatioTemporal(times=self.times,
                                       values=values,
                                       enable_write_symlink=False,
                                       reference_originals=False,
                                       n_seg=self.n_seg,
                                       # these probably get overwritten anyway.
                                       scenario=self.scenario,
                                       name=self.name)

def cast_to_parameter(v):
    if isinstance(v,Parameter):
        return v
    else:
        # casting rules:
        # This may get fancier over time, with potential support for
        # xarray, pandas, numpy, blah, blah.
        # Not at this point, though.
        return ParameterConstant(v)
    
# Options for defining parameters and substances:
# 1. as before - list attributes of the class
#    this is annoying because you can't alter the lists easily/safely 
#    until after object instantiation
# 2. as a method, returning a list or dict.  This gets closer, but 
#    then you can't modify values - it's stuck inside a method
# 3. init_* methods called on instantiation.  Just have to be clear
#    about the order of steps, what information is available when, etc.
#    you have just as much information available as when defining things
#    at class definition time.

class NamedObjects(OrderedDict):
    """ 
    utility class for managing collections of objects which
    get a name and a reference to the scenario or hydro
    """
    def __init__(self,sort_key=None,
                 cast_value=lambda x: x,
                 **kw):
        """
        sort_key: when iterating over the items, return sorted based on this
           attribute of each item.

        cast_value: a function which is applied to all values coming in via __setitem__

        a single additional keyword argument: a name-value pair which is set on 
          incoming values, after cast_value is applied.
        """
        super(NamedObjects,self).__init__()
        assert len(kw)==1
        
        self.parent_name=list(kw.keys())[0]
        self.parent=kw[self.parent_name]
        self.cast_value=cast_value
        
        self.sort_key=sort_key

    def normalize_key(self,k):
        try:
            return k.lower()
        except AttributeError:
            return k
   
    def __contains__(self,k):
        return super(NamedObjects,self).__contains__(self.normalize_key(k))
    
    def __setitem__(self,k,v):
        v=self.cast_value(v)
        v.name=k
        setattr(v,self.parent_name,self.parent) # v.scenario=self.scenario
        super(NamedObjects,self).__setitem__(self.normalize_key(k),v)
    def __getitem__(self,k):
        return super(NamedObjects,self).__getitem__(self.normalize_key(k))
    def __delitem__(self,k):
        return super(NamedObjects,self).__delitem__(self.normalize_key(k))
        
    def clear(self):
        for key in list(self.keys()):
            del self[key]

    def __iter__(self):
        """ optionally applies an extra level of sorting based on
        self.sort_key
        """
        orig=super(NamedObjects,self).__iter__()
        if self.sort_key is None:
            return orig
        else:
            # is this sort stable? as of python 2.2, yes!
            entries=list(orig)
            real_sort_key=lambda k: self.sort_key(self[k])
            entries.sort( key=real_sort_key )
            return iter(entries)
    # other variants are defined in terms of iter, so only
    # have to change the ordering in __iter__.
    # except that might have changed...
    def values(self):
        return [self[k] for k in self]
    def __add__(self,other):
        a=NamedObjects(**{self.parent_name:self.parent})
        for src in self,other:
            for v in src.values():
                a[v.name]=v
        return a

    # would be nice to change __iter__ behavior since name
    # is already an attribute on the items, but __iter__ is
    # central to the inner workings of other dict methods, and
    # hard to override safely.

class DispArray(object):
    """
    input file manual appendix, page 90 says dispersions file is
    time[i4], [ndisp,nqt] matrix of 'f4' - for each time step.
    nqt is total number of exchanges.
    """
    def __init__(self,name=None,substances=None,data=None,times=None):
        """
        typ. usage:
        scenario.dispersions['subtidal_K']=DispArray(substances='.*',data=xxx)

        name: label for the dispersion array, max len 20 char
        substances: list of substance names or patterns for which this array applies.
          can be a str, which is coerced to [str].  Interpreted as regular expression.
        Options for data, times:
          constant in time: data is an array of dispersion coefficients, one per exchange.
             times is None.
          unsteady: data is a 2D array, [time,exchange], and times gives corresponding
             timesteps in system time units, similar to ParameterSpatioTemporal, i.e. seconds since
             reference time.
        """
        if name is not None:
            self.name=name[:20]
        else:
            self.name=None
        if isinstance(substances,str):
            substances=[substances]
        self.patts=substances
        self.data=data
        self.times=times
        
    def matches(self,name):
        for patt in self.patts:
            if re.match(patt,name,flags=re.IGNORECASE):
                return True
        return False
    def text(self,write_supporting=True):
        if write_supporting:
            self.write_supporting()
        return "XXX PARAMETERS '{self.name}' ALL BINARY_FILE '{self.supporting_file}'".format(self=self)
    @property
    def safe_name(self):
        """ reformatted self.name which can be used in filenames 
        """
        return self.name.replace(' ','_')
    def supporting_file(self):
        """ base name of the supporting binary file (dir name will come from scenario)
        """
        return self.scenario.name + "-" + self.safe_name + ".par"

    def write_supporting(self):
        with open(os.path.join(self.scenario.base_path,self.supporting_file),'wb') as fp:
            # leading 'i4' with value 0.
            # I didn't see this in the docs, but it's true of the 
            # .par files written by the GUI. probably this is to make the format
            # the same as a segment function with a single time step.
            if self.times is None:  # Write constant in time.
                fp.write(np.array(0,dtype='i4').tobytes())
                fp.write(self.data.astype('f4').tobytes())
            else:
                for ti,t in enumerate(self.times):
                    fp.write(np.array(t,dtype='i4').tobytes())
                    fp.write(self.data[ti,:].astype('f4').tobytes())

def map_nef_names(nef):
    subst_names=nef['DELWAQ_PARAMS'].getelt('SUBST_NAMES',[0])

    elt_map={}
    real_map={} # map nc variable names to original names
    new_count=defaultdict(lambda: 0) # map base names to counts
    for i,name in enumerate(subst_names):
        name=name.decode()
        new_name=qnc.sanitize_name(name.strip()).lower()
        new_count[new_name]+=1
        if new_count[new_name]>1:
            new_name+="%03i"%(new_count[new_name])
        elt_map['SUBST_%03i'%(i+1)] = new_name
        real_map[new_name]=name
    return elt_map,real_map

        

class Scenario(scriptable.Scriptable):
    """
    A class wrapper for a Delft D-Water Quality (DWAQ)
    simulation.
    """
    name="tbd" # this is used for the basename of the various files.
    desc=('line1','line2','line3')

    # system clock unit. 
    time0=None
    # time0=datetime.datetime(1990,8,5) # defaults to hydro value
    scu=datetime.timedelta(seconds=1)
    time_step=None  # these are taken from hydro, unless specified otherwise
    start_time=None # 
    stop_time=None  # 

    log=logging # python logging 

    # backward differencing,
    # .60 => second and third keywords are set 
    # => no dispersion across open boundary
    # => lower order at boundaries
    integration_option="15.60"

    #  add quantities to default
    DEFAULT=DEFAULT
    mon_output= (DEFAULT,'SURF','LocalDepth') # monitor file
    grid_output=('SURF','LocalDepth')              # grid topo
    hist_output=(DEFAULT,'SURF','LocalDepth') # history file
    map_output =(DEFAULT,'SURF','LocalDepth')  # map file
    stat_output =() # default to no stat output

    map_formats=['nefis']
    history_formats=['nefis','binary']

    water_grid=None # if using old style layered bed then this should probably be 'water-grid'
    # if using old style layered bed this should be set to something. None indicates no old-style
    # bed and disables defining deep BCs. If bottom_grid is None but bottom_layers is set, assume
    # that DelwaqG will be used and allocate DelwaqG initial conditions accordingly.
    bottom_grid=None
    bottom_layers=[] # thickness of layers, used for delwaqg
    delwaqg_initial=None

    @property
    def n_bottom_layers(self):
        return len(self.bottom_layers)

    # not fully handled, but generally trying to overwrite
    # a run without setting this to True should fail.
    overwrite=False

    # if set, initial conditions are taken from a restart file given here
    restart_file=None
    
    # settings related to paths - a little sneaky, to allow for shorthand
    # to select the next non-existing subdirectory by setting base_path to
    # "auto"
    _base_path="dwaq"
    @property
    def base_path(self):
        if self._base_path=='auto':
            self._base_path=self.auto_base_path()            
        return self._base_path
    @base_path.setter
    def base_path(self,v):
        self._base_path=v

    # tuples of name, segment id
    # some confusion about whether that's a segment id or list thereof
    # monitor_areas=[ ('single (0)',[1]),
    #                 ('single (1)',[2]),
    #                 ('single (2)',[3]),
    #                 ('single (3)',[4]),
    #                 ('single (4)',[5]) ]
    # dwaq bug(?) where having transects but no monitor_areas means history
    # file with transects is not written.  so always include a dummy:
    # the output code handles adding 1, so these should be stored zero-based.
    monitor_areas=( ('dummy',[0]), ) 

    # e.g. ( ('gg_outside', [24,26,-21,-27,344] ), ...  )
    # where negative signs mean to flip the sign of that exchange.
    # note that these exchanges are ONE-BASED - this is because the
    # sign convention is wrapped into the sign, so a zero exchange would
    # be ambiguous.
    monitor_transects=()

    # These really shouldn't be this large.  in the update WaqModel they all
    # default to 0.0, much more sensible.
    base_x_dispersion=1.0 # m2/s
    base_y_dispersion=0.0 # m2/s not used in unstructured, but nonzero breaks scheme 24 performance.
    base_z_dispersion=1e-7 # m2/s

    # these default to simulation start/stop/timestep
    map_start_time=None
    map_stop_time=None
    map_time_step=None
    # likewise
    hist_start_time=None
    hist_stop_time=None
    hist_time_step=None
    # and more
    mon_start_time=None
    mon_stop_time=None
    mon_time_step=None

    # the 'type' of all deep sediment BCs, and the prefix for BC id,
    # followed by 1-based element number
    bottom_bc_prefix='deep bed'

    def __init__(self,hydro=None,**kw):
        self.log=logging.getLogger(self.__class__.__name__)

        # list of dicts.  moving towards standard setting of loads via src_tags
        # should be populated in init_substances(), gets used 
        self.src_tags=[]
        
        self.dispersions=NamedObjects(scenario=self)
        self.velocities=NamedObjects(scenario=self)

        self.hydro=hydro

        self.inp=InpFile(scenario=self)

        # set attributes here, before the init code might want to
        # use these settings (e.g. start/stop times)
        for k,v in iteritems(kw):
            try:
                getattr(self,k)
                setattr(self,k,v)
            except AttributeError:
                raise Exception("Unknown Scenario attribute: %s"%k)

        if len(self.bottom_layers) and not self.bottom_grid:
            # This can be set to None in which case DelwaqG will end up pulling
            # depth-uniform initial conditions from water column parameters with
            # S1* names.
            self.delwaqg_initial=make_delwaqg_dataset(self)

        self.parameters=self.init_parameters()
        if self.hydro is not None:
            self.hydro_parameters=self.init_hydro_parameters()
        self.substances=self.init_substances()
        assert self.substances is not None, "init_substances() did not return anything."
        self.init_bcs()
        self.init_loads()

    # scriptable interface settings:
    cli_options="hp:"
    def cli_handle_option(self,opt,val):
        if opt=='-p':
            print("Setting base_path to '%s'"%val)
            self.base_path=val
        else:
            super(Scenario,self).cli_handle_option(opt,val)
        
    def auto_base_path(self):
        ymd=datetime.datetime.now().strftime('%Y%m%d')        
        prefix='dwaq%s'%ymd
        for c in range(100):
            base_path=prefix+'%02d'%c
            if not os.path.exists(base_path):
                return base_path
        else:
            raise Exception("Possible run-away directory naming")
        
    @property
    def n_substances(self):
        return len(self.substances)
    @property
    def n_active_substances(self):
        return len( [sub for sub in self.substances.values() if sub.active] )
    @property
    def n_inactive_substances(self):
        return len( [sub for sub in self.substances.values() if not sub.active] )
    
    @property
    def multigrid_block(self):
        """ 
        inserted verbatim in section 3 of input file.
        """
        # appears that for a lot of processes, hydro must be dense wrt segments
        # exchanges need not be dense, but sparse exchanges necessitate explicitly
        # providing the number of layers.  And that brings us to this stanza:

        # sparse_layers is actually an input flag to the aggregator
        # assert not self.hydro.sparse_layers
        # instead, test this programmatically, and roughly the same as how dwaq will test
        self.hydro.infer_2d_elements()
        kmax=self.hydro.seg_k.max()
        if self.hydro.n_seg != self.hydro.n_2d_elements*(kmax+1):
            raise Exception("You probably mean to be running with segment-dense hydrodynamics")
        
        num_layers=self.hydro.n_seg / self.hydro.n_2d_elements
        if self.hydro.vertical != self.hydro.SIGMA:
            return """MULTIGRID
  ZMODEL NOLAY %d
END_MULTIGRID"""%num_layers
        else:
            return " ; sigma layers - no multigrid stanza"

    _hydro=None
    @property
    def hydro(self):
        return self._hydro
    @hydro.setter
    def hydro(self,value):
        self.set_hydro(value)

    def set_hydro(self,hydro):
        self._hydro=hydro

        if hydro is None:
            return

        self._hydro.scenario=self

        # I think it's required that these match
        self.time0 = self._hydro.time0

        # Other time-related values needn't match, but the hydro
        # is a reasonable default if no value has been set yet.
        if self.time_step is None:
            self.time_step=self._hydro.time_step
        if self.start_time is None:
            self.start_time=self.time0+self.scu*self._hydro.t_secs[0]
        if self.stop_time is None:
            self.stop_time =self.time0+self.scu*self._hydro.t_secs[-1]

    def init_parameters(self):
        params=NamedObjects(scenario=self,cast_value=cast_to_parameter)
        params['ONLY_ACTIVE']=ParameterConstant(1) # almost always a good idea.
        
        return params
    def init_hydro_parameters(self):
        """ parameters which come directly from the hydro, and are
        written out in the same way that process parameters are 
        written.
        """
        if self.hydro:
            # in case hydro is re-used, make sure that this call gets a fresh
            # set of parameters.  some leaky abstraction going on...
            return self.hydro.parameters(force=True)
        else:
            self.log.warning("Why requesting hydro parameters with no hydro?")
            assert False # too weird
            return NamedObjects(scenario=self,cast_value=cast_to_parameter)

    def init_substances(self):
        # sorts active substances first.
        return NamedObjects(scenario=self,sort_key=lambda s: not s.active)

    def text_thatcher_harleman_lags(self):
        return """;
; Thatcher-Harleman timelags
0 ; no lags
        """

    def init_bcs(self):
        self.bcs=[]

        # 2017-03-17: moved this bit of boiler plate from a bunch of subclasses
        # to here in a step towards standardizing the BC settings.
        boundaries=self.hydro.boundary_defs()
        ids=[b['id'] for b in boundaries]

        for src_tag in self.src_tags:
            # conc. defaults to 1.0
            self.add_bc(src_tag['items'],src_tag['tracer'],src_tag.get('value',1.0))

    def deep_sediment_boundary_defs(self):
        n_2d=self.hydro.n_2d_elements
        bdefs=np.zeros(n_2d, Hydro.boundary_dtype)
        prefix=self.bottom_bc_prefix
        for elt in range(n_2d):
            bdefs[elt]['id'] ="%s %d"%(prefix, elt+1)
            bdefs[elt]['name']="%s %d"%(prefix, elt+1)
            bdefs[elt]['type']=prefix
        return bdefs

    def set_deep_sediment_bc_from_ic(self, subs, subset=[]):
        """
        For the given substances set the deep boundary in a layered sediment bed to the previously
        configured IC for them.
        """
        assert self.bottom_grid is not None,"Trying to set deep BCs, but bottom_grid is not configured"
        
        self.hydro.infer_2d_elements()
        n_2d = self.hydro.n_2d_elements
                        
        for scalar in subset:
            if scalar not in subs: continue
            sub=subs[scalar]

            if sub.initial.seg_values is not None:
                # Should be the initial condition in the bottom sediment layer.
                values = sub.initial.seg_values[-n_2d:]
                if np.all(values==values[0]):
                    # Might save some space/work
                    print("Spatially variable was actually constant for %s"%scalar)
                    values=values[0]
            else:
                values = sub.initial.default
                
            if np.isscalar(values):
                # Can use the 'type' to write a single BC value for all deep sediment BCs
                # when this gets written out it ends up with 'ITEM 'sunnyvale' 'valero' ... CONCENTRATION <scalar> 
                self.src_tags.append(dict(tracer=scalar,items=[self.bottom_bc_prefix],value=values))
            else:
                # Possible that there is a shorthand here
                deep_bcs = self.deep_sediment_boundary_defs()
                assert len(deep_bcs) == len(values)
                for item,value in zip(deep_bcs,values):
                    self.src_tags.append(dict(tracer=scalar,items=[item['id'].decode()],value=value))

    def add_bc(self,*args,**kws):
        bc=BoundaryCondition(*args,**kws)
        bc.scenario=self
        self.bcs.append(bc)
        return bc

    def init_loads(self):
        """
        Set up discharges (locations of point sources/sinks), and
        corresponding loads (e.g. mass/time for a substance at a source)
        """
        self.discharges=[]
        self.loads=[]

    def add_discharge(self,on_exists='exception',*arg,**kws):
        disch=Discharge(*arg,**kws)
        disch.scenario=self
        disch.update_fields()
        exists=False
        for other in self.discharges:
            if other.load_id==disch.load_id:
                if on_exists=='exception':
                    raise Exception("Discharge with id='%s' already exists"%other.load_id)
                elif on_exists=='ignore':
                    self.log.info("Discharge id='%s' exists - skipping duplicate"%other.load_id)
                    return other
                else:
                    assert False 
        self.discharges.append(disch)
        return disch

    def add_load(self,*args,**kws):
        load=Load(*args,**kws)
        load.scenario=self
        self.loads.append(load)
        return load
    
    def add_monitor_from_shp(self,shp_fn,naming='elt_layer',point_layers=True):
        """
        For each feature in the shapefile, add a monitor area.
        shp_fn: path to shapefile
        naming: generally, a field name from the shapefile giving the name of the monitor area.
          special case when point_layers is True and the shapefile has points, naming can
          be "elt_layer" in which case individual segments are monitored separately, and
          named like elt123_layer45.
        """
        assert self.hydro,"Set hydro before calling add_monitor_from_shp"

        locations=wkb2shp.shp2geom(shp_fn)
        self.hydro.infer_2d_elements()

        g=self.hydro.grid()
        new_areas=[] # generate as list, then assign as tuple
        names={}
        for n,segs in self.monitor_areas:
            names[n]=True # record names in use to avoid duplicates

        for i,rec in enumerate(locations):
            geom=rec['geom']

            if point_layers and (geom.type=='Point'):
                xy=np.array(geom.coords)[0]
                # would be better to have select_cells_nearest use
                # centroids instead of circumcenters, but barring
                # that, throw the net wide and look at up to 40
                # cells.
                elt=g.select_cells_nearest(xy,inside=True,count=40)
                if elt is None:
                    self.log.warning("Monitor point %s was not found inside a cell"%xy)
                    # Fall back to nearest
                    elt=g.select_cells_nearest(xy,inside=False)

                segs=np.nonzero( self.hydro.seg_to_2d_element==elt )[0]
                for layer,seg in enumerate(segs):
                    if naming=='elt_layer':
                        name="elt%d_layer%d"%(elt,layer)
                    else:
                        name="%s_layer%d"%(rec[naming],layer)
                    if name not in names:
                        new_area=(name,[seg])
                        self.monitor_areas = self.monitor_areas + (new_area,)
                        names[name]=True
                    else:
                        self.log.warning("Duplicate requests to monitor %s"%name)
            else:
                try:
                    name=rec[naming]
                except:
                    # This used to name everything mon%d, but for easier use
                    # and compatibility with older code, use geometry type
                    name="%s%d"%(geom.type.lower(),len(self.monitor_areas))
                self.add_monitor_for_geometry(name=name,geom=geom)
                
    def add_monitor_for_geometry(self,name,geom):
        """
        Add a monitor area for elements intersecting the shapely geometry geom.
        """
        # make sure the name is unique
        for n,segs in self.monitor_areas:
            assert name!=n
        
        g=self.hydro.grid()
        # bitmask over 2D elements
        self.log.info("Selecting elements in polygon '%s'"%name)
        # better to go by center, so that non-intersecting polygons
        # yield non-intersecting sets of elements and segments
        # 2019-09-12: use centroid instead of center in case the grid has weird geometry
        # 2020-01-31: use representative points. It's possible for centroid not to fall within
        #      the polygon.  Don't use full polygon, because that will pick up adjacent 
        #      cells that share an edge.
        elt_sel=g.select_cells_intersecting(geom,by_center='representative') 

        # extend to segments:
        seg_sel=elt_sel[ self.hydro.seg_to_2d_element ] & (self.hydro.seg_to_2d_element>=0)         
        segs=np.nonzero( seg_sel )[0]

        new_area= (name,segs) 

        self.log.info("Added %d monitored segments for %s"%(len(segs),name))
        self.monitor_areas = self.monitor_areas + ( new_area, )

    def add_transects_from_shp(self,shp_fn,naming='count',clip_to_poly=True,
                               on_boundary='warn_and_skip',on_edge=False,
                               add_station=False):
        """
        Add monitor transects from a shapefile.
        By default transects are named in sequence.  
        Specify a shapefile field name in 'naming' to pull user-specified
        names for the transects.
        on_edge: indicates that the shapefile is made up of nodes already following
          edges of the grid.  In theory not needed, but included here for compatibility
          with ZZ code.
        add_station: if True, select the deepest adjacent cell and add a monitoring station
          with the same name
        """
        locations=wkb2shp.shp2geom(shp_fn)
        g=self.hydro.grid()

        if clip_to_poly:
            poly=g.boundary_polygon()
            
        new_transects=[] # generate as list, then assign as tuple

        for i,rec in enumerate(locations):
            geom=rec['geom']

            if geom.type=='LineString':
                if clip_to_poly:
                    clipped=geom.intersection(poly)

                    # rather than assume that clipped comes back with
                    # the same orientation, and multiple pieces come
                    # back in order, manually re-assemble the line
                    if clipped.type=='LineString':
                        segs=[clipped]
                    else:
                        segs=clipped.geoms

                    all_dists=[]
                    for seg in segs:
                        for xy in seg.coords:
                            all_dists.append( geom.project( geometry.Point(xy) ) )
                    # sorting the distances ensures that orientation is same as
                    # original
                    all_dists.sort()

                    xy=[geom.interpolate(d) for d in all_dists]
                else:
                    xy=np.array(geom.coords)
                    
                if naming=='count':
                    name="transect%04d"%i
                else:
                    name=rec[naming]
                exchs=self.hydro.path_to_transect_exchanges(xy,on_boundary=on_boundary,on_edge=on_edge)
                new_transects.append( (name,exchs) )
            else:
                self.log.warning("Not ready to handle geometry type %s"%geom.type)
        self.log.info("Added %d monitored transects from %s"%(len(new_transects),shp_fn))
        self.monitor_transects = self.monitor_transects + tuple(new_transects)

        if add_station:
            self.add_station_for_transects(new_transects) # Not yet implement for Scenario

    def add_area_boundary_transects(self,exclude='dummy'):
        """
        create monitor transects for the common boundaries between a subset of
        monitor areas. this assumes that the monitor areas are distinct - no
        overlapping cells (in fact it asserts this).
        The method and the non-overlapping requirement apply only for areas which
        do *not* match the exclude regex.
        """
        areas=[a[0] for a in self.monitor_areas]
        if exclude is not None:
            areas=[a for a in areas if not re.match(exclude,a)]

        mon_areas=dict(self.monitor_areas)

        seg_to_area=np.zeros(self.hydro.n_seg,'i4')-1

        for idx,name in enumerate(areas):
            # make sure of no overlap:
            assert np.all( seg_to_area[ mon_areas[name] ] == -1 )
            # and label to this area:
            seg_to_area[ mon_areas[name] ] = idx

        poi0=self.hydro.pointers - 1

        exch_areas=seg_to_area[poi0[:,:2]]
        # fix up negatives in poi0
        exch_areas[ poi0[:,:2]<0 ] = -1

        # convert to tuples so we can get unique pairs
        exch_areas_tupes=set( [ tuple(x) for x in exch_areas if x[0]!=x[1] and x[0]>=0 ] )
        # make the order canonical 
        canon=set()
        for a,b in exch_areas_tupes:
            if a>b:
                a,b=b,a
            canon.add( (a,b) )
        canon=list(canon) # re-assert order

        names=[]
        exch1s=[]

        for a,b in canon:
            self.log.info("%s <-> %s"%(areas[a],areas[b]))
            name=areas[a][:9] + "__" + areas[b][:9]
            self.log.info("  name: %s"%name)
            names.append(name)

            fwd=np.nonzero( (exch_areas[:,0]==a) & (exch_areas[:,1]==b) )[0]
            rev=np.nonzero( (exch_areas[:,1]==a) & (exch_areas[:,0]==b) )[0]
            exch1s.append( np.concatenate( (fwd+1, -(rev+1)) ) )
            self.log.info("  exchange count: %d"%len(exch1s[-1]))

        # and add to transects:
        transects=tuple(zip(names,exch1s))
        self.monitor_transects=self.monitor_transects + transects
        
    def add_transect(self,name,exchanges):
        """ Append a transect definition for logging.
        """
        self.monitor_transects = self.monitor_transects + ( (name,exchanges), )
        
    def ensure_base_path(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def write_inp(self):
        """
        Write the inp file for delwaq1/delwaq2
        """
        self.ensure_base_path()
        # parameter files are also written along the way
        self.inp.write()

    def write_hydro(self):
        self.ensure_base_path()
        self.hydro.write()

    _pdb=None
    @property
    def process_db(self):
        if self._pdb is None:
            self._pdb = waq_process.ProcessDB(scenario=self)
        return self._pdb
    def lookup_item(self,name):
        return self.process_db.substance_by_id(name)

    def as_datetime(self,t):
        """
        t can be a datetime object, an integer number of seconds since time0,
        or a floating point datenum
        """
        if np.issubdtype(type(t),np.integer):
            return self.time0 + t*self.scu
        elif np.issubdtype(type(t),np.floating):
            return utils.to_datetime(t) #num2date(t)
        elif isinstance(t,datetime.datetime):
            return t
        else:
            raise WaqException("Invalid type for datetime: {}".format(type(t)))

    def fmt_datetime(self,t):
        """ 
        return datetime formatted as text.
        format is part of input file configuration, but 
        for now, stick with 1990/08/15-12:30:00

        t is specified as in as_datetime() above.
        """
        return self.as_datetime(t).strftime('%Y/%m/%d-%H:%M:%S')

    #-- Access to output files
    def nef_history(self):
        hda=os.path.join( self.base_path,self.name+".hda")
        hdf=os.path.join( self.base_path,self.name+".hdf")
        if os.path.exists(hda):
            if nefis.nef_lib() is None:
                self.log.warning("Nefis library not configured -- will not read nef history")
                return None
            return nefis.Nefis(hda, hdf)
        else:
            return None
    def nef_map(self):
        ada=os.path.join(self.base_path, self.name+".ada")
        adf=os.path.join(self.base_path, self.name+".adf")
        if os.path.exists(ada):
            return nefis.Nefis( ada,adf)
        else:
            return None

    #  netcdf versions of those:
    def nc_map(self,nc_kwargs={}):
        nef=self.nef_map()
        try:
            elt_map,real_map = map_nef_names(nef)

            nc=nefis_nc.nefis_to_nc(nef,element_map=elt_map,nc_kwargs=nc_kwargs)
            for vname,rname in iteritems(real_map):
                nc.variables[vname].original_name=rname
            # the nefis file does not contain enough information to get
            # time back to a real calendar, so rely on the Scenario's
            # version of time0
            if 'time' in nc.variables:
                nc.time.units='seconds since %s'%self.time0.strftime('%Y-%m-%d %H:%M:%S')
        finally:
            nef.close()
        return nc

    # try to find the common chunks of code between writing ugrid
    # nc output and the history output

    def ugrid_map(self,nef=None,nc_kwargs={}):
        return self.ugrid_nef(mode='map',nef=nef,nc_kwargs=nc_kwargs)

    def ugrid_history(self,nef=None,nc_kwargs={}):
        return self.ugrid_nef(mode='history',nef=nef,nc_kwargs=nc_kwargs)

    default_ugrid_output_settings=['quickplot_compat']
    
    def ugrid_nef(self,mode='map',nef=None,nc_kwargs={},output_settings=None):
        """ Like nc_map, but write a netcdf file more ugrid compliant.
        this is actually pretty different, as ugrid requires that 3D
        field is organized into a horizontal dimension (i.e element)
        and vertical dimension (layer).  the original nefis code
        just gives segment.
        nef: supply an already open nef.  Note that caller is responsible for closing it!
        mode: 'map' use the map output
              'history' use history output
        """
        if output_settings is None:
            output_settings=self.default_ugrid_output_settings

        if nef is None:
            if mode=='map':
                nef=self.nef_map()
            elif mode=='history':
                nef=self.nef_history()
            close_nef=True
        else:
            close_nef=False
        if nef is None: # file didn't exist
            self.log.info("NEFIS file didn't exist. Skipping ugrid_nef()")
            return None
            
        flowgeom=self.flowgeom()
        mesh_name="FlowMesh" # sync with sundwaq for now.

        if flowgeom is not None:
            nc=flowgeom.copy(**nc_kwargs)
        else:
            nc=qnc.empty(**nc_kwargs)
        nc._set_string_mode('fixed') # required for writing to disk

        self.hydro.infer_2d_elements()

        try:
            if mode=='map':
                seg_k = self.hydro.seg_k
                seg_elt = self.hydro.seg_to_2d_element
                n_locations=len(seg_elt)
            elif mode=='history':
                # do we go through the location names, trying to pull out elt_k? no - as needed,
                # use self.monitor_areas.

                # maybe the real question is how will the data be organized in the output?
                # if each history output can be tied to a single segment, that's one thing.
                # Could subset the grid, or keep the full grid and pad the data with fillvalue.
                # but if some/all of the output goes to multiple segments, then what?
                # keep location_name in the output?
                # maybe we skip any notion of ugrid, and instead follow a more CF observed features
                # structure?

                # also, whether from the original Scenario, or by reading in the inp file, we can get
                # the original map between location names and history outputs
                # for the moment, classify everything in the file based on the first segment
                # listed

                # try including the full grid, and explicitly output the mapping between history
                # segments and the 2D+z grid.
                hist_segs=[ma[1][0] for ma in self.monitor_areas]
                seg_k=self.hydro.seg_k[ hist_segs ]
                seg_elt=self.hydro.seg_to_2d_element[hist_segs]

                # Need to handle transects - maybe that's handled entirely separately.
                # even so, the presence of transects will screw up the matching of dimensions
                # below for history output.
                # pdb.set_trace()

                # i.e. current output with transect:
                # len(seg_elt)==135
                # but we ended up creating an anonymous dimension d138
                # (a) could consult scenario to get the count of transects
                # (b) is there anything in the NEFIS file to indicate transect output?
                # (c) could use the names as a hint
                #     depends on the input file, but currently have things like eltNNN_layerNN
                #     vs. freeform for transects

                # what about just getting the shape from the LOCATIONS field?
                shape,dtype = nef['DELWAQ_PARAMS'].getelt('LOCATION_NAMES',shape_only=True)
                n_locations=shape[1]
                if n_locations>len(seg_elt):
                    self.log.info("Looks like there were %d transects, too?"%(n_locations - len(seg_elt)))
                elif n_locations<len(seg_elt):
                    self.log.warning("Weird - fewer output locations than anticipated! %d vs %d"%(n_locations,
                                                                                                  len(seg_elt)))
            else:
                assert False

            n_layers= seg_k.max() + 1

            # elt_map: 'SUBST_001' => 'oxy'
            # real_map: 'saturoxy' => 'SaturOXY'
            elt_map,real_map = map_nef_names(nef)

            # check for unique element names
            name_count=defaultdict(lambda: 0)
            for group in nef.groups():
                for elt_name in group.cell.element_names:
                    name_count[elt_name]+=1

            # check for unique unlimited dimension:
            n_unl=0
            for group in nef.groups():
                # there are often multiple unlimited dimensions.
                # hopefully just 1 unlimited in the RESULTS group
                if 0 in group.shape and group.name=='DELWAQ_RESULTS':
                    n_unl+=1

            # give the user a sense of how many groups are being
            # written out:
            self.log.info("Elements to copy from NEFIS:")
            for group in nef.groups():
                for elt_name in group.cell.element_names:
                    nef_shape,nef_type=group.getelt(elt_name,shape_only=True)
                    vshape=group.shape + nef_shape
                    self.log.info("  %s.%s: %s (%s)"%(group.name,
                                                      elt_name,
                                                      vshape,nef_type))

            for group in nef.groups():
                g_shape=group.shape
                grp_slices=[slice(None)]*len(g_shape)
                grp_dim_names=[None]*len(g_shape)

                # infer that an unlimited dimension in the RESULTS
                # group is time.
                if 0 in g_shape and group.name=='DELWAQ_RESULTS':
                    idx=list(g_shape).index(0)
                    if n_unl==1: # which will be named
                        grp_dim_names[idx]='time'

                for elt_name in group.cell.element_names:
                    # print("elt name is",elt_name)

                    # Choose a variable name for this element
                    if name_count[elt_name]==1:
                        vname=elt_name
                    else:
                        vname=group.name + "_" + elt_name

                    if vname in elt_map:
                        vname=elt_map[vname]
                    else:
                        vname=vname.lower()

                    self.log.info("Writing variable %s"%vname)
                    subst=self.lookup_item(vname) # may be None!
                    if subst is None:
                        self.log.info("No metadata from process library on %s"%repr(vname))

                    # START time-iteration HERE
                    # for large outputs, need to step through time
                    # assume that only groups with 'time' as a dimension
                    # (as detected above) need to be handled iteratively.
                    # 'time' assumed to be part of group shape.
                    # safe to always iterate on time.

                    # nef_shape is the shape of the element subject to grp_slices,
                    # as understood by nefis, before squeezing or projecting to [cell,layer]
                    nef_shape,value_type=group.getelt(elt_name,shape_only=True)
                    self.log.debug("nef_shape: %s"%nef_shape )
                    self.log.debug("value_type: %s"%value_type )

                    if value_type.startswith('f'):
                        fill_value=np.nan
                    elif value_type.startswith('S'):
                        fill_value=None
                    else:
                        fill_value=-999

                    nef_to_squeeze=[slice(None)]*len(nef_shape)
                    if 1: # squeeze unit element dimensions
                        # iterate over just the element portion of the shape
                        squeeze_shape=list( nef_shape[:len(g_shape)] )
                        for idx in range(len(g_shape),len(nef_shape)):
                            if nef_shape[idx]==1:
                                nef_to_squeeze[idx]=0
                            else:
                                squeeze_shape.append(nef_shape[idx])
                    else: # no squeeze
                        squeeze_shape=list(nef_shape)

                    self.log.debug("squeeze_shape: %s"%squeeze_shape)
                    self.log.debug("nef_to_squeeze: %s"%nef_to_squeeze)

                    # mimics qnc naming - will come back to expand 3D fields
                    # and names
                    dim_names=[qnc.anon_dim_name(size=l) for l in squeeze_shape]
                    for idx,name in enumerate(grp_dim_names): # okay since squeeze only does elt dims
                        if name:
                            dim_names[idx]=name

                    # special handling for results, which need to be mapped 
                    # back out to 3D
                    proj_shape=list(squeeze_shape)
                    if group.name=='DELWAQ_RESULTS' and self.hydro.n_seg in squeeze_shape:
                        seg_idx = proj_shape.index(self.hydro.n_seg)
                        proj_shape[seg_idx:seg_idx+1]=[self.hydro.n_2d_elements,
                                                       n_layers]
                        # the naming of the layers dimension matches assumptions in ugrid.py
                        # not sure how this is supposed to be specified
                        dim_names[seg_idx:seg_idx+1]=["nFlowElem","nFlowMesh_layers"]

                        # new_value=np.zeros( new_shape, value_type )
                        # new_value[...]=fill_value

                        # this is a little tricky, but seems to work.
                        # map segments to (elt,layer), and all other dimensions
                        # get slice(None).
                        # vmap assumes no group slicing, and is to be applied
                        # to the projected array (projection does not involve
                        # any slices on the nefis src side)
                        vmap=[slice(None) for _ in proj_shape]
                        vmap[seg_idx]=seg_elt
                        vmap[seg_idx+1]=seg_k
                        # new_value[vmap] = value
                    elif group.name=='DELWAQ_RESULTS' and n_locations in squeeze_shape:
                        # above and below: n_locations used to be len(seg_elt)
                        # but it's still writing out things like location_names with
                        # an anonymous dimension
                        seg_idx = proj_shape.index(n_locations)
                        # note that nSegment is a bit of a misnomer, might have some transects
                        # in there, too.
                        dim_names[seg_idx]="nSegment"
                    else:
                        vmap=None # no projection

                    for dname,dlen in zip(dim_names,proj_shape):
                        if dname=='time':
                            # if time is not specified as unlimited, it gets
                            # included as the fastest-varying dimension, which
                            # makes writes super slow.
                            nc.add_dimension(dname,0)
                        else:
                            nc.add_dimension(dname,dlen)

                    # most of the time goes into writing.
                    # typically people optimize chunksize, but HDF5 is
                    # throwing an error when time chunk>1, so it's
                    # hard to imagine any improvement over the defaults.
                    ncvar=nc.createVariable(vname,np.dtype(value_type),dim_names,
                                            fill_value=fill_value,
                                            complevel=2,
                                            zlib=True)

                    if vmap is not None:
                        nc.variables[vname].mesh=mesh_name
                        # these are specifically the 2D horizontal metadata
                        if 'quickplot_compat' in output_settings:
                            # as of Delft3D_4.01.01.rc.03, quickplot only halfway understands
                            # ugrid, and actually does better when location is not specified.
                            self.log.info('Dropping location for quickplot compatibility')
                        else:
                            nc.variables[vname].location='face' # 
                        nc.variables[vname].coordinates="FlowElem_xcc FlowElem_ycc"

                    if subst is not None:
                        if hasattr(subst,'unit'):
                            # in the process table units are parenthesized
                            units=subst.unit.replace('(','').replace(')','')
                            # no guarantee of CF compliance here...
                            nc.variables[vname].units=units
                        if hasattr(subst,'item_nm'):
                            nc.variables[vname].long_name=subst.item_nm
                        if hasattr(subst,'aggrega'):
                            nc.variables[vname].aggregation=subst.aggrega
                        if hasattr(subst,'groupid'):
                            nc.variables[vname].group_id=subst.groupid

                    if 'time' in dim_names:
                        # only know how to deal with time as the first index
                        assert dim_names[0]=='time'
                        self.log.info("Will iterate over %d time steps"%proj_shape[0])

                        total_tic=t_last=time.time()
                        read_sum=0
                        write_sum=0
                        for ti in range(proj_shape[0]):
                            read_sum -= time.time()
                            value_slice=group.getelt(elt_name,[ti])
                            read_sum += time.time()

                            if vmap is not None:
                                proj_slice=np.zeros(proj_shape[1:],value_type)
                                proj_slice[...]=fill_value
                                proj_slice[tuple(vmap[1:])]=value_slice
                            else:
                                proj_slice=value_slice
                            write_sum -= time.time()
                            ncvar[ti,...] = proj_slice
                            write_sum += time.time()

                            if (time.time() - t_last > 2) or (ti+1==proj_shape[0]):
                                t_last=time.time()
                                self.log.info('  time step %d / %d'%(ti,proj_shape[0]))
                                self.log.info('  time for group so far: %fs'%(t_last-total_tic))
                                self.log.info('  reading so far: %fs'%(read_sum))
                                self.log.info('  writing so far: %fs'%(write_sum))

                    else:
                        value=group.getelt(elt_name)
                        if vmap is not None:
                            proj_value=value[tuple(vmap)]
                        else:
                            proj_value=value
                        # used to have extraneous[?] names.append(Ellipsis)
                        ncvar[:]=proj_value

                    setattr(ncvar,'group_name',group.name)
            ####

            for vname,rname in iteritems(real_map):
                nc.variables[vname].original_name=rname
            # the nefis file does not contain enough information to get
            # time back to a real calendar, so rely on the Scenario's
            # version of time0
            if 'time' in nc.variables:
                nc.time.units='seconds since %s'%self.time0.strftime('%Y-%m-%d %H:%M:%S')
                nc.time.standard_name='time'
                nc.time.long_name='time relative to model time0'

        finally:
            if close_nef:
                nef.close()

        # cobble together surface h, depth info.
        if 'time' in nc.variables:
            t=nc.time[:]
        else:
            t=None # not going to work very well...or at all

        z_bed=nc.FlowElem_bl[:]
        # can't be sure of what is included in the output, so have to try some different
        # options

        if 1:
            if mode=='map':
                etavar=nc.createVariable('eta',np.float32,['time','nFlowElem'],
                                         zlib=True)
                etavar.standard_name='sea_surface_height_above_geoid'
                etavar.mesh=mesh_name

                for ti in range(len(nc.dimensions['time'])):
                    # due to a possible DWAQ bug, we have to be very careful here
                    # depths in dry upper layers are left at their last-wet value,
                    # and count towards totaldepth and localdepth.  That's fixed in
                    # DWAQ now.
                    if 'totaldepth' in nc.variables:
                        depth=nc.variables['totaldepth'][ti,:,0]
                    elif 'depth' in nc.variables:
                        depth=np.nansum(nc.variables['depth'][ti,:,:],axis=2)
                    else:
                        # no freesurface info.
                        depth=-z_bed[None,:] 

                    z_surf=z_bed + depth
                    etavar[ti,:]=z_surf
            elif mode=='history':
                # tread carefully in case there is nothing useful in the history file.
                if 'totaldepth' in nc.variables and 'nSegment' in nc.dimensions:
                    etavar=nc.createVariable('eta',np.float32,['time',"nSegment"],
                                             zlib=True)
                    etavar.standard_name='sea_surface_height_above_geoid'

                    # some duplication when we have multiple layers of the
                    # same watercolumn
                    # can only create eta for history output, nan for transects
                    pad=np.nan*np.ones(n_locations-len(seg_elt),'f4')
                    for ti in range(len(nc.dimensions['time'])):
                        depth=nc.variables['totaldepth'][ti,:]
                        z_surf=z_bed[seg_elt]
                        z_surf=np.concatenate( (z_surf,pad) )+depth
                        etavar[ti,:]=z_surf
                else:
                    self.log.info('Insufficient info in history file to create eta')

        if 1: # extra mapping info for history files
            pad=-1*np.ones( n_locations-len(seg_elt),'i4')
            if mode=='history':
                nc['element']['nSegment']=np.concatenate( (seg_elt,pad) )
                nc['layer']['nSegment']=np.concatenate( (seg_k,pad) )
                if flowgeom:
                    xcc=flowgeom.FlowElem_xcc[:]
                    ycc=flowgeom.FlowElem_ycc[:]
                    nc['element_x']['nSegment']=np.concatenate( (xcc[seg_elt],np.nan*pad) )
                    nc['element_y']['nSegment']=np.concatenate( (ycc[seg_elt],np.nan*pad) )

        # extra work to make quickplot happy
        if (mode=='map') and ('quickplot_compat' in output_settings):
            # add in some attributes and fields which might make quickplot happier
            # Add in node depths
            
            # g=unstructured_grid.UnstructuredGrid.from_ugrid(nc)
            x_nc=self.flowgeom_ds()
            g=unstructured_grid.UnstructuredGrid.from_ugrid(x_nc)

            # dicey - assumes particular names for the fields:
            if 'FlowElem_zcc' in nc and 'Node_z' not in nc:
                self.log.info('Adding a node-centered depth via interpolation')
                nc['Node_z']['nNode']=g.interp_cell_to_node(nc.FlowElem_zcc[:])
                nc.Node_z.units='m'
                nc.Node_z.positive='up',
                nc.Node_z.standard_name='sea_floor_depth',
                nc.Node_z.long_name="Bottom level at net nodes (flow element's corners)"
                nc.Node_z.coordinates=nc[mesh_name].node_coordinates
                
            # need spatial attrs for node coords
            node_x,node_y = nc[mesh_name].node_coordinates.split()
            nc[node_x].units='m'
            nc[node_x].long_name='x-coordinate of net nodes'
            nc[node_x].standard_name='projection_x_coordinate'
            
            nc[node_y].units='m',
            nc[node_y].long_name='y-coordinate of net nodes'
            nc[node_y].standard_name='projection_y_coordinate'

            for k,v in six.iteritems(nc.variables):
                if 'location' in v.ncattrs():
                    self.log.info("Stripping location attribute from %s for quickplot compatibility"%k)
                    v.delncattr('location')

                # some of the grids being copied through are missing this, even though waq_scenario
                # is supposed to write it out.
                if 'standard_name' in v.ncattrs() and v.standard_name=='ocean_sigma_coordinate':
                    v.formula_terms="sigma: nFlowMesh_layers eta: eta depth: FlowElem_bl"
                    
        return nc

    _flowgeom=None
    def flowgeom(self):
        """ Returns a netcdf dataset with the grid geometry, or None
        if the data is not around.
        """
        if self._flowgeom is None:
            fn=self.hydro.flowgeom_filename
            if os.path.exists(fn):
                self._flowgeom=qnc.QDataset(fn)
        
        return self._flowgeom
    _flowgeom_ds=None
    def flowgeom_ds(self):
        """ Returns a netcdf dataset with the grid geometry, or None
        if the data is not around.  Returns as an xarray Dataset.
        """
        if self._flowgeom_ds is None:
            fn=self.hydro.flowgeom_filename
            if os.path.exists(fn):
                self._flowgeom_ds=xr.open_dataset(fn)
        
        return self._flowgeom_ds
    
    #-- Command line access
    def cmd_write_runid(self):
        """
        Label run name in the directory, needed for some delwaq2 (confirm?)

        Maybe this can be supplied on the command line, too?
        """
        self.ensure_base_path()

        if self.use_bloom:
            with open(os.path.join(self.base_path,'runid.eco'),'wt') as fp:
                fp.write("{}\n".format(self.name))
                fp.write("y\n")
        else:
            with open(os.path.join(self.base_path,'runid.waq'),'wt') as fp:
                fp.write("{}\n".format(self.name))
                fp.write("y\n") # maybe unnecessary for non-bloom run.

    def cmd_write_bloominp(self):
        """
        Copy supporting bloominp file for runs using BLOOM algae
        """
        if not os.path.exists(self.original_bloominp_path):
            # pdb.set_trace()
            self.log.warning("BLOOM not found (%s)! Tread carefully"%self.original_bloominp_path)
            return
        
        dst=os.path.join(self.base_path,'bloominp.d09')
        if not self.overwrite:
            assert not os.path.exists(dst)
        shutil.copyfile(self.original_bloominp_path,dst)

    def cmd_write_inp(self):
        """
        Write inp file and supporting files (runid, bloominp.d09) for
        delwaq1/2
        """
        self.ensure_base_path()

        self.log.debug("Writing inp file")
        self.write_inp()

        self.cmd_write_bloominp()
        self.cmd_write_runid()

        if self.n_bottom_layers and not self.bottom_grid:
            self.write_delwaqg()

    def write_delwaqg(self):
        write_delwaqg_parameters(self) # will check for delwaqg_initials

    def cmd_write_hydro(self):
        """
        Create hydrodynamics data ready for input to delwaq1
        """

        self.ensure_base_path()

        self.log.info("Writing hydro data")
        self.write_hydro()

    def cmd_default(self):
        """
        Prepare all inputs for delwaq1 (hydro, runid, inp files)
        """
        self.cmd_write_hydro()
        self.cmd_write_inp()

    def cmd_write_nc(self):
        """ Transcribe binary or NEFIS to NetCDF for a completed DWAQ run 
        """
        self.write_binary_his_nc() or self.write_nefis_his_nc()
        # binary is faster and doesn't require dwaq libraries, but
        # does not know about units.
        self.write_binary_map_nc() or self.write_nefis_map_nc()
        
    def write_nefis_his_nc(self):
        nc2_fn=os.path.join(self.base_path,'dwaq_hist.nc')
        nc2=self.ugrid_history(nc_kwargs=dict(fn=nc2_fn,overwrite=True))
        # if no history output or nefis is not setup, no nc2.
        if nc2:
            nc2.close()
            return True
        else:
            return False

    def write_binary_his_nc(self):
        """ If binary history output is present, write that out to netcdf, otherwise
        return False
        """
        his_fn=os.path.join(self.base_path,self.name+".his")
        his_nc_fn=os.path.join(self.base_path,'dwaq_hist.nc')
        
        if not os.path.exists(his_fn): return False
        ds=dio.his_file_xarray(his_fn)
        if os.path.exists(his_nc_fn):
            os.unlink(his_nc_fn)
        ds.to_netcdf(his_nc_fn)
        
    def write_nefis_map_nc(self):
        nc_fn=os.path.join(self.base_path,'dwaq_map.nc')
        nc=self.ugrid_map(nc_kwargs=dict(fn=nc_fn,overwrite=True))
        # if no map output, no nc
        if nc:
            nc.close()
            return True
        else:
            return False
        
    def write_binary_map_nc(self):
        """
        Transcribe binary formatted map output from completed dwaq
        run to a ugrid-esque netcdf file.  Currently assumes sigma
        coordinates!
        """
        if 'binary' not in self.map_formats:
            return False

        import stompy.model.delft.io as dio
        map_fn=os.path.join(self.base_path,self.name+".map")
        map_ds=dio.read_map(map_fn,self.hydro)

        dio.map_add_z_coordinate(map_ds,total_depth='TotalDepth',coord_type='sigma',
                                 layer_dim='layer')

        map_ds.to_netcdf(os.path.join(self.base_path,"dwaq_map.nc"))
        return True

    use_bloom=False
    def cmd_delwaq1(self):
        """
        Run delwaq1 preprocessor
        """
        if self.use_bloom:
            bloom_part="-eco {}".format(self.bloom_path)
        else:
            bloom_part=""

        cmd=[self.delwaq1_path,
             "-waq", bloom_part,
             "-p",self.proc_path]
        #cmd='{} -waq {} -p {}'.format(self.delwaq1_path,
        #                              bloom_part,
        #                              self.proc_path)
        self.log.info("Running delwaq1:")
        self.log.info("  "+ " ".join(cmd))

        t_start=time.time()
        try:
            ret=subprocess.check_output(cmd,shell=False,cwd=self.base_path,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            self.log.error("problem running delwaq1")
            self.log.error("output: ")
            self.log.error("-----")
            self.log.error(exc.output)
            self.log.error("-----")
            raise WaqException("delwaq1 exited early.  check lst and lsp files")
        self.log.info("delwaq1 ran in %.2fs"%(time.time() - t_start))

        nerrors=nwarnings=-1
        # dwaq likes to draw boxes with code page 437
        for line in ret.decode('cp437','ignore').split("\n"):
            if 'Number of WARNINGS' in line:
                nwarnings=int(line.split()[-1])
            elif 'Number of ERRORS during input' in line:
                nerrors=int(line.split()[-1])
        if nerrors > 0 or nwarnings>0:
            print( ret )
            raise WaqException("delwaq1 found %d errors and %d warnings"%(nerrors,nwarnings))
        elif nerrors < 0 or nwarnings<0:
            print( ret)
            raise WaqException("Failed to find error/warning count")

    def cmd_delwaq2(self,output_filename=None,delwaq2name=None):
        """
        Run delwaq2 (computation)
        delwaq2name: temporarily override the path to the delwaq2 executable.
        this can be done more generally by setting self.delwaq2_path.
        """
        cmd=[delwaq2name or self.delwaq2_path,self.name]
        if not output_filename:
            output_filename= os.path.join(self.base_path,'delwaq2.out')

        t_start=time.time()
        with open(output_filename,'wt') as fp_out:
            self.log.info("Running delwaq2 - might take a while...")
            self.log.info("  " + " ".join(cmd))
            
            sim_time=(self.stop_time-self.start_time).total_seconds()
            tail=MonTail(os.path.join(self.base_path,self.name+".mon"),
                         log=self.log,sim_time_seconds=sim_time)
            try:
                try:
                    ret=subprocess.check_call(cmd,shell=False,cwd=self.base_path,stdout=fp_out,
                                              stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as exc:
                    raise WaqException("delwaq2 exited with an error code - check %s"%output_filename)
            finally:
                tail.stop()

        self.log.info("delwaq2 ran in %.2fs"%(time.time()-t_start))

        # return value is not meaningful - have to scrape the output
        with open(output_filename,'rt') as fp:
            for line in fp:
                if 'Stopping the program' in line:
                    raise WaqException("Delwaq2 stopped early - check %s"%output_filename)
        self.log.info("Done")
            
    # Paths for Delft tools:
    @property
    def delft_path(self):
        # on linux probably one directory above bin directory
        if 'DELFT_SRC' not in os.environ:
            raise WaqException("Environment variable DELFT_SRC not defined")
        return os.environ['DELFT_SRC']
    @delft_path.setter
    def delft_path(self,p):
        os.environ['DELFT_SRC']=p
    @property
    def delft_bin(self):
        if 'DELFT_BIN' in os.environ:
            return os.environ['DELFT_BIN']
        return os.path.join(self.delft_path,'bin')
    @delft_bin.setter
    def delft_bin(self,p):
        os.environ['DELFT_BIN']=p
    @property
    def delwaq1_path(self):
        return os.path.join(self.delft_bin,'delwaq1')
    # ZZ had changed this so that an alternate delwaq2 could be specified.
    # instead, allow overwriting delwaq2_path. Note that, following ZZ's
    # convention, a relative path here is interpreted relative to delft_bin.
    _delwaq2_basename="delwaq2"
    @property
    def delwaq2_path(self):
        return os.path.join(self.delft_bin,self._delwaq2_basename)
    @delwaq2_path.setter
    def delwaq2_path(self,p):
        self._delwaq2_basename=p

    _share_path=None
    @property
    def share_path(self):
        if self._share_path is None:
            # this is where it would live for a freshly compiled, not installed
            # delft3d:
            return os.path.join(self.delft_path,'engines_gpl/waq/default')
        else:
            return self._share_path
    @share_path.setter
    def share_path(self,p):
        self._share_path=p
    @property
    def bloom_path(self):
        return os.path.join(self.share_path,'bloom.spe')
    @property
    def original_bloominp_path(self):
        # this gets copied into the model run directory
        return os.path.join(self.share_path,'bloominp.d09')
    @property
    def proc_path(self):
        return os.path.join(self.share_path,'proc_def')

    # plot process diagrams
    def cmd_plot_process(self,run_name='dwaq'):
        """ Build a process diagram and save to file.  Sorry, you have no voice 
        in choosing the filename
        """
        pd = process_diagram.ProcDiagram(waq_dir=self.base_path)
        pd.render_dot()
    def cmd_view_process(self,run_name='dwaq'):
        """ Build a process diagram and display
        """
        pd = process_diagram.ProcDiagram(waq_dir=self.base_path)
        pd.view_dot()

class InpFile(object):
    """ define/access/generate the text input file for delwaq1 and delwaq2.
    """
    def __init__(self,scenario):
        self.log=logging.getLogger(self.__class__.__name__)
        self.scenario=scenario

    def default_filename(self):
        return os.path.join(self.scenario.base_path,
                            self.scenario.name+".inp")

    def write(self,fn=None):
        inp_fn=fn or self.default_filename()

        with open(inp_fn,'wt') as fp:
            fp.write(self.text())

    def text(self):
        return "".join( [self.text_header(),
                         self.text_block01(),
                         self.text_block02(),
                         self.text_block03(),
                         self.text_block04(),
                         self.text_block05(),
                         self.text_block06(),
                         self.text_block07(),
                         self.text_block08(),
                         self.text_block09(),
                         self.text_block10()])

    def text_header(self):
        header="""1000 132 ';'    ; width of input and output, comment
;
; Type of DELWAQ input file:
; DELWAQ_VERSION_4.91
; Option for printing the report: verbose
; PRINT_OUTPUT_OPTION_4

"""
        return header

    @property
    def n_substances(self):
        return self.scenario.n_substances
    @property
    def n_active_substances(self):
        return self.scenario.n_active_substances
    @property
    def n_inactive_substances(self):
        return self.scenario.n_inactive_substances

    def fmt_time0(self):
        return self.scenario.fmt_datetime(self.scenario.time0) # e.g. "1990.08.05 00:00:00"
    def fmt_scu(self):
        # no support yet for clock units < 1 second
        assert(self.scenario.scu.microseconds==0)
        # it's important that the output be exactly this wide!
        return "{:8}s".format(self.scenario.scu.seconds + 86400*self.scenario.scu.days)
    def fmt_substance_names(self):
        return "\n".join( ["    {:4}  '{}'".format(1+idx,s.name)
                           for idx,s in enumerate(self.scenario.substances.values())] )

    @property
    def integration_option(self):
        return self.scenario.integration_option

    @property
    def desc(self):
        return self.scenario.desc

    def text_block01(self):
        block01="""; first block: identification
'{self.desc[0]}'
'{self.desc[1]}'
'{self.desc[2]}'
'T0: {time0}  (scu={scu})'
;
; substances file: n/a
; hydrodynamic file: n/a
;
; areachar.dat: n/a
;
  {self.n_active_substances}  {self.n_inactive_substances}    ; number of active and inactive substances

; Index  Name
{substances}
;
#1 ; delimiter for the first block
"""
        assert len(self.desc)==3,"Scenario.desc must have exactly 3 strings"
        return block01.format(self=self,
                              time0=self.fmt_time0(),scu=self.fmt_scu(),
                              substances=self.fmt_substance_names())

    @property
    def text_start_time(self):
        return self.scenario.fmt_datetime(self.scenario.start_time)
    @property
    def text_stop_time(self):
        return self.scenario.fmt_datetime(self.scenario.stop_time)

    @property
    def text_map_start_time(self):
        return self.scenario.fmt_datetime(self.scenario.map_start_time or 
                                          self.scenario.start_time)
    @property
    def text_map_stop_time(self):
        return self.scenario.fmt_datetime(self.scenario.map_stop_time or 
                                          self.scenario.stop_time)

    @property
    def text_hist_start_time(self):
        return self.scenario.fmt_datetime(self.scenario.hist_start_time or 
                                          self.scenario.start_time)
    @property
    def text_hist_stop_time(self):
        return self.scenario.fmt_datetime(self.scenario.hist_stop_time or 
                                          self.scenario.stop_time)
    @property
    def text_mon_start_time(self):
        return self.scenario.fmt_datetime(self.scenario.mon_start_time or 
                                          self.scenario.start_time)
    @property
    def text_mon_stop_time(self):
        return self.scenario.fmt_datetime(self.scenario.mon_stop_time or 
                                          self.scenario.stop_time)

    @property
    def time_step(self):
        return self.scenario.time_step
    @property
    def map_time_step(self):
        return self.scenario.map_time_step or self.scenario.time_step
    @property
    def hist_time_step(self):
        return self.scenario.hist_time_step or self.scenario.time_step
    @property
    def mon_time_step(self):
        return self.scenario.mon_time_step or self.scenario.time_step

    # start_time="1990/08/05-12:30:00"
    # stop_time ="1990/08/15-12:30:00"

    def text_monitor_areas(self):
        lines=["""
 1     ; monitoring points/areas used
 {n_points}   ; number of monitoring points/areas
""".format( n_points=len(self.scenario.monitor_areas) )]

        for name,segs in self.scenario.monitor_areas:
            # These can get quite long, so wrap the list of segments.
            # DWAQ can handle up to 1000 characters/line, but might as well
            # stop at 132 out of kindness.
            lines.append("'{}' {} {}".format(name,len(segs),
                                             textwrap.fill(" ".join(["%d"%(i+1) for i in segs]),
                                                           width=132)))

        return "\n".join(lines)

    def text_monitor_transects(self):
        n_transects=len(self.scenario.monitor_transects)

        if n_transects==0:
            return " 2     ; monitoring transects not used;\n"

        if len(self.scenario.monitor_areas)==0:
            # this is a real problem, though the code above for text_monitor_areas
            # has a kludge where it adds in a dummy monitoring area to avoid the issue
            raise Exception("DWAQ may not output transects when there are no monitor areas")

        lines=[" 1   ; monitoring transects used",
               " {n_transects} ; number of transects".format(n_transects=n_transects) ]
        for name,exchs in self.scenario.monitor_transects:
            # The 1 here is for reporting net flux.
            # split exchanges on multiple lines -- fortran may not appreciate
            # really long lines.
            lines.append("'{}' 1 {}".format(name,len(exchs)))
            lines+=["   %d"%i
                    for i in exchs]
        return "\n".join(lines)
    
    def text_monitor_start_stop(self):
        # not sure if the format matters - this was originally using dates like
        # 1990/08/05-12:30:00
        text="""; start time      stop time     time step 
 {self.text_mon_start_time}       {self.text_mon_stop_time}       {self.mon_time_step:08}      ; monitoring
 {self.text_map_start_time}       {self.text_stop_time}       {self.map_time_step:08}      ; map, dump
 {self.text_hist_start_time}       {self.text_hist_stop_time}       {self.hist_time_step:08}      ; history
"""
        return text.format(self=self)

    def text_block02(self):
        block02="""; 
; second block of model input (timers)
; 
; integration timers 
; 
 86400  'ddhhmmss' 'ddhhmmss' ; system clock in sec, aux in days
 {self.integration_option}    ; integration option
 {self.text_start_time}      ; start time 
 {self.text_stop_time}       ; stop time 
 0                  ; constant timestep 
 {self.time_step:07}      ; time step
;
{monitor_areas}
{monitor_transects}
{monitor_start_stop}
;
#2 ; delimiter for the second block
"""
        return block02.format(self=self,
                              monitor_areas=self.text_monitor_areas(),
                              monitor_transects=self.text_monitor_transects(),
                              monitor_start_stop=self.text_monitor_start_stop())
              
    @property
    def n_segments(self):
        return self.scenario.hydro.n_seg
    grid_layout=2 # docs suggest NONE would work, but seems to fail

    @property
    def multigrid_block(self):
        return self.scenario.multigrid_block

    @property
    def atr_filename(self):
        # updated to now include all of the attribute block
        return "com-{}.atr".format(self.scenario.name)

    #@property
    #def act_filename(self):
    #    return "com-{}.act".format(self.scenario.name)

    # when using existing hydro data these are usually symlinks,
    # but symlinks are not simple on windows filesystems. Support
    # an option to instead write an absolute or full path
    
    @property
    def vol_filename(self):
        #return "com-{}.vol".format(self.scenario.name)

        # 1. Hydro objects can write their data out. This can be
        #    new file, or a symlink to an original file.
        #    Hydro.vol_filename is the full path to this file.
        
        # 2. HydroFiles objects reference existing files, typically by
        #    reading a hyd file.
        #    HydroFiles.get_path('volumes-file') is a relative path
        #
        # Both of those are either absolute or relative to the working
        # directory.
        #
        # "com-<scenario>.vol" is a default name put in the inp file.

        fn=self.scenario.hydro.vol_filename 
        abs_fn=os.path.abspath(fn) # easier to compare paths
        abs_inp_path = os.path.abspath(self.scenario.base_path)
        rel_vol_path=os.path.relpath(abs_fn,abs_inp_path)
        return rel_vol_path

    @property
    def flo_filename(self):
        return os.path.relpath(os.path.abspath(self.scenario.hydro.flo_filename),
                               os.path.abspath(self.scenario.base_path))
        
        #return "com-{}.flo".format(self.scenario.name)

    @property
    def are_filename(self):
        #return "com-{}.are".format(self.scenario.name)
        return os.path.relpath(os.path.abspath(self.scenario.hydro.are_filename),
                               os.path.abspath(self.scenario.base_path))

    @property
    def poi_filename(self):
        return "com-{}.poi".format(self.scenario.name)

    @property
    def len_filename(self):
        return "com-{}.len".format(self.scenario.name)

    def text_block03(self):
        block03="""; 
; third block of model input (grid layout)
 {self.n_segments}      ; number of segments
{self.multigrid_block}       ; multigrid block
 {self.grid_layout}        ; grid layout not used
;
; features
INCLUDE '{self.atr_filename}'  ; attributes file
;
; volumes
;
-2  ; first volume option
'{self.vol_filename}'  ; volumes file
;
#3 ; delimiter for the third block
"""
        return block03.format( self=self )

    @property
    def n_exch_x(self):
        return self.scenario.hydro.n_exch_x
    @property
    def n_exch_y(self):
        return self.scenario.hydro.n_exch_y
    @property
    def n_exch_z(self):
        return self.scenario.hydro.n_exch_z

    @property 
    def n_dispersions(self):
        # each dispersion array can have a name (<=20 characters)
        #  and if there are any, then we have to know which array
        #  if any goes with each substance
        return len(self.scenario.dispersions)

    @property 
    def n_velocities(self):
        # same deal as dispersions
        return len(self.scenario.velocites)

    @property
    def dispersions_declaration(self):
        """ the count and substance assignment for dispersion arrays
        """
        lines=[" {} ; dispersion arrays".format(len(self.scenario.dispersions))]
        if len(self.scenario.dispersions):
            subs=list( self.scenario.substances.keys() )[:self.scenario.n_active_substances]
            assignments=np.zeros(len(subs),'i4') # 1-based

            for ai,a in enumerate(self.scenario.dispersions.values()):
                lines.append(" '{}'".format(a.name))
                for subi,sub in enumerate(subs):
                    if a.matches(sub):
                        assignments[subi]=ai+1 # to 1-based
            lines.append( " ".join(["%d"%assign for assign in assignments])  + " ; assign to substances" )
        else:
            self.log.info("No dispersion arrays, will skip assignment to substances")
        return "\n".join(lines)

    @property
    def velocities_declaration(self):
        """ the count and substance assignment for velocity arrays
        """
        lines=[" {} ; velocity arrays".format(len(self.scenario.velocities))]
        if len(self.scenario.velocities):
            subs=list( self.scenario.substances.keys() )[:self.scenario.n_active_substances]
            assignments=np.zeros(len(subs),'i4') # 1-based

            for ai,a in enumerate(self.scenario.velocities.values()):
                lines.append(" '{}'".format(a.name))
                for subi,sub in enumerate(subs):
                    if a.matches(sub):
                        assignments[subi]=ai+1 # to 1-based
            lines.append( " ".join(["%d"%assign for assign in assignments])  + " ; assign to substances" )
        else:
            self.log.info("No velocity arrays, will skip assignment to substances")
        return "\n".join(lines)
    
    @property
    def dispersions_definition(self):
        if len(self.scenario.dispersions)==0:
            return ""
        else:
            # This code is currently limited to constant in time.
            # for time-variable, the option here is not 1, but
            # 3.
            disps=self.scenario.dispersions.values()
            
            unsteady=np.array([d.times is not None for d in disps])
            if unsteady.min()!=unsteady.max():
                raise Exception("Multiple dispersion arrays specified, with some "
                                "constant and some unsteady. It's just too much.")
            unsteady=unsteady[0]
            hydro=self.scenario.hydro

            lines=['; Data option']

            if unsteady:
                # page 81 of D-Water Quality Description Input File Manual (1.5.4)
                lines.append( '3 ; information comes as time functions')

                if len(disps)>1:
                    self.log.warning("Brave soul! Multiple dispersion arrays have not been tested")
                    
                for disp in disps:
                    # RH: assuming that the multiple blocks correspond to potentially
                    # multiple dispersion arrays. But not sure about that.  Manual
                    # isn't clear, and there is at least the possibility of multiple
                    # dispersion arrays in a single block (but that would further require
                    # each to have the same time dimension).
                    lines.append( '2 ; this block -- information at breakpoints with linear time interpolation')
                    lines.append( '%d ; number of items, equals number of exchanges'%hydro.n_exch )
                    
                    lines.append( '; exchanges ids (all of them...)')
                    all_j=np.arange(hydro.n_exch)+1
                    per_line=10
                    while all_j.size:
                        lines.append( " ".join( [str(j) for j in all_j[:per_line]] ) )
                        all_j=all_j[per_line:]
                    lines.append( '%d ; number of breakpoints'%len(disp.times))
                    # code below treats x and z separately. weird. not sure.
                    # with all 3, I get an error here on the second value.  So try just one.
                    # that runs. Appears to do the right thing.
                    lines.append( '1.0 ; scale factors')
                    for ti,t in enumerate(disp.times):
                        lines.append( '%d ; time integer at breakpoint %d'%(t,ti) )
                        for exch_i in range(hydro.n_exch):
                            lines.append( "%.3e"%disp.data[ti,exch_i] )
            else:
                lines.append( '1 ; information is constant and provided without defaults')

                # add x direction:
                lines.append("1.0 ; scale factor for x")

                for exch_i in range(hydro.n_exch_x):
                    vals=[disp.data[exch_i] for disp in disps]
                    lines.append( " ".join(["%.3e"%v for v in vals]) + "; Kx")

                assert hydro.n_exch_y==0 # not implemented

                lines.append("1.0 ; scale factor for z")

                for exch_i in range(hydro.n_exch_x+hydro.n_exch_y,hydro.n_exch):
                    vals=[disp.data[exch_i] for disp in disps]
                    lines.append( " ".join(["%.3e"%v for v in vals]) + "; Kz" )

            # Write them out to a separate text file
            disp_filename='dispersions.dsp'
            with open(os.path.join(self.scenario.base_path,disp_filename),'wt') as fp:
                fp.write("\n".join(lines))
            return "INCLUDE '{}'".format(disp_filename)

    @property
    def velocities_definition(self):
        if len(self.scenario.velocities)==0:
            return ""
        else:
            lines=['1 ; ASCII data',
                   '1 ; information is constant in time, provided without defaults']

            hydro=self.scenario.hydro

            # add x direction:
            lines.append("1.0 ; scale factor for x")
            velos=self.scenario.velocities.values()

            for exch_i in range(hydro.n_exch_x):
                vals=[velo.data[exch_i] for velo in velos]
                lines.append( " ".join(["%.3e"%v for v in vals]) )

            assert hydro.n_exch_y==0 # not implemented
                
            lines.append("1.0 ; scale factor for z")

            for exch_i in range(hydro.n_exch_x+hydro.n_exch_y,hydro.n_exch):
                vals=[velo.data[exch_i] for velo in velos]
                lines.append( " ".join(["%.3e"%v for v in vals]) )
                
            # Write them out to a separate text file
            velo_filename='velocities.dat'
            with open(os.path.join(self.scenario.base_path,velo_filename),'wt') as fp:
                fp.write("\n".join(lines))
            return "INCLUDE '{}'".format(velo_filename)
        
    @property
    def base_x_dispersion(self):
        return self.scenario.base_x_dispersion
    @property
    def base_y_dispersion(self):
        return self.scenario.base_y_dispersion
    @property
    def base_z_dispersion(self):
        return self.scenario.base_z_dispersion

    def text_block04(self):
        block04="""; 
; fourth block of model input (transport)
 {self.n_exch_x}  ; exchanges in direction 1
 {self.n_exch_y}  ; exchanges in direction 2
 {self.n_exch_z}  ; exchanges in direction 3
; 
 {self.dispersions_declaration} ; dispersions
 {self.velocities_declaration} ; velocities
; 
 1  ; first form is used for input 
 0  ; exchange pointer option
'{self.poi_filename}'  ; pointers file
; 
 1  ; first dispersion option nr - these constants will be added in.
 1.0 1.0 1.0   ; scale factors in 3 directions
 {self.base_x_dispersion} {self.base_y_dispersion} {self.base_z_dispersion} ; dispersion in x,y,z directions
{self.dispersions_definition}
; 
 -2  ; first area option
'{self.are_filename}'  ; area file
; 
 -2  ; first flow option
'{self.flo_filename}'  ; flow file
; Velocities
{self.velocities_definition}
; Lengths
  1  ; length vary
 0   ; length option
'{self.len_filename}'  ; length file
;
#4 ; delimiter for the fourth block
"""
        return block04.format(self=self)

        # including explicit dispersion arrays:
        # this page: http://oss.deltares.nl/web/delft3d/delwaq/-/message_boards/view_message/583767;jsessionid=3C8F18A0BB9B95EE1FFE77F72764DD77
        # shows a text format
        # the GUI puts

    @property
    def text_boundary_defs(self):
        # a triple of strings for each boundary exchange
        # first is id, must be unique in 20 characters
        # second is name, freeform
        # third is type, will be matched with first 20 characters
        # to group boundaries together.

        # I used to think that these were labeled in order of their
        # appearance in pointers, but 2017-10-13, they appear to be
        # in accordance with the boundary segment.  This is [soon]
        # handled in boundary_defs()
        lines=[]

        #boundary_count=0
        bc_sets=[self.scenario.hydro.boundary_defs()]
        
        if self.scenario.bottom_grid is not None: # self.scenario.n_bottom_layers>0:
            bc_sets.append(self.scenario.deep_sediment_boundary_defs())
            
        for bc_set in bc_sets:
            for bdry in bc_set:
                #boundary_count+=1
                lines.append("'{}' '{}' '{}'".format( bdry['id'].decode(),
                                                      bdry['name'].decode(),
                                                      bdry['type'].decode() ) )
                
        # This used to get returned in-line, but maybe cleaner to put into separate
        # file.
        dir_name=self.scenario.base_path                    

        local_name = 'boundary_defs.txt'            
        fn=os.path.join(dir_name,local_name)
        
        data = "\n".join(lines)        
        with open(fn,'wt') as fp:
            fp.write(data)
        return "INCLUDE '%s' ; boundary definition file"%local_name
    
    @property
    def n_boundaries(self):
        return self.scenario.hydro.n_boundaries

    @property
    def text_overridings(self):
        lines=[ "{:5}    ; Number of overridings".format(self.n_boundaries) ]
        for bi in range(self.n_boundaries):
            lines.append( "  {:9}     00000000000000   ; Left-right 1".format(bi+1) )
        return "\n".join(lines)

    # Boundary condition definitions:

    @property
    def text_thatcher_harleman_lags(self):
        # not ready to build in explicit handling of this, so for
        # now delegate to Scenario (so at least customization is
        # centralized there)
        return self.scenario.text_thatcher_harleman_lags()

    @property
    def text_bc_items(self):
        lines=[]
        for bc in self.scenario.bcs:
            lines.append( bc.text() )
        return "\n".join(lines)

    def text_block05(self):
        block05="""; 
; fifth block of model input (boundary condition)
{self.text_boundary_defs}
{self.text_thatcher_harleman_lags}
{self.text_bc_items}
;
 #5 ; delimiter for the fifth block
"""
        return block05.format(self=self)

    @property
    def n_discharges(self):
        return len(self.scenario.discharges)

    @property
    def text_discharge_names(self):
        lines=[]
        for disch in self.scenario.discharges:
            # something like that - 
            lines.append( disch.text() )
        return "\n".join(lines)

    @property
    def text_discharge_items(self):
        lines=[]
        for load in self.scenario.loads:
            lines.append( load.text() )
        return "\n".join(lines)

    @property
    def par_filename(self):
        return self.scenario.name+".par"

    @property
    def vdf_filename(self):
        return "com-{}.vdf".format(self.scenario.name)

    def text_block06(self):
        block06="""; 
; sixth block of model input (discharges, withdrawals, waste loads)
   {self.n_discharges} ; number of waste loads/continuous releases
{self.text_discharge_names}
{self.text_discharge_items}
;
 #6 ; delimiter for the sixth block
""".format(self=self)
        return block06

    def text_block07(self):
        lines=['; seventh block of model input (process parameters)']

        for param in self.scenario.parameters.values():
            lines.append( param.text(write_supporting=True) )
        for param in self.scenario.hydro_parameters.values():
            # hydro.write() takes care of writing its own parameters
            lines.append( param.text(write_supporting=False) )

        lines.append("#7 ; delimiter for the seventh block")
        return "\n".join(lines)

    def text_block08(self):
        # unclear how to add spatially varying initial condition
        # in new style.
        return self.text_block08_old()

    def text_block08_new(self):
        """ new style initial conditions - NOT USED """

        defaults="\n".join([" {:e} ; {}".format(s.initial.default,s.name)
                            for s in self.scenario.substances.values() ])
        lines=["; ",
               "; eighth block of model input (initial conditions) ",
               " MASS/M2 ; unit for inactive substances",
               " INITIALS ",
               # list of substances
               " ".join( [" {} ".format(s)
                          for s in self.scenario.substances.values()] )]
        # pick up here.
        raise Exception("New style initial condition code not implemented yet")

        return "\n".join(lines)
        
    def text_block08_old(self):
        """ old-style initial conditions.  note this is old-style, not
        old code.  This is the version currently used!
        """
        lines=["; ",
               "; eighth block of model input (initial conditions) ",
               " MASS/M2 ; unit for inactive substances"]
        if self.scenario.restart_file is not None:
            res_file=self.scenario.restart_file
            # These filenames get corrupted in delwaq, so try to be a
            # little smart
            fns=glob.glob(res_file+'*')
            if res_file not in fns:
                if len(fns)==1:
                    self.scenario.log.warning("Will copy corrupt restart file to specified restart file")
                    shutil.copyfile(fns[0],res_file)
                elif len(fns)==0:
                    raise Exception("Restart file '%s' not found (including corrupted variants)"%res_file)
                else:
                    raise Exception("Restart file '%s' not found, but multiple corrupt alternatives"%res_file)
            
            lines+=[ " 0 ; Restart file",
                     "%s ; Binary restart file"%res_file ]
        else:
            lines+=[ " 1 ; initial conditions follow"]

            # are any initial conditions spatially varying?
            # if so, then skip defaults and specify all substances, everywhere
            for s in self.scenario.substances.values():
                if s.initial.seg_values is not None:
                    lines+=self.text_ic_old_spatially_varying()
                    break
            else:
                # otherwise, just give defaults:
                defaults="\n".join([" {:e} ; {}".format(s.initial.default,s.name)
                                for s in self.scenario.substances.values() ])
                lines+=[ " 2 ; all values with default",
                         "{self.n_substances}*1.0 ; scale factors".format(self=self),
                         defaults,
                         " 0  ; overridings"]

        lines+=[ ";",
                 " #8 ; delimiter for the eighth block"]

        return "\n".join(lines)

    def text_ic_old_spatially_varying(self):
        """ return lines for initial conditions when they are spatially varying
        """
        # use transpose, so that it's easy to write defaults when we have them,
        # and spatially varying when needed
        subs=self.scenario.substances.values()

        lines=["TRANSPOSE",
               "1 ; without defaults",
               "1.0 ; scaling for all substances"]

        # "{}*1.0 ; no scaling".format(len(subs))] # doesn't work!

        # with TRANSPOSE we get only one scale factor total.

        if self.scenario.bottom_grid is not None:
            n_seg_sediment=self.scenario.n_bottom_layers * self.scenario.hydro.n_2d_elements
        else:
            n_seg_sediment = 0 # might have bed state in DelwaqG, but that is handled elsewhere.
            
        # RH: Assuming that old-style ICs with a layered bed need to specify values for both
        # water segments and bed segments all together.
        n_seg_total = self.scenario.hydro.n_seg + n_seg_sediment
        
        for s in subs:
            if s.initial.seg_values is not None:
                # Write this out to a separate file, and INCLUDE it
                supporting_fn='initial-%s.dat'%s.name
                with open(os.path.join(self.scenario.base_path,supporting_fn),'wt') as fp:
                    fp.write(" ; spatially varying for {}\n".format(s.name) )
                    seg_values=s.initial.seg_values
                    if len(seg_values) < n_seg_total:
                        print("Padding %s with zeros for sediment segments"%s.name)
                        seg_values =np.concatenate( [seg_values, np.zeros(n_seg_total - len(seg_values))])
                    supp_lines= [" %f"%val for val in seg_values]
                    fp.write("\n".join(supp_lines))

                lines.append("INCLUDE '%s' ; spatially varying for %s"%(supporting_fn,s.name) )
            else:
                lines.append( " %d*%f ; default for %s"%(n_seg_total, 
                                                         s.initial.default,s.name) )
        return lines

    def text_block09(self):
        lines=[";",
               " ; ninth block of model input (specification of output)",
               "1 ; output information in this file" ]
        MONITOR='monitor'
        GRID='grid dump'
        HIS='history'
        MAP='map'

        outputs=[self.scenario.mon_output,
                 self.scenario.grid_output,
                 self.scenario.hist_output,
                 self.scenario.map_output]
        for spec,output_type in zip(outputs,
                                    [MONITOR,GRID,HIS,MAP]):
            spec=list(np.unique(spec))
            if output_type in [MONITOR,HIS]:
                weighing=" ' '"
            else:
                weighing=""
            vnames=["  '{name}' {weighing}".format(name=name,weighing=weighing)
                    for name in spec
                    if name!=DEFAULT]
            if len(spec)==0:
                lines.append('0 ; no output for {}'.format(output_type))
            elif DEFAULT in spec:
                if vnames:
                    lines.append(" 2 ; all substances and extra output, {}".format(output_type))
                    lines.append(" {} ; number of extra".format(len(vnames)))
                    lines+=vnames
                else:
                    lines.append('  1 ; only default, {} output'.format(output_type))
            else:
                lines.append("  3 ; only extras, {} output".format(output_type))
                lines.append("{} ; number of extra".format(len(vnames)))
                lines+=vnames

        if 1: # allow formats to be configurable, too
            lines.append( "%d ; binary history file"%int('binary' in self.scenario.history_formats) )
            lines.append( "%d ; binary map file    "%int('binary' in self.scenario.map_formats)     )
            lines.append( "%d ; nefis history file "%int('nefis'  in self.scenario.history_formats) )
            lines.append( "%d ; nefis map file     "%int('nefis'  in self.scenario.map_formats)     )
        else: # old hardcoded approach:
            lines += ["  1 ; binary history file on",
                      "  0 ; binary map     file on",
                      "  1 ; nefis  history file on",
                      "  1 ; nefis  map     file on"]
        lines+=["; ",
                " #9 ; delimiter for the ninth block"]

        return "\n".join(lines)

    def text_block10(self):
        lines=[";",
               "; Statistical output"]

        for sub in self.scenario.stat_output:
            lines+=["output-operation 'STADAY'"
                    "  substance '%s'"%sub,
                    "  suffix    ' '",
                    "  time-parameter 'TINIT' 'START'",
                    "  time-parameter 'PERIOD' '0000/00/01-00:00:00'",
                    "end-output-operation"]
        
        lines+=["#10 ; delimiter for the tenth block "]
        return "\n".join(lines)

    
##

class WaqModelBase(scriptable.Scriptable):
    """
    New-style
    rather than requiring all information to be provided up front,
    this version of the class follows the simpler approach of
    the HydroModel class, where the model state can be built up
    as needed, loaded, written, etc.

    Also transitions to broader usage of np.datetime64 instead of python datetime.

    WaqModelBase is for code common to both offline and online D-WAQ runs.
    WaqModel is the subclass for offline-specific code, and WaqOnlineModel
    handles specifics of running online under a DFlowModel instance.
    """
    name="waqmodel" # this is used for the basename of the various files.

    # simplify imports
    Sub=Substance

    # python datetime giving the reference time for the run.
    # system clock unit. Trying to move away from python datetime, just use
    # numpy datetime64 to be consistent with pandas, numpy, xarray.
    scu64=np.timedelta64(1,'s')
    # but provide a read-only field for queries by Hydro.
    @property
    def scu(self):
        return datetime.timedelta(seconds=self.scu64/np.timedelta64(1,'s'))

    # These should all be np.datetime64, unlike WaqScenario
    time0=None
    start_time=None # 
    stop_time=None  # 

    time_step=None  # taken from hydro, unless specified otherwise


    # These are not fully supported for WaqOnlineModel, but included
    # here to facilitate setup of runs that could be online or offline.
    mon_output= (DEFAULT,'SURF','LocalDepth') # monitor file
    grid_output=('SURF','LocalDepth')              # grid topo
    hist_output=(DEFAULT,'SURF','LocalDepth') # history file
    map_output =(DEFAULT,'SURF','TotalDepth','LocalDepth')  # map file
    # Stat output is currently not very configurable, and just allows
    # specifying which variables to use. See text_block10() above
    # for the specifics that are used.
    stat_output=() # defaults to none

    delwaqg_initial=None

    # easier to handle multi-platform read/write with binary, compared to nefis output
    map_formats=('binary',)
    history_formats=('binary',)

    def __init__(self,**kw):
        self.log=logging.getLogger(self.__class__.__name__)

        # list of dicts.  moving towards standard setting of loads via src_tags
        # should be populated in init_substances(), gets used 
        self.src_tags=[]
        # Not sure whether dispersions and velocities can be utilized in
        # online simulations at this time (2021-06-21)
        self.dispersions=NamedObjects(scenario=self)
        self.velocities=NamedObjects(scenario=self)
        self.processes=[]
        self.parameters=self.init_parameters()
        self.substances=self.init_substances()
        assert self.substances is not None, "init_substances() did not return anything."
        
        utils.set_keywords(self,kw)

        self.init_bcs()
        self.init_loads()

    # scriptable interface settings:
    cli_options="hp:"
    def cli_handle_option(self,opt,val):
        if opt=='-p':
            print("Setting base_path to '%s'"%val)
            self.base_path=val
        else:
            super(Scenario,self).cli_handle_option(opt,val)
        
    @property
    def n_substances(self):
        return len(self.substances)
    @property
    def n_active_substances(self):
        return len( [sub for sub in self.substances.values() if sub.active] )
    @property
    def n_inactive_substances(self):
        return len( [sub for sub in self.substances.values() if not sub.active] )

    def add_process(self,name):
        self.processes.append(name)
    
    def init_parameters(self):
        params=NamedObjects(scenario=self,cast_value=cast_to_parameter)
        params['ONLY_ACTIVE']=1 # almost always a good idea.
        self.add_process("DYNDEPTH")
        self.add_process("TOTDEPTH")
        
        return params
    
    def init_hydro_parameters(self):
        """ parameters which come directly from the hydro, and are
        written out in the same way that process parameters are 
        written.
        """
        if self.hydro:
            # in case hydro is re-used, make sure that this call gets a fresh
            # set of parameters.  some leaky abstraction going on...
            return self.hydro.parameters(force=True)
        else:
            self.log.warning("Why requesting hydro parameters with no hydro?")
            raise Exception("Tried to initialize hydro parameters without a hydro instance")

    def init_substances(self):
        # sorts active substances first.
        return NamedObjects(scenario=self,sort_key=lambda s: not s.active)

    def init_bcs(self):
        self.bcs=[]

        # RH 2019-09-09: moving to more procedural approach, unsure of whether this
        # should be included.
        if 0:
            # 2017-03-17: moved this bit of boiler plate from a bunch of subclasses
            # to here in a step towards standardizing the BC settings.
            boundaries=self.hydro.boundary_defs()
            ids=[b['id'] for b in boundaries]

            for src_tag in self.src_tags:
                # conc. defaults to 1.0
                self.add_bc(src_tag['items'],src_tag['tracer'],src_tag.get('value',1.0))

    def add_bc(self,*args,**kws):
        bc=BoundaryCondition(*args,**kws)
        bc.scenario=self
        self.bcs.append(bc)
        return bc

    def init_loads(self):
        """
        Set up discharges (locations of point sources/sinks), and
        corresponding loads (e.g. mass/time for a substance at a source)
        """
        self.discharges=[]
        self.loads=[]

    def add_discharge(self,on_exists='exception',*arg,**kws):
        disch=Discharge(*arg,**kws)
        disch.scenario=self
        disch.update_fields()
        exists=False
        for other in self.discharges:
            if other.load_id==disch.load_id:
                if on_exists=='exception':
                    raise Exception("Discharge with id='%s' already exists"%other.load_id)
                elif on_exists=='ignore':
                    self.log.info("Discharge id='%s' exists - skipping duplicate"%other.load_id)
                    return other
                else:
                    assert False 
        self.discharges.append(disch)
        return disch

    def add_load(self,*args,**kws):
        load=Load(*args,**kws)
        load.scenario=self
        self.loads.append(load)
        return load

    # Requires further refactoring to separate online/offline specific
    # code
    def add_monitor_from_shp(self,shp_fn,naming='elt_layer',point_layers=True):
        raise Exception("add_monitor_from_shp is sub-class dependent, and apparently not implemented")

    def add_transect(self,name,exchanges):
        raise Exception("add_transect is sub-class dependent, and apparently not implemented")

    def add_area_boundary_transects(self,exclude='dummy'):
        raise Exception("add_are_boundary_transects is sub-class dependent, and apparently not implemented")

    def add_transects_from_shp(self,shp_fn,naming='count',clip_to_poly=True,
                               on_boundary='warn_and_skip',on_edge=False):
        raise Exception("add_transects_from_shp is sub-class dependent, and apparently not implemented")

    def add_monitor_for_geometry(self,name,geom):
        raise Exception("add_monitor_for_geometry is sub-class dependent, and apparently not implemented")
    
    _pdb=None
    @property
    def process_db(self):
        if self._pdb is None:
            self._pdb = waq_process.ProcessDB(scenario=self)
        return self._pdb
    def lookup_item(self,name):
        return self.process_db.substance_by_id(name)

    def as_datetime(self,t):
        """
        t can be a datetime object, an integer number of seconds since time0,
        or a floating point datenum
        """
        if np.issubdtype(type(t),np.integer):
            return utils.to_datetime(self.time0 + t*self.scu64)
        else:
            return utils.to_datetime(t)

    def fmt_datetime(self,t):
        """ 
        return datetime formatted as text.
        format is part of input file configuration, but 
        for now, stick with 1990/08/15-12:30:00

        t is specified as in as_datetime() above.
        """
        return self.as_datetime(t).strftime('%Y/%m/%d-%H:%M:%S')

    def flowgeom_ds(self):
        # Unclear whether we really need flowgeom_ds, or if a grid
        # is sufficient.  For online simulations, would rather just
        # reference self.model.grid, but it's not quite the same as flowgeom_ds
        raise Exception("flowgeom_ds not implemented in base class")

    def cmd_write_bloominp(self):
        """
        Copy supporting bloominp file for runs using BLOOM algae
        """
        if not os.path.exists(self.original_bloominp_path):
            # pdb.set_trace() 
            self.log.warning("BLOOM not found (%s)! Tread carefully"%self.original_bloominp_path)
            return
        
        dst=os.path.join(self.base_path,'bloominp.d09')
        if os.path.exists(dst) and not self.overwrite:
            raise Exception("%s exists, but overwrite is False"%dst)
        
        shutil.copyfile(self.original_bloominp_path,dst)

    # Paths for Delft tools:
    @property
    def delft_path(self):
        # on linux probably one directory above bin directory
        if 'DELFT_SRC' not in os.environ:
            raise WaqException("Environment variable DELFT_SRC not defined")
        return os.environ['DELFT_SRC']
    @delft_path.setter
    def delft_path(self,p):
        os.environ['DELFT_SRC']=p
    @property
    def delft_bin(self):
        if 'DELFT_BIN' in os.environ:
            return os.environ['DELFT_BIN']
        return os.path.join(self.delft_path,'bin')
    @delft_bin.setter
    def delft_bin(self,p):
        os.environ['DELFT_BIN']=p

    _share_path=None
    @property
    def share_path(self):
        
        if self._share_path is not None:
            return self._share_path
        elif 'DELFT_SHARE' in os.environ:
            return os.environ['DELFT_SHARE']
        else:
            # this is where it would live for a freshly compiled, not installed
            # delft3d:
            return os.path.join(self.delft_path,'engines_gpl/waq/default')

    @share_path.setter
    def share_path(self,p):
        self._share_path=p
    @property
    def bloom_path(self):
        return os.path.join(self.share_path,'bloom.spe')
    @property
    def original_bloominp_path(self):
        # this gets copied into the model run directory
        return os.path.join(self.share_path,'bloominp.d09')

    _waq_proc_def=None
    @property
    def waq_proc_def(self):
        if self._waq_proc_def is not None:
            return self._waq_proc_def
        return os.path.join(self.share_path,'proc_def')
    
    @waq_proc_def.setter
    def waq_proc_def(self,value):
        if value.endswith('.def') or value.endswith('.dat'):
            raise ValueError("waq_proc_def should not include the extension")
        self._waq_proc_def=value

    # plot process diagrams
    def cmd_plot_process(self,run_name='dwaq'):
        """ Build a process diagram and save to file.  Sorry, you have no voice 
        in choosing the filename
        """
        pd = process_diagram.ProcDiagram(waq_dir=self.base_path)
        pd.render_dot()
    def cmd_view_process(self,run_name='dwaq'):
        """ Build a process diagram and display
        """
        pd = process_diagram.ProcDiagram(waq_dir=self.base_path)
        pd.view_dot()


class WaqModel(WaqModelBase):
    """
    code for D-WAQ setup specific to running offline
    """
    desc=('DWAQ','n/a','n/a') # 3 arbitrary lines of text for description

    overwrite=False 

    # if set, initial conditions are taken from a restart file given here
    restart_file=None
    
    # backward differencing,
    # .60 => second and third keywords are set 
    # => no dispersion across open boundary
    # => lower order at boundaries
    integration_option="15.60"

    #  add quantities to default
    DEFAULT=DEFAULT

    water_grid=None # from old style layered bed. Leave None
    bottom_grid=None
    bottom_layers=[] # thickness of layers, used for delwaqg

    # settings related to paths - a little sneaky, to allow for shorthand
    # to select the next non-existing subdirectory by setting base_path to
    # "auto"
    _base_path="dwaq"
    @property
    def base_path(self):
        if self._base_path=='auto':
            self._base_path=self.auto_base_path()            
        return self._base_path
    @base_path.setter
    def base_path(self,v):
        self._base_path=v

    # tuples of name, segment id
    # some confusion about whether that's a segment id or list thereof
    # monitor_areas=[ ('single (0)',[1]),
    #                 ('single (1)',[2]),
    #                 ('single (2)',[3]),
    #                 ('single (3)',[4]),
    #                 ('single (4)',[5]) ]
    # dwaq bug(?) where having transects but no monitor_areas means history
    # file with transects is not written.  so always include a dummy:
    # the output code handles adding 1, so these should be stored zero-based.
    monitor_areas=( ('dummy',[0]), ) 

    # e.g. ( ('gg_outside', [24,26,-21,-27,344] ), ...  )
    # where negative signs mean to flip the sign of that exchange.
    # note that these exchanges are ONE-BASED - this is because the
    # sign convention is wrapped into the sign, so a zero exchange would
    # be ambiguous.
    monitor_transects=()

    # These used to default to 1,1, and 1e-7.
    # safer to default to 0.0, make the user select something non-zero.
    base_x_dispersion=0.0 # m2/s
    base_y_dispersion=0.0 # m2/s
    base_z_dispersion=0.0 # m2/s

    # these default to simulation start/stop/timestep
    map_start_time=None
    map_stop_time=None
    map_time_step=None
    # likewise
    hist_start_time=None
    hist_stop_time=None
    hist_time_step=None
    # and more
    mon_start_time=None
    mon_stop_time=None
    mon_time_step=None

    # getter/setter handle grabbing parameters from the hydro when
    # set
    _hydro=None

    def auto_base_path(self):
        ymd=datetime.datetime.now().strftime('%Y%m%d')        
        prefix='dwaq%s'%ymd
        for c in range(100):
            base_path=prefix+'%02d'%c
            if not os.path.exists(base_path):
                return base_path
        else:
            raise Exception("Possible run-away directory naming")
        
    @property
    def multigrid_block(self):
        """ 
        inserted verbatim in section 3 of input file.
        """
        # appears that for a lot of processes, hydro must be dense wrt segments
        # exchanges need not be dense, but sparse exchanges necessitate explicitly
        # providing the number of layers.  And that brings us to this stanza:

        # sparse_layers is actually an input flag to the aggregator
        # assert not self.hydro.sparse_layers
        # instead, test this programmatically, and roughly the same as how dwaq will test
        self.hydro.infer_2d_elements()
        kmax=self.hydro.seg_k.max()
        if self.hydro.n_seg != self.hydro.n_2d_elements*(kmax+1):
            raise Exception("You probably mean to be running with segment-dense hydrodynamics")
        
        num_layers=self.hydro.n_seg / self.hydro.n_2d_elements
        if self.hydro.vertical != self.hydro.SIGMA:
            return """MULTIGRID
  ZMODEL NOLAY %d
END_MULTIGRID"""%num_layers
        else:
            return " ; sigma layers - no multigrid stanza"

    @property
    def hydro(self):
        return self._hydro
    @hydro.setter
    def hydro(self,value):
        self.set_hydro(value)

    @classmethod
    def load(cls,path,load_hydro=True):
        """
        Working towards a similar ability as in DFlowModel, where an existing
        run can be loaded.
        """
        model=cls(base_path=path)
        model.overwrite=False
        # currently very little here...

        if load_hydro:
            # Try to guess what the right hyd file is
            model.load_hydro()
        return model

    def load_hydro(self):
        hyds=glob.glob(os.path.join(self.base_path,"*.hyd"))
        if len(hyds)==1:
            self.hydro=HydroFiles(hyd_path=hyds[0])
        else:
            log.info("Could not detect a load-able hyd file")
        return self.hydro
    
    def set_hydro(self,hydro):
        self._hydro=hydro

        if hydro is None:
            return

        self._hydro.scenario=self

        # I think it's required that these match
        self.time0 = self._hydro.time0

        # Other time-related values needn't match, but the hydro
        # is a reasonable default if no value has been set yet.
        start,stop,dt=hydro.timeline_data()
        
        if self.time_step is None:
            # note that this is a dwaq-formatted time step, as an integer.
            # DDDHHMMSS
            self.time_step=hydro.time_step

        if self.start_time is None:
            self.start_time=utils.to_dt64(start)
            self.log.info(" start time updated from hydro: %s"%self.start_time)
        if self.stop_time is None:
            self.stop_time=utils.to_dt64(stop)
            self.log.info(" stop time update from hydro: %s"%self.stop_time)
            
        if self.stop_time<self.start_time:
            self.log.warning("Stop time is before than start time")
            
        self.hydro_parameters=self.init_hydro_parameters()
        self.log.info("Parameters gleaned from hydro: %s"%self.hydro_parameters)

    def text_thatcher_harleman_lags(self):
        return """;
; Thatcher-Harleman timelags
0 ; no lags
        """

    def add_monitor_from_shp(self,shp_fn,naming='elt_layer',point_layers=True):
        """
        For each feature in the shapefile, add a monitor area.
        shp_fn: path to shapefile
        naming: generally, a field name from the shapefile giving the name of the monitor area.
          special case when point_layers is True and the shapefile has points, naming can
          be "elt_layer" in which case individual segments are monitored separately, and
          named like elt123_layer45.
        """
        assert self.hydro,"Set hydro before calling add_monitor_from_shp"

        locations=wkb2shp.shp2geom(shp_fn)
        self.hydro.infer_2d_elements()

        g=self.hydro.grid()
        new_areas=[] # generate as list, then assign as tuple
        names={}
        for n,segs in self.monitor_areas:
            names[n]=True # record names in use to avoid duplicates

        for i,rec in enumerate(locations):
            geom=rec['geom']

            if point_layers and (geom.type=='Point'):
                xy=np.array(geom.coords)[0]
                # would be better to have select_cells_nearest use
                # centroids instead of circumcenters, but barring
                # that, throw the net wide and look at up to 40
                # cells.
                elt=g.select_cells_nearest(xy,inside=True,count=40)
                if elt is None:
                    self.log.warning("Monitor point %s was not found inside a cell"%xy)
                    # Fall back to nearest
                    elt=g.select_cells_nearest(xy,inside=False)

                segs=np.nonzero( self.hydro.seg_to_2d_element==elt )[0]
                for layer,seg in enumerate(segs):
                    if naming=='elt_layer':
                        name="elt%d_layer%d"%(elt,layer)
                    else:
                        name="%s_layer%d"%(rec[naming],layer)
                    if name not in names:
                        new_area=(name,[seg])
                        self.monitor_areas = self.monitor_areas + (new_area,)
                        names[name]=True
                    else:
                        self.log.warning("Duplicate requests to monitor %s"%name)
            else:
                try:
                    name=rec[naming]
                except:
                    name="mon%d"%len(self.monitor_areas)
                self.add_monitor_for_geometry(name=name,geom=geom)
                
    def add_monitor_for_geometry(self,name,geom):
        """
        Add a monitor area for elements intersecting the shapely geometry geom.
        """
        # make sure the name is unique
        for n,segs in self.monitor_areas:
            assert name!=n
        
        g=self.hydro.grid()
        # bitmask over 2D elements
        self.log.info("Selecting elements in polygon '%s'"%name)
        # better to go by center, so that non-intersecting polygons
        # yield non-intersecting sets of elements and segments
        # 2019-09-12: use centroid instead of center in case the grid has weird geometry
        # 2020-01-31: use representative points. It's possible for centroid not to fall within
        #      the polygon.  Don't use full polygon, because that will pick up adjacent 
        #      cells that share an edge.
        elt_sel=g.select_cells_intersecting(geom,by_center='representative') # few seconds

        # extend to segments:
        seg_sel=elt_sel[ self.hydro.seg_to_2d_element ] & (self.hydro.seg_to_2d_element>=0)         
        segs=np.nonzero( seg_sel )[0]

        new_area= (name,segs) 

        self.log.info("Added %d monitored segments for %s"%(len(segs),name))
        self.monitor_areas = self.monitor_areas + ( new_area, )

    def add_transects_from_shp(self,shp_fn,naming='count',clip_to_poly=True,
                               on_boundary='warn_and_skip',on_edge=False,
                               add_station=False):
        """
        Add monitor transects from a shapefile.
        By default transects are named in sequence.  
        Specify a shapefile field name in 'naming' to pull user-specified
        names for the transects.
        """
        if on_edge:
            # RH 2020-01-31: on_edge is from ZZ code. I'm not clear on the purpose,
            #  it's now included for back-compatibilty in the WaqScenario class, but
            #  moving forward I'm trying to discourage it unless a specific use-case
            #  arises.
            self.log.warning("add_transects_from_shp: on_edge was set. In WaqModel this does nothing")
                             
        assert self.hydro,"Set hydro before calling add_transects_from_shp"
        locations=wkb2shp.shp2geom(shp_fn)
        g=self.hydro.grid()

        if clip_to_poly:
            poly=g.boundary_polygon()
            
        new_transects=[] # generate as list, then assign as tuple

        for i,rec in enumerate(locations):
            geom=rec['geom']

            if geom.type=='LineString':
                if clip_to_poly:
                    clipped=geom.intersection(poly)

                    # rather than assume that clipped comes back with
                    # the same orientation, and multiple pieces come
                    # back in order, manually re-assemble the line
                    if clipped.type=='LineString':
                        segs=[clipped]
                    else:
                        segs=clipped.geoms

                    all_dists=[]
                    for seg in segs:
                        for xy in seg.coords:
                            all_dists.append( geom.project( geometry.Point(xy) ) )
                    # sorting the distances ensures that orientation is same as
                    # original
                    all_dists.sort()

                    xy=[geom.interpolate(d) for d in all_dists]
                else:
                    xy=np.array(geom.coords)
                    
                if naming=='count':
                    name="transect%04d"%i
                else:
                    name=rec[naming]
                exchs=self.hydro.path_to_transect_exchanges(xy,on_boundary=on_boundary)
                new_transects.append( (name,exchs) )
            else:
                self.log.warning("Not ready to handle geometry type %s"%geom.type)
        self.log.info("Added %d monitored transects from %s"%(len(new_transects),shp_fn))
        self.monitor_transects = self.monitor_transects + tuple(new_transects)
        if add_station:
            self.add_station_for_transects(new_transects) # Not yet implement for Scenario

    def add_station_for_transects(self,transects):
        stations=[]

        self.hydro.infer_2d_links()

        elt_depth=self.hydro.element_depth(t_secs=0)

        for tran in transects:
            name,exchs=tran
            name='s_'+name # hopefully doesn't make it too long...
            exch0s = np.abs(exchs) - 1
            links = np.unique(self.hydro.exch_to_2d_link['link'][exch0s])
            elts = np.unique(self.hydro.links[links])
            elt = elts[ np.argmax( elt_depth[elts] ) ]

            segs=np.nonzero( self.hydro.seg_to_2d_element==elt )[0] # probably slow
            stations.append( (name,segs) )
        self.monitor_areas = self.monitor_areas + tuple(stations)
            
    def add_area_boundary_transects(self,exclude='dummy'):
        """
        create monitor transects for the common boundaries between a subset of
        monitor areas. this assumes that the monitor areas are distinct - no
        overlapping cells (in fact it asserts this).
        The method and the non-overlapping requirement apply only for areas which
        do *not* match the exclude regex.
        """
        assert self.hydro,"Set hydro before calling add_area_boundary_transects"
        
        areas=[a[0] for a in self.monitor_areas]
        if exclude is not None:
            areas=[a for a in areas if not re.match(exclude,a)]

        mon_areas=dict(self.monitor_areas)

        seg_to_area=np.zeros(self.hydro.n_seg,'i4')-1

        for idx,name in enumerate(areas):
            # make sure of no overlap:
            assert np.all( seg_to_area[ mon_areas[name] ] == -1 )
            # and label to this area:
            seg_to_area[ mon_areas[name] ] = idx

        poi0=self.hydro.pointers - 1

        exch_areas=seg_to_area[poi0[:,:2]]
        # fix up negatives in poi0
        exch_areas[ poi0[:,:2]<0 ] = -1

        # convert to tuples so we can get unique pairs
        exch_areas_tupes=set( [ tuple(x) for x in exch_areas if x[0]!=x[1] and x[0]>=0 ] )
        # make the order canonical 
        canon=set()
        for a,b in exch_areas_tupes:
            if a>b:
                a,b=b,a
            canon.add( (a,b) )
        canon=list(canon) # re-assert order

        names=[]
        exch1s=[]

        for a,b in canon:
            self.log.info("%s <-> %s"%(areas[a],areas[b]))
            name=areas[a][:9] + "__" + areas[b][:9]
            self.log.info("  name: %s"%name)
            names.append(name)

            fwd=np.nonzero( (exch_areas[:,0]==a) & (exch_areas[:,1]==b) )[0]
            rev=np.nonzero( (exch_areas[:,1]==a) & (exch_areas[:,0]==b) )[0]
            exch1s.append( np.concatenate( (fwd+1, -(rev+1)) ) )
            self.log.info("  exchange count: %d"%len(exch1s[-1]))

        # and add to transects:
        transects=tuple(zip(names,exch1s))
        self.monitor_transects=self.monitor_transects + transects
        
    def add_transect(self,name,exchanges):
        """ Append a transect definition for logging.
        """
        self.monitor_transects = self.monitor_transects + ( (name,exchanges), )
        
    def ensure_base_path(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def write_inp(self):
        """
        Write the inp file for delwaq1/delwaq2
        """
        self.ensure_base_path()
        # parameter files are also written along the way
        for process in self.processes:
            self.parameters['ACTIVE_'+process]=1
        inp=InpFile(scenario=self)
        inp.write()

    def write_hydro(self):
        self.ensure_base_path()
        self.hydro.overwrite=self.overwrite
        self.hydro.write()
    
    #-- Access to output files
    def nef_history(self):
        hda=os.path.join( self.base_path,self.name+".hda")
        hdf=os.path.join( self.base_path,self.name+".hdf")
        if os.path.exists(hda):
            if nefis.nef_lib() is None:
                self.log.warning("Nefis library not configured -- will not read nef history")
                return None
            return nefis.Nefis(hda, hdf)
        else:
            return None
    def nef_map(self):
        ada=os.path.join(self.base_path, self.name+".ada")
        adf=os.path.join(self.base_path, self.name+".adf")
        if os.path.exists(ada):
            return nefis.Nefis( ada,adf)
        else:
            return None

    #  netcdf versions of those:
    def nc_map(self,nc_kwargs={}):
        nef=self.nef_map()
        try:
            elt_map,real_map = map_nef_names(nef)

            nc=nefis_nc.nefis_to_nc(nef,element_map=elt_map,nc_kwargs=nc_kwargs)
            for vname,rname in iteritems(real_map):
                nc.variables[vname].original_name=rname
            # the nefis file does not contain enough information to get
            # time back to a real calendar, so rely on the Scenario's
            # version of time0
            if 'time' in nc.variables:
                nc.time.units='seconds since %s'%self.time0.strftime('%Y-%m-%d %H:%M:%S')
        finally:
            nef.close()
        return nc

    def hist_ds(self):
        """
        Shifting to a slightly more standarized way of accessing history output
        """
        # this is the file that we postprocess and write out
        hist_nc_fn=os.path.join(self.base_path,'dwaq_hist.nc')

        if not os.path.exists(hist_nc_fn):
            self.cmd_write_his_nc()
            
        assert os.path.exists(hist_nc_fn),"Trouble writing history nc file"
        return xr.open_dataset(hist_nc_fn)

    def map_ds_path(self):
        return os.path.join(self.base_path,"dwaq_map.nc")
        
    def map_ds(self):
        """
        Shifting to a slightly more standarized way of accessing map output
        """
        map_nc_fn=self.map_ds_path()
        
        if not os.path.exists(map_nc_fn):
            self.cmd_write_map_nc()
            assert os.path.exists(map_nc_fn),"Trouble writing map nc file"
        return xr.open_dataset(map_nc_fn)
    
    # try to find the common chunks of code between writing ugrid
    # nc output and the history output
    def ugrid_map(self,nef=None,nc_kwargs={}):
        return self.ugrid_nef(mode='map',nef=nef,nc_kwargs=nc_kwargs)

    def ugrid_history(self,nef=None,nc_kwargs={}):
        return self.ugrid_nef(mode='history',nef=nef,nc_kwargs=nc_kwargs)

    default_ugrid_output_settings=['quickplot_compat']
    
    def ugrid_nef(self,mode='map',nef=None,nc_kwargs={},output_settings=None):
        """ Like nc_map, but write a netcdf file more ugrid compliant.
        this is actually pretty different, as ugrid requires that 3D
        field is organized into a horizontal dimension (i.e element)
        and vertical dimension (layer).  the original nefis code
        just gives segment.
        nef: supply an already open nef.  Note that caller is responsible for closing it!
        mode: 'map' use the map output
              'history' use history output
        """
        if output_settings is None:
            output_settings=self.default_ugrid_output_settings

        if nef is None:
            if mode=='map':
                nef=self.nef_map()
            elif mode=='history':
                nef=self.nef_history()
            close_nef=True
        else:
            close_nef=False
        if nef is None: # file didn't exist
            self.log.info("NEFIS file didn't exist. Skipping ugrid_nef()")
            return None
            
        flowgeom=self.flowgeom()
        mesh_name="FlowMesh" # sync with sundwaq for now.

        if flowgeom is not None:
            nc=flowgeom.copy(**nc_kwargs)
        else:
            nc=qnc.empty(**nc_kwargs)
        nc._set_string_mode('fixed') # required for writing to disk

        self.hydro.infer_2d_elements()

        try:
            if mode=='map':
                seg_k = self.hydro.seg_k
                seg_elt = self.hydro.seg_to_2d_element
                n_locations=len(seg_elt)
            elif mode=='history':
                # do we go through the location names, trying to pull out elt_k? no - as needed,
                # use self.monitor_areas.

                # maybe the real question is how will the data be organized in the output?
                # if each history output can be tied to a single segment, that's one thing.
                # Could subset the grid, or keep the full grid and pad the data with fillvalue.
                # but if some/all of the output goes to multiple segments, then what?
                # keep location_name in the output?
                # maybe we skip any notion of ugrid, and instead follow a more CF observed features
                # structure?

                # also, whether from the original Scenario, or by reading in the inp file, we can get
                # the original map between location names and history outputs
                # for the moment, classify everything in the file based on the first segment
                # listed

                # try including the full grid, and explicitly output the mapping between history
                # segments and the 2D+z grid.
                hist_segs=[ma[1][0] for ma in self.monitor_areas]
                seg_k=self.hydro.seg_k[ hist_segs ]
                seg_elt=self.hydro.seg_to_2d_element[hist_segs]

                # Need to handle transects - maybe that's handled entirely separately.
                # even so, the presence of transects will screw up the matching of dimensions
                # below for history output.
                # pdb.set_trace()

                # i.e. current output with transect:
                # len(seg_elt)==135
                # but we ended up creating an anonymous dimension d138
                # (a) could consult scenario to get the count of transects
                # (b) is there anything in the NEFIS file to indicate transect output?
                # (c) could use the names as a hint
                #     depends on the input file, but currently have things like eltNNN_layerNN
                #     vs. freeform for transects

                # what about just getting the shape from the LOCATIONS field?
                shape,dtype = nef['DELWAQ_PARAMS'].getelt('LOCATION_NAMES',shape_only=True)
                n_locations=shape[1]
                if n_locations>len(seg_elt):
                    self.log.info("Looks like there were %d transects, too?"%(n_locations - len(seg_elt)))
                elif n_locations<len(seg_elt):
                    self.log.warning("Weird - fewer output locations than anticipated! %d vs %d"%(n_locations,
                                                                                                  len(seg_elt)))
            else:
                assert False

            n_layers= seg_k.max() + 1

            # elt_map: 'SUBST_001' => 'oxy'
            # real_map: 'saturoxy' => 'SaturOXY'
            elt_map,real_map = map_nef_names(nef)

            # check for unique element names
            name_count=defaultdict(lambda: 0)
            for group in nef.groups():
                for elt_name in group.cell.element_names:
                    name_count[elt_name]+=1

            # check for unique unlimited dimension:
            n_unl=0
            for group in nef.groups():
                # there are often multiple unlimited dimensions.
                # hopefully just 1 unlimited in the RESULTS group
                if 0 in group.shape and group.name=='DELWAQ_RESULTS':
                    n_unl+=1

            # give the user a sense of how many groups are being
            # written out:
            self.log.info("Elements to copy from NEFIS:")
            for group in nef.groups():
                for elt_name in group.cell.element_names:
                    nef_shape,nef_type=group.getelt(elt_name,shape_only=True)
                    vshape=group.shape + nef_shape
                    self.log.info("  %s.%s: %s (%s)"%(group.name,
                                                      elt_name,
                                                      vshape,nef_type))

            for group in nef.groups():
                g_shape=group.shape
                grp_slices=[slice(None)]*len(g_shape)
                grp_dim_names=[None]*len(g_shape)

                # infer that an unlimited dimension in the RESULTS
                # group is time.
                if 0 in g_shape and group.name=='DELWAQ_RESULTS':
                    idx=list(g_shape).index(0)
                    if n_unl==1: # which will be named
                        grp_dim_names[idx]='time'

                for elt_name in group.cell.element_names:
                    # print("elt name is",elt_name)

                    # Choose a variable name for this element
                    if name_count[elt_name]==1:
                        vname=elt_name
                    else:
                        vname=group.name + "_" + elt_name

                    if vname in elt_map:
                        vname=elt_map[vname]
                    else:
                        vname=vname.lower()

                    self.log.info("Writing variable %s"%vname)
                    subst=self.lookup_item(vname) # may be None!
                    if subst is None:
                        self.log.info("No metadata from process library on %s"%repr(vname))

                    # START time-iteration HERE
                    # for large outputs, need to step through time
                    # assume that only groups with 'time' as a dimension
                    # (as detected above) need to be handled iteratively.
                    # 'time' assumed to be part of group shape.
                    # safe to always iterate on time.

                    # nef_shape is the shape of the element subject to grp_slices,
                    # as understood by nefis, before squeezing or projecting to [cell,layer]
                    nef_shape,value_type=group.getelt(elt_name,shape_only=True)
                    self.log.debug("nef_shape: %s"%nef_shape )
                    self.log.debug("value_type: %s"%value_type )

                    if value_type.startswith('f'):
                        fill_value=np.nan
                    elif value_type.startswith('S'):
                        fill_value=None
                    else:
                        fill_value=-999

                    nef_to_squeeze=[slice(None)]*len(nef_shape)
                    if 1: # squeeze unit element dimensions
                        # iterate over just the element portion of the shape
                        squeeze_shape=list( nef_shape[:len(g_shape)] )
                        for idx in range(len(g_shape),len(nef_shape)):
                            if nef_shape[idx]==1:
                                nef_to_squeeze[idx]=0
                            else:
                                squeeze_shape.append(nef_shape[idx])
                    else: # no squeeze
                        squeeze_shape=list(nef_shape)

                    self.log.debug("squeeze_shape: %s"%squeeze_shape)
                    self.log.debug("nef_to_squeeze: %s"%nef_to_squeeze)

                    # mimics qnc naming - will come back to expand 3D fields
                    # and names
                    dim_names=[qnc.anon_dim_name(size=l) for l in squeeze_shape]
                    for idx,name in enumerate(grp_dim_names): # okay since squeeze only does elt dims
                        if name:
                            dim_names[idx]=name

                    # special handling for results, which need to be mapped 
                    # back out to 3D
                    proj_shape=list(squeeze_shape)
                    if group.name=='DELWAQ_RESULTS' and self.hydro.n_seg in squeeze_shape:
                        seg_idx = proj_shape.index(self.hydro.n_seg)
                        proj_shape[seg_idx:seg_idx+1]=[self.hydro.n_2d_elements,
                                                       n_layers]
                        # the naming of the layers dimension matches assumptions in ugrid.py
                        # not sure how this is supposed to be specified
                        dim_names[seg_idx:seg_idx+1]=["nFlowElem","nFlowMesh_layers"]

                        # new_value=np.zeros( new_shape, value_type )
                        # new_value[...]=fill_value

                        # this is a little tricky, but seems to work.
                        # map segments to (elt,layer), and all other dimensions
                        # get slice(None).
                        # vmap assumes no group slicing, and is to be applied
                        # to the projected array (projection does not involve
                        # any slices on the nefis src side)
                        vmap=[slice(None) for _ in proj_shape]
                        vmap[seg_idx]=seg_elt
                        vmap[seg_idx+1]=seg_k
                        # new_value[vmap] = value
                    elif group.name=='DELWAQ_RESULTS' and n_locations in squeeze_shape:
                        # above and below: n_locations used to be len(seg_elt)
                        # but it's still writing out things like location_names with
                        # an anonymous dimension
                        seg_idx = proj_shape.index(n_locations)
                        # note that nSegment is a bit of a misnomer, might have some transects
                        # in there, too.
                        dim_names[seg_idx]="nSegment"
                    else:
                        vmap=None # no projection

                    for dname,dlen in zip(dim_names,proj_shape):
                        if dname=='time':
                            # if time is not specified as unlimited, it gets
                            # included as the fastest-varying dimension, which
                            # makes writes super slow.
                            nc.add_dimension(dname,0)
                        else:
                            nc.add_dimension(dname,dlen)

                    # most of the time goes into writing.
                    # typically people optimize chunksize, but HDF5 is
                    # throwing an error when time chunk>1, so it's
                    # hard to imagine any improvement over the defaults.
                    ncvar=nc.createVariable(vname,np.dtype(value_type),dim_names,
                                            fill_value=fill_value,
                                            complevel=2,
                                            zlib=True)

                    if vmap is not None:
                        nc.variables[vname].mesh=mesh_name
                        # these are specifically the 2D horizontal metadata
                        if 'quickplot_compat' in output_settings:
                            # as of Delft3D_4.01.01.rc.03, quickplot only halfway understands
                            # ugrid, and actually does better when location is not specified.
                            self.log.info('Dropping location for quickplot compatibility')
                        else:
                            nc.variables[vname].location='face' # 
                        nc.variables[vname].coordinates="FlowElem_xcc FlowElem_ycc"

                    if subst is not None:
                        if hasattr(subst,'unit'):
                            # in the process table units are parenthesized
                            units=subst.unit.replace('(','').replace(')','')
                            # no guarantee of CF compliance here...
                            nc.variables[vname].units=units
                        if hasattr(subst,'item_nm'):
                            nc.variables[vname].long_name=subst.item_nm
                        if hasattr(subst,'aggrega'):
                            nc.variables[vname].aggregation=subst.aggrega
                        if hasattr(subst,'groupid'):
                            nc.variables[vname].group_id=subst.groupid

                    if 'time' in dim_names:
                        # only know how to deal with time as the first index
                        assert dim_names[0]=='time'
                        self.log.info("Will iterate over %d time steps"%proj_shape[0])

                        total_tic=t_last=time.time()
                        read_sum=0
                        write_sum=0
                        for ti in range(proj_shape[0]):
                            read_sum -= time.time()
                            value_slice=group.getelt(elt_name,[ti])
                            read_sum += time.time()

                            if vmap is not None:
                                proj_slice=np.zeros(proj_shape[1:],value_type)
                                proj_slice[...]=fill_value
                                proj_slice[tuple(vmap[1:])]=value_slice
                            else:
                                proj_slice=value_slice
                            write_sum -= time.time()
                            ncvar[ti,...] = proj_slice
                            write_sum += time.time()

                            if (time.time() - t_last > 2) or (ti+1==proj_shape[0]):
                                t_last=time.time()
                                self.log.info('  time step %d / %d'%(ti,proj_shape[0]))
                                self.log.info('  time for group so far: %fs'%(t_last-total_tic))
                                self.log.info('  reading so far: %fs'%(read_sum))
                                self.log.info('  writing so far: %fs'%(write_sum))

                    else:
                        value=group.getelt(elt_name)
                        if vmap is not None:
                            proj_value=value[tuple(vmap)]
                        else:
                            proj_value=value
                        # used to have extraneous[?] names.append(Ellipsis)
                        ncvar[:]=proj_value

                    setattr(ncvar,'group_name',group.name)
            ####

            for vname,rname in iteritems(real_map):
                nc.variables[vname].original_name=rname
            # the nefis file does not contain enough information to get
            # time back to a real calendar, so rely on the Scenario's
            # version of time0
            if 'time' in nc.variables:
                nc.time.units='seconds since %s'%self.time0.strftime('%Y-%m-%d %H:%M:%S')
                nc.time.standard_name='time'
                nc.time.long_name='time relative to model time0'

        finally:
            if close_nef:
                nef.close()

        # cobble together surface h, depth info.
        if 'time' in nc.variables:
            t=nc.time[:]
        else:
            t=None # not going to work very well...or at all

        z_bed=nc.FlowElem_bl[:]
        # can't be sure of what is included in the output, so have to try some different
        # options

        if 1:
            if mode=='map':
                etavar=nc.createVariable('eta',np.float32,['time','nFlowElem'],
                                         zlib=True)
                etavar.standard_name='sea_surface_height_above_geoid'
                etavar.mesh=mesh_name

                for ti in range(len(nc.dimensions['time'])):
                    # due to a possible DWAQ bug, we have to be very careful here
                    # depths in dry upper layers are left at their last-wet value,
                    # and count towards totaldepth and localdepth.  That's fixed in
                    # DWAQ now.
                    if 'totaldepth' in nc.variables:
                        depth=nc.variables['totaldepth'][ti,:,0]
                    elif 'depth' in nc.variables:
                        depth=np.nansum(nc.variables['depth'][ti,:,:],axis=2)
                    else:
                        # no freesurface info.
                        depth=-z_bed[None,:] 

                    z_surf=z_bed + depth
                    etavar[ti,:]=z_surf
            elif mode=='history':
                # tread carefully in case there is nothing useful in the history file.
                if ('totaldepth' in nc.variables) and ('nSegment' in nc.dimensions):
                    etavar=nc.createVariable('eta',np.float32,['time',"nSegment"],
                                             zlib=True)
                    etavar.standard_name='sea_surface_height_above_geoid'

                    # some duplication when we have multiple layers of the
                    # same watercolumn
                    # can only create eta for history output, nan for transects
                    pad=np.nan*np.ones(n_locations-len(seg_elt),'f4')
                    for ti in range(len(nc.dimensions['time'])):
                        depth=nc.variables['totaldepth'][ti,:]
                        z_surf=z_bed[seg_elt]
                        z_surf=np.concatenate( (z_surf,pad) )+depth
                        etavar[ti,:]=z_surf
                else:
                    self.log.info('Insufficient info in history file to create eta')

        if 1: # extra mapping info for history files
            pad=-1*np.ones( n_locations-len(seg_elt),'i4')
            if mode=='history':
                nc['element']['nSegment']=np.concatenate( (seg_elt,pad) )
                nc['layer']['nSegment']=np.concatenate( (seg_k,pad) )
                if flowgeom:
                    xcc=flowgeom.FlowElem_xcc[:]
                    ycc=flowgeom.FlowElem_ycc[:]
                    nc['element_x']['nSegment']=np.concatenate( (xcc[seg_elt],np.nan*pad) )
                    nc['element_y']['nSegment']=np.concatenate( (ycc[seg_elt],np.nan*pad) )

        # extra work to make quickplot happy
        if (mode=='map') and ('quickplot_compat' in output_settings):
            # add in some attributes and fields which might make quickplot happier
            # Add in node depths
            
            # g=unstructured_grid.UnstructuredGrid.from_ugrid(nc)
            x_nc=self.flowgeom_ds()
            g=unstructured_grid.UnstructuredGrid.from_ugrid(x_nc)

            # dicey - assumes particular names for the fields:
            if 'FlowElem_zcc' in nc and 'Node_z' not in nc:
                self.log.info('Adding a node-centered depth via interpolation')
                nc['Node_z']['nNode']=g.interp_cell_to_node(nc.FlowElem_zcc[:])
                nc.Node_z.units='m'
                nc.Node_z.positive='up',
                nc.Node_z.standard_name='sea_floor_depth',
                nc.Node_z.long_name="Bottom level at net nodes (flow element's corners)"
                nc.Node_z.coordinates=nc[mesh_name].node_coordinates
                
            # need spatial attrs for node coords
            node_x,node_y = nc[mesh_name].node_coordinates.split()
            nc[node_x].units='m'
            nc[node_x].long_name='x-coordinate of net nodes'
            nc[node_x].standard_name='projection_x_coordinate'
            
            nc[node_y].units='m',
            nc[node_y].long_name='y-coordinate of net nodes'
            nc[node_y].standard_name='projection_y_coordinate'

            for k,v in six.iteritems(nc.variables):
                if 'location' in v.ncattrs():
                    self.log.info("Stripping location attribute from %s for quickplot compatibility"%k)
                    v.delncattr('location')

                # some of the grids being copied through are missing this, even though waq_scenario
                # is supposed to write it out.
                if 'standard_name' in v.ncattrs() and v.standard_name=='ocean_sigma_coordinate':
                    v.formula_terms="sigma: nFlowMesh_layers eta: eta depth: FlowElem_bl"
                    
        return nc

    _flowgeom_ds=None
    def flowgeom_ds(self):
        """ Returns a netcdf dataset with the grid geometry, or None
        if the data is not around.  Returns as an xarray Dataset.
        """
        assert self.hydro,"Must set hydro before requesting flowgeom_ds"
        if self._flowgeom_ds is None:
            fn=self.hydro.flowgeom_filename
            if os.path.exists(fn):
                self._flowgeom_ds=xr.open_dataset(fn)
        
        return self._flowgeom_ds
    
    #-- Command line access
    def cmd_write_runid(self):
        """
        Label run name in the directory, needed for some delwaq2 (confirm?)

        Maybe this can be supplied on the command line, too?
        """
        self.ensure_base_path()

        if self.use_bloom:
            fn=os.path.join(self.base_path,'runid.eco')
        else:
            fn=os.path.join(self.base_path,'runid.waq')

        if os.path.exists(fn) and not self.overwrite:
            raise Exception("%s exists, but overwrite is False"%fn)
        
        with open(fn,'wt') as fp:
            fp.write("%s\n"%self.name)
            fp.write("y\n")
    def cmd_write_inp(self):
        """
        Write inp file and supporting files (runid, bloominp.d09) for
        delwaq1/2
        """
        self.ensure_base_path()

        self.log.debug("Writing inp file")
        self.write_inp()

        self.cmd_write_bloominp()
        self.cmd_write_runid()

    def cmd_write_hydro(self):
        """
        Create hydrodynamics data ready for input to delwaq1
        """

        self.ensure_base_path()

        self.log.info("Writing hydro data")
        self.write_hydro()

    def cmd_default(self):
        """
        Prepare all inputs for delwaq1 (hydro, runid, inp files)
        """
        self.cmd_write_hydro()
        self.cmd_write_inp()

    def cmd_write_nc(self):
        """ Transcribe binary or NEFIS to NetCDF for a completed DWAQ run 
        """
        self.cmd_write_his_nc()
        self.cmd_write_map_nc()
        
    def cmd_write_his_nc(self):
        self.write_binary_his_nc() or self.write_nefis_his_nc()
        
    def cmd_write_map_nc(self):
        # binary is faster and doesn't require dwaq libraries, but
        # does not know about units.
        self.write_binary_map_nc() or self.write_nefis_map_nc()

    def write_binary_his_nc(self):
        """ If binary history output is present, write that out to netcdf, otherwise
        return False
        """
        his_fn=os.path.join(self.base_path,self.name+".his")
        his_nc_fn=os.path.join(self.base_path,'dwaq_hist.nc')
        
        if not os.path.exists(his_fn): return False
        ds=dio.his_file_xarray(his_fn)
        if os.path.exists(his_nc_fn):
            os.unlink(his_nc_fn)
        ds.to_netcdf(his_nc_fn)
        
    def write_nefis_his_nc(self):
        nc2_fn=os.path.join(self.base_path,'dwaq_hist.nc')
        nc2=self.ugrid_history(nc_kwargs=dict(fn=nc2_fn,overwrite=True))
        # if no history output or nefis is not setup, no nc2.
        if nc2:
            nc2.close()
            return True
        else:
            return False
        
    def write_nefis_map_nc(self):
        nc_fn=os.path.join(self.base_path,'dwaq_map.nc')
        nc=self.ugrid_map(nc_kwargs=dict(fn=nc_fn,overwrite=True))
        # if no map output, no nc
        if nc:
            nc.close()
            return True
        else:
            return False
        
    def write_binary_map_nc(self,output_fn=None,overwrite=None):
        """
        Transcribe binary formatted map output from completed dwaq
        run to a ugrid-esque netcdf file.  Currently assumes sigma
        coordinates!
        """
        if 'binary' not in self.map_formats:
            return False

        if overwrite is None: overwrite=self.overwrite

        if output_fn is None:
            output_fn=os.path.join(self.base_path,"dwaq_map.nc")
            
        if os.path.exists(output_fn) and not overwrite:
            raise Exception("While writing binary map nc to %s: file exists, overwrite is False"%out_fn)
        

        from . import io as dio
        map_fn=os.path.join(self.base_path,self.name+".map")
        map_ds=dio.read_map(map_fn,self.hydro)

        dio.map_add_z_coordinate(map_ds,total_depth='TotalDepth',coord_type='sigma',
                                 layer_dim='layer')

        if os.path.exists(output_fn):
            if overwrite:
                os.unlink(output_fn)
            else:
                raise Exception("While writing binary map nc to %s: file exists, overwrite is False"%out_fn)
        map_ds.to_netcdf(output_fn)
        return True

    use_bloom=False
    def cmd_delwaq1(self):
        """
        Run delwaq1 preprocessor
        """
        if self.use_bloom:
            bloom_part="-eco {}".format(self.bloom_path)
        else:
            bloom_part=""

        cmd=[self.delwaq1_path,
             "-waq", bloom_part,
             "-p",self.waq_proc_def]
        self.log.info("Running delwaq1:")
        self.log.info("  "+ " ".join(cmd))

        t_start=time.time()
        try:
            ret=subprocess.check_output(cmd,shell=False,cwd=self.base_path,stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            self.log.error("problem running delwaq1")
            self.log.error("output: ")
            self.log.error("-----")
            self.log.error(exc.output)
            self.log.error("-----")
            raise WaqException("delwaq1 exited early.  check lst and lsp files")
        self.log.info("delwaq1 ran in %.2fs"%(time.time() - t_start))

        nerrors=nwarnings=-1
        # dwaq likes to draw boxes with code page 437
        for line in ret.decode('cp437','ignore').split("\n"):
            if 'Number of WARNINGS' in line:
                nwarnings=int(line.split()[-1])
            elif 'Number of ERRORS during input' in line:
                nerrors=int(line.split()[-1])
        if nerrors > 0 or nwarnings>0:
            ret=ret.decode() # sometimes this comes back as bytes, and prints better if decoded
            print( ret )
            raise WaqException("delwaq1 found %d errors and %d warnings"%(nerrors,nwarnings))
        elif nerrors < 0 or nwarnings<0:
            print( ret)
            raise WaqException("Failed to find error/warning count")

    def cmd_delwaq2(self,output_filename=None):
        """
        Run delwaq2 (computation)
        """
        cmd=[self.delwaq2_path,self.name]
        if not output_filename:
            output_filename= os.path.join(self.base_path,'delwaq2.out')

        t_start=time.time()
        with open(output_filename,'wt') as fp_out:
            self.log.info("Running delwaq2 - might take a while...")
            self.log.info("  " + " ".join(cmd))
            
            sim_time=(self.stop_time-self.start_time) / np.timedelta64(1,'s')
            tail=MonTail(os.path.join(self.base_path,self.name+".mon"),
                         log=self.log,sim_time_seconds=sim_time)
            try:
                try:
                    ret=subprocess.check_call(cmd,shell=False,cwd=self.base_path,stdout=fp_out,
                                              stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as exc:
                    raise WaqException("delwaq2 exited with an error code - check %s"%output_filename)
            finally:
                tail.stop()

        self.log.info("delwaq2 ran in %.2fs"%(time.time()-t_start))

        # return value is not meaningful - have to scrape the output
        with open(output_filename,'rt') as fp:
            for line in fp:
                if 'Stopping the program' in line:
                    raise WaqException("Delwaq2 stopped early - check %s"%output_filename)
        self.log.info("Done")
    @property
    def delwaq1_path(self):
        return os.path.join(self.delft_bin,'delwaq1')
    @property
    def delwaq2_path(self):
        return os.path.join(self.delft_bin,'delwaq2')

    # Boundary conditions
    def find_discharge_bc(self,name):
        """
        Find a BC by name, if needed create the discharge entry, 
        and return the text label for the discharge
        """
        seg = self.bc_name_to_segment(name,layer='bed')

        # Just a reminder that at least some versions of the DWAQ setup handled multiple
        # sites dumping into the same segment. If that is still needed, it needs to be
        # handled either here in add_load().
        
        #source_segs={} # name => discharge id
        #potw_multi = []

        # the same segment can receive multiple loads, so stick to seg-<id>
        # for naming here, as opposed to naming discharge points after a
        # specific source.
        source_seg="seg-%d"%(seg+1)
    
        self.add_discharge(seg_id=seg,load_id=source_seg,on_exists='ignore')
        return source_seg

    def bc_name_to_segment(self,name,layer='bed'):
        # This stuff should be cached if it's not already:
        boundaries=self.hydro.boundary_defs()
        allitems = [boundary.decode("utf-8") for boundary in set(boundaries['type'])]
        group_boundary_links = self.hydro.group_boundary_links() # read boundary location and name information from DFM .bnd file
          
        g=self.hydro.grid()
        self.hydro.infer_2d_elements()

        bdn = np.nonzero(group_boundary_links['name']==k) #getting index for boundary
        assert(len(bdn)==1)                
        bdn = np.asscalar(np.asarray(bdn)) #bdn is in an annoying tuple type
        line = shapely.wkt.loads(group_boundary_links['attrs'][bdn]['geom'])
        xutm = line.xy[0][0]
        yutm = line.xy[1][0]                
        xy=np.array( [xutm, yutm] )
        elt=g.select_cells_nearest(xy)
        if layer=='bed':
            layer=-1
        elif layer=='surface':
            layer=0
        else:
            layer=int(layer)
        seg=np.nonzero( self.hydro.seg_to_2d_element==elt )[0][layer]

        
class HydroOnlineProxy(Hydro):
    """
    Place to put methods that make online coupled hydro look like regular
    hydro.
    """
    def __init__(self, scenario, dflow_model):
        self.scenario = scenario
        self.dflow_model = dflow_model
        
    def extrude_element_to_segment(self,V):
        """ V: [n_2d_elements] array
        returns [n_seg] array
        """
        layers = max(1,self.dflow_model.n_layers) # in case it's 2D == 0 layers
        return np.tile(V,layers)

    def grid(self):
        return self.dflow_model.grid
    
class WaqOnlineModel(WaqModelBase):
    """
    code for D-WAQ setup specific to running online, under a
    D-Flow hydro model
    """
    # design is evolving, but probably bcs for an online model
    # should live in the hydro object.
    bcs="DO NOT USE"

    # Segment function (spatiotemporal parameters) have to be put onto a curvilinear
    # grid for online coupling. Currently this is a cartesian grid with constant
    # resolution, specified here
    seg_function_resolution=500.0
    
    def __init__(self,model,**k):
        self.model=model
        self.hydro = HydroOnlineProxy(scenario=self,dflow_model=model)
        super(WaqOnlineModel,self).__init__(**k)
        
        self.sub_file = 'sources.sub'
        self.sub_path = os.path.join(self.model.run_dir, self.sub_file)

        self.coupled_parameters=self.init_coupled_parameters()
        

    #time_step=None  # these are taken from hydro, unless specified otherwise
    
    @property
    def start_time(self):
        return utils.to_dt64(self.model.run_start)

    @property
    def stop_time(self):
        return utils.to_dt64(self.model.run_stop)

    @property
    def time0(self):
        return utils.to_dt64(self.model.ref_date)

    @property
    def waq_proc_def(self):
        if self.model.waq_proc_def:
            # DFlowModel has been assuming that we provide
            # the full proc_def.def, but waq_scenario code
            # assume no extension.
            proc_def=self.model.waq_proc_def
            if proc_def.endswith('.def'):
                proc_def=proc_def[:-4]
            return proc_def
        else:
            return os.path.join(self.share_path,'proc_def')
        
    def init_coupled_parameters(self):
        """
        For online coupling there are some minor gymnastics needed to
        get DFM parameters into DWAQ. See the section of the dia file starting
          ** INFO   : Data from hydrodynamics available for water quality
        to see which parameters are linked.

        This method sets up self.coupled_parameters, which only serves to
        add entries in the substances file. That triggers DFM to supply
        those values.
        """
        params=NamedObjects(scenario=self,cast_value=cast_to_parameter)
        # All of the current known options:
        # params['Tau']=1
        # params['TauFlow']=1
        # params['Velocity']=1
        if self.model.mdu.get_bool('physics','Salinity'):
            params['salinity']=1 
        if self.model.mdu.get_bool('physics','Temperature'):
            params['temp']=1 
        params['vwind']=1
        #params['winddir']=1
        #params['rain']=1
        return params
    
    def write_waq(self):
        """
        Writes .sub file for Delwaq model, and adds Delwaq configuration details to Dflow .mdu file.
        Reorders self.substances
        Initial conditions, boundary conditions are not [yet] handled here.

        Non-constant parameters are [WIP] handled here by appending to external forcing file. 
        handled here.

        This is called during DFM.update_config(), since it modifies the MDU.
        
        See online_dwaq.py in the proof-of-concept manual setup for a specialization
        of write_waq() that appends dwaq tracer info to an existing external forcing file.
        """
        self.log.info('Updating mdu with Delwaq settings...')
        # add reference to sub-file in .mdu
        self.model.mdu['processes', 'SubstanceFile'] = self.sub_file
        self.model.mdu['processes', 'DtProcesses'] = 300  # TODO hard-coded to match DtUser in template .mdu
        self.model.mdu['processes', 'ProcessFluxIntegration'] = 1  # 1 = Delwaq, 2 = Dflow

        # this should be handled by the caller.
        # self.model.set_run_dir(self.model.run_dir, mode='create')

        self.fix_substance_order()

        self.log.info('Writing Delwaq model files...')
        with open(self.sub_path, 'wt') as f:
            for s in self.substances:
                self.write_substance(f, self.substances[s])

            for name in self.parameters:
                # The offline model enables processes by setting parameters
                # with names ACTIVE_<process>
                # here we change those to processes
                if name.upper().startswith("ACTIVE_"):
                    if name.upper()!="ACTIVE_ONLY":
                        self.log.info("Translating parameter %s to process"%name)
                        self.add_process(name.upper().replace('ACTIVE_',''))
                    continue
                # Only constant parameters go in the substances file
                # Note that this also helps with coupled parameters. If a parameter
                # like temp is given a non-constant value it will not show
                # up here (thus not triggering the use of a DFM value), but will just
                # show up in the external forcing file.
                if isinstance(self.parameters[name],ParameterConstant):
                    self.write_param(f, self.parameters[name])

            for name in self.coupled_parameters:
                # These should only be constant valued, and are here just to request
                # that DFM fill in the values during the run.
                self.write_param(f, self.coupled_parameters[name])
                
            self.write_processes(f)

            # Additional output can be specified in the substance file but affects both map and history files.
            # Separate outputs can be specified but it requires an extra file and adding an entry to
            # the MDU.
            extra_outputs = np.unique( self.hist_output + self.map_output )
            for out_var in extra_outputs:
                if out_var==DEFAULT or out_var in self.substances: continue
                f.write("output '%s'\n"%out_var)
                # Could fetch description in NEFIS or csvs
                f.write("  description '%s extra output'\n"%out_var)
                f.write("end-output\n")
            
        for attr in ['mon_output','grid_output','stat_output',
                     'map_formats','history_formats']:
            if getattr(self,attr) != getattr(WaqModelBase,attr):
                self.log.info("WaqOnlineModel: '%s' has been modified, but is not [yet] handled for online coupling"
                              %attr)

    def write_waq_forcing(self):
        """
        Make edits to external forcing file for online waq run. Entries that
        were added to the model config as BC objects should be added in write_waq(),
        and will get written out by the DFM driver as part of writing other BCs.
        This is for 'out-of-band' edits to external forcing file.
        """
        # WAQ parameters that go in the external forcing file:
        with open(self.model.mdu.filepath(('external forcing','ExtForceFile')),'at') as fp:
            fp.write("\n\n")
            for name in self.parameters:
                if isinstance(self.parameters[name],ParameterSpatial):
                    self.write_parameter_spatial(fp,self.parameters[name])
                elif isinstance(self.parameters[name],ParameterTemporal):
                    self.write_parameter_temporal(fp,self.parameters[name])
                elif isinstance(self.parameters[name],ParameterSpatioTemporal):
                    self.write_parameter_spatiotemporal(fp,self.parameters[name])

    
    def write_waq_spatial(self,fp,quantity,data_fn,xyn):
        """
        Write an xyz dataset and add as spatial data source in fp (presumably
        external forcing file).
        data_fn is expected to be relative to the run directory (not the script
        directory).
        """
        np.savetxt(os.path.join(self.model.run_dir,data_fn),
                   xyn,fmt="%.6g")
        fp.write("\n".join(["QUANTITY=%s"%quantity,
                            "FILENAME=%s"%data_fn,
                            "FILETYPE=7",
                            "METHOD=4",
                            "OPERAND=O\n"]))
        
    def write_parameter_spatial(self, fp, param):
        xy=self.model.grid.cells_centroid()
        name=param.name
        assert name!="unnamed"
        
        if param.per_segment is not None:
            pdb.set_trace()
            # Hopefully seg_values is 2D, with no vertical variation.
            conc = something
        else:
            pdb.set_trace()
            
        xyn=np.c_[xy, conc]

        self.write_waq_spatial(fp,
                               quantity="waqparameter"+name,
                               data_fn="PARAM-%s.xyn"%name,
                               xyn=xyn)

    def write_parameter_temporal(self,fp, param):
        print("Would be writing temporal parameter to external forcing file")
        # e.g. RadSurf
        name=param.name
        tim_fn="%s.tim"%name
        tim_path = os.path.join(self.model.run_dir,tim_fn)

        # expecting xr.DataArray
        ds=xr.Dataset()
        times=np.asarray(param.times)
        if not np.issubdtype(times.dtype,np.datetime64):
            print("Whoa hoss. Expected parameter times to be np.datetime64")
            pdb.set_trace()
        ds['time']=('time',),times
        ds[name]=('time',),param.values
        self.model.write_tim(ds[name],tim_path)
        
        fp.write( "\n".join( [
            "QUANTITY=waqfunction%s"%name,
            "FILENAME=%s"%tim_fn,
            "FILETYPE=1",
            "METHOD=1", # linear interpolation. 0 for block
            "OPERAND=O\n"
            ]))
            
    def write_parameter_spatiotemporal(self, fp, param):
        """
        Convert a segment function to 2D raster netcdf and add to external forcing file.
        This is not exact! Incoming data is on the grid, but we have to write a curvilinear
        grid
        """
        # assert param.values is not None,"Expected seg function data in param.values"
        
        seg_fn="seg-%s.nc"%param.name

        name=param.name

        if 1: # write the cartesian netcdf 
            # param.values:  2 x 10 x 49996 => time x layer x cell

            # Could be done ahead of time and cached.
            cell_to_pixels=self.model.grid.interp_cell_to_raster_function(dx=self.seg_function_resolution,
                                                                          dy=self.seg_function_resolution)
            template_fld = cell_to_pixels(None)

            segfunc_ds=xr.Dataset()

            times=np.asarray(param._times)
            if isinstance(times[0],str): # some files comes in with ASCII datetimes
                times=times.astype(np.datetime64)
            if not np.issubdtype(times.dtype,np.datetime64):
                # may get inputs that are in 'dwaq time' (e.g. seconds since reference time)
                print("Whoa hoss. Expected segment function times to be np.datetime64")
                pdb.set_trace()
            segfunc_ds['time']=('time',),times

            # It's possible that DFM would understand a proper SGRID-compliant file. But I know that
            # it understands this, which is simpler.
            segfunc_ds.attrs.update(dict(grid_type='IRREGULAR',
                                         coordinate_system=self.model.projection or 'EPSG:26910',
                                         Conventions='CF-1.0'))
            X,Y = template_fld.XY()
            segfunc_ds['y'] =('M','N'), Y
            segfunc_ds['x'] =('M','N'), X
            segfunc_ds['y'].attrs.update(dict(long_name='northing',standard_name='projection_y_coordinate',
                                              units='m'))
            segfunc_ds['x'].attrs.update(dict(long_name='easting',standard_name='projection_x_coordinate',
                                              units='m'))

            data = np.zeros( (segfunc_ds.dims['time'],segfunc_ds.dims['M'], segfunc_ds.dims['N']),
                             np.float32)
            n_2d_elts=self.model.grid.Ncells()
            for tidx,t in utils.progress(enumerate(times)):
                if param.values is not None:
                    data_on_grid=param.values[tidx,:]
                elif param.func_t is not None:
                    data_on_grid=param.func_t(t)
                    
                # param.values is theoretically coming in as n_seg, but can also come in
                # as n_2d_elements. Either way, make it [layers,elements], and layers might be
                # 1-long or n_layers long
                data_on_grid = data_on_grid.reshape([-1,n_2d_elts])
                    
                col_range=data_on_grid.max(axis=0) - data_on_grid.min(axis=0)
                assert np.all(col_range<1e-12),"Segment function has 3D data, but online coupling can only represent 2D"
                data_2d=data_on_grid[0,:] # or average if we tolerate some variation.
                data[tidx,:,:] = cell_to_pixels(data_2d).F

            segfunc_ds[name]=('time','M','N'), data
            segfunc_ds[name].attrs.update(dict(long_name=name,
                                               grid_mapping='projected_coordinate_system'))
            seg_path=os.path.join(self.model.run_dir,seg_fn)
            segfunc_ds.to_netcdf(seg_path)
            self.log.info("Wrote %s to %s"%(param.name, seg_path))

        fp.write("\n".join([
            "QUANTITY=waqsegmentfunction%s"%name,
            "FILENAME=%s"%seg_fn,
            "VARNAME=%s"%name,
            "FILETYPE=11",
            "METHOD=3",
            "OPERAND=O\n"]) )

                
    def fix_substance_order(self):
        # A bit tricky: we need to know the order of WAQ substances before source/sink
        # BCs can be written, and that order is some unknown function of the order
        # of substances in the substance file, initial conditions, and other BCs.
        # So here we force the order of all these things to be the same
        # Note that this gets sub names in lower case
        subs=[sub for sub in self.substances]
        
        # Modify model.bcs to put these last
        # This is annoying... No guarantee that there *are* BCs for all scalars. So
        # if there is only a BC for one sub, then that sub may get placed first even
        # if canon_order has it later.
        waq_bcs=   [bc for bc in self.model.bcs if getattr(bc,'scalar','_dummy_').lower() in subs]
        nonwaq_bcs=[bc for bc in self.model.bcs if getattr(bc,'scalar','_dummy_').lower() not in subs]
        bc_subs=[bc.scalar.lower() for bc in waq_bcs]

        subs.sort()  # Sort the substances alphabetically
        # sub order is then bc subs first, alphabetically, then non-bc subs, then inactive subs,
        # each in alphabetic order.  only active subs will have bcs, so no conflict there.
        sub_ordered=[s for s in subs if s in bc_subs]
        sub_ordered+=[s for s in subs if (s not in bc_subs) and self.substances[s].active]
        sub_ordered+=[s for s in subs if (s not in bc_subs) and not self.substances[s].active]

        for s in sub_ordered:
            self.substances.move_to_end(s)
        
        # Now reorder the bcs to follow:
        # Reorder the waq_bcs to follow alphabetic order
        waq_bc_order=np.argsort([sub_ordered.index(bc.scalar.lower()) for bc in waq_bcs])
        waq_bcs=[waq_bcs[i] for i in waq_bc_order]
        
        new_bcs=nonwaq_bcs + waq_bcs
        assert len(new_bcs)==len(self.model.bcs),"Sanity lost"
        self.model.bcs=new_bcs

    def write_substance(self, f, substance):
        """Writes to opened .sub file f for a particular substance"""
        item=self.process_db.substance_by_id(substance.name)
        if item is None:
            description="N/A"
            conc_unit="N/A"
        else:
            description=item.item_nm
            conc_unit=item.unit

        waste_unit="N/A"
        
        s = f"substance '{substance.name}' {'in' * (not substance.active)}active\n" \
            f"  description        '{description}'\n" \
            f"  concentration-unit '{conc_unit}'\n" \
            f"  waste-load-unit    '{waste_unit}'\n" \
            f"end-substance\n"
        f.writelines(s)
        return 0

    def write_param(self, f, param):
        """Writes to opened .sub file f for a particular parameter"""
        if param.name.lower()=='only_active':
            self.log.info("Ignoring only_active for online WAQ configuration")
            return
        item=self.process_db.substance_by_id(param.name)

        if item is None:
            item_nm=param.name
            unit="n/a"
        else:
            item_nm=item.item_nm
            unit=item.unit
        # Need to query database to find the description and unit
        s = f"parameter '{param.name}'\n" \
            f"  description '{item_nm}'\n" \
            f"  unit '{unit}'\n" \
            f"  value {param.value}\n" \
            f"end-parameter\n"
        f.writelines(s)
        return 0

    def write_processes(self, f):
        """Writes all active processes to .sub file"""
        s=["active-processes"]
        skip_for_online = ['vertdisp'] 
        for proc_name in self.processes:
            if proc_name.lower() in skip_for_online:
                self.log.info("Ignoring process %s for online WAQ configuration (right?)"%proc_name)
                continue
            proc=self.process_db.process_by_id(proc_name)
            if proc is None:
                desc=proc_name
            else:
                # proc_name from the database is really a descriptive phrase
                desc=proc.proc_name
            s.append( f"\tname '{proc_name}'  '{desc}'" )
        s.append("end-active-processes")
        f.writelines("\n".join(s))
        f.write("\n")
        return 0

    def add_bc(self,*args,**kws):
        self.model.add_ScalarBC(*args,**kws)
    
    def add_bcs(self, bc):
        """Add bcs to DelwaqModel object (currently not used, can just add to DFlowModel)"""
        raise Exception("Thought this was unused - probably call add_bc() instead")
        if isinstance(bc, list):
            [self.add_bcs(b) for b in bc]
        else:
            assert isinstance(bc, DelwaqScalarBC), f"BC type {type(bc)} cannot be handled by Delwaq model."
            self.bcs.append(bc)
        
    def update_command(self,cmd):
        """
        Let the WAQ setup alter and extend command line arguments for
        invoking dflowfm. This is also where a custom dll would be 
        specified
        """
        return cmd+["--processlibrary",self.waq_proc_def+".def"]

    def add_monitor_from_shp(self,shp_fn,naming=None,point_layers=None):
        """
        For each feature in the shapefile, add a monitor area.
        shp_fn: path to shapefile.
        Online output is not configurable in the same ways as offline DWAQ.
        Online the choices are points, which go into the history file, or non-overlapping
        regions which go into mass balance areas.
        
        naming: field name from the shapefile giving the name of the monitor area.
          for compatibilty with offline code, naming can be "elt_layer".
        """
        if naming=='elt_layer':
            self.log.warning("monitoring for online waq doesn't support naming='elt_layer'")
            
        locations=wkb2shp.shp2geom(shp_fn)
        
        for i,rec in enumerate(locations):
            geom=rec['geom']

            # Default name
            # This used to name everything mon%d, but for easier use
            # and compatibility with older code, use geometry type
            name="%s%d"%(geom.type.lower(),len(self.model.mon_points))

            if naming is not None and naming in rec:
                name=rec[naming]
            self.add_monitor_for_geometry(name=name,geom=geom)
                
    def add_monitor_for_geometry(self,name,geom):
        """
        Add a monitor area for elements intersecting the shapely geometry geom.
        """
        # make sure the name is unique
        for rec in self.model.mon_points:
            assert name!=rec.get('name','__unnamed')
            # getting a warning about elementwise comparison...
            if not isinstance(name,str):
                import pdb
                pdb.set_trace()

        self.model.mon_points.append( dict(name=name,geom=geom) )
    
    def add_transects_from_shp(self,shp_fn,naming='count',clip_to_poly=None,
                               on_boundary=None,on_edge=None):
        """
        Add monitor transects from a shapefile.
        By default transects are named in sequence.  
        Specify a shapefile field name in 'naming' to pull user-specified
        names for the transects.

        online model punts details to hydro_model and dflow_model.
        
        on_edge, on_boundary, clip_to_poly are accepted but ignored
        """
        if clip_to_poly is not None:
            self.log.warning("Online-coupled WAQ does not concern itself with clip_to_poly")
        if on_boundary is not None:
            self.log.warning("Online-coupled WAQ does not concern itself with on_boundary")
        if on_edge is not None:
            self.log.warning("Online-coupled WAQ does not concern itself with on_edge")
            
        locations=wkb2shp.shp2geom(shp_fn)

        new_transects=[] # generate as list, then assign as tuple

        for i,rec in enumerate(locations):
            geom=rec['geom']

            if naming=='count':
                name="transect%04d"%i
            else:
                name=rec[naming]
            new_transects.append( dict(name=name,geom=geom) )
            
        self.log.info("Added %d monitored transects from %s"%(len(new_transects),shp_fn))
        self.model.add_monitor_sections(new_transects)

    
def write_delwaqg_parameters(scen):
    """
    scen: a Scenario or WaqModelBase instance
    layers: list or array of layer thicknesses in meters.
    """
    layers = scen.bottom_layers
    fn = os.path.join(scen.base_path,"delwaqg.parameters")

    if scen.delwaqg_initial is not None:
        initial_fn="delwaqg.initials"
        # write a restart file
        dio.create_restart(res_fn=os.path.join(scen.base_path,initial_fn),
                           hyd=scen.hydro,
                           map_net_cdf=True,
                           map_fn=scen.delwaqg_initial,
                           state_vars=delwaqg_substances)
    else:
        initial_fn="none"

    lines=[]
    lines.append("'delwaqg.map'   # map file for output of bed concentration")
    lines.append("'delwaqg.restart' # write restart information to here")
    # 2023-07-24: Waiting to catch up with Pradeep before getting into generating
    # initial conditions file.
    lines.append("'%s'  # separate file for initial conditions"%initial_fn)
    # 
    lines.append("%d  # count of layers"%len(layers))
    for thick in layers:
        # [m], [m2/s], [m2/s]
        # lines.append(" %g %g %g"%(thickness[i],diffusion[i],bioturb[i]))
        # Try specifying only the thickness.
        lines.append(" %g"%thick)

    # 2023-07-24: Likewise, discuss before worrying about zones and ICs in this file.
    # Unclear whether zones will work correctly in online coupling. 
    #lines.append("'zones.dwq'  # file defining zones")

    ## For each relevant substance,
    #for sub in dwaqg_subs:
    #    lines.append("'%s'  %g"%(sub.name, sub.ic.value))
    
    with open(fn,'wt') as fp:
        for l in lines:
            fp.write(l+"\n")


# Expected order of delwaqg substances in restart file.
delwaqg_substances = [
    'CH4-pore', 'DOC-pore', 'DON-pore', 'DOP-pore', 'DOS-pore', 'NH4-pore', 
    'NO3-pore', 'OXY-pore', 'PO4-pore', 'Si-pore', 'SO4-pore', 'SUD-pore', 
    'AAP-bulk', 'APATP-bulk', 'FeIIIpa-bulk', 'Opal-bulk', 'POC1-bulk', 'POC2-bulk', 
    'POC3-bulk', 'POC4-bulk', 'PON1-bulk', 'PON2-bulk', 'PON3-bulk', 'PON4-bulk', 
    'POP1-bulk', 'POP2-bulk', 'POP3-bulk', 'POP4-bulk', 'POS1-bulk', 'POS2-bulk', 
    'POS3-bulk', 'POS4-bulk', 'SUP-bulk', 'VIVP-bulk',
]
            
def make_delwaqg_dataset(scen):
    """
    Half of the process for passing 3D initial conditions to delwaqg.
    Returns a dataset with the grid, layers, and required substances included.
    Caller can edit that as needed and then pass to write_delwaqg_map()
    (or maybe as an extra argument to write_delwaqg_parameters()).
    """
    layers = scen.bottom_layers
    grid = scen.hydro.grid()

    ds=grid.write_xarray() # create_restart expects face and layer dimensions

    ds['layer_thickness']=('layer',), layers
    ds['layer_thickness'].attrs['long_name']="Thickness of sediment layers in DelwaqG"
    ds['layer_thickness'].attrs['units']='m'

    for sub in delwaqg_substances:
        # create_restart expects "native" dwaq order of 'layer','face'
        ds[sub] = ('layer','face'), np.zeros( (ds.dims['layer'], ds.dims['face']), np.float32)
        if '-pore' in sub:
            ds[sub].attrs['units']='g/m3 of porewater'
        else:
            ds[sub].attrs['units']='g/m3 bulk'
            
    # DelwaqG does not use the names.
    # Must get the order correct
    
    # read title
    # read nosysini, nosegini (substance count, segment count)
    # read substances names into synameinit array, but they are never used.
    # read data into sedconc array ~ [layer, substance, segment]
    # dissolved substances (the first 12) are scaled by porosity.
    # re units: the code that reads in WC parameters appears to assume
    #   g/m2 input, which is then divided by total bed thickness to
    #   get g/m3.
    # dissolved substances are assumed to come in as a porewater concentration,
    # and scaling by porisity then gives a bulk concentration.

    # Order:
    # Names don't matter. For simplicity, use the same names as in the map and restart
    # files:

    # create_restart() expects this.
    ds['header']="Initial conditions file for DelwaqG"

    return ds


class InpReader:
    def __init__(self,fn):
        self.fn=fn
        with open(self.fn,'rt') as fp:
            self.fp = fp
            self.tok = self.toker()
            self.read()
            self.fp = None
    def skip_to_section(self,sec):
        while 1:
            line = self.fp.readline()
            if line=="": break
            if line.startswith(f'#{sec}'):
                return True
        return False
    
    def toker(self):
        while 1:
            line=self.fp.readline()
            if line=="": break
            line=line.split(';')[0].strip()
            # break on white space except when inside single or double quotes
            for t in shlex.split(line): 
                #print("Token: ",t)
                yield t

    def next_int(self): return int(next(self.tok))
    def next_str(self): return next(self.tok)
    def next_float(self): return float(next(self.tok))
    
    def read(self):
        self.read_section0()
        assert self.skip_to_section(1)
        self.next_int() ; self.next_str() ; self.next_str()
        self.next_float()
        while 1:
            t=self.next_str()
            if t[0] in "01232456789": break
        self.next_str()
        assert 0==self.next_int() # timestep option
        self.next_str() # time step value
        self.read_mon()
        self.read_transects()

    def read_section0(self):
        header=self.fp.readline() # first line sets comment char, can't be read through toker
        tokens = header.split()
        self.max_input_width  = int(tokens[0])
        self.max_output_width = int(tokens[1])
        self.comment_char = tokens[2] # still has quotes
        # ignore trailing comment
        
        self.desc=[self.next_str(), self.next_str(), self.next_str()]
        self.time_string = self.next_str()
        print("Time string: ",self.time_string)
        # T0: 2013/08/01-00:00:00  (scu=       1s)
        self.time0 = np.datetime64(
            datetime.datetime.strptime(
                self.time_string[4:23]
                .replace('/','-').replace('.','-'),
                "%Y-%m-%d-%H:%M:%S"))
        self.scu_seconds = int(self.time_string[30:38])

        self.n_active = self.next_int()
        self.n_passive = self.next_int()
        self.substances=[]
        for _ in range(self.n_active):
            self.next_int() # 1-based index
            self.substances.append(Substance(name=self.next_str(),active=True))
        for _ in range(self.n_passive):
            self.next_int() # 1-based index
            self.substances.append(Substance(name=self.next_str(),active=False))
        
    def read_mon(self):
        has_mon=self.next_int()
        n_mon = self.next_int()
        self.monitor_areas=[]
        for mon_idx in range(n_mon):
            name=self.next_str()
            count=self.next_int()
            segs=[self.next_int() for _ in range(count)]
            self.monitor_areas.append( (name,segs))
        print(f"{len(self.monitor_areas)} mon areas")
        
    def read_transects(self):
        has_tran=self.next_int()
        if has_tran>0:
            n_tran = self.next_int()
            print("n_tran",n_tran)
        else:
            n_tran=0
        self.monitor_transects=[]
        for _ in range(n_tran):
            name=self.next_str()
            report_net=self.next_int() # we always use 1.
            count=self.next_int()
            exchs=[self.next_int() for _ in range(count)]
            self.monitor_transects.append( (name,exchs))
        print(f"{len(self.monitor_transects)} transects")

    def get_transect_by_name(self,name):
        for tran in self.monitor_transects:
            if name==tran[0]:
                return tran
        return None
    
        
