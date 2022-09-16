import os
import re
import datetime
import numpy as np
import glob
import logging
log=logging.getLogger("ptm_config")

from ... import utils

class PtmConfig(object):
    """
    A work in progress for configuring, running, loading FISH-PTM
    runs.
    """
    run_dir=None    
    end_time=None
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        self.regions=[]
        self.releases=[]
        self.groups=[]

    @staticmethod
    def load(path):
        cfg=PtmConfig()
        cfg.run_dir=path
        cfg.read()
        return cfg

    def read(self):
        """
        INCOMPLETE!
        set parameters as much as possible based on info in the files
        in self.run_dir
        """
        with open(os.path.join(self.run_dir,'FISH_PTM.inp'),'rt') as fp:
            lines=fp.readlines()
            def eat(s):
                if lines[0].strip()==s:
                    del lines[:1]
                else:
                    raise Exception("Expected %s but found %s"%(s,lines[0]))
            def keyvalue(s):
                while lines:
                    # not perfect -- will pick up a comment inside quotes.
                    line=re.sub(r'--.*$','',lines[0]).strip()
                    if not line:
                        del lines[:1]
                        continue
                    k,v=line.split('=',1)
                    del lines[:1]
                    if k.strip()==s:
                        return v.strip()
                    else:
                        raise Exception("Expected %s= but found %s"%(s,lines[0]))
            def keystring(s): # removes matching quotes
                v=keyvalue(s)
                if v[0]==v[-1]:
                    return v[1:-1]
                else:
                    raise Exception("Expected matching quotes, but got %s"%v)
            def keydate(s):
                v=keystring(s)
                return utils.to_dt64(datetime.datetime.strptime(v,'%Y-%m-%d %H:%M:%S'))
            eat('GLOBAL INFORMATION')
            self.end_time=keydate('END_TIME')
            self.restart_dir=keystring('RESTART_DIR')
    def is_complete(self,groups='all',tol=np.timedelta64(0,'h')):
        """
        Return true if it appears that this PTM run has completed.
        NOT VERY ROBUST.
        Assumes that all groups have output (or at least bin index entries) through
        the end of the run.  if the output interval is not even with end_time,
        this will give a false negative.
        Supply a positive timedelta64 for tol to accept output within tolerance of the
        end of the run as complete. Some runs appear to miss the last time step of
        output.

        sets self.last_output to the time of the last output, to allow a rough
        estimate of percent complete.

        groups: 'all' check on all groups. 'first' just check first valid bin.idx file 
          we find.
        """
        self.first_output=None
        self.last_output=None
        n_short=0
        idxs=glob.glob(os.path.join(self.run_dir,'*_bin.idx'))
        if len(idxs)==0:
            return False

        for idx in idxs:
            with open(idx,'rt') as fp:
                lines=fp.readlines()
                dts=[]
                for line in [ lines[0],lines[-1]]:
                    try:
                        year,month,day,hour,minute,offset,count = [int(s) for s in line.split()]
                        dt=datetime.datetime(year=year,month=month,day=day,
                                             hour=hour,minute=minute)
                        dt=utils.to_dt64(dt)
                        dts.append(dt)
                    except ValueError:
                        dts.append(None)
                dt_first,dt_last=dts
                
                if dt_last is None:
                    n_short+=1 # invalid index file
                    continue
                elif (self.last_output is None) or (self.last_output<dt_last):
                    self.last_output=dt_last
                
                if dt_first is None:
                    log.error("Couldn't parse the first output line")
                elif (self.first_output is None) or (self.first_output>dt_first):
                    self.first_output=dt_first
                    
                if dt+tol<self.end_time:
                    log.debug("IDX %s appears short: %s vs %s"%(idx,dt,self.end_time))
                    n_short+=1
            if groups=='first':
                break
        return n_short==0
            
    def bin_files(self):
        fns=glob.glob(os.path.join(self.run_dir,"*_bin.out"))
        fns.sort()
        return fns
        
    def config_text(self):
        self.lines=[]
        self.add_global()
        self.add_transects()
        self.add_regions()
        self.add_release_distribution_sets()
        self.add_release_timing()
        self.add_behaviors()
        self.add_output_sets()
        self.add_groups()
        
        return "\n".join(self.lines)
    
    def add_global(self):
        self.lines +=[
        """\
GLOBAL INFORMATION
   END_TIME = '{0.end_time_str}'
   RESTART_DIR = 'none'
   TIME_STEP_SECONDS = 180.
 
   -- deactivation logicals ---
   REACHED_OPEN_BOUNDARY = 'true'
   REACHED_FLOW_BOUNDARY = 'true'
   ENTRAINED_BY_VOLUME_SINK = 'false'
   CROSSED_LINE = 'false'
   DEPOSITED_ON_BED = 'false'
   CONSOLIDATED_ON_BED = 'false'
 
   -- kill logicals ---
   REACHED_OPEN_BOUNDARY = 'true'
   REACHED_FLOW_BOUNDARY = 'true'
   ENTRAINED_BY_VOLUME_SINK = 'false'
   CROSSED_LINE = 'false'
   DEPOSITED_ON_BED = 'false'
   CONSOLIDATED_ON_BED = 'false'
 
   -- line information --- 
   NLINES = 0
        """.format(self)]
    @property
    def end_time_str(self):
        return utils.to_datetime(self.end_time).strftime("%Y-%m-%d %H:%M:%S")
    @property
    def rel_time_str(self):
        return utils.to_datetime(self.rel_time).strftime("%Y-%m-%d %H:%M:%S")

    def add_transects(self):
        self.lines+=["""\
TRANSECT INFORMATION -- applies to tidal surfing
   NTRANSECTS = 0
"""]

    def add_regions(self):
        self.lines.append("REGION INFORMATION")
        self.lines.append( "   NREGIONS = {nregions}".format(nregions=len(self.regions)) )
        for i,r in enumerate(self.regions):
            self.lines.append("   --- region %d ---"%i)
            self.lines += r
    def add_release_distribution_sets(self):
        self.lines.append("""RELEASE DISTRIBUTION INFORMATION
   NRELEASE_DISTRIBUTION_SETS = {num_releases}
""".format(num_releases=len(self.releases)))
    
        for i,rel in enumerate(self.releases):
            self.lines.append("\n   -- release distribution set %d ---"%i)
            self.lines+=rel

    def add_release_timing(self):
        self.lines+=["""\
RELEASE TIMING INFORMATION
   NRELEASE_TIMING_SETS = 3
   -- release timing set 1 ---        
     RELEASE_TIMING_SET = 'once'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'single'
     INACTIVATION_TIME = 'none'
   -- release timing set 2 ---        
     RELEASE_TIMING_SET = 'flowbased'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'interval'
          NINTERVALS = 1000000
          RELEASE_INTERVAL_HOURS = 1.0
          INACTIVATION_TIME = 'none'
   -- release timing set 3 ---        
     RELEASE_TIMING_SET = 'interval'
     INITIAL_RELEASE_TIME = '{rel_time_str}'
     RELEASE_TIMING = 'interval'
       NINTERVALS = 100000
       RELEASE_INTERVAL_HOURS = 1.0
     INACTIVATION_TIME = 'none'""".format(rel_time_str=self.rel_time_str)
          ]
    def add_behaviors(self):
        self.lines+=["""\
BEHAVIOR INFORMATION
   NBEHAVIOR_PROFILES = 0
   NBEHAVIORS = 2

   -- behavior 1 ---
     BEHAVIOR_SET = 'down5000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'down_5mm_per_s.inp'
  
   -- behavior 2 ---
     BEHAVIOR_SET = 'up5000'
     BEHAVIOR_DIMENSION = 'vertical'
     BEHAVIOR_TYPE = 'specified'
     BEHAVIOR_FILENAME = 'up_5mm_per_s.inp'

"""]


    def add_output_sets(self):
        self.lines+=["""\
OUTPUT INFORMATION 
   NOUTPUT_SETS = 3

   -- output set 1 ---  for debugging
   OUTPUT_SET = '6min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 0.10
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 0.50
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 1.00
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 0.50
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 0.50
   STATE_OUTPUT_INTERVAL_HOURS = 0.10
   NUMBER_OF_VARIABLES_OUTPUT = 6
     VARIABLE_OUTPUT = 'velocity'
     VARIABLE_OUTPUT = 'salinity'
     VARIABLE_OUTPUT = 'layer'
     VARIABLE_OUTPUT = 'water_depth'
     VARIABLE_OUTPUT = 'water_level'
     VARIABLE_OUTPUT = 'bed_elevation'

     NODATA_VALUE = -999.0

   -- output set 2 --- for short time scales
   OUTPUT_SET = '15min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 0.25
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 0.50
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 1.00
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 0.50
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 0.50
   STATE_OUTPUT_INTERVAL_HOURS = 'none'

   -- output set 3 --- close to production
   OUTPUT_SET = '30min_output'
   FLAG_LOG_LOGICAL = 'true'
   BINARY_OUTPUT_INTERVAL_HOURS = 0.5
   ASCII_OUTPUT_INTERVAL_HOURS = 'none' 
   HISTOGRAM_OUTPUT_INTERVAL_HOURS = 'none'
   STATISTICS_OUTPUT_INTERVAL_HOURS = 24
   CONCENTRATION_OUTPUT_INTERVAL_HOURS = 24.0
   REGION_COUNT_OUTPUT_INTERVAL_HOURS = 24.
   REGION_COUNT_UPDATE_INTERVAL_HOURS = 24.
   STATE_OUTPUT_INTERVAL_HOURS = 'none'

"""]


    def add_groups(self):
        self.lines.append("""
PARTICLE GROUP INFORMATION 
   NGROUPS = {num_groups}
""".format(num_groups=len(self.groups)))
        for i,group in enumerate(self.groups):
            self.lines += ["   --- group %d ---"%i]
            self.lines += group

    def clean(self):
        print("Cleaning")
        for patt in ["*.out",
                     "*.log",
                     "*.txt",
                     "fort.*",
                     "*release_log",
                     "*.idx"]:
            for fn in glob.glob(os.path.join(self.run_dir,patt)):
                os.unlink(fn)

    def write(self):
        try:
            os.path.exists(self.run_dir) or os.makedirs(self.run_dir)
        except os.FileExistsError:
            print(f"Weird - {self.run_dir} exists ({os.path.exists(self.run_dir)}), but makedirs failed.")
            raise
        
        self.write_config()
        self.write_method()
    def write_config(self):
        with open(os.path.join(self.run_dir,"FISH_PTM.inp"),'wt') as fp:
            fp.write(self.config_text())

    def method_text(self):
        return """\
 MAX_HORIZONTAL_ADVECTION_SUBSTEPS = 10
 MAX_HORIZONTAL_DIFFUSION_SUBSTEPS = 10
 GRID_TYPE = 'unstructured'
 ADVECTION_METHOD = 'streamline'
   NORMAL_VELOCITY_GRADIENT = 'constant'
 VERT_COORD_TYPE = 'z-level'
 HORIZONTAL_DIFFUSION_METHOD = 'constant'
   CONSTANT_HORIZONTAL_EDDY_DIFFUSIVITY = 0.01
 VERTICAL_ADVECTION_METHOD = 'streamline'
 MIN_VERTICAL_EDDY_DIFFUSIVITY = 0.00001
 MAX_VERTICAL_EDDY_DIFFUSIVITY = 0.10000
 MAX_VERTICAL_DIFFUSION_SUBSTEPS = 100
 MIN_VERTICAL_DIFFUSION_TIME_STEP = 1.0
 RANDOM_NUMBER_DISTRIBUTION = 'normal'
 SPECIFY_RANDOM_SEED = 'true'
   SPECIFIED_RANDOM_SEED = 1
 REMOVE_DEAD_PARTICLES = 'false'
 SUBGRID_BATHY = 'false'
            
"""
    def write_method(self):
        with open(os.path.join(self.run_dir,'FISH_PTM_method.inp'),'wt') as fp:
            fp.write(self.method_text())
