import os
import glob
from ... import utils

class PtmConfig(object):
    """
    A work in progress for configuring, running, loading FISH-PTM
    runs.
    """
    run_dir=None    
    end_time=None
    def __init__(self):
        self.regions=[]
        self.releases=[]
        self.groups=[]

    @staticmethod
    def load(path):
        cfg=PtmConfig()
        cfg.run_dir=path
        return cfg
        
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
        os.path.exists(self.run_dir) or os.makedirs(self.run_dir)
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
