# A first cut at a class that embodies many simulations sharing the same
# physical domain.
import os, shutil, sys, getopt, glob, datetime
from shapely import wkb, geometry
import logging

from .depender import rule,make,clear,DependencyGraph
from ...grid import (paver,orthomaker,trigrid)
from ...spatial import linestring_utils, wkb2shp
from . import sunreader
from ... import parse_interval
import interp_depth
import instrument,adcp,forcing
import field
import edge_depths

import domain_plotting

try:
    from osgeo import ogr,osr
except ImportError:
    print("falling back to direct ogr in domain")
    import ogr,osr

from safe_pylab import *
from numpy import *
import numpy as np # for transition to better style...

from numpy import random

def empty_create(target,deps=None):
    fp = open(target,'wt')
    fp.close()


def relpath(path, start=os.curdir):
    """Return a relative version of a path
    copied from python 2.6 posixpath (unavailable in 2.5)
    """

    if not path:
        raise ValueError("no path specified")

    start_list = os.path.abspath(start).split(os.sep)
    path_list = os.path.abspath(path).split(os.sep)

    # Work out how much of the filepath is shared by start and path.
    i = len(os.path.commonprefix([start_list, path_list]))

    rel_list = [os.pardir] * (len(start_list)-i) + path_list[i:]
    if not rel_list:
        return os.curdir
    return os.path.join(*rel_list)


def mylink(src,dst):
    # given two paths that are either absolute or relative to
    # pwd, create a symlink that includes the right number of
    # ../..'s
    if os.path.isabs(src): # no worries
        os.symlink(src,dst)
    else:
        pre = relpath(os.path.dirname(src),os.path.dirname(dst))
        os.symlink( os.path.join(pre,os.path.basename(src)), dst )

def ensure_real_file( fn ):
    """ If fn is a symlink, replace it with a copy of the target
    """
    if os.path.islink(fn):
        shutil.copyfile(fn,'tmp')
        os.unlink(fn)
        shutil.move('tmp',fn)

class VirtualInstrument(object):
    """ Define locations of 'virtual' instruments which can be
    realized from simulation output data.
    """
    def __init__(self,name,xy):
        self.name = name
        self.xy = xy


class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)


default_suntans_dat_template_txt = """########################################################################
#
#  Input file for SUNTANS.
#
########################################################################
Nkmax   		1	# Number of vertical grid points
stairstep		0	# 1 if stair-stepping, 0 if partial-stepping
rstretch		1	# Stretching factor for vertical grid (1<=rstretch<1.1)
CorrectVoronoi		2	# Whether or not to correct Voronoi points 1: dist-ratio 2: angle
VoronoiRatio		85.0	# Adjust the voronoi points by this amount if 1 then = centroid.
vertgridcorrect 	0 	# Correct vertical grid if Nkmax is too small
IntDepth 		0	# 1 if interpdepth, 0 otherwise
dzsmall			0	# obsolete
scaledepth 		0 	# Scale the depth by scalefactor
scaledepthfactor 	0 	# Depth scaling factor (to test deep grids with explicit methods)
thetaramptime	        86400   # Timescale over which theta is ramped from 1 to theta (fs theta only)
theta			0.55	# 0: fully explicit, 1: fully implicit
thetaS			0.55	# For scalar advection
thetaB			0.55	# For scalar advection
beta                    0.00078   # Coefficient of expansivity of salt
kappa_s                 0	# Vertical mass diffusivity
kappa_sH                0       # Horizontal mass diffusivity
gamma 			0  	# Coefficient of expansivity of temperature.
kappa_T                 0       # Vertical thermal diffusivity
kappa_TH                0       # Horizontal thermal diffusivity
nu 			1e-6  	# Laminar viscosity of water (m^2 s^-1) (w*w*dt=1e-3*1e-3*90)
nu_H 			0.0     # Horizontal laminar viscosity of water (m^2 s^-1) (.1*.1*90)
tau_T			0.0 	# Wind shear stress
z0T	                0.0 	# Top roughness
z0B	                0.0001	# Bottom roughness
CdT	               	0.0	# Drag coefficient at surface
CdB	                0.0	# Drag coefficient at bottom
CdW			0.0       # Drag coefficient at sidewalls
turbmodel		1	# Turbulence model (0 for none, 1 for MY25)
dt 			120      # Time step
Cmax 			10      # Maximum permissible Courant number
nsteps 		        20000
ntout   		10     # How often to output data 
ntprog   		1 	# How often to report progress (in %)
ntconserve 		1	# How often to output conserved data
nonhydrostatic		0	# 0 = hydrostatic, 1 = nonhydrostatic
cgsolver		1	# 0 = GS, 1 = CG
maxiters		5000	# Maximum number of CG iterations
qmaxiters		2000	# Maximum number of CG iterations for nonhydrostatic pressure
hprecond                1       # 1 = preconditioned  0 = not preconditioned
qprecond		2	# 2 = Marshall et al preconditioner(MITgcm), 1 = preconditioned, 0 = not preconditioned
epsilon			1e-10 	# Tolerance for CG convergence
qepsilon		1e-10	# Tolerance for CG convergence for nonhydrostatic pressure
resnorm			0	# Normalized or non-normalized residual
relax			1.0	# Relaxation parameter for GS solver.	
amp 			0       # amplitude
omega 			0	# frequency
flux 			0	# flux
timescale		0	# timescale for open boundary condition
volcheck		0	# Check for volume conservation
masscheck		0	# Check for mass conservation
nonlinear		3	# 2 2nd order central, 1 if 1st Order upwind for horizontal velocity, 0 for u^(n+1)=u^n
newcells		0	# 1 if adjust momentum in surface cells as the volume changes, 0 otherwise
wetdry			1       # 1 if wetting and drying, 0 otherwise
Coriolis_f              9.36e-5 # Coriolis frequency f=2*Omega*sin(phi)
sponge_distance	        0 	# Decay distance scale for sponge layer
sponge_decay	        0 	# Decay time scale for sponge layer
readSalinity		1	# Whether or not to read initial salinity profile from file InitSalinityFile
readTemperature		0	# Whether or not to read initial temperature profile from file InitTemperatureFile
start_day	        0       # Offset for tides from start of simulation year (in days) GMT!!!
start_year           2005       # start day is relative to the beginning of this year
########################################################################
#
#  Grid Files
#
########################################################################
pslg   oned.dat 	# Planar straight line graph (input)
points points.dat	# Vertices file (input)
edges edges.dat		# Edge file (input)
cells cells.dat		# Cell centered file (input)
depth depth.dat	# Depth file for interpolation (if INTERPDEPTH=1) (input)
edgedepths edgedepths.dat
celldata celldata.dat	# Cell-centered output (output)
edgedata edgedata.dat	# Edge-centered output (output)
vertspace vertspace.dat	# Vertical grid spacing (output)
topology topology.dat	# Grid topology data
########################################################################
#
#  Output Data Files
#
########################################################################
FreeSurfaceFile   	      	fs.dat
HorizontalVelocityFile 	      	u.dat
VerticalVelocityFile 		w.dat
SalinityFile 			s.dat
BGSalinityFile 			s0.dat
TemperatureFile			T.dat
PressureFile			q.dat
VerticalGridFile 		g.dat
ConserveFile			e.dat
ProgressFile	        	step.dat
StoreFile			store.dat
StartFile			start.dat
EddyViscosityFile		nut.dat
ScalarDiffusivityFile		kappat.dat
# LagrangianFile			lagra.dat
########################################################################
#
# Input Data Files
#
########################################################################
InitSalinityFile	mbay_salinity.dat
InitTemperatureFile	Tinit.dat
TideInput               tidecomponents.dat
BoundaryInput           boundaries.dat
TideOutput              tidexy.dat
########################################################################
#
# For output of data
#
########################################################################
ProfileVariables	husnb      # Only output free surface and currents
DataLocations	dataxy.dat       # dataxy.dat contains column x-y data
ProfileDataFile	profdata.dat     # Information about profiles is in profdata.dat
ntoutProfs		10      # 10 minutes. Output profile data every n time steps
NkmaxProfs		0       # Only output the top 1 z-level
numInterpPoints 	1        # Output data at the nearest neighbor.
"""

def default_suntans_dat_template():
    return sunreader.SunConfig(text=default_suntans_dat_template_txt)
    
class Domain(domain_plotting.DomainPlotting):
    """
    A recipe for creating a ready to run set of input files
    Requires a suntans.dat.template file in datadir
    """
    np = 1
    # some commands can utilize mpi - it has to be specified on the command line
    # before the command (python mydomain.py -m my_command datadir ...)
    # experimental! (but what isn't?)
    mpicomm = False
    
    # when spinning up a fine grid to match the timing of a
    # coarse grid (in preparation for copying the coarse grid
    # salinity to the fine grid), this is the lead time used:
    hydro_spinup_days = 3

    # coordinate reference - defaults to UTM Zone 10, NAD83, meters
    spatial_ref='EPSG:26910' # the projection of the grid
    lonlat_ref = 'WGS84' # the specific system to use for lat/lon inputs

    # Thalwegging: starting from any inflow edges, search (breadth-first)
    # for a cell that is at least as deep as thalweg_depth, then set up
    # to thalweg_max_cells along that path to thalweg_depth.
    # set thalweg_max_cells = 0 to disable
    thalweg_max_cells = 0 # disabled.
    thalweg_depth = 0.0  # positive down, and must account for bathy_offset
    
    enable_edge_based_bathymetry = True
    # whether thin bed cells should be removed.  if this is not enabled, you probably
    # want to have a nonzero dz_lump_bed_frac in suntans.dat
    trim_depths = True

    private_rundata = False

    # Names of other config files:
    sedi_dat = "sedi.dat"
    sedi_ls_dat = "sedi_ls.dat"
    wave_dat = "wave.dat"
    contiguous_dat = "contiguous.dat"

    def __init__(self,datadir = 'rundata'):
        # self.rundata_dir is where all the global grid, bc, etc. information is kept,
        # plus partitioned grids for each number of processors.
        # if this is a private_rundata Domain, though, these are kept in the initial
        # directory of a particular run.
        self.log=logging.getLogger(self.__class__.__name__)
        self.rundata_dir = datadir

        self.conf = self.suntans_config()

        # sediment configuration:
        self.sedi_conf = self.sedi_config() # old sediment code - not really supported at this point
        self.sedi_ls_conf = self.sedi_ls_config() # new sediment code.

    def sedi_config(self):
        return None # disabled by default 

    def sedi_ls_config(self):
        return None # disabled by default 
    
    def suntans_config(self):
        """ Returns a SunConfig object used as the base suntans.dat.

        the usage here is migrating - it used to be that if the template
        file did not exist, it would be initialized from code like this,
        and if it did exist, it would be used as is.  but that allows
        the domain code and the suntans.dat file to get out of sync, and
        it's too confusing to know which is in charge.

        so now the base suntans.dat always comes from here.  the file is
        used only to timestamp when the settings have changed.

        so this method starts with the base suntans.dat above, and allows
        programmatic modifications.  code elsewhere takes care of writing
        this out to the suntans_dat_template_file() if there is a relevant
        change.
        """
        conf = default_suntans_dat_template()

        # Make changes based on settings that are already known
        if self.enable_edge_based_bathymetry:
            conf['edgedepths'] = 'edgedepth.dat'
        else:
            del conf['edgedepths']

        # assume for now that we're always specifying the exact layers
        conf['rstretch']=0
        conf['Nkmax'] = len(self.vertspace())

        return conf

    def virtual_instruments(self):
        return []

    def original_grid_dir(self):
        return os.path.join(self.rundata_dir,'original_grid')
    def depth_untrimmed_file(self):
        return os.path.join(self.rundata_dir,self.conf['depth'] + '.untrimmed')
    def edgedepth_untrimmed_file(self):
        return os.path.join(self.rundata_dir,self.conf['edgedepths'] + '.untrimmed')
    
    def depth_thalwegged_file(self):
        return os.path.join(self.rundata_dir,'depth.dat.thalwegged')
    def edgedepth_thalwegged_file(self):
        return os.path.join(self.rundata_dir,self.conf['edgedepths'] + '.thalwegged')
    
    def depth_file(self):
        return os.path.join(self.rundata_dir,self.conf['depth'])
    def edgedepth_file(self):
        return os.path.join(self.rundata_dir,self.conf['edgedepths'])
    
    def global_bc_grid_dir(self):
        return self.rundata_dir
    def vertspace_dat_in_file(self):
        return os.path.join(self.rundata_dir,'vertspace.dat.in')
    
    def partitioned_grid_dir(self):
        if self.private_rundata:
            return os.path.join(self.rundata_dir)
        else:
            return os.path.join(self.rundata_dir,'np%03d'%self.np)
        
    def salinity_file(self):
        readSalinity = self.conf.conf_int('readSalinity')
        if readSalinity == 1:
            # just reads vertical profile
            return os.path.join(self.partitioned_grid_dir(),self.conf.conf_str("InitSalinityFile"))
        elif readSalinity == 2:
            return os.path.join(self.partitioned_grid_dir(),self.conf.conf_str("InitSalinityFile")+".0")
        else:
            return None
    def temperature_file(self):
        readTemperature = self.conf.conf_int('readTemperature')
        if readTemperature == 1:
            # just reads vertical profile
            return os.path.join(self.partitioned_grid_dir(),self.conf.conf_str("InitTemperatureFile"))
        elif readTemperature == 2:
            return os.path.join(self.partitioned_grid_dir(),self.conf.conf_str("InitTemperatureFile")+".0")
        else:
            return None

    def sedi_file(self):
        return os.path.join(self.rundata_dir, self.sedi_dat)
    
    def sedi_ls_file(self):
        return os.path.join(self.rundata_dir,self.sedi_ls_dat)
            
    def dataxy_file(self):
        return os.path.join(self.rundata_dir,'dataxy.dat')
    
    def suntans_dat_template_file(self):
        return os.path.join(self.original_grid_dir(),'suntans.dat')

    def create_suntans_dat_template(self,target,deps):
        self.log.debug("Writing self.conf to %s"%target)
        
        if not os.path.exists(self.rundata_dir):
            os.mkdir(self.rundata_dir)

        self.conf.write_config(filename=target,check_changed=True)
        self.log.debug("Done writing self.conf")

    def vertspace(self):
        """ definitive source for z-layer elevations.  these are the 
        thickness of the cells, starting from 0 going from the surface to
        the bed. 
        """
        # typically this would be overridden in the subclass
        return np.loadtxt("path/to/vertspace.dat.in")

    def create_vertspace_dat_in(self,target,deps):
        # 2015-10-08: try having the definitive z vertical spacing
        # set as a method, self.zlevels().
        # In order to avoid unnecessarily modifying file timestamps,
        # this method compares self.zlevels() and the contents of
        # an existing vertspace.dat.in, updating the file if they differ.

        vals=self.vertspace()

        if os.path.exists(target):
            target_vals=np.loadtxt(target)
            if len(vals) == len(target_vals) and np.allclose(vals,target_vals):
                self.log.debug("vertspace file %s is up to date"%target)
                return
        vals.tofile(target,"\n")
                  
    ## Create the original, global grid
    def interior_lines(self):
        """ If there are internal guidelines for the grid creation.  

        This would be for forcing cells to be aligned with some isobath, or
        for defining sections for flux analysis.
        """
        return []

    def tweak_paver(self):
        """ gives subclasses a chance to make minor changes to the paving instance
        before a grid is created
        """
        pass
    def prep_paver(self):
        """ Create the paving instance, get everything ready to start paving.
        Useful for debugging the paving process.
        """
        rings = self.shoreline()
        degenerates = self.interior_lines()
        density = self.density()
        
        p = paver.Paving(rings,density,degenerates=degenerates)
        self.p = p
        self.tweak_paver()
        
    def grid_create(self,target,deps):
        if not os.path.exists(self.original_grid_dir()):
            os.makedirs( self.original_grid_dir() )

        self.prep_paver()
        
        self.p.verbose = 1
        self.p.pave_all()
        self.p.renumber()
        
        self.p.write_suntans(self.original_grid_dir())
    
    def grid_read(self):
        return orthomaker.OrthoMaker( suntans_path=self.original_grid_dir() )

    def global_markers_create(self,target,deps):
        """ mark edges according to the forcing in the global grid.
        Try to avoid actually loading any data, just figure out which
        edges/cells will be forced.
        """
        # in the past, this copied the files over, but if there was an error
        # while marking them, the unmarked files were left.
        # now, try to catch any exceptions long enough to remove the unmarked
        # grid
        # another option would be to read in the original grid, and write
        # the marked grid, but this way there is at least a chance of
        # realizing that the marks have not changed, and the grid can be
        # left as is (so modification doesn't trigger unnecessary repartitioning
        # and so forth).

        remove_on_exception = []
        
        try:
            # Copy the original grid to here, then update with boundary markers
            for f in ['cells.dat','edges.dat','points.dat']:
                self.log.debug("Copying %s to %s"%(f,self.global_bc_grid_dir()))

                fn = os.path.join(self.global_bc_grid_dir(),f)
                remove_on_exception.append(fn)
                src = os.path.join( self.original_grid_dir(),f )
                dst = fn
                try:
                    shutil.copy(os.path.join(self.original_grid_dir(),f), fn)
                except:
                    self.log.error("Problem while copying %s => %s"%(src,dst))
                    raise

            # and copy the template suntans.dat to here:
            sdat_template = self.suntans_dat_template_file()
            sdat_global_bc = os.path.join(self.global_bc_grid_dir(),'suntans.dat')

            if not os.path.exists( sdat_global_bc ):
                self.log.debug("Copying suntans.dat template to global BC directory")
                shutil.copy(sdat_template,sdat_blobal_bc)

            # Read the original grid, but write a global grid into 
            # global_bc_grid_dir()
            gforce = self.global_forcing( datadir=self.global_bc_grid_dir() )
            gforce.update_grid()
        except:
            self.log.error("Exception while marking grid - removing potentially unmarked grid files")
            for f in remove_on_exception:
                if os.path.exists( f ):
                    os.unlink(f)
            raise 

    def boundaries_dat_create(self,target,deps):
        gforce = self.global_forcing(datadir=self.partitioned_grid_dir())
        gforce.write_boundaries_dat()

    # Spatial reference convenience routines:
    geo_to_local = None
    local_to_geo = None
    def convert_geo_to_local(self,lonlat):
        self.set_transforms()
        x,y,z = self.geo_to_local.TransformPoint(lonlat[0],lonlat[1],0)
        return [x,y]
    def convert_local_to_geo(self,xy):
        self.set_transforms()
        lon,lat,z = self.geo_to_local.TransformPoint(xy[0],xy[1],0)
        return [lon,lat]
        
    def set_transforms(self,force=False):
        if force or (self.geo_to_local is None or self.local_to_geo is None):
            local = osr.SpatialReference() ; local.SetFromUserInput(self.spatial_ref)
            geo = osr.SpatialReference() ; geo.SetFromUserInput(self.lonlat_ref)
            self.geo_to_local = osr.CoordinateTransformation(geo,local)
            self.local_to_geo = osr.CoordinateTransformation(local,geo)
            
    # Utility methods for defining freshwater sources:
    def add_flows_from_shp(self,gforce,flows_shp,bc_type='bc_type'):
        """ Reads flow sources from the given shapefile and adds them to the
        global forcing

        Since flows will use domain-dependent datasources, this will make calls
        back to self.flow_datasource(driver,{fields}) to get the relevant datasource.

        if bc_type is given, it is the name of a text field in the shapefile
        which gives the type of element which is forced:
         BOUNDARY: boundary edges
         BED: bed cell (which is currently turned into surface cell)
         others...
        """
        if not os.path.exists(flows_shp):
            self.log.error("While trying to open shapefile %s"%flows_shp)
            raise Exception("Failed to find flows shapefile")

        flows=wkb2shp.shp2geom(flows_shp)
        groups=gforce.add_groups_bulk(flows,bc_type_field=bc_type)

        # zero = forcing.Constant('zero',0)

        for feat_id in range(len(flows)):
            grp=groups[feat_id]

            fields={ fld:flows[fld][feat_id]
                     for fld in flows.dtype.names }

            self.log.debug("Processing flow data from shapefile: %s"%fields.get('name','n/a'))
            
            # geo = fields['geom']

            # might be either a point or a line - just process all the points
            # points = np.array(geo)

            # if bc_type is not None:
            #     typ=fields[bc_type]
            # else:
            #     typ="FORCE_Q"
            # 
            # if typ=='FORCE_BED_Q':
            #     typ='FORCE_SURFACE_Q'
            #     print "Will fake the bed source with a surface source"

            # # Define the forcing group:
            # if typ in ['FORCE_SURFACE_Q','FORCE_BED_Q']:
            #     grp = gforce.new_group( nearest_cell=points[0] )
            # else:
            #     if len(points) == 1:
            #         grp = gforce.new_group( nearest=points[0] )
            #     else:
            #         # just use endpoints to define an arc along the boundary.
            #         grp = gforce.new_group( points=[points[0],
            #                                         points[-1]] )

            self.attach_datasources_to_group(grp,fields)

            # flow = self.flow_datasource(fields['driver'],fields)
            # 
            # grp.add_datasource(flow,typ)
            # if grp.edge_based():
            #     grp.add_datasource(zero,"FORCE_S")
            # # currently surface fluxes can only have zero salinity
        
    def partitioned_create(self,target,deps):
        # setup the partitioned subdirectory, then partition
        if not os.path.exists(self.partitioned_grid_dir()):
            os.mkdir(self.partitioned_grid_dir())

        if os.path.abspath(self.global_bc_grid_dir()) != os.path.abspath(self.partitioned_grid_dir()):
            for f in ['cells.dat','edges.dat','points.dat','depth.dat','suntans.dat']:
                self.log.debug("Copying %s"%f)
                shutil.copy(os.path.join(self.global_bc_grid_dir(),f),
                            os.path.join(self.partitioned_grid_dir(),f))

            # these may exist - copy if they do:
            section_input_file = self.conf.conf_str("SectionsInputFile")
            if section_input_file:
                section_inputs = [section_input_file]
            else:
                section_inputs = []

            if self.sedi_conf:
                self.sedi_conf.write_config( os.path.join( self.global_bc_grid_dir(),self.sedi_dat) )
            if self.sedi_ls_conf:
                self.sedi_ls_conf.write_config( os.path.join( self.global_bc_grid_dir(),self.sedi_ls_dat) )
                
            for f in [self.wave_dat,self.contiguous_dat] + section_inputs:
                if os.path.exists( os.path.join(self.global_bc_grid_dir(),f) ):
                    self.log.debug("Copying %s"%f)
                    shutil.copy(os.path.join(self.global_bc_grid_dir(),f),
                                os.path.join(self.partitioned_grid_dir(),f))


            if self.conf.conf_int('outputPTM',0) > 0:
                srcfile = os.path.join(self.global_bc_grid_dir(),'ptm.in')
                if os.path.exists(srcfile):
                    shutil.copy(srcfile, self.partitioned_grid_dir())

            shutil.copy(self.dataxy_file(), os.path.join(self.partitioned_grid_dir()))
        else:
            # for some files like sedi.dat may need to create them even when there is
            # no copying to an npXX directory to worry about.
            if self.sedi_conf:
                self.sedi_conf.write_config( os.path.join( self.partitioned_grid_dir(),self.sedi_dat) )
            if self.sedi_ls_conf:
                self.sedi_ls_conf.write_config( os.path.join( self.partitioned_grid_dir(),self.sedi_ls_dat))

        rstretch=self.conf.conf_float('rstretch')
        if rstretch == 0.0:
            if not os.path.exists( os.path.join(self.partitioned_grid_dir(),'vertspace.dat.in')):
                self.log.debug("Copying vertspace.dat.in from %s to %s"%(self.vertspace_dat_in_file(),
                                                                         self.partitioned_grid_dir()))
                shutil.copy(self.vertspace_dat_in_file(),
                            os.path.join(self.partitioned_grid_dir(),'vertspace.dat.in'))
        else:
            self.log.debug("No need for vertspace.dat.in because rstretch=%s"%rstretch)
        
        sun = sunreader.SunReader(self.partitioned_grid_dir())
        sun.domain_decomposition(np=self.np)

    def file_copy(self,target,deps):
        shutil.copy(deps[0],target)
        
    def run_create(self):
        """ create the time-invariant portios of a run in rundata/npNNN
        """
        # There are too many things going on in here - making it impossible
        # for subclasses to override selectively.
        clear()
        
        # the grid doesn't depend on anyone
        original_grid_file = os.path.join( self.original_grid_dir(), 'edges.dat')
        rule( original_grid_file,[],self.grid_create )

        # template file: if it doesn't exist, create one with the default contents
        # which also takes care of creating rundata
        rule( self.suntans_dat_template_file(), [], self.create_suntans_dat_template, always_run=True )
        
        # 2 copies of the suntans.dat file
        # can we get rid of this one?
        # og_suntans_dat = os.path.join(self.original_grid_dir(),'suntans.dat')
        suntans_dat = os.path.join(self.global_bc_grid_dir(),'suntans.dat')

        # rule( og_suntans_dat, self.suntans_dat_template_file(), lambda target,deps: shutil.copyfile(deps[0],target))
        sdtf = self.suntans_dat_template_file()
        self.log.debug("Template file: %s"%sdtf)
        rule( self.global_bc_grid_dir(),[], lambda target,deps: os.mkdir(self.global_bc_grid_dir()))
        rule( suntans_dat,  [self.suntans_dat_template_file(),self.global_bc_grid_dir()],
              lambda target,deps: shutil.copyfile(deps[0],target))
        
        global_markers_file = os.path.join(self.global_bc_grid_dir(),'edges.dat')
        rule(global_markers_file, [original_grid_file, suntans_dat],
             self.global_markers_create )

        ## BATHYMETRY

        # several steps depend on knowing the dz distribution from vertspace.dat.in -
        if self.conf.conf_float('rstretch') == 0.0:
            vertspace_req = [ self.vertspace_dat_in_file() ]
        else:
            vertspace_req = []

        self.log.debug("vertspace_req is %s"%vertspace_req )
        
        ## First, the recipe for edge-based depths, if enabled:
        if self.enable_edge_based_bathymetry:
            self.log.info("Adding rules for edge-based bathymetry")
            rule( self.edgedepth_untrimmed_file(), [original_grid_file] + self.extra_depth_depends(),
                  self.edgedepth_untrimmed_create )

            if self.thalweg_max_cells > 0:
                # global markers are in here so we can add thalwegs for freshwater sources
                rule( self.edgedepth_thalwegged_file(),
                      [self.edgedepth_untrimmed_file(),global_markers_file],
                      self.edgedepth_thalwegged_create )
                file_to_trim = self.edgedepth_thalwegged_file()
            else:
                file_to_trim = self.edgedepth_untrimmed_file()

            # the trimmed rule should be the same for everyone, though
            rule( self.edgedepth_file(),
                  [suntans_dat,file_to_trim]+vertspace_req,
                  self.edgedepth_trimmed_create )

            # In the case of edge-based bathymetry, we still want to generate cell depths since
            # they will be read in, but they are basically slaved to the edge bathymetry
            rule( self.depth_file(),
                  [self.edgedepth_file(),original_grid_file],
                  self.depth_from_edges_create )
        else: ## Only cell-based bathymetry  
            ## Then the recipe for cell-based depths:
            rule( self.depth_untrimmed_file(), [original_grid_file] + self.extra_depth_depends(),
                  self.depth_untrimmed_create )

            if self.thalweg_max_cells > 0:
                # global markers are in here so we can add thalwegs for freshwater sources
                rule( self.depth_thalwegged_file(),
                      [self.depth_untrimmed_file(),global_markers_file],
                      self.depth_thalwegged_create )
                file_to_trim = self.depth_thalwegged_file()
            else:
                file_to_trim = self.depth_untrimmed_file()

            # the trimmed rule should be the same for everyone, though
            rule( self.depth_file(),
                  [suntans_dat,file_to_trim] + vertspace_req,
                  self.depth_trimmed_create )


        partitioned_file = os.path.join(self.partitioned_grid_dir(),'edges.dat.0')

        partition_deps = [global_markers_file, self.depth_file(), suntans_dat, self.dataxy_file()] + vertspace_req

        # give subclasses the option of creating a vertspace.dat.in file
        rule( self.vertspace_dat_in_file(), [self.suntans_dat_template_file()], self.create_vertspace_dat_in )

        rule( partitioned_file, partition_deps, self.partitioned_create )

        boundaries_dat_file = os.path.join( self.partitioned_grid_dir(),'boundaries.dat.0')

        # No longer create boundary data for npNNN - this will be created on a per-run
        # basis
        # rule( boundaries_dat_file, [partitioned_file,suntans_dat],
        #       self.boundaries_dat_create)

        # the profile points - should depend on the original grid so that we can make sure
        # all points are within the grid
        rule( self.dataxy_file(), original_grid_file, self.create_dataxy_file)

        if self.enable_edge_based_bathymetry:
            # this used to get included regardless - now only do it if we are really using
            # edge depths, and fix the code to handle the case when the edge depths are missing.
            partitioned_edgedepths_file = self.partitioned_edgedepths_file()
            edge_deps = [partitioned_edgedepths_file]
            # include the step that fixes up the edge depths after partitioning.
            rule( partitioned_edgedepths_file, [self.edgedepth_file(), partitioned_file],
                  self.partitioned_edgedepths_create )
        else:
            edge_deps = []
            # why was this here??
            #rule( partitioned_edgedepths_file, [partitioned_file],
            #      self.partitioned_edgedepths_create )
            
        other_deps = []

        if 0: # do this on a per-run basis - see cmd_replace_initcond()
            if self.salinity_file():
                rule( self.salinity_file(), [partitioned_file,suntans_dat] + edge_deps,
                      self.salinity_create)
                other_deps.append( self.salinity_file() )

            if self.temperature_file():
                rule( self.temperature_file(), [partitioned_file,suntans_dat] + edge_deps,
                      self.temperature_create )
                other_deps.append( self.temperature_file() )

        # flux analysis sections
        if self.section_endpoints is not None:
            rule( self.sections_input_file(), [global_markers_file], self.sections_create )
            rule( self.partitioned_sections_input_file(), [self.sections_input_file()],
                  self.file_copy )
            other_deps.append( self.partitioned_sections_input_file() )

        # boundaries_dat_file
        rule('total',[partitioned_file,self.dataxy_file() ] + edge_deps + other_deps)
            
        make('total')


    def partitioned_edgedepths_file(self):
        # return os.path.join(self.partitioned_grid_dir(),'edgedepths.dat.0')
        return os.path.join(self.partitioned_grid_dir(),self.conf['edgedepths']) + ".0"

    edgedepths_delete_friction_strips = False
    def partitioned_edgedepths_create(self,target,deps):
        sun = sunreader.SunReader( self.partitioned_grid_dir() )
        if self.enable_edge_based_bathymetry:
            edgedepth_file,partitioned_file = deps

            # The EdgeDepthWriter needs the global edge depths available, so symlink those
            # into the partitioned directory
            link_to_global_edgedepths = os.path.join(self.partitioned_grid_dir(),
                                                     os.path.basename(edgedepth_file))
            if not os.path.exists(link_to_global_edgedepths):
                mylink(edgedepth_file,link_to_global_edgedepths)

            # no direct check here to see that we're writing out the file that we think we are...
            #  partitioned_file coming in is hopefully from the same place as partitioned_grid_dir,
            #  and edgedepth_file is hopefully the same as sun.file_path('edgedepths',proc)

            ew = edge_depths.EdgeDepthWriter(delete_friction_strips=self.edgedepths_delete_friction_strips)
            ew.run( sun=sun )
                
        else:
            partitioned_file = deps
            # pull edge depths from the shallower of the two neighboring cells
            sun = sunreader.SunReader( self.partitioned_grid_dir() )
            global_grid = sun.grid()
            global_depth = loadtxt( self.depth_file() )

            for proc in range(sun.num_processors()):
                localg = sun.grid(proc)
                edgedata = sun.edgedata(proc)

                fp = open(sun.file_path('edgedepths',proc),'wt')

                for j in range(localg.Nedges()):
                    # find the global edge for this edge:
                    global_j = global_grid.find_edge( localg.edges[j,:2] )
                    nc1,nc2 = global_grid.edges[global_j,3:5]
                    if nc1 < 0:
                        nc1 = nc2
                    elif nc2 < 0:
                        nc2 = nc1
                    de = min( abs(global_depth[nc1,2]),
                              abs(global_depth[nc2,2]) )
                    
                    fp.write("%.6f %.6f %.6f\n"%(edgedata[j,4], edgedata[j,5], de) )
                

    

    dataxy_shps = []
    dataxy_fraction = 0.00
    def create_dataxy_file(self,target,deps):
        """
        For the long delta, we choose a small number of points randomly, plus
        exact points for the Polaris cruise stations.  Also include exact locations
        of the instruments from the mar-09 deployment
        
        deps: [original_grid_file]
        """
        grid = trigrid.TriGrid(suntans_path = self.original_grid_dir())
        vcenters = grid.vcenters()

        data_points = [] # zeros( (0,2), float64)

        for typ,label,points in self.enumerate_dataxy_shps():
            data_points.append( points )

        if len(data_points) > 0:
            data_points = concatenate(data_points)
        else:
            data_points = zeros( (0,2), float64 )
        
        # Now put all of those on the nearest cell center
        # everybody gets mapped to nearest
        for i in range(len(data_points)):
            c = grid.closest_cell( data_points[i] )
            data_points[i] = vcenters[c]

        # And some random samples:
        if self.dataxy_fraction > 0.0:
            R = random.random(grid.Ncells())

            random_points = vcenters[ R < self.dataxy_fraction ]

            if len(random_points)>0:
                if len(data_points) > 0:
                    data_points = concatenate( (data_points,random_points) )
                else:
                    data_points = random_points

        savetxt( self.dataxy_file(), data_points )
        
    # New depth approach - the subclass provides a depth field
    def extra_depth_depends(self):
        return []
    def depth_field(self):
        """ a Field instance, returning elevations in natural units.  The
        results here will be negated (to get soundins) and offset by 5m
        """
        raise Exception("domain::depth_field() must be provided by subclasses")

    def depth_untrimmed_create(self,target,deps):
        """ target: something like ../depth.dat.untrimmed
            deps[0]: a file in the original grid dir
            """
        f = self.depth_field()
        g = self.grid_read()
        vc = g.vcenters()
        # just evalute the depth field at the cell centers.  For edge-based depths, this is
        # sufficient - for good cell-based depths, this should be an integration over the area
        # of the cell.

        # Note that the field gives us elevations, as in positive is up, and 0 is the datum 0.
        depths = f.value( vc )


        # fix the sign and offset:
        offset = sunreader.read_bathymetry_offset() # 5m
        depths = -depths + offset

        # combine to one array:
        xyz = concatenate( (vc,depths[:,newaxis]), axis=1)
        savetxt(target,xyz)

        bad_count=np.sum(np.isnan(depths))
        if bad_count:
            raise Exception("%d of %d cell-center depths were nan - see %s"%(bad_count,
                                                                             len(depths),
                                                                             target))

    def depth_from_edges_create(self,target,deps):
        """ When bathymetry is specified on the edges, then this method can be used to
        create cell bathymetry directly from the edges.  Assumes that any thalwegging
        and bed level trimming has been done in the edge bathymetry.
        """
        g = self.grid_read()
        edge_depths = loadtxt(self.edgedepth_file())
        cell_depths = zeros( (g.Ncells(),3), float64)
        for i in range(g.Ncells()):
            jlist = g.cell2edges(i)
            cell_depths[i] = edge_depths[jlist,2].max()
        cell_depths[:,:2] = g.vcenters()
            
        savetxt(target,cell_depths)
            
    def edgedepth_untrimmed_create(self,target,deps):
        """ target: something like ../edgedepths.dat.untrimmed
            deps[0]: a file in the original grid dir
            """
        f = self.depth_field()
        g = self.grid_read()

        all_edge_depths = zeros( g.Nedges() )
        for e in range(g.Nedges()):
            if e % 10000 == 0:
                self.log.debug("Edge depths: %d / %d"%(e,g.Nedges()))
            all_edge_depths[e] = f.value_on_edge(g.points[g.edges[e,:2]])

        offset = sunreader.read_bathymetry_offset()
        all_edge_depths = -all_edge_depths + offset

        edge_field = field.XYZField(X=g.edge_centers(),
                                    F=all_edge_depths)
        edge_field.write_text(target)

        bad_count=np.sum(np.isnan(all_edge_depths))
        if bad_count:
            raise Exception("%d of %d edge depths were nan - see %s"%(bad_count,
                                                                      len(all_edge_depths),
                                                                      target))

    def edgedepth_thalwegged_create(self,thalwegged,deps):
        edge_untrimmed_file,global_markers_file = deps
        threshold = self.thalweg_depth

        # Load the grid and find all flow-forced boundaries (type 2)
        g = trigrid.TriGrid(suntans_path = os.path.dirname(global_markers_file) )
        to_thalweg = nonzero( g.edges[:,2] == 2 )[0]

        # Load untrimmed depth data:
        xyz = loadtxt(edge_untrimmed_file)

        self.log.info("Edge thalwegs...")
        for j in to_thalweg:
            self.log.debug("  Carving from edge %d"%j)
            g.carve_thalweg(xyz[:,2],threshold,j,mode='edges',max_count=self.thalweg_max_cells)

        savetxt(thalwegged,xyz)
        
    def depth_thalwegged_create(self,thalwegged,deps):
        untrimmed_file,global_markers_file = deps
        threshold = self.thalweg_depth

        # Load the grid and find all flow-forced boundaries (type 2)
        g = trigrid.TriGrid(suntans_path = os.path.dirname(global_markers_file) )
        to_thalweg = nonzero( g.edges[:,2] == 2 )[0]

        # Load untrimmed depth data:
        xyz = loadtxt(untrimmed_file)

        self.log.info( "Thalwegs...")
        for j in to_thalweg:
            self.log.debug( "  Carving from edge %d"%j)
            g.carve_thalweg(xyz[:,2],threshold,j,mode='cells',max_count=self.thalweg_max_cells)

        fp = open(thalwegged,'wt')
        for i in range(len(xyz)):
            fp.write("%9.2f %9.2f %5.3f\n"%(xyz[i,0],xyz[i,1],xyz[i,2]))
        fp.close()
                    
    def depth_trimmed_create(self,trimmed_file, deps):
        suntans_dat, untrimmed_file  = deps[:2]

        if self.trim_depths:
            interp = interp_depth.DepthInterp()
            interp.create_trimmed(trimmed_file,suntans_dat,untrimmed_file)
        else:
            shutil.copyfile(untrimmed_file,trimmed_file)

    def edgedepth_trimmed_create(self,edge_trimmed_file,deps):
        suntans_dat, edge_untrimmed_file = deps[:2]

        if self.trim_depths:
            interp = interp_depth.DepthInterp()
            interp.create_trimmed(edge_trimmed_file,suntans_dat,edge_untrimmed_file)
        else:
            shutil.copyfile(edge_untrimmed_file,edge_trimmed_file)

    section_endpoints = None
    def sections_input_file(self):
        return os.path.join( self.global_bc_grid_dir(), self.conf.conf_str('SectionsInputFile') )
    def partitioned_sections_input_file(self):
        return os.path.join( self.partitioned_grid_dir(), self.conf.conf_str('SectionsInputFile') )
            
    def sections_create(self,sections_input_file,deps):
        """ Create section_defs.dat based on section_endpoints
        """
        if self.section_shp is not None:
            # populate section_endpoints from section_shp
            ods = ogr.Open(self.section_shp)
            layer = ods.GetLayer(0)
            self.section_endpoints = []

            while 1:
                feat = layer.GetNextFeature()
                if feat is None:
                    break
                linestring = wkb.loads( feat.GetGeometryRef().ExportToWkb() )
                self.section_endpoints.append( array(linestring) )
            
        if self.section_endpoints is None:
            return

        fp = open(sections_input_file,"wt")

        fp.write("%d\n"%len(self.section_endpoints))

        # Need to load the global grid...  very slow........
        sun = sunreader.SunReader( os.path.join(self.global_bc_grid_dir() ) )
        self.log.info("Loading global grid.  Patience, friend")
        g = sun.grid()
        self.log.info("Finding shortest paths for sections" )
        # old code - just allowed endpoints -
        # new code - allow path
        for section_path in self.section_endpoints:
            self.log.debug("section: %s"%(section_path) )

            this_path = []
            for i in range(len(section_path)-1):
                a,b = section_path[i:i+2]
                n1 = g.closest_point(a)
                n2 = g.closest_point(b)
                
                this_path.append(g.shortest_path(n1,n2))

            this_path = concatenate( this_path )
            uniq = concatenate( ( [True],
                                   diff(this_path) != 0) )
            
            linestring = this_path[ uniq ]
                
            fp.write("%d\n"%len(linestring))

            fp.write(" ".join( [str(n) for n in linestring] ) + "\n")
        fp.close()
            
    ### Scalar initial conditions
    def salinity_create(self,datadir):
        """ default implementation calls self.initial_salinity() to figure out
        the salinity distribution.  
        """
        sun = sunreader.SunReader(datadir)
        readSalinity = sun.conf.conf_int('readSalinity')
        if readSalinity == 1:
            dimensions = 1
        else:
            dimensions = 3

        sun.write_initial_salinity(self.initial_salinity,dimensions=dimensions)
        
    def temperature_create(self,datadir):
        """ default implementation calls self.initial_temperature() to figure out
        the temperature distribution.  
        """
        sun = sunreader.SunReader(datadir)
        readTemperature = sun.conf.conf_int('readTemperature')
        if readTemperature == 1:
            dimensions = 1
        else:
            dimensions = 3
            
        sun.write_initial_temperature(self.initial_temperature,dimensions=dimensions)

    ### Virtual Instruments
    def find_jd0(self,sun):
        year = sun.conf.conf_int('start_year')
        return date2num( datetime.datetime(year,1,1) )
    
    def get_instrument(self,name_or_inst,sun,chain_restarts=1):
        """ returns a ctd-like instrument with ADCP-like instrument as inst.adcp
        """
        if isinstance(name_or_inst,str):
            # look up the instrument instance:
            vi = self.virtual_instruments()
            all_labels = [v.name for v in vi]

            i = all_labels.index(name_or_inst)

            virt = vi[i]
        else:
            virt = name_or_inst

        pnt = virt.xy
        label = virt.name

        self.log.debug("Creating virtual instrument at %s"%label)

        ### First, figure out grid-based metadata that will be the same
        #   for all of the sunreader instances (in case we are chaining backwards)

        # This can be cached in the virtual instrument instance:
        try:
            prof_i,proc,cell,depth,bathymetry,nkmax,nk = virt.cached_invariant_data
            print("Using cached data for bathymetry, nk, etc.")
        except AttributeError:
            # get a freesurface timeseries to show at the same time:
            prof_i = sun.xy_to_profile_index(pnt)
            print( "Found profile index %d"%prof_i)

            print( "Finding closest cell")
            proc,cell = sun.closest_cell(pnt)
            depth = sun.depth_for_cell(cell,proc)
            # depth: negative, and without the bathymetry offset.
            # depth_for_cell returns the value from the file, and suntans doesn't
            # respect signs in the bathy input file, so we just have to force the
            # sign here, and let depth be the bed elevation, a negative number.
            depth = -abs(depth)
            # then bathymetry is what is physically out there
            bathymetry = depth + sun.bathymetry_offset()

            nkmax = sun.conf_int('Nkmax')
            nk = sun.Nk(proc)[cell]


            virt.cached_invariant_data = prof_i,proc,cell,depth,bathymetry,nkmax,nk
        
        ### Then per-sunreader instance things:
        all_records = []
        all_records_wc = []
        all_adcps = [] # U,V,t for each period
        
        if chain_restarts:
            suns = sun.chain_restarts()
        else:
            suns = [sun]
            
        jd0 = self.find_jd0(sun)
        
        for sun in suns: # sun.chain_restarts():
            ## Figure out the times we want:
            prof_absdays = sun.timeline(units='absdays',output='profile')

            self.log.info("Processing sun run JD %g-%g"%(prof_absdays[0]-jd0,
                                                         prof_absdays[-1]-jd0))

            sun_t0 = sun.time_zero()

            self.log.debug("  Reading freesurface timeseries")
            h = sun.profile_data('FreeSurfaceFile')[:,prof_i]

            ctops = sun.h_to_ctop(h)

            # The top of the 'bottom' region:
            # take 2 cells, as long as they are valid
            top_of_bottom = maximum(nk-2,ctops)

            # and the bottom of the top region:
            bottom_of_top = minimum(ctops+2,nk)

            # these should be timestep,profile point,z-level
            salt = sun.profile_data('SalinityFile')[:,prof_i,:]
            nut  = sun.profile_data('EddyViscosityFile')[:,prof_i,:]
            # this should be timestep,{u,v,w}, profile point, z-level
            U    = sun.profile_data('HorizontalVelocityFile')[:,:,prof_i,:]

            # To allow for running this while a simulation is still going,
            # we have to be careful about how many profile steps there are.
            n_p_steps = [ h.shape[0], len(prof_absdays), U.shape[0], salt.shape[0] ]
            n_steps = min(n_p_steps)
    

            # create a record array that will have depth, salinity_surface
            # salinity_bed, jd
            records = zeros( (n_steps,),
                             dtype=[('jd','f8'),
                                    ('h','f8'),
                                    ('Depth','f8'),
                                    ('salinity_surface','f8'),
                                    ('salinity_bed','f8'),
                                    ('salinity_avg','f8'),
                                    ('u_surface','f8'),
                                    ('u_bed','f8'),
                                    ('v_surface','f8'),
                                    ('v_bed','f8'),
                                    ('u_avg','f8'),
                                    ('v_avg','f8'),
                                    ('int_udz','f8'),
                                    ('int_vdz','f8'),
                                    ('k_top','i4'),
                                    ] )
            # and for values that have a vertical dimension:
            # shares the same timeline as records.
            records_wc = zeros( (n_steps,nk),
                                dtype=[('u','f8'),
                                       ('v','f8'),
                                       ('salinity','f8'),
                                       ('nut','f8'),
                                       ('z_top','f8'),
                                       ('z_bot','f8')] )

                                
            records['jd'] = prof_absdays[:n_steps] - jd0
            # h is NAVD88-bathymetry_offset(), so adjust back to NAVD88
            records['h' ] = h[:n_steps] + sun.bathymetry_offset()
            # and the difference is the depth
            records['Depth'] = h[:n_steps] - depth

            # tops of each z-level - sun.z_levels() gives positive values
            # so negate to get proper elevations
            z_tops = -concatenate( ( [0],sun.z_levels()) )
            # adjust the bottom z-level
            # nk is the number of cells in the water column - so
            # z_top[0] is the top of z-level 0
            # z_top[nk] is the top of z-level nk, blah blah blah.
            # this used to be nk+1

            # keep going back and forth on whether there should be a minus
            # sign here.
            z_tops[nk] = depth
            # take the means
            z_mids = 0.5*(z_tops[:-1] + z_tops[1:])

            ## Put together some ADCP-like records
            adcpU = U[:n_steps,0,:].copy() # copy so nans don't pollute summations below
            adcpV = U[:n_steps,1,:].copy()
            adcpU[:,nk:] = nan
            adcpV[:,nk:] = nan
            adcpt = prof_absdays[:n_steps]-jd0
            # these will get nan'd out in the loop below, but go ahead and put
            # them in the list now.
            all_adcps.append( [adcpU,adcpV,adcpt])
            ##

            ## For now, do both the adcp style output and a 3D record style output
            records_wc['u'] = U[:n_steps,0,:nk]
            records_wc['v'] = U[:n_steps,1,:nk]
            records_wc['salinity'] = salt[:n_steps,:nk]
            records_wc['nut']  = nut[:n_steps,:nk]

            

            for i in range(n_steps):

                # define vectors that will do various averages on the inputs
                z_tops_i = z_tops.copy()
                # adjust the surface height
                z_tops_i[ctops[i]] = h[i]
                records['k_top'][i] = ctops[i]
                # thickness of each z-level
                dzz = diff( z_tops_i )

                k=arange(nkmax)

                ## some regions:
                surface = dzz* ((k>=ctops[i]) & (k<bottom_of_top[i]))
                bed     = dzz* ((k>=top_of_bottom[i]) & (k<nk))
                water   = dzz* ((k>=ctops[i]) & (k<nk))

                # Do some averaging...
                records['salinity_surface'][i] = sum( salt[i] * surface ) / sum(surface)
                records['salinity_bed'][i] = sum( salt[i] * bed ) / sum(bed)
                records['salinity_avg'][i] = sum( salt[i] * water ) / sum(water)
                records['u_surface'][i] = sum( U[i,0] * surface ) / sum(surface)
                records['u_bed'][i] = sum(U[i,0] * bed ) / sum(bed)
                records['u_avg'][i] = sum(U[i,0] * water) / sum(water)
                records['v_surface'][i] = sum( U[i,1] * surface ) / sum(surface)
                records['v_bed'][i] = sum(U[i,1] * bed ) / sum(bed)
                records['v_avg'][i] = sum(U[i,1] * water) / sum(water)
                records['int_udz'][i] = sum(U[i,0] * water)
                records['int_vdz'][i] = sum(U[i,1] * water)


                # NaN out the dry/empty cells in the adcp data
                adcpU[i,0:ctops[i]] = nan
                adcpV[i,0:ctops[i]] = nan

                # And just to make life easy, include the dimensions of the z-levels
                records_wc['z_top'][i,:] = z_tops_i[:nk] + sun.bathymetry_offset()
                records_wc['z_bot'][i,:] = z_tops_i[1:nk+1] + sun.bathymetry_offset()
                # doctor up the water column output:
                for col in 'z_top','z_bot','u','v','salinity','nut':
                    records_wc[col][i,0:ctops[i]] = nan
                

            # profile outputs can overlap (when full output interval is not a
            # multiple of profile output interval), so we can get duplicate
            # data.
            if len(all_records) > 0:
                last_output_jd = all_records[-1]['jd'][-1]
                first_new_output = searchsorted( records['jd'], last_output_jd, side='right')
                if first_new_output < len(records):
                    self.log.debug("Trimming a bit off the beginning of profile data from a restart")
                    all_records.append( records[first_new_output:] )
                    all_records_wc.append( records_wc[first_new_output:] )
                else:
                    self.log.info("A restart didn't have any new profile data, so we skipped it")
            else:
                all_records.append(records)
                all_records_wc.append(records_wc)
            
        records = concatenate( all_records )
        records_wc = concatenate( all_records_wc )

        jd_base = sun.conf.conf_int('start_year')
        virt_ctd =  instrument.Instrument(records=records,comment=label,records_wc=records_wc,
                                          xy=pnt,bathymetry=bathymetry,
                                          jd_base = jd_base)

        # copy everything because there seem to be some errors otherwise
        all_adcpU = array( transpose( concatenate( [adcpU for adcpU,adcpV,adcpt in all_adcps] ) ),
                           copy=1)
        all_adcpV = array( transpose( concatenate( [adcpV for adcpU,adcpV,adcpt in all_adcps] ) ),
                           copy=1 )
        all_adcpt = concatenate( [adcpt for adcpU,adcpV,adcpt in all_adcps] )

        virt_ctd.adcp = adcp.AdcpMoored.from_values( all_adcpU,
                                                     all_adcpV,
                                                     z_mids,
                                                     all_adcpt, label)
             
        return virt_ctd

    ### METHODS for MANIPULATING RUNS ###

    def copy_to_datadir(self,dest_dir,src_dir=None,symlink=True):
        """  Symlink/Copy the requisite files from a rundata/npNNN directory
        to the given dest_dir.

        If symlink is true, then most files that are not expected to change
        (grid-related) are symlinked, and the others are copied.

        This is the python equivalent of copy_for_restart, without the restart
        part.

        sedi.dat and wave.dat files, if they exist in rundata/, will be copied
        to the new directory.

        Currently this isn't hooked into depender - it needs to be clearer
          how depender deals with a collection of files.

        if src_dir is not given, it is assumed to be the partitioned_grid_dir()
        corresponding to self.np
        """
        if src_dir is None:
            src_dir = self.partitioned_grid_dir()

        if symlink:
            maybe_link = mylink
        else:
            maybe_link = shutil.copyfile

        if os.path.abspath(src_dir) == os.path.abspath(dest_dir):
            print( "copy_to_datadir: nothing to do")
            return
        
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        for f in glob.glob(os.path.join(dest_dir,'*')):
            if os.path.isdir(f) and not os.path.islink(f):
                shutil.rmtree(f)
            else:
                os.unlink(f)

        # the @ prefix means it will use suntans.dat to look up the actual filename.
        # otherwise use the filename as given.
        per_processor = ['@topology','@BoundaryInput','@cells','@celldata','@edges','@edgedata']
        global_files  = ['@vertspace','@DataLocations','@points','@cells','@edges']

        # If edgedepths is set, we need to copy them, too
        if self.conf.conf_str('edgedepths') is not None:
            per_processor.append('@edgedepths')
        
        # Temperature & salinity depend on configuration:
        read_salt=self.conf.conf_int('readSalinity') ; read_temp=self.conf.conf_int('readTemperature')
        
        if read_salt == 1:
            global_files.append('@InitSalinityFile')
        elif read_salt == 2:
            per_processor.append('@InitSalinityFile')
            
        if read_temp == 1:
            global_files.append('@InitTemperatureFile')
        elif read_temp == 2:
            per_processor.append('@InitTemperatureFile')

        ## particle tracking:
        if self.conf.conf_int('outputPTM',0) > 0:
            if os.path.exists( os.path.join(src_dir,'ptm.in') ):
                global_files.append('ptm.in')

        # Pre-processor files:
        for f in per_processor:
            if f[0]=='@':
                datname = self.conf.conf_str(f[1:])
            else:
                datname = f
                
            for p in range(self.np):
                # may have to mangle the paths for symlink
                src_file = os.path.join(src_dir,datname+".%i"%p)
                if os.path.exists(src_file):
                    maybe_link( src_file,
                                os.path.join(dest_dir,datname+".%i"%p) )

        # Global files:
        for f in global_files:
            if f[0] == '@':
                datname = self.conf.conf_str(f[1:])
            else:
                datname = f
            src_file = os.path.join(src_dir,datname)
            if os.path.exists(src_file):
                maybe_link( src_file,
                            os.path.join(dest_dir,datname) )

        # Section analysis
        section_input_file = self.conf.conf_str("SectionsInputFile")

        if section_input_file is not None and os.path.exists( os.path.join(src_dir,section_input_file)):
            print( "Copying section definitions")
            maybe_link( os.path.join(src_dir,section_input_file),
                        os.path.join(dest_dir,section_input_file) )

        # Sediment, Waves:
        for dat_file in [self.sedi_dat,self.sedi_ls_dat,self.wave_dat]:
            if os.path.exists(os.path.join(src_dir,dat_file)):
                print( "Copying sediment data file")
                maybe_link( os.path.join(src_dir ,dat_file),
                            os.path.join(dest_dir,dat_file) )
            
        # BC Forcing data:
        if os.path.exists( os.path.join(src_dir,'datasources') ):
            print( "Copying boundary forcing data")
            shutil.copytree( os.path.join(src_dir,'datasources'),
                             os.path.join(dest_dir,'datasources') )

        if os.path.exists( os.path.join(src_dir,'fs_initial.dat') ):
            maybe_link( os.path.join(src_dir,'fs_initial.dat'),
                        os.path.join(dest_dir,'fs_initial.dat') )
        
        # suntans.dat is copied, not linked
        shutil.copy( os.path.join(src_dir,'suntans.dat'),
                     os.path.join(dest_dir,'suntans.dat') )

    def setup_startfiles(self,src_dir,dest_dir,symlink=True):
        """ Create symlinks from src_dir/storefiles to
        dest_dir/startfiles, and copy suntans.dat from the
        previous run
        """
        # clean them up so the string substitutions below are safe
        src_dir = src_dir.rstrip('/')
        dest_dir = dest_dir.rstrip('/')
        
        start = self.conf.conf_str('StartFile')
        store = self.conf.conf_str('StoreFile')
        
        if symlink:
            copier = mylink
        else:
            copier = shutil.copy

        for p in range(self.np):
            copier( os.path.join(src_dir,store+".%i"%p),
                    os.path.join(dest_dir,start+".%i"%p) )

        # handles store_sedi and friends:
        for f in glob.glob(os.path.join(src_dir,"store_*.dat.*")):
            new_f = f.replace("store_","start_")
            new_f = new_f.replace(src_dir,dest_dir)
            copier( f, new_f)

        shutil.copy( os.path.join(src_dir,'suntans.dat'),
                     os.path.join(dest_dir,'suntans.dat') )
        # the old one may be write-protected - force writable
        os.chmod( os.path.join(dest_dir,'suntans.dat'), 0o644 )

    def find_np_from_dir(self,datadir):
        # reads np from the first topology file
        topo_f = os.path.join( datadir,self.conf.conf_str('topology')+".0" )
        fp = open(topo_f,'rb')
        np = int( fp.readline().split()[0] )
        fp.close()
        return np

    def log_arguments(self,new_dir):
        fp = open( os.path.join(new_dir,'history.txt'), 'a+')

        def safely(s):
            if s.find(' ')>=0:
                return '"' + s + '"'
            else:
                return s

        fp.write("Created/updated with:  %s\n"%(" ".join(map(safely,sys.argv))))
        fp.close()


    def main(self,args=None):
        """ parse command line and start making things
        """
        def usage():
            print("python blah.py [-n #] [-tmM] [command]")
            print("    -n     Specify number of processors")
            print("    -i     Ignore timestamps")
            print("    -m     parallelize using mpi - indicates that the script is being invoked by mpirun or friends ")
            print("    -M     parallelize using mpi, and take care of queuing the process with the given number of processes from -n")
            print(" Command is one of:")
            print("  setup [<datadir>] - prepare the given directory for a run")
            print("       If no directory is given, the files are left in rundata/npNNN")
            print("  continue <old> <new> [+<duration>] - setup for continuing an ended or failed run in a new directory")
            print("  match_spinup <spin_dir> <new_dir> [date] - create a short run that ends at the same time")
            print("       as the spin_dir, or coincides with a full output step near/before the given date")
            print("  continue_with_spin <old> <spin> <new> [+<duration>] - like continue, but some scalars (currently only salt)")
            print("  write_instruments <datadir> - in a 'instruments' subdirectory, write virtual instrument data")
            print("       will be replaced with values interpolated from spin, and")
            print("       will replace symlinks to Storefiles with copies")
            print("  make_shps - write contour and boundary shapefiles")
            print("  plot_forcing <datadir>- dump plots of the various forcing datasources")

        if args is None:
            args = sys.argv[1:]

        self.check_timestamps = 1
        try:
            opts,rest = getopt.getopt(args, "n:imM:")
        except getopt.GetoptError:
            usage()
            sys.exit(1)

        for opt,val in opts:
            if opt == '-n':
                self.np = int(val)
            elif opt == '-i':
                self.check_timestamps = 0
            elif opt == '-m':
                import mpi4py.MPI
                self.mpicomm = mpi4py.MPI.COMM_WORLD
            elif opt == '-M':
                self.mpicomm = "SHELL"
            else:
                usage()
                sys.exit(1)

        if len(rest) > 0:
            cmd = rest[0]
            rest = rest[1:]
        else:
            # default action: 
            cmd = 'setup'

        # try making all output unbuffered:
        sys.stdout = Unbuffered(sys.stdout)
            
        if self.mpicomm == 'SHELL':
            print( "Will start the as an mpi job with %d processes"%self.np)
            print( "argv: ",sys.argv)
            # the interpreter tends to get stripped out - this may cause some problems for scripts
            # that are directly executable and specify some extra stuff to the interpreter...
            sub_args = ['python']
            for a in sys.argv:
                if a=='-M':
                    # indicate to the subprocess that it *in* mpi
                    a='-m'
                sub_args.append(a)
            sunreader.MPIrunner(sub_args,np=self.np,wait=0)
            print( "Job queued if you're lucky.")
        else:
            DependencyGraph.check_timestamps = self.check_timestamps
            self.invoke_command(cmd,rest)

    def run_has_completed(self,datadir):
        """ look at step.dat and return true if it exists and appears to represent a completed run
        """
        if os.path.exists(datadir+"/step.dat"):
            sun = sunreader.SunReader(datadir)
            sd = sun.step_data()
            return sd['step'] == sd['total_steps']
        return False
    def run_can_be_continued(self,datadir):
        """ slightly more permissive than run_has_completed - call this if you just want to make
        sure that a run was started, and got far enough such that continuing it makes sense.
        This does *not* check to see whether the given run is still running.

        The decision is made based on whether step.dat says that more than one step has been output
        (since the first step to be output is the initial condition...)
        """
        if os.path.exists(datadir+"/step.dat"):
            sun = sunreader.SunReader(datadir)
            sd = sun.step_data()
            return sd['steps_output'] > 1
        return False
    def run_crashed(self,datadir):
        """ Check to see if the given run crashed, as in the computations went unstable (as
        opposed to a case where it was killed by pbs, segfault'd, etc.)
        this depends on newer suntans code which differentiates storefile and crashfile.
        """
        crash = datadir+"/crash.dat.0"
        store = datadir+"/store.dat.0"
        return os.path.exists(datadir+"/crash.dat.0") and os.stat(crash).st_mtime > os.stat(store).st_mtime
        

    def invoke_command(self,cmd,args):
        meth_name = "cmd_" + cmd

        f = getattr(self,meth_name)
        f(*args)
        
    def cmd_write_instruments(self,datadir):
        dest_dir = os.path.join( datadir, 'instruments')
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
            
        sun = sunreader.SunReader(datadir)
            
        for inst in self.virtual_instruments():
            print( "processing instrument ",inst.name)
            v_inst = self.get_instrument(inst.name,sun,chain_restarts=1)
            fname = os.path.join( dest_dir,
                                  v_inst.safe_name() + ".inst" )
            v_inst.save( fname )

    def enumerate_dataxy_shps(self):
        """ Reads in the dataxy_shps, returns an iterator over
        profile points, (type,label,points)
        type = 'point' | 'transect'
        points = array( [[x,y],...] ), length 1 for single point samples

        Note that it returns the points as specified in the shapefiles,
        not nudged to be at cell centers.
        """
        count = 0
        for shp in self.dataxy_shps:
            print( "Loading profile points / transects from %s"%shp)
            if not os.path.exists(shp):
                raise Exception("Profile point shapefile not found: %s"%shp)

            feats=wkb2shp.shp2geom(shp)
            for idx,r in enumerate(feats):
                try:
                    name=r['name']
                except AttributeError:
                    name='feat%d'%idx
                    
                if r['geom'].geom_type=='Point':
                    coords=np.array(r['geom'].coords)
                    ftyp='point'
                elif r['geom'].geom_type=='LineString':
                    ftyp='transect'
                    coords=np.array(r['geom'].coords)
                    try:
                        resolution = r['resolution']
                        coords=linestring_utils.upsample_linearring(coords,resolution,closed_ring=0)
                    except AttributeError:
                        pass
                yield ftyp,name,coords
            
            # ods = ogr.Open(shp)
            # layer = ods.GetLayer(0)
            # 
            # while 1:
            #     count += 1
            #     feat = layer.GetNextFeature()
            #     if feat is None:
            #         break
            #     try:
            #         name = feat.GetField('name')
            #     except ValueError:
            #         name = "nameless%05d"%count
            # 
            #     g = wkb.loads( feat.GetGeometryRef().ExportToWkb() )
            #     if g.type == 'Point':
            #         # everybody gets mapped to nearest
            #         yield 'point',name,array(g.coords)
            #     elif g.type == 'LineString':
            #         points = array(g)
            #         resolution = feat.GetField('resolution')
            # 
            #         points = linestring_utils.upsample_linearring(points,resolution,closed_ring=0)
            #         yield 'transect',name,points
            #     else:
            #         raise Exception("create_dataxy can only handle Point or LineString layers, not %s"%g.type)

    def cmd_write_matlab(self,datadir,chain_restarts=1,insts=None):
        """ Like write_instruments, but
         (a) writes matlab files
         (b) grabs locations to dump from the self.dataxy_shps

         insts: generally leave it alone - this can be populated with a list of
         VirtualInstrument instances, in which case some data can be cached between
         processing restarts.
        """
        dest_dir = os.path.join( datadir, 'matlab')

        if not chain_restarts:
            dest_dir += '-single'
        
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        # Build the list of VirtualInstruments:
        if insts is None:
            insts = []
            for ptype,label,points in self.enumerate_dataxy_shps():
                if ptype == 'point':
                    insts.append( VirtualInstrument(name=label,xy=points[0]) )
                else:
                    print( "Not ready for transects")
        else:
            print("Will use supplied list of instruments")

        sun = sunreader.SunReader(datadir)

        if chain_restarts:
            chained = sun.chain_restarts()
            
            for restart in chained:
                self.cmd_write_matlab(restart.datadir,chain_restarts=0,insts=insts)
            
            single_dirs =[ os.path.join( s.datadir, 'matlab-single') for s in chained]
            self.join_matlab_outputs(dest_dir,single_dirs)
        else:
            # compare timestamps
            step_mtime = os.stat( sun.file_path('ProgressFile') ).st_mtime
            
            for inst in insts:
                mat_fn = os.path.join( dest_dir,
                                       instrument.sanitize(inst.name) ) + ".mat"
                
                print("Checking timestamp of ",mat_fn)
                if os.path.exists(mat_fn) and os.stat( mat_fn ).st_mtime > step_mtime:
                    print("Using pre-existing output")
                    continue
                
                v_inst = self.get_instrument(inst,sun,chain_restarts=chain_restarts)
                fname = os.path.join( dest_dir,
                                      v_inst.safe_name() + ".mat" )
                v_inst.save_matlab( fname )
                
    def join_matlab_outputs(self,dest_dir,single_dirs):
        """ Go through each of the single_dirs, and concatenate the data across the different
        directories, creating a new matlab output file in dest_dir
        """

        for matfile in glob.glob(single_dirs[0]+'/*.mat'):
            base_matfile = os.path.basename(matfile)

            print("Processing instrument %s"%base_matfile)
            mat_pieces = [os.path.join(sd,base_matfile) for sd in single_dirs]
            target_mat = os.path.join(dest_dir,base_matfile)
            
            instrument.Instrument.splice_matfiles(target_mat, mat_pieces)
                    
    def cmd_setup(self):
        """ This handles all the steps to get a partitioned grid with BC markers
        and bathymetry ready, storing the result in rundata/npNNN
        """
        self.run_create()

    def cmd_initialize_run(self,datadir,interval):
        """  Prepare a grid to simulate the given period
        """
        if self.private_rundata:
            self.rundata_dir = datadir
            
        self.cmd_setup() # make sure the partitioned grid is ready to go

        # safer to initialize a run with a copy of the grid, in case things get regridded
        # or partitioned later on.
        if not self.private_rundata:
            self.copy_to_datadir(datadir,symlink=False)
        else:
            print("Private rundata: files are ready to go")
            
        self.log_arguments(datadir)

        sun = sunreader.SunReader(datadir)
        def_start,def_end = sun.simulation_period()
        start_date,end_date = parse_interval.parse_interval(interval,
                                                            default_start = def_start,
                                                            default_end   = def_end)
        if 0: # 
            sun.conf.set_simulation_period(start_date,end_date)
            sun.conf.write_config()
        else:
            if not self.conf.is_grid_compatible(sun.conf):
                raise Exception("Looks like the differences between these suntans.dat files would make incompatible grids")
            self.conf.set_simulation_period(start_date,end_date)
            self.conf.write_config(os.path.join(datadir,'suntans.dat'))

        self.cmd_replace_forcing(datadir)
        self.cmd_replace_initcond(datadir)


    def cmd_run(self,datadir,wait=1,max_retries="0"):
        """ retry: if true, then if the simulation stops but doesn't produce a crash file,
        try to restart the run and keep going.

        if it is necessary to restart, the earlier runs will be renamed, and the final 
        run will be named datadir.
        """
        wait = (wait in (True, 1,'yes','on','wait','sync'))

        if wait:
            print("Will wait for run to complete")

        sun = sunreader.SunReader(datadir)

        max_retries = int(max_retries)

        prev_count=0
        while 1:
            sun.run_simulation(wait=wait)

            if not wait:
                print("Not waiting around for that to finish")
                break
            
            if self.run_has_completed(datadir):
                print("Looks like the run really did complete")
                break
            # otherwise - did it crash, or just disappear?
            if not self.run_can_be_continued(datadir):
                print("The run didn't finish, but didn't get far enough along to consider continuing")
                break
            if self.run_crashed(datadir):
                print("The run crashed - will not continue")
            if prev_count >= max_retries:
                print("Too many continuations: %d"%prev_count)
                break
            # So it's viable to restart this run.
            old_dir = datadir + "-pre%02d"%prev_count
            prev_count += 1
            os.rename(datadir,old_dir)
            self.cmd_setup_completion(old_dir,datadir)

    def cmd_setup_completion(self,incomplete_dir,next_dir):
        """ Given a run in incomplete_dir which is... incomplete,
        setup a new run in next_dir which will reach the originally intended
        ending time.
        """
        self.np = self.find_np_from_dir(incomplete_dir)

        self.copy_to_datadir(src_dir=incomplete_dir,dest_dir=next_dir,symlink=True)
        self.setup_startfiles(incomplete_dir,next_dir)

        ## Find when it was supposed to finish:
        old_sun = sunreader.SunReader(incomplete_dir)
        old_start,old_end = old_sun.simulation_period()

        ## And when the restart starts:
        sun = sunreader.SunReader(next_dir)
        new_start,new_end = sun.simulation_period()

        # So we have the start and end dates that we actually want it to run, but since
        # we're dealing with a restart, we just want to skip straight to specifying the
        # duration of the run.
        sim_days = date2num(old_end) - date2num(new_start)
        print("Old run was supposed to end %s"%old_end.strftime('%Y-%m-%d %H:%M'))
        print("New restart claims to begin %s"%new_start.strftime('%Y-%m-%d %H:%M'))
        print("So the simulation will be run for %g days"%sim_days)

        sun.conf.set_simulation_duration_days(sim_days)
        sun.conf.write_config()
        self.log_arguments(next_dir)

    def cmd_complete_run(self,incomplete_dir,next_dir,*args):
        """ setup_completion, then run it
        specify 'nowait' as an extra argument to queue the run and return
        """
        self.cmd_setup_completion(incomplete_dir,next_dir)
        self.cmd_run(next_dir,*args)
        
        
    def cmd_continue(self,old_dir,new_dir,interval=None,symlink=True,
                     base_config=None):
        """
        interval: specification of what the simulation period should be
        symlink: create symbolic links from StartFile in the

        base_config: if specified, a SunConfig object which will be updated
         with the right period.  Use to submit different settings than are in
         <old_dir>/suntans.dat
        """
        if self.private_rundata:
            # should we chain this back to get to the first one?
            self.rundata_dir = old_dir
            
        # determine np from old_dir
        old_np = self.find_np_from_dir(old_dir)
        self.np = old_np
        print("Found np=%i"%self.np)

        # First step is the non-time varying files
        # actually, these are typically not that big - a small fraction of
        # even one step of output, so forget the symlinking and just copy.
        self.copy_to_datadir(src_dir=old_dir,dest_dir=new_dir,symlink=False)
        
        # Second link the storefiles and copy old_dir/suntans.dat over
        self.setup_startfiles(old_dir,new_dir,symlink=symlink)

        ## Make any changes to the run time:
        sun = sunreader.SunReader(new_dir)
        def_start,def_end = sun.simulation_period()
        
        start_date,end_date = parse_interval.parse_interval(interval,
                                                            default_start = def_start,
                                                            default_end   = def_end)
        # in this case, it's a restart and we're not allowed to change the start date
        if start_date != def_start:
            print(interval)
            raise Exception("For restarts you can only specify :end_date, :+duration or +duration")

        # So we have the start and end dates that we actually want it to run, but since
        # we're dealing with a restart, we just want to skip straight to specifying the
        # duration of the run.
        sim_days = date2num(end_date) - date2num(start_date)
        print("So the simulation will be run for %g days"%sim_days)

        if base_config:
            conf = base_config
            conf.copy_t0(sun.conf)
        else:
            conf = sun.conf
        
        conf.set_simulation_duration(delta = end_date - start_date)
        conf.write_config(filename=sun.conf.filename)

        self.log_arguments(new_dir)
        self.cmd_replace_forcing(new_dir)

    def cmd_replace_forcing(self,data_dir):
        """
        Replace the forcing data - this will pick up the fact that it's
        a restart, and will construct BC data for the whole period
        (so it actually duplicates the BC data for the simulation period
        that has already run)
        removes the old forcing files first, so that we don't
        overwrite older files that are symlinked in
        """
        for f in glob.glob(os.path.join(data_dir, self.conf.conf_str('BoundaryInput')+"*")):
            os.unlink(f)
        gforce = self.global_forcing(datadir=data_dir)
        gforce.write_boundaries_dat()

        self.log_arguments(data_dir)

    def cmd_replace_initcond(self,datadir):
        sun = sunreader.SunReader(datadir)
        scalars = sun.scalars()
        if self.salinity_file() and 'SalinityFile' in scalars:
            self.salinity_create(datadir)

        if self.temperature_file() and 'TemperatureFile' in scalars:
            self.temperature_create(datadir) 

    def cmd_match_spinup(self,spin_dir,new_dir,end_date=None):
        # self.hydro_spinup_days
        if self.private_rundata:
            self.rundata_dir = new_dir
            
        self.run_create()
        
        # First, load the spinup run to figure out the timing for this run
        spin_sun = sunreader.SunReader(spin_dir)

        spin_outputs = date2num(spin_sun.time_zero()) + spin_sun.timeline(units='days',output='grid')

        ## Figure out the ending time for the transition run
        if end_date is not None:
            # new format - YYYYMMDDHHMMSS - or a prefix thereof
            end_datetime = parse_interval.parse_date_or_interval(end_date)
            end_absdays = date2num(end_datetime)
            i = max(searchsorted(spin_outputs,end_absdays)-1,0)
            end_absdays = spin_outputs[i]
        else:
            end_absdays = spin_outputs[-1]
            
        end_date = num2date(end_absdays)
        
        print("This run will end on %s to match a spinup output time"%(end_date.strftime('%Y-%m-%d %H:%M:%S')))

        start_date = end_date - datetime.timedelta( self.hydro_spinup_days )

        ## Find a full grid output from the spinup run that can be used to initialize the
        #  transition run
        i = max(searchsorted(spin_outputs,date2num(start_date))-1,0)
        start_absdays = spin_outputs[i]
        start_date = num2date(start_absdays)

        print("This run will start on %s, to match a spinup output time"%(start_date))
        print("Hydro spinup time is %g days. This run will be %g days"%(self.hydro_spinup_days,
                                                                        end_absdays - start_absdays))

        # First, copy the non time-varying files:
        self.copy_to_datadir(new_dir)

        # Doctor up the suntans.dat file to reflect the period we want:
        sunconf = sunreader.SunConfig( os.path.join(new_dir, 'suntans.dat') )
        sunconf.set_simulation_period(start_date,end_date)
        sunconf.write_config()

        # Now setup the initial conditions - for now this is done the same way as
        # for a spinup run -

        self.cmd_replace_forcing(new_dir)
        self.cmd_copy_spin_scalars(spin_dir,new_dir)
        self.log_arguments(new_dir)
        

    def cmd_continue_with_spin(self,trans_dir,spin_dir,new_dir,interval=None):
        """ trans_dir: the full resolution, transition run (for hydro spinup)
            spin_dir: the coarse resolution run that provides an updated scalar field
            new_dir: where the create the combined run.
        """
        if self.private_rundata:
            self.rundata_dir = trans_dir
            
        self.cmd_continue(trans_dir,new_dir,interval,symlink=False)
        self.cmd_copy_spin_scalars(spin_dir,new_dir)
        self.log_arguments(new_dir)

    def cmd_change_dt(self,data_dir,dt):
        sun = sunreader.SunReader(data_dir)
        sun.modify_dt(float(dt))
        self.log_arguments(data_dir)
        
                               
    def cmd_copy_spin_scalars(self,spin_dir,new_dir,unsymlink=True):
        """ Overwrite scalar initialization in new_dir with values interpolated in space,
        nearest in time, from spin_dir.

        if new_dir has StartFiles, those will be used.
        otherwise, initial salinity will be overwritten.

        unsymlink: if the StartFiles are symlinked, copy them into a real file
        same goes for initial salinity
        """
        sun = sunreader.SunReader(new_dir)
        spin_sun = sunreader.SunReader(spin_dir)

        ## Do we have StartFiles?
        start_file_fn = sun.file_path('StartFile',0)
        if os.path.exists(start_file_fn):
            print( "Copy_spin_scalars: StartFile exists.")

            startfile = sunreader.StoreFile(sun,0,startfile=1)
            my_absdays = date2num( startfile.time() )
            startfile.close()

            mapper = sunreader.spin_mapper(spin_sun=spin_sun,
                                           target_sun=sun,
                                           target_absdays=my_absdays,
                                           scalar='salinity')

            for proc in range(self.np):
                if unsymlink:
                    ensure_real_file(sun.file_path('StartFile',proc))

                startfile = sunreader.StoreFile(sun,proc,startfile=1)
                startfile.overwrite_salinity(mapper)
                startfile.close()
        else:
            print("Copy_spin_scalars: no start file - will overwrite salinity initial conditions")
            read_salinity = sun.conf_int('readSalinity')
            if read_salinity != 2:
                print("Modifying suntans.dat: readSalinity  %d => %d"%(read_salinity,2))
                sun.conf.set_value('readSalinity',2)
                sun.conf.write_config()

            # This needs to share code with the copy_salinity functionality.
            # Maybe one function that's not in either class, that takes a spinup run and
            # a target grid and date and returns the interpolated salt field.  then each
            # implementation can write it out however they want.

            mapper = sunreader.spin_mapper(spin_sun=spin_sun,
                                           target_sun=sun,
                                           target_absdays= date2num(sun.time_zero()),
                                           scalar='salinity')

            if unsymlink:
                # remove existing files, and they will be recreated
                for proc in range(sun.num_processors()):
                    fname = sun.file_path("InitSalinityFile",proc)
                    if os.path.exists(fname):
                        print("removing old initial salinity file",fname)
                        os.unlink(fname)
                        
            sun.write_initial_salinity(mapper,dimensions=3,func_by_index=1)
                
        self.log_arguments(new_dir)

    def cmd_make_shps(self,depth_fn=None):
        print("Assume that we're running in the directory above 'rundata'")

        if depth_fn is None:
            depth_fn = 'rundata/depth.dat.untrimmed'

        if self.private_rundata:
            raise Exception("Sorry - make_shps is not currently compatible with private_rundata")

        g = trigrid.TriGrid(suntans_path='rundata/original_grid')

        # just the shoreline:
        g.write_shp('grid-boundaries.shp',overwrite=1)

        if os.path.exists(depth_fn):
            ## And some depth contours
            # Read the untrimmed depth data
            xyz = loadtxt(depth_fn)
            cell_depths=xyz[:,2]

            # Plot a useful set of contours
            g.write_contours_shp('grid-contours.shp',cell_depths,
                                 array([4,5,6,7,8,9,10,12,15,20,25]),
                                 overwrite=1)
        else:
            print("No depth information - skipping contours")

    def cmd_make_subdomain_shps(self,datadir):
        """ Write boundary shapefiles for each of the subdomains in the given
        data directory.
        """
        sun = sunreader.SunReader(datadir)

        Nprocs = sun.num_processors()
        for proc in range(Nprocs):
            print("Subdomain %d/%d\n"%(proc,Nprocs))

            g = sun.grid(proc)
            g.write_shp(os.path.join(datadir,'grid-boundaries%03i.shp'%proc),overwrite=1)


    def cmd_plot_datasources(self,datadir):
        """ Kind of like plot_forcing, but reads each of the datasource files and makes a plot, hoping that
        the first line is a useful comment
        """
        plotdir = os.path.join(datadir,'plots_datasources')

        if not os.path.exists(plotdir):
            os.mkdir( plotdir )
        else:
            for f in os.listdir(plotdir):
                os.unlink( os.path.join(plotdir,f) )

        sun = sunreader.SunReader(datadir)
        start_date,end_date = sun.simulation_period()

        for ds_fn in glob.glob(os.path.join(datadir,'datasources','*')):
            print("Reading datasource %s"%ds_fn)

            ds = forcing.read_datasource(ds_fn,sun)

            clf()
            ds.plot_overview(start_date,end_date)
            savefig( os.path.join(plotdir,os.path.basename(ds_fn)+".png") )
        
    def cmd_plot_forcing(self,datadir):
        """ Make plots of all the forcing variables for the run in datadir.
        This will reload the data, rather than reading from boundaries.dat.*
        """

        plotdir = os.path.join(datadir,'forcing')

        if not os.path.exists(plotdir):
            os.mkdir( plotdir )
        else:
            for f in os.listdir(plotdir):
                os.unlink( os.path.join(plotdir,f) )

        sun = sunreader.SunReader(datadir)
        start_date,end_date = sun.simulation_period()

        gforce = self.global_forcing( datadir=datadir )

        i = 0
        for d in gforce.datasources:
            clf()
            d.plot_overview(tmin=start_date,tmax=end_date)
            savefig(os.path.join(plotdir,"datasource%04i.pdf"%i))
            i+=1

    ##### Scalar releases
    def scalar_release_regions(self):
        """ Return closed polygons (assumed closed) representing the release regions.
        Used for defining how cell scalar value are set during the release (and
        potentially during grid generation)

        The linestrings here should *not* repeat the last vertex - they are assumed
        closed.
        """
        return []
    
        
    # update start.dat with a scalar release:
    #  before calling this you'd have to do a continue, then this command
    #  will be run in the new run.
    def process_scalar_release(self,sun,mapper,unsymlink=True,species='temperature'):
        """ sun: sunreader instance
            mapper: function of the form f(proc,cell,k) which returns concentration for the given cell
            unsymlink: if symlinked start files should be copied, rather than editing the original store files.
            species: 'temperature', 'salinity' for regular scalars
              'sediNNN' for suspended sediment concentration in given sediment species
              'bedNNN' for bed mass in given layer
        """
        # branch off if this isn't for a regular scalar quantity.
        if species.find('sedi') == 0:
            species = int(species[ len('sedi'):])
            self.process_ssc_release(sun,mapper,unsymlink,species)
            return
        elif species.find('bed') == 0:
            NL = int(species[ len('bed'):])
            self.process_bed_release(sun,mapper,unsymlink,NL)
            return
            
        start_file_fn = sun.file_path('StartFile',0)
        if not os.path.exists(start_file_fn):
            raise Exception("scalar_release:: failed to find StartFile %s."%start_file_fn)
        
        for proc in range(self.np):
            if unsymlink:
                ensure_real_file(sun.file_path('StartFile',proc))

            startfile = sunreader.StoreFile(sun,proc,startfile=1)
            if species=='temperature':
                startfile.overwrite_temperature(mapper)
            elif species=='salinity':
                startfile.overwrite_salinity(mapper)
            else:
                raise Exception("unknown species %s"%species)
            startfile.close()
            print("Overwrote %s"%species)

    def process_bed_release(self,sun,mapper,unsymlink,layer):
        """ see process_scalar_release.
        even though the layer is specified, the mapper must still accept a k argument, which
        will be the layer"""
        print("Will write over sediment bed layer %d"%layer)
        
        for proc in range(self.np):
            start_file_fn = sunreader.SediStoreFile.filename(sun,proc,startfile=1)
            if unsymlink:
                ensure_real_file(start_file_fn)
                
            sedi_startfile = sunreader.SediStoreFile(sun,proc,startfile=1)
            sedi_startfile.overwrite_bed(mapper,layer=layer)
            sedi_startfile.close()
        
    def process_ssc_release(self,sun,mapper,unsymlink,species):
        """ see process_scalar_release """
        print("Will write over suspended sediment concentration for species %d"%species)
        
        for proc in range(self.np):
            start_file_fn = sunreader.SediStoreFile.filename(sun,proc,startfile=1)
            if unsymlink:
                ensure_real_file(start_file_fn)
                
            sedi_startfile = sunreader.SediStoreFile(sun,proc,startfile=1)
            sedi_startfile.overwrite_ssc(mapper,species=species)
            sedi_startfile.close()


    def cmd_scalar_release_polygon(self,datadir,n=0,unsymlink=True,species='temperature'):
        """ Defaults to the first release region
        datadir: the restart to operate on (currently does not support release with initial conditions)
        n: id of the scalar release region
        unsymlink: should symlinked start.dat files be copied first
        """
        n = int(n)
        # cast to float because shapely is not so smart about that...
        release_region = array( self.scalar_release_regions()[n],float64 )

        # make it into an easily queryable polygon:
        poly = geometry.Polygon(release_region)

        sun = sunreader.SunReader(datadir)

        print("scalar poly is",release_region)
        print(" with area ",poly.area)
        
        def mapper(proc,cell,k):
            g = sun.grid(proc)
            pnt = geometry.Point( g.vcenters()[cell])
            if poly.contains(pnt):
                print("wrote some temp")
                return 1.0
            else:
                return 0.0
        return self.process_scalar_release(sun,mapper=mapper,unsymlink=unsymlink,species=species)
        
    def cmd_scalar_release_gaussian_2D(self,datadir,xy,sd,unsymlink=True,species='temperature'):
        """ create a 2D gaussian plume, centered at the given x,y point (which should be given
        as either a [x,y] list, or a string <x>,<y>.
        sd is the standard deviation [sd_x,sd_y], and may be infinite in one dimension.
        if a third value is included for sd, it will be used as the peak value of the distribution,
        otherwise 1.0 is assumed.
        """
        if type(xy) == str:
            xy = [float(s) for s in xy.split(',')]
        if type(sd) == str:
            sd = [float(s) for s in sd.split(',')]

        if len(sd) == 3:
            peak = sd[2]
        else:
            peak = 1.0
            
        sun = sunreader.SunReader(datadir)

        def mapper(proc,cell,k):
            g = sun.grid(proc)
            pnt = g.vcenters()[cell]
            return peak * exp( -(pnt[0]-xy[0])**2 / (2*sd[0]**2) - (pnt[1]-xy[1])**2 / (2*sd[1]**2) )
        return self.process_scalar_release(sun,mapper,unsymlink,species)
        
    def cmd_batch_continue(self,base_dir,batch_dir,interval):
        """ base_dir  - a completed run, from which this one will continue
            batch_dir - the name for the new runs, including trailing two digit
               id for the first of the continuations.
            interval - how long each continuation should be.  probably for ponds this
               should be +2d
        """
        if not self.run_can_be_continued(base_dir):
            print("Sorry - the base run does not appear to be complete.")
            return 

        import re
        m = re.match('(.*[^0-9])([0-9]+)',batch_dir)
        prefix = m.group(1)
        ndigits = len(m.group(2))
        starting_index = int(m.group(2))

        index = starting_index
        prev_dir = base_dir
        
        while 1:
            cont_dir = prefix + "%0*i"%(ndigits,index)
            print("Checking status of %s"%cont_dir)

            if not self.run_has_completed(cont_dir):
                print("Preparing %s for run"%cont_dir)
                self.cmd_continue(prev_dir,cont_dir,interval)
                print("Running")
                self.cmd_run(cont_dir)

                if not self.run_has_completed(cont_dir):
                    print("That run appears not to have finished.  Bailing out")
                    break
            else:
                print("Run %s appears to have already run"%cont_dir)
                
            index += 1
            prev_dir = cont_dir
            
    def cmd_batch(self,datadir,interval):
        """  A combination of initialize_run and batch_continue -
        datadir is taken as the base name, and will have a five digit index appended
        """
        initial_dir = "%s00000"%datadir
        continue_dir = "%s00001"%datadir

        
        self.cmd_initialize_run(initial_dir,interval)
        self.cmd_run(initial_dir)

        sun = sunreader.SunReader(initial_dir)
        dstart,dend = sun.simulation_period()
        interval_days = date2num(dend) - date2num(dstart)
        interval_str = "+%fd"%interval_days

        self.cmd_batch_continue(initial_dir,continue_dir,interval_str)
        
    def cmd_resymlink(self,*datadirs):
        """ for each of the given datadirs, see if there are chained runs
        where constant files can be turned back into symlinks.  tries to
        symlink back to the original file, not just the previous run
        """
        for datadir in datadirs:
            print("Checking %s"%datadir)
            sun = sunreader.SunReader(datadir)
            chain = sun.chain_restarts()
            if len(chain) <= 1:
                print("No chain.")
                continue
            
