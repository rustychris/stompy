from __future__ import print_function
from __future__ import division

# These were added by 2to3, but seem to cause more issues than they solve
# from builtins import zip
# from builtins import str
# from builtins import map
# from builtins import range
# from builtins import object
# from past.utils import old_div

import re
import pprint
import sys, os, shutil, time

import hashlib, pickle

try:
    try:
        from osgeo import osr
    except:
        import osr
except ImportError:
    print("GDAL unavailable")
    osr = "unavaiable"
    
from numpy import ma
import glob,datetime
import subprocess

from . import transect

from ...grid import (trigrid,orthomaker)
from . import forcing
from ...spatial import field

import mmap
from numpy import *

from ...spatial.linestring_utils import upsample_linearring

try:
    import pytz
    utc = pytz.timezone('utc')
except ImportError:
    print("couldn't load utc timezone")
    utc = None
    
try:
    from safe_pylab import *
    import safe_pylab as pylab
except:
    print("Plotting disabled")
    
from numpy.random import random

# configurable, per-user or host settings:
import local_config

# use absolute paths since we need to change directories to invoke vdatum
REALSIZE = 8
REALTYPE = float64

def read_bathymetry_offset():
    ## Figure out the constant offset in the bathymetry from NAVD88
    
    import sys
    comp_path = os.path.join( os.environ['HOME'], "classes/research/spatialdata/us/ca/suntans/bathymetry/compiled")
    if comp_path not in sys.path:
        sys.path.append(comp_path)

    import depth_offset
    bathymetry_offset = depth_offset.navd88_highwater

    #print "Bathymetry has constant offset of %gm"%bathymetry_offset
    return bathymetry_offset


def msl_to_navd88(lonlat_locs):
    if msl_to_navd88.warning:
        print(msl_to_navd88.warning)
        msl_to_navd88.warning = None
    return 0.938*ones( (lonlat_locs.shape[0],) )
    # return apply_vdatum('LMSL','MLLW',lonlat_locs)

def apply_vdatum(src_vdatum,dest_vdatum,lonlat_locs):
    """ given a vector of lon/lat yx pairs, return the height that must be added to MSL to get
    a NAVD88 measurement
    """
    
        
    tmp_in = "/tmp/tmpinput"
    tmp_out = "/tmp/tmpoutput"

    in_fp = open(tmp_in,'wt')

    for i in range(lonlat_locs.shape[0]):
        in_fp.write("%d, %f, %f, 0.0\n"%(i,lonlat_locs[i,0],lonlat_locs[i,1]))
    in_fp.close()
    vdatum_dir = local_config.vdatum_dir

    command = "cd %s ; java VDatum -lonlat -heights -hin WGS84 -units meters -vin %s -vout %s %s %s"%(
        vdatum_dir,
        src_vdatum, dest_vdatum,
        tmp_in,tmp_out)

    # print "About to run vdatum command:"
    # print command
    res = subprocess.call(command,shell=True)
    if res:
        print("Something probably went wrong there.  Returned ",res)
        raise Exception("Subcommand vdatum failed")

    offsets = zeros( (lonlat_locs.shape[0]), float64)

    fp_out = open(tmp_out,'rt')

    for i in range(lonlat_locs.shape[0]):
        stupid_id,x,y,z = [float(s) for s in fp_out.readline().split(',')]
        offsets[i] = z
    fp_out.close()

    return offsets

msl_to_navd88.warning = "WARNING: assuming constant 0.938m MSL-NAVD88"

import subprocess
import socket

class MPIrunner(object):
    """ Try to figure out if we're running on mpich1 or mpich2
    """
    def __init__(self,cmdlist,np=4,no_mpi_for_1=True,wait=1):
        """ np: number of processors
            no_mpi_for_for_1: if np==1 and this is true, run the command
              directly without any mpi

            Now tries to figure out if we're on bobstar, in which case submit
            jobs via qsub
        """
        mpi_version = None

        if np == 1 and no_mpi_for_1:
            mpi_version = 0

        host = socket.gethostname()
            
        # bobstar? (or other qsub based machine)
        if mpi_version is None:
            if host == 'head0.localdomain':
                print("good luck - we're running on bobstar")
                mpi_version = 'bobstar'
            elif host.startswith('eddy.'):
                print("looks like we're on eddy")
                mpi_version = 'eddy'
            elif host == 'sunfish':
                print("Hi sunfish!")
                mpi_version='sunfish'
            elif host.startswith('trestles'):
                print("hi trestles.")
                # seems that trestles can run qsub from nodes
                # now, so no need to test whether we're in
                # a qsub job. (but this does not handle the
                # case where the script is running inside
                # a qsub job that already has the right np,
                # and really just want to run mpirun_rsh right away..)
                mpi_version = 'trestles'

        # running locally on mac:
        if mpi_version is None and sys.platform == 'darwin':
            mpi_version = 'darwin'


        # look for some direct ways to start it:
        if mpi_version is None:
            print("Trying versions of mpich")
            for cmd,version in [('mpich2version',2),
                                ('mpich2version',1),
                                ('mpichversion.mpich-shmem',1)]:
                try:
                    pipe = subprocess.Popen(cmd,stdout=subprocess.PIPE)
                    version_info = pipe.communicate()[0]
                    mpi_version = version
                    break
                except OSError as exc:
                    #print "While trying to figure out the mpich version, the command %s"%cmd
                    #print "raised an exception"
                    #print exc
                    pass

        # might be open mpi - 
        if mpi_version is None:
            print("Trying Open MPI")
            try:
                pipe = subprocess.Popen(['mpirun','-V'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                stdout_txt,stderr_txt = pipe.communicate()
                if b"Open MPI" in stderr_txt + stdout_txt:
                    print("Found mpi is openmpi")
                    mpi_version="openmpi"
                else:
                    print("OpenMPI mpirun printed something else")
                    print(stderr_txt)
                    print(stdout_txt)
            except OSError as exc:
                print("While checking for Open MPI, the command %s"%cmd)
                print("raised an exception")
                print(exc)
                
        if mpi_version is None:
            raise Exception("MPI wasn't found")
            
        print("Using mpi version %s"%mpi_version)
        print("  PWD: ",os.getcwd())
        print("  np: ",np)
        print("  command: "," ".join(cmdlist))
    
        def must_wait():
            if not wait:
                print("Sorry - have to wait on non-qsub platforms")

        if mpi_version == 0:
            must_wait()
            subprocess.call(cmdlist)
        elif mpi_version == 1:
            must_wait()
            if host=='marvin':
                mpirun = "mpirun.mpich-shmem"
            else:
                mpirun = "mpirun"
            subprocess.call([mpirun,'-np','%d'%np] + cmdlist)
        elif mpi_version == 2:
            must_wait()
            subprocess.call(['mpiexec','-np','%d'%np] + cmdlist)
        elif mpi_version == 'darwin':
            must_wait()
            subprocess.call(['mpiexec','-np','%d'%np,'-host','localhost'] + cmdlist)
        elif mpi_version =='bobstar':
            self.run_bobstar(np,cmdlist,wait=wait)
        elif mpi_version == 'eddy':
            self.run_eddy(np,cmdlist,wait=wait)
        elif mpi_version in ['trestles']:
            self.run_trestles(np,cmdlist,wait=wait,mpi_version=mpi_version)
        elif mpi_version=='sunfish':
            self.run_sunfish(np,cmdlist,wait=wait)
        elif mpi_version=='openmpi':
            must_wait()
            print("Running via openmpi")
            subprocess.call(['mpirun','-n','%d'%np] + cmdlist)
        else:
            raise Exception("Failed to find a way to run this! mpi version is %s"%mpi_version)


    def run_bobstar(self,np,cmdlist,wait=True):
        self.run_qsub(np,cmdlist,
                      ppn=8,walltime=None,
                      wait=wait)
    def run_sunfish(self,np,cmdlist,wait=True):
        self.run_qsub(np,cmdlist,ppn=8,walltime="24:00:00",wait=wait,
                      mpi_template="mpirun -np %(np)i",)
    def run_eddy(self,np,cmdlist,wait=True):
        if 0: # the more standard but old way on eddy:
            # eddy policy is 24 hour jobs
            self.run_qsub(np,cmdlist,
                          ppn=8,walltime='24:00:00',
                          mpi_template='/usr/bin/mpirun -np %(np)i',
                          wait=wait)
        else:
            self.run_qsub(np,cmdlist,
                          ppn=8,walltime='24:00:00',
                          mpi_template='/home/rusty/bin/mpiexec --comm pmi -n %(np)i',
                          wait=wait)
            
        
    def run_trestles(self,np,cmdlist,wait=True,mpi_version='trestles'):
        """ trestles has 32 cores per node, but they also have a shared queue
        which allows for using a smaller number of cores (and getting charged
        for the smaller number).  it defaults to 48 hours, but I think that
        it charges for all of walltime, even if the job finishes early (annoying).
        for testing, force walltime = 1.0
        """
        if np > 1:
            mpi_template="mpirun_rsh -np %(np)i -hostfile $PBS_NODEFILE"
        else:
            # sometimes run serial jobs via qsub to have more resources
            mpi_template=""

            
        if np < 32:
            # use one node from the shared queue
            q = 'shared'
            ppn = np
        else:
            q = 'normal'
            ppn = 32

        # just guess if not given...
        walltime=os.environ.get('WALLTIME','1:00:00')
            
        self.run_qsub(np,cmdlist,
                      ppn=ppn,walltime=walltime,save_script=False,
                      queue=q,
                      mpi_template=mpi_template, # "mpirun_rsh -np %(np)i -hostfile $PBS_NODEFILE",
                      wait=wait)
            
    def run_qsub(self,np,cmdlist,ppn=8,walltime=None,save_script=False,
                 queue=None,
                 mpi_template="mpirun -np %(np)i -machine vapi",
                 wait=True):
        """ Simple wrapper for qsub -
        note: some systems have a very small default walltime but allow for
        arbitrarily large walltime, while others have a generous default but
        don't allow anything bigger.

        save_script: if true, write the script out to qsub_script.sh, and exit.
        wait: if False queue and it return, otherwise watch qstat for it to finish.
        """
        print("submitting job: walltime will be: ",walltime)
        
        # first figure out how many nodes:
        nnodes = int( (np-1) / ppn + 1 )
        if walltime is None:
            walltime = ""
        else:
            walltime = ",walltime=%s"%walltime


        if queue is not None:
            qtxt ="#PBS -q %s\n"%queue
        else:
            qtxt ="# no queue specified"

        mpi_command = mpi_template%locals()
        
        script =  """#!/bin/bash
%s
#PBS -N rusty
#PBS -l nodes=%i:ppn=%i%s
#PBS -V

DISPLAY=""
export DISPLAY

cd %s
echo Run started at `date`
time %s %s
echo Run ended at `date`
"""%(qtxt,nnodes,ppn,walltime,os.getcwd(),mpi_command," ".join(cmdlist))

        if save_script:
            fp = open("qsub_script.sh","wt")
            fp.write(script)
            fp.close()
            sys.exit(1)

        print("------Script------")
        print(script)
        print("------------------")

        # it would be nice to run it interactively and just wait, but
        # qsub says we have to be on a terminal to do that (and I think
        # that interactive jobs also don't run the script).
        # so capture the job number from qsub:
        # 9552.scyld.localdomain
        proc = subprocess.Popen(['qsub'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        out,err = proc.communicate( script )
        job_id = out.strip()
        print("qsub returned with code %s. job id is '%s'"%(proc.returncode,job_id))

        if job_id=="":
            print("Didn't get a job id back from qsub.")
            print("stdout was:")
            print(out)
            print("stderr was:")
            print(err)
            return

        if not wait:
            print("Not waiting.  job id is %s"%job_id)
            return
        
        # poll the job via qstat

        # increase sleep time as we go
        sleep_time = 5.0
        sleep_r = 1.2
        sleep_time_max = 600

        last_status = None
        
        while 1:
            proc = subprocess.Popen(['qstat',job_id],stdout=subprocess.PIPE)
            out,err = proc.communicate()
            print("qstat returned: ",out)
            lines = out.split("\n")
            if len(lines)>=3:
                # print "Job status line is ",lines[2]
                stat_line = lines[2]
                status = stat_line.split()[4]
                print('status is ',status)
            else:
                print("Assuming that no response means that it's finished")
                status = 'C'

            if status == 'C':
                print("Completed!")
                break
            else:
                if last_status == status:
                    sleep_time = min( sleep_r * sleep_time, sleep_time_max)
                else:
                    sleep_time = 5.0
                last_status = status
                
                print("sleeping %f seconds"%sleep_time)
                time.sleep( sleep_time )

class SuntansCrash(object):
    horizontal_courant = 1
    vertical_courant   = 2
    
    def site(self):
        g = self.sun.grid(processor = self.processor)
        if self.description == SuntansCrash.vertical_courant:
            return g.vcenters()[self.cell_id]
        else:
            endpoints = g.points[g.edges[self.edge_id,:2],:2]
            return endpoints.mean(axis=0) # compute midpoint
        
    def plot(self):
        # plot the location of the cell on top of the
        # bathymetry, and with a second plot showing the
        # stage of the nearest profile point

        # plotting a crash site
        self.sun.plot_bathymetry(procs = [self.processor] )

        site = self.site()
        plot( [site[0]],[site[1]],'ro' )
        annotate('Offender',site)

    def closest_profile_point(self):
        site = self.site()

        pnts = self.sun.profile_points()
        
        dists = sqrt( (pnts[:,0]-site[0])**2 + (pnts[:,1]-site[1])**2 )
        best = argmin(dists)
        print("Closest profile point is %g away"%dists[best])

        self.closest_profile_location = pnts[best]
        self.closest_profile_index = best
        return best

    def plot_nearest_freesurface(self):
        prof_point = self.closest_profile_point()
        prof_data = self.sun.profile_data('FreeSurfaceFile')
        point_data = prof_data[:,prof_point]
        prof_time = self.sun.timeline(units='days',output='profile')
        plot(prof_time,point_data)

    def full_plot(self):
        
        clf()
        subplot(212)
        self.plot_nearest_freesurface()
        
        subplot(211)
        self.plot_vicinity()
        
    def plot_vicinity(self):
        self.plot()
        
        self.closest_profile_point()

        p = self.closest_profile_location
        plot( [p[0]],[p[1]],'bo')
        annotate('Profile loc',p)


    def depth_info(self):
        
        if self.description == self.horizontal_courant:
            print("Horizontal courant number violation")
            print("edge_id: ",self.edge_id)
            g = self.sun.grid(processor = self.processor)

            cells = g.edges[self.edge_id,3:5]
        elif self.description == self.vertical_courant:
            print("Vertical courant number violation")
            print("cell_id: ",self.cell_id)
            print("dz: ",self.dz)
            cells = [self.cell_id]
        else:
            print("Unknown crash type ",self.description)

        print("Z-level of crash: ",self.z_id)

        fs = self.storefile().freesurface()
        
        for c in cells:
            print("Cell %i"%c)
            print("Depth[cell=%i]: %g"%(c,self.sun.depth_for_cell(c,self.processor)))
            if self.z_id == 0:
                print("Top of this z-level: 0.0")
            else:
                print("Top of this z_level: %g"%(self.sun.z_levels()[self.z_id-1]))
            print("Bottom of this z_level: %g"%(self.sun.z_levels()[self.z_id]))
            print("Freesurface[cell=%i]: %g"%(c,fs[c]))

    _storefile = None
    def storefile(self):
        if self._storefile is None:
            self._storefile = StoreFile(self.sun, self.processor)
        return self._storefile


def spin_mapper(spin_sun,target_sun,target_absdays,scalar='salinity',nonnegative=1):
    """ return a function that takes a proc, cell and k-level from target_sun
    and returns a scalar value from the spin_sun.
    the timestep will be chosen to be as close as possible to the starting
    time of target_sun
    """
    ### figure out the step to use:
    # the time of this storefile:
    
    spin_times = date2num( spin_sun.time_zero()) + spin_sun.timeline(units='days')

    step = argmin( abs(spin_times-target_absdays) )

    print("Difference between target time and spinup time: %gs"%( 24*3600*(target_absdays - spin_times[step]) ))

    ### map target z-levels to the spin_sun z-levels
    z_levels = target_sun.z_levels()
    spin_z_levels = spin_sun.z_levels()

    # could be refined some, to use middles of the cells
    k_to_spin_k = searchsorted(spin_z_levels,z_levels)

    ### as needed, figure out how our cells map onto the
    #   processors and cells of the spin_run.  Since we
    #   may not need all of the spin_sun's domains, 
    #   do this dynamically
    proc_cell_to_spin_cell = {} # (target_proc,target_cell) -> (spin_proc,spin_cell_id)
    cell_salinity = {} # spin proc -> salt[spin_c,spin_k]

    grids = [None] * target_sun.num_processors()
    spin_ctops = [None] * spin_sun.num_processors()
    
    def mapper(p,c,k):
        # populate per processor things as needed
        if grids[p] is None:
            grids[p] = target_sun.grid(p)
        
        # first figure out the spin-cell
        if (p,c) not in proc_cell_to_spin_cell:
            # The real way:
            proc_cell_to_spin_cell[(p,c)] = spin_sun.closest_cell(grids[p].vcenters()[c])
    
        spin_proc,spin_c = proc_cell_to_spin_cell[(p,c)]

        # map my k onto the spinup-k:
        try:
            spin_k = k_to_spin_k[k]
        except:
            print("Ran into trouble while trying to map p=%d c=%d k=%d"%(p,c,k))
            print("through the map ")
            print(k_to_spin_k)
            raise

        # get the ctop values for the spinup run:
        if spin_ctops[spin_proc] is None:
            spin_ctops[spin_proc] = spin_sun.ctop(spin_proc,step)
            # print "Read ctops for spinup: spin_proc=%d  length=%s\n"%(spin_proc,spin_ctops[spin_proc].shape)

        # how deep is that cell?
        spin_nk = spin_sun.Nk(spin_proc)[spin_c]

        # if the spinup cell is dry, pull from the highest wet cell
        try:
            if spin_k < spin_ctops[spin_proc][spin_c]:
                spin_k = spin_ctops[spin_proc][spin_c]
        except IndexError:
            print("Trouble with production run (%d,%d,%d) mapping to spinup (%d,%d,??) shaped %s"%(p,c,k,spin_proc,spin_c,spin_ctops[spin_proc].shape))
            raise
        
        # if it's too deep, truncate
        if spin_k >= spin_nk:
            spin_k = spin_nk - 1

        if spin_proc not in cell_salinity:
            g,cell_salinity[spin_proc] = spin_sun.cell_salinity(spin_proc,step)
        s = cell_salinity[spin_proc][spin_c,spin_k]

        if nonnegative and s<0.0:
            s=0.0
        return s 
    return mapper


class SediStoreFile(object):
    """ Like StoreFile, but for reading/writing sediment restart data
    """
    hdr_bytes=3*4
    
    @staticmethod
    def filename(sun,processor,startfile=0):
        # this is currently hardcoded in sedi.c and phys.c - so we have to hardcode here.
        if startfile:
            return os.path.join(sun.datadir,'start_sedi.dat.%d'%processor)
        else:
            return os.path.join(sun.datadir,'store_sedi.dat.%d'%processor)

    def __init__(self,sun,processor,startfile=0):
        self.sun = sun
        self.proc = processor
        self.grid = sun.grid(processor)

        self.fn = self.filename(sun,processor,startfile)
        self.fp = open(self.fn,'rb+')

        # pre-compute strides
        self.stride_Nc = self.grid.Ncells()
        self.stride_Nc_Nk = self.sun.Nk(self.proc).sum()

        # read the header info:
        header = fromstring(self.fp.read(self.hdr_bytes),int32)
        self.timestep = header[0]
        self.nclasses = header[1]
        self.nlayers = header[2]

    def seek_ssc(self,species):
        self.fp.seek(self.hdr_bytes + species* self.stride_Nc_Nk * REALSIZE)
    def seek_bed(self,layer):
        self.fp.seek(self.hdr_bytes + self.nclasses*self.stride_Nc_Nk * REALSIZE \
                     + layer * self.stride_Nc)
        
    def read_ssc(self,species):
        self.seek_ssc(species)
        return fromstring( self.fp.read(self.stride_Nc_Nk * REALSIZE), REALTYPE)
    def read_bed(self,layer):
        self.seek_bed(layer)
        return fromstring( self.fp.read(self.stride_Nc * REALSIZE), REALTYPE)
    def write_ssc(self,species,ssc):
        data = ssc.astype(REALTYPE)
        self.seek_ssc(species)
        self.fp.write( data.tostring() )
        self.fp.flush()
    def write_bed(self,layer,bed):
        data = bed.astype(REALTYPE)
        self.seek_bed(layer)
        self.fp.write( data.tostring() )
        self.fp.flush()
        
    def overwrite_ssc(self,func,species):
        """ iterate over all the cells and set the ssc
        in each one, overwriting the existing ssc data

        signature for the function func(cell id, k-level)
        """
        # read the current data:
        ssc = self.read_ssc(species)

        i = 0 # linear index into data
        Nk = self.sun.Nk(self.proc)

        for c in range(self.grid.Ncells()):
            for k in range(Nk[c]):
                ssc[i] = func(self.proc,c,k)
                i+=1

        self.write_ssc(species,ssc)
        
    def overwrite_bed(self,func,layer):
        bed = self.read_bed(layer)

        for c in range(self.grid.Ncells()):
            bed[c] = func(self.proc,c,layer)

        self.write_bed(layer,bed)
        
    def close(self):
        self.fp.close()
        self.fp = None
        

class StoreFile(object):
    """ Encapsulates reading of store.dat files, either for restarts or
    for crashes

    New: support for overwriting portions of a storefile
    """

    def __init__(self,sun,processor,startfile=0,filename=None):
        """ startfile: if true, choose filename using StartFile 
        instead of StoreFile
        """
        self.sun = sun
        self.proc = processor

        if filename is None:
            if startfile:
                self.fn = self.sun.file_path('StartFile',self.proc)
            else:
                self.fn = self.sun.file_path('StoreFile',self.proc)
        else:
            self.fn=filename   

        self.fp = open(self.fn,'rb+')

        # lazy loading of the strides, in case we just want the
        # timestep
        self.blocks_initialized = False
        
    def initialize_blocks(self):
        if not self.blocks_initialized:
            # all data is lazy-loaded
            self.grid = self.sun.grid(self.proc)

            # pre-compute strides
            Nc = self.grid.Ncells()
            Ne_Nke = self.sun.Nke(self.proc).sum()
            Nc_Nk = self.sun.Nk(self.proc).sum()
            Nc_Nkp1 = (self.sun.Nk(self.proc) + 1).sum()

            # and define the structure of the file:
            blocks = [
                ['timestep', int32, 1],
                ['freesurface', REALTYPE, Nc],
                ['ab_hor_moment', REALTYPE, Ne_Nke],
                ['ab_vert_moment', REALTYPE, Nc_Nk],
                ['ab_salinity', REALTYPE, Nc_Nk],
                ['ab_temperature', REALTYPE, Nc_Nk],
                ['ab_turb_q', REALTYPE, Nc_Nk],
                ['ab_turb_l', REALTYPE, Nc_Nk],
                ['turb_q', REALTYPE, Nc_Nk],
                ['turb_l', REALTYPE, Nc_Nk],
                ['nu_t', REALTYPE, Nc_Nk],
                ['K_t', REALTYPE, Nc_Nk],
                ['u', REALTYPE, Ne_Nke],
                ['w', REALTYPE, Nc_Nkp1],
                ['p_nonhydro', REALTYPE, Nc_Nk],
                ['salinity', REALTYPE, Nc_Nk],
                ['temperature', REALTYPE, Nc_Nk],
                ['bg_salinity', REALTYPE, Nc_Nk]]


            # and then rearrange to get block offsets and sizes ready for reading
            block_names = [b[0] for b in blocks]
            block_sizes = array( [ ones(1,b[1]).itemsize * b[2] for b in blocks] )
            block_offsets = block_sizes.cumsum() - block_sizes

            expected_filesize = block_sizes.sum()
            actual_filesize = os.stat(self.fn).st_size

            if expected_filesize != actual_filesize:
                raise Exception("Mismatch in filesize: %s != %s"%(expected_filesize, actual_filesize))

            self.block_names = block_names
            self.block_sizes = block_sizes
            self.block_offsets = block_offsets
        self.block_types = [b[1] for b in blocks]
        
        self.blocks_initialized = True

    def close(self):
        self.fp.close()
        self.fp = None
        
    def read_block(self,label):
        # special handling for timestep - can skip having to initialized
        # too much
        if label == 'timestep':
            self.fp.seek(0)
            s = self.fp.read( 4 )
            return fromstring( s, int32 )
        else:
            # 2014/7/13: this line was missing - not sure how it ever
            # worked - or if this is incomplete code??
            self.initialize_blocks()
            i = self.block_names.index(label)
            self.fp.seek( self.block_offsets[i] )
            s = self.fp.read( self.block_sizes[i] )
            return fromstring( s, self.block_types[i] )

    def write_block(self,label,data):
        i = self.block_names.index(label)
        self.fp.seek( self.block_offsets[i] )
        data = data.astype(self.block_types[i])
        self.fp.write( data.tostring() )
        self.fp.flush()
        
    def timestep(self):
        return self.read_block('timestep')[0]

    def time(self):
        """ return a datetime corresponding to our timestep """
        return self.sun.time_zero() + datetime.timedelta( self.timestep() * self.sun.conf_float('dt')/(24.*3600) )

    def freesurface(self):
        return self.read_block('freesurface')

    def u(self):
        blk = self.read_block('u')
        Nke = self.sun.Nke(self.proc)
        Nke_cumul = Nke.cumsum() - Nke
        
        full_u = nan*ones( (self.grid.Nedges(),Nke.max()) )

        for e in range(self.grid.Nedges()):
            full_u[e,0:Nke[e]] = blk[Nke_cumul[e]:Nke_cumul[e]+Nke[e]]
            
        return full_u
            
    def overwrite_salinity(self,func):
        """ iterate over all the cells and set the salinity
        in each one, overwriting the existing salinity data

        signature for the function func(cell id, k-level)
        """
        # read the current data:
        salt = self.read_block('salinity')

        # for starters, just call out to func once for each
        # cell, but pass it the cell id so that func can
        # cache locations for each grid cell.

        i = 0 # linear index into data
        Nk = self.sun.Nk(self.proc)

        for c in range(self.grid.Ncells()):
            for k in range(Nk[c]):
                salt[i] = func(self.proc,c,k)
                i+=1

        self.write_block('salinity',salt)

    def overwrite_temperature(self,func):
        """ iterate over all the cells and set the temperature
        in each one, overwriting the existing salinity data

        signature for the function func(proc, cell id, k-level)
        """
        # read the current data:
        temp = self.read_block('temperature')

        # for starters, just call out to func once for each
        # cell, but pass it the cell id so that func can
        # cache locations for each grid cell.

        i = 0 # linear index into data
        Nk = self.sun.Nk(self.proc)

        for c in range(self.grid.Ncells()):
            for k in range(Nk[c]):
                temp[i] = func(self.proc,c,k)
                i+=1

        self.write_block('temperature',temp)

    def copy_salinity(self,spin_sun):
        """ Overwrite the salinity record with a salinity record
        taken from the spin_sun run.  Step selects which full-grid
        output to use, or if not specified choose the step
        closest to the time of this storefile.
        """
        ### figure out the step to use:
        # the time of this storefile:
        my_absdays = date2num( sun.time() ) 

        mapper = spin_mapper(spin_sun,
                             self.sun,
                             my_absdays,scalar='salinity')
        
        self.overwrite_salinity(mapper)

    # the rest of the fields still need to be implemented, but the pieces are mostly
    # here.  see SunReader.read_cell_z_level_scalar() for a good start, and maybe
    # there is a nice way to refactor this and that.
    



class GenericConfig(object):
    """ Handles reading and writing of suntans.dat formatted files.
    """
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
                key = m.group(2).lower()
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
        if isinstance(value,float) or isinstance(value,floating):
            value = "%.12g"%value
        else:
            value = str(value)
        return value

    def set_value(self,key,value):
        """ Update a value in the configuration.  Setting an item to None will
        comment out the line if it already exists, and omit the line if it does
        not yet exist.
        """
        key = key.lower()
        
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
        

class SediLsConfig(GenericConfig):
    def __init__(self,filename=None,text=None):
        if filename is None and text is None:
            text = """
########################################################################
#
#  Default input file for the L. Sanford based sediment module in SUNTANS.
# see LS2008 for variable meanings
########################################################################
r_consolidate     3.4722e-5   # 3.0  / 86400. # [1/s]
r_swell           3.4722e-7   # 3.0 / 86400 / 100;  # [1/s]
beta              1.36e-4     # [m/s/Pa]
gamma0            0.002       # [-]
rho_solid         2650        # dry weight [kg/m3]
tau_cs            0.125       # [Pa] critical stress of erosion for sand
tau_c_min         0.02        # [Pa] - minimum floc critical stress for erosion
phi_mud_solid_min 0.03        # mud solids fraction [-]

Nspecies		2	# Number of sediment species

# Settling velocities - positive is rising
ws000             -0.0005
ws001             -0.01
flags000           SED_MUD
flags001           SED_SAND

bed_fraction000   0.9
bed_fraction001   0.1



Nlayers                26
dm000                 0.05 # nominal mass in each layer [kg/m2]
dm001                 0.05
dm002                 0.05
dm003                 0.05
dm004                 0.05
dm005                 0.05
dm006                 0.05
dm007                 0.05
dm008                 0.05
dm009                 0.05
dm010                 0.05
dm011                 0.05
dm012                 0.05
dm013                 0.05
dm014                 0.05
dm015                 0.05
dm016                 0.05
dm017                 0.05
dm018                 0.05
dm019                 0.05
dm020                 0.05
dm021                 0.05
dm022                 0.05
dm023                 0.05
dm024                 0.05
dm025                 3.75
"""
        super(SediLsConfig,self).__init__(filename=filename,text=text)

class SediConfig(GenericConfig):
    def __init__(self,filename=None,text=None):
        if filename is None and text is None:
            text = """
########################################################################
#
#  Default input file for the sediment module in SUNTANS.
#
########################################################################
NL			1	# Number of bed layers (MAX = 5)
spwght                  2.65    # Specific weight of the sediment particle
diam                    0.0001  # Mean diameter of the sediment particle (m)
gamma                   0.2     # Coefficient for flocculated settling velcoity 
Chind                   0.1     # Concentration (in volumetric fraction) criterion for hindered settling velocity
Cfloc                   0.02    # Concentration (in volumetric fraction) criterion for flocculated settling velcoity
k                       0.0002  # Constant coefficient for settling velocity as a function of conc.
Kb                      0.001    # Bottom length scale
Kagg                    0.15     # Aggregation coefficient (dimensionless)
Kbrk                    0.0002   # Break-up coefficient (dimensionless)
Fy                      0.0000000001  #Yield strength (N)
nf                      2.5           #Fractal dimension (0~3, usually 2~3)
q                       0.5           #Constant coefficient in the breakup formulation
Nsize                   2             #Number of the size classes
diam_min                0.00008       #Minimum sediment size
diam_max                0.00024        #Maximum sediment size
Dp                      0.000004      #Diameter of the primary particle (in m)            
Dl1                     180000     # Dry density (g/m^3)
Tcsl1                   0.06   # Critical Shear Stress (N/m^2)
E1                      0.03   # Erosion Rate Constant (g/m^2/s)
alpha1                  6.5     # Empirical coef. for the erosion rate
cnsd1                   0.001   # Consolidation rate (g/m^2/s)
hl1                     1.00      # Layer thickness (m)
"""
        super(SediConfig,self).__init__(filename=filename,text=text)
        
# the current process for handling dates uses pylab datenums, but since these
# have units of days, and days have a non-power of 2 number of seconds in them,
# it can't represent times accurately.  the error is usually on the order of
# 10 us - this parameter describes the expected resolution of values which
# have been converted to floating point - and where possible the conversion
# back from floating point should round at this precision
# to make it a bit easier to know that the arithmetic will be exact, this
# is specified as an integer inverse of the precision 
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
        keys = ['nkmax', 
                'stairstep',
                'rstretch',
                'correctvoronoi',
                'voronoiratio',	
                'vertgridcorrect',
                'intdepth',
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

class SunReader(object):
    """
    Encapsulates reading of suntans output data
    """

    sun_exec = local_config.sun_exec

    EMPTY = 999999
    
    def __init__(self,datadir='.'):
        self.datadir = datadir

        self.load_suntans_dat()

        # lazy-load
        self._step_data = None
        self._profile_points = None
        self._starting_time_step = None
        self._profile_data = {}
        self._z_levels = None
        # per processor stuff
        self._grids = {}
        self._topos = {}
        self._bathy = {}
        self._edgedata = {}
        self._celldata = {}
        self._topo_edges = None
        self._cell_velocity = {}

    _shared_grid = None
    def share_grid(self,other_sun):
        """ Queries for grid information will be redirected to the given
        instance - useful for a series of restarts where there's no need to
        read the grid in multiple times.
        No checking is done to make sure that the given instance actually has the
        same grid as this instance - caveat emptor.
        """
        self._shared_grid = other_sun
        
    def load_suntans_dat(self):
        self.conf = SunConfig( os.path.join(self.datadir,'suntans.dat') )

    def save_to_folder(self,folder):
        """ Write everything we know about the run into the given folder.
        At a minimum this is suntans.dat, points.dat, cells.dat, edges.dat
        """

        folder = os.path.abspath(folder)
        orig_folder = os.path.abspath(self.datadir)
        
        # suntans.dat - copied from our existing suntans.dat unless they are
        # the same file.
        if folder != orig_folder:
            print("Copying suntans.dat")
            shutil.copyfile( os.path.join(orig_folder,'suntans.dat'),
                             os.path.join(folder,'suntans.dat') )
        else:
            print("Skipping suntans.dat - it's the same file")

        # grid:
        g = self.grid()
        print("Renumbering before save")
        g.renumber()
        print("Writing renumbered grid")
        g.write_suntans( folder )

    # old interface - just delegate to newer class
    def conf_float(self,key):
        return self.conf.conf_float(key)
    def conf_int(self,key):
        return self.conf.conf_int(key)
    def conf_str(self,key):
        return self.conf.conf_str(key)

    
    def modify_dt(self,dt):
        """ Changes dt, and tries to alter the simulation period
        accordingly.  This means changing the number of steps to
        cover the same period of time, and updating start_year/day
        so that boundaries.c gets the same real-world time for the
        start of the simulation
        """
        # we can't really trust start_year/day at the beginning,
        # but try to preserve the duration:
        duration = self.conf.simulation_seconds()
        eff_t0 = self.effective_time_zero()
        self.conf.set_value('dt',float(dt))
        # Now this will use the new value of dt to figure out nsteps
        self.conf.set_simulation_period(eff_t0,eff_t0 + duration/(24*3600.))
        self.conf.write_config()

    def grid(self,processor=None,readonly=True):
        """  if processor is None, return a TriGrid object for the entire domain
        otherwise, return one processor's grid.

        if readonly is True, enable space-savings which reuse
         things like the points array
        """
        if self._shared_grid:
            return self._shared_grid.grid(processor)
        
        if processor not in self._grids:
            self._grids[processor] = orthomaker.OrthoMaker(suntans_reader=self,processor=processor,readonly=readonly)

        return self._grids[processor]

    def proc_nonghost_cells(self,proc):
        """ returns an array of cell indices which are *not* ghost cells """
        ng_cells = []
        g = self.grid(proc)
        
        for i in range(g.Ncells()):
            if not self.proc_cell_is_ghost(proc,i):
                ng_cells.append(i)
        return array(ng_cells)
    def proc_cell_is_ghost(self,proc,i):
        """ Returns true if the specified cell is a ghost cell.
        """
        cdata = self.celldata(proc)
        edges = self.grid(proc).edges

        marks = edges[cdata[i,5:8].astype(int32),2]
        return any(marks==6)
        
    warned_profile_filesize = 0
    def steps_available(self,output='grid'):
        """ How many timesteps have been written out
        (not how many have been run, though, since generally only a small
        fraction are actually written out to disk)
        """
        if output=='grid':
            return self.step_data()['steps_output']
        else:
            # profile data:
            from_step_dat = old_div((self.step_data()['step'] - self.starting_time_step()),self.conf_int('ntoutProfs'))
            prof_dat_size = os.stat( self.file_path('FreeSurfaceFile') + ".prof" ).st_size
            n_prof_pnts = len( self.profile_points() )
            n_steps_in_file = old_div(prof_dat_size, (n_prof_pnts * REALSIZE))

            if n_steps_in_file != from_step_dat and not self.warned_profile_filesize:
                print("Filesize suggests %i profile timesteps, but step.dat suggests %i"%(n_steps_in_file,from_step_dat))
                self.warned_profile_filesize = 1 

            return n_steps_in_file
            
        
    def step_data(self,reload=False):
        """ Return a dict with information parsed from step.dat, or None if there is no step.dat file
        """
        if self._step_data is None or reload:
            try:
                raw = open(self.datadir + "/step.dat","rt").read()
            except IOError:
                return None
            
            m = re.match("On (\d+) of (\d+), t=([0-9\.]+) \(([0-9\.]+)% Complete, (\d+) output\)",
                         raw)
            if m is None:
                print("------step.dat------")
                print(raw)
                print("--------------------")
                
                raise Exception("Failed to parse step.dat.  Probably transient - try again")
            self._step_data = {'step':int(m.group(1)),
                               'total_steps':int(m.group(2)),
                               'model_time':float(m.group(3)),
                               'percent':float(m.group(4)),
                               'steps_output':int(m.group(5))
                               }
        return self._step_data
        
    def domain_decomposition(self,np=None):
        """ Once the grid and depths are in place, call on sun to
        break it out into multiple domains
        """
        if np is None:
            np = self.num_processors()
        # possible that this is the first time to run the decomposition
        if np is None:
            print("Couldn't detect number of processors - will use np=1")
            np = 1


        cmd = [self.sun_exec,'-g','-vvv','--datadir=%s'%os.path.abspath(self.datadir)]
        
        MPIrunner(cmd,np=np)

        # and check to see that it worked - or at least created a topology.dat.0 file
        if not os.path.exists( self.file_path('topology',0) ):
            raise Exception("Appears that partitioning failed - probably couldn't find sun (%s) or MPI"%self.sun_exec)

    def run_simulation(self,np=None,wait=1):
        """ run it!
        if wait is true, will not return until simulation process completes
        """
        if np is None:
            np = self.num_processors()
            if np is None:
                print("Couldn't detect number of processors - will use np=1")
                np = 1

        cmd = [self.sun_exec,'-s','-vvv','--datadir=%s'%os.path.abspath(self.datadir)]

        if self.starting_time_step() != 0:
            print("Will be a restart")
            cmd.append('-r')

        if wait:
            t_start = time.time()
            
        MPIrunner(cmd,np=np,wait=wait)

        if wait:
            t_elapsed = time.time() - t_start
            self.log_runtime(np,t_elapsed)

    def log_runtime(self,np,t_elapsed):
        """ For now, just log as many of the relevant details as possible to a file in datadir.
        Depending on how teragrid charges for stuff, this may get expanded into something that
        can approximate the anticipated run time.

        possible factors:
         - the sun executable
         - all of sedi.dat, wave.dat
         - the grid (discerned from depth.dat)
         - Nkmax, dt, nonlinear, beta==0, gamma==0
         - np
        """
        details = {}

        for datfile in ["sedi.dat","wave.dat"]:
            fn = os.path.join(self.datadir,datfile)
            if os.path.exists(fn):
                fp = open(fn,"rt")
                hsh = hashlib.md5() # md5.new()
                hsh.update(fp.read())
                fp.close()
                details[datfile ] = hsh.hexdigest()
            else:
                details[datfile] = "missing"
        fp = open(self.sun_exec,"rb")
        hsh = hashlib.md5() ; hsh.update(fp.read())
        details['sun'] = hsh.hexdigest() # md5.new(fp.read()).hexdigest()
        fp.close()

        # get some idea of grid size just from the first processor:
        cdata = self.celldata(0)
        details['Nc0'] = "%d"%len(cdata)
        details['Nc3D0'] = "%d"%cdata[4].sum()
        details['np'] = "%d"%np
        details['dt'] = "%f"%self.conf_float('dt')
        details['nonlinear'] = "%d"%self.conf_int('nonlinear')
        for s in ['beta','gamma']:
            if self.conf_float(s)!=0:
                details[s] = "yes"
            else:
                details[s] = "no"

        # And write it out...
        log_fp = open(os.path.join(self.datadir,"timing.out"),"wt")
        for k in details:
            log_fp.write("%s = %s\n"%(k,details[k]))
        log_fp.write("elapsed = %f\n"%t_elapsed)
        log_fp.close()
        
        

    # The actual data:
    #   Most data is defined at the voronoi centers
    #   Some values (free surface), or defined only at the surface,
    #    while others are defined for every z-level
    #   Some values have a time axis as well.
    # Want something like:
    #   fs = sdata.freesurface()
    #   # fs has a time axis, which can be referenced
    #   # 
    #   gridplot( fs.grid(time=213.4) )
    #   gridplot( fs.grid(step=100) )

    # Maybe the grid can handle most of the plotting, and we just
    #  pass it the grid data?
    # Then the freesurface would just map cell indices to a scalar
    #  value.
    # Surface velocity would map cell indices to a 3-vector
    def cell_velocity_fread(self,processor,time_step):
        """ Returns velocity values in an array Nc x Nkmax x 3
        for now, only a single time_step can be specified
        returns tuple of grid, velocity[cell,z_level,component]
        """
        g = self.grid(processor)

        Ncells = g.Ncells()

        u_name = self.file_path("HorizontalVelocityFile",processor)

        nsteps = self.steps_available()

        nkmax = self.conf_int('Nkmax')
        
        fp = open(u_name,"rb")

        if time_step >= nsteps:
            print("Using last time step instead of whatever you asked for")
            time_step = nsteps - 1
        if time_step < 0:
            time_step = nsteps + time_step
            if time_step < 0:
                print("Clamping time step to be non-negative")
                time_step = 0

        frame_size = nkmax*3*Ncells*REALSIZE
        fp.seek(time_step*frame_size)
        raw = fp.read(frame_size)

        # assume it was run on the same machine, so endian-ness
        # is the same
        values = fromstring(raw,REALTYPE)
        results = values.reshape( (nkmax,3,Ncells) )
        results = swapaxes(results,1,2) # => z-level, cell index, component
        results = swapaxes(results,0,1) # => cell index, z-level, component
        return g,results

    
    def cell_velocity(self,processor,time_step=None):
        """ Like cell_velocity(), but try to memory map the file, and returns all
        timesteps
        
        returns tuple of grid, velocity[time,cell,z_level,component]
        """
        g = self.grid(processor)

        if processor not in self._cell_velocity:
            Ncells = g.Ncells()

            u_name = self.file_path("HorizontalVelocityFile",processor)
            nkmax = self.conf_int('Nkmax')

            # Choose the number of steps based on the file size -
            # A single step will take Ncells*Nkmax*3*REALSIZE
            frame_size = nkmax*3*Ncells*REALSIZE

            nbytes = os.stat(u_name).st_size

            # self.steps_available()
            nsteps = nbytes // frame_size

            final_shape = (nsteps,nkmax,3,Ncells)

            results = memmap(u_name, dtype=REALTYPE, mode='r', shape=final_shape)

            # time, k, component, cell
            results = swapaxes(results,2,3) # => time, z-level, cell index, component
            results = swapaxes(results,1,2) # => time, cell index, z-level, component
            self._cell_velocity[processor] = results
        else:
            results = self._cell_velocity[processor]
        
        if time_step is not None:
            results = results[time_step,...]

        return g,results

    def cell_nuT(self,processor,time_step):
        return self.cell_scalar('EddyViscosityFile',processor,time_step)
    
    def cell_salinity(self,processor,time_step=None):
        """ Read salinity values into an array Nc x Nkmax
        for now, only a single time_step can be specified
        
        returns tuple of grid, salinity[cell,z_level]
        """
        return self.cell_scalar('SalinityFile',processor,time_step)

    def cell_temperature(self,processor,time_step=None):
        """ Read temperature values into an array Nc x Nkmax
        for now, only a single time_step can be specified
        
        returns tuple of grid, salinity[cell,z_level]
        """
        return self.cell_scalar('TemperatureFile',processor,time_step)

    _cell_scalars = None
    def cell_scalar(self,filename_name,processor,time_step=None):
        """ Read a cell-based scalar value from suntans output.
        filename_name: the suntans.dat field that has the setting we want.
          SalinityFile
          TemperatureFile
        """
        if self._cell_scalars is None:
            self._cell_scalars = {}

        if filename_name not in self._cell_scalars:
            self._cell_scalars[filename_name] = {}
            
        g = self.grid(processor)
        if processor not in self._cell_scalars[filename_name]:
            Ncells = g.Ncells()

            s_name = self.file_path(filename_name,processor)

            nsteps = self.steps_available()
            nkmax = self.conf_int('Nkmax')

            frame_size = Ncells * nkmax * REALSIZE
            nbytes = os.stat(s_name).st_size
            data_shape = ( nbytes//frame_size,nkmax,Ncells)

            try:
                full_scal = memmap(s_name,
                                   dtype=REALTYPE,
                                   mode='r',
                                   shape=data_shape)
                # print "Successfully mapped %s"%s_name
            except mmap.error:
                print("Looks like we can't memory map the files.  Going to be slow...")
                print("Size of %s is %d bytes"%(s_name,bytes))
                # is there another option, like a dynamically read, file-backed array?
                fp = open(s_name,'rb')
                full_scal = fromstring(fp.read(),dtype=REALTYPE ).reshape( data_shape )
                fp.close()
            
            self._cell_scalars[filename_name][processor] = swapaxes(full_scal,1,2)

        full_scal = self._cell_scalars[filename_name][processor]

        nsteps = full_scal.shape[0]

        if time_step is not None:
            if time_step >= nsteps:
                time_step = nsteps - 1
            if time_step < 0:
                time_step = nsteps + time_step
                if time_step < 0:
                    time_step = 0
            return g,full_scal[time_step,:,:]
        else:
            return g,full_scal[:,:,:]

    def close_files(self):
        """ Close things like memmap'd scalar files
        """
        if self._cell_scalars is not None:
            for filename in self._cell_scalars:
                for proc in list(self._cell_scalars[filename].keys()):
                    del self._cell_scalars[filename][proc]

        if self._freesurface is not None:
            for proc in list(self._freesurface.keys()):
                del self._freesurface[proc]

        if self._cell_velocity is not None:
            for proc in list(self._cell_velocity.keys()):
                del self._cell_velocity[proc]

        if self._celldata is not None:
            for proc in list(self._celldata.keys()):
                del self._celldata[proc]

        if self._edgedata is not None:
            for proc in list(self._edgedata.keys()):
                del self._edgedata[proc]

    def ctop(self,processor,time_step):
        h = self.freesurface(processor,[time_step])[0]
        return self.h_to_ctop(h)
    def h_to_ctop(self,h,dzmin=None):
        if dzmin is None:
            # recent suntans code does this
            # the 2 is from #define DZMIN_SURFACE 2*DZMIN
            # so self.dzmin should reflect #DEFINE DZMIN 0.001 (or whatever it currently is)
            dzmin = 2*self.dzmin
        ctops = searchsorted(self.z_levels() - dzmin, -h)
        return ctops

    _freesurface = None
    def freesurface(self,processor,time_step=None):
        """ Returns freesurface values in an array (len(time_step),Nc)
        (or a 1-d array if time_step is a scalar).
        if time_step is not specified, returns freesurface for all cells, all timesteps
        in the file.
        """
        
        g = self.grid(processor)

        if self._freesurface is None:
            self._freesurface = {}
            
        if processor not in self._freesurface:
            Ncells = g.Ncells()

            fs_name = self.file_path("FreeSurfaceFile",processor)

            frame_size = Ncells * REALSIZE
            nbytes = os.stat(fs_name).st_size
            data_shape = ( nbytes//frame_size,Ncells)

            try:
                self._freesurface[processor] = memmap(fs_name, dtype=REALTYPE, mode='r', shape=data_shape)
            except mmap.error:
                fp = open(fs_name,'rb')
                self._freesurface[processor] = fromstring(fp.read(),dtype=REALTYPE ).reshape( data_shape )
                fp.close()


        if time_step is None:
            return self._freesurface[processor]
        else:
            return self._freesurface[processor][time_step,:]
        
    def file_path(self,conf_name,processor=None):
        base_name = self.conf_str(conf_name)
        if base_name is None:
            # raise Exception,"File path configuration not found for %s"%conf_name
            base_name = conf_name
        if processor is not None and conf_name not in ('points','DataLocations'):
            base_name += ".%i"%processor

        return self.datadir+"/"+base_name

    def profile_points(self,force=False):
        if self._profile_points is None or force:
            prof_points_fp = open(self.file_path("DataLocations"))

            xy_points = []
            for line in prof_points_fp:
                this_point = [float(s) for s in line.split()]
                if len(this_point) == 2:
                    xy_points.append(this_point)

            self._profile_points = array(xy_points)
        return self._profile_points

    _profdata = None
    def profdata(self):
        """ Reads the profdata.dat file,
        """
        if self._profdata is None:
            fn = self.file_path('ProfileDataFile')
            fp = open(fn,'rb')

            # data format:
            #  (4 byte int)numTotalDataPoints: Number of data points found on all processors.  Note that
            #      that this could be different from the number specified since some may lie outside the domain.
            #  (4 byte int)numInterpPoints: Number of nearest neighbors to each point used for interpolation.
            #  (4 byte int)NkmaxProfs: Number of vertical levels output in the profiles.
            #  (4 byte int)nsteps: Total number of time steps in the simulation.
            #  (4 byte int)ntoutProfs: Frequency of profile output.  This implies a total of nsteps/ntoutProfs are output.
            #  (8 byte double)dt: Time step size
            #  (8 byte double array X NkmaxProfs)dz: Contains the vertical grid spacings.
            #  (4 byte int array X numTotalDataPoints)allIndices: Contains the indices of each point that determines its
            #      original location in the data file.  This is mostly for debugging since the output data is resorted
            #      so that it is in the same order as it appeared in the data file.
            #  (4 byte int array X 2*numTotalDataPoints)dataXY: Contains the original data points at (or near) which profiles
            #      are output.
            #  (8 byte double array X numTotalDataPoints*numInterpPoints)xv: Array containing the x-locations of the nearest
            #      neighbors to the dataXY points.  If numInterpPoints=3, then the 3 closest neighbors to the point
            #      (dataXY[2*i],dataXY[2*i+1]) are (xv[3*i],yv[3*i]), (xv[3*i+1],yv[3*i+1]), (xv[3*i+2],yv[3*i+2]).
            #  (8 byte double array X numTotalDataPoints*numInterpPoints)yv: Array containing the y-locations of the nearest
            #      neighbors to the dataXY points (see xv above).

            pdata = {}
            
            hdr_ints = fromstring(fp.read(5*4),int32)
            pdata['numTotalDataPoints'] = hdr_ints[0]
            pdata['numInterpPoints'] = hdr_ints[1]
            pdata['NkmaxProfs'] = hdr_ints[2]
            pdata['nsteps'] = hdr_ints[3]
            pdata['ntoutProfs'] = hdr_ints[4]

            pdata['dt'] = fromstring(fp.read(REALSIZE),REALTYPE)
            pdata['dzz'] = fromstring(fp.read(REALSIZE*pdata['NkmaxProfs']),REALTYPE)
            pdata['allIndices'] = fromstring(fp.read(4*pdata['numTotalDataPoints']),int32)
            # Wait a second - this file doesn't even have proc/cell info...
            dataxy = fromstring(fp.read(REALSIZE*2*pdata['numTotalDataPoints']),REALTYPE)
            # pdata['dataXY_serial'] = dataxy # needs to be reshaped
            pdata['dataXY'] = dataxy.reshape( (-1,2) )

            print("About to read coordinates, file position is",fp.tell())
            
            xvyv = fromstring(fp.read(2*REALSIZE*pdata['numTotalDataPoints']*pdata['numInterpPoints']),
                              REALTYPE)
            pdata['xvyv'] = xvyv
            pdata['xy'] = xvyv.reshape( (2,-1) ).transpose()
            
            self._profdata = pdata
        return self._profdata
    def nkmax_profs(self):
        nkmax_profs = self.conf_int('NkmaxProfs')
        if nkmax_profs == 0:
            nkmax_profs = self.conf_int('Nkmax')
        return nkmax_profs
    
    def profile_data(self,scalar,timestep=None):
        """ scalar is one of HorizontalVelocityFile,
        FreeSurfaceFile, etc"""
        if scalar not in self._profile_data:
            prof_pnts = self.profile_points()
            prof_len = prof_pnts.shape[0]

            prof_fname = self.file_path(scalar) + ".prof"

            if not os.path.exists(prof_fname):
                return None

            ## Figure out the shape of the output:
            #  I'm assuming that profile data gets spat out in the same
            #  ordering of dimensions as regular grid-based data
            
            shape_per_step = []

            # profiles.c writes u first then v, then w, each with a
            # separate call to Write3DData()
            if scalar == 'HorizontalVelocityFile':
                shape_per_step.append(3)
            
            # the outer loop is over profile points
            shape_per_step.append(prof_len)

            # This used to drop the z-level dimension for 2-D runs, but
            # moving forward, seems better to always include z even if
            # there's only one layer, so post-processing scripts don't have
            # to special case it.
            
            ## And does it have z-levels? if so, that is the inner-most
            #  loop, so the last dimension of the array
            if scalar != 'FreeSurfaceFile':
                nkmax_profs = self.nkmax_profs() 
                shape_per_step.append(nkmax_profs)

            # better to use the size of the specific file we're opening:
            # NOT this way:
            # profile_steps = self.steps_available(output='profile')
            # but this way:
            prof_dat_size = os.stat( prof_fname).st_size
            bytes_per_step = REALSIZE * prod( array(shape_per_step) )
            n_steps_in_file = int( prof_dat_size//bytes_per_step )

            final_shape = tuple([n_steps_in_file] + shape_per_step)
            
            # print "Total shape of profile data: ",final_shape

            if self.conf_int('numInterpPoints') != 1:
                raise Exception("Sorry - please set numInterpPoints to 1")

            if 1:
                # print "Trying to memory map the data.."
                data = memmap(prof_fname, dtype=REALTYPE, mode='r', shape=final_shape)
            else:
                prof_fp = open(prof_fname,"rb")
                data = fromstring(prof_fp.read(),float64)
                prof_fp.close()
                
                data = data.reshape(*final_shape)
            
            self._profile_data[scalar] = data

        data = self._profile_data[scalar]
        
        if timestep is not None:
            if timestep >= data.shape[0]:
                print("Bad timestep %d, last valid step is %d"%(timestep,data.shape[0]-1))
                timestep = data.shape[0] - 1
            return data[timestep]
        else:
            return data

    def scalars(self):
        """ Returns a list of names for scalar outputs, i.e. 
        [SalinityFile, TemperatureFile]
        """
        scals = []
        if float(self.conf['beta']) != 0:
            scals.append('SalinityFile')
        if float(self.conf['gamma']) != 0:
            scals.append('TemperatureFile')
        
        scals.append('EddyViscosityFile')
        return scals
    
    # Profile processing:
    
    def profile_to_transect(self,xy,absdays_utc,scalar):
        """ Extract data from profile dumps close to the given times and
        locations, then construct a Transect instance.

        xy is an Nx2 vector of utm coordinates
        absdays_utc is an N vector of UTC abs-days, from date2num
        scalar identifies what variable is output - 'SalinityFile'

        For now, no interpolation is done, and the transect will have the
        actual xy, times and depths from the simulation.

        In the future there could be options for interpolating in time,
        horizontal space, and/or vertical space, such that the resulting
        transect would be congruent with some other transect
        """

        ## Allocate
        N = len(xy)

        new_xy = zeros( (N,2), float64 )
        new_times = zeros( N, float64 )

        if self.nkmax_profs() != self.conf_int('Nkmax'):
            # This isn't hard, just too lazy to do it right now.
            raise Exception("Not quite smart enough to handle profiles that are different Nkmax than grid")

        new_scalar = zeros( (self.conf_int('Nkmax'),N), float64 )
        z_levels = concatenate( ([0],-self.z_levels()) )
        z_interfaces = repeat( z_levels[:,newaxis],N,axis=1 )
        mask = zeros( new_scalar.shape,bool )

        ## Get timeline:

        t0 = self.conf.time_zero()
        t0_absdays = date2num( t0 )

        prof_absdays = t0_absdays + self.timeline(units='days',output='profile')

        pnts = self.profile_points()
        
        # this memory maps the profile data file, so there is no advantage in pulling
        # just a single timestep.
        prof_data = self.profile_data(scalar)
        prof_h = self.profile_data('FreeSurfaceFile')

        if len(prof_data.shape) == 2:
            new_shape = (prof_data.shape[0],prof_data.shape[1],1)
            prof_data = prof_data.reshape( new_shape )

        # prof_data: [step,point,z-level]

        for i in range(N):
            # what timestep is this closest to?
            prof_step = searchsorted(prof_absdays,absdays_utc[i])

            # what profile point is it closest to?
            prof_loc = self.xy_to_profile_index(xy[i])
            # and find the bathymetry there:
            proc,cell = self.closest_cell(xy[i])
            cdata = self.celldata(proc)
            bathy = -cdata[cell,3]

            # read that profile
            fs_height = prof_h[prof_step,prof_loc]
            new_xy[i] = pnts[prof_loc]
            new_times[i] = prof_absdays[prof_step]
            new_scalar[:,i] = prof_data[prof_step,prof_loc,:]
            
            # and take care of a possible thin cell at the surface:
            Nk = cdata[cell,4].astype(int32)
            ctop = min( self.h_to_ctop(fs_height), Nk-1)
            z_interfaces[:ctop+1,i] = fs_height
            z_interfaces[Nk:,i] = bathy

            mask[:ctop,i] = True
            mask[Nk:,i] = True

        # apply mask to scalar
        new_scalar = ma.array(new_scalar,mask=mask)

        if 0:
            # Transect assumes that data are located at the given nodes, so we need to
            # roughly translate the bin-based values of SUNTANS into centered points.
            # This does not take into account the location of the freesurface or the bed
            # within the given z-level.
            z_level_centers = self.bathymetry_offset()-self.z_levels() + 0.5*self.dz()
            return transect.Transect(new_xy,new_times,
                                     z_level_center,
                                     new_scalar,
                                     desc='Suntans output: %s'%scalar)
        else:
            # New transect code allows for zonal scalar -
            z_interfaces = z_interfaces + self.bathymetry_offset()
            return transect.Transect(new_xy,new_times,
                                     z_interfaces,
                                     new_scalar,
                                     desc='Suntans output: %s'%scalar)

    def map_local_to_global(self,proc):
        gglobal=self.grid()
        glocal=self.grid(proc)

        l2g=np.zeros( glocal.Ncells(), 'i4')
        for li in range(glocal.Ncells()):
            l2g[li] = gglobal.find_cell( glocal.cells[li] )
        return l2g
        
    # in-core caching in addition to filesystem caching
    _global_to_local = None
    def map_global_cells_to_local_cells(self,cells=None,allow_cache=True,check_chain=True,
                                        honor_ghosts=False):
        """ Map global cell indices to local cell indices.
        if cells is None, return a mapping for all global cells

        if cells is None, and allow_cache is true, attempt to read/write
         a cached mapping as global_to_local.bin
        
        if honor_ghosts is True, then make the mapping consistent with the "owner"
        of each cell, rather than just a processor which contains that cell.
        """
        # Map global cells to processors, then iterate over processors, calculating
        # surface velocity, and averaging over the relevant cells.

        if cells is None and allow_cache:
            if self._global_to_local is not None:
                print("using in-core caching for global to local mapping")
                return self._global_to_local
            
            if check_chain:
                datadirs = [s.datadir for s in self.chain_restarts()]
            else:
                datadirs = [self.datadir]

            for datadir in datadirs[::-1]:
                cache_fn = os.path.join(datadir,'global_to_local.bin')

                if os.path.exists(cache_fn):
                    fp = open(cache_fn,'rb')
                    global_to_local = None
                    try:
                        global_to_local = pickle.load(fp)
                    finally:
                        fp.close()
                    if global_to_local is not None:
                        return global_to_local
            cache_fn = os.path.join(self.datadir,'global_to_local.bin')
        else:
            cache_fn = None
            
        grid = self.grid()

        if cells is None:
            print("Will map all cells")
            cells = arange(grid.Ncells())
            all_cells = True
        else:
            all_cells = False
            
        global_to_local = zeros( len(cells), [('global',int32),
                                              ('proc',int32),
                                              ('local',int32)])
        global_to_local['global'] = cells
        global_to_local['proc'] = -1

        for processor in range(self.num_processors()):
            print("P%d"%processor, end=' ')
            local_g = self.grid(processor)
            if all_cells:
                # faster to loop over the local cells
                if honor_ghosts:
                    local_cells=self.proc_nonghost_cells(processor)
                else:
                    local_cells=range(local_g.Ncells())
                for i in local_cells:
                    gi = grid.find_cell( local_g.cells[i] )
                    gtl = global_to_local[gi]
                    if gtl['proc'] >= 0:
                        continue
                    gtl['proc'] = processor
                    gtl['local'] = i
            else:
                for gtl in global_to_local:
                    if gtl['proc'] >= 0:
                        continue
                    try:
                        i = local_g.find_cell( grid.cells[gtl['global']] )
                        gtl['proc'] = processor
                        gtl['local'] = i
                    except trigrid.NoSuchCellError:
                        pass
        print("done mapping")

        if cache_fn is not None:
            fp = open(cache_fn,'wb')
            pickle.dump(global_to_local,
                        fp)
            fp.close()
            self._global_to_local = global_to_local
            
        return global_to_local

    def cell_values_local_to_global(self,cell_values=None,func=None):
        """ Given per-processor cell values (for the moment, only supports
        2-D cell-centered scalars) return an array for the global cell-centered
        data
        """
        g2l = self.map_global_cells_to_local_cells()
        gg = self.grid()

        print("Compiling local data to global array")

        g_data = None # allocate lazily so we know the dtype to use
        

        for p in range(self.num_processors()):
            if cell_values:
                local_values = cell_values[p]
            else:
                local_values = func(p)
                
            # not terribly efficient, but maybe okay...
            local_g2l = g2l[ g2l['proc'] == p ]

            if g_data is None:
                g_data = zeros( gg.Ncells(), dtype=local_values.dtype)
                
            g_data[ local_g2l['global'] ] = local_values[ local_g2l['local'] ]
        print("Done compiling local data to global array")

        return g_data

    def read_section_defs(self):
        fp = open(self.file_path('sectionsinputfile'),'rt')

        def tok_gen(fp):
            for line in fp:
                for snip in line.split():
                    yield snip
                    
        token = tok_gen(fp).__next__

        Nsections = int(token())

        sections = [None]*Nsections
        for nsec in range(Nsections):
            Nnodes = int(token())
            nodes = []
            for n in range(Nnodes):
                nodes.append( int(token()) )
            sections[nsec] = nodes
        return sections
        
    def full_to_transect(self,xy,absdays,scalar_file,min_dx=10.0):
        """ Construct a Transect from full grid scalar output, where xy is a sequence of points
        giving the transect, absdays a sequence of times, and scalar which field should be
        read.

        For now, it's not smart enough to know how big the cells are and do a proper line-cell
        intersection, so you have to specify a min_dx, and as long as that is significantly
        smaller than the grid size, it will pick up all the cells with a significant intersection
        with the transect.

        also the handling of the freesurface and timesteps are lacking.  The freesurface is used
        only to decide ctop - it is not used to truncate the surface cell.
        
        And no interpolation in time is done - only the nearest timestep is extracted.
        """
        xy = asarray(xy)
        absdays = asarray(absdays)
        
        utm_points,sources = upsample_linearring(xy,density=200.0,closed_ring=0,return_sources=1)

        # construct an interpolated set of times, estimating a timestamp for each of the newly
        # interpolated utm_points.
        absdays_expanded = interp(sources, arange(len(absdays)),absdays)

        utm_deltas = sqrt(sum(diff(utm_points,axis=0)**2,axis=1))
        utm_dists = concatenate( ([0],cumsum(utm_deltas)) )

        ## choose a timestep:
        timeline = date2num(self.time_zero()) + self.timeline(output='grid',units='days')

        steps = searchsorted(timeline,absdays_expanded) # the output right after the requested date

        # adjust to whichever step closer:
        for i in range(len(steps)):
            if steps[i]>0 and timeline[steps[i]] - absdays_expanded[i] > absdays_expanded[i] - timeline[steps[i]-1]:
                steps[i] -= 1

        g = self.grid()

        global_cells = []

        for xy in utm_points:
            global_cells.append( g.closest_cell(xy) )

        # Now remove any duplicates    
        global_cells = array(global_cells)
        valid = (global_cells[:-1] != global_cells[1:] )
        valid = concatenate( (valid,[True]) )

        global_cells = global_cells[valid]
        utm_dists = utm_dists[valid]
        utm_points = utm_points[valid]
        steps = steps[valid]
        absdays_expanded = absdays_expanded[valid]

        nkmax = self.conf_int('nkmax')

        scalar = zeros( (len(global_cells),nkmax), float64 )
        local_proc_cells = zeros( (len(global_cells),2), int32 ) - 1

        bathy_offset = self.bathymetry_offset()
        interface_elevs = bathy_offset + concatenate( ([0], -self.z_levels()) ) # Nk + 1 entries!
        elev_fs = zeros( len(global_cells), float64 ) # these will be corrected for bathy_offset
        elev_bed = zeros( len(global_cells), float64 )

        # elevations of interfaces for each watercolumn
        elev_per_column = zeros( (len(global_cells),len(interface_elevs)), float64)
        
        for proc in range(self.num_processors()):
            # print "Reading salinity from processor %d"%proc
            local_g, full_scal = self.cell_scalar(scalar_file,proc) # [Ntimesteps,cells,z-level]
            fs = self.freesurface(proc) #  [Ntimesteps,cells]
            celldata = self.celldata(proc)

            for gc_i in range(len(global_cells)):
                gc = global_cells[gc_i]
                step = steps[gc_i]

                if local_proc_cells[gc_i,0] >= 0:
                    continue # already been read
                try:
                    i = local_g.find_cell( g.cells[gc] )
                    local_proc_cells[gc_i,0] = proc
                    local_proc_cells[gc_i,1] = i

                    ktop = self.h_to_ctop(fs[step,i])
                    kmax = int(celldata[i,4])
                    
                    elev_fs[gc_i] = fs[step,i] + bathy_offset
                    elev_bed[gc_i] = -celldata[i,3] + bathy_offset
                    scalar[gc_i,:] = full_scal[step,i,:]
                    scalar[gc_i,:ktop] = nan
                    scalar[gc_i,kmax:] = nan
                    
                    elev_per_column[gc_i,:] = interface_elevs
                    elev_per_column[gc_i,ktop] = elev_fs[gc_i]
                    elev_per_column[gc_i,kmax] = elev_bed[gc_i]
                except trigrid.NoSuchCellError:
                    continue

        ## Make that into a transect:
        scalar = ma.masked_invalid(scalar)

        # ideally we'd include the time-varying freesurface elevation, too...
        t = transect.Transect(xy=utm_points,
                              times = timeline[steps],
                              elevations=elev_per_column.T,
                              scalar=scalar.T,
                              dists=utm_dists)

        t.trim_to_valid()
        return t

    _finder = None
    def xy_to_profile_index(self,xy):
        if self._finder is None:
            pnts = self.profile_points()
            data = arange(len(pnts))
            finder = field.XYZField(pnts,data)
            finder.build_index()
            self._finder = finder

        return self._finder.nearest(xy)

    def timeline(self,units='seconds',output='grid'):
        """
        units: seconds, minutes, hours, days
          or absdays, which returns matplotlib-style datenums
        output: grid - timeseries for grid outputs
                profile - timeseries for profile outputs
        times are measured from sun.time_zero() except for absdays
          which is the matplotlib absolute time unit, decimal days
          since something...

        Note that in some cases the outputs are not evenly spaced,
        particularly when ntout does not divide evenly into nsteps
        and when the starting time step was not on an integral number
        of ntouts
        """
        steps_output = self.steps_available(output)
        
        if output=='grid':
            output_interval = self.conf_int('ntout')
        elif output=='profile':
            output_interval = self.conf_int('ntoutProfs')
        else:
            raise Exception("bad output spec for timeline: %s"%output)

        dt = output_interval*self.conf_float('dt')
        if output == 'grid':
            # a little tricky, as the starting step may not be an integral number of
            # ntout.
            offset = self.starting_time_step() % self.conf_int('ntout')
            steps = arange(0,steps_output)*output_interval + self.starting_time_step() - offset
            steps[0] += offset
            # may also have a straggler
            last_step = StoreFile(self,0).timestep()
            if last_step != steps[-1]:
                steps = concatenate( (steps,[last_step]) )
            tseries_seconds = steps*self.conf_float('dt')
        else:
            # profiles are output when the step is a multiple of the output interval, and only starting
            # with the end of the first step:
            # this expression rounds starting_time_step+1 up to the next even multiple of output_interval.
            first_prof_output_step = output_interval * \
                                     ((self.starting_time_step() + output_interval)//output_interval)
            tseries_seconds = arange(0,steps_output)*dt + first_prof_output_step*self.conf_float('dt')

        offset = 0
        
        if units == 'seconds':
            divisor = 1.
        elif units == 'minutes':
            divisor = 60.
        elif units == 'hours':
            divisor = 3600.
        elif units == 'days':
            divisor = 24*3600
        elif units =='absdays':
            divisor = 24*3600
            offset = date2num( self.time_zero() )
        else:
            raise Exception("Bad time unit: %s"%units)
        
        return offset + tseries_seconds/divisor

    class Topology(object):
        pass
    
    def topology(self,processor):
        if processor not in self._topos:
            topo_path = self.file_path('topology',processor)

            try:
                fp = open(topo_path)
            except IOError:
                return None
            
            def int_iter(fp):
                for line in fp:
                    for num in map(int,line.split()):
                        yield num
            # just returns integer after integer...
            nums = int_iter(fp)

            topo = SunReader.Topology()
            topo.filename = topo_path

            topo.num_processors = next(nums)
            topo.num_neighbors =  next(nums)
            topo.neighbor_ids = [next(nums) for i in range(topo.num_neighbors)]

            topo.cellsend = [None]*topo.num_neighbors
            topo.cellrecv = [None]*topo.num_neighbors
            topo.edgesend = [None]*topo.num_neighbors
            topo.edgerecv = [None]*topo.num_neighbors

            for i in range(topo.num_neighbors):
                num_cellsend = next(nums)
                num_cellrecv = next(nums)
                num_edgesend = next(nums)
                num_edgerecv = next(nums)
                topo.cellsend[i] = array([next(nums) for j in range(num_cellsend)])
                topo.cellrecv[i] = array([next(nums) for j in range(num_cellrecv)])
                topo.edgesend[i] = array([next(nums) for j in range(num_edgesend)])
                topo.edgerecv[i] = array([next(nums) for j in range(num_edgerecv)])
            # 3 and 6 come from the limits in grid.h, MAXBCTYPES-1 and MAXMARKS-1
            topo.celldist = array([next(nums) for i in range(3)])
            topo.edgedist = array([next(nums) for i in range(6)])

            grid = self.grid(processor)
            topo.cellp = array([next(nums) for i in range(grid.Ncells())])
            topo.edgep = array([next(nums) for i in range(grid.Nedges())])
            self._topos[processor] = topo
            
        return self._topos[processor]

    def sendrecv_edges(self,data):
        """ data: list with num_processors() elements, each being iterable with Nedges()
        elements.

        Exchange data as described by the topology files, overwriting entries in the given arrays.
        """
        for proc in range(self.num_processors()):
            print("sendrecv for proc %d"%proc)

            topo = self.topology(proc)

            for i in range(topo.num_neighbors):
                nbr = topo.neighbor_ids[i]
                print("  receiving from %d"%nbr)
                topo_nbr = self.topology(nbr)
                # find the edge ids that they're supposed to send to us:
                me_to_them = topo_nbr.neighbor_ids.index(proc)
                # These are the indices, as known by the neighbor:
                nbr_edges_to_send = topo_nbr.edgesend[me_to_them]
                # And the indices as known to me:
                my_edges_to_recv = topo.edgerecv[i]
                data[proc][my_edges_to_recv] = data[nbr][nbr_edges_to_send]
                
    def num_processors(self):
        t = self.topology(0)
        if t:
            return t.num_processors
        else:
            return None

    def time_zero(self):
        """ return python datetime for the when t=0.  This has been moved
        into SunConfig, and here we just delegate to that.
        """
        return self.conf.time_zero()
    
    def simulation_period(self,end_is_last_output=True):
        """ return a pair of python datetime objects for the start and end of the simulation
        This includes the offset due to a restart, and gives the ending datetime of when the
        simulation would finish, irrespective of how many steps have been run so far.

        there are some corner cases here that get fuzzy, depending on which files
        are availale for determining the simulation period.  Read the code for details,
        but in general, life is easier if ntout divides into nsteps evenly.

        where possible, end_is_last_output determines how to choose the exact
        definition of the end date.  
        """
        start_fn = self.file_path('StartFile',0)
        store_fn = self.file_path('StoreFile',0)

        step_data = self.step_data()

        if os.path.lexists(start_fn):
            if os.path.exists(store_fn):
                # I don't remember the exact reason that it's better to
                # use storefiles, except that when runs are moved around,
                # storefiles are more likely to still exist, while links to
                # startfiles get broken.  But it's possible that the storefile
                # is empty, it's not a restart, and we'd have been better off
                # to reader

                # From a storefile and step_data, can work back to get
                # starting time
                sf = StoreFile(self,processor=0)
                last_output_date = sf.time()

                grid_outputs = step_data['steps_output']
                # note that this is duration from first to last output.
                run_duration = self.conf.timestep() * int(self.conf['ntout']) * (grid_outputs-1)
                start_date = last_output_date - run_duration
            elif os.path.exists(start_fn):
                # So it's presumably restart:
                start_date,end_date = self.conf.simulation_period()

                if self.starting_time_step()==0:
                    raise Exception("%s looks like a restart, but can't find Start or Store file"%self.datadir)

                restart_offset = self.starting_time_step() * self.conf.timestep()
                start_date += restart_offset
            else:
                raise Exception("Looks like a restart, but store and start files are missing")
        else:
            # presumably not a restart, and the configured period is what we want.
            start_date,end_date = self.conf.simulation_period()
            
        nsteps = int(self.conf['nsteps'])
        ntout  = int(self.conf['ntout'])
        
        if end_is_last_output:
            # round down to integer number of ntout periods:
            duration = self.conf.timestep() * ntout * (nsteps//ntout)
        else:
            duration = self.conf.timestep() * nsteps
            
        end_date = start_date + duration
        
        return start_date,end_date

    def starting_time_step(self):
        if self._starting_time_step is None:
            start_file = self.file_path('StartFile',0)
            if os.path.exists(start_file):
                # print "This is a restart."
                fp = open(start_file,'rb')
                x = fromstring(fp.read(4),int32)
                fp.close()
                self._starting_time_step = x[0]
            else:
                self._starting_time_step = 0
        return self._starting_time_step

    def parent_datadir(self):
        """ If this is a restart, return the datadir of the original
        (if possible - this requires that StartFile is a symlink!)
        if this is a restart, but the parent run can't be found, returns
        -1.
        if it's not a restart, return False
        """
        start_file = self.file_path('StartFile',0)
        if os.path.exists(start_file):
            if os.path.islink(start_file):
                parent_dir = os.path.dirname( os.path.realpath( start_file ) )
                if parent_dir == self.datadir:
                    # shouldn't ever happen, but occasionally runs are corrupted like this.
                    return False
                else:
                    return parent_dir
            else:
                print("  It's a restart, but no symlink")
                return -1
        else:
            return False

    _restarts = None
    def chain_restarts(self,max_count=None):
        """ returns something that can be iterated over to get
        sunreader instances ending with this
        one.  only goes back as far as the symlink trail of startfiles will
        allow
        the last one will be self - not a copy of self.
        """
        if self._restarts is None:
            suns = []
            sun = self
            while sun is not None and (max_count is None or len(suns) < max_count):
                suns.insert(0,sun)

                parent_dir = sun.parent_datadir()
                # print "Checking parent_dir",parent_dir
                if isinstance(parent_dir,str):
                    sun = SunReader(parent_dir)
                else:
                    sun = None

            print("Found %d chained restarts"%len(suns))
            if max_count is not None: # only cache this when we got all of them
                self._restarts = suns
            else:
                return suns
            
        if max_count is not None and len(self._restarts > max_count):
            return self._restarts[-max_count:]
        else:
            return self._restarts

    def effective_time_zero(self):
        """ return datetime for the when t=0, adjusted to be consistent with
        possible changes in dt between restarts.
        """
        suns = self.chain_restarts()[:-1]

        if len(suns) > 0:
            # The step that each one ended on
            ending_steps = array( [StoreFile(s,0).timestep() for s in suns] )

            num_steps = ending_steps.copy()
            # here we assume that first that we got back (which may *not* be the
            # actual first simulation - we may have lost the trail of symlinks)
            # has a timestep representative of all runs up to that point
            num_steps[1:] -= num_steps[:-1]

            # And the possibly varying dt
            dts   = array( [s.conf_float('dt') for s in suns] )

            sim_seconds = num_steps*dts

            total_past_seconds = sum(sim_seconds)

            # This should be the real time of our start 
            t = suns[0].time_zero() + datetime.timedelta( total_past_seconds/(24.*3600.) )
            t0 = t - datetime.timedelta( self.starting_time_step()*self.conf_float('dt') / (24.*3600) )
        else:
            t0 = self.time_zero()

        return t0

    def bathymetry_offset(self):
        """ a bit hackish - a constant offset is subtracted from all NAVD88 bathymetry
        to ensure that there aren't issues with the way that suntans reads bathymetry
        as absolute value.

        this code peaks into the bathymetry processing code to intuit what offset was
        added to the bathymetry
        """
        return read_bathymetry_offset()
        
    # Datum helpers
    def srs_text(self):
        return "EPSG:26910"

    def srs(self):
        proj = osr.SpatialReference()
        proj.SetFromUserInput(self.srs_text())
        return proj
    
    def xform_suntans_to_nad83(self):
        nad83_proj = osr.SpatialReference()
        nad83_proj.SetFromUserInput('NAD83')
        xform = osr.CoordinateTransformation(self.srs(),nad83_proj)
        return xform
    
    def xform_nad83_to_suntans(self):
        nad83_proj = osr.SpatialReference()
        nad83_proj.SetFromUserInput('NAD83')
        xform = osr.CoordinateTransformation(nad83_proj,self.srs())
        return xform
    
    def xform_suntans_to_wgs84(self):
        wgs84_proj = osr.SpatialReference()
        wgs84_proj.SetFromUserInput('WGS84')
        xform = osr.CoordinateTransformation(self.srs(),wgs84_proj)
        return xform
    
    def xform_wgs84_to_suntans(self):
        wgs84_proj = osr.SpatialReference()
        wgs84_proj.SetFromUserInput('WGS84')
        xform = osr.CoordinateTransformation(wgs84_proj,self.srs())
        return xform

    def mllw_to_navd88(self,utm_locs):
        return self.vdatum_for_utm('MLLW','NAVD88',utm_locs)
    def msl_to_navd88(self,utm_locs):
        return self.vdatum_for_utm('LMSL','NAVD88',utm_locs)
    def vdatum_for_utm(self,src_vdatum,dest_vdatum,utm_locs):
        """ given a vector of utm xy pairs, return the height that must be added to go from
        the first vertical datum to the second.
        """

        lonlat_locs = zeros( utm_locs.shape, float64 )
        xform = self.xform_suntans_to_nad83()

        for i in range(utm_locs.shape[0]):
            lon,lat,dummy = xform.TransformPoint(utm_locs[i,0],utm_locs[i,1])
            lonlat_locs[i,:] = [lon,lat]

        return apply_vdatum(src_vdatum,dest_vdatum,lonlat_locs)


    def plot_bathymetry(self,procs=None,ufunction=None,**kwargs):
        def f(proc):
            bath,gr = self.read_bathymetry(proc)
            if ufunction:
                bath = ufunction(bath)
            return bath

        return self.plot_scalar(f,procs,**kwargs)

    def plot_edge_vector(self,u,proc,offset=0.0,**kwargs):
        """ quiver plot on edges.  u is a scalar, and the vector will be
        constructed using the edge normals
        offset shifts the origin along the edge to facilitate
        multiple vectors on one edge
        kwargs passed on to quiver.
        """
        if u.ndim != 1:
            raise Exception("velocity vector has shape %s - should be 1-D"%str(u.shape))
        
        # g = self.grid(proc)
        edata = self.edgedata(proc)
        vec_u = u[:,newaxis] * edata[:,2:4]
        vec_origin = edata[:,4:6] + offset*(edata[:,0])[:,newaxis] * (edata[:,3:1:-1] * [-1,1])
        # slide origin over so we can put multiple arrows on
        # the same edge:

        # g.plot()
        quiver(vec_origin[:,0],
               vec_origin[:,1],
               vec_u[:,0],
               vec_u[:,1],
               **kwargs)
        
    def plot_scalar(self,scalar_for_proc,procs=None,clip=None,cmap=None,vmin=None,vmax=None):
        """ takes care of setting all regions to the same
        normalization scale.  nan valued cells will be skipped
        """
        if procs is None:
            procs = range(self.num_processors())

        pdatas = []
        clim = [ inf,-inf]
        for proc in procs:
            gr = self.grid(proc)
            scalar = scalar_for_proc(proc)

            pdata = gr.plot_scalar( scalar, clip=clip,cmap=cmap )

            if pdata:
                valid = isfinite(scalar)
                if any(valid):
                    this_clim = scalar[valid].min(), scalar[valid].max()
                    if this_clim[0] < clim[0]:
                        clim[0] = this_clim[0]
                    if this_clim[1] > clim[1]:
                        clim[1] = this_clim[1]

                pdatas.append( pdata )

        if vmin is not None:
            clim[0] = vmin
        if vmax is not None:
            clim[1] = vmax
            
        for p in pdatas:
            p.set_clim( *clim )

        return pdatas[0]

    def count_3d_cells(self):
        count = 0
        for p in range(self.num_processors()):
            cdata = self.celldata(p)
            ng = self.proc_nonghost_cells(p)
            count += cdata[ng,4].sum()
        return count
            
    def celldata(self,proc):
        # 0,1: x,y voronoi center
        # 2: area
        # 3: depth at voronoi center
        # 4: Nk - number of levels
        # 5-7: edge indexes
        # 8-10: cell neighbors
        # 11-13: dot(cell-outware,edge-nx/ny)
        # 14-16: distances from edges to voronoi center
        if self._shared_grid:
            return self._shared_grid.celldata(proc)
        
        if proc not in self._celldata:
            Ncells = self.grid(proc).Ncells()
            celldata_name = self.file_path("celldata",proc)

            fp = open(celldata_name,"rb")
            
            cdata = []

            for i in range(Ncells):
                cdata.append( list( [float(s) for s in fp.readline().split() ] ) )

            cdata = array(cdata)
            
            self._celldata[proc] = cdata
        return self._celldata[proc]
    
    def write_celldata(self,processor):
        f = self.file_path('celldata',processor)    
        fp = open(f,'wt')
        g = self.grid(processor)
        Nc = g.Ncells()
        cdata = self.celldata(processor)
        for i in range(Nc):
            fp.write("%.6f %.6f  %.6f  %.6f  %d  %d %d %d  %d %d %d  %d %d %d %.6f %.6f %.6f\n"%tuple(cdata[i,:]))
        fp.close()

    def Nk(self,proc):
        return self.celldata(proc)[:,4].astype(int32)

    def Nke(self,proc):
        return self.edgedata(proc)[:,6].astype(int32)

    def Nkc(self,proc):
        return self.edgedata(proc)[:,7].astype(int32)


    def edge_bathymetry(self,proc):
        """ If edge depths are defined separately, return an
        xyz array of them for the given processor.  Otherwise,
        return None
        """
        fn = self.file_path('edgedepths',proc)
        if os.path.exists(fn):
            return loadtxt(fn)

    def read_bathymetry(self,proc):
        gr = self.grid(proc)
        return self.celldata(proc)[:,3],gr
        
        
    def depth_for_cell(self,cell_id,proc):
        bath,gr = self.read_bathymetry(proc)
        return bath[cell_id]

    dzmin=0.001 # someday could read this from a file...

    _dz = None
    def dz(self):
        if self._dz is None:
            vsp = open(self.file_path('vertspace'),'rt')
            self._dz = array(list(map(float,vsp.read().split())))
        return self._dz

    def z_levels(self):
        """ returns bottoms of the z-levels, but as *soundings*, and not
        including z=0.
        """
        if self._z_levels is None:
            dz_list = self.dz()
            self._z_levels = cumsum(dz_list)
        return self._z_levels

    def primary_boundary_datasource(self):
        """ return the datasource that forces the largest number of edges.
        This is a hack to find the datasource that forces the freesurface on
        the ocean boundary.
        """
        best_fg = None
        
        for proc in range(self.num_processors()):
            f = forcing.read_boundaries_dat(self,proc)

            for fg in f.forcing_groups:
                if best_fg is None or len(best_fg.edges) < len(fg.edges):
                    best_fg = fg
                    
        ds_index = best_fg.hydro_datasource()
        return best_fg.gforce.datasources[ds_index]
            
    def boundary_forcing(self,proc=None):
        """ if proc is not given, return forcing from the first processor that has
        some forced cells

        this could probably get factored into boundary_inputs:BoundaryWriter
        """
        if proc is None:
            for proc in range(self.num_processors()):
                f = forcing.read_boundaries_dat(self,proc)
                if f.has_forced_edges():
                    return f
        else:
            return forcing.read_boundaries_dat(self,proc)
            

    def topo_edges(self):
        if self._topo_edges is None:
            self._topo_edges = []
            
            for p in range(self.num_processors()):
                fp = open(self.file_path('topology',p),'rt')

                def tokgen():
                    while 1:
                        buff = fp.readline()
                        if buff == "":
                            return
                        for t in buff.split():
                            yield int(t)

                tok = tokgen()
                nprocs = next(tok)
                nneighs = next(tok)
                for n in range(nneighs):
                    neigh = next(tok)
                    if p < neigh:
                        self._topo_edges.append( (p,neigh) )
        return self._topo_edges
        
    def show_topology(self,procs_per_node=4,topo_edges=None):

        if topo_edges is None:
            topo_edges = self.topo_edges()

        # load the graph:
        # the graph is stored just as a set of edges, with
        # the processors numbered 0-<nprocs-1>

        cla()
        
        
        # graph the processor connectivity graph:
        # round up:
        n_nodes = 1 + (self.num_processors()-1)//procs_per_node
        
        nodes = 2*arange(n_nodes) # space nodes out twice as much as cores
        cores = arange(procs_per_node)
        
        x,y = meshgrid(cores,nodes)
        # I want an array that maps proc_number to an xy pair
        proc_locs = transpose( array( (x.ravel(),y.ravel()), float64 ))

        # and randomly perturb so we can see all the lines:
        proc_locs[:,:1] = proc_locs[:,:1] + 0.4*(random( proc_locs[:,:1].shape ) - 0.5)

        # now proc 0-3 are on a line, 4-7, etc.
        for i in range(self.num_processors()):
            pylab.annotate( "%i"%i, proc_locs[i] )
        pylab.plot(proc_locs[:,0],proc_locs[:,1],'ro')

        for e in topo_edges:
            locs = proc_locs[array(e),:]
            pylab.plot( locs[:,0],locs[:,1],'b-' )
        pylab.axis('equal')
        x1,x2,y1,y2 = pylab.axis()
        x1 = x1 - 0.05*(x2 - x1)
        x2 = x2 + 0.05*(x2 - x1)
        y1 = y1 - 0.05*(y2 - y1)
        y2 = y2 + 0.05*(y2 - y1)
        
        pylab.axis( [x1,x2,y1,y2] )

    def remap_processors(self,procs_per_node=4,do_plot=False):
        import pymetis

        # create the adjacency graph in the way that
        # pymetis likes it
        adj = [None]*s.num_processors()
        topo_edges = s.topo_edges()

        for a,b in topo_edges:
            if adj[a] is None:
                adj[a] = []
            if adj[b] is None:
                adj[b] = []
            adj[a].append(b)
            adj[b].append(a)

        n_nodes = 1 + old_div((s.num_processors() - 1), procs_per_node)

        cuts,parts = pymetis.part_graph(n_nodes,adjacency=adj)
        print(parts)
        # create a mapping of old proc nunmber to new proc number

        #parts = array(parts)
        mapping = -1*ones(s.num_processors())
        # mapping[i] gives the new processor number for proc i
        count_per_node = zeros(n_nodes)
        for i in range(len(parts)):
            # old proc i
            my_node = parts[i]
            new_proc = my_node * procs_per_node + count_per_node[my_node]
            mapping[i] = new_proc

            count_per_node[my_node]+=1

        # now create a new topo-edges array so we can graph this...
        new_topo_edges = mapping[array(s.topo_edges())]
        new_topo_edges = new_topo_edges.astype(int32)

        if do_plot:
            pylab.clf()
            pylab.subplot(121)
            s.show_topology()
            pylab.subplot(122)
            s.show_topology(topo_edges=new_topo_edges)
        

    def parse_output(self,output_name=None):
        """
        reads the output from a run, hopefully with at least
        -vv verbosity.
        If the run crashed, sets self.crash to a crash object
        Sets self.status to one of 'done','crash','running'

        this is a work in progress (but you knew that, right?)
        """
        if output_name is None:
            output_name = sun_dir+'/output'
        run_output = open(output_name)

        self.status = 'running'
        self.crash = None
        
        while 1:
            l = run_output.readline()
            if not l:
                break

            if l.find('Run is blowing up!') >= 0:
                self.status = 'crash'
                m = re.match(r'Time step (\d+): Processor (\d+), Run is blowing up!',l)
                if not m:
                    print("Failed to match against")
                    print(l)
                else:
                    # got a crash
                    crash = SuntansCrash()
                    crash.sun = self

                    crash.step = int(m.group(1))
                    crash.processor = int(m.group(2))

                    l = run_output.readline()

                    for i in range(100): # search for CFL details up to 100 lines away
                        
                        ### Vertical Courant number:
                        m = re.match(r'Courant number problems at \((\d+),(\d+)\), Wmax=([-0-9\.]+), dz=([0-9\.]+) CmaxW=([0-9\.]+) > ([0-9\.]+)',
                                     l)
                        if m:
                            crash.cell_id = int(m.group(1))
                            crash.z_id    = int(m.group(2))
                            crash.w_max   = float(m.group(3))
                            crash.dz      = float(m.group(4))
                            crash.cmax_w  = float(m.group(5))
                            crash.cmax_w_lim = float(m.group(6))
                            crash.description = SuntansCrash.vertical_courant
                            break
                        
                        ### Horizontal Courant number:
                        m = re.match(r'Courant number problems at \((\d+),(\d+)\), Umax=([-0-9\.]+), dx=([0-9\.]+) CmaxU=([0-9\.]+) > ([0-9\.]+)',
                                     l)
                        if m:
                            crash.edge_id = int(m.group(1))
                            crash.z_id    = int(m.group(2))
                            crash.u_max   = float(m.group(3))
                            crash.dx      = float(m.group(4))
                            crash.cmax_u  = float(m.group(5))
                            crash.cmax_u_lim = float(m.group(6))
                            crash.description = SuntansCrash.horizontal_courant
                            break

                        print("Hmm - maybe this isn't a vertical courant number issue")
                        l = run_output.readline()

                    self.crash = crash
                break
    def write_bov(self,label,proc,dims,data):
        """
        Write a binary file that will hopefully be readable by
        visit through some naming conventions, and can be read
        back into sunreader.
        label: a name containing no spaces or dashes that describes
          what the data is (e.g. m2_fs_amp )
        dims: a list identifying the dimensions in order.
           [z_level, cell, time_step]
           
        data: an array that matches the described dimensions.

        Currently there are only two grids defined in visit:
           2D cells
           3D cells, with z-level
        """
        # Enforce this ordering on the dimensions, which comes from the
        # ordering of dimensions in suntans scalar output
        required_order = ['time_step','z_level','cell']
        given_order = [required_order.index(s) for s in dims]
        if sorted(given_order) != given_order:
            raise Exception("Order of dimensions must be time_step, cell, z_level")

        # Enforce the expected size of each dimension:
        g = self.grid(proc)
        
        for i in range(len(dims)):
            if dims[i] == 'time_step':
                print("Assuming that number of timesteps is okay")
            elif dims[i] == 'cell' and data.shape[i] != g.Ncells():
                print("WARNING: cell dimension - data shape is %i but grid reports %i"%(data.shape[i],g.Ncells()))
            elif dims[i] == 'z_level' and data.shape[i] != self.conf_int('Nkmax'):
                print("WARNING: z_level dimension - data shape is %i but Nkmax is %i"%(
                    data.shape[i],self.conf_int('Nkmax')))
            
        if data.dtype != float64:
            print("Converting to 64-bit floats")
            data = data.astype(float64)
        
        formatted_name = os.path.join(self.datadir,label + "-" + "-".join(dims) + ".raw.%i"%proc)
        print("Writing to %s"%formatted_name)
        fp = open(formatted_name,'wb')
        fp.write(data.tostring())
        fp.close()

    def harm_decomposition(self,consts=['constant','M2'],ref_data=None,phase_units='minutes',
                           skip=0.5):
        """  Perform a harmonic decomposition on the freesurface, using
        the given constituents, and write the results to
        <const name>_<phase or amp>-cell.raw.<proc>

        Phase is relative to cos(t), t in simulation time.

        At some point ref_data may be used to specify a timeseries that can also
        be decomposed, and whose amp/phase will be used as a reference for normalizing
          the others...

        or set ref_data='forcing' to take the reference to be the forcing on the first forced
           cell (i.e. it will loop over processors, and take the first cell with forcing data)
        """
        import harm_field

        if ref_data == 'forcing':
            # this matches the timeline used in harm_field:
            # times of freesurface output, using the second half of the run.
            # for forcing it would be okay to use the entire run, but I
            # think it's more consistent to decompose the forcing at the
            # same times as the cell values
            t = self.timeline()[self.steps_available()//2:] 

            forcing = None
            for proc in range(self.num_processors()):
                forcing = self.boundary_forcing(proc)
                if forcing.n_bcells > 0:
                    print("Getting forcing data for boundary cell 0, processor %s"%proc)
                    ref_data = forcing.calc_forcing(times=t,units='seconds')
                    break
            if forcing is None:
                raise Exception("No forced boundary cells were found")

        if ref_data:
            ref_t,ref_vals = ref_data

            # need to get the right omegas, a bit kludgy.
            import harm_plot,harm_decomp
            hplot = harm_plot.HarmonicPlot(sun=self,consts=consts)
            my_omegas = hplot.omegas
            print("Calculating decomposition for forcing data")
            ref_comps = harm_decomp.decompose(ref_t,ref_vals,my_omegas)
        else:
            ref_comps = None

            
        for proc in range(self.num_processors()):
            print("Decomposition for processor %i"%proc)
            
            harm_field
            hplot = harm_field.HarmonicField(self,proc=proc,consts=consts,skip=skip)
            print("  Calculating decomposition")
            amps,phase = hplot.calc_harmonics()

            if ref_comps is not None:
                amps = amps/ref_comps[:,0]
                phase = phase - ref_comps[:,1]

            for i in range(len(consts)):
                print("  Writing %s"%consts[i])
                self.write_bov('%s_amp'%consts[i],proc=proc,dims=['cell'],data=amps[:,i])

                phase_data = phase[:,i]
                if phase_units == 'radians':
                    pass
                elif phase_units == 'degrees':
                    phase_data *= (180./pi)
                elif phase_units == 'minutes':
                    omega = hplot.omegas[i]
                    if omega > 0.0:
                        phase_data *= 1.0 / (60.0*omega)
                
                self.write_bov('%s_phase'%consts[i],proc=proc,dims=['cell'],data=phase_data)

    def edgedata(self,processor):
        # 0: edge length
        # 1: dg, voronoi length
        # 2,3: nx, ny, edge normal
        # 4,5: x,y position of center
        # 6: Nke
        # 7: Nkc
        # 8: cell nc1 - [nx,ny] points *toward* this cell.
        # 9: cell nc2 - upwind cell for positive face velocity
        # 10,11: gradf - face number for nc1, nc2
        # 12: marker
        # 13,14: point indexes

        if self._shared_grid:
            return self._shared_grid.edgedata(processor)
        
        if processor not in self._edgedata:
            g = self.grid(processor)
            Ne = g.Nedges()
            
            f = self.file_path('edgedata',processor)    
            fp = open(f,'rt')

            edgedata = zeros([Ne,15],float64)
            for i in range(Ne):
                edgedata[i,:] = list(map(float,fp.readline().split()))
            self._edgedata[processor] = edgedata
        return self._edgedata[processor]
    
    def write_edgedata(self,processor):
        f = self.file_path('edgedata',processor)    
        fp = open(f,'wt')
        g = self.grid(processor)
        Ne = g.Nedges()
        edata = self.edgedata(processor)
        for i in range(Ne):
            fp.write("%f %f %f %f %f %f %d %d %d %d %d %d %d %d %d\n"%tuple(edata[i,:]))
        fp.close()
            
    def write_random_dataxy(self,fraction=0.05):
        """ randomly choose the given fraction of cells, and write a set of profile
        points (dataxy.dat) accordingly.
        """
        # refuse to overwrite an existing one:
        dataxy_path = self.file_path('DataLocations')
        if os.path.exists(dataxy_path):
            raise Exception("Please remove existing profile locations file %s first"%dataxy_path)
        fp = open(dataxy_path,'wt')
        
        for proc in range(self.num_processors()):
            print("choosing cell centers from processor %i"%proc)
            g = self.grid(proc)

            sampling = random( g.Ncells() )
            chosen = find( sampling < fraction )
            chosen_points = g.vcenters()[chosen]
            
            for winner in chosen_points:
                fp.write( "%g %g\n"%(int(winner[0]),int(winner[1])) )
        fp.close()


    def write_initial_salinity(self,func,dimensions=3,func_by_index=0):
        """ write salinity initialization for all processors based on the
        values returned by func
        dimensions=3: full field, func = func(x,y,z)
        dimensions=1: vertical profile, func = func(z)
        """
        self.write_initial_cond(func,'InitSalinityFile',dimensions,func_by_index=func_by_index)
        
    def write_initial_temperature(self,func,dimensions=3,func_by_index=0):
        self.write_initial_cond(func,'InitTemperatureFile',dimensions,func_by_index=func_by_index)

    def copy_scalar_to_initial_cond(self,sun_in,
                                    scalar_file_in='SalinityFile',
                                    scalar_file_out='InitSalinityFile',
                                    step=-1):
        """ For copying the output of one run to the initialization of another.
        Right now, the grids must be identical, including the subdomain decomposition
        which makes this method pretty much useless since suntans can just do a restart

        sun_in: a sun instance for the last run
        
        scalar_file_in: which suntans.dat field has the filename for reading the
          last run's scalar

        scalar_file_out: which suntans.dat field has the filename for the scalar the
          we are writing out

        step: number of the step to read from the last run.  negative means count
          back from the end.
        """

        # First, some sanity checks:
        if self.num_processors() != sun_in.num_processors():
            raise Exception("Mismatch in number of processors: %d vs %d"%(self.num_processors(),
                                                                          sun_in.num_processors()))
        
        for proc in range(self.num_processors()):
            fname_in = sun_in.file_path(scalar_file_in,proc)
            fname_out = self.file_path(scalar_file_out,proc)
            print("Transcribing scalar from %s to %s"%(fname_in,fname_out))

            # Read the old data:
            g_in,scalar_in = sun_in.cell_scalar(scalar_file_in,proc,step)
            g_out = self.grid(proc)

            # more sanity checks:
            if any(g_in.cells != g_out.cells):
                raise Exception("Cell arrays don't match!")
            
            Nc = self.grid(proc).Ncells()
            Nk = self.Nk(proc)

            column_starts = Nk.cumsum() - Nk
            scalar = zeros( Nk.sum(), REALTYPE )

            for i in range(Nc):
                for k in range(Nk[i]):
                    # not positive about the ordering of indices for scalar_in
                    scalar[column_starts[i] + k] = scalar_in[i,k]
            fp = open(fname,'wb')
            fp.write( scalar.tostring() )
            fp.close()
        
    def write_initial_cond(self,func,scalar_file='InitSalinityFile',dimensions=3,func_by_index=0):
        """ need to interpret func_by_index - if true, then the function actually takes
        a cell and k, and returns the value, not by physical coordinates.
        """

        if not func_by_index:
            # vertical locations
            nkmax = self.conf_int('Nkmax')
            z_levels = concatenate( ([0],self.z_levels()) )
            mid_elevations = 0.5*(z_levels[1:] + z_levels[:-1])

        if dimensions == 3:
            for proc in range(self.num_processors()):
                fname = self.file_path(scalar_file,proc)
                print("Writing initial condition to %s"%fname)

                g = self.grid(proc)
                if not func_by_index:
                    # these are the xy locations
                    centers = g.vcenters()
                Nc = self.grid(proc).Ncells()
                Nk = self.Nk(proc)

                column_starts = Nk.cumsum() - Nk
                scalar = zeros( Nk.sum(), REALTYPE )

                for i in range(Nc):
                    for k in range(Nk[i]):
                        if func_by_index:
                            scalar[column_starts[i] + k] = func(proc,i,k)
                        else:
                            scalar[column_starts[i] + k] = func(centers[i,0],centers[i,1],mid_elevations[k])
                fp = open(fname,'wb')
                fp.write( scalar.tostring() )
                fp.close()
        else: # dimensions == 1
            fname = self.file_path(scalar_file)
            scalar = zeros( nkmax, REALTYPE)
            for k in range(nkmax):
                scalar[k] = func(mid_elevations[k])

            fp = open(fname,'wb')
            fp.write( scalar.tostring() )
            fp.close()

    def read_cell_scalar_log(self,label,proc=None,time_step=None):
        """ a bit of a hack - this is for reading data logged by the logger.c functions.
        needs to be abstracted out to the same can be used for reading scalar initialization,
        or even more general where we can specify whether all z-levels or only valid z-levels
        are in the file (then it could be used for reading scalar output, too)
        """
        
        if time_step is not None:
            fname = '%s-time_step-cell.raw'%label
        else:
            fname = '%s-cell.raw'%label

        fname = os.path.join(self.datadir,fname)
        
        if proc is not None:
            fname += '.%d'%proc

        fp = open(fname,'rb')

        Nc = self.grid(proc).Ncells()
        real_size = 8

        time_stride = real_size * Nc

        if time_step:
            fp.seek(time_stride*time_step)

        cell_scalar = fromstring(fp.read(time_stride),float64)
        
        return cell_scalar
        


    def read_edge_scalar(self,label,proc=None,time_step=None):
        """ if time_step is None, assume it's not time varying.
        if time_step is 'all', then read all timesteps.
        otherwise, read a single timestep as specified
        """
        if time_step is not None:
            fname = '%s-time_step-edge.raw'%label
        else:
            fname = '%s-edge.raw'%label

        if proc is not None:
            fname += '.%d'%proc

        Ne = self.grid(proc).Nedges()
        real_size = 8

        full_fname = os.path.join(self.datadir,fname)
        if time_step == 'all': # try memory mapping:
            frame_size = Ne * REALSIZE
            nbytes = os.stat(full_fname).st_size
            data_shape = (nbytes//frame_size,Ne)

            edge_scalar = memmap(full_fname, dtype=REALTYPE, mode='r', shape=data_shape)
        else:
            fp = open(full_fname,'rb')
            time_stride = real_size * Ne
            if time_step:
                fp.seek(time_stride*time_step)

            edge_scalar = fromstring(fp.read(time_stride),float64)
            fp.close()
        return edge_scalar
        
    def read_edge_z_level_scalar(self,label,proc=None,time_step=None):
        if time_step is not None:
            fname = '%s-time_step-edge-z_level.raw'%label
        else:
            fname = '%s-edge-z_level.dat'%label

        if proc is not None:
            fname += '.%d'%proc

        fp = open(os.path.join(self.datadir,fname),'rb')

        Nkc = self.Nkc(proc)
        Ne = self.grid(proc).Nedges()
        real_size = 8

        time_stride = real_size * Nkc.sum()

        if time_step:
            fp.seek(time_stride*time_step)

        raw_scalar = fromstring(fp.read(time_stride),float64)

        # rewrite scalar in a nice x[j][k] sort of way.  Note that this means
        # the [j] array is a list, not a numpy array:
        column_starts = Nkc.cumsum() - Nkc

        edgescalar = [raw_scalar[column_starts[i]:column_starts[i]+Nkc[i]] for i in range(Ne)]
        return edgescalar

    def read_cell_z_level_scalar(self,label=None,fname=None,proc=None,time_step=None):
        """ if fname is None, determines a file name based on the label and the structure
        of the data. Note that if fname is specified, proc must still be specified as needed
        but it will *not* be appended to the filename
        """
        if fname is None:
            if time_step is not None:
                fname = '%s-time_step-cell-z_level.raw'%label
            else:
                fname = '%s-cell-z_level.raw'%label

            if proc is not None:
                fname += '.%d'%proc

        fp = open(os.path.join(self.datadir,fname),'rb')

        Nk = self.Nk(proc)
        Nc = self.grid(proc).Ncells()
        real_size = 8

        time_stride = real_size * Nk.sum()

        if time_step:
            fp.seek(time_stride*time_step)

        raw_scalar = fromstring(fp.read(time_stride),float64)

        # rewrite scalar in a nice x[j][k] sort of way.  Note that this means
        # the [j] array is a list, not a numpy array:
        column_starts = Nk.cumsum() - Nk

        cellscalar = [raw_scalar[column_starts[i]:column_starts[i]+Nk[i]] for i in range(Nc)]
        return cellscalar

    def closest_profile_point(self,p):
        pnts = self.profile_points()
        
        dists = sqrt( (pnts[:,0]-p[0])**2 + (pnts[:,1]-p[1])**2 )
        return argmin(dists)
        

    def closest_cell(self,xy,full=0):
        """ Return proc,cell_id for the closest cell to the given point, across
        all processors.  Slow...
         full==0: each subdomain will only consider cells that contain the closest global
           point.  as long as all points are part of a cell, this should be fine.
         full==1: if the closest point isn't in a cell, consider *all* cells.  
        """
        ids = []
        dists = []

        # Read the full grid once, build its index.
        #print "Closest cell - reading full grid"
        gfull = self.grid()
        #print "building its index"
        gfull.build_index()

        i = gfull.closest_point(xy)
        #print "Got closest index %d"%i

        c_list = [] # a list of cell ids, one per processor
        dist_list = [] # the distances, same deal
        
        for p in range(self.num_processors()):
            #print "Checking on processor %i"%p
            #print "Reading its grid"
            g = self.grid(p)

            # c_p = g.closest_cell(xy,full=full)

            ## this is basically the body of trigrid::closest_cell()
            try:
                cells = list( g.pnt2cells(i) )
            except KeyError:
                if not full:
                    # note that this processor didn't have a good match
                    c_list.append(-1)
                    dist_list.append(inf)
                    continue
                else:
                    print("This must be on a subdomain.  The best point wasn't in one of our cells")
                    cells = list(range(g.Ncells()))

            # found some candidate cells -
            # choose based on distance to vcenter
            cell_centers = g.vcenters()[cells]
            dists = ((xy-cell_centers)**2).sum(axis=1)

            # closest cell on proc p
            chosen = cells[argmin(dists)]

            c_list.append( chosen )
            dist_list.append( dists.min() )

        # 
        dist_list = array(dist_list)
        best_p = argmin(dist_list)
        best_c = c_list[best_p]
        
        return best_p,best_c
    
    def grid_stats(self):
        """ Print a short summary of grid information
        This needs to be run on a domain that has already been decomposed
        into subdomains (such that celldata.dat.* files exist)
        """
        procs = list(range(self.num_processors()))

        twoD_cells = 0
        threeD_cells = 0
        
        for p in procs:
            print("...Reading subdomain %d"%p)
            g=self.grid(p)
            nonghost = self.proc_nonghost_cells(p)
            twoD_cells += len(nonghost)

            threeD_cells += sum(self.Nk(p)[nonghost])
            
        print("Total 2D cells: %d"%twoD_cells)
        print("Total 3D cells: %d"%threeD_cells)
        
        # Some distance metrics:
        dg_per_proc = []
        
        for p in procs:
            print("...Reading subdomain %d"%p)
            edgedata = self.edgedata(p)

            valid = (edgedata[:,12] == 0) | ((edgedata[:,12]==5) & (edgedata[:,8]>=0))
            dg_per_proc.append( edgedata[valid,1] )
            
        dg_total = concatenate(dg_per_proc)
        print("Spacing between adjacent voronoi centers [m]:")
        print("  Mean=%f  Median=%f  Min=%f  Max=%f"%(mean(dg_total),median(dg_total),
                                                      dg_total.min(), dg_total.max()))

    def calc_Cvol_distribution(self):
        """ from storefile, calculate some simple statistics for the distribution
        of volumetric courant number.  prints them out and returns a list of
        [ ('name',value), ... ]
        """

        # Choose a full-grid output to analyze - since we need the storefile
        # as well, we pretty much have to choose the last full grid output
        n = self.steps_available() - 1
        dt = self.conf_float('dt')

        all_Cvol = [] # will append single Cvol numbers
        all_df = []   # will append arrays for valid edges
        all_dg = []   # same

        dz = self.dz()
        dzmin_surface = 2*self.dzmin # the dzmin cutoff for merging a surface cell with the one below it.
        z = -self.z_levels() # elevation of the bottom of each z level



        for proc in range(self.num_processors()):
            print("Proc %d"%proc)    

            cdata = self.celldata(proc)
            edata = self.edgedata(proc)
            Ac = cdata[:,2]
            bed = -cdata[:,3]

            valid_edge = (edata[:,12] == 0) | ( (edata[:,12] == 5) & (edata[:,8]>=0))
            all_df.append( edata[valid_edge,0] )
            all_dg.append( edata[valid_edge,1] )

            store = StoreFile(self,proc)
            h = store.freesurface()
            u = store.u()

            # calculate dz, dzf
            g = self.grid(proc)
            ctops = self.h_to_ctop(h,dzmin_surface)
            etops = zeros( g.Nedges(), int32 )
            h_edge= zeros( g.Nedges(), float64 )
            bed_edge = zeros( g.Nedges(), float64 )
            Nke = edata[:,6].astype(int32)

            dzf = [None] * g.Nedges()

            for j in range( g.Nedges() ):
                # we don't have exact information here, but assume that the top, non-zero velocity
                # in u corresponds to the top edge.
                nc1,nc2 = edata[j,8], edata[j,9]
                if nc1 < 0:
                    nc = nc2
                    bed_edge[j] = bed[nc2]
                elif nc2 < 0:
                    nc = nc1
                    bed_edge[j] = bed[nc1]
                else:
                    bed_edge[j] = max( bed[nc1], bed[nc2] )
                    nz = nonzero( u[j] )[0]
                    if len(nz) > 0:
                        u_for_upwind = u[j][nz[0]]
                        if u_for_upwind > 0:
                            nc = nc2
                        else:
                            nc = nc1
                    else:
                        if h[nc1] > h[nc2]:
                            nc = nc1
                        else:
                            nc = nc2
                etops[j] = ctops[nc]
                h_edge[j] = h[nc]

                # fill in dzf:
                this_dzf = dz[:Nke[j]].copy() 
                this_dzf[:etops[j]] = 0.0 # empty cells above fs
                this_dzf[etops[j]] = h_edge[j] - z[etops[j]] # recalc surface cell
                this_dzf[Nke[j]-1] -= bed_edge[j] - z[Nke[j]-1] # trim bed layer (which could be same as surface)

                dzf[j] = this_dzf


            # Now loop through cells, and get volumetric courant numbers:
            for i in self.proc_nonghost_cells(proc): # range(g.Ncells()):
                Nk = int(cdata[i,4])
                dzc = dz[:Nk].copy()
                dzc[:ctops[i]] = 0.0 # empty cells above fs
                dzc[ctops[i]] = h[i] - z[ctops[i]] # recalc surface cell
                dzc[Nk-1] -= bed[i] - z[Nk-1] # trim bed layer (which could be same as surface)

                for k in range(ctops[i],Nk):
                    V = Ac[i] * dzc[k]
                    Q = 0
                    for j,normal in zip( cdata[i,5:8].astype(int32), cdata[i,11:14]):
                        df = edata[j,0]
                        if k == ctops[i]:
                            # surface cell gets flux from edges here and up
                            # (which may be below ctop, anyway)
                            kstart = etops[j]
                        else:
                            kstart = k

                        for kk in range(kstart,min(k+1,Nke[j])): 
                            # so now kk is guaranteed to be valid layer for the edge j
                            if u[j][kk] * normal > 0: # it's an outflow
                                Q += u[j][kk] * normal * df * dzf[j][kk]
                    C = Q * dt / V
                    all_Cvol.append(C)


        all_Cvol = array(all_Cvol)
        all_df = concatenate( all_df )
        all_dg = concatenate( all_dg )

        # and calculate the effect of substepping to get an effective mean Cvol:
        nsubsteps = ceil(all_Cvol.max())
        dt =  self.conf_float('dt')
        sub_dt = dt/nsubsteps
        mean_sub_Cvol = mean( all_Cvol/nsubsteps )

        # print "Mean Cvol: %f"%all_Cvol.mean()
        # print "Mean df: %f"%all_df.mean()
        # print "Mean dg: %f"%all_dg.mean()

        return [ ('Mean Cvol',all_Cvol.mean()),
                 ('Mean df',all_df.mean()),
                 ('Mean dg',all_dg.mean()),
                 ('dt',dt),
                 ('sub_dt',sub_dt),
                 ('nsubsteps',nsubsteps),
                 ('Mean subCvol',mean_sub_Cvol)]
    
    def all_dz(self,proc,time_step):
        """ Return a 2-D array of dz values all_dz[cell,k]
        dry cells are set to 0, and the bed and freesurface height are
        taken into account.  Useful for depth-integrating.
        """
        cdata = self.celldata(proc)
        
        Nk = cdata[:,4].astype(int32)
        bed = -cdata[:,3]

        all_dz = (self.dz())[newaxis,:].repeat(len(Nk),axis=0)
        
        z = -self.z_levels()

        h = self.freesurface(proc,[time_step])[0]
        ctops = self.h_to_ctop(h)

        for i in range(len(Nk)):
            all_dz[i,:ctops[i]] = 0.0 # empty cells above fs
            all_dz[i,Nk[i]:] = 0.0    # empty cells below bed
            all_dz[i,ctops[i]] = h[i] - z[ctops[i]] # recalc surface cell
            all_dz[i,Nk[i]-1] -= bed[i] - z[Nk[i]-1] # trim bed layer (which could be same as surface)
        return all_dz

    def averaging_weights(self,proc,time_step,ztop=None,zbottom=None,dz=None):
        """ Returns weights as array [Nk] to average over a cell-centered quantity
        for the range specified by ztop,zbottom, and dz.

        range is specified by 2 of the 3 of ztop, zbottom, dz, all non-negative.
        ztop: distance from freesurface
        zbottom: distance from bed
        dz: thickness

        if the result would be an empty region, return nans.

        this thing is slow! - lots of time in adjusting all_dz
        """
        cdata = self.celldata(proc)
        
        Nk = cdata[:,4].astype(int32)
        bed = -cdata[:,3]

        one_dz = self.dz()
        all_dz = one_dz[newaxis,:].repeat(len(cdata),axis=0)
        all_k = arange(len(one_dz))[None,:].repeat(len(cdata),axis=0)
        
        z = -self.z_levels()

        h = self.freesurface(proc,[time_step])[0]

        # adjust bed and 
        # 3 choices here..
        # try to clip to reasonable values at the same time:
        if ztop is not None:
            if ztop != 0:
                h = h - ztop # don't modify h
                # don't allow h to go below the bed
                h[ h<bed ] = bed
            if dz is not None:
                # don't allow bed to be below the real bed.
                bed = maximum( h - dz, bed)
        if zbottom is not None:
            # no clipping checks for zbottom yet.
            if zbottom != 0:
                bed = bed + zbottom # don't modify bed!
            if dz is not None:
                h = bed + dz

        # so now h and bed are elevations bounding the integration region
        
        ctops = self.h_to_ctop(h)
        # default h_to_ctop will use the dzmin appropriate for the surface,
        # but at the bed, it goes the other way - safest just to say dzmin=0,
        # and also clamp to known Nk
        cbeds = self.h_to_ctop(bed,dzmin=0) + 1 # it's an exclusive index
        cbeds[ cbeds > Nk ] = Nk[ cbeds > Nk ]

        if 0: # non vectorized - takes 0.8s per processor on the test run
            for i in range(len(Nk)):
                all_dz[i,:ctops[i]] = 0.0 # empty cells above fs
                all_dz[i,cbeds[i]:] = 0.0    # empty cells below bed - might be an error...
                all_dz[i,ctops[i]] = h[i] - z[ctops[i]] # recalc surface cell
                all_dz[i,cbeds[i]-1] -= bed[i] - z[cbeds[i]-1] # trim bed layer (which could be same as surface)
        else: # attempt to vectorize - about 70x speedup
            # seems that there is a problem with how dry cells are handled -
            # for the exploratorium display this ending up with a number of cells with
            # salinity close to 1e6.
            # in the case of a dry cell, ctop==cbed==Nk[i]
            # 
            drymask = (all_k < ctops[:,None]) | (all_k>=cbeds[:,None])
            all_dz[drymask] = 0.0
            ii = arange(len(cdata))
            all_dz[ii,ctops] = h - z[ctops]
            all_dz[ii,cbeds-1] -= bed - z[cbeds-1]

        
        # make those weighted averages
        # have to add extra axis to get broadcasting correct
        all_dz = all_dz / sum(all_dz,axis=1)[:,None]
        return all_dz

