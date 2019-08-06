"""
Adaptation of sunreader.StoreFile, to work with updated SuntansModel
class.
"""
import logging as log
from ... import utils
import numpy as np

class StoreFile(object):
    """ Encapsulates reading of store.dat files, either for restarts or
    for crashes

    New: support for overwriting portions of a storefile
    """
    def __init__(self,model,proc,startfile=0,filename=None):
        """ startfile: if true, choose filename using StartFile 
        instead of StoreFile
        """
        self.sun = model
        self.proc = proc

        if filename is None:
            if startfile:
                self.fn = self.sun.file_path('StartFile',self.proc)
            else:
                self.fn = self.sun.file_path('StoreFile',self.proc)
        else:
            self.fn=filename

        try:
            self.fp = open(self.fn,'rb+')
        except OSError:
            log.warning("Could not open %s for update, will try read-only"%self.fn)
            self.fp = open(self.fn,'rb')

        # lazy loading of the strides, in case we just want the
        # timestep
        self.blocks_initialized = False

    def initialize_blocks(self):
        if not self.blocks_initialized:
            # all data is lazy-loaded
            self.grid = self.sun.subdomain_grid(self.proc)

            # pre-compute strides
            Nc = self.grid.Ncells()
            Ne_Nke = self.sun.Nke(self.proc).sum()
            Nc_Nk = self.sun.Nk(self.proc).sum()
            Nc_Nkp1 = (self.sun.Nk(self.proc) + 1).sum()

            REALTYPE=np.float64

            # and define the structure of the file:
            blocks = [
                ['timestep', np.int32, 1],
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
            block_sizes = np.array( [ np.ones(1,b[1]).itemsize * b[2] for b in blocks] )
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
            return np.fromstring( s, np.int32 )
        else:
            # 2014/7/13: this line was missing - not sure how it ever
            # worked - or if this is incomplete code??
            self.initialize_blocks()
            i = self.block_names.index(label)
            self.fp.seek( self.block_offsets[i] )
            s = self.fp.read( self.block_sizes[i] )
            return np.fromstring( s, self.block_types[i] )

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
        # scale to microseconds so numpy doesn't complain about
        # floating point time arithmetic, but we still get
        # sufficient resolution.
        dt=int(float(self.sun.config['dt']) * 1e6) * np.timedelta64(1,'us')
        return self.sun.time0+self.timestep() * dt

    def freesurface(self):
        return self.read_block('freesurface')

    def u(self):
        blk = self.read_block('u')
        Nke = self.sun.Nke(self.proc)
        Nke_cumul=Nke.cumsum() - Nke

        full_u=np.nan*np.ones((self.grid.Nedges(),Nke.max()))

        for e in range(self.grid.Nedges()):
            full_u[e,0:Nke[e]] = blk[Nke_cumul[e]:Nke_cumul[e]+Nke[e]]

        return full_u

    def overwrite_salinity(self,func):
        """ iterate over all the cells and set the salinity
        in each one, overwriting the existing salinity data

        signature for the function func(cell id, k-level)
        """
        # read the current data:
        salt=self.read_block('salinity')

        # for starters, just call out to func once for each
        # cell, but pass it the cell id so that func can
        # cache locations for each grid cell.

        i=0 # linear index into data
        Nk=self.sun.Nk(self.proc)

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

