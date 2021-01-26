"""
Tools for reading and plotting output from Ed Gross's FISH-PTM model

original code from Matt Rayson's soda library:
https://github.com/mrayson/soda

"""

import os
import time
import numpy as np
import xarray as xr
from datetime import datetime
import matplotlib.pyplot as plt
from ...spatial import wkb2shp
from ... import memoize, utils
import pandas as pd

class PtmBin(object):
    # when True, load particle data as memory map rather than np.fromstring.
    use_memmap=True
    fp=None
    def __init__(self,fn,release_name=None,idx_fn='auto'):
        self.fn = fn

        if release_name is None:
            release_name = os.path.basename(fn)
            release_name = release_name.replace("_bin.out","")
        self.release = release_name

        self.fp = open(self.fn,'rb')
        self.fn_bytes = os.stat(self.fn).st_size

        self.read_bin_header()

        # File Format:
        #  int32: Nattr number of attributes
        #  Nattr*[int32 char80 char80]: attribute index, type, name
        #  Ntimesteps* {
        #    6*int32: year, month, day, hour, minute, Npart(t)
        #    Npart(t)* {
        #       int32, 3*float64, int32: id, xyz, active

        self.offsets = {} # map timestep => start of date header for that timestep
        self.offsets[0] = self.fp.tell()

        if idx_fn=='auto':
            idx_fn=fn.replace('_bin.out','_bin.idx')
        if os.path.exists(idx_fn):
            self.idx_fn=idx_fn
            self.read_index()
        else:
            # Get the time information
            self.getTime()

    def __del__(self):
        if self.fp is not None:
            self.fp.close()
            self.fp=None

    def read_index(self):
        # 20x faster
        df=pd.read_csv(self.idx_fn,sep='\s+',
                       names=['year','month','day','hour','minute','offset','count']) # 5ms
        df['time']=pd.to_datetime(df[['year','month','day','hour','minute']])
        # mimic what getTime and scan would have done:
        self.time=df.time.dt.to_pydatetime()
        self.offsets=dict(zip(df.index.values,df.offset.values))
                          
    def read_bin_header(self):
        self.Nattr = int( np.fromstring(self.fp.read(4),np.int32) )

        # print "Nattr: ",self.Nattr

        atts = []
        for i in range(self.Nattr):
            idx = int( np.fromstring( self.fp.read(4), np.int32) )
            type_str = self.fp.read(80).strip()
            name_str = self.fp.read(80).strip()
            atts.append( (idx,type_str,name_str) )
        self.atts=atts

    def scan_to_timestep(self,ts):
        """ Return true if successful, False if ts is beyond end of file.
        Set the file pointer to the beginning of the requested timestep.
        if the beginning of that timestep is at or beyond the end of the file
        return False, signifying that ts does not exist.
        """
        
        if ts<0:
            nsteps=self.count_timesteps()
            ts=nsteps+ts
            assert ts>=0
            
        if ts not in self.offsets:
            for ts_scan in range(1,ts+1):
                if ts_scan not in self.offsets:
                    # if we don't have the offset of this step, go to the one
                    # before, and find out how big the previous frame was.
                    self.fp.seek( self.offsets[ts_scan-1])
                    tstep_header = np.fromstring( self.fp.read( 6*4 ), np.int32 )
                    Npart = tstep_header[5]
                    # print "Step %d has %d particles"%(ts_scan-1,Npart)
                    frame = 6*4 + Npart * (2*4 + 3*8)
                    self.offsets[ts_scan] = self.offsets[ts_scan-1] + frame
                    if self.offsets[ts_scan] >= self.fn_bytes:
                        #print "Hit end of file"
                        return False
        if self.offsets[ts] >= self.fn_bytes:
            return False

        self.fp.seek(self.offsets[ts])
        return True

    def count_timesteps(self):
        saved_pos = self.fp.tell()

        valid_ts = -1
        while 1:
            if self.scan_to_timestep(valid_ts+1):
                # next one is valid, keep going
                valid_ts += 1
            else:
                # valid_ts+1 doesn't exist, so valid_ts is the last valid timestep
                break

        self.fp.seek(saved_pos)
        # possible that this is 0!
        return valid_ts + 1

    def dt_seconds(self):
        """
        Return the bin file output interval in decimal seconds.
        """
        dnum1,data = self.read_timestep(0)
        dnum2,data = self.read_timestep(1)
        return (dnum2-dnum1).total_seconds()

    def read_timestep(self,ts=0):
        """ returns a datenum and the particle array
        """
        if not self.scan_to_timestep(ts):
            return None,None

        # Read the time
        dnum,Npart = self.readTime()

        part_dtype = [('id','i4'),
                      ('x','3f8'),
                      ('active','i4')]
        part_size = 2*4 + 3*8

        # print "reading %d particles"%Npart
        if self.use_memmap:
            data=np.memmap( self.fn,dtype=part_dtype, offset=self.fp.tell(),
                            mode='r',shape=(Npart,) )
        else:
            data = np.fromstring( self.fp.read( part_size * Npart), dtype=part_dtype)
        return dnum,data

    def readTime(self):
        """
        Reads the time header for one step and returns a datetime object
        """
        tstep_header = np.fromstring( self.fp.read( 6*4 ), np.int32 )

        year,month,day,hour,minute,Npart = tstep_header

        if minute == 60:
            hour += 1
            minute = 0
        if hour == 24:
            hour = 0
            day += 1

        return datetime(year,month,day,hour,minute),Npart


    def getTime(self):
        """
        Returns a list of datetime objects
        """
        self.nt = self.count_timesteps()
        self.time=[]
        for ts in range(self.nt):
            self.scan_to_timestep(ts)
            t,npart = self.readTime()
            self.time.append(t)

    def plot(self,ts,ax=None,zoom='auto',fontcolor='k',update=True,
             mask=slice(None),marker='.',color='m',**kwargs):
        """
        Plots the current time step
        """

        # Check for the plot handle
        if (not update) or ('p_handle' not in self.__dict__):
            # Initialize the plot
            if ax==None:
                ax = plt.gca()
            h1 = ax.plot([],[],marker=marker,linestyle='None',color=color,**kwargs)
            self.p_handle=h1[0]
            self.title = ax.set_title("",fontdict={'color':fontcolor})

        # Now just update the plot
        t,parts = self.read_timestep(ts=ts)
        x = parts['x'][mask,0]
        y = parts['x'][mask,1]
        if zoom=='auto':
            zoom= [x.min(), x.max(), y.min(), y.max()]

        self.p_handle.set_xdata(x)
        self.p_handle.set_ydata(y)
        self.title=ax.set_title('Particle positions at %s'%(datetime.strftime(t,'%Y-%m-%d %H:%M:%S')))
        if zoom:
            ax.set_adjustable('datalim')
            ax.set_aspect('equal')
            ax.axis(zoom)

    # -- caching of particle -> cell mapping
    def grid_to_cell_cache_fn(self,grid):
        grid_key=memoize.memoize_key( grid.nodes['x'][ grid.cells['nodes'].clip(-1) ] )
        return self.fn+".cells-"+grid_key+".nc"

    def precompute_cells(self,grid,force=False):
        """
        The cached cell info is about 15% the size of the original
        bin.out file. ad-hoc binary and netcd yield about the same
        file size.

        grid: UnstructuredGrid. Will compute the cell index (or -1)
           for each particle at each time step.

        currently mapping a grid to a cache name is somewhat expensive,
         so if extensive access to cached data in performance critical
         sections are needed, this will need an option to directly
         specify the grid_key.

        force: when False, use cached data when possible, otherwise
          recompute.

        returns the cached filename, a netcdf file.
        """
        cell_cache_fn=self.grid_to_cell_cache_fn(grid)
        if not force and os.path.exists(cell_cache_fn):
            return cell_cache_fn

        n_steps=self.count_timesteps()
        dnums=[]
        xys=[]

        # loop once to gather all points
        for ts in utils.progress(range(n_steps)):
            dnum,parts=self.read_timestep(ts)
            if ts%100==0:
                print(f"{ts} {dnum} {len(parts)}")
            dnums.append(dnum)
            xys.append( parts['x'][:,:2].copy() )
        all_xy=np.concatenate(xys)

        # compute cells:
        t=time.time()
        # be sure to insert these as regular int
        all_cell=grid.points_to_cells(all_xy)
        elapsed=time.time() - t
        print("Python mapping time: %.3fs"%elapsed)

        ds=xr.Dataset()
        ds['cell']=('particle_loc',),all_cell.astype(np.int32)
        ds['dnum']=('time',),utils.to_dt64(np.array(dnums))
        counts=np.array( [len(xy) for xy in xys] )
        ds['count']=('time',),counts
        ds['offset']=('time',),np.cumsum(counts)-counts
        ds.to_netcdf(cell_cache_fn,mode='w')

        return cell_cache_fn

def release_log_dataframe(fn):
    """
    Parse release_log into pandas DataFrame
    """
    # for short releases, infer_datetime_format slows it down.
    # Might be faster if there are many releases.
    return pd.read_csv(fn,sep='\s+',
                       names=['id','gid','x','y','z','k','cell','date','time'],
                       parse_dates=[ ['date','time'] ],
                       infer_datetime_format=False)
    
class ReleaseLog(object):
    def __init__(self,fn):
        self.data=release_log_dataframe(fn)
        self.intervals=self.to_intervals(self.data)
    def to_intervals(self,data):
        # group by interval
        grped=data.groupby('date_time')
        intervals=pd.DataFrame(dict(time=grped['date_time'].first(),
                                    id_min=grped['id'].min(),
                                    id_max=grped['id'].max(),
                                    gid_min=grped['gid'].min(),
                                    gid_max=grped['gid'].max(),
                                    count=grped['id'].size()))
        return intervals
    
class PtmState(object):
    """
    Probably ought to share some code with PtmBin.  
    """
    REAL=np.float32
    step_header_dtype=np.dtype( [('year',np.int32),
                                 ('month',np.int32),
                                 ('day',np.int32),
                                 ('hour',np.int32),
                                 ('minute',np.int32),
                                 ('Npart',np.int32)] )
    
    def __init__(self,fn):
        self.fn=fn
        self.step_offsets={}
        self.fp=open(self.fn,'rb')

        self.read_header()
        
    def read_header(self):
        self.fp.seek(0)

        Nfields=np.fromfile(self.fp,np.int32,count=1)[0]
        attr_dtype=[('attr_id',np.int32),
                    ('name','S80')]
        attrs=np.fromfile(self.fp,attr_dtype,Nfields)

        
        part_dtype=[('id',np.int32),
                    ('x',np.float64,3),
                    ('status',np.int32)]

        for attr_name in attrs['name']:
            attr_name=attr_name.decode('latin').strip()
            if attr_name=='velocity':
                part_dtype.append( ('u',self.REAL,3) )
            elif attr_name=='depth_avg_vel':
                part_dtype.append( ('uavg',self.REAL,2) )
            elif attr_name=='hor_swim_vel':
                part_dtype.append( ('uswim',self.REAL,2) )
            else:
                part_dtype.append( (attr_name,self.REAL) )

        self.part_dtype=np.dtype(part_dtype)
        self.step_offsets[0]=self.fp.tell()

    def read_step_header(self):
        """
        read the per-timestep header from the current location of self.fp,
        leaving fp just after the header
        """
        return np.fromfile(self.fp,self.step_header_dtype,1)[0]
        
    def scan_to_step(self,step):
        start_step=step
        # find a known offset location
        while start_step not in self.step_offsets:
            assert start_step>=0
            start_step-=1
        while start_step<step:
            self.fp.seek(self.step_offsets[start_step])
            hdr=self.read_step_header()
            start_step+=1
            framesize=( self.step_header_dtype.itemsize+
                        hdr['Npart']*self.part_dtype.itemsize)
            self.step_offsets[start_step]=self.step_offsets[start_step-1]+framesize

        self.fp.seek(self.step_offsets[step])
    def read_step(self,step):
        self.scan_to_step(step)
        hdr=self.read_step_header()
        particles=np.fromfile(self.fp,self.part_dtype,hdr['Npart'])
        return hdr,particles
    def read_timestep(self,step):
        """
        make it look more like PtmBin 
        """
        hdr,particles=self.read_step(step)

        if hdr['minute'] == 60:
            hdr['hour'] += 1
            hdr['minute'] = 0
        if hdr['hour'] == 24:
            hdr['hour'] = 0
            hdr['day'] += 1

        return datetime(hdr['year'],hdr['month'],hdr['day'],hdr['hour'],hdr['minute']),particles

class PtmConcentration(object):
    """ Read time series concentration field output, e.g.
    instantaneous concentration and deposited
    """
    hdr_dtype=np.dtype( [('npoly','<i4'),('part_mass','<f4'), ('nodata','<f4'),
                         ('month','<i4'), ('day','<i4'), ('year','<i4'), ('hour','<f4')] )
    conc_dtype=np.dtype([('i','<i4'),('conc','<f4'),('vol','<f4')])
    
    def __init__(self,fn):
        self.fn=fn
        self.stride=0 # initilized below
        # go ahead and read the first record to know how large the records are.
        self.fp=open(self.fn,'rb')

        hdr=self.read_hdr()
        self.npoly=hdr['npoly']
        self.stride=self.hdr_dtype.itemsize + hdr['npoly']*self.conc_dtype.itemsize
        self.nsteps=os.stat(self.fn).st_size / self.stride
        
    def __del__(self):
        if self.fp is not None:
            self.fp.close()
            self.fp=None
        
    def seek_step(self,step):
        if self.stride==0:
            assert step==0
        assert step<self.nsteps,"Cannot seek to %d, only %d available"%(step,self.nsteps)
        self.fp.seek(step*self.stride)
        
    def read_hdr(self):
        # read first frame
        hdr=np.fromfile(self.fp, count=1, dtype=self.hdr_dtype)[0]
        return hdr
    
    def read_data(self):
        return np.fromfile(self.fp,count=self.npoly,dtype=self.conc_dtype)
    
    def read_timestep(self,step):
        self.seek_step(step)
        hdr=self.read_hdr()
        data=self.read_data()
        return hdr,data
            
def shp2pol(shpfile,outdir):
    """
    Converts a polygon shape file to a *.pol file

    Uses the first polygon in the file and the filename as the polygon name
    """
    # RH: converted to stompy methods, but untested
    poly=wkb2shp.shp2geom(shpfile)['geom'][0]
    polynameext =   os.path.basename(shpfile)
    polyname,ext = os.path.splitext(polynameext)
    outfile = '%s/%s.pol'%(outdir,polyname)

    geom2pol(poly,outfile)

def geom2pol(poly,outfile,polyname=None):
    xy=np.array(poly.exterior.coords)
    numverts = xy.shape[0]
    if polyname is None:
        polyname=os.path.basename(outfile).replace('.pol','')
        
    with open(outfile,'w') as f:
        f.write('POLYGON_NAME\n')
        f.write('%s\n'%polyname)
        f.write('NUMBER_OF_VERTICES\n')
        f.write('%d\n'%numverts)

        for ii in range(numverts):
            f.write('%6.10f %6.10f\n'%(xy[ii,0],xy[ii,1]))

def calc_agebin(binfile,ncfile,polyfile,ntout):
    """
    Calculate the from a binary file and save to netcdf
    """
    # Load the polygon from a shapefile
    # xypoly,field = readShpPoly(polyfile)
    recs=wkb2shp.shp2geom(polyfile)
    xypoly=[np.array( p.exterior.coords )
            for p in recs['geom']]

    # Load the binary file object
    PTM = PtmBin(binfile)

    # Count the number of particles from the first time step
    time,pdata = PTM.read_timestep()
    N = pdata.shape[0]
    import pdb
    pdb.set_trace()
    tsec = othertime.SecondsSince(PTM.time)
    dt = tsec[1] - tsec[0]

    # Initialize the age particle object
    raise Exception("Particle age has not been ported from soda/suntanspy")
    Age = ParticleAge(xypoly[0],N)

    # Loop through
    outctr = ntout
    ncctr=0
    for tt in range(PTM.nt-1):
        # Read the current time step
        time,pdata = PTM.read_timestep(ts=tt)

        # Update the age variable
        Age.update_age(pdata['x'][:,0],pdata['x'][:,1],pdata['x'][:,2],dt)

        # Write to netcdf
        if outctr==ntout:
            Age.write_nc(time,ncctr,ncfile=ncfile)
            ncctr+=1
            outctr=0
        outctr+=1

    print('Done.')

#hydrofile = '../InputFiles/untrim_hydro.nc'
#ptmfile = '../InputFiles/line_specify_bin.out'
#outfile = '../InputFiles/FISH_PTM.mov'
#
## Load the particle binary file
#pr = PtmBin(ptmfile)
#
#fig=plt.figure()
#ax=plt.gca()
#
## Load and plot the grid
#
#pr.plot(0,ax=ax)
#def updateLocation(ii):
#    pr.plot(ii,ax=ax)
#    return(pr.p_handle,pr.title)
#
#anim = animation.FuncAnimation(fig, updateLocation,\
#    frames=pr.nt, interval=2, blit=True)
#plt.show()
