"""
Tools for reading and plotting output from Ed Gross's FISH-PTM model

original code from Matt Rayson's soda library:
https://github.com/mrayson/soda

"""

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from ...spatial import wkb2shp
#from soda.utils.particles import ParticleAge
#import soda.utils.othertime

class PtmBin(object):
    def __init__(self,fn,release_name=None):
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

        # Get the time information
        self.getTime()

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
        dnum1,data = self.read_timestep(0)
        dnum2,data = self.read_timestep(1)
        return (dnum2-dnum1)

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


            
def shp2pol(shpfile,outdir):
    """
    Converts a polygon shape file to a *.pol file

    Uses the first polygon in the file and the filename as the polygon name
    """
    # RH: converted to stompy methods, but untested
    poly=wkb2shp.shp2geom(shpfile)['geom'][0]

    xy=np.array(poly.exterior.coords)
    numverts = xy.shape[0]

    polynameext =   os.path.basename(shpfile)
    polyname,ext = os.path.splitext(polynameext)

    outfile = '%s/%s.pol'%(outdir,polyname)

    print('Writing polygon to: %s...'%outfile)

    f = open(outfile,'w')
    f.write('POLYGON_NAME\n')
    f.write('%s\n'%polyname)
    f.write('NUMBER_OF_VERTICES\n')
    f.write('%d\n'%numverts)

    for ii in range(numverts):
        f.write('%6.10f %6.10f\n'%(xy[ii,0],xy[ii,1]))

    f.close()
    print('Done.')

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
