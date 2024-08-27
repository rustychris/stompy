# Class for representing an RBR CTD
from __future__ import print_function

import numpy as np

import re
import seawater
import copy
import sqlite3
import pytz
import datetime
from matplotlib.dates import date2num,num2date
import xarray as xr
from .. import utils

class Calibration(object):
    """ a container for calibration information - these aren't
    actually used, though
    """
    def __init__(self,txt,coefs,units):
        self.txt = txt
        self.coefs = coefs
        self.units = units
    @staticmethod
    def from_dict(d):
        coefs=[]
        for ci in range(9):
            cN='c%d'%ci
            if cN not in d:
                break
            if d[cN] is not None:
                coefs.append(d[cN])
        cal = Calibration(txt='equation=%s'%d['equation'],coefs=coefs,units=d['units'])
        cal.equation=d['equation']
        return cal

class Rbr(object):
    def __init__(self,dat_file,instrument_tz=pytz.utc,target_tz=pytz.utc):
        """
        dat_file: path to a .hex or .rsk file.
        instrument_tz: a tzinfo object describing how to interpret times 
          in the dat file.  
        target_tz: a tzinfo describing what time zone to report results in.
        """
        self.timebases=[]
        self.filename=dat_file
        self.instrument_tz=instrument_tz
        self.target_tz=target_tz
        if self.filename:
            self.read()

    def read(self):
        raise Exception("use RbrHex or RbrRsk")

    def clean_name(self,s):
        return s.replace(' ','_').replace('-','_')

    def synthesize_fields(self):
        """ In case we have conductivity, temp and pressure but not salinity, calculate
        it here.
        """
        if 'Salinity' in self.columns:
            return
        
        if 'Cond' in self.columns and \
           'Temp' in self.columns:
            if 'Pres' in self.columns:
                Pres=self.Pres
            else:
                Pres=10.13 * np.ones_like(self.Cond)

            # conductivity starts as mS/cm, needs to be a ratio
            # to R =  C(S,T,P)/C(35,15(IPTS-68),0), which from
            # http://www.kayelaby.npl.co.uk/general_physics/2_7/2_7_9.html
            # is 42.90, I think...  some question as to whether S in the above
            # ratio is g/kg, or something else...
            salt = seawater.salt( self.Cond / 42.90,
                                  self.Temp,
                                  Pres )
            self.columns.append('Salinity')
            self.Ncolumns += 1
            self.data = np.concatenate( (self.data,salt[:,None]),axis=1)
            self.update_fields()

    def update_fields(self):
        """  set fields like self.t, self.cond, etc. which reference slices
        of self.data
        """
        self.t = self.data[:,0]
        self.column_vars=[]
        for c in range(self.Ncolumns):
            aname=self.clean_name(self.columns[c])
            self.column_vars.append(aname)
            setattr( self,aname, self.data[:,c+1] )

    def autotrim(self):
        """ Trim the timeseries to reflect when it looks like it was actually
        in the water
        """

        if 'Cond' in self.columns:
            p_anom = (self.Cond - median(self.Cond)) / self.Cond.std()

            good = np.where(abs(p_anom) < 0.5)[0]

            start = good[0]
            stop =  good[-1]
        elif 'Pres' in self.columns:
            p_anom = (self.Pres - median(self.Pres)) / self.Pres.std()

            good = np.where(abs(p_anom) < 0.1)[0]

            start = good[0]
            stop =  good[-1]
        else:
            print("%s doesn't have pressure or conductivity"%self.comment)
            return

        self.data = self.data[start:stop]
        self.update_fields()

    def remove_all_spikes(self,columns=['Cond','Salinity','SpecCond']):

        for c in columns:
            try:
                ci = self.columns.index(c)
            except ValueError:
                continue

            self.remove_spikes(ci)

    def remove_spikes(self,ci,method='d2',d2_threshold=30):
        """ attempt to automatically remove the spikes.
        not the best idea, but hopefully saves some time for a quick look
        at data

         d2_threshold=number of standard deviations in 2nd derivative
             to consider an outlier
        """

        if method == 'd2':
            while 1:
                d2 = np.diff(self.data[:,ci+1],2)
                d2_norm = np.abs(d2 / d2.std())

                bad = np.argmax(d2_norm)
                if d2_norm[bad] > 30:
                    self.data[bad+1,ci+1] = np.nan
                else:
                    break

    def to_xarray(self):
        ds=xr.Dataset()
        ds['time']=('time',),utils.to_dt64(self.data[:,0])
        for icol,col in enumerate(self.columns):
            ds[col]=('time',),self.data[:,1+icol]

        for field in ['averaging','instrument_tz','Nchannels',
                      'txt_logger_time','txt_host_time',
                      'txt_sample_period','txt_logging_start']:
            val=getattr(self,field)
            if field.startswith('txt'):
                val=val.strip()
            ds[field]=val
        return ds


class RbrRsk(Rbr):
    # dtype([('tstamp', '<i8'), ('channel01', '<f8'), ('channel02', '<f8'), ('channel03', '<f8'), ('channel04', '<f8')])
    #da=r.data_array()

    def read(self):
        self.conn = sqlite3.connect(self.filename)
        self.curs = self.conn.cursor()
        self.read_headers()
        self.read_calibrations()
        self.read_extras()
        self.read_data()


    def read_headers(self):
        self.txt_model = self.curs.execute('select model from instruments').fetchone()[0]
        self.ruskin = ('Ruskin' in self.txt_model)

        self.curs.execute('select startTime,endTime from epochs')
        self.epoch_start,self.epoch_end = self.curs.fetchone()

        # parse the ones that seem important:
        self.logging_start=self.instrument_tz.localize(
            datetime.datetime.fromtimestamp(self.epoch_start/1000.0))
        self.logging_end  =self.instrument_tz.localize(
            datetime.datetime.fromtimestamp(self.epoch_end/1000.0) )

        self.dt_s=self.curs.execute('select samplingPeriod/1000.0 from schedules').fetchone()[0]

        # Number of channels =  4, number of samples =   5110, mode: Logging Complete
        self.Nchannels = self.curs.execute('select count(*) from channels').fetchone()[0]
        self.Nsamples = self.curs.execute('select count(*) from data').fetchone()[0]
        
    def read_calibrations(self):
        self.calibrations = [None]*self.Nchannels
        self.curs.execute("""select channelID,equation,c0,c1,c2,c3,units 
                               from calibrations cal join channels chan
                                 on (cal.channelOrder=chan.channelID)""")
        names=[d[0] for d in self.curs.description]

        for row in self.curs:
            d=dict(zip(names,row))
            self.calibrations[d['channelID']-1]=Calibration.from_dict(d) 

    def read_extras(self):
        """ fields I don't want to mess with yet """
        self.averaging='n/a'
        # self.timebases

    def read_data_raw(self):
        """ shove the data table into a numpy array
        """
        all_data=self.curs.execute('select * from data order by tstamp asc').fetchall()

        dtype_desc=[]
        self.columns=[]
        for field_desc in self.curs.description:
            field_name=field_desc[0]
            #if field_name=='tstamp':
            #    typ='i8'
            #else:
            #    typ='f8'
            #dtype_desc.append((field_name,typ))
            self.columns.append(field_name)

        self.columns=self.columns[1:] # omit timestamp

        self.Ncolumns=len(self.columns)
        return np.array(all_data) # ,dtype=dtype_desc)

    def read_data(self):
        self.data=self.read_data_raw()
        
        # timezone handling here is a real pain.

        # start off with data[:,0] has milliseconds since the unix 
        # epoch, in the local time at start of deployment

        # datetime.datetime.fromtimestamp assumes that the value is always
        # in UTC, but defaults to *converting* to a localtime (can't remember
        # if it leaves the result naive or aware)
        
        # first step, tell that we want the result in UTC, so it won't do
        # any conversions, but it's not really UTC so drop it.
        dt_a=datetime.datetime.fromtimestamp( self.data[0,0]/1000.0,pytz.utc).replace(tzinfo=None)
        # print "dt_a: ",dt_a.strftime('%c'),dt_a.tzinfo 
        # now we have "2013/10/17 08:00:00 AM"

        # get the tzinfo to set the correct interpretation - it's not sufficient
        # to just replace tzinfo with self.instrument_tz. Note that this
        # will fail if the time is ambiguous.
        dt0_aware=self.instrument_tz.localize(dt_a)
        # print "dt0_aware: ",dt0_aware.strftime('%c'),dt0_aware.tzinfo
        # now we have "2013/10/17 08:00:00 AM EDT"
        
        dt0_naive=dt0_aware.astimezone(self.target_tz).replace(tzinfo=None)
        #print "dt0_naive: ",dt0_naive.strftime('%c'),dt0_naive.tzinfo
        # that gets us to "2013/10/17 07:00:00 AM"
        dn0=date2num( dt0_naive )

        self.data[:,0] = dn0 + (self.data[:,0]-self.data[0,0])/(1000.0*86400)

        # for consistency with RbrHex, uses names 
        # 'Cond','Temp','Pres','Salinity'
        # put some nicer names in there
        self.curs.execute("""select channelID,longName from channels
                              where not isDerived order by channelID""")
        ch_map=dict([ ("channel%02d"%cid,str(shortname)[:4]) for cid,shortname in self.curs])
        ch_map['tstamp']='tstamp'

        for ci,colname in enumerate(self.columns):
            self.columns[ci]=ch_map.get(colname,colname)

        #new_dtype=[ (ch_map[fname],ftype) for fname,ftype in data.dtype.descr]
        #data.dtype=np.dtype(new_dtype)

        #dn_py=date2num(datetime.datetime(1970,1,1)) + data['tstamp']/86400000.
        #dn_mat=dn_py+366
        #new_fields=[('dn_py',dn_py),
        #            ('dn_mat',dn_mat)]
        #data=array_append.recarray_add_fields(data,new_fields)

        self.update_fields()
        self.synthesize_fields()

class RbrHex(Rbr):
    """ subclass for reading hex files.
    """

    def read(self):
        self.fp = open(self.filename,'rt')

        self.read_headers()
        self.read_calibrations()
        self.read_extras()

        self.read_data()

        self.txt_trailing = self.fp.read()
        self.fp.close()

    def read_headers(self):
        self.txt_model = self.fp.readline()
        self.ruskin = ('Ruskin' in self.txt_model)

        self.txt_host_time = self.fp.readline()
        self.txt_logger_time = self.fp.readline()
        self.txt_logging_start = self.fp.readline()
        self.txt_logging_end   = self.fp.readline()
        self.txt_sample_period = self.fp.readline()
        self.txt_Nchannels = self.fp.readline()
        self.txt_float_format = self.fp.readline()

        # parse the ones that seem important:
        
        # Logging start 09/02/24 17:00:00
        log,start,date,time = self.txt_logging_start.split()
        self.logging_start = self.parse_datetime(date,time)

        log,end,date,time = self.txt_logging_end.split()
        self.logging_end   = self.parse_datetime(date,time)

        m=re.match('Sample period\s+(.*)',self.txt_sample_period)
        if m is not None:
            sample_period=m.group(1).strip()
            if sample_period=='6Hz profiling':
                self.dt_s=1./6
            else:
                # something like '00:00:30'
                h,m,s=[int(p) for p in sample_period.split(':')]
                self.dt_s=h*3600+60*m+s

        # Number of channels =  4, number of samples =   5110, mode: Logging Complete
        m = re.match(r'Number of channels\s*=\s*(\d+),\s*number of samples\s*=\s*(\d+)',self.txt_Nchannels)
        self.Nchannels = int(m.group(1))
        self.Nsamples = int(m.group(2))
        
    def read_calibrations(self):
        self.calibrations = []
        for i in range(self.Nchannels):
            self.calibrations.append( self.read_calibration() )

    def read_calibration(self):
        txt_cal = self.fp.readline() + self.fp.readline()+self.fp.readline()+self.fp.readline()
        parts = txt_cal.split()
        coefs = np.array( map(float,parts[2:6]) )
        units = parts[6]

        return Calibration(txt_cal,coefs,units)

    def read_extras(self):
        """ fields I don't want to mess with yet """
        while 1:
            l = self.fp.readline().strip()

            if l.find('Correction to conductivity') == 0:
                pass
            elif l.find('COMMENT:') == 0:
                self.comment = l.split(':')[1].strip()
            elif l.find('Averaging:') == 0:
                parts = l.split()
                if len(parts)==2 and parts[1] == 'NONE':
                    self.averaging = None
                else:
                    self.averaging = int(parts[1])
            elif l.find('Number of bytes in header')==0:
                self.num_bytes_header = int( l.split()[-1] )
            elif l.find('Logger start:Start of logging') == 0:
                pass
            #elif l.find('Atmospheric') == 0:
            #    pass
            #elif l.find('Memory type:') == 0:
            #    pass
            elif l.strip()=="":
                break
            else:
                pass

            # Lines like:
            # Timestamp 2014/05/14 14:54:55 at sample 3 of type: TIME STAMP
            # Reset stamp 2014/05/14 17:26:18 at sample 54486 of type: STOP STAMP

            m=re.match('(Timestamp|Reset stamp) (\S+) (\S+) at sample (\d+) of type: (.*) STAMP',l)
            if m is not None:
                # print "Got a timestamp"
                if m.group(5) in ('TIME','STOP','GAIN'):
                    # [ [type,datetime,1-based sample], ... ]
                    self.timebases.append( [
                        m.group(5),
                        self.parse_datetime(m.group(2),m.group(3)),
                        int( m.group(4) )
                        ] )
                else:
                    print("Bad type",repr(m.group(5)))
            else:
                #print "Not a timestamp: ",repr(l)
                pass
            
        # # Sometimes there are messages about auto-gain here.  
        # while 1:
        #     if self.fp.readline().strip() == "":
        #         break
            
        self.columns = self.fp.readline().split() # after date and time
        self.Ncolumns = len(self.columns)

    def parse_datetime(self,d,t):
        try:
            dt=datetime.datetime.strptime(d+" "+t,'%y/%m/%d %H:%M:%S')
        except:
            dt=datetime.datetime.strptime(d+" "+t,'%Y/%m/%d %H:%M:%S')
        return self.instrument_tz.localize(dt).astimezone(self.target_tz).replace(tzinfo=None)
                                  
    def read_data(self):
        self.data = np.zeros( (self.Nsamples,1+len(self.columns)),np.float64 )

        saved = self.fp.tell()

        nbytes_line = self.fp.readline()
        if nbytes_line.find('Number of bytes of data')==0:
            parts = nbytes_line.split()
            self.num_bytes_data = int(parts[5])
            self.fp.readline() # some pipe characters
            self.raw = self.fp.read().replace("\n","").replace("\r","")
            self.parse_raw()

            if self.dt_s is not None and self.dt_s>=1:
                # maybe it's safer just to assume all samples were output,
                # and synthesize time one go.
                # assumes that the first timestamp is correct
                print("Overwriting time with even steps of dt_s")
                self.data[:,0] = self.data[0,0] + self.dt_s * np.arange(len(self.data)) / 86400.
        else:
            # simple text format
            for i in range(self.Nsamples):
                parts = self.fp.readline().split()
                self.data[i,0] = date2num( self.parse_datetime(parts[0],parts[1]) )

                for c in range(len(self.columns)):
                    self.data[i,1+c] = float(parts[2+c])

        self.update_fields()
        self.synthesize_fields()

    def parse_raw(self):
        # very iffy on these
        # there can be some extra samples at the end (24 bytes worth in one case),
        # but I think that using the STOP STAMP to choose the last sample is
        # a good way to go.
        # for 018503_20140514_1726.hex, +36 was good, dropping 3 samples
        #   
        # for 018503_20140514_1446.hex, +24 is probably correct.
        #
        #   Logger start:Start of logging 
        #   Timestamp 2014/05/14 07:24:12 at sample 1 of type: TIME STAMP
        #   Timestamp 2014/05/14 07:24:12 at sample 1 of type: GAIN STAMP channel=4 gain=1 (500NTU)
        #   Timestamp 2014/05/14 07:24:13 at sample 4 of type: TIME STAMP
        #   Number of bytes in header 48
        #   
        #                                  Cond       Temp       Pres     Turb-a 
        #   Number of bytes of data 1936356
        #   |     |     |     |     |     |     |     |        # 24 bytes per line
        #   140514072405140514220000000001000000000000000001   # HEADER
        #   00000000000000C104000000000000000000000000000032   # HEADER
        #   00000054494D140514072412FFE30E94DBEE01A7DB0003C2   # time stamp, sample
        #   FFE01C94DB7F01AC300003F2FFE0E194DAFF01AF5200034A   # sample,   , sample
        #   00000054494D140514072413FFDF8594DA8901B2260003DA   # time stamp, sample
        #   FFE1F894DA2501B45D0003B2FFE15B94D9B701B64E0003CA

        # Seems like the format is that each distinct Timestamp occupies one 
        # frame (12 bytes).  Timestamps are written before the associated sample
        # data.  When there are multiple timestamps for the same sample, of the same type,
        # all are written out.  Sequential timestamps, same sample, different type lead to
        # only one timestamp written out.

        # Comparing to a matlab file from Ruskin, there's a little discrepancy.
        # When Ruskin converts ....1726.hex, sample 2 (0-based) is nan, and the
        # rest of the series is offset by 1 sample relative to this code.
        # That would mean Ruskin interprets a timestamp in the datastream to replace
        # the sample, not just appear before the sample.  FWIW, the extra sample at the
        # end, which this code includes, has reasonable data.
        # As long as the file doesn't have lots of events with timestamps in the middle
        # of it, this is a minor issue.

        # includes some timestamps
        binary=self.raw.decode('hex')[self.num_bytes_header:]

        bytes=np.fromstring(binary,np.uint8)

        # snip out the timestamp entries
        sel=np.ones(len(bytes),np.bool_)
        offset=0
        bytes_per_frame=3*self.Ncolumns
        for tb in self.timebases:
            stamp_type,dt,sample=tb
            if stamp_type=='GAIN':
                # assume that GAIN STAMPs always come with regular stamps
                # so we can ignore them (since the code here isn't dealing with NTU cal)
                continue
            sample-=1 # to 0-based
            sel[bytes_per_frame*(offset+sample):bytes_per_frame*(offset+sample+1)]=False
            offset+=1
        bytes=bytes[sel]

        # expand 3 byte samples to 4 byte samples..
        expanded=np.zeros(len(bytes)*4/3,np.uint8)
        expanded[0::4]=bytes[0::3]
        expanded[1::4]=bytes[1::3]
        expanded[2::4]=bytes[2::3]

        # possible that the file is truncated, and does not have an integer
        # number of frames.  
        raw_fields=(np.fromstring(expanded.tostring(),'>i4') / 256)
        extra=len(raw_fields) % self.Ncolumns
        if extra:
            print("There were %d extra triples of data - truncating"%extra)
            raw_fields=raw_fields[:-extra]
        raw_fields=raw_fields.reshape([-1,self.Ncolumns])
        unit_fields=raw_fields/(2.0**24)
        unit_fields=unit_fields[:self.Nsamples]
        
        for i,cal in enumerate(self.calibrations):
            poly_coeffs=cal.coefs[::-1]
            if self.columns[i]=='Temp':
                # from Mark Vist, RBR, and the RBRsolo manual
                X=np.log(1/(unit_fields[:,i]%1.0) - 1)
                Y = np.polyval(poly_coeffs,X) 
                self.data[:,i+1]=1/Y - 273.15
            else:
                self.data[:,i+1]=np.polyval(poly_coeffs,unit_fields[:,i])
                
        # And the timebase:
        # as it stands, it's possible for the end to be off by up to one
        # sample.  A STOP might be output at a fractional time, truncating
        # that last period.

        base_times=[]
        base_samples=[]

        # in some cases there isn't a start - probably it started right away
        # and didn't generate an event - so use logging start
        # to be safe, include start of logging
        base_times.append( date2num(self.logging_start) )
        base_samples.append(0)

        for stamp_type,dtime,idx in self.timebases:
            dnum=date2num(dtime)
            if stamp_type=='STOP':
                # djanky.  but reproduces matlab results
                nsteps=(dnum-base_times[-1])*86400/self.dt_s
                if nsteps%1.0>0.01:
                    nsteps=np.ceil(nsteps)
                else:
                    nsteps=np.floor(nsteps)
                dnum=base_times[-1] + nsteps*self.dt_s/86400.
            base_times.append( dnum )
            base_samples.append(idx-1)

        # And add a last timestamp, based on the sample rate, in cases there wasn't a STOP STAMP.
        if self.dt_s is not None:
            base_samples.append(len(self.data))
            # possibly one-off here:
            base_times.append( base_times[-1]+ self.dt_s*(base_samples[-1]-base_samples[-2])/86400. )

        sample_i=np.arange(len(self.data)) # indices where we want times.
        self.data[:,0]=np.interp(sample_i,base_samples,base_times,left=np.nan,right=np.nan)
        
    @staticmethod
    def concatenate(Rs):
        R=RbrHex(None) # make a blank Rbr
        R.dat_file="merged"
        R.sources=Rs
        R.__dict__.update(Rs[0].__dict__) # copy most stuff from the first

        R.txt_logging_end=Rs[-1].txt_logging_end
        R.logging_end=Rs[-1].logging_end
        R.Nsamples = sum([r.Nsamples for r in Rs])

        R.timebases=[]
        sample_offset=0
        for r in Rs:
            for tb in r.timebases:
                tb=copy.deepcopy(tb)
                tb[2]+=sample_offset
                R.timebases.append(tb)
            sample_offset+=r.Nsamples

        R.data=np.concatenate( [r.data for r in Rs])
        R.update_fields()
        return R
            
class RbrText(Rbr):
    def read(self):
        self.fp=open(self.filename,'rt')

        self.skip_headers()
        self.parse_field_names()
        self.read_data()
        self.update_fields()

    def skip_headers(self):
        # scan to header row
        while 1:
            line=self.fp.readline()
            if line.startswith('NumberOfSamples'):
                break
            elif line=='':
                raise Exception("Never found NumberOfSamples")
        self.fp.readline()

    def parse_field_names(self):
        """ populate self.fields as list of tuples of ('name',parser)
        """
        headers=self.fp.readline()

        fields=[] 

        def gen_parser(s):
            return float(s.strip())
        def time_parser(s): # time to fractional day
            # ala '16:23:42.000'
            h,m,s = [float(p) for p in s.strip().split(':')]
            return ((s/60.+m)/60.+h)/24
        def date_parser(s):
            # ala '01-Apr-2015'
            return date2num( datetime.datetime.strptime(s,'%d-%b-%Y') )

        fields.append( ('date',date_parser) )
        fields.append( ('time',time_parser) )
        n_fields=(len(headers)-24)/14
        # when values overflow their intended width, the alignment fails, so parse the data
        # by split.
        # however, some field names have spaces, so parse the field names by column
        for fld_i in range( n_fields ):
            fld_s=24+fld_i*14
            fld_e=fld_s+14
            fields.append( (headers[fld_s:fld_e].strip(),gen_parser) )
        self.fields=fields

        self.columns=[fld[0] for fld in self.fields[2:]]
        self.Ncolumns=len(self.columns)

    def read_data(self):
        """ 
        self.data is [Nsamples,Nfields]
        self.columns: list of data column names 
        """
        samples=[]

        for line in self.fp:
            if line.strip()=='':
                break
            sample=[]
            for (fld,parser),val in zip(self.fields,line.split()):
                sample.append(parser(val))
            # special handling for date/time
            sample[:2]=[sample[0]+sample[1]]
            samples.append(sample)
        self.data=np.array(samples)


        
def load(fn,**kwargs):
    """ try to detect whether fn is a Rsk or hex file, 
    and return a corresponding instance
    kwargs:
     instrument_tz=pytz.utc
     target_tz=pytz.utc
    """
    if fn.lower().endswith('rsk'):
        return RbrRsk(fn,**kwargs)
    else:
        return RbrHex(fn,**kwargs)
