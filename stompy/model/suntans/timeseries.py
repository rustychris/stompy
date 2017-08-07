"""
DEPRECATED

Encapsulate some time series methods into a class.  pandas and xarray
largely make this obsolete, and further efforts would be better off
creating some utility method for operating on those data structures
as opposed to creating a new data structure.
"""

from __future__ import print_function

import numpy as np

import datetime, time
from matplotlib.dates import date2num,num2date
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

class Timeseries(object):
    """ generic timeseries handling
    t is stored as floating point absolute days, as defined by pylab's date2num
    (which is 1 greater than the mx.DateTime class idea of absdays, which is
    days since 0001-01-01 midnight, GMT)
    d days into year y can be approximated here by:
    floor((y-1)*365.2426) + d
    """
    def __init__(self,t,x):
        self.t = t
        self.x = x

    @staticmethod
    def splice(ts1,ts2,verbose=0):
        """ Merge two timeseries.  combined dt will be the smaller of the two median timesteps.
        """
        ts1_dt = np.median(np.diff(ts1.t))
        ts2_dt = np.median(np.diff(ts2.t))
        
        basic_dt = min( ts1_dt, ts2_dt)

        if verbose:
            print("Splicing with dt=%f (%f vs %f)"%(basic_dt,ts1_dt,ts2_dt))

        splice_t = np.concatenate( (ts1.t,ts2.t) )
        splice_x = np.concatenate( (ts1.x,ts2.x) )
        
        order = np.argsort(splice_t)
        splice_t = splice_t[order]
        splice_x = splice_x[order]

        dirty_splice = Timeseries(t=splice_t,
                                  x=splice_x)
        dirty_splice.sanitize(verbose=verbose)
        return dirty_splice

        
    @staticmethod
    def load_csv(fn,date_fmt=None,skiprows=0,delimiter=",",force_year_before="now"):
        """ Load data from a two-column csv file.
        By default, the first column is absdays and the second column is the data.
        skiprows is same as for loadtxt.  date_fmt can be used to get datetime.datetime to handle
        parsing date strings to absdays.  
        
        force_year_before: if the date_fmt has a %y (2-digit year), make sure that the year portion of
          all dates are the current year or before.
        """
        if date_fmt is None or date_fmt.find("%y")<0:
            force_year_before = False # skip past checks
        elif force_year_before == 'now':
            force_year_before = time.localtime().tm_year 
        else:
            pass  # assume a numeric year was given

        def parse_date(s):
            # original date format is MM/DD/YYYY
            dt = datetime.datetime.strptime(s,date_fmt)
            if force_year_before and dt.year > force_year_before:
                dt = datetime.datetime( dt.year-100, dt.month, dt.day, dt.hour, dt.minute, dt.second)
            return date2num(dt) 
        converters={}
        if date_fmt is not None:
            converters[0] = parse_date
        def float_or_nan(s):
            if s == "":
                return np.nan
            else:
                return float(s)
        converters[1] = float_or_nan
            
        d = np.loadtxt(fn,skiprows=skiprows,converters=converters,delimiter=delimiter)
        return Timeseries(d[:,0],d[:,1])

    def write_csv(self,fn,date_fmt=None,headers=None):
        fp = open(fn,'wt')
        if headers is not None:
            fp.write(",".join(headers)+"\n")
            
        for i in range(len(self.t)):
            if date_fmt is not None:
                d = num2date(self.t[i])
                d = d.strftime(date_fmt)
            else:
                d = "%f"%self.t[i]
            fp.write("%s,%f\n"%(d,self.x[i]))
        fp.close()
        

    def t_in(self,units='absdays'):
        base_t = num2date(self.t[0])

        if units=='yeardays':
            base_t = date2num( datetime.datetime(base_t.year,1,1) )
            return self.t - base_t
        if units=='absdays':
            return self.t
        if units=='years':
            year1 = base_t.year 
            yearN = num2date(self.t[-1]).year + 1

            years = np.arange(year1,yearN+1)
            days  = np.array([date2num( datetime.datetime(y)) for y in years])

            return np.interp(self.t,days,years)
            
    def plot(self,t_units=None):
        if pylab is None:
            print("NO plotting.  couldn't load pylab")
            return
        
        if t_units:
            t = self.t_in(t_units)
        else:
            t = self.t
            t_units = 'abs. days'
            
        plt.plot(t,self.x)
        plt.xlabel(t_units)

    def plot_dates(self,*args,**kwargs):
        plt.plot_date(self.t,self.x,*args,**kwargs)
        plt.gcf().autofmt_xdate()

    def clip(self,t_start,t_stop):
        """ return a new timeseries that covers only the given period.
        if the start/stop dates do not coincide with a datapoint, expand
        the interval such that it contains t_start and t_stop

        t_start and t_stop should python datetime instances
        """
        t_start = date2num( t_start )
        t_stop = date2num( t_stop )
        
        i_start = np.searchsorted( self.t, t_start, side='right') - 1
        i_stop  = np.searchsorted( self.t, t_stop,  side='left') + 1

        return Timeseries(t = self.t[i_start:i_stop],
                          x = self.x[i_start:i_stop])

    def sanitize(self,max_missing_samples=None,verbose=0):
        """ Ensure that the timesteps are all equal, and fill any gaps up to max_missing_samples.
        Throws an exception if there is a gap larger than max_missing_samples.  If dt is
        variable (more than 1% difference between median and max), re-interpolate linearly, using the median dt.
        """
        if len(self.t) == 0:
            raise Exception("No data found!")
        elif len(self.t) == 1:
            raise Exception("Timeseries has one point - this is probably not what you want!")
            
        all_dt = np.diff(self.t)
        basic_dt = np.median(all_dt)
        missing = np.nonzero( all_dt > 1.01*basic_dt)[0]
        too_short = np.nonzero( all_dt < 0.99*basic_dt)[0]

        if max_missing_samples is not None and all_dt.max() > max_missing_samples * basic_dt:
            raise Exception("sanitize: too much data missing!")

        if len(missing) > 0 or len(too_short) > 0:
            if verbose:
                if len(missing) > 0:
                    print("Some missing samples, will interpolate")
                elif len(too_short) > 0:
                    print("Some small timesteps, will re-interpolate")
                    print("Median dt: %f  Min dt: %f  count(dt<%f)=%d"%(basic_dt,all_dt.min(),0.99*basic_dt,len(too_short)))
            nsteps = 1 + np.round( (self.t[-1] - self.t[0])/ basic_dt )
            new_t = np.linspace(self.t[0],self.t[-1],nsteps)

            f = interp1d(self.t, self.x)

            self.t = new_t
            self.x = f(new_t)
            
        
        
class IEPFile(Timeseries):
    def __init__(self,filename):
        self.source = filename

        fp = open(filename,'rt')

        self.dss_parts = fp.readline()
        self.n_records = int(fp.readline())
        self.sample_type = fp.readline().split()[1]
        self.units = fp.readline().split()[1]

        t = np.zeros(self.n_records,np.float64)
        x = np.zeros_like(t)

        for i in range(self.n_records):
            datepart,timepart,flowpart = fp.readline().split()

            d = datetime.datetime.strptime(datepart,'%d%b%Y')
            h = float(timepart[:2])
            m = float(timepart[2:])
            d = d + datetime.timedelta(((m/60) +h)/24.)

            t[i] = date2num(d)
            x[i] = float(flowpart)

        if self.units == 'CFS':
            x *= 0.028316847
        Timeseries.__init__(self,t,x)



class UcsdText(Timeseries):
    source = '/home/rusty/classes/research/suntans/forcing/salinity/scripps-shore/Farallons_1925-200912.txt'
    def __init__(self,filename=None,var_name='SURF_SALT_PSU'):
        if filename is not None:
            self.source = filename

        fp = open(self.source,'rt')

        ## This used to work, but the more recent files don't have the quotes.
        # while 1:
        #     l = fp.readline().strip()
        #     if not (len(l) == 0 or l[0] == '"'):
        #         break

        ## instead, look for a pair of blank lines
        while 1:
            l = fp.readline().strip()
            if len(l) != 0:
                continue
            l = fp.readline().strip()
            if len(l) == 0:
                break

        headers = fp.readline().strip().split()

        value_i = headers.index(var_name)

        dates = []
        values = []
        
        for line in fp:
            cols = []
            for v in line.split():
                cols.append( float(v.strip('U')) )

            y,m,d = cols[:3]
            dates.append( date2num( datetime.datetime(y,m,d)) )
            values.append( cols[value_i] )

        t = np.array(dates)
        x = np.array(values)
                
        Timeseries.__init__(self,t,x)
            

def fill_holes(data,max_missing_samples=None,basic_dt=None):
    # remove nans:
    if len(data) == 0:
        return data # nothing to do
    
    invalid = np.any(np.isnan(data[:,1:]),axis=1)
    if np.sum(~invalid) == 0:
        return data # nothing to do
    
    data = data[~invalid]
    if len(data)<2:
        print("not enough data to do anything")
        return data
    
    all_dt = np.diff(data[:,0])
    if basic_dt is None:
        basic_dt = np.median(all_dt)
    missing = np.nonzero( all_dt > 1.5*basic_dt )[0]

    if max_missing_samples is not None and all_dt.max() > max_missing_samples * basic_dt:
        raise Exception("fill_holes: too much data missing!")
    
    if len(missing) > 0:
        nsteps = 1 + np.round( (data[-1,0] - data[0,0]) / basic_dt )
        new_t = np.linspace(data[0,0],data[-1,0],nsteps)

        f = interp1d(data[:,0], data,axis=0)

        new_data = f(new_t)
        new_data[:,0] = new_t
        return new_data
    else:
        return data


if __name__ == '__main__':
    fn = '/home/rusty/classes/research/suntans/forcing/flows/ndoi-1994-2009.txt'
    
    ndoi = IEPFile(fn)

    ndoi.plot(t_units='years')
