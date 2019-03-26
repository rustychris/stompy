"""
Functions to take a time series of waterlevels and compute
tidal datums
"""

import numpy as np

# constants
TM2=12.4200*60 # M2 period in minutes. 12h25.2m 745.2 minutes
TLM=29.53059*24*60 # Lunar month in minutes 29d12h44m
T57M2=TM2*57 # 28.5 lunar days, 57 M2 periods, in minutes

def find_mllw(timeseries):
    """
    Find MLLW from a month long tidal record
    
    timeseries: [N,2] array, [:,0] datenums, [:,1] water level samples

    Translated from Vitalii Sheremet's matlab code sh_tidemllw.m

    Only consumes the integer multiples of a lunar month of data
    """
    return find_tidal_datum(timeseries,stat='min',daily=True)

def find_mlw(timeseries):
    return find_tidal_datum(timeseries,stat='min',daily=False)
def find_mhw(timeseries):
    return find_tidal_datum(timeseries,stat='max',daily=False)
def find_mhhw(timeseries):
    return find_tidal_datum(timeseries,stat='max',daily=True)

def find_tidal_datum(timeseries,stat,daily=False):
    """
    generic workings for tidal extrema datums
    stat: 'min' or 'max'
    """
        
    t = timeseries[:,0]
    h = timeseries[:,1]
    
    # median seems safer than mode with floating point data
    dt=np.median(np.diff(t)*24*60) # time step of the record in minutes
    nm2=TM2/dt      # fractional samples per TM2
    
    h1=h-h.mean() # height anomaly

    i0 = np.nonzero( h1[:-1]*h1[1:] < 0)[0][0] # first zero crossing

    Nmonths = int( (t[-1] - t[i0])*24*60 / T57M2 )

    # Low Water find minimum in each TM2 segment 
    jm=np.zeros(57*Nmonths,np.int32) # indices to low water within each M2 period

    for k in range(57*Nmonths):
        i1=int(i0+np.round(k * nm2)) # index of kth m2
        i2=int(i0+np.round((k+1) * nm2))
        if stat is 'min':
            jm[k] = i1 + np.argmin( h[i1:i2] )
        elif stat is 'max':
            jm[k] = i1 + np.argmax( h[i1:i2] )
        else:
            raise Exception("Stat %s not understodd"%stat)
    h_agg = h[jm] # h extrema aggregated per M2 period

    if not daily:
        return h_agg.mean()
    else:
        # [RH]: why compute the pairs two different ways?
        # This is a departure from V.S. code, and maybe
        # a departure from the 'correct' way - have to go
        # back to MLLW documentation...
        if len(h_agg)%2:
            h_agg = h_agg[:-1] # trim to even number of M2 periods
        h_agg_by_day = h_agg.reshape( (-1,2) )

        if stat is 'min':
            daily_agg = h_agg_by_day.min(axis=1)
        else:
            daily_agg = h_agg_by_day.max(axis=1)

        return daily_agg.mean()
