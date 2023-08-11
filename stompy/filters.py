"""
General zero-phase filter.  Thanks to Ed Gross for most
of this.  It's mostly a python translation of matlab code he
provided. (this is a generalization of the code in tidal_filter.py)
"""

from __future__ import print_function

import numpy as np
from scipy.signal.filter_design import butter
import scipy.signal

from scipy.signal import filtfilt, lfilter
import warnings

def lowpass(data,in_t=None,cutoff=None,order=4,dt=None,axis=-1,causal=False):
    """
    data: vector of data
    in_t: sample times
    cutoff: cutoff period in the same units as in_t

    returns vector same as data, but with high frequencies removed
    """
    
    # Step 1: Determine dt from data or from user if specified
    if dt is None:
        dt=np.median(np.diff(in_t))
    dt=float(dt) # make sure it's not an int
    cutoff=float(cutoff)

    Wn = dt / cutoff 

    B,A = butter(order, Wn)

    if not causal:
        # scipy filtfilt triggers some warning message about tuple
        # indices.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_filtered = filtfilt(B,A,data,axis=axis)
    else:
        data_filtered = lfilter(B,A,data,axis=axis)

    return data_filtered


def lowpass_gotin(data,in_t_days,*args,**kwargs):
    print("Use lowpass_godin() instead of lowpass_gotin()")
    return lowpass_godin(data,in_t_days,*args,**kwargs)

def lowpass_godin(data,in_t_days=None,ends='pass',
                  mean_dt_h = None,axis=-1,
                  *args,**kwargs):
    """ Approximate Godin's tidal filter
    Note that in preserving the length of the dataset, the ends aren't really
    valid

    data: array suitable to pass to np.convolve
    in_t_days: timestamps in decimal days.  This is only used to establish
    the time step, which is assumed to be constant.
    dt_h: directly specify timestep in hours. 

    ends:
    'pass' no special treatment at the ends.  The first and last ~37
      hours will be contaminated by end-effects.
    'nan' will replace potentially contaminated end samples with nan

    *args,**kwargs are allowed but ignored.  They are present to make it
    easier to slip this method in to replace others without having to change
    the call signature
    """
    if mean_dt_h is None:
        mean_dt_h = 24*np.mean(np.diff(in_t_days))

    # how many samples are in 24 hours?
    N24 = int(round(24. / mean_dt_h))
    # and in 25 hours?
    N25 = int(round(25. / mean_dt_h))

    A24 = np.ones(N24) / float(N24)
    A25 = np.ones(N25) / float(N25)

    # numpy convolve only takes 1D inputs.
    # so iterate over any extra dimensions
    iter_shape=list(data.shape)
    del iter_shape[axis]
    data=data.copy()
    for idx in np.ndindex(*iter_shape):
        expanded=list(idx)
        expanded.insert(axis,slice(None))
        expanded=tuple(expanded)
        slc=data[expanded]
        
        if ends=='nan':
            # Add nan at start/end, which will carry through
            # the convolution to mark any samples affected
            # by the ends
            slc=np.concatenate( ( [np.nan],slc,[np.nan] ) )
        slc=np.convolve(slc,A24,'same')
        slc=np.convolve(slc,A24,'same')
        slc=np.convolve(slc,A25,'same')

        if ends=='nan':
            slc=slc[1:-1]
        data[expanded]=slc            

    return data

def lowpass_fir(x,winsize,ignore_nan=True,axis=-1,mode='same',use_fft=False,
                nan_weight_threshold=0.49999,window='hanning',
                dim='time'):
    """
    In the absence of exact filtering needs, choose the window 
    size to match the cutoff period.  Signals with a frequency corresponding to
    that cutoff period will be attenuated to about 50% of their original
    amplitude, corresponding to the -6dB point.

    Rolloff is about 18dB/octave, though highly scalloped so it's not as
    simple as with a Butterworth filter.  That 18dB/octave is roughly the same as
    a 3rd order butterworth, but it's a bit unclear on exactly where the 18dB/octave
    rolloff holds.
    
    x: ndarray
    winsize: integer - how long the hanning window is
    axis: the axis along which to apply the filter
    mode: same as for scipy.signal convolve operations
    use_fft: using the fft is faster, but sometimes less robust
    nan_weight_threshold: items with a weight less than this will be marked nan
      the default value is slightly less than half, to avoid numerical roundoff
      issues with 0.49999999 < 0.5
    dim: if x is a xarray.DataArray, gives the dimension name along which to
      filter.

    N.B. In the case of an input with nan only at the beginning and end, to 
    guarantee that the output will have the same nans as the input winsize must
    be odd. Otherwise there can be nan-weights exactly==0.500 on both ends.
    """
    # hanning windows have first/last elements==0.
    # but it's counter-intuitive - so force a window with nonzero
    # elements matching the requested size
    import xarray as xr
    if isinstance(winsize,np.ndarray):
        win=winsize
        winsize=len(win)
    else:
        if isinstance(winsize, np.timedelta64):
            assert isinstance(x,xr.DataArray)
            dts=np.diff(x.time)
            dts.sort()
            dt=dts[len(dts)//2]
            winsize=max(1,int(winsize / dt))
            
        if window=='hanning':
            win=np.hanning(winsize+2)[1:-1]
        elif window=='boxcar':
            win=np.ones(winsize)
        else:
            raise Exception("window must be one of hanning, boxcar")
    win/=win.sum()

    if use_fft:
        convolve=scipy.signal.fftconvolve
    else:    
        convolve=scipy.signal.convolve

    if isinstance(x,xr.DataArray):
        x_da=x
        x=x_da.values
        if dim is not None:
            axis=x_da.get_axis_num(dim)
    else:
        x_da=None
        
    slices=[None]*x.ndim
    slices[axis]=slice(None)
    win=win[tuple(slices)] # expand to get the right broadcasting

    if ignore_nan:
        x=x.copy()
        valid=np.isfinite(x)
        x[~valid]=0.0
    
    result=convolve( x, win, mode) 

    if ignore_nan:
        weights=convolve( valid.astype('f4'),win,mode)
        has_weight=(weights>0)
        result[has_weight] = result[has_weight] / weights[has_weight]
        result[~has_weight]=0 # redundant, but in case of roundoff.
        result[weights<nan_weight_threshold]=np.nan

    if x_da is not None:
        result_da=x_da.copy()
        result_da.values[...] = result
        return result_da
    else:
        return result

# xarray time series versions:
def lowpass_xr(da,cutoff,**kw):
    """
    Like lowpass(), but ds is a data array with a time coordinate,
    and cutoff is a timedelta64.
    """
    data=da.values
    time_secs=(da.time.values-da.time.values[0])/np.timedelta64(1,'s')
    cutoff_secs=cutoff/np.timedelta64(1,'s')

    axis=da.get_axis_num('time')
    
    data_lp=lowpass(data,time_secs,cutoff_secs,axis=axis,**kw)
    da_lp=da.copy(deep=True)
    da_lp.values[:]=data_lp
    da_lp.attrs['comment']="lowpass at %g seconds"%(cutoff_secs)
    return da_lp

    
