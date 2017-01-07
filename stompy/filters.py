"""
General zero-phase filter.  Thanks to Ed Gross for most
of this.  It's mostly a python translation of matlab code he
provided. (this is a generalization of the code in tidal_filter.py)
"""

import numpy as np
from scipy.signal.filter_design import butter
import scipy.signal

from scipy.signal import filtfilt, lfilter

def lowpass(data,in_t=None,cutoff=None,order=4,dt=None,axis=-1,causal=False):
    """
    data: vector of data
    in_t: sample times
    cutoff: cutoff period in the same units as in_t

    returns vector same as data, but with high frequencies removed
    """
    
    # Step 1: Determine dt from data
    dt=dt or np.median(np.diff(in_t))
    dt=float(dt)
    cutoff=float(cutoff)

    Wn = dt / cutoff 

    B,A = butter(order, Wn)

    if not causal:
        data_filtered = filtfilt(B,A,data,axis=axis)
    else:
        data_filtered = lfilter(B,A,data,axis=axis)

    return data_filtered


def lowpass_gotin(data,in_t_days,*args,**kwargs):
    """ Approximate Gotin's tidal filter
    Note that in preserving the length of the dataset, the ends aren't really
    valid
    """
    mean_dt_h = 24*mean(diff(in_t_days))

    # how many samples are in 24 hours?
    N24 = round(24. / mean_dt_h)
    # and in 25 hours?
    N25 = round(25. / mean_dt_h)

    A24 = ones(N24) / float(N24)
    A25 = ones(N25) / float(N25)

    data = convolve(data,A24,'same')
    data = convolve(data,A24,'same')
    data = convolve(data,A25,'same')

    return data
    

def lowpass_fir(x,winsize,ignore_nan=True,axis=-1,mode='same',use_fft=False,
                nan_weight_threshold=0.5):
    """
    x: ndarray
    winsize: integer - how long the hanning window is
    axis: the axis along which to apply the filter
    mode: same as for scipy.signal convolve operations
    use_fft: using the fft is faster, but sometimes less robust
    nan_weight_threshold: items with a weight less than this will be marked nan
    """
    # not sure why hanning windows have first/last elements==0
    # but it's counter-intuitive - so force a window with nonzero
    # elements matching the requested size
    win=np.hanning(winsize+2)[1:-1]
    win/=win.sum()

    if use_fft:
        convolve=scipy.signal.fftconvolve
    else:    
        convolve=scipy.signal.convolve

    slices=[None]*x.ndim
    slices[axis]=slice(None)
    win=win[slices] # expand to get the right broadcasting

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
    return result
