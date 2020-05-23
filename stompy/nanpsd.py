import numpy as np
try:
    from numpy import nanmean
except ImportError:
    # obsolete location
    from scipy.stats import nanmean

# initial implementation 90ms for 1132 sample window,
# compared to 0.258ms for holey_psd
# changing to nanmean -> 6ms.
# mean takes just 3ms.

def nan_detrend(x,order=1):
    valid=np.isfinite(x)
    coord=np.nonzero(valid)[0]
    order=min(order,len(coord)-1)
    if order>=0:
        coeffs=np.polyfit(coord,x[valid],order)
        return x-np.polyval(coeffs,np.arange(len(x)))
    else:
        return x

def autocorrcoef(x,maxlags=None):
    N = len(x)

    #default lag is N-1
    if maxlags == None:
        maxlags = N - 1
    else:
        assert maxlags < N, 'lag must be less than len(x)'
    assert(np.isrealobj(x))
    
    #create an autocorrelation array with same length as lag
    r = np.zeros(1+maxlags, dtype=float)

    x=x-np.nanmean(x)
    
    for k in range(0, maxlags+1):
        nk = N - k - 1
        a=x[0:nk+1]
        b=x[k:k+nk+1]
        valid=np.isfinite(a*b)

        denom=np.sqrt( np.mean( a[valid]**2 ) ) * np.sqrt( (b[valid]**2).mean() )
        r[k]=np.mean( a[valid]*b[valid] ) / denom 
    return r


def autocovariance(x, maxlags=None,
                   normalize=None):
    """
    Correlation function - similar to xcorr.  Copied from
    spectrum.correlation.CORRELATION, and simplified for
    the present application.
    assumes autocorrelation, not cross-correlation.
    norm is unbiased,x is real.

    By default this is autocovariance.
    normalize: 'corr' - divide by var(x).
      'corrcoef' - calculate a correlation coefficient for each lag
    """
    N = len(x)

    if np.all(np.isfinite(x)):
        my_mean=np.mean
    else:
        my_mean=nanmean

    #default lag is N-1
    if maxlags == None:
        maxlags = N - 1
    else:
        assert maxlags < N, 'lag must be less than len(x)'
    assert(np.isrealobj(x))
    
    #create an autocorrelation array with same length as lag
    r = np.zeros(1+maxlags, dtype=float)

    for k in range(0, maxlags+1):
        nk = N - k - 1

        # for an unbiased estimate would have to get fancier,
        # counting the number of missing samples.
        # for biased, though, it's easy:
        # biased - doesn't really make a big difference
        if normalize=='corrcoef':
            valid=np.isfinite(x[0:nk+1]*x[k:k+nk+1])
            r[k]=np.corrcoef(x[0:nk+1],x[k:k+nk+1])[1,0]
        else:
            r[k]=my_mean(x[0:nk+1]*x[k:k+nk+1])
    
    if normalize=='corr':
        r/=r[0]
        
    return r

def correlogrampsd(X, lag, NFFT=None):
    """
    PSD estimate using correlogram method.
    taken from spectrum, simplified for real-valued autocorrelation
    """
    N = len(X)
    assert lag<N, 'lag must be < size of input data'

    if NFFT == None:
        NFFT = N
    psd = np.zeros(NFFT, dtype=complex)
    
    # Window should be centered around zero. Moreover, we want only the
    # positive values. So, we need to use 2*lag + 1 window and keep values on 
    # the right side.
    w = np.hanning(2*lag+1)[lag+1:]

    # compute the cross covariance
    rxy = autocovariance(X, lag)
    
    # keep track of the first elt.
    psd[0] = rxy[0]
    
    # create the first part of the PSD
    psd[1:lag+1] = rxy[1:] * w
    
    # create the second part. 
    psd[-1:NFFT-lag-1:-1] = rxy[1:].conjugate() * w

    # real, and not abs??
    # probably because this is the spectrum of the autocorrelation - 
    # the phase is very important
    psd = np.real(np.fft.fft(psd))

    return psd


def psd_correl(data,Fs=1,NFFT=None,scale_by_freq=True,lag=None,detrend=1):
    """ a mlab.psd workalike, but based on the correlogram.
    """
    lag=lag or len(data)/10
    
    if detrend is not None:
        data=nan_detrend(data,order=detrend)

    Pxx=correlogrampsd(X=data,lag=lag,NFFT=NFFT)
    NFFT=len(Pxx)
    # since real valued:
    # though this is probably where it should *not* get flipped
    dc_comp=Pxx[0]
    Pxx = Pxx[NFFT/2:]*2
    Pxx[0] /= 2.
    Pxx = np.append(Pxx, dc_comp) # putting the DC part at the end??

    if scale_by_freq:
        Pxx /= Fs
        
    N=NFFT # not entirely sure about this...
    df=float(Fs)/N
    kx=df*np.arange(len(Pxx)) # was 0,1+np.floor(N/2.0)

    # in a few cases, it seems that we get negative values,
    # at least in the spectrum code on which this is based
    # (usually for very sparse signals).
    # Also, as written, Pxx is backwards, with the DC component
    # at the last index.
    Pxx=np.abs(Pxx[::-1])
    return Pxx,kx

# input size 1131
# kx  566
# Pxx: 567
# Pxx from correlogrampsd is 1131 long
