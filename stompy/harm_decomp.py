# harmonic decomposition
from __future__ import print_function

import numpy as np
from numpy.linalg import norm,qr,pinv

from . import tide_consts    


###
def recompose(t,comps,omegas):
    d = np.zeros(t.shape,np.float64)
    
    for i in range(len(omegas)):
        d += comps[i,0] * np.cos(t*omegas[i] - comps[i,1])
    return d
            
def decompose(t,h,omegas):
    """ 
    take an arbitrary timeseries defined by times t and values h plus a list
    of N frequencies omegas, which must be ANGULAR frequencies (don't forget the 2pi)
    
    return comps as an Nx2 array, where comps[:,0] are the amplitudes and comps[:,1]
    are the phases.

    super cheap caching: remembers the last t and omegas, and if they are the same
    it will reuse the matrix from before.
    """
    t=np.asanyarray(t)
    omegas=np.asanyarray(omegas)

    valid=np.isfinite(t)&np.isfinite(h)
    t=t[valid]
    h=h[valid]
    
    def sim(a,b):
        if a is b:
            return True
        if a is None or b is None:
            return False
        return (a.shape == b.shape) and np.allclose(a,b)
    if sim(decompose.cached_t,t) and sim(decompose.cached_omegas,omegas):
        Ainv = decompose.cached_Ainv
    else:
        # A is a matrix of basis functions - two (cos/sin) for each frequency
        n_bases = 2*len(omegas)
        basis_len = len(h)

        # form the linear system
        # each column of A is a basis function
        A = np.zeros( (basis_len,n_bases), np.float64)

        for i in range(len(omegas)):
            A[:,2*i] = np.cos(omegas[i]*t)
            A[:,2*i+1] = np.sin(omegas[i]*t)

        Ainv = pinv(A)

        decompose.cached_Ainv=Ainv
        decompose.cached_t = t.copy()
        decompose.cached_omegas = omegas.copy()
        
        # and can we say anything about the conditioning of A ?
        def cond_num(L):
            return norm(L,ord=2)*norm(pinv(L),ord=2)

        # sort of arbitrary...
        cnum = cond_num(A)
        if cnum > 10:
            print("Harmonic decomposition: condition number may be too high: ",cnum)
        
    x=np.dot(Ainv,h)

    # now rows are constituents, and we get the cos/sin as two columns
    comps = np.reshape(x,(len(omegas),2))

    # now transform cos/sin into amp/phase
    x_amps = np.sqrt( comps[:,0]**2 + comps[:,1]**2 )

    #
    x_phis = np.arctan2( comps[:,1], comps[:,0] )

    # rewrite comps using the amp/phase
    comps[:,0] = x_amps
    comps[:,1] = x_phis

    if 0:
        # check to see how close we are:
        recomposed = recompose(t,comps,omegas)
        rms = np.sqrt( ((h - recomposed)**2).mean() )

        print("RMS Error:",rms)
    
    return comps

def noaa_37_names():
    """ 
    return names of the 37 constituents provided in NOAA harmonic data
    """
    return ["M2","S2","N2","K1","M4","O1","M6","MK3","S4","MN4","NU2",
            "S6","MU2","2N2","OO1","LAM2","S1","M1","J1","MM","SSA",
            "SA","MSF","MF","RHO","Q1","T2","R2","2Q1","P1","2SM2",
            "M3","L2","2MK3","K2","M8","MS4"]

def noaa_37_omegas():
    """
    return frequencies in rad/sec for the 37 NOAA constituents
    """
    return names_to_omegas(noaa_37_names())

def names_to_omegas(names):
    """
    return speed in radians/sec for the named constituents
    """
    idx = [tide_consts.const_names.index(n) for n in names]
    omega_deg_per_hour = tide_consts.speeds[idx]
    omega_per_sec = omega_deg_per_hour * (1./3600) * (1/360.)
    return 2*np.pi*omega_per_sec


def decompose_noaa37(t,h):
    return decompose(t,h,noaa_37_omegas())


def select_omegas(T,omegas=None,factor=0.25):
    """
    T: duration of a timeseries in seconds
    omegas: an array of angular frequencies in rad/s, defaults to
    the 37 constituents used in NOAA predictions.

    returns a subset of angular frequencies which are resolvable in the
    given period of data.  This is based on eliminating pairs of
    constituents whose beat frequency is less than factor times
    the reciprocal of T.  In each such pair, the order of the incoming
    omegas is used as a prioritization.
    """
    # length of the data

    # min_beat
    # Roughly, to distinguish two frequencies f_a,f_b, must be able
    # to resolve their beat frequency, (f_b-f_a)
    # Figure we need a quarter period(?) to resolve that
    min_omega_beat=factor * (2*np.pi/T)

    if omegas is None:
        omegas=noaa_37_omegas() # rad/s
        
    # ability to differentiate two frequencies in a time series of
    # length T is proportional to the ratio of T to 1/delta f.

    sel_omegas=np.concatenate( ([0],omegas) )
    sel_omegas.sort()

    while 1:
        omega_beat=np.diff(sel_omegas)

        if omega_beat.min() < min_omega_beat:
            idx=np.argmin(omega_beat)
            if idx==0:
                T_a=np.inf
            else:
                T_a=2*np.pi/(3600*sel_omegas[idx])
            T_b=2*np.pi/(3600*sel_omegas[idx+1])

            # print( ("Periods %.2fh and %.2fh have beat period %.2fh,"
            #         " too long to be resolved by %.2fh time"
            #         " series")%( T_a,T_b,
            #                      2*np.pi/(3600*omega_beat.min()),
            #                      T/3600 ))
            
            # drop the one later in the original list of frequencies, and never DC.
            if idx==0:
                drop=1
            else:
                rank_a=np.nonzero( omegas==sel_omegas[idx] )[0][0]
                rank_b=np.nonzero( omegas==sel_omegas[idx+1] )[0][0]

                if rank_a<rank_b:
                    drop=idx+1
                else:
                    drop=idx
            # print("Will drop period of %.2fh"%(2*np.pi/3600/sel_omegas[drop]))
            sel_omegas=np.concatenate( (sel_omegas[:drop],
                                        sel_omegas[drop+1:]) )
        else:
            break
    sel_omegas=sel_omegas[1:] # drop DC now.
    return sel_omegas




decompose.cached_t = None
decompose.cached_omegas = None

