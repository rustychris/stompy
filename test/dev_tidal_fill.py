from stompy import utils
from stompy import filters
from stompy import harm_decomp

def fill_tidal_data(da,fill_time=True):
    """
    Extract tidal harmonics from an incomplete xarray DataArray, use
    those to fill in the gaps and return a complete DataArray.

    Uses all 37 of the standard NOAA harmonics, may not be stable
    with short time series.
    
    A 5-day lowpass is removed from the harmonic decomposition, and added
    back in afterwards.

    Assumes that the DataArray has a 'time' coordinate with datetime64 values.

    The time dimension must be dense enough to extract an exact time step
    
    If fill_time is True, holes in the time coordinate will be filled, too.
    """
    diffs=np.diff(da.time)
    dt=np.median(diffs)

    if fill_time:
        gaps=np.nonzero(diffs>1.5*dt)[0]
        pieces=[]
        last=0
        for gap_i in gaps:
            # gap_i=10 means that the 10th diff was too big
            # that means the jump from 10 to 11 was too big
            # the preceding piece should go through 9, so
            # exclusive of gap_i
            pieces.append(da.time.values[last:gap_i])
            pieces.append(np.arange( da.time.values[gap_i],
                                     da.time.values[gap_i+1],
                                     dt))
            last=gap_i+1
        pieces.append(da.time.values[last:])
        dense_times=np.concatenate(pieces)
        dense_values=np.nan*np.zeros(len(dense_times),np.float64)
        dense_values[ np.searchsorted(dense_times,da.time.values) ] = da.values
        da=xr.DataArray(dense_values,
                        dims=['time'],coords=[dense_times])
    else:
        pass 

    dnums=utils.to_dnum(da.time)
    data=da.values

    # lowpass at about 5 days, splitting out low/high components
    winsize=int( np.timedelta64(5,'D') / dt )
    data_lp=filters.lowpass_fir(data,winsize)
    data_hp=data - data_lp

    valid=np.isfinite(data_hp)
    omegas=harm_decomp.noaa_37_omegas() # as rad/sec

    harmonics=harm_decomp.decompose(dnums[valid]*86400,data_hp[valid],omegas)

    dense=harm_decomp.recompose(dnums*86400,harmonics,omegas)

    data_recon=utils.fill_invalid(data_lp) + dense

    data_filled=data.copy()
    missing=np.isnan(data_filled)
    data_filled[missing] = data_recon[missing]

    fda=xr.DataArray(data_filled,coords=[da.time],dims=['time'])
    return fda

# #

# Fill gaps in data with significant tidal variation
da_orig=xr.open_dataset('/opt/data/delft/sfb_dfm_v2/runs/wy2013/jersey-raw.nc')

da=da_orig.stream_flow_mean_daily.copy(deep=True)

sel=utils.within(utils.to_dnum(da_orig.time),
                 [734791.50788189541,734793.34447256383])

da_short=da_orig.isel(time=sel)

fda=fill_tidal_data(da_short.stream_flow_mean_daily)

plt.figure(1).clf()
plt.plot(fda.time,fda,'g-')
plt.plot(da.time,da)
plt.axis( (734791.50788189541,734793.34447256383,
           -287214.44701689074,191235.44801606232) )

##

def select_omegas(T,omegas=None,factor=0.25):
    """
    T: timedelta64 giving duration of a timeseries
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
        omegas=harm_decomp.noaa_37_omegas() # rad/s
        
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

            print( ("Periods %.2fh and %.2fh have beat period %.2fh,"
                    " too long to be resolved by %.2fh time"
                    " series")%( T_a,T_b,
                                 2*np.pi/(3600*omega_beat.min()),
                                 T/3600 ))
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
            print("Will drop period of %.2fh"%(2*np.pi/3600/sel_omegas[drop]))
            sel_omegas=np.concatenate( (sel_omegas[:drop],
                                        sel_omegas[drop+1:]) )
        else:
            break
    sel_omegas=sel_omegas[1:] # drop DC now.
    return sel_omegas

T=(da_short.time[-1] - da_short.time[0])/np.timedelta64(1,'s')
omegas=select_omegas(T,factor=0.05)
