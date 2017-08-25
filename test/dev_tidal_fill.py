from stompy import utils
from stompy import filters
from stompy import harm_decomp

# Fill gaps in data with significant tidal variation
da_orig=xr.open_dataset('/opt/data/delft/sfb_dfm_v2/runs/wy2013/jersey-raw.nc')

da=da_orig.stream_flow_mean_daily.copy(deep=True)

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

##

fda=fill_tidal_data(da_orig.stream_flow_mean_daily)

plt.figure(1).clf()
plt.plot(fda.time,fda,'g-')
plt.plot(da.time,da)
plt.axis( (734980.96027543489,
           734983.42080451641,
           -287214.44701689074,
           191235.44801606232) )
