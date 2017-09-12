"""
Access SFEI ERDDAP copy of USGS SF Bay Water Quality data.

Note that SFEI ERDDAP is not necessarily up to date!
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from stompy import (utils,xr_utils)
from stompy.spatial import proj_utils

usgs_erddap="http://sfbaynutrients.sfei.org/erddap/tabledap/usgs_sfb_nutrients"


def cruise_dataset(start,stop):
    """
    Fetches USGS SF Bay water quality cruises from SFEI ERDDAP, munges the
    data to some degree and returns in an xarray dataset.

    start, stop: dates bracketing the period of interest.
    """
    full_remote_usgs_ds=xr.open_dataset(usgs_erddap)

    # Limit to requested period
    start=utils.to_dt64(start)
    stop=utils.to_dt64(stop)
    time_slc=slice( *np.searchsorted( full_remote_usgs_ds['s.time'].values,
                                      [start,stop]) )
    remote_usgs_ds=full_remote_usgs_ds.isel(s=time_slc)

    # Drop the annoying s. prefix
    renames=dict([ (v,v[2:]) for v in remote_usgs_ds.data_vars
                   if v.startswith('s.') ] )
    # some sort of xarray or numpy bug.  the first copy sorts out something deep in the internals.
    # see xarray bug report #1253.
    _dummy=remote_usgs_ds.copy(deep=True)
    # second copy can then proceed correctly
    ds0=remote_usgs_ds.copy(deep=True)

    ds=ds0.rename(renames)

    # add dates:
    # have to be careful about time zones here.  Add 7 hours before rounding to
    # date to get PST days, within 1hr.
    # Also note that xarray sneakily pushes everything to datetime[ns],
    # so even though we cast to M8[D], xarray immediately makes that midnight
    # UTC on the given date, which then displays as 1600 or 1700 on the previous
    # day.
    ds['date']=(ds.time-np.timedelta64(7*3600,'s')).astype('M8[D]')

    # At this point, ds also respects the ordering.  so far so good

    # Kludge missing distances:
    bad=np.isnan(ds['Distance_from_station_36'].values)
    # for weird reasons, must use .values here!  hmm - doesn't always work.
    ds['Distance_from_station_36'].values[ bad ] = 999

    # A little dicey within each profile, notably it doesn't
    # explicitly force the depths to be in order.
    # Using Distance_from_station_36 yields a better ordering...
    ds4=xr_utils.redimension(ds.copy(deep=True),
                             ['date','Distance_from_station_36'],
                             intragroup_dim='prof_sample',
                             save_mapping=True,
                             inplace=True)

    if 1:
        # There are a few variables which are specific to a station, so no need to carry
        # them around in full dimension:
        spatial=['StationNumber',
                 'latitude','longitude',
                 'StationName']
        for fld in spatial:
            # This fails because if a cast has no surface sample, we get
            # nan values.
            # ds4[fld] = ds4[fld].isel(drop=True,date=0,prof_sample=0)
            # Instead, aggregate over the dimension.  min() picks out nan
            # values.  median can't handle strings.  max() appears ok.
            # In some cases this is still a problem, ie on hpc, can get a
            # TypeError because missing values are nan, and cannot be compared
            # to byte or string.
            # Because there are some lurking inconsistencies with the type
            # of strings coming in as bytes vs. strings, we still have some
            # weird logic here
            try:
                # this should work if the field is already a float
                ds4[fld] = ds4[fld].max(dim='date').max(dim='prof_sample')
            except TypeError:
                # maybe it's a string?
                try:
                    ds4[fld] = ds4[fld].fillna(b'').max(dim='date').max(dim='prof_sample')
                except TypeError:
                    ds4[fld] = ds4[fld].fillna('').max(dim='date').max(dim='prof_sample')
            
        ds4=ds4.set_coords(spatial)
     
        # And some nutrient variables which do not have a vertical dimension
        # with the exception of 3 samples, there is at most 1 nutrient sample
        # per water column.
     
        nutrients=['nh','p','si','nn','NO2','ext_coeff','ext_coeff_calc']
    
        for fld in nutrients:
            valid=np.isfinite(ds4[fld].values)
            max_count=np.sum( valid, axis=2).max()
            if 0: # show some add'l info on how often there is more than 1 sample per profile:
                print("%s: max per cast %d"%(fld,max_count))
                for mc in range(max_count):
                    print("    %d casts have %d samples"%( np.sum( (mc+1)==np.sum( valid,axis=2 ) ),
                                                           mc+1))
            ds4[fld] = ds4[fld].mean(dim='prof_sample')

    # The rest will get sorted by depth
    ds5=xr_utils.sort_dimension(ds4,'depth','prof_sample')
    # add a bit of CF-style metadata - possible that this could be copied from the
    # ERDDAP data...
    ds5.depth.attrs['positive']='down'

    # Go ahead and add UTM coordinates
    utm_xy=proj_utils.mapper('WGS84','EPSG:26910')( np.array( [ds5.longitude,ds5.latitude] ).T )
    ds5['x']=( ds5.longitude.dims, utm_xy[:,0] )
    ds5['y']=( ds5.longitude.dims, utm_xy[:,1] )
    
    ds5=ds5.set_coords(['time','x','y','depth'])

    # add more metadata
    ds5['prof_sample'].attrs['long_name']='Profile (vertical) dimension'

    return ds5
    
## --- 

# Directly query USGS system:
# And get some discrete chl from USGS data
# easiest to go straight to the source 

# this is from executing the request on the USGS Site, and grabbing the form
# data which was posted:

# Included for reference for future development of a proper function
if 0:
    import StringIO
    import datetime
    import requests
    from bs4 import BeautifulSoup, Comment
    import re

    form_vars="""\
    col:realdate
    col:stat
    col:depth
    col:dscrchl
    col:calcchl
    col:salin
    col:temp
    col:sigt
    dstart:1990
    p11:
    p12:
    p21:
    p22:
    p31:
    p32:
    type1:year
    type2:---
    type3:---
    value1:2012
    value2:
    value3:
    comp1:ge
    comp2:gt
    comp3:gt
    conj2:AND
    conj3:AND
    sort1:fulldate
    asc1:on
    sort2:stat
    asc2:on
    sort3:---
    asc3:on
    out:comma
    parm:on
    minrow:0
    maxrow:99999
    ftype:easy
    """

    params=[s.split(':') for s in form_vars.split()]

    url="https://sfbay.wr.usgs.gov/cgi-bin/sfbay/dataquery/query16.pl"

    result=requests.post(url,params)

    text=result.text

    soup = BeautifulSoup(text, 'html.parser')
    #comments = soup.findAll(text=lambda text:isinstance(text, Comment))
    #[comment.extract() for comment in comments]

    data = soup.find('pre')
    data1=data.get_text()

    # Remove HTML comments
    data2=re.sub(r'<!--[^>]*>','',data1)

    df = pd.read_csv(StringIO.StringIO(data2),skiprows=[1],parse_dates=["Date"] )

    # Shorten names
    df1=df.rename(columns={'Station Number':'Station',
                           'Discrete Chlorophyll':'dChl ug/L',
                           'Calculated Chlorophyll':'cChl ug/L',
                           'Salinity':'sal',
                           'Temperature':"T",
                           'Sigma-t':'sigmat'})
    
