"""
Access SFEI ERDDAP copy of USGS SF Bay Water Quality data.

Note that SFEI ERDDAP is not necessarily up to date!
"""
import os
import six

import logging
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

import datetime
import requests
import pandas as pd
import hashlib
from bs4 import BeautifulSoup, Comment
import re

from ... import (utils,xr_utils,memoize)
from ...spatial import proj_utils

StringIO=six.StringIO


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

# available columns:
usgs_sfbay_columns=[ ('realdate','Date (MM/DD/YYYY)'),
                     ('jdate','Julian Date (YYYYDDD)'),
                     ('days','Days since 01/01/1990'), # hardcoded 1990 as dstart
                     ('decdate','Decimal Date'),
                     ('time','Time of Day '),
                     ('stat','Station Number'),
                     ('dist','Distance from Station 36'),
                     ('depth','Depth'),
                     ('dscrchl','Discrete Chlorophyll'),
                     ('chlrat','Chlorophyll a/a+PHA ratio'),
                     ('fluor','Fluorescence'),
                     ('calcchl','Calculated Chlorophyll'),
                     ('dscroxy','Discrete Oxygen'),
                     ('oxy','Oxygen electrode output'),
                     ('oxysat','Oxygen Saturation %'),
                     ('calcoxy','Calculated Oxygen'),
                     ('dscrspm','Discrete SPM'),
                     ('obs','Optical Backscatter'),
                     ('calcspm','Calculated SPM'),
                     ('dscrexco','Measured Extinction Coeff'),
                     ('excoef','Calculated Extinction Coeff'),
                     ('salin','Salinity'),
                     ('temp','Temperature'),
                     ('sigt','Sigma-t'),
                     # ('height','Tide Height at SF'),
                     ('no2','Nitrite'),
                     ('no32','Nitrate+Nitrite'),
                     ('nh4','Ammonium'),
                     ('po4','Phosphate'),
                     ('si','Silicate'),
                     ]


@memoize.memoize()
def station_locs():
    this_dir=os.path.dirname(__file__)
    # The web API doesn't provide lat/lon
    return pd.read_csv(os.path.join(this_dir,'usgs_sfbay_station_locations.csv'),
                       skiprows=[1]).set_index('StationNumber')

def station_number_to_lonlat(s):
    return station_locs().loc[float(s),[ 'longitude','latitude']].values


# This is a snapshot to give a sense of some of the options, but not everything
# (very little, really) is exposed in the method below
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

from .common import periods

def query_usgs_sfbay(period_start, period_end, cache_dir=None, days_per_request='M'):
    """
    Download (and locally cache) data from the monthly cruises of the USGS R/V Polaris 
    and R/V Peterson.

    Returns data as pandas DataFrame.
    """
    # Handle longer periods:
    if days_per_request is not None:
        logging.info("Will break that up into pieces")
        dfs=[]
        for interval_start,interval_end in periods(period_start,period_end,days_per_request):
            df=query_usgs_sfbay(interval_start,interval_end,cache_dir=cache_dir,days_per_request=None)
            if df is not None:
                dfs.append(df)
        return pd.concat(dfs)
    
    params=[]

    for column,text in usgs_sfbay_columns:
        params.append( ('col',column) )

    params += [('dstart','1990'),
               ('p11',''),
               ('p12',''),
               ('p21',''),
               ('p22',''),
               ('p31',''),
               ('p32','')]

    comps=dict(type1='---',value1='',comp1='gt',
               type2='---',value2='',comp2='gt',
               type3='---',value3='',comp3='gt')

    filter_count=0

    for t,comp in [ (period_start,'ge'),
                    (period_end,'lt') ]:
        if t is not None:
            filter_count+=1
            comps['type%d'%filter_count]='jdate'
            comps['comp%d'%filter_count]=comp
            comps['value%d'%filter_count]=str(utils.to_jdate(t))

    for fld in ['type','value','comp']:
        for comp_i in [1,2,3]:
            fld_name=fld+str(comp_i)
            params.append( (fld_name, comps[fld_name] ) )

    params+= [ ('conj2','AND'),
               ('conj3','AND'),
               ('sort1','fulldate'),
               ('asc1','on'),
               ('sort2','stat'),
               ('asc2','on'),
               ('sort3','---'),
               ('asc3','on'),
               ('out','comma'),
               ('parm','on'),
               ('minrow','0'),
               ('maxrow','99999'),
               ('ftype','easy')
    ]

    if cache_dir is not None:
        fmt="%Y%m%d"
        cache_file=os.path.join(cache_dir,'usgs_sfbay_%s_%s.csv'%(utils.to_datetime(period_start).strftime(fmt),
                                                                  utils.to_datetime(period_end).strftime(fmt)))
    else:
        cache_file=None

    def fetch():
        logging.info("Fetch %s -- %s"%(period_start,period_end))
        url="https://sfbay.wr.usgs.gov/cgi-bin/sfbay/dataquery/query16.pl"
        result=requests.post(url,params)
        text=result.text
        soup = BeautifulSoup(text, 'html.parser')
        data = soup.find('pre')
        data1=data.get_text()
        data2=re.sub(r'<!--[^>]*>','',data1) # Remove HTML comments
        return data2
    
    if cache_file is None:
        data2=fetch()
    else:
        if not os.path.exists(cache_file):
            data2=fetch()
            with open(cache_file,'wt') as fp:
                fp.write(data2)
        else:
            # print("Reading from cache")
            logging.info("Cached %s -- %s"%(period_start,period_end))
            with open(cache_file,'rt') as fp:
                data2=fp.read()
                
    df = pd.read_csv(StringIO(data2),skiprows=[1],parse_dates=["Date"] )
    if len(df)==0:
        return None

    # get a real timestamp per station.
    minutes=df.Time.values%100
    hours=df.Time.values//100
    time_of_day=(hours*3600+minutes*60).astype(np.int32) * np.timedelta64(1,'s')
    df['time']=df['Date']+time_of_day
    del df['Time']
    
    # merge in lat/lon
    lonlats=[station_number_to_lonlat(s) for s in df['Station Number']]
    lonlats=np.array(lonlats)
    df['longitude']=lonlats[:,0]
    df['latitude']=lonlats[:,1]
    
    return df


def usgs_sfbay_dataset(start_date, end_date,
                       cache_dir=None, days_per_request='M'):
    """
    Like query_usgs_sfbay, but return xarray Dataset
    Also convert time to UTC (and save original time to time_local).
    """
    polaris=query_usgs_sfbay(period_start=start_date,
                             period_end=end_date,
                             cache_dir=cache_dir,
                             days_per_request=days_per_request)
    
    #hier=polaris.set_index(['Date','Station Number','Depth'])
    # there were 10 rows, 2017-04-04, stations 35 and 36, with duplicate
    # entries. Like they measured the same location, same day, 1 hour apart.
    hier=polaris.groupby(['Date','Station Number','Depth']).first()
    if len(hier) != len(polaris):
        logging.warning("After grouping by date, station and depth, there were some duplicates.")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")


    ds=xr.Dataset.from_dataframe(hier)
    ds=ds.rename({'Station Number':'station','Depth':'depth','Date':'cruise'})
    ds['date']=ds['cruise']

    ds=ds.set_coords(['Julian Date','Days since 1/1/1990','Decimal Date','time',
                      'Distance from 36','longitude','latitude'])
    
    def agg_field(ds,fld,agg):
        with warnings.catch_warnings():
            # ignore RuntimeWarning due to all-nan slices
            # and FutureWarning for potential NaT!=NaT comparison
            warnings.simplefilter('ignore')
            vmin=ds[fld].min(dim=agg)
            vmax=ds[fld].max(dim=agg)
            # funny comparisons to check for either nan/nat or that they
            # are equal.
            if np.any( (vmin==vmin) & (vmin!=vmax) ):
                print("Will not group %s"%fld)
            else:
                ds[fld]=vmin

    # fields that only vary by cruise:
    agg=['station','depth']
    for fld in ['Julian Date','Days since 1/1/1990','Decimal Date']:
        agg_field(ds,fld,agg)

    # fields that vary only by station
    agg=['cruise','depth']
    for fld in ['Distance from 36','longitude','latitude']:
        agg_field(ds,fld,agg)

    # field that do not vary with depth
    agg=['depth']
    for fld in ['time', # maybe?
                'Measured Extinction Coefficient',
                'Calculated Extinction Coefficient']:
        agg_field(ds,fld,agg)

    # Convert times to utc
    ds['time_local']=ds['time'].copy()
    ds.time_local.attrs['timezone']='America/Los (PST/PDT)'
    
    import pytz, datetime
    local = pytz.timezone ("America/Los_Angeles")

    def loc_to_utc(t):
        if utils.isnat(t): return t
        naive = utils.to_datetime(t)
        local_dt = local.localize(naive, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utils.to_dt64(utc_dt)

    tloc=ds.time_local.values
    tutc=ds.time.values
    for idx in np.ndindex(tloc.shape):
        tutc[idx]=loc_to_utc(tloc[idx])
    ds.time.attrs['timezone']='UTC'

    return ds
