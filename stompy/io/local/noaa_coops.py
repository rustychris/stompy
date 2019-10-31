import os
import datetime

import numpy as np
import xarray as xr
import requests
import logging

log=logging.getLogger('noaa_coops')

from ... import utils
from .common import periods

all_products=dict(
    water_level="water_level",
    air_temperature="air_temperature",
    water_temperature="water_temperature",
    wind="wind",
    air_pressure="air_pressure",
    air_gap="air_gap",
    conductivity="conductivity",
    visibility="visibility",
    humidity="humidity",
    salinity="salinity",
    hourly_height="hourly_height",
    high_low="high_low",
    daily_mean="daily_mean",
    monthly_mean="monthly_mean",
    one_minute_water_level="one_minute_water_level",
    predictions="predictions",
    datums="datums",
    currents="currents")

all_datums=dict(
    CRD="Columbia River Datum",
    MHHW="Mean Higher High Water",
    MHW="Mean High Water",
    MTL="Mean Tide Level",
    MSL="Mean Sea Level",
    MLW="Mean Low Water",
    MLLW="Mean Lower Low Water",
    NAVD="North American Vertical Datum",
    STND="Station Datum")

def coops_json_to_ds(json,params):
    """ Mold the JSON response from COOPS into a dataset
    """
    
    ds=xr.Dataset()
    if 'metadata' in json:
        meta=json['metadata']
    
        ds['station']=( ('station',), [meta['id']])
        for k in ['name','lat','lon']:
            val=meta[k]
            if k in ['lat','lon']:
                val=float(val)
            ds[k]= ( ('station',), [val])
    else:
        # predictions do not come back with metadata
        ds['station']= ('station',),[params['station']]

    times=[]
    values=[]
    qualities=[]

    if 'data' in json:
        data=json['data']
    elif 'predictions' in json:
        # Why do they present predictions data in such a different format?
        data=json['predictions']
        
    for row in data:
        # {'f': '0,0,0,0', 'q': 'v', 's': '0.012', 't': '2010-12-01 00:00', 'v': '0.283'}
        try:
            values.append(float(row['v']))
        except ValueError:
            values.append(np.nan)
        times.append( np.datetime64(row['t']) )
        # for now, ignore flags, verified status.
    ds['time']=( ('time',),times)
    ds[params['product']]=( ('station','time'), [values] )

    bad_count=np.sum( np.isnan(values) )
    if bad_count:
        log.warning("%d of %d data values were missing"%(bad_count,len(values)))
        
    if params['product'] in ['water_level','predictions']:
        ds[params['product']].attrs['datum'] = params['datum']
        
    return ds


def coops_dataset(station,start_date,end_date,products,
                  days_per_request=None,cache_dir=None):
    """
    bare bones retrieval script for NOAA Tides and Currents data.
    In particular, no error handling yet, doesn't batch requests, no caching,
    can't support multiple parameters, no metadata, etc.

    days_per_request: break up the request into chunks no larger than this many
    days.  for hourly data, this should be less than 365.  for six minute, I think
    the limit is 32 days.
    """

    ds_per_product=[]

    for product in products:
        ds=coops_dataset_product(station=station,
                                 product=product,
                                 start_date=start_date,
                                 end_date=end_date,
                                 days_per_request=days_per_request,
                                 cache_dir=cache_dir)
        if ds is not None:
            ds_per_product.append(ds)
    ds_merged=xr.merge(ds_per_product,join='outer')
    return ds_merged

def coops_dataset_product(station,product,
                          start_date,end_date,days_per_request='M',
                          cache_dir=None,refetch_incomplete=True,
                          clip=True):
    """
    Retrieve a single data product from a single station.
    station: string or numeric identifier for COOPS station
    product: string identifying the variable to retrieve, such as "water_level".
      See all_products at the top of this file.
    start_date,end_date: period to retrieve, as python datetime, matplotlib datenum,
    or numpy datetime64.
    days_per_request: batch the requests to fetch smaller chunks at a time.
    if this is an integer, then chunks will start with start_date, then start_date+days_per_request,
    etc.
      if this is a string, it is interpreted as the frequency argument to pandas.PeriodIndex.
    so 'M' will request month-aligned chunks.  this has the advantage that requests for different
    start dates will still be aligned to integer periods, and can reuse cached data.
   
    cache_dir: if specified, save each chunk as a netcdf file in this directory,
      with filenames that include the gage, period and products.  The directory must already
      exist.

    returns an xarray dataset, or None if no data could be fetched

    refetch_incomplete: if True, if a dataset is pulled from cache but appears incomplete
      with respect to the start_date and end_date, attempt to fetch it again.  Not that incomplete
      here is meant for realtime data which has not yet been recorded, so the test is only
      between end_date and the last time stamp of retrieved data.

    clip: if true, return only data within the requested window, even if more data was fetched.
    """
    fmt_date=lambda d: utils.to_datetime(d).strftime("%Y%m%d %H:%M")
    base_url="https://tidesandcurrents.noaa.gov/api/datagetter"

    # not supported by this script: bin
    datums=['NAVD','MSL']

    datasets=[]

    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        if cache_dir is not None:
            begin_str=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
            end_str  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

            cache_fn=os.path.join(cache_dir,
                                  "%s_%s_%s_%s.nc"%(station,
                                                    product,
                                                    begin_str,
                                                    end_str))
        else:
            cache_fn=None

        ds=None
        if (cache_fn is not None) and os.path.exists(cache_fn):
            log.info("Cached   %s -- %s"%(interval_start,interval_end))
            ds=xr.open_dataset(cache_fn)
            if refetch_incomplete:
                # This will fetch a bit more than absolutely necessary
                # In the case that this file is up to date, but the sensor was down,
                # we might be able to discern that if this was originally fetched
                # after another request which found valid data from a later time.
                if ds.time.values[-1]<min(utils.to_dt64(interval_end),
                                          end_date):
                    log.warning("   but that was incomplete -- will re-fetch")
                    ds=None
        if ds is None:
            log.info("Fetching %s -- %s"%(interval_start,interval_end))

            params=dict(begin_date=fmt_date(interval_start),
                        end_date=fmt_date(interval_end),
                        station=str(station),
                        time_zone='gmt', # always!
                        application='stompy',
                        units='metric',
                        format='json',
                        product=product)
            if product in ['water_level','hourly_height',"one_minute_water_level","predictions"]:
                while 1:
                    # not all stations have NAVD, so fall back to MSL
                    params['datum']=datums[0] 
                    req=requests.get(base_url,params=params)
                    try:
                        data=req.json()
                    except ValueError: # thrown by json parsing
                        log.warning("Likely server error retrieving JSON data from tidesandcurrents.noaa.gov")
                        data=dict(error=dict(message="Likely server error"))
                        break
                    if (('error' in data)
                        and (("datum" in data['error']['message'].lower())
                             or (product=='predictions'))):
                        # Actual message like 'The supported Datum values are: MHHW, MHW, MTL, MSL, MLW, MLLW, LWI, HWI'
                        # Predictions sometimes silently fail, as if there is no data, but really just need
                        # to try MSL.
                        log.debug(data['error']['message'])
                        datums.pop(0) # move on to next datum
                        continue # assume it's because the datum is missing
                    break
            else:
                req=requests.get(base_url,params=params)
                data=req.json()

            if 'error' in data:
                msg=data['error']['message']
                if "No data was found" in msg:
                    # station does not have this data for this time.
                    log.warning("No data found for this period")
                else:
                    # Regardless, if there was an error we got no data.
                    log.warning("Unknown error - got no data back.")
                    log.debug(data)
                    
                log.debug("URL was %s"%(req.url))
                continue

            ds=coops_json_to_ds(data,params)
            if cache_fn is not None:
                ds.to_netcdf(cache_fn)

        if len(datasets)>0:
            # avoid duplicates in case they overlap
            ds=ds.isel(time=ds.time.values>datasets[-1].time.values[-1])
        datasets.append(ds)

    if len(datasets)==0:
        # could try to construct zero-length dataset, but that sounds like a pain
        # at the moment.
        return None 

    if len(datasets)>1:
        # data_vars='minimal' is needed to keep lat/lon from being expanded
        # along the time axis.
        dataset=xr.concat( datasets, dim='time',data_vars='minimal')
    else:
        dataset=datasets[0].copy(deep=True)
    # better not to leave these lying around open
    for d in datasets:
        d.close()

    if clip:
        time_sel=(dataset.time.values>=start_date) & (dataset.time.values<end_date)
        dataset=dataset.isel(time=time_sel)
        
    return dataset
