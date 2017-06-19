import datetime

import numpy as np
import xarray as xr
import requests
import logging

log=logging.getLogger('noaa_coops')

from ... import utils

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
    meta=json['metadata']
    data=json['data']
    
    ds=xr.Dataset()
    ds['station']=( ('station',), [meta['id']])
    for k in ['name','lat','lon']:
        val=meta[k]
        if k in ['lat','lon']:
            val=float(val)
        ds[k]= ( ('station',), [val])

    times=[]
    values=[]
    qualities=[]
    
    for row in data:
        # {'f': '0,0,0,0', 'q': 'v', 's': '0.012', 't': '2010-12-01 00:00', 'v': '0.283'}
        values.append(float(row['v']))
        times.append( np.datetime64(row['t']) )
        # for now, ignore flags, verified status.
    ds['time']=( ('time',),times)
    ds[params['product']]=( ('station','time'), [values] )

    if params['product'] == 'water_level':
        ds[params['product']].attrs['datum'] = params['datum']
        
    return ds


def coops_dataset(station,start_date,end_date,products,
                  days_per_request=None):
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
        ds_per_product.append( coops_dataset_product(station=station,
                                                     product=product,
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     days_per_request=days_per_request) )
    # punt for the moment - no real support for multiple datasets...
    assert len(products)==1
    return ds_per_product[0]

def coops_dataset_product(station,product,
                          start_date,end_date,days_per_request=None):
    """
    Retrieve a single data product from a single station.
    station: string or numeric identifier for COOPS station
    product: string identifying the variable to retrieve.  See all_products at 
      the top of this file.
    start_date,end_date: period to retrieve, as python datetime, matplotlib datenum,
      or numpy datetime64.
    days_per_request: batch the requests to fetch smaller chunks at a time.

    returns an xarray dataset
    """
    fmt_date=lambda d: utils.to_datetime(d).strftime("%Y%m%d %H:%M")
    base_url="https://tidesandcurrents.noaa.gov/api/datagetter"

    # not supported by this script: bin
    datums=['NAVD','MSL']

    def periods(start_date,end_date,days_per_request):
        start_date=utils.to_datetime(start_date)
        end_date=utils.to_datetime(end_date)

        if days_per_request is None:
            yield (start_date,end_date)
        else:
            interval=datetime.timedelta(days=days_per_request)

            while start_date<end_date:
                next_date=min(start_date+interval,end_date)
                yield (start_date,next_date)
                start_date=next_date

    datasets=[]

    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        log.info("Fetching %s -- %s"%(interval_start,interval_end))

        params=dict(begin_date=fmt_date(interval_start),
                    end_date=fmt_date(interval_end),
                    station=str(station),
                    time_zone='gmt', # always!
                    application='stompy',
                    units='metric',
                    format='json',
                    product=product)
        if product in ['water_level','hourly_height',"one_minute_water_level"]:
            while 1:
                # not all stations have NAVD, so fall back to MSL
                params['datum']=datums[0] 
                req=requests.get(base_url,params=params)
                data=req.json()
                if ('error' in data) and ("datum" in data['error']['message'].lower()):
                    # Actual message like 'The supported Datum values are: MHHW, MHW, MTL, MSL, MLW, MLLW, LWI, HWI'
                    log.info(data['error']['message'])
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
                continue

        ds=coops_json_to_ds(data,params)    
        datasets.append(ds)

    if len(datasets)==0:
        # could try to construct zero-length dataset, but that sounds like a pain
        # at the moment.
        return None 

    if len(datasets)>1:
        dataset=xr.concat( datasets, dim='time')
    else:
        dataset=datasets[0]
    return dataset
