import numpy as np
import xarray as xr
import requests

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
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     days_per_request=days_per_request) )
    # punt for the moment - no real support for multiple datasets...
    assert len(products)==1
    return ds_per_product[0]

def coops_dataset_product(station,start_date,end_date,days_per_request=None):
    fmt_date=lambda d: utils.to_datetime(d).strftime("%Y%m%d %H:%M")
    base_url="https://tidesandcurrents.noaa.gov/api/datagetter"

    # not supported by this script: bin
    datums=['NAVD','MSL']

    if days_per_request is None:
        period_slices=[ (start_date,end_date) ]
    else:

        
        params=dict(begin_date=fmt_date(start_date),
                    end_date=fmt_date(end_date),
                    station=str(station),
                    time_zone='gmt', # always!
                    application='SFEI',
                    units='metric',
                    format='json',
                    product=product)
        if product in ['water_level','hourly_height',"one_minute_water_level"]:
            for datum in datums:
                params['datum']=datum # not all stations have this, though.
                req=requests.get(base_url,params=params)
                if 'error' in req.json():
                    continue # assume it's because the datum is missing
                break
        else:
            req=requests.get(base_url,params=params)

        data=req.json()

    ds=coops_json_to_ds(data,params)    
    return ds
