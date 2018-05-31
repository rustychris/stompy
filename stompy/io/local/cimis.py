"""
Programmatic access to CIMIS data

The CIMIS API requires a key, which can be obtained for free.  That key will be 
read in from CIMIS_KEY in the environment, or can be passed into these functions
as the cimis_key keyword argument
"""

from __future__ import print_function
import time
import numpy as np
import xarray as xr
import datetime
import requests
import os

from ... import utils

from .common import periods

## 

url="http://et.water.ca.gov/api/data"

def cimis_json_to_xr(data):
    """
    Reformat a JSON result from CIMIS into an xarray dataset.
    """
    df=xr.Dataset()

    data2=data['Data']['Providers'][0]

    # Owner, Records, Type, Name
    df.attrs['data_owner'] = data2['Owner']
    df.attrs['station_type'] = data2['Type']
    df.attrs['service_name'] = data2['Name']

    records=data2['Records']
    # 768 records..

    if len(records)==0:
        print("No data")
        return None

    df['Date']= ( ('Date',), [ "%s %s"%(rec['Date'],rec['Hour'])
                               for rec in records] )

    def cnv(s):
        try:
            return float(s)
        except (ValueError,TypeError):
            return np.nan

    for field in ['HlyAirTmp', 'HlyEto','HlyNetRad',
                  'HlyPrecip','HlyRelHum','HlyResWind',
                  'HlySolRad','HlyWindDir','HlyWindSpd']:
        # use zip to transpose the lists
        qc,value = zip( *[ (rec[field]['Qc'],cnv(rec[field]['Value']))
                           for rec in records ] )
        df[field]=( ('Date',), np.array(value) )
        df[field+'_qc'] = ( ('Date',), np.array(qc) )

        df[field].attrs['units']=records[0][field]['Unit']

    df.attrs['station_num']=int(records[0]['Station'])

    return df

def cimis_fetch_station_metadata(station,df=None,cimis_key=None):
    """
    Return an xr.Dataset with station metadata for the station ID
    (integer) supplied.
    cimis_key is not needed, but accepted.
    """
    df=df or xr.Dataset()
    req=requests.get("http://et.water.ca.gov/api/station/%s"%station,
                     headers=dict(Accept='application/json'))
    station_meta=req.json()

    # add station metadata to attrs:
    stn=data['Stations'][0]

    df.attrs['elevation'] = float(stn['Elevation'])
    df.attrs['is_active'] = stn['IsActive']
    df.attrs['station_name']=stn['Name']
    lat=float(stn['HmsLatitude'].split('/')[1]) #  u"37\xba35'56N / 37.598758"
    lon=float(stn['HmsLongitude'].split('/')[1])

    df.attrs['latitude']=lat
    df.attrs['longitude']=lon
    return df

def cimis_fetch_to_xr(stations, # Union City
                      start_date,end_date,
                      fields=None,
                      station_meta=True,
                      days_per_request=None,cache_dir=None,
                      cimis_key=None):
    if cimis_key is None:
        try:
            cimis_key=os.environ['CIMIS_KEY']
        except KeyError:
            raise Exception("cimis_key was not supplied and CIMIS_KEY was not in the environment")
        
    if fields is None:
        fields=['hly-air-tmp','hly-eto',
                'hly-net-rad','hly-precip',
                'hly-rel-hum','hly-res-wind',
                'hly-sol-rad','hly-wind-dir',
                'hly-wind-spd']

    if isinstance(stations,np.int) or isinstance(stations,str):
        stations=[stations]

    stations=[str(s) for s in stations] 

    start_date,end_date=[ utils.to_datetime(d).strftime('%Y-%m-%d')
                          for d in (start_date,end_date) ]

    all_ds=[]
    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        req=requests.get(url,params=dict(appKey=cimis_key,
                                         targets=",".join(stations), 
                                         startDate=start_date,
                                         endDate=end_date,
                                         unitOfMeasure='M', # metric please
                                         dataItems=",".join( fields ) ))
        ds=cimis_json_to_xr(req.json())
        all_ds.append(ds)
        time.sleep(1.5) # be kind

    if 1:
        return all_ds # DBG

    if len(all_ds)==1:
        ds=all_ds[0]
    elif len(add_ds)==0:
        return None
    else:
        ds=xr.concat( all_ds, dim='time')

    if station_meta:
        if len(stations)!=1:
            print("Can only record station metadata for a single station")
        else:
            cimis_fetch_station_metadata(stations[0],df=ds)

    return ds

## 

