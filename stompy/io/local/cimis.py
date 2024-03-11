"""
Programmatic access to CIMIS data

The CIMIS API requires a key, which can be obtained for free.  That key will be 
read in from CIMIS_KEY in the environment, or can be passed into these functions
as the cimis_key keyword argument

cimis_fetch_to_xr(), documented below, is the main entry point
"""

from __future__ import print_function
import time
import numpy as np
import xarray as xr
import datetime
import requests
from requests.compat import json

import os
import logging as log

from ... import utils

from .common import periods

## 

# Web service requests URL
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

    if records[0]['Scope']=='daily':
        df['Date']= ( ('time',), [ rec['Date']
                                   for rec in records] )
        datetimes=[ datetime.datetime.strptime(s,"%Y-%m-%d")
                    for s in df.Date.values ]
    else: # hourly
        df['Date']= ( ('time',), [ "%s %s"%(rec['Date'],rec['Hour'])
                                   for rec in records] )
        # annoying convention where midnight is 2400.
        datetimes=[ (datetime.datetime.strptime(s.replace('2400','0000'),
                                                '%Y-%m-%d %H%M')
                     + datetime.timedelta(days=int(s.endswith('2400'))))
                    for s in df.Date.values]
    dt64s=[utils.to_dt64(dt) for dt in datetimes]
    df['time']=('time',),np.asarray(dt64s,'M8[ns]') # appease pandas

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
        df[field]=( ('time',), np.array(value) )
        df[field+'_qc'] = ( ('time',), np.array(qc) )

        df[field].attrs['units']=records[0][field]['Unit']

    df.attrs['station_num']=int(records[0]['Station'])

    return df

def cimis_fetch_station_metadata(station,df=None,cimis_key=None,cache_dir=None):
    """
    Return an xr.Dataset with station metadata for the station ID
    (integer) supplied.
    cimis_key is not needed, but accepted.
    """
    if df is None:
        df=xr.Dataset()
    if cache_dir is not None:
        assert os.path.exists(cache_dir)

        # Be nice and make a cimis subdirectory
        cache_sub_dir=os.path.join(cache_dir,'cimis')
        os.path.exists(cache_sub_dir) or os.mkdir(cache_sub_dir)
        cache_fn=os.path.join(cache_sub_dir,"station_metadata-%s.json"%station)
    else:
        cache_fn=None

    # The Latin-1 business here is because CIMIS uses 0xBA for a degree sign, and
    # that trips up python unicode.  Latin-1 in theory means don't transform any
    # bytes -- just write it out, and pretend we all agree on the high byte symbols.
    if (cache_fn is not None) and os.path.exists(cache_fn):
        log.info("Station metadata from cache")
        with open(cache_fn,'rb') as fp:
            station_meta=json.loads(fp.read().decode('Latin-1'))
    else:
        log.info("Station metadata from download")
        req=requests.get("http://et.water.ca.gov/api/station/%s"%station,
                         headers=dict(Accept='application/json'))
        if cache_fn is not None:
            with open(cache_fn,'wb') as fp:
                fp.write(req.text.encode('Latin-1'))
        station_meta=req.json()

    # add station metadata to attrs:
    stn=station_meta['Stations'][0]

    df.attrs['elevation'] = float(stn['Elevation'])
    df.attrs['is_active'] = stn['IsActive']
    df.attrs['station_name']=stn['Name']
    lat=float(stn['HmsLatitude'].split('/')[1]) #  u"37\xba35'56N / 37.598758"
    lon=float(stn['HmsLongitude'].split('/')[1])

    df.attrs['latitude']=lat
    df.attrs['longitude']=lon
    return df

def cimis_fetch_to_xr(stations, 
                      start_date,end_date,
                      fields=None,
                      station_meta=True,
                      days_per_request="10D",cache_dir=None,
                      cimis_key=None):
    """
    Download data for one or more CIMIS stations.

    stations: a single value or a list of values giving the numeric id of the station(s)
      IDs can be supplied as an integer or string.  e.g. 171, [171], "171", ["168",142]

    start_date,end_date: period to download.  Can be specified as floating point datenum,
     python datetime, or numpy datetime64.

    fields: None to fetch all known fields, or ["hly-eto","hly-sol-rad",...] to get a subset.
    
    station_meta: Download station metadata 

    days_per_request: break the request up into chunks no longer than this many days.
      Optionally specify a pandas PeriodIndex string (e.g. "10D" for 10 day periods).  This
      has the added effect of making the requests line up to even multiples of the given 
      period, which allows for caching results even when requests are not for the same
      period.

    cache_dir: path to a directory which already exists, to hold cached data.  This is 
      currently only used when days_per_request is a pandas-style string.  Providing
      this path enables caching.

    cimis_key: CIMIS data download requires an API key.  This parameter will override
      the CIMIS_KEY environment variable.


    Returns: an xarray Dataset
    """
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
        fields_label="all"
    else:
        fields_label="_".join(fields)

    if isinstance(stations,np.integer) or isinstance(stations,str):
        stations=[stations]

    stations=[str(s) for s in stations] 

    interval_dt64=np.timedelta64(int(days_per_request[:-1]), days_per_request[-1])
    # quantize to standard t0
    start_date=utils.floor_dt64( utils.to_dt64(start_date), interval_dt64)
    end_date=utils.ceil_dt64( utils.to_dt64(end_date), interval_dt64)
    
    start_date,end_date=[ utils.to_datetime(d).strftime('%Y-%m-%d')
                          for d in (start_date,end_date) ]

    all_ds=[]
    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        if cache_dir is not None:
            begin_str=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
            end_str  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

            # Coming up with a good cache name is a bit tricky since it might have a bunch
            # of fields, and a bunch of stations.
            # Most typical situation is fetching all fields for one station, and at least
            # that usage yields reasonable filenames here
            cache_fn=os.path.join(cache_dir,
                                  "%s_%s_%s_%s.nc"%("_".join(stations),
                                                    fields_label,
                                                    begin_str,
                                                    end_str))
        else:
            cache_fn=None

        if (cache_fn is not None) and os.path.exists(cache_fn):
            log.info("Cached   %s -- %s"%(interval_start,interval_end))
            ds=xr.open_dataset(cache_fn)
        else:
            log.info("Requesting %s to %s"%(interval_start,interval_end))
            log.info("CIMIS: %s"%url)
            params=dict(appKey=cimis_key,
                        targets=",".join(stations), 
                        startDate=begin_str,
                        endDate=end_str,
                        unitOfMeasure='M', # metric please
                        # is this causing problems?
                        dataItems=",".join( fields ))
            log.info(str(params))
            try:
                req=requests.get(url,params=params)
            except requests.ConnectError:
                log.warning("CIMIS connction error")
                time.sleep(1.0)
                continue
            
            try:
                req_json=req.json()
            except json.JSONDecodeError:
                log.warning("CIMIS request json decode error: ")
                log.warning(f"  url was: {req.url}")
                log.warning(req.text[:200])
                time.sleep(1.0)
                continue
                
            if isinstance(req_json,str):
                log.warning("CIMIS request json is just a string?: ")
                log.warning(f"  url was: {req.url}")
                log.warning(req_json)
                time.sleep(1.0)
                continue
            
            try:
                ds=cimis_json_to_xr(req_json)
            except Exception:
                log.warning("While requesting from:")
                log.warning(req.url)
                raise
            
            if (ds is not None) and (cache_fn is not None):
                ds.to_netcdf(cache_fn)
            time.sleep(1.0) # be kind

        if ds is not None:
            all_ds.append(ds)

    if len(all_ds)==1:
        ds=all_ds[0]
    elif len(all_ds)==0:
        return None
    else:
        ds=xr.concat(all_ds, dim='time')
        # account for overlaps
        sel=utils.select_increasing(ds.time.values)
        [x.close() for x in all_ds] # reduce the number of netcdf handles left open
        ds=ds.isel(time=sel)

    if station_meta:
        if len(stations)!=1:
            print("Can only record station metadata for a single station")
        else:
            cimis_fetch_station_metadata(stations[0],df=ds,cache_dir=cache_dir)

    return ds

