import requests
import pandas as pd
import xarray as xr
import numpy as np
import pickle
import logging as log
import json
import os
from ... import utils

from .common import periods

##

root="https://environment.data.gov.uk/flood-monitoring"

def get_tide_gauges_json(cache_dir):
    if cache_dir is not None:
        cache_fn=os.path.join(cache_dir,"uk_tides/stations.json")
        if os.path.exists(cache_fn):
            log.debug("Tide gauge list from cache (%s)"%cache_fn)
            with open(cache_fn,'rt') as fp:
                return fp.read()
    else:
        cache_fn=None

    all_stations=root+"/id/stations?type=TideGauge"
    log.debug("Fetching tide gauge list")
    res=requests.get(all_stations)

    if cache_fn is not None:
        dname=os.path.dirname(cache_fn)
        if not os.path.exists(dname):
            os.makedirs(dname)
            
        with open(cache_fn,'wt') as fp:
            fp.write(res.text)
    return res.text
        

def get_tide_gauges(cache_dir):
    return json.loads(get_tide_gauges_json(cache_dir))

def find_station(cache_dir,label=None,station=None):
    stations=get_tide_gauges(cache_dir=cache_dir)

    def match(rec):
        if (label is not None) and (rec['label']==label):
            return True
        if (station is not None) and (rec['stationReference']==station):
            return True
        return False
            
    for rec in stations['items']:
        if match(rec) and rec['measures'][0]['unitName']=='mAOD':
            return rec

    log.warning("No AOD measure found")
    for rec in stations['items']:
        if match(rec):
            return rec

    log.error("No Station found for %s"%label)
    return None

# To fetch a specific station for a specific time
def fetch_tides(start_date,end_date,cache_dir,
                station=None,label=None,
                days_per_request='5D',cache_only=False):
    """
    Retrieve a single data product from a single station.
    station: id like E72124
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
    """
    if station is not None:
        station_meta=find_station(station=station,cache_dir=cache_dir)
    else:
        assert label is not None,"Specify one of station or label"
        station_meta=find_station(label=label,cache_dir=cache_dir)
    if station_meta is not None:
        station=station_meta['stationReference']
        
    fmt_date=lambda d: utils.to_datetime(d).strftime("%Y-%m-%d")

    datasets=[]

    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        if cache_dir is not None:
            begin_str=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
            end_str  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

            cache_fn=os.path.join(cache_dir,
                                  'uk_tides',
                                  "%s_%s_%sa.nc"%(station,
                                                  begin_str,
                                                  end_str))
        else:
            cache_fn=None

        ds=None
        log.debug("cache_fn: %s"%cache_fn)
        
        if (cache_fn is not None) and os.path.exists(cache_fn):
            log.debug("Cached   %s -- %s"%(interval_start,interval_end))
            ds=xr.open_dataset(cache_fn)
        elif cache_only:
            continue
        if (not cache_only) and (ds is None):
            log.info("Fetching %s -- %s"%(interval_start,interval_end))

            params=dict(startdate=fmt_date(interval_start),
                        enddate=fmt_date(interval_end),
                        _limit=2000)

            url=f"{root}/id/stations/{station}/readings"
            
            req=requests.get(url,params=params)
            try:
                data=req.json()
            except ValueError: # thrown by json parsing
                log.warning("Likely server error retrieving JSON data from environment.data.gov.uk")
                data=dict(error=dict(message="Likely server error"))
                break

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

            ds=json_to_ds(data,params)

            if ds is not None:
                if cache_fn is not None:
                    dname=os.path.dirname(cache_fn)
                    if not os.path.exists(dname):
                        os.makedirs(dname)
                    ds.to_netcdf(cache_fn)
            else:
                continue
        # seems these don't come in order
        ds=ds.sortby(ds.time)

        if len(datasets)>0:
            # avoid duplicates in case they overlap
            ds=ds.isel(time=ds.time.values>datasets[-1].time.values[-1])
        datasets.append(ds)

    if len(datasets)==0:
        # could try to construct zero-length dataset, but that sounds like a pain
        # at the moment.
        return None 

    if len(datasets)>1:
        dataset=xr.concat( datasets, dim='time')
    else:
        dataset=datasets[0].copy(deep=True)
    # better not to leave these lying around open
    for d in datasets:
        d.close()

    # add in metadata
    dataset['lat']=(), station_meta['lat']
    dataset['lon']=(), station_meta['long']
    dataset['label']=(), station_meta['label']
    dataset['station']=(),station_meta['stationReference']
    dataset['value'].attrs['units']=station_meta['measures'][0]['unitName']

    return dataset
    
def json_to_ds(data,params):
    if len(data['items']) == data['meta']['limit']:
        log.warning("May have hit limit - should pull shorter windows")
    
    rows=[]
    for d in data['items']:
        try:
            t=np.datetime64(d['dateTime'].strip('Z'))
        except Exception:
            t=None

        rec=dict(time=t,
                 measure=d['measure'],
                 value=d['value'])
        rows.append(rec)

    if len(rows)==0:
        return None
    
    df=pd.DataFrame(rows)
    xds=xr.Dataset()
    xds['time']=('time',),df.time
    xds['value']=('time',),df.value

    measure_uniq=df.measure.unique()

    if len(measure_uniq)==1:
        xds['measure']=(),measure_uniq[0]
    else:
        log.warning("Multiple types of measures")
        xds['measure']=('time'),df.measure

    return xds

##

