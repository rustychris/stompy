import datetime
import os
import logging
import re
import six

#from six.moves import cPickle

import numpy as np
import xarray as xr
import pandas as pd
from pandas.compat import StringIO
import requests

log=logging.getLogger('cdec')

from ... import utils
from .common import periods

try:
    import seawater
except ImportError:
    seawater=None

def cdec_df_to_ds(df):
    ds=xr.Dataset()

    times=[datetime.datetime.strptime(s,"%Y%m%d %H%M") for s in df['DATE TIME']]
    ds['time']=('time',),times
    if not times:
        # force type, which for empty list defaults to float64
        ds['time']=ds.time.values.astype('<M8')

    # make the adjustment to UTC, even though this is daily data...
    ds.time.values[:] += np.timedelta64(8,'h')
    # not convention, but a nice reminder
    ds.time.attrs['timezone']='UTC'

    sensor_num=df['SENSOR_NUMBER'][0]
    vname='sensor%04d'%sensor_num
    ds[vname]=('time',),df['VALUE']
    ds[vname].attrs['units']=df['UNITS'][0]
    ds[vname].attrs['station']=df['STATION_ID'][0]
    return ds


def cdec_dataset(station,start_date,end_date,sensor,
                 days_per_request='M',duration='E',
                 cache_dir=None,clip=True,cache_only=False):
    """
    Retrieval script for CDEC csv

    Retrieve one sensor from a single station.
    station: string, e.g. "BKS"

    sensor: integer, e.g. 70 for discharge, pumping

    start_date,end_date: period to retrieve, as python datetime, matplotlib datenum,
    or numpy datetime64.

    days_per_request: batch the requests to fetch smaller chunks at a time.
    if this is an integer, then chunks will start with start_date, then start_date+days_per_request,
    etc.
      if this is a string, it is interpreted as the frequency argument to pandas.PeriodIndex.
    so 'M' will request month-aligned chunks.  this has the advantage that requests for different
    start dates will still be aligned to integer periods, and can reuse cached data.

    duration: CDEC duration codes. H: hourly, E: event (I think this means the frequency
      of the original observations.).  D: daily.

    cache_dir: if specified, save each chunk as a netcdf file in this directory,
      with filenames that include the gage, period and products.  The directory must already
      exist.

    clip: if True, then even if more data was fetched, return only the period requested.

    cache_only: If true, only read from cache, not attempting to fetch any new data.

    returns an xarray dataset.  note that TIMES ARE UTC in the returned dataset.

    CDEC returns all data in PST, but for consistency with other io.local modules
    those are converted to UTC in this script
    """
    start_date=utils.to_dt64(start_date)
    end_date=utils.to_dt64(end_date)

    # URL is something like:
    # http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet?Stations=BKS&SensorNums=70&dur_code=D&Start=2018-07-05&End=2018-10-05

    params=dict(Stations=station,
                SensorNums=str(sensor),
                dur_code=duration)

    # Only for small requests of recent data:
    base_url="http://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"

    datasets=[]

    last_url=None

    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        params['Start']=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
        params['End']  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

        base_fn="%s_%s%s_%s_%s.nc"%(station,
                                    str(sensor), duration,
                                    params['Start'],
                                    params['End'])
        if cache_dir is not None:
            cache_fn=os.path.join(cache_dir,base_fn)
        else:
            cache_fn=None

        if (cache_fn is not None) and os.path.exists(cache_fn):
            log.info("Cached   %s -- %s"%(interval_start,interval_end))
            ds=xr.open_dataset(cache_fn)
        elif cache_only:
            log.info("Cache only - no data for %s -- %s"%(interval_start,interval_end))
            continue
        else:
            log.info("Fetching %s"%(base_fn))
            req=requests.get(base_url,params=params)
            df=pd.read_csv(StringIO(req.text),na_values=['---'])
            if len(df)==0:
                continue

            # debugging a parsing problem
            if not (np.issubdtype(df['VALUE'].dtype,np.integer) or
                    np.issubdtype(df['VALUE'].dtype,np.floating)):
                # in case they change their nan string
                log.warning("Some VALUE items may not have been parsed")
            ds=cdec_df_to_ds(df)
            ds.attrs['url']=req.url

            if cache_fn is not None:
                ds.to_netcdf(cache_fn)

        # in case returned data is inclusive of the requested date range
        # avoid overlap
        if len(datasets):
            ds=ds.isel(time=ds.time>datasets[-1].time[-1])
        datasets.append(ds)

    if len(datasets)==0:
        # could try to construct zero-length dataset, but that sounds like a pain
        # at the moment.
        log.warning("   no data for station %s for any periods!"%station)
        return None 

    if len(datasets)>1:
        # it's possible that not all variables appear in all datasets
        dataset=datasets[0]
        for other in datasets[1:]:
            dataset=dataset.combine_first(other)
        for stale in datasets:
            stale.close() # maybe free up FDs?
    else:
        dataset=datasets[0]

    if clip:
        time_sel=(dataset.time.values>=start_date) & (dataset.time.values<end_date)
        dataset=dataset.isel(time=time_sel)

    dataset.load() # force read into memory before closing files
    for d in datasets:
        d.close()

    return dataset


# def add_salinity(ds):
#     assert seawater is not None
#     for v in ds.data_vars:
#         if v.startswith('specific_conductance'):
#             salt_name=v.replace('specific_conductance','salinity')
#             if salt_name not in ds:
#                 print("%s => %s"%(v,salt_name))
#                 salt=seawater.eos80.salt(ds[v].values/1000. / seawater.constants.c3515,
#                                          25.0, # temperature - USGS adjusts to 25degC
#                                          0) # no pressure effects
#                 ds[salt_name]=ds[v].dims, salt
# 
# def station_metadata(station,cache_dir=None):
#     if cache_dir is not None:
#         cache_fn=os.path.join(cache_dir,"meta-%s.pkl"%station)
# 
#         if os.path.exists(cache_fn):
#             with open(cache_fn,'rb') as fp:
#                 meta=cPickle.load(fp)
#             return meta
# 
#     url="https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=%s"%station
# 
#     resp=requests.get(url)
# 
#     m=re.search(r"Latitude\s+([.0-9&#;']+\")",resp.text)
#     lat=m.group(1)
#     m=re.search(r"Longitude\s+([.0-9&#;']+\")",resp.text)
#     lon=m.group(1)
# 
#     def dms_to_dd(s):
#         s=s.replace('&#176;',' ').replace('"',' ').replace("'"," ").strip()
#         d,m,s =[float(p) for p in s.split()]
#         return d + m/60. + s/3600.
#     lat=dms_to_dd(lat)
#     # no mention of west longitude, but can assume it is west.
#     lon=-dms_to_dd(lon)
#     meta=dict(lat=lat,lon=lon)
# 
#     if cache_dir is not None:
#         with open(cache_fn,'wb') as fp:
#             cPickle.dump(meta,fp)
#     return meta
