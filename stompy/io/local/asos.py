# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:04:47 2022

@author: rusty
"""
import os
import datetime

import numpy as np
import xarray as xr
import requests
import logging

log=logging.getLogger('noaa_coops')

from ... import utils
#from .common import periods




def asos_dataset(station,
                 start_date,end_date,
                 cache_dir=None,refetch_incomplete=True,
                 clip=True):
    """
    Retrieve ASOS data for one station
    station: 4 letter string identifier (i.e. KHWD for Hayward airport)
    start_date,end_date: period to retrieve, as python datetime, matplotlib datenum,
    or numpy datetime64.   
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
    start_date=utils.to_dt64(start_date)
    end_date=utils.to_dt64(end_date)
    
    fmt_date=lambda d: utils.to_datetime(d).strftime("%Y%m%d %H:%M")
    base_url="https://www.ncei.noaa.gov/pub/data/asos-fivemin/"
    # remainder of URL: 6401-2022/64010KHWD202201.dat

    datasets=[]

    # ASOS data is provided in 1-month chunks
    for interval_start,interval_end in periods(start_date,end_date,'1M'):
        int_start_dt=utils.to_datetime(interval_start)
        int_end_dt  =utils.to_datetime(interval_end)
        if cache_dir is not None:
            begin_str=int_start_dt.strftime('%Y-%m-%d')
            end_str  =int_end_dt.strftime('%Y-%m-%d')
            cache_fn=os.path.join(cache_dir,
                                  "%s_%s_%s_%s.nc"%(station,
                                                    begin_str))
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
                # For ASOS data, may need to make this a bit more relaxed, depending
                # on the exact timestamp of the last datum.
                if ds.time.values[-1]<min(utils.to_dt64(interval_end),
                                          end_date):
                    log.warning("   but that was incomplete -- will re-fetch")
                    ds=None
        if ds is None:

            year=int_start_dt.year
            month=int_start_dt.month
            
            url=base_url+f"6401-{year}/64010{station}{year:04d}{month:02d}.dat"
            ds=asos_url_to_dataset(url)
            if cache_fn is not None:
                if os.path.exists(cache_fn):
                    # simply overwriting often does not work, so try removing first
                    os.unlink(cache_fn)
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

    dataset['time'].attrs['timezone']='UTC'
        
    return dataset

def asos_url_to_dataset(url):
    log.info("Fetching %s -- %s: %s"%(interval_start,interval_end,url))

    req=requests.get(url)
    #data=req.json()
    #ds=asos_dat_to_ds(data)

#%%
# for post July 1998 data, format defined in
#  https://www.ncei.noaa.gov/pub/data/asos-fivemin/td6401b.txt

# Dev
url="https://www.ncei.noaa.gov/pub/data/asos-fivemin/6401-2022/64010KHWD202201.dat"

fn="J:\work2022\SFEI-2022\64010KHWD202201.dat"
