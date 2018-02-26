import datetime
import os
import logging

import numpy as np
import xarray as xr
import pandas as pd
import requests

log=logging.getLogger('usgs_nwis')

from ... import utils
from .. import rdb
from .common import periods

try:
    import seawater
except ImportError:
    seawater=None


def nwis_dataset_collection(stations,*a,**k):
    """
    Fetch from multiple stations, glue together to a combined dataset.
    The rest of the options are the same as for nwis_dataset()
    """
    ds_per_site=[]
    for station in stations:
        ds=nwis_dataset(station,*a,**k)
        ds['site']=('site',),[station]
        ds_per_site.append(ds)

    # And now glue those all together, but no filling of gaps yet.
    # would need to add 'site' as a dimension on data variables for this to work
    # dataset=ds_per_gage[0]
    #for other in ds_per_gage[1:]:
    #    dataset=dataset.combine_first(other)

    # As cases of missing data come up, this will have to get smarter about padding
    # individual sites.
    return xr.concat( ds_per_site, dim='site')
        
def nwis_dataset(station,start_date,end_date,products,
                 days_per_request=None,frequency='realtime',
                 cache_dir=None):
    """
    Retrieval script for USGS waterdata.usgs.gov
    
    Retrieve one or more data products from a single station.
    station: string or numeric identifier for COOPS station.

    product: string identifying the variable to retrieve.  See all_products at 
    the top of this file.
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

    returns an xarray dataset.

    frequency: defaults to "realtime" which should correspond to the original 
      sample frequency.  Alternatively, "daily" which access daily average values.

    Note that names of variables are inferred from parameter codes where possible,
    but this is not 100% accurate with respect to the descriptions provided in the rdb,
    notably "Discharge, cubic feet per second" may be reported as 
    "stream_flow_mean_daily"
    """
    params=dict(site_no=station,
                format='rdb')

    for prod in products:
        params['cb_%05d'%prod]='on'

    # Only for small requests of recent data:
    # base_url="https://waterdata.usgs.gov/nwis/uv"
    # Otherwise it redirects to here:
    if frequency=='realtime':
        base_url="https://nwis.waterdata.usgs.gov/usa/nwis/uv/"
    elif frequency=='daily':
        base_url="https://waterdata.usgs.gov/nwis/dv"
    else:
        raise Exception("Unknown frequency: %s"%(frequency))
    
    params['period']=''

    # generator for dicing up the request period

    datasets=[]

    last_url=None
    
    for interval_start,interval_end in periods(start_date,end_date,days_per_request):

        params['begin_date']=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
        params['end_date']  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

        if cache_dir is not None:
            cache_fn=os.path.join(cache_dir,
                                  "%s_%s_%s_%s.nc"%(station,
                                                    "-".join(["%d"%p for p in products]),
                                                    params['begin_date'],
                                                    params['end_date']))
        else:
            cache_fn=None
            
        if (cache_fn is not None) and os.path.exists(cache_fn):
            log.info("Cached   %s -- %s"%(interval_start,interval_end))
            ds=xr.open_dataset(cache_fn)
        else:
            log.info("Fetching %s -- %s"%(interval_start,interval_end))
            req=requests.get(base_url,params=params)
            data=req.text
            ds=rdb.rdb_to_dataset(text=data)
            if ds is None: # There was no data there
                log.warning("    no data found for this period")
                continue
            ds.attrs['url']=req.url

            if cache_fn is not None:
                ds.to_netcdf(cache_fn)
                
        # USGS returns data inclusive of the requested date range - leading to some overlap
        if len(datasets):
            ds=ds.isel(time=ds.time>datasets[-1].time[-1])
        datasets.append(ds)

    if len(datasets)==0:
        # could try to construct zero-length dataset, but that sounds like a pain
        # at the moment.
        return None 

    if len(datasets)>1:
        # it's possible that not all variables appear in all datasets
        # dataset=xr.concat( datasets, dim='time')
        dataset=datasets[0]
        for other in datasets[1:]:
            dataset=dataset.combine_first(other)
    else:
        dataset=datasets[0]
    return dataset


def add_salinity(ds):
    assert seawater is not None
    for v in ds.data_vars:
        if v.startswith('specific_conductance'):
            salt_name=v.replace('specific_conductance','salinity')
            if salt_name not in ds:
                print("%s => %s"%(v,salt_name))
                salt=seawater.eos80.salt(ds[v].values/1000. / seawater.constants.c3515,
                                         25.0, # temperature - USGS adjusts to 25degC
                                         0) # no pressure effects
                ds[salt_name]=ds[v].dims, salt

