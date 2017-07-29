import datetime

import numpy as np
import xarray as xr
import requests
import logging

log=logging.getLogger('usgs_nwis')

from ... import utils
from .. import rdb

##

def nwis_dataset(station,start_date,end_date,products,
                  days_per_request=None):
    """
    Retrieval script for USGS waterdata.usgs.gov
    
    Retrieve one or more data products from a single station.
    station: string or numeric identifier for COOPS station
    product: string identifying the variable to retrieve.  See all_products at 
    the top of this file.
    start_date,end_date: period to retrieve, as python datetime, matplotlib datenum,
    or numpy datetime64.
    days_per_request: batch the requests to fetch smaller chunks at a time.

    returns an xarray dataset.

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
    base_url="https://nwis.waterdata.usgs.gov/usa/nwis/uv/"
    # ?format=rdb&begin_date=2012-08-01&cb_00060=on&site_no=11337190&end_date=2012-08-15&period=&cb_00010=on

    params['period']=''

    # generator for dicing up the request period
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

        params['begin_date']=utils.to_datetime(interval_start).strftime('%Y-%m-%d')
        params['end_date']  =utils.to_datetime(interval_end).strftime('%Y-%m-%d')

        req=requests.get(base_url,params=params)
        data=req.text

        ds=rdb.rdb_to_dataset(text=data)

        # USGS returns data inclusive of the requested date range - leading to some overlap
        if len(datasets):
            ds=ds.isel(time=ds.time>datasets[-1].time[-1])
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
