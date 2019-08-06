"""
Methods related to downloading and processing HYCOM data.
"""
import os
import time
import datetime
import pandas as pd
import numpy as np
import xarray as xr

import logging
from ... import utils
log=logging.getLogger('hycom')

def fetch_range(lon_range, lat_range, time_range, cache_dir):
    """
    lon_range: [lon_min,lon_max]
    lat_range: [lat_min,lat_max]
    time_range: [time_min,time_max]
    returns a list of local netcdf filenames, one per time step.

    Limitations:
     * not ready for time ranges that span multiple HYCOM experiments.
    """
    # deprecated
    #times=pd.DatetimeIndex(start=time_range[0],end=time_range[1],freq='D')
    # modern call
    
    times=pd.date_range(start=utils.floor_dt64(time_range[0],np.timedelta64(24,'h')),
                        end=utils.ceil_dt64(time_range[1],np.timedelta64(24,'h')),
                        freq='D')

    last_ds=None
    lon_slice=None
    lat_slice=None

    lon_range=np.asarray(lon_range)

    filenames=[]

    for idx,t in enumerate(times):
        log.info("t=%s  %d/%d"%(t,idx,len(times)))
        time_str = t.strftime('%Y%m%d%H')
        cache_name=os.path.join( cache_dir,
                                 "%s-%.2f_%.2f_%.2f_%.2f.nc"%(time_str,
                                                              lon_range[0],lon_range[1],
                                                              lat_range[0],lat_range[1]) )
        if not os.path.exists(cache_name):
            log.info("Fetching %s"%cache_name)
            try:
                fetch_one_day(t,cache_name,lon_range,lat_range)
            except HycomException:
                log.warning("HYCOM download failed -- will continue with other days")
                continue
            time.sleep(1)
        filenames.append(cache_name)

    return filenames

class HycomException(Exception):
    pass

def hycom_bathymetry(t,cache_dir):
    """
    Return a dataset for the hycom bathymetry, suitable for the given date.
    currently doesn't do anything with the date, and always returns the most
    recent bathymetry.
    """
    url="ftp://ftp.hycom.org/datasets/GLBb0.08/expt_93.0/topo/depth_GLBb0.08_09m11.nc"
    local_fn=os.path.join(cache_dir,url.split("/")[-1])
    if not os.path.exists(local_fn):
        utils.download_url(url,local_fn,log=log,on_abort='remove')

    ds=xr.open_dataset(local_fn)
    # make it look like the older file
    invalid=1e6<np.where( np.isnan(ds.depth.values),0,ds.depth.values)
    ds.depth.values[invalid]=np.nan
    ds['bathymetry']=ds['depth']
    return ds

def fetch_one_day(t,output_fn,lon_range,lat_range):
    """
    Download physical variables from hycom for the lat/lon ranges,
    and the 24 hours following the given time t.

    Raises HycomException if the download fails, leaving any partial
    or bad download in output_fn+"-FAIL"

    returns None
    """
    t=utils.to_datetime(t)

    # this is going to be a problem, since they switch experiments mid-day.
    # that's unkind.
    if t>=datetime.datetime(2017,2,1,0,0) and t<datetime.datetime(2017,6,1,12,0):
        ncss_base_url="http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.8"
    elif t>=datetime.datetime(2017,6,1,12,0) and t<=datetime.datetime(2017,10,1,9,0):
        ncss_base_url="http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_57.7"
    elif t>=datetime.datetime(2017,10,1,12,0) and t<=datetime.datetime(2018,3,17,9,0):
        # previously had 2018-03-20 09:00 as the end, but looks like it is earlier
        ncss_base_url="http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9"
    elif t>=datetime.datetime(2018,3,17,0,0): 
        # New - testing!
        ncss_base_url="http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_93.0"
    else:
        raise Exception("Not ready for other experiments")

    variables=["salinity_bottom","surf_el","water_temp_bottom",
               "water_u_bottom","water_v_bottom","salinity",
               "water_temp","water_u","water_v"]
    var_args=[ ('var',v) for v in variables ]
    loc_args=[ ('north',"%.3f"%lat_range[1]),
               ('south',"%.3f"%lat_range[0]),
               ('west',"%.3f"%lon_range[0]),
               ('east',"%.3f"%lon_range[1]) ]
    time_fmt="%Y-%m-%dT%H:%M:%SZ"
    time_args=[ ('time_start', t.strftime(time_fmt)),
                ('time_end', (t+datetime.timedelta(days=1)).strftime(time_fmt)),
                ('timeStride', "1") ]
    etc_args=[ ('disableProjSubset',"on"),
               ('horizStride',"1"),
               ('vertCoord',''),
               ('LatLon',"true"),
               ('accept',"netcdf4")]

    params=var_args+loc_args+time_args+etc_args
    valid=True

    # DBG:
    logging.info("About to download from hycom:")
    logging.info(ncss_base_url)
    logging.info(params)
    t=time.time()
    try:
        utils.download_url(ncss_base_url,local_file=output_fn,
                           log=logging,params=params,timeout=1800)
    except KeyboardInterrupt:
        raise
    except:
        logging.error("fetching hycom:", exc_info=True)
        valid=False

    elapsed=time.time()-t
    logging.info("download_url: elapsed time %.1fs"%elapsed)

    if valid:
        with open(output_fn,'rb') as fp:
            head=fp.read(1000)
            if ( (b'500 Internal Server Error' in head)
                 or (b'No such file' in head) ):
                logging.error("HYCOM download failed with server error")
                valid=False
    if not valid:
        logging.error("renaming failed hycom download")
        if os.path.exists(output_fn):
            os.rename(output_fn,output_fn+"-FAIL")
        else:
            with open(output_fn+"-FAIL",'wt') as fp:
                fp.write("Failed to download.  see stompy/io/local/hycom.py")
        raise HycomException("HYCOM download failed")

    
