"""
Methods related to downloading and processing HYCOM data.
"""
import os
import time
import datetime
import pandas as pd
import numpy as np
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
    times=pd.date_range(start=time_range[0],end=time_range[1],freq='D')

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
                filenames.append(cache_name)
            except HycomException:
                log.warning("HYCOM download failed -- will continue with other days")
            time.sleep(1)
        else:
            filenames.append(cache_name)

    return filenames

class HycomException(Exception):
    pass

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
    elif t>=datetime.datetime(2017,10,1,12,0) and t<=datetime.datetime(2018,3,20,9,0):
        ncss_base_url="http://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_92.9"
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

    
