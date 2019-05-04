"""
Methods related to downloading and processing HYCOM data.
"""
import os
import re
import time
from stompy.spatial import field
import datetime
import numpy as np
import xarray as xr
import logging
from ... import utils
log=logging.getLogger('coamps')

missing_files= [ 'cencoos_4km/2017/2017091612/',
                 'cencoos_4km/2017/2017091700/']

def known_missing(recs):
    """
    There are some skipped dates in the COAMPS data online.  This checks
    a download URL against a list of known missing files, returning true
    if the URL is expected to be missing.
    """
    # Get a representative URL
    url=list(recs.values())[0]['url']
    for patt in missing_files:
        if re.search(patt,url):
            return True
    return False

def coamps_files(start,stop,cache_dir,fields=['wnd_utru','wnd_vtru','pres_msl']):
    """ 
    Generate urls, filenames, and dates for
    fetching or reading COAMPS data

    Tries to pull the first 12 hours of runs, but if a run is known to be missing,
    will pull later hours of an older run.
    
    returns a generator, which yields for each time step of coamps output
    {'wnd_utru':{'url=..., local=..., timestamp=...}, ...}

    """
    dataset_name="cencoos_4km"
    # round to days
    start=start.astype('M8[D]')
    stop=stop.astype('M8[D]') + np.timedelta64(1,'D')

    # The timestamps we're trying for
    target_hours=np.arange(start,stop,np.timedelta64(1,'h'))
    
    for hour in target_hours:
        day_dt=utils.to_datetime(hour)

        # Start time of the ideal run:
        run_start0=day_dt - datetime.timedelta(hours=day_dt.hour%12)

        # runs go for 48 hours, so we have a few chances to get the
        # same output timestamp
        for step_back in [0,12,24,36]:
            run_start=run_start0-datetime.timedelta(hours=step_back)

            # how many hours into this run is the target datetime?
            hour_of_run = int(round((day_dt - run_start).total_seconds() / 3600))

            run_tag="%04d%02d%02d%02d"%(run_start.year,
                                        run_start.month,
                                        run_start.day,
                                        run_start.hour)
            base_url=("http://www.usgodae.org/pub/outgoing/fnmoc/models/"
                      "coamps/calif/cencoos/cencoos_4km/%04d/%s/")%(run_start.year,run_tag)
            
            recs=dict()

            for field_name in fields:
                if field_name in ['wnd_utru','wnd_vtru']:
                    elev_code=105 # 0001: surface?  0105: above surface  0100: pressure?
                    elev=100
                elif field_name in ['rltv_hum','air_temp']:
                    # these are available at 20, while winds seems only at 100 and higher
                    elev_code=105
                    elev=20
                elif field_name in ['sol_rad']:
                    elev_code=1
                    elev=0
                else:
                    # pressure at sea level
                    elev_code=102
                    elev=0
                url_file=("US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_"
                          "%03d"
                          "00F0NL"
                          "%s_%04d_%06d-000000%s")%(hour_of_run,run_tag,elev_code,elev,field_name)

                output_fn=os.path.join(cache_dir, dataset_name, url_file)
                recs[field_name]=dict(url=base_url+url_file,
                                      local=output_fn,
                                      timestamp=hour)
            if known_missing(recs):
                continue
            yield recs
            break
        else:
            raise Exception("Couldn't find a run for date %s"%day_dt.strftime('%Y-%m-%d %H:%M'))


def fetch_coamps_wind(start,stop, cache_dir, **kw):
    """
    start,stop: datetime64
    cache_dir: pre-existing cache directory
    kw: can pass fields=['wnd_utru',...] on through to coamps_files()
    returns a list of local netcdf filenames, one per time step.


    Download all COAMPS outputs between the given np.datetime64()s.
    Does not do any checking against available data, so requesting data
    before or after what is available will fail.
    """
    files=[]
    
    for recs in coamps_files(start,stop,cache_dir,**kw):
        for field_name in recs:
            rec=recs[field_name]
            output_fn=rec['local']
            files.append(output_fn)
            if os.path.exists(output_fn):
                log.debug("Skip %s"%os.path.basename(output_fn))
                continue

            log.info("Fetch %s"%os.path.basename(output_fn))

            output_dir=os.path.dirname(output_fn)
            os.path.exists(output_dir) or os.makedirs(output_dir)
            utils.download_url(rec['url'],output_fn,on_abort='remove')
            
            time.sleep(2) # be a little nice.
    return files


def coamps_press_windxy_dataset(bounds,start,stop,cache_dir):
    """
    Downloads COAMPS winds for the given period (see fetch_coamps_wind),
    trims to the bounds xxyy, and returns an xarray Dataset.
    """
    fetch_coamps_wind(start,stop,cache_dir)

    pad=10e3
    crop=[bounds[0]-pad,bounds[1]+pad,
          bounds[2]-pad,bounds[3]+pad]

    dss=[] 

    for recs in coamps_files(start,stop,cache_dir):
        timestamp=recs['wnd_utru']['timestamp']
        timestamp_dt=utils.to_datetime(timestamp)
        timestamp_str=timestamp_dt.strftime('%Y-%m-%d %H:%M')
        # use the local file dirname to get the same model subdirectory
        # i.e. cencoos_4km
        cache_fn=os.path.join(os.path.dirname(recs['pres_msl']['local']),
                              "%s.nc"%timestamp_dt.strftime('%Y%m%d%H%M'))
        if not os.path.exists(cache_fn):
            print(timestamp_str)

            # load the 3 fields:
            wnd_utru=field.GdalGrid(recs['wnd_utru']['local'])
            wnd_vtru=field.GdalGrid(recs['wnd_vtru']['local'])
            pres_msl=field.GdalGrid(recs['pres_msl']['local'])

            # Reproject to UTM: these come out as 3648m resolution, compared to 4km input.
            # Fine.  366 x 325.  Crops down to 78x95
            wnd_utru_utm=wnd_utru.warp("EPSG:26910").crop(crop)
            wnd_vtru_utm=wnd_vtru.warp("EPSG:26910").crop(crop)
            pres_msl_utm=pres_msl.warp("EPSG:26910").crop(crop)

            ds=xr.Dataset()
            ds['time']=timestamp
            x,y = wnd_utru_utm.xy()
            ds['x']=('x',),x
            ds['y']=('y',),y
            # copy, in hopes that we can free up ram more quickly
            ds['wind_u']=('y','x'), wnd_utru_utm.F.copy()
            ds['wind_v']=('y','x'), wnd_vtru_utm.F.copy()
            ds['pres']=('y','x'), pres_msl_utm.F.copy()

            ds.to_netcdf(cache_fn)
            ds.close()
        ds=xr.open_dataset(cache_fn)
        ds.load() # force load of data
        ds.close() # and close out file handles
        dss.append(ds) # so this is all in ram.

    ds=xr.concat(dss,dim='time')
    return ds

