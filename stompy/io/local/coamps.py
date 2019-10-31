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
                 'cencoos_4km/2017/2017091700/',
                 'cencoos_4km/2018/2018013112/',
                 'cencoos_4km/2018/2018013012/'
]
# specifically in 2018, these are missing:
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013112/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00900F0NL2018013112_0105_000100-000000wnd_utru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013112/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_01000F0NL2018013112_0105_000100-000000wnd_utru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013112/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_01100F0NL2018013112_0105_000100-000000wnd_utru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013112/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00800F0NL2018013112_0105_000100-000000wnd_utru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013112/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00700F0NL2018013112_0105_000100-000000wnd_utru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013012/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00300F0NL2018013012_0105_000100-000000wnd_vtru
# http://www.usgodae.org/pub/outgoing/fnmoc/models/coamps/calif/cencoos/cencoos_4km/2018/2018013012/US058GMET-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_00100F0NL2018013012_0105_000100-000000wnd_utru


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
            if step_back != 0:
                log.warning("Trying to find data for %s, step back is %d"%(str(day_dt),step_back))
                
            run_start=run_start0-datetime.timedelta(hours=step_back)

            # how many hours into this run is the target datetime?
            hour_of_run = int(round((day_dt - run_start).total_seconds() / 3600))

            run_tag="%04d%02d%02d%02d"%(run_start.year,
                                        run_start.month,
                                        run_start.day,
                                        run_start.hour)
            # 2019-10-12: appears to have changed to https, though the certificate
            # is not valid.
            base_url=("https://www.usgodae.org/pub/outgoing/fnmoc/models/"
                      "coamps/calif/cencoos/cencoos_4km/%04d/%s/")%(run_start.year,run_tag)
            
            recs=dict()
            
            for field_name in fields:
                if field_name in ['wnd_utru','wnd_vtru']:
                    cat="MET"
                    elev_code=105 # 0001: surface?  0105: above surface  0100: pressure?
                    elev=100
                elif field_name in ['rltv_hum','air_temp']:
                    cat="MET"
                    # these are available at 20, while winds seems only at 100 and higher
                    elev_code=105
                    elev=20
                elif field_name in ['sol_rad','grnd_sea_temp']:
                    elev_code=1
                    elev=0
                else:
                    # pressure at sea level
                    elev_code=102
                    elev=0
                    
                if field_name in ['grnd_sea_temp']:
                    cat="COM"
                else:
                    cat="MET"

                url_file=("US058G%s-GR1dyn.COAMPS-CENCOOS_CENCOOS-n3-c1_"
                          "%03d"
                          "00F0NL"
                          "%s_%04d_%06d-000000%s")%(cat,hour_of_run,run_tag,elev_code,elev,field_name)

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
    import requests
    files=[]
    
    for recs in coamps_files(start,stop,cache_dir,**kw):
        for field_name in recs:
            rec=recs[field_name]
            output_fn=rec['local']
            if os.path.exists(output_fn):
                files.append(output_fn)
                log.debug("Skip %s"%os.path.basename(output_fn))
                continue

            log.warning("Fetch %s"%os.path.basename(output_fn))

            output_dir=os.path.dirname(output_fn)
            os.path.exists(output_dir) or os.makedirs(output_dir)
            try:
                # 2019-10-12: certificate doesn't match, so don't verify.
                utils.download_url(rec['url'],output_fn,on_abort='remove',verify=False)
                files.append(output_fn)
            except (requests.HTTPError,requests.ConnectionError) as exc:
                # 2019-10-12: getting ConnectionError - not sure if this is
                # a different failure mode, or different version of requests.
                log.error("Failed to download %s, will move on (%s)"%(rec['url'],exc))
            
            time.sleep(1) # be a little nice.  not as nice as it used to be.
    return files

def coamps_press_windxy_dataset(bounds,start,stop,cache_dir):
    return coamps_dataset(bounds,start,stop,cache_dir,
                          fields=['wnd_utru','wnd_vtru','pres_msl'])

def coamps_dataset(bounds,start,stop,cache_dir,
                   fields=['wnd_utru','wnd_vtru','pres_msl'],
                   missing='omit',
                   fetch=True):
    """
    Downloads COAMPS winds for the given period (see fetch_coamps_wind),
    trims to the bounds xxyy, and returns an xarray Dataset.

    fields: these are the coamps names for the fields.  for legacy reasons,
      some of them are remapped in the dataset
      wnd_utru => wind_u
      wnd_vtru => wind_v
      pres_msl => pres

    missing:'omit' - if any of the fields are missing for a timestep, omit that timestep.
        None: will raise an error when GdalGrid fails.
        may in the future also have 'nan' which would fill that variable with nan.

    fetch: if False, rely on cached data.
    """
    fields=list(fields)
    fields.sort()

    if fetch:
        fetch_coamps_wind(start,stop,cache_dir,fields=fields)

    pad=10e3
    crop=[bounds[0]-pad,bounds[1]+pad,
          bounds[2]-pad,bounds[3]+pad]

    dss=[] 

    renames={'wnd_utru':'wind_u',
             'wnd_vtru':'wind_v',
             'pres_msl':'pres'}
    
    for recs in coamps_files(start,stop,cache_dir,fields=fields):
        timestamp=recs[fields[0]]['timestamp']
        timestamp_dt=utils.to_datetime(timestamp)
        timestamp_str=timestamp_dt.strftime('%Y-%m-%d %H:%M')
        # use the local file dirname to get the same model subdirectory
        # i.e. cencoos_4km
        cache_fn=os.path.join(os.path.dirname(recs[fields[0]]['local']),
                              "%s-%s.nc"%(timestamp_dt.strftime('%Y%m%d%H%M'),
                                          "-".join(fields)))
        if not os.path.exists(cache_fn):
            print(timestamp_str)
            ds=xr.Dataset()
            ds['time']=timestamp

            # check to see that all files exists:
            # this will log the missing urls, which can then be added to the missing
            # list above if they are deemed really missing
            missing_urls=[ recs[field_name]['url']
                           for field_name in fields
                           if not os.path.exists(recs[field_name]['local'])]
            if len(missing_urls)>0 and missing=='omit':
                log.warning("Missing files for time %s: %s"%(timestamp_str,", ".join(missing_urls)))
                continue
            
            for i,field_name in enumerate(fields):
                # load the grib file
                raw=field.GdalGrid(recs[field_name]['local'])
                # Reproject to UTM: these come out as 3648m resolution, compared to 4km input.
                # Fine.  366 x 325.  Crops down to 78x95
                raw_utm=raw.warp("EPSG:26910").crop(crop)

                if i==0: # take spatial details from the first field
                    x,y = raw_utm.xy()
                    ds['x']=('x',),x
                    ds['y']=('y',),y

                ds_name=renames.get(field_name,field_name)
                # copy, in hopes that we can free up ram more quickly
                ds[ds_name]=('y','x'), raw_utm.F.copy()

            ds.to_netcdf(cache_fn)
            ds.close()
        ds=xr.open_dataset(cache_fn)
        ds.load() # force load of data
        ds.close() # and close out file handles
        dss.append(ds) # so this is all in ram.

    ds=xr.concat(dss,dim='time')
    return ds

