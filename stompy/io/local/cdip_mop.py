"""
Simple caching layer on THREDDS access to CDIP MOP
wave data. Useful for US West Coast wave climate data,
hourly resolution of many wave parameters.
"""

import os
import numpy as np
from .common import periods
import logging
log=logging.getLogger('cdip_mop')

import xarray as xr

def hindcast_dataset(station,start_date,end_date,cache_dir=None,
                     variables=['waveHs'],clip='True',
                     days_per_request='M'):
    """
    station: string, such as 'SM141' (which Pescadero State Beach, San Mateo county)
    start_date,end_date: np.datetime64
    clip: 
      False: keep duration of cached blocks
      True: clip python-style, exclusive of end date
      'inclusive': clip, but include end date

    variables: which variables to fetch. pass None for all variables.
    """
    url="http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/%s_hindcast.nc"%station

    ds=None
    t=None

    chunks=[]
    
    for interval_start,interval_end in periods(start_date,end_date,days_per_request):
        base_fn="%s_%s_%s_%s.nc"%(station,
                                  "-".join(["%s"%p for p in variables]),
                                  interval_start.strftime('%Y%m%d'),interval_end.strftime('%Y%m%d'))

        if cache_dir is not None:
            cache_fn=os.path.join(cache_dir,base_fn)
        else:
            cache_fn=None

        if (cache_fn is not None) and os.path.exists(cache_fn):
            chunk_ds=xr.open_dataset(cache_fn)
            chunk_ds.load()
            chunks.append(chunk_ds.copy())
            chunk_ds.close()
        else:
            # Fetch it
            if ds is None: # open on demand
                log.info("Fetching cdip/mop metadta")
                ds=xr.open_dataset(url)
                t=ds.waveTime.values

            cmip_start,cmip_stop=np.searchsorted(t,
                                                 [np.datetime64(interval_start),
                                                  np.datetime64(interval_end)   ])

            if variables is not None:
                ds_sel=ds
            else:
                ds_sel=ds[variables]
                
            ds_sel=ds_sel[variables].isel(waveTime=slice(cmip_start,cmip_stop+1))
            ds_sel.load()
            chunks.append(ds_sel)
            if cache_fn is not None:
                ds_sel.to_netcdf(cache_fn)
    if len(chunks)==0:
        return None
    elif len(chunks)==1:
        ds=chunks[0]
    else:
        # eliminate any time overlaps:
        clipped=[chunks[0]]
        for chunk in chunks[1:]:
            sel=chunk.waveTime.values>clipped[-1].waveTime.values[-1]
            clipped.append( chunk.isel(waveTime=sel) )
        ds=xr.concat(clipped,dim='waveTime')

    if not clip:
        pass
    else:
        if clip=='inclusive':
            sel=(ds.waveTime.values>=start_date)&(ds.waveTime.values<=end_date)
        else:
            sel=(ds.waveTime.values>=start_date)&(ds.waveTime.values<end_date)
        ds=ds.isel(waveTime=sel)

    # Consistent with other modules in stompy
    ds=ds.rename({'waveTime':'time'})
    # But copy just in case...
    ds['waveTime']=('time',),ds.time.values
    return ds
        
            

