"""
Thin wrapper to return xarray Dataset for an ADCP, after parsing
using the regular rdradcp code.
"""
import xarray as xr
import numpy as np
from .. import utils
from . import rdradcp

def adcp_to_dataset(adcp):
    ds=xr.Dataset()
    ds.attrs['adcp']=adcp
    
    # convert to xarray
    for field in adcp.bin_data.dtype.names:
        dims=('time','bin')
        
        if adcp.bin_data[field].ndim==3:
            dims=dims+('beam',)
            
        ds[field]=dims, adcp.bin_data[field]

    for field in adcp.ensemble_data.dtype.names:
        dims=('time',)
        if adcp.ensemble_data[field].ndim==2:
            dims=dims+('beam',)
        ds[field]=dims,adcp.ensemble_data[field]

    ds['config']=(),adcp.config
    ds['name']=(),adcp.name

    # ds['bin_dist']=adcp.config.bin1_dist + adcp.config.cell_size*np.arange(ds.dims['bin'])
    ds['range']=('bin',),adcp.config.ranges

    ds['time']=('time',), utils.to_dt64(ds.mtime.values)

    # Copy config to a dummy variable in addition to adcp.config
    ds['config']=(),0
    cfg=ds.attrs['adcp'].config
    for k in cfg.__dict__:
        ds.config.attrs[k]=getattr(cfg,k)
    
    return ds

def rdradcp_xr(*a,**kw):
    """
    Read raw T-RDI ADCP data, return as an xarray Dataset
    """
    adcp=rdradcp.rdradcp(*a,**kw)
    ds=adcp_to_dataset(adcp)
    ds.attrs['src']=kw.get('name',None) or a[0]
    return ds

