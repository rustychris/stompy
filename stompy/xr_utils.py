import xarray as xr
import numpy as np

def gradient(ds,varname,coord):
    # rather than assume that consecutive data points are valid,
    # fit a line to the data values per water column
    daC,daz = xr.broadcast(ds[varname],ds[coord])

    z=daz.values
    C=daC.values
    assert z.shape==C.shape
    
    newdims=[dim for dim in daC.dims if dim!=coord]
    newshape=[len(daC[dim]) for dim in newdims]
    newdims,newshape
    result=np.zeros(newshape,'f8')

    for idx in np.ndindex(*newshape):
        colC=C[idx]
        colz=z[idx]
        assert colC.ndim==colz.ndim==1 # sanity
        valid=np.isfinite(colC*colz)
        if np.sum(valid)>1:
            mb=np.polyfit(colz[valid],colC[valid],1)
            result[idx]=mb[0]
        else:
            result[idx]=np.nan
    return xr.DataArray(result,coords=[ds[dim] for dim in newdims],name='d%s/d%s'%(varname,coord))

def find_var(nc,pred):
    for fld in nc:
        try:
            if pred(nc[fld]):
                return fld
        except:
            pass
    return None
