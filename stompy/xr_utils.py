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

def redimension(ds,new_dims,
                intragroup_dim=None,
                inplace=False,
                save_mapping=False):
    """ 
    copy ds, making new_dims into the defining 
    dimensions for variables which share shape with new_dims.
    
    each entry in new_dims must have the same dimension, and
    must be unidimensional
    
    Example:
    Dataset:
      coordinates
        sample  [0,1,2..100]
      data variables
        date(sample)      [...]
        station(sample)   [...]
        depth(sample)     [...]
        salinity(sample)  [...]
    
    We'd like to end up with 
      salinity(date,station,profile_sample)
      depth(date,station,profile_sample)

    Or
    Dataset:
      coordinates
        time [...]
        item [...]
      data variables
        x(item)        [...]
        y(item)        [...]
        z(item)        [...]
        salinity(time,time) [...,...]

    Which you want to become

    Dataset:
      coordinates
        time [.]
        x [.]
        y [.]
        zi [.]
      data variables
        z(x,y,zi)             [...]
        salinity(time,x,y,zi) [....]

    In other words, replace item with three orthogonal dimensions.  Two of the
    orthogonal dimensions have existing coordinates, and the third is an index
    to elements within the bin defined by x,y.

    save_mapping: create an additional variable in the output which stores the
    mapping of the linear dimension to the new, orthogonal dimensions
    """
    if not inplace:
        ds=ds.copy()

    lin_dim=ds[new_dims[0]].dims[0]# the original linear dimension
    orig_dims=[ ds[vname].values.copy()
                for vname in new_dims ]
    Norig=len(orig_dims[0]) # length of the original, linear dimension

    uni_new_dims=[ np.unique(od) for od in orig_dims]

    # note that this is just the shape that will replace occurences of lin_dim
    new_shape=[len(und) for und in uni_new_dims]

    # build up an index array 
    new_idxs=[ np.searchsorted(und,od)
               for und,od in zip( uni_new_dims, orig_dims ) ]

    if intragroup_dim is not None:
        # here we need to first count up the max number within each 'bin'
        # so new_idxs
        count_per_group=np.zeros(new_shape,'i4')
        intra_idx=np.zeros(Norig,'i4')
        for orig_idx,idxs in enumerate(zip(*new_idxs)):
            intra_idx[orig_idx] = count_per_group[idxs] 
            count_per_group[ idxs ]+=1
        n_intragroup=count_per_group.max() # 55 in the test case

        # add in the new dimension
        new_shape.append(n_intragroup)
        new_idxs.append(intra_idx)

    # negative means missing.  at this point, intragroup_dim has not been taken care of
    mapper=np.zeros(new_shape,'i4') - 1
    mapper[ tuple(new_idxs) ] = np.arange(Norig)

    # install the new coordinates - first the grouped coordinates
    for nd,und in zip(new_dims,uni_new_dims):
        del ds[nd] # doesn't like replacing these in one go
        ds[nd]= ( (nd,), und )
    if intragroup_dim is not None:
        # and second the new intragroup coordinate:
        new_dims.append(intragroup_dim)
        ds[intragroup_dim] = ( (intragroup_dim,), np.arange(n_intragroup) )

    for vname in ds.data_vars:
        if lin_dim not in ds[vname].dims:
            print("Skipping %s"%vname)
            continue
        print(vname)

        var_new_dims=[]
        var_new_slice=[]
        mask_slice=[]
        for d in ds[vname].dims:
            if d==lin_dim:
                var_new_dims += new_dims
                var_new_slice.append( mapper )
                mask_slice.append( mapper<0 )
            else:
                var_new_dims.append(d)
                var_new_slice.append(slice(None))
                mask_slice.append(slice(None))
        var_new_dims=tuple(var_new_dims)
        var_new_slice=tuple(var_new_slice)

        # this is time x nSegment
        # ds[vname].values.shape # 10080,1494

        # This is the beast:
        new_vals=ds[vname].values[var_new_slice]
        mask=np.zeros_like(new_vals,'b1')
        mask[mask_slice] = True

        new_vals=np.ma.array(new_vals,mask=mask)

        # assumes a float-valued variable! maybe better to use masked array?
        ds[vname]=( var_new_dims, new_vals )

    if save_mapping:
        ds['mapping']= ( new_dims, mapper)

    return ds


def sort_dimension(ds,sort_var,sort_dim,inplace=False):
    """
    sort_var: variable whose value will be used to sort items along sort_dim.
    sort_dim must be in sort_var.dims
    only variables with dimensions the same or a superset of sort_var.dims
    can/will be sorted.
    """
    if not inplace:
        ds=ds.copy()
        
    #if ds[sort_var].ndim>1:
    # the interesting case
    # want to sort within each 'bin'
    
    sort_var_dims=ds[sort_var].dims
    sort_var_dimi = sort_var_dims.index(sort_dim)
    new_order=ds[sort_var].argsort(axis=sort_var_dimi).values

    # this only works for variables with a superset of sort_var's
    # dimensions (or the same).
    # i.e. depth(date,station,prof_sample)
    # can't be used to sort a variable of (station,prof_sample)
    # but it can be used to sort a variable of (analyte,date,station,prof_sample)
    for v in ds.data_vars:
        for d in sort_var_dims:
            compat=True
            if d not in ds[v].dims:
                print("%s not compatible with dimensions for sorting"%v)
                compat=False
        if not compat: continue

        # build up transpose
        trans_dims=[]
        for d in ds[v].dims:
            if d not in sort_var_dims:
                trans_dims.append(d)
        n_extra=len(trans_dims)
        trans_dims+=sort_var_dims

        orig_dims=ds[v].dims
        tmp_trans=ds[v].transpose(*trans_dims)

        vals=tmp_trans.values
        # actually a tricky type of indexing to handle:
        # new_order.shape: (23, 37, 52)
        # luckily numpy knows how to do this relatively efficiently:
        idxs=np.ix_( *[np.arange(N) for N in vals.shape] )
        idxs=list(idxs)
        idxs[n_extra+sort_var_dimi]=new_order
        tmp_trans.values=vals[tuple(idxs)]
        ds[v].values=tmp_trans.transpose(*orig_dims)
    return ds
