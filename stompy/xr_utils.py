import logging
log=logging.getLogger('xr_utils')

from collections import OrderedDict

import xarray as xr
import six
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

    intragroup_dim: introduce an additional dimension to enumerate the original
    data which map to the same new coordinates.
    """
    if not inplace:
        ds=ds.copy()

    lin_dim=ds[new_dims[0]].dims[0]# the original linear dimension
    orig_dims=[ ds[vname].values.copy()
                for vname in new_dims ]
    Norig=len(orig_dims[0]) # length of the original, linear dimension

    uni_new_dims=[ np.unique(od) for od in orig_dims]

    for und in uni_new_dims:
        try:
            if np.any(und<0):
                log.warning("New dimensions have negative values -- will continue but you probably want to drop those first")
        except TypeError:
            # some versions of numpy/xarray will not compare times to 0,
            # triggering a TypeError
            pass

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
    # mapper: array of the shape of the new dimensions, with each entry giving the linear
    # index into the original dimension
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
            # print("Skipping %s"%vname)
            continue
        # print(vname)

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

        # This is the beast: but now it's including some crap values at the beginning
        new_vals=ds[vname].values[var_new_slice]
        mask=np.zeros_like(new_vals,'b1')
        mask[tuple(mask_slice)] = True

        new_vals=np.ma.array(new_vals,mask=mask)

        old_attrs=OrderedDict(ds[vname].attrs)
        # This seems to be dropping the attributes
        ds[vname]=( var_new_dims, new_vals )
        for k in old_attrs:
            if k != '_FillValue':
                ds[vname].attrs[k] = old_attrs[k]

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
                # print("%s not compatible with dimensions for sorting"%v)
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


def first_finite(da,dim):
    # yecch.
    valid=np.isfinite(da.values)
    dimi=da.get_axis_num('prof_sample') 
    first_valid=np.argmax( valid, axis=dimi)
    new_shape=[ slice(length)
                for d,length in enumerate(da.shape)
                if d!=dimi ]
    indexers=np.ogrid[ tuple(new_shape) ]
    indexers[dimi:dimi]=[first_valid]

    da_reduced=da.isel(**{dim:0,'drop':True})
    da_reduced.values=da.values[tuple(indexers)]
    return da_reduced


# Helper for z_from_sigma
def decode_sigma(ds,sigma_v):
    """
    ds: Dataset
    sigma_v: sigma coordinate variable.
    return DataArray of z coordinate implied by sigma_v
    """
    import re
    formula_terms=sigma_v.attrs['formula_terms']
    terms={}
    for hit in re.findall(r'\s*(\w+)\s*:\s*(\w+)', formula_terms):
        terms[hit[0]]=ds[hit[1]]

    # this is where xarray really shines -- it will promote z to the
    # correct dimensions automatically, by name
    # This ordering of the multiplication puts laydim last, which is
    # assumed in some other [fragile] code.
    # a little shady, but its helpful to make the ordering here intentional
    # For temporary compatibility with multi-ugrid, force compute on all
    eta=terms['eta'].compute()
    bedlevel=terms['bedlevel'].compute()
    sigma=terms['sigma'].compute()
    
    z=(eta-bedlevel)*sigma + bedlevel

    return z


def z_from_sigma(dataset,variable,interfaces=False,dz=False):
    """
    Create a z coordinate for variable as a Dataset from the given dataset

    interfaces: False => do nothing related to layer boundaries
            variable name => use the given variable to define interfaces between layers.
            True => try to infer the variable, fallback to even spacing otherwise.
     if interfaces is anything other than False, then the return value will be a Dataset
     with the centers in a 'z_ctr' variable and the interfaces in a 'z_int'

    dz: implies interfaces, and includes a z_dz variable giving thickness of each layer.
    """
    da=dataset[variable]
    da_dims=da.dims

    if dz:
        assert interfaces is not False,"Currently must enable interfaces to get thickness dz"

    # Variables which are definitely sigma, and might be the one we're looking for
    sigma_vars=[v for v in dataset.variables
                if dataset[v].attrs.get('standard_name',None) == 'ocean_sigma_coordinate']

    # xr data arrays
    sigma_ctr_v=None # sigma coordinate for centers
    sigma_int_v=None # sigma coordinate for interfaces

    for v in sigma_vars:
        if set(dataset[v].dims)<=set(da_dims):
            assert sigma_ctr_v is None,"Multiple matches for layer center sigma coordinate"
            sigma_ctr_v=dataset[v]
    assert sigma_ctr_v is not None,"Failed to find a layer-center sigma coordinate"

    # With the layer center variable known, can search for layer interfaces
    if interfaces is False:
        pass
    else:
        if interfaces is True:
            maybe_int_vars=sigma_vars
        else:
            # Even when its specified, check to see that it has the expected form
            maybe_int_vars=[interfaces]

        for v in maybe_int_vars:
            ctr_dims=set(sigma_ctr_v.dims)
            int_dims=set(dataset[v].dims)

            ctr_only = list(ctr_dims - int_dims)
            int_only = list(int_dims - ctr_dims)

            if (len(ctr_only)!=1) or (len(int_only)!=1):
                continue
            if len(dataset[ctr_only[0]])+1==len(dataset[int_only[0]]):
                assert sigma_int_v is None,"Multiple matches for layer interface sigma coordinate"
                sigma_int_v=dataset[v]

    z_ctr=decode_sigma(dataset,sigma_ctr_v)
    if sigma_int_v is not None:
        z_int=decode_sigma(dataset,sigma_int_v)

    result=xr.Dataset()
    result['z_ctr']=z_ctr

    if interfaces is not False:
        result['z_int']=z_int
    if dz is not False:
        dz=xr.ones_like( z_ctr )
        dz.values[...]=np.diff( z_int, axis=z_int.get_axis_num(int_only[0]))
        result['z_dz']=dz

    return result


def bundle_components(ds,new_var,comp_vars,frame,comp_names=None):
    """
    ds: Dataset
    new_var: name of the vector-valued variable to create
    comp_vars: list of variables, one-per component
    frame: name to give the component dimension, i.e. the name of the
     reference frame
    comp_names: list same length as comp_vars, used to name the components.
    """
    vector=xr.concat([ds[v] for v in comp_vars],dim=frame)
    # That puts xy as the first dimension, but I'd rather it last
    dims=vector.dims
    roll_dims=dims[1:] + dims[:1]
    ds[new_var]=vector.transpose( *roll_dims )
    if comp_names is not None:
        ds[frame]=(frame,),comp_names



def concat_permissive(srcs,**kw):
    """
    Small wrapper around xr.concat which fills in nan 
    coordinates where they are missing, in case some
    of the incoming datasets have more metadata than others. 
    """
    extra_coords=set()
    for src in srcs:
        extra_coords |= set(src.coords)

    expanded_srcs=[]

    for src in srcs:
        for extra in extra_coords:
            if extra not in src:
                src=src.assign_coords(**{extra:np.nan})
        expanded_srcs.append(src)

    return xr.concat(expanded_srcs,**kw)

def structure_to_dataset(arr,dim,extra={}):
    """
    Convert a numpy structure array to a dataset.
    arr: structure array.
    dim: name of the array dimension.  can be a tuple with multiple dimension
      names if arr.ndim>1.
    extra: dict optionally mapping specific fields to additional dimensions
     within that field.
    """
    if isinstance(dim,six.string_types):
        dim=(dim,)
    ds=xr.Dataset()
    for fld in arr.dtype.names:
        if fld in extra:
            extra_dims=extra[fld]
        else:
            extra_dims=['d%02d'%d for d in arr[fld].shape[1:]]
        ds[fld]=dim+tuple(extra_dims),arr[fld]
    return ds

def decode_geometry(ds,field,replace=True, on_error='pass'):
    from shapely import geometry
    node_counts=ds[ds[field].attrs['node_count']]
    coordx,coordy=ds[field].attrs['node_coordinates'].split()
    try:
        node_x=ds[coordx]
        node_y=ds[coordy]
    except KeyError:
        print("node_coordinates %s %s do not exist"%(coordx,coordy))
        if on_error=='pass':
            return
        else:
            raise
            
    node_stops=np.cumsum(node_counts)
    node_starts=node_stops-node_counts

    # For simplicity assume a 1D array of geometries
    geoms=np.empty(node_counts.shape,dtype=object)
    geom_type=ds[field].attrs['geometry_type']

    warned=False
    
    for idx in np.ndindex(*node_counts.shape):
        slc=slice(node_starts.values[idx],node_stops.values[idx])
        if geom_type=='line':
            pnts=np.c_[ node_x.values[slc], node_y.values[slc]]
            if pnts.shape[0]>1:
                geom=geometry.LineString(pnts)
            else:
                if not warned:
                    print("Some lines are degenerate")
                    warned=True
                geom=None
        else:
            raise Exception("Unhandled geometry type %s"%geom_type)
        geoms[idx]=geom

    if replace:
        old_attrs=dict(ds[field].attrs)
        ds[field]=node_counts.dims, geoms
        ds[field].attrs.update(old_attrs)
    return geoms

def xr_open_with_rename(fn,renames,**kw):
    """
    Work around netcdf files with variables like z(time,z,lat,lon).
    These are valid netcdf but break xarray.
    """
    import netCDF4
    ds=xr.open_dataset(fn,drop_variables=renames.keys(),**kw)
    ds_nc = netCDF4.Dataset(fn)
    for v in renames:
        nc_var=ds_nc[v]
        ds[renames[v]]=nc_var.dimensions,nc_var[...]
    ds_nc.close()
    ds=ds.set_coords(renames.values())    
    return ds
