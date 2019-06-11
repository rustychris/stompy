"""
WORK IN PROGRESS
 -- does not do anything useful at this point
"""

import six
import xarray as xr

from xr.core.pycompat import (
    PY3, OrderedDict, basestring, iteritems, suppress)

class DFMMergeMap(xr.backends.AbstractDataStore):
    def __init__(self,file_pattern):
        """
        Open a set of DFM map files spread across subdomains,
        merge them on the fly.
        """
        self.file_pattern=file_pattern
        self.update_file_list()
        self.scan_datasets()
    def update_file_list(self):
        self.file_list=glob.glob(self.file_pattern)
        self._dss={} # index into self.file_list => xr.Dataset()
    def dss(self,i):
        if i not in self._dss:
            self._dss[i]=xr.open_dataset(self.file_list[i])
        return self._dss[i]
    def each_ds(self):
        for i in range(len(self.file_list)):
            yield self.dss(i)

    def scan_datasets(self):
        dims0=self.dss(0).dims
        max_elem_global=max([ ds.FlowElemGlobalNr.values.max() for ds in self.each_ds()])
        min_elem_global=min([ ds.FlowElemGlobalNr.values.min() for ds in self.each_ds()])

        my_dim_list=[]

        for d in dims0:
            if d!='nFlowElem':
                size=dims0[d]
            else:
                size=max_elem_global
            my_dim_list.append( (d,size) )
        self._dimensions=xr.core.utils.FrozenOrderedDict(my_dim_list)

    def get_variables(self):
        dsvars=[]
        for k,v in iteritems(self.dss(0)):
            # Here NetCDF4Store calls self.open_store_variable(k, v)
            # on the Variable objects, which seems to just unpack
            # and repack the data.  We don't need to be as nit-picky
            # about the details, but I think this is where we'll
            # be replacing some of the data.

            if 'nFlowElem' in v.dimensions:
                as_var=self.merge_var(k,v)
            else:
                as_var=xr.as_variable(v)
            dsvars.append( (k,as_var) )

        return xr.core.utils.FrozenOrderedDict(dsvars)

    def merge_var(self,name,var0):
        v_dims=var0.dimensions

        # HERE - how do we use this thing:
        # data=indexing.LazilyOuterIndexedArray(NetCDF4ArrayWrapper(... ))
        # Hopefully can get away with only reimplementating NetCDF4ArrayWrapper,
        # Having it call back into this datastore to piece things together.
        v_data=v.data #
        v_attributes = OrderedDict((k, v.getncattr(k))
                                   for k in v.ncattrs())

        return xr.Variable(v_dims, v_data, v_attributes)


    def load(self):
        ds0 = self.dss(0)
        # variables: Frozen(OrderedDict( xarray.Variable objects))
        # atributes: Frozen(OrderedDict( (key,value) tuples)

        return ( xr.core.utils.FrozenOrderedDict(ds0.variables),
                 xr.core.utils.FrozenOrderedDict(ds0.attrs) )

    def get_dimensions(self):
        return self._dimensions

    @classmethod
    def open(cls,filename_or_obj):
        assert isinstance(filename_or_obj,six.string_types)
        return cls(file_pattern=filename_or_obj)

def open_merged_dataset(file_pattern,
                        mask_and_scale=True, decode_times=True,
                        concat_characters=True, decode_coords=True, drop_variables=None):
    store=DFMMergeMap.open(file_pattern)

    ds = xr.conventions.decode_cf(
        store, mask_and_scale=mask_and_scale, decode_times=decode_times,
        concat_characters=concat_characters, decode_coords=decode_coords,
        drop_variables=drop_variables)

    return ds

nc0_fn='runs/hor_002/DFM_OUTPUT_flowfm/flowfm_0000_20120801_000000_map.nc'
store0=xr.backends.netCDF4_.NetCDF4DataStore.open(nc0_fn)
ds0=xr.open_dataset(nc0_fn)

store=DFMMergeMap.open('runs/hor_002/DFM_OUTPUT_flowfm/*0_map.nc')
ds=open_merged_dataset('runs/hor_002/DFM_OUTPUT_flowfm/*0_map.nc')

print(ds)

## 

# Come up with mapping for multiple domains to one.
# at least with 53925 output, we get FlowElemDomain, 0-based.
# FlowElemGlobalNr: 1-based.
# FlowLinkDomain
# note that original input grid and "interpreted" grid are not identical.







## 

# backends.netcdf4_
class NetCDF4DataStore(WritableCFDataStore, DataStorePickleMixin):
    """Store for reading and writing data via the Python-NetCDF4 library.

    This store supports NetCDF3, NetCDF4 and OpenDAP datasets.
    """

    def __init__(self, netcdf4_dataset, mode='r', writer=None, opener=None,
                 autoclose=False, lock=HDF5_LOCK):

        if autoclose and opener is None:
            raise ValueError('autoclose requires an opener')

        _disable_auto_decode_group(netcdf4_dataset)

        self._ds = netcdf4_dataset
        self._autoclose = autoclose
        self._isopen = True
        self.format = self.ds.data_model
        self._filename = self.ds.filepath()
        self.is_remote = is_remote_uri(self._filename)
        self._mode = mode = 'a' if mode == 'w' else mode
        if opener:
            self._opener = functools.partial(opener, mode=self._mode)
        else:
            self._opener = opener
        super(NetCDF4DataStore, self).__init__(writer, lock=lock)

    @classmethod
    def open(cls, filename, mode='r', format='NETCDF4', group=None,
             writer=None, clobber=True, diskless=False, persist=False,
             autoclose=False, lock=HDF5_LOCK):
        import netCDF4 as nc4
        if (len(filename) == 88 and
                LooseVersion(nc4.__version__) < "1.3.1"):
            warnings.warn(
                'A segmentation fault may occur when the '
                'file path has exactly 88 characters as it does '
                'in this case. The issue is known to occur with '
                'version 1.2.4 of netCDF4 and can be addressed by '
                'upgrading netCDF4 to at least version 1.3.1. '
                'More details can be found here: '
                'https://github.com/pydata/xarray/issues/1745')
        if format is None:
            format = 'NETCDF4'
        opener = functools.partial(_open_netcdf4_group, filename, mode=mode,
                                   group=group, clobber=clobber,
                                   diskless=diskless, persist=persist,
                                   format=format)
        ds = opener()
        return cls(ds, mode=mode, writer=writer, opener=opener,
                   autoclose=autoclose, lock=lock)

    def open_store_variable(self, name, var):
        with self.ensure_open(autoclose=False):
            dimensions = var.dimensions
            data = indexing.LazilyOuterIndexedArray(
                NetCDF4ArrayWrapper(name, self))
            attributes = OrderedDict((k, var.getncattr(k))
                                     for k in var.ncattrs())
            _ensure_fill_value_valid(data, attributes)
            # netCDF4 specific encoding; save _FillValue for later
            encoding = {}
            filters = var.filters()
            if filters is not None:
                encoding.update(filters)
            chunking = var.chunking()
            if chunking is not None:
                if chunking == 'contiguous':
                    encoding['contiguous'] = True
                    encoding['chunksizes'] = None
                else:
                    encoding['contiguous'] = False
                    encoding['chunksizes'] = tuple(chunking)
            # TODO: figure out how to round-trip "endian-ness" without raising
            # warnings from netCDF4
            # encoding['endian'] = var.endian()
            pop_to(attributes, encoding, 'least_significant_digit')
            # save source so __repr__ can detect if it's local or not
            encoding['source'] = self._filename
            encoding['original_shape'] = var.shape

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            dsvars = FrozenOrderedDict((k, self.open_store_variable(k, v))
                                       for k, v in
                                       iteritems(self.ds.variables))
        return dsvars

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            attrs = FrozenOrderedDict((k, self.ds.getncattr(k))
                                      for k in self.ds.ncattrs())
        return attrs

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            dims = FrozenOrderedDict((k, len(v))
                                     for k, v in iteritems(self.ds.dimensions))
        return dims

    def get_encoding(self):
        with self.ensure_open(autoclose=True):
            encoding = {}
            encoding['unlimited_dims'] = {
                k for k, v in self.ds.dimensions.items() if v.isunlimited()}
        return encoding

    def set_dimension(self, name, length, is_unlimited=False):
        with self.ensure_open(autoclose=False):
            dim_length = length if not is_unlimited else None
            self.ds.createDimension(name, size=dim_length)

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            if self.format != 'NETCDF4':
                value = encode_nc3_attr_value(value)
            _set_nc_attribute(self.ds, key, value)

    def set_variables(self, *args, **kwargs):
        with self.ensure_open(autoclose=False):
            super(NetCDF4DataStore, self).set_variables(*args, **kwargs)

    def encode_variable(self, variable):
        variable = _force_native_endianness(variable)
        if self.format == 'NETCDF4':
            variable = _encode_nc4_variable(variable)
        else:
            variable = encode_nc3_variable(variable)
        return variable

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):
        datatype = _get_datatype(variable, self.format)
        attrs = variable.attrs.copy()

        fill_value = attrs.pop('_FillValue', None)

        if datatype is str and fill_value is not None:
            raise NotImplementedError(
                'netCDF4 does not yet support setting a fill value for '
                'variable-length strings '
                '(https://github.com/Unidata/netcdf4-python/issues/730). '
                "Either remove '_FillValue' from encoding on variable %r "
                "or set {'dtype': 'S1'} in encoding to use the fixed width "
                'NC_CHAR type.' % name)

        encoding = _extract_nc4_variable_encoding(
            variable, raise_on_invalid=check_encoding,
            unlimited_dims=unlimited_dims)
        if name in self.ds.variables:
            nc4_var = self.ds.variables[name]
        else:
            nc4_var = self.ds.createVariable(
                varname=name,
                datatype=datatype,
                dimensions=variable.dims,
                zlib=encoding.get('zlib', False),
                complevel=encoding.get('complevel', 4),
                shuffle=encoding.get('shuffle', True),
                fletcher32=encoding.get('fletcher32', False),
                contiguous=encoding.get('contiguous', False),
                chunksizes=encoding.get('chunksizes'),
                endian='native',
                least_significant_digit=encoding.get(
                    'least_significant_digit'),
                fill_value=fill_value)
            _disable_auto_decode_variable(nc4_var)

        for k, v in iteritems(attrs):
            # set attributes one-by-one since netCDF4<1.0.10 can't handle
            # OrderedDict as the input to setncatts
            _set_nc_attribute(nc4_var, k, v)

        target = NetCDF4ArrayWrapper(name, self)

        return target, variable.data

    def sync(self, compute=True):
        with self.ensure_open(autoclose=True):
            super(NetCDF4DataStore, self).sync(compute=compute)
            self.ds.sync()

    def close(self):
        if self._isopen:
            # netCDF4 only allows closing the root group
            ds = find_root(self.ds)
            if ds._isopen:
                ds.close()
            self._isopen = False
