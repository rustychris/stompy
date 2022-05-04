from __future__ import print_function

from six import iteritems
import netCDF4
import os
import uuid
from .. import utils
import numpy as np
try:
    from collections.abc import Iterable
except ImportErro:
    from collections import Iterable
    
from scipy import interpolate
from scipy.signal import decimate

class QncException(Exception):
    pass

def to_str(s):
    if not isinstance(s,str) and isinstance(s,bytes):
        s=s.decode() # in case py3 and s is bytes
    return s

def sanitize_name(s):
    """
    make s suitable for use as a variable or dimension
    name
    """
    return to_str(s).replace(' ','_').replace('/','_')
    
def as_tuple(x):
    if isinstance(x,tuple):
        return x
    elif isinstance(x,list):
        return tuple(x)
    else:
        return (x,)

def anon_dim_name(size,**kws):
    """
    Name given to on-demand dimensions.
    kws: unused, but might include the type?
    """
    return 'd%d'%size

class QuickVar(object): # wraps netcdf variable
    # predefine QuickVar attributes for the sake of  __setattr__
    _nc=None
    _v=None
    _transpose=None
    # _converter=None
    def __init__(self,nc,v,transpose=None):
        self.__dict__['_nc']=nc
        self.__dict__['_v']=v
        # self.__dict__['_converter']=converter
        if transpose is None:
            transpose=range(len(v.dimensions))
        self.__dict__['_transpose']=transpose
    def __getitem__(self,k):
        """ the options are similar to indexing a record array,
        but strings refer to dimensions
        """

        # if k is a string or tuple of strings, transpose
        # and return a new quick var.
        # if k is a dict, no transposition, but map the values to
        # slices

        # otherwise delegate to var.

        # this makes list arguments and tuple arguments
        # appear the same, but in the spirit of following
        # numpy semantics, those should not be the same.
        if isinstance(k,dict):
            dims=self.dims
            slices=[ k.get(d,slice(None))
                     for d in self.dims ]
            k=tuple(slices)

        if not isinstance(k,tuple):
            k=(k,)

        k=tuple(to_str(ki) for ki in k)
        
        if isinstance(k[0],str):
            return self._transpose_by_names(k)
        else:
            myv=self._v

            # first, make k explicitly cover all dimensions
            #try:
            k=list(k) # make a list so we can modify it
            #except TypeError: # k just a single slice or index
            #    k=[k]

            for ki,kk in enumerate(k):
                if kk is Ellipsis:
                    expand_slc=slice(ki,ki+1)
                    expansion=[slice(None)]*(myv.ndim-(len(k)-1))
                    k[expand_slc] = expansion
                    break
            else:
                while len(k)< myv.ndim:
                    k.append( slice(None) )

            # k is still a set of slices on the transposed data
            untranspose=[ self._transpose.index(i) 
                          for i in range(myv.ndim) ]
            k_untransposed=[k[j] for j in untranspose]

            # retrieve the data, possibly having netcdf subset it
            # important to tuplify - different semantics than 
            # a list
            pulled=self._rich_take(myv,tuple(k_untransposed))

            # if none of the slices are a singleton, then we'd just 
            # apply our transpose and be done.
            # .. say the initial dims were [time,cell,z]
            #    and self._transpose=[1,0,2] # cell,time,z
            #    and k=[0,slice(None),slice(None)]
            # which means cell=0,all time, all z
            # then k_transposed=[slice(None),0,slice(None)]
            # pulled has dimensions of time,z
            # retranspose=[0,1]
            # compared to transpose which was [1,0,2]
            #  so of transpose, 
            retranspose=[i for i in self._transpose 
                         if (isinstance(k_untransposed[i],slice) or
                             isinstance(k_untransposed[i],Iterable))]
            # and renumber via cheesy np trick
            if len(retranspose):
                retranspose=np.argsort(np.argsort(retranspose))
                return pulled.transpose(retranspose)
            else:
                return pulled
    def _rich_take(self,myv,idxs):
        # allow more relaxed semantics, in particular, grabbing an
        # out of order or repeated set of indices

        new_idxs=[] # introduce some mangling
        post_idxs=[] # then use this to undo mangling

        for idx in idxs:
            post=slice(None)

            if isinstance(idx, Iterable):
                idx=np.asarray(idx)
                if ( idx.ndim!=1 or 
                     (len(idx)>1 and np.any(np.diff(idx)<=0)) ):
                    post=idx # have numpy do it.
                    idx=slice(None)
            new_idxs.append(idx)
            if not np.isscalar(idx):
                post_idxs.append(post)

        new_idxs=tuple(new_idxs)
        post_idxs=tuple(post_idxs)
        result=myv[new_idxs]
        if len(post_idxs):
            result=result[post_idxs] 
        return result
            
    def __setitem__(self,k,v):
        """
        limited support here - just nc.varname[slices]=value
        TODO: allow k to be a bitmask index.
        TODO: allow k to be dimensions, to transpose on the fly
        TODO: allow self to already have a transpose
        """
        myv=self._v
        # for now, no support for assigning into a transposed array
        if np.all( self._transpose == range(len(myv.dimensions)) ):
            # when k is a bitmask, this is super slow.
            if myv.size<1e6: # do it in python, in memory.
                value=myv[:]
                value[k]=v
                myv[:]=value
            else:
                myv[k]=v
        else:
            raise QncException("Tranpose is set - not ready for that")
        
    def __getattr__(self,attr):
        return getattr(self._v,attr)
    def __setattr__(self,attr,value):
        # ony pre-existing attributes are set on this object
        # all others are passed on to the real variable.
        if attr in self.__dict__:
            self.__dict__[attr]=value
        else:
            return setattr(self._v,attr,value)
    def _transpose_by_names(self,k):
        if isinstance(k,str):
            k=(k,)
        new_transpose=[self._v.dimensions.index(kk) for kk in k]
        return QuickVar(self._nc,self._v,new_transpose)
    def __len__(self):
        return len(self._v)
    @property
    def dims(self):
        return self.dimensions
    @property
    def dimensions(self):
        return self._v.dimensions

    def as_datenum(self):
        import nc_utils
        return nc_utils.cf_time_to_datenum(self._v)
            
class QDataset(netCDF4.Dataset):
    # 'varlen' will try to use variable length arrays to store strings
    # 'fixed' will add a strNN dimension, and strings are padded out to that size
    # 'varlen' may not be supported on as wide a range of library versions.
    # neither is well-tested at this point
    
    def __init__(self,*args,**kws):
        # seems that it has to be done this way, otherwise the setattr/getattr
        # overrides don't see it.
        super(QDataset,self).__init__(*args,**kws)
        self._set_string_mode('varlen') # 'fixed'
        self.__dict__['_auto_dates']=True

    def _set_string_mode(self,mode):
        self.__dict__['_string_mode']=mode

    def __getattr__(self,attr):
        # only called when attribute is not found in dict
        if attr in self.variables:
            return QuickVar(self,self.variables[attr])
        raise AttributeError(attr)

    class VarProxy(object):
        """ represents a variable about to be defined, but
        waiting for dimension names and data, via setattr.
        """
        def __init__(self,dataset,varname):
            self.dataset=dataset
            self.varname=to_str(varname)
        def __setitem__(self,dims,data):
            """ syntax for creating dimensions and variables at the
            same time.

            nc['var_name']['dim1','dim2','dim3']=np.array(...)
            """
            return self.create(dims,data)

        def create(self,dims,data,**kwargs):
            if self.varname in self.dataset.variables:
                print( "Trying to setitem on varname=%s"%(self.varname) )
                print( "Existing dataset state:" )
                self.dataset._dump()
                raise QncException("Can't create variable twice: %s"%self.varname)

            attrs=kwargs.get('attrs',{})

            data=np.asarray(data)
            # create dimensions on-demand:
            dims=as_tuple(dims)

            dims=tuple(to_str(di) for di in dims)
            
            if len(dims) and dims[-1]==Ellipsis:
                dims=dims[:-1] #drop the ellipsis
                # add new dimensions named for their size, according to the
                # trailing dimensions of data 
                extras=[anon_dim_name(size=n) for n in data.shape[len(dims):]]
                if extras:
                    dims=dims + tuple(extras)
                    #print "New dimensions: "
                    #print dims

            # A college try at dereferencing objects
            if data.dtype==object:
                if np.all( [isinstance(v,str) or isinstance(v,unicode) for v in data.ravel()] ):
                    data=data.astype('S')
            
            # and handle strings by also adding an extra generic dimension
            dtype_str=data.dtype.str
            # print "Dtype: ",dtype_str
            if dtype_str.startswith('|S') or dtype_str.startswith('S'):
                # print "It's a string!"
                slen=data.dtype.itemsize

                if slen>1 and self.dataset._string_mode=='fixed':
                    dims=dims + ("str%d"%slen,)
                    new_shape=data.shape + (slen,)
                    new_data=np.fromstring(data.tostring(),'S1').reshape(new_shape)
                    data=new_data
                    # print "New data dtype: ",data.dtype

            # get smart about np datetime64 values
            if self.dataset._auto_dates and (data.dtype.type == np.datetime64):
                # CF conventions can't deal with nanoseconds...
                # this will lose sub-second values - could try going with floats
                # instead??
                data=data.astype('M8[s]').astype(np.int64)
                # Assumes UTC
                attrs['units']='seconds since 1970-01-01 00:00:00'

            data_shape=list(data.shape)
            var_dtype=data.dtype

            for dim_idx,dim_name in enumerate(dims):
                self.dataset.add_dimension(dim_name,data.shape[dim_idx])

            variable=self.dataset.createVariable(self.varname,
                                                 var_dtype,
                                                 dims,**kwargs)
            variable[:]=data
            for k,v in iteritems(attrs):
                setattr(variable,k,v)

    def add_dimension(self,dim_name,length):
        """ 
        create dimension if it doesn't exist, otherwise check that
        the length requested matches what does exist.
        """
        if dim_name in self.dimensions:
            assert ( self.dimensions[dim_name].isunlimited() or
                     (len(self.dimensions[dim_name]) == length) )
        else:
            self.createDimension(dim_name,length)

    def __getitem__(self,k):
        if k in self.variables:
            return QuickVar(self,self.variables[k])
        else:
            return self.VarProxy(dataset=self,varname=k)

    def __setitem__(self,k,val):
        # shorthand for dimensionless variable 
        self[k][()]=val

    def __contains__(self,k):
        return k in self.variables
    
    def alias(self,**kwargs):
        """ had been just copying the variables.  But why not just 
        update self.variables?  This works even if not writing
        to the file.
        """
        for k,v in iteritems(kwargs):
            if 0: # deep copy:
                self[k][self.variables[v].dimensions] = self.variables[v][:]

                for attr_name in self.variables[v].ncattrs():
                    setattr(self.variables[k],attr_name,
                            getattr(self.variables[v],attr_name))
            else:
                self.variables[k]=self.variables[v]

    def interpolate_dimension(self,int_dim,int_var,new_coordinate,
                              max_gap=None,gap_fields=None,
                              int_mode='nearest'):
        """ 
        return a new dataset as a copy of this one, but with
        the given dimension interpolated according to varname=values
        
        typically this would be done to a list of datasets, after which
        they could be appended.

        it can also be used to collapse a 'dependent' coordinate into
        an independent coordinate - e.g. if depth bins are a function of
        time, this can be used to interpolate onto a constant depth axis,
        which will also remove the time dimension from that depth variable.

        max_gap: jumps in the source variable greater than max_gap are filled
        with nan (or -99 if int valued).  For now this is only supported when
        int_dim has just one dimension
        gap_fields: None, or a list of variable names to be masked based on gaps.

        int_mode: 
          'nearest' - grab the integer value from the nearest sample
          may add 'linear' in the future, which would cast to float
        """
        result=empty()

        int_ncvar=self.variables[int_var]
        if len(int_ncvar.dimensions)>1:
            print( "Will collapse %s"%int_var)
            if max_gap:
                raise QncException("max_gap not implemented for multi-dimensional variable")
        else:
            if max_gap:
                gapped=np.zeros(new_coordinate.shape,'b1')
                deltas=np.diff(int_ncvar[:])

                gap_idx=np.nonzero(deltas>max_gap)[0]
                for idx in gap_idx:
                    gap_left=int_ncvar[idx]
                    gap_right=int_ncvar[idx+1]
                    print( "Gap is %f - %f (delta %f)"%(gap_left,gap_right,gap_right-gap_left) )
                    to_mask=slice(*np.searchsorted( new_coordinate, [gap_left,gap_right] ))
                    gapped[to_mask]=True
                    
        for varname in self.variables.keys():
            dim_names=self.variables[varname].dimensions

            if int_dim in dim_names:
                merge_idx=dim_names.index(int_dim)
            else:
                merge_idx=None

            if varname==int_var:
                # takes care of matching the coordinate variable.  
                # but there will be trouble if there are multiple
                # coordinate variables and you try to concatenate
                # use int_dim here instead of dim_names, because we might
                # be collapsing the interpolant variable.
                result[varname][int_dim]=new_coordinate
            elif merge_idx is not None:
                # it's an array-valued variable, and includes 
                # the dimension over which we want to interpolate

                int_all_dims=self.variables[int_var].dimensions
                src_var=self.variables[varname]
                src_val=src_var[:]

                # masked values don't work so well with the 
                # interpolation:
                if isinstance(src_val,np.ma.core.MaskedArray):
                    print( "Filling masked src data" )
                    if 'int' in src_val.dtype.name:
                        src_val=src_val.filled(-1)
                    else:
                        src_val=src_val.filled(np.nan)

                if len(int_all_dims)==1:
                    if ('int' in src_val.dtype.name) and (int_mode=='nearest'):
                        def interper(coord):
                            idxs=utils.nearest(self.variables[int_var][:],coord)
                            slices=[slice(None)]*src_val.ndim
                            slices[merge_idx]=idxs
                            return src_val[slices]
                    else:
                        interper=interpolate.interp1d(self.variables[int_var][:],
                                                      src_val,
                                                      axis=merge_idx,
                                                      bounds_error=False,
                                                      assume_sorted=False)
                    new_val=interper(new_coordinate)

                    if max_gap and ( (gap_fields is None) or (varname in gap_fields)):
                        if 'int' in new_val.dtype.name:
                            new_val[gapped] = -99
                        else:
                            new_val[gapped] = np.nan
                    result[varname][dim_names]=new_val
                else:
                    # here's the tricky part when it comes to collapsing
                    # dimensions
                    # self.variables[int_var] - this is multidimensional
                    # basically we want to iterate over elements in 

                    # iterate over all other dimensions of the int_var
                    int_values=self.variables[int_var][:]
                    int_dim_idx=self.variables[int_var].dimensions.index(int_dim)

                    # preallocate the result, since we have to fill it in bit-by-bit
                    dest_shape=list(src_var.shape)
                    dest_shape[merge_idx] = len(new_coordinate)
                    dest=np.zeros(dest_shape,dtype=src_val.dtype)

                    # start with a stupid, slow way:
                    # Assume that there is one extra dimension and it's the first one.

                    # int_ncvar has multiple dimensions, i.e. depth ~ bin,time
                    # so there is an assumption here that some variable to be interpolated
                    # like u=u(bin,time)
                    # could there be something that depended on bin, but not time?
                    # there aren't any in the existing adcp data

                    for extra in range(int_ncvar.shape[0]):
                        interper=interpolate.interp1d(int_ncvar[extra,:],
                                                      src_val[extra,:],
                                                      axis=merge_idx-1, # account for extra
                                                      bounds_error=False,
                                                      assume_sorted=False)
                        dest[extra,:]=interper(new_coordinate)
                    result[varname][dim_names]=dest

            else: # non-dimensioned attributes here
                result[varname][dim_names]=self.variables[varname][:]
            self.copy_ncattrs_to(result)
        return result

    def copy(self,skip=[],fn=None,**create_args):
        """ make a deep copy of self, into a writable, diskless QDataset
        if fn is given, target is a netCDF file on disk.
        """
        if fn is not None:
            new=empty(fn,**create_args)
        else:
            new=empty()
        for dimname in self.dimensions.keys():
            if dimname not in skip:
                new.createDimension(dimname,len(self.dimensions[dimname]))
        for varname in self.variables.keys():
            if varname not in skip:
                ncvar=self.variables[varname]
                new[varname][ncvar.dimensions] = ncvar[:]
        self.copy_ncattrs_to(new)
        return new
    def copy_ncattrs_to(self,new):
        for varname in self.variables.keys():
            myvar=self.variables[varname]
            if varname not in new.variables:
                continue
            newvar = new.variables[varname]
            for attr in myvar.ncattrs():
                if attr != '_FillValue':
                    # _FillValue can only be set at var creation time
                    # This approach was failing on valid_range, 2017-03-17
                    # setattr(newvar,attr,getattr(myvar,attr))
                    newvar.setncattr(attr,getattr(myvar,attr))
    def select(self,**kwargs):
        new=empty()

        for varname in self.variables.keys():
            dim_names=self.variables[varname].dimensions
            if len(dim_names)==0:
                newvar=new.createVariable(varname,self.variables[varname].dtype,())
                newvar[...]=self.variables[varname][...]
            else:
                slices=[slice(None)]*len(dim_names)

                for slc_dim,slc_sel in iteritems(kwargs):
                    if slc_dim in dim_names:
                        slc_idx=dim_names.index(slc_dim)
                        slices[slc_idx]=slc_sel
                    else:
                        print( "slice dimension %s not in dim_names %s"%(slc_dim,dim_names) )

                new[varname][dim_names]=self.variables[varname][slices]
        self.copy_ncattrs_to(new)
        return new
    def within(self,**kwargs):
        selects={}
        for slc_varname,slc_range in iteritems(kwargs):
            slc_var=self.variables[slc_varname]
            assert( len(slc_var.dimensions)==1 )
            
            selects[slc_var.dimensions[0]]=utils.within(slc_var[:],slc_range,as_slice=True)
        return self.select(**selects)
    def _dump(self):
        print( self._desc() )

    def _desc(self):
        """ returns pretty printed description of dataset similar to 
        output of ncdump
        """
        lines=[ "%s %s {"%( self.file_format.lower(), "unknown_filename" ) ]
        lines.append( self._desc_dims() )
        lines.append( self._desc_vars() )
        lines.append( "}" )

        return "\n".join(lines)

    def _desc_dims(self):
        lines=["dimensions:"]
        for k,v in iteritems(self.dimensions):
            lines.append("    %s = %s ;"%(k,len(v) ))
        return "\n".join(lines)
    def _desc_vars(self,max_attr_len=20,max_n_attrs=7):
        lines=["variables:"]

        for k,v in iteritems(self.variables):
            try:
                typ=v.dtype.name
            except AttributeError:
                typ=str(v.dtype)
            lines.append( "    %s %s(%s) ;"%( typ, k, ",".join( v.dimensions )) )
            for attr_name in v.ncattrs()[:max_n_attrs]:
                a_val=getattr(v,attr_name)
                if isinstance(a_val,str) and len(a_val)>max_attr_len:
                    a_val = a_val[:max_attr_len] + "... [%d more bytes]"%(len(a_val)-max_attr_len)
                    a_val = '"' + a_val + '"'
                lines.append('         %s:%s = %s'%(k,attr_name,a_val))
            if len(v.ncattrs()) > max_n_attrs > 0:
                lines.append('         ... %d more'%(len(v.ncattrs())-max_n_attrs))
        return "\n".join(lines)

def empty(fn=None,overwrite=False,**kwargs):
    if fn is None:
        return QDataset(uuid.uuid1().hex,'w',diskless=True,**kwargs)
    else:
        if os.path.exists(fn):
            if overwrite:
                os.unlink(fn)
            else:
                raise QncException('File %s already exists'%fn)
        return QDataset(fn,'w',**kwargs)

def concatenate(ncs,cat_dim,skip=[],new_dim=None):
    """ ncs is an ordered list of QDataset objects
    If a single QDataset is given, it will be copied at the metadata
    level
    new_dim: if given, then fields not having cat_dim, but differing
     between datasets, will be concatenated along new_dim.

    for convenience, elements of gdms which are None are silently dropped
    """
    ncs=filter(None,ncs)
    N=len(ncs)
    if N==1:
        return ncs[0].copy()
    if N==0:
        return empty()
    
    result=empty()
    
    for varname in ncs[0].variables.keys():
        if varname in skip:
            continue

        dim_names=ncs[0].variables[varname].dimensions
    
        if cat_dim in dim_names:
            cat_idx=dim_names.index(cat_dim)
        else:
            cat_idx=None
            
        parts=[nc.variables[varname][:] for nc in ncs]
        if cat_idx is not None:
            result[varname][dim_names]=np.concatenate(parts,axis=cat_idx)
        else:
            constant=True
            for n in range(1,N):
                if np.any(parts[0]!=parts[n]):
                    constant=False
                    break

            if not constant:
                if new_dim is None:
                    raise QncException("Non-concatenated variable %s "\
                                    "does not match %s != %s"%(varname,
                                                               parts[0],
                                                               parts[n]))
                else:
                    print( "Variable values of %s will go into new dimension %s"%(varname,
                                                                                  new_dim) )
                    result[varname][ [new_dim]+list(dim_names) ]=np.array(parts)
            else:
                result[varname][dim_names]=parts[0]

    # attrs are copied from first element
    ncs[0].copy_ncattrs_to(result)
    return result
    


# Functional manipulations of QDataset:

def downsample(ds,dim,stride,lowpass=True):
    """ Lowpass variables along the given dimension, and resample
    at the given stride.
    lowpass=False   => decimate, no lowpass
    lowpass=<float> => lowpass window size is lowpass*stride
    """
    lowpass=float(lowpass)
    winsize=int(lowpass*stride)

    new=empty()

    for var_name in ds.variables:
        ncvar=ds.variables[var_name]
        val=ncvar[:]
        if dim in ncvar.dimensions:
            dim_idx=ncvar.dimensions.index(dim)
            # should be possible to use the faster scipy way,
            # but it is having issues with setting nan's in the
            # output - maybe something is getting truncated or 
            # reshaped??
            if True: # lowpass!=1: # older, slower way:
                if lowpass:
                    import lp_filter
                    val_lp=lp_filter.lowpass_fir(val,winsize,axis=dim_idx,nan_weight_threshold=0.5)
                else:
                    val_lp=val
                slcs=[slice(None)]*len(ncvar.dimensions)
                slcs[dim_idx]=slice(None,None,stride)
                val=val_lp[slcs]
            else: # scipy.signal way:
                kws=dict(q=stride,ftype='fir',axis=dim_idx)
                val_valid=decimate(np.isfinite(val).astype('f4'),**kws)
                val=decimate(val,**kws)
                val[val_valid<0.1] = np.nan
                new[var_name+'_msk'][ncvar.dimensions]=val_valid # DBG

        new[var_name][ncvar.dimensions] = val

    ds.copy_ncattrs_to(new)
    return new

def mat_to_nc(mat,dim_map={},autosqueeze=True):
    nc=empty()

    length_to_dim={}

    if autosqueeze:
        sq=np.squeeze
    else:
        sq=lambda x: x

    for k,v in iteritems( dim_map ):
        if isinstance(v,str):
            v=[v]

        if k in mat:
            x=sq(mat[k])
            if x.ndim != len(v):
                raise QncException("dim_map: field %s - %s doesn't match with %s"%(k,v,x.shape))
            for dim_name,size in zip(v,x.shape):
                length_to_dim[size]=dim_name
    for k,v in iteritems(mat):
        if k.startswith('_'):
            print( "Skipping %s"%k)
            continue
        v=sq(v)
        if not isinstance(v,np.ndarray):
            print( "Don't know how to deal with variable of type %s"%str(type(v)))
            continue
        if v.ndim==0:
            setattr(nc,k,v.item())
            continue
        if k in dim_map:
            dims=dim_map[k]
        else:
            dims=[]
            for size in v.shape:
                if size not in length_to_dim:
                    length_to_dim[size]='d%d'%size
                dims.append(length_to_dim[size])
        # special handling for some datatypes:
        if v.dtype == np.dtype('O'):
            # sometimes this is just ragged array
            # for some reason mat files can have a 3-dimensional variable reported
            # as an array of 2-d arrays.
            # TODO
            print( "%s is not a simple array. Skipping"%k )
            continue
        nc[k][dims]=v
    return nc



def linear_to_orthogonal_nc(nc_src,lin_dim,ortho_dims,nc_dst=None):
    """ 
    copy a dataset, changing a linear dimension into a pair of orthogonal
    dimensions
    """
    if isinstance(nc_src,str):
        nc=qnc.QDataset(nc_src)
    else:
        nc=nc_src
        
    if nc_dst is None:
        nc2=qnc.empty()
    elif isinstance(nc_dst,str):
        nc2=qnc.empty(fn=nc_dst)
    else:
        nc2=nc_dst

    ortho_coords=[np.unique(nc.variables[d][:]) for d in ortho_dims]
    ortho_shape =[len(oc) for oc in ortho_coords]
    map1=np.searchsorted(ortho_coords[0],nc.variables[ortho_dims[0]][:])
    map2=np.searchsorted(ortho_coords[1],nc.variables[ortho_dims[1]][:])

    for d in ortho_dims:
        vals=np.unique(nc.variables[d][:])
        nc2[d][d]=vals
        nc2.variables
    # nc2.set_coords(ortho_dims)

    for v in nc.variables:
        print("Processing %s"%v)
        if v in ortho_dims: 
            continue
        create_kwargs={}
        if '_FillValue' in nc.variables[v].ncattrs():
            create_kwargs['fill_value']=nc.variables[v]._FillValue
            
        if lin_dim in nc.variables[v].dimensions:
            old_val=nc.variables[v][:]
            new_dims=[] ; new_shape=[] ; new_slices=[]
            for d in nc.variables[v].dimensions:
                if d==lin_dim:
                    new_dims+=ortho_dims
                    new_shape+=ortho_shape
                    new_slices+=[map1,map2]
                else:
                    new_dims.append(d)
                    new_shape.append(len(nc.variables[d]))
                    new_slices+=[slice(None)]

            new_val=np.zeros(new_shape,old_val.dtype)
            new_val[:] = np.nan
            new_val[tuple(new_slices)] = old_val

            # dims=[nc2[d] for d in new_dims]
            nc2[v].create(new_dims,new_val,**create_kwargs)
        else:
            #print "COPY",v,nc[v].dims
            data=nc.variables[v][:]
            if data.dtype==object:
                data=data.astype('S')
            nc2[v].create(nc.variables[v].dimensions,data,**create_kwargs)

        for att in nc.variables[v].ncattrs():
            if att == '_FillValue': # can't be added after the fact
                continue
            setattr(nc2.variables[v],att,getattr(nc.variables[v],att))
            
    if isinstance(nc_src,str):
        nc.close()
    return nc2

def ortho_to_transect_nc(src_nc,src_x_var,src_y_var,transect_xy,dst_nc=None):
    """ Extract a transect to a new dataset
    """
    if isinstance(src_nc,str):
        src_nc=qnc.QDataset(src_nc)
        close_src=True
    else:
        close_src=False
        
    if dst_nc is None:
        dst_nc=qnc.empty()
    elif isinstance(dst_nc,str):
        dst_nc=qnc.empty(fn=dst_nc)
    else:
        pass
    
    src_xy=np.array( [src_nc.variables[src_x_var][:],src_nc.variables[src_y_var][:]] ).T

    elti_sel=[]
    for xy in transect_xy:
        dists=utils.dist(xy,src_xy)
        elti_sel.append(np.argmin(dists))
    elti_sel=np.array(elti_sel)    

    station_dim=src_nc.variables['x'].dimensions[0]
    print("Station dim inferred to be %s"%station_dim)        

    # extract profile data from the original netcdf, building a new xarray
    for field in src_nc.variables:
        print(field,end=" ")
        fvar=src_nc[field]
        if station_dim in fvar.dims:
            dst_nc[field][fvar.dims]=fvar[{station_dim:elti_sel}]
        else:
            dst_nc[field][fvar.dims]=fvar[:]
    print()

    # and derived coordinates
    dst_nc['distance'][station_dim]=utils.dist_along(dst_nc[src_x_var][:],
                                                     dst_nc[src_y_var][:])
    # trans=trans.set_coords(['distance'])
    return dst_nc


# enhancements:
#   read some local config file, and remap variables and units
#   to the local user's preferences.
#   this would include translating time variables to a common standard,
#   adding aliases like 'u'->'east_vel', with rules based on a combination
#   of standard names, long names, units, and variable names.
