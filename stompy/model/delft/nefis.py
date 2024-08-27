"""
a NEFIS reader in python using ctypes to link in the nefis library
at run time.

"""
# TODO: 
#  cleaner interface for reopening/caching/etc.  The NEFIS library
#  refuses to open a file a second time.  Could support a 'reopen'
#  mechanism - need to flush any local, python state.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import pdb

def to_str(s):
    if not isinstance(s,str):
        s=s.decode()
    return s
def to_bytes(s):
    if not isinstance(s,bytes):
        s=s.encode()
    return s

log=logging.getLogger(__name__)

from ctypes import * 

# better to get this from an environment variable
def load_nef_lib():
    """
    Find and load the dll for NEFIS.
    This has to come from a compiled D-WAQ installation.

    Tries these locations:
     $HOME/code/delft/d3d/master/src/lib
     $PYTHON_DIR/lib
     $D3D_HOME/lib

    return None if the DLL cannot be found
    """
    if 'NEFIS_LIB' in os.environ:
        try:
            nefis_lib=os.environ['NEFIS_LIB']
            return cdll.LoadLibrary(nefis_lib)
        except OSError:
            log.warning("Failed to load nefis DLL - read/write not enabled")
            log.warning("Used env. var NEFIS_LIB='%s'"%nefis_lib)
            return None
    
    if sys.platform.startswith('linux'):
        basenames=['libNefisSO.so','libnefis.so']
    elif sys.platform=='darwin':
        # this is for OSX
        basenames=['libNefisSO.dylib']
    else:
        log.warning("Need to add support in nefis.nef_lib() for platform=%s"%sys.platform)
        return None

    fail='_fail_' # to test for missing environment variables
    locations=[]
    if 'D3D_HOME' in os.environ:
        locations.append( os.path.join(os.environ['D3D_HOME'],'lib') )
    if 'DELFT_HOME' in os.environ:
        locations.append( os.path.join(os.environ['DELFT_HOME'],'lib') )

    if (sys.platform=='linux') and ('LD_LIBRARY_PATH' in os.environ):
        locations.extend( os.environ['LD_LIBRARY_PATH'].split(':') )

    for prefix in locations:
        for basename in basenames:
            nefis_lib=os.path.join(prefix,basename)
            try:
                return cdll.LoadLibrary(nefis_lib)
            except OSError:
                continue
    log.warning("Failed to load nefis DLL - read/write not enabled")
    log.warning("Tried to load from %s"%(locations))
    log.warning("Tried basenames %s"%(basenames))
    return None

_nef_lib=False # False=> uninitialized, None=> not found
def nef_lib():
    global _nef_lib
    if _nef_lib is False:
        _nef_lib=load_nef_lib()
    return _nef_lib

class NefisException(Exception):
    def __init__(self,code,msg):
        self.code=code
        self.msg=msg
        super(NefisException,self).__init__(self.__str__())

    def __str__(self):
        return "{}: {}".format(self.code,self.msg)

class NefisMaxSizeException(NefisException):
    def __init__(self,msg):
        super(NefisMaxSizeException,self).__init__(code=0,msg=msg)

def fmt_shape(shape):
    def fmt(l):
        if l:
            return "%d"%l
        else:
            return "unl"
    return ",".join( [fmt(x) for x in shape] )

class NefisElement(object):
    def __init__(self,nefis,name,typ,nbytsg,quantity,unit,desc,shape):
        self.nefis=nefis
        self.name=to_str(name)
        self.typ=to_str(typ.strip())
        self.nbytsg=nbytsg
        self.quantity=quantity.strip()
        self.unit=to_str(unit.strip())
        self.desc=to_str(desc.strip())
        self.shape=shape

        self.np_type_code=None
        if self.typ=='REAL':
            if self.nbytsg==4:
                self.np_type_code='f4'
            else:
                self.np_type_code='f8'
        elif self.typ=='INTEGER':
            if self.nbytsg==8:
                self.np_type_code='i8'
            elif self.nbytsg==4:
                self.np_type_code='i4'
            elif self.nbytsg==2:
                self.np_type_code='i2'
            elif self.nbytsg==1:
                self.np_type_code='i1'
        elif self.typ=='CHARACTE':
            self.np_type_code='S%d'%self.nbytsg

        if self.np_type_code is None:
            raise NefisException(0,"Unknown type: {}, {} bytes".format(self.typ,self.nbytsg))

    def __str__(self):
        return "<NefisElement {} {} [{}]>".format(self.name,self.typ,
                                                  fmt_shape(self.shape))


class NefisGroup(object):
    # name, cellname,ndim,shape,order
    def __init__(self,nefis,name,cellname,shape,order):
        self.nefis=nefis
        self.name=to_str(name.strip())
        self.cellname=to_str(cellname.strip())
        self.shape=shape
        self.order=order

    @property
    def ndim(self):
        return len(self.shape)

    def __str__(self):
        return "<NefisGroup {}:{}, [{}]>".format(self.name,self.cellname,
                                                 fmt_shape(self.shape))
    def __repr__(self):
        return str(self)

    def unl_length(self):
        """ find out length of a unlimited dimension """
        grpnam=create_string_buffer(to_bytes(self.name),17)
        maxi=c_int(-1)
        
        err=nef_lib().Inqmxi(byref(self.nefis.fd),byref(grpnam),byref(maxi))
        if err:
            self.nefis.with_err(err)
        return maxi.value

    @property
    def cell(self):
        cname=self.cellname
        for c in self.nefis.cells():
            if c.name.strip() == cname.strip():
                return c
        return None

    def getelt(self,element,read_shape=None,shape_only=False):
        """
        read data from this group
        read_shape: select slices to read.  Note that there
        are two levels of shapes here - the shape of the group (often
        just a single, unlimited axis), and the shape of the element.
        
        read_shape is intended to cover slices for both, with the
        group-level dimensions coming first.

        return_shape: if True, figure out the shape of the result and return that as
         a tuple.  Takes read_shape into account.  Does not actually read any data.
        """
        uindex=np.ones((self.ndim,3),'i4')

        elt=self.nefis.get_element(element)
        elt_shape=list(elt.shape)
        itemsize=elt.nbytsg
        
        total_ndim=self.ndim + len(elt_shape)

        # read_shape is expanded to be a list of the same length
        # as the total number of dimensions - cell and element.
        # one Ellipsis can be included, and elements can be slices 
        # or integer scalars.
        if read_shape is None:
            read_shape=[] # default to reading entire element
        else:
            read_shape=list(read_shape)
        while len(read_shape) < total_ndim:
            read_shape.append(slice(None))

        # replace ellipsis with repeated slice(None)
        if Ellipsis in read_shape:
            ell_idx=read_shape.index(Ellipsis)
            expand=[slice(None)]*(total_ndim-len(read_shape)+1)
            read_shape[ell_idx:ell_idx+1]=expand

        # pad dimensions with full slices
        if len(read_shape) < total_ndim:
            read_shape=read_shape + [slice(None)]*( total_ndim-len(read_shape) )

        # put slices into uindex
        cell_shape=[]
        for idx in range(self.ndim):
            slc=read_shape[idx]
            
            size=self.shape[idx] or self.unl_length()

            # beware one-off errors and one-based / zero-based

            if not isinstance(slc,slice): # must be int-like?
                uindex[idx,0]=slc+1
                uindex[idx,1]=slc+1
                uindex[idx,2]=1
                # omit from cell_shape
            else:
                # assume for the moment that it's 1 based.
                uindex[idx,0]=1 + (slc.start or 0)
                uindex[idx,1]=slc.stop or size
                uindex[idx,2]=slc.step or 1
                cell_shape.append(uindex[idx,1] - uindex[idx,0] + 1)

        uindex=np.ascontiguousarray(uindex)
        uindex_ref=uindex.ctypes

        # always read in the natural order...
        uorder=np.arange(self.ndim,dtype='i4')+1
        uorder=np.ascontiguousarray(uorder)
        uorder_ref=uorder.ctypes

        # This is the result, with any cell-level slices applied,
        # but element-level slices come later.
        total_result_shape=cell_shape + elt_shape[::-1]

        if shape_only:
            return total_result_shape,elt.np_type_code

        buflen=c_int(np.prod(total_result_shape)*itemsize)

        data_nbytes=np.prod(total_result_shape)*itemsize
        u32_max=np.iinfo(np.uint32).max
        if data_nbytes>u32_max:
            msg="Buffer size for getelt (%d) would exceed uint32.max (%d)"%(data_nbytes,u32_max)
            raise NefisMaxSizeException(msg)

        data=np.ones(np.prod(total_result_shape),elt.np_type_code)
        data=np.ascontiguousarray(data)
        data_ref=data.ctypes

        grp_name=create_string_buffer(to_bytes(self.name))
        elm_name=create_string_buffer(to_bytes(element))

        err = nef_lib().Getelt(byref(self.nefis.fd),
                               byref(grp_name), byref(elm_name),
                               uindex_ref, uorder_ref,
                               byref(buflen),
                               data_ref)
        if err:
            self.nefis.with_err(err)

        data=data.reshape( *total_result_shape )

        # The Ellipsis skips over the cell-level slices
        # whatever they are
        post_slice=tuple([Ellipsis]+read_shape[self.ndim:])

        # now apply element level slices:
        data=data[ post_slice ]
        return data

    # Attributes
    def attrs(self):
        return self.attrs_int() + self.attrs_real() + self.attrs_str()

    def attrs_int(self):
        nef=nef_lib()
        return self.attrs_gen( lambda: c_int(-1),
                               nef.Inqfia, nef.Inqnia )

    def attrs_real(self):
        nef=nef_lib()
        return self.attrs_gen( lambda: c_float(-1),
                               nef.Inqfra, nef.Inqnra )

    def attrs_str(self):
        nef=nef_lib()
        return self.attrs_gen( lambda: create_string_buffer(17),
                               nef.Inqfsa, nef.Inqnsa )

    def attrs_gen(self,typ_factory,fn_first,fn_next):
        attrs=[]

        while True:
            if len(attrs)==0:
                fn=fn_first
            else:
                fn=fn_next

            grpnam=create_string_buffer(self.name.encode,17)
            attnam=create_string_buffer(17)
            attval=typ_factory()
        
            err= fn(byref(self.nefis.fd),byref(grpnam),byref(attnam),byref(attval))
            if err in [6016,6018,6020]:
                # different error codes for each type
                break
            elif err:
                self.nefis.with_err(err)

            attrs.append( attname.value, attval.value )
        return attrs
 

class NefisCell(object):
    def __init__(self,nefis,name,nbytes,element_names):
        self.nefis=nefis
        self.name=to_str(name.strip())
        self.nbytes=nbytes
        self.element_names=[to_str(s) for s in element_names]
    def __str__(self):
        return "<NefisCell {}, {} bytes>".format(self.name,self.nbytes)

class Nefis(object):
    def __init__(self,dat_fn,def_fn=None):
        self.elements={}
        self.dat_fn=dat_fn
        self.def_fn=def_fn

        self.open()

    def with_err(self,err):
        if err:
            buff=create_string_buffer(1025)
            nef_lib().nefis_error(0,byref(buff))
            raise NefisException(code=err,msg=buff.value)

    def open(self):
        self.fd=c_int(0)

        endian=c_char(b'N')
        access=c_char(b'r')

        # from nefis, oc.c, line 276 - looks like combined data/definition
        # files are loaded by specifying the same filename for both.
        nlib=nef_lib()
        if nlib is None:
            # This happens when either the library itself could not be found, or
            # on Linux it can happen when the library is found, but LD_LIBRARY_PATH
            # was not set *before python started*, and the linker fails to load
            # library dependencies.
            raise NefisException(code=-999,msg="NEFIS library could not be loaded.")
        else:
            self.with_err(nef_lib().Crenef(byref(self.fd), 
                                           self.dat_fn.encode(), 
                                           (self.def_fn or self.dat_fn).encode(),
                                           byref(endian), access))
    def close(self):
        nlib=nef_lib()
        if nlib is not None:
            self.with_err( nlib.Clsnef(byref(self.fd)) )
        self.fd=None

    def __del__(self):
        # no guarantees, but can help...
        if self.fd is not None:
            self.close()

    def reopen(self):
        self.close()
        self.open()

    header_max_len=128
    def get_header(self):
        # get basic header info:
        # init as long, null-terminated, so nefis knows it's long enough
        header_buff=create_string_buffer(b" "*self.header_max_len)
        self.with_err( nef_lib().Gethdf(byref(self.fd),
                                  byref(header_buff)) )
        # => 'Deltares, NEFIS Definition File; 5.00.00'
        return header_buff.value

    _groups=None
    def groups(self):
        if self._groups is None:
            groups=[]
            
            nef=nef_lib()
            while 1:
                grpnam=create_string_buffer(b" "*16)
                celnam=create_string_buffer(b" "*16)
                grpndm=c_int(5) # 5 is max - will be set to actual
                grpdms=np.ones(10,'i4')
                grpord=np.ones(10,'i4')

                if len(groups)==0:
                    fn=nef.Inqfgr
                else:
                    fn=nef.Inqngr

                err=fn(byref(self.fd),
                       byref(grpnam),
                       byref(celnam),
                       byref(grpndm),
                       grpdms.ctypes,
                       grpord.ctypes) 
                if err in [-6028,-6027]:
                    break # no more groups.
                elif err:
                    self.with_err(err)
                groups.append( NefisGroup(nefis=self,
                                          name=grpnam.value,
                                          cellname=celnam.value,
                                          shape=grpdms[:grpndm.value],
                                          order=grpord[:grpndm.value]) )
            self._groups=groups
        return self._groups

    def __getitem__(self,k):
        """ quick access to groups
        """
        for grp in self.groups():
            if grp.name==k:
                return grp
        raise KeyError(k)

    def get_element(self,name):
        name=name.strip()
        if name not in self.elements:
            self.elements[name]=self.get_element_real(name)
        return self.elements[name]

    def get_element_real(self,name):
        elmnam=(c_char*17)() ; elmnam.value=to_bytes(name)
        elmtyp=create_string_buffer(b" "*9)
        nbytsg=c_int(0)

        elmqty=(c_char*17)() ; elmqty.value=b" "*16
        elmunt=(c_char*17)() ; elmunt.value=b" "*16
        desc=(c_char*65)() ; desc.value=b" "*64 
        elmndm=c_int(5)
        elmdms=np.ones(5,'i4')

        self.with_err( nef_lib().Inqelm(byref(self.fd),byref(elmnam), # inputs
                                        byref(elmtyp), # outputs
                                        byref(nbytsg),
                                        byref(elmqty),byref(elmunt),
                                        byref(desc),
                                        byref(elmndm),
                                        elmdms.ctypes) )
        return NefisElement(nefis=self,
                            name=name,
                            typ=elmtyp.value,
                            nbytsg=nbytsg.value,
                            quantity=elmqty.value,
                            unit=elmunt.value,
                            desc=desc.value,
                            shape=elmdms[:elmndm.value])


    _cells=None
    def cells(self):
        if self._cells is None:
            nef=nef_lib()
            cells=[]
            
            while 1:
                celnam=create_string_buffer(b" "*16)
                nelems_alloc=200 # RH 20171031: had been 100, which is now too small for some runs.
                nelems=c_int(nelems_alloc) 
                nbytes=c_int(0)
                elmnms=((c_char*17)*nelems_alloc)()
     
                if len(cells)==0:
                    fn=nef.Inqfcl
                else:
                    fn=nef.Inqncl
                err = fn(byref(self.fd),
                         byref(celnam),
                         byref(nelems),
                         byref(nbytes),
                         byref(elmnms))
                if err in [-6025,-6026]:
                    break
                else:
                    self.with_err(err)

                cells.append( NefisCell( nefis=self,
                                         name=celnam.value,
                                         nbytes=nbytes.value,
                                         element_names=[ elmnms[idx].value[:16].strip()
                                                         for idx in range(nelems.value) ] ) )
            self._cells=cells
        return self._cells
    
    def summary(self):
        return "\n".join( [self.group_summary(),self.cell_summary()] )
    def group_summary(self):
        lines=[]
        lines.append("GROUPS")
    
        for grp in self.groups():
            lines.append("  {}:{} [{}]".format(grp.name,grp.cellname, fmt_shape(grp.shape) ))
        return "\n".join(lines)
    def cell_summary(self):
        lines=[]
        lines.append("CELLS")

        for cel in self.cells():
            lines.append("  {}: {} bytes".format(cel.name,cel.nbytes))
            for elt_name in cel.element_names:
                elt=self.get_element(elt_name)
                lines.append("     {}:{} [{}]".format(elt.name, elt.typ,fmt_shape(elt.shape)))
                if elt.desc or elt.unit:
                    lines.append("       {} {}".format(elt.desc,elt.unit))
        return "\n".join(lines)

