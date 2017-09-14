from __future__ import print_function

import nose
import nefis
import numpy as np

def get_nef():
    return nefis.Nefis('data/trim-tut_fti_waq.dat',
                       'data/trim-tut_fti_waq.def')

def test_open():
    nef=nefis.Nefis('data/trim-tut_fti_waq.dat',
                    'data/trim-tut_fti_waq.def')
    assert(nef)
    nef.close()

def test_double_open():
    nef1=get_nef()
    assert(nef1)
    try:
        nef2=get_nef()
        assert(False)
    except nefis.NefisException as exc:
        assert(exc.code==8001)
        assert("has already been opened" in exc.msg)
    nef1.close()

def test_get_header():
    nef=get_nef()
    s=nef.get_header()
    assert(s)
    assert(s.startswith('Deltares'))
    nef.close()

def test_get_groups():
    nef=get_nef()
    groups=nef.groups()
    print(groups)
    assert(len(groups) == 5)
    # make sure map-const shows up somewhere
    for g in groups:
        if g.name.strip()=='map-const':
            break
    else:
        assert(False)
    nef.close()

def test_group_str():
    nef=get_nef()
    groups=nef.groups()
    grp_str=str(groups[0])
    assert( grp_str=="<NefisGroup map-const:map-const, [unl]>" )
    nef.close()

def test_get_elements():
    nef=get_nef()
    elt=nef.get_element('U1')
    assert(elt)
    nef.close()

def test_elt_str():
    nef=get_nef()
    elt=nef.get_element('U1')
    assert( str(elt) == '<NefisElement U1 REAL [22,15,10]>' )
    nef.close()

def test_get_cells():
    nef=get_nef()
    cells=nef.cells()
    assert(cells)
    nef.close()

def test_cell_str():
    nef=get_nef()
    cell=nef.cells()[0]
    assert( str(cell) == "<NefisCell map-const, 19704 bytes>" )
    nef.close()

def test_summary():
    nef=get_nef()
    summary=nef.summary()
    assert( 'CELLS' in summary)
    assert( 'GROUPS' in summary)
    assert( 'map-const:map-const [unl]' in summary)
    assert( 'map-const: 19704 bytes' in summary)
    nef.close()

def test_group_getitem():
    nef=get_nef()
    grp=nef['map-series']
    assert(grp)
    assert(grp.name=='map-series')
    nef.close()

def test_unl_length():
    nef=get_nef()
    assert( nef['map-series'].unl_length() == 11 )
    nef.close()

# Attribute tests - at this point, don't have any confirmation of
# nefis files that have attributes - can only test that these routines
# complete.
def test_int_attrs():
    nef=get_nef()
    attrs=nef.groups()[0].attrs_int()
    print(attrs)
    nef.close()

def test_real_attrs():
    nef=get_nef()
    attrs=nef.groups()[0].attrs_real()
    print(attrs)
    nef.close()

def test_str_attrs():
    nef=get_nef()
    attrs=nef.groups()[0].attrs_str()
    print(attrs)
    nef.close()

def test_getelt():
    nef=get_nef()
    data=nef['map-series'].getelt('U1')
    nef.close()

    assert(data is not None)
    # time should be first index, and this dataset starts with a quiescent
    # field
    assert(np.allclose(data[0,:],0))

    # weak check on dimensions - 6,3 is a masked cell, so should be 
    # zero for all times and vertical indices.
    assert( np.all( data[:,:,6,3] == 0.0 ) )

def test_getelt_with_shape():
    nef=get_nef()
    # all slices specified, one of them a subslice
    data=nef['map-series'].getelt('U1',[slice(1,3),slice(None),slice(None),slice(None)])
    assert(data.shape[0]==2)

    # default for trailing dimensions
    data=nef['map-series'].getelt('U1',[slice(1,3)])
    assert(data.shape[0]==2)
    assert( len(data.shape)==4 )

    # single element - not a slice. 
    data=nef['map-series'].getelt('U1',[1])
    assert(data.shape[0]==10)
    assert( len(data.shape)==3 )

    # ellipsis
    data=nef['map-series'].getelt('U1',[Ellipsis,2,3])
    print("Actually ellipsis data shape",data.shape)
    assert( data.shape == (11,10) )

    nef.close()

def test_character_element():
    nef=nefis.Nefis('data/tut_fti_waq.hda',
                    'data/tut_fti_waq.hdf')
    
    sub_names=nef['DELWAQ_PARAMS'].getelt('SUBST_NAMES',[0])
    assert(sub_names[0].strip()=='IM1')
    

def test_invalid_file():
    try:
        nef=nefis.Nefis('data/tut_fti_waq-bal.his')
    except nefis.NefisException as exc:
        assert(exc.code==8023)
        return
    assert(False)
    
def test_del():
    nef=get_nef()
    nef.close()
    del nef
    

if __name__ == '__main__':
    nose.main()
