"""
Converting between nefis and netcdf
In a separate module to separate dependencies

Relies on the qnc wrapper of netcdf4
"""

from collections import defaultdict
from ...io import qnc

def nefis_to_nc(nef,squeeze_unl=True,squeeze_element=True,
                short_if_unique=True,to_lower=True,unl_name='time',
                element_map={},nc_kwargs={},nc=None):
    """
    nef: an open Nefis object
    squeeze_unl: unit length unlimited dimensions in groups are dropped
      groups can't be dimensionless, so parameter values are often in
      a unit-length group.
    short_if_unique: element names which appear in only one group get
      just the element name.  others are group_element
    to_lower: make names lower case
    squeeze_element: unit dimensions in elements are also removed
    unl_name: if there is a unique unlimited dimension length (ignoring
      unit lengths if squeeze_unl is set) - use this name for it.
    element_map: map original element names to new names.  this matches
      against element names before to_lower (but after strip), and the results
      will *not* be subject to to_lower.
    nc_kwargs: dict of argument to pass to qnc.empty
    nc: altenatively, an already open QDataset
    """
    if nc is None:
        nc=qnc.empty(**nc_kwargs)
    # required for writing to disk
    nc._set_string_mode('fixed')

    # string handling is a little funky -
    # still working out whether it should be varlen or fixed.
    # fixed makes the string length a new dimension, just annoying
    # to get strings back from that.
    # varlen ends up having an object dtype, which may not write out
    # well with netcdf.

    # check for unique element names
    name_count=defaultdict(lambda: 0)
    for group in nef.groups():
        for elt_name in group.cell.element_names:
            name_count[elt_name]+=1

    # check for unique unlimited dimension:
    n_unl=0
    for group in nef.groups():
        if 0 in group.shape and (group.unl_length()>1 or not squeeze_unl):
            n_unl+=1

    for group in nef.groups():
        # print group.name

        g_shape=group.shape
        grp_slices=[slice(None)]*len(g_shape)
        grp_dim_names=[None]*len(g_shape)

        if 0 in g_shape: # has an unlimitied dimension
            idx=list(g_shape).index(0)
            if group.unl_length()==1 and squeeze_unl: # which will be squeezed
                grp_slices[idx]=0
            elif n_unl==1 and unl_name: # which will be named
                grp_dim_names[idx]=unl_name

        for elt_name in group.cell.element_names:
            # print elt_name

            if name_count[elt_name]==1 and short_if_unique:
                vname=elt_name
            else:
                vname=group.name + "_" + elt_name

            if vname in element_map:
                vname=element_map[vname]
            elif to_lower:
                vname=vname.lower()

            value=group.getelt(elt_name)
            # apply slices 
            value=value[tuple(grp_slices)]

            if squeeze_element:
                # slices specific to this element
                val_slices=[slice(None)]*len(value.shape)
                # iterate over just the element portion of the shape
                for idx in range(len(g_shape),len(val_slices)):
                    if value.shape[idx]==1:
                        val_slices[idx]=0
                value=value[val_slices]

            # mimics qnc naming.
            names=[qnc.anon_dim_name(size=l) for l in value.shape]
            for idx,name in enumerate(grp_dim_names):
                if name:
                    names[idx]=name
            names.append(Ellipsis)

            nc[vname][names] = value
            setattr(nc.variables[vname],'group_name',group.name)
    return nc

