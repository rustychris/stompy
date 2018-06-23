# Methods for representing and manipulating transects
# in xarray datasets.

# Transects being a dataset with the form
#  x(s), y(s), z(s,k) or z(k)
#  u(s,k), v(s,k), etc.
# where x(s),y(s) describes a polyline.

# names are somewhat standardized, but can generally be provided
# U: vector valued velocity in geographic coordinate frame
#  the component dimension is last by convention, and its name
#  describes the coordinate system (xy, roz, tran)
# V<n>: component of a velocity (typ e, n, u)
# <N>_avg: depth averaged value
import six
import numpy as np
import xarray as xr
from .spatial import linestring_utils
from . import utils

def vert_dim(tran):
    # placeholder - need a more robust way of inferring which dimension
    # is vertical
    return 'laydim'

def infer_z_dz(tran,z_ctr='z_ctr',z_dz='z_dz',update=True):
    if z_dz in tran:
        return tran[z_dz]
    # center to center spacing:
    dctr=tran[z_ctr].diff(dim='laydim')
    dz_middle=0.5*(dctr.isel(laydim=slice(None,-1)) +
                   dctr.isel(laydim=slice(1,None)))
    dz=xr.concat( [ dz_middle.isel(laydim=slice(None,1)),
                    dz_middle,
                    dz_middle.isel(laydim=slice(-1,None)) ],
                  dim='laydim')
    dz.name=z_dz
    if update:
        tran[z_dz]=dz
    return dz

def depth_int(tran,v):
    # integrate the variable named by v in the vertical
    if isinstance(v,six.string_types):
        v=tran[v]
    z_dz=infer_z_dz(tran)
    return (v*z_dz).sum(dim=vert_dim(tran))

def depth_avg(tran,v):
    # average the variable named by v in the vertical
    integrated=depth_int(tran,v)
    depth = depth_int(tran,1)
    return integrated/depth

def Qleft(tran):
    quv=depth_int(tran,'U')

    # unit normals left of transect:
    left_normal=linestring_utils.left_normals(np.c_[tran.x_sample,tran.y_sample])
    # flow per-width normal to transect:
    qnorm=np.sum( (quv.values * left_normal),axis=1 )

    Q=np.sum( tran.dx_sample.values * qnorm )
    return Q

def add_rozovski_angles(tran,src,name='roz_angle'):
    quv=depth_int(tran,src)

    # direction of flow, as mathematical angle (radians
    # left of east)
    roz_angle=np.arctan2( quv.values[...,1],
                          quv.values[...,0])
    tran[name]=('sample',),roz_angle

def add_rozovski(tran,src='U',dst='Uroz',frame='roz',comp_names=['downstream','left']):
    add_rozovski_angles(tran,src)

    vec_norm=xr.concat( [np.cos(tran.roz_angle),
                         np.sin(tran.roz_angle)],
                        dim=frame).transpose('sample',frame)

    tran['roz_vec_norm']=vec_norm

    R = np.array( [[np.cos(tran.roz_angle),-np.sin(tran.roz_angle)],
                   [np.sin(tran.roz_angle),np.cos(tran.roz_angle)]] ).transpose(2,0,1)
    src_frame=tran[src].dims[-1]
    transform=src_frame + '_to_' + frame
    tran[transform]=('sample',src_frame,frame),R

    # xarray dot() doesn't collapse dimensions correctly
    # Do it by hand, and in this order so that roz is at the end of the dimensions.
    tran[dst]=(tran[src]*tran[transform]).sum(dim=src_frame)


#--- Functions related to reading section_hydro.txt data

def section_hydro_parse(filename):
    """
    The text processing part of reading section_hydro_txt
    """
    all_sections=[]

    with open(filename,'rt') as fp:
        line=fp.readline().strip()

        while 1: # Loop over transects
            if not line:
                break

            # "section 8B"
            assert line.startswith("section ")
            section_name=line.split(' ')[1]
            line=fp.readline().strip()

            water_columns=[]
            while 1: # Loop over locations within one transect:
                if not line:
                    break

                if line.startswith("section "):
                    break
                # Parse this -- water column data
                # "0.000000 647104.061850 4185777.338411 380 25 25 -9.420000 9.420000"
                dist_m,x,y,idx,k_bed,k_surf,z_bed,z_surf = [float(s)
                                                            for s in line.split()]
                k_bed=int(k_bed) ; k_surf=int(k_surf)

                bins=[]

                for k in range(k_bed,k_surf+1):
                    # each bin "20 -0.108014 0.015937 0.500000"
                    items=fp.readline().strip().split()

                    k=int(items[0])
                    u,v,dz=[float(s) for s in items[1:]]
                    bins.append( dict(k=k,u=u,v=v,dz=dz) )
                water_column=dict(dist_m=dist_m,x=x,y=y,idx=idx,k_bed=k_bed,k_surf=k_surf,
                                  z_bed=z_bed,z_surf=z_surf,bins=bins)
                water_columns.append(water_column)

                line=fp.readline().strip()

            # Close out this section
            all_sections.append( (section_name,water_columns) )

    return all_sections

def section_hydro_parsed_to_transect(section,filename):
    """
    Convert python datastructures in section to a transect Dataset
    following conventions in this module.

    This most basic conversion leaves the layer indices intact, creating
    larger arrays with more nans.
    """
    ds=xr.Dataset()
    ds.attrs['name']=section[0]
    ds.attrs['source']="%s:%s"%(filename,section[0])
    ds.attrs['filename']=filename

    profiles=section[1]

    wet_profiles=[ prof for prof in profiles if prof['k_bed']<prof['k_surf'] ]
    k_min=min( [p['k_bed'] for p in wet_profiles] )
    k_max=max( [p['k_surf'] for p in wet_profiles] )

    ds['z_surf']=('sample',), np.array([p['z_surf'] for p in wet_profiles]) # z_surf is positive up
    ds['z_bed'] =('sample',), np.array([-p['z_bed'] for p in wet_profiles]) # p['z_bed'] is positive down

    z_min=ds.z_bed.values.min()
    z_max=ds.z_surf.values.max()

    ds['sample']=('sample',), np.arange(len(wet_profiles))

    ds['layer']=('layer',),np.arange(0,k_max+1)
    ds['interface']=('interface',),np.arange(0,k_max+2)
    ds['k']=ds.layer

    empty=lambda:np.nan*np.ones( (len(ds['sample']),len(ds['layer'])), np.float64)
    Ve=empty()
    Vn=empty()
    Vu=empty()
    z_ctr=empty()
    dz=empty()
    z_int=np.nan*np.ones( (len(ds['sample']),len(ds['interface'])), np.float64)

    for sample_i,p in enumerate(wet_profiles):
        # represent discrete cells directly
        # thickness of each layer in the model output

        # +1 here to go from inclusive to exclusive end index.
        k_slice=slice(p['k_bed'],p['k_surf']+1)
        # Same, but for interfaces
        kint_slice=slice(p['k_bed'],p['k_surf']+2)
        dzs=[ b['dz'] for b in p['bins'] ]

        dz[sample_i,k_slice]=dzs

        # Add a zero thickness at the top to help with going from bins to
        # interfaces n'th item is the elevation of the bottom of the n'th
        # layer, relative to the surface, with an extra item for the top of the
        # top layer.
        interface_elevations=(-p['z_bed']) + np.cumsum(np.array([0] + dzs))
        z_int[sample_i,kint_slice]=interface_elevations
        z_ctr[sample_i,k_slice]=interface_elevations[:-1] + 0.5*np.array(dzs)

        # should be within text tolerances.
        assert np.abs(interface_elevations[-1] - p['z_surf']) < 0.01,"Layers did not add up"

        Ve[sample_i,k_slice]=[b['u'] for b in p['bins']]
        Vn[sample_i,k_slice]=[b['v'] for b in p['bins']]
        Vu[sample_i,k_slice] = 0 # not reported

    ds['Ve']=('sample','layer'), Ve
    ds['Vn']=('sample','layer'), Vn
    ds['Vu']=('sample','layer'), Vu
    ds['z_dz']=('sample','layer'), dz
    ds['z_int']=('sample','interface'),z_int
    ds['z_ctr']=('sample','layer'),z_ctr

    xy=np.array([ [p['x'],p['y']] for p in wet_profiles])
    ds['x_sample']=('sample',),xy[:,0]
    ds['y_sample']=('sample',),xy[:,1]

    ds['d_sample']=('sample',),utils.dist_along(xy)

    # Out of place -- should be refactored
    # ll=proj_utils.mapper('EPSG:26910','WGS84')(xy)
    # ds['lon']=('sample',),ll[:,0]
    # ds['lat']=('sample',),ll[:,1]
    return ds

def section_hydro_to_transects(filename):
    """
    Parse section_hydro.txt (output from extract_velocity_section)
    to a transect following the conventions
    in this module.

    A single section_hydro.txt can contain an arbitrary number of
    transects -- this will read them all, and return a list datasets.
    The name of each transect is put in a name attribute on the respective
    datasets
    """
    all_sections=section_hydro_parse(filename)
    all_ds=[]

    for section in all_sections:
        ds=section_hydro_parsed_to_transect(section,filename)
        all_ds.append(ds)
    return all_ds

def section_hydro_names(filename):
    all_sections=section_hydro_parse(filename)
    return [sec[0] for sec in all_sections]

def section_hydro_to_transect(filename,name=None,index=None):
    all_sections=section_hydro_parse(filename)

    if index is not None:
        if index<len(all_sections):
            section=all_sections[index]
        else:
            return None
    elif name is not None:
        for section in all_sections:
            if name is not None and section[0]==name:
                break
        else:
            return None

    return section_hydro_parsed_to_transect(section,filename)

def resample_z(tran,new_z):
    """
    Resample z coordinate to the given vector new_z [N].
    """
    # Resample to evenly spaced vertical axis.  Will be shifting to be positive up, relative to
    # water surface, below.

    ds=xr.Dataset()

    z_dim='layer'

    ds['sample']=tran['sample']
    ds['z_ctr']=(z_dim,),new_z
    ds['z_dz']=(z_dim,),utils.center_to_interval(new_z)

    for v in tran.data_vars:
        var=tran[v]
        if z_dim not in var.dims:
            ds[v]=var
            continue
        elif v in ['z_ctr','z_dz']:
            continue # gets overwritten
        dims=var.dims # dimension names don't change
        shape=[ len(ds[d]) for d in dims]

        # For the moment, can assume that there are two dimensions,
        # and the first is sample.
        new_val=np.nan*np.ones( shape, np.float64 )

        # iter_shape=[ len(tran[d]) for d in dims
        iter_shape=var.shape

        z_num=list(dims).index(z_dim)

        if len(dims)==1:
            # print("Not sure how to resample %s"%v)
            continue
        # Not quite there -- this isn't smart enough to get the interfaces
        _,src_z,src_dz = xr.broadcast(var,tran['z_ctr'],tran['z_dz'])
        for index in np.ndindex( *iter_shape ):
            if index[z_num]>0:
                continue
            index=list(index)
            index[z_num]=slice(None)
            my_src_z=src_z.values[index]
            my_src_dz=src_dz.values[index]
            my_src=var.values[index]
            my_src_bottom=my_src_z-0.5*my_src_dz
            my_src_top=my_src_z+0.5*my_src_dz

            src_valid=np.isfinite(my_src_z+my_src)
            Nsrc_valid=src_valid.sum()

            my_src_ints=np.concatenate( (my_src_bottom[src_valid],
                                         my_src_top[src_valid][-1:]) )

            bins=np.searchsorted(my_src_ints,new_z)
            bad=(bins<1)|(bins>Nsrc_valid) # not sure about that top clip
            # Make these safe, and 0-referenced
            bins=bins.clip(1,Nsrc_valid)-1
            # But record which ones are not valid
            new_val[index]=np.where( bad,np.nan,var.values[index][src_valid][bins] )

        ds[v]=dims,new_val

    return ds


def lplt():
    """ lazy load plotting library """
    import matplotlib.pyplot as plt
    return plt

def plot_scalar(tran,v,ax=None,**kw):
    plt=lplt()
    from stompy.plot import plot_utils

    ax=ax or plt.gca()

    if isinstance(v,six.string_types):
        v=tran[v]

    x,y,scal,dz=xr.broadcast(tran.d_sample,tran.z_ctr,v,tran.z_dz)

    # important to use .values, as xarray will otherwise muck with
    # the indexing
    # coll_u=plot_utils.pad_pcolormesh(x.values,y.values,scal.values,ax=ax)
    # But we have some additional information on how to pad Y, so do that
    # here.

    # Move to numpy land
    X=x.values
    Y=y.values
    Dz=dz.values

    # Expands the vertical coordinate in the vertical
    Ybot=Y-0.5*Dz
    Yexpand=np.concatenate( (Ybot,Ybot[:,-1:]), axis=1)
    Yexpand[:,-1]=np.nan
    Yexpand[:,1:]=np.where( np.isfinite(Yexpand[:,1:]),
                            Yexpand[:,1:],
                            Y+0.5*Dz)
    # Expands the horizontal coordinate in the vertical
    Xexpand=np.concatenate( (X,X[:,-1:]), axis=1)

    # And expand in the horizontal
    def safe_midpnt(a,b):
        ab=0.5*(a+b)
        invalid=np.isnan(ab)
        ab[invalid]=a[invalid]
        invalid=np.isnan(ab)
        ab[invalid]=b[invalid]
        return ab

    dx=utils.center_to_interval(X[:,0])
    Xexpand2=np.concatenate( (Xexpand-0.5*dx[:,None], Xexpand[-1:,:]+0.5*dx[-1:,None]), axis=0)
    Yexpand2=np.concatenate( (Yexpand[:1,:],
                              safe_midpnt(Yexpand[:-1],Yexpand[1:]),
                              Yexpand[-1:,:]), axis=0)

    coll=ax.pcolor(Xexpand2,Yexpand2,scal.values,**kw)

    return coll

