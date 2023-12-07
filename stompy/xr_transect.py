"""
Methods for representing and manipulating transects
in xarray datasets.

Transects being a dataset with the form
 x(s), y(s), z(s,k) or z(k)
 u(s,k), v(s,k), etc.
where x(s),y(s) describes a polyline.

names are somewhat standardized, but can generally be provided
U: vector valued velocity in geographic coordinate frame
 the component dimension is last by convention, and its name
 describes the coordinate system (xy, roz, tran)
V<n>: component of a velocity (typ e, n, u)
<N>_avg: depth averaged value

Remaining design decisions:
1. Make it more standardized between one variable per component, or
   vector-valued variables.  Vector-valued variables are cleaner
   mathematically, though it seems CF tends towards one variable
   per-component.
2. Standardize how intervals and bounds are represented.  Having bounds
   makes indexing more annoying, but keeps the dimension names
   consistent.

"""
import six
import numpy as np
import xarray as xr
from .spatial import linestring_utils
from . import utils

def vert_dim(tran):
    # placeholder - need a more robust way of inferring which dimension
    # is vertical
    return 'layer'

def get_z_dz(tran):
    """
    Populate tran.z_dz with the thickness of each layer.
    This is a signed quantity, dependent on the ordering
    z_ctr, i.e. same sign as np.diff(z_ctr)
    """
    if 'z_dz' in tran:
        return tran.z_dz

    v_dim=vert_dim(tran)

    if 'z_int' in tran:
        int_dim='interface'
        # get numpy values just for consistency with the else clause
        # dz=tran.z_int.diff(dim=int_dim).rename({int_dim:v_dim})
        dz=tran.z_int.diff(dim=int_dim).values
        dz[dz==0]=np.nan # not sure this the right place to do this..
    else:
        # center to center spacing:
        axis=tran.z_ctr.get_axis_num(v_dim)

        #dctr=tran[z_ctr].diff(dim=v_dim)
        dctr=np.diff(tran.z_ctr.values,axis=axis)

        def b(i):
            dims=[]
            for dim in range(tran.z_ctr.ndim):
                if dim==axis:
                    dims.append(i)
                else:
                    dims.append(slice(None))
            return tuple(dims)

        # This drops a layer top&bottom in z-layers
        # have to use .values to keep xr from trying to line up
        # indices.
        dz_middle=0.5*(dctr[b(slice(None,-1))]
                       +
                       dctr[b(slice(1,None))] )

        dz=np.concatenate( [ dz_middle[b(slice(None,1))],
                             dz_middle,
                             dz_middle[b(slice(-1,None))] ],
                           axis=axis)
        # Tricky - the copying above is fine when the valid values go all the way
        # to index 0 and N of dz.  When the ends are nan, then we have to copy
        # internally.  This is a gross way of doing that without an explicit loop,
        # essentially saying dz[:-1]=dz[1:], where dz[:-1] is nan, and dealing with
        # arbitrary dimensions.  ick.
        dz[b(slice(None,-1))] = np.where( np.isnan(dz[b(slice(None,-1))]),
                                          dz[b(slice(1,None))],
                                          dz[b(slice(None,-1))])
        dz[b(slice(1,None))]  = np.where( np.isnan(dz[b(slice(1,None))]),
                                          dz[b(slice(None,-1))],
                                          dz[b(slice(1,None))])

    tran['z_dz']=tran.z_ctr.dims,dz

    signs=np.sign(dz)
    # allow for 0 and nan, just not 1 and -1
    if np.nanmin(signs)<0 and np.nanmax(signs)>0:
        assert False,"Problem with mixed signs of dz"

    return tran['z_dz']

def get_d_sample(tran):
    if 'd_sample' not in tran:
        xy=np.c_[ tran.x_sample.values,
                  tran.y_sample.values ]
        tran['d_sample']=('sample',),utils.dist_along(xy)

    return tran['d_sample']

def get_dx_sample(tran):
    if 'dx_sample' not in tran:
        tran['dx_sample']=('sample',),utils.center_to_interval( get_d_sample(tran).values )
    return tran['dx_sample']

def depth_int(tran,v):
    """
    integrate the variable named by v in the vertical
    note that regardless of the order or sign of the vertical
    coordinate, this will use non-negative values of dz.
    """
    if isinstance(v,six.string_types):
        v=tran[v]
    z_dz=get_z_dz(tran)
    # adjust for get_z_dz returning negative dz (i.e. positive
    # z coordinate, but ordered surface to bed, or a positive
    # down coordinate, ordered bed to surface
    sign=np.sign( np.nanmean(z_dz) )

    return sign*(v*z_dz).sum(dim=vert_dim(tran))

def depth_avg(tran,v):
    """
    average the variable named by v in the vertical.
    """
    if isinstance(v,six.string_types):
        v=tran[v]
    
    integrated=depth_int(tran,v)
    # tempting to define depth as depth_int(tran,1), but
    # if, for example, z_dz is horizontally constant and
    # valid elements are denoted just by finite tran[v], that
    # would fail.
    depth = depth_int(tran,1*np.isfinite(v))
    return integrated/depth

def d_dz(tran,fld):
    """
    Calculate depth-averaged vertical gradient of a tran[fld].
    Plenty of room for improvement -- this approach fits a line
    to the distribution of fld in each water column.
    zero-thickness layers are ignored, but there no weighting of
    thicker layers versus thinner layers.  TODO.
    """
    gradients=[]
    scal,z,dz=xr.broadcast(tran[fld], tran.z_ctr, tran.z_dz)

    x_dim=[d for d in scal.dims if d!='layer'][0]
    
    for i in range(tran.dims[x_dim]):
        col_sel={x_dim:i}
        col_scal=scal.isel(col_sel)
        col_z=z.isel(col_sel)
        col_dz=dz.isel(col_sel)
        valid=np.isfinite(col_scal) & (col_dz != 0.0)
        if valid.sum()==0:
            gradients.append(np.nan)
        elif valid.sum()==1:
            gradients.append(0) # arguably also nan
        else:
            mb=np.polyfit(col_z[valid],col_scal[valid],1)
            gradients.append(mb[0])
            
    grad=xr.DataArray(np.array(gradients),dims=(x_dim))
    return grad
 
def lateral_int(tran,v):
    """
    tran: xarray dataset transect
    v: DataArray or name of variable in tran
    integrates v laterally, using dx_sample
    """
    if isinstance(v,six.string_types):
        v=tran[v]
    dx=get_dx_sample(tran)
    return (v*dx).sum(dim='sample')

def total_int(tran,v):
    return lateral_int(tran,depth_int(tran,v))

def total_avg(tran,v):
    return total_int(tran,v) / total_int(tran,1.0)

def left_normal(tran):
    # unit normals left of transect:
    left_normal=linestring_utils.left_normals(np.c_[tran.x_sample,tran.y_sample])
    # in some cases there are repeated points in x,y, leading to nan here.
    utils.fill_invalid(left_normal[:,0])
    utils.fill_invalid(left_normal[:,1])
    # re-normalize magnitudes
    mags=utils.mag(left_normal).clip(1e-5,np.inf)
    left_normal /= mags[:,None]
    return left_normal

def Qleft(tran):
    """
    Calculate flow with positive meaning towards the 'left', i.e.
    if looking from the start to the end of the transect.
    """
    quv=depth_int(tran,'U')
    ln=left_normal(tran)
    # flow per-width normal to transect:
    qnorm=np.sum( (quv.values * ln),axis=1 )

    Q=np.sum( get_dx_sample(tran).values * qnorm )
    return Q

def add_rozovski_angles(tran,src,name='roz_angle',force_left=False):
    """
    Calculate per-water column mean flow direction a al Rozovski.
    tran: transect Dataset
    src: vector-valued velocity variable
    name: field to save the angle to.
    force_left: if true, angles which would put a unit vector in the 
    opposite direction of right-to-left flow are flipped.  Useful if
    a transect has some eddying or recirculation, and it's necessary
    to retain the net 'downstream' sense of the angles.
    """
    quv=depth_int(tran,src)
    
    # direction of flow, as mathematical angle (radians
    # left of east)
    roz_angle=np.arctan2( quv.values[...,1],
                          quv.values[...,0])
    if force_left:
        ln=left_normal(tran)
        qnorm=np.sum( quv.values*ln, axis=1)
        roz_angle[qnorm<0] += np.pi
    
    tran[name]=('sample',),roz_angle

def add_rozovski(tran,src='U',dst='Uroz',frame='roz',comp_names=['downstream','left'],
                 force_left=False):
    add_rozovski_angles(tran,src,force_left=force_left)
    add_rotated(tran,src=src,dst=dst,frame=frame,comp_names=comp_names,
                angle_field='roz_angle')

def add_normal_tangential(tran,src='U',dst='Utrn',frame='trn',comp_names=['downstream','left']):
    """
    Add velocity components normal and tangential to the section. Downstream/normal
    component assumes the section is oriented left to right while looking downstream. 
    Tangential component is positive to the left (to maintain right-handed coordinate
    system).
    """
    xy=np.c_[ tran.x_sample.values, tran.y_sample.values]
    dxy=np.vstack( [ xy[1]-xy[0], xy[2:]-xy[:-2], xy[-1]-xy[-2]] )
    angle_fld=frame+'_angle'
    tran[angle_fld]=('sample',),np.arctan2(dxy[:,0],-dxy[:,1]) # includes rotation to get normal
    add_rotated(tran,src=src,dst=dst,frame=frame,angle_field=angle_fld)

def add_rotated(tran,src='U',dst='Uroz',frame='roz',comp_names=['downstream','left'],
                angle_field='roz_angle'):
    """
    Rotate the variable 'src' by the angle 'angle_field', putting the result into
    'dst', and naming the new coordinate dimension 'frame', with component labels
    'comp_names'.
    Defaults are suitable for Rozovski rotation.
    Modifies tran in place, with the new velocities in a new variable.
    """
    angle=tran[angle_field]
    vec_norm=xr.concat( [np.cos(angle),
                         np.sin(angle)],
                        dim=frame).transpose('sample',frame)

    tran[frame+'_vec_norm']=('sample','xy'),vec_norm.values

    R = np.array( [[np.cos(angle),-np.sin(angle)],
                   [np.sin(angle),np.cos(angle)]] ).transpose(2,0,1)
    src_frame=tran[src].dims[-1]
    transform=src_frame + '_to_' + frame
    tran[transform]=('sample',src_frame,frame),R

    tran[frame]=(frame,),comp_names
    # xarray dot() doesn't collapse dimensions correctly
    # Do it by hand, and in this order so that roz is at the end of the dimensions.
    tran[dst]=(tran[src]*tran[transform]).sum(dim=src_frame)
    # xarray sum doesn't preserve the nans, so replace them here:
    tran[dst].values[np.isnan(tran[src].values)]=np.nan


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
    ds['U']=('sample','layer','xy'), np.concatenate( (Ve[:,:,None], Vn[:,:,None]), axis=2)

    ds['z_dz']=('sample','layer'), dz
    ds['z_int']=('sample','interface'),z_int
    ds['z_ctr']=('sample','layer'),z_ctr

    xy=np.array([ [p['x'],p['y']] for p in wet_profiles])
    ds['x_sample']=('sample',),xy[:,0]
    ds['y_sample']=('sample',),xy[:,1]

    # pre-calculate derived fields
    get_d_sample(ds) # ['d_sample']=('sample',),utils.dist_along(xy)
    get_dx_sample(ds)

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

def resample_z(tran,new_z,save_original=None,new_z_positive='same'):
    """
    Resample z coordinate to the given vector new_z [N].
    
    new_z_positive: dictates sign of new_z.
      'same': it's the same as tran.z_ctr
      'up': positive up
      'down': positive down

    the current handling for order of z is not good.

    new_z is taken to be in the target sign convention.
    """
    # had been a comment about resampling to positive up, but the code
    # doesn't actually do that.  instead, assumes that new_z is the
    # same reference and sign as tran.z_ctr
    ds=xr.Dataset()

    z_dim='layer'

    z_ctr_pos=tran.z_ctr.attrs.get('positive','up')
    if new_z_positive=='same':
        new_z_positive=z_ctr_pos
        
    if new_z_positive!=z_ctr_pos:
        z_flip=-1
    else:
        z_flip= 1

    # print("Existing z_ctr_pos %s  new z pos %s z_flip %s"%(z_ctr_pos,new_z_positive,z_flip))

    ds['sample']=tran['sample']
    ds['z_ctr']=(z_dim,),new_z
    ds['z_dz']=(z_dim,),utils.center_to_interval(new_z)

    ds['z_ctr'].attrs.update(tran.z_ctr.attrs)
    ds['z_ctr'].attrs['positive']=new_z_positive

    for v in tran.data_vars:
        var=tran[v]
        if z_dim not in var.dims:
            ds[v]=var
            continue
        elif v in ['z_ctr','z_dz']:
            if save_original is not None:
                # This is where we could be doing something related to
                # saving the original vertical values.  not that interesting
                # for vertical, so for now just pass
                pass
            continue # gets overwritten

        dims=var.dims # dimension names don't change
        # any new dimensions we need to copy?
        for d in dims:
            if (d not in ds) and (d not in [z_dim]):
                # unclear how to deal with things like wdim.  For now
                # it will get copied, but then it will not be valid.
                ds[d]=tran[d]
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
        _,src_z,src_dz = xr.broadcast(var,tran['z_ctr'],get_z_dz(tran))

        # sgn is used to get the src data into increasing z coordinate
        # so if the order is top-to-bottom, sgn*src_z is positive-down
        #    if order is bottom-to-top, sgn*src_z is positive-up
        all_sgns=np.sign(src_dz).values.ravel()
        # some of these may be nan - just look past those
        all_sgns=all_sgns[ np.isfinite(all_sgns) ]
        if all_sgns.max()>0:
            sgn=1
        elif all_sgns.min()<0:
            sgn=-1
        else:
            raise Exception("All signs are 0?")
        assert np.all( sgn*all_sgns>=0 )

        for index in np.ndindex( *iter_shape ):
            if index[z_num]>0:
                continue
            index=list(index)
            index[z_num]=slice(None)
            index=tuple(index)
            my_src_z=src_z.values[index]
            my_src_dz=src_dz.values[index] # this has a nan
            my_src=var.values[index]
            my_src_bottom=my_src_z-0.5*my_src_dz # for each layer
            my_src_top=my_src_z+0.5*my_src_dz

            # all-nan columns are not valid below because 'bad' bins are still
            # accessed.  Could rewrite to avoid the partial indexing by src_valid
            # early on.  in theory we should always have valid z if we have valid
            # var, but in the case of sontek files, we can have valid SNR, but for
            # bins below the bed or otherwise invalid, and z will b reported as nan.
            if np.all(np.isnan(my_src + my_src_z)):
                new_val[index]=np.nan
                continue
            src_valid=np.isfinite(my_src_z+my_src)
            Nsrc_valid=src_valid.sum()

            # interfaces in src z coordinate
            my_src_ints=np.concatenate( (my_src_bottom[src_valid],
                                         my_src_top[src_valid][-1:]) )

            # use sgn to make sure this is increasing
            # and new_z is flipped maybe to match flipping of src, and
            # maybe again to get the desired output sign.
            bins=np.searchsorted(sgn*my_src_ints,z_flip*sgn*new_z)
            bad=(bins<1)|(bins>Nsrc_valid) # not sure about that top clip
            # Make these safe, and 0-referenced
            bins=bins.clip(1,Nsrc_valid)-1
            # But record which ones are not valid
            new_val[index]=np.where( bad,np.nan,var.values[index][src_valid][bins] )

        ds[v]=dims,new_val
        ds[v].attrs.update(var.attrs)

    ds.attrs.update(tran.attrs)
    return ds

def resample_d(tran,new_xy,save_original=None):
    """
    tran: xr_transect style Dataset
    new_xy: [N,2] points for resampling horizontal transect dimension
    save_original: if not None, a string prefix for saving original xy information
    """
    # need a function which takes per-sample data from ds_in,
    # returns per-sample data at new_xy
    old_xy=np.c_[tran.x_sample, tran.y_sample]

    # per point in new_xy, an array of averaging weights to apply across samples
    weights=np.zeros( (len(new_xy),len(old_xy)), np.float64)
    # for categorical data, just have to choose one input element
    # -1 => no match
    selectors=np.zeros(len(new_xy), np.int32)-1

    # rather than iterate over all new points, check first to see if some
    # new points fall off the end of the input, and should be left nan.
    dists0=utils.dist(new_xy,old_xy[0,:])
    distsN=utils.dist(new_xy,old_xy[-1,:])
    new_start=np.argmin(dists0)
    new_stop =np.argmin(distsN)
    if new_start>new_stop:
        # print("Resampling: flip transect to match order of new points")
        new_start,new_stop = new_stop,new_start

    for row in range(new_start,new_stop+1):
        pnt=new_xy[row]
        # start with simple -- choose nearest point in input
        dists=utils.dist(pnt,old_xy)
        best=np.argmin(dists)
        weights[row,best]=1.0
        selectors[row]=best

    ds=xr.Dataset()
    ds['sample']=('sample',),np.arange(len(new_xy))
    sample_dim='sample'

    ds['x_sample']=('sample',),new_xy[:,0]
    ds['y_sample']=('sample',),new_xy[:,1]
    ds['d_sample']=('sample',),utils.dist_along(new_xy)

    for v in tran.data_vars:
        var=tran[v]
        v_dest=v # name of the variable to write
        if sample_dim not in var.dims:
            ds[v]=var
            continue
        elif v in ['x_sample','y_sample','d_sample','dx_sample']:
            if save_original is None:
                continue # manually supplied above
            else:
                v_dest=save_original+v
                # print("Beware - trying to save original location info to %s"%v_dest)

        dims=var.dims # dimension names don't change
        # any new dimensions we need to copy?
        for d in dims:
            if (d not in ds) and (d not in [sample_dim]):
                # unclear how to deal with things like wdim.  For now
                # it will get copied, but then it will not be valid.
                ds[d]=tran[d]
        shape=[ len(ds[d]) for d in dims]

        new_val=np.zeros( shape, var.dtype )
        if np.issubdtype(var.dtype,np.floating):
            new_val[...]=np.nan
        sample_num=list(dims).index(sample_dim)

        # handling below is simplified by casting dates to floats.
        if np.issubdtype(var.dtype,np.datetime64):
            cast=utils.to_dnum
            uncast=utils.to_dt64
        else:
            def cast(x): return x
            def uncast(x): return x

        for index in np.ndindex( *var.shape ):
            if index[sample_num]>0:
                continue # really only iterating over the non-sample dimensions
            # replace sample index with slice(None)
            index=list(index)
            index[sample_num]=slice(None)

            my_src=cast(var.values[tuple(index)])
            if not np.issubdtype(my_src.dtype,np.floating):
                print("Variable %s will be treated as a category"%var.name)
                my_dst=my_src[selectors]
                my_dst[selectors<0]=None
            else:
                valid=np.isfinite(my_src)

                my_dst=(my_src[valid]*weights[:,valid]).sum(axis=1)
                weight_sum=weights[:,valid].sum(axis=1)

                bad=(weight_sum==0)
                my_dst[bad]=np.nan
                my_dst[~bad]=my_dst[~bad]/weight_sum[~bad]

            new_val[tuple(index)]=my_dst

        ds[v_dest]=dims,uncast(new_val)

    return ds


def extrapolate_vertical(tran,var_methods,eta=0,z_bed='z_bed',save_original=False):
    """
    Extrapolate each water column in the vertical to span the
    full bed-to-surface range.

    eta: positive-up scalar for elevation of freesurface

    z_bed: name of variable holding a per-sample bed elevation in the same
     coordinate system as z_ctr.

    var_methods: awkward, but something like
    [ ('Uroz',dict(roz=0),'linear','constant'),
      ('Uroz',dict(roz=1),'linear','constant') ]
    where each tuple describe a variable and how it is extrapolated.
    'Uroz': variable
    dict(roz=0): sub-component of variable, or None to ignore.
    'linear': how to extrapolate between the bed and the first valid data point.
       in this case, linearly ramp from 0 at the bed up to the first value.
    'constant': how to extrapolate between the top valid data point and the
       free surface.
    Alternative:
      'pow(0.167)': fit alpha in u~ alpha * z.a.b.^0.167

    will resample in the vertical to make sure the full range of elevations is
    in z_ctr.

    save_original: a copy of each variable will be made with a _nofill suffix,
     after resampling but before filling.

    returns a new dataset
    """
    z_sgn=1
    if tran.z_ctr.attrs.get('positive','up')=='down':
        z_sgn=-1

    if np.median(np.diff(tran.z_ctr.values*z_sgn))<0:
        order='top_down'
        to_bottom_up=slice(None,None,-1)
    else:
        order='bottom_up'
        to_bottom_up=slice(None,None,1)

    # does not (yet) pad z up to eta.
    z_bed=z_sgn*tran['z_bed'] # assumed same coordinates as z_ctr, and a value per sample

    # Resample z -- otherwise we'd have to check each column to see if it was deep enough.
    z_max=np.max(eta)
    z_min=np.min(z_bed)
    new_z=np.linspace(z_min,z_max,tran.dims['layer'])
    # but keep the same sign/order as before:
    new_z=z_sgn*new_z[to_bottom_up]
    ds=resample_z(tran,new_z)

    for data_var,isel_kw,bed_mode,surface_mode in var_methods:
        data=ds[data_var]

        if save_original:
            save_var=data_var+"_nofill"
            if save_var not in ds:
                ds[save_var]=ds[data_var].copy(deep=True)
                
        if isel_kw:
            data=data.isel(**isel_kw)

        u,z=xr.broadcast( data, # ds['Uroz'].sel(roz='downstream'),
                          ds['z_ctr'] )

        for s in range(ds.dims['sample']):
            # treat each individually
            # get the data and elevations in a common order and sign:
            # bottom-up, positive=up
            eta_col=eta # could be extended to pull from spatially variable eta
            z_bed_col=z_bed.values[s]
            u_col=u[s,to_bottom_up]
            z_col=z_sgn*z[s,to_bottom_up]
            u_valid=np.nonzero(np.isfinite(u_col.values))[0]
            if len(u_valid)==0:
                continue # empty water column

            top_valid_idx=u_valid[-1]
            bottom_valid_idx=u_valid[0]
            eta_idx=np.searchsorted(z_col,eta_col)
            bed_idx=np.searchsorted(z_col,z_bed_col)

            # remove spurious data outside depth range
            u_col[eta_idx:]=np.nan
            u_col[:bed_idx]=np.nan

            def powfit(mode):
                beta=float(mode[4:-1])
                # Just fit the power curve once
                zab=(z_col-z_bed_col).clip(1e-6)
                # least squares solution:
                alpha=( np.sum( (zab**beta*u_col.values)[u_valid] )
                        /
                        np.sum( (zab**(2*beta))[u_valid] ) )
                return alpha*zab**beta
                
            # Surface:
            if surface_mode=='constant':
                u_col[top_valid_idx+1:eta_idx] = u_col[top_valid_idx]
            elif surface_mode.startswith('pow('):
                u_pow=powfit(surface_mode)
                u_col[top_valid_idx+1:eta_idx]=u_pow[top_valid_idx+1:eta_idx]
            else:
                raise Exception("Unknown surface_mode %s"%surface_mode)

            # Bed:
            N=bottom_valid_idx - bed_idx
            if N>0:
                if bed_mode=='linear':
                    u_col[bed_idx:bottom_valid_idx]=np.linspace(0,u_col[bottom_valid_idx],N+1)[:-1]
                elif bed_mode.startswith('pow('):
                    u_pow=powfit(bed_mode)
                    u_col[bed_idx:bottom_valid_idx]=u_pow[bed_idx:bottom_valid_idx]

    ds.attrs.update(tran.attrs)
    history=ds.attrs.get('history',"")+"extrapolate_vertical"
    ds.attrs['history']=history
    return ds

def lplt():
    """ lazy load plotting library """
    import matplotlib.pyplot as plt
    return plt


def interp_to_grid(tran,v,expand_x=True,expand_y=True):
    """
    Return dense matrix for X,Y and V (from v, or tran[v] if v is str)
    expand_x: defaults to 1 more value in the X dimension than in V, suitable for
    passing to pcolormesh.
    expand_y: defaults to 1 more value in the Y dimension than in V, for pcolormesh
    """
    if isinstance(v,six.string_types):
        v=tran[v]

    x,y,scal,dz=xr.broadcast(get_d_sample(tran),tran.z_ctr,v,get_z_dz(tran))

    # important to use .values, as xarray will otherwise muck with
    # the indexing
    # coll_u=plot_utils.pad_pcolormesh(x.values,y.values,scal.values,ax=ax)
    # But we have some additional information on how to pad Y, so do that
    # here.

    # Move to numpy land
    X=x.values
    Y=y.values
    Dz=dz.values

    if expand_y:
        # Expands the vertical coordinate in the vertical
        Ybot=Y-0.5*Dz
        Yexpand=np.concatenate( (Ybot,Ybot[:,-1:]), axis=1)
        Yexpand[:,-1]=np.nan
        Yexpand[:,1:]=np.where( np.isfinite(Yexpand[:,1:]),
                                Yexpand[:,1:],
                                Y+0.5*Dz)
        # Expands the horizontal coordinate in the vertical
        Xexpand=np.concatenate( (X,X[:,-1:]), axis=1)
    else:
        Yexpand=Y
        Xexpand=X
        
    # And expand in the horizontal
    def safe_midpnt(a,b):
        ab=0.5*(a+b)
        invalid=np.isnan(ab)
        ab[invalid]=a[invalid]
        invalid=np.isnan(ab)
        ab[invalid]=b[invalid]
        return ab

    if expand_x:
        dx=utils.center_to_interval(X[:,0])
        Xexpand2=np.concatenate( (Xexpand-0.5*dx[:,None], Xexpand[-1:,:]+0.5*dx[-1:,None]), axis=0)
        Yexpand2=np.concatenate( (Yexpand[:1,:],
                                  safe_midpnt(Yexpand[:-1],Yexpand[1:]),
                                  Yexpand[-1:,:]), axis=0)
    else:
        Xexpand2=Xexpand
        Yexpand2=Yexpand
    return Xexpand2,Yexpand2,scal.values

def plot_scalar_pcolormesh(tran,v,ax=None,xform=None,**kw):
    """
    This approximates a sigma coordinate grid in the slice.
    Since that's not the natural staggering for a transect,
    it requires some extrapolation which will probably look
    bad for grids where a specific layer index can jump around
    in the vertical (e.g. variable resolution ADCP).
    """
    plt=lplt()
    from stompy.plot import plot_utils

    ax=ax or plt.gca()

    X,Y,V = interp_to_grid(tran,v)

    if xform:
        X,Y=xform(X,Y)
    coll=ax.pcolor(X,Y,V,**kw)

    return coll

def plot_scalar_polys(tran,v,ax=None,xform=None,**kw):
    """
    A more literal interpretation of how to plot a transect, with no
    interpolation of vertical coordinates
    xform: func(X,Y) applies a transformation to coordinates before plotting
    """
    plt=lplt()

    from matplotlib import collections
    from stompy.plot import plot_utils

    ax=ax or plt.gca()

    if isinstance(v,six.string_types):
        v=tran[v]

    x,y,scal,dz=xr.broadcast(get_d_sample(tran),tran.z_ctr,v,get_z_dz(tran))

    # important to use .values, as xarray will otherwise muck with
    # the indexing

    # Move to numpy land
    X=x.values
    Y=y.values
    Dz=dz.values
    
    if y.attrs.get('positive','up')=='down':
        Y=-Y
        
    # I think Dz is getting contaminated at the top/bottom
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

    dx=utils.center_to_interval(X[:,0])
    Xexpand2=np.concatenate( (Xexpand-0.5*dx[:,None], Xexpand[-1:,:]+0.5*dx[-1:,None]), axis=0)

    polys=[]
    values=[]

    V=v.values
    for samp,zi in np.ndindex( v.shape ):
        if np.isnan(V[samp,zi]):
            continue

        poly=np.array([ [Xexpand2[samp,  zi],Yexpand[samp,zi]  ],
                        [Xexpand2[samp+1,zi],Yexpand[samp,zi]  ],
                        [Xexpand2[samp+1,zi],Yexpand[samp,zi+1]],
                        [Xexpand2[samp,  zi],Yexpand[samp,zi+1]] ] )
        if xform:
            x,y=xform(poly[:,0],poly[:,1])
            poly=np.c_[x,y]
        
        polys.append(poly)
        values.append(V[samp,zi])

    values=np.array(values)

    coll=collections.PolyCollection(polys,array=values,**kw)
    ax.add_collection(coll)
    ax.axis('auto')

    return coll

plot_scalar=plot_scalar_polys

def surface_bed_linestrings(tran,v,xform=None,dzmin=0.005):
    """
    A more literal interpretation of how to plot a transect, with no
    interpolation of vertical coordinates
    xform: func(X,Y) applies a transformation to coordinates before plotting
    Should be compatible with plot_scalar_polys.
    variable must be specified,and is used to infer wet/dry
    """
    if isinstance(v,six.string_types):
        v=tran[v]

    x,y,scal,dz=xr.broadcast(get_d_sample(tran),tran.z_ctr,v,get_z_dz(tran))

    # important to use .values, as xarray will otherwise muck with
    # the indexing

    # Move to numpy land
    X=x.values
    Y=y.values
    Dz=dz.values
    V=v.values
    
    if y.attrs.get('positive','up')=='down':
        Y=-Y
        
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
    dx=utils.center_to_interval(X[:,0])
    Xexpand2=np.concatenate( (Xexpand-0.5*dx[:,None], Xexpand[-1:,:]+0.5*dx[-1:,None]), axis=0)

    coords=[] # [dist,bed_elev,surface_elev]
    #breakpoint()
    for samp in np.arange(V.shape[0]):
        # sometimes validity is just encoded in Y coordinate, sometimes
        # in scalar.
        wet = np.isfinite(Y[samp,:] * V[samp,:])
        if np.all(~wet):
            coords.append([np.nan,np.nan,np.nan])
            continue
        wet_i=np.nonzero(wet)[0]
        # add a point on the left and right.
        # May have to adjust if Dz is negative
        if np.abs(Yexpand[samp,wet_i[0]] - Yexpand[samp,wet_i[-1]+1])<dzmin:
            coords.append([np.nan,np.nan,np.nan])
        else:                              
            coords.append( [Xexpand2[samp,  wet_i[0]], Yexpand[samp,wet_i[0]], Yexpand[samp,wet_i[-1]+1]] )
            coords.append( [Xexpand2[samp+1,wet_i[0]], Yexpand[samp,wet_i[0]], Yexpand[samp,wet_i[-1]+1]] )

    coords=np.array(coords)        
    if xform:
        x,ybed=xform(coords[:,0],coords[:,1])
        x,ysurf=xform(coords[:,0],coords[:,2])
        coords[:,0]=x
        coords[:,1]=ybed
        coords[:,2]=ysurf

    return coords

def contour(tran,v,*args,**kwargs):
    return contour_like(tran,v,'contour',*args,**kwargs)

def contourf(tran,v,*args,**kwargs):
    return contour_like(tran,v,'contourf',*args,**kwargs)
    
def contour_like(tran,v,meth,*args,**kwargs):
    if isinstance(v,six.string_types):
        v=tran[v]

    x,y,scal,dz=xr.broadcast(get_d_sample(tran),tran.z_ctr,v,get_z_dz(tran))
    if 'ax' in kwargs:
        ax=kwargs.pop('ax')
    else:
        ax=lplt().gca()
    # x is always full, but y can have nan, and that triggers a warning.
    # appears to be okay to just fill with 0.
    yvals=y.values.copy()
    yvals[np.isnan(yvals)]=0.0

    if y.attrs.get('positive','up')=='down':
        yvals*=-1

    f=getattr(ax,meth)
    return f(x.values,yvals,scal.values,*args,**kwargs)

# Code related to averaging multiple transects
def transects_to_segment(trans,unweight=True,ax=None,n_nodes=2):
    """
    trans: list of transects per xr_transect
    unweight: if True, follow ADCPy and thin dense clumps of pointer.

    return a segment [ [x0,y0],[x1,y1] ] approximating the
    points
    
    n_nodes: if 2, just fit a line. if >2, fit a polyline with the given
    number of nodes. Current algorithm smooths the curves and evenly
    spaces the nodes. to capture a sharp turn you'll need a lot of nodes
    (10 or more)

    if ax is supplied, it is a matplotlib Axes into which the steps
    of this method are plotted.
    """
    from stompy.spatial import linestring_utils
    all_xy=[]
    all_dx=[]

    all_dx=[get_dx_sample(tran).values
            for tran in trans]
    median_dx=np.median(np.concatenate(all_dx))

    for tran_i,tran in enumerate(trans):
        xy=np.c_[ tran.x_sample.values,tran.y_sample.values]

        if ax:
            ax.plot(xy[:,0],xy[:,1],marker='.',label='Input %d'%tran_i)

        if unweight:
            # resample_linearring() allows for adding new points, which can be
            # problematic if the GPS jumps around, adding many new points on a
            # rogue line segment.
            # downsample makes sure that clusters are thinned, but does not add
            # new points between large jumps.
            xy=linestring_utils.downsample_linearring(xy,density=3*median_dx,
                                                      closed_ring=False)
        all_xy.append(xy)

    all_xy=np.concatenate(all_xy)
    if ax:
        ax.plot(all_xy[:,0],all_xy[:,1],'bo',label='Unweighted')

    C=np.cov(all_xy.T)
    vec=utils.to_unit(C[:,0])

    centroid=all_xy.mean(axis=0)
    dist_along=np.dot((all_xy-centroid),vec)
    dist_range=np.array( [dist_along.min(), dist_along.max()] )

    seg=centroid + dist_range[:,None]*vec
    if ax:
        ax.plot(seg[:,0],seg[:,1],'k-o',lw=5,alpha=0.5,label='Segment')
        ax.legend()
        
    if n_nodes>2:
        from stompy import filters
        xy_seq=all_xy[ np.argsort(dist_along) ]
        # hanning filter -- twitchy at the ends but otherwise decent.
        xy_lp=filters.lowpass_fir(xy_seq,len(xy_seq)//n_nodes,axis=0)
        dists=utils.dist_along(xy_lp)
        d_sampled=np.linspace(dists[0],dists[-1],n_nodes)
        x_sampled=np.interp(d_sampled,dists,xy_lp[:,0])
        y_sampled=np.interp(d_sampled,dists,xy_lp[:,1])
        seg=np.c_[x_sampled, y_sampled]
        
    return seg

def resample_to_common_z(trans,dz=None,save_original=None):
    """
    apply resample_z to a list of transects, making the respective
     vertical coordinates compatible.   may not be smart enough
     when it comes to transects with mixed sign conventions
    """

    all_dz=[ np.abs(get_z_dz(tran).values.ravel())
             for tran in trans]
    all_dz=np.concatenate( all_dz )

    if all_dz.min()<0:
        assert all_dz.max()<=0,"Looks like mixing sign conventions"
    if all_dz.max()>0:
        assert all_dz.min()>=0,"Looks like mixing sign conventions"
    if dz is None:
        # generally want to retain most of the vertical
        # resolution, but not minimum dz since there could be
        # some partial layers, near-field layers, etc.
        # even 10th percentile may be small.
        dz_finite=all_dz[np.isfinite(all_dz)]
        # some datasets have dz=0 to mark nonexistent cells
        dz_pos=dz_finite[dz_finite>0]
        dz=np.percentile(dz_pos,10)

    # Get the maximum range of valid vertical
    z_bnds=[]
    for tran in trans:
        V,z_full,z_dz = xr.broadcast(tran.Ve, tran.z_ctr, get_z_dz(tran))
        valid=np.isfinite(V.values * z_dz.values * z_full.values)
        z_valid=z_full.values[valid]
        dz2_valid=z_dz.values[valid]/2.0
        z_low=z_valid - dz2_valid
        z_high=z_valid + dz2_valid
        z_bnds.append( [z_low.min(), z_high.max()] )

    z_bnds=np.concatenate(z_bnds)
    z_min=np.nanmin(z_bnds)
    z_max=np.nanmax(z_bnds)

    # Resample each transect in the vertical:
    new_z=np.linspace(z_min,z_max,int(round((z_max-z_min)/dz)))

    ds_resamp=[resample_z(tran,new_z,save_original=save_original)
               for tran in trans]
    return ds_resamp

def resample_to_common(trans,dz=None,dx=None,resample_x=True,resample_z=True,
                       seg=None,save_original='orig_'):
    """
    trans: list of xr_transect Datasets.
    dx: length scale for horizontal resampling, defaults to median.  Pass 0
     to use seg as is.
    dz: length scale for vertical resampling.  defaults to 10th percentile.
    resample_x: can be set to false to skip horizontal resampling if all transects
     already have the same horizontal coordinates
    resample_z: can be set to false to skip vertical resampling if all transects
     already have the same vertical coordinates.
    seg: the linestring of the new transect.  defaults to fitting a line.
      can also pass an integer>=2, which will fit a polyline with the given
      number of nodes.

    save_original: if not None, a prefix for saving coordinates before resampling.
    """
    if resample_z:
        trans=resample_to_common_z(trans,dz=dz,save_original=save_original)

    if resample_x:
        if seg is None:
            seg=transects_to_segment(trans)
        elif np.isscalar(seg):
            assert seg>=2
            seg=transects_to_segment(trans,n_nodes=seg)

        if dx is None:
            # Define the target vertical and horizontal bins
            all_dx=[get_dx_sample(tran).values
                    for tran in trans]
            median_dx=np.median(np.concatenate(all_dx))
            dx=median_dx

        if dx>=0:
            # Keep this general, so that segment is allowed to have more than
            # just two vertices
            new_xy = linestring_utils.resample_linearring(seg,dx,closed_ring=False)
        else:
            new_xy = seg

        trans=[resample_d(tran,new_xy,save_original=save_original)
               for tran in trans]
    return trans

def average_transects(trans,dz=None,dx=None,resample_x=True,resample_z=True,
                      seg=None):
    """
    See resample_to_common for description of most of the parameters.

    returns a new Dataset.
    """
    # Get representative range of timestamps, before resample_to_common
    # screws up the time dimension.
    min_time=None
    max_time=None
    for ds in trans:
        if 'time' in ds:
            ds_min=ds.time.values.min()
            ds_max=ds.time.values.max()
            min_time=min( min_time or ds_min, ds_min)
            max_time=max( max_time or ds_max, ds_max)
    
    trans=resample_to_common(trans,dz=dz,dx=dx,resample_x=resample_x,resample_z=resample_z,
                             seg=seg)

    # Do the actual averaging
    combined=xr.concat(trans,dim='repeat')
    # here we could also calculate variance in the future
    ds_average=combined.mean(dim='repeat')

    # copy metadata over too
    for var in ds_average.data_vars:
        ds_average[var].attrs.update(trans[0][var].attrs)

    if min_time is not None:
        ds_average['start_time']=(),min_time
        ds_average['end_time']=(),max_time
        # easier than a proper average, and probably just as useful.
        ds_average['time']=(),min_time+(max_time-min_time)/2
    return ds_average

def calc_secondary_strength(tran,name='secondary'):
    """
    a somewhat ad-hoc measure of secondary circulation.
    in each water column, sort the velocities, and find the
    zero crossing.  in the unsorted data, take the mean of the
    velocities in the bins above the zero crossing.

    the goal is to be somewhat immune to different elevations
    of the flow reversal, and immune to noisy data.

    resulting value is a velocity.
    """
    if np.nanstd(tran.z_dz)>1e-10:
        print("Circulation strength assumes that z_dz is evenly spaced")

    circ_velocity=np.zeros(tran.dims['sample'],np.float64)
    # for positive-up z coordinates, starting from the surface going to the bed,
    # a positive velocity at the surface yields a positive strength
    flip_sgn=1

    # ADCP: positive:down, first bin is at the surface, so ultimately want
    # flip_sgn=1.
    # suntans: positive:up, first bin is at the surface. so ultimately want
    # flip_sgn=1.
    if tran.z_ctr.attrs.get('positive','up')=='down':
        flip_sgn*=-1
    # if the first bin is at the bed
    if tran.z_dz.mean()>0:
        flip_sgn*=-1

    get_z_dz(tran)
    all_u_left,z_dz= xr.broadcast(tran.Uroz.isel(roz=1), tran.z_dz)

    for samp in range(tran.dims['sample']):
        u_left=all_u_left.isel(sample=samp).values
        valid_left=np.isfinite(u_left) & (z_dz.isel(sample=samp).values!=0.0)
        u_left=u_left[valid_left]
        u_left_sort=np.sort(u_left)
        mid_idx=np.searchsorted(u_left_sort,0)
        if mid_idx>0:
            circ_velocity[samp]=flip_sgn*u_left[:mid_idx].mean()
        else:
            pass # leave as zero
    if name is not None:
        tran[name]=('sample',),circ_velocity
        tran[name].attrs['units']='m s-1'
        tran[name].attrs['description']='Average left-ward velocity in upper water column'
    return circ_velocity

def shift_vertical(tran,delta):
    """
    Add a vertical delta to all vertical coordinates.
    Regardless of the sign convention in tran, delta is interpreted
    postive up.
    """
    def shift(fld,default):
        if fld not in tran: return
        sign=tran[fld].attrs.get('positive',default)
        if sign=='up':
            fac=1
        else:
            fac=-1
        tran[fld].values[:] += fac*delta
    shift('eta','up')
    shift('z_bed','up')
    shift('z_ctr','up')
    shift('z_top','up')
    shift('z_bot','up')
    # less standard... probably don't belong
    shift('depth_m','down')
    shift('dv','down')
    shift('z_r','down')
    shift('z_w','down')
    
    
def mask_bed(tran,v,depth_fraction=0.9,z_top=0.0):
    """
    tran: transect dataset
    v: variable to mask out
    depth_fraction: bins with a z_ctr below z_top - 0.9(z_top - z_bed)
    will be set to nan.
    """
    # Velocity within 10% of the bed also deleted for sidelobe 
    # contamination.
    V,z_ctr,z_bed,z_top=xr.broadcast(v,tran.z_ctr,tran.z_bed,z_top)
    pos=tran.z_ctr.attrs.get('positive','up')
    if pos=='up': 
        z_sgn=1
    else:
        z_sgn=-1
    depth=(z_top - z_sgn*z_bed).clip(0.)
    mask=(z_sgn*z_ctr) < (z_top - 0.90*depth)
    v.values[mask]=np.nan

def pos_up(tran,vname):
    if tran[vname].attrs.get('positive','up')=='up':
        z_sgn=1
    else:
        z_sgn=-1
    return z_sgn*tran[vname]


def casts_to_transect(df_casts,f_dist,f_depth):
    """
    Convert pandas dataframe with cast data to a transect.
    f_dist: field holding the per-station distance, i.e. d_sample.
    Must be constant within a cast.
    f_depth: field holding depth/elevation.  Will end up in z_ctr.
    """
    df_casts=df_casts.set_index(f_dist)

    dss_cast=[]
    for d in df_casts.index.unique():
        # Don't set depth as index
        df_cast=df_casts.loc[d].reset_index() # .set_index(f_depth)
        ds_cast=xr.Dataset.from_dataframe(df_cast)
        ds_cast['d_sample']=(), d
        dss_cast.append(ds_cast)
    ds_casts=xr.concat(dss_cast,dim='sample')
    ds_casts=ds_casts.rename_dims({'index':'layer'}).rename_vars({f_depth:'z_ctr'})
    ds_casts=ds_casts.set_coords(['d_sample','z_ctr'])
    return ds_casts
