from __future__ import print_function 
import six
import os
import logging
log=logging.getLogger('utils')

try:
    import xlrd
except ImportError:
    log.info('xlrd unavailable')
    xlrd=None

import functools
import numpy as np
try:
    import pandas as pd
except ImportError:
    log.warning("pandas unavailable")
    pd=None

try:
    import xarray as xr
except ImportError:
    log.warning("xarray unavailable")
    xr=None

from collections import OrderedDict,Iterable
import sys
from scipy.interpolate import RectBivariateSpline,interp1d
from . import filters
import re

import datetime
import itertools
from matplotlib.dates import num2date,date2num

try:
    from shapely import geometry
except ImportError:
    log.warning("shapely unavailable")

def path(append):
    if append not in sys.path:
        sys.path.append(append)
        
def add_to(instance):
    def decorator(f):
        import types
        f = types.MethodType(f, instance, instance.__class__)
        setattr(instance, f.func_name, f)
        return f
    return decorator

class Bucket(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

def records_to_array(records):
    # Convert that to an array
    # deprecated - this one was pulled from some specific use case.
    rectype = []
    if len(records) == 0:
        recarray = np.zeros(0)
    else:        
        for k in records[0].keys():
            if k=='date':
                t=object
            elif k in ['inst','line']:
                t=np.int32
            else:
                t=np.float64
            rectype.append((k,t))

        recarray = np.zeros(len(records),dtype=rectype)
        for i,rec in enumerate(records):
            for k,v in rec.iteritems():
                recarray[i][k] = v
    return recarray

def hashes_to_array(records,float_fill=np.nan,int_fill=-99,uint_fill=0):
    # convert a list of dicts to a struct array
    # loops over all records to get the superset of keys
    rectype = []
    L=len(records)
    if L == 0:
        return np.zeros(0)
    
    def dtype_fill_value(t):
        if t in (np.dtype('f4'),np.dtype('f8')):
            return float_fill
        elif t in (np.dtype('i1'),np.dtype('i2'),np.dtype('i4'),np.dtype('i8')):
            return int_fill
        elif t in (np.dtype('c8'),np.dtype('c16')):
            return float_fill + 1j*float_fill
        else:
            return 0

    name_to_col={}
    rectype=[]
    coldata=[] #  [None]*len(records)
    for reci,rec in enumerate(records):
        for k,v in rec.iteritems():
            if (v is not None) and (k not in name_to_col):
                t=np.dtype(type(v))
                name_to_col[k]=len(rectype)

                # when v is a fixed length list, t is problematic b/c
                # we'd like to make a compound type, rather than an object
                # array

                if isinstance(v,str):
                    print("field %s is a string"%k)
                    new_coldata=['']*L
                else:
                    if isinstance(v, Iterable):
                        # handle compound type
                        # get numpy to figure out the root type
                        test_v=np.array(v)
                        t=(test_v.dtype,test_v.shape)
                        new_coldata=np.zeros(L,dtype=t)
                        new_coldata[:]=dtype_fill_value(t)
                    else:
                        # straight ahead numeric arrays
                        new_coldata=np.ones(L,dtype=t) 
                        new_coldata[:]=dtype_fill_value(t)

                coldata.append(new_coldata)
                rectype.append( (k,t) )
            if v is not None:
                coldata[name_to_col[k]][reci]=v

    for col_name,col_i in name_to_col.iteritems():
        if rectype[col_i][1]==np.dtype('S'): # now we know length...
            # print("Converting %s to array after delay"%col_name)
            coldata[col_i]=np.array(coldata[col_i])
            rectype[col_i] = (col_name,coldata[col_i].dtype)

    recarray = np.zeros(L,dtype=rectype)
    for col_name,col_i in name_to_col.iteritems():
        recarray[col_name][:] = coldata[col_i]
    return recarray


def bounds(pnts):
    """ 
    returns array [{lower,upper},pnts.shape[-1]]
    """
    lower=pnts
    upper=pnts
                   
    while lower.ndim> 1:
        lower=lower.min(axis=0)
        upper=upper.max(axis=0)
    return np.array([lower,upper])
        
def center_to_interval(c):
    """
    c: coordinates of centers,
    d: sizes of intervals, with the first/last interval
    assumed same as neighbors
    """
    d=np.ones_like(c)
    d[1:-1] = abs(0.5*(c[2:] - c[:-2]))
    d[0]=d[1] ; d[-1]=d[-2]
    return d

def center_to_edge(c,dx_single=None,axis=0):
    """
    take 'cell' center locations c, and infer boundary locations.
    first/last cells get width of the first/last inter-cell spacing.
    if there is only one sample and dx_single is specified, use that
    for width.  otherwise error.
    """
    crot=np.rollaxis(c,axis,0)
    new_shape=(crot.shape[0]+1,) + crot.shape[1:]
    d=np.ones(new_shape) # (len(c)+1)
    d[1:-1,...] = 0.5*(crot[1:,...] + crot[:-1,...])
    if len(crot)>1:
        d[0,...]=crot[0,...]-0.5*(crot[1,...]-crot[0,...])
        d[-1,...]=crot[-1,...]+0.5*(crot[-1,...]-crot[-2,...])
    elif dx_single:
        d[0,...]=crot[0,...]-0.5*dx_single
        d[1,...]=crot[0,...]+0.5*dx_single
    else:
        raise Exception("only a single data point given to center to edge with no dx_single")
    d=np.rollaxis(d,0,axis+1) # weird that it takes the +1...
    return d

def center_to_edge_2d(X,Y,dx_single=None,dy_single=None):
    if X.ndim==Y.ndim==1:
        Xpad=center_to_edge(X,dx_single=dx_single)
        Ypad=center_to_edge(Y,dx_single=dy_single)
    elif X.ndim==Y.ndim==2:
        def expand(X):
            newX=np.zeros( (X.shape[0]+1,X.shape[1]+1),'f8')
            # interior points are easy:
            newX[1:-1,1:-1] = 0.25*(X[:-1,:-1]+X[1:,:-1]+X[1:,1:]+X[:-1,1:])
            # and these are just kind of cheesy...
            newX[1:-1,0] = 2*newX[1:-1,1] - newX[1:-1,2]
            newX[1:-1,-1] = 2*newX[1:-1,-2]-newX[1:-1,-3]
            newX[0,1:-1] = 2*newX[1,1:-1]-newX[2,1:-1]
            newX[-1,1:-1] =2*newX[-2,1:-1]-newX[-3,1:-1]

            newX[0,0]=2*newX[1,1]-newX[2,2]
            newX[-1,-1]=2*newX[-2,-2]-newX[-3,-3]
            newX[0,-1]=2*newX[1,-2]-newX[2,-3]
            newX[-1,0]=2*newX[-2,1]-newX[-3,2]
            return newX
        Xpad=expand(X)
        Ypad=expand(Y.T).T
    else:
        raise Exception("Not ready for mixed 1d 2d dimensions")
    return Xpad,Ypad
    
    
class BruteKDE(object):
    def __init__(self,values,weights,bw):
        self.values=values
        self.weights=weights
        self.bw=bw
        self.norm_factor=np.sum(self.weights)*np.sqrt(np.pi)*bw
    def __call__(self,x):
        res=np.zeros_like(x)
        for idx in np.ndindex(res.shape):
            res[idx]=np.sum( self.weights*np.exp( -((self.values-x[idx])/self.bw)**2 ) )
        return res/self.norm_factor

def quantize(a,stride,axis=0,reducer=np.mean):
    # first truncate to an even multiple of stride
    N=(a.shape[axis]//stride)*stride
    slices=[ slice(None) ]*a.ndim
    slices[axis]=slice(N)
    a=a[slices]
    dims=list(a.shape)
    dims[axis:axis+1] = [N//stride,stride]
    return reducer( a.reshape(dims), axis=axis+1)

def within(item,ends,as_slice=False,fmt='auto'):
    """
    original version defaulted to bitfield, but could be forced
    to return a slice with as_slice.
    as_slice overrides fmt, and is the same as fmt='slice'
    fmt: auto, slice, mask, index
    """
    if as_slice:
        fmt='slice'
    
    if fmt=='auto':
        if all(np.diff(item) > 0): 
            fmt='slice'
        else:
            fmt='mask'

    if fmt in ('mask','index'):
        sel=(item>=ends[0])&(item<=ends[1])
        if fmt=='mask':
            return sel
        else:
            return np.nonzero(sel)[0]
    else:
        return slice(*np.searchsorted(item,ends))

def within_2d(vecs,xxyy):
    return within(vecs[:,0],xxyy[:2],fmt='mask') & within(vecs[:,1],xxyy[2:],fmt='mask')


def expand_xxyy(xxyy,factor):
    dx=xxyy[1] - xxyy[0]
    dy=xxyy[3] - xxyy[2]
    return [ xxyy[0] - dx*factor,
             xxyy[1] + dx*factor,
             xxyy[2] - dy*factor,
             xxyy[3] + dy*factor] 

def dice_interval(subinterval,overlap_fraction,start,end=None):
    """ 
    subinterval gives a duration, say 90 [s]
    overlap_fraction=0 means end of one interval is start of the next,
    overlap_fraction=0.5 means middle of one interval is start of next.
    start is either a scalar, or a pair of scalars
    if start is a scalar, end must be specified also as a scalar

    yields [substart,subend] pairs 
    """
    if end is None:
        start,end=start
        
    if subinterval is None:
        yield [start,end]
    else:
        # truncates towards zero
        # With overlap:
        Nsubs = 1 + int( (end-start-subinterval)/((1-overlap_fraction)*subinterval) )

        # find a subinterval that gives exactly this number of subwindows
        subinterval=(end-start)/((Nsubs-1)*(1-overlap_fraction)+1.0)
        advance = (1-overlap_fraction)*subinterval
        
        for n in range(Nsubs):
            p_start = start + n*advance
            yield (p_start,p_start+subinterval)


def fill_invalid(A,axis=0,ends='constant'):
    """ 
    ends:
    'constant'  missing values at the ends will take nearest valid value
    'linear' missing values will be extrapolated with a linear fit through the first/last valid values
    """
    # rotate the operational index to be first:
    new_order=(np.arange(A.ndim)+axis)%A.ndim
    revert_order=np.argsort(new_order)

    Atrans=A.transpose(new_order)
    i=np.arange(Atrans.shape[0])

    if ends is 'constant':
        kwargs={}
    else:
        kwargs=dict(left=np.nan,right=np.nan)
        
    # iterates over indices into the non-fill axes
    for idx in np.ndindex(Atrans.shape[1:]):
        Aslice=Atrans[(slice(None),)+idx]
        valid=np.isfinite(Aslice)
        if any(valid):
            Aslice[~valid]=np.interp(i[~valid],i[valid],Aslice[valid],**kwargs)
        if ends is 'linear':
            if np.isnan(Aslice[0]) or np.isnan(Aslice[-1]):
                mb = np.polyfit( i[valid][ [0,-1] ],
                                 Aslice[valid][ [0,-1] ], 1)
                missing=np.isnan(Aslice)
                Aslice[missing]=np.polyval(mb,i[missing])
            
    return Atrans.transpose(revert_order)


def select_increasing(x):
    """ 
    Return a bitmask over x removing any samples which are
    less than or equal to the largest preceding sample
    """
    mask=np.ones(len(x),np.bool8)
    last=None # -np.inf doesn't work for weird types that can't compare to float
    for i in range(len(x)):
        if last is not None and x[i] <= last:
            mask[i]=False
        else:
            last=x[i]
    return mask


# Enough screwing around with scipy interpolation -
# really just want a basic 2-D linear interpolation
# that doesn't need kid gloves around nans.
def interp_bilinear(x,y,z):
    if x[0]>x[-1]:
        x=x[::-1]
        z=z[::-1,:]
    if y[0]>y[-1]:
        y=y[::-1]
        z=z[:,::-1]
    z=z.copy()
    invalid=~np.isfinite(z)
    z[invalid]=0
    
    z_interper=RectBivariateSpline(x,y,z,kx=1,ky=1)
    invalid_interper=RectBivariateSpline(x,y,invalid,kx=1,ky=1)

    def interper(xx,yy):
        # slight change in behavior - 
        # if xx,yy are vectors, iterate over pairs
        # otherwise, assume they are as the output of meshgrid
        if xx.ndim==1:
            zs=[z_interper(xi,yi) for xi,yi in zip(xx,yy)]
            invalids=[invalid_interper(xi,yi) for xi,yi in zip(xx,yy)]
            zs=np.concatenate(zs).ravel()
            invalids=np.concatenate(invalids).ravel()
        else:
            zs=z_interper(xx,yy)
            invalids=invalid_interper(xx,yy)

        zs[invalids>0] = np.nan
        return zs
    return interper

def interp_near(x,sx,sy,max_dx=None):
    src_idx=np.searchsorted(sx,x) # gives the index for the element *after* x
    right_idx=src_idx.clip(0,len(sx)-1)
    left_idx=(src_idx-1).clip(0,len(sx)-1)

    dx=np.minimum( np.abs(sx[right_idx] - x),
                   np.abs(x - sx[left_idx]) )

    y_at_x=np.interp(x,sx,sy)
    # drop things off the end and things too spaced apart
    y_at_x[ (dx>max_dx) | (dx<0) ] = np.nan                 
    return y_at_x

def nearest(A,x,max_dx=None):
    """ 
    like searchsorted, but return the index of the nearest value,
    not just the first value greater than x.
    if max_dx is given, then return -1 for items where the nearest
    match was more than max_dx away
    """
    N=len(A)
    xi_right=np.searchsorted(A,x).clip(0,N-1) # the index of the value to the right of x
    xi_left=(xi_right-1).clip(0,N-1)
    dx_right=np.abs(x-A[xi_right])
    dx_left=np.abs(x-A[xi_left])
    
    xi=xi_right
    sel_left=dx_left < dx_right
    if xi.ndim:
        xi[sel_left] = xi_left[sel_left]
    else:
        if sel_left:
            xi=xi_left
    if max_dx is not None:
        dx_best=np.minimum(dx_right,dx_left)
        xi=np.where(dx_best<=max_dx,xi,-1)
    return xi

def nearest_val(A,x):
    """ return something like x, but with each element (or the scalar)
    replaced by the nearest value in the sorted 1-D array A
    """
    return A[nearest(A,x)]

def mag(vec):
    vec = np.asarray(vec)
    return np.sqrt( (vec**2).sum(axis=-1))

def to_unit(vecs):
    return vecs / mag(vecs)[...,None]

def dist(a,b=None):
    if b is not None:
        a=a-b
    return mag(a)


def dist_along(x,y=None):
    if y is None:
        x,y = x[:,0],x[:,1]

    # and a distance along transect
    steps=np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return np.concatenate( ( [0],
                             np.cumsum(steps) ) )


def point_line_distance(point,line):
    """
    point: [nd] array
    line [2,nd] array
    """
    # find the point-line distance
    delta = point - line[0]
    vec = to_unit(line[1] - line[0])
    delta -= np.dot(delta,vec) * vec
    return mag(delta)

# rotate the given vectors/points through the CCW angle in radians
def rot_fn(angle):
    R = np.array( [[np.cos(angle),-np.sin(angle)],
                   [np.sin(angle),np.cos(angle)]] )
    def fn(pnts):
        pnts=np.asarray(pnts)
        orig_shape=pnts.shape
        # could make the multi-dimensional side smarter...
        pnts=pnts.reshape([-1,2])
        pnts=np.tensordot(R,pnts,axes=(1,-1) ).transpose() 
        pnts=pnts.reshape(orig_shape)
        return pnts
    return fn

def rot(angle,pnts):
    return rot_fn(angle)(pnts)

def signed_area(points):
    i = np.arange(points.shape[0])
    ip1 = (i+1)%(points.shape[0])
    return 0.5*(points[i,0]*points[ip1,1] - points[ip1,0]*points[i,1]).sum()


## Tide-related functions

def find_slack(jd,u,leave_mean=False,which='both'):
    # returns ([jd, ...], 'high'|'low')
    dt=jd[1]-jd[0]

    u=filters.lowpass_fir(u,
                            winsize=1+np.round(2./(dt*24)))
    if not leave_mean:
        u-=filters.lowpass_fir(u,
                                 winsize=1+np.round(33./(dt*24)))

    missing=np.isnan(u)
    u[missing]=np.interp(jd[missing],
                         jd[~missing],u[~missing])

    # transition from ebb/0 to flood, or the other way around
    sel_low=(u[:-1]<=0) & (u[1:]>0)
    sel_high=(u[:-1]>0) & (u[1:]<=0)
    if which=='both':
        sel=sel_low|sel_high
    elif which=='high':
        sel=sel_high
    elif which=='low':
        sel=sel_low
    else:
        assert(False)
    
    b=np.nonzero(sel)[0]
    jd_slack=jd[b]-u[b]/(u[b+1]-u[b])*dt
    if u[0]<0:
        start='ebb'
    else:
        start='flood'
    return jd_slack,start
    

def hour_tide(jd,u,jd_new=None,leave_mean=False):
    fn=hour_tide_fn(jd,u,leave_mean=leave_mean)

    if jd_new is None:
        jd_new=jd
    return fn(jd_new)

def hour_tide_fn(jd,u,leave_mean=False):
    """ Return a function for extracting tidal hour 
    from the time/velocity given.
    Use the _fn version if making repeated calls with different jd_new,
    but the same jd,u
    """
    #function hr_tide=hour_tide(jd,u,[jd_new],[leave_mean]);
    #  translated from rocky's m-files
    #   generates tidal hours starting at slack water, based on 
    #   u is a vector, positive is flood-directed velocity
    #   finds time of "slack" water
    #   unless leave_mean=True, removes low-pass velocity from record
    
    jd_slack,start=find_slack(jd,u,leave_mean=leave_mean,which='both')
    # left/right here allow for one more slack crossing 
    hr_tide=np.interp(jd,
                      jd_slack,np.arange(len(jd_slack))*6,
                      left=-0.01,right=len(jd_slack)-0.99)
    if start=='flood':
        hr_tide += 6 # starting on an ebb

    print("start is",start)
    hr_tide %= 12

    # angular interpolation - have to use scipy interp1d for complex values
    arg=np.exp(1j*hr_tide*np.pi/6.0)
    def fn(jd_new):
        argi=interp1d(jd,arg,bounds_error=False)(jd_new)
        hr_tide=(np.angle(argi)*6/np.pi) % 12.0
        return hr_tide
    return fn


def find_phase(jd,u,which='ebb',**kwargs):
    slacks,start = find_slack(jd,u,which='both',**kwargs)
    if start==which: # starts with partial phase - drop it
        slacks=slacks[1:]
    if len(slacks) % 2:
        slacks=slacks[:-1]
    return np.array( [slacks[::2],slacks[1::2]] ).T


def quadmesh_interp_one(X,Z,V,x,z):
    # assumes that x is independent of z, but not the other way 
    mesh_x=X[0,:]
    col=np.interp(x,
                  mesh_x,np.arange(len(mesh_x)),left=np.nan,right=np.nan)
    if np.isnan(col):
        return np.nan
    col_i=int(col) # Truncate
    alpha=col-col_i
    z_left=Z[:,col_i]
    z_right=Z[:,col_i+1]
    mesh_z=(1-alpha)*z_left + alpha*z_right
    if mesh_z[0] > mesh_z[-1]:
        row=np.searchsorted(-mesh_z,-z)-1 
    else:
        row=np.searchsorted(mesh_z,z)-1 
    if row<0 or row>len(mesh_z)-2:
        return np.nan
    return V[row,col]
def quadmesh_interp(X,Z,V,x,z):
    result=np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        result[idx] = quadmesh_interp_one(X,Z,V,x[idx],z[idx])
    return result



def resample_to_common(A,Z,
                       z=None,
                       dz=None,n_samples=None,
                       max_z=None,min_z=None,
                       left=None,right=None):
    """ given data in A, with the coordinates of one axis dependent on 
    the other, resample, so that all elements of that axis have the
    same coordinates.
    in other words, A~[time,depth]
    Z~[time,depth]
    but you want z~[depth].

    for now, z has to be the second axis.

    Z can either be the per-element z values, or just a pair
    of values giving the evenly spaced range.
    """
    if max_z is None:
        max_z=Z.max()
    if min_z is None:
        min_z=Z.min()

    if z is None:
        if n_samples is not None:
            z=np.linspace(min_z,max_z,n_samples)
        else:
            z=np.arange(min_z,max_z,dz)

    new_A=np.zeros( (A.shape[0],len(z)), A.dtype)

    for i in range(A.shape[0]):
        if A.shape[1]==2:
            old_z=np.linspace( data2['range_m'][i,0],
                               data2['range_m'][i,1],
                               data2['echo'].shape[1])
        else:
            old_z=Z[i,:]
        new_A[i,:]=np.interp(z,
                             old_z,data2['echo'][i,:],
                             left=left,right=right)
    return z,new_A


def principal_theta(vec,eta=None,positive='flood',detrend=False,
                    ambiguous='warn'):
    """
    vec: 2D velocity data, last dimension must be {x,y} component.
    eta: if specified, freesurface data with same time dimension as vec.
      used in conjunction with positive to resolve the ambiguity in theta.
      this is approximate at best, esp. since there is no time information.
      the assumption is that the tides are between standing and progressive,
      such that u*h and u*dh/dt are both positive for flood-positive u.

    ambiguous: 'warn','error','standing','progressive','nan' see code.
    """
    # vec just needs to have a last dimensions of 2.
    vec=vec.reshape([-1,2])
    if detrend:
        vbar=vec.mean(axis=0)
        vec=vec-vbar[None,:]
    svdU,svdS,svdVh = np.linalg.svd( np.cov( vec.T ) )

    theta=np.arctan2( svdU[0,1],svdU[0,0] )
    if eta is not None:
        unit=np.array( [np.cos(theta),np.sin(theta)] )
        U=np.dot(vec-vec.mean(axis=0),unit)

        dhdt=eta[1:] - eta[:-1]
        u_h = np.dot( U, eta-eta.mean() )
        u_dh = np.dot( 0.5*(U[1:] + U[:-1]), dhdt-dhdt.mean() )
        # standing wave: u_h near zero, u_dh positive
        if (u_h<0) and (u_dh<0):
            if positive=='flood':
                theta+=np.pi
        elif (u_h>0) and (u_dh>0):
            if positive=='ebb':
                theta+=np.pi
        elif ambiguous=='standing':
            if (u_dh>0) != (positive=='flood'):
                theta+=np.pi
        elif ambiguous=='progressive':
            if (u_h>0) != (positive=='flood'):
                theta+=np.pi
        else:
            if ambiguous=='warn':
                log.warning("principal_theta: flood direction still ambiguous")
            elif ambiguous=='error':
                raise principal_theta.Exception("u_h: %f  u_dh: %f"%(u_h,u_dh))
            elif ambiguous=='nan':
                return np.nan

    return theta
class PrincipalThetaException(Exception):
    pass
principal_theta.Exception=PrincipalThetaException

def principal_vec(vec,**kw):
    theta=principal_theta(vec,**kw)
    return np.array( [np.cos(theta),np.sin(theta)] )

def rotate_to_principal(vec,eta=None,positive='flood',detrend=False,
                        ambiguous='warn'):
    theta=principal_theta(vec,eta,positive,detrend,ambiguous=ambiguous)
    # theta gives the direction of flood-positive currents.  To get that
    # component into the 'x' component, rotate by -theta
    return rot(-theta,vec)

def bootstrap_resample(X, n=None):
    """ 
    Bootstrap resample an array_like
    credits to http://nbviewer.ipython.org/gist/aflaxman/6871948

    Parameters:
    
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None

    Results:

    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

def bootstrap_stat(X,n_pop=10000,n_elements=None,pop_stat=np.mean,
                   bootstrap_stat=lambda l: (np.var(l),np.mean(l)) ):
    # resample X by simple resample-with-replacement, N times,
    # for reach resample, compute pop_stat (i.e. the mean, the variance...)
    # then for the collection of population stats, compute the bootstrap_stat.
    pop_stats=[None]*n_pop
    for i in range(n_pop):
        res=bootstrap_resample(X,n=n_elements)
        pop_stats[i] = pop_stat(res) 
    return bootstrap_stat( np.array(pop_stats) )

def model_skill(xmodel,xobs,ignore_nan=True):
    """ 
    Wilmott 1981 model skill metric
    """
    # Weird - random data gets a score of 0.43 or so - 
    #  if the prediction is too small by a factor of 10, the skill is still about
    #  the same.  In fact if the prediction is 0, it will still get a score of 0.43.
    # but if the predictions are too large, or have a markedly different mean, then
    # the score gets much closer to 0.

    if ignore_nan:
        sel=np.isfinite(xmodel+xobs)
    else:
        sel=slice(None)

    num = np.sum( (xmodel - xobs)[sel]**2 )
    den = np.sum( (np.abs(xmodel[sel] - xobs[sel].mean()) + np.abs(xobs[sel] - xobs[sel].mean()))**2 )
    
    skill = 1 - num / den
    return skill

def break_track(xy,waypoints,radius_min=400,radius_max=800,min_samples=10):
    """ 
    xy: coordinate sequence of trackline
    waypoints: collection of waypoints

    return pairs of indices giving start/end of transects, split by waypoints.
    """
    breaks=[]

    for waypt in waypoints:
        dists = mag( xy[:] - waypt)

        in_max=np.concatenate( ( [False],
                                 dists<radius_max ,
                                 [False]) )

        delta_max=np.diff(1*in_max)
        enter_idxs=np.nonzero( delta_max>0 )[0]
        exit_idxs =np.nonzero( delta_max<0 )[0]


        for enter_i,exit_i in zip(enter_idxs,exit_idxs):
            best_idx=np.argmin(dists[enter_i:exit_i]) + enter_i
            if dists[best_idx] < radius_min:
                breaks.append( best_idx )

    breaks.append(0)
    breaks.append( len(dists) )
    breaks=np.sort(breaks)
    print(breaks)
    breaks=np.array( breaks )
    sections=np.array( [ breaks[:-1],
                         breaks[1:] ] ).T

    short=(sections[:,1] - sections[:,0])<min_samples
    return sections[~short,:]

class forwardTo(object):
    """
    credits:
    http://code.activestate.com/recipes/510402-attribute-proxy-forwarding-attribute-access/

    A descriptor based recipe that makes it possible to write shorthands
    that forward attribute access from one object onto another.

    >>> class C(object):
    ...     def __init__(self):
    ...         class CC(object):
    ...             def xx(self, extra):
    ...                 return 100 + extra
    ...             foo = 42
    ...         self.cc = CC()
    ...
    ...     localcc = forwardTo('cc', 'xx')
    ...     localfoo = forwardTo('cc', 'foo')
    ...
    >>> print C().localcc(10)
    110
    >>> print C().localfoo
    42

    Arguments: objectName - name of the attribute containing the second object.
               attrName - name of the attribute in the second object.
    Returns:   An object that will forward any calls as described above.
    """
    def __init__(self, objectName, attrName):
        self.objectName = objectName
        self.attrName = attrName

    def __get__(self, instance, owner=None):
        return getattr(getattr(instance, self.objectName), self.attrName)

    def __set__(self, instance, value):
        setattr(getattr(instance, self.objectName), self.attrName, value)

    def __delete__(self, instance):
        delattr(getattr(instance, self.objectName), self.attrName)


# Three representations for datetimes (ignoring date-only and timezone issues)
#   floating point datenums, in the standard of matplotlib, which matlab
#     standard + 366 (pretty sure)
#   datetime object.  i.e. import datetime ; datetime.datetime(2013,11,4,12,0)
#   numpy datetime64 (including various precisions)

# There are 3 "containers":
#  numpy array
#  scalar
#  pandas object with index
#    this one is a bit weird

def to_dnum(x):
    # Unwrap pandas data:
    # used to silently grab the index.
    if pd is not None:
        if isinstance(x,pd.DataFrame):
            log.warning("to_dnum should be given either a series or an index, not a dataframe")
            assert isinstance(x.index,pd.DatetimeIndex)
            x=x.index
        if isinstance(x,pd.DatetimeIndex) or isinstance(x,pd.Series):
            x=x.values

    # Unwrap xarray data
    if xr is not None:
        if isinstance(x,xr.DataArray):
            x=x.values

    if isinstance(x,datetime.datetime) or isinstance(x,datetime.date):
        return date2num(x)

    if np.isscalar(x):
        if isinstance(x,float):
            return x

        if isinstance(x,np.datetime64):
            return dt64_to_dnum(x)
        
        assert False
    else:
        if pd is not None and isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
            x=x.index.values

        if np.issubdtype(x.dtype,np.float):
            return x
        if isinstance(x[0],datetime.datetime) or isinstance(x[0],datetime.date):
            return date2num(x)
        if np.issubdtype(x.dtype,np.datetime64):
            return dt64_to_dnum(x)
        assert False

def to_dt64(x):
    if pd is not None and isinstance(x,pd.DataFrame) or isinstance(x,pd.Series):
        x=x.index.values

    # isscalar is too specific - only for numpy scalars
    if not isinstance(x,np.ndarray):
        if isinstance(x,float):
            x=num2date(x) # now a datetime
        elif isinstance(x,str):
            if 'since' in x:
                return cf_string_to_dt64(x)
            else:
                return np.datetime64(x)

        if isinstance(x,datetime.datetime) or isinstance(x,datetime.date):
            x=np.datetime64(x)

        if np.issubdtype(x,np.datetime64):
            return x

        assert False
    elif isinstance(x,pd.Timestamp):
        return x.to_datetime64()
    else:
        if np.issubdtype(x.dtype, np.float):
            x=num2date(x)

        x=np.asarray(x)
        if isinstance( x.flat[0], datetime.datetime ) or isinstance(x.flat[0],datetime.date):
            x=np.array(x,'M8[ns]')
        
        if np.issubdtype(x.dtype,np.datetime64):
            return x

        assert False

def to_unix(t):
    """ 
    Convert t to unix epoch time, defined as the number of seconds
    since 1970-01-01 00:00:00.  The result is a float or int, and is *not*
    distinguishable a priori from datenums.  For that reason, numerical values
    passed to the other date converters are assumed to be datenums.

    Also note that if you want to double-check the results of this function
    via python's datetime module, use datetime.datetime.utcfromtimestamp(x).
    Otherwise you are likely to get some unwelcome time zone conversions.
    """
    if 1:
        # I think this is actually pretty good.
        unix0=np.datetime64('1970-01-01 00:00:00')
        return (to_dt64(t) - unix0)/np.timedelta64(1,'s')
    else:
        dt=to_datetime(t)
        dt0=datetime.datetime(1970, 1, 1)
        return (dt - dt0).total_seconds()

def cf_string_to_dt64(x):
    """ return a seconds-based numpy datetime 
    from something like 
    ``1000 seconds since 1983-01-09T12:43:10``

    This is conditionally called from to_datetime(), too.

    A timezone, either as a trailing 'Z' or -0:00 is allowed,
    but other timezones are not (since that would introduce an 
    ambiguity as to whether to adjust to UTC, or leave in 
    another timezone)
    """
    duration,origin = x.split(" since ")
    count,units = duration.split()

    if origin.endswith('Z'):
        origin=origin[:-1]
    else:
        # does it have a +0800 style time zone?
        time_part=origin[10:]
        for sep in "+-":
            if sep in time_part:
                tz_part=time_part[ time_part.index(sep):]
                break
        else:
            tz_part=None
        if tz_part:
            tz_part=tz_part.replace(':','')
            assert float(tz_part)==0
            # extra 1 for the separator
            origin=origin[:-1-len(tz_part)]

    tzero = np.datetime64(origin,'s')

    mult = dict(seconds=1,
                minutes=60,
                hours=3600,
                days=86400)[units]
    delta=np.timedelta64(int( float(count) * mult),'s')
    return tzero + delta


def to_datetime(x):
    try:
        x=x.asm8 # if it's a scalar pandas timestamp
    except AttributeError:
        pass

    # Unwrap xarray data
    if xr is not None and isinstance(x,xr.DataArray):
        x=x.values
    
    if isinstance(x,float):
        return num2date(x)
    if isinstance(x,datetime.datetime):
        return x
    if isinstance(x,str):
        if 'since' in x:
            x=cf_string_to_dt64(x)
        else:
            # see if numpy will parse it
            # a bit dangerous and not as smart as pandas, but
            # at the moment we're trying to avoid a pandas dependency
            # here.
            x=np.datetime64(x)
    
    # isscalar is not so general - it does *not* mean not array, it means
    # the value *is* a numpy scalar.
    if np.isscalar(x):
        if np.issubdtype(x,np.datetime64):
            ts = (x - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            return datetime.datetime.utcfromtimestamp(ts)
    else:
        if np.issubdtype(x.dtype,np.float):
            return num2date(x)
        if np.issubdtype(x.dtype,np.datetime64):
            ts = (x - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
            # return datetime.datetime.utcfromtimestamp(ts) # or do we have to vectorize?
            return [datetime.datetime.utcfromtimestamp(x) 
                    for x in ts] 
        if len(x) and isinstance(x.flatten[0],datetime.datetime):
            return x

# some conversion help for datetime64 and python datenums
# pandas includes some of this functionality, but trying to 
# keep utils.py pandas-free (no offense, pandas)

def dt64_to_dnum(dt64):
    # get some reference points:

    dt1=datetime.datetime.fromordinal(1) # same as num2date(1)

    for reftype,ref_to_days in [('M8[us]',86400000000),
                                ('M8[D]',1)]:
        dt64_1=np.datetime64(dt1).astype(reftype)
        # integer microseconds since day 1
        try:
            diff=dt64.astype(reftype) - dt64_1
        except TypeError:
            continue # numpy won't convert a value in days to microseconds.

        # need to convert to days, as floating point
        delta=np.timedelta64(ref_to_days,'us')
        break

    dnum=1.0+diff/delta
    return dnum

def dnum_to_dt64(dnum,units='us'):
    reftype='m8[%s]'%units

    # how many units are in a day?
    units_per_day=np.timedelta64(1,'D').astype(reftype)
    # datetime for the epoch
    dt_ref=datetime.datetime(1970,1,1,0,0)
    
    offset = ((dnum-dt_ref.toordinal())*units_per_day).astype(reftype)
    return np.datetime64(dt_ref) + offset

def dnum_jday0(dnum):
    # return a dnum for the first of the year
    dt=to_datetime(dnum)
    dt0=datetime.datetime(dt.year,1,1)
    return to_dnum(dt0)

def invert_permutation(p):
    '''
    Returns an array s, where s[i] gives the index of i in p.
    The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    see http://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    
    this version, using np.arange instead of xrange, is 23% faster than the
    np.put() version on that stackoverflow question.
    '''
    s = np.zeros(p.size, p.dtype) # np.zeros is better than np.empty here, at least on Linux
    s[p] = np.arange(p.size) 
    return s

def circumcenter(p1,p2,p3):
    ref = p1
    
    p1x = p1[...,0] - ref[...,0] # ==0.0
    p1y = p1[...,1] - ref[...,1] # ==0.0
    p2x = p2[...,0] - ref[...,0]
    p2y = p2[...,1] - ref[...,1]
    p3x = p3[...,0] - ref[...,0]
    p3y = p3[...,1] - ref[...,1]

    vc = np.zeros( p1.shape, np.float64)
    
    # taken from TRANSFORMER_gang.f90
    dd=2.0*((p1x-p2x)*(p1y-p3y) -(p1x-p3x)*(p1y-p2y))
    b1=p1x**2+p1y**2-p2x**2-p2y**2
    b2=p1x**2+p1y**2-p3x**2-p3y**2 
    vc[...,0]=(b1*(p1y-p3y)-b2*(p1y-p2y))/dd + ref[...,0]
    vc[...,1]=(b2*(p1x-p2x)-b1*(p1x-p3x))/dd + ref[...,1]
    
    return vc

def poly_circumcenter(points):
    """ 
    unbiased (mostly) estimate of circumcenter, by computing circumcenter
    of consecutive groups of 3 points
    """
    triples=np.array(list(circular_n(points,3)))
    ccs=circumcenter(triples[:,0],triples[:,1],triples[:,2])
    return np.array(ccs).mean(axis=0)

def rms(v):
    return np.sqrt( np.mean( v**2 ) )


def circular_pairs(iterable):
    """
    like pairwise, but closes the loop.
    s -> (s0,s1), (s1,s2), (s2, s3), ..., (sN,s0)
    """
    a, b = itertools.tee(iterable)
    b = itertools.cycle(b)
    next(b, None)
    return six.moves.zip(a, b)

def circular_n(iterable,n):
    iters = list(itertools.tee(iterable,n))
    for i in range(1,n):
        iters[i]=itertools.cycle(iters[i])
        [next(iters[i],None) for count in range(i)]
    return six.moves.zip( *iters )
        

def cdiff(a,n=1,axis=-1):
    """ 
    Like np.diff, but include difference from last element back
    to first.
    """ 
    assert n==1 # not ready for higher order
    # assert axis==-1 # also not ready for weird axis

    result=np.zeros_like(a)
    d=np.diff(a,n=n,axis=axis)

    # using [0] instead of 0 means that axis is preserved
    # so the concatenate is easier
    last=np.take(a,[0],axis=axis) - np.take(a,[-1],axis=axis) 

    # this is the part where we have to assume axis==-1
    #return np.concatenate( [d,last[...,None]] )

    return np.concatenate( [d,last] ) 


def enumerate_groups(keys):
    """
    given an array of labels, return, in increasing value of key,
    yield tuples (key,idxs) such that np.all(keys[idxs]==key)
    """
    order=np.argsort(keys) 
    breaks=1+np.nonzero( np.diff( keys[order] ) )[0]
    for group in np.split( order,breaks ):
        group.sort()
        yield keys[group[0]],group
    
def moving_average_nearest(x,y,n):
    """ 
    x,y: 1-D arrays
    n: integer giving size of the averaging window

    Returns a new array with each element calculated as the
    mean of the n nearest (in terms of x coordinate) input values.

    Very inefficient implementation!
    """
    out=np.zeros_like(y)
    for i in range(len(x)):
        dists=np.abs(x-x[i])
        choose=np.argsort(dists)[:n]
        out[i]=np.mean(y[choose]) 
    return out


def touch(fname, times=None):
    """ 
    Like unix touch.  Thanks to 
    http://stackoverflow.com/questions/1158076/implement-touch-using-python
    """
    with open(fname, 'a'):
        os.utime(fname, times)


def orient_intersection(lsA,lsB,delta=1.0):
    """ 
    lsA,lsB: LineString geometries with exactly one intersecting point.
    returns 
    1: if lsA crosses lsB left to right, when looking from the start towards the
    end of lsB.
    -1: the other case.

    delta: current implementation uses a finite difference, and delta specifies 
    the scale of that difference.
    """
    pnt=lsA.intersection(lsB)
    fA=lsA.project(pnt)
    fB=lsB.project(pnt)

    delta=1.0 # [m] ostensibly
    # assuming
    pntAplus=lsA.interpolate(fA+delta)
    pntBplus=lsB.interpolate(fB+delta)

    # if [pntBplus,pnt,pntAplus] is oriented CCW, then lsA crosses
    # left to right with respect to lsB
    pnts=np.array( [pntBplus.coords[0],
                    pnt.coords[0],
                    pntAplus.coords[0]] )
    area=signed_area(pnts)
    assert area!=0.0
    if area<0:
        return -1
    else:
        return 1
        


# Some XLS-related functions:
def alpha_to_zerobased(s):
    value=0
    for i,c in enumerate(s):
        value=26*value + (ord(c) - ord('A') + 1)
    # A really means 1, except for the last character
    # for which A means 0.  So the above is a 1-based value,
    # even though it doesn't look like it.
    return value-1 # make it 0-based

def parse_xl_address(s):
    m=re.match('([A-Z]+)([0-9]+)',s)
    return (alpha_to_zerobased(m.group(1)),
            int(m.group(2))-1)

def parse_xl_range(s):
    # note that this does *NOT* change to be exclusive indexing
    return [parse_xl_address(part)
            for part in s.split(':')]

def as_sheet(sheet,fn):
    if isinstance(sheet,str) and fn is not None:
        sheet=xlrd.open_workbook(fn).sheet_by_name(sheet)
    return sheet

def cell_region_to_array(sheet,xl_range,dtype='O',fn=None):
    start,stop = parse_xl_range(xl_range)
    data=[]

    sheet=as_sheet(sheet,fn)
    
    # have to add 1 because these come in as inclusive
    # indices, but range assumes exclusive stop
    for row in range(start[0],stop[0]+1):
        row_data=[]
        for col in range(start[1],stop[1]+1):
            row_data.append( sheet.cell_value(col,row) )
        data.append(row_data)

    return np.array(data,dtype=dtype).T



def cell_region_to_df(sheet,xl_range,idx_ncols=0,idx_nrows=0,fn=None):
    chunk=cell_region_to_array(sheet,xl_range,fn=fn)

    columns=chunk[:idx_nrows,idx_ncols:].T
    if idx_nrows==1:
        columns=columns[:,0]
        
    df=pd.DataFrame( chunk[idx_nrows:,idx_ncols:],
                     index=chunk[idx_nrows:,:idx_ncols],
                     columns=columns)
    return df


def uniquify_paths(fns):
    # split each filename into parts, and include only enough of the trailing parts
    # make each name unique, and omit trailing parts that are the same for all fns.

    # reverse to make the indexing easier
    splits = [ fn.split('/')[::-1] for fn in fns]
    

    # Trying to identify a,b such that splits[a:b] uniquely identify each source
    a=0
    
    # first trim identical prefixes (which were originally suffixes)
    done = 0
    for a in range(len(splits[0])):
        for i in range(1,len(splits)):
            if splits[0][a] != splits[i][a]:
                log.info("Suffix %d differs"%a)
                done=1
                break
        if done:
            break

    b=a+1
    for b in range(a+1,len(splits[0])):
        done = 1
        for i in range(1,len(splits)):
            if splits[0][a:b] == splits[i][a:b]:
                # still not uniquely identified
                done = 0
                break
        if done:
            break
    trimmed = [ '/'.join(split[a:b][::-1]) for split in splits]
    return trimmed


# Used to be in array_append
def array_append( A, b ):
    """ 
    append b to A, where b.shape == A.shape[1:]
    Attempts to make this fast by dynamically resizing the base array of
    A, and returning the appropriate slice.
    """

    # a bit more complicated because A may have a different column ordering
    # than A.base (due to a previous transpose, probably)
    # can compare strides to see if our orderings are the same.

    # possible that it's not a view, or
    # the base array isn't big enough, or
    # the layout is different and it would just get confusing, or
    # A is a slice on other dimensions, too, which gets too confusing.
    
    if A.base is None or type(A.base) == str \
           or A.base.size == A.size or A.base.strides != A.strides \
           or A.shape[1:] != A.base.shape[1:]:
        new_shape = list(A.shape)

        # make it twice as long as A, and in case the old shape was 0, add 10
        # in for good measure.
        new_shape[0] = new_shape[0]*2 + 10  
        
        base = np.zeros( new_shape, dtype=A.dtype)
        base[:len(A)] = A

        # print "resized based array to %d elements"%(len(base))
    else:
        base = A.base

    A = base[:len(A)+1]
    if A.dtype.isbuiltin:
        A[-1] = b
    else:
        # recarray's get tricky, and the corner cases are not clear to me.
        # if b is a 0-dimensional recarray, list(b) doesn't work.
        # so just punt and try both.
        try: # if type(b) == numpy.void:
            val=b.tolist()
        except AttributeError:
            # cover cases where the new value isn't an ndarray, but a list or tuple.
            val=list(b)
        A[-1] = val
    return A

def array_concatenate( AB ):
    """
    similiar to array_append, but B.shape[1:] == A.shape[1:]

    while the calling convention is similar to concatenate, it currently only supports
    2 arrays
    """
    A,B = AB
    
    if A.base is None or type(A.base) == str \
           or A.base.size == A.size or A.base.strides != A.strides \
           or len(A) + len(B) > len(A.base) \
           or A.shape[1:] != A.base.shape[1:]:
        new_shape = list(A.shape)

        # make it twice as long as A, and in case the old shape was 0, add 10
        # in for good measure.
        new_shape[0] = max(new_shape[0]*2 + 10,
                           new_shape[0] + len(B))
        
        base = np.zeros( new_shape, dtype=A.dtype)
        base[:len(A)] = A
    else:
        base = A.base

    lenA = len(A)
    A = base[:lenA+len(B)]
    A[lenA:] = B
        
    return A
    
    
def concatenate_safe_dtypes( ab ):
    """ 
    Concatenate two arrays, but allow for the dtypes to be different.  The
    fields are taken from the first array - matching fields in subsequent arrays
    are copied, others discarded.
    """
    a,b = ab # for now, force just two arrays
    
    result = np.zeros( len(a)+len(b), a.dtype)
    result[:len(a)] = a

    for name in b.dtype.names:
        if name in a.dtype.names:
            result[ len(a):len(a)+len(b)][name] = b[name]
    return result

def recarray_del_fields(A,old_fields):
    new_dtype=[fld
               for fld in A.dtype.descr
               if fld[0] not in old_fields]

    new_A=np.zeros( len(A), dtype=new_dtype)
    
    for name in new_A.dtype.names:
        new_A[name]=A[name]

    return new_A
    
def recarray_add_fields(A,new_fields):
    """ 
    A: a record array
    new_fields: [ ('name',data), ... ]
    where data must be the same length as A.  So far, no support for
    non-scalar values
    """
    new_dtype=A.dtype.descr
    for name,val in new_fields:
        # handle non-scalar fields
        # assume that the first dimension is the "record" dimension
        new_dtype.append( (name,val.dtype,val.shape[1:] ) )
    new_names=[name for name,val in new_fields]
    new_values=[val for name,val in new_fields]
    new_A=np.zeros( len(A), dtype=new_dtype)
    
    for name in new_A.dtype.names:
        try:
            new_A[name]=new_values[new_names.index(name)]
        except ValueError:
            new_A[name]=A[name]

    return new_A


def isnat(x):
    """ 
    datetime64 analog to isnan.
    doesn't yet exist in numpy - other ways give warnings
    and are likely to change.  
    """
    return x.astype('i8') == np.datetime64('NaT').astype('i8')


def group_by_sign_hysteresis(Q,Qlow=0,Qhigh=0):
    """
    A seemingly over-complicated way of solving a simple problem.
    Breaking a time series into windows of positive and negative values 
    (e.g. delimiting flood and ebb in a velocity or flux time series).
    With the complicating factor of the time series sometimes hovering 
    close to zero.
    This function introduces some hysteresis, while preserving the exact
    timing of the original zero-crossing.  Qlow (typ. <0) and Qhigh (typ>0)
    set the hysteresis band.  Crossings only count when they start below Qlow 
    and end above Qhigh.  Within that period, there could be many small amplitude
    crossings - the last one is used.  This is in contrast to standard hysteresis
    where the timing would corresponding to crossing Qhigh or Qlow.  

    returns two arrays, pos_windows = [Npos,2], neg_windows=[Nneg,2]
    giving the indices in Q for respective windows
    """
    states=[]

    for idx in range(len(Q)):
        if Q[idx]<=Qlow:
            states.append('L')
        elif Q[idx]<=0:
            states.append('l')
        elif Q[idx]<=Qhigh:
            states.append('h')
        else:
            states.append('H')
    statestr="".join(states)

    p2n=[]
    for m in re.finditer( r'H[hl]*L',statestr):
        sub=m.group(0)
        # the last h-l transition
        # 1 for the leading H
        idx=m.span()[0] + 1 + re.search(r'[Hh][lL]+$',sub).span()[0]
        p2n.append(idx)

    n2p=[]
    for m in re.finditer( r'L[hl]*H',statestr):
        sub=m.group(0)
        # the last h-l transition
        idx=m.span()[0] + 1 + re.search(r'[Ll][hH]+$',sub).span()[0]
        n2p.append(idx)

    if p2n[0]<n2p[0]:
        # starts positive, so the first full window is negative
        neg_windows=zip(p2n,n2p)
        pos_windows=zip(n2p,p2n[1:])
    else:
        # starts negative, first full window is positive
        pos_windows=zip(n2p,p2n)
        neg_windows=zip(p2n,n2p[1:])
    pos_windows=np.array(list(pos_windows))
    neg_windows=np.array(list(neg_windows))
    return pos_windows,neg_windows


def point_in_polygon( pgeom, randomize=False ):
    """ return a point that lies inside the given [multi]polygon
    geometry

    randomize: if True, pick points randomly, but fallback to deterministic
    approach when the geometry is to sparse
    """
    if randomize:
        env = pgeom.envelope
        if env.area/pgeom.area > 1e4:
            print("Sliver! Going to non-random point_in_polygon code")
            return point_in_polygon(pgeom,False)
        
        env_pnts = np.array(env.exterior.coords)
        minx,miny = env_pnts.min(axis=0)
        maxx,maxy = env_pnts.max(axis=0)

        while 1:
            x = minx + np.random.random()*(maxx-minx)
            y = miny + np.random.random()*(maxy-miny)
            pnt_geom = geometry.Point(x,y)
            if pgeom.contains(pnt_geom):
                return np.array([x,y])
    else:
        # print "holding breath for point_in_polygon"
        # Try a more robust way of choosing the point:
        c = pgeom.centroid
        min_x,min_y,max_x,max_y = pgeom.bounds
        horz_line = geometry.LineString([[min_x-10.0,c.y],
                                         [max_x+10.0,c.y]])
        full_ix = horz_line.intersection( pgeom )

        # intersect it with the polygon:
        if full_ix.type == 'MultiLineString':
            lengths = [l.length for l in full_ix.geoms]
            best = np.argmax(lengths)
            ix = full_ix.geoms[best]
        else:
            ix = full_ix
        # then choose the midpoint:
        midpoint = np.array(ix.coords).mean(axis=0)
        return midpoint
