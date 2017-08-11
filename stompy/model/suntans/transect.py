# This is not in a more general place because it's old code
# and probably doesn't reflect more recent design decisions
# (e.g. using xarray).  It's shoved into this SUNTANS 
# folder to support old code in sunreader.

# development of mimicking transects from profile data

import sys
# from safe_pylab import *
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

from ...plot.contour_transect import contourf_t,contour_t

try:
    import cPickle as pickle
except ImportError:
    import pickle

class Transect(object):
    """ a 2-D data container, representing a series of casts to constant
    depths, where each cast has a recorded xy lateral location and time.

    depths can take several forms, where N,M = scalar.shape, (M stations, N measurement per water column)
      [N]: elevation at center of cell, or measurement point
      [N+1]: elevation of interface between control volumes
      [M,N]: elevation at center of cell, variable across watercolumns
      [M,N+1]: elevation of interfaces between control volumes, variable across water columns.

    2012/04/25: still developing all of those choices... caveat emptor.
    """
    
    def __init__(self,xy,times,elevations,scalar,desc="",dists=None):
        self.xy = xy
        self.times = times

        # scalar should be masked, but if not, mask it based on nan
        if not isinstance(scalar,ma.MaskedArray):
            scalar = ma.masked_invalid(scalar,copy=False)
        self.scalar = scalar
        self.desc = desc

        if dists is None:
            self.compute_distances()
        else:
            self.dists = dists

        # manufacture a consistent set of elevation information:
        if elevations.ndim == 1:
            if elevations.shape[0] != scalar.shape[0]:
                raise Exception("1-D elevation array length doesn't match scalar")
            
            # duplicate for each watercolumn
            elevations = np.repeat(elevations[:,None], # 0 index is z, 1 index is xy
                                   [scalar.shape[1]],axis=1 )

        elif elevations.ndim == 2:
            pass
        else:
            raise Exception("Bad ndim for elevation data")
            
        if elevations.shape[1] != scalar.shape[1]:
            raise Exception("Bad x-shape for elevation data")
        if elevations.shape[0] == scalar.shape[0]:
            self.center_elevations = elevations
            # and synthesize interfaces - a bit weak...
            # but this should be consistent with the way that it seems polaris cruises are
            # recorded - the measurements start 1m down, but the polaris code copies that measurement
            # to the freesurface h=0.  the last measurement appears to be taken at the bed
            breaks = 0.5*(elevations[:-1] + elevations[1:])
            self.interface_elevations = concatenate( ([ elevations[0] ],
                                                      breaks,
                                                      [ elevations[-1] ]) )
            # but that needs to be adjusted for watercolumns that don't span the whole range of z-levels
            for i in range(scalar.shape[1]):
                valid = np.nonzero(~self.scalar.mask[:,i])[0]
                if len(valid) > 1:
                    # if there is only one valid value, let it stand - there's no good way to dream up
                    # a nonzero dz for it
                    # top of the top valid cell - 
                    self.interface_elevations[ valid[0],i] = self.center_elevations[ valid[0],i]
                    # bottom of the last valid cell
                    self.interface_elevations[ valid[-1] + 1,i] = self.center_elevations[ valid[-1],i]
            self.scalar_location = 'centers'
        elif elevations.shape[0] == scalar.shape[0] + 1:                                                  
            self.interface_elevations = elevations
            self.center_elevations = 0.5*( elevations[1:] + elevations[:-1] )
            self.scalar_location = 'zones'
        else:
            raise Exception("Bad shape for elevation data")
            
    def compute_distances(self):
        """populate self.dist with distance along transect"""
        d = np.sqrt( (np.diff(self.xy,axis=0)**2).sum(axis=1) )
        self.dists = np.concatenate( ([0.0], np.cumsum( d )) )

    def do_contour(self,plotter,*args,**kwargs):
        # D,X = meshgrid(self.center_elevations,self.dists)
        D = self.center_elevations
        X = np.repeat(self.dists[None,:],D.shape[0],axis=0)
        return plotter(X.T,D.T,self.scalar.T,*args,**kwargs)
            
    def contourf(self,*args,**kwargs):
        return self.do_contour(contourf,*args,**kwargs)

    def contour(self,*args,**kwargs):
        return self.do_contour(contour,*args,**kwargs)

    def contourf_t(self,*args,**kwargs):
        return self.do_contour(contourf_t,*args,**kwargs)
                        
    def contour_t(self,*args,**kwargs):
        return self.do_contour(contour_t,*args,**kwargs)
        
    def plot_surface(self,labelA=True,labelB=True):
        plt.scatter(self.xy[:,0],self.xy[:,1],60,self.scalar[0,:],linewidth=0)
        if labelA:
            if labelA == True:
                labelA = str(self.dists[0])
            plt.annotate(labelA, self.xy[0,:] )
        if labelB:
            if labelB == True:
                labelB = str(self.dists[-1])
            plt.annotate(labelB, self.xy[-1,:] )

    def scatter(self,**kwargs):
        # D,X = meshgrid(self.elevations,self.dists)
        # WARNING: untested with new elevations code
        X = np.repeat(self.dists[None,:],D.shape[0],axis=0)
        x = X.ravel()
        y = self.center_elevations.ravel()
        s = np.transpose(self.scalar).ravel()
        
        plt.scatter(x,y,60,s,lw=0)

    def trim_to_valid(self):
        valid_z_levels = np.nonzero( np.any(~self.scalar.mask,axis=1) )[0]
        start_z = valid_z_levels[0]
        end_z = valid_z_levels[-1]
        
        self.center_elevations = self.center_elevations[start_z:end_z+1,:]
        self.interface_elevations = self.interface_elevations[start_z:end_z+2,:]
        
        self.scalar = self.scalar[start_z:end_z+1,:]

    def bed_elevations(self):
        """ Returns a vector of the best guess of the bed elevation
        at each cast
        """
        elevs = np.zeros( self.scalar.shape[1], np.float64 )
        for i in range(len(elevs)):
            k_bot = np.nonzero(np.isfinite(self.scalar[:,i]))[0][-1] # index inclusive!
            elevs[i] = self.interface_elevations[k_bot+1,i]
        return elevs

    def surface_elevations(self):
        """ Returns a vector of the best guess of the surface elevation
        at each cast.  For now, this is basically just 0.
        """
        elevs = np.zeros( self.scalar.shape[1], np.float64 )
        for i in range(len(elevs)):
            k_top = np.nonzero( np.isfinite(self.scalar[:,i] ) )[0][0]
            elevs[i] = self.interface_elevations[k_top,i]

        return elevs

    def d_dz(self):
        """ Estimate the vertical gradient at each watercolumn by fitting a line 
        """
        grads = np.zeros( self.scalar.shape[1], np.float64)
        for i in range(len(grads)):
            valid = ~self.scalar.mask[:,i]
            [m,b] = np.polyfit( self.center_elevations[valid,i], self.scalar[valid,i],1 )
            grads[i] = m
        return grads

    def vertical_average(self,z_top=None,z_bot=None,dz=None):
        """ integrates scalar over some vertical range.  z_top starts from the surface, going down
        z_bot starts from the bed, going up.
        dz: thickness of layer, if only one of z_top or z_bot are given

        e.g.
        surface 1m:  z_top=0  dz=1.0
        bed 1m: z_bot=0, dz=1.0

        full water column:
        z_top=0, z_bot=0
        """

        # returns an array of scalar values, same length as self.scalar.shape[1],
        # and with the same lateral information as self.xy, but taken as the
        # average of the top dz of the watercolumn at each station.

        vals = np.zeros( self.scalar.shape[1], np.float64 )

        bed_elevations = self.bed_elevations()
        surface_elevations = self.surface_elevations()

        for i in range(len(vals)):
            # get z_top and z_bot into the same datum as depth, namely 0 at (or near) the surface
            # and negative elevations going towards the bed.

            # for now, depth is found just by the bottom valid scalar entry
            depth = bed_elevations[i]
            eta = surface_elevations[i]

            if z_bot is not None and z_top is not None:
                elev_bot = depth + z_bot
                elev_top = eta - z_top
            elif z_bot is not None:
                elev_bot = depth + z_bot
                elev_top = depth + z_bot + dz
            elif z_top is not None:
                elev_top = eta - z_top
                elev_bot = eta - z_top - dz
            else:
                raise Exception("Must specify two of z_top,z_bot and dz")

            if self.scalar_location == 'centers':
                # WARNING: untested with new elevation code
                # The places we want values:
                this_wc = self.center_elevations[:,i]
                eval_z = this_wc[  (this_wc <= elev_top ) & (this_wc>=elev_bot) ]
                if len(eval_z) == 0 or eval_z[0] < elev_top:
                    eval_z = np.concatenate( ([elev_top],eval_z) )

                if len(eval_z) == 0 or eval_z[-1] > elev_bot:
                    eval_z = np.concatenate( (eval_z,[elev_bot]) )

                # interp likes increasing functions
                eval_z = eval_z[::-1]

                # limit this to the places in th water column that we have real data:
                valid = np.nonzero(self.scalar.mask[:,0]==False)[0]
                valid=valid[::-1]

                # default for interp is to extrapolate constant value - just what we want.
                eval_scal = np.interp(eval_z, self.center_elevations[valid,i], self.scalar[valid,i] )

                vals[i] = np.trapz(eval_scal,eval_z) / (eval_z[-1] - eval_z[0])
            elif self.scalar_location == 'zones':
                interfaces = self.interface_elevations[:,i].copy()
                clipped = interfaces.clip(elev_bot,elev_top)
                dzz = -np.diff(clipped) # negate since its stored surface to bed
                scal_column = self.scalar[:,i]
                valid = np.isfinite(scal_column)
                vals[i] = np.sum(scal_column[valid] * dzz[valid]) / np.sum(dzz[valid])
            else:
                raise Exception("bad scalar location: %s"%self.scalar_location)
        return vals
            

class TransectTimeSeries(object):
    """ Collect a series of Transects.  Each transect is assumed to have the same time for
    all points.  For starters, it really just offers a way to store them together, and query
    one based on a timestamp.  Eventually this will handle efficient storage and retrieval
    of sets of transects.

    Assumes that the xy points are constant across timesteps, but depths may change
    """
    def __init__(self,transects):
        self.transects = transects

        self.index_transects()

    def index_transects(self):
        """ Extract a datenum for each transect
        """
        self.transects_dnums = np.array( [tran.times[0] for tran in self.transects] )
        # so that we can run searchsorted directly and get the index of the nearest
        # transect
        self.threshold_dnums = transect_dnums[:-1] + 0.5*np.diff(transect_dnums)

    def save(self,filename):
        fp = open(filename,"wb")
        pickle.dump(self.transects,fp)
        fp.close()

    @staticmethod
    def load(filename):
        fp = open(filename,'rb')
        transects = pickle.load(fp)
        fp.close()
        return TransectTimeSeries(transects=transects)

    def fetch_by_date(self,dnum):
        ti = np.searchsorted(self.threshold_dnums,dnum)
        return self.transects[ti]
        
def isolate_downcasts(xy,times,z,scalar,z_surface='auto',min_cast_samples=10):
    if z_surface == 'auto':
        z_surface = percentile(z,95) - 0.3 # 0.3m slush factor
    dz = np.concatenate( ([0],diff(z)) )
    mask = ((z<z_surface) & (dz<0))

    ## divide into individual casts
    cast_starts = np.nonzero( mask[1:] & (~mask[:-1]))[0]
    cast_ends = np.nonzero( mask[:-1] & (~mask[1:]))[0]
    # remove partials at start and end
    if mask[0]:
        cast_ends = cast_ends[1:]
    if mask[-1]:
        cast_starts = cast_starts[:-1]

    cast_Nsamples = cast_ends - cast_starts
    valid_casts = cast_Nsamples > min_cast_samples

    
def cast_timeseries_to_transect(xy,times,z,scalar,z_surface='auto',min_cast_samples=10):
    """
    Take timeseries from a bunch of casts, produce a transect object.
    For a timeseries with N observations:
    xy: [N,2]
    times: [N]
    z: [N] - positive up
    scalar: [N]

    options:
    z_surface: if auto, try to figure out the surface elevation.  used to remove
       points while instrument is at surface, and separate casts
    min_cast_samples: casts with fewer than this number of samples are discarded
    """
    if z_surface == 'auto':
        z_surface = np.percentile(z,95) - 0.3 # 0.3m slush factor
    dz = np.concatenate( ([0],diff(z)) )
    mask = ((z<z_surface) & (dz<0))

    ## divide into individual casts
    cast_starts = np.nonzero( mask[1:] & (~mask[:-1]))[0]
    cast_ends = np.nonzero( mask[:-1] & (~mask[1:]))[0]
    # remove partials at start and end
    if mask[0]:
        cast_ends = cast_ends[1:]
    if mask[-1]:
        cast_starts = cast_starts[:-1]

    cast_Nsamples = cast_ends - cast_starts
    valid_casts = cast_Nsamples > min_cast_samples
    cast_starts = cast_starts[valid_casts]
    cast_ends = cast_ends[valid_casts]
    max_cast_samples = cast_Nsamples.max()

    Ncasts = len(cast_starts)

    # repackage scalar and z into MxN array
    scalar_mn = np.zeros( (Ncasts,max_cast_samples), scalar.dtype)
    z_mn = np.zeros( (Ncasts,max_cast_samples), float64)
    scalar_mn[...] = nan
    z_mn[...] = nan
    times_m = np.zeros(Ncasts, np.float64)
    xy_m = np.zeros( (Ncasts,2), np.float64)
    for c,(s,e)in enumerate(zip(cast_starts,cast_ends)):
            scalar_mn[c,:e-s] = scalar[s:e]
            z_mn[c,:e-s] = z[s:e]
            times_m[c] = times[s]
            xy_m[c,:] = xy[s]
    tr = Transect(xy=xy_m,times=times_m,elevations=z_mn.T,scalar=scalar_mn.T)
    return tr

