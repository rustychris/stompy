"""
Methods for resampling unstructured, 3D hydro to a regular grid,
with user-defined integration in the vertical.
"""
import os
import hashlib
import logging

import six
import six.moves.cPickle as pickle

log=logging.getLogger('sample_to_rectilinear')

import numpy as np

from . import ugrid
from ..spatial import field

class Structurer(object):
    """ Spatial average and interpolate unstructured data
    to a structured grid
    """
    # simple caching - to make it easier for outside scripts to submit
    # separate requests for u and v, but we calculate them only once
    last_calculated_variable = (None,None) # (tindex, var_name)
    last_calculated_data = None

    # if no data points fall within the query radius, then we fallback
    # to choosing the nearest N points and averaging
    fallback_interpolate_N = 4
    cache_regions = True

    mask_land = True

    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

        self.prepare()

    def set_regions(self,utm,radii):
        """ Set the centers and radii for the averages.
        utm: [...,2]
        radii: [...] -- if radii is a scalar, it will be repeated for all points.

        """
        log.info("Setting region")
        self.utm_shaped = utm

        if np.asarray(radii).ndim==0:
            radii = radii*np.ones_like(utm[...,0])
        self.radii_shaped=radii

        cache_fields = ['utm','radii','processors','cell_infos','proc_to_index']

        if self.cache_regions:
            log.info("Checking for cached regions:")
            # hash utm and radii - then see if there's a matching pickle, which
            # should contain
            # self.{utm,radii,processors,cell_infos,proc_to_index,
            region_hash = hashlib.md5()
            region_hash.update( utm.tostring() )
            region_hash.update( radii.tostring() )
            # include a notion of the grid geometry too
            region_hash.update( self.vcf.X.tobytes() )
            cache_basename = "region_"+region_hash.hexdigest()+".bin"
            log.info("Cache filename is %s"%cache_basename)
        else:
            cache_basename = None

        if cache_basename:
            cache_dirs = self.cache_directories()
            for datadir in cache_dirs:
                cache_fn = os.path.join(datadir,cache_basename)
                if os.path.exists(cache_fn):
                    log.info("YES - will load that cache file!")
                    try:
                        fp = open(cache_fn,'rb')
                        data = pickle.load(fp)
                        for cf in cache_fields:
                            setattr(self,cf,data[cf])
                        fp.close()
                        return
                    except Exception as exc:
                        log.exception("Failed to load cached region info:")
            self.cache_fn = cache_fn = os.path.join(cache_dirs[0],cache_basename)

        # preprocess that some - figure out which processors are relevant

        # easier to deal with straight arrays
        self.utm = utm.reshape([-1,2])
        self.radii = radii.ravel()
        self.processors = set()

        grid = self.global_grid()

        # struct array to hold the metadata
        self.cell_infos = np.zeros( len(self.radii), dtype=[('masked',np.bool_),
                                                            # count of all contributors - for averaging
                                                            ('count',np.int32),
                                                            # hash proc=>array of local indexes
                                                            ('locals',object)])
        # maps a processor to a list of output indexes which use it.
        self.proc_to_index = [ [] for p in range(self.Nsubdomains())]

        self.cell_infos['masked'] = False

        if self.mask_land:
            # quick sweep of exterior
            grid_poly=grid.boundary_polygon()
            ext_xy=np.array(grid_poly.exterior)
            from matplotlib.path import Path
            p=Path(ext_xy)
            # doesn't care about order.  on the line counts as outside
            wet=p.contains_points(self.utm[:,:])
            self.cell_infos['masked']=~wet

        for i in range(len(self.radii)):
            if self.cell_infos['masked'][i]: continue

            if i % 1000 == 0:
                log.info("Finding averaging inputs for %d/%d "%(i,len(self.radii)))

            # Find all cells within the given radius:
            P = self.utm[i]
            if self.mask_land:
                # mask points which aren't really on water.
                # with the patch check above, this is just for islands.
                c = grid.select_cells_nearest(P,inside=True)
                if c is None:
                    self.cell_infos[i]['masked'] = True
                    continue

            # what processors are touched by this region?
            selected = self.vcf.within_r(P,self.radii[i])
            # always get at least the fallback interpolation
            # count
            if len(selected) < self.fallback_interpolate_N:
                # fall back to nearest points
                extras = self.vcf.nearest(P,count = self.fallback_interpolate_N )
                sel=list(selected)
                for n in extras:
                    if n not in sel:
                        sel.append(n)
                    if len(sel)>=self.fallback_interpolate_N:
                        break
                selected=np.array(sel)

            my_g2l = self.global_to_local[selected]
            my_procs = set( my_g2l['proc'] )
            self.cell_infos['count'][i] = len(my_g2l)
            self.cell_infos['locals'][i] = {}
            for p in my_procs:
                self.proc_to_index[p].append(i)
                self.cell_infos['locals'][i][p] = my_g2l['local'][ my_g2l['proc'] == p]

        if cache_fn:
            log.info("Saving that to a cache")
            fp = open(cache_fn,'wb')
            to_cache = {}
            for cf in cache_fields:
                to_cache[cf] = getattr(self,cf)
            pickle.dump(to_cache,fp)
            fp.close()

    def process(self,tindex,var_name):
        """
        tindex: integer indexing sun.timeline()

        var_name: for now, each variable is handled somewhat manually.
          supported variables:
            U_top1m - returns u,v components of current in top 1m of water column.
            u_wind, v_wind - u,v components of wind, read from raw output files
              like wind_u-time_step-cell.raw.0
            NEXT: h - seasurface height
        """
        this_variable = (tindex,var_name)
        if self.last_calculated_variable == this_variable:
            return self.last_calculated_data

        if var_name == 'U_top1m':
            # will have to get reshaped later...
            self.var_data = meanU = np.zeros( (len(self.utm),2), np.float64 )
        elif var_name in ['u_wind','v_wind','eta']:
            self.var_data = mean_scal = np.zeros( len(self.utm), np.float64 )
        else:
            raise Exception("unsupported variable requested: %s"%var_name)

        grid = self.global_grid()

        for proc in range(self.Nsubdomains()):
            my_output_indexes = self.proc_to_index[proc]
            if len(my_output_indexes) == 0:
                log.info("no outputs from processor %d"%proc)
                continue

            if var_name == 'U_top1m':
                weights = self.averaging_weights(proc=proc,time_step=tindex,ztop=0,dz=1.0)
                # problem: weights is cell, layer, but U is layer, cell
                # fixed - it was missing metadata on dv.
                U = self.cell_velocity(processor=proc,time_step = tindex)
                
                # this does the vertical averaging in each water column
                U[ weights==0 ] = 0.0 # avoid nan contamination
                data_for_proc = np.sum( U[:,:,:2] * weights[:,:,None],axis=1)
                print("data_for_proc %d finite %d nan"%
                      (np.isfinite(data_for_proc).sum(),
                       np.isnan(data_for_proc).sum()))
            elif var_name=='eta':
                data_for_proc = self.read_cell_scalar(label='eta',processor=proc,time_step = tindex)
            elif var_name in ['u_wind','v_wind']:
                if var_name == 'u_wind':
                    label = 'wind_u'
                else:
                    label = 'wind_v'
                data_for_proc = self.read_cell_scalar(label=label,processor=proc,time_step = tindex)
            for i in my_output_indexes:
                # array of local indices for this output, on this processor
                local_i = self.cell_infos[i]['locals'][proc]
                self.var_data[i] += np.sum(data_for_proc[local_i],axis=0)

        # to avoid division by zero - division by nan doesn't print warnings
        count=self.cell_infos['count'].astype('f8')
        count[ count==0 ] = np.nan

        # be careful that this may be vector data, or scalar..
        if self.var_data.ndim == 1:
            self.var_data /= count[:]
        elif self.var_data.ndim == 2:
            self.var_data /= count[:,None]
        else:
            raise Exception("really? tensors?")

        # correct shape
        correct_shape = list(self.radii_shaped.shape) + list(self.var_data.shape)[1:]
        self.var_data = self.var_data.reshape(correct_shape)

        # cache for repetitive calls
        self.last_calculated_variable = this_variable
        self.last_calculated_data = self.var_data

        return self.var_data

class UgridAverager(Structurer):
    """
    Uses the first mesh name found in the file
    """
    cache_dirs=None
    def __init__(self,nc_fns,**kwargs):
        self.nc_fns = nc_fns
        self._global_grid=kwargs.pop('global_grid',None)
        Structurer.__init__(self,**kwargs)

    def prepare(self):
        log.info("Loading individual ugrids")
        # not enough metadata or smarts to figure out some of these
        # names
        self.ugrids=[]
        self.grids=[]

        for nc_fn in self.nc_fns:
            ug = ugrid.UgridXr(nc_fn,
                               face_eta_vname='eta',
                               face_u_vname='uc',
                               face_v_vname='vc')
            ug.nc.z_r.attrs['positive']='down'
            ug.nc.z_w.attrs['positive']='down'
            ug.nc.dv.attrs['positive']='down'
            self.ugrids.append(ug)
            self.grids.append(ug.grid)

        if self._global_grid is None:
            gg=self.grids[0].copy()
            for g in self.grids[1:]:
                gg.add_grid(g,merge_nodes='auto')
            self._global_grid=gg
        else:
            gg=self._global_grid

        # this should be on the *global grid*
        vc = gg.cells_center()
        self.vcf = field.XYZField(X=vc,F=np.arange(len(vc)))
        self.vcf.build_index()

        self.global_to_local = np.zeros( gg.Ncells(),
                                         dtype = [('global',np.int32),
                                                  ('proc',np.int32),
                                                  ('local',np.int32)])
        self.global_to_local['global'][:]=np.arange(gg.Ncells())
        self.global_to_local['proc'][:]=-1

        for p,g in enumerate(self.grids):
            gcc=g.cells_center()
            for c,xy in enumerate(gcc):
                global_c=gg.select_cells_nearest(xy,inside=True)
                if global_c is None: continue
                if self.global_to_local['proc'][global_c]>=0: continue
                self.global_to_local['proc'][global_c]=p
                self.global_to_local['local'][global_c]=c

    def cache_directories(self):
        """ return list, in anti-chronological order, of possible
        directories holding the cached regions.
        the first element should be where any new cache files should go.
        """
        if self.cache_dirs is None:
            self.cache_dirs=[ os.path.dirname(self.nc_fn) ]
        return self.cache_dirs

    _global_grid=None
    def global_grid(self):
        if self._global_grid is None:
            raise Exception("Need some code to calc global grid or pass it in")
        return self._global_grid

    def Nsubdomains(self):
        return len(self.ugrids)

    def averaging_weights(self,proc,time_step,ztop=None,dz=None,zbottom=None):
        return self.ugrids[proc].vertical_averaging_weights(time_slice=time_step,ztop=ztop,dz=dz,zbottom=zbottom)

    def cell_velocity(self,processor,time_step):
        return self.ugrids[processor].get_cell_velocity(time_step)

    def read_cell_scalar(self,label,processor,time_step):
        return self.ugrids[processor].get_cell_scalar(label=label,time_step=time_step)

    def datenum(self,time_step):
        """ should be referenced to UTC
        """
        return self.available_steps()[time_step]

    def available_steps(self):
        return self.ugrids[0].times()
