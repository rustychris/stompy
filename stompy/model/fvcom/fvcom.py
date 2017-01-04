import netCDF4
import memoize as memo
import numpy as np
import trigrid
import pytz

class FvcomNC(object):
    tz='US/Eastern'
    tzvalue='-0500'
    
    def __init__(self,nc_file,**kwargs):
        self.nc_file = nc_file
        self.nc=netCDF4.Dataset(self.nc_file)
        self.__dict__.update(kwargs)

    # pickle protocol interface - since Dataset cannot be pickled.
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['nc']
        return d
    def __setstate__(self,d):
        self.__dict__.update(d)
        self.nc=netCDF4.Dataset(self.nc_file)
                    
    @property
    @memo.memoize()
    def time(self):
        nc_t_var=self.nc.variables['time']
        units,origin = nc_t_var.units.split(" since ")
        if origin[-1] in '0123456789':
            origin = origin + self.tzvalue
        tzero = np.datetime64(origin,'s')

        days=self.nc.variables['Itime'][:] * np.timedelta64(1,'D')
        msecs=self.nc.variables['Itime2'][:]*np.timedelta64(1,'ms')
        t=tzero+days+msecs
        return t

    @property
    @memo.memoize()
    def points(self):
        return np.array( [self.nc.variables['x'][:],
                          self.nc.variables['y'][:]], dtype='f8' ).T
    @property
    @memo.memoize()
    def centers(self):
        return np.array( [self.nc.variables['xc'][:],
                          self.nc.variables['yc'][:]] ).T
    @property
    @memo.memoize()
    def trigrid(self):
        cells=self.nc.variables['nv'][:].T - 1
        points=self.points.copy()
        g=trigrid.TriGrid(points=points,cells=cells)
        areas=g.areas()
        sel=np.nonzero(areas<0)[0]
        cells[sel,:] = cells[sel,::-1]

        # make a clean one with the new cells
        g=trigrid.TriGrid(points=points,cells=cells)
        g.make_edges_from_cells()
        return g

    # convenience for getting cell centered mean quantities
    @property
    @memo.memoize()
    def hc(self):
        h=self.nc.variables['h'][:]
        return h[self.trigrid.cells].mean(axis=1)

    @memo.memoize(lru=2)
    def zetac(self,tidx):
        zeta=self.nc.variables['zeta'][tidx]
        return zeta[self.trigrid.cells].mean(axis=1)

    @memo.memoize(lru=2)
    def uvw(self,tidx,c=slice(None),sigma_layer=slice(None)):
        u=self.nc.variables['u'][tidx,sigma_layer,c] 
        v=self.nc.variables['v'][tidx,sigma_layer,c] 
        w=self.nc.variables['ww'][tidx,sigma_layer,c]
        
        return np.concatenate( (u[...,None],v[...,None],w[...,None]), axis=-1 )

    @memo.memoize(lru=2)
    def to_sigma_layer(self,tidx,c,X):
        """ 0-based sigma layer index
        """
        z=X[2]
        h=self.hc[c]
        zeta=self.zetac(tidx)[c]

        # [-1,0]
        sigma=(z-(-h))/(zeta-(-h)) - 1
        siglevels=self.nc.variables['siglev'][:,self.trigrid.cells[c]].mean(axis=1)
        layer=np.searchsorted(-siglevels,-sigma).clip(0,len(siglevels)-2)
        return layer
