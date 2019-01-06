"""
Automate parts of setting up a DFlow hydro model.

TODO:
  allow for setting grid bathy from the model instance
"""
import os,shutil,glob,inspect
import six
import logging
log=logging.getLogger('DFlowModel')

import copy

import numpy as np
import xarray as xr
from shapely import geometry

import stompy.model.delft.io as dio
from stompy import xr_utils
from stompy.io.local import noaa_coops, hycom
from stompy import utils, filters, memoize
from stompy.spatial import wkb2shp, proj_utils
from stompy.model.delft import dfm_grid
import stompy.grid.unstructured_grid as ugrid

from . import io as dio

class BC(object):
    name=None
    _geom=None
    # set geom_type in subclasses to limit the matching geometries
    # to just 'Point', 'LineString', etc.   Avoids conflicts if
    # there are multiple features with the same name
    geom_type=None

    # not sure if I'll keep these -- may be better to query at time of use
    grid_edge=None
    grid_cell=None

    # some BCs allow 'add', which just applies a delta to a previously
    # set BC.
    mode='overwrite'

    # extend the data before/after the model period by this much
    pad=np.timedelta64(24,'h')

    def __init__(self,name,model=None,**kw):
        """
        Create boundary condition object.  Note that no work should be done
        here, as the rest of the model data is not yet in place, and this
        instance does not even have access yet to its geometry or other
        shapefile attributes.  model should either be passed in, or assigned
        immediately by caller, since most later steps rely on access to a model
        object.
        """
        self.model=model # may be None!
        self.name=name
        self.filters=[]

        utils.set_keywords(self,kw)
        # above line should replace this stanza:
        #   for k in kw:
        #       try:
        #           getattr(self,k)
        #       except AttributeError:
        #           raise Exception("Setting attribute %s failed because it doesn't exist on %s"%(k,self))
        #       self.__dict__[k]=kw[k]

        for f in self.filters:
            f.setup(self)

    # A little goofy - the goal is to make geometry lazily
    # fetched against the model gazetteer, but it makes
    # get/set operations awkward
    @property
    def geom(self):
        if (self._geom is None) and (self.model is not None):
            kw={}
            if self.geom_type is not None:
                kw['geom_type']=self.geom_type
            self._geom=self.model.get_geometry(name=self.name,**kw)
        return self._geom
    @geom.setter
    def geom(self,g):
        if isinstance(g,np.ndarray):
            if g.ndim==1:
                g=geometry.Point(g)
            elif g.ndim==2:
                g=geometry.LineString(g)
            else:
                raise Exception("Not sure how to convert %s to a shapely geometry"%g)
        self._geom=g

    # Utilities for specific types of BCs which need more information
    # about the grid
    def get_inward_normal(self,grid_edge=None):
        """
        Query the grid based on self.grid_edge to find the unit
        normal vector for this velocity BC, positive pointing into
        the domain.
        """
        if grid_edge is None:
            grid_edge=self.grid_edge
        assert grid_edge is not None
        return self.model.grid.edges_normals(grid_edge,force_inward=True)
    def get_depth(self,grid_edge=None):
        """
        Estimate the water column depth associated with this BC.
        This is currently limited to a constant value, calculated for
        self.grid_edge.
        For the purposes here, this is a strictly positive quantity.
        """
        if grid_edge is None:
            grid_edge=self.grid_edge
        assert grid_edge is not None

        # This feels like it should be somewhere else, maybe in DFlowModel?
        h=-self.model.edge_depth(self.grid_edge,datum='eta0')
        if h<=0:
            log.warning("Depth for velocity BC is %f, should be >0"%h)
        return h

    # Below are more DFM specific methods which have not yet been
    # refactored
    def write(self):
        log.info("Writing feature: %s"%self.name)

        self.write_pli()
        self.write_config()
        self.write_data()

    def write_config(self):
        log.warning("Boundary condition '%s' has no write_config method"%self.name)
    def write_data(self):
        log.warning("Boundary condition '%s' has no write_data method"%self.name)

    def filename_base(self):
        """
        filename base (no extension, relative to model run_dir) used to construct
        other filenames.
        """
        return self.name

    def pli_filename(self):
        """
        Name of polyline file, relative to model run_dir
        """
        return self.filename_base() + '.pli'

    def write_pli(self):
        if self.geom is not None:
            assert self.geom.type=='LineString'
            pli_data=[ (self.name, np.array(self.geom.coords)) ]
            pli_fn=os.path.join(self.model.run_dir,self.pli_filename())
            dio.write_pli(pli_fn,pli_data)

    def default_tim_fn(self):
        """
        full path for a time file matched to the first node of the pli.
        This is only used as a default tim output path when none is
        specified.
        """
        return os.path.join(self.model.run_dir,self.filename_base() + "_0001.tim")

    def default_t3d_fn(self):
        """
        same as above, but for t3d
        """
        return os.path.join(self.model.run_dir,self.filename_base() + "_0001.t3d")

    def write_tim(self,da,fn=None):
        """
        Write a DFM tim file based on the timeseries in the DataArray.
        da must have a time dimension.  No support yet for vector-values here.
        """
        ref_date,start,stop = self.model.mdu.time_range()
        dt=np.timedelta64(60,'s') # always minutes
        # self.model.mdu.t_unit_td64()
        elapsed_time=(da.time.values - ref_date)/dt 

        data=np.c_[elapsed_time,da.values]
        if fn is None:
            fn=self.default_tim_fn()

        np.savetxt(fn,data)

    def write_t3d(self,da,z_bed,fn=None):
        """
        Write a 3D boundary condition for a feature from a vertical profile (likely
           ROMS or HYCOM data)
         - most of the time writing boundaries is here
         - DFM details for rev52184:
             the LAYERS line is silently truncated to 100 characters.
             LAYER_TYPE=z assumes a coordinate of 0 at the bed, positive up

        we assume that the incoming data has no nan, has a positive-up
        z coordinate with 0 being model datum (i.e. NAVD88)
        """
        ref_date,t_start,t_stop = self.model.mdu.time_range()

        # not going to worry about 3D yet.  see ocean_dfm.py
        # for some hints.
        assert da.ndim==2

        # new code gets an xr dataset coming in with z coordinate.
        # old code did some cleaning on ROMS data.  no more.

        # Do sort the vertical
        dz=np.diff(da.z.values)
        if np.all(dz>0):
            log.debug("Vertical order ok")
        elif np.all(dz<0):
            log.debug("3D velo flip ertical order")
            da=da.isel(z=slice(None,None,-1))

        if np.median(da.z.values) > 0:
            log.warning("Weak sign check suggests t3d input data has wrong sign on z")

        max_line_length=100 # limitation in DFM on the whole LAYERS line
        # 7 is '_2.4567'
        # -1 for minor bit of safety
        max_layers=(max_line_length-len("LAYERS=")) // 7 - 1

        # This should be the right numbers, but reverse order
        # that's probably not right now...
        sigma = (z_bed - da.z.values) / z_bed

        # Force it to span the full water column
        # used to allow it to go slightly beyond, but
        # in trying to diagnose a 3D profile run in 52184, limit
        # to exactly 0,1
        # well, maybe that's not necessary -- before trying to do any resampling
        # here, maybe go ahead and let it span too far
        bed_samples=np.nonzero(sigma<=0)[0]
        surf_samples=np.nonzero(sigma>=1.0)[0]
        slc=slice(bed_samples[-1],surf_samples[0]+1)
        da=da.isel(z=slc)
        sigma=sigma[slc]
        sigma[0]=0.0 # min(0.0,sigma[0])
        sigma[-1]=1.0 # max(1.0,sigma[-1])

        assert np.all(np.diff(sigma)>0),"Need more sophisticated treatment of sigma in t3d file"
        assert len(sigma)<=max_layers

        #     remapper=lambda y: np.interp(np.linspace(0,1,max_layers),
        #                                  np.linspace(0,1,len(sigma)),y)
        #     # Just because the use of remapper below is not compatible
        #     # with vector quantities at this time.
        #     assert da_sub.ndim-1 == 1

        sigma_str=" ".join(["%.4f"%s for s in sigma])

        # This line is truncated at 100 characters in DFM r52184.
        layer_line="LAYERS=%s"%sigma_str
        assert len(layer_line)<max_line_length

        # NB: this is independent of the TUnit setting in the MDU, because
        # it is written out in the file (see below).
        elapsed_minutes=(da.time.values - ref_date)/np.timedelta64(60,'s')

        ref_date_str=utils.to_datetime(ref_date).strftime('%Y-%m-%d %H:%M:%S')

        if fn is None:
            fn=self.default_t3d_fn()

        assert da.dims[0]=='time' # for speed-up of direct indexing

        # Can copy this to other node filenames if necessary
        with open(fn,'wt') as fp:
            fp.write("\n".join([
                "LAYER_TYPE=sigma",
                layer_line,
                "VECTORMAX=%d"%(da.ndim-1), # default, but be explicit
                "quant=velocity",
                "quantity1=velocity", # why is this here?
                "# start of data",
                ""]))
            for ti,t in enumerate(elapsed_minutes):
                fp.write("TIME=%g minutes since %s\n"%(t,ref_date_str))
                # Faster direct indexing:
                # The ravel will interleave components - unclear if that's correct.
                data=" ".join( ["%.3f"%v for v in da.values[ti,:].ravel()] )
                fp.write(data)
                fp.write("\n")

    def as_data_array(self,data,quantity='value'):
        """
        Convert several types into a consistent DataArray ready to be
        post-processed and then written out.

        Conversion rules:
        dataarray => no change
        dataset => pull just the data variable, either based on quantity, or if there
          is a single data variable that is not a coordinate, use that.
        constant => wrap in a DataArray with no time dimension.
          used to create a two-point timeseries, but if that is needed it should be moved
          to model specific code.
        """
        if isinstance(data,xr.DataArray):
            data.attrs['mode']=self.mode
            return data
        elif isinstance(data,xr.Dataset):
            if len(data.data_vars)==1:
                da=data[data.data_vars[0]]
                da.attrs['mode']=self.mode
                return da
            else:
                raise Exception("Dataset has multiple data variables -- not sure which to use: %s"%( str(data.data_vars) ))
        elif isinstance(data,(np.integer,np.floating,int,float)):
            # # handles expanding a constant to the length of the run
            # ds=xr.Dataset()
            # ds['time']=('time',),np.array( [self.data_start,self.data_stop] )
            # ds[quantity]=('time',),np.array( [data,data] )
            # da=ds[quantity]
            da=xr.DataArray(data)
            da.attrs['mode']=self.mode
            return da
        else:
            raise Exception("Not sure how to cast %s to be a DataArray"%data)

    # Not all BCs have a time dimension, but enough do that we have some general utility
    # getters/setters at this level
    # Note that data_start, data_stop are from the point of view of the data source,
    # e.g. a model starting on 2015-01-01 could have a 31 day lag, such that
    # data_start is actually 2014-12-01.
    _data_start=None
    _data_stop =None
    @property
    def data_start(self):
        if self._data_start is None:
            return self.transform_time_input(self.model.run_start-self.pad)
        else:
            return self._data_start
    @data_start.setter
    def data_start(self,v):
        self._data_start=v

    @property
    def data_stop(self):
        if self._data_stop is None:
            return self.transform_time_input(self.model.run_stop+self.pad)
        else:
            return self._data_stop
    @data_stop.setter
    def data_stop(self,v):
        self._data_stop=v

    def transform_time_input(self,t):
        for filt in self.filters:
            t=filt.transform_time_input(t)
        return t
    def transform_output(self,da):
        """
        Apply filter stack to da, including model-based time zone
        correction of model is set.
        """
        for filt in self.filters[::-1]:
            da=filt.transform_output(da)
        da=self.to_model_timezone(da)
        return da
    def to_model_timezone(self,da):
        if 'time' in da.dims and self.model is not None:
            da.time.values[:]=self.model.utc_to_native(da.time.values)
        return da

    def src_data(self):
        raise Exception("src_data must be set in subclass")

    def data(self):
        da=self.src_data()
        da=self.as_data_array(da)
        da=self.transform_output(da)
        return da

    # if True, bokeh plot will include time series for intermediate
    # data as filters are applied
    bokeh_show_intermediate=True
    def write_bokeh(self,filename=None,path=".",title=None,mode='cdn'):
        """
        Write a bokeh html plot for this dataset.
        path: folder in which to place the plot.
        filename: relative or absolute filename.  defaults to path/{self.name}.html
        mode: this is passed to bokeh, 'cdn' yields small files but requires an internet
         connection to view them.  'inline' yields self-contained, larger (~800k) files.
        """
        import bokeh.io as bio # output_notebook, show, output_file
        import bokeh.plotting as bplt

        bplt.reset_output()

        if title is None:
            title="Name: %s"%self.name

        p = bplt.figure(plot_width=750, plot_height=350,
                        title=title,
                        active_scroll='wheel_zoom',
                        x_axis_type="datetime")

        if self.bokeh_show_intermediate:
            da=self.src_data()
            self.plot_bokeh(da,p,label="src")
            for filt in self.filters[::-1]:
                da=filt.transform_output(da)
                self.plot_bokeh(da,p,label=filt.label())
            da=self.to_model_timezone(da)
            self.plot_bokeh(da,p)
        else:
            da=self.data()
            self.plot_bokeh(da,p)
        if filename is None:
            filename="bc_%s.html"%self.name
        output_fn=os.path.join(path,filename)
        bio.output_file(output_fn,
                        title=title,
                        mode=mode)
        bio.save(p) # show the results

    #annoying, but bokeh not cycle colors automatically
    _colors=None
    def get_color(self):
        if self._colors is None:
            from bokeh.palettes import Dark2_5 as palette
            import itertools
            self._colors=itertools.cycle(palette)
        return six.next(self._colors)
    def plot_bokeh(self,da,plot,label=None):
        """
        Generic plotting implementation -- will have to override for complicated
        datatypes
        """
        plot.yaxis.axis_label = da.attrs.get('units','n/a')
        if label is None:
            label=self.name
        if 'time' in da.dims:
            plot.line( da.time.values.copy(), da.values.copy(), legend=label,
                       color=self.get_color())
        else:
            from bokeh.models import Label
            label=Label(x=70, y=70, x_units='screen', y_units='screen',
                        text="No plotting for %s (%s)"%(label,self.__class__.__name__))
            plot.add_layout(label)

class BCFilter(object):
    """
    Transformation/translations that can be applied to
    a BC
    """
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
    def setup(self,bc):
        """
        This is where you might increase the pad
        """
        self.bc=bc
    def transform_time_input(self,t):
        """
        Transform the externally requested time to what the data source
        should provide
        """
        return t
    def transform_output(self,da):
        """
        Whatever dataarray comes back from the source, apply the necessary
        transformations (including the inverse of the time_input transform)
        """
        return da
    def label(self):
        return self.__class__.__name__

class LowpassGodin(BCFilter):
    min_pad=np.timedelta64(5*24,'h')
    def setup(self,bc):
        super(LowpassGodin,self).setup(bc)
        if self.bc.pad<self.min_pad:
            self.bc.pad=self.min_pad
    def transform_output(self,da):
        assert da.ndim==1,"Only ready for simple time series"
        from ... import filters
        da.values[:]=filters.lowpass_godin(da.values,
                                           utils.to_dnum(da.time))
        return da

class Lowpass(BCFilter):
    cutoff_hours=None
    # if true, replace any nans by linear interpolation, or
    # constant extrapolation at ends
    fill_nan=True
    def transform_output(self,da):
        assert da.ndim==1,"Only ready for simple time series"
        from ... import filters
        assert self.cutoff_hours is not None,"Must specify lowpass threshold cutoff_hors"
        dt_h=24*np.median(np.diff(utils.to_dnum(da.time.values)))
        log.info("Lowpass: data time step is %.2fh"%dt_h)
        data_in=da.values

        if np.any(~np.isfinite(data_in)):
            if self.fill_nan:
                log.info("Lowpass: %d of %d data values will be filled"%( np.sum(~np.isfinite(data_in)),
                                                                          len(data_in) ))
                data_in=utils.fill_invalid(data_in,ends='constant')
            else:
                log.error("Lowpass: %d of %d data values are not finite"%( np.sum(~np.isfinite(data_in)),
                                                                           len(data_in) ))
        da.values[:]=filters.lowpass(data_in,cutoff=self.cutoff_hours,dt=dt_h)

        assert np.all(np.isfinite(da.values)),("Lowpass: %d of %d output data values are not finite"%
                                               ( np.sum(~np.isfinite(da.values)),
                                                 len(da.values) ))
        return da

class Lag(BCFilter):
    def __init__(self,lag):
        self.lag=lag
    def transform_time_input(self,t):
        return t+self.lag
    def transform_output(self,da):
        da.time.values[:]=da.time.values-self.lag
        return da
class Transform(BCFilter):
    def __init__(self,fn):
        self.fn=fn
    def transform_output(self,da):
        da.values[:]=self.fn(da.values)
        return da

class RoughnessBC(BC):
    shapefile=None
    data_array=None # xr.DataArray
    def __init__(self,shapefile=None,**kw):
        if 'name' not in kw:
            kw['name']='roughness'

        super(RoughnessBC,self).__init__(**kw)
        self.shapefile=shapefile
    def write_config(self):
        with open(self.model.ext_force_file(),'at') as fp:
            lines=["QUANTITY=frictioncoefficient",
                   "FILENAME=%s"%self.xyz_filename(),
                   "FILETYPE=7",
                   "METHOD=4",
                   "OPERAND=O",
                   "\n"
                   ]
            fp.write("\n".join(lines))

    def xyz_filename(self):
        return self.filename_base()+".xyz"

    def src_data(self):
        if self.shapefile is not None:
            shp_data=wkb2shp.shp2geom(self.shapefile)
            coords=np.array( [np.array(pnt) for pnt in shp_data['geom'] ] )
            n=shp_data['n']
            da=xr.DataArray(n,dims=['location'],name='n')
            da=da.assign_coords(x=xr.DataArray(coords[:,0],dims='location'))
            da=da.assign_coords(y=xr.DataArray(coords[:,1],dims='location'))
            da.attrs['long_name']='Manning n'
        elif self.data_array is not None:
            da=self.data_array
                
        return da

    def write_data(self):
        data_fn=os.path.join(self.model.run_dir,self.xyz_filename())
        xyz=self.data()
        np.savetxt(data_fn,xyz)

    def write_bokeh(self,filename=None,path=".",title=None,mode='cdn'):
        """
        Write a bokeh html plot for this dataset.  RoughnessBC has specific
        needs here.
        path: folder in which to place the plot.
        filename: relative or absolute filename.  defaults to path/{self.name}.html
        mode: this is passed to bokeh, 'cdn' yields small files but requires an internet
         connection to view them.  'inline' yields self-contained, larger (~800k) files.
        """
        import bokeh.io as bio # output_notebook, show, output_file
        import bokeh.plotting as bplt

        bplt.reset_output()

        if title is None:
            title="Name: %s"%self.name

        p = bplt.figure(plot_width=750, plot_height=750,
                        title=title,
                        active_scroll='wheel_zoom')
        p.match_aspect=True # aiming for analog to axis('equal')

        da=self.data()
        self.plot_bokeh(da,p)
        if filename is None:
            filename="bc_%s.html"%self.name
        output_fn=os.path.join(path,filename)
        bio.output_file(output_fn,
                        title=title,
                        mode=mode)
        bio.save(p) # save the results

    def plot_bokeh(self,da,plot,label=None):
        if label is None:
            label=self.name
        rough=da.values

        from bokeh.models import LinearColorMapper,ColorBar

        color_mapper=LinearColorMapper(palette="Viridis256",
                                       low=rough.min(), high=rough.max())
        from matplotlib import cm
        cmap=cm.viridis
        norm_rough=(rough-rough.min())/(rough.max()-rough.min())
        mapped=[cmap(v) for v in norm_rough]
        colors = [
            "#%02x%02x%02x" % (int(m[0]*255),
                               int(m[1]*255),
                               int(m[2]*255))
            for m in mapped ]

        plot.scatter(da.x.values.copy(), da.y.values.copy(), radius=3,
                     fill_color=colors, line_color=None,legend=label)

        color_bar = ColorBar(color_mapper=color_mapper, 
                             label_standoff=12, border_line_color=None, location=(0,0))
        plot.add_layout(color_bar, 'right')

class StageBC(BC):
    # If other than None, can compare to make sure it's the same as the model
    # datum.
    datum=None
    geom_type='LineString'

    def __init__(self,z=None,**kw):
        super(StageBC,self).__init__(**kw)
        self.z=z

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=waterlevelbnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

    def filename_base(self):
        """
        Make it clear in the filenames what is being forced
        """
        return super(StageBC,self).filename_base()+"_ssh"

    def src_data(self):
        return self.z

    def write_data(self):
        # just write a single node
        self.write_tim(self.data())

class FlowBC(BC):
    dredge_depth=-1.0
    Q=None
    geom_type='LineString'

    def __init__(self,Q=None,**kw):
        super(FlowBC,self).__init__(**kw)
        self.Q=Q

    def filename_base(self):
        return super(FlowBC,self).filename_base()+"_Q"

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=dischargebnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

    def write_pli(self):
        super(FlowBC,self).write_pli()

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for flow BC %s"%self.name)
            dfm_grid.dredge_boundary(self.model.grid,
                                     np.array(self.geom.coords),
                                     self.dredge_depth)
        else:
            log.info("Dredging disabled")

    def src_data(self):
        # probably need some refactoring here...
        return self.Q

    def write_data(self):
        self.write_tim(self.data())

class SourceSinkBC(BC):
    # The grid, at the entry point, will be taken down to this elevation
    # to ensure that prescribed flows are not prevented due to a dry cell.

    # could allow this to come in as a point, though it is probably not
    # supported in the code below at this point.
    geom_type=None
    z='bed'

    dredge_depth=-1.0
    def __init__(self,Q=None,**kw):
        """
        Q: one of:
          a constant value in m3/s
          an xarray DataArray with a time index.
        """
        super(SourceSinkBC,self).__init__(**kw)
        self.Q=Q

    def filename_base(self):
        return super(SourceSinkBC,self).filename_base()+"_Q"

    def write_config(self):
        assert self.Q is not None

        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=discharge_salinity_temperature_sorsin",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=1", # how is this different than method=3?
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

    def write_pli(self):
        super(SourceSinkBC,self).write_pli()

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for flow BC %s"%self.name)
            dfm_grid.dredge_discharge(self.model.grid,
                                      np.array(self.geom.coords),
                                      self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_data(self):
        self.write_tim(self.data())
    def src_data(self):
        assert self.Q is not None
        return self.Q

class WindBC(BC):
    """
    Not yet fully updated
    """
    wind=None
    def __init__(self,**kw):
        if 'name' not in kw:
            # commonly applied globally, so may not have a geographic name
            kw['name']='wind'
        super(WindBC,self).__init__(**kw)
    def write_pli(self):
        assert self.geom is None,"Spatially limited wind not yet supported"
        return # nothing to do

    def default_tim_fn(self):
        # different than super class because typically no nodes
        return os.path.join(self.model.run_dir,self.filename_base() + ".tim")

    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=windxy",
                   "FILENAME=%s.tim"%self.filename_base(),
                   "FILETYPE=2",
                   "METHOD=1",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))
    def write_data(self):
        self.write_tim(self.data())
    def src_data(self):
        assert self.wind is not None
        return self.wind
    def plot_bokeh(self,da,plot,label=None):
        # this will have to get smarter time...
        # da will almost certainly have an xy dimension for the two components.
        # for now, we assume no spatial variation, and plot two time series
        if label is None:
            label=self.name
        for xy in [0,1]:
            plot.line( da.time.values.copy(),
                       da.isel(xy=xy).values.copy(),
                       legend=label+"-"+"xy"[xy],
                       color=self.get_color())

class ScalarBC(BC):
    scalar=None
    value=None
    def __init__(self,**kw):
        """
        name: feature name
        model: HydroModel instance
        scalar: 'salinity','temperature', other
        value: floating point
        """
        super(ScalarBC,self).__init__(**kw)
    def src_data(self):
        # Base implementation does nothing
        return self.value

class VerticalCoord(object):
    """
    A placeholder for now, but potentially a place to describe the
    vertical coordinate structure
    """
    pass

class SigmaCoord(VerticalCoord):
    sigma_growth_factor=1


class HydroModel(object):
    mpi_bin_dir=None # same, but for mpiexec.  None means use dfm_bin_dir
    mpi_bin_exe='mpiexec'
    num_procs=1
    run_dir="." # working directory when running dflowfm
    cache_dir=None

    run_start=None
    run_stop=None

    mdu_basename='flowfm.mdu'

    mdu=None
    grid=None

    projection=None # string like "EPSG:26910"
    z_datum=None

    # this is only used for setting utc_to_native, and native_to_utc
    utc_offset=np.timedelta64(0,'h') # -8 for PST

    def __init__(self):
        self.log=log
        self.bcs=[]
        self.extra_files=[]
        self.gazetteers=[]

        self.mon_sections=[]
        self.mon_points=[]

    def add_extra_file(self,path,copy=True):
        self.extra_files.append( (path,copy) )

    def write_extra_files(self):
        for f in self.extra_files:
            path,copy = f
            if copy:
                tgt=os.path.join( self.run_dir, os.path.basename(path))
                if not (os.path.exists(tgt) and os.path.samefile(tgt,path)):
                    shutil.copyfile(path,tgt)
                else:
                    log.info("Extra file %s points to the target.  No-op"%path)

    def copy(self,deep=True):
        """
        Make a copy of this model instance.
        """
        # Starting point is just python deepcopy, but can customize
        # as needed.
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k in ['log']: # shallow for some object
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def create_with_mode(self,path,mode='create'):
        """
        path: absolute, or relative to pwd
        mode: 'create'  create the folder if it doesn't exist
         'pristine' create, and clear anything already in there
         'noclobber' create, and fail if it already exists.
         'existing' assert that the path exists, but do nothing to it.
        """
        if mode=='create':
            if not os.path.exists(path):
                os.makedirs(path)
        elif mode=='pristine':
            if os.path.exists(path):
                # shutil.rmtree(path)
                # rather than going scorched earth, removed the contents of
                # the directory.  this plays nicer with processes which
                # may be working in that directory.
                for p in os.listdir(path):
                    fp=os.path.join(path,p)
                    if os.path.isdir(fp):
                        shutil.rmtree(fp)
                    else:
                        os.unlink(fp)
            else:
                os.makedirs(path)
        elif mode=='noclobber':
            assert not os.path.exists(path),"Directory %s exists, but mode is noclobber"%path
            os.makedirs(path)
        elif mode=='askclobber':
            if os.path.exists(path):
                import sys
                sys.stdout.write("Directory %s exists.  overwrite? [y/n] "%path)
                sys.stdout.flush()
                resp=six.moves.input()
                if resp.lower()!='y':
                    raise Exception("Directory %s exists -- failing out"%path)
                return self.create_with_mode(path,'pristine')
            else:
                os.makedirs(path)
        elif mode=='existing':
            assert os.path.exists(path),"Directory %s does not exist"%path

    def set_run_dir(self,path,mode='create'):
        """
        Set the working directory for the simulation.
        See create_with_mode for details on 'mode' parameter.
        set_run_dir() supports an additional mode "clean",
        which removes files known to be created during the
        script process, as opposed to 'pristine' which deletes
        everything.
        """
        self.run_dir=path
        if mode=="clean":
            self.create_with_mode(path,"create")
            self.clean_run_dir()
        else:
            self.create_with_mode(path,mode)

    def clean_run_dir(self):
        """
        Clean out most of the run dir, deleting files known to be
        created by DFlowModel
        """
        patts=['*.pli','*.tim','*.t3d','*.mdu','FlowFM.ext','*_net.nc','DFM_*', '*.dia',
               '*.xy*','initial_conditions*','dflowfm-*.log']
        for patt in patts:
            matches=glob.glob(os.path.join(self.run_dir,patt))
            for m in matches:
                if os.path.isfile(m):
                    os.unlink(m)
                elif os.path.isdir(m):
                    shutil.rmtree(m)
                else:
                    raise Exception("What is %s ?"%m)

    def set_cache_dir(self,path,mode='create'):
        """
        Set the cache directory, mainly for BC data.
        See create_with_mode for details on 'mode' parameter
        """
        self.create_with_mode(path,mode)

    def set_grid(self,grid):
        if isinstance(grid,six.string_types):
            grid=dfm_grid.DFMGrid(grid)
        self.grid=grid

    default_grid_target_filename='grid_net.nc'
    def grid_target_filename(self):
        """
        The filename, relative to self.run_dir, of the grid.  Not guaranteed
        to exist, and if no grid has been set, or the grid has no filename information,
        this will default to self.default_grid_target_filename
        """
        if self.grid is None or self.grid.filename is None:
            return self.default_grid_target_filename
        else:
            return os.path.basename(self.grid.filename)

    def add_monitor_sections(self,sections):
        """
        sections: list or array of features.  each feature
        must have a 'geom' item giving the shapely geometry as a
        LineString.  the feature name is pulled from a 'name'
        item if it exists, otherwise 'obs_sec_NNN'
        """
        self.mon_sections.extend(sections)
    def add_monitor_points(self,points):
        """
        points: list or array of features, must have a 'geom' item giving
        the shapely geometry as a Point.  if there is a 'name' item,
        that will be used to name the feature, otherwise it will be given
        a numeric name 'obs_pnt_NNN'
        """
        self.mon_points.extend(points)

    def write(self):
        # Make sure instance data has been pushed to the MDUFile, this
        # is used by write_forcing() and write_grid()
        assert self.grid is not None,"Must call set_grid(...) before writing"
        self.update_config()
        log.info("Writing MDU to %s"%self.mdu.filename)
        self.write_config()
        self.write_extra_files()
        self.write_forcing()
        # Must come after write_forcing() to allow BCs to modify grid
        self.write_grid()

    def write_grid(self):
        raise Exception("Implement in subclass")
    def write_forcing(self):
        for bc in self.bcs:
            self.write_bc(bc)

    def write_bc(self,bc):
        if isinstance(bc,MultiBC):
            bc.enumerate_sub_bcs()
            for sub_bc in bc.sub_bcs:
                self.write_bc(sub_bc)
        else:
            raise Exception("BC type %s not handled by class %s"%(bc.__class__,self.__class__))

    def partition(self):
        if self.num_procs<=1:
            return

        cmd="--partition:ndomains=%d %s"%(self.num_procs,self.mdu['geometry','NetFile'])
        self.run_dflowfm(cmd)

        # similar, but for the mdu:
        gen_parallel=os.path.join(self.dfm_bin_dir,"generate_parallel_mdu.sh")
        cmd="%s %s %d 6"%(gen_parallel,os.path.basename(self.mdu.filename),self.num_procs)
        utils.call_with_path(cmd,self.run_dir)

    _dflowfm_exe=None
    @property
    def dflowfm_exe(self):
        if self._dflowfm_exe is None:
            return os.path.join(self.dfm_bin_dir,self.dfm_bin_exe)
        else:
            return self._dflowfm_exe
    @dflowfm_exe.setter
    def dflowfm_exe(self,v):
        self._dflowfm_exe=v
        
    def run_dflowfm(self,cmd):
        # Names of the executables
        dflowfm=os.path.join(self.dfm_bin_dir,"dflowfm")

        if self.num_procs>1:
            mpi_bin_dir=self.mpi_bin_dir or self.dfm_bin_dir
            mpiexec=os.path.join(mpi_bin_dir,"mpiexec")
            real_cmd="%s -n %d %s %s"%(mpiexec,self.num_procs,dflowfm,cmd)
        else:
            real_cmd="%s %s"%(dflowfm,cmd)

        self.log.info("Running command: %s"%real_cmd)
        utils.call_with_path(real_cmd,self.run_dir)

    def run_model(self):
        self.run_dflowfm(cmd="-t 1 --autostartstop %s"%os.path.basename(self.mdu.filename))

    def add_gazetteer(self,shp_fn):
        """
        Register a shapefile for resolving feature locations.
        shp_fn: string, to be loaded as shapefile, or a structure array with a geom field.
        """
        if not isinstance(shp_fn,np.ndarray):
            shp_fn=wkb2shp.shp2geom(shp_fn)
        self.gazetteers.append(shp_fn)
    def get_geometry(self,**kws):
        """
        The gazetteer interface for BC geometry.  given criteria as keyword arguments,
        i.e. name='Old_River', return the matching geometry from the gazetteer as
        a shapely geometry.
        if no match, return None.  Error if more than one match
        """
        hits=self.match_gazetteer(**kws)
        if hits:
            assert len(hits)==1
            return hits[0]['geom']
        else:
            return None
    def match_gazetteer(self,**kws):
        """
        search all gazetteers with criteria specified in keyword arguments,
        returning a list of shapefile records (note that this is a python
        list of numpy records, not a numpy array, since shapefiles may not
        have the same fields).
        return empty list if not hits
        """
        hits=[]
        for gaz in self.gazetteers:
            for idx in range(len(gaz)):
                if self.match_feature(kws,gaz[idx]):
                    hits.append( gaz[idx] )
        return hits
    def match_feature(self,kws,feat):
        """
        check the critera in dict kws against feat, a numpy record as
        returned by shp2geom.
        there is special handling for several values:
          'geom_type' is the geom_type attribute of the geometry itself,
          e.g. 'LineString' or 'Point'
        """
        for k in kws:
            if k=='geom_type':
                feat_val=feat['geom'].geom_type
            else:
                try:
                    feat_val=feat[k]
                except KeyError:
                    return False
                except ValueError: # depending on type of feat can get either
                    return False
            if feat_val==kws[k]:
                continue
            else:
                return False
        return True

    # some read/write methods which may have to refer to model state to properly
    # parse inputs.
    def read_bc(self,fn):
        """
        Read a new-style BC file into an xarray dataset
        """
        return dio.read_dfm_bc(fn)

    def read_tim(self,fn,time_unit=None,columns=['val1','val2','val3']):
        """
        Parse a tim file to xarray Dataset.  This needs to be a model method so
        that we know the units, and reference date.  Currently, this immediately
        reads the file, which may have to change in the future for performance
        or ease-of-use reasons.

        time_unit: 'S' for seconds, 'M' for minutes.  Relative to model reference
        time.

        returns Dataset with 'time' dimension, and data columns labeled according
        to columns.
        """
        if time_unit is None:
            # time_unit=self.mdu['time','Tunit']
            # always minutes, unless overridden by caller
            time_unit='M'

        ref_time,_,_ = self.mdu.time_range()
        return dio.read_dfm_tim(fn,time_unit=time_unit,
                                ref_time=ref_time,
                                columns=columns)

    def add_FlowBC(self,**kw):
        self.add_bcs(FlowBC(model=self,**kw))
    def add_SourceSinkBC(self,*a,**kw):
        self.add_bcs(SourceSinkBC(*a,model=self,**kw))
    def add_StageBC(self,**kw):
        self.add_bcs(StageBC(model=self,**kw))
    def add_WindBC(self,**kw):
        self.add_bcs(WindBC(model=self,**kw))
    def add_RoughnessBC(self,**kw):
        self.add_bcs(RoughnessBC(model=self,**kw))
    # def add_Structure(self,**kw): # only for DFM now.

    def add_bcs(self,bcs):
        """
        Add BC objects to this models definition.

        bcs: None (do nothing), one BC instance, or a list of BC instances
        """
        if bcs is None:
            return
        if isinstance(bcs,BC):
            bcs=[bcs]
        for bc in bcs:
            assert (bc.model is None) or (bc.model==self),"Not expecting to share BC objects"
            bc.model=self
        self.bcs.extend(bcs)

    def utc_to_native(self,t):
        return t+self.utc_offset
    def native_to_utc(self,t):
        return t-self.utc_offset

    @property
    @memoize.member_thunk
    def ll_to_native(self):
        """
        Project array of longitude/latitude [...,2] to
        model-native (e.g. UTM meters)
        """
        return proj_utils.mapper('WGS84',self.projection)

    @property
    @memoize.member_thunk
    def native_to_ll(self):
        """
        Project array of x/y [...,2] coordinates in model-native
        project (e.g. UTM meters) to longitude/latitude
        """
        return proj_utils.mapper(self.projection,'WGS84')

    # Some BC methods need to know more about the domain, so DFlowModel
    # provides these accessors
    def edge_depth(self,j,datum=None):
        """
        Return the bed elevation for edge j, in meters, positive=up.
        """
        z=self.grid.nodes['depth'][ self.grid.edges['nodes'][j] ].min()
        if z>0:
            log.warning("Edge %d has positive depth %.2f"%(j,z))

        if datum is not None:
            if datum=='eta0':
                z+=self.initial_water_level()
        return z

# Functions for manipulating DFM input/output

def extract_transect(ds,line,grid=None,dx=None,cell_dim='nFlowElem',
                     include=None,rename=True,add_z=True,name=None):
    """
    Extract a transect from map output.

    ds: xarray Dataset
    line: [N,2] polyline
    grid: UnstructuredGrid instance, defaults to loading from ds, although this
      is typically much slower as the spatial index cannot be reused
    dx: sample spacing along line
    cell_dim: name of the dimension
    include: limit output to these data variables
    rename: if True, follow naming conventions in xr_transect
    """
    missing=np.nan
    assert dx is not None,"Not ready for adaptively choosing dx"
    if grid is None:
        grid=dfm_grid.DFMGrid(ds)

    from stompy.spatial import linestring_utils
    line_sampled=linestring_utils.resample_linearring(line,dx,closed_ring=False)
    N_sample=len(line_sampled)

    # Get the mapping from sample index to cell, or None if
    # the point misses the grid.
    cell_map=[ grid.select_cells_nearest( line_sampled[samp_i], inside=True)
               for samp_i in range(N_sample)]
    # to make indexing more streamlined, replace missing cells with 0, but record
    # who is missing and nan out later.  Note that this need to be None=>0, to avoid
    # changing index of 0 to something else.
    cell_mask=[ c is None for c in cell_map]
    cell_map_safe=[ c or 0 for c in cell_map]

    if include is not None:
        exclude=[ v for v in ds.data_vars if v not in include]
        ds_orig=ds
        ds=ds_orig.drop(exclude)

    new_ds=ds.isel(**{cell_dim:cell_map_safe})

    # Record the intended sampling location:
    new_ds['x_sample']=(cell_dim,),line_sampled[:,0]
    new_ds['y_sample']=(cell_dim,),line_sampled[:,1]
    distance=utils.dist_along(line_sampled)
    new_ds['d_sample']=(cell_dim,),distance
    # And some additional spatial data:
    dx_sample=utils.center_to_interval(distance)

    new_ds['dx_sample']=(cell_dim,),dx_sample
    new_ds['d_sample_bnd']=(cell_dim,'two'), np.array( [distance-dx_sample/2,
                                                        distance+dx_sample/2]).T
    new_ds=new_ds.rename({cell_dim:'sample'})

    if add_z:
        new_ds.update( xr_utils.z_from_sigma(new_ds,'ucx',interfaces=True,dz=True) )

    # need to drop variables with dimensions like nFlowLink
    to_keep_dims=set(['wdim','laydim','two','three','time','sample'])
    to_drop=[]
    for v in new_ds.variables:
        if (set(new_ds[v].dims) - to_keep_dims):
            to_drop.append(v)

    new_ds=new_ds.drop(to_drop)

    xr_utils.bundle_components(new_ds,'U',['ucx','ucy'],'xy',['N','E'])
    xr_utils.bundle_components(new_ds,'U_avg',['ucxa','ucya'],'xy',['N','E'])

    if rename:
        new_ds=new_ds.rename( {'ucx':'Ve',
                               'ucy':'Vn',
                               'ucz':'Vu',
                               'ucxa':'Ve_avg',
                               'ucya':'Vn_avg',
                               's1':'z_surf',
                               'FlowElem_bl':'z_bed',
                               'laydim':'layer'} )

    # Add metadata if missing:
    if (name is None) and ('name' not in new_ds.attrs):
        new_ds.attrs['name']='DFM Transect'
    elif name is not None:
        new_ds.attrs['name']=name
    if 'filename' not in new_ds.attrs:
        new_ds.attrs['filename']=new_ds.attrs['name']
    if 'source' not in new_ds.attrs:
        new_ds.attrs['source']=new_ds.attrs['source']

    return new_ds

class OTPSHelper(object):
    # water columns shallower than this will have a velocity calculated
    # based on this water column depth rather than their actual value.
    min_h=5.0

    otps_model=None
    # slightly larger than default pad. probably unnecessary
    pad=2*np.timedelta64(24,'h')

    def __init__(self,otps_model,**kw):
        self.otps_model=otps_model # something like OhS
    def dataset(self):
        """
        extract h,u,v from OTPS.
        returns a xr.Dataset with time,U,V,u,v,h,Q,unorm
          U,V: east/north transport in m2/s
          u,v: east/north velocity in m/s, relative to model depth.
          h: tidal freesurface
          Q: inward-postive flux in m3/s
          unorm: inward-positive velocity in m/s
        """
        from stompy.model.otps import read_otps

        ds=xr.Dataset()
        times=np.arange( self.data_start,
                         self.data_stop,
                         15*np.timedelta64(60,'s') )
        log.debug("Will generate tidal prediction for %d time steps"%len(times))
        ds['time']=('time',),times
        modfile=read_otps.model_path(self.otps_model)
        xy=np.array(self.geom.coords)
        ll=self.model.native_to_ll(xy)
        # Note z=1.0 to get transport values in m2/s
        pred_h,pred_U,pred_V=read_otps.tide_pred(modfile,lon=ll[:,0],lat=ll[:,1],
                                                 time=times,z=1.0)
        pred_h=pred_h.mean(axis=1)
        pred_U=pred_U.mean(axis=1)
        pred_V=pred_V.mean(axis=1)

        ds['U']=('time',),pred_U
        ds['V']=('time',),pred_V
        ds['water_level']=('time',),pred_h

        # need a normal vector and a length.  And make sure normal vector is pointing
        # into the domain.
        L=utils.dist(xy[0],xy[-1])
        j=self.model.grid.select_edges_nearest( 0.5*(xy[0]+xy[-1]) )
        grid_n=self.get_inward_normal(j)
        Q=L*(grid_n[0]*pred_U + grid_n[1]*pred_V)
        ds['Q']=('time',),Q

        # u,v,unorm need model depth
        edge_depth=max(self.get_depth(j),self.min_h)
        # no adjustment for changing freesurface.  maybe later.
        ds['u']=ds.U/edge_depth
        ds['v']=ds.V/edge_depth
        ds['unorm']=ds.Q/(L*edge_depth)
        ds.attrs['mode']=self.mode
        return ds

class OTPSStageBC(StageBC,OTPSHelper):
    def __init__(self,**kw):
        super(OTPSStageBC,self).__init__(**kw)

    # write_config same as superclass
    # filename_base same as superclass

    def src_data(self):
        return self.dataset()['water_level']

    def write_data(self): # DFM IMPLEMENTATION!
        self.write_tim(self.data())



class OTPSFlowBC(FlowBC,OTPSHelper):
    def __init__(self,**kw):
        super(OTPSFlowBC,self).__init__(**kw)

    # write_config same as superclass
    # filename_base same as superclass

    def src_data(self):
        return self.dataset()['Q']

    def write_data(self): # DFM IMPLEMENTATION!
        self.write_tim(self.data())

class VelocityBC(BC):
    """
    expects a dataset() method which provides a dataset with time, u,v, and unorm
    (positive into the domain).

    dflowfm notes:
    BC setting edge-normal velocity (velocitybnd), uniform in the vertical.
    positive is into the domain.
    """
    # write a velocitybnd BC
    def write_config(self):
        old_bc_fn=self.model.ext_force_file()

        with open(old_bc_fn,'at') as fp:
            lines=["QUANTITY=velocitybnd",
                   "FILENAME=%s"%self.pli_filename(),
                   "FILETYPE=9",
                   "METHOD=3",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))
    def filename_base(self):
        """
        Make it clear in the filenames what is being forced
        """
        return super(VelocityBC,self).filename_base()+"_vel"
    def write_data(self):
        raise Exception("Implement write_data() in subclass")

class OTPSVelocityBC(VelocityBC,OTPSHelper):
    """
    Force 2D transport based on depth-integrated transport from OTPS.
    """
    def __init__(self,**kw):
        super(OTPSVelocityBC,self).__init__(**kw)

    def src_data(self):
        return self.dataset()['unorm']

    def write_data(self):
        da=self.data()
        if 'z' in da.dims:
            self.write_t3d(da,z_bed=self.model.edge_depth(self.grid_edge))
        else:
            self.write_tim(da)

class OTPSVelocity3DBC(OTPSVelocityBC):
    """
    Force 3D transport based on depth-integrated transport from OTPS.
    This is a temporary shim to test setting a 3D velocity BC.

    It is definitely wrong.  Don't use this yet.
    """
    def velocity_ds(self):
        ds=super(OTPSVelocity3DBC,self).velocity_ds()

        # so there is a 'unorm'
        z_bed=self.model.edge_depth(self.grid_edge)
        z_surf=1.0

        assert z_bed<0 # should probably use self.get_depth() instead.

        # pad out a bit above/below
        # and try populating more levels, in case things are getting chopped off
        N=10
        z_pad=10.0
        ds['z']=('z',), np.linspace(z_bed-z_pad,z_surf+z_pad,N)
        sig=np.linspace(-1,1,N)

        new_unorm,_=xr.broadcast(ds.unorm,ds.z)
        ds['unorm']=new_unorm

        # Add some vertical structure to test 3D nature of the BC
        delta=xr.DataArray(0.02*sig,dims=['z'])
        ds['unorm'] = ds.unorm + delta

        return ds

class MultiBC(BC):
    """
    Break up a boundary condition spec into per-edge boundary conditions.
    Hoping that this can be done in a mostly opaque way, without exposing to
    the caller that one BC is being broken up into many.
    """
    def __init__(self,cls,**kw):
        self.saved_kw=dict(kw) # copy
        # These are all passed on to the subclass, but only the
        # known parameters are kept for MultiBC.
        # if at some we need to pass parameters only to MultiBC, but
        # not to the subclass, this would have to check both ways.
        keys=list(kw.keys())
        for k in keys:
            try:
                getattr(self,k)
            except AttributeError:
                del kw[k]

        super(MultiBC,self).__init__(**kw)
        self.cls=cls
        self.sub_bcs="not yet!" # not populated until self.write()

    def filename_base(self):
        assert False,'This should never be called, right?'

    def write(self):
        # delay enumeration until now, so we have the most up-to-date
        # information about the model, grid, etc.
        self.enumerate_sub_bcs()

        for sub_bc in self.sub_bcs:
            sub_bc.write()

    def enumerate_sub_bcs(self):
        # dredge_grid already has some of the machinery
        grid=self.model.grid

        edges=dfm_grid.polyline_to_boundary_edges(grid,np.array(self.geom.coords))

        self.model.log.info("MultiBC will be applied over %d edges"%len(edges))

        self.sub_bcs=[]

        for j in edges:
            seg=grid.nodes['x'][ grid.edges['nodes'][j] ]
            sub_geom=geometry.LineString(seg)
            # This slightly breaks the abstraction -- in theory, the caller
            # can edit all of self's values up until write() is called, yet
            # here we are grabbing the values at time of instantiation of self.
            # hopefully it doesn't matter, especially since geom and model
            # are handled explicitly.
            sub_kw=dict(self.saved_kw) # copy original
            sub_kw['geom']=sub_geom
            sub_kw['name']="%s%04d"%(self.name,j)
            sub_kw['grid_edge']=j
            j_cells=grid.edge_to_cells(j)
            assert j_cells.min()<0
            assert j_cells.max()>=0
            sub_kw['grid_cell']=j_cells.max()

            assert self.model is not None,"Why would that be?"
            assert sub_geom is not None,"Huh?"

            sub_bc=self.cls(model=self.model,**sub_kw)
            self.sub_bcs.append(sub_bc)


# HYCOM
class HycomMultiBC(MultiBC):
    """
    Common machinery for pulling spatially variable fields from hycom
    """
    # according to website, hycom runs for download are non-tidal, so
    # don't worry about filtering
    # Data is only daily, so go a bit longer than a usual tidal filter
    lp_hours=0
    pad=np.timedelta64(4,'D')
    cache_dir=None

    def __init__(self,cls,ll_box=None,**kw):
        self.ll_box=ll_box
        self.data_files=None
        super(HycomMultiBC,self).__init__(cls,**kw)
        if self.cache_dir is None:
            self.log.warning("You probably want to pass cache_dir for hycom download")

    def enumerate_sub_bcs(self):
        if self.ll_box is None:
            # grid=self.model.grid ...
            raise Exception("Not implemented: auto-calculating ll_box")
        self.populate_files()
        super(HycomMultiBC,self).enumerate_sub_bcs()
    
        # adjust fluxes...
        self.populate_values()

    def populate_files(self):
        self.data_files=hycom.fetch_range(self.ll_box[:2],self.ll_box[2:],
                                          [self.data_start,self.data_stop],
                                          cache_dir=self.cache_dir)

    def init_bathy(self):
        """
        populate self.bathy, an XYZField in native coordinates, with
        values as hycom's positive down bathymetry.
        """
        # TODO: download hycom bathy on demand.
        hy_bathy=self.hy_bathy=xr.open_dataset( os.path.join(self.cache_dir,'depth_GLBa0.08_09.nc') )
        lon_min,lon_max,lat_min,lat_max=self.ll_box

        sel=((hy_bathy.Latitude.values>=lat_min) &
             (hy_bathy.Latitude.values<=lat_max) &
             (hy_bathy.Longitude.values>=lon_min) &
             (hy_bathy.Longitude.values<=lon_max))

        bathy_xyz=np.c_[ hy_bathy.Longitude.values[sel],
                         hy_bathy.Latitude.values[sel],
                         hy_bathy.bathymetry.isel(MT=0).values[sel] ]
        bathy_xyz[:,:2]=self.model.ll_to_native(bathy_xyz[:,:2])

        from ...spatial import field
        self.bathy=field.XYZField(X=bathy_xyz[:,:2],F=bathy_xyz[:,2])



class HycomMultiScalarBC(HycomMultiBC):
    """
    Extract 3D salt, temp from Hycom
    """
    scalar=None

    def __init__(self,**kw):
        super(HycomMultiScalarBC,self).__init__(self.ScalarProfileBC,**kw)

    class ScalarProfileBC(ScalarBC):
        cache_dir=None # unused now, but makes parameter-setting logic cleaner
        _dataset=None # supplied by factory
        def dataset(self):
            self._dataset.attrs['mode']=self.mode
            return self._dataset
        def src_data(self):# was dataarray()
            da=self.dataset()[self.scalar]
            da.attrs['mode']=self.mode
            return da

    def populate_values(self):
        """ Do the actual work of iterating over sub-edges and hycom files,
        interpolating in the vertical.

        Desperately wants some refactoring with the velocity code.
        """
        sun_var=self.scalar
        if sun_var=='salinity':
            hy_scalar='salinity'
        elif sun_var=='temperature':
            hy_scalar='water_temp'

        # Get spatial information about hycom files
        hy_ds0=xr.open_dataset(self.data_files[0])
        if 'time' in hy_ds0.water_u.dims:
            hy_ds0=hy_ds0.isel(time=0)
        # makes sure lon,lat are compatible with water velocity
        _,Lon,Lat=xr.broadcast(hy_ds0.water_u.isel(depth=0),hy_ds0.lon,hy_ds0.lat)
        hy_xy=self.model.ll_to_native(Lon.values,Lat.values)

        self.init_bathy()

        # Initialize per-edge details
        self.model.grid._edge_depth=self.model.grid.edges['edge_depth']
        layers=self.model.layer_data(with_offset=True)

        # In order to get valid data even when the hydro model has a cell
        # that lines up with somewhere dry in hycom land, limit the search below
        # to wet cells
        hy_wet=np.isfinite(hy_ds0[hy_scalar].isel(depth=0).values)

        for i,sub_bc in enumerate(self.sub_bcs):
            sub_bc.edge_center=np.array(sub_bc.geom.centroid)
            hyc_dists=utils.dist( sub_bc.edge_center, hy_xy )
            # lazy way to skip over dry cells.  Note that velocity differs
            # here, since it's safe to just use 0 velocity, but a zero
            # salinity can creep in and wreak havoc.  
            hyc_dists[~hy_wet]=np.inf
            row,col=np.nonzero( hyc_dists==hyc_dists.min() )
            row=row[0] ; col=col[0]
            sub_bc.hy_row_col=(row,col) # tuple, so it can be used directly in []

            # initialize the datasets
            sub_bc._dataset=sub_ds=xr.Dataset()
            # assumes that from each file we use only one timestep
            sub_ds['time']=('time',), np.ones(len(self.data_files),'M8[m]')

            sub_ds[sun_var]=('time','layer'), np.zeros((sub_ds.dims['time'],layers.dims['Nk']),
                                                       np.float64)
            sub_bc.edge_depth=edge_depth=self.model.grid.edge_depths()[sub_bc.grid_edge] # positive up

            # First, establish the geometry on the suntans side, in terms of z_interface values
            # for all wet layers.  below-bed layers have zero vertical span.  positive up, but
            # shift back to real, non-offset, vertical coordinate
            sun_z_interface=(-self.model.z_offset)+layers.z_interface.values.clip(edge_depth,np.inf)
            sub_bc.sun_z_interfaces=sun_z_interface
            # And the pointwise data from hycom:
            hy_layers=hy_ds0.depth.values.copy()
            sub_bc.hy_valid=valid=np.isfinite(hy_ds0[hy_scalar].isel(lat=row,lon=col).values)
            hycom_depths=hy_ds0.depth.values[valid]
            # possible that hy_bed_depth is not quite correct, and hycom has data
            # even deeper.  in that case just pad out the depth a bit so there
            # is at least a place to put the bed velocity.
            if len(hycom_depths)!=0:
                sub_bc.hy_bed_depth=max(hycom_depths[-1]+1.0,self.bathy(hy_xy[sub_bc.hy_row_col]))
                sub_bc.hycom_depths=np.concatenate( [hycom_depths, [sub_bc.hy_bed_depth]])
            else:
                # edge is dry in HYCOM -- be careful to check and skip below.
                sub_bc.hycom_depths=hycom_depths
                # for scalars, pray this never gets used...
                # don't use nan in case it participates in a summation with 0, but
                # make it yuge to make it easier to spot if it is ever used
                print("Hmm - got a dry hycom edge, even though should be skipping those now")
                sub_bc._dataset[sun_var].values[:]=100000000.

        # Populate the scalar data, outer loop is over hycom files, since
        # that's most expensive
        for ti,fn in enumerate(self.data_files):
            hy_ds=xr.open_dataset(fn)
            if 'time' in hy_ds.dims:
                # again, assuming that we only care about the first time step in each file
                hy_ds=hy_ds.isel(time=0)
            print(hy_ds.time.values)

            scalar_val=hy_ds[hy_scalar].values
            scalar_val_bottom=hy_ds[hy_scalar+'_bottom'].values

            for i,sub_bc in enumerate(self.sub_bcs):
                hy_depths=sub_bc.hycom_depths
                sub_bc._dataset.time.values[ti]=hy_ds.time.values
                if len(hy_depths)==0:
                    continue # already zero'd out above.
                row,col=sub_bc.hy_row_col
                z_sel=sub_bc.hy_valid

                sun_dz=np.diff(-sub_bc.sun_z_interfaces)
                sun_valid=sun_dz>0

                sub_scalar_val=np.concatenate([ scalar_val[z_sel,row,col],
                                                scalar_val_bottom[None,row,col] ])

                # integrate -- there isn't a super clean way to do this that I see.
                # but averaging each interval is probably good enough, just loses some vertical
                # accuracy.
                interval_mean_val=0.5*(sub_scalar_val[:-1]+sub_scalar_val[1:])
                valdz=np.concatenate( ([0],np.cumsum(np.diff(hy_depths)*interval_mean_val)) )
                sun_valdz=np.interp(-sub_bc.sun_z_interfaces, hy_depths, valdz)
                sun_d_veldz=np.diff(sun_valdz)

                sub_bc._dataset[sun_var].values[ti,sun_valid]=sun_d_veldz[sun_valid]/sun_dz[sun_valid]
            hy_ds.close() # free up netcdf resources


class HycomMultiVelocityBC(HycomMultiBC):
    """
    Special handling of multiple hycom boundary segments to
    enforce specific net flux requirements.
    Otherwise small errors, including quantization and discretization,
    lead to a net flux.
    """
    def __init__(self,**kw):
        super(HycomMultiVelocityBC,self).__init__(self.VelocityProfileBC,**kw)

    class VelocityProfileBC(VelocityBC):
        cache_dir=None # unused now, but makes parameter-setting logic cleaner
        _dataset=None # supplied by factory
        def dataset(self):
            self._dataset.attrs['mode']=self.mode
            return self._dataset
        def update_Q_in(self):
            """calculate time series flux~m3/s from self._dataset,
            updating Q_in field therein.
            Assumes populate_velocity has already been run, so 
            additional attributes are available.
            """
            ds=self.dataset()
            sun_dz=np.diff(-self.sun_z_interfaces)
            # u ~ [time,layer]
            Uint=(ds['u'].values[:,:]*sun_dz[None,:]).sum(axis=1)
            Vint=(ds['v'].values[:,:]*sun_dz[None,:]).sum(axis=1)
                                
            Q_in=self.edge_length*(self.inward_normal[0]*Uint +
                                   self.inward_normal[1]*Vint)
            ds['Q_in'].values[:]=Q_in
            ds['Uint'].values[:]=Uint
            ds['Vint'].values[:]=Vint

    def populate_values(self):
        """ Do the actual work of iterating over sub-edges and hycom files,
        interpolating in the vertical, projecting as needed, and adjust the overall
        fluxes
        """
        # The net inward flux in m3/s over the whole BC that we will adjust to.
        target_Q=np.zeros(len(self.data_files)) # assumes one time step per file

        # Get spatial information about hycom files
        hy_ds0=xr.open_dataset(self.data_files[0])
        if 'time' in hy_ds0.water_u.dims:
            hy_ds0=hy_ds0.isel(time=0)
        # makes sure lon,lat are compatible with water velocity
        _,Lon,Lat=xr.broadcast(hy_ds0.water_u.isel(depth=0),hy_ds0.lon,hy_ds0.lat)
        hy_xy=self.model.ll_to_native(Lon.values,Lat.values)

        self.init_bathy()

        # Initialize per-edge details
        self.model.grid._edge_depth=self.model.grid.edges['edge_depth']
        layers=self.model.layer_data(with_offset=True)

        for i,sub_bc in enumerate(self.sub_bcs):
            sub_bc.inward_normal=sub_bc.get_inward_normal()
            sub_bc.edge_length=sub_bc.geom.length
            sub_bc.edge_center=np.array(sub_bc.geom.centroid)

            # skip the transforms...
            hyc_dists=utils.dist( sub_bc.edge_center, hy_xy )
            row,col=np.nonzero( hyc_dists==hyc_dists.min() )
            row=row[0] ; col=col[0]
            sub_bc.hy_row_col=(row,col) # tuple, so it can be used directly in []

            # initialize the datasets
            sub_bc._dataset=sub_ds=xr.Dataset()
            # assumes that from each file we use only one timestep
            sub_ds['time']=('time',), np.ones(len(self.data_files),'M8[m]')
            # getting tricky here - do more work here rather than trying to push ad hoc interface
            # into the model class
            # velocity components in UTM x/y coordinate system
            sub_ds['u']=('time','layer'), np.zeros((sub_ds.dims['time'],layers.dims['Nk']),
                                                   np.float64)
            sub_ds['v']=('time','layer'), np.zeros((sub_ds.dims['time'],layers.dims['Nk']),
                                                   np.float64)
            # depth-integrated transport on suntans layers, in m2/s
            sub_ds['Uint']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)
            sub_ds['Vint']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)
            # project transport to edge normal * edge_length to get m3/s
            sub_ds['Q_in']=('time',), np.nan*np.ones(sub_ds.dims['time'],np.float64)

            sub_bc.edge_depth=edge_depth=self.model.grid.edge_depths()[sub_bc.grid_edge] # positive up

            # First, establish the geometry on the suntans side, in terms of z_interface values
            # for all wet layers.  below-bed layers have zero vertical span.  positive up, but
            # shift back to real, non-offset, vertical coordinate
            sun_z_interface=(-self.model.z_offset)+layers.z_interface.values.clip(edge_depth,np.inf)
            sub_bc.sun_z_interfaces=sun_z_interface
            # And the pointwise data from hycom:
            hy_layers=hy_ds0.depth.values.copy()
            sub_bc.hy_valid=valid=np.isfinite(hy_ds0.water_u.isel(lat=row,lon=col).values)
            hycom_depths=hy_ds0.depth.values[valid]
            # possible that hy_bed_depth is not quite correct, and hycom has data
            # even deeper.  in that case just pad out the depth a bit so there
            # is at least a place to put the bed velocity.
            if len(hycom_depths)!=0:
                sub_bc.hy_bed_depth=max(hycom_depths[-1]+1.0,self.bathy(hy_xy[sub_bc.hy_row_col]))
                sub_bc.hycom_depths=np.concatenate( [hycom_depths, [sub_bc.hy_bed_depth]])
            else:
                # edge is dry in HYCOM -- be careful to check and skip below.
                sub_bc.hycom_depths=hycom_depths
                sub_bc._dataset['u'].values[:]=0.0
                sub_bc._dataset['v'].values[:]=0.0
                sub_bc._dataset['Uint'].values[:]=0.0
                sub_bc._dataset['Vint'].values[:]=0.0

        # Populate the velocity data, outer loop is over hycom files, since
        # that's most expensive
        for ti,fn in enumerate(self.data_files):
            hy_ds=xr.open_dataset(fn)
            if 'time' in hy_ds.dims:
                # again, assuming that we only care about the first time step in each file
                hy_ds=hy_ds.isel(time=0)
            print(hy_ds.time.values)

            water_u=hy_ds.water_u.values
            water_v=hy_ds.water_v.values
            water_u_bottom=hy_ds.water_u_bottom.values
            water_v_bottom=hy_ds.water_v_bottom.values

            for i,sub_bc in enumerate(self.sub_bcs):
                hy_depths=sub_bc.hycom_depths
                sub_bc._dataset.time.values[ti]=hy_ds.time.values
                if len(hy_depths)==0:
                    continue # already zero'd out above.
                row,col=sub_bc.hy_row_col
                z_sel=sub_bc.hy_valid

                sun_dz=np.diff(-sub_bc.sun_z_interfaces)
                sun_valid=sun_dz>0
                for water_vel,water_vel_bottom,sun_var,trans_var in [ (water_u,water_u_bottom,'u','Uint'),
                                                                      (water_v,water_v_bottom,'v','Vint') ]:
                    sub_water_vel=np.concatenate([ water_vel[z_sel,row,col],
                                                   water_vel_bottom[None,row,col] ])

                    # integrate -- there isn't a super clean way to do this that I see.
                    # but averaging each interval is probably good enough, just loses some vertical
                    # accuracy.
                    interval_mean_vel=0.5*(sub_water_vel[:-1]+sub_water_vel[1:])
                    veldz=np.concatenate( ([0],np.cumsum(np.diff(hy_depths)*interval_mean_vel)) )
                    sun_veldz=np.interp(-sub_bc.sun_z_interfaces, hy_depths, veldz)
                    sun_d_veldz=np.diff(sun_veldz)

                    sub_bc._dataset[sun_var].values[ti,sun_valid]=sun_d_veldz[sun_valid]/sun_dz[sun_valid]
                    # might as well calculate flux while we are here
                    # explicit flux:
                    # sub_bc._dataset[trans_var].values[ti]=(sub_bc._dataset[sun_var]*sun_dz).sum()
                    # but we've already done the integration
                    sub_bc._dataset[trans_var].values[ti]=sun_veldz[-1]
            hy_ds.close() # free up netcdf resources

        # project transport onto edges to get fluxes
        total_Q=0.0
        total_flux_A=0.0
        for i,sub_bc in enumerate(self.sub_bcs):
            Q_in=sub_bc.edge_length*(sub_bc.inward_normal[0]*sub_bc._dataset['Uint'].values +
                                     sub_bc.inward_normal[1]*sub_bc._dataset['Vint'].values)
            sub_bc._dataset['Q_in'].values[:]=Q_in
            total_Q=total_Q+Q_in
            # edge_depth here reflects the expected water column depth.  it is the bed elevation, with
            # the z_offset removed (I hope), under the assumption that a typical eta is close to 0.0,
            # but may be offset as much as -10.
            # edge_depth is positive up.  here assume typical eta=0
            total_flux_A+=sub_bc.edge_length*(-sub_bc.edge_depth).clip(0,np.inf)

        Q_error=total_Q-target_Q
        vel_error=Q_error/total_flux_A
        print("Velocity error: %.6f -- %.6f m/s"%(vel_error.min(),vel_error.max()))
        print("total_flux_A: %.3e"%total_flux_A)

        # And apply the adjustment, and update integrated quantities
        adj_total_Q=0.0
        for i,sub_bc in enumerate(self.sub_bcs):
            # seems like we should be subtracting vel_error, but that results in a doubling
            # of the error?
            sub_bc._dataset['u'].values[:,:] -= vel_error[:,None]*sub_bc.inward_normal[0]
            sub_bc._dataset['v'].values[:,:] -= vel_error[:,None]*sub_bc.inward_normal[1]
            sub_bc.update_Q_in()
            adj_total_Q=adj_total_Q+sub_bc._dataset['Q_in']
        adj_Q_error=adj_total_Q-target_Q
        adj_vel_error=adj_Q_error/total_flux_A
        print("Post-adjustment velocity error: %.6f -- %.6f m/s"%(adj_vel_error.min(),adj_vel_error.max()))


class NOAAStageBC(StageBC):
    station=None # integer station
    product='water_level' # or 'predictions'
    cache_dir=None
    def src_data(self):
        ds=self.fetch_for_period(self.data_start,self.data_stop)
        return ds['z']
    def write_bokeh(self,**kw):
        defaults=dict(title="Stage: %s (%s)"%(self.name,self.station))
        defaults.update(kw)
        super(NOAAStageBC,self).write_bokeh(**defaults)
    def fetch_for_period(self,period_start,period_stop):
        """
        Download or load from cache, take care of any filtering, unit conversion, etc.
        Returns a dataset with a 'z' variable, and with time as UTC
        """
        ds=noaa_coops.coops_dataset(station=self.station,
                                    start_date=period_start,
                                    end_date=period_stop,
                                    products=[self.product],
                                    days_per_request='M',cache_dir=self.cache_dir)
        ds=ds.isel(station=0)
        ds['z']=ds[self.product]
        ds['z'].attrs['units']='m'
        return ds

class NwisBC(object):
    cache_dir=None
    product_id="set_in_subclass"

    def __init__(self,station,**kw):
        """
        station: int or string station id, e.g. 11455478
        """
        self.station=str(station)
        super(NwisBC,self).__init__(**kw)

class NwisStageBC(NwisBC,StageBC):
    product_id=65 # gage height
    def src_data(self):
        ds=self.fetch_for_period(self.data_start,self.data_stop)
        return ds['z']
    def write_bokeh(self,**kw):
        defaults=dict(title="Stage: %s (%s)"%(self.name,self.station))
        defaults.update(kw)
        super(NwisStageBC,self).write_bokeh(**defaults)
    def fetch_for_period(self,period_start,period_stop):
        """
        Download or load from cache, take care of any filtering, unit conversion, etc.
        Returns a dataset with a 'z' variable, and with time as UTC
        """
        from ...io.local import usgs_nwis
        ds=usgs_nwis.nwis_dataset(station=self.station,start_date=period_start,
                                  end_date=period_stop,
                                  products=[self.product_id],
                                  cache_dir=self.cache_dir)
        ds['z']=('time',), 0.3048*ds['height_gage']
        ds['z'].attrs['units']='m'
        return ds

class NwisFlowBC(NwisBC,FlowBC):
    product_id=60 # discharge
    def src_data(self):
        ds=self.fetch_for_period(self.data_start,self.data_stop)
        return ds['Q']
    def write_bokeh(self,**kw):
        defaults=dict(title="Flow: %s (%s)"%(self.name,self.station))
        defaults.update(kw)
        super(NwisFlowBC,self).write_bokeh(**defaults)
    def fetch_for_period(self,period_start,period_stop):
        """
        Download or load from cache, take care of any filtering, unit conversion, etc.
        Returns a dataset with a 'z' variable, and with time as UTC
        """
        from ...io.local import usgs_nwis
        
        ds=usgs_nwis.nwis_dataset(station=self.station,start_date=period_start,
                                  end_date=period_stop,
                                  products=[self.product_id],
                                  cache_dir=self.cache_dir)
        ds['Q']=('time',), 0.028316847*ds['stream_flow_mean_daily']
        ds['Q'].attrs['units']='m3 s-1'
        return ds



class DFlowModel(HydroModel):
    # If these are the empty string, then assumes that the executables are
    # found in existing $PATH
    dfm_bin_dir="" # .../bin  giving directory containing dflowfm
    dfm_bin_exe='dflowfm'

    # flow and source/sink BCs will get the adjacent nodes dredged
    # down to this depth in order to ensure the impose flow doesn't
    # get blocked by a dry edge. Set to None to disable.
    dredge_depth=-1.0

    def __init__(self,*a,**kw):
        super(DFlowModel,self).__init__(*a,**kw)
        self.structures=[]

    def write_forcing(self,overwrite=True):
        bc_fn=self.ext_force_file()
        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)
        super(DFlowModel,self).write_forcing()

    default_grid_target_filename='grid_net.nc'
    def grid_target_filename(self):
        """
        The filename, relative to self.run_dir, of the grid.  Not guaranteed
        to exist, and if no grid has been set, or the grid has no filename information,
        this will default to self.default_grid_target_filename
        """
        if self.grid is None or self.grid.filename is None:
            return self.default_grid_target_filename
        else:
            grid_fn=self.grid.filename
            if not grid_fn.endswith('_net.nc'):
                if grid_fn.endswith('.nc'):
                    grid_fn=grid_fn.replace('.nc','_net.nc')
                else:
                    grid_fn=grid_fn+"_net.nc"
            return os.path.basename(grid_fn)
    def write_grid(self):
        """
        Write self.grid to the run directory.
        Must be called after MDU is updated.  Should also be called
        after write_forcing(), since some types of BCs can update
        the grid (dredging boundaries)
        """
        dest=os.path.join(self.run_dir, self.mdu['geometry','NetFile'])
        dfm_grid.write_dfm(self.grid,dest,overwrite=True)

    def ext_force_file(self):
        return os.path.join(self.run_dir,self.mdu['external forcing','ExtForceFile'])

    def load_mdu(self,fn):
        self.mdu=dio.MDUFile(fn)

    @classmethod
    def load(cls,fn):
        """
        Populate Model instance from an existing run
        """
        fn=cls.to_mdu_fn(fn) # in case fn was a directory
        if fn is None:
            # no mdu was found
            return None
        model=DFlowModel()
        model.load_mdu(fn)
        try:
            model.grid = ugrid.UnstructuredGrid.read_dfm(model.mdu.filepath( ('geometry','NetFile') ))
        except FileNotFoundError:
            log.warning("Loading model from %s, no grid could be loaded"%fn)
            model.grid=None
        model.set_run_dir(os.path.dirname(fn),mode='existing')
        # infer number of processors based on mdu files
        # Not terribly robust if there are other files around..
        sub_mdu=glob.glob( fn.replace('.mdu','_*.mdu') )
        if len(sub_mdu)>0:
            model.num_procs=len(sub_mdu)
        else:
            # probably better to test whether it has even been processed
            model.num_procs=1

        ref,start,stop=model.mdu.time_range()
        model.run_start=start
        model.run_stop=stop
        return model

    @classmethod
    def to_mdu_fn(cls,path):
        """
        coerce path that is possibly a directory to a best guess
        of the MDU path.  file paths are left unchanged. returns None
        if path is a directory but no mdu files is there.
        """
        # all mdu files, regardless of case
        if not os.path.isdir(path):
            return path
        fns=[os.path.join(path,f) for f in os.listdir(path) if f.lower().endswith('.mdu')]
        # assume shortest is the one that hasn't been partitioned
        if len(fns)==0:
            return None
        
        unpartitioned=np.argmin([len(f) for f in fns])
        return fns[unpartitioned]
        
    @classmethod
    def run_completed(cls,fn):
        """
        fn: path to mdu file.  will attempt to guess the right mdu if a directory
        is provided, but no guarantees.

        returns: True if the file exists and the folder contains a run which
          ran to completion. Otherwise False.
        """
        if not os.path.exists(fn):
            return False
        model=cls.load(fn)
        return (model is not None) and model.is_completed()
    def is_completed(self):
        """
        return true if the model has been run.
        this can be tricky to define -- here completed is based on
        a report in a diagnostic that the run finished.
        this doesn't mean that all output files are present.
        """
        root_fn=self.mdu.filename[:-4] # drop .mdu suffix
        if self.num_procs>1:
            dia_fn=root_fn+'_0000.dia'
        else:
            dia_fn=root_fn+'.dia'
            
        assert dia_fn!=self.mdu.filename,"Probably case issues with %s"%dia_fn
                                          
        if not os.path.exists(dia_fn):
            return False
        # Read the last 1000 bytes
        with open(dia_fn,'rb') as fp:
            fp.seek(0,os.SEEK_END)
            tail_size=min(fp.tell(),1000)
            fp.seek(-tail_size,os.SEEK_CUR)
            # This may not be py2 compatible!
            tail=fp.read().decode(errors='ignore')
        return "Computation finished" in tail
    
    def update_config(self):
        """
        Update fields in the mdu object with data from self.
        """
        if self.mdu is None:
            self.mdu=dio.MDUFile()

        self.mdu.set_time_range(start=self.run_start,stop=self.run_stop)
        self.mdu.set_filename(os.path.join(self.run_dir,self.mdu_basename))

        self.mdu['geometry','NetFile'] = self.grid_target_filename()

        # Try to allow for the caller handling observation and cross-section
        # files externally or through the interface -- to that end, don't
        # overwrite ObsFile or CrsFile, but if internally there are point/
        # line observations set, make sure that there is a filename there.
        if len(self.mon_points)>0 and not self.mdu['output','ObsFile']:
            self.mdu['output','ObsFile']="obs_points.xyn"
        if len(self.mon_sections)>0 and not self.mdu['output','CrsFile']:
            self.mdu['output','CrsFile']="obs_sections.pli"

    def write_config(self):
        # Assumes update_config() already called
        self.write_structures() # updates mdu
        self.write_monitors()
        self.mdu.write()

    def write_monitors(self):
        self.write_monitor_points()
        self.write_monitor_sections()

    def write_monitor_points(self):
        fn=self.mdu.filepath( ('output','ObsFile') )
        with open(fn,'at') as fp:
            for i,mon_feat in enumerate(self.mon_points):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_pnt_%03d"%i
                xy=np.array(mon_feat['geom'])
                fp.write("%.3f %.3f '%s'\n"%(xy[0],xy[1],name))
    def write_monitor_sections(self):
        fn=self.mdu.filepath( ('output','CrsFile') )
        with open(fn,'at') as fp:
            for i,mon_feat in enumerate(self.mon_sections):
                try:
                    name=mon_feat['name']
                except KeyError:
                    name="obs_sec_%03d"%i
                xy=np.array(mon_feat['geom'])
                dio.write_pli(fp,[ (name,xy) ])

    def add_Structure(self,**kw):
        self.structures.append(kw)

    def write_structures(self):
        structure_file='structures.ini'
        if len(self.structures)==0:
            return

        self.mdu['geometry','StructureFile']=structure_file

        with open( self.mdu.filepath(('geometry','StructureFile')),'wt') as fp:
            for s in self.structures:
                lines=[
                    "[structure]",
                    "type         = %s"%s['type'],
                    "id           = %s"%s['name'],
                    "polylinefile = %s.pli"%s['name'],
                    "door_height  = %.3f"%s['door_height'],
                    "lower_edge_level = %.3f"%s['lower_edge_level'],
                    "opening_width = %.3f"%s['opening_width'],
                    "sill_level     = %.3f"%s['sill_level'],
                    "horizontal_opening_direction = %s"%s['horizontal_opening_direction'],
                    "\n"
                ]
                fp.write("\n".join(lines))
                pli_fn=os.path.join(self.run_dir,s['name']+'.pli')
                feat=self.match_gazetteer(name=s['name'])
                assert len(feat)==1
                geom=feat[0]['geom']
                assert geom.type=='LineString'
                pli_data=[ (s['name'], np.array(geom.coords)) ]
                dio.write_pli(pli_fn,pli_data)

    def write_bc(self,bc):
        if isinstance(bc,StageBC):
            self.write_stage_bc(bc)
        elif isinstance(bc,FlowBC):
            self.write_flow_bc(bc)
        elif isinstance(bc,SourceSinkBC):
            self.write_source_bc(bc)
        elif isinstance(bc,WindBC):
            self.write_wind_bc(bc)
        elif isinstance(bc,RoughnessBC):
            self.write_roughness_bc(bc)
        else:
            super(DFlowModel,self).write_bc(bc)

    def write_tim(self,da,file_path):
        """
        Write a DFM tim file based on the timeseries in the DataArray.
        da must have a time dimension.  No support yet for vector-values here.
        file_path is relative to the working directory of the script, not
        the run_dir.
        """
        ref_date,start,stop = self.mdu.time_range()
        dt=np.timedelta64(60,'s') # always minutes
        
        if len(da.dims)==0:
            # raise Exception("Not implemented for constant waterlevel...")
            pad=np.timedelta64(86400,'s')
            times=np.array([start-pad,stop+pad])
            values=np.array([da.values.item(),da.values.item()])
        else:
            times=da.time.values
            values=da.values
        elapsed_time=(times - ref_date)/dt 
        data=np.c_[elapsed_time,values]

        np.savetxt(file_path,data)

    def write_stage_bc(self,bc):
        self.write_gen_bc(bc,quantity='stage')

    def write_flow_bc(self,bc):
        self.write_gen_bc(bc,quantity='flow')

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            dfm_grid.dredge_boundary(self.grid,
                                     np.array(bc.geom.coords),
                                     self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_source_bc(self,bc):
        self.write_gen_bc(bc,quantity='source')

        if self.dredge_depth is not None:
            # Additionally modify the grid to make sure there is a place for inflow to
            # come in.
            log.info("Dredging grid for source/sink BC %s"%bc.name)
            dfm_grid.dredge_discharge(self.grid,
                                      np.array(bc.geom.coords),
                                      self.dredge_depth)
        else:
            log.info("dredging disabled")

    def write_gen_bc(self,bc,quantity):
        """
        handle the actual work of writing flow and stage BCs.
        quantity: 'stage','flow','source'
        """
        bc_id=bc.name+"_" + quantity

        #self.write_pli()
        assert bc.geom.type=='LineString'
        pli_data=[ (bc_id, np.array(bc.geom.coords)) ]
        pli_fn=bc_id+'.pli'
        dio.write_pli(os.path.join(self.run_dir,pli_fn),pli_data)

        #self.write_config()
        with open(self.ext_force_file(),'at') as fp:
            lines=[]
            method=3 # default
            if quantity=='stage':
                lines.append("QUANTITY=waterlevelbnd")
            elif quantity=='flow':
                lines.append("QUANTITY=dischargebnd")
            elif quantity=='source':
                lines.append("QUANTITY=discharge_salinity_temperature_sorsin")
                method=1 # not sure how this is different
            else:
                assert False
            lines+=["FILENAME=%s"%pli_fn,
                    "FILETYPE=9",
                    "METHOD=%d"%method,
                    "OPERAND=O",
                    ""]
            fp.write("\n".join(lines))

        #self.write_data()
        da=bc.data()
        assert len(da.dims)<=1,"Only ready for dimensions of time or none"
        tim_path=os.path.join(self.run_dir,bc_id+"_0001.tim")
        self.write_tim(da,tim_path)

    def write_wind_bc(self,bc):
        assert bc.geom is None,"Spatially limited wind not yet supported"

        tim_fn=bc.name+".tim"
        tim_path=os.path.join(self.run_dir,tim_fn)

        # write_config()
        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=windxy",
                   "FILENAME=%s"%tim_fn,
                   "FILETYPE=2",
                   "METHOD=1",
                   "OPERAND=O",
                   "\n"]
            fp.write("\n".join(lines))

        self.write_tim(bc.data(),tim_path)

    def write_roughness_bc(self,bc):
        # write_config()
        xyz_fn=bc.name+".xyz"
        xyz_path=os.path.join(self.run_dir,xyz_fn)

        with open(self.ext_force_file(),'at') as fp:
            lines=["QUANTITY=frictioncoefficient",
                   "FILENAME=%s"%xyz_fn,
                   "FILETYPE=7",
                   "METHOD=4",
                   "OPERAND=O",
                   "\n"
                   ]
            fp.write("\n".join(lines))

        # write_data()
        da=bc.data()
        xyz=np.c_[ da.x.values,
                   da.y.values,
                   da.values ]
        np.savetxt(xyz_path,xyz)

    def initial_water_level(self):
        """
        some BC methods which want a depth need an estimate of the water surface
        elevation, and the initial water level is as good a guess as any.
        """
        return float(self.mdu['geometry','WaterLevIni'])

    def map_outputs(self):
        """
        return a list of map output files
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_map.nc'))
        fns.sort()
        return fns
    def his_output(self):
        """
        return path to history file output
        """
        output_dir=self.mdu.output_dir()
        fns=glob.glob(os.path.join(output_dir,'*_his.nc'))
        assert len(fns)==1
        return fns[0]
        
    

import sys
if sys.platform=='win32':
    cls=HydroModel
    cls.dfm_bin_exe="dflowfm-cli.exe"
    cls.mpi_bin_exe="mpiexec.exe"
