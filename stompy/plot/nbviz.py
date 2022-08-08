# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 05:00:18 2022

@author: rusty
"""
import os
import matplotlib.pyplot as plt
from . import plot_utils
from .. import utils
from ..grid import unstructured_grid

from ipywidgets import Button, Layout, jslink, IntText, IntSlider, AppLayout, Output
import ipywidgets as widgets

# NBViz: top level. holds a list of datasets, list of layers
#    manages the top level widget layout.
# Layer is a combination of a dataset, a variable or expression name,
#    and a plotter.  
# Dataset: roughly a netcdf file. 
#   has a list of meshes, in the VisIt sense. 
#   has a list of variables. Each variable has dimensions.
# Not yet clear on how dimensions and sliders should be managed.
#  Would like to support a default where dimensions are manipulated
#  at the dataset level, shared across layers based on that dataset.
#  But flexible enough to share a dimension slider across datasets, 
#  or have dimension sliders specific to a layer.
# Maybe this can use the observe methods?
# Probably best to separate Layer, which is more like plot-type, from
# PlotVar. It's really plotvar that needs to know dimensions.

class Fig:
    """ Manages a single matplotlib figure
    """
    num=None
    figsize=(9.25,7)
    def __init__(self,**kw):
        utils.set_keywords(self,kw)        
        self.fig,self.ax=plt.subplots(num=self.num,clear=1,figsize=self.figsize)
        self.ax.set_adjustable('datalim')

        self.caxs=[]
                   
    def get_cax(self):
        # Colorbars. This will need to be more dynamic, with some way for
        # plots to request colorbar axis, and dispose of it later.      
        n_cax=len(self.caxs)
        cax=self.fig.add_axes([0.9,0.1+0.3*n_cax,0.02,0.25])
        self.caxs.append(cax)
        return cax
    def redraw(self):
        self.fig.canvas.draw()        
    def save_frame(self,fn):
        fn=os.path.abspath(fn)
        print("Saving to %s"%fn)
        self.fig.savefig(fn)
    def do_tight_layout(self):
        self.fig.tight_layout()

class Dataset:
    """
    Generic unstructured_grid / xarray plotting
    for jupyter notebook
    """
    def __init__(self,**kw):
        utils.set_keywords(self,kw)
        
    def add_layer(self,variable):
        raise Exception("overload")        

class UGDataset(Dataset):
    # ccoll=None
    # cell_var=None
    # ecoll=None
    # edge_var=None
    # ncoll=None
    # node_var=None
    
    def __init__(self,ds,grid=None,**kw):
        super().__init__(**kw)
        self.ds=ds
        if grid is None:
            grid=unstructured_grid.UnstructuredGrid.read_ugrid(ds)            
        self.grid=grid
        
        # select field to plot for cells:
        self.dim_selectors=dict(time=self.ds.dims['time']-5)
        #self.set_cell_var('eta')

    def create_layer(self,variable):
        if variable in self.available_cell_vars():
            return UGCellLayer(self,variable)
        else:
            print("Not sure how to handle variable for layer: ",variable)
            return None

    def available_cell_vars(self):
        meta=self.grid.nc_meta
        cell_vars=[]
        for v in self.ds:
            if meta['face_dimension'] in self.ds[v].dims:
                cell_vars.append(v)
        # need more generic expression handling.
        if ('eta' in cell_vars) and ('bed_elev' in cell_vars):
            cell_vars.append('depth')
        # also should filter out variables that are known to be
        # part of the grid definition.
        return cell_vars
                
    def select_cell_data(self,variable,dims):
        if variable == 'depth': # Will get generalized to Expression
            return self.select_data('eta',dims) - self.select_data('bed_elev',dims)
        return self.select_data(variable,dims)
    def var_dims(self,v):
        if v=='depth':
            return self.ds['eta'].dims
        else:
            return self.ds[v].dims
    def select_data(self,varname,dims):
        v=self.ds[varname]
        isels={k:dims[k]
               for k in dims
               if k in v.dims}
        if len(isels):
            v=v.isel(**isels)
        return v
    
    # def update_dim_sliders(self):
    #     # all dimensions that might need to be indexed:
    #     active_dims=[]
    #     for v in [self.cell_var]: # edge var, node var
    #         if v is None: continue
    #         for d in self.var_dims(v):
    #             meta=self.grid.nc_meta
    #             if d in [meta['face_dimension'],
    #                      meta['edge_dimension'],
    #                      meta['node_dimension']]:
    #                 continue
    #             if d in active_dims: continue
    #             active_dims.append(d)
                
    #     # HERE: update to manage slider widgets
    #     #if len(active_dims)>len(self.dim_axs):
    #     #    print("More dimensions than sliders...")
    #     # self.widgets=[]
    #     # for dim_ax,dim in zip(self.dim_axs,active_dims):
    #     #     dim_ax.cla()
    #     #     widget=mw.Slider(dim_ax,dim,valmin=0,
    #     #                      valmax=self.ds.dims[dim]-1,
    #     #                      valstep=1.0)
    #     #     if dim in self.dim_selectors:
    #     #         widget.set_val(self.dim_selectors[dim])
    #     #     widget.on_changed( lambda val: self.update_dim(dim,val) )
    #     #     self.widgets.append(widget)
    #     # for k in list(self.dim_selectors):
    #     #     if k not in active_dims:
    #     #         del self.dim_selectors[k]
        

class Layer:
    pass

class UGLayer(Layer):
    def __init__(self,ds,variable):
        self.ds=ds
        self.variable=variable
        self.dims=None # uninitialized.
    
class UGCellLayer(UGLayer):
    """
    pseudocolor plot of cell-centered values for unstructured grid
    dataset.
    """
    ccoll=None
        
    def init_plot(self,Fig):
        # okay if it is an empty dict, just not None
        assert self.dims is not None
        
        self.Fig=Fig
        self.ccoll=self.ds.grid.plot_cells(values=self.get_data(),
                                           cmap='turbo',ax=self.Fig.ax)
        self.cax=self.Fig.get_cax()
        self.ccbar=plot_utils.cbar(self.ccoll,cax=self.cax)
    def get_data(self):
        return self.ds.select_cell_data(self.variable,self.dims)
    
    def update_dims(self,dims):
        # dims: dim => index
        # If any of the dims that control this layer have changed,
        # fetch new data and update the plot.
        update_plot=False
        if self.dims is None:
            #Copy, as dims may get updated later.
            self.dims=dict(dims) # really this should be slimmed down to the active dims for the layer
            update_plot=True
            return
        for k in self.dims:
            if k not in dims: 
                # generally will want to get all dimensions passed in, but
                # open to the idea that, aside from initialization, we'd only
                # get updated dims.
                continue
            if self.dims[k]==dims[k]:
                continue
            update_plot=True
            self.dims[k]=dims[k]
        if update_plot:
            self.update_arrays()
            self.redraw() # maybe too proactive
    def redraw(self):
        if self.Fig is not None and self.ccoll is not None:
            self.Fig.redraw()
    def update_arrays(self):
        # Would like to be smarter -- have callers figure out who
        # should be updated, and this just becomes the final draw 
        # call. probably keep a list of layers, broadcast dim
        # changes, and let layers notify of update.
        if self.ccoll is not None:
            self.ccoll.set_array(self.get_data())
    
    def available_vars(self):
        # to show options for changing the variable of an existing layer
        # will need to also be smarter about updating self.dims, in case
        # new variable has different dimensions. 
        return self.ds.available_cell_vars()
    
    def set_variable(self,v):
        if v==self.variable: return
        print("set_variable: ",v)
        self.variable=v
        
        # This feels a bit too early. Would be nice to be more JIT.
        if self.ccoll is not None:
            data=self.get_data()
            self.ccoll.norm.autoscale(data)
            self.ccoll.set_array(data)
            self.Fig.redraw()


                         
# use while debugging to see how layouts are working
def create_expanded_button(description, button_style):
    return Button(description=description, button_style=button_style, 
                  layout=Layout(height='auto', width='auto'))

class NBViz(widgets.AppLayout):
    """
    Manage databases, layers, and a plot window.
    This is roughly the top level, but unlike visit there is
    only one window/figure in which results are shown. 
    - May allow that one figure to have multiple axes, so keep
      ax separate
    """
    # Will try just being a subclass of the top-level
    # layout. That will streamline display logic, but
    # not sure what the other implications are.
    active_layer=None
    
    def __init__(self,datasets=None):
        # at the moment datasets can only be ugrid-ish xr.Dataset,
        # to be wrapped in UGDataset
        if datasets is not None:
            self.datasets=[self.data_to_Dataset(d) for d in datasets]
        else:
            self.datasets=[]
        assert len(self.datasets)==1,"Not ready for 0 or multiple datasets"
        self.layers=[]
        self.Fig=Fig()
        # May need a display call here to get the figure to show up?
        
        # this will need to get smarter
        # dynamically find out which dimensions are active, 
        # allow for dimensions to be shared or not shared between
        # datasets and layers within datasets.
        # show sliders for active dimensions.
        # and dynamically update min/max for dimensions.
        self.active_dims={'time':0}
        
        # create widgets and call super().__init__
        self.init_layout()

    def data_to_Dataset(self,d):
        " wrap original data of some form in an appropriate NBViz dataset"
        return UGDataset(d)
    
    def init_layout(self):
        ds=self.datasets[0] # DEV

        # hard coded time dimension handling
        time_min=0
        time_max=ds.ds.dims['time']-1
        
        self.play = widgets.Play(
            value=50,
            min=time_min,
            max=time_max,
            step=1,
            interval=50,
            description="Press play",
            disabled=False
        )
        self.time_slider = widgets.IntSlider(min=time_min,max=time_max) 
        widgets.jslink((self.play, 'value'), (self.time_slider, 'value'))
        time_control=widgets.HBox([self.play, self.time_slider])
        
        def on_time_change(change):
            # Currently all layers will get update_dim for all dimensions.
            self.active_dims['time']=change['new']
            for layer in self.layers:
                layer.update_dims(self.active_dims)
            self.Fig.redraw()
            if self.save_enabled.value:
                self.save_frame()
                
        self.time_slider.observe(on_time_change, names='value')
        
                   
        self.var_selector = widgets.Dropdown(options=["No active layer"],
                                             description="Var:")
        self.var_selector.observe(self.on_var_change,names='value')
        
        global_controls=self.global_controls_widget()
        
        # so far have failed to use MPL figure as a widget.
        super().__init__(header=time_control,
                         left_sidebar=self.var_selector,
                         center=None, # create_expanded_button('Center', 'warning'),
                         right_sidebar=global_controls,
                         footer=None)
        # displaying the viz object in the notebook then displays the 
        # widgets.
    def global_controls_widget(self):
        # buttons, toggles, etc. that control the global state of the figure
        show_axes=widgets.Checkbox(value=True,
                                   description='Show axes')
        show_axes.observe(self.on_show_axes_change,names='value')
        tight_layout=widgets.Button(description="Tight layout")
        tight_layout.on_click(self.do_tight_layout)
        self.save_enabled=widgets.Checkbox(value=False,
                                           description="")
        self.save_path=widgets.Text(value='image-%04d.png',
                                    placeholder='path for saved plots',
                                    description='Save image:',
                                    disabled=True)
        # might not work - may need to do it python side.
        def cb(change,save_path=self.save_path):
            save_path.disabled=not change['new']
        self.save_enabled.observe(cb,names='value')
        
        vbox=widgets.VBox([widgets.HBox([show_axes,tight_layout]),
                           widgets.HBox([self.save_enabled,self.save_path])])
        return vbox
    def save_frame(self):
        fn_template=self.save_path.value
        fn=fn_template.format(**self.active_dims)
        self.Fig.save_frame(fn)
    def on_show_axes_change(self,change):
        if self.Fig is not None:
            self.Fig.ax.axis(change['new'])
    def do_tight_layout(self,b):
        self.Fig.do_tight_layout()
    def add_layer(self,ds,variable):
        layer=ds.create_layer(variable)
        if layer is None:
            print("Could not create layer")
        # will have to be smarter once active_dims actually represents
        # the active layer. and active_dims will probably show all
        # dimensions, but we'll show sliders only for the ones for the
        # active layer.
        self.layers.append(layer)
        layer.update_dims(self.active_dims)
        layer.init_plot(self.Fig)
        self.activate_layer(layer)
        
    def on_var_change(self,change):
        if self.active_layer is None: 
            print("var change but no active layer")
            return
        self.active_layer.set_variable(change['new'])

    def activate_layer(self,layer):
        if self.active_layer == layer: return
        self.deactivate_layer()
        self.active_layer=layer

        # I think it works to just directly set these -- the traitlets
        # will take care of the rest?       
        # small problem: setting the options will force a change to the
        # value, and currently that sets a value that doesn't plot (face_node)
        # So disable the callback while changing options:
        self.var_selector.unobserve(self.on_var_change,names='value')
        
        self.var_selector.options=self.active_layer.available_vars()
        self.var_selector.value=layer.variable
        self.var_selector.observe(self.on_var_change,names='value')

    def deactivate_layer(self):
        self.active_layer=None
        
        
# NEXT -
#   Add edge layer class, create one manually.
