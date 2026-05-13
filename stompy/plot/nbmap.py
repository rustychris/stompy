# Set up generic figure
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from .. import utils
from ..grid import unstructured_grid

class GenericMap:
    cmaps=['viridis','plasma','turbo','coolwarm']

    cmap0=None
    var0=None
    time0=None
    layer0=None
    clim0=None # as tuple or list
    cax_pos=[0.05,0.1,0.02,0.25]
    
    def __init__(self,ds,**kw):
        utils.set_keywords(self,kw)
        self.ds = ds
        self.g = unstructured_grid.UnstructuredGrid.read_ugrid(ds)

        # plottable tracers:
        self.spatial_vars = [s for s in self.ds.data_vars 
                             if 'face' in self.ds[s].dims] 

        # initialize figure
        self.fig,self.ax = plt.subplots(figsize=(10,8))
        self.cax = self.fig.add_axes(self.cax_pos)
        
        self.ax.set_adjustable('datalim')
        self.ax.set_position([0,0,1,1])
        self.ax.xaxis.set_visible(0)
        self.ax.yaxis.set_visible(0)
        self.g.plot_boundary(lw=0.5,color='k',ax=self.ax)
        self.cell_coll = self.g.plot_cells(values=np.zeros(self.g.Ncells()),ax=self.ax,lw=0.5,ec='face')
        plt.colorbar(self.cell_coll,cax=self.cax) 
                
        self.textbox = self.ax.text(0.01,0.98,"Initial",va='top',transform=self.ax.transAxes)
        
        # widgets
        self.widget_time=widgets.IntSlider(value=self.ds.sizes['time']-1,
                                           min=0,max=self.ds.sizes['time']-1,description='Time step:')
        self.widget_var  =widgets.Dropdown(options=self.spatial_vars,value=self.spatial_vars[0],description="Variable:")
        self.widget_layer=widgets.IntSlider(value=0,min=0,max=self.ds.sizes['layer']-1,description='Layer:')
        self.widget_output = widgets.Output(layout={'border': '1px solid black'})
        self.widget_clim=widgets.FloatRangeSlider(value=[0, 1.0], min=0, max=1.0, step=0.01, description='Color range:')
        self.widget_cmap=widgets.Dropdown(options=self.cmaps,value=self.cmaps[0],description="Colormap:")

        # overrides from caller:
        if self.cmap0 is not None:
            self.widget_cmap.value=self.cmap0
        if self.var0 is not None:
            self.widget_var.value=self.var0
        if self.time0 is not None:
            self.widget_time.value=self.time0
        if self.clim0 is not None:
            if self.clim0[0]<self.widget_clim.max:            
                self.widget_clim.min = self.clim0[0]
            self.widget_clim.max = self.clim0[1] # will get overwritten, but need it at least 
            self.widget_clim.min = self.clim0[0] # to cover requested lower/upper
            self.widget_clim.lower=self.widget_clim.min
            self.widget_clim.upper=self.widget_clim.max
            
        if self.layer0 is not None:
            self.widget_layer.value=self.layer0
            
        @self.widget_output.capture(clear_output=False)
        def on_button_clicked(b):
            #widget_output.clear_output()
            with self.widget_output: # necessary?
                #print("Updating...")
                scal = self.ds[self.widget_var.value]
                isel_kw={}
                if 'time' in scal.dims:
                    isel_kw['time'] = self.widget_time.value
                    t_str = utils.strftime(self.ds.time.values[isel_kw['time']])
                    self.textbox.set_text(f"Time:  {t_str}")
                if 'layer' in scal.dims:
                    isel_kw['layer'] = self.widget_layer.value
                data = scal.isel(**isel_kw)
                self.cell_coll.set_array(data)
                data_min=np.nanmin(data.values)
                data_max=np.nanmax(data.values)        
                if data_min<self.widget_clim.max:
                    self.widget_clim.min=data_min # can't set min>max, even transiently
                    self.widget_clim.max=data_max
                else:
                    self.widget_clim.max=data_max
                    self.widget_clim.min=data_min
                    
                clim=[self.widget_clim.lower,self.widget_clim.upper]
                self.cell_coll.set_clim(clim)
                self.cell_coll.set_cmap(self.widget_cmap.value)
                #self.cax.cla()
                self.fig.canvas.draw()
        
        self.widget_update=widgets.Button(description="Update")
        self.widget_update.on_click(on_button_clicked)

        on_button_clicked(None)
        display(self.widget_time,self.widget_var,self.widget_layer,self.widget_clim,self.widget_cmap,
                self.widget_update,self.widget_output)
