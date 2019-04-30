"""
Tools related to comparing time series, typically model-obs or model-model.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stompy import filters
from matplotlib import dates

from .. import (xr_utils, utils)

def combine_sources(all_sources,dt=np.timedelta64(900,'s')):
    # For many plots and metrics need a common timeline -- 
    # Get them on common time frames

    t_min=t_max=None
    for da in all_sources:
        if t_min is None or da.time[0]<t_min:
            t_min=da.time[0]
        if t_max is None or da.time[-1]>t_max:
            t_max=da.time[-1]
    dt=np.timedelta64(900,"s")  # compare at 15 minute intervals.
    resample_bins=np.arange(utils.floor_dt64(t_min,dt),
                            utils.ceil_dt64(t_max,dt)+dt,
                            dt)
    bin_labels=resample_bins[:-1]

    def resample(da):
        # groupby_bins allows for specifying the exact bins and labels,
        # simplifying concatenation below.
        da_r=(da.groupby_bins(da.time,resample_bins,labels=bin_labels)
              .mean()
              .rename(time_bins='time')
              .to_dataset())
        return da_r

    resampled=[resample(da) for da in all_sources]

    field_name=all_sources[0].name
    combined=xr_utils.concat_permissive(resampled,dim='source')[field_name]
    return combined


def assemble_comparison_data(models,observations,model_labels=None):
    # models: list of HydroModel instances
    # observations: list of DataArrays representing time series
    #   the first observation must have lon and lat fields
    #   defining where to extract data from in the model.
    # returns a list of dataarrays, and a combined dataset

    if model_labels is None:
        if len(models)==1:
            model_labels=["Model"]
        else:
            model_labels=["Model %d"%(i+1) for i,mod in enumerate(models)]

    # Extract relevant variable and location from model
    base_obs=observations[0] # defines the variable and location for extracting model data
    model_data=[] # a data array per model
    for model,label in zip(models,model_labels):
        ds=model.extract_station(ll=[base_obs.lon,base_obs.lat])
        if base_obs.name=='water_level':
            da=ds['eta']
            da.name='water_level' # having the same name helps later
        else:
            raise Exception("Not yet ready")
        da=da.assign_coords(label=label)
        model_data.append(da)
        
    # Annotate the sources with labels
    for i,da in enumerate(observations):
        if 'name' in da.attrs:
            label=da.attrs['name']
        else:
            label="Obs %d"%i
        da=da.assign_coords(label=label)
        observations[i]=da

    all_sources=model_data+observations
    combined=combine_sources(all_sources)
    return all_sources,combined


def calc_metrics(x,ref):
    """
    x, ref: DataArrays with common time dimension
    """
    metrics={}
    metrics['bias']=(x-ref).mean()
    valid=np.isfinite( (x+ref).values )
    metrics['r'] = np.corrcoef( x.values[valid],ref.values[valid])[0,1]
    metrics['lag']= utils.find_lag_xr(x,ref) 
    metrics['lag_s']=metrics['lag']/np.timedelta64(1,'s')
    metrics['amp']=np.std(x.values[valid]) / np.std(ref.values[valid])
    return metrics    



def fix_date_labels(ax):
    xfmt = dates.DateFormatter('%Y-%m-%d')
    xax=ax.xaxis
    xax.set_major_formatter(xfmt)
    xax.set_major_locator(dates.WeekdayLocator(byweekday=(1),
                          interval=1))
    
def calibration_figure_3panel(all_sources,combined,
                              metric_x=0,metric_ref=1,
                              offset_source=0,scatter_x_source=0,
                              num=None,
                              styles=None):
    gs = gridspec.GridSpec(2, 3)
    fig=plt.figure(figsize=(9,7),num=num)
    ts_ax = fig.add_subplot(gs[0, :])
    lp_ax = fig.add_subplot(gs[1, :-1])
    scat_ax=fig.add_subplot(gs[1, -1])

    offsets=combined.mean(dim='time').values
    offsets-=offsets[offset_source]

    if styles is None:
        styles=[{}]*len(all_sources)

    if 1: # Tidal time scale plot:
        ax=ts_ax
        for src_i,src in enumerate(all_sources):
            ax.plot(src.time,src.values-offsets[src_i],
                    label=combined.label.isel(source=src_i).item(),
                    **styles[src_i])
        ax.legend(fontsize=8,loc='upper left')

    # Scatter:
    if 1:
        ax=scat_ax
        for i in range(len(combined.source)):
            if i==scatter_x_source: continue
            kw={}
            style=styles[i]
            for k in ['color','zorder']:
                if k in style:
                    kw[k]=style[k]
            ax.plot(combined.isel(source=scatter_x_source)-offsets[scatter_x_source],
                    combined.isel(source=i)-offsets[i],
                    '.',ms=1.5,**kw)
        ax.set_xlabel(combined.label.isel(source=scatter_x_source).item())
        
    # Metrics
    if metric_x is not None:
        ax=scat_ax # text on same plot as scatter
        metrics=calc_metrics(x=combined.isel(source=metric_x),
                             ref=combined.isel(source=metric_ref))
        lines=["Ampl: %.3f"%metrics['amp'],
               "Lag: %.1f min"%( metrics['lag_s']/60.)]
        ax.text(0.05,0.95,"\n".join(lines),va='top',transform=ax.transAxes)

    # Lowpass:
    if 1:
        ax=lp_ax
        t=combined.time.values

        def lp(x): 
            x=utils.fill_invalid(x)
            dn=utils.to_dnum(t)
            cutoff=36/24.
            x_lp=filters.lowpass(x,dn,cutoff=cutoff)
            mask= (dn<dn[0]+2*cutoff) | (dn>dn[-1]-2*cutoff)
            x_lp[mask]=np.nan
            return x_lp
        for i in range(len(combined.source)):
            ax.plot(t,lp(combined.isel(source=i).values)-offsets[i],
                    label=combined.label.isel(source=i).item(),
                    **styles[i])
        #ax.legend(fontsize=8)

    # ts_ax.set_title(model.run_dir)
    fix_date_labels(ts_ax)
    fix_date_labels(lp_ax)
    return fig
