"""
Tools related to comparing time series, typically model-obs or model-model.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import logging as log
import matplotlib.gridspec as gridspec
from stompy import filters
from matplotlib import dates
from scipy.stats import spearmanr

from .. import (xr_utils, utils)

def period_union(sources):
    t_min=t_max=None
    for da in sources:
        if t_min is None or da.time[0]<t_min:
            t_min=da.time.values[0]
        if t_max is None or da.time[-1]>t_max:
            t_max=da.time.values[-1]
    return t_min,t_max

def period_intersection(sources):
    t_min=t_max=None
    for da in sources:
        if t_min is None or da.time[0]>t_min:
            t_min=da.time.values[0]
        if t_max is None or da.time[-1]<t_max:
            t_max=da.time.values[-1]
    return t_min,t_max

def combine_sources(all_sources,dt=np.timedelta64(900,'s'),min_period=True):
    """
    Resample multiple DataArray time series to common timebase.
    all_sources: list of xr.DataArray()
    dt: each input is resample at this time step.
    min_period: True => use the 
    """
    # For many plots and metrics need a common timeline -- 
    # Get them on common time frames
    empty=[len(da)==0 for da in all_sources]
    
    if min_period:
        if np.any(empty):
            print("Empty time series")
            return None
        t_min,t_max=period_intersection(all_sources)
    else:
        if np.all(empty):
            print("All empty time series")
            return None
        t_min,t_max=period_union(all_sources)
        
    dt=np.timedelta64(900,"s")  # compare at 15 minute intervals.
    resample_bins=np.arange(utils.floor_dt64(t_min,dt),
                            utils.ceil_dt64(t_max,dt)+dt,
                            dt)

    if len(resample_bins)<2:
        log.warning("No overlapping data")
        return None
    bin_labels=resample_bins[:-1]

    # All data arrays get renamed to the field name of the first one
    field_name=all_sources[0].name

    def resample(da):
        # groupby_bins allows for specifying the exact bins and labels,
        # simplifying concatenation below.
        da=da.rename(field_name)
        # having trouble with groupby_bins
        #
        da['dnum']=('time',),utils.to_dnum(da.time)
        bins=utils.to_dnum(resample_bins)
        # dim='time' is needed for vector-valued data to indicate not to
        # take the mean across vector components, just within bins on the
        # time axis
        da_r=(# ada.groupby_bins(da.time,resample_bins,labels=bin_labels)
            da.groupby_bins('dnum',bins,labels=bin_labels)
              .mean(dim='time')
              #.rename(time_bins='time')
            .rename(dnum_bins='time')
              .to_dataset())
        return da_r

    resampled=[resample(da) for da in all_sources]

    combined=xr_utils.concat_permissive(resampled,dim='source')[field_name]
    return combined


def assemble_comparison_data(models,observations,model_labels=None,
                             extract_options={}):
    """
    Extract data from one or more model runs to match one or more observations
    
    models: list of HydroModel instances
    observations: list of DataArrays representing time series
      the first observation must have lon and lat fields
      defining where to extract data from in the model.

    returns a tuple: ( [list of dataarrays], combined dataset )
    """
    if model_labels is None:
        if len(models)==1:
            model_labels=["Model"]
        else:
            model_labels=[]
            for m in i,models in enumerate(models):
                try:
                    model_labels.append( model.label )
                except AttributeError:
                    model_labels.append("Model %d"%(i+1))
    else:
        assert len(model_labels)>=len(models),"Not enough model labels supplied"
    
    # Extract relevant variable and location from model
    base_obs=observations[0] # defines the variable and location for extracting model data
    model_data=[] # a data array per model
    for model,label in zip(models,model_labels):
        if base_obs.name=='water_level':
            ds=model.extract_station(ll=[base_obs.lon,base_obs.lat],
                                     data_vars=['eta'], # can give drastic speedup
                                     **extract_options)
            da=ds['eta']
            da.name='water_level' # having the same name helps later
        elif base_obs.name=='flow':
            assert False,"this has not been written yet"
            # extract_section currently only for DFM, and only by name
            ds=model.extract_section(ll=[base_obs.lon,base_obs.lat],
                                     data_vars=['cross_section_discharge'],
                                     **extract_options)
            da=ds['cross_section_discharge'] # that's a DFM name...
            da.name='flow' # having the same name helps later
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


def calc_metrics(x,ref,combine=False):
    """
    x, ref: DataArrays with common dimension.

    if that dimension is time, some additional time-series metrics
    are calculated (namely lag).

    straight arrays can be passed in, in which case no time-related
    processing will be done.
    """
    if not isinstance(x,xr.DataArray):
        x=xr.DataArray(x)
    if not isinstance(ref,xr.DataArray):
        ref=xr.DataArray(ref)

    x_orig=x
    ref_orig=ref
        
    if combine:
        combined=combine_sources([x,ref])
        x=combined.isel(source=0)
        ref=combined.isel(source=1)
        
    metrics={}
    metrics['bias']=np.nanmean( (x-ref).values )
    valid=np.isfinite( (x+ref).values )
    metrics['r'] = np.corrcoef( x.values[valid],ref.values[valid])[0,1]
    if 'time' in x.dims and 'time' in ref.dims:
        metrics['lag']= utils.find_lag_xr(x_orig,ref_orig) 
        metrics['lag_s']=metrics['lag']/np.timedelta64(1,'s')
    metrics['amp']=np.std(x.values[valid]) / np.std(ref.values[valid])

    metrics['wilmott']=utils.model_skill(x.values,ref.values)
    metrics['murphy']=utils.murphy_skill(x.values,ref.values)
    metrics['spearman_rho'],metrics['spearman_p']=spearmanr(x.values,ref.values)
    
    return metrics    


def fix_date_labels(ax,nticks=3):
    xfmt = dates.DateFormatter('%Y-%m-%d')
    xax=ax.xaxis
    xax.set_major_formatter(xfmt)
    xax.set_major_locator(dates.AutoDateLocator(minticks=nticks,maxticks=nticks+1,
                                                interval_multiples=False))
    
def calibration_figure_3panel(all_sources,combined=None,
                              metric_x=1,metric_ref=0,
                              offset_source=0,scatter_x_source=0,
                              num=None,trim_time=False,
                              lowpass=True,
                              styles=None):
    """
    all_sources: list of DataArrays to compare.
    combined: those same dataarrays interpolated to common time, or none to automatically
      do this.
    metric_x: index of the 'model' data in combined.
    metric_ref: index of the 'observed' data in combined.
    scatter_x_ref: which item in combined to use for the x axis of the scatter.

    lowpass: if True, the lower left panel is a lowpass of the data, otherwise
    it will be used for the text metrics instead of overlaying them on the scatter.
    
    These default to having the reference observations as the first element, and the
    primary model output second.

    trim_time: truncate all sources to the shortest common time period
    """
    N=np.arange(len(all_sources))
    if metric_ref<0:
        metric_ref=N[metric_ref]
    if scatter_x_source<0:
        scatter_x_source=N[scatter_x_source]
        
    gs = gridspec.GridSpec(5, 3)
    fig=plt.figure(figsize=(9,7),num=num)
    ts_ax = fig.add_subplot(gs[:-3, :])
    lp_ax = fig.add_subplot(gs[-3:-1, :-1])
    scat_ax=fig.add_subplot(gs[-3:-1, 2])
    txt_ax= fig.add_subplot(gs[-1,:])

    labels=list(combined.label.values)
    
    if trim_time:
        t_min,t_max=period_intersection(all_sources)
        new_sources=[]
        for src in all_sources:
            tsel=(src.time.values>=t_min)&(src.time.values<=t_max)
            new_sources.append( src.isel(time=tsel) )
        all_sources=new_sources
        
    if combined is None:
        combined=combine_sources(all_sources)
        
    offsets=combined.mean(dim='time').values
    if offset_source is not None:
        offsets-=offsets[offset_source]
    else:
        # no offset to means.
        offsets*=0

    if styles is None:
        styles=[{}]*len(all_sources)

    if 1: # Tidal time scale plot:
        ax=ts_ax
        for src_i,src in enumerate(all_sources):
            ax.plot(src.time,src.values-offsets[src_i],
                    label=labels[src_i],
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
        ax.set_xlabel(labels[scatter_x_source])
        
    # Metrics
    if metric_x is not None:
        ax=txt_ax

        if metric_x=='all':
            metric_x=[i for i in range(len(all_sources)) if i!=metric_ref]
        else:
            metric_x=np.atleast_1d(metric_x)

        df=pd.DataFrame()
        recs=[]
        for mx in metric_x:
            rec=calc_metrics(x=combined.isel(source=mx)-offsets[mx],
                             ref=combined.isel(source=metric_ref)-offsets[metric_ref])
            rec['bias']+=offsets[mx] - offsets[metric_ref]
            recs.append(rec)
        df=pd.DataFrame(recs)
        df['label']=[labels[i] for i in metric_x]
        del df['lag']
        df=df.set_index('label')
        with pd.option_context('expand_frame_repr', False,
                               'precision',3):
            tbl=str(df)
            
        plt.setp(list(ax.spines.values()),visible=0)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)
        
        ax.text(0.05,0.95,tbl,va='top',transform=ax.transAxes,
                family='monospace',fontsize=8)

    # Lowpass:
    has_lp_data=False
    if lowpass:
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
            y=lp(combined.isel(source=i).values)-offsets[i]
            if np.any(np.isfinite(y)):
                has_lp_data=True
                ax.plot(t, y, label=labels[i], **styles[i])
    fix_date_labels(ts_ax,4)
    
    # zoom to common period
    ts_ax.axis(xmin=combined.time.values[0],
               xmax=combined.time.values[-1])

    if has_lp_data:
        fix_date_labels(lp_ax,2)
    else:
        lp_ax.xaxis.set_visible(0)
        lp_ax.yaxis.set_visible(0)
        if lowpass:
            lp_ax.text(0.5,0.5,"Insufficient data for low-pass",transform=lp_ax.transAxes,
                       ha='center',va='center')
    fig.subplots_adjust(hspace=0.4)
    txt_ax.patch.set_visible(0)
    return fig
