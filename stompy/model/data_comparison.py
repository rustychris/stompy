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

from stompy.model import hydro_model as hm

from stompy import (xr_utils, utils)

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
    min_period: True => time period defined by the intersection of all the sources
      otherwise use the union of all source time periods
    """
    t_min=None
    t_max=None
    for src in all_sources:
        if len(src.time)==0:
            continue
        if (t_min is None) or (t_min>src.time.min()):
            t_min=src.time.min()
        if (t_max is None) or (t_max<src.time.max()):
            t_max=src.time.max()
    new_sources=[]
    
    for src in all_sources:
        if isinstance(src, hm.BC):
            # Now get the real data.
            src.data_start=t_min
            src.data_stop=t_max
            new_sources.append(src.data())
        else:
            new_sources.append(src)
    sources=new_sources
    
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

    # Force to ns precision to appease pandas.
    resample_bins=np.arange(utils.floor_dt64(t_min,dt),
                            utils.ceil_dt64(t_max,dt)+dt,
                            dt).astype('M8[ns]')

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

        # check for datetimes that are not ns precision
        
        bins=utils.to_dnum(resample_bins)
        # dim='time' is needed for vector-valued data to indicate not to
        # take the mean across vector components, just within bins on the
        # time axis
        # This is slow, but more general than a hand-rolled numpy solution
        # The da.groupby_bins portion is causing some heartache with new pandas.
        # it claims that non-nanosecond times are being forced to nanosecond
        # times. But when I look they appear to already be nanosecond times.
        # I think it's the bin_labels.
        da_r=(da.groupby_bins('dnum',bins,labels=bin_labels)
              .mean(dim='time')
              .rename(dnum_bins='time')
              .to_dataset())
        return da_r

    resampled=[resample(da) for da in all_sources]

    combined=xr_utils.concat_permissive(resampled,dim='source')[field_name]
    return combined


def assemble_comparison_data(models,observations,model_labels=None,
                             period='model',
                             extract_options={}):
    """
    Extract data from one or more model runs to match one or more observations
    
    models: list of HydroModel instances
    observations: list of DataArrays representing time series
      the first observation must have lon and lat fields
      defining where to extract data from in the model.

      alternatively, can pass BC object, allowing the auto-download and 
      translate code for BCs to be reused for managing validation data.

    the first observation determines what data is extracted from the
    model. if a dataarray, it should have a name of water_level or flow.
    if a BC object, then the class of the object (FlowBC,StageBC) determines
    what to extract from the model.

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

    # Collect inferred options for extracting model data, which
    # can later be overridden by extract_options
    loc_extract_opts=dict()
        
    # Convert BC instances into dataarrays
    new_obs=[]
    for oi,obs in enumerate(observations):
        if isinstance(obs,hm.BC):
            # Have to decide at this point what period of data to request
            if period=='model': # the first model, no chaining
                period=[models[0].run_start,models[0].run_stop]
                
            bc=obs
            bc.data_start=period[0]
            bc.data_stop=period[1]
            
            obs=bc.data()
            if oi==0:
                # This BC/dataarray will define where model data is extracted.
                # so try to get location information if it exists
                loc_extract_opts['name']=bc.name
                # could get fancy and try to query the gazetteer, but for now
                # just assume BC had a good name, that will match the output
                    
        new_obs.append(obs)

    orig_obs=observations
    observations=new_obs
    
    # Extract relevant variable and location from model
    base_obs=observations[0] # defines the variable and location for extracting model data
    base_var=base_obs.name # e.g. 'water_level', 'flow'

    try:
        loc_extract_opts['lon']=base_obs.lon
        loc_extract_opts['lat']=base_obs.lat
    except AttributeError:
        pass

    try:
        loc_extract_opts['x']=base_obs.x
        loc_extract_opts['y']=base_obs.y
    except AttributeError:
        pass

    if base_var=='water_level':
        loc_extract_opts['data_vars']=['water_level']
        # there are numerous very similar standard names, mostly depending
        # on the datum.  the models never know the true datum, so it's
        # arbitrary exactly which standard name is used.
    elif base_var=='flow':
        loc_extract_opts['data_vars']=['cross_section_discharge']
        # Not that many people use this...  but it's the correct one.
    elif base_var=='salinity':
        loc_extract_opts['data_vars']=['salinity']
    elif base_var=='inorganic_nitrogen_(nitrate_and_nitrite)':
        loc_extract_opts['data_vars']=['ZNit','NO3']  # want to extract both to calculate age and compare with nitrogen
    else:
        raise Exception("Not ready to extract variable %s"%base_var)
    
    loc_extract_opts.update(extract_options)
    
    model_data=[] # a data array per model
    for model,label in zip(models,model_labels):
        if base_var=='flow':
            ds=model.extract_section(**loc_extract_opts)
        else:
            ds=model.extract_station(**loc_extract_opts)

        if ds is None:
            print("No data extracted from model.  omitting")
            continue
            
        assert len(loc_extract_opts['data_vars'])>=1,"otherwise missing some data"
        tgt_vars=loc_extract_opts['data_vars']
        for tgt_var in tgt_vars:
            try:
                da=ds[tgt_var]
            except KeyError:
                # see if the variable can be found based on standard-name
                for dv in ds.data_vars:
                    if ds[dv].attrs.get('standard_name','')==tgt_var:
                        da=ds[dv]
                        da.name=tgt_var
                        break
                else:
                    raise Exception("Could not find %s by name or standard_name"%(tgt_var))

            da.name=base_var # having the same name helps later
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
        
    metrics['spearman_rho'],metrics['spearman_p']=spearmanr(x.values[valid],ref.values[valid])
    
    return metrics    


def fix_date_labels(ax,nticks=3):
    xfmt = dates.DateFormatter('%Y-%m-%d')
    xax=ax.xaxis
    xax.set_major_formatter(xfmt)
    xax.set_major_locator(dates.AutoDateLocator(minticks=nticks,maxticks=nticks+1,
                                                interval_multiples=False))
    
def calibration_figure_3panel(all_sources,combined=None,
                              metric_x=1,metric_ref=0,
                              offset_source=None,scatter_x_source=0,
                              num=None,fig=None,trim_time=False,
                              lowpass=True,
                              styles=None,ylabel=None,
                              offset_method='mean'):
    """
    all_sources: list of DataArrays to compare.
    combined: those same dataarrays interpolated to common time, or none to automatically
      do this.
    metric_x: index of the 'model' data in combined.
    metric_ref: index of the 'observed' data in combined.
    offset_source: if not None, specify the index of the source to which other
      sources will be shifted to
    scatter_x_ref: which item in combined to use for the x axis of the scatter.

    lowpass: if True, the lower left panel is a lowpass of the data, otherwise
    it will be used for the text metrics instead of overlaying them on the scatter.
    
    These default to having the reference observations as the first element, and the
    primary model output second.

    trim_time: truncate all sources to the shortest common time period

    
    offset_method: 'mean' calculates offsets between stations by mean.  'median'
     by median, which can be better when a source has noise or model crashes and
     corrupts values at the end.
    """
    N=np.arange(len(all_sources))
    if metric_ref<0:
        metric_ref=N[metric_ref]
    if scatter_x_source<0:
        scatter_x_source=N[scatter_x_source]

    if trim_time:
        t_min,t_max=period_intersection(all_sources)
        new_sources=[]
        for src in all_sources:
            tsel=(src.time.values>=t_min)&(src.time.values<=t_max)
            new_sources.append( src.isel(time=tsel) )
        all_sources=new_sources
        
    if combined is None:
        combined=combine_sources(all_sources,min_period=trim_time)
        if combined is None:
            log.warning("Combined sources was None -- likely no overlap between data sets")
            return None

    labels=list(combined.label.values)

    gs = gridspec.GridSpec(5, 3)
    if fig is not None:
        fig.clf()
    else:
        fig=plt.figure(figsize=(9,7),num=num)
    #plt.tight_layout()
    ts_ax = fig.add_subplot(gs[:-3, :])
    lp_ax = fig.add_subplot(gs[-3:-1, :-1])
    scat_ax=fig.add_subplot(gs[-3:-1, 2])
    if lowpass:
        txt_ax= fig.add_subplot(gs[-1,:])
    else:
        txt_ax=lp_ax

    if offset_method=='mean':
        offsets=combined.mean(dim='time').values
    elif offset_method=='median':
        offsets=combined.median(dim='time').values
    else:
        raise Exception("offset_method=%s is not understood"%offset_method)
    
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
            # When reading live output, it's possible for the length of
            # the time dimension and the data to get out of sync.  slc
            # clips to the shorter of the two.
            label=labels[src_i]
            if offsets[src_i]!=0.0:
                label="%s %+.2f"%(label,-offsets[src_i])
            slc=slice(None,min(src.time.shape[0],src.values.shape[0]))
            ax.plot(src.time.values[slc],src.values[slc]-offsets[src_i],
                    label=label,
                    **styles[src_i])
        ax.legend(fontsize=8,loc='upper left')
        if ylabel is not None:
            ax.set_ylabel(ylabel)

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
        # 2023-05-16: with recent pandas there is both display.precision and
        # styler.format.precision
        with pd.option_context('expand_frame_repr', False,
                               'display.precision',3):
            tbl=str(df)
            
        plt.setp(list(ax.spines.values()),visible=0)
        ax.xaxis.set_visible(0)
        ax.yaxis.set_visible(0)

        if lowpass:
            fontsize=8
            x=0.05
        else:
            # less horizontal space
            fontsize=6.5
            x=-0.05
        ax.text(x,0.95,tbl,va='top',transform=ax.transAxes,
                family='monospace',fontsize=fontsize,zorder=3)

    # Lowpass:
    has_lp_data=False
    if lowpass:
        ax=lp_ax
        t=combined.time.values

        def lp(x): 
            x=utils.fill_invalid(x)
            dn=utils.to_dnum(t)
            # cutoff for low pass filtering, must be 2 * cutoff days after start or before end of datenums
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
        if ylabel is not None:
            ax.set_ylabel(ylabel)
            
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
