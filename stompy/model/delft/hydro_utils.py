"""
Tools for additional manipulations and data munging related
to Hydro objects, but not central to typical D-WAQ hydro
usage.
"""
import datetime

import numpy as np
import xarray as xr

from ... import utils

def extract_water_level(hydro,xy,start_time,end_time):
    """ Not sure about the code location here...
    Try to pull together grid geometry to map point xy to an element,
    then to the set of segments, evaluate their volumes within the given
    time range, normalize by planform area, and pack the resulting time 
    series of free surface elevation into an xarray Dataset.  too much
    going on...
    """
    start_idx=hydro.datetime_to_index(start_time)
    end_idx  =hydro.datetime_to_index(end_time  )

    time_indexes=np.arange(start_idx,end_idx)

    g=hydro.grid()
    elt=g.select_cells_nearest(xy)
    hydro.infer_2d_elements()

    segs=np.nonzero( (hydro.seg_to_2d_element==elt) )[0]

    V=[]
    for t_idx in time_indexes:
        V.append( np.sum( hydro.volumes(hydro.t_secs[t_idx])[segs] ) )
    t_secs=hydro.t_secs[time_indexes]

    t_dts=[ hydro.time0 + datetime.timedelta(seconds=int(t_sec))
            for t_sec in t_secs ]

    seg_areas=hydro.planform_areas().data[segs]
    assert np.allclose( seg_areas[0], seg_areas )

    ds=xr.Dataset()

    ds['station']=( ('station',), [0] )
    ds['x']=( ('station',),[xy[0]] )
    ds['y']=( ('station',),[xy[1]] )

    ds['time']=( ('time',), t_dts )

    ds['water_depths']= ( ('station','time'), (np.array(V) / seg_areas[0])[None,:] )

    bottom_depth=hydro.bottom_depths().data[segs[-1]]

    ds['bottom_depth'] = ( ('station',), [bottom_depth] )
    ds['water_level'] = ds.water_depths + bottom_depth

    return ds

def extract_velocity(hydro,xy,start_time,end_time):
    """ Not sure about the code location here...
    Try to pull together grid geometry to map point xy to an element,
    then to the set of segments, then exchanges, estimate a cell-centered
    velocity based on the exchange fluxes. sketchy.
    """
    start_idx=hydro.datetime_to_index(start_time)
    end_idx  =hydro.datetime_to_index(end_time)
    time_indexes=np.arange(start_idx,end_idx)

    g=hydro.grid()
    elt=g.select_cells_nearest(xy)
    hydro.infer_2d_elements()

    segs=np.nonzero( (hydro.seg_to_2d_element==elt) )[0]


    hydro.infer_2d_elements()
    poi0=hydro.pointers-1

    ds=xr.Dataset()

    ds['station']=( ('station',), [0] )
    ds['bin']=( ('bin',), np.arange(len(segs)) )
    ds['segment'] = ( ('bin',), segs)

    ds['x']=( ('station',),[xy[0]] )
    ds['y']=( ('station',),[xy[1]] )

    t_secs=hydro.t_secs[time_indexes]

    t_dts=[ hydro.time0 + datetime.timedelta(seconds=int(t_sec))
            for t_sec in t_secs ]
    ds['time']=( ('time',), t_dts )


    # This part gets a bit sketchy -- no guarantee that there is enough
    # data in the waq output to get proper velocity.

    # these map segments to the time-independent data for extracting velocity
    seg_exchs={} # segment => array of horizontal exchange indices
    seg_matrix={} # segment => matrix of exchange normals

    def exch_normal(exch):
        elt_from,elt_to = hydro.seg_to_2d_element[ poi0[exch,:2] ]
        vec=g.cells_center()[elt_to] - g.cells_center()[elt_from]
        return vec / utils.mag(vec)

    for seg in segs:
        # only worry about horizontal exchanges
        seg_exch,sign = np.nonzero( poi0[:hydro.n_exch_x+hydro.n_exch_y,:2]==seg )

        seg_exchs[seg]=seg_exch
        seg_matrix[seg]=np.array( [exch_normal(e) for e in seg_exch] )


    U=np.zeros( (len(time_indexes),len(segs),2 ), 'f8' )
    Udavg=np.zeros( (len(time_indexes),2 ), 'f8' )
    residuals=np.zeros( (len(time_indexes),len(segs) ), 'f8' )

    for ti, t_idx in enumerate(time_indexes):
        print(ti)
        t_sec=hydro.t_secs[t_idx]
        flows=hydro.flows(t_sec)
        areas=hydro.areas(t_sec)
        vols=hydro.volumes(t_sec)

        for seg_i,seg in enumerate(segs):
            seg_exch=seg_exchs[seg]
            Bcol = flows[seg_exch] / areas[seg_exch]

            seg_uv,residual,rank,singular=np.linalg.lstsq(seg_matrix[seg],Bcol)

            rel_error=residual / utils.mag(seg_uv)
            if rel_error>0.05: # probably have to relax that..
                hydro.log.warning("Relative error in velo reconstruction %.2f"%rel_error)

            U[ti,seg_i,:] = seg_uv
            residuals[ti,seg_i] = residual
        Udavg[ti,:] = (U[ti,:,:] * vols[segs,None]).sum(axis=0) / vols[segs].sum()


    ds['u'] = ( ('time','bin'), U[:,:,0] )
    ds['v'] = ( ('time','bin'), U[:,:,1] )
    ds['u_davg'] = ( ('time',), Udavg[:,0] )
    ds['v_davg'] = ( ('time',), Udavg[:,1] )
    ds['residual'] = ( ('time','bin'), residuals )

    return ds

