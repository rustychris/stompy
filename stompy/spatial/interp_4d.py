"""
Explore 4D interpolation on the computational grid (or mildly aggregated 
form thereof)
"""

import numpy as np
import pandas as pd
import logging
log=logging.getLogger(__name__)

from .. import utils
from ..grid import unstructured_grid
from ..model import unstructured_diffuser

## 

def interp_to_time_per_ll(df,tstamp,lat_col='latitude',lon_col='longitude',
                          value_col='value'):
    """ 
    interpolate each unique src/station to the given
    tstamp, and include a time_offset
    """
    tstamp=utils.to_dnum(tstamp)

    def interp_col(grp):
        # had been dt64_to_dnum
        dns=utils.to_dnum(grp.time.values)
        value=np.interp( tstamp,dns,grp[value_col] )
        dist=np.abs(tstamp-dns).min()
        return pd.Series([value,dist],
                         [value_col,'time_offset'])

    # for some reason, using apply() ignores as_index 
    return df.groupby([lat_col,lon_col],as_index=False).apply(interp_col).reset_index()


def weighted_grid_extrapolation(g,samples,alpha=1e-5,
                                x_col='x',y_col='y',value_col='value',weight_col='weight',
                                cell_col=None,
                                edge_depth=None,cell_depth=None,
                                return_weights=False):
    """ 
    g: instance of UnstructuredGrid
    samples: DataFrame, with fields x,y,value,weight
    (or other names given by *_col)
    if cell_col is specified, this gives 0-based indices to the grid's
    cells, and speeds up the process.
    if x_col is None, or samples doesn't have x_col, point-data will be used
    from the grid geometry.

    alpha: control spatial smoothing.  Lower value is smoother

    returns extrapolated data in array of size [Ncells]
    """

    if x_col not in samples.columns:
        x_col=None
        y_col=None
        
    D=unstructured_diffuser.Diffuser(g,edge_depth=edge_depth,cell_depth=cell_depth)
    D.set_decay_rate(alpha)
    Dw=unstructured_diffuser.Diffuser(g,edge_depth=edge_depth,cell_depth=cell_depth)
    Dw.set_decay_rate(alpha)

    for i in range(len(samples)):
        if i%1000==0:
            log.info("%d/%d samples"%(i,len(samples)))
            
        rec=samples.iloc[i]
        if weight_col is None:
            weight=1.0
        else:
            weight=rec[weight_col]
        if x_col is not None:
            xy=rec[[x_col,y_col]].values
        else:
            xy=None
            
        if cell_col is not None:
            cell=int(rec[cell_col])
        else:
            cell=D.grid.point_to_cell(xy) or D.grid.select_cells_nearest(xy)

        if xy is None:
            xy=D.grid.cells_centroid([cell])[0]
            
        D.set_flux(weight*rec[value_col],cell=cell,xy=xy)
        Dw.set_flux(weight,cell=cell,xy=xy)

    log.warning("Construct 1st linear system")
    D.construct_linear_system()
    log.warning("Solve 1st linear system")
    D.solve_linear_system(animate=False)
    log.warning("Construct 2nd linear system")
    Dw.construct_linear_system()
    log.warning("Solve 2nd linear system")
    Dw.solve_linear_system(animate=False)

    C=D.C_solved
    W=Dw.C_solved
    T=C / W
    if return_weights:
        return T,W
    else:
        return T


