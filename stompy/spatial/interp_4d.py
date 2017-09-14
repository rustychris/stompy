"""
Explore 4D interpolation on the computational grid (or mildly aggregated 
form thereof)
"""

import numpy as np
import pandas as pd

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
                                return_weights=False):
    """ 
    g: instance of UnstructuredGrid
    samples: DataFrame, with fields x,y,value,weight
    (or other names given by *_col)
    alpha: control spatial smoothing.  Lower value is smoother

    returns extrapolated data in array of size [Ncells]
    """
    D=unstructured_diffuser.Diffuser(g)
    D.set_decay_rate(alpha)
    Dw=unstructured_diffuser.Diffuser(g)
    Dw.set_decay_rate(alpha)

    for i in range(len(samples)):
        rec=samples.iloc[i]
        weight=rec[weight_col]
        xy=rec[[x_col,y_col]].values
        cell=D.grid.point_to_cell(xy) or D.grid.select_cells_nearest(xy)
        D.set_flux(weight*rec[value_col],cell=cell)
        Dw.set_flux(weight,cell=cell)

    D.construct_linear_system()
    D.solve_linear_system(animate=False)
    Dw.construct_linear_system()
    Dw.solve_linear_system(animate=False)

    C=D.C_solved
    W=Dw.C_solved
    T=C / W
    if return_weights:
        return T,W
    else:
        return T


