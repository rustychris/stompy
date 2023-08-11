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

from scipy.sparse import linalg 
## 

def interp_to_time_per_ll(df,tstamp,lat_col='latitude',lon_col='longitude',
                          value_col='value'):
    """ 
    interpolate each unique src/station to the given
    tstamp, and include a time_offset
    """
    assert np.all(np.isfinite(df[value_col].values))
    
    tstamp=utils.to_dnum(tstamp)

    def interp_col(grp):
        # had been dt64_to_dnum
        dns=utils.to_dnum(grp.time.values)
        value=np.interp( tstamp,dns,grp[value_col] )
        dist=np.abs(tstamp-dns).min()
        return pd.Series([value,dist],
                         [value_col,'time_offset'])

    # for some reason, using apply() ignores as_index 
    result=df.groupby([lat_col,lon_col],as_index=False).apply(interp_col).reset_index()
    assert np.all(np.isfinite(result[value_col].values))
    return result

class PreparedExtrapolation:
    """
    Save some intermediate state to allow much faster extrapolation when the
    shape of the system does not change.
    The weighted extrapolation only uses flux BCs, which are set entirely in
    the rhs.
    Alpha cannot be dynamically changed.
    """
    def __init__(self, g, alpha=1e-5, edge_depth=None, cell_depth=None):
        self.g = g
        self.alpha = alpha
        
        self.D=unstructured_diffuser.Diffuser(self.g,edge_depth=edge_depth,cell_depth=cell_depth)
        self.D.set_decay_rate(self.alpha)

        self.prepare()
    def prepare(self):
        self.D.construct_linear_system()
        self.A = self.D.A.tocsr()

    def process(self, samples, 
                x_col='x',y_col='y',value_col='value',weight_col='weight', cell_col=None,
                return_weights=False):
        # B[:,0]: rhs for values
        # B[:,1]: rhs for weights
        D=self.D
        B=np.zeros( (D.Ncalc, 2), np.float64) # possible it will be faster if this is sparse.


        if weight_col is None:
            weights=np.ones(len(samples))
        else:
            weights=samples[weight_col].values

        values=samples[value_col].values

        if cell_col is not None:
            cells=samples[cell_col].values
        else:
            xys=samples[[x_col,y_col]].values
            cells=[ (D.grid.point_to_cell(xy) or D.grid.select_cells_nearest(xy))
                    for xy in xys]

        for i in range(len(samples)):
            weight_flux=weights[i]
            value_flux =weight_flux*values[i]
            
            # Flux boundary conditions:
            cell=cells[i]
            mic=D.c_map[cell]
            B[mic,0] -= value_flux/(D.area_c[cell]*D.dzc[cell]) * D.dt
            B[mic,1] -= weight_flux/(D.area_c[cell]*D.dzc[cell]) * D.dt
        
        # Solve both at once:
        CW_solved=linalg.spsolve(self.A,B)
        C=CW_solved[:,0]
        W=CW_solved[:,1]
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(W))
        assert np.all(W>0)
    
        T=C / W
        if return_weights:
            return T,W
        else:
            return T

    def process_raw(self, cell_weights, cell_values=None, cell_value_weights=None,
                    return_weights=False):
        """
        Even more direct. If we have per-cell value and weight, can just 
        directly use those.
        values can be unscaled, or can be supplied already scaled by weight.
        """
        D=self.D
        B=np.zeros( (D.Ncalc, 2), np.float64)
        assert D.Ncalc == len(cell_weights)

        time_per_vol=-D.dt / (D.area_c*D.dzc)

        B[:,1] = cell_weights * time_per_vol
        
        if cell_values is not None:
            B[:,0] = cell_weights * cell_values * time_per_vol
        elif cell_value_weights is not None:
            B[:,0] = cell_value_weights * time_per_vol
            
        # Solve both at once:
        CW_solved=linalg.spsolve(self.A,B)
        C=CW_solved[:,0]
        W=CW_solved[:,1]
        assert np.all(np.isfinite(C))
        assert np.all(np.isfinite(W))
        assert np.all(W>=0)
    
        T=C / W
        if return_weights:
            return T,W
        else:
            return T

        

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
            assert weight>0.0
            
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

    log.info("Construct 1st linear system")
    D.construct_linear_system()
    log.info("Solve 1st linear system")
    D.solve_linear_system(animate=False)
    log.info("Construct 2nd linear system")
    Dw.construct_linear_system()
    log.info("Solve 2nd linear system")
    Dw.solve_linear_system(animate=False)

    C=D.C_solved
    W=Dw.C_solved

    assert np.all(np.isfinite(C))
    assert np.all(np.isfinite(W))
    assert np.all(W>0)
    
    T=C / W
    if return_weights:
        return T,W
    else:
        return T


