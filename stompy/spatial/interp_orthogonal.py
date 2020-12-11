"""
Anisotropic orthogonal interpolation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import utils
from . import linestring_utils, interp_4d, wkb2shp, field
from ..grid import unstructured_grid
from ..model import unstructured_diffuser
import stompy.grid.quad_laplacian as quads
from scipy import sparse

from shapely import geometry

def poly_to_grid(poly,nom_res):
    xy=np.array(poly.exterior)
    
    gen=unstructured_grid.UnstructuredGrid(max_sides=len(xy))
    
    nodes,edges=gen.add_linestring(xy, closed=True)
    gen.add_cell(nodes=nodes[:-1]) # drop repeated node
    gen.orient_cells() # polygon may be reversed

    sqg=quads.SimpleQuadGen(gen,nom_res=nom_res,cells=[0],execute=False)
    sqg.execute()
    return sqg.qgs[0].g_final

class OrthoInterpolator(object):
    """
    Given either a curvilinear grid or a boundary for a curvilinear 
    grid, interpolate point data anisotropically and generate a raster
    DEM.
    Solves a laplacian on the nodes of the curvilinear grids. Input
    points are assigned to the nearest node.
    """
    nom_res=None # resolution for new grid if a polygon is specified

    anisotropy=0.05 # lower values means less radial diffusion
    # alpha=1.0  # lower values mean smoother results

    background_field=None

    # No support yet for weights.
    # background_weight=0.02
    
    def __init__(self,region,samples=None,**kw):
        utils.set_keywords(self,kw)

        self.region=region

        if isinstance(region,unstructured_grid.UnstructuredGrid):
            self.grid=self.region
        else:
            assert self.nom_res is not None
            self.grid=poly_to_grid(self.region,self.nom_res)

        if samples is not None:
            self.samples=samples
        else:
            self.samples=pd.DataFrame()

        if self.background_field is not None:
            self.add_background_samples()

        self.result = self.solve()
        
    def add_background_samples(self):
        bnodes=self.grid.boundary_cycle()
        bg_samples=pd.DataFrame()
        xy=self.grid.nodes['x'][bnodes]
        bg_samples['x']=xy[:,0]
        bg_samples['y']=xy[:,1]
        bg_samples['value']=self.background_field(xy)
        # bg_samples['weight']=self.background_weight
        self.samples=pd.concat( [self.samples,bg_samples] )
    
    def solve(self):
        grid=self.grid
        dirich_idxs=[grid.select_nodes_nearest(xy)
                     for xy in self.samples.loc[:,['x','y']].values]
        dirich_vals=self.samples['value'].values
        dirich={idx:val for idx,val in zip(dirich_idxs,dirich_vals)}

        # Recover the row/col indexes of the quads:
        node_idxs,ijs=grid.select_quad_subset(grid.nodes['x'][0])
        
        ij_span = ijs.max(axis=0) - ijs.min(axis=0)
        # Arbitrarily set i to be the larger dimension.  Need
        # this to be consistent to apply anisotropy
        if ij_span[0]<ij_span[1]:
            ijs=ijs[:,::-1]
        # force start at 0
        ijs-= ijs.min(axis=0)
        nrows,ncols=1+ijs.max(axis=0) # ij max is 1 less than count

        # 2D index array to simplify things below
        patch_nodes=np.zeros( (nrows,ncols),np.int32)
        assert nrows*ncols==len(node_idxs)
        patch_nodes[ijs[:,0],ijs[:,1]]=node_idxs
        
        # Build the matrix:
        N=len(node_idxs)
        M=sparse.dok_matrix( (N,N), np.float64)
        b=np.zeros(N,np.float64)
        K=[1,self.anisotropy]

        # could be faster but no biggie right now.
        for row in range(nrows):
            for col in range(ncols):
                n=patch_nodes[row,col]
                if n in dirich:
                    M[n,n]=1
                    b[n]=dirich[n]
                else:
                    M[n,n]=-2*(K[0]+K[1])
                    if row==0:
                        M[n,patch_nodes[row+1,col]]=2*K[0]
                    elif row==nrows-1:
                        M[n,patch_nodes[row-1,col]]=2*K[0]
                    else:
                        M[n,patch_nodes[row-1,col]]=K[0]
                        M[n,patch_nodes[row+1,col]]=K[0]
                    if col==0:
                        M[n,patch_nodes[row,col+1]]=2*K[1]
                    elif col==ncols-1:
                        M[n,patch_nodes[row,col-1]]=2*K[1]
                    else:
                        M[n,patch_nodes[row,col-1]]=K[1]
                        M[n,patch_nodes[row,col+1]]=K[1]

        soln=sparse.linalg.spsolve(M.tocsr(),b)
        
        return soln
        
    def plot_result(self,**kw):
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        ccoll=self.grid.plot_cells(values=self.result,ax=ax,cmap='jet',
                                   **kw)
        scat=ax.scatter( self.samples['x'] ,self.samples['y'], 40,
                         self.samples['value'], cmap='jet',
                         norm=ccoll.norm)
        ax.axis('equal')
        plt.colorbar(ccoll)

        scat.set_lw(0.5)
        scat.set_edgecolor('k')

        return fig,ax,[ccoll,scat]

    def field(self):
        fld=field.XYZField(X=self.grid.nodes['x'],F=self.result)
        fld._tri=self.grid.mpl_triangulation()
        return fld
        
    def rasterize(self,dx=None,dy=None):
        if dx is None:
            dx=self.nom_res/2 # why not?
        if dy is None:
            dy=dx
            
        fld=self.field()
        
        return fld.to_grid(self.grid.bounds(),dx=dx,dy=dy)
        
    
