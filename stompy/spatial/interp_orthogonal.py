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

def simple_quad_gen_to_grid(sqg,aniso=None):
    """
    Combine the patches in the given instance of 
    SimpleQuadGen, assigning 'K' to the edges.
    sqg.gen.cells should have a field 'anisotropy'
    which specifies how much smaller the off-axis
    diffusion coefficient is than the on-axis
    diffusion coefficient.
    Returns a grid with 'K' defined on edges.
    """
    joined=None
    for qg in sqg.qgs:
        grid=qg.g_final.copy()
        Klong=1.0
        try:
            Kshort=qg.gen.cells['anisotropy'][0]
        except ValueError:
            Kshort=aniso

        if len(qg.right_i)>len(qg.right_j):
            j_long=grid.edges['orient']==0
        else:
            j_long=grid.edges['orient']==90
        K=np.where( j_long, Klong, Kshort)
        grid.add_edge_field('K',K,on_exists='overwrite')
        grid.add_edge_field('long',j_long,on_exists='overwrite')

        grid.orient_cells()

        if joined:
            node_map,edge_map,cell_map=joined.add_grid(grid,merge_nodes='auto',
                                                       tol=0.01)
            joined.edges['K'][edge_map] = 0.5*(joined.edges['K'][edge_map] + grid.edges['K'])
        else:
            joined=grid
    return joined

class OrthoInterpolator(object):
    """
    Given either a curvilinear grid or a boundary for a curvilinear 
    grid, interpolate point data anisotropically and generate a raster
    DEM.
    Solves a laplacian on the nodes of the curvilinear grids. Input
    points are assigned to the nearest node.
    """
    nom_res=None # resolution for new grid if a polygon is specified

    anisotropy=0.05 # lower values means less lateral diffusion
    # alpha=1.0  # lower values mean smoother results

    background_field=None

    # If True, only samples contained in the grid outline are retained.
    clip_samples=True

    # No support yet for weights.
    # background_weight=0.02

    # tuples of (polygon,[Klon,Klat])
    # Samples within the polygon are eliminated, and the given diffusivities
    # installed. Useful if the local point distribution is too gnarly for
    # anisotropy to handle
    overrides=()
    
    def __init__(self,region,samples=None,**kw):
        """
        region: curvilinear UnstructuredGrid instance or 
         shapely.Polygon suitable for automatic quad generation
         (simple, 4 smallest internal angles are corners)
         or SimpleQuadGen instance that has been executed, and
         optionally has a cell-field called 'anisotropy'

        samples: if given, a pd.DataFrame with x,y, and value
        background field: if given, a field.Field that can be
        queried to get extra data along boundaries.
        """
        utils.set_keywords(self,kw)

        self.region=region

        if isinstance(region,unstructured_grid.UnstructuredGrid):
            self.grid=self.region
        elif isinstance(region,quads.SimpleQuadGen):
            # aniso is only used if sqg.gen doesn't have anisotropy
            self.grid=simple_quad_gen_to_grid(region,aniso=self.anisotropy)
        else:
            assert self.nom_res is not None
            self.grid=poly_to_grid(self.region,self.nom_res)

        if samples is not None:
            # Clipping only applies to these samples, not background
            # samples.  Otherwise we have to worry about boundary
            # points falling just outside the grid boundary.
            if self.clip_samples:
                boundary=self.grid.boundary_polygon()
                sel=[boundary.contains(geometry.Point(xy))
                     for xy in samples[['x','y']].values]
                samples=samples.iloc[sel,:]
            self.samples=samples
        else:
            self.samples=pd.DataFrame()

        if self.background_field is not None:
            self.add_background_samples()

        for geom,Kxy in self.overrides:
            # Which samples to drop:
            to_drop = np.array([ geom.contains(geometry.Point(xy))
                                 for xy in self.samples[ ['x','y'] ].values ])
            j_sel=self.grid.select_edges_intersecting(geom)
            # Not quite right -- doesn't respect changing orientations
            # this is just a global orient, right?
            j_long=self.grid.edges['long']
            
            self.grid.edges['K'][j_sel & j_long   ]=Kxy[0]
            self.grid.edges['K'][j_sel & (~j_long)]=Kxy[1]
            
            self.samples=self.samples[~to_drop]
            
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

        samples=self.samples
            
        dirich_idxs=[grid.select_nodes_nearest(xy)
                     for xy in samples.loc[:,['x','y']].values]
        dirich_vals=samples['value'].values
        dirich={idx:val for idx,val in zip(dirich_idxs,dirich_vals)}

        # Recover the row/col indexes of the quads:
        # Note that the order of ijs corresponds to node_idxs,
        # not natural node index.
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
        patch_nodes=np.zeros( (nrows,ncols),np.int32)-1
        # Map row,col back to grid.node index
        patch_nodes[ijs[:,0],ijs[:,1]]=node_idxs
        
        if nrows*ncols!=len(node_idxs):
            print("Brave new territory. Nodes are not in a dense rectangle")
        if not np.all(patch_nodes>=0):
            print("Yep, brave new territory.")
            
        # Build the matrix:
        N=len(node_idxs)
        M=sparse.dok_matrix( (N,N), np.float64)
        b=np.zeros(N,np.float64)

        # With the SQG code, there are two differences:
        #  grid may not be dense
        #  K is already given on edges, rather than just
        #  by grid direction

        try:
            Kedge=grid.edges['K']
            Kij=None
            print("Will use K from edges")
        except:
            Kedge=None
            Kij=[1,self.anisotropy]
            print("Will use K by grid orientation")

        # For now we only handle cases where the quad subset
        # includes the whole grid.
        # That means that the matrix here is indexed by grid.nodes,
        # rather than going through node_idxs
        
        # could be faster but no biggie right now.
        # While we iterate over nrows and ncols of patch_nodes,
        #  the matrix itself is constructed in terms of grid.node
        #  indexes.
        for row in range(nrows):
            for col in range(ncols):
                n=patch_nodes[row,col]
                if n<0:
                    continue
                
                if n in dirich:
                    M[n,n]=1
                    b[n]=dirich[n]
                else:
                    # For each cardinal direction either None,
                    # or a (node,K) tuple
                    if row==0:
                        node_north=-1
                    else:
                        node_north=patch_nodes[row-1,col]
                    if row==nrows-1:
                        node_south=-1
                    else:
                        node_south=patch_nodes[row+1,col]
                    if col==0:
                        node_west=-1
                    else:
                        node_west=patch_nodes[row,col-1]
                    if col==ncols-1:
                        node_east=-1
                    else:
                        node_east=patch_nodes[row,col+1]

                    # mirror missing nodes for a no-flux BC
                    if node_north<0: node_north=node_south
                    if node_south<0: node_south=node_north
                    if node_west<0: node_west=node_east
                    if node_east<0: node_east=node_west
                    nbrs=[node_north,node_south,node_west,node_east]
                    assert np.array(nbrs).min()>=0
                    if Kedge is not None:
                        Ks=[Kedge[grid.nodes_to_edge(n,nbr)]
                            for nbr in nbrs ]
                    else:
                        Ks=[Kij[0], Kij[0], Kij[1], Kij[1]]

                    M[n,n]=-np.sum(Ks)
                    for nbr,K in zip(nbrs,Ks):
                        M[n,nbr] = M[n,nbr] + K

        # This soln is indexed by node_idxs
        soln=sparse.linalg.spsolve(M.tocsr(),b)
        self.b=b # 0 for computational, and value for dirichlet.
        
        return soln
        
    def plot_result(self,num=1,**kw):
        plt.figure(num).clf()
        fig,ax=plt.subplots(num=num)

        ccoll=self.grid.contourf_node_values(self.result,32,cmap='jet',
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
        
    
