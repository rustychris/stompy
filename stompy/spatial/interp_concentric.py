import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import utils
from . import linestring_utils, interp_4d, wkb2shp, field
from ..grid import unstructured_grid, exact_delaunay
from ..model import unstructured_diffuser

from shapely import geometry

class ConcentricInterpolator(object):
    """
    Given a polygon and a set of samples within that polygon,
    interpolate the sample points across the interior of the
    polygon anisotropically based on distance from polygon
    boundary.
    """
    dx=None # discretized spacing of contours
    simplification=1.0 # length scale for simplifying

    min_area=0.1 # cells in triangulation with less than this area are omitted
    d_min_rel=0.01 # adjacent cells with close circumcenters get merged

    anisotropy=0.05 # lower values means less radial diffusion
    alpha=1.0  # lower values mean smoother results

    background_field=None
    background_weight=0.02
    
    def __init__(self,region,samples,**kw):
        utils.set_keywords(self,kw)

        self.region=region
        self.samples=samples

        if self.dx is None:
            # reasonable starting point
            self.dx=np.sqrt(region.area)/20

        self.rings=self.create_rings()
        self.tri  =self.create_triangulation()
        self.grid = self.create_solution_grid()

        if self.background_field is not None:
            self.add_background_samples()

        self.result = self.solve()
        
    def create_rings(self):
        inset=0
        rings=[]

        while True:
            buffered=self.region.buffer(-inset)
            if self.simplification>0:
                buffered=buffered.simplify(self.simplification)

            if buffered.area <= 0.0:
                break

            if buffered.type=='MultiPolygon':
                geoms=buffered.geoms
            else:
                geoms=[buffered]

            for geom in geoms:
                for g in [geom.exterior] + list(geom.interiors):
                    if g is None: continue
                    r=np.array( g )
                    rings.append(r)
            inset+=self.dx
        return rings

    def plot_rings(self):
        # Discretize in rings
        plt.figure(1).clf()
        fig,ax=plt.subplots(num=1)

        plot_wkb.plot_wkb(self.region,ax=ax,zorder=-1)
        scat=ax.scatter( self.samples['x'], self.samples['y'], 20, self.samples['value'], cmap='jet')

        ax.axis('equal')

        for r in self.rings:
            ax.plot( r[:,0], r[:,1], 'k-',lw=0.5)

    def create_triangulation(self):
        g=unstructured_grid.UnstructuredGrid(max_sides=3)

        for ring in self.rings:
            ring=linestring_utils.upsample_linearring(ring,self.dx)
            nodes=[g.add_or_find_node(x=x)
                   for x in ring]
            for a,b in utils.circular_pairs(nodes):
                if a==b: continue # probably repeated start/end node
                try:
                    g.add_edge(nodes=[a,b])
                except g.GridException:
                    pass
                
        tri=exact_delaunay.Triangulation()
        tri.init_from_grid(g)
        return tri

    def create_solution_grid(self):
        g2=self.tri.copy()
        g2.cells['_area']=np.nan
        g2.cells['_center']=np.nan

        areas=g2.cells_area()

        for c in np.nonzero(areas<=self.min_area)[0]:
            g2.delete_cell(c)

        centroids=g2.cells_centroid()
        for c in g2.valid_cell_iter():
            if self.region.contains( geometry.Point(centroids[c]) ): continue
            g2.delete_cell(c)

        g2.edge_to_cells(recalc=True)
        g2.delete_orphan_edges()
        g2.delete_orphan_nodes()
        g2.renumber()

        g2.modify_max_sides(5)

        g2.merge_cells_by_circumcenter(d_min_rel=self.d_min_rel)

        g2.delete_orphan_edges()
        g2.delete_orphan_nodes()
        g2.renumber()
        g2.edge_to_cells(recalc=True)
        g2.cells_center(refresh=True)

        return g2

    def add_background_samples(self):
        e2c=self.grid.edge_to_cells()
        bcells=e2c[ e2c.min(axis=1)<0,: ].max(axis=1)
        bcenters=self.grid.cells_centroid(bcells)
        bg_samples=pd.DataFrame()
        bg_samples['x']=bcenters[:,0]
        bg_samples['y']=bcenters[:,1]
        bg_samples['value']=self.background_field(bcenters) 
        bg_samples['weight']=self.background_weight
        self.samples=pd.concat( [self.samples,bg_samples] )
    
    def solve(self):
        edge_depth=np.where( self.grid.edges['constrained'], self.anisotropy, 1.0 )

        res=interp_4d.weighted_grid_extrapolation(self.grid,self.samples,alpha=self.alpha,
                                                  x_col='x',y_col='y',value_col='value',weight_col='weight',
                                                  edge_depth=edge_depth)
        return res

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

    def rasterize(self,dx,dy=None):
        if dy is None:
            dy=dx
        return field.rasterize_grid_cells(self.grid,values=self.result,
                                          dx=dx,dy=dy,stretch=True)
        
