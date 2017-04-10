from stompy.grid import orthogonalize, front, unstructured_grid, exact_delaunay

from stompy.model.delft import dfm_grid
from stompy import utils

import matplotlib.pyplot as plt

try:
    reload
except NameError:
    from importlib import reload

## 
reload(unstructured_grid)
reload(exact_delaunay)
reload(dfm_grid)

## 

g=dfm_grid.DFMGrid("data/lsb_combined_v14_net.nc")

zoom=(578260, 579037, 4143970., 4144573)
point_in_poly=(578895, 4144375)
point_on_edge=(579003., 4144517.)
j_init=g.select_edges_nearest(point_on_edge)

dim_par=10.0 # 10m wide for edges "parallel" to j
dim_perp=20.0 # 20m long for edges "perpendicular" to j.

g.edge_to_cells()

## 

plt.figure(1).clf()

fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,clip=zoom)

ax.text(point_in_poly[0],point_in_poly[1],'START')

g.plot_edges(mask=[j_init],color='r',lw=2)

if 0:
    coll=g.plot_halfedges(values=af.grid.edges['cells'],clip=zoom,
                          ax=ax)
    coll.set_clim([-3,0])
    coll.set_lw(0)

# The idea is to start with an edge adjacent to an unfilled
# area (and with an existing cell on the other side).

# This is the starting point for paving
# paving will only use rectangles.


## 
af.plot_summary(label_nodes=False,clip=zoom)
ax.axis(zoom)

## 

# HERE:
# building the triangulation for the existing grid is way too slow -
# looks like it will take an hour? for 55k nodes.
# also might be quadratic.
# 1.3, 3.5, 5.4, 6.7
# so either need to forge ahead without building the triangulation, or
# look into how to optimize it.  should at least be possible to avoid
# quadratic.
reload(front)

af=front.AdvancingQuads(grid=g,scale=dim_par,perp_scale=dim_perp)
af.grid.edges['para']=0 # avoid issues during dev

af.add_existing_curve_surrounding(point_in_poly)
af.orient_quad_edge(j_init,af.PARA)

##

# without a richer way of specifying the scales, have to start
# with marked edges
class QuadCutoffStrategy(front.Strategy):
    def metric(self,site,scale_factors):
        return 1.0 # ?
    def execute(self,site):
        """
        Apply this strategy to the given Site.
        Returns a dict with nodes,cells which were modified 
        """
        jnew=site.grid.add_edge(nodes=[site.abcd[0],site.abcd[3]],
                                para=site.grid.edges['para'][site.js[1]] )
        cnew=site.grid.add_cell(nodes=[site.a,site.b,site.c,site.d])

        return {'edges': [jnew],
                'cells': [cnewc] }

QuadCutoff=QuadCutoffStrategy()

class QuadSite(front.FrontSite):
    def __init__(self,af,nodes):
        self.af=af
        self.grid=af.grid
        assert len(nodes)==4
        self.abcd = nodes

        self.js=[ self.grid.nodes_to_edge(nodes[:2]),
                  self.grid.nodes_to_edge(nodes[1:3]),
                  self.grid.nodes_to_edge(nodes[2:])]
        
    def metric(self):
        return 1.0 # ?
    def points(self):
        return self.grid.nodes['x'][ self.abcd ]
    
    # def internal_angle(self): ...
    # def edge_length(self): ...
    # def local_length(self): ...

    def plot(self,ax=None):
        ax=ax or plt.gca()
        points=self.grid.nodes['x'][self.abcd]
        return ax.plot( points[:,0],points[:,1],'r-o' )[0]
    
    def actions(self):
        return [QuadCutoff] # ,FloatLeft,FloatRight,FloatBoth,NonLocal?]


def enumerate_sites(self):
    sites=[]
    # FIX: This doesn't scale!
    valid=(self.grid.edges['cells'][:,:]==self.grid.UNMESHED)& (self.grid.edges['para']!=0)[:,None]
    J,Orient = np.nonzero(valid)

    for j,orient in zip(J,Orient):
        if self.grid.edges['deleted'][j]:
            continue
        he=self.grid.halfedge(j,orient)
        he_nxt=he.fwd()
        a=he.rev().node_rev()
        b=he.node_rev()
        c=he.node_fwd()
        d=he.fwd().node_fwd()

        sites.append( QuadSite(self,nodes=[a,b,c,d]) )
    return sites

sites=enumerate_sites(af)

##

# Try moving this into the site implementation -- if this is okay, then
# do the same with TriangleSite
def resample_neighbors(self):
    """ may update site! 
    """
    a,b,c,d = self.abcd
    # could extend to something more dynamic, like triangle does
    local_para=self.af.para_scale
    local_perp=self.af.perp_scale

    g=self.af.grid
    
    if g.edges['para'][self.js[1]] == self.af.PARA:
        scale=local_perp
    else:
        scale=local_para

    for n,anchor,direction in [ (a,b,-1),
                                (d,c,1) ]:
        if ( (self.grid.nodes['fixed'][n] == self.af.SLIDE) and
             self.grid.node_degree(n)<=2 ):
            n_res=self.af.resample(n=n,anchor=anchor,scale=scale,direction=direction)
            if n!=n_res:
                self.log.info("resample_neighbors changed a node")
                if n==a:
                    site.abcd[0]=n_res
                else:
                    site.abcd[3]=n_res
    # return True                

site=sites[0]
resample_neighbors(site)

##
import time

# Need to see why the DT is quadratic...
cdt=front.ShadowCDT(g,ignore_existing=True)

t_last=time.time()
for ni,n in enumerate(g.valid_node_iter()):
    if ni%100==0:
        elapsed=time.time()-t_last
        t_last=time.time()
        print("Nodes: %d/%d %.2fs per 100"%(ni,g.Nnodes(),elapsed))
    cdt.after_add_node(g,'add_node',n,x=g.nodes['x'][n])
            
##

def next100():
    for ni,n in enumerate(g.valid_node_iter()):
        if ni<300:
            continue
        if ni>=400:
            break
        cdt.after_add_node(g,'add_node',n,x=g.nodes['x'][n])


# Where's the time:
# 8.2s:
# 3.3s in check_local_delaunay.
# 1.5 in edge_to_cells.
# 0.9 in topo_sort_adjacent_nodes
# 1.4 in propagating_flip.

# without checks, it is better...
# but not great.

##

from scipy import spatial

t=time.time()
elapsed=time.time() - t

# 0.5s
print "%d points took %.2fs"%(g.Nnodes(), elapsed)

##

# outputs which are useful for us:
# sdt.vertices # [Nc,3]
# sdt.neighbors # [Nc,3]

# What state would need to be set up?
# node['x']
# edges['cells'] (with INF_CELL), edges['nodes']
# shadow_cdt: nodemap_g_to_local
reload(exact_delaunay)
reload(front)

self=front.ShadowCDT(g)
# self.bulk_init_from_grid(self,g)

# HERE - problem in the imported triangulation.

