from stompy.grid import orthogonalize, front, unstructured_grid, exact_delaunay

from stompy.model.delft import dfm_grid

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

## 

zoom=(578260, 579037, 4143970., 4144573)

plt.figure(1).clf()

fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,clip=zoom)

point_in_poly=(578895, 4144375)
point_on_edge=(579003., 4144517.)
ax.text(point_in_poly[0],point_in_poly[1],'START')

j=g.select_edges_nearest(point_on_edge)

g.plot_edges(mask=[j],color='r',lw=2)

g.plot_halfedges(values=af.grid.edges['cells'],clip=zoom,
                 ax=ax)


## 

dim_par=10.0 # 10m wide for edges "parallel" to j
dim_perp=20.0 # 20m long for edges "perpendicular" to j.

# The idea is to start with an edge adjacent to an unfilled
# area (and with an existing cell on the other side).

# This is the starting point for paving
# paving will only use rectangles.

reload(front)

af=front.AdvancingQuads(grid=g,scale=dim_par,perp_scale=dim_perp)

af.plot_summary(label_nodes=False,clip=zoom)
ax.axis(zoom)

## 
self=af
x=point_in_poly

# def add_existing_curve_surrounding(self,x):

# Get the nodes:
pc=self.grid.enclosing_nodestring(x,self.grid.Nnodes())
if pc is None:
    raise Exception("No ring around this rosey")

curve_idx=self.add_curve( front.Curve(self.grid.nodes['x'][pc],closed=True) )
curve=self.curves[curve_idx]

# update those nodes to reflect their relationship to this curve.
self.grid.nodes['oring']=curve_idx


self.grid.nodes['ring_f']=curve.distances[:-1] 

for n in pc:
    degree=self.grid.node_degree(n)
    assert degree >= 2
    if degree==2:
        self.grid.nodes['fixed']=self.SLIDE
    else:
        self.grid.nodes['fixed']=self.RIGID


# Any need to update edges['cells'] ?


# for na,nb in utils.circulate_pairs(pc):
#             pairs=zip( np.arange(Ne),
#                        (np.arange(Ne)+1)%Ne)
#             for na,nb in pairs:
#                 self.grid.add_edge( nodes=[nodes[na],nodes[nb]],
#                                     cells=[self.grid.UNMESHED,
#                                            self.grid.UNDEFINED] )



## 

