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

#- # 

g=dfm_grid.DFMGrid("data/lsb_combined_v14_net.nc")

zoom=(578260, 579037, 4143970., 4144573)
point_in_poly=(578895, 4144375)
point_on_edge=(579003., 4144517.)
j_init=g.select_edges_nearest(point_on_edge)

dim_par=10.0 # 10m wide for edges "parallel" to j
dim_perp=20.0 # 20m long for edges "perpendicular" to j.

g.edge_to_cells()

#  plt.figure(1).clf()
#  
#  fig,ax=plt.subplots(num=1)
#  
#  g.plot_edges(ax=ax,clip=zoom)
#  
#  ax.text(point_in_poly[0],point_in_poly[1],'START')
#  
#  g.plot_edges(mask=[j_init],color='r',lw=2)
#  
#  if 0:
#      coll=g.plot_halfedges(values=af.grid.edges['cells'],clip=zoom,
#                            ax=ax)
#      coll.set_clim([-3,0])
#      coll.set_lw(0)
#  
#  # The idea is to start with an edge adjacent to an unfilled
#  # area (and with an existing cell on the other side).
#  
#  # This is the starting point for paving
#  # paving will only use rectangles.
#  
#  
#  ## 
#  af.plot_summary(label_nodes=False,clip=zoom)
#  ax.axis(zoom)

## 

reload(front)

af=front.AdvancingQuads(grid=g.copy(),scale=dim_par,perp_scale=dim_perp)
af.grid.edges['para']=0 # avoid issues during dev

af.add_existing_curve_surrounding(point_in_poly)
af.orient_quad_edge(j_init,af.PARA)

## 

af.loop(46)

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

af.grid.plot_edges(ax=ax,clip=zoom)
af.grid.plot_edges(mask=[j_init],color='r',lw=2)

if 0:
    af.plot_summary(label_nodes=False,clip=zoom)


zoom2=(578939.26355999336,
       579037.73609418306,
       4144480.9287384176,
       4144557.3495081947)


#g.plot_nodes(labeler=lambda n,rec: str(n),
#             clip=zoom2,ax=ax)
# ax.axis(zoom2)
ax.axis(zoom)

## 

# af.loop:
site=af.choose_site()

site.plot()
fig.canvas.draw()

##
self=af
# Fails here:

# trying to merge some edges - (121999, 122071)
# they have incompatible cells (one -1, one -2), *and* 122071
# is part of the site (which might be okay...)
# 121999: -2,-99 - this is probably not right.  the -99 should be -1.
# 122071: -1,-1 - one of those should be -2.
# When it starts, the outside is all -99, but ought to be -1.
# fixed the starting state, and at loop 46, looks correct.
# but that last loop mucks with the cell labels of 122072 and
# 122071
# happens during resample_neighbors.

# resample_neighbors calls free_span, and we should probably
# tell it about the other end of the site as an extra stop.
# the current problem is likely the fault of split_edge
ret=self.resample_neighbors(site)


##

zoom2=(578294.20525725419, 578455.3281456246, 4143970.8466474125, 4144095.8879623255)
af.grid.plot_edges(clip=zoom2,labeler=lambda i,r: str(i))
af.grid.plot_halfedges(clip=zoom2,labeler=lambda j,s: str(af.grid.edges['cells'][j,s]))
ax.axis(zoom2)
fig.canvas.draw()

## 
actions=site.actions()
metrics=[a.metric(site) for a in actions]
bests=np.argsort(metrics)

## 
# Problem 1: going around a bend it starts making trapezoids, 
#   never really recovers.  A more nuanced cost function with angles
#   would help here.

# Enhancement: there is an obvious place to put a triangle around a bend.
#   will get there eventually.

# Problem 2: Doesn't know how to steop.

