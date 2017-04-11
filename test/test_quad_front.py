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
#  
#  ## 

reload(front)

af=front.AdvancingQuads(grid=g,scale=dim_par,perp_scale=dim_perp)
af.grid.edges['para']=0 # avoid issues during dev

af.add_existing_curve_surrounding(point_in_poly)
af.orient_quad_edge(j_init,af.PARA)

## 

# HERE: second time around it doesn't pick up a site!
sites=af.enumerate_sites()
site=sites[0]
af.resample_neighbors(site)

for a in site.actions():
    res = a.execute(site)
    
    break

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

g.plot_edges(ax=ax,clip=zoom)
g.plot_edges(mask=[j_init],color='r',lw=2)

if 0:
    af.plot_summary(label_nodes=False,clip=zoom)

site.plot(ax=ax)


zoom2=(578939.26355999336,
       579037.73609418306,
       4144480.9287384176,
       4144557.3495081947)

#ax.axis(zoom)

g.plot_nodes(labeler=lambda n,rec: str(n),
             clip=zoom2,ax=ax)
ax.axis(zoom2)

## 

# [55515, 55509] 
# nodes=np.unique(g.edges['nodes'][ res['edges'] ])

# while it might be good to do some tweaking really specific to the 
# quad case before a more blind optimization, at least try the 
# generic approach first.

## 

n=55515
f=af.cost_function(n)
print "Cost: ",f(af.grid.nodes['x'][n])

## 

# HERE
# This runs, and yet doesn't decrease the cost at all...
#import pdb
#pdb.run("af.relax_node(n)")

# it comes up with a new_f=1753, beyond the slide_limits of
# array([ 1645.5365554 ,  1713.80292975])
# and way beyond reason.

# some sort of bug with the cost function

af.relax_node(n)

