reload(unstructured_grid)
reload(exact_delaunay)
reload(front)

import test_front
reload(test_front)
from test_front import *

## 
rings=sine_sine_rings()
density = field.ConstantField(25.0)


af=front.AdvancingTriangles()
af.set_edge_scale(density)

af.add_curve(rings[0],interior=False)
for ring in rings[1:]:
    af.add_curve(ring,interior=True)
af.initialize_boundaries()

af.loop()

## 
zoom=(3685.3576744887459, 3766.6617074119663, -106.27412460553033, -45.230532144628569)
zoom=(2391.5283204797729, 2467.0778002411375, -175.48232649105583, -118.75928967022534)

af.plot_summary(label_nodes=False)

site=af.choose_site()
site.plot()

plt.axis(zoom)

##

# this is okay
# af.resample_neighbors(site)
# This fails:
pdb.run("front.Resample.execute(site)")

## -----------
