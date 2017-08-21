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

# gets pretty far, then corrupts some shoreline
# while getting an intersecting constraint error
af.loop(283)

# 

##
af.loop(1)
zoom=(3685.3576744887459, 3766.6617074119663, -106.27412460553033, -45.230532144628569)
zoom=(2391.5283204797729, 2467.0778002411375, -175.48232649105583, -118.75928967022534)
zoom=(617.5288764629438, 851.27615881930399, 122.37679637174006, 241.99027866141051)
af.plot_summary(label_nodes=False)

site=af.choose_site()
site.plot()

plt.axis(zoom)

##

# this is what fails.
af.resample_neighbors(site)
