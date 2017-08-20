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

# two problems:
# 1: resample is looking like the "best" child, so it doesn't
#    actually get anywhere.  In some cases, NonLocal has the same
#    issue.  cost in some of these cases is just None.
# 2: edges are getting corrupted somehow
#    not exactly corrupted, but nearly colinear due to a NonLocal
# in the bad step, the metric is 2.96. 
af=test_basic_setup()

af.log.setLevel(logging.INFO)
af.cdt.post_check=False
af.current=af.root=front.DTChooseSite(af)
## 
for step in range(2):
    if not af.current.children:
        break # we're done?
    
    if not af.current.best_child(): # cb=cb
        assert False

plt.clf()
af.plot_summary()

        
