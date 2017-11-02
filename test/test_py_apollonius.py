import numpy as np
from stompy.spatial import field

def test_basic():
    pa=field.PyApolloniusField()

    pa.insert([0,0],10)
    pa.insert([100,100],5)

    g=pa.to_grid(100,100)

    # good to test a single point, too
    pa( [10,10] )

##

def test_larger():
    # About 1.5s on basic macbook
    pa=field.PyApolloniusField()

    for it in range(1000):
        xy=1000*np.random.random(2)
        pa.insert(xy,10)

    g=pa.to_grid(200,200)

