from stompy.spatial import field

def test_basic():
    pa=field.PyApolloniusField()

    pa.insert([0,0],10)
    pa.insert([100,100],5)

    g=pa.to_grid(100,100)

    # good to test a single point, too
    pa( [10,10] )

