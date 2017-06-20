from stompy.plot import cmap

def test_list_gradients():
    assert len(cmap.list_gradients())

def test_load_gradient():
    assert cmap.load_gradient('hot_desaturated.cpt') is not None
