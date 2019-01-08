import stompy.grid.unstructured_grid as ugrid

def test_read_ras2d():
    g2=ugrid.UnstructuredGrid.read_ras2d('data/Ex1.g01.hdf')
