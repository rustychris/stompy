import os

from stompy.spatial.constrained_delaunay import ConstrainedXYZField

data_dir=os.path.join(os.path.dirname(__file__),'data')

def test_basic():
    basedir = data_dir

    s = ConstrainedXYZField.read_shps([os.path.join(basedir,'scale.shp'),
                                       os.path.join(basedir,'scale-lines.shp')],
                                      value_field='scale')

    sg = s.to_grid(nx=500,ny=500)


