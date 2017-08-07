import os
import numpy as np

from stompy.spatial import field


datadir=os.path.join( os.path.dirname(__file__), 'data')

#depth_bin_file = '/home/rusty/classes/research/spatialdata/us/ca/suntans/bathymetry/compiled2/final.bin'

def test_xyz():
    depth_bin_file = os.path.join(datadir,'depth.xyz')

    f = field.XYZText(fname=depth_bin_file)
    f.build_index()

    center = np.array([  563379.6 , 4196117. ])

    elev = f.inv_dist_interp(center,
                             min_n_closest=8,
                             min_radius=3900.0)



##

def test_lin_interp():
    X=np.array([[0.,0.],[10.,0.],[10.,10.],[0.,10.]])
    F=np.array([1.,2.,3.,4.])

    f = field.XYZField(X=X,F=F)

    elev = f.interpolate( [2,3] )

    out=f.interpolate(X)
    assert np.allclose(out,F)


