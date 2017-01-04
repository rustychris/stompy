from numpy import *
import field
reload(field)

#depth_bin_file = '/home/rusty/classes/research/spatialdata/us/ca/suntans/bathymetry/compiled2/final.bin'
depth_bin_file = '/home/rusty/data/sfbay/depth-compiled/depth-narrow-delta.bin'


f = field.XYZField.read(depth_bin_file)
f.build_index(index_type='stree')

center = array([  563379.6 , 4196117. ])

elev = f.inv_dist_interp(center,
                         min_n_closest=8,
                         min_radius=3900.0)
                         clip_max= -2,
                         default = -2)






