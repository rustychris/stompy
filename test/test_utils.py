import numpy as np
from stompy.utils import segment_segment_intersection

def test_segment_segment_intersection():
    x1=segment_segment_intersection( np.array( [ [0,0],[1,0]] ),
                                    np.array( [ [0,0],[0,1]] ) )
    assert np.allclose( x1, [0,0])

    x2=segment_segment_intersection( np.array( [ [0,0],[1,0]] ),
                                     np.array( [ [1,0],[0,0]] ) )
    assert x2 is None

    x3=segment_segment_intersection( np.array( [ [0,0],[1,0]] ),
                               np.array( [ [0,1],[1,1]] ) )
    assert x3 is None

    x4=segment_segment_intersection( np.array( [ [0,0],[1,1]] ),
                                    np.array( [ [1,0],[0,1]] ) )
    assert np.allclose(x4,[0.5,0.5])

