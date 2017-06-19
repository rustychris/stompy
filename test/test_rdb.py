import os
import numpy as np
import nose

from stompy.io import rdb, rdb_codes

datadir=os.path.join(os.path.dirname(__file__),'data')

def test_basic():
    fn=os.path.join(datadir,'test_rdb.rdb')
    r=rdb.Rdb(source_file=fn)

def test_xarray():
    ds1=rdb.rdb_to_dataset(os.path.join(datadir,'test_rdb.rdb'))
    ds2=rdb.rdb_to_dataset(os.path.join(datadir,'coyote.rdb'))
    
    assert ds2.time.dtype==np.datetime64

def test_codes():
    pcodes=rdb_codes.parm_codes()
    scodes=rdb_codes.stat_codes()
    pcodes.loc[60]


