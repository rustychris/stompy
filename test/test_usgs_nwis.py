import os
import shutil
import time

import numpy as np
from stompy.io.local import usgs_nwis

## 

def test_basic():
    # This requires internet access!
    ds=usgs_nwis.nwis_dataset(station="11337190",
                              start_date=np.datetime64('2012-08-01'),
                              end_date  =np.datetime64('2012-10-01'),
                              products=[60, # "Discharge, cubic feet per second"
                                        10], # "Temperature, water, degrees Celsius"
                              days_per_request=30)


def test_provisional():
    usgs_a8_site="372512121585801"

    ds=usgs_nwis.nwis_dataset(usgs_a8_site,
                              np.datetime64("2015-12-10"),
                              np.datetime64("2015-12-20"),
                              products=[72178])

def test_missing():
    station="11162765"
    t_start=np.datetime64('2016-10-01')
    t_stop=np.datetime64('2016-12-01')
    # This period has some missing data identified by '***' which 
    # caused problems in older versions of rdb.py
    ds=usgs_nwis.nwis_dataset(station,
                              t_start,t_stop,
                              products=[95,90860],
                              days_per_request=20)

def test_caching():
    station="11162765"
    t_start=np.datetime64('2016-10-01')
    t_stop=np.datetime64('2016-12-01')

    cache_dir='tmp_cache'
    if os.path.exists(cache_dir):
        # Start clean
        shutil.rmtree(cache_dir)

    os.mkdir(cache_dir)

    timings=[]
    for trial in [0,1]:
        t0=time.time()
        ds=usgs_nwis.nwis_dataset(station,
                                  t_start,t_stop,
                                  products=[95,90860],
                                  days_per_request='10D',cache_dir=cache_dir)
        timings.append(time.time() - t0)

    assert timings[0]>5*timings[1]
