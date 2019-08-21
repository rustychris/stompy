from __future__ import print_function
import os
import shutil
import six
import numpy as np
from stompy.io.local import cimis

# os.environ['CIMIS_KEY']='FILL_THIS_IN'

def test_cimis():
    """
    Fetch CIMIS data for station 171.  This requires a network connection
    and defining CIMIS_KEY (freely available application key) in the environment
    or above in this file.
    """
    # Start anew
    print("Get pristine cache dir")
    cache_dir='test_cache'
    os.path.exists(cache_dir) and shutil.rmtree(cache_dir)
    os.mkdir(cache_dir)

    print("Fetch first time")
    # 2/5/2001 is start of record for union city
    period=[np.datetime64('2016-01-01'),
            np.datetime64('2016-03-01')]
    df=cimis.cimis_fetch_to_xr(171,period[0],period[1],station_meta=True,
                               days_per_request='10D',cache_dir=cache_dir)
    print("Fetch second time")
    df2=cimis.cimis_fetch_to_xr(171,period[0],period[1],station_meta=True,
                                days_per_request='10D',cache_dir=cache_dir)
    print("Done")
    return df2
    
