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



