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
