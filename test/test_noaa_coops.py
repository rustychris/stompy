import numpy as np
from stompy.io.local import noaa_coops

##
ds=noaa_coops.coops_dataset("9414290",
                            np.datetime64("2015-12-10"),
                            np.datetime64("2016-02-28"),
                            ["water_level","water_temperature"],
                            days_per_request=30)

# The merge is failing, but the real problem is that the water_level data
# has duplicates within itself.
