import numpy as np
import six
from stompy.io.local import coamps
six.moves.reload_module(coamps)


bounds=[340000, 610000, 3980000, 4294000]

ds=coamps.coamps_dataset( bounds=bounds,
                          start=np.datetime64("2018-01-01 00:00:00"),
                          stop= np.datetime64("2018-02-01 00:00:00"),
                          cache_dir="/home/rusty/src/sfb_ocean/wind/cache",
                          fields=['wnd_utru','wnd_vtru'],
                          fetch=False)

# that call first does
# def fetch_coamps_wind(start,stop, cache_dir, **kw):
#   to do the downloading.

# then iterates over def coamps_files(start,stop,cache_dir,fields=['wnd_utru','wnd_vtru','pres_msl']):
# to figure out what to compile
##
six.moves.reload_module(coamps)
files=coamps.coamps_files(start=np.datetime64("2018-01-01 00:00:00"),
                          stop= np.datetime64("2018-02-01 00:00:00"),
                          cache_dir="/home/rusty/src/sfb_ocean/wind/cache",
                          fields=['wnd_utru','wnd_vtru'])
 
files=list(files)
