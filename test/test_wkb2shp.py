import os
import numpy as np
from stompy.spatial import wkb2shp
from stompy import utils
import six
six.moves.reload_module(wkb2shp)

def test_load_shp():
    feats=wkb2shp.shp2geom("data/dem_sources.shp")
    assert np.any(~utils.isnat(feats['start_date']))
    
def test_load_shp_query():
    feats=wkb2shp.shp2geom("data/dem_sources.shp")
    queries=["priority < 100",
             "start_date is null"]
    # This appears not to be supported yet.  unclear.
    # 'start_date is null or (cast(start_date as character) < "2014-09-01")']

    for query in queries:
        print(query)
        feat_sel=wkb2shp.shp2geom("data/dem_sources.shp",
                                  query=query)
        assert len(feats)>len(feat_sel)

