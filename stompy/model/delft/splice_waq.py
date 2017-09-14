# invoke with python -m delft.splice_waq
# (assuming that delft is on PYTHONPATH
# use waq_scenario machinery to stitch a bunch of subdomains into
# a single subdomain.
# sort of a repurposing of the DWaqAggregator.

import os
import logging
import numpy as np

logger = logging.getLogger()

from ..suntans import sunreader
from . import waq_scenario
from . import sundwaq
from ... import scriptable

class SpliceScenario(waq_scenario.Scenario):
    pass 

class SunSplicer(scriptable.Scriptable):
    def cmd_merge(self,sun_dir,agg_shp="agg-cells.shp",name='global'):
        """ sun_dir [agg_shp] [name]
        sun_dir: path to directory containing suntans.dat
        agg_shp: path, absolute or relative to sun_dir, to shapefile definining
         aggregation.  Defaults to creating a non-aggregating shapefile.
        name: the name under the dwaq directory for the output.

        Combines dwaq-formatted output from multiple processors into a single
        domain.  Calls suntans_post and make_agg_shp as needed.
        """
        sun_dir=os.path.abspath(sun_dir)

        self.cmd_suntans_post(sun_dir)
        sun=sunreader.SunReader(sun_dir)

        run_prefix="sun"
        dwaq_path=os.path.join(sun.datadir,'dwaq')

        agg_shp=os.path.join(sun_dir,agg_shp)
        self.cmd_make_agg_shp(sun_dir,agg_shp)

        agg=waq_scenario.HydroMultiAggregator(agg_shp=agg_shp,
                                              run_prefix="sun",
                                              path=os.path.join(sun_dir,'dwaq'))

        scen=SpliceScenario(hydro=agg)
        scen.base_path=os.path.join(dwaq_path,"global")
        scen.name="spliced"

        scen.write_hydro()

    def cmd_make_agg_shp(self,sun_dir,agg_shp="grid-cells.shp"):
        """ sun_dir [agg_shp]
        If agg_shp does not exist, write a shapefile which retains all global
        cells (i.e. no aggregation)
        """
        # agg_shp is relative to sun_dir, unless it is an absolute
        # path
        
        agg_shp=os.path.join(sun_dir,agg_shp) # make it absolute

        if not os.path.exists(agg_shp):
            sun=sunreader.SunReader(sun_dir)
            g=sun.grid()
            fields=np.zeros(g.Ncells(),dtype=[('name',object)])
            fields['name'] =["%d"%c for c in range(g.Ncells())] 
            g.write_cells_shp(agg_shp,overwrite=True,fields=fields)

    def cmd_suntans_post(self,sun_dir):
        """ sun_dir
        sun_dir: path to directory containing suntans.dat
        Write flowgeom.nc for each subdomain of dwaq output under the given
        directory.  
        """
        sun=sunreader.SunReader(sun_dir)
        sundwaq.postprocess(sun=sun,force=False)


if __name__ == '__main__':
    splicer=SunSplicer()
    splicer.main()
