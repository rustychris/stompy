"""
Tests related to aggregating WAQ hydro
 on hold, as the current aggregation code uses netcdf net files to 
 figure out the partitions, which aren't available for the FTI
 tutorial case.
"""
from __future__ import print_function

import os
import shutil

import nose
import numpy as np
from . import waq_scenario


PC=waq_scenario.ParameterConstant
Sub=waq_scenario.Substance
IC=waq_scenario.Initial


class Testbox(waq_scenario.Scenario):
    base_path=os.path.join(os.path.dirname(__file__),'testbox')
    name="bloombox02-test"

    integration_option="15 NODISP-AT-BOUND LOWER-ORDER-AT-BOUND DISP-AT-NOFLOW"

    def cmd_default(self):
        self.cmd_write_runid()
        self.cmd_write_hydro()

def test_agg():
    dwaq_path="data/waq_hyd" # /tut_fti_waq.hyd
    hydro=waq_scenario.DwaqAggregator("box_defs.shp",
                                      "r17",dwaq_path,nprocs=None)

    scen=TextBox(hydro=hydro)

    if os.path.exists(scen.base_path):
        print("base path:",scen.base_path)
        shutil.rmtree(scen.base_path)

    scen.cmd_default()

    shutil.rmtree(scen.base_path)


if __name__=='__main__':
    nose.main()
