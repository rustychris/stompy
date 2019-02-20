#!/usr/bin/env python
"""
Command-line tool to infer missing boundary fluxes, and update them
in a flow file.

This applies to non-flow BCs in DFM, i.e. sea boundary conditions and
discharges.

"""
from __future__ import print_function

import argparse
import sys,os
import numpy as np

import stompy.model.delft.waq_scenario as waq

parser = argparse.ArgumentParser(description='Adjust DFM D-WAQ output to add missing BC fluxes.')

parser.add_argument('hyd_fn', metavar='dfm_out.hyd', type=str,
                    help='path to hyd file')

args = parser.parse_args()

hyd_fn=args.hyd_fn

print("Opening hydro from %s"%hyd_fn)
hydro=waq.HydroFiles(hyd_fn)

print("Adjusting flows, updating %s in place"%hydro.get_path('flows-file'))

hydro.adjust_boundaries_for_conservation()

print("Done")
