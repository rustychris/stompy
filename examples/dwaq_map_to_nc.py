#!/usr/bin/env python
"""
Command-line tool to convert a binary map output to netcdf.
"""
from __future__ import print_function

import argparse
import sys,os
import numpy as np

import stompy.model.delft.io as dio

parser = argparse.ArgumentParser(description='Convert D-WAQ binary map output to NetCDF.')

parser.add_argument('map_fn', metavar='somefile.map', type=str,
                    help='path to map file output')
parser.add_argument('hyd_fn', metavar='other.hyd', type=str,
                    help='path to hyd file')
parser.add_argument('--totaldepth',default='TotalDepth',
                    help='output variable to use as total depth. none to disable sigma coordinate')

args = parser.parse_args()
# DBG args=parser.parse_args(['--totaldepth','none',"wy2011.map","com-wy2011.hyd"])

map_fn=args.map_fn
hyd_fn=args.hyd_fn


output_fn=map_fn.replace('.map','.nc')
if os.path.exists(output_fn):
    print("Output file '%s' exists.  Aborting"%output_fn)
    sys.exit(1)

print("Reading map data and grid")
map_ds=dio.read_map(map_fn,hyd_fn)

if args.totaldepth != 'none':
    total_depth=args.totaldepth

    print("Adding minor metadata")

    if total_depth not in map_ds:
        print("Fabricating a total-depth variable to allow ugrid-ish output")
        map_ds[total_depth]=('time','layer','face'),np.ones( (len(map_ds.time),
                                                              len(map_ds.layer),
                                                              len(map_ds.face)), '<i1')
    
dio.map_add_z_coordinate(map_ds,total_depth=total_depth,coord_type='sigma',
                         layer_dim='layer')

print("Writing to %s"%output_fn)

map_ds.to_netcdf(output_fn)

