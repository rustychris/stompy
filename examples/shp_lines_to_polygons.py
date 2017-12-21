#!/usr/bin/env python
from __future__ import print_function

from stompy.spatial import join_features
from optparse import OptionParser

try:
    from osgeo import ogr
except ImportError:
    import ogr

# # How to use this:
# ### Load shapefile
# ods = ogr.Open("/home/rusty/classes/research/spatialdata/us/ca/suntans/shoreline/noaa-medres/pacific_medium_shoreline-cropped.shp")
# output = "/home/rusty/classes/research/spatialdata/us/ca/suntans/shoreline/noaa-medres/pacific_medium_shoreline-cropped-merged.shp"
# orig_layer = ods.GetLayer(0)
# ## process it
# process_layer(orig_layer,output)

if __name__ == '__main__':

    parser = OptionParser(usage="usage: %prog [options] input.shp output.shp")
    parser.add_option("-p", "--poly",
                      help="create polygons from closed linestrings",
                      action="store_true",
                      dest='create_polygons',default=False)
    parser.add_option("-a", "--arc", dest="close_arc", default=False,
                      action="store_true",
                      help="close the largest open linestring with a circular arc")
    parser.add_option("-t","--tolerance", dest="tolerance", type="float", default=0.0,
                      metavar="DISTANCE",
                      help="Tolerance for joining two endpoints, in geographic units")
    parser.add_option("-m","--multiple", dest="single_feature", default=True,
                      action="store_false",metavar="SINGLE_FEATURE")
    
    (options, args) = parser.parse_args()
    input_shp,output_shp = args
    
    ods = ogr.Open(input_shp)
    layer = ods.GetLayer(0)

    join_features.process_layer(layer,output_shp,
                                create_polygons=options.create_polygons,close_arc=options.close_arc,
                                tolerance=options.tolerance,single_feature=options.single_feature)
    
