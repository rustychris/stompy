"""
write polygons from a shapefile into an easy to parse text file
expects shapefiles with polygons with no interior rings, and 
a field 'name'
"""
from __future__ import print_function

import numpy as np
from ...spatial import wkb2shp

import shapely.wkb,shapely.geometry

import sys
import os.path
import argparse
import glob

name_field='name'

def write_polygons_to_textfile(layer,output_name):
    """ Given an array of polygons as returned from
    wkb2shp, write a text file
    of the format:

    <npolygons>
    {repeat npolygons times:}
    <name of polygon>
    <nvertices for polygon i>
    {repeat nvertices times}
    <x> <y>
    """
    fp = open(output_name,'wt')

    npolygons = len(layer)

    fp.write("%d\n"%npolygons)

    for i in range(npolygons):
        feat = layer[i]
        geom=feat['geom']
        assert geom.type=='Polygon',"All features must be polygons"
        assert len(geom.interiors)==0,"No support for interior rings"

        points = np.array(geom.exterior.coords)

        name = ""
        try:
            name=feat[name_field]
        except IndexError:
            name = "POLYGON%03d"%(i+1)
        fp.write("%s\n"%name)
        fp.write("%d\n"%points.shape[0])
        for i in range(points.shape[0]):
            fp.write("%.0f %.0f\n"%(points[i,0],points[i,1]))
    fp.close()

def write_polygons_to_folder(layer,output_dir):
    """ Given an ogr Polygon layer, write a text file
    for each polygon of the format:

     POLYGON_NAME
     Delta
     NUMBER_OF_VERTICES
     8
     EASTING           NORTHING (free format)
     601055.5219414893 4218083.957446809
     595972.2453457447 4208341.010638298
     648711.2400265958 4164709.5531914895
     ....

    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    npolygons = len(layer)
    print("npoly=%d"%npolygons)

    roster_fp = open(os.path.join(output_dir,"groups.txt"),'wt')
    roster_fp.write(" GROUP#  FILENAME\n")

    for i in range(npolygons):
        feat = layer[i]
        name = feat[name_field]
        output_name = os.path.join(output_dir,"%s.pol"%name)

        roster_fp.write(" %-7i %s\n"%(i+1,os.path.basename(output_name)))

        with open(output_name,'wt') as fp:
            geom=feat['geom']
            assert geom.type=='Polygon',"All features must be polygons"
            assert len(geom.interiors)==0,"No support for interior rings"
            points = np.array(geom.exterior.coords)

            # Write the data:
            fp.write(" POLYGON_NAME\n %s\n"%name)
            fp.write(" NUMBER_OF_VERTICES\n %d\n"%points.shape[0])
            fp.write(" EASTING   NORTHING (free format)\n")
            for i in range(points.shape[0]):
                fp.write(" %.4f %.4f\n"%(points[i,0],points[i,1]))
    roster_fp.close()

### run it

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("shp",help="Shapefile path, e.g. 'data/polygons.shp'")
    parser.add_argument("prefix",help="Output prefix" )
    parser.add_argument("-m",help="Multi-file output")
    args=parser.parse_args()

    multifile_output = True

    ## get to work...
    layer=wkb2shp.shp2geom(args.shp)
    output_txt = args.prefix + ".txt"

    write_polygons_to_textfile(layer,output_txt)

    if multifile_output:
        # and multifile output to a directory
        if not os.path.exists(args.prefix):
            os.mkdir(args.prefix)
        # clear out old files -- RH that seems a bit risky
        # [os.unlink(f) for f in glob.glob(out_prefix+'/*')]
        write_polygons_to_folder(layer,args.prefix)

