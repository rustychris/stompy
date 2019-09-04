import logging as log
import argparse
import matplotlib
import os
import datetime
import .waq_scenario as waq
from ...spatial import wkb2shp

# Clean inputs
def clean_shapefile(shp_in):
    """
    break multipolygons into individual polygons.
    shp_in: path to aggregation polygon shapefile.
    returns either shp_in, unchanged, if there were no changes needed, or
      writes a new shapefile with suffix -cleaned.shp, which has edits
      to shp_in.
    """
    geoms=wkb2shp.shp2geom(shp_in)

    multi_count=0
    
    new_geoms=[]
    for fi,feat in enumerate(geoms):
        if feat['geom'].type=='Polygon':
            new_geoms.append(feat['geom'])
        else:
            multi_count+=1
            for g in feat['geom'].geoms:
                new_geoms.append(g)
    if multi_count:
        cleaned=shp_in.replace('.shp','-cleaned.shp')
        assert cleaned!=agg_grid_shp
        wkb2shp.wkb2shp(cleaned,new_geoms,overwrite=True)

        return cleaned
    else:
        return shp_in

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Manipulate transport data in D-WAQ format.')

    parser.add_argument("-a", "--aggregate", help="Path to shapefile definining aggregation polygons",default=None,type=str)
    parser.add_argument("-h", "--hyd", help="Path to hyd file for input", default=None,type=str,required=True)
    parser.add_argument("-o", "--output", help="Path to hyd file output", default="output/output.hyd")

    # these options copied in from another script, here just for reference, and possible consistency
    # in how arguments are named and described.
    
    #parser.add_argument("-g", "--grid",help="Path to DWAQ grid geometry netcdf.",default=None,required=True)
    #parser.add_argument("-r", "--reference", help="Reference date for DWAQ run (YYYY-MM-DDTHH:MM)", default=None, required=True)
    #parser.add_argument("-s", "--start", help="Date of start of output (YYYY-MM-DDTTHH:MM)",default=None)
    #parser.add_argument("-e", "--end", help="Date of end of output (YYYY-MM-DDTHH:MM)",default=None)
    #parser.add_argument("-d", "--data",help="Input data",nargs='+')
    #parser.add_argument("-i", "--interval",help="Time step in output, suffix 's' for seconds, 'D' for days", default='1D')

    args=parser.parse_args()

    # For now, all operations starts with reading the existing hydro
    hydro_orig=waq.HydroFiles(hyd_fn)

    # both specifies that the operation is aggregation, and what the aggregation geometry
    # is
    if args.aggregate: 
        



#----------



# Processing:

# remove multipolygons from inputs 
shp=clean_shapefile(agg_grid_shp)    

# create object representing aggregated hydrodynamics
# sparse_layers: for z-layer inputs this can be True, in which cases cells are only output for the
#    layers in which they are above the bed.  Usually a bad idea.  Parts of DWAQ assume
#    each 2D cell exists across all layers
# agg_boundaries: if True, multiple boundary inputs entering a single aggregated cell will be
#   merged into a single boundary input.  Generally best to keep this as False.
hydro_agg=waq.HydroAggregator(hydro_in=hydro_orig,
                              agg_shp=shp,
                              sparse_layers=False,
                              agg_boundaries=False)


# The code to write dwaq hydro is wrapped up in the code to write a dwaq model inp file,
# so we pretend to set up a dwaq simulation, even though the goal is just to write
# the hydro.
name=os.path.basename(output_fn.replace('.hyd',''))
class Writer(waq.Scenario):
    name=name
    desc=(name,
          agg_grid_shp,
          'aggregated')
    # output directory inferred from output hyd path
    base_path=os.path.dirname(output_fn)

# Define the subset of timesteps to write out, in this case the
# whole run.
sec=datetime.timedelta(seconds=1)
start_time=hydro_agg.time0+hydro_agg.t_secs[ 0]*sec
stop_time =hydro_agg.time0+hydro_agg.t_secs[-1]*sec

# probably would have been better to just pass name, desc, base_path in here,
# rather than using a shell subclass.
writer=Writer(hydro=hydro_agg,
              start_time=start_time,
              stop_time=stop_time)

# This step is super slow.  Watch the output directory for progress.
# Takes ~20 hours on HPC for the full wy2013 run.
writer.cmd_write_hydro()
