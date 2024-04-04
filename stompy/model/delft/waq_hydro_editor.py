"""
Command line interface to several manipulations of D-WAQ formatted 
hydrodynamic data.

Typical invocation:

  python -m stompy.model.delft.waq_hydro_editor -h

(to get help message).

  python -m stompy.model.delft.waq_hydro_editor -i path/to/com-foo.hyd -a agg.shp -o output_agg/output

Read existing, serial DWAQ hydro, aggregate according to polygons in agg.shp, and write to output_agg/

  python -m stompy.model.delft.waq_hydro_editor -i output_agg/com-output.hyd -c

Check DWAQ hydro for continuity.  Relative errors are typically around 1e-8, which is machine precision
for the 32-bit floats that are used. Aggregated, spliced and low-pass hydro can accumulate larger errors,
especially in the presence of wetting and drying.

  python -m stompy.model.delft.waq_hydro_editor -i path/com-input.hyd -l -o output_lp/output

Remove tides, write to output_lp.  Currently the interval for the lowpass is not exposed on the command
line, and the filter is hard-wired to be a Butterworth IIR filter.

  python -m stompy.model.delft.waq_hydro_editor -m path/to/flowfm.mdu -s -o output_splice/output

Splice a multiprocessor run into a single D-WAQ hydro dataset.  Note that in the case of an MPI run,
the original D-Flow FM mdu file is specified with -m, instead of providing the hyd file.


Caveats:

* Lowpass loads the entire dataset into RAM, so it cannot handle large or very long simulations as input.
* While it should be possible to use this script with DFM output in MapFormat=1 or MapFormat=4, there are some
  subtle differences.  It was originally written for MapFormat=1, and more recently adapted to handle 
  MapFormat=4.
* There are some places where the code makes assumptions about undocumented details of the DFM output.  It has
  been developed against rev 53925, which is probably ca. late 2017.

"""
import logging as log
import argparse
import numpy as np
import matplotlib
import os
import datetime
from . import waq_scenario as waq
from . import dflow_model as dfm
from ...spatial import wkb2shp

# Clean inputs
def clean_shapefile(shp_in):
    """
    Break multipolygons into individual polygons.
    :param shp_in: path to aggregation polygon shapefile.
    :type shp_in: str

    :return: either shp_in, unchanged, if there were no changes needed, or
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
    
def main(args=None):
    parser=argparse.ArgumentParser(description='Manipulate transport data in D-WAQ format.')
    one_of=parser.add_mutually_exclusive_group()

    parser.add_argument("-i", "--hyd", help="Path to hyd file for input", default=None,type=str)
    parser.add_argument("-m", "--mdu", help="Path to mdu file (for splicing)", default=None,type=str)

    one_of.add_argument("-a", "--aggregate", help="Path to shapefile definining aggregation polygons",default=None,type=str)
    one_of.add_argument("-c", "--continuity", help="Check continuity by comparing fluxes and volumes", action='store_true')
    one_of.add_argument("-l", "--lowpass", help="Low-pass filter", action='store_true')
    # TODO: splice output should default to mimicking what a serial run would use.
    one_of.add_argument("-s", "--splice", help="Splice an MPI run into a single DWAQ hydro dataset",action='store_true')
    
    parser.add_argument("-o", "--output", help="Path and run name for file output", default="output/output")
    parser.add_argument("-p", "--pass-parameters",help="Pass parameters through without low-pass filter",action="store_true")
    
    parser.add_argument("--keep-cells",
                        help=("When splicing skip regeneration of cells. DFM typically regenerates cells"
                              "on startup, which can introduce inconsistency between the net file and the"
                              "output. By default the same regeneration is applied here, but can be disabled"
                              "with this option"),
                        action='store_true')
    parser.add_argument("--write-only",
                        help=("Comma-separated list of file types to write. e.g. vol,srf,flo"),
                        default=None,type=str)

    parser.add_argument("--tolerance",
                        help="Distance tolerance for merging vertexes in shapefile input",
                        default=0.0,type=float)
    
    parser.add_argument("--nudging",
                        help=("Max iterations adjusting cell mapping. 0 disables entirely. Nudging tries to "
                              "assign input cells to aggregated cells so that the faces in the output also"
                              "exist in the input aggregation regions. This may not be possible, in which"
                              "case use a value of 0"),
                        default=5,type=int)
    
    # these options copied in from another script, here just for reference, and possible consistency
    # in how arguments are named and described.
    
    #parser.add_argument("-g", "--grid",help="Path to DWAQ grid geometry netcdf.",default=None,required=True)
    #parser.add_argument("-r", "--reference", help="Reference date for DWAQ run (YYYY-MM-DDTHH:MM)", default=None, required=True)
    #parser.add_argument("-s", "--start", help="Date of start of output (YYYY-MM-DDTTHH:MM)",default=None)
    #parser.add_argument("-e", "--end", help="Date of end of output (YYYY-MM-DDTHH:MM)",default=None)
    #parser.add_argument("-d", "--data",help="Input data",nargs='+')
    #parser.add_argument("-i", "--interval",help="Time step in output, suffix 's' for seconds, 'D' for days", default='1D')

    args=parser.parse_args(args=args)

    # Most operations starts with reading the existing hydro:
    if not args.splice:
        hydro_orig=waq.HydroFiles(args.hyd)
    else:
        # splice code defines the input below
        hydro_orig=None
        
    # Only some actions produce new output
    hydro_out=None
    
    # both specifies that the operation is aggregation, and what the aggregation geometry
    # is
    if args.aggregate:
        # split multipolygons to multiple polygons
        agg_shp=clean_shapefile(args.aggregate)    
        # create object representing aggregated hydrodynamics
        # sparse_layers: for z-layer inputs this can be True, in which cases cells are only output for the
        #    layers in which they are above the bed.  Usually a bad idea.  Parts of DWAQ assume
        #    each 2D cell exists across all layers
        # agg_boundaries: if True, multiple boundary inputs entering a single aggregated cell will be
        #   merged into a single boundary input.  Generally best to keep this as False.
        hydro_out=waq.HydroAggregator(hydro_in=hydro_orig,
                                      agg_shp=agg_shp,
                                      agg_shp_tolerance=args.tolerance,
                                      max_nudge_iterations=args.nudging,
                                      sparse_layers=False,
                                      agg_boundaries=False)
    if args.splice:
        assert args.mdu is not None,"Must specify MDU path"
        
        # In theory it's possible to splice MPI and aggregate at the same time.
        # for simplicity and to avoid nasty bugs, keep those steps separate.
        # load the DFM run and specify its grid as the agg_shp (this is a short
        # cut for just using the cells of an existing grid as the aggregation
        # geometry).
        model=dfm.DFlowModel.load(args.mdu)
        run_prefix=model.mdu.name
        run_dir=model.run_dir
        dest_grid=model.grid
        if not args.keep_cells:
            dest_grid.make_cells_from_edges()
        hydro_out=waq.HydroMultiAggregator(run_prefix=run_prefix,
                                           path=run_dir,
                                           agg_shp=dest_grid,
                                           agg_boundaries=False)

    if args.continuity:
        def err_callback(time_index,summary):
            log.info("Time index: %d: max volume error: %.3e  relative error: %.3e"%( time_index,
                                                                                      np.abs(summary['vol_err']).max(),
                                                                                      summary['rel_err'].max()) )

        hydro_orig.check_volume_conservation_incr(err_callback=err_callback)

    if args.lowpass:
        # TO-DO: expose these parameters on the command line
        hydro_out=waq.FilterAll(original=hydro_orig,
                                filter_type='butter',
                                filter_parameters=(not args.pass_parameters),
                                lp_secs=86400*36./24)

    # The code to write dwaq hydro is wrapped up in the code to write a dwaq model inp file,
    # so we pretend to set up a dwaq simulation, even though the goal is just to write
    # the hydro.
    if hydro_out is not None:
        if args.output is not None:
            out_name=os.path.basename(args.output.replace('.hyd',''))
            out_path=os.path.dirname(args.output)
            assert out_path!='',"Must specify a path/name combination, like path/to/name"

            # Define the subset of timesteps to write out, in this case the
            # whole run.
            sec=datetime.timedelta(seconds=1)
            start_time=hydro_out.time0+hydro_out.t_secs[ 0]*sec
            stop_time =hydro_out.time0+hydro_out.t_secs[-1]*sec

            # probably would have been better to just pass name, desc, base_path in here,
            # rather than using a shell subclass.
            writer=waq.Scenario(name=out_name,
                                desc=(out_name,"n/a","n/a"), # not used for hydro output
                                hydro=hydro_out,
                                start_time=start_time,
                                stop_time=stop_time,
                                base_path=out_path)

            # This step is super slow.  Watch the output directory for progress.
            # Takes ~20 hours on HPC for the full wy2013 run.
            if args.write_only is None:
                writer.cmd_write_hydro()
            else:
                for file_type in args.write_only.split(','):
                    log.info("Writing %s file"%file_type)
                    if file_type=='hyd':
                        hydro_out.write_hyd()
                    elif file_type=='vol':
                        hydro_out.write_vol()
                    elif file_type=='poi':
                        hydro_out.write_poi()
                    elif file_type=='links':
                        hydro_out.write_2d_links()
                    elif file_type=='bnd_links':
                        hydro_out.write_boundary_links()
                    elif file_type=='atr':
                        hydro_out.write_atr()
                    elif file_type=='srf':
                        hydro_out.write_srf()
                    elif file_type=='params':
                        hydro_out.write_parameters()
                    elif file_type=='geom':
                        hydro_out.write_geom()
                    elif file_type=='are':
                        hydro_out.write_are()
                    elif file_type=='flo':
                        hydro_out.write_flo()
                    elif file_type=='len':
                        hydro_out.write_len()
                    else:
                        raise Exception("Unknown hydro output file type: %s"%file_type)
        else:
            log.info("No output file given -- will not write out results.")
        

if __name__ == '__main__':
    main()
    
        
