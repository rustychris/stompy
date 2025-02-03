from . import depth_average
from ... import utils
import os

def main(argv=None):
    import argparse
    
    parser = argparse.ArgumentParser(description='Postprocess OpenFOAM results, esp. depth-averaging')

    parser.add_argument("-c", "--case", help="Path to case, default is working directory",
                        default=".")
    parser.add_argument("-U", "--velocity", help="Enable depth-averaged velocity output",
                        action="store_true")
    parser.add_argument("-s","--stage", help="Enable stage (water surface elevation) output",
                        action="store_true")
    parser.add_argument("-t","--time", help="Select output times as comma-separated time names, or 'all'",
                        default="all")
    parser.add_argument("-r","--res", help="Output raster cell size (positive), or count in x (negative)",
                        default=20.0, type=float)
    parser.add_argument("-f","--force", help="Force overwrite existing files",action='store_true')

    args = parser.parse_args(argv)

    pf = depth_average.PostFoam(sim_dir=args.case)
    
    if args.time=='all':
        times = pf.available_times(omit_zero=True)
    else:
        times = args.time.split(",")
    print(f"{len(times)} time step(s) to process")

    raster_info = pf.set_raster_parameters(dx=args.res)

    print(f"Output rasters will be {raster_info['ny']}x{raster_info['nx']}")
    output_path=os.path.join(args.case,"postProcessing",'python')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(f"Writing output to {output_path}")
        
    for t in utils.progress(times,msg="Time steps: %s"):
        if args.velocity:
            fn=os.path.join(output_path,f"U_{t}.tif")
            if args.force or not os.path.exists(fn):
                fld_U=pf.to_raster('U',t)
                # should put the velocity components into bands.
                fld_U.write_gdal(fn,overwrite=True)
        if args.stage:
            fn=os.path.join(output_path,f"stage_{t}.tif")
            if args.force or not os.path.exists(fn):
                fld_stage=pf.to_raster('wse',t)
                fld_stage.write_gdal(fn,overwrite=True)
        
if __name__ == '__main__':
    main()
