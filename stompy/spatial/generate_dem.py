"""
Command line interface to generating a DEM from a collections of
source data described in a shapefile.
"""
from stompy.spatial import field, wkb2shp
import os, glob
import numpy as np
import logging
import subprocess
log=logging.getLogger("generate_dem")

# global variable holding the composite field
dataset=None

def create_dataset(args):
    """
    Parse the shapefile and source definitions, but do not render
    any of the DEM.
    """
    paths=args.path

    # Find out the projection, so we can reproject rasters as needed
    sources,projection=wkb2shp.shp2geom(args.shapefile,return_srs=True)

    # field.CompositeField has a generic interface for sources, via
    # a 'factory' method.
    def factory(attrs):
        xyxy=attrs['geom'].bounds
        geo_bounds=[xyxy[0], xyxy[2], xyxy[1], xyxy[3]]

        if attrs['src_name']=='':
            # for some data_mode, don't need a source, but upstream
            # code expects something. here is something.
            return field.ConstantField(0.0)

        if attrs['src_name'].startswith('py:'):
            expr=attrs['src_name'][3:]
            # something like 'ConstantField(-1.0)'
            # a little sneaky... make it look like it's running
            # after a "from stompy.spatial.field import *"
            # and also it gets fields of the shapefile
            field_hash=dict(field.__dict__)
            # convert the attrs into a dict suitable for passing to eval
            attrs_dict={}
            for name in attrs.dtype.names:
                attrs_dict[name]=attrs[name]
            return eval(expr,field_hash,attrs_dict)
        
        # Otherwise assume src_name is a file name or file pattern.
        for p in paths:
            full_path=os.path.join(p,attrs['src_name'])
            files=glob.glob(full_path)
            if len(files)>1:
                mrf=field.MultiRasterField(files)
                return mrf
            elif len(files)==1:
                # Is this an okay place to test for projection?
                # HERE: attrs may have 'projection', and want to use that if present to override
                # when loading here:
                if 'projection' in attrs.dtype.names:
                    source_projection=attrs['projection']
                    if source_projection=='':
                        source_projection=None # defaults to GDAL's projection info
                else:
                    source_projection=None

                gg=field.GdalGrid(files[0],geo_bounds=geo_bounds,target_projection=projection,
                                  source_projection=source_projection)
                gg.default_interpolation='linear'
                return gg
        
        log.warning("Source %s was not found -- ignoring"%attrs['src_name'])
        return None
        
    comp_field=field.CompositeField(shp_fn=args.shapefile,
                                    shp_query=args.query,
                                    target_date=args.date,
                                    factory=factory,
                                    priority_field='priority',
                                    data_mode='data_mode',
                                    alpha_mode='alpha_mode')
    return comp_field


def process_tile(args,mask_poly=None,overwrite=False):
    fn,xxyy,res = args

    bleed=150 # pad out the tile by this much to avoid edge effects

    if overwrite or (not os.path.exists(fn)):
        #try:
        xxyy_pad=[ xxyy[0]-bleed,
                   xxyy[1]+bleed,
                   xxyy[2]-bleed,
                   xxyy[3]+bleed ]
        dem=dataset.to_grid(dx=res,dy=res,bounds=xxyy_pad,mask_poly=mask_poly)

        if bleed!=0:
            dem=dem.crop(xxyy)
        if overwrite and os.path.exists(fn):
            os.unlink(fn)
        dem._projection=dataset.projection
        dem.write_gdal(fn)
    else:
        log.info("File exists")

def config_logging(args):
    levels=[logging.ERROR,logging.WARNING,logging.INFO,logging.DEBUG]
    if args.verbose>=len(levels): level=levels[-1]
    else: level=levels[args.verbose]
    log.setLevel(level)
    handler=logging.StreamHandler()
    log.addHandler(handler)
    bf = logging.Formatter('[{asctime} {name} {levelname:8s}] {message}',
                           style='{')
    handler.setFormatter(bf)
    for handler in log.handlers:
        handler.setLevel(level)

##

def parse_args(argv):
    import argparse

    parser=argparse.ArgumentParser(description='Generate DEM from multiple sources.')

    parser.add_argument("-s", "--shapefile", help="Shapefile defining sources")
    parser.add_argument("-o", "--output", help="Directory for writing output, default 'output'", default="output")
    parser.add_argument("-p", "--path", help="Add to search path for source datasets",
                        action='append',default=["."])
    parser.add_argument("-m", "--merge", help="Merge the tiles once all have been rendered",action='store_true')
    parser.add_argument("-r", "--resolution", help="Output resolution, default 10", default=10.0,type=float)
    parser.add_argument("-b", "--bounds",help="Bounds for output xmin xmax ymin ymax",nargs=4,type=float)
    parser.add_argument("-t", "--tilesize",help="Make tiles NxN, default 1000",default=None,type=int)
    parser.add_argument("-v", "--verbose",help="Increase verbosity",default=1,action='count')
    parser.add_argument("-f", "--force",help="Overwrite existing tiles",action='store_true')
    parser.add_argument("-q", "--query",help="Query to select subset of features",default=None)
    parser.add_argument("-d", "--date",help="Target date",default=None,type=np.datetime64)
    parser.add_argument("-g", "--grid",help="Mask region by grid outline",default=None)
    parser.add_argument("--buffer",help="Buffer distance beyond grid",default=100.0)

    return parser.parse_args(args=argv)
    
def main(argv=None):
    global dataset
    args=parse_args(argv)
    
    config_logging(args)
    
    dataset=create_dataset(args)

    dem_dir=args.output
    if not os.path.exists(dem_dir):
        log.info("Creating output directory %s"%dem_dir)
        os.makedirs(dem_dir)

    res=args.resolution
    if args.tilesize is not None:
        tile_x=tile_y=res*args.tilesize
    else:
        tile_x=tile_y=None

    total_bounds=args.bounds

    if args.grid is not None:
        from ..grid import cli
        # mimic the format in grid.cli
        fmt,path=args.grid.split(':')
        log.info("Reading %s as %s"%(path,fmt))
        if fmt in cli.ReadGrid.formats:
            grid=cli.ReadGrid.formats[fmt][1](fmt,path)
        else:
            log.error("Did not understand format %s"%fmt)
            log.error("Read formats are: %s"%(cli.ReadGrid.format_list()))
            raise Exception("Bad grid format")

        poly=grid.boundary_polygon()
        poly_buff=poly.buffer(args.buffer)

        if total_bounds is None:
            xyxy=np.array(poly_buff.bounds)
            total_bounds=[xyxy[0],xyxy[2],xyxy[1],xyxy[3]]
            log.info('Bounds will default to grid+buffer, %s'%str(total_bounds))
    else:
        poly_buff=None
    
    # round out to tiles
    if tile_x is not None:
        total_tile_bounds= [np.floor(total_bounds[0]/tile_x) * tile_x,
                            np.ceil(total_bounds[1]/tile_x) * tile_x,
                            np.floor(total_bounds[2]/tile_y) * tile_y,
                            np.ceil(total_bounds[3]/tile_y) * tile_y ]

        calls=[]
        for x0 in np.arange(total_tile_bounds[0],total_tile_bounds[1],tile_x):
            for y0 in np.arange(total_tile_bounds[2],total_tile_bounds[3],tile_y):
                xxyy=(x0,x0+tile_x,y0,y0+tile_y)
                fn=os.path.join(dem_dir,"tile_res%g_%.0f_%.0f.tif"%(res,x0,y0))
                calls.append( [fn,xxyy,res] )
    else:
        fn=os.path.join(dem_dir,"output_res%g.tif"%(res))
        calls=[ [fn,total_bounds,res] ]

    log.info("%d tiles"%len(calls))
    
    if 0:
        # this won't work for now, as dataset is created in the __main__ stanza
        p = Pool(4)
        p.map(f, calls )
    else:
        for i,call in enumerate(calls):
            log.info("Call %d/%d %s"%(i,len(calls),call[0]))
            process_tile(call,mask_poly=poly_buff,overwrite=args.force)

    if args.merge:
        if tile_x is None:
            log.warning("Cannot merge when not tiling")
        else:
            # and then merge them with something like:
            # if the file exists, its extents will not be updated.
            output_fn=os.path.join(dem_dir,'merged.tif')
            os.path.exists(output_fn) and os.unlink(output_fn)

            log.info("Merging %d tiles using gdal_merge.py"%len(calls))

            # Try importing gdal_merge directly, which will more reliably
            # find the right library since if we got this far, python already
            # found gdal okay.  Unfortunately it's not super straightforward
            # to get the right way of importing this, since it's intended as
            # a script and not a module.
            try:
                # This way seems deprecated, and fails outright on 2022-03
                # windows install.
                #from Scripts import gdal_merge
                # This is maybe sanctioned, tho
                from osgeo_utils import gdal_merge
            except ImportError:
                log.info("Failed to import gdal_merge, will try subprocess",exc_info=args.verbose>1)
                gdal_merge=None

            tiles=glob.glob("%s/tile*.tif"%dem_dir)
            cmd=["python","gdal_merge.py","-init","nan","-a_nodata","nan",
                 "-o",output_fn]+tiles

            log.info(" ".join(cmd))

            if gdal_merge:
                gdal_merge.main(argv=cmd[1:])
            else:
                # more likely that gdal_merge.py is on PATH, than the script itself will
                # be seen by python, so drop python, and invoke script directly.
                subprocess.call(" ".join(cmd[1:]),shell=True)


if __name__=='__main__':
    main()

