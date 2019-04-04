"""
Command line interface to the grid generation code.
"""
from __future__ import print_function

import sys,os,getopt

try:
    import stompy
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__),'..','..'))

try:
    from osgeo import ogr
except ImportError:
    import ogr

import shapely.wkb
import matplotlib
# Avoid interactive display in case we're on a server
matplotlib.use('Agg')

# Cairo doesn't play well with some conda installations.
import pylab 
import numpy as np

# no relative imports here, since we're trying to make this a
# command line program.
from stompy.grid import (trigrid, paver)
from stompy.spatial import (field, join_features, wkb2shp)

# Maybe we have CGAL, and can use the constrained delaunay field -
try:
    from stompy.spatial import constrained_delaunay
except ImportError:
    constrained_delaunay = False

from stompy.grid import optimize_grid
from stompy.grid.geom_types import ogr2text

class FileNotFound(Exception):
    pass

class ExitException(Exception):
    """ raised to signal the command line 
    app to exit
    """
    def __init__(self,val):
        super(ExitException,self).__init__(val)
        self.value=val
def exit(val):
    raise ExitException(val)

class Tom(object):
    scale_shps = None
    tele_scale_shps = None
    # NB: this is adjusted by scale_factor before being handed to ApolloniusGraph
    effective_tele_rate = 1.1
    boundary_shp = None
    plot_interval = None
    checkpoint_interval = None
    smooth = 1
    simplify_tolerance=0.0 # length scale for geometry simplification before smoothing
    resume_checkpoint_fn = None
    dump_checkpoint=False #
    verbosity=1
    dry_run=0
    optimize = None
    interior_shps = None
    output_shp = None
    slide_interior = 1
    scale_factor = 1.0
    scale_ratio_for_cutoff = 1.0
    output_path="."
    relaxation_iterations=4
    
    # These are not currently mutable from the command line
    # but could be.
    checkpoint_fn = "checkpoint.pav"
    plot_fn = "partial-grid.pdf"
    boundary_poly_shp = "processed-boundary.shp"
    smoothed_poly_shp = "smoothed-shoreline.shp"
    linestring_join_tolerance = 1.0
    scale_shp_field_name='scale'
    density_map = None

    
    # non-customizable instance variables
    original_boundary_geo = None
    
    def __init__(self):
        self.scale_shps = []
        self.tele_scale_shps = []

    def usage(self):
        print("tom.py   -h                   # show this help message      ")
        print("         -b boundary.shp      # boundary shapefile          ")
        print("         -i interior.shp      # interior paving guides      ")
        print("         --slide-interior     # Allow nodes on interior lines to slide [default]")
        print("         --rigid-interior     # Force nodes on interior lines to stay put")
        print("         -s scale.shp         # scale shapefile             ")
        if field.has_apollonius:
            print("         -a telescoping_scale.shp # auto telescoping scale shapefile")
            print("         -t N.NN              # telescoping rate - defaults to 1.1")
        else:
            print("         [DISABLED] -a telescoping_scale.shp")
        print("         -f N.NN              # factor for adjusting scale globally")
        print("         -I N                 # relaxation iterations, default %d"%self.relaxation_iterations)
        print("         -C N.NN              # smoothing: min number of cells across a channel")
        print("         -p N                 # output interval for plots   ")
        print("         -c N                 # checkpoint interval         ")
        print("         -d                   # disable smoothing ")
        print("         -D N.NN              # simplification length scale before smoothing")
        print("         -o                   # enable optimization ")
        print("         -O path              # set output path")
        print("         -r checkpoint.pav    # resume from a checkpoint    ")
        print("         -R checkpoint.pav    # load a checkpoint and output plot and shapefile")
        print("         -v N                 # set verbosity level N")
        print("         -n                   # ready the shoreline, but don't mesh it")
        print("         -m x1,y1,x2,y2,dx,dy # output raster of scale field")
        
        print("         -g output.shp        # output shapefile of grid")
        
        print(" boundary.shp: ")
        print("   A shapefile containing either lines or a single polygon.  If ")
        print("   lines, they must join together within a tolerance of 1.0 units")
        print(" scale.shp:")
        print("   A shapefile containing points with a field 'scale', which gives")
        print("   the desired length of the edges in a region.")
        print("   If the CGAL-python library is available, multiple shapefiles can ")
        print("   be specified, including LineString layers.")
        print(" interior.shp:")
        print("   A shapefile containg line segments which will be used to guide the orientation of cells")
        print(" output interval N: ")
        print("   Every N steps of the algorithm create a PDF of the grid so far")
        print(" checkpoint interval N:")
        print("   Every N steps of the algorithm make a backup of the grid and all")
        print("   intermediate information")
        print(" resume from checkpoint")
        print("   Loads a previously saved checkpoint file and will continue where it")
        print("   left off.")
        print(" verbosity level N:")
        print("   Defaults to 1, which prints step numbers.")
        print("   0: almost silent")
        print("   1: ~1 line per step")
        print("   2: ~30 lines per step")
        print("   3: ~100 lines per step and will try to plot intermediate stages")
        print(" raster of scale field: the given region will be rasterized and output to scale-raster.tif")
        
        
    def run(self,argv):
        try:
            opts,rest = getopt.getopt(argv[1:],'hb:s:a:t:i:c:r:R:dv:np:om:i:f:g:C:O:D:I:',
                                      ['slide-interior',
                                       'rigid-interior'])
        except getopt.GetoptError as e:
            print(e)
            print("-"*80)
            self.usage()
            exit(1)
            
        for opt,val in opts:
            if opt == '-h':
                self.usage()
                exit(1)
            elif opt == '-s':
                self.scale_shps.append(val)
            elif opt == '-a':
                self.tele_scale_shps.append(val)
            elif opt == '-t':
                self.effective_tele_rate = float(val)
            elif opt == '-f':
                self.scale_factor = float(val)
            elif opt == '-D':
                self.simplify_tolerance=float(val)
            elif opt == '-b':
                self.boundary_shp = val
            elif opt == '-p':
                self.plot_interval = int(val)
            elif opt == '-c':
                self.checkpoint_interval = int(val)
            elif opt == '-C':
                self.scale_ratio_for_cutoff = float(val)
            elif opt == '-I':
                self.relaxation_iterations = int(val)
            elif opt == '-r':
                self.resume_checkpoint_fn = val
            elif opt == '-R':
                self.resume_checkpoint_fn = val
                self.dump_checkpoint=True
            elif opt == '-d':
                self.smooth = 0
            elif opt == '-v':
                self.verbosity = int(val)
            elif opt == '-n':
                self.dry_run=1
            elif opt == '-o':
                self.optimize = 1
            elif opt == '-O':
                self.output_path=val
            elif opt == '-m':
                self.density_map = val
            elif opt == '-i':
                if not self.interior_shps:
                    self.interior_shps = []
                self.interior_shps.append( val )
            elif opt == '-g':
                self.output_shp = val
            elif opt == '--slide-interior':
                self.slide_interior = 1
            elif opt == '--rigid-interior':
                self.slide_interior = 0

        self.check_parameters()

        log_fp = open('tom.log','wt')
        log_fp.write( "TOM log:\n")
        log_fp.write( " ".join(argv) )
        log_fp.close()

        if not self.resume_checkpoint_fn:
            bound_args = self.prepare_boundary()
            density_args = self.prepare_density()

            args = {}
            args.update(bound_args)
            args.update(density_args)
            args['slide_internal_guides'] = self.slide_interior

            # Wait until after smoothing to add degenerate interior lines
            # args.update(self.prepare_interiors())

            self.p = paver.Paving(**args)
            self.p.verbose = self.verbosity

            self.p.scale_ratio_for_cutoff = self.scale_ratio_for_cutoff

            if self.smooth:
                self.p.smooth()
                # and write out the smoothed shoreline
                wkb2shp.wkb2shp(os.path.join(self.output_path,self.smoothed_poly_shp),
                                [self.p.poly],overwrite=True)

            int_args = self.prepare_interiors()

            if 'degenerates' in int_args:
                for degen in int_args['degenerates']:
                    self.p.clip_and_add_degenerate_ring( degen )
        else:
            self.p = paver.Paving.load_complete(self.resume_checkpoint_fn)
            self.p.verbose = self.verbosity

        self.p.relaxation_iterations=self.relaxation_iterations
        
        if self.dry_run:
            print("dry run...")
        elif self.density_map:
            f = self.p.density
            x1,y1,x2,y2,dx,dy = map(float,self.density_map.split(','))
            bounds = np.array( [[x1,y1],[x2,y2]] )
            rasterized = f.to_grid(dx=dx,dy=dy,bounds=bounds)
            rasterized.write_gdal( "scale-raster.tif" )
        elif self.dump_checkpoint:
            # write grid as shapefile
            if self.output_shp:
                print("Writing shapefile with %d features (edges)"%(self.p.Nedges()))
                self.p.write_shp(self.output_shp,only_boundaries=0,overwrite=1)
            self.plot_intermediate()
        else:
            starting_step = self.p.step
            self.create_grid()

            final_pav_fn=os.path.join( self.output_path,'final.pav')
            final_pdf_fn=os.path.join( self.output_path,'final.pdf')

            if (not os.path.exists(final_pav_fn)) or self.p.step > starting_step:
                self.p.write_complete(final_pav_fn)
            if (not os.path.exists(final_pdf_fn)) or self.p.step > starting_step:
                self.plot_intermediate(fn=final_pdf_fn,color_by_step=False)

            # write grid as shapefile
            if self.output_shp:
                print("Writing shapefile with %d features (edges)"%(self.p.Nedges()))
                self.p.write_shp(self.output_shp,only_boundaries=0,overwrite=1)
                # by reading the suntans grid output back in, we should get boundary edges
                # marked as 1 - self.p probably doesn't have these markers
                g = trigrid.TriGrid(suntans_path='.')
                g.write_shp('trigrid_write.shp',only_boundaries=0,overwrite=1)

            if self.optimize:
                self.run_optimization()
                self.p.write_complete('post-optimize.pav')
                self.plot_intermediate(fn='post-optimize.pdf')
                
    def check_parameters(self):
        """ make sure that the command line arguments are consistent and that
        we have what we need to proceed
        """
        show_usage = 0

        if self.resume_checkpoint_fn is None:
            if len(self.scale_shps) + len(self.tele_scale_shps) == 0:
                print("ERROR: Must specify scale shapefile or telescoping scale shapefile")
                show_usage = 1
            if self.boundary_shp is None:
                print("ERROR: Must specify boundary shapefile")
                show_usage = 1
            if len(self.tele_scale_shps)>0 and not field.has_apollonius:
                print("ERROR: Telescoping scales supplied, but Apollonius is not available")
                show_usage = 1
            if not self.smooth:
                print("WARNING: You are brave to disable smoothing.")
                print("         without smoothing the scale may be larger than the shoreline permits")
        else:
            if (len(self.scale_shps) + len(self.tele_scale_shps)) > 0 or self.boundary_shp is not None:
                print("ERROR: with resume (-r), no scale or boundary may be specified")
                show_usage = 1

        if show_usage:
            print("-"*80)
            self.usage()
            exit(1)

    def prepare_boundary(self):
        """ Prepare a polygon from the boundary shapefile - joining lines
        and nesting polygons as needed.
        Returns a dict of the arguments that need to be passed to Paving.
        """
        
        ods = ogr.Open(self.boundary_shp)
        if ods is None:
            raise Exception("Didn't find shapefile %s"%self.boundary_shp)
        
        layer = ods.GetLayer(0)

        n_features = layer.GetFeatureCount()
        feat = layer.GetNextFeature()
        geo = feat.GetGeometryRef()
        gtype = geo.GetGeometryType()
        feat_is_multi_polygon = False

        if gtype == ogr.wkbLineString:
            if self.boundary_poly_shp == self.boundary_shp:
                print("Boundary shapefile has same name as processed shapefile -")
                print("Exiting to avoid overwriting it")
                exit(1)
            join_features.process_layer(layer,self.boundary_poly_shp,
                                        tolerance=self.linestring_join_tolerance,
                                        create_polygons=True,
                                        close_arc=False)
        elif gtype == ogr.wkbPolygon:
            if n_features > 1:
                print("Sorry - no support yet for multiple polygons - must be a single polygon with holes")
                exit(1)
            self.boundary_poly_shp = self.boundary_shp
        elif gtype == ogr.wkbMultiPolygon:
            feat_is_multi_polygon = True
            self.boundary_poly_shp = self.boundary_shp
        else:
            print("Boundary shapefile has geometry of type '%s'"%ogr2text[gtype])
            print("Must be LineString or Polygon")
            exit(1)

        # Read the polygonized shoreline geometry and keep it around for plotting:
        ods = ogr.Open(self.boundary_poly_shp)
        layer = ods.GetLayer(0)
        original_geometries = []
        if feat_is_multi_polygon:
            feat = layer.GetNextFeature()
            polygons = shapely.wkb.loads( feat.GetGeometryRef().ExportToWkb() )
            max_area = 0.
            for poly in polygons.geoms:
                if poly.area > max_area:
                    largest_polygon = poly
            original_geometries.append( largest_polygon )
        else: 
            while 1:
                feat = layer.GetNextFeature()
                if feat is None:
                    break
                original_geometries.append( shapely.wkb.loads( feat.GetGeometryRef().ExportToWkb()) )

        self.original_boundary_geo = original_geometries

        # RH: 2019-02-14: In an effort to be robust against very small stepsizes,
        # try to coarsen such that the input does not contain any edges
        # shorter than min_edge_length
        # Old code:
        # return {'shp':self.boundary_poly_shp}
        # New code:
        geom=wkb2shp.shp2geom(self.boundary_poly_shp)['geom'][0]
        if self.simplify_tolerance>0:
            geom=geom.simplify(self.simplify_tolerance)
        return {'geom':geom}
    
    def prepare_interiors(self):
        if self.interior_shps is None or len(self.interior_shps)==0:
            return {}
        
        degens = []
        for shp in self.interior_shps:
            ods = ogr.Open(shp)
            if ods is None:
                raise FileNotFound("%s was not found"%shp)
            layer = ods.GetLayer(0)

            while 1:
                feat = layer.GetNextFeature()
                if not feat:
                    break
                geom = shapely.wkb.loads( feat.GetGeometryRef().ExportToWkb() )
                degens.append( np.array(geom) )
        print("Got degenerate edges:")
        print(degens)
        return {'degenerates':degens}
        
    def prepare_density(self):
        """ Load the density field and return a dict of the args to pass to Paving
        """
        if len(self.scale_shps)>0:
            if constrained_delaunay:
                density = constrained_delaunay.ConstrainedXYZField.read_shps(self.scale_shps,value_field=self.scale_shp_field_name)
            else:
                if len(self.scale_shps) == 1:
                    density = field.XYZField.read_shp(self.scale_shps[0],value_field=self.scale_shp_field_name)
                else:
                    # really this could be implemented very easily - just been too lazy to do it.
                    raise Exception("constrained_delaunay unavailable, but you specified multiple scale files")
        else:
            density = None
            
        if len(self.tele_scale_shps)>0:
            internal_tele_rate = 1+ (self.effective_tele_rate-1)/self.scale_factor
            
            tele_density = field.ApolloniusField.read_shps(self.tele_scale_shps,
                                                           value_field=self.scale_shp_field_name,
                                                           r=internal_tele_rate,
                                                           redundant_factor=0.9)
            if density is None:
                density = tele_density
            else:
                density = field.BinopField( density, np.minimum, tele_density)
                
        if self.scale_factor != 1.0:
            density = self.scale_factor * density
        return {'density':density}

    def plot_intermediate(self,fn=None,color_by_step=True):
        fig = pylab.figure(figsize=(100,100))
        ax = pylab.axes( [0,0,1,1] )

        # In larger domains, this tends to become blocky when written out to PDF
        # (maybe some sort of level-of-detail thing?)
        # better to skip it - we still have the boundary plotted below

        coll=self.p.plot() # plots edges, colored by age
        if not color_by_step:
            coll.set_array(None)
            coll.set_edgecolor('k')

        pylab.axis('equal')
        if fn is None:
            fn = self.plot_fn%self.p.__dict__
            
        pylab.savefig("temp-plot.pdf")
        os.rename("temp-plot.pdf",fn)
        pylab.close(fig)

    def create_grid(self):
        print("Starting grid generation")

        p = self.p
        
        try:
            while 1: 
                if not p.choose_and_fill():
                    break
                if self.plot_interval and p.step%self.plot_interval == 0:
                    self.plot_intermediate()
                if self.checkpoint_interval and p.step%self.checkpoint_interval == 0:
                    print("Saving checkpoint data to %s"%self.checkpoint_fn)
                    p.write_complete("temp-checkpoint.pav")
                    os.rename("temp-checkpoint.pav",self.checkpoint_fn)
                    print("Done with checkpoint")

            print("Done with creating cells")
            print("Renumbering:")
            p.renumber()
            print("Writing suntans output")
            p.write_suntans(self.output_path)
        except paver.FillFailed:
            print("Paver failed.")
            print("plotting the aftermath")
            self.plot_intermediate('crash.pdf')
            print("Saving post-crash checkpoint")
            p.write_complete('crash.pav')
            print("Goodbye.")
            raise

    def run_optimization(self):
        # May have some empty unpaved rings sitting around.
        self.p.clean_unpaved()
        opter = optimize_grid.GridOptimizer(self.p)

        opter.stats()

        opter.full()

        opter.stats()
        
        print("Done with optimizing")

        print("Overwriting suntans output with optimized grid.")
        self.p.renumber()
        print("Writing suntans output")
        self.p.write_suntans('.')

                
# For debugging
#tom = Tom()
#tom.run(['interactive','-s','scale.shp','-b','grid-boundaries.shp','-n'])

if __name__ == '__main__':
    try:
        Tom().run(sys.argv)
    except ExitException as exc:
        sys.exit(exc.value)
