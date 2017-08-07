"""
OLD -- this module has not been fully modernized, and most likely will not 
import.

 forcing.py: construct forcing (and eventually initialization?) for suntans runs.

 the total process is something like this:
   1) Write a forcing_conf.py file that goes with a specific grid
      This file constructs groups of edges that will be forced and the datasets
      that are used to force them.  Only need the original points/edges/cells.dat,
      not the partitioning.  This description is built up using classes from this
      file.

   2) For a particular partitioning, run forcing_conf.py partition to dice up the forcing
      information and write boundaries.dat.nnn files for each partition.

   done.
""" 
from __future__ import print_function
import sys, os, glob

import logging

import netCDF4

from . import sunreader
from . import timeseries
from ...grid import trigrid
from ... import filters as lp_filter
from ...spatial import wkb2shp


try:
    if netCDF4.__version__ >= '1.2.6':
        # kludge:
        # recent netCDF4 isn't compatible with cfunits due to renaming
        # some datetime internals, which cfunits tries to reach in and grab.
        # monkey patch in shame.
        nct = netCDF4.netcdftime.netcdftime = netCDF4.netcdftime
        nct._DateFromNoLeapDay = nct.DatetimeNoLeap
        nct._DateFromAllLeap   = nct.DatetimeAllLeap
        nct._DateFrom360Day    = nct.Datetime360Day
    
    from cfunits.units import Units
except ImportError:
    Units=None

try:
    import qnc
except ImportError:
    qnc=None

from ... import (tide_consts, utils)
# cache_handler relies on some pieces which are no longer easily accessible.
# from cache_handler import urlopen_delay as urlopen

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from scipy.interpolate import interp1d

import datetime
from shapely import geometry

forcing_dir= os.path.join( os.environ['HOME'], "models/forcing")

try:
    from shapely.prepared import prep as geom_prep
except ImportError:
    geom_prep = None

hydro_forcings = ['FORCE_H','FORCE_U', 'FORCE_U_VECTOR', 'FORCE_Q','FORCE_SURFACE_Q']

class GlobalForcing(object):
    def __init__(self,datadir=None,sun=None,proc=None):
        self.log=logging.getLogger(self.__class__.__name__)

        if sun:
            self.sun = sun
        else:
            if datadir is None:
                datadir='.'
            self.sun = sunreader.SunReader(datadir)

        # if this is None, we're looking at forcing for the whole domain
        # otherwise, it's forcing after domain decomposition
        self.proc = proc
        self.forcing_groups = []
        self.datasources = []

    # some functios to make it look more like the old forcing class
    def has_forced_edges(self):
        """ true if there are actual forced edges or cells """
        return len(self.forcing_groups) > 0
    
    def uses_gages(self):
        """ kludge: guess if it uses gages based on whether the datasources
        are timeseries or harmonics
        """
        for ds in self.datasources:
            if isinstance(ds,Timeseries):
                return True
        return False

    def uses_predictions(self):
        """ kludge: guess if it uses harmonics based on whether the datasources
        are timeseries or harmonics
        """
        for ds in self.datasources:
            if isinstance(ds,Harmonics):
                return True
        return False
                   
    def new_group(self,**kwargs):
        """ create an edge group, add it to the GlobalForcing, and return the object
        """
        self.forcing_groups.append( ForcingGroup(self,**kwargs) )
        return self.forcing_groups[-1]

    def add_groups_bulk(self,defs=None,shp=None,bc_type_field='bc_type'):
        """ defs: array as returned by wkb2shp.shp2geom
        has a geom field which will be used to match against edges/cells.
        if there is a field matching bc_type_field, it will be used to 
        discern the type of forcing, and specifically whether to identify
        cells or edges.
        """
        if defs is None:
            assert shp is not None
            defs=wkb2shp.shp2geom(shp)

        if bc_type_field in defs.dtype.names:
            bc_types=defs[bc_type_field]
        else:
            # default to edges
            bc_types=['BOUNDARY']*len(defs)

        groups=[None]*len(defs)

        warn_on_fake=True

        for feat_id in range(len(defs)):
            fields={ fld:defs[fld][feat_id]
                     for fld in defs.dtype.names }
            geo = fields['geom']
            points = np.array(geo)

            typ=bc_types[feat_id]

            if typ=='BED':
                typ='SURFACE'
                if warn_on_fake:
                    self.log.debug("Will fake the bed source with a surface source")
                    warn_on_fake=False

            # Define the forcing group:
            if typ in ['SURFACE','BED']:
                grp = self.new_group( nearest_cell=points[0] )
            else:
                if len(points) == 1:
                    grp = self.new_group( nearest=points[0] )
                else:
                    # just use endpoints to define an arc along the boundary.
                    grp = self.new_group( points=[points[0],
                                                  points[-1]] )
            groups[feat_id]=grp

        return groups

    def add_datasource(self,ds):
        """ returns the index of the datasource
        """
        if ds not in self.datasources:
            self.datasources.append( ds )
            ds.filename ='%04d'%self.datasources.index(ds)

            # UPDATE: go ahead and try moving as much of the downloading
            #   and time-consuming work to right before the data is written
            #   out.  This should help avoid unnecessary processing when
            #   all we need to know is which edges are being forced.

            # OLD NOTES:
            # this could also be moved to right before writing out config files,
            # so that information from other datasources or which forcing_group
            # a datasource is tied to could be utilized.
            # the downside is that we might want to call plot() or similar before
            # or instead of writing a config file.
            # Also, with the new Kriging code, ds.prepare is where the Kriging class
            # adds its subsources into the global forcing.
            ds.prepare(self)
            
        return self.datasources.index(ds)

    def write_fs_initial_dat(self):
        print("Attempting to figure out a good initial freesurface")
        best_ds=None
        best_n_edges=0
        
        for fg in self.forcing_groups:
            for bc_type,ds in fg.datasources:
                if bc_type == 'FORCE_H':
                    n_edges = fg.Nedges()
                    if n_edges > best_n_edges:
                        best_ds = ds
                        best_n_edges = n_edges
        if best_n_edges > 0:
            print("Found a good datasource for getting the freesurface")
            # evaluate the datasource at the desired time,
            start = self.sun.time_zero()
            start_val = self.datasources[best_ds].calc_values( date2num(start) )

            fp = open( os.path.join(self.sun.datadir,'fs_initial.dat'), 'wt')
            fp.write("%8f\n"%start_val)
            fp.close()
            
    def write_boundaries_dat(self):
        """ write boundaries.dat.* for each processor
        if possible, also write an fs_initial.dat file with a reasonable choice
        of initial freesurface elevation.
        """
        # The trick here is to map global edges to local edges
        # luckily the points don't change
        self.write_fs_initial_dat()

        # Write out the datasources first, in their own directory
        ds_subdir = 'datasources'
        
        datasource_dir = os.path.join( self.sun.datadir,ds_subdir)
        if not os.path.exists(datasource_dir):
            os.mkdir(datasource_dir)
            
        for i,d in enumerate(self.datasources):
            ## d.filename is now populated at time of creation to ease referencing from one datasource
            ## to another
            ds_path = os.path.join( self.sun.datadir, ds_subdir, d.filename )
            fp = open( ds_path, 'wt')
            d.write_config(fp,self.sun)
            fp.close()

        for proc in range(self.sun.num_processors()):
            print("Writing boundary data for processors %d"%proc)

            self.write_boundaries_dat_proc(proc)
            
    def write_boundaries_dat_proc(self,proc):
        #print "loading global grid"
        g = self.sun.grid()
        #print "loading per-proc grid"
        gproc = self.sun.grid(proc)

        fp = open(self.sun.file_path('BoundaryInput',proc),'wt')

        fp.write("BOUNDARY_FORCING 6\n")
        # for now, each processor gets all of the data sections

        #print "Mapping groups"
        # first cycle through to figure out which groups have edges on this processor
        mapped_groups = [fg.map_to_grid(g,gproc) for fg in self.forcing_groups]

        mapped_groups = [fg for fg in mapped_groups if fg.nonempty()]

        # Split those into edge-based and cell-based:
        edge_based = [fg for fg in mapped_groups if fg.edge_based()]
        cell_based = [fg for fg in mapped_groups if fg.cell_based()]

        fp.write("ITEMLIST_COUNT %d\n"%(len(edge_based)+len(cell_based)))

        ## Write Edge based
        for fg in edge_based:
            fg.write_config(fp,self.sun)
            
        ## Write Cell based
        all_cells = []
        for fg in cell_based:
            fg.write_config(fp,self.sun)
            if fg.cells is not "all":
                all_cells.append( fg.cells )

        if len(all_cells)>0:
            all_cells = concatenate(all_cells)
            if len(all_cells) > len(unique(all_cells)):
                print("All cells for proc %d: %s"%(proc,all_cells))
                print("Looks like there are duplicates, for which we are not prepared!")
                raise Exception("Duplicate cells in forcing")

        fp.close()

    def update_grid(self,target_path=None):
        """ Given the forcing groups defined, rewrite edges.dat, adjusting
        edge markers as necessary.

        Currently this only operates on the global grid, so if anything
        changes, you will have to repartition
 
        target_path: if specified, write the new grid here, instead of overwriting the
          old one.

        Returns 1 if the grid changed and needs to be repartitioned, 0 otherwise
        """
        g = self.sun.grid()

        # keep track of who should be marked
        marked = []
        changed = 0
        
        for fg in self.forcing_groups:
            edges = fg.edges
            if edges is None or len(edges)==0:
                continue

            if fg.hydro_bctype() in ["FORCE_H"]:
                marker = 3
            elif fg.hydro_bctype() in ["FORCE_U","FORCE_U_VECTOR","FORCE_Q"]:
                marker = 2
            else:
                print("update_grid: no hydrodynamic forcing found")
                continue

            if len(edges) == 0:
                print("WARNING: forcing group %s has no edges"%fg)
                continue
            
            marked.append(edges)
            print("update_grid: edges is ",edges)
            if any( g.edges[edges,2] != marker ):
                print("Writing in new markers=%d"%marker)
                changed = 1
                g.edges[edges,2] = marker

        # the ones we expect to have >1 markers
        if len(marked) > 0:
            marked = concatenate( marked )
        else:
            marked = array([],int32)
            
        extras = setdiff1d( where(g.edges[:,2] > 1)[0], marked )

        if len(extras)>0:
            print("There were extra markers in places - they will be set to closed=1")
            g.edges[extras,2] = 1
            changed = 1

        if changed or target_path is not None:
            if target_path is None:
                target_path = self.sun.datadir
                
            print("There were changes - writing out new global grid")
            g.write_suntans(target_path)
            print("Reloading grid")
            self.sun = sunreader.SunReader(target_path)
            print("You will need to repartition the grid!")
            return 1
        else:
            print("No changes in grid markers.")
            return 0
        

class ForcingGroup(object):
    """  A group of features (cells or edges) that will get the same forcing.
    """
    def __init__(self,
                 gforce,
                 edges=None, # list of ids of edges - *must* match the global-ness of
                             # GlobalForcing.
                 nodes=None,     # pair of global node ids, connected by shortest path along edges on the boundary
                 points=None,   # pair of coordinates, connect by shortest path
                 nearest=None,    # single coordinate pair, force the one edge closest
                 cells=None, # list of cell ids, or 'all'
                 boundary_cells=False, # if true, after finding boundary edges, use the cells just inside.
                 nearest_cell=None,# choose cell closest to the given coord pair
                 edges_in_polygon = None): # edges with centers within the given shapely.Polygon 
        
        self.gforce = gforce
        self.edges = None
        self.cells = None

        # Record how the edges/cells were specified:
        self.spec = {'edges':edges,
                     'nodes':nodes,
                     'points':points,
                     'nearest':nearest,
                     'cells':cells,
                     'nearest_cell':nearest_cell,
                     'edges_in_polygon':edges_in_polygon}
                     
        # each datasource is a tuple of (bctype,datasource_id)
        self.datasources = []
        
        if edges is not None:
            # if type(edges) == str and edges == 'all':
            #     g = self.gforce.sun.grid()
            #     self.edges = arange(g.Nedges())
            # else:
            #     self.edges = edges
            self.edges = edges 
        elif nodes is not None:
            self.edges = self.find_edges(nodes_ccw)
        elif points is not None:
            g = self.gforce.sun.grid()
            nodes = [g.closest_point(p,boundary=1) for p in points]

            if len(nodes) > 2:
                raise Exception("For now, line segments must be 1 or 2 nodes only")

            if nodes[0] == nodes[1]:
                self.edges = self.find_nearest_edges(points[0])
            else:
                self.edges = self.find_edges(nodes)
        elif nearest is not None:
            self.edges = self.find_nearest_edges(nearest)
        elif cells is not None:
            self.cells = cells
        elif nearest_cell is not None:
            self.cells = self.find_nearest_cell(nearest_cell)
        elif edges_in_polygon is not None:
            self.edges = self.find_edges_in_polygon(edges_in_polygon)

        if boundary_cells:
            if self.cells is not None:
                raise Exception("boundary_cells specified, but cells have already been chosen")
            if self.edges is None:
                raise Exception("boundary_cells specified, but no edges have been chosen")
            if self.edges is 'all':
                raise Exception("can't use boundary_cells with edges='all'")
            g = self.gforce.sun.grid()
            edges = array(self.edges)
            self.cells = g.edges[edges,3]
            self.edges = None

        if (self.edges is None or len(self.edges) == 0) and (self.cells is None or len(self.cells) == 0):
            print("ForcingGroup(edges=%s,"%edges)
            print("             nodes=%s,"%nodes)
            print("             points=%s,"%points)
            print("             nearest=%s,"%nearest)
            print("             cells=%s)"%cells)
            print("FOUND NO EDGES")

    def Nedges(self):
        if self.edge_based():
            if self.edges is 'all':
                return self.gforce.sun.grid().Nedges()
            else:
                return len(self.edges)
        else:
            return 0
    def Ncells(self):
        if self.cell_based():
            if self.cells is 'all':
                return self.gforce.sun.grid().Ncells()
            else:
                return len(self.cells)
        else:
            return 0
        
    def edge_based(self):
        return (self.edges is not None)
    def cell_based(self):
        return (self.cells is not None)
    def nonempty(self):
        return (self.edges is not None and len(self.edges)>0) or \
               (self.cells is not None and len(self.cells)>0)
    
    def copy(self):
        fg = ForcingGroup(gforce = self.gforce,
                          edges = self.edges,
                          cells = self.cells)
        fg.datasources = deepcopy( self.datasources )
        
        fg.spec = {'copy':self}
        
        return fg

    def find_edges(self,nodes):
        # get the global grid:
        g = self.gforce.sun.grid()

        # print "Searching for shortest path along boundary between nodes %d %d"%(nodes[0],nodes[1])
        
        path = g.shortest_path(nodes[0],nodes[1],boundary_only=1)
        edges = []
        
        for i in range(len(path)-1):
            edges.append(g.find_edge( path[i:i+2] ))
        return array(edges)

    
    def find_nearest_edges(self,point):
        g = self.gforce.sun.grid()
        n = g.closest_point(point, boundary=1)

        possible_edges = g.pnt2edges(n)
        best_dist = inf
        best_edge = -1

        for e in possible_edges:
            if g.edges[e,2] == 0:
                # skip internal edges, but allow for edges that
                # are currently marked 1 (closed), b/c in the future
                # this code may be responsible for setting edge markers.
                continue

            if g.edges[e,0] == n:
                nbr = g.edges[e,1]
            else:
                nbr = g.edges[e,0]

            dist = norm( g.points[nbr,:2] - point )
            if dist < best_dist:
                best_dist = dist
                best_edge = e
        if best_edge < 0:
            raise Exception("Didn't find a good edge near %s"%point)

        return array( [best_edge] )

    def find_edges_in_polygon(self,edges_in_polygon):
        g = self.gforce.sun.grid()
        ec = g.edge_centers()

        if geom_prep is not None:
            poly = geom_prep(edges_in_polygon)
        else:
            poly = edges_in_polygon
            
        in_poly = []
        for j in range(g.Nedges()):
            if j % 100000 == 0:
                print("%d / %d edge centers checked"%(j,g.Nedges()))
            if poly.contains(geometry.Point(ec[j])):
                in_poly.append(j)
        return array( in_poly )

    def find_nearest_cell(self,nearest_cell):
        g = self.gforce.sun.grid()
        c = g.closest_cell( nearest_cell )
        return [c]
        
    def add_datasource(self,ds,bctype):
        """ ties the given datasource to this group, for the given boundary
        condition type ( 'FORCE_H','FORCE_WIND', etc.)
        """
        # gets the integer index
        # if type(ds) == tuple:
        #     myds = (bctype,
        #             self.gforce.add_datasource(ds[0]),
        #             self.gforce.add_datasource(ds[1]))
        # else:
        
        myds = (bctype,self.gforce.add_datasource(ds))
        self.datasources.append(myds)

        self.check_bctypes()

    def hydro_bctype(self):
        """ returns the bctype of just hydrodynamic forcing, if it is set
        """
        for tup in self.datasources:
            if tup[0] in hydro_forcings:
                return tup[0]
        return None

    def hydro_datasource(self):
        """ returns the datasource that specifies hydrodynamics on this
        group, or None if none exists
        """
        for tup in self.datasources:
            if tup[0] in hydro_forcings:
                print("NB: just returning the first datasource.  there could be a second one")
                return tup[1]
        return None

    def check_bctypes(self):
        # at most, one hydro forcing:
        n_hydro_forcing = 0

        for tup in self.datasources:
            if tup[0] in hydro_forcings:
                n_hydro_forcing += 1

        if n_hydro_forcing > 1:
            raise Exception("Looks like more than one type of hydrodynamic forcing")
        return True

    def write_config(self,fp,sun):
        fp.write("BEGIN_ITEMLIST\n")
        if self.edges is 'all':
            fp.write("  ITEM_TYPE ALL_EDGES\n")
            # for now, all_cells and all_edges only works for 2-D fields
            fp.write("  DIMENSIONS xy\n")
        elif self.edges is not None:
            fp.write("  ITEM_TYPE EDGE\n")
            fp.write("  ITEM_COUNT %d\n"%len(self.edges))
            fp.write("  ITEMS")
            for e in self.edges:
                fp.write(" %d"%e)
            fp.write("\n")
        elif self.cells is 'all':
            fp.write("  ITEM_TYPE ALL_CELLS\n")
            # see note above for ALL_EDGES
            fp.write("  DIMENSIONS xy\n")
        else:
            fp.write("  ITEM_TYPE CELL\n")

            fp.write("  ITEM_COUNT %d\n"%len(self.cells))
            fp.write("  ITEMS")
            for c in self.cells:
                fp.write(" %d"%c)
            fp.write("\n")

            
        fp.write("  BC_COUNT %d\n"%len(self.datasources))
        for tup in self.datasources:
            ds = self.gforce.datasources[tup[1]]
            fp.write("    BCTYPE %s\n"%tup[0])
            fp.write("    DATA %s\n"%ds.filename)

        fp.write("END_ITEMLIST\n")

    def map_to_grid(self,oldg,newg):
        """ requires that the point arrays are the same, and then translates edge or cell indices,
        making a copy of self
        """
        c = self.copy()
        
        if c.edges is not None and c.edges is not 'all':
            #print "Mapping edges"
            if 'edges' in self.spec and self.spec['edges'] is 'all':
                #print "Fast mapping of edges='all'"
                c.edges = arange(newg.Nedges())
            else:
                new_edges = []

                for e in c.edges:
                    try:
                        new_e = newg.find_edge( oldg.edges[e,:2])
                        new_edges.append(new_e)
                    except trigrid.NoSuchEdgeError:
                        pass
                c.edges = array(new_edges)

        if c.cells is not None and c.cells is not 'all':
            print("Mapping cells")
            if 'cells' in self.spec and self.spec['cells'] is 'all':
                #print "Fast mapping of cells='all'"
                c.cells = arange(newg.Ncells())
            else:
                new_cells = []

                for i in c.cells:
                    try:
                        new_c = newg.find_cell( oldg.cells[i])
                        print("Mapped global cell %d to local cell %d on grids %s,%s"%(i, new_c, oldg.processor,newg.processor))
                        new_cells.append(new_c)
                    except trigrid.NoSuchCellError:
                        pass
                c.cells = array(new_cells)
        #print "done with mapping"

        return c

class DataSource(object):
    n_components = 1
    def __init__(self,label):
        self.label = label
        
    def write_config(self,fp,sun):
        raise Exception("Missing!")
    
    def prepare(self,gforce):
        pass

    def plot(self,t,**kwargs):
        " t should be a vector of absdays "
        v = self.calc_values(t)
        plot(t,v,**kwargs)

    def calc_values(self,t):
        raise Exception("Missing!")

    def plot_overview(self,tmin,tmax):
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        
        axvline( date2num(tmin),c='k')
        axvline( date2num(tmax),c='k')
        grid()
        title(self.label)

        gca().xaxis_date()
        gcf().autofmt_xdate()
        

class Constant(DataSource):
    def __init__(self,label,value):
        DataSource.__init__(self,label)
        
        self.value = value
        
    def write_config(self,fp,sun):
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  CONSTANT\n")
        fp.write("  VALUE %g\n"%self.value)
        fp.write("END_DATA\n")

    def calc_values(self,t):
        return self.value * ones_like(t)

    def plot_overview(self,tmin,tmax):
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        plot( [date2num(tmin),date2num(tmax)],[self.value,self.value], 'r')
        annotate( "Constant: %f"%self.value,
                  [0.5*(date2num(tmin)+date2num(tmax)),self.value] )
        DataSource.plot_overview(self,tmin,tmax)


class Constant2Vector(DataSource):
    n_components = 2
    def __init__(self,label,value1,value2):
        DataSource.__init__(self,label)
        self.value = array([value1,value2])
        
    def write_config(self,fp,sun):
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  CONSTANT_2VEC\n")
        fp.write("  VALUE %g %g\n"%(self.value[0],self.value[1]))
        fp.write("END_DATA\n")

    def calc_values(self,t):
        return self.value * ones( (len(t),2) )

    def plot_overview(self,tmin,tmax):
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        quiver( [0.5*(date2num(tmin)+date2num(tmax))],[0],[self.value[0]],[self.value[1]] )

        annotate( "Constant 2-vector: (%f,%f)"%(self.value[0],self.value[1]),
                  [0.5*(date2num(tmin)+date2num(tmax)),0.0] )
        axis(ymin=-1,ymax=1)
        
        DataSource.plot_overview(self,tmin,tmax)
        
class Constant3Vector(DataSource):
    """ For wave forcing, the values are Hsig, thetamean, sigma_mean
    """
    n_components = 3
    def __init__(self,label,value1,value2,value3):
        DataSource.__init__(self,label)
        self.value = array([value1,value2,value3])
        
    def write_config(self,fp,sun):
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  CONSTANT_3VEC\n")
        fp.write("  VALUE %g %g %g\n"%(self.value[0],self.value[1],self.value[2]))
        fp.write("END_DATA\n")

    def calc_values(self,t):
        return self.value * ones( (len(t),3) )

    def plot_overview(self,tmin,tmax):
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays

        annotate( "Constant 3-vector: (%f,%f,%f)"%(self.value[0],self.value[1],self.value[2]),
                  [0.5*(date2num(tmin)+date2num(tmax)),0.0] )
        axis(ymin=-1,ymax=1)
        
        DataSource.plot_overview(self,tmin,tmax)


class Harmonics(DataSource):
    start_year = 2000 # really this should be set to sun.conf_int('start_year')

    def __init__(self,label,omegas=None,phases=None,amplitudes=None):
        DataSource.__init__(self,label)

        self.omegas = array(omegas)
        self.phases = array(phases)
        self.amplitudes = array(amplitudes)

    def prepare(self,gforce):
        self.start_year =  gforce.sun.conf_int('start_year')

    def calc_values(self,t):
        """ t should be a datenum, python.datetime / pylab style.
        self.omegas are in rad/sec
        self.phases are radians, relative t=0 at midnight, 1/1/<start year>
        """
        t = array(t)

        # convert to yeardays:
        t = t - date2num( datetime.datetime(self.start_year,1,1) )
        # and yearseconds
        t = 24*3600.*t
        
        if t.shape:
            v = (self.amplitudes * cos(self.omegas*t[...,newaxis]+self.phases)).sum(axis=1)
        else:
            v = (self.amplitudes * cos(self.omegas*t+self.phases)).sum()
            v = float(v)
            
        return v
    
    def write_config(self,fp,sun):
        if len(self.omegas) != len(self.phases) or len(self.phases) != len(self.amplitudes):
            raise Exception("Mismatch in number of harmonics")

        
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")

        fp.write("  HARMONICS\n")
        nc =  len(self.omegas)

        fp.write("  CONSTITUENTS_COUNT %d\n"%nc)
        fp.write("  OMEGAS")
        for omega in self.omegas:
            fp.write( " %7g"%omega)
        fp.write("\n")

        fp.write("  PHASES")
        for phase in self.phases:
            fp.write( " %7g"%phase)
        fp.write("\n")

        fp.write("  AMPLITUDES")
        for amp in self.amplitudes:
            fp.write( " %7g"%amp)
        fp.write("\n")
        
        fp.write("END_DATA\n")


class OtisHarmonics(Harmonics):
    """ Read the output of OTPS extract_HC and write suntans boundaries.c compatible
    harmonics.
    """ 
    def __init__(self,label,otis_output,h_offset=0.0):
        self.otis_output = otis_output

        omegas,phases,amplitudes = self.parse_otis(otis_output)
        self.h_offset = h_offset # this is taken care of during prepare()
        
        # print omegas,phases,amplitudes
        Harmonics.__init__(self,label=label,omegas=omegas,phases=phases,amplitudes=amplitudes)
        
    def parse_otis(self,fn):
        fp = open(fn,'rt')
        model = fp.readline()
        units = fp.readline()
        if units.strip() != 'Elevations (m)':
            print("Expected meters - units line was",units)
        headers = fp.readline().split()
        values = {}
        line = fp.readline() # first line of numbers
        for h,s in zip(headers,line.split()):
            values[h] = float(s)

        constituents = [s[:-4] for s in headers if s[-4:] == '_amp']
        amplitudes = array(  [values[c+'_amp'] for c in constituents] )
        phases = array( [values[c+'_ph'] for c in constituents] )
        phases *= pi/180 # convert to radians
        
        self.constituents = constituents

        omegas = []
        tide_db_indexes = []
        
        # find this constituent in the tide database, so we can get speed and
        # eventually equilibrium phase
        for constituent in constituents:
            # hopefully the constituent names are consistent - we just have to upcase
            # the otis names:
            i = tide_consts.const_names.index(constituent.upper())
            speed = tide_consts.speeds[i] # degrees per hour
            period = 1./(speed/360.)
            omega = speed/3600. * pi/180.
            omegas.append( omega ) # rad/s
            tide_db_indexes.append( i )
            # print constituent,i,speed,omega,period
        omegas = array(omegas)
        self.tide_db_indexes = array(tide_db_indexes)

        # since we have to correct these during prepare()
        self.original_phases = phases.copy()
        self.original_amps = amplitudes.copy()
        
        return omegas,phases,amplitudes
    
    def prepare(self,gforce):
        Harmonics.prepare(self,gforce)
        self.adjust_for_year(self.start_year)

        if self.h_offset != 0.0:
            self.omegas =     concatenate( [self.omegas,[0.0]])
            self.phases =     concatenate( [self.phases,[0.0]])
            self.amplitudes = concatenate( [self.amplitudes,[self.h_offset]] )
            
    def adjust_for_year(self,start_year):
        if start_year not in tide_consts.years:
            raise Exception('constants for prediction year are not available')

        year_i = tide_consts.years.searchsorted(start_year)

        # extract just the constituents OTIS provided, just for this year
        v0u=tide_consts.v0u[self.tide_db_indexes,year_i]
        lun_nod=tide_consts.lun_nodes[self.tide_db_indexes,year_i]

        # self.phases = (v0u*(pi/180)) - self.original_phases
        self.phases = (v0u*(pi/180)) - self.original_phases
        self.amplitudes = lun_nod*self.original_amps



class Timeseries(DataSource):
    def __init__(self,label,t0=None,dt=None,data=None,lag_s=None):
        DataSource.__init__(self,label)

        if t0 is not None and not isinstance(t0,datetime.datetime):
            raise Exception("t0 should be a datetime.  It was %s"%t0)
        
        self.t0 = t0
        self.dt = dt # should be in seconds
        self.data = data

        if lag_s is not None:
            self.lag_s = lag_s
        else:
            self.lag_s = 0.0

        # if we already have the pieces, go ahead and populate self.absdays
        if self.data is not None and self.t0 is not None and self.dt is not None:
            self.absdays = date2num(self.t0) + arange(len(self.data))*self.dt/(24*3600.)

    def calc_values(self,t):
        """ Evaluate this datasource at the given times t, an array of absdays
        """
        self.get_data()
        # so t is in absdays, self.t0 is offset in seconds between
        my_t = date2num(self.t0) + (self.dt/(24.*3600.)) *arange(len(self.data))

        v = interp(t, my_t, self.data, left=self.data[0], right = self.data[-1] )

        return v
        
    def write_config(self,fp,sun):
        self.get_data()
        
        # t0 in the file is seconds since the beginning of the year the simulation
        # started
        base_absdays = date2num( datetime.datetime( self.sun.time_zero().year, 1, 1) )

        t0_sun_seconds = (date2num(self.t0) - base_absdays)*24*3600
        
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  TIMESERIES\n")
        fp.write("  SAMPLE_COUNT %d\n"%len(self.data))
        fp.write("  DT %g\n"%self.dt)
        fp.write("  TZERO %g\n"%t0_sun_seconds)
        fp.write("  VALUES")
        for v in self.data:
            fp.write(" %g"%v)
        fp.write("\n")
        fp.write("END_DATA\n")

    def set_times(self,sun,Tabsdays):
        """ convenience routine: set t0 and dt from an array of absdays"""
        if len(Tabsdays)>0:
            self.t0 = num2date( Tabsdays[0] )
        
            if len(Tabsdays) > 1:
                t_first = sunreader.dt_round( num2date(Tabsdays[0]) )
                t_last  = sunreader.dt_round( num2date(Tabsdays[-1]) )
                tdelta = t_last - t_first
                total_seconds = tdelta.days*86400 + tdelta.seconds + tdelta.microseconds*1e-6
                dt_s = total_seconds / (len(Tabsdays)-1)
                
                # self.dt = 24*3600*median(diff(Tabsdays))
                self.dt = dt_s
            else:
                sim_start,sim_end = sun.simulation_period()
                # make it one timestep the length of the simulation
                self.dt = 24*3600*( date2num(sim_end) - date2num(sim_start))
            print("TimeSeries: Found t0 = ",date2num(self.t0))
            print("TimeSeries: Found dt = ",self.dt)
        else:
            raise Exception("No data at all found for this data source: %s"%str(self))
        
    def prepare(self,gforce):
        self.sun = gforce.sun

    def get_data(self):
        """ populate self.data, according to the period of the simulation.
        """
        if self.data is None:
            sim_start,sim_end = self.sun.simulation_period()

            ## This used to use the real t0 for sim_start - the beginning of the whole series
            #  of runs.  This is no good for long-term runs since every time we have to reconstruct
            #  the entire history.  Not sure why that was a good idea in the past...

            # but for a restart, sim_start will be the beginning of the
            # restart, so pull out the real t0 from sun
            # sim_start = datetime.datetime(self.sun.conf_int('start_year'),1,1) + datetime.timedelta(self.sun.conf_float('start_day'))

            # a negative lag means that at model time t, the forcing is from
            # real time t+delta, 
            lag_days = self.lag_s / (24*3600.0)
            sim_start += datetime.timedelta(-lag_days)
            sim_end   += datetime.timedelta(-lag_days)

            absdays, values = self.raw_data(sim_start,sim_end)

            # here is where we 'lie' about the timing of gage data
            self.absdays = absdays + lag_days

            # set the variables that Timeseries will need:
            self.set_times(self.sun,self.absdays)
            self.data = values

    def plot_overview(self,tmin,tmax):
        self.get_data()
        
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        if min(self.data) < 0 and max(self.data)>0: # go with linear
            plot(self.absdays,self.data,'r')
        else:
            if min(self.data) <  0 and max(self.data)<0:
                data = -self.data
                ylabel('negated')
            else:
                data = self.data

            # attempt to be clever about applying log scale:
            log_range = log10(data.max()) - log10(data.min())
            
            if log_range < 1.5:
                # go back to linear, with original sign
                ylabel('')
                plot(self.absdays,self.data,'r')
            else:
                if log_range > 3.5:
                    ymin = data.max() / 10**3.5
                    data = clip(data,ymin,inf)
                
                gca().set_yscale('log')
                plot(self.absdays,data,'r')
            
        DataSource.plot_overview(self,tmin,tmax)

class TimeseriesFunction(Timeseries):
    """ A timeseries, but the data is supplied as a callable function
    """
    def __init__(self,label,func,dt):
        Timeseries.__init__(self,label)
        
        self.func = func
        self.dt = dt # seconds! used to be interpreted as days

    def raw_data(self,sim_start,sim_end):
        dt_days = self.dt / 86400.0
        absdays = arange(date2num(sim_start),date2num(sim_end)+dt_days,dt_days)

        data = self.func(absdays)

        return absdays, data

    def plot_overview(self,tmin,tmax):
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        annotate( "TimeseriesFunction", [self.absdays[0],self.data[0]])
        Timeseries.plot_overview(self,tmin,tmax)


class TimeseriesVector(Timeseries):    
    def write_config(self,fp,sun):
        self.get_data()

        base_absdays = date2num( datetime.datetime( sun.time_zero().year,1,1 ) )
        t0_sun_seconds = (date2num(self.t0) - base_absdays) *24*3600
        
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  TIMESERIES_%dVEC\n"%self.n_components)
        fp.write("  SAMPLE_COUNT %d\n"%len(self.data))
        fp.write("  DT %g\n"%self.dt)
        fp.write("  TZERO %g\n"%t0_sun_seconds)
        fp.write("  VALUES")
        for v in self.data:
            for i in range(self.n_components):
                fp.write(" %g"%v[i] )
        fp.write("\n")
        fp.write("END_DATA\n")
        

class Timeseries2Vector(TimeseriesVector):
    n_components = 2
    
    def plot_overview(self,tmin,tmax):
        self.get_data()
        
        # plot something representative into a single axes,
        # for the given simulation period, as specified by absdays
        quiver(self.absdays,0*self.absdays,
               self.data[:,0],self.data[:,1])
        axis(ymin=-1,ymax=1)
        DataSource.plot_overview(self,tmin,tmax)

    def plot_components(self,tmin,tmax):
        #self.get_data()
        
        plot(self.absdays,self.data[:0],'r')
        plot(self.absdays,self.data[:0],'b')
        #DataSource.plot_overview(self,tmin,tmax)

class Timeseries3Vector(TimeseriesVector):
    n_components = 3

    def plot_overview(self,tmin,tmax):
        self.get_data()
        
        plot(self.absdays,self.data[:,0],'r')
        plot(self.absdays,self.data[:,1],'g')
        plot(self.absdays,self.data[:,2],'b')
        DataSource.plot_overview(self,tmin,tmax)

import gage_data, opendap

class NoaaHarmonics(Harmonics):
    """ Specify harmonic constituents fetched from NOAA

    usage: NoaaHarmonics('Point Reyes, CA',2006)

    n_consts: if this is a number, take the first n_consts constituents
              if a list, it is the names of the constituents desired
    """
    def __init__(self,station_name,n_consts=None,
                 amplification=1.0,
                 raise_h=0.0,
                 lag_s=0.0,
                 include_bathy_offset=1):

        self.station_name = station_name
        
        self.gage = gage_data.gage(self.station_name)
        self.n_consts = n_consts

        self.amplification = 1.0
        self.raise_h = raise_h
        self.lag_s = lag_s
        self.include_bathy_offset = include_bathy_offset

        self.omegas = None
        self.amplitudes = None
        self.phases = None

        Harmonics.__init__(self,"%s harmonics"%station_name)
            
    def prepare(self,gforce):
        self.sun = gforce.sun

    def get_data(self):
        if self.omegas is not None:
            # assume that we've already loaded stuff
            return
        
        # Grab the constituents:
        start_year = self.sun.conf_int('start_year')
        self.start_year = start_year
        
        [self.omegas,
         self.amplitudes,
         self.phases] = opendap.convert_noaa_to_otis_xtides(int(self.gage.external_id),
                                                            start_year)

        self.amplitudes *= self.amplification
   
        if self.lag_s != 0.0:
            print("NoaaHarmonics: lag is not yet supported")
            
        if self.n_consts is not None:
            self.choose_constituents(self.n_consts)

        offset = self.raise_h
        if self.include_bathy_offset:
            offset -= sunreader.read_bathymetry_offset()

        # The sign here has gone back and forth, but Kevin says that it worked
        # for him recently (5/2011) as offset += msl_to_navd88.  
        offset += self.gage.msl_to_navd88()

        if offset != 0.0:
            # add a constant term to the harmonic decomposition
            self.omegas      = concatenate( (self.omegas,[0.0] ) )
            self.amplitudes  = concatenate( (self.amplitudes,[offset]) )
            self.phases      = concatenate( (self.phases,[0.0] ) )


    def choose_constituents(self,n_consts):
        if type(n_consts) == int:
            self.omegas = self.omegas[:n_consts]
            self.amplitudes = self.amplitudes[:n_consts]
            self.phases = self.phases[:n_consts]
        elif type(n_consts) == list:
            noaa_names, consts = opendap.get_prediction_consts(int(self.gage.external_id))

            idxs = array([noaa_names.index(const_name) for const_name in n_consts])
            self.omegas = self.omegas[idxs]
            self.amplitudes   = self.amplitudes[idxs]
            self.phases = self.phases[idxs]

            # sanity checking, print out the periods ~[h] of the selected
            # constituents
            hours = 1.0 / (self.omegas * 3600 / (2*pi))
            print("Selected tidal periods [h]:",hours)
        else:
            raise "Bad n_consts: %s "%n_consts



class NoaaGage(Timeseries):
    """ Get NOAA gage data and create a timeseries Datasource with it
    amplification will multiply by the given factor, centered around the
    mean of the highest and lowest tides
    raise_h adds the given amount to the freesurface, and lag_s introduces
    a lag specified in seconds (i.e. if the model lags reality, specify a
    negative lag here so that the forcing is shifted back in time)

    include_bathy_offset: automatically incorporate the bathymetry offset, as queried
      from the sunreader instance
    """
    
    max_missing_samples = 5
    def __init__(self,station_name,
                 amplification=1.0,
                 raise_h=0.0,
                 lag_s=0.0,
                 include_bathy_offset=1):
        Timeseries.__init__(self,label="%s observed tidal height"%station_name,
                            lag_s=lag_s)

        self.station_name = station_name
        self.amplification = amplification
        self.raise_h = raise_h
        self.include_bathy_offset = include_bathy_offset

        self.gage = gage_data.gage(station_name)

    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for the real start/end dates
        given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering

        should return a vector of AbsDays
        """
        print("Fetch data for gage: ",self.gage)
        print("    start: ",sim_start)
        print("    stop:  ",sim_end)
        
        vals = self.gage.data(sim_start,sim_end,'h')
        vals = timeseries.fill_holes(vals,max_missing_samples=self.max_missing_samples)
        absdays = vals[:,0]
        values   = vals[:,1]
        
        # remember some of the intermediate values just in case we need to debug
        self.raw_h = values

        # filter:
        #self.h = tidal_filter.filter_tidal_data(self.raw_h, absdays*24*3600)
        self.h = lp_filter.lowpass(self.raw_h,
                                   absdays*24, # Time in hours
                                   cutoff=3.0) # cutoff of 3 hours

        if self.include_bathy_offset:
            self.h -= sunreader.read_bathymetry_offset()

        # add amplification (note that this is pretty much wrong, since we really
        # ought to amplify it around MSL, which we probably aren't coming anywhere
        # close to computing.  At least averaging the max/min aren't as vulnerable
        # to the period of the gage data
        mean_h = 0.5*(self.h.max() + self.h.min())
        self.h = mean_h + (self.h - mean_h)*self.amplification + self.raise_h

        return absdays, self.h


class CompositeNoaaGage(NoaaGage):
    lp_hours = 35.0
    
    def __init__(self,*args,**kwargs):
        self.backup_station_name = kwargs.pop('backup_station_name',None)
        NoaaGage.__init__(self,*args,**kwargs)

    def fill_in_missing_data(self,sim_start,sim_end,data):
        # Find the holes that are bigger than our limit for filling in just by interpolation
        # Same logic as in timeseries.py
        
        # First, get everything onto a common time line, with nans for the missing spots
        basic_dt = median( diff(data[:,0]) )

        t0 = data[0,0]
        tN = data[-1,0]

        if date2num(sim_start) < t0:
            # push back the start time, but as a whole number of timesteps
            nsteps = int(ceil( (t0 - date2num(sim_start))/basic_dt))
            t0 = t0 - nsteps*basic_dt
        if date2num(sim_end) > tN:
            # push ahead the end time, as a whole number of timesteps
            nsteps = int(ceil( (date2num(sim_end) - tN)/basic_dt))
            tN = tN + nsteps*basic_dt

        nsteps = int(ceil( (tN-t0)/basic_dt))
        new_data = zeros( (nsteps,2), float64 )
        new_data[:,0] = linspace(t0,tN,nsteps)

        new_data[:,1] = nan
        # populate the original data, just choosing the nearest timestep (thus the 0.5*dt)
        new_data[searchsorted( new_data[:,0]+0.5*basic_dt, data[:,0] ),1] = data[:,1]

        # Small missing chunks we just interpolate over:
        missing = isnan(new_data[:,1])
        idx = arange(len(missing))

        # find the indices into new_data for each place a gap starts or ends
        # this gives the index right before the transition
        gap_bounds, = nonzero( (missing[:-1] != missing[1:]) )

        if missing[0]:
            gap_bounds = concatenate( ([-1],gap_bounds) )
        if missing[-1]:
            gap_bounds = concatenate( (gap_bounds, [missing.shape[0]-1] ) )

        if len(gap_bounds) %2 != 0:
            raise Exception("How can there be an odd number of gap bounds?")

        gap_starts = gap_bounds[0::2]+1  # index of first missing
        gap_ends   = gap_bounds[1::2]+1  # index after last missing
        
        # Fill in short gaps with interpolation:
        i_to_interp = []
        f = interp1d(data[:,0],data[:,1])
        
        for i in range(len(gap_starts)):
            # For gaps that have valid data on both sides, and aren't too long, we interpolate
            if gap_starts[i] != 0 and \
               gap_ends[i] != data.shape[0] and \
               (gap_ends[i] - gap_starts[i] <= self.max_missing_samples):
                data[gap_starts[i]:gap_ends[i],1] = f( data[gap_starts[i]:gap_ends[i],0] )
            else:
                i_to_interp.append(i)

        if len(i_to_interp) == 0:
            return data

        print("Will have to go to the backup datasource")
        
        gap_starts = gap_starts[i_to_interp]
        gap_ends   = gap_ends[i_to_interp]

        mask = zeros( (new_data.shape[0]), bool8 )
        mask[:] = False
        for i in range(len(gap_starts)):
            mask[gap_starts[i]:gap_ends[i]] = True
            
        backup_t = new_data[mask,0]
        
        backup_t0 = num2date(backup_t[0])
        backup_tN = num2date(backup_t[-1])

        # First, we get harmonic predictions for this period:
        
        my_harmonic_vals = self.gage.data( backup_t0, backup_tN, 'p' )
        f = interp1d( my_harmonic_vals[:,0], my_harmonic_vals[:,1] )

        new_data[mask,1] = f( new_data[mask,0] )

        # Second, come back and add some subtidal fluctuations
        if self.backup_station_name is not None:
            print("Getting subtidal from %s"%self.backup_station_name)
            backup_gage = gage_data.gage(self.backup_station_name)
            # harmonics from that station:
            backup_harmonics = backup_gage.data( backup_t0,backup_tN,'p')

            # observed tides:
            backup_observations  = backup_gage.data( backup_t0,backup_tN,'h')
            f = interp1d( backup_observations[:,0], backup_observations[:,1],bounds_error=False,fill_value=nan)

            backup_subtidal = f(backup_harmonics[:,0]) - backup_harmonics[:,1]

            f = interp1d(backup_harmonics[:,0],backup_subtidal)
            
            new_data[mask,1] += f( new_data[mask,0] )

        return new_data


    def raw_data(self,sim_start,sim_end):
        print("Fetch data for gage: ",self.gage)
        print("    start: ",sim_start)
        print("    stop:  ",sim_end)
        
        vals = self.gage.data(sim_start,sim_end,'h')

        self.primary_vals = vals
        # Intervene, and check for possibly missing data, take care of fill_holes stuff
        vals = self.fill_in_missing_data(sim_start,sim_end,vals)

        # And back to the usual:
        absdays = vals[:,0]
        values   = vals[:,1]
        
        # remember some of the intermediate values just in case we need to debug
        self.raw_h = values

        # filter:
        #self.h = tidal_filter.filter_tidal_data(self.raw_h, absdays*24*3600)
        self.h = lp_filter.lowpass(self.raw_h, absdays*24, cutoff=3.0)        

        if self.include_bathy_offset:
            self.h -= sunreader.read_bathymetry_offset()

        # add amplification (note that this is pretty much wrong, since we really
        # ought to amplify it around MSL, which we probably aren't coming anywhere
        # close to computing.  At least averaging the max/min aren't as vulnerable
        # to the period of the gage data
        mean_h = 0.5*(self.h.max() + self.h.min())
        self.h = mean_h + (self.h - mean_h)*self.amplification + self.raise_h

        return absdays, self.h



class MergeTidalTimeseriesFilter(Timeseries):
    """
    taken largely from CompositeNoaaGage, but updated to work with the DatabaseGage
    classes.

    represents a time series by combining two gages.  When the primary gage has
    data, all is well.  gaps smaller than linear_gap_days are filled by linear
    interpolation just using the primary gage's data.

    gaps larger than that will be queried from the secondary gage.  the secondary
    timeseries will be adjusted to match the primary at the start and end of the
    gap.
    """
    max_missing_samples = 5
    lp_hours = 35.0

    # as much as possible pull the timestep from the primary or secondary gages,
    # but if there are no hints, use this value:
    default_dt_days = 360./86400.
    
    def __init__(self,label,primary_gage,secondary_gage,offset=0.0):
        self.pri = primary_gage
        self.sec = secondary_gage
        self.offset = offset
        
        super(MergeTidalTimeseriesFilter,self).__init__(label)

    def raw_data(self,sim_start,sim_end):
        # use the query interface for the primary/secondary gages, to avoid
        # any interpolation happening too soon
        sim_start = date2num(sim_start)
        sim_end = date2num(sim_end)

        vals = self.pri.query(sim_start, sim_end,
                              interpolate_gaps = True, # just gets enough to data to allow for interpolation,
                              extrapolate = False,  # not sure about that one
                              autopopulate = True)
        
        # Intervene, and check for possibly missing data, take care of fill_holes stuff
        self.vals = vals = self.fill_in_missing_data(sim_start,sim_end,vals)
        vals[:,1] += self.offset
        
        # And back to the usual:
        return vals[:,0],vals[:,1]
        
    def fill_in_missing_data(self,sim_start,sim_end,data):
        # Find the holes that are bigger than our limit for filling in just by interpolation
        # Same logic as in timeseries.py
        
        # First, get everything onto a common time line, with nans for the missing spots
        # if there isn't any good data, this will be nan:
        basic_dt = median( diff(data[:,0]) )
        if isnan(basic_dt):
            try:
                basic_dt = self.pri.parms['dt_s'] / 86400.
            except:
                pass
        if isnan(basic_dt):
            try:
                basic_dt = self.sec.parms['dt_s'] / 86400.
            except:
                pass
        if isnan(basic_dt):
            print("Having to fall back to default dt for tide datasource")
            basic_dt = self.default_dt_days

        t0 = data[0,0]
        tN = data[-1,0]

        ## The idea here is to make sure that t0<= sim_start, tN >= sim_end,
        #  that the dt in the new data is the same as the basic_dt of the old
        #  data, and t0 falls on an integral time step of the new data.

        print("t0:",num2date(t0))
        print("tN:",num2date(tN))

        # is 0 if t0 is early enough, or some negative number if we need
        # more steps before data starts.
        step_start = min(0, int( -ceil( (data[0,0]-sim_start)/basic_dt)))

        # keep data through at least what we have, and extend to the simulation period if
        # that's longer
        tEnd = max( data[-1,0],sim_end)
        
        # count steps based solely on duration and timestep, not how many steps came in
        # (since they might not be evenyl spaced, missing, etc.)
        # this is _inclusive_
        step_end = int( ceil( (tEnd - data[0,0])/basic_dt))
            
        nsteps = step_end + 1 - step_start # 
        new_data = zeros( (nsteps,2), float64 )
        new_data[:,0] = arange(step_start,step_end+1) * basic_dt + data[0,0]

        new_data[:,1] = nan
        # place the original data in reasonable spots in the old data,
        # just choosing the nearest new-data timestep (thus 0.5*dt)
        new_data[searchsorted( new_data[:,0]+0.5*basic_dt, data[:,0] ),1] = data[:,1]

        # Small missing chunks we just interpolate over:
        missing = isnan(new_data[:,1])
        idx = arange(len(missing))

        # find the indices into new_data for each place a gap starts or ends
        # this gives the index right before the transition
        gap_bounds, = nonzero( (missing[:-1] != missing[1:]) )

        if missing[0]:
            gap_bounds = concatenate( ([-1],gap_bounds) )
        if missing[-1]:
            gap_bounds = concatenate( (gap_bounds, [missing.shape[0]-1] ) )

        if len(gap_bounds) %2 != 0:
            raise Exception("How can there be an odd number of gap bounds?")

        gap_starts = gap_bounds[0::2]+1  # index of first missing
        gap_ends   = gap_bounds[1::2]+1  # index after last missing
        
        # Fill in short gaps with interpolation:
        i_to_interp = []
        f = None
        if len(data) > 1: # this fails when there isn't enough data
            valid = ~isnan(data[:,1])
            if sum(valid) > 1:
                f = interp1d(data[valid,0],data[valid,1])
                f_tmin = data[valid,0].min()
                f_tmax = data[valid,0].max()
            # otherwise, can't do basic linear interpolation - will fall back to
            # the backup datasource below:
        
        for i in range(len(gap_starts)):
            # For gaps that have valid data on both sides, and aren't too long, we interpolate
            gap_t_start = new_data[gap_starts[i],0] # time of first missing sample
            gap_t_end   = new_data[gap_ends[i]-1,0] # time of last missing sample
            if f and \
               gap_t_start > f_tmin and \
               gap_t_end < f_tmax and \
               (gap_ends[i] - gap_starts[i] <= self.max_missing_samples):
                new_data[gap_starts[i]:gap_ends[i],1] = f( new_data[gap_starts[i]:gap_ends[i],0] )
            else:
                # fill in by going to backup datasource
                # to match the DC & linear component, query for some extra data
                # choose an index into the real data before the period to fill in
                left_w_overlap = max(gap_starts[i]-self.max_missing_samples,0)
                # and an index into the real data after the period to fill
                right_w_overlap = min(gap_ends[i]+self.max_missing_samples,len(new_data))
                # pull the whole chunk
                win_with_overlap = new_data[left_w_overlap:right_w_overlap]

                t_start = new_data[left_w_overlap,0] 
                t_end = new_data[right_w_overlap-1,0]

                # NB: the slightest bit of round off and backup_vals will not entirely
                # cover the range of dates - so request and extra basic_dt on each end
                backup_vals = self.sec.query( t_start-basic_dt, t_end+basic_dt, autopopulate=True,interpolate_gaps=True )

                fbackup = interp1d( backup_vals[:,0], backup_vals[:,1] )

                t_fill = new_data[gap_starts[i]:gap_ends[i],0]

                # and narrow that down to valid data in the overlapping region
                overlap_data = win_with_overlap[ ~isnan(win_with_overlap[:,1]) ]


                # if we only have data on oneside of the gap, fit only the DC component -
                # otherwise fit a line
                if len(overlap_data) > 0:
                    if all(overlap_data[:,0]<= new_data[gap_starts[i],0]) or \
                       all(overlap_data[:,0]>= new_data[gap_ends[i],0]):
                        degree = 0
                    else:
                        degree = 1
                    error = overlap_data[:,1] - fbackup( overlap_data[:,0] )
                    fit = polyfit( overlap_data[:,0],
                                   error,
                                   degree )
                    correction = polyval(fit,t_fill)
                else:
                    correction = 0.0

                new_data[gap_starts[i]:gap_ends[i],1] = fbackup( t_fill ) + correction

        return new_data



                          
def read_cimis_csv(filename,col_names):
    """ returns an array of data, with the requested columns plus
    an absdays column.
    """
    if filename is None:
        filename = os.path.join( os.environ['HOME'], 'classes/research/suntans/forcing/wind/cimis/union_city_171_2005_2010.csv')

    fp = open(filename,'rt')
    headers = [s.strip() for s in fp.readline().split(",")]
    # date/time always pulled:
    date_col = headers.index('Date')
    hour_col = headers.index('Hour')

    col_indexes = [headers.index(col_name) for col_name in col_names]

    records = []
    
    for line in fp:
        cols = line.split(",")
        d = datetime.datetime.strptime(cols[date_col].strip(),"%m/%d/%Y")
        hour_minute = cols[hour_col].strip()
        h = int(hour_minute[:2])
        m = int(hour_minute[2:])

        record = [date2num(d) + (h+m/60.)/24.]

        for col_index in col_indexes:
            try:
                val = float( cols[col_index] )
            except ValueError:
                val = nan
            record.append(val)
        
        records.append(record)
    fp.close()

    return array(records)

    
def read_Japanese_met(filename,col_names):
    """ returns an array of data, with the requested columns plus
    an absdays column. The date is given in julian day of year starting
    at 0 on Jan. 1. 
    A single row is manually added to each wind file to specify the
    year of the data.
    """

    if filename is None:
        print("No wind filename given")
        exit(1)

    # hardwired column definitions
    column_list = ["Day","Precip (mm)","Air Temp (deg C)","Wind Speed (m/s)",
                   "Wind Dir (0-360)"]

    col_indexes = [column_list.index(col_name) for col_name in col_names]

    fp = open(filename,'rt')
    year_string = fp.readline().split()
    year = int(year_string[0])
    # date/time always pulled:

    # get time of Jan 1 of input year at 0:00
    d = datetime.datetime.strptime(year_string[0] + "-01-01 00:00","%Y-%m-%d %H:%M")
    Jan1_of_year = date2num(d)

    records = []
    
    for line in fp:
        cols = line.split()
        record = []
        record.append( Jan1_of_year + float(cols[0]))

        for col_index in col_indexes:
            try:
                val = float( cols[col_index] )
            except ValueError:
                val = nan
            record.append(val)
        
        records.append(record)
    fp.close()

    return array(records)


class CimisEvapPrecip(Timeseries):
    """ References a CSV file from CIMIS to get hourly precipitation, and
    a climatology of evaporation, creating a timeseries where evaporation is
    positive, and precipitation is negative, in m/s
    """
    def __init__(self,name,csv_file=None,lag_s=0.0):
        """ Load CIMIS precip data and merge with evap. climatology.  if csv_file is
        not specified, it defaults to a 2005-2010 dataset.
        """
        self.name = name
        self.csv_file = csv_file
        
        Timeseries.__init__(self,label="%s CIMIS evap-precip"%name,
                            lag_s=lag_s)
        
    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for C:\Program Files\GnuWin32the real start/end dates
        given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering
        """
        data = read_cimis_csv(self.csv_file,['Precip (mm)'])
        
        # nan's get replaced by 0
        valid = isfinite(data[:,1])
        
        data[~valid,1] = 0.0

        # convert to a rate in m/s
        data[0,1] = 0.0
        #            mm/period   -> m       dt [d]      h    seconds
        data[1:,1] = data[1:,1] * 1e-3 / ( diff(data[:,0])*24*3600 )

        # just to make sure that we have evenly spaced data
        vals = timeseries.fill_holes(data)

        # trim down to the period in question:
        valid = (vals[:,0] > date2num(sim_start) -1) & (vals[:,0] < date2num(sim_end) + 1)
        vals = vals[valid,:]
        
        # and add in evaporation:
        evap = SFBayMeanEvaporation()
        # combine, with evaporation being positive, precip negative
        vals[:,1] = evap.interp_to_absdays(vals[:,0]) - vals[:,1]
        
        return vals[:,0], vals[:,1]
        

class JapaneseEvapPrecip(Timeseries):
    """ References Japanese met file to get precipitation
        Calculated precipitation is negative, in m/s
        Hardwired evaporation to zero.
    """
    def __init__(self,name,tab_delim_file=None,lag_s=0.0):

        self.name = name
        self.tab_delim_file = tab_delim_file
        
        Timeseries.__init__(self,label="%s Japanese precip"%name,
                            lag_s=lag_s)
        
    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for C:\Program Files\GnuWin32the real start/end dates
        given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering
        """
        data = read_Japanese_met(self.tab_delim_file,['Precip (mm)'])
        
        # nan's get replaced by 0
        valid = isfinite(data[:,1])
        
        data[~valid,1] = 0.0

        # convert to a rate in m/s
        data[0,1] = 0.0
        #            mm/period   -> m       dt [d]      h    seconds
        data[1:,1] = data[1:,1] * 1e-3 / ( diff(data[:,0])*24*3600 )

        # just to make sure that we have evenly spaced data
        vals = timeseries.fill_holes(data)

        # trim down to the period in question:
        valid = (vals[:,0] > date2num(sim_start) -1) & (vals[:,0] < date2num(sim_end) + 1)
        vals = vals[valid,:]
        
        # evaporation is hardwired to zero, subtract precip from zero evap to get net evap:
        vals[:,1] = -vals[:,1]
        
        return vals[:,0], vals[:,1]
        
    
    
class CimisWind(Timeseries2Vector):
    """ References a CSV file downloaded from wwwcimis.water.ca.gov
    Choose 'CSV with Headers' when saving the file, and save to metric units.
    """
    def __init__(self,name,csv_file=None,lag_s=0.0):
        """ Read wind from a single station CIMIS csv file.  If csv_file is not
        given, defaults to a Union City, 2005-2010 dataset.
        """
        self.name = name
        self.csv_file = csv_file
        
        Timeseries2Vector.__init__(self,label="%s CIMIS wind"%name,
                            lag_s=lag_s)
        
    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for the real start/end dates
        given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering
        """
        data = read_cimis_csv(self.csv_file,['Wind Speed (m/s)',
                                             'Wind Dir (0-360)'])
        
        # remove nan values - they'll be interpolated in
        valid = isfinite(data[:,1])
        
        data = data[valid,:]

        # 
        wind_dir = (180 + 90-data[:,2]) * pi / 180. # radians CCW from +x, velocity vector
        wind_spd = data[:,1].copy()
        
        data[:,1] = cos(wind_dir) * wind_spd
        data[:,2] = sin(wind_dir) * wind_spd
        
        vals = timeseries.fill_holes(data)

        # trim down to the period in question:
        valid = (vals[:,0] > date2num(sim_start) -1) & (vals[:,0] < date2num(sim_end) + 1)
        vals = vals[valid,:]
        
        return vals[:,0], vals[:,1:]
        
    
class JapaneseWind(Timeseries2Vector):
    """ References a tab delimited file 
    units are m/s and degrees.
    """
    def __init__(self,name,tab_delim_file=None,lag_s=0.0):
        """ Read wind from a single station Japanese weather file.  
        """
        self.name = name
        self.tab_delim_file = tab_delim_file
        
        Timeseries2Vector.__init__(self,label="%s Japanese wind"%name,
                            lag_s=lag_s)
        
    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for the real 
	start/end dates given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering
        """
        data = read_Japanese_met(self.tab_delim_file,['Wind Speed (m/s)',
                                                      'Wind Dir (0-360)'])
        
        # remove nan values - they'll be interpolated in
        valid = isfinite(data[:,1])
        
        # 
        wind_dir = (180 + 90-data[:,2]) * pi / 180. # radians CCW from +x, velocity vector
        wind_spd = data[:,1].copy()
        
        data[:,1] = cos(wind_dir) * wind_spd
        data[:,2] = sin(wind_dir) * wind_spd
        
        vals = timeseries.fill_holes(data)

        # trim down to the period in question:
        valid = (vals[:,0] > date2num(sim_start) -1) & (vals[:,0] < date2num(sim_end) + 1)
        vals = vals[valid,:]
        
        return vals[:,0], vals[:,1:]

# class NDBCWind(Timeseries2Vector):
#     """ Retrieve wind timeseries from National Data Buoy Center
#     buoys.  At least for 46026, just beyond the sand bar outside Golden
#     Gate, this is hourly, near-realtime data, measured at 5m above sea surface.
#     """
     
    
class NoaaWind(Timeseries2Vector):
    """ Get NOAA wind data and create a timeseries Datasource with it
    lag_s introduces
    a lag specified in seconds (i.e. if the model lags reality, specify a
    negative lag here so that the forcing is shifted back in time)
    """
    def __init__(self,station_name,lag_s=0.0,lowpass_hours=0):
        Timeseries2Vector.__init__(self,label="%s vector wind"%station_name,
                            lag_s=lag_s)

        self.station_name = station_name
        self.gage = gage_data.gage(station_name)
        self.lowpass_hours = lowpass_hours

    def raw_data(self,sim_start,sim_end):
        """ return the array of times and array of values for the real start/end dates
        given.
        code in here should *not* perform the lagging
        this *is* the right place to perform amplification or shifts in value,
        or tidal filtering
        """
        if self.lowpass_hours > 0:
            td = datetime.timedelta(2* self.lowpass_hours / 24.0)
            sim_start = sim_start - td
            sim_end   = sim_end   + td
            
        vals = self.gage.data(sim_start,sim_end,'w')
        vals = timeseries.fill_holes(vals)
        
        absdays   = vals[:,0]
        uv_values   = vals[:,1:]

        # hopefully 4th order is okay -
        # typical cutoff will probably be ~1h or less, and typical data sampling rate
        # will be 10 per hour.
        if len(absdays) < 12:
            print("NOAA Wind Data for station %s, period %s - %s is no good"%(self.station_name,
                                                                              sim_start,sim_end))
            print(absdays)
            print(uv_values)
            print("Last request key: ",self.gage.last_request_key)
            raise Exception("Missing forcing data")
        if self.lowpass_hours > 0:
            uv_values[:,0] = lp_filter.lowpass(uv_values[:,0],absdays, self.lowpass_hours / 24.0, order=4 )
            uv_values[:,1] = lp_filter.lowpass(uv_values[:,1],absdays, self.lowpass_hours / 24.0, order=4 )
        
        return absdays, uv_values

class SFBayMeanEvaporation(Timeseries):
    """ Kludge for estimating evaporation rates based at Oakland airport.

    Eventually this will need to be calculated using some form of the Penman
    equation.

    These data are in several places on the web, notably
    http://www.calclim.dri.edu/ccda/comparative/avgpan.html

    while the units are not given, I'm pretty sure it should be inches/month.
    
    """

    # the Burlingame data (OAK and SFO data are calculated, this is measured)
    orig_data = array( [1.27, 1.81, 3.60, 5.28, 6.85,  7.82,  8.42,  7.39,  5.74,  3.78,  1.98, 1.28] )
    
    # Burlingame:            [1.27  1.81  3.60  5.28  6.85  7.82  8.42  7.39  5.74  3.78  1.98  1.28]
    # Oakland AP:            [1.8   2.3   3.8   4.8   5.7   6.4   6.4   6.0   5.4   4.0   2.4   1.8 ]
    # SFO:                   [1.7   2.4   3.8   5.3   6.4   7.1   6.7   6.6   5.9   4.4   2.4   1.7 ]
    # Grizzly Island:        [1.45  2.25  4.00  5.72  8.07  9.82 10.69  8.93  6.88  4.33  2.10  1.55]
    # Mandeville Isl, Delta: [1.10  2.38  4.77  6.95  8.55 10.44 11.22  9.71  7.41  5.12  2.47  1.13 ]
    # Panoche Cr, San Jose:  [1.74  2.86  5.72  7.50 11.83 13.58 15.04 14.29 10.45  7.61  2.72  1.81 ]
    # Tracy Pumps:           [1.54  2.48  5.31  8.16 12.00 14.88 16.92 14.52 10.62  6.59  2.95  1.47]
        
    def __init__(self,lag_s=0.0):
        Timeseries.__init__(self,label="SF Bay evaporation, monthly climatology",
                            lag_s=lag_s)

    def raw_data(self,sim_start,sim_end):
        # Take the original data at the middle of each month, so here we need
        # to synthesize the ides-timeseries between sim_start and sim_end

        # remember months are counted 1-based
        start_month = sim_start.year * 12 + (sim_start.month - 1)
        end_month   = sim_end.year*12 + (sim_end.month - 1)

        # a bit of padding to make sure we have a measurement before the beginning
        # and after the end.
        all_months = arange( start_month - 1, end_month + 2 )

        mapping = all_months % 12 # indexes the monthly average values - NOT IDES!

        # the Timeseries code wants a constant time-step -
        start_absday = date2num(datetime.datetime( all_months[0] // 12, (all_months[0] % 12) + 1, 15 ) )
        end_absday   = date2num(datetime.datetime( all_months[-1] // 12, (all_months[-1] % 12) + 1, 15 ) )

        absdays = linspace(start_absday,end_absday,len(all_months))

        # Just inch/month -> meters/second
        evap_mps = self.orig_data[mapping] * 9.6587355e-09

        return absdays,evap_mps

    def interp_to_absdays(self,absdays):
        """ Return evaporation rates, sampled onto the given absdays values.
        This is used by precipitation datasources which combine with evaporation
        """
        sim_start = num2date(absdays[0])
        sim_end   = num2date(absdays[-1])

        my_absdays,my_data = self.raw_data(sim_start,sim_end)
        f = interp1d(my_absdays,my_data)

        return f(absdays)
        
        

class NDOI(Timeseries):
    """ Flow data for net delta output index.  This can probably be ignored, and use the FlowCsvMgd class instead.
    """
    ndoi_fn = os.path.join(forcing_dir,"flows/ndoi-1994-2009.txt")
    dayflow_fn = os.path.join(forcing_dir,"flows/dayflow.csv")

    def __init__(self,
                 amplification=1.0,
                 lag_s=0.0):
        Timeseries.__init__(self,label="NDOI",lag_s=lag_s)
        
        self.amplification = amplification

    def raw_data(self,sim_start,sim_end):
        # NDOI from IEP appears to no longer be supported.  Switching to data from dayflow, which
        # should be about the same, although there is some small discrepancy
        if 0:
            ts = timeseries.IEPFile(self.ndoi_fn)
            # important to clip because the timeseries determines the starting point
            # for year days by looking at the year of the first entry in the timeseries.
        else:
            d = loadtxt(self.dayflow_fn,
                        skiprows=1,delimiter=',',
                        converters={0: lambda s: date2num(datetime.datetime.strptime(s,'%d-%b-%y')) } )
            # [cfs] => [m3/s]
            d[:,1] *= 0.028316847

            ts = timeseries.Timeseries( d[:,0],d[:,1] )
            
        ts = ts.clip(sim_start,sim_end)

        return ts.t_in(units='absdays'), ts.x * self.amplification

class FlowCsvMgd(Timeseries):
    """ Basic CSV format for flows.
    Assumes first line is headers, subsequent lines are YYYY-MM-DD HH:MM,24.5234
    where the flow is in mgd.
    timestamps are assumed to be at the center of the averaging interval, already adjusted
    to be UTC.  The suntans boundaries.c code doesn't do it yet, but it would be more appropriate
    to take the nearest value rather than interpolating between values (because these data are
    generally daily averages already).
    """
    def __init__(self,csv_fn,amplification=1.0,lag_s=0.0):
        Timeseries.__init__(self,label=os.path.basename(csv_fn),lag_s=lag_s)
        self.amplification = amplification
        self.fn = csv_fn
        
    def raw_data(self,sim_start,sim_end):
        ts = timeseries.Timeseries.load_csv(self.fn,
                                            skiprows=1,date_fmt="%Y-%m-%d %H:%M")
        
        # important to clip because the timeseries determines the starting point
        # for year days by looking at the year of the first entry in the timeseries.
        ts = ts.clip(sim_start,sim_end)

        #                                amplify and convert [mgd] -> [m3/s]
        return ts.t_in(units='absdays'), ts.x * self.amplification * 0.043812636


class NCFlow(Timeseries):
    """ Read flow data from a netcdf file
    """
    def __init__(self,nc_fn,amplification=1.0,lag_s=0.0):
        super(NCFlow,self).__init__(label=os.path.basename(nc_fn),lag_s=lag_s)
        self.amplification = amplification
        self.fn = nc_fn
        
    def raw_data(self,sim_start,sim_end):
        if not os.path.exists(self.fn):
            raise Exception("Forcing file doesn't exist: %s"%self.fn)
        nc=qnc.QDataset(self.fn)
        
        # assumes that time is called time, and is CF-like
        t=nc.time.as_datenum()
        Q=nc.flow[:]
        # convert to m3/s
        Q=Units.conform(Q,Units(nc.flow.units),Units('m3/s'))

        sel=utils.within(t,[date2num(sim_start),date2num(sim_end)])
        t=t[sel]
        Q=Q[sel]

        #ts = timeseries.Timeseries(t,Q)
        #ts = ts.clip(sim_start,sim_end)

        #                                amplify 
        # return ts.t_in(units='absdays'), ts.x * self.amplification 
        return t,Q


class EBDA_MDF(Timeseries):
    """ Flow data for the East Bay Dischargers Assoc.
    This is data from Mike Connor, mconnor@ebda.org, for MDF (Marina Dechlorination Facility),
    which includes EBDA contributors, plus some LAVWMA, less some diversions.  I'm pretty sure
    it's the last point before it's pumped into the bay.

    This can also be ignored, and use FlowCsvMgd instead.
    """
    fn = os.path.join(forcing_dir,"flows/ebda/ebda_mdf_flow.csv")

    def __init__(self,
                 amplification=1.0,
                 lag_s=0.0):
        Timeseries.__init__(self,label="EBDA",lag_s=lag_s)
        
        self.amplification = amplification

    def read_timeseries(self):
        fp = open(self.fn,'rt')
        fp.readline() # column names

        t = []
        f = [] 
        for line in fp:
            date,flow = line.split(',')
            flow = float(flow) * 0.043812636 # [MGD]->[m3/s]
            date = date2num( datetime.datetime.strptime(date,'%m/%d/%Y') ) + 0.5

            t.append(date)
            f.append(flow)

        return timeseries.Timeseries(array(t),array(f))
            
    def raw_data(self,sim_start,sim_end):
        ts = self.read_timeseries()
        # important to clip because the timeseries determines the starting point
        # for year days by looking at the year of the first entry in the timeseries.
        ts = ts.clip(sim_start,sim_end)

        return ts.t_in(units='absdays'), ts.x * self.amplification
    
class SJWWTP(Timeseries):
    """ Flow data for San Jose wastewater treatment plant inputs into
    Artesian slough.  Data obtained from Peter.Schafer@sanjoseca.gov

    Likewise, use FlowCsvMgd instead.
    """
    fn=os.path.join(forcing_dir,"flows/sjwwtp2005_2009.csv")

    def __init__(self,
                 amplification=1.0,
                 lag_s=0.0):
        Timeseries.__init__(self,label="SJWWTP",lag_s=lag_s)
        
        self.amplification = amplification

    def read_timeseries(self):
        fp = open(self.fn,'rt')
        fp.readline() # header info
        fp.readline() # column names

        t = []
        f = [] 
        for line in fp:
            date,flow = line.split(',')
            flow = float(flow) * 0.043812636 # [MGD]->[m3/s]
            date = date2num( datetime.datetime.strptime(date,'%m/%d/%Y') ) + 0.5

            t.append(date)
            f.append(flow)

        return timeseries.Timeseries(array(t),array(f))
            
            
    def raw_data(self,sim_start,sim_end):
        ts = self.read_timeseries()
        # important to clip because the timeseries determines the starting point
        # for year days by looking at the year of the first entry in the timeseries.
        ts = ts.clip(sim_start,sim_end)

        return ts.t_in(units='absdays'), ts.x * self.amplification

from rdb import Rdb

class UsgsGage(Timeseries):
    """ fetches data from waterdata.usgs.gov
    for now, just get daily information.
    """
    def __init__(self,station_code,lag_s = None,label=None,amplification=1.0):
        if label is None:
            label = "USGS streamflow, #%s"%station_code
        Timeseries.__init__(self, label=label, lag_s=lag_s)
        self.station_code = station_code
        self.amplification = amplification

    def raw_data(self,sim_start,sim_end):
        # cb_00060=on gives us daily mean
        sim_start=utils.to_datetime(sim_start)
        sim_end  =utils.to_datetime(sim_end)

        begin_date = sim_start.strftime("%Y-%m-%d")
        end_date   = (sim_end+datetime.timedelta(1)).strftime("%Y-%m-%d")

        self.url = "http://waterdata.usgs.gov/nwis/dv?referred_module=sw&" + \
                   "site_no=%d&cb_00060=on&begin_date=%s&end_date=%s&format=rdb"%(self.station_code,
                                                                                  begin_date,end_date)
        print(self.url)
        fp = urlopen(self.url)

        self.reader = Rdb(fp=fp)

        # this assumes that only one numeric data column is in the rdb file
        daily_mean_cfs = self.reader.data()
        
        cumecs = daily_mean_cfs * 0.028316847 * self.amplification
        absdays = self.reader['datetime']

        # fake any missing data by interpolation:
        invalid = isnan(cumecs)
        if any(invalid):
            print("USGS gage %s has some missing data.  will attempt to interpolate"%self.station_code)
            cumecs[invalid] = interp( absdays[invalid],
                                      absdays[~invalid],cumecs[~invalid],
                                      left = cumecs[~invalid][0], right = cumecs[~invalid][-1] )

        return absdays,cumecs


                
class CompositeUsgsGage(Timeseries):
    """ A weighted average of multiple usgs gages.  This is used in conjunction with the watershed-based
    flow forcing where each un-gaged source is correlated to gaged sources using watershed area.
    """
    def __init__(self,gage_ids,weights,lag_s = None,label=None,amplification=1.0):
        if label is None:
            label = "USGS streamflow, composite"
        Timeseries.__init__(self, label=label, lag_s=lag_s)
        self.gage_ids = gage_ids
        self.weights = weights
        self.amplification = amplification
        
    def raw_data(self,sim_start,sim_end):
        all_absdays = []
        all_cumecs = []

        for gage_id,weight in zip(self.gage_ids,self.weights):
            one_gage = UsgsGage(station_code=gage_id,label="temp",amplification=self.amplification)
            absdays,cumecs = one_gage.raw_data(sim_start,sim_end)

            if len(all_absdays) > 0:
                # For now, assert that they are all the same size and time period
                if len(absdays) != len(all_absdays[0]):
                    print(sim_start,sim_end)
                    print("While processing composite gage for %s"%self.label)
                    raise Exception("Lengths of absdays didn't match")
                if any( absdays != all_absdays[0] ):
                    print(sim_start,sim_end)
                    print("While processing composite gage for %s"%self.label)
                    raise Exception("Values of absdays didn't match")
                    
            all_absdays.append(absdays)
            all_cumecs.append(cumecs)

        absdays = all_absdays[0]
        total_cumecs = 0*all_cumecs[0]
        for cumecs,weight in zip(all_cumecs,self.weights):
            total_cumecs += weight * cumecs
            
        return absdays,total_cumecs

class KrigedSource(DataSource):
    """ A spatially variable field, based on Kriging between a given set
    of sources.  The other sources may themselves be time-varying.
    """
    def __init__(self,label,station_list):
        """ label: string giving short descriptive name of this field
            station_list: [ ([x,y],datasource),  ... ]
        """
        DataSource.__init__(self,label)
        
        self.station_list = station_list
        
    def write_config(self,fp,sun):
        fp.write("# %s\n"%self.label)
        fp.write("BEGIN_DATA\n")
        fp.write("  KRIGED\n")
        fp.write("  STATION_COUNT %d\n"%len(self.station_list))
        for xy,subsrc in self.station_list:
            fp.write("  STATION_SPEC %s %f %f\n"%(subsrc.filename, xy[0], xy[1]) )
        fp.write("END_DATA\n")
        
    def prepare(self,gforce):
        """ For Kriging sources, this is where subsources are registered, and
        we can get the proper references for them before write_config is called
        """
        for xy,subsrc in self.station_list:
            gforce.add_datasource(subsrc)


## Filters for modifying and combining data sources:            
class LowpassTimeseries(Timeseries):
    pad_factor = 2.0 # assume that the filter transients decay within time pad_factor*cutoff_days
    
    def __init__(self,source,cutoff_days,order=4):
        """ Returns a new forcing timeseries object which is a low-passed
        version of the source timeseries

        handles fetching a bit of extra data to pad out the input before filtering,
        """
        
        super(LowpassTimeseries,self).__init__(label="LP"+source.label)
        
        self.source=source
        self.cutoff_days = cutoff_days
        self.order = order

    def raw_data(self,start_datetime,end_datetime):
        """ Fetch data from the underlying source for a slightly larger time window,
        make sure it's evenly spaced, low-pass filter, truncate, and return
        """
        pad = datetime.timedelta(self.cutoff_days * self.pad_factor)

        source_times,source_data = self.source.raw_data(start_datetime - pad,
                                                        end_datetime+pad)

        lp_data = lp_filter.lowpass(source_data,source_times,self.cutoff_days,order=self.order)

        i_start = searchsorted(source_times,date2num(start_datetime),side='left')
        i_end =  searchsorted(source_times,date2num(end_datetime),side='right')

        # need to include an extra sample to completely enclose the range
        i_start = max(0,i_start-1)
        i_end += 1

        return source_times[i_start:i_end],lp_data[i_start:i_end]
        
class FillByLastValid(Timeseries):
    """ Wrap a timeseries, and when there is missing data, use the most
    recent valid data from before the missing data.
    """
    stride_days=1.0
    max_backwards_days=30
    
    def __init__(self,source,stride_days=None,max_backwards_days=None,fallback=0.0):
        """ source: a Timeseries object
        stride_days: when the request period starts with invalid data, this
        gives the stride for checking past periods
        max_backwards_days: if no valid data is found within this amount of
          time, then leading invalid values are given the next valid value.
        if there are no valid values anywhere, then returns fallback
        """
        super(FillByLastValid,self).__init__(label="Fill"+source.label)
        self.source=source
        if stride_days:
            self.stride_days=stride_days
        if max_backwards_days:
            self.max_backwards_days=max_backwards_days
        self.fallback=fallback

    def raw_data(self,start_datetime,end_datetime):
        """ Fetch data from the underlying source for a slightly larger time window,
        make sure it's evenly spaced, low-pass filter, truncate, and return
        """
        source_times,source_data = self.source.raw_data(start_datetime,
                                                        end_datetime)
        
        if isnan(source_data[0]):
            print("FillByLastValid: looking backwards in time")
            Nbacks=int(self.max_backwards_days/self.stride_days)
            last_valid = nan
            
            for i in range(1,Nbacks+1):
                start_dt=start_datetime - datetime.timedelta(i*self.stride_days)
                end_dt=start_datetime - datetime.timedelta((i-1)*self.stride_days)
                back_times,back_data = self.source.raw_data(start_dt,end_dt)
                if any( isfinite(back_data) ):
                    last_valid= back_data[isfinite(back_data)][-1]
                    break
            if isnan(last_valid):
                print("FillByLastValid: found no past, useable data.")

                if any(isfinite(source_data)):
                    last_valid=source_data[isfinite(source_data)][0]
                else:
                    print("No valid data anywhere - using fallback value")
                    last_valid=self.fallback
        else:
            last_valid=source_data[0]

        for i in range(len(source_data)):
            if isnan(source_data[i]):
                source_data[i]=last_valid
            else:
                last_valid=source_data[i]
                
        return source_times,source_data
    

class ShiftTimeseries(Timeseries):
    """ Apply time/value shift/scaling
    """
    
    def __init__(self,source,amplify=1.0,delay_s=0.0,offset=0.0,center=None):
        """ Returns a new forcing timeseries object which is has time/value shifts
        relative the source timeseries.

        amplify scales the data about center, which if unspecified is taken as the mean.
         (i.e. good for tides, bad for wind)
        delay_s will shift the data in time
        """
        super(ShiftTimeseries,self).__init__(label="Shift"+source.label)
        
        self.source=source
        self.amplify = amplify
        self.delay_s = delay_s
        self.offset = offset
        self.center = center

    def raw_data(self,start_datetime,end_datetime):
        """ Fetch data from the underlying source for a slightly larger time window,
        make sure it's evenly spaced, low-pass filter, truncate, and return
        """
        delay_delta = datetime.timedelta(self.delay_s/86400.)

        print("ShiftTimeSeries: end_datetime: %s"%end_datetime)
        print("                 shifted end: %s"%(end_datetime - delay_delta))
        
        source_times,source_data = self.source.raw_data(start_datetime - delay_delta,
                                                        end_datetime - delay_delta)

        source_times += self.delay_s/86400.

        print("Resulting range of data: %s - %s"%( num2date(source_times[0]),
                                                   num2date(source_times[-1])))

        if self.center is None:
            center = mean(source_data)
        else:
            center = self.center

        new_data = (source_data - center)*self.amplify + center + self.offset

        return source_times,new_data

        
def read_boundaries_dat(sun,proc):
    
    fp = open(sun.file_path('BoundaryInput',proc),'rt')
    
    gforce = GlobalForcing(sun=sun,proc=proc)

    # simple tokenizer
    #  able to handle comments that start with a # 
    def tok_gen():
        for line in fp:
            for t in line.split():
                if t[0] == '#':
                    break # skip the rest of the line
                yield t
    tok = tok_gen().__next__
    def tok_tag(s):
        t = tok()
        if t != s:
            print("Expected %s, got %s"%(s,t))
    def tok_int(tag=None):
        if tag:
            tok_tag(tag)
        return int(tok())
    def tok_float(tag=None):
        if tag:
            tok_tag(tag)
        return float(tok())
    def tok_str(tag=None):
        if tag:
            tok_tag(tag)
        return tok()


    version = tok_int('BOUNDARY_FORCING')
    print("reading boundaries.dat version %d"%version)

    if version == 2:
        # format, something like:
        ntides = tok_int()
        ncells = tok_int()
        ngages = tok_int()
        gage_t0 = tok_float()
        gage_dt = tok_float()
        ngage_steps = tok_int()

        if ntides > 0:
            raise Exception("New forcing code not tested with old format and harmonics")

        omegas = [tok_float() for tide_i in range(ntides)]

        gage_weights = zeros( (ncells,ngages), float64 )
        for c in range(ncells):
            # this is where we should be doing something smarter...
            for x in range(ntides*6):
                tok_float()

            # this is what we want:
            for gi in range(ngages):
                gage_weights[c] = [tok_float() for x in ngages]

        datasources = []
        if ngages > 0:
            gage_data = zeros( (ngage_steps,ngages,3), float64)
            
            for gage_step in range(ngage_steps):
                for gage_i in range(ngages):
                    # read u,v,h
                    gage_data[gage_step,gage_i] = [tok_float(),tok_float(),tok_float()]
                    

            for i in range(ngages):
                datasources.append( Timeseries("gage%i"%i,
                                               t0=gage_t0,
                                               dt=gage_dt,
                                               data=gage_data[:,gage_i],
                                               lag_s=0) )

        # For now, assume all edges are getting the same weight
        # Still, the old code starts with the boundary edges, gets the boundary cells,
        # and those are what are listed in the boundaries.dat files.  
        raise Exception("Really not prepared for reading the old forcing file.")

            
            
        # BOUNDARY_FORCING version
        #    <number of tidal components>
        #    <number of boundary cells>
        #    <number of gages>
        #    <gage t0 - simulation_start, in seconds>
        #    <gage timestep>
        #    <num gage timesteps>
        #    ntides * <omega>
        #    ncells * [ ntides * <uamp> 
        #               ntides * <uphase>
        #               ntides * <vamp>
        #               ntides * <vphase>
        #               ntides * <hamp>
        #               ntides * <hphase>
        #               ngages * <gage weight> ]
        #    ngagetimesteps * [  ngages * [ u,v,h ] ]

    elif version == 6:
        # read in the datasources first, stored into a dict:
        ds_dir = os.path.join(sun.datadir,'datasources')
        datasources = {}
        for f in glob.glob(os.path.join(ds_dir,"*")):
            ds_name = os.path.basename(f)
            try:
                datasources[ds_name] = read_datasource(f,sun)
            except Exception as e:
                print("Couldn't read datasource %s (file %s)"%(ds_name,f))
                datasources[ds_name] = None
            
        itemlist_count = tok_int('ITEMLIST_COUNT')
        print("itemlist_count is" ,itemlist_count)
        for itemlist_index in range(itemlist_count):
            tok_tag('BEGIN_ITEMLIST')

            # identify which model elements are being forced, and define a forcing
            # group
            item_type = tok_str('ITEM_TYPE')

            if item_type in ('EDGE','CELL'):
                item_count = tok_int('ITEM_COUNT')
                tok_tag('ITEMS')
                items = [tok_int() for i in range(item_count)]
                if item_type == 'EDGE':
                    group = gforce.new_group(edges = items)
                elif item_type == 'CELL':
                    group = gforce.new_group(cells = items)
                else:
                    raise Exception("unknown item type %s"%item_type)
            elif item_type in ('ALL_CELLS','ALL_EDGES'):
                items = item_type
                dimensions = tok_str('DIMENSIONS')
                if item_type == 'ALL_CELLS':
                    group = gforce.new_group(cells='all')
                elif item_type == 'ALL_EDGES':
                    group = gforce.new_group(edges='all')
            
            # then read what parameters are forced, and what datasource is used.
            bc_count = tok_int('BC_COUNT')
            for bc_index in range(bc_count):
                bctype = tok_str('BCTYPE')
                data_index = tok_str('DATA')
                group.add_datasource(datasources[data_index],bctype)
            tok_tag('END_ITEMLIST')
        
    else:
        # READ DATA SECTIONS:
        data_count = tok_int('DATA_COUNT')
        datasources = [None] * data_count
        # print "Reading %d data sections"%data_count

        for data_i in range(data_count):
            dsource = None
            tok_tag('BEGIN_DATA')
            dtype = tok()

            if dtype in ('TIMESERIES','TIMESERIES_2VEC'):
                sample_count = tok_int('SAMPLE_COUNT')
                dt = tok_float('DT')
                t0_sun_seconds = tok_float('TZERO')
                # convert to datetime:
                base_date = datetime.datetime(sun.time_zero().year,1,1)
                t0 = base_date + datetime.timedelta(t0_sun_seconds / (24.*3600.))
                
                tok_tag('VALUES')
                if dtype == 'TIMESERIES':
                    values = zeros( sample_count, float64)
                    for i in range(sample_count):
                        values[i] = tok_float()
                    dsource = Timeseries("timeseries%i"%data_i,
                                         t0=t0,
                                         dt=dt,
                                         data=values,lag_s=0)
                else:
                    values = zeros( (sample_count,2), float64)
                    for i in range(sample_count):
                        values[i,0] = tok_float()
                        values[i,1] = tok_float()
                    dsource = Timeseries2Vector("timeseries2vec%i"%data_i,
                                                t0=t0,
                                                dt=dt,
                                                data=values,lag_s=0)
            elif dtype == 'CONSTANT':
                value = tok_float('VALUE')
                dsource = Constant("const%i"%data_i,
                                   value=value)
                
            elif dtype == 'HARMONICS':
                constituents_count = tok_int('CONSTITUENTS_COUNT')
                omegas = zeros(constituents_count,float64)
                phases = zeros_like(omegas)
                amps   = zeros_like(omeags)

                for sec,vals in zip(['OMEGAS','PHASES','AMPLITUDES'],
                                    [omegas,phases,amps]):
                    tok_tag(sec)
                    for i in range(constituents_count):
                        vals[i] = tok_float()
                dsource = Harmonics("harmonics%i"%data_i,
                                    omegas=omegas,
                                    phases=phases,
                                    amplitudes = amplitudes)
            else:
                raise Exception("Unrecognized data type: %s"%dtype)

            datasources[data_i] = dsource
            tok_tag('END_DATA')

        # READ EDGELIST SECTIONS:
        edgelist_count = tok_int('EDGELIST_COUNT')

        for elist_i in range(edgelist_count):
            tok_tag('BEGIN_EDGELIST')
            edge_count = tok_int('EDGE_COUNT')
            edges = zeros(edge_count,int32)
            tok_tag('EDGES')
            for i in range(edge_count):
                edges[i] = tok_int()

            group = gforce.new_group(edges = edges)

            bc_count = tok_int('BC_COUNT')

            bcs = [None]*bc_count
            for i in range(bc_count):
                tok_tag('BCTYPE')
                bctype = tok()
                dsource_index = tok_int('DATA')

                group.add_datasource(datasources[dsource_index],bctype)

            tok_tag('END_EDGELIST')

        if version == 5:
            print("Version 5, reading cell lists")
            celllist_count = tok_int('CELLLIST_COUNT')

            for clist_i in range(celllist_count):
                tok_tag('BEGIN_CELLLIST')
                cell_count = tok_int('CELL_COUNT')
                cells = zeros(cell_count,int32)
                tok_tag('CELLS')
                for i in range(cell_count):
                    cells[i] = tok_int()

                group = gforce.new_group(cells = cells)

                bc_count = tok_int('BC_COUNT')

                bcs = [None]*bc_count
                for i in range(bc_count):
                    tok_tag('BCTYPE')
                    bctype = tok()
                    dsource_index = tok_int('DATA')

                    group.add_datasource(datasources[dsource_index],bctype)

                tok_tag('END_CELLLIST')
    return gforce


def read_datasource(fn,sun):
    #
    fp = open(fn,'rt')

    # Read any header lines, and concatenate to make a comment
    comment = [os.path.basename(fn)]
    while 1:
        txt = fp.readline().strip()
        if len(txt) == 0:
            pass
        elif txt[0] == '#':
            comment.append( txt[1:].strip() )
        else:
            break

    comment = " ".join(comment)

    # simple tokenizer
    #  able to handle comments that start with a # 
    def tok_gen():
        for t in txt.split():
            if t[0] == '#':
                break
            yield t
            
        for line in fp:
            for t in line.split():
                if t[0] == '#':
                    break # skip the rest of the line
                yield t
                
    tok = tok_gen().__next__
    def tok_tag(s):
        t = tok()
        if t != s:
            print("Expected %s, got %s"%(s,t))
    def tok_int(tag=None):
        if tag:
            tok_tag(tag)
        return int(tok())
    def tok_float(tag=None):
        if tag:
            tok_tag(tag)
        return float(tok())


    tok_tag('BEGIN_DATA')
    ds_type = tok()

    if ds_type == "TIMESERIES":
        sample_count = tok_int('SAMPLE_COUNT')
        dt = tok_float('DT')

        t0_sun_seconds = tok_float('TZERO')
        # convert to datetime:
        base_date = datetime.datetime(sun.time_zero().year,1,1)
        t0 = base_date + datetime.timedelta(t0_sun_seconds / (24.*3600.))

        tok_tag('VALUES')
        data = array( [tok_float() for i in range(sample_count)] )
        tok_tag('END_DATA')
        dsource = Timeseries(comment,
                             t0=t0,
                             dt=dt,
                             data=data,lag_s=0)
    else:
        dsource = Constant(comment + "FAKE", value=1)

    return dsource
    
