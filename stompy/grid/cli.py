"""
Grid manipulation related command line interface.

Access this with something like:
  python -m stompy.grid.cli <args>
"""
# why has python gone through three iterations of argument parsing modules?
from __future__ import print_function
import sys
import argparse
import logging as log

from . import unstructured_grid

ops=[]
stack=[]

class Op(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        #print '%r %r %r' % (namespace, values, option_string)
        ops.append( (self,values) )

class ReadGrid(Op):
    formats={}
    def run(self,args):
        fmt,path=args[0].split(':')
        log.info("Reading %s as %s"%(path,fmt))

        if fmt in self.formats:
            g=self.formats[fmt][1](fmt,path)
        else:
            log.error("Did not understand format %s"%fmt)
            log.error("Read formats are: %s"%(self.format_list()))
            sys.exit(1)
        stack.append(g)
        log.info("Read grid (%d cells, %d edges, %d nodes)"%(g.Ncells(),g.Nedges(),g.Nnodes()))
    @classmethod
    def format_list(cls):
        fmt_names=list(cls.formats.keys())
        fmt_names.sort()
        return ", ".join(fmt_names)
        
ReadGrid.formats['suntans_classic']=['SUNTANS (classic)',
                                     lambda fmt,path: unstructured_grid.UnstructuredGrid.read_suntans_classic(path)]
ReadGrid.formats['suntans_hybrid']=['SUNTANS (hybrid)',
                                     lambda fmt,path: unstructured_grid.UnstructuredGrid.read_suntans_hybrid(path)]
ReadGrid.formats['suntans']=['SUNTANS (auto)',
                                     lambda fmt,path: unstructured_grid.UnstructuredGrid.read_suntans(path)]
ReadGrid.formats['ugrid']=['UGRID netCDF',
                           lambda fmt,path: unstructured_grid.UnstructuredGrid.from_ugrid(path)]
ReadGrid.formats['untrim']=['UnTRIM',
                            lambda fmt,path: unstructured_grid.UnTRIM08Grid(path)]
ReadGrid.formats['dfm']=['DFM netCDF (*_net.nc)',
                         lambda fmt,path: unstructured_grid.UnstructuredGrid.read_dfm(fn=path)]
ReadGrid.formats['sms']=['SMS grd',
                         lambda fmt,path: unstructured_grid.UnstructuredGrid.read_sms(path)]
ReadGrid.formats['ras2d']=(
    ['RAS2D h5', lambda fmt,path: unstructured_grid.UnstructuredGrid.read_ras2d(path,subedges='subedges') ]
)

class Dualify(Op):
    def run(self,args):
        g=stack.pop()
        gd=g.create_dual(center='circumcenter',create_cells=True,remove_disconnected=True)
        stack.append(gd)

# TODO: switch format handling to be more like ReadGrid
class WriteGrid(Op):
    clobber=False
    
    def run(self,args):
        fmt,path=args[0].split(':')
        log.info("Writing %s as %s"%(path,fmt))

        g=stack[-1] # by default, do not pop from stack

        if fmt in ['suntans_classic','suntans']:
            g.write_suntans(path,overwrite=self.clobber)
        elif fmt=='suntans_hybrid':
            g.write_suntans_hybrid(path,overwrite=self.clobber)
        elif fmt=='ugrid':
            g.write_ugrid(path,overwrite=self.clobber)
        elif fmt=='untrim':
            g.write_untrim08(path,overwrite=self.clobber)
        elif fmt=='cell_shp':
            g.write_cells_shp(path,overwrite=self.clobber)
        elif fmt=='boundary_shp':
            g.write_shore_shp(path,overwrite=self.clobber)
        elif fmt=='edge_shp':
            g.write_edges_shp(path,overwrite=self.clobber)
        elif fmt=='node_shp':
            g.write_node_shp(path,overwrite=self.clobber)
        elif fmt=='fishptm':
            g.write_ptm_gridfile(path,overwrite=self.clobber)
        elif fmt=='dfm':
            from stompy.model.delft import dfm_grid
            if not path.endswith('_net.nc'):
                log.warning("Writing DFM grid to filename not ending in '_net.nc'")
            dfm_grid.write_dfm(g,path,overwrite=self.clobber)
        else:
            log.error("Did not understand format %s"%fmt)
            log.error("Possible formats are: %s"%self.format_list())
            sys.exit(1)
    @classmethod
    def format_list(cls):
        return "suntans_classic, suntans_hybrid, ugrid, untrim, cell_shp, boundary_shp, edge_shp, node_shp, fishptm, dfm"

class SetClobber(Op):
    def run(self,args):
        WriteGrid.clobber=True

parser = argparse.ArgumentParser(description='Manipulate unstructured grids.')

parser.add_argument("-i", "--input", help="Read a grid, fmt one of %s"%ReadGrid.format_list(),
                    metavar="fmt:path",
                    nargs=1,action=ReadGrid)
parser.add_argument("-o", "--output", help="Write a grid, fmt one of %s"%WriteGrid.format_list(),
                    metavar="fmt:path",
                    nargs=1,action=WriteGrid)
parser.add_argument("-d", "--dual", help="Convert to dual of grid",
                    nargs=0,action=Dualify)
parser.add_argument("-c", "--clobber", help="Allow overwriting of existing grids",
                    nargs=0,action=SetClobber)


def parse_and_run(cmd=None):
    # In case there are repeated calls from a script:
    del ops[:]
    del stack[:]

    if cmd is not None:
        # allows for calling cli.py in a script, but with the exact same commandline.
        # Except that this doesn't work?  It gives an error related to ipython arguments
        # and the real sys.argv, rather than the argv I just specified.
        argv=cmd.split()
        print("Parsing %r"%argv)
        args=parser.parse_args(argv)
    else:
        args=parser.parse_args()

    for impl,args in ops:
        impl.run(args)

if __name__ == '__main__':
    parse_and_run()

