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

parser = argparse.ArgumentParser(description='Manipulate unstructured grids.')

ops=[]
stack=[]

class Op(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        #print '%r %r %r' % (namespace, values, option_string)
        ops.append( (self,values) )

class ReadGrid(Op):
    def run(self,args):
        fmt,path=args[0].split(':')
        log.info("Reading %s as %s"%(path,fmt))

        if fmt=='suntans':
            g=unstructured_grid.SuntansGrid(path)
        elif fmt=='ugrid':
            g=unstructured_grid.UnstructuredGrid.from_ugrid(path)
        elif fmt=='untrim':
            g=unstructured_grid.UnTRIM08Grid(path)
        elif fmt=='dfm':
            from stompy.model.delft import dfm_grid
            if not path.endswith('_net.nc'):
                log.warning("Read DFM grid from filename not ending in '_net.nc'")
            g=dfm_grid.DFMGrid(path)
        else:
            log.error("Did not understand format %s"%fmt)
            sys.exit(1)
        stack.append(g)
        log.info("Reading grid (%d cells, %d edges, %d nodes)"%(g.Ncells(),g.Nedges(),g.Nnodes()))

class WriteGrid(Op):
    clobber=False
    def run(self,args):
        fmt,path=args[0].split(':')
        log.info("Writing %s as %s"%(path,fmt))

        g=stack[-1] # by default, do not pop from stack

        if fmt=='suntans':
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
            sys.exit(1)

class SetClobber(Op):
    def run(self,args):
        WriteGrid.clobber=True

parser.add_argument("-i", "--input", help="Read a grid",
                    nargs=1,action=ReadGrid)
parser.add_argument("-o", "--output", help="Write a grid",
                    nargs=1,action=WriteGrid)
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

