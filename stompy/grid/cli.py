"""
Grid manipulation related command line interface.  

Access this with something like:
  python -m stompy.grid.cli <args>
"""
# why has python gone through three iterations of argument parsing modules?
import sys
import argparse
import logging as log

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
            stack.append(g)
            log.info("Reading grid (%d cells, %d edges, %d nodes)"%(g.Ncells(),g.Nedges(),g.Nnodes()))
        else:
            log.error("Did not understand format %s"%fmt)
            sys.exit(1)

class WriteGrid(Op):
    def run(self,args):
        fmt,path=args[0].split(':')
        log.info("Writing %s as %s"%(path,fmt))

        g=stack[-1] # by default, do not pop from stack

        if fmt=='suntans':
            g.write_suntans(path)
        elif fmt=='suntans_hybrid':
            g.write_suntans_hybrid(path)
        elif fmt=='ugrid':
            g.write_ugrid(path)
        elif fmt=='untrim':
            g.write_untrim08(path)
        elif fmt=='cell_shp':
            g.write_cells_shp(path)
        elif fmt=='boundary_shp':
            g.write_shore_shp(path)
        elif fmt=='edge_shp':
            g.write_edges_shp(path)
        elif fmt=='node_shp':
            g.write_node_shp(path)
        elif fmt=='fishptm':
            g.write_ptm_gridfile(path)
        elif fmt=='dfm':
            from stompy.model.delft import dfm_grid
            if not path.endswith('_net.nc'):
                log.warning("Writing DFM grid to filename not ending in '_net.nc'")
            dfm_grid.write_dfm(g,path)
        else:
            log.error("Did not understand format %s"%fmt)
            sys.exit(1)


parser.add_argument("-i", "--input", help="Read a grid",
                    nargs=1,action=ReadGrid)
parser.add_argument("-o", "--output", help="Write a grid",
                    nargs=1,action=WriteGrid)

if __name__ == '__main__':
    args = parser.parse_args()
    from . import unstructured_grid

    for impl,args in ops:
        impl.run(args)
        

    
    

