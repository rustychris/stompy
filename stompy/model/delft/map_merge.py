"""
Merge DFM map files from multiprocessor to single domain.

In theory dfmoutput mapmerge infiles...
does this, but experience has shown it is not reliable.

"""


class MapMerge(object):
    def __init__(self,infiles,outfile,global_grid):
        self.global_grid=global_grid
        self.outfile=outfile
        self.infiles=infiles
    def write_merged(self):
        self.write_grid()
        self.write_data()
    def write_grid(self):
        # here - could decide on dialect, naming.
        ds=self.global_grid.write_to_xarray()
        ds.to_netcdf(self.outfile)

        
        
