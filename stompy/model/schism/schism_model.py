from ...grid import multi_ugrid, unstructured_grid
import numpy as np
import glob, os

class MultiSchism(multi_ugrid.MultiUgrid):
    def __init__(self,grid_fn,paths,xr_kwargs={}):
        if not isinstance(grid_fn,unstructured_grid.UnstructuredGrid):
            self.grid=unstructured_grid.UnstructuredGrid.read_gr3(grid_fn)
        else:
            self.grid=grid_fn
            
        if isinstance(paths,str):
            paths=glob.glob(paths)
            # more likely to get datasets in order of processor rank
            # with a sort.
            paths.sort()
        self.paths=paths
        self.dss=self.load(**xr_kwargs)
        self.n_ranks=len(self.dss)
        
        # avoid loading grids -- in SCHISM the subdomain grids are
        # not as cleanly independent, but we have explicit mappings

        self.rev_meta={ 'nSCHISM_hgrid_node':'node_dimension',
                        'nSCHISM_hgrid_edge':'edge_dimension',
                        'nSCHISM_hgrid_face':'face_dimension', }

        self.create_global_grid_and_mapping()
        
    def create_global_grid_and_mapping(self):
        self.node_l2g=[] # per rank, 
        self.edge_l2g=[]
        self.cell_l2g=[]

        for rank in range(self.n_ranks):
            ds=self.dss[rank]
            nc_fn=self.paths[rank]
            
            fn=os.path.join(os.path.dirname(nc_fn),'local_to_global_%04d'%rank)
            fp=open(fn,'rt')
            sizes=[int(s) for s in fp.readline().strip().split()]
            comment=fp.readline()
            Ncell=int(fp.readline().strip())
            assert Ncell==ds.dims['nSCHISM_hgrid_face']
            c_map=np.loadtxt(fp,np.int32,max_rows=Ncell) - 1
            Nnode=int(fp.readline().strip())
            assert Nnode==ds.dims['nSCHISM_hgrid_node']
            n_map=np.loadtxt(fp,np.int32,max_rows=Nnode) - 1
            Nedge=int(fp.readline().strip())
            assert Nedge==ds.dims['nSCHISM_hgrid_edge']
            j_map=np.loadtxt(fp,np.int32,max_rows=Nedge) - 1

            # From the source file, these all include the local index.
            # verified that those are sequential, so with no further
            # munging we're done:
            self.node_l2g.append(n_map[:,1])
            self.edge_l2g.append(j_map[:,1])
            self.cell_l2g.append(c_map[:,1])
