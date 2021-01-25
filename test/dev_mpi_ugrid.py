"""
Use dask.Array to lazily combine subdomain outputs
"""
import os,glob
import copy
import numpy as np
import xarray as xr
from stompy import utils
from stompy.grid import unstructured_grid
import matplotlib.pyplot as plt


## 

out_dir="/home/rusty/src/csc/dflowfm/runs/v03regroup_20190115/DFM_OUTPUT_flowfm"
map_files=glob.glob(os.path.join(out_dir,'*0???_map.nc'))

##

class MultiVar(object):
    """ 
    Proxy for a single variable of a MultiUgrid instance.
    i.e. DataArray
    """
    def __init__(self,mu,sub_vars):
        self.mu=mu
        self.sub_vars=sub_vars
        
    # Still possible to request
    def __repr__(self):
        return "MultiVar wrapper around %s"%repr(self.sub_vars[0])
    def __str__(self):
        return "MultiVar wrapper around %s"%str(self.sub_vars[0])
    def isel(self,**kwargs):
        return MultiVar(self.mu,
                        [da.isel(**kwargs) for da in self.sub_vars])
    def sel(self,**kwargs):
        return MultiVar(self.mu,
                        [da.sel(**kwargs) for da in self.sub_vars])
    @property
    def values(self):
        """
        Combine subdomain values
        """
        sv0=self.sub_vars[0] # template sub variable
        res_dims=sv0.dims

        meta=self.mu.grids[0].nc_meta
        rev_meta={meta[k]:k for k in meta} # Reverse that
        shape=[]

        l2g=None

        left_idx=[]
        right_idx=[]
        
        for dim_i,dim in enumerate(res_dims):
            if rev_meta.get(dim,None)=='face_dimension':
                shape.append( self.mu.grid.Ncells() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                left_idx.append( lambda proc: self.mu.cell_l2g[proc] )
            elif rev_meta.get(dim,None)=='edge_dimension':
                shape.append( self.mu.grid.Nedges() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                left_idx.append( lambda proc: self.mu.edge_l2g[proc] )
            elif rev_meta.get(dim,None)=='node_dimension':
                shape.append( self.mu.grid.Nnodes() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                left_idx.append( lambda proc: self.mu.node_l2g[proc] )
            else:
                shape.append( sv0.shape[dim_i] )
                left_idx.append( lambda proc: slice(None) )
            right_idx.append( lambda proc: slice(None) ) # no subsetting on rhs for now.

        result=np.zeros( shape, sv0.dtype)

        # Copy subdomains to global:

        for proc,sv in enumerate(self.sub_vars):
            # In the future may want to control which subdomain provides
            # a value in ghost cells, by having some values of the mapping
            # negative, and they get filtered out here.
            left_slice =tuple( [i(proc) for i in left_idx ])
            right_slice=tuple( [i(proc) for i in right_idx])
            result[left_slice]=sv.values[right_slice]
        return result

    @property
    def dims(self):
        return self.sub_vars[0].dims
        
class MultiUgrid(object):
    """
    Given a list of netcdf files, each having a subdomain in ugrid, and
    possibly also having output data on the respective grids,
    Generate a global grid, and provide an interface approximating
    xarray.Dataset that performs the subdomain->global domain translation
    on the fly.
    """
    # HERE:
    # Need to figure these out so that at the very least
    # .values can invoked the right mappings.
    # one step better is to figure out that sometimes a variable doesn't
    # have any Multi-dimensions, and we can return a proper xr result
    # straight away.
    node_dim=None
    edge_dim=None
    cell_dim=None

    # Unclear if there is a situation where subdomains have to be merged with
    # a nonzero tolerance
    merge_tol=0.0
    
    def __init__(self,paths,cleanup_dfm=False,
                 **grid_kwargs):
        self.paths=paths
        self.dss=[xr.open_dataset(p) for p in paths]
        self.grids=[unstructured_grid.UnstructuredGrid.read_ugrid(ds,**grid_kwargs) for ds in self.dss]

        if cleanup_dfm:
            for g in self.grids:
                unstructured_grid.cleanup_dfm_multidomains(g)

        self.create_global_grid_and_mapping()

    def create_global_grid_and_mapping(self):
        self.node_l2g=[]
        self.edge_l2g=[]
        self.cell_l2g=[]

        for gnum,g in enumerate(self.grids):
            if gnum==0:
                self.grid=self.grids[0].copy()
                n_map=np.arange(self.grid.Nnodes())
                j_map=np.arange(self.grid.Nedges())
                c_map=np.arange(self.grid.Ncells())
            else:
                n_map,j_map,c_map = self.grid.add_grid(g,merge_nodes='auto',
                                                       tol=self.merge_tol)
            self.node_l2g.append(n_map)
            self.edge_l2g.append(j_map)
            self.cell_l2g.append(c_map)
        
    def __getitem__(self,k):
        # return a proxy object - can't do the translation until .values is called.
        if k in list(self.dss[0].variables.keys()):
            return MultiVar(self,
                            [ds[k] for ds in self.dss])
        else:
            raise KeyError("%s is not an existing variable"%k)
    
    def __getattr__(self,k):
        # Two broad cases here
        #  attempting to get a variable
        #     => delegate to getitem
        #  attempting some operation, that we probably don't know how to complete
        try: 
            return self.__getitem__(k)
        except KeyError:
            raise Exception("%s is not an existing variable or known method"%k)

    def __str__(self):
        return "MultiFile Layer on top of %s"%str(self.dss[0])
    def __repr__(self):
        return str(self)

    def isel(self,**kwargs):
        subset=copy.copy(self)
        subset.dss=[ds.isel(**kwargs) for ds in self.dss]
        return subset
    def sel(self,**kwargs):
        subset=copy.copy(self)
        subset.dss=[ds.sel(**kwargs) for ds in self.dss]
        return subset

    
mu=MultiUgrid(map_files,cleanup_dfm=True)

# this part is ok
# plt.figure(1).clf()
# mu.grid.plot_edges(lw=3,alpha=0.4)

## 
    
ucmag=mu['mesh2d_ucmag'].isel(time=-1).values

plt.figure(1).clf()

ccoll=mu.grid.plot_cells(values=ucmag,cmap='jet')

plt.axis('tight')
plt.axis('equal')

# basically works, but ghost cells need to be dealt with.
