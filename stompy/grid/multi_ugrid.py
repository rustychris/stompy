"""
Open ugrid-ish subdomains and approximate a global domain

Currently this is pretty slow, due to the add_grid() step.

The ghost-cell handling is also very basic, and does not
properly distinguish ghost cells along land boundaries.

TODO: handling isel for partitioned dimensions
  handling DFM output with FlowLink vs. NetLink
  dims should report the merged dimensions sizes, not the first subdomain.
"""

import glob
import copy
import xarray as xr
import numpy as np
from . import unstructured_grid
import logging
log=logging.getLogger('multi_ugrid')

class MultiVar(object):
    """ 
    Proxy for a single variable of a MultiUgrid instance.
    i.e. DataArray.
    Handles isel() calls by dispatching non-partitioned dimensions
    """
    _size=None
    def __init__(self,mu,sub_vars):
        self.mu=mu
        self.sub_vars=sub_vars

        # sv_dims: list of dimensions of the underlying sub_vars
        #    these may change by isel() that eliminate non-partition
        #    dimensions.
        # dims: active dimensions of the MultiVar. These may change by
        #    isel() that eliminate partition dimensions. In that case
        #    the result may no longer be MultiVar.
        # Need to think about this more. Which of these should be
        # used in shape_and_indices?
        self.sv_dims=self.dims=self.sub_vars[0].dims
        self.part_dims={}
        for dim in self.dims:
            if self.mu.rev_meta.get(dim,None) in ['face_dimension','node_dimension','edge_dimension']:
                self.part_dims[dim]=slice(None)
        
    # Still possible to request
    def __repr__(self):
        return "MultiVar wrapper around %s"%repr(self.sub_vars[0])
    def __str__(self):
        return "MultiVar wrapper around %s"%str(self.sub_vars[0])

    def isel(self,**kwargs):
        # Apply indexing, returning either a more restricted MultiVar,
        # or if the selection includes the partitioned dimension, then
        # return a vanilla Dataset.
        
        # Break up the requested indices into partitioned and non-partitioned
        # dimensions:
        part_kwargs={}
        nonpart_kwargs={}

        for key in kwargs:
            val=kwargs[key]
            if key not in self.part_dims:
                nonpart_kwargs[key]=val
            else:
                part_kwargs[key]=val

        # Apply the nonpartitioned selections:
        mv=MultiVar(self.mu,                                                                                                                      
                    [da.isel(**nonpart_kwargs) for da in self.sub_vars])

        if len(part_kwargs)==0:
            return mv
        assert len(part_kwargs)<=1,"Not ready for multiple partitioned dimensions on one var"

        # Come back and apply the partitioned selections
        for key in part_kwargs:
            val=part_kwargs[key]
            if self.mu.rev_meta[key]=='face_dimension':
                g2l=self.mu.cell_g2l
            elif self.mu.rev_meta[key]=='node_dimension':
                g2l=self.mu.node_g2l
            elif self.mu.rev_meta[key]=='edge_dimension':
                g2l=self.mu.edge_g2l
            else:
                raise Exception("Mapping global-to-local not implemented for %s"%key)

            # val could be an int, a sequence of ints, or a slice.
            # while numpy allows a multidimension index array, xarray does
            # not, and we'll follow that same constraint.
            if isinstance(val,slice):
                # Slices can be no-copy on a regular dataset, but here we 
                # have to revert to copying, and convert the slice to a
                # sequence. 
                val=range(len(g2l))[val]

            val=np.asanyarray(val)
            if val.shape==(): # scalar
                proc,loc=g2l[val]
                sv=mv.sub_vars[proc].isel(**{key:loc})
                return sv
            else:
                svs=[mv.sub_vars[proc].isel(**{key:loc})
                     for proc,loc in g2l[val]]
                return xr.concat(svs,dim=key)
    
    def sel(self,**kwargs):
        return MultiVar(self.mu,
                        [da.sel(**kwargs) for da in self.sub_vars])

    _size=None
    @property
    def size(self):
        if self._size is None:
            self._size=sum([v.size for v in self.sub_vars])
        return self._size

    @property
    def ndim(self):
        return self.sub_vars[0].ndim
            
    def shape_and_indexes(self):
        """
        Determine the shape of the combined variable, and how to 
        index the source datasets.
        returns ( shape, left_idx, right_idx)
        each of left_idx and right_idx are a list of functions
        each function takes subdomain index (0-based), and returns the
        corresponding entry for indexing.  left_idx takes care of the
        multi-domain merging by including an index array for dimensions
        which get merged across subdomains.
        right_idx handles ghost entries for merged dimensions, and collected
        subsetting for all dimensions.
        """
        sv0=self.sub_vars[0] # template sub variable

        shape=[]

        l2g=None

        left_idx=[]
        right_idx=[]
        
        for dim_i,dim in enumerate(self.dims):
            right=lambda proc: slice(None)
            
            if self.mu.rev_meta.get(dim,None)=='face_dimension':
                shape.append( self.mu.grid.Ncells() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                # without ghost-handling:
                # left=lambda proc: self.mu.cell_l2g[proc]
                # With ghost-handling:
                def face_left(proc):
                    c_map=self.mu.cell_l2g[proc]
                    sel=c_map>=0
                    return c_map[sel]
                def face_right(proc):
                    c_map=self.mu.cell_l2g[proc]
                    sel=c_map>=0
                    return sel
                
                left=face_left
                right=face_right
            elif self.mu.rev_meta.get(dim,None)=='edge_dimension':
                shape.append( self.mu.grid.Nedges() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                left=lambda proc: self.mu.edge_l2g[proc]
            elif self.mu.rev_meta.get(dim,None)=='node_dimension':
                shape.append( self.mu.grid.Nnodes() )
                assert l2g is None,"Can only concatenate on one parallel dimension"
                left=lambda proc: self.mu.node_l2g[proc]
            else:
                # Check for differing lengths across sub vars
                sv_lengths =[sv.shape[dim_i] for sv in self.sub_vars]
                max_length = max(sv_lengths)
                if max_length != min(sv_lengths):
                    fill_value = self.infer_fill_value(sv0)
                    log.info("Ragged shapes for %s, filling with %s"%(sv0.name,fill_value))
                
                shape.append( max_length )

                # When we assumed that shape was the same across procs
                # left=lambda proc: slice(None)
                # Our new, less innocent and naiive, understanding of the world:
                # Also note that variable binding is important here!
                # sv_lengths is passed in so that as sv_lengths changes in later
                # loop iterations, the lambda will hold onto the original value.
                left=lambda proc, lengths=sv_lengths: slice(0,lengths[proc])
                
            right_idx.append( right ) # no subsetting on rhs for now.
            left_idx.append( left )
            
        return shape,left_idx,right_idx
    @property
    def shape(self):
        return self.shape_and_indexes()[0]
    
    @property
    def values(self):
        """
        Combine subdomain values
        """
        shape,left_idx,right_idx=self.shape_and_indexes()
        
        sv0=self.sub_vars[0] # template sub variable
        
        result=np.full( shape, self.infer_fill_value(sv0), sv0.dtype)
        if result.size==0:
            # empty range. Nothing to fill. Causes issues if we try to attempt
            return result
        
        # Copy subdomains to global:
        
        for proc,sv in enumerate(self.sub_vars):
            # In the future may want to control which subdomain provides
            # a value in ghost cells, by having some values of the mapping
            # negative, and they get filtered out here.

            # Another annoyance here is the possibility that with grid
            # topology some subdomains can have different shapes like
            # max_faces.
            left_slice =tuple( [i(proc) for i in left_idx ])
            right_slice=tuple( [i(proc) for i in right_idx])
            result[left_slice]=sv.values[right_slice]
        return result

    def infer_fill_value(self,sv):
        if 'start_index' in sv.attrs:
            return sv.attrs['start_index'] - 1
        if np.issubdtype(sv.dtype, np.floating):
            return np.nan
        return 0
    
    @property
    def attrs(self):
        return self.sub_vars[0].attrs
    
    def __array__(self):
        """ This lets numpy-expecting functions accept this franken-array
        """
        return self.values

    def __len__(self):
        shape,left_idx,right_idx=self.shape_and_indexes()
        return shape[0]


    def to_dataarray(self): # old name
        return self.compute()
    
    def compute(self):
        """
        Materialize to a non-partitioned DataArray.
        """
        # Would be nice to put coordinates together, too.
        # Have to be careful to drop or merge any coordinates
        # that span a partitioned dimension, though.
        da=xr.DataArray(self.values,dims=self.dims,
                        name=self.sub_vars[0].name,
                        attrs=self.sub_vars[0].attrs)
        return da

    # Incomplete attempt to look like a DataArray in math operations
    # I think xarray is sometimes too greedy in trying to handle operations, though.
    # having some issues with DataArray + MultiVar
    def __add__(self,other):
        return self.compute()+other
    def __radd__(self,other):
        return other+self.compute()
    def __sub__(self,other):
        return self.compute()-other
    def __rsub__(self,other):
        return other-self.compute()
    def __mul__(self,other):
        return self.compute()*other
    def __rmul__(self,other):
        return other*self.compute()
    def __truediv__(self,other):
        return self.compute()/other
    def __rtruediv__(self,other):
        return other/self.compute()
    
        
        
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
    
    def __init__(self,paths,cleanup_dfm='auto',xr_kwargs={},grid=None,
                 match_grid_tol=1e-3,clone_from=None,
                 **grid_kwargs):
        """
        paths: 
            list of paths to netcdf files
            single glob pattern
            list of xr.Dataset

        cleanup_dfm: True: remove extra bits common in DFM output that either
         lead to duplicate edges, or cannot be handled by multi_ugrid. If 'auto'
         then check for 'Deltares' in Conventions.

        grid: instead of building a new grid, match the partitioned datasets
          to an existing grid. Caveat emptor -- be sure the supplied grid is an
          exact match!  match_grid_tol gives the tolerance in matching up nodes.
          note that edges may not preserve orientation, and cells may not preserve
          exact node order.

        clone_from: a MultiUgrid object with exactly matching structure (same number
          of subdomains, identical subdomain grids). In this case just copy the
          mapping information from the existing MultiUgrid, and replace the datasets.        

        ** (grid_kwargs): keyword arguments passed to read_ugrid.
        xr_kwargs: dict of arguments passed to xr.open_dataset.
        """
        if isinstance(paths,str):
            paths=glob.glob(paths)
            # more likely to get datasets in order of processor rank
            # with a sort.
            paths.sort()
        # list of str vs list of xr.Dataset is handled in self.load()
        self.paths=paths # paths might be xr.Datasets!
        self.dss=self.load(**xr_kwargs)

        if clone_from is not None:
            self.grids=clone_from.grids
            self.rev_meta=clone_from.rev_meta
            self.node_l2g=clone_from.node_l2g
            self.edge_l2g=clone_from.edge_l2g
            self.cell_l2g=clone_from.cell_l2g
            self.grid=clone_from.grid
            return
        
        self.grids=[unstructured_grid.UnstructuredGrid.read_ugrid(ds,**grid_kwargs) for ds in self.dss]

        # Build a mapping from dimension to ugrid role -- used by MultiVar to
        # decide how to aggregate
        meta=self.grids[0].nc_meta
        self.rev_meta={meta[k]:k for k in meta} # Reverse that

        if cleanup_dfm=='auto':
            cleanup_dfm=('Deltares' in self.dss[0].attrs.get('Conventions',"")
                          or 'D-Flow FM' in self.dss[0].attrs.get('source',""))
            
        if cleanup_dfm:
            for g in self.grids:
                unstructured_grid.cleanup_dfm_multidomains(g)
                # Also remove extra fields that depend on max_sides but that we
                # don't use.
                # Would be better to either support these, or detect them based on
                # netcdf dimensions
                for f in ['mesh2d_face_x_bnd','mesh2d_face_y_bnd']:
                    if f in g.cells.dtype.names:
                        log.warning("Dropping extra cell field %s to avoid max_sides issues"%f)
                        g.delete_cell_field(f)
                    
            # kludge DFM output (ver. 2021.03) has nNetElem and nFlowElem, which appear to
            # both be for the cell dimension
            for ds in self.dss:
                if ( ('nFlowElem' not in ds.dims) or
                     ('nNetElem' not in ds.dims)):
                    break
                if ds.dims['nFlowElem']!=ds.dims['nNetElem']:
                    log.warning("Expected dimensions nFlowElem and nNetElem to be duplicates, but %d!=%d"%
                                (ds.dims['nFlowElem'],ds.dims['nNetElem']))
                    break
            else:
                self.rev_meta['nFlowElem']='face_dimension'

        self.create_global_grid_and_mapping(grid=grid,match_grid_tol=match_grid_tol)
        # make the dimension mapping available
        self.grid.nc_meta=meta
        
    def load(self,**xr_kwargs):
        if isinstance(self.paths[0],str):
            return [xr.open_dataset(p,**xr_kwargs) for p in self.paths]
        elif isinstance(self.paths[0],xr.Dataset):
            return self.paths
        else:
            raise Exception("Unsure whether paths has Datasets or paths (%s)"%self.paths[0])
    
    def reload(self):
        """
        Close and reopen individual datasets, in case unlimited dimensions (i.e. time) have
        been extended.  Does not recompute the grid.
        """
        self.close()
        self.dss=self.load()

    def close(self):
        for ds in self.dss:
            ds.close() 

    def create_global_grid_and_mapping(self,grid=None,match_grid_tol=1e-3):
        """
        grid: if given, a pre-existing global grid. subdomains will be matched
          exactly to this grid.
        """
        if grid is None:
            generate=True
        else:
            self.grid=grid.copy()
            # initialize to 0 for unset.
            self.grid.add_cell_field( 'ghostness', np.zeros(self.grid.Ncells(), np.int32),
                                      on_exists='overwrite')
            # initialize to -1 for unset.
            self.grid.add_cell_field( 'proc', np.zeros( self.grid.Ncells(), np.int32)-1,
                                      on_exists='overwrite')
            generate=False
            
        self.node_l2g=[]
        self.edge_l2g=[]
        self.cell_l2g=[]

        # initialize 
        for gnum,g in enumerate(self.grids):
            # ghost cells:
            e2c=g.edge_to_cells()
            bnd_edge=np.nonzero( e2c.min(axis=1) < 0)[0]
            bnd_cell=e2c[bnd_edge,:].max(axis=1)
            # boundary and potential ghost cells get -1, else 0.
            # not quite there, since there are boundary cells that
            # are ghost and non-ghost. better to have a count of
            # neighbors for each cell?
            # Revisit.  For now, ghostness is 0 for unset, and more
            # positive the more likely cell is to be real
            ghostness=100*np.ones(g.Ncells(), np.int32)
            ghostness[bnd_cell] -= 1

            if generate:
                if gnum==0:
                    self.grid=self.grids[0].copy()
                    n_map=np.arange(self.grid.Nnodes())
                    j_map=np.arange(self.grid.Nedges())
                    c_map=np.arange(self.grid.Ncells())
                    self.grid.add_cell_field( 'ghostness', ghostness )
                    self.grid.add_cell_field( 'proc', np.zeros( g.Ncells(), np.int32) )
                else:
                    n_map,j_map,c_map = self.grid.add_grid(g,merge_nodes='auto',
                                                           tol=self.merge_tol)
                    # c_map will be g.Ncells(), mapping to global idx.
                    # either ghostness not set, or the existing value is ghostier
                    # than new value:
                    sel_proc=self.grid.cells['ghostness'][c_map] < ghostness
                    c_map=np.where(sel_proc, c_map, -1)
                    # just the selected cells get this proc
                    self.grid.cells['proc'][c_map[sel_proc]]=gnum
                    self.grid.cells['ghostness'][c_map[sel_proc]]=ghostness[sel_proc]
            else:
                # i.e. g.nodes['x'][n] == self.grid.nodes['x'][node_map[n]]
                n_map,j_map,c_map = g.match_to_grid(self.grid,tol=match_grid_tol)
                assert np.all(n_map>=0)
                assert np.all(j_map>=0)
                assert np.all(c_map>=0)
                # Would be nice to factor this out
                sel_proc=self.grid.cells['ghostness'][c_map] < ghostness
                c_map=np.where(sel_proc, c_map, -1)
                self.grid.cells['ghostness'][c_map] = ghostness
                self.grid.cells['proc'][c_map]=gnum

            self.node_l2g.append(n_map)
            self.edge_l2g.append(j_map)
            self.cell_l2g.append(c_map)

    # TODO: likely abstract out commonality here
    _cell_g2l=None
    @property
    def cell_g2l(self):
        if self._cell_g2l is None:
            cell_g2l=np.zeros((self.grid.Ncells(),2),np.int32)
            for proc,l2g in enumerate(self.cell_l2g):
                valid=l2g>=0
                cell_g2l[l2g[valid],0]=proc
                cell_g2l[l2g[valid],1]=np.arange(len(l2g))[valid]
            self._cell_g2l=cell_g2l
        return self._cell_g2l

    _node_g2l=None
    @property
    def node_g2l(self):
        if self._node_g2l is None:
            node_g2l=np.zeros((self.grid.Nnodes(),2),np.int32)
            for proc,l2g in enumerate(self.node_l2g):
                valid=l2g>=0
                node_g2l[l2g[valid],0]=proc
                node_g2l[l2g[valid],1]=np.arange(len(l2g))[valid]
            self._node_g2l=node_g2l
        return self._node_g2l

    _edge_g2l=None
    @property
    def edge_g2l(self):
        if self._edge_g2l is None:
            edge_g2l=np.zeros((self.grid.Nedges(),2),np.int32)
            for proc,l2g in enumerate(self.edge_l2g):
                valid=l2g>=0
                edge_g2l[l2g[valid],0]=proc
                edge_g2l[l2g[valid],1]=np.arange(len(l2g))[valid]
            self._edge_g2l=edge_g2l
        return self._edge_g2l

    def __iter__(self):
        """ iterate over variable names
        """
        return self.dss[0].__iter__()
    
    def __getitem__(self,k):
        """
        Returns a proxy object (MultiVar) - delaying the partition translation until .values is called.
        note that k may be either a single var name in which case the result
        would be a DataArray, or a sequence of var names in which case the result
        would be a Dataset. Test for validity pro-actively to avoid delayed error.

        Currently this proxies all variables. There is some subtlety to what happens for 
        non-partitioned variables. As it stands, they are proxied, and an eventual call
        to .values will actually read the values from all subdomains and overwrite the
        target data repeatedly. Effectively then a non-partitioned variable gets the values 
        of the last subdomain. This could be slightly more performant by only reading
        the first subdomain, or more conservative by checking for equivalence across
        subdomains.
        """
        valid=False
        varnames=list(self.dss[0].variables.keys())
        if k in varnames:
            valid=True
        elif k in self.dims:
            if k in self.rev_meta:
                meta=self.rev_meta[k]
                if meta=='face_dimension':
                    return np.arange(self.cell_g2l.shape[0])
                elif meta=='edge_dimension':
                    return np.arange(self.edge_g2l.shape[0])
                elif meta=='node_dimension':
                    return np.arange(self.node_g2l.shape[0])
                else:
                    raise KeyError("Not ready for on-the-fly coordinate for partitioned dimension...")
            else:
                return np.arange(self.dss[0].dims[k])
            
        elif isinstance(k,list):
            valid=np.all( [kk in varnames for kk in k] )
            
        if valid:
            # Is this partitioned?
            partitioned=False
            for d in self.dss[0][k].dims:
                if d in self.rev_meta:
                    partitioned=True
                    break
            if partitioned:
                return MultiVar(self,
                                [ds[k] for ds in self.dss])
            else:
                # Should all be the same, so use the first domain
                return self.dss[0][k]
        else:
            raise KeyError("%s is not an existing variable"%k)

    def __setitem__(self,k,v):
        dims,values=v
        for dim in dims:
            if dim in self.rev_meta:
                raise Exception("Assigning to partitioned dimension is not supported")
        # might be overkill to assign to all, but otherwise we get into a situation where
        # a coordinate that was created through this route would not be available in some
        # subdomains.
        for ds in self.dss:
            ds[k]=v
    
    def __getattr__(self,k):
        # Two broad cases here
        #  attempting to get a variable
        #     => delegate to getitem
        #  attempting some operation, that we probably don't know how to complete
        try: 
            return self.__getitem__(k)
        except KeyError:
            raise AttributeError("%s is not an existing variable or known method"%k)

    def __str__(self):
        return "MultiFile Layer on top of %s"%str(self.dss[0])
    def __repr__(self):
        return str(self)

    @property
    def dims(self):
        return self.dss[0].dims

    def __setstate__(self,state):
        for k in state:
            setattr(self,k,state[k])

    # These should give the correct names, but won't have the correct
    # dimensions for partitioned variables.
    @property
    def data_vars(self):
        return self.dss[0].data_vars

    @property
    def variables(self):
        return self.dss[0].variables

    def isel(self,**kwargs):
        """
        Partial handling of subselection at the MultiUgrid level. Subsetting non-partitioned
        dimensions is batched to each of the underlying Datasets. Subsetting on a single
        partitioned dimension is handled by materializing the selection on each
        data variable. This currently forces the creation of a vanilla Dataset, and
        all partitioned variables that are not selected will be dropped.
        """
        
        part_kwargs={}
        nonpart_kwargs={}

        for key in kwargs:
            val=kwargs[key]
            if key not in self.rev_meta:
                nonpart_kwargs[key]=val
            else:
                part_kwargs[key]=val

        # Apply the nonpartitioned selections:                                                                                                                   
        sub=copy.copy(self)
        sub.dss=[ds.isel(**nonpart_kwargs) for ds in self.dss]

        if len(part_kwargs)==0:
            return sub

        # Not able to support a mix of aggregated dimensions and
        # non-aggregated dimensions.
        # So at this point if you isel on a partitioned dimension, the
        # result is a simple Dataset, and any variables related to a partitioned
        # dimension that wasn't isel'd is dropped.
        result=xr.Dataset()

        for dv in sub.data_vars:
            # if this variable has no partitioned dimensions, grab a
            # copy from the first subdomain.
            part_dims=[d for d in sub[dv].dims if d in self.rev_meta]
            if len(part_dims)==0:
                result[dv]=sub.dss[0][dv]
            else:
                # if the variable has partitioned dimensions that are not part of
                # the isel call, drop it.  We're not ready to have a mix of partitioned
                # and unpartitioned values.
                # The more complete way to do this is to:
                #   1. check if any partitioned dimensions are not selected.
                #   2. if all partitioned dimensions are selected, then the code below is
                #      fine, and we get a vanilla Dataset
                #   3. if partitioned dimensions remain, the newly non-partitioned dimension
                #      are dropped from self.rev_meta, and the newly non-partitioned variables
                #      assigned to the first sub-dataset.
                #      For this to work, then MultiVar.values needs to only pull from the
                #      first dataset when no dimensions are partitioned.
                
                free_part_dims=[d for d in part_dims if d not in part_kwargs]
                if free_part_dims:
                    log.info("Dropping %s because it has unselected partitioned dimensions"%dv)
                    continue
                # narrow the partitioned dimensions to those that actually
                # exist for this variable.
                v_part_kwargs={d:part_kwargs[d] for d in part_kwargs if d in sub[dv].dims}
                result_var=sub[dv].isel(**v_part_kwargs)
                # piece together new dimensions.
                result[dv]=result_var.dims,result_var.values

        return result

    def sel(self,**kwargs):
        subset=copy.copy(self)
        subset.dss=[ds.sel(**kwargs) for ds in self.dss]
        return subset
    def drop(self,*args,**kwargs):
        subset=copy.copy(self)
        subset.dss=[ds.drop(*args,**kwargs) for ds in self.dss]
        return subset

    
    def compute(self,vars=None):
        if vars is None:
            vars = self.data_vars
        ds=xr.Dataset()
        for v in vars:
            ds[v] = self[v].compute()
        return ds
    
