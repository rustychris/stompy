"""
Take a netcdf file in UGRID-ish format, with multiple subdomains, and write out
one with a subdomain.  Requires that a common nodes array be used in order to
unambiguously discern global edge/face relationships.

1/21/2016: seems to have issues with corrupting depth data.  Tested 
on a transcribed-global.nc file from boffinator, grabbing the full
grid and plotting cell scalar of depth shows mismatch.  Maybe not 
properly reordered?
"""
from __future__ import print_function
import six
import os,sys
import ugrid
import numpy as np
import netCDF4
import re

class UgridTranscriber(object):
    """ common utility methods useful when transcribing ugrid-based
    netcdf files
    """
    var_kwargs = dict(zlib=True,complevel=2)
    var_kwargs_re = None
    time_var = 'time'
    layers_name = "nMeshGlobal_layers"

    def __init__(self,nc_in_fn):
        self.var_kwargs_re = {} # regular expression -> dict of variable create options
        self.topology_variables={}

        self.nc_in_fn=nc_in_fn
        
        self.nc_in = netCDF4.Dataset(self.nc_in_fn)
        self.ug = ugrid.Ugrid(self.nc_in)
        self.submeshes = self.ug.mesh_names()

    def var_options(self,varname):
        for patt,opts in self.var_kwargs_re.iteritems():
            if re.match(patt,varname):
                print("%s => options are %s"%(varname,opts))
                return opts
        return self.var_kwargs
        
    def copy_nodes(self,meshvar):
        """ While nodes are global, still have to set the node names as
        an attribute on the mesh, so mesh gets passed in.
        """
        # check that a single points array is used, and copy it over:
        node_coordinates = None
        for mesh in self.submeshes:
            mesh_var = self.nc_in.variables[mesh]
            if node_coordinates:
                if node_coordinates != mesh_var.node_coordinates:
                    raise Exception("Can only deal with a single global node_coordinates variable")
            else:
                node_coordinates = mesh_var.node_coordinates

        # Copy over the node_coordinates data:
        node_x_coordinates, node_y_coordinates = node_coordinates.split()
        meshvar.node_coordinates = node_coordinates

        # check to see whether the nodes have already been copied:
        if node_x_coordinates not in self.topology_variables:
            self.copy_nc_dimension('nMeshGlobal_node')
            self.copy_nc_variable(node_x_coordinates)
            self.copy_nc_variable(node_y_coordinates)
            self.topology_variables[node_x_coordinates]=1
            self.topology_variables[node_y_coordinates]=1
        
    def copy_non_mesh_variables(self):
        self.copy_nc_variable(self.time_var)

    def copy_nc_dimension(self,name):
        dim_orig = self.nc_in.dimensions[name]
        if dim_orig.isunlimited():
            length = None
        else:
            length = len(dim_orig)
        self.nc_out.createDimension(name,length)

    def copy_nc_variable(self,name,new_name=None):
        new_name = new_name or name
        
        var_orig = self.nc_in.variables[name]

        for dim_name in var_orig.dimensions:
            if not self.nc_out.dimensions.has_key(dim_name):
                print("Copying dimension %s on-demand"%dim_name)
                self.copy_nc_dimension(dim_name)

        var_new = self.nc_out.createVariable(new_name,
                                             var_orig.dtype,
                                             var_orig.dimensions,
                                             **self.var_kwargs)

        for ncattr in var_orig.ncattrs():
            setattr(var_new,ncattr, getattr(var_orig,ncattr))

        var_new[...] = var_orig[...]
    def is_topology_variable(self,name):
        return (name in self.topology_variables) or ( ("Mesh_"+name) in self.topology_variables)

    def mesh_face_dim(self,mesh):
        return self.ug.mesh_face_dim(mesh)
    def mesh_edge_dim(self,mesh):
        return self.ug.mesh_edge_dim(mesh)


class MergeUgridSubgrids(UgridTranscriber):
    prefix_data='Mesh_' # should a Mesh_ prefix be kept for data
    prefix_topo='Mesh_'
    
    mode='w'
    def __init__(self,nc_in_fn,nc_out_fn,include_re=None,**kwargs):
        super(MergeUgridSubgrids,self).__init__(nc_in_fn=nc_in_fn)
        
        self.nc_out_fn = nc_out_fn
        self.include_re = include_re
        self.__dict__.update(kwargs)

        self.load_datasets()
        self.copy_non_mesh_variables()
        self.copy_topology()
        self.copy_edge_data()
        self.copy_face_data()
        self.nc_out.close()

    def load_datasets(self):
        #self.nc_in = netCDF4.Dataset(self.nc_in_fn)
        #ug = ugrid.Ugrid(self.nc_in)
        #self.submeshes = ug.mesh_names()
        self.nc_out = netCDF4.Dataset(self.nc_out_fn,mode=self.mode)

    def copy_topology(self):
        self.new_mesh = new_mesh = self.nc_out.createVariable('Mesh',int)
        new_mesh.cf_role = 'mesh_topology'
        new_mesh.long_name = "Topology data of 2D unstructured mesh"
        new_mesh.dimension = 2

        # keep a list of topology variables, to distinguish from geometry
        # later on
        
        self.copy_nodes(new_mesh)
        self.copy_edges()
        self.copy_faces()
        self.copy_vertical()

    def copy_edges(self):
        self.select_edges_from_submesh = select_edges_from_submesh = [None] * len(self.submeshes)
        self.submesh_edge_ranges = submesh_edge_ranges = np.zeros( (len(self.submeshes),2), np.int32)

        self.copy_nc_dimension('Two')

        # hash map of node duples+, in ascending order, to new edge indexes
        new_edge_idx = 0 # number of unique faces so far.
        new_edge_map = {}
        max_edges_per_submesh = 0

        for meshi,mesh in enumerate(self.submeshes):
            print("Indexing edges from ",mesh)
            submesh_edge_ranges[meshi,0] = new_edge_idx
            mesh_var = self.nc_in.variables[mesh]
            fn_con = self.nc_in.variables[ mesh_var.edge_node_connectivity ][...]
            fn_con.sort(axis=1)
            select_edges_from_submesh[meshi] = selected = np.zeros( len(fn_con), np.bool_ )

            tupes = [tuple(nnn) for nnn in fn_con]
            max_edges_per_submesh = max(max_edges_per_submesh,len(tupes))

            for e,k in enumerate(tupes):
                if k not in new_edge_map:
                    new_edge_map[k] = new_edge_idx
                    new_edge_idx += 1
                    selected[e] = True
                else:
                    pass
            submesh_edge_ranges[meshi,1] = new_edge_idx

        self.nc_out.createDimension(self.prefix_topo+'nedge',new_edge_idx)

        # And repeat, this time copying topology
        varname=self.prefix_topo+'edge_nodes'
        edge_nodes = self.nc_out.createVariable(varname,int32,
                                                [self.prefix_topo+'nedge','Two'],
                                                **self.var_options(varname))
        self.new_mesh.edge_node_connectivity = self.prefix_topo+'edge_nodes'
        self.topology_variables[self.prefix_topo+'edge_nodes']=1
        
        for meshi,mesh in enumerate(self.submeshes):
            print("Copying edge topology from ",mesh)
            mesh_var = self.nc_in.variables[mesh]
            fn_con = self.nc_in.variables[ mesh_var.edge_node_connectivity ][...]

            # more efficient to bitmap the input, but write to a slice
            edge_nodes[submesh_edge_ranges[meshi,0]:submesh_edge_ranges[meshi,1],:] = fn_con[select_edges_from_submesh[meshi],:]
        
    def copy_faces(self):
        # records which elements should actually be copied over
        # note that this doesn't respect suntans version of what is a ghost cell -
        # it assumes that ghost cells have the same data as the real cell, and it
        # just takes the first cell it finds
        self.select_from_submesh = select_from_submesh = [None] * len(self.submeshes) 
        self.submesh_face_ranges = submesh_face_ranges = np.zeros( (len(self.submeshes),2), np.int32)

        # hash map of node triples+, in ascending order, to new face indexes
        new_face_idx = 0 # number of unique faces so far.
        new_face_map = {}
        max_faces_per_submesh = 0

        for meshi,mesh in enumerate(self.submeshes):
            print("Indexing faces from ",mesh)
            submesh_face_ranges[meshi,0] = new_face_idx
            mesh_var = self.nc_in.variables[mesh]
            fn_con = self.nc_in.variables[ mesh_var.face_node_connectivity ][...]
            fn_con.sort(axis=1)
            select_from_submesh[meshi] = selected = np.zeros( len(fn_con), np.bool_ )

            tupes = [tuple(nnn) for nnn in fn_con]
            max_faces_per_submesh = max(max_faces_per_submesh,len(tupes))

            for f,k in enumerate(tupes):
                if k not in new_face_map:
                    new_face_map[k] = new_face_idx
                    new_face_idx += 1
                    selected[f] = True
                else:
                    pass
            submesh_face_ranges[meshi,1] = new_face_idx

        self.nc_out.createDimension(self.prefix_topo+'nmax_face_nodes',fn_con.shape[1])
        self.nc_out.createDimension(self.prefix_topo+'nface',new_face_idx)

        # And repeat, this time copying topology
        varname=self.prefix_topo+'face_nodes'
        face_nodes = self.nc_out.createVariable(varname,np.int32,
                                                [self.prefix_topo+'nface',
                                                 self.prefix_topo+'nmax_face_nodes'],
                                                **self.var_options(varname) )
        self.new_mesh.face_node_connectivity = self.prefix_topo+'face_nodes'
        self.topology_variables[self.prefix_topo+'face_nodes']=1
        
        for meshi,mesh in enumerate(self.submeshes):
            print("Copying face topology from ",mesh)
            mesh_var = self.nc_in.variables[mesh]
            fn_con = self.nc_in.variables[ mesh_var.face_node_connectivity ][...]

            # more efficient to bitmap the input, but write to a slice
            face_nodes[submesh_face_ranges[meshi,0]:submesh_face_ranges[meshi,1],:] = fn_con[select_from_submesh[meshi],:]

    def copy_vertical(self):
        """ pretty rough here - this isn't really encoded in the netcdf files at this
        point.  at this point it just copies from the first submesh <submesh>_layers
        to the new global mesh

        also - the dimensions have to be renamed...
        """
        self.copy_nc_variable(self.layers_name)

        # if it has bounds - copy those, too
        layers = self.nc_in.variables[self.layers_name]
        if 'bounds' in layers.ncattrs():
            self.copy_nc_variable(layers.bounds)

    def copy_face_data(self):
        # Scan for cell-based variables in the first mesh -
        # accepts variables with the first dimension of edge index, and somewhere
        # it has time
        mesh0 = self.submeshes[0]

        face_dim=self.mesh_face_dim(mesh0)

        for varname in self.nc_in.variables.keys():
            v = self.nc_in.variables[varname]
            if not ('mesh' in v.ncattrs() and v.mesh == mesh0):
                continue
            print("copy_face_data: variable is on this mesh...")
            
            if face_dim in v.dimensions: #and self.time_var in v.dimensions:
                if self.include_re is not None:
                    if re.match(self.include_re,varname) is None:
                        print("skipping %s"%varname)
                        continue
                
                name = varname.replace(mesh0+'_','')
                if self.is_topology_variable(name):
                    continue
                print("Copying face-centered variable %s"%name)
                self.copy_nc_submesh_variable(name)

    def copy_edge_data(self):
        # Scan for edge-based variables in the first mesh -
        # accepts variables with the first dimension of edge index, and somewhere
        # it has time
        mesh0 = self.submeshes[0]
        edge_dim=self.mesh_edge_dim(mesh0)

        for varname in self.nc_in.variables.keys():
            if not varname.startswith(mesh0):
                continue
            v = self.nc_in.variables[varname]
            
            if edge_dim in v.dimensions: # and self.time_var in v.dimensions:
                if self.include_re is not None:
                    if re.match(self.include_re,varname) is None:
                        print("skipping %s"%varname)
                        continue
                
                name = varname.replace(mesh0+'_','')
                if self.is_topology_variable(name):
                    continue
                print("Copying edge-centered variable %s"%name)
                self.copy_nc_submesh_variable(name)
        
    def copy_nc_submesh_variable(self,name):
        """ name: string name of the variable to be copied, assumed
        to be named '<submesh>_<name>' on the input netcdf.
        will be written out to self.prefix_data + <name>
        """
        submesh = self.submeshes[0]
        submesh_var = self.nc_in.variables[submesh]
        var_orig_sub = self.nc_in.variables[submesh +"_"+name]

        new_dims = []
        for dim in var_orig_sub.dimensions:
            new_dim = self.prefix_topo + dim.replace(submesh+'_','')
            new_dims.append( new_dim )
            if new_dim not in self.nc_out.dimensions:
                # pray that it's not an unlimited or mesh-varying dimension
                print("Creating new dimension %s"%new_dim)
                self.nc_out.createDimension(new_dim,len(self.nc_in.dimensions[dim]))

        varname=self.prefix_data+name
        my_var_kwargs = dict(self.var_options(varname))
        if '_FillValue' in submesh_var.ncattrs():
            # this has to be done at createVariable time
            my_var_kwargs['fill_value'] = submesh_var._FillValue

        var_new = self.nc_out.createVariable(varname,var_orig_sub.dtype,
                                             new_dims,**my_var_kwargs)

        for ncattr in var_orig_sub.ncattrs():
            if ncattr == '_FillValue':
                pass
            elif ncattr == 'mesh':
                setattr(var_new,ncattr,'Mesh')
            elif ncattr == 'coordinates':
                old_val=getattr(var_orig_sub,ncattr)
                new_vars=[ self.prefix_topo+old_var.replace(submesh+'_','')
                           for old_var in old_val.split()]
                new_val=" ".join(new_vars)
                setattr(var_new,ncattr, new_val)
            else:
                setattr(var_new,ncattr, getattr(var_orig_sub,ncattr))

        for submeshi,submesh in enumerate(self.submeshes):
            sys.stdout.write(" %s"%submesh )
            sys.stdout.flush()
            var_orig_sub = self.nc_in.variables[submesh +"_"+name]

            slices = []
            for dimi,dim in enumerate(new_dims):
                if dim == self.prefix_topo + 'nface':
                    # nsub_faces = var_orig_sub.shape[dimi] # unused??
                    slices.append( slice(self.submesh_face_ranges[submeshi,0],
                                         self.submesh_face_ranges[submeshi,1]) )
                    data_select = self.select_from_submesh[submeshi]
                elif dim == self.prefix_topo+'nedge':
                    slices.append( slice(self.submesh_edge_ranges[submeshi,0],
                                         self.submesh_edge_ranges[submeshi,1]) )
                    data_select = self.select_edges_from_submesh[submeshi]
                else:
                    slices.append( slice(None) )

            # apparently inefficient to have netcdf handle the bitmap select,
            # especially since it is typically very dense.
            data = var_orig_sub[...]
            var_new[tuple(slices)] = data[data_select]
        print()

class VerticalAverage(UgridTranscriber):
    """ Copy a ugrid netcdf, but apply a selective vertical average
    on all depth-varying quantities
    """
    avg_prefix="surface_"
    avg_only=False # only populate the actual averaged quantities, no mesh/topo/etc.
    mode="w" # open mode for the netCDF output dataset
    
    def __init__(self,nc_in_fn,nc_out_fn,ztop=None,zbottom=None,dz=None,**kwargs):
        super(VerticalAverage,self).__init__(nc_in_fn=nc_in_fn)

        self.ztop=ztop
        self.zbottom=zbottom
        self.dz=dz
        
        self.nc_out_fn = nc_out_fn
        self.__dict__.update(kwargs)

        self.load_datasets()
        if not self.avg_only:
            self.copy_non_mesh_variables() # primarily time

        self.meshvars={} # map mesh names to netcdf variables
        
        for mesh in self.submeshes:
            if not self.avg_only:
                self.copy_topology(mesh)
            self.copy_edge_data(mesh)
            self.copy_face_data(mesh)
            
        self.nc_out.close()
        
    def load_datasets(self):
        # this should actually happen automatically by netCDF4
        # if os.path.exists(self.nc_out_fn):
        #     os.unlink(self.nc_out_fn)
        self.nc_out = netCDF4.Dataset(self.nc_out_fn,mode=self.mode)
        
    def copy_topology(self,mesh):
        self.meshvars[mesh] = new_mesh = self.nc_out.createVariable(mesh,int)
        new_mesh.cf_role = 'mesh_topology'
        new_mesh.long_name = "Topology data of 2D unstructured mesh"
        new_mesh.dimension = 2
        self.topology_variables[mesh]=1
        
        self.copy_nodes(new_mesh)
        self.copy_edges(mesh)
        self.copy_faces(mesh)
        self.copy_vertical()

    def copy_edges(self,mesh):
        srcmesh=self.nc_in.variables[mesh]
        dstmesh=self.meshvars[mesh]
        
        # copies dimensions on-demand
        self.copy_nc_variable(srcmesh.edge_node_connectivity)
        dstmesh.edge_node_connectivity=srcmesh.edge_node_connectivity
        self.topology_variables[srcmesh.edge_node_connectivity]=1
        
    def copy_faces(self,mesh):
        dstmesh=self.meshvars[mesh]
        srcmesh=self.nc_in.variables[mesh]
        # copies dimensions on-demand
        self.copy_nc_variable(srcmesh.face_node_connectivity)
        dstmesh.face_node_connectivity=srcmesh.face_node_connectivity
        self.topology_variables[srcmesh.face_node_connectivity]=1
        
    def copy_vertical(self):
        # check to see if it has already been copied
        if self.layers_name not in self.topology_variables:
            self.copy_nc_variable(self.layers_name)
            self.topology_variables[self.layers_name]=1
        
            # if it has bounds - copy those, too
            layers = self.nc_in.variables[self.layers_name]
            if 'bounds' in layers.ncattrs():
                self.copy_nc_variable(layers.bounds)
                self.topology_variables[layers.bounds]=1

    def copy_edge_data(self,mesh):
        print("No edge data yet!")

    def copy_face_data(self,mesh):
        # Scan for cell-based variables in the first mesh -
        # accepts variables with the first dimension of edge index, and somewhere
        # it has time
        face_dim=self.mesh_face_dim(mesh)
        vert_dim = self.nc_in.variables[self.layers_name].dimensions[0]
        
        for varname in self.nc_in.variables.keys():
            v = self.nc_in.variables[varname]
            
            if not ('mesh' in v.ncattrs() and v.mesh==mesh):
                continue
            name = varname.replace(mesh+'_','')
            if self.is_topology_variable(name):
                continue

            if face_dim in v.dimensions:
                if self.include_re is not None:
                    if re.match(self.include_re,varname) is None:
                        print("skipping %s"%varname)
                        continue
                    
                if vert_dim in v.dimensions:
                    if self.time_var in v.dimensions:
                        self.copy_averaged_face_variable(varname,self.avg_prefix+varname)
                    else:
                        print("WARNING: %s has vertical, but no time - don't know how to average that"%varname)
                else:
                    if not self.avg_only:
                        print("Copying face-centered variable %s"%name)
                        self.copy_nc_variable(varname)
                    
    def copy_averaged_face_variable(self,varname,new_name=None):
        """ assumes that varname has both self.layers_name and self.time_var
        as dimensions (well, their respective dimensions)
        """
        new_name = new_name or varname
        print("Averaging %s => %s"%(varname,new_name))
        
        var_orig = self.nc_in.variables[varname]

        avg_dims=[]
        
        for dim_name in var_orig.dimensions:
            if dim_name == self.layers_name:
                continue
            avg_dims.append(dim_name)
            if not self.nc_out.dimensions.has_key(dim_name):
                print("Copying dimension %s on-demand"%dim_name)
                self.copy_nc_dimension(dim_name)

        var_new = self.nc_out.createVariable(new_name,
                                             var_orig.dtype,
                                             avg_dims,
                                             **self.var_options(new_name))

        for ncattr in var_orig.ncattrs():
            setattr(var_new,ncattr, getattr(var_orig,ncattr))

        if 'mesh' in var_orig.ncattrs():
            mesh=var_orig.mesh
        else:
            mesh=varname.split('_')[0]
            print("bad UGRID - missing mesh attribute.  will guess %s"%mesh)
            
        time_var=self.nc_in.variables[self.time_var]

        print("#"*len(time_var))
        for step in range(len(time_var)):
            sys.stdout.write('.') ; sys.stdout.flush()
            weights=self.ug.vertical_averaging_weights(step,ztop=self.ztop,zbottom=self.zbottom,dz=self.dz,
                                                       mesh_name=mesh)
            # gets tricky here - for now, stick with the most obvious layout, rather than
            # trying to dynamically find out the right indexing scheme.
            # not very flexible, but will work for files generated by this code.
            var_one_step=var_orig[:,:,step]
            var_new[:,step] = np.sum(var_one_step*weights,axis=1)
        print()

import pdb, traceback, sys

if __name__ == '__main__':
    fn,fn_out = sys.argv[1:3]

    if len(sys.argv)>3:
        include_re = sys.argv[3]
    else:
        include_re = None
    
    try:
        MergeUgridSubgrids(fn,fn_out,include_re=include_re)
    except:
        typ,value,tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
# where should depth be copied?
#   copy_faces, or copy_face_data?
#  copy_face_data looks for things index by [face,...,time,...]
#     so currently skips depth
#   if that were to ignore the absence of a time dimension, then it
#     would have to be smarter about also ignoring topology
#  depth is most like face_x, face_y, but those are also skipped.

# so how to distinguish face_x,face_y, and depth from topology?
#  topology always has a second dimension, and is integer valued,
#   has a cf_role attribute

