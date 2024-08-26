# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:47:01 2024

@author: rusty
"""
import h5py
import numpy as np
from ...grid import unstructured_grid

def ras62d_to_dataset(ras6_out_fn,area_name=None,load_full=False):
    """
    Load RAS6-esque model output into an xarray Dataset resembling ugrid
    output.
    
    ras6_out_fn: path to HDF5 output in RAS6-esque format.
    area_name: 2D area name, defaults to first area and raises exception
    if there are multiple areas.
    load_full: when False, leave HDF data lazily loaded when possible. 
      Otherwise load all data into memory and close the HDF handle.
    """
    ras6_grid=unstructured_grid.UnstructuredGrid.read_ras2d(ras6_out_fn,
                                                            twod_area_name=area_name)

    # Come back and get some results:
    ras6_h5 = h5py.File(ras6_out_fn, 'r')
    area_name=ras6_grid.twod_area_name # in case we're using the default.
    
    ras6_ds=ras6_grid.write_to_xarray()
    if 'cell_z_min' in ras6_grid.cells.dtype.names:
        ras6_ds['bed_elev'] = ('face',),ras6_grid.cells['cell_z_min']
    
    # Is this RAS6 or RAS2025?
    unsteady_base=('/Results/Unsteady/Output/Output Blocks/'
                   'Base Output/Unsteady Time Series')
    try:
        ras6_h5[unsteady_base]
        version="RAS6"
    except KeyError:
        version="RAS2025"

    if version=='RAS6':        
        area_base=(unsteady_base + f'/2D Flow Areas/{area_name}/')
    elif version=='RAS2025':
        unsteady_base = '/Results/Output Blocks/Base Output'
        # maybe "Mesh" is actually the twod_area_name
        area_base = unsteady_base + f'/2D Flow Areas/Mesh/'
    else:
        raise Exception("Bad version: "+version)
        
    n_nonvirtual=ras6_grid.Ncells()

    def LD(v):
        if load_full:
            return np.asarray(v) # I think this forces a load
        else:
            return v
    
    # 2025 differences:
    # /Results/Output Blocks/Base Output/2D Flow Areas/Mesh
    # grid has 1809 faces, 3903 edges.
    # h5 has 2185 faces.
    u_face=None
    for var_name in ras6_h5[area_base]:
        v=ras6_h5[area_base+var_name]
        
        # Appears that names are decoded, but attribute values are
        # left as bytestrings.
        if v.attrs.get('Can Plot','n/a') != b'True':
            continue
        if v.attrs.get('Columns','n/a')==b'Cells':
            ras6_ds[var_name]=('time','face'),np.asanyarray(LD(v))[:,:n_nonvirtual]
        elif v.attrs.get('Columns','n/a')==b'Faces':
            data = LD(v)
            if data.shape[1] > ras6_ds.dims['edge']:
                print(f"Truncating {var_name}")
                data = data[:,:ras6_ds.dims['edge']]
            ras6_ds[var_name]=('time','edge'),data
            if var_name=='Face Velocity':
                u_face=data
        else:
            import pdb
            pdb.set_trace()
    
    # This isn't going to scale -- will need to make the visualization
    # smarter.
    # synthesize cell center velocity -- normals may be off, though.
    M=ras6_grid.interp_perot_matrix()
    #      [3618 x 3903] * [3600 x 3903].T
    if u_face is not None:
        # u_face comes in as HDF dataset -- convert to numpy to get .T
        # That part won't scale very well.
        u_face_a=np.asanyarray(u_face)
        UVcell=M.dot(u_face_a.T).T.reshape( [-1,n_nonvirtual,2] )
        ras6_ds['ucx']=('time','face'), UVcell[:,:,0]
        ras6_ds['ucy']=('time','face'), UVcell[:,:,1]
    
    time_days=ras6_h5[unsteady_base+'/Time']
    ras6_ds['time']=('time',), time_days
    ras6_ds['time'].attrs['units']='d'
    try:
        ras6_ds['time_step']=('time',), ras6_h5[unsteady_base+'/Time Step']
    except KeyError:
        pass
    
    if load_full:
        ras6_h5.close()
    ras6_ds.attrs['grid'] = ras6_grid
    
    return ras6_ds
