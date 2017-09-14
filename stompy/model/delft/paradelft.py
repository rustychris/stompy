"""
Tools to read Delft output into paraview.

Note that this probably needs to be run from within paraview or pvpython.
"""

from __future__ import print_function

import numpy as np
import time

from ...io import qnc
from . import dfm_grid

try:
    import paraview
    import paraview.numpy_support as pnp
    import vtk
except ImportError:
    print("vtk/paraview import failed. Paradelft not available")

# this works, but it's super slow (25.92s to load one timestep)
# starting point: 25.92s
# caching the netcdf and trigrid instances: 25.82s.
# note that running in paraview is a bit slower than running in local python.
# 23.20 / 25.91s is spent building cells.
# call cell_to_nodes once per 2D cell, and cache the type ids...
# 5.29 / 8.04s is spent building cells.

import memoize

#@memoize.memoize(lru=1)
def load_data(ncmap_fn):
    ncmap=qnc.QDataset(ncmap_fn)
    g=dfm_grid.DFMGrid(ncmap)
    return ncmap,g

wedge_type     =vtk.vtkWedge().GetCellType()    
hexahedron_type=vtk.vtkHexahedron().GetCellType()

def load_map(output=None,ncmap_fn=None,tidx=None):
    output=output or vtk.vtkUnstructuredGrid()

    if ncmap_fn is None:
        ncmap_fn="/home/rusty/models/delft/nms/nms_00_project.dsproj_data/nms_hydro_00_output/DFM_OUTPUT_nms_hydro_00/nms_hydro_00_map.nc"

    ncmap,g = load_data(ncmap_fn)

    if tidx is None:
        tidx=len(ncmap.time)-1 # force to last time step

    # Still have about 1.6s in all of this stuff
    Nk=len(ncmap.dimensions['laydim'])

    # 3-D locations of nodes
    Zbed=ncmap.NetNode_z[:]
    Zsurf_cells=ncmap.s1[tidx,:] # unfortunately this is at cell centers, not nodes.

    Zsurf=g.interp_cell_to_node(Zsurf_cells)
    Znode=np.zeros( (g.Nnodes(),Nk+1), 'f8' )

    Zbed=np.minimum(Zsurf-0.2,Zbed)

    alpha=np.linspace(0,1,Nk+1)[None,:]

    # this is where the positive up stuff is assumed
    Znode[:,:] = (1-alpha)*Zbed[:,None] + alpha*Zsurf[:,None]

    # map nodes to linear index
    node_idx=np.arange(Znode.size).reshape(Znode.shape)

    # assemble the 3-d points array and write to a vtk file:
    all_z=Znode.ravel()
    all_x=np.repeat(g.nodes['x'][:,0],Nk+1)
    all_y=np.repeat(g.nodes['x'][:,1],Nk+1)

    max_Ncells_3d=g.Ncells() * Nk # upper bound

    pts=vtk.vtkPoints()
    xyz=np.ascontiguousarray( np.array([all_x,all_y,all_z]).T )
    arr=pnp.numpy_to_vtk(xyz,deep=1)
    pts.SetData(arr) # do we have to specify 'Points' here?

    output.SetPoints(pts)
    output.Allocate(max_Ncells_3d, 1000)

    t=time.time()

    # this is always the most expensive part.
    # but it could be cached, and just update the points with 
    # different timesteps, or not even worry about changing geometry.
    for c in range(g.Ncells()):
        nodes=g.cell_to_nodes(c)

        if len(nodes)==3:
            for k in range(Nk):
                # assuming that nodes is CCW, then we start with the top layer
                connectivity = [ node_idx[nodes[0],k],
                                 node_idx[nodes[1],k],
                                 node_idx[nodes[2],k],
                                 node_idx[nodes[0],k+1],
                                 node_idx[nodes[1],k+1],
                                 node_idx[nodes[2],k+1]] 
                cell_type=wedge_type 
                pointIds = vtk.vtkIdList()
                for pointId,conn in enumerate(connectivity):
                    pointIds.InsertId(pointId, conn)
                output.InsertNextCell(cell_type,pointIds)

        elif len(nodes)==4:
            # arrays are small, actually slower to construct connectivity via
            # ndarray
            for k in range(Nk):
                # 5.29s for cell building
                connectivity =  [ node_idx[nodes[0],k+1],
                                  node_idx[nodes[1],k+1],
                                  node_idx[nodes[2],k+1],
                                  node_idx[nodes[3],k+1],
                                  node_idx[nodes[0],k],
                                  node_idx[nodes[1],k],
                                  node_idx[nodes[2],k],
                                  node_idx[nodes[3],k] ] 
                
                cell_type=hexahedron_type
                pointIds = vtk.vtkIdList()
                for pointId,conn in enumerate(connectivity):
                    pointIds.InsertId(pointId, conn)
                output.InsertNextCell(cell_type,pointIds)

        else:
            raise Exception("Only know how to translate 3,4 sided cells")

    print("Elapsed for building cells: %.2f"%(time.time() - t))

    t=time.time()

    ncells3d=output.GetNumberOfCells()

    # add cell velocity:
    cellData=output.GetCellData() # dataSetAttributes

    u=ncmap.variables['ucx'][tidx,:,:].ravel()
    v=ncmap.variables['ucy'][tidx,:,:].ravel()
    w=ncmap.variables['ucz'][tidx,:,:].ravel()

    # 0.01
    print("Reading data: %.2f"%(time.time() - t))

    U=np.ascontiguousarray( np.array([u,v,w]).T )
    arr=pnp.numpy_to_vtk(U,deep=1)
    arr.SetName("cell_velocity")
    cellData.AddArray(arr)

    # scalars
    for nc_name,pv_name in [ ('sa1','salinity'),
                             ('rho','rho') ]:
        scal=ncmap.variables[nc_name][tidx,:,:]
        scal=scal[:,:Nk] # in case it's rho and has an extra layer.
        scal=scal.ravel()

        arr=pnp.numpy_to_vtk( np.ascontiguousarray(scal),deep=1)
        arr.SetName(pv_name)
        cellData.AddArray(arr)

    # 0.33
    print("Elapsed total for cell data: %.2f"%(time.time() - t))


# paste something like:
#  import delft.paradelft as pdf
#  #..sadasdfasdfasdfasdf
#  pdf.load_map(self.GetOutput(),tidx=3)

# into a vtkUnstructuredGrid ProgrammableSource


