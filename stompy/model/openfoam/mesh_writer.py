try:
    import pyvista as pv
except ModuleNotFoundError:
    print("pyvista not installed. mesh_writer will not work")
from . import mesh_ops
import numpy as np

def mesh_cell_to_polydata(cIdx, xyz, face_nodes, face_cells, cell_faces):
    node_count=0
    node_map={}

    points=[]
    faces=[]
    
    for signed_fIdx in cell_faces[cIdx]:
        if signed_fIdx==mesh_ops.NO_FACE: break
        if signed_fIdx<0:
            fIdx=~signed_fIdx
        else:
            fIdx= signed_fIdx
        f_nodes = face_nodes[fIdx]
        f_nodes = f_nodes[f_nodes>=0]
        face=[len(f_nodes)]
        for f_n in f_nodes:
            if f_n not in node_map:
                node_map[f_n]=node_count
                node_count+=1
                points.append( xyz[f_n] )
            face.append(node_map[f_n])
        faces.append(face)
    
    faces = np.concatenate(faces)

    mesh = pv.PolyData(points, faces)
    return mesh



