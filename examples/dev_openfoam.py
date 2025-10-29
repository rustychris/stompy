from stompy.model.openfoam import depth_average
import numpy as np
from stompy import utils

import six

sim_dir="y:/DWR/fishpassage-DWR-7/fishpassage-DWR-7-7cms-local"
## 
six.moves.reload_module(depth_average)
pf = depth_average.PostFoam(sim_dir=sim_dir)

# OF generated:
#centers0=pf.cell_centers(proc)
import time

center_fn="test-centers.dat"
t=time.time()
pf.compute_cell_centers_py(proc,center_fn)
print("Elapsed: ",time.time()-t)



import pandas as pd
df = pd.read_csv(center_fn,sep=r'\s+',names=['type','x','y','z'])
centers_py = df[ ['x','y','z'] ].values
err = centers0 - centers_py
print("RMSE: " , np.sqrt(np.mean(err**2)))

##

# about 25s for a processor with 35k faces, almost all the time
# in calc_face_center

def calc_face_center(face):
    VSMALL=1e-14
    if len(face)==3:
        return face.mean(axis=0),0.5*np.cross(face[1]-face[0],face[2]-face[0])
    else:    
        sumN=np.zeros(3,np.float64)
        sumA=0.0
        sumAc=np.zeros(3,np.float64)

        fCentre = face.mean(axis=0)

        nPoints=face.shape[0]
        for pi in range(nPoints):
            p1=face[pi]
            p2=face[(pi+1)%nPoints]

            centroid3 = p1 + p2 + fCentre
            area_norm = np.cross( p2 - p1, fCentre - p1)
            area = utils.mag(area_norm)

            sumN += area_norm;
            sumA += area;
            sumAc += area*centroid3;

        return (1.0/3.0)*sumAc/(sumA + VSMALL), 0.5*sumN

def cell_center_py(facefile, xyz, owner, neigh):
    """
    facefile:
    xyz: pointfile.values.reshape([-1,3])
    owner: [Nfaces] index of face's owner cell
    neigh: [NInternalFaces]  index of faces's neighbor cell
    """
    # replicate cell center calculation from openfoam-master/src/meshTools/primitiveMeshGeometry.C
    faces=[] # [N,{xyz}] array per face

    VSMALL=1e-14
    nfaces = facefile.nfaces
    # WRONG ncells = owner.shape[0]
    ncells = 1+max(owner.max(),neigh.max()) # or get it from owner file
    n_internal = neigh.shape[0]        

    face_ctr=np.zeros((nfaces,3),np.float64)
    face_area=np.zeros((nfaces,3),np.float64)

    if 1: # get face centers
        # 20 s for one domain
        for fIdx in utils.progress(range(nfaces)):
            face_nodes = facefile.faces[fIdx]["id_pts"][:]
            face = xyz[list(face_nodes)]
            face_ctr[fIdx],face_area[fIdx] = calc_face_center(face)

    if 1: # estimated cell centers
        cell_est_centers = np.zeros( (ncells,3), np.float64)
        cell_n_faces = np.zeros( ncells, np.int32)

        for j,ctr in enumerate(face_ctr):
            c_own = owner[j]
            cell_est_centers[c_own] += ctr
            cell_n_faces[c_own] += 1

        for j,ctr in enumerate(face_ctr[:n_internal]):
            c_nbr = neigh[j]
            cell_est_centers[c_nbr] += ctr
            cell_n_faces[c_nbr] += 1

        cell_est_centers[:] /= cell_n_faces[:,None]

    if 1: # refined cell centers
        cell_centers=np.zeros_like(cell_est_centers)
        cell_volumes=np.zeros(ncells,np.float64)

        def mydot(a,b): # fighting with numpy to get vectorized dot product
            return (a*b).sum(axis=-1)

        pyr3Vol_own = mydot(face_area, face_ctr - cell_est_centers[owner]).clip(VSMALL)
        pyrCtr_own = (3.0/4.0)*face_ctr + (1.0/4.0)*cell_est_centers[owner]
        for j in range(nfaces):
            cell_centers[owner[j]] += pyr3Vol_own[j,None] * pyrCtr_own[j]
            cell_volumes[owner[j]] += pyr3Vol_own[j]

        # note sign flip to account for nbr normal
        pyr3Vol_nbr = mydot(face_area[:n_internal], cell_est_centers[neigh] - face_ctr[:n_internal]).clip(VSMALL)
        pyrCtr_nbr = (3.0/4.0)*face_ctr[:n_internal] + (1.0/4.0)*cell_est_centers[neigh]

        for j in range(n_internal):
            cell_centers[neigh[j]] += pyr3Vol_nbr[j,None] * pyrCtr_nbr[j]
            cell_volumes[neigh[j]] += pyr3Vol_nbr[j]

        cell_centers /= cell_volumes[:,None]
        cell_volumes *= 1.0/3.0

    return cell_centers, cell_volumes

proc=0
centers0=pf.cell_centers(proc)

faces0 = pf.face_centers(proc)

facefile = pf.read_facefile(proc)
pointfile = pf.read_pointfile(proc)
owner = pf.read_owner(proc)
neigh = pf.read_neighbor(proc)

ctrs,vols = cell_center_py(facefile, pointfile.values.reshape([-1,3]), owner.values, neigh.values)
