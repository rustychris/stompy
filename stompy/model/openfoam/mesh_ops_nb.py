import os
import numpy as np
from ... import utils
from scipy.spatial import Delaunay
from collections import defaultdict

from numba import njit
from numba.typed import List
    
import time

from .mesh_ops import FACE_MAX_NODES, CELL_MAX_FACES, NO_FACE, VSMALL


@njit
def mesh_face_center_areas_nb(xyz,face_nodes, face_cells, cell_faces):
    # same calc as face_center_area_py, but takes array data
    
    nfaces=len(face_nodes)

    #face_node_count = (face_nodes>=0).sum(axis=1)
    #face_node_count = np.zeros(nfaces,np.int32)
    face_ctr=np.zeros((nfaces,3),np.float64)
    face_area=np.zeros((nfaces,3),np.float64)

    for fIdx in range(nfaces):
        face = List()
        ctr=np.zeros(3,np.float64)
        for f_n in face_nodes[fIdx]:
            if f_n<0: break
            p = xyz[f_n]
            face.append(p)
            ctr+=p

        ctr /= len(face)
        if len(face)==3:
            area = 0.5*np.cross(face[1]-face[0],face[2]-face[0])
        else:    
            sumN=np.zeros(3,np.float64)
            sumA=0.0
            sumAc=np.zeros(3,np.float64)

            #fCentre = face.sum(axis=0) / face.shape[0]

            nPoints=len(face)
            for pi in range(nPoints):
                p1=face[pi]
                p2=face[(pi+1)%nPoints]

                centroid3 = p1 + p2 + ctr
                area_norm = np.cross( p2 - p1, ctr - p1)
                area = np.sqrt(np.sum(area_norm**2))

                sumN += area_norm;
                sumA += area;
                sumAc += area*centroid3;

            ctr = (1.0/3.0)*sumAc/(sumA + VSMALL)
            area = 0.5*sumN
        face_ctr[fIdx] = ctr
        face_area[fIdx] = area

    return face_ctr, face_area

def mesh_cell_volume_centers_nb(xyz, face_nodes, face_cells, cell_faces):
    """
    xyz: pointfile.values.reshape([-1,3])
    face_nodes: [Nfaces,FACE_MAX_NODES] padded array with -1 for missing
    face_cells: [Nfaces,2] owner,neighbor cell (-1 for boundary)
    cell_faces: [Ncells, CELL_MAX_FACES] fIdx, ~fIdx, or NO_FACE

    very similar to cell_volume_centers in depth_average, but takes padded arrays
    and the order of return values matches the function name
    """
    # replicate cell center calculation from openfoam-master/src/meshTools/primitiveMeshGeometry.C
    print("top of mesh_cell_volume_centers_nb")

    VSMALL=1e-14
    nfaces = len(face_nodes)
    ncells = len(cell_faces)
    
    # boundary faces not necessarily sorted to the end since the mesh is being updated.
    # NOT VALID n_internal = neigh.shape[0] 

    if 1: # get face centers
        face_ctr,face_area = mesh_face_center_areas_nb(xyz,face_nodes,face_cells,cell_faces)

    if 1: # estimated cell centers
        cell_est_centers = np.zeros( (ncells,3), np.float64)
        cell_n_faces = np.zeros( ncells, np.int32)

        for j in range(nfaces):
            c_own = face_cells[j][0]
            cell_est_centers[c_own] += face_ctr[j]
            cell_n_faces[c_own] += 1
            c_nbr = face_cells[j][1]
            if c_nbr>=0:
                cell_est_centers[c_nbr] += face_ctr[j]
                cell_n_faces[c_nbr] += 1

        cell_est_centers[:] /= cell_n_faces[:,None]

    if 1: # refined cell centers
        cell_centers=np.zeros_like(cell_est_centers)
        cell_volumes=np.zeros(ncells,np.float64)

        for j in range(nfaces):
            c_own = face_cells[j][0]
            c_nbr = face_cells[j][1]
            
            # For owner:
            pyr3Vol = (face_area[j]*(face_ctr[j]-cell_est_centers[c_own])).sum()
            pyr3Vol = max(pyr3Vol,VSMALL)
            pyrCtr = (3.0/4.0)*face_ctr[j] + (1.0/4.0)*cell_est_centers[c_own]

            cell_centers[c_own] += pyr3Vol * pyrCtr
            cell_volumes[c_own] += pyr3Vol

            if c_nbr>=0:
                # note sign flip to account for nbr normal
                pyr3Vol = (face_area[j] * (cell_est_centers[c_nbr] - face_ctr[j])).sum()
                pyr3Vol = max(pyr3Vol,VSMALL)
                pyrCtr = (3.0/4.0)*face_ctr[j] + (1.0/4.0)*cell_est_centers[c_nbr]

                cell_centers[c_nbr] += pyr3Vol * pyrCtr
                cell_volumes[c_nbr] += pyr3Vol

        cell_centers /= cell_volumes[:,None]
        cell_volumes *= 1.0/3.0
        
    print("end of mesh_cell_volume_centers_nb")

    return cell_volumes, cell_centers


# about 8x faster than python version. but maybe has some bad corner cases?
# On case_dir="../../../fishpassage-Bombac/fp-DWR-Bombac-5pools-xyz5"
# with dx=0.05
# Gets through the columns but then row 21 yields "Failed to find face to flip"
# for proc 0.
# try without njit
@njit(fastmath=True)
def mesh_slice_nb(slice_normal, slice_offset, cell_mapping, xyz, face_nodes, face_cells, cell_faces):
    slice_normal=slice_normal.astype(np.float64)
    tol = 1e-10
    if cell_mapping is None:
        cell_mapping = List(np.arange(len(cell_faces)))

    xyz=List(xyz)
    face_nodes=List(face_nodes)
    face_cells=List(face_cells)
    cell_faces=List(cell_faces)
        
    # identify which faces to slice:
    if 1:
        side_xyz = List() # np.full(len(xyz),0,dtype=np.float32)
        for i,pnt in enumerate(xyz):
            offset_pnt = np.dot(pnt,slice_normal) - slice_offset
            side_pnt = np.int32( np.sign(offset_pnt) )
            # nudge onto slice plane
            if np.abs(offset_pnt)<tol:
                side_pnt = 0
            side_xyz.append(side_pnt)

        #cmp_xyz = side_xyz<0
        # face_nodes is padded with -1

        # okay if some of these do not end up getting sliced, say if they are tangent or coplanar to
        # slice. just be fast here

        faces_to_slice_mask = np.full(len(face_nodes),False)
        for fIdx in range(len(face_nodes)):
            has_neg=False
            has_pos=False
            for n in face_nodes[fIdx]:
                if n<0:
                    break
                if side_xyz[n]<0:
                    has_neg=True
                else:
                    has_pos=True
                if has_neg and has_pos==True:
                    faces_to_slice_mask[fIdx]=True
                    break
        faces_to_slice = np.nonzero(faces_to_slice_mask)[0]
            

    # Filtering step - in the case of warped faces, it's possible to have something like
    # size_xyz=[-1,1,-1,1]. Forcing those nodes to be on the slice plane leads to issues
    # with adjacent faces. Instead, triangulate.
    if 1:
        n_face_orig = faces_to_slice.shape[0]
        
        for fIdx in faces_to_slice:
            nodes = face_nodes[fIdx]
            # node_count = np.sum(nodes>=0)
            for node_i in range(FACE_MAX_NODES):
                if nodes[node_i]<0:
                    node_count=node_i
                    break
            else:
                node_count=FACE_MAX_NODES
                
            nodes = nodes[:node_count]

            # Simplify logic below by forcing sides_nodes[0]==-1
            for i_start in range(node_count):
                if side_xyz[nodes[i_start]]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue

            last_side=side_xyz[nodes[i_start]]
            crossings=0
            coplanar=0
            for i in range(node_count):
                i_this=(i_start+i)%node_count
                i_next=(i_this+1)%node_count
                side_this=side_xyz[nodes[i_this]]
                side_next=side_xyz[nodes[i_next]]
                if side_this==0 and side_next==0:
                    coplanar+=1
                    continue
                if side_next==0:
                    continue
                if last_side!=side_next:
                    crossings+=1
                last_side=side_next

            if crossings>2 or (crossings>0 and coplanar>0):
                # Break fIdx into triangles. No changes to xyz, but face_nodes, face_cells, cell_faces get
                # updated.
                print(f"Triangulating warped face on slice plane for fIdx={fIdx}")
                (xyz, face_nodes, face_cells, cell_faces) = mesh_triangulate_nb(fIdx, 
                                                                                xyz, face_nodes, face_cells, cell_faces)

        tri_faces = np.arange(n_face_orig,len(face_nodes))
        if len(tri_faces):
            faces_to_slice = np.concatenate((faces_to_slice,tri_faces))
        
    # slices actually have to be per edge, not face.
    sliced_edges={} # (edge small node, edge large node) => new node
    sliced_cells={} # cell index => new cell clipped off of it
    
    cell_n_faces = np.zeros(len(cell_faces),np.int32)
    for cIdx,faces in enumerate(cell_faces):
        cell_n_faces[cIdx] = (faces!=NO_FACE).sum()
    
    n_face_orig = len(face_nodes)
    n_node_orig = len(xyz)
    
    for fIdx in faces_to_slice:
        nodes = face_nodes[fIdx]
        for node_i in range(FACE_MAX_NODES):
            if nodes[node_i]<0:
                node_count=node_i
                break
        else:
            node_count=FACE_MAX_NODES
        nodes = nodes[:node_count]
    
        # use side_xyz for fine-grained control

        if 1: # Simplify logic below by forcing sides_nodes[0]==-1
            for i_start in range(node_count):
                if side_xyz[nodes[i_start]]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue
        
        # Split into two lists of nodes, one for the part of the face with cmp_xyz True (less
        # than offset), and one for the part of the face on or above the slice
        # for the moment don't worry about warped faces that could have more than 2 crossings
        nodes_neg = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_neg=0
        nodes_pos = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_pos=0
        last_side = side_xyz[nodes[i_start]]
        assert last_side!=0,"Should be -1, but definitely not 0"
    
        coplanar_count=0 # how many edges of the face are on the slice plane
    
        for ii in range(node_count):
            i_this=(i_start+ii)%node_count
            i_next=(i_this+1)%node_count
            # where should node i+1 go?
            # cases for -1/0/+1:
            #   Easy cases:
            #     -1 to  1
            #     -1 to -1
            #      1 to -1
            #      1 to  1
    
            # Tricky cases:
            #   -1 to 0   Add to last_side
            #   1 to 0    Add to last_side
            #   0 to 0    Add to last_side and note coplanar edge
            
            #   0 to -1   If last_side is -1, add holding[1:] to neg, clear holding
            #             If last_side is 1:
            #   0 to 1    I
    
            # nodes on the slice
            #
            #                E------D                    F--E
            #                |      |                    |  |
            # --D------C-- --F------C-- ----C----  --G---D--C--
            #   |      |     |      |      / \       |      |
            #   |      |     |      |     /   \      |      |
            #   A------B     A------B    A-----B     A------B
            #
            # The goals would be:
            #   case I: one of the faces ends up empty. No slicing.
            #     if the nodes go to pos face, make sure order is correct.
            #   case II: create two faces, use existing nodes on slice
            #   case III: one of the faces ends up empty. No slicing.
            #   case IV: bad mesh... this one is tough because the edge
            #      GD can go to either side depending on whether FDG is
            #      concave or DGA. Flag it as a bad face and punt for now.
            #
            #  e.g. case I:
            #    figure out the starting side. If the side_nodes[0]==0,
            #    rotate to always get side_nodes[0]==-1
            
            side_this=side_xyz[nodes[i_this]]
            side_next=side_xyz[nodes[i_next]]
            
            if side_this==-1 and side_next==-1:
                nodes_neg[count_neg]=nodes[i_next]
                count_neg+=1
                # last_side unchanged
            elif side_this==1 and side_next==1:
                nodes_pos[count_pos]=nodes[i_next]
                count_pos+=1
                # last_side unchanged
            elif side_next==0:
                if side_this==0:
                    coplanar_count+=1 # for warning below
                    
                if last_side==-1:
                    nodes_neg[count_neg]=nodes[i_next]
                    count_neg+=1
                else: # last_side==1
                    assert last_side==1
                    nodes_pos[count_pos]=nodes[i_next]
                    count_pos+=1
                # last_side unchanged
            elif side_this==0: # leaving slice plane
                # nodes[i] already on last_side.
                if side_next!=last_side:
                    # need to add nodes[i] to side_next, too
                    if side_next==-1:
                        nodes_neg[count_neg]=nodes[i_this] # i, not i_next
                        count_neg+=1
                    elif side_next==1:
                        nodes_pos[count_pos]=nodes[i_this] # i, not i_next
                        count_pos+=1
                    else:
                        assert False,"Really?"
                else:
                    pass # node on slice then return to last_side is fine.
                    
                if side_next==-1:
                    nodes_neg[count_neg]=nodes[i_next]
                    count_neg+=1
                elif side_next==1:
                    nodes_pos[count_pos]=nodes[i_next]
                    count_pos+=1
                else:
                    assert False,"How?"
                last_side = side_next
            elif side_this * side_next < 0: # Clear split
                # if cmp_nodes[i] != cmp_nodes[i_next]:  # SPLIT
                n_small,n_large = nodes[i_this],nodes[i_next]
                if n_small>n_large:
                    n_small,n_large = n_large,n_small
                k=(n_small,n_large)
    
                if k not in sliced_edges:
                    value = np.dot(xyz[nodes[i_this]],slice_normal) - slice_offset
                    next_value = np.dot(xyz[nodes[i_next]],slice_normal) - slice_offset
                    assert next_value!=value
                    alpha = (0-value)/(next_value-value)
                    new_point = (1-alpha)*xyz[nodes[i_this]] + alpha*xyz[nodes[i_next]]
                    new_node = len(xyz)
                    xyz.append( new_point )
                    side_xyz.append(0) # on the slice by construction
                    sliced_edges[k]=new_node
                else:
                    new_node = sliced_edges[k]
                # New node goes on both
                nodes_neg[count_neg]=new_node
                count_neg+=1
                nodes_pos[count_pos]=new_node
                count_pos+=1
                if side_next==-1:
                    nodes_neg[count_neg]=nodes[i_next]
                    count_neg+=1
                else:
                    assert side_next==1
                    nodes_pos[count_pos]=nodes[i_next]
                    count_pos+=1
                last_side = side_next
    
        if count_neg==0 or count_pos==0:
            continue # no slice once tangent/coplanar handled.
    
        assert coplanar_count==0,"Not ready for bad cells that are sliced and coplanar"
        assert count_neg>=3 
        assert count_pos>=3
    
        face_nodes[fIdx]=nodes_neg
        newFIdx = len(face_nodes)
        face_nodes.append(nodes_pos)
        face_cells.append(face_cells[fIdx].copy())
    
        # At this stage, just update the faces
        # and keep them all tied to the original cell, and note that the
        # cell has to be split. Sorting into a new cell handled below.
        # Orientation of nodes_pos should be consistent with nodes_neg
        # so 1s-complement for neigh is correct.
    
        cIdx = face_cells[fIdx][0] # owner
        assert cIdx>=0
        sliced_cells[cIdx] = -1 # sliced, but no new cell has been created
        tmp=cell_faces[cIdx]
        tmp[cell_n_faces[cIdx]]=newFIdx
        cell_n_faces[cIdx]+=1
    
        cIdx = face_cells[fIdx][1] # neighbor
        if cIdx>=0:
            sliced_cells[cIdx] = -1
            cell_faces[cIdx][cell_n_faces[cIdx]]=~newFIdx # ~ for neighbor reference
            cell_n_faces[cIdx]+=1
            assert cIdx>face_cells[fIdx][0] # the actual invariant for face orientation
    
    # Sort the clipped cells -- safer not to assume they're sorted. Don't want to
    # assume that everything is maintained upper-triangular and properly sorted
    cells_to_sort = list(sliced_cells.keys())
    cells_to_sort.sort()
    
    for cIdx in cells_to_sort:
        # extra cast is just a guess.
        faces = cell_faces[cIdx][:cell_n_faces[cIdx]].astype(np.int32) 
        cell_face_neg = np.full(CELL_MAX_FACES,NO_FACE,np.int32)
        count_neg=0
        cell_face_pos = np.full(CELL_MAX_FACES,NO_FACE,np.int32)
        count_pos=0
        new_face_edges=[] # collect edges to make new face
    
        for signed_fIdx in faces:
            if signed_fIdx<0:
                fIdx = ~signed_fIdx
            else:
                fIdx = signed_fIdx
    
            f_nodes = face_nodes[fIdx]
            f_nodes = f_nodes[f_nodes>=0]
            # is_neg = (np.dot(slice_normal,xyz[f_nodes].mean(axis=0))<slice_offset)
            is_neg_sum = 0.0
            for f_n in f_nodes:
                is_neg_sum += np.dot(slice_normal, xyz[f_n])
            is_neg = is_neg_sum/len(f_nodes) < slice_offset
            
            if is_neg:
                #print(f"Face {fIdx} assigned to original cell as {signed_fIdx}")
                cell_face_neg[count_neg]=signed_fIdx
                count_neg+=1
            else:
                #print(f"Face {fIdx} assigned to new cell as {signed_fIdx}")
                cell_face_pos[count_pos]=signed_fIdx
                count_pos+=1
    
            # since original faces can be tangent/coincident with slice plane,
            # not sufficient to check that it's a new face. 
            if not is_neg: 
                nodeA=nodeB=-1
                # on_slice_count = (side_xyz[f_nodes]==0).sum()
                on_slice_count = 0
                for f_n in f_nodes:
                    if side_xyz[f_n]==0:
                        on_slice_count+=1
                        
                if on_slice_count>2:
                    print(f"WARNING: face {fIdx} has {on_slice_count} nodes on slice, and is part of a cell to be sorted")
                    #import pdb
                    #pdb.set_trace() # not sure what to do in this case
                if (fIdx<n_face_orig) and (on_slice_count<2):
                    continue
                
                for f_i in range(len(f_nodes)):
                    assert f_nodes[f_i]>=0 # f_nodes is truncated above
                    
                    if side_xyz[f_nodes[f_i]]==0:
                        # created from or on slice
                        # on sliced edge the new nodes must be adjacent
                        # weird sliced faces with edges that were on the slice will be
                        # bad here.
                        if f_i==0 and side_xyz[f_nodes[f_i+1]]!=0:
                            # last-to-first edge is the one.
                            # pretty sure this overly restrictive. face could have come in with a node
                            # on the slice. Even if it's a new face, one of the nodes could have already
                            # been on the slice.
                            #assert f_nodes[-1]>=n_node_orig,"More trouble with slice nodes"
                            nodeA = f_nodes[-1]
                            nodeB = f_nodes[0]
                            assert nodeA!=nodeB
                        else: 
                            nodeA=f_nodes[f_i]
                            assert f_i+1<len(f_nodes),"Trouble with slice nodes"
                            nodeB=f_nodes[f_i+1]
                            assert nodeA!=nodeB
                        break
                assert nodeA>=0
                assert nodeB>=0
                assert side_xyz[nodeA]==0
                assert side_xyz[nodeB]==0
    
                # Should be able to infer orientation
                # The invariants for orientation: face normals follow righthand rule,
                # and the normals points out of the cell with a lower index.
                # Try to stick with that, rather than assuming it always points away from owner.
                # So I want the edges to make a face normal pointing out of the neg cell
                if signed_fIdx>=0:
                    # face_nodes[fIdx] are in the desired order
                    # These two clauses are suspect, and just checking the sign may not be
                    # enough. Also possible that invariants were broken earlier. Orientation of
                    # new faces matches the old, but it's possible that cells are being created
                    # in the right order, that face_cells isn't proparly updated, dunno
                    pass
                else:
                    nodeB,nodeA = nodeA,nodeB
                new_face_edges.append( [nodeA,nodeB] )
    
        # Stuff new_face_edges in here by matching
        # The ordering is tedious
        # Could resort to angle-sorting, esp. since the slice plane is handy.
        
        # Create the new face's nodes:
        if 1:
            new_face_node = np.full(FACE_MAX_NODES,-1, dtype=np.int32)
            new_face_node[:2] = new_face_edges[0]
            new_face_edges[0] = [-1,-1]
        
            nfn_count=2
            for edge_count in range(1,len(new_face_edges)):
                to_match = new_face_node[nfn_count-1]
                for edge_i in range(1,len(new_face_edges)):
                    if new_face_edges[edge_i][0]!=to_match: continue # already used or no match
                    new_face_node[nfn_count] = new_face_edges[edge_i][1]
                    nfn_count+=1
                    new_face_edges[edge_i] = [-1,-1]
                    break
            #assert new_face_node[nfn_count-1]==new_face_node[0]
            new_face_node[nfn_count-1]=-1
        
        new_fIdx = len(face_nodes)
        face_nodes.append(new_face_node) 

        # new_cIdx always greater than cIdx, so this maintains face-normal invariant.
        cell_face_neg[count_neg]=new_fIdx
        count_neg+=1
        cell_face_pos[count_pos]=~new_fIdx
        count_pos+=1
        
        # Create new cell and update face_cells and cell_face adjacency info
        new_cIdx = np.int32(len(cell_faces))
        cell_faces[cIdx][:] = cell_face_neg
        cell_faces.append(cell_face_pos)
        cell_mapping.append(cell_mapping[cIdx])
        face_cells.append(np.array([cIdx,new_cIdx],np.int32))
        
        # Update face_cells. Only have to deal with those in cell_face_pos. Faces in
        # cell_face_neg already point to the correct cIdx. This will overwrite
        # new_cIdx with new_cIdx for the slice face, just fyi
        for signed_fIdx in cell_face_pos:
            if signed_fIdx==NO_FACE: break
            if signed_fIdx<0:
                fIdx=~signed_fIdx
                face_cells[fIdx][1] = new_cIdx
            else:
                fIdx=signed_fIdx
                face_cells[fIdx][0] = new_cIdx
            # Some extra work to maintain the invariant that the face normal points to
            # the higher indexed cell.
            cOwn, cNbr = face_cells[fIdx] # ownership before flipping
            if cNbr>=0 and cNbr<cOwn:
                # Due to changes in cell indexes, face must point in opposite direction
                face_cells[fIdx][0]=cNbr
                face_cells[fIdx][1]=cOwn
                f_nodes = face_nodes[fIdx]
                f_nodes = f_nodes[f_nodes>=0]
                face_nodes[fIdx][:len(f_nodes)] = f_nodes[::-1] # flip orientation

                for f_i in range(CELL_MAX_FACES):
                    if cell_faces[cOwn][f_i] == fIdx:
                        cell_faces[cOwn][f_i] = ~fIdx
                        break
                else:
                    print("Failed to find face to flip")
                    print(f"cOwn: {cOwn} looking for fIdx:{fIdx} in ",cell_faces[cOwn])
                    print(type(fIdx))
                    assert False,"Failed to find face to flip"
                    
                for f_i in range(CELL_MAX_FACES):
                    if cell_faces[cNbr][f_i] == ~fIdx:
                        cell_faces[cNbr][f_i] = fIdx
                        break
                else:
                    print("Failed to find face to flip")
                    print(f"cNbr: {cNbr} looking for ~fIdx:{~fIdx} in ",cell_faces[cNbr])
                    print(type(fIdx))
                    assert False,"Failed to find face to flip"

    return cell_mapping, (xyz, face_nodes, face_cells, cell_faces)

@njit
def mesh_cell_bboxes_nb(xyz,face_nodes,face_cells,cell_faces):
    n_faces = len(face_nodes)
    face_xyz_min = np.zeros((n_faces,3),np.float64)
    face_xyz_max = np.zeros((n_faces,3),np.float64)

    for fIdx in range(n_faces):
        face_xyz_min[fIdx,:] = xyz[face_nodes[fIdx][0]]
        face_xyz_max[fIdx,:] = xyz[face_nodes[fIdx][0]]
        for f_n in face_nodes[fIdx][1:]:
            if f_n<0: break
            face_xyz_min[fIdx] = np.minimum(face_xyz_min[fIdx],xyz[f_n])
            face_xyz_max[fIdx] = np.maximum(face_xyz_max[fIdx],xyz[f_n])
            
    n_cells = len(cell_faces)
    cell_bounds = np.zeros((n_cells,6), np.float64)
    for cIdx in range(n_cells):
        fIdx=cell_faces[cIdx][0]
        cell_bounds[cIdx,::2] = face_xyz_min[fIdx]
        cell_bounds[cIdx,1::2] = face_xyz_max[fIdx]
        for fIdx in cell_faces[cIdx][1:]:
            if fIdx==NO_FACE: break
            cell_bounds[cIdx,::2 ] = np.minimum(cell_bounds[cIdx,::2], face_xyz_min[fIdx])
            cell_bounds[cIdx,1::2] = np.maximum(cell_bounds[cIdx,1::2],face_xyz_max[fIdx])
                
    return cell_bounds


@njit
def mesh_triangulate_nb(fIdx, 
                        xyz, face_nodes, face_cells, cell_faces):
    # assume incoming face is convex in the plane, such that triangulation is
    # simple. Not always a great assumption...
    nodes = face_nodes[fIdx]
    nodes = nodes[nodes>=0] # this will be a copy, so it's safe to mutate face_nodes below

    assert len(nodes)>=3
    
    owner,nbr = face_cells[fIdx]
    n_face_owner = (cell_faces[owner]!=NO_FACE).sum()
    if nbr>=0:
        n_face_nbr = (cell_faces[nbr]!=NO_FACE).sum()
    
    for i_tri in range(len(nodes)-2):
        tri_nodes = np.full(FACE_MAX_NODES,-1,np.int32)
        tri_nodes[0] = nodes[0]
        tri_nodes[1] = nodes[i_tri+1]
        tri_nodes[2] = nodes[i_tri+2]

        if i_tri==0:
            face_nodes[fIdx] = tri_nodes
        else:
            newFIdx = len(face_nodes)
            face_nodes.append(tri_nodes)
            face_cells.append(face_cells[fIdx].copy())
            cell_faces[owner][n_face_owner]=newFIdx
            n_face_owner+=1

            if nbr>=0:
                cell_faces[nbr][n_face_nbr]=~newFIdx # ~ for neighbor reference
                n_face_nbr+=1

    return xyz, face_nodes, face_cells, cell_faces
