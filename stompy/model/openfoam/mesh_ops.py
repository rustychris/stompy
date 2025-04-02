import os
import numpy as np
from numba import njit
from fluidfoam.readof import OpenFoamFile
from ... import utils

FACE_MAX_NODES=20
CELL_MAX_FACES=20
NO_FACE=np.iinfo(np.int32).max


#mesh_state = (xyz, face_nodes, face_cells, cell_faces)


def load_mesh_state(case_dir,precision=15):
    # Load proc mesh                    
    verbose=False
    meshpath = os.path.join(case_dir,'constant/polyMesh')
    # owner.nb_faces, boundary, values, nb_cell
    owner = OpenFoamFile(meshpath, name="owner", verbose=verbose)
    facefile = OpenFoamFile(meshpath, name="faces", verbose=verbose)
    pointfile = OpenFoamFile(meshpath,name="points",precision=precision,
        verbose=verbose)
    neigh = OpenFoamFile(meshpath, name="neighbour", verbose=verbose)

    face_cells=np.zeros((len(owner.values),2),np.int32)
    face_cells[:,0] = owner.values
    n_interior = len(neigh.values)
    face_cells[:n_interior,1] = neigh.values
    face_cells[n_interior:,1] = -1 # boundary

    xyz=pointfile.values.reshape([-1,3])
    # xyz=np.tensordot(xyz,rot,[-1,0]) # move this to separate function

    # At what point do I need to track cells?
    # ultimately, it's cells that have data

    cell_faces = np.full( (owner.nb_cell,CELL_MAX_FACES), NO_FACE, dtype=np.int32 )
    cell_n_faces = np.zeros( owner.nb_cell, np.int32)
    
    for fIdx,cIdx in enumerate(neigh.values):
        cell_faces[cIdx,cell_n_faces[cIdx]] = ~fIdx  # one's complement indicates we're the nbr
        cell_n_faces[cIdx] += 1
    for fIdx,cIdx in enumerate(owner.values):
        cell_faces[cIdx,cell_n_faces[cIdx]] = fIdx  # one's complement indicates we're the nbr
        cell_n_faces[cIdx] += 1

    face_nodes=np.full( (facefile.nfaces,FACE_MAX_NODES), -1, np.int32)
    for fIdx in range(facefile.nfaces):
        id_pts = facefile.faces[fIdx]["id_pts"]
        face_nodes[fIdx,:len(id_pts)]=id_pts

    return (xyz,face_nodes,face_cells,cell_faces)

def mesh_rotate(rot,xyz,face_nodes,face_cells,cell_faces):
    xyz=np.tensordot(xyz,rot,[-1,0])
    if np.linalg.det(rot)<0:
        print("WARNING: rotation does not preserve handedness")
    return (xyz,face_nodes,face_cells,cell_faces)

@njit
def mesh_face_center_areas(xyz,face_nodes, face_cells, cell_faces):
    # same calc as face_center_area_py, but takes array data
    face_node_count = (face_nodes>=0).sum(axis=1)
    VSMALL=1e-14
    nfaces=face_nodes.shape[0]
    face_ctr=np.zeros((nfaces,3),np.float64)
    face_area=np.zeros((nfaces,3),np.float64)

    for fIdx in range(nfaces):
        face = xyz[face_nodes[fIdx,:face_node_count[fIdx]]]

        if len(face)==3:
            ctr = (face[0] + face[1] + face[2]) / 3.0
            area = 0.5*np.cross(face[1]-face[0],face[2]-face[0])
        else:    
            sumN=np.zeros(3,np.float64)
            sumA=0.0
            sumAc=np.zeros(3,np.float64)

            fCentre = face.sum(axis=0) / face.shape[0]

            nPoints=face.shape[0]
            for pi in range(nPoints):
                p1=face[pi]
                p2=face[(pi+1)%nPoints]

                centroid3 = p1 + p2 + fCentre
                area_norm = np.cross( p2 - p1, fCentre - p1)
                area = np.sqrt(np.sum(area_norm**2))

                sumN += area_norm;
                sumA += area;
                sumAc += area*centroid3;

            ctr = (1.0/3.0)*sumAc/(sumA + VSMALL)
            area = 0.5*sumN
        face_ctr[fIdx] = ctr
        face_area[fIdx] = area

    return face_ctr, face_area
    
def mesh_cell_volume_centers(xyz, face_nodes, face_cells, cell_faces):
    """
    xyz: pointfile.values.reshape([-1,3])
    face_nodes: [Nfaces,FACE_MAX_NODES] padded array with -1 for missing
    face_cells: [Nfaces,2] owner,neighbor cell (-1 for boundary)
    cell_faces: [Ncells, CELL_MAX_FACES] fIdx, ~fIdx, or NO_FACE

    very similar to cell_volume_centers in depth_average, but takes padded arrays
    and the order of return values matches the function name
    """
    # replicate cell center calculation from openfoam-master/src/meshTools/primitiveMeshGeometry.C

    faces=[] # [N,{xyz}] array per face

    VSMALL=1e-14
    nfaces = face_nodes.shape[0]
    ncells = cell_faces.shape[0]
    # boundary faces not necessarily sorted to the end since the mesh is being updated.
    # NOT VALID n_internal = neigh.shape[0] 

    if 1: # get face centers
        face_ctr,face_area = mesh_face_center_areas(xyz,face_nodes,face_cells,cell_faces)

    if 1: # estimated cell centers
        cell_est_centers = np.zeros( (ncells,3), np.float64)
        cell_n_faces = np.zeros( ncells, np.int32)

        for j in range(nfaces):
            c_own = face_cells[j,0]
            cell_est_centers[c_own] += face_ctr[j]
            cell_n_faces[c_own] += 1
            c_nbr = face_cells[j,1]
            if c_nbr>=0:
                cell_est_centers[c_nbr] += face_ctr[j]
                cell_n_faces[c_nbr] += 1

        cell_est_centers[:] /= cell_n_faces[:,None]

    if 1: # refined cell centers
        owner = face_cells[:,0]
        cell_centers=np.zeros_like(cell_est_centers)
        cell_volumes=np.zeros(ncells,np.float64)

        def mydot(a,b): # fighting with numpy to get vectorized dot product
            return (a*b).sum(axis=-1)

        # For owner:
        pyr3Vol = mydot(face_area, face_ctr - cell_est_centers[owner]).clip(VSMALL)
        pyrCtr = (3.0/4.0)*face_ctr + (1.0/4.0)*cell_est_centers[owner]
        for j in range(nfaces):
            cell_centers[owner[j]] += pyr3Vol[j,None] * pyrCtr[j]
            cell_volumes[owner[j]] += pyr3Vol[j]

        neigh = face_cells[:,1]
        internal = neigh>=0 # rather than :n_internal
        # note sign flip to account for nbr normal

        pyr3Vol[~internal] = np.nan
        pyr3Vol[ internal] = mydot(face_area[internal], cell_est_centers[neigh[internal]] - face_ctr[internal]).clip(VSMALL)
        pyrCtr[~internal] = np.nan
        pyrCtr[ internal] = (3.0/4.0)*face_ctr[internal] + (1.0/4.0)*cell_est_centers[neigh[internal]]

        for j,nbr in enumerate(face_cells[:,1]):
            if nbr>=0:
                cell_centers[nbr] += pyr3Vol[j,None] * pyrCtr[j]
                cell_volumes[nbr] += pyr3Vol[j]

        cell_centers /= cell_volumes[:,None]
        cell_volumes *= 1.0/3.0

    return cell_volumes, cell_centers

def mesh_cell_summary(cIdx,mesh_state):
    return mesh_check_cell(cIdx,True,mesh_state)

def mesh_check_cell(cIdx,verbose,mesh_state):
    if verbose:
        print(f"cIdx={cIdx}")
    (xyz, face_nodes, face_cells, cell_faces) = mesh_state

    faces=cell_faces[cIdx]
    faces=faces[faces!=NO_FACE]
    # triangular prism - expect 5 faces
    if verbose:
        print(f"  cell_faces: ", faces)

    abs_faces = [(f if f>=0 else ~f) for f in faces]
    cell_nodes = np.unique(np.concatenate([face_nodes[f] for f in abs_faces]))
    cell_nodes = cell_nodes[cell_nodes>=0]
    if verbose:
        print("  cell_nodes: ",cell_nodes)

    # approx face normal calc
    face_centers=[]
    face_normals=[]
    for f in faces:
        flip = f<0
        if flip:
            f=~f
        f_nodes = face_nodes[f]
        f_nodes = f_nodes[f_nodes>=0]
        ctr=xyz[f_nodes].mean(axis=0)
        face_centers.append( ctr )
        nrm=np.cross(xyz[f_nodes[0]] - ctr, xyz[f_nodes[1]] - ctr)
        if flip:
            nrm *= -1 # should make all normals pointing out
        face_normals.append( nrm )
    cell_ctr=np.mean(face_centers,axis=0)
    outwards = [np.dot(face_centers[i] - cell_ctr,face_normals[i])
                for i in range(len(faces))]
    # note that if the coordinate transformation somehow flips handedness
    # these come out with wrong sign.
    if verbose:
        print("Outward components: "," ".join([f"{comp:.2e}" for comp in outwards])) # expect all positive

    # are face_cells and cell_faces consistent?
    fc_count = np.sum(face_cells==cIdx)
    cf_count = np.sum(cell_faces[cIdx]!=NO_FACE)
    if fc_count != cf_count:
        if verbose:
            print(f"Mismatch fc_count:{fc_count} != cf_count:{cf_count}")
        #import pdb
        #pdb.set_trace()

    return np.all(np.array(outwards)>0)


def mesh_slice(slice_normal, slice_offset, cell_mapping, xyz, face_nodes, face_cells, cell_faces):
    tol = 1e-10
    if cell_mapping is None:
        cell_mapping = np.arange(len(cell_faces))
        
    # identify which faces to slice:
    if 1:
        offset_xyz = np.dot(xyz,slice_normal) - slice_offset
        side_xyz = np.sign(offset_xyz).astype(np.int32)
        # nudge onto slice plane
        side_xyz[ np.abs(offset_xyz)< tol ] = 0
        cmp_xyz = side_xyz<0
        # face_nodes is padded with -1
        cmp_face_0 = cmp_xyz[face_nodes[:,0]]
        cmp_face_same = np.all( (cmp_xyz[face_nodes[:,1:]] == cmp_face_0[:,None]) | (face_nodes[:,1:]<0),
                                axis=1)
        # okay if some of these do not end up getting sliced, say if they are tangent or coplanar to
        # slice. just be fast here
        faces_to_slice = np.nonzero(~cmp_face_same)[0]

    # Filtering step - in the case of warped faces, it's possible to have something like
    # size_xyz=[-1,1,-1,1]. Forcing those nodes to be on the slice plane leads to issues
    # with adjacent faces. Instead, triangulate.
    if 1:
        n_face_orig = faces_to_slice.shape[0]
        
        for fIdx in faces_to_slice:
            nodes = face_nodes[fIdx]
            node_count = np.sum(nodes>=0)
            nodes = nodes[:node_count]
            side_nodes = side_xyz[nodes]

            last_side = side_nodes[0] 
            # Simplify logic below by forcing sides_nodes[0]==-1
            for i in range(node_count):
                if side_nodes[i]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue
            if i!=0: # NB: only works if nodes and side_nodes are trimmed
                nodes = np.roll(nodes,-i)
                side_nodes = np.roll(side_nodes,-i)

            last_side=side_nodes[0]
            crossings=0
            coplanar=0
            for i in range(node_count):
                i_next=(i+1)%node_count
                side_this=side_nodes[i]
                side_next=side_nodes[i_next]
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
                (xyz, face_nodes, face_cells, cell_faces) = mesh_triangulate(fIdx, 
                                                                             xyz, face_nodes, face_cells, cell_faces)

        tri_faces = np.arange(n_face_orig,face_nodes.shape[0])
        if len(tri_faces):
            faces_to_slice = np.concatenate((faces_to_slice,tri_faces))
        
    #print(f"At offset {slice_offset} will slice {len(faces_to_slice)} faces")

    # slices actually have to be per edge, not face.
    sliced_edges={} # (edge small node, edge large node) => new node
    sliced_cells={} # cell index => new cell clipped off of it

    cell_n_faces = (cell_faces!=NO_FACE).sum(axis=1)

    n_face_orig = face_nodes.shape[0]
    n_node_orig = xyz.shape[0]

    for fIdx in faces_to_slice:
        #import pdb
        #pdb.set_trace()
        nodes = face_nodes[fIdx]
        node_count = np.sum(nodes>=0)
        nodes = nodes[:node_count]

        # use side_xyz for fine-grained control
        #cmp_nodes = cmp_xyz[nodes[:node_count]]
        side_nodes = side_xyz[nodes]
        if 1: # Simplify logic below by forcing sides_nodes[0]==-1
            for i in range(node_count):
                if side_nodes[i]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue
            if i!=0:
                # NB: only works if nodes and side_nodes are trimmed
                nodes = np.roll(nodes,-i)
                side_nodes = np.roll(side_nodes,-i)
        
        # Split into two lists of nodes, one for the part of the face with cmp_xyz True (less
        # than offset), and one for the part of the face on or above the slice
        # for the moment don't worry about warped faces that could have more than 2 crossings
        nodes_neg = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_neg=0
        nodes_pos = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_pos=0
        last_side = side_nodes[0] 
        assert last_side!=0,"Should be -1, but definitely not 0"

        coplanar_count=0 # how many edges of the face are on the slice plane

        #if np.any(side_nodes==0):
        #    print(f"{fIdx}: has {np.sum(side_nodes==0)} nodes on slice surface")

        for i in range(node_count):
            i_next=(i+1)%node_count
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
            
            side_this=side_nodes[i]
            side_next=side_nodes[i_next]
            
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
                        nodes_neg[count_neg]=nodes[i] # i, not i_next
                        count_neg+=1
                    elif side_next==1:
                        nodes_pos[count_pos]=nodes[i] # i, not i_next
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
                n_small,n_large = nodes[i],nodes[i_next]
                if n_small>n_large:
                    n_small,n_large = n_large,n_small
                k=(n_small,n_large)

                if k not in sliced_edges:
                    value = np.dot(xyz[nodes[i]],slice_normal) - slice_offset
                    next_value = np.dot(xyz[nodes[i_next]],slice_normal) - slice_offset
                    assert next_value!=value
                    alpha = (0-value)/(next_value-value)
                    new_point = (1-alpha)*xyz[nodes[i]] + alpha*xyz[nodes[i_next]]
                    new_node = xyz.shape[0]
                    xyz = utils.array_append(xyz,new_point)
                    side_xyz = utils.array_append(side_xyz,0) # on the slice by construction
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
        face_nodes = utils.array_append(face_nodes, nodes_pos)
        face_cells = utils.array_append(face_cells, face_cells[fIdx])

        # At this stage, just update the faces
        # and keep them all tied to the original cell, and note that the
        # cell has to be split. Sorting into a new cell handled below.
        # Orientation of nodes_pos should be consistent with nodes_neg
        # so 1s-complement for neigh is correct.

        cIdx = face_cells[fIdx,0] # owner
        assert cIdx>=0
        sliced_cells[cIdx] = -1 # sliced, but no new cell has been created
        cell_faces[cIdx,cell_n_faces[cIdx]]=newFIdx
        cell_n_faces[cIdx]+=1

        cIdx = face_cells[fIdx,1] # neighbor
        if cIdx>=0:
            sliced_cells[cIdx] = -1
            cell_faces[cIdx,cell_n_faces[cIdx]]=~newFIdx # ~ for neighbor reference
            cell_n_faces[cIdx]+=1
            assert cIdx>face_cells[fIdx,0] # the actual invariant for face orientation

    # Sort the clipped cells -- safer not to assume they're sorted. Don't want to
    # assume that everything is maintained upper-triangular and properly sorted
    cells_to_sort = list(sliced_cells.keys())
    cells_to_sort.sort()

    for cIdx in cells_to_sort:
        faces = cell_faces[cIdx,:cell_n_faces[cIdx]]
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
            is_neg = (np.dot(slice_normal,xyz[f_nodes].mean(axis=0))<slice_offset)
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
                on_slice_count = (side_xyz[f_nodes]==0).sum()
                if on_slice_count>2:
                    print(f"WARNING: face {fIdx} has {on_slice_count} nodes on slice, and is part of a cell to be sorted")
                    import pdb
                    pdb.set_trace() # not sure what to do in this case
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

        new_face_edges_bak = list(new_face_edges)
        # Stuff new_face_edges in here by matching
        # The ordering is tedious
        # Could resort to angle-sorting, esp. since the slice plane is handy.

        # Create the new face's nodes:
        if 1:
            new_face_node = np.full(FACE_MAX_NODES,-1)
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
            assert new_face_node[nfn_count-1]==new_face_node[0]
            new_face_node[nfn_count-1]=-1

            
        new_fIdx = face_nodes.shape[0]
        face_nodes = utils.array_append(face_nodes,new_face_node)
                                        
        # new_cIdx always greater than cIdx, so this maintains face-normal invariant.
        cell_face_neg[count_neg]=new_fIdx
        count_neg+=1
        cell_face_pos[count_pos]=~new_fIdx
        count_pos+=1

        # Create new cell and update face_cells and cell_face adjacency info
        new_cIdx = len(cell_faces)
        cell_faces[cIdx,:] = cell_face_neg
        cell_faces = utils.array_append(cell_faces,cell_face_pos)
        cell_mapping = utils.array_append(cell_mapping, cell_mapping[cIdx])
        face_cells = utils.array_append(face_cells,np.array([cIdx,new_cIdx]))

        #import pdb
        #pdb.set_trace()
        
        # Update face_cells. Only have to deal with those in cell_face_pos. Faces in
        # cell_face_neg already point to the correct cIdx. This will overwrite
        # new_cIdx with new_cIdx for the slice face, just fyi
        for signed_fIdx in cell_face_pos:
            if signed_fIdx==NO_FACE: break
            if signed_fIdx<0:
                fIdx=~signed_fIdx
                face_cells[fIdx,1] = new_cIdx
            else:
                fIdx=signed_fIdx
                face_cells[fIdx,0] = new_cIdx
            # Some extra work to maintain the invariant that the face normal points to
            # the higher indexed cell.
            cOwn, cNbr = face_cells[fIdx] # ownership before flipping
            if cNbr>=0 and cNbr<cOwn:
                # Due to changes in cell indexes, face must point in opposite direction
                face_cells[fIdx] = [cNbr,cOwn]
                f_nodes = face_nodes[fIdx]
                f_nodes = f_nodes[f_nodes>=0]
                face_nodes[fIdx,:len(f_nodes)] = f_nodes[::-1] # flip orientation
                
                for f_i in range(CELL_MAX_FACES):
                    if cell_faces[cOwn,f_i] == fIdx:
                        cell_faces[cOwn,f_i] = ~fIdx
                        break
                else:
                    assert False,"Failed to find face to flip"
                for f_i in range(CELL_MAX_FACES):
                    if cell_faces[cNbr,f_i] == ~fIdx:
                        cell_faces[cNbr,f_i] = fIdx
                        break
                else:
                    assert False,"Failed to find face to flip"

    return cell_mapping, (xyz, face_nodes, face_cells, cell_faces)


def mesh_cell_bboxes(xyz,face_nodes,face_cells,cell_faces):
    dense_faces = xyz[face_nodes]
    dense_faces[face_nodes<0,: ] = np.nan
    face_xyz_min = np.nanmin(dense_faces,axis=1)
    face_xyz_max = np.nanmax(dense_faces,axis=1)

    cell_abs_face = np.where(cell_faces<0, ~cell_faces, cell_faces)
    cell_abs_face[cell_faces==NO_FACE] = -1
    
    dense_face_min = face_xyz_min[cell_abs_face]
    dense_face_max = face_xyz_max[cell_abs_face]
    dense_face_min[cell_faces==NO_FACE,:] = np.nan
    dense_face_max[cell_faces==NO_FACE,:] = np.nan
        
    cell_xyz_min = np.nanmin(dense_face_min,axis=1)
    cell_xyz_max = np.nanmax(dense_face_max,axis=1)
    
    n_cells = cell_faces.shape[0]
    cell_bounds = np.zeros((n_cells,6), np.float64)
    cell_bounds[:,::2] = cell_xyz_min
    cell_bounds[:,1::2] = cell_xyz_max

    return cell_bounds

    
# (xyz, face_nodes, face_cells, cell_faces) =
def mesh_triangulate(fIdx, 
                     xyz, face_nodes, face_cells, cell_faces):
    # assume incoming face is convex in the plane, such that triangulation is
    # simple. Not always a great assumption...
    nodes = face_nodes[fIdx]
    nodes=nodes[nodes>=0] # this will be a copy, so it's safe to mutate face_nodes below

    owner,nbr = face_cells[fIdx,:]
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
            face_nodes = utils.array_append(face_nodes, tri_nodes)
            face_cells = utils.array_append(face_cells, face_cells[fIdx])
            cell_faces[owner,n_face_owner]=newFIdx
            n_face_owner+=1

            if nbr>=0:
                cell_faces[nbr,n_face_nbr]=~newFIdx # ~ for neighbor reference
                n_face_nbr+=1

    
    return xyz, face_nodes, face_cells, cell_faces
