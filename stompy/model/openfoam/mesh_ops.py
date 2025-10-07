import os
import numpy as np
from ... import utils
from ...spatial import triangulate_polygon
from collections import defaultdict

from . import mesh_writer

try:
    from . import mesh_ops_cy
except ModuleNotFoundError:
    print("assume cython not installed. buggy accelerated code not activated")

import time
from fluidfoam.readof import OpenFoamFile

# Did run into an issue with MR long v3 that hit the cell-faces limit of 20
FACE_MAX_NODES=40
CELL_MAX_FACES=40
NO_FACE=np.iinfo(np.int32).max
VSMALL=1e-14

# Throughout,
#    mesh_state = (xyz, face_nodes, face_cells, cell_faces)

array_append=utils.array_append

def load_mesh_state(case_dir,precision=15):
    # Load proc mesh                    
    verbose=False
    meshpath = os.path.join(case_dir,'constant/polyMesh')
    t=time.time()
    # owner.nb_faces, boundary, values, nb_cell
    owner = OpenFoamFile(meshpath, name="owner", verbose=verbose)
    facefile = OpenFoamFile(meshpath, name="faces", verbose=verbose)
    pointfile = OpenFoamFile(meshpath,name="points",precision=precision,
        verbose=verbose)
    neigh = OpenFoamFile(meshpath, name="neighbour", verbose=verbose)

    print(f"Time to read mesh files: {time.time()-t:.3f}s")

    t=time.time()
    face_cells=np.zeros((len(owner.values),2),np.int32)
    face_cells[:,0] = owner.values
    n_interior = len(neigh.values)
    face_cells[:n_interior,1] = neigh.values
    face_cells[n_interior:,1] = -1 # boundary

    xyz=pointfile.values.reshape([-1,3])
    # xyz=np.tensordot(xyz,rot,[-1,0]) # move this to separate function

    # At what point do I need to track cells?
    # ultimately, it's cells that have data

    n_cells = max(owner.nb_cell, neigh.nb_cell)
    cell_faces = np.full( (n_cells, CELL_MAX_FACES), NO_FACE, dtype=np.int32 )
    cell_n_faces = np.zeros( n_cells, np.int32)
    
    for fIdx,cIdx in enumerate(neigh.values):
        cell_faces[cIdx,cell_n_faces[cIdx]] = ~fIdx  # one's complement indicates we're the nbr
        cell_n_faces[cIdx] += 1
    for fIdx,cIdx in enumerate(owner.values):
        cell_faces[cIdx,cell_n_faces[cIdx]] = fIdx  # non-negative indicates we're the owner
        cell_n_faces[cIdx] += 1

    face_nodes=np.full( (facefile.nfaces,FACE_MAX_NODES), -1, np.int32)
    for fIdx in range(facefile.nfaces):
        id_pts = facefile.faces[fIdx]["id_pts"]
        face_nodes[fIdx,:len(id_pts)]=id_pts

    print(f"Time to convert mesh representation: {time.time()-t:.3f}s")
    return (xyz,face_nodes,face_cells,cell_faces)

def mesh_bbox(case_dir,precision=15):
    # Load proc mesh                    
    verbose=False
    meshpath = os.path.join(case_dir,'constant/polyMesh')
    pointfile = OpenFoamFile(meshpath, name="points", precision=precision, verbose=verbose)

    xyz=pointfile.values.reshape([-1,3])
    return np.array([xyz.min(axis=0),xyz.max(axis=0)])

def mesh_rotate(rot,xyz,face_nodes,face_cells,cell_faces):
    xyz=np.tensordot(xyz,rot,[-1,0])
    if np.linalg.det(rot)<0:
        print("WARNING: rotation does not preserve handedness")
    return (xyz,face_nodes,face_cells,cell_faces)

def mesh_face_center_areas(xyz,face_nodes, face_cells, cell_faces):
    # same calc as face_center_area_py, but takes array data
    nfaces=len(face_nodes)
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

def mesh_check_adjacency(mesh_state):
    print("Checking adjacency")
    xyz, face_nodes, face_cells, cell_faces = mesh_state

    for fIdx in range(face_nodes.shape[0]):
        # check owner
        if fIdx not in cell_faces[face_cells[fIdx,0],:]:
            import pdb
            pdb.set_trace()
        if face_cells[fIdx,1]>=0:
            if ~fIdx not in cell_faces[face_cells[fIdx,1],:]:
                import pdb
                pdb.set_trace()
            
    for cIdx in range(cell_faces.shape[0]):
        for signed_fIdx in cell_faces[cIdx,:]:
            if signed_fIdx==NO_FACE:
                break
            elif signed_fIdx<0:
                if cIdx!=face_cells[~signed_fIdx,1]:
                    import pdb
                    pdb.set_trace()
            else:
                if cIdx!=face_cells[signed_fIdx,0]:
                    import pdb
                    pdb.set_trace()
    return True
            
    
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
        #ctr=xyz[f_nodes].mean(axis=0)
        ctr=np.zeros(3,np.float64)
        for n in f_nodes:
            ctr += xyz[n]
        ctr /= len(f_nodes)
        
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
        if np.all(outwards==0):
            import pdb
            pdb.set_trace()

    # are face_cells and cell_faces consistent?
    fc_count = np.sum(face_cells==cIdx)
    cf_count = np.sum(cell_faces[cIdx]!=NO_FACE)
    if fc_count != cf_count:
        if verbose:
            print(f"Mismatch fc_count:{fc_count} != cf_count:{cf_count}")
        #import pdb
        #pdb.set_trace()

    return np.all(np.array(outwards)>0)

def mesh_cell_is_closed(cIdx,mesh_state, four_okay=False):
    """
    four_okay: when called from mesh_clean_duplicate_triples, it's 
    expected that some pairs of faces will get triangulated to
    create duplicate edges. Those appear here as one edge in a cell
    with 4 adjacent faces. specify four_okay=True when this is expected
    and should just print an info message
    """
    # for starters, don't worry about normals, just that each
    # edge should occur twice
    edges=defaultdict(lambda: 0) # key is ordered vertices

    xyz,face_nodes,face_cells,cell_faces = mesh_state
    for fIdx in cell_faces[cIdx]: # signed
        if fIdx==NO_FACE: break
        if fIdx<0:
            fIdx = ~fIdx
        f_nodes=face_nodes[fIdx]
        for i,nA in enumerate(f_nodes):
            nB = f_nodes[i+1]
            if nB<0:
                k=(nA,f_nodes[0])
            else:
                k=(nA,nB)
            if k[0]>k[1]:
                k=(k[1],k[0])
            edges[k] += 1
            if nB<0: break
    is_closed=True
    for k in edges:
        count=edges[k]
        if count==2: continue
        if four_okay and count==4:
            pass # print(f"  {cIdx=} has edge {k} with {edges[k]} occurrences. Hopefully will merge")
        else:
            print(f"Appears {cIdx=} is not closed - edge {k} has {count} occurrences")
            is_closed=False
            import pdb
            pdb.set_trace()
    return is_closed
        

def mesh_slice(slice_normal, slice_offset, cell_mapping, xyz, face_nodes, face_cells, cell_faces):
    #mesh_check_adjacency([xyz,face_nodes,face_cells,cell_faces])
    #assert np.all(face_cells<0, axis=1).sum()==0  # right? but it fails...
    
    #print(f"mesh_slice(slice_normal={slice_normal}, slice_offset={slice_offset})")

    t0=time.time()
    
    tol = 1e-10
    if cell_mapping is None:
        cell_mapping = np.arange(len(cell_faces), dtype=np.int32)
    else:
        assert cell_mapping.shape[0] == len(cell_faces)
        cell_mapping = cell_mapping.astype(np.int32)

    # identify which faces to slice:
    if 1:
        offset_xyz = np.dot(xyz,slice_normal) - slice_offset
        side_xyz = np.sign(offset_xyz).astype(np.int32)
        # nudge onto slice plane
        side_xyz[ np.abs(offset_xyz)< tol ] = 0
        cmp_xyz = side_xyz<0
        # face_nodes is padded with -1

        # okay if some of these do not end up getting sliced, say if they are tangent or coplanar to
        # slice. just be fast here

        # numpy approach (numba approach in alternative implementation)
        cmp_face_0 = cmp_xyz[face_nodes[:,0]]
        cmp_face_same = np.all( (cmp_xyz[face_nodes[:,1:]] == cmp_face_0[:,None]) | (face_nodes[:,1:]<0),
                                axis=1)
        faces_to_slice = np.nonzero(~cmp_face_same)[0]
            
    # Filtering step - in the case of warped faces, it's possible to have something like
    # side_xyz=[-1,1,-1,1]. Forcing those nodes to be on the slice plane leads to issues
    # with adjacent faces. Instead, triangulate.
    if 1:
        n_face_orig = face_nodes.shape[0] # had been wrongly faces_to_slice
        
        for fIdx in faces_to_slice:
            #DBG
            # if fIdx==883911 and np.allclose(slice_offset, -9.775):
            #     # probably about to fail with new_face_edges having just 2 entries.
            #     import pdb
            #     pdb.set_trace()
            #/DBG
            
            nodes = face_nodes[fIdx]
            # node_count = np.sum(nodes>=0)
            for node_i in range(FACE_MAX_NODES):
                if nodes[node_i]<0:
                    node_count=node_i
                    break
            else:
                node_count=FACE_MAX_NODES
                
            nodes = nodes[:node_count]
            side_nodes = side_xyz[nodes]

            # Simplify logic below by forcing sides_nodes[0]==-1
            for i_start in range(node_count):
                if side_nodes[i_start]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue

            last_side=side_nodes[i_start]
            crossings=0
            coplanar=0
            for i in range(node_count):
                i_this=(i_start+i)%node_count
                i_next=(i_this+1)%node_count
                side_this=side_nodes[i_this]
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
                assert np.all(face_nodes[:,:3]>=0)
                assert np.all(face_cells<0, axis=1).sum()==0 

        tri_faces = np.arange(n_face_orig,face_nodes.shape[0])
        if len(tri_faces):
            faces_to_slice = np.concatenate((faces_to_slice,tri_faces))

    t_filtering=time.time() - t0
    #print(f"Filtering: {t_filtering:.3f}s")
    t0=time.time()
    
    if len(faces_to_slice)==0:
        #print("Nothing to slice - early return")
        return cell_mapping, (xyz, face_nodes, face_cells, cell_faces)

    print(f"At offset {slice_offset} will slice {len(faces_to_slice)} faces")

    # slices actually have to be per edge, not face.
    sliced_edges={} # (edge small node, edge large node) => new node
    sliced_cells={} # cell index => new cell clipped off of it

    cell_n_faces = (cell_faces!=NO_FACE).sum(axis=1).astype(np.int32)

    n_face_orig = face_nodes.shape[0]
    n_node_orig = xyz.shape[0]

    assert np.all(face_nodes[:,:3]>=0)
    assert np.all( face_cells<0, axis=1).sum()==0 # failing here

    for fIdx in faces_to_slice:
        nodes = face_nodes[fIdx]
        for node_i in range(FACE_MAX_NODES):
            if nodes[node_i]<0:
                node_count=node_i
                break
        else:
            node_count=FACE_MAX_NODES
        nodes = nodes[:node_count]
        assert len(nodes)>=3 # having some corruption issues...

        # use side_xyz for fine-grained control
        side_nodes = side_xyz[nodes]
        if 1: # Simplify logic below by forcing sides_nodes[0]==-1
            for i_start in range(node_count):
                if side_nodes[i_start]==-1:
                    break
            else:
                # no nodes clearly on neg side, no splice to do
                continue
        
        # Split into two lists of nodes, one for the part of the face with cmp_xyz True (less
        # than offset), and one for the part of the face on or above the slice
        nodes_neg = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_neg=0
        nodes_pos = np.full(FACE_MAX_NODES,-1,dtype=np.int32)
        count_pos=0
        last_side = side_nodes[i_start] 
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
            
            side_this=side_nodes[i_this]
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
                    new_node = xyz.shape[0]
                    xyz = array_append(xyz,new_point)
                    side_xyz = array_append(side_xyz,0) # on the slice by construction
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

        #DBG - make sure we're not creating a degenerate triangle
        # for new_faces in [nodes_neg,nodes_pos]:
        #     tmp_nodes = new_faces[new_faces>=0]
        #     tmp_nodes.sort()
        #     assert np.all(tmp_nodes[1:] - tmp_nodes[:-1])>0
        #     assert(len(tmp_nodes)>=3)
        #/DBG
                
        face_nodes[fIdx]=nodes_neg
        newFIdx = len(face_nodes)
        face_nodes = array_append(face_nodes, nodes_pos)
        face_cells = array_append(face_cells, face_cells[fIdx])

        # At this stage, just update the faces
        # and keep them all tied to the original cell, and note that the
        # cell has to be split. Sorting into a new cell handled below.
        # Orientation of nodes_pos should be consistent with nodes_neg
        # so 1s-complement for neigh is correct.

        cIdx = face_cells[fIdx,0] # owner
        assert cIdx>=0
        sliced_cells[cIdx] = -1 # sliced, but no new cell has been created
        # this is failing with cell_n_faces[cIdx]==20, but dimensions is 20.
        if cell_n_faces[cIdx]==CELL_MAX_FACES:
            raise Exception(f"cell_n_faces[cIdx=={cIdx}]=={CELL_MAX_FACES} - cannot add another face")
        cell_faces[cIdx,cell_n_faces[cIdx]]=newFIdx
        cell_n_faces[cIdx]+=1

        cIdx = face_cells[fIdx,1] # neighbor
        if cIdx>=0:
            sliced_cells[cIdx] = -1
            cell_faces[cIdx,cell_n_faces[cIdx]]=~newFIdx # ~ for neighbor reference
            cell_n_faces[cIdx]+=1
            assert cIdx>face_cells[fIdx,0] # the actual invariant for face orientation

    t_slice_face=time.time() - t0
    print(f"Slice faces: {t_slice_face:.3f}s")

    t0=time.time()

    # This is the most expensive part, e.g. sort cells: 1.5s, clean: 0.38s, slice faces: 0.2s, filter 0.1s
    # split to separate method
    # cython gets sort cells from 1.5 to 0.5
    
    # Sort the clipped cells -- safer not to assume they're sorted. Don't want to
    # assume that everything is maintained upper-triangular and properly sorted
    mesh_slice_fix_cells=mesh_slice_fix_cells_py
    #mesh_slice_fix_cells=mesh_ops_cy.mesh_slice_fix_cells
    
    cells_to_sort = np.array(list(sliced_cells.keys()))
    cells_to_sort.sort()
    new_cell_start_idx=cell_faces.shape[0] # track new cells added during mesh_slice_fix_cells
    cell_mapping, xyz, face_nodes, face_cells, cell_faces = mesh_slice_fix_cells(cells_to_sort.astype(np.int32),
                                                                                 side_xyz.astype(np.int32),
                                                                                 cell_n_faces.astype(np.int32),
                                                                                 n_face_orig,
                                                                                 cell_mapping.astype(np.int32),
                                                                                 xyz,
                                                                                 face_nodes.astype(np.int32),
                                                                                 face_cells.astype(np.int32),
                                                                                 cell_faces.astype(np.int32))


    t_sort_cells=time.time() - t0
    print(f"Sort cells: {t_sort_cells:.3f}s")
    t0=time.time()
    assert np.all(face_cells[:,0]>=0) # all faces must have an owner

    maybe_disconnected=np.concatenate([cells_to_sort,
                                       list(range(new_cell_start_idx,cell_faces.shape[0]))])
    cell_mapping, mesh_state = mesh_split_disconnected_cells(maybe_disconnected,
                                                             cell_mapping,
                                                             (xyz,face_nodes,face_cells,cell_faces))
    assert np.all(mesh_state[1][:,:3]>=0)
    # DBG
    #mesh_check_adjacency(mesh_state)
    #assert np.all(mesh_state[2][:,0]>=0) # all faces must have an owner
    # /DBG

    t_cleanup = time.time() - t0
    print(f"Clean: {t_cleanup:3f}s")
    return cell_mapping, mesh_state



#-------------------

def mesh_slice_fix_cells_py(cells_to_sort, side_xyz, cell_n_faces, n_face_orig, 
                            cell_mapping, xyz, face_nodes, face_cells, cell_faces):
    cells_to_sort.sort()

    assert np.all(face_nodes[:,:3]>=0)
    assert np.all(face_cells[:,0]>=0) # all faces must have an owner. Failing here.

    for cIdx in cells_to_sort:
        verbose=False
        faces = cell_faces[cIdx,:cell_n_faces[cIdx]]
        #if verbose:
        #    print("  Faces: ",faces)
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
            #if verbose:
            #    print(f"  fIdx={fIdx} ({signed_fIdx})")

            f_nodes = face_nodes[fIdx]
            f_nodes = f_nodes[f_nodes>=0]

            f_nodes_sides=side_xyz[f_nodes]
            is_neg = f_nodes_sides.mean()<0
            if f_nodes_sides.min()<0 and f_nodes_sides.max()>0:
                import pdb
                pdb.set_trace()

            #is_neg_coords = (np.dot(slice_normal,xyz[f_nodes].mean(axis=0))<slice_offset)
            #if is_neg != is_neg_coords:
            #    print(f"HEY - side_xyz shows {is_neg=} but from the coordinates {is_neg_coords=}")

            #if verbose:
            #    print(f"  relative to slice_offset: ", np.dot(slice_normal,xyz[f_nodes].mean(axis=0))-slice_offset)
            if is_neg:
                if verbose:
                    print(f"    Face {fIdx} is_neg, assign to original cell as {signed_fIdx}")
                cell_face_neg[count_neg]=signed_fIdx
                count_neg+=1
            else:
                if verbose:
                    print(f"    Face {fIdx} !is_neg, assign to new cell as {signed_fIdx}")
                cell_face_pos[count_pos]=signed_fIdx
                count_pos+=1

            if verbose:
                print("    Nodes: ",f_nodes)
                print("    sides: ",side_xyz[f_nodes])
                
            # since original faces can be tangent/coincident with slice plane,
            # not sufficient to check that it's a new face. 
            if not is_neg: 
                nodeA=nodeB=-1
                on_slice_count = (side_xyz[f_nodes]==0).sum()
                if on_slice_count>2:
                    print(f"WARNING: face {fIdx} has {on_slice_count} nodes on slice, and is part of a cell to be sorted")
                    #import pdb
                    #pdb.set_trace() # not sure what to do in this case
                if (fIdx<n_face_orig) and (on_slice_count<2):
                    if verbose:
                        print(f"    not added. original face and {on_slice_count=}")
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
                    #if verbose:
                    #    print(f"    signed_fIdx non-negative, keeping  order, will add edges {nodeA},{nodeB}")
                    pass
                else:
                    nodeB,nodeA = nodeA,nodeB
                    #if verbose:
                    #    print(f"    signed_fIdx negative, flipping edge order, will add edge {nodeA},{nodeB}")
                    
                new_face_edges.append( [nodeA,nodeB] )

        #if verbose:
        #    print("    All edges:")
        #    for edge in new_face_edges:
        #        print(f"      {edge}")
        #if verbose:
        #    import pdb
        #    pdb.set_trace()
        #    verbose=False
            
        # Stuff new_face_edges in here by matching
        # The ordering is tedious
        # Could resort to angle-sorting, esp. since the slice plane is handy.

        # Create the new face's nodes:
        new_face_edges_count=len(new_face_edges)
        matched_count=0
        new_faces_count=0 # how many new faces are created by the slice of this cell
        while matched_count<new_face_edges_count:
            new_face_node = np.full(FACE_MAX_NODES,-1)

            # Find an unmatched edge
            for edge_i in range(len(new_face_edges)):
                if new_face_edges[edge_i][0]<0: continue # already used
                new_face_node[:2] = new_face_edges[edge_i]
                new_face_edges[edge_i] = [-1,-1]
                matched_count+=1 # first match is free!
                break
            else:
                print("matched_count suggested we still unmatched edges, but couldn't find it")
                import pdb
                pdb.set_trace()

            nfn_count=2
            for edge_count in range(1,len(new_face_edges)): # finite iterations
                to_match = new_face_node[nfn_count-1] # looking for an edge starting with the end of the previous
                assert to_match>=0
                
                for edge_i in range(len(new_face_edges)): # scan all the edges
                    if new_face_edges[edge_i][0]!=to_match:
                        continue # already used or no match
                    # we have a match
                    new_face_node[nfn_count] = new_face_edges[edge_i][1]
                    nfn_count+=1
                    matched_count+=1
                    new_face_edges[edge_i] = [-1,-1]
                    break
                else:
                    # Could have multiple cycles, no need to keep looking in this
                    # loop. Outer loop will pick them up. 
                    break
            assert new_face_node[nfn_count-1]==new_face_node[0]
            new_face_node[nfn_count-1]=-1
            assert (new_face_node>=0).sum() >= 3,"Possible duplicate faces, maybe bad triangulation of warped faces"
            
            new_fIdx = face_nodes.shape[0]
            face_nodes = array_append(face_nodes,new_face_node)
            new_faces_count+=1

            # new_cIdx always greater than cIdx, so this maintains face-normal invariant.
            cell_face_neg[count_neg]=new_fIdx
            count_neg+=1
            cell_face_pos[count_pos]=~new_fIdx
            count_pos+=1

            if matched_count<new_face_edges_count:
                print("HEYO - slice created more than one cycle on a cell")
                #import pdb
                #pdb.set_trace()

        assert np.all(face_cells[:,0]>=0) # all faces must have an owner

        # Create new cell and update face_cells and cell_face adjacency info
        new_cIdx = len(cell_faces)
        cell_faces[cIdx,:] = cell_face_neg
        cell_faces = array_append(cell_faces,cell_face_pos)
        cell_mapping = array_append(cell_mapping, cell_mapping[cIdx])
        for i in range(new_faces_count):
            face_cells = array_append(face_cells,np.array([cIdx,new_cIdx]))

        assert np.all(face_cells[:,0]>=0) # all faces must have an owner
            
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
        #DBG
        assert mesh_cell_is_closed(cIdx,(xyz,face_nodes,face_cells,cell_faces))
        assert mesh_cell_is_closed(new_cIdx,(xyz,face_nodes,face_cells,cell_faces))
        #/DBG
    return cell_mapping, xyz, face_nodes, face_cells, cell_faces

#-------------------






def mesh_split_disconnected_cells(cIdxs,cell_mapping,mesh_state):
    (xyz,face_nodes,face_cells,cell_faces) = mesh_state
    for cIdx in cIdxs:
        fIdxs = []
        for fIdx in cell_faces[cIdx]:
            if fIdx==NO_FACE: break
            if fIdx<0: fIdxs.append(~fIdx)
            else: fIdxs.append(fIdx)
        node_to_faces=defaultdict(list)
        face_to_fi={}
        for fi,fIdx in enumerate(fIdxs):
            face_to_fi[fIdx]=fi
            for nIdx in face_nodes[fIdx]:
                if nIdx<0: break
                node_to_faces[nIdx].append(fIdx)
                
        component=np.full(len(fIdxs),-1)
        n_components=0
        for fi,fIdx in enumerate(fIdxs):
            if component[fi]>=0: continue # already visited
            component[fi]=n_components
            stack=[fIdx] # stack holds faces that have been labeled, but nbrs not queued
            while stack:
                fIdx_trav=stack.pop()
                for n in face_nodes[fIdx_trav]:
                    if n<0: break
                    for nbr_fIdx in node_to_faces[n]:
                        nbr_i=face_to_fi[nbr_fIdx]
                        if component[nbr_i]<0:
                            component[nbr_i]=n_components
                            stack.append(nbr_fIdx)
                        else:
                            assert component[nbr_i] == n_components
            n_components+=1
        if n_components!=1:
            print("EXPERIMENTAL: split_disconnected_cells")
            #import pdb
            #pdb.set_trace()

            # This is rare, and even rarer that there would be more than two components
            # Don't worry about quadratic loops...
            # 1. create a new set of cell_faces rows
            orig_cell_faces = cell_faces[cIdx].copy()
            new_cell_face_rows=[]
            cells_to_check=[]
            for new_comp_i in range(n_components):
                new_cell_face_row = np.full(CELL_MAX_FACES,NO_FACE)
                new_cell_face_count=0
                new_cell_idx=cIdx if new_comp_i==0 else len(cell_faces) # reference cell_faces as it grows
                for fi,comp in enumerate(component):
                    if comp==new_comp_i:
                        signed_fIdx = orig_cell_faces[fi] # cell_faces being mutated, use orig_cell_faces
                        new_cell_face_row[new_cell_face_count] = signed_fIdx
                        new_cell_face_count+=1
                        if new_comp_i>0: # need to change face_cells, too
                            side = 0 if signed_fIdx>=0 else 1
                            assert face_cells[fIdxs[fi],side]==cIdx
                            face_cells[fIdxs[fi],side]=new_cell_idx

                if new_comp_i==0:
                    cell_faces[cIdx] = new_cell_face_row
                else:
                    cell_faces = array_append(cell_faces, new_cell_face_row)
                    cell_mapping = array_append(cell_mapping, cell_mapping[cIdx])
                cells_to_check.append(new_cell_idx)
            # loop over fIdxs and check for any flips
            mesh_maybe_flip_faces(fIdxs, (xyz, face_nodes, face_cells, cell_faces)) # inplace

            #DBG
            for check_cIdx in cells_to_check:
                assert mesh_cell_is_closed(check_cIdx,(xyz,face_nodes,face_cells,cell_faces))
            #mesh_check_adjacency([xyz, face_nodes, face_cells, cell_faces])
            #/DBG
            
    return cell_mapping, (xyz,face_nodes,face_cells,cell_faces)

def mesh_maybe_flip_faces(fIdxs, mesh_state):
    # Some extra work to maintain the invariant that the face normal points to
    # the higher indexed cell.
    xyz,face_nodes,face_cells,cell_faces = mesh_state
    for fIdx in fIdxs:
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


def mesh_face_center_areanormal(face_xyz):
    ctr=face_xyz.mean(axis=0)

    if len(face_xyz)==3:
        area = 0.5*np.cross(face_xyz[1]-face_xyz[0],face_xyz[2]-face_xyz[0])
    else:    
        sumN=np.zeros(3,np.float64)
        sumA=0.0
        sumAc=np.zeros(3,np.float64)

        nPoints=len(face_xyz)
        for pi in range(nPoints):
            p1=face_xyz[pi]
            p2=face_xyz[(pi+1)%nPoints]

            centroid3 = p1 + p2 + ctr
            area_norm = np.cross( p2 - p1, ctr - p1)
            area = np.sqrt(np.sum(area_norm**2))

            sumN += area_norm;
            sumA += area;
            sumAc += area*centroid3;

        ctr = (1.0/3.0)*sumAc/(sumA + VSMALL)
        area = 0.5*sumN
    return ctr, area

# delaunay triangulation
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

    # project to a plane in order to compute delaunay triangulation
    # This is not a common step, okay to be expensive
    face_xyz=xyz[nodes]
    face_center, face_areanormal = mesh_face_center_areanormal(face_xyz)
    face_k_axis = utils.to_unit(face_areanormal)
    face_i_axis = utils.to_unit(np.cross(face_k_axis,face_xyz[0]))
    face_j_axis = utils.to_unit(np.cross(face_k_axis,face_i_axis))
    rot=np.stack([face_i_axis,face_j_axis,face_k_axis],axis=-1)

    # [x0 y0 z0] . [ xi xj xk ]
    # [x1 y1 z1]   [ yi yj yk ]  = 
    #              [ zi zj zk ]
    # Can drop k coordinate, unless we need deviation from the plane
    face_ij = face_xyz @ rot[:,:2]

    # Get it working with external Delaunay - could reimplement if needed
    #simplices = Delaunay(face_ij).simplices
    simplices = triangulate_polygon.constrained_delaunay(face_ij)

    for i_tri in range(len(nodes)-2):
        tri_nodes = np.full(FACE_MAX_NODES,-1,np.int32)
        tri_nodes[:3] = nodes[simplices[i_tri]]

        if i_tri==0:
            face_nodes[fIdx] = tri_nodes
        else:
            newFIdx = len(face_nodes)
            face_nodes = array_append(face_nodes, tri_nodes)
            face_cells = array_append(face_cells, face_cells[fIdx])
            cell_faces[owner,n_face_owner]=newFIdx
            n_face_owner+=1

            if nbr>=0:
                cell_faces[nbr,n_face_nbr]=~newFIdx # ~ for neighbor reference
                n_face_nbr+=1

    return xyz, face_nodes, face_cells, cell_faces


def mesh_renumber_faces(xyz,face_nodes,face_cells,cell_faces):
    # deleted faces have face_nodes[:,0] < 0
    valid_faces = face_nodes[:,0]>=0

    face_map = np.full(face_nodes.shape[0],np.iinfo(np.int32).max)
    face_map[valid_faces] = np.arange(valid_faces.sum())

    # Updates for renumbering
    face_nodes=face_nodes[valid_faces,:]
    face_cells=face_cells[valid_faces,:]

    pos_faces=(cell_faces>=0) & (cell_faces!=NO_FACE)
    neg_faces=(cell_faces<0)
    cell_faces[pos_faces] = face_map[cell_faces[pos_faces]]
    cell_faces[neg_faces] = ~face_map[~cell_faces[neg_faces]]

    assert np.all(face_cells<0, axis=1).sum()==0 # right?
    
    return xyz,face_nodes,face_cells,cell_faces


def mesh_duplicate_triples(xyz,face_nodes,face_cells,cell_faces):
    # Brute-force scan for repeated triplets
    # Maybe 30 seconds?
    def norm_triple(abc):
        return tuple(np.sort(abc))
    
    triples_to_faces=defaultdict(list)
    for fIdx in utils.progress(range(face_nodes.shape[0]),msg="Find duplicate triples %s"):
        nodes = list(face_nodes[fIdx])
        nodes=nodes[:nodes.index(-1)]
        if len(nodes)==3:
            #nodes = np.roll(nodes,-np.argmin(nodes)) # to resolve ambiguity
            # norm_triple now handles the ambiguity
            triples_to_faces[norm_triple(nodes)].append(fIdx)
        else:
            nodes += nodes[:2]
            for offset in range(len(nodes)-2):
                #triple = tuple(nodes[offset:offset+3])
                triple = norm_triple(nodes[offset:offset+3])
                triples_to_faces[triple].append(fIdx)
    # another 5 seconds
    dupe_triples={}
    for k in triples_to_faces:
        if len(triples_to_faces[k])>1:
            dupe_triples[k]=triples_to_faces[k]
    return dupe_triples

def merge_duplicate_triples(fIdxA,fIdxB,*mesh_state):
    """
    Given two triangular faces with the same nodes, combine, removing
    both of them from a shared cell. All operations are in-place, so
    mesh_state is not returned.
    Will leave fIdxB deleted, denoted by face_nodes[fIdxB,0]<0
    """
    xyz,face_nodes,face_cells,cell_faces = mesh_state
    
    cellsA = face_cells[fIdxA] # e.g. array([151425, 152171], dtype=int32)
    cellsB = face_cells[fIdxB] # e.g. array([152171, 152921], dtype=int32)

    # Find the common cell, wire it all up, delete the other face (annoying)
    cell_middle=-1
    for cellsAone in cellsA:
        for cellsBone in cellsB:
            if cellsAone == cellsBone:
                cell_middle = cellsAone
                break
        if cell_middle>=0: break

    if cell_middle<0:
        print("Congruent faces but no shared cell. Maybe a bug?  Maybe irrelevant?  Continuing, but anxious")
        return

    # Remove both faces from cell_middle
    new_cell_faces=[]
    found=0
    for signed_f in cell_faces[cell_middle]:
        if signed_f==NO_FACE: break

        if (signed_f<0):
            f=~signed_f
        else:
            f=signed_f
        if f in (fIdxA,fIdxB):
            #if f==fIdxA:
            #    print(f"Middle cell reference to fIdxA is {signed_f}")
            found+=1
        else:
            new_cell_faces.append(signed_f)

    assert found==2
    new_cell_faces += [NO_FACE,NO_FACE]
    
    # UPDATE middle_cell. it no longer references either face A or faceB
    cell_faces[cell_middle,:len(new_cell_faces)]=new_cell_faces 

    # Delete fIdxB -- will have to renumber
    # Link fIdxA to opposite
    if cellsA[0]==cell_middle:
        A_mid_side=0
    else:
        A_mid_side=1

    if cellsB[0]==cell_middle:
        B_mid_side=0
    else:
        B_mid_side=1

    cell_opp_A = face_cells[fIdxA,1-A_mid_side]
    # B's other cell
    cell_opp_B = face_cells[fIdxB,1-B_mid_side]

    # UPDATE fIdxA, fIdxB, and the other cells
    # point B_opp at its replacement, fIdxA
    flipA=False
    if cell_opp_B>=0: # check it's interior
        if (cell_middle-cell_opp_A) * (cell_opp_B-cell_opp_A)<0:
            flipA=True

        for i,signed_f in enumerate(cell_faces[cell_opp_B]):
            if signed_f<0:
                f=~signed_f
            else:
                f=signed_f
            if f != fIdxB:
                continue
            
            # I *think* that if face_cells[f,1] == cell, then cell_faces[cell,i]=~f
            if A_mid_side==0:
                #print("Adding fIdxA as positive")
                cell_faces[cell_opp_B,i] = fIdxA
            else:
                #print("Adding fIdxA as negative")
                cell_faces[cell_opp_B,i] = ~fIdxA
            break
        else:
            raise Exception("Failed to patch up fIdxB opposite cell")
    else:
        # We don't have to worry about B, but it's possible that fIdxA was pointing
        # *into* A, but now it needs to point out of the domain
        if cell_opp_A>cell_middle:
            flipA = True
            
    if flipA:
        # Flip the sequence of face_nodes[fIdxA,:]
        fA_nodes=face_nodes[fIdxA,:]
        for fA_count,n in enumerate(fA_nodes):
            if n<0: break
        face_nodes[fIdxA,:fA_count] = face_nodes[fIdxA,:fA_count][::-1]

        # Flip the complement on cell_faces[cell_opp_{A,B},i]
        for cell_opp in [cell_opp_A,cell_opp_B]:
            if cell_opp<0: continue
            for i,signed_f in enumerate(cell_faces[cell_opp]):
                if fIdxA==signed_f or fIdxA==~signed_f:
                    cell_faces[cell_opp,i] = ~cell_faces[cell_opp,i]
                    break
            else:
                print("Failed to flipA")
                import pdb
                pdb.set_trace()
        face_cells[fIdxA,1-A_mid_side] = cell_opp_B
        face_cells[fIdxA,A_mid_side] = cell_opp_A
    else:
        # face A, on the side that was facing the middle cell, now faces face B's outside cell
        face_cells[fIdxA,A_mid_side] = cell_opp_B
        assert face_cells[fIdxA,1-A_mid_side] == cell_opp_A
        
    # face B will be discarded, no need to update its cells
    assert not np.any((cell_faces==fIdxB) | (cell_faces==~fIdxB) )
    face_cells[fIdxB,:] = -1
    face_nodes[fIdxB,:] = -1

    # It's possible that cell_opp_A and cell_opp_B are both -1. fIdxA 
    # is no longer owned by anyone, so it should be marked for deletion, too
    if cell_opp_A<0 and cell_opp_B<0:
        assert np.all( face_cells[fIdxA]<0 )
        face_nodes[fIdxA]=-1 # this is specifically what the renumber code looks for
        return
    
    if 1:
        for cell_opp in [cell_opp_A,cell_opp_B]:
            if cell_opp<0: continue
            #valid = mesh_cell_summary(cell_opp,(xyz,face_nodes,face_cells,cell_faces))
            #print(f"Mesh check: {cell_opp=} {valid=}")
            
            # Likely that the triangulation will have technically invalidated some of these cells.
            # the existing mesh_cell_check asserts that faces have a valid normal relative to the
            # cell center. For this usage, might be better to look at the cell-cell vector and
            # and the face normal, assert that they are more parallel than anti-parallel
            #if not valid:
            #    import pdb
            #    pdb.set_trace()

# 90 on the first pass
# 3 (but really just 1) on second pass
def mesh_clean_duplicate_triples(xyz,face_nodes,face_cells,cell_faces):
    if 0: # debug checking
        mesh_state=(xyz,face_nodes,face_cells,cell_faces)
        for cIdx in utils.progress(range(len(cell_faces)), msg="Check cells are closed %s"):
            mesh_cell_is_closed(cIdx,mesh_state)
            
    dupe_triples = mesh_duplicate_triples(xyz,face_nodes,face_cells,cell_faces)
    print(f"{len(dupe_triples)} triples with more than one face on first pass")

    assert np.all(face_cells<0, axis=1).sum()==0 # right?
    
    # Triangulate them
    faces_to_triangulate=[]
    for dupe_triple in dupe_triples:
        faces_to_triangulate += dupe_triples[dupe_triple]
    faces_to_triangulate=np.unique(faces_to_triangulate) # 160
    for fIdx in faces_to_triangulate:
        for cIdx in face_cells[fIdx,:]:
            if cIdx>=0:
                mesh_cell_is_closed(cIdx,(xyz,face_nodes,face_cells,cell_faces))
                
        n_before=face_nodes.shape[0]
        xyz,face_nodes,face_cells,cell_faces = mesh_triangulate(fIdx, 
                                                                xyz, face_nodes, face_cells, cell_faces)
        #print(f"mesh_clean: triangulate fIdx={fIdx} into add'l {np.arange(n_before,face_nodes.shape[0])}")

        #DBG
        for cIdx in face_cells[fIdx,:]:
            if cIdx>=0:
                mesh_cell_is_closed(cIdx,(xyz,face_nodes,face_cells,cell_faces),four_okay=True)
        #/DBG

    assert np.all(face_cells<0, axis=1).sum()==0 # right?
        
    dupe_triples = mesh_duplicate_triples(xyz,face_nodes,face_cells,cell_faces)
    print(f"{len(dupe_triples)} triples with more than one face on second pass")

    for dupe_triple in dupe_triples:
        print(dupe_triple, xyz[dupe_triple[1]])
        for fIdx in dupe_triples[dupe_triple]:
            nodes = face_nodes[fIdx]
            print(f"{fIdx}: {nodes[nodes>=0]}")

    # Ideally we can remove these faces from their shared cell, merge into a single
    # face.
    
    if 1:
        mesh_state=(xyz,face_nodes,face_cells,cell_faces)
        for cIdx in utils.progress(range(len(cell_faces)), msg="Check cells are closed %s"):
            mesh_cell_is_closed(cIdx,mesh_state,four_okay=True)

    if len(dupe_triples)>0:
        for dupe_triple in dupe_triples:
            fIdxs = dupe_triples[dupe_triple]
            assert len(fIdxs)==2
            fIdxA,fIdxB = fIdxs
            merge_duplicate_triples(fIdxA,fIdxB, xyz,face_nodes,face_cells,cell_faces)

        dupe_triples = mesh_duplicate_triples(xyz,face_nodes,face_cells,cell_faces)
        print(f"{len(dupe_triples)} triples with more than one face on third pass")

        xyz,face_nodes,face_cells,cell_faces = mesh_renumber_faces(xyz,face_nodes,face_cells,cell_faces)

    # I'm guessing that we need to renumber here - should have just happened
    assert np.all(face_cells<0, axis=1).sum()==0 # right?
        
    if 1:
        mesh_state=(xyz,face_nodes,face_cells,cell_faces)
        for cIdx in utils.progress(range(len(cell_faces)), msg="Check cells are closed %s"):
            mesh_cell_is_closed(cIdx,mesh_state,four_okay=False)

    return xyz,face_nodes,face_cells,cell_faces
    
def sample_lines(normal, origins, weights_dok, bbox, xyz, face_nodes, face_cells, cell_faces):
    # collapse each face onto a plane defined by the normal
    # calculate signed areas and whether the polygon is valid (not self-intersecting)
    # create spatial index
    # Cast a ray for each origin:
    #  project origin onto same plane
    #  find all faces that it intersects via spatial index

    # How to make this robust in light of nonplanar faces and floating point issues?
    # Option A: walk the cells -- involved and hard to make fast
    #   Is it that hard?  Already have cell bboxes, so reasonably fast to find potential
    #   cells.
    #   
    # Option B: triangulate all faces... helps some, but ...
    # Option C: Start simple -- most of the time the intersections are clean. Each
    #           face entering a cell is paired with a face leaving the same cell.
    #           Calculate a distance from origin for each intersection, sort the faces
    #           by distance. Most of the time that will map to a consistent sequence
    #           of cells. If the sequence is not consistent, could just drop those cells,
    #           include them.
    # 
    
    for row in utils.progress(range(origins.shape[0]),msg="Sample lines %s"):
        for col in range(origins.shape[1]):
            pix = row*origins.shape[1] + col # row-major ordering of pixels
            point=origins[row,col]
            #x=fld_x[col] ; y=fld_y[row]
            (cIdxs, cell_lengths) = sample_line(normal, point, bbox, *mesh_state)
            for cIdx,weight in zip(cIdxs, cell_lengths):
                weights_dok[pix,cIdx] = weight
                
def sample_line(normal, point, bbox, xyz, face_nodes, face_cells, cell_faces):
    # returns cells, lengths
    assert np.all(normal==np.r_[0,0,1])  # for starters
    # bbox is xxyyzz
    hits = (bbox[:,0]<=point[0]) & (bbox[:,1]>=point[0]) & (bbox[:,2]<=point[1]) & (bbox[:,3]>=point[2])
    hits = np.nonzero(hits)[0]
    if len(hits)==0:
        return [],[]

    # Could get a list of faces from the cells, compute intersections for all
    # of them.
    # maybe triangulate them all

def mesh_copy(mesh_state):
    return [a.copy() for a in mesh_state]
