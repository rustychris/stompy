"""
WIP, lots of annotations and catering to memoryviews. 
"""
import cython
from cython.cimports.cython import view

import numpy as np
from collections import defaultdict
import time

print("Hello")

FACE_MAX_NODES=40
CELL_MAX_FACES=40
NO_FACE=np.iinfo(np.int32).max

def array_append_int32_1d( A: cython.int[:], b: cython.int) -> cython.int[:]:
    """
    append b to A, where b.shape == A.shape[1:]
    Attempts to make this fast by dynamically resizing the base array of
    A, and returning the appropriate slice.

    if b is None, zeros are appended to A

    """
    # compared to stompy.utils, this one is for memoryviews within cython
    # assume that there have not been any transpositions, we only deal with
    # numeric types.
    assert A.base is not None

    base: cython.int[:]
    i: cython.Py_ssize_t
    
    if A.base.shape[0] == A.shape[0]:
        # make it twice as long as A, and in case the old shape was 0, add 10
        # in for good measure.

        base_array = view.array(shape=(A.shape[0]*2+10,), itemsize=cython.sizeof(cython.int), format="i")
        base = base_array

        for i in range(A.shape[0]):
            base[i] = A[i]
    else:
        base = A.base

    i=A.shape[0]
    base[i]=b
    A = base[:i+1]
    return A

def array_append_int32_2d( A: cython.int[:,:], b: cython.int[:]) -> cython.int[:,:]:
    """
    append b to A, where b.shape == A.shape[1:]
    Attempts to make this fast by dynamically resizing the base array of
    A, and returning the appropriate slice.

    if b is None, zeros are appended to A

    """
    # compared to stompy.utils, this one is for memoryviews within cython
    # assume that there have not been any transpositions, we only deal with
    # numeric types.
    
    # possible that it's not a view, or
    # the base array isn't big enough, or
    # the layout is different and it would just get confusing, or
    # A is a slice on other dimensions, too, which gets too confusing.
    # ignore differences in strides - we just need a growable array

    assert A.base is not None
    assert A.shape[1] == b.shape[0]

    base: cython.int[:,:]
    i: cython.Py_ssize_t
    
    if A.base.shape[0] == A.shape[0]:
        # make it twice as long as A, and in case the old shape was 0, add 10
        # in for good measure.

        base_array = view.array(shape=(A.shape[0]*2+10,A.shape[1]), itemsize=cython.sizeof(cython.int), format="i")
        base = base_array

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                base[i,j] = A[i,j]
    else:
        base = A.base

    i=A.shape[0]
    for j in range(A.shape[1]):
        base[i,j]=b[j]
        
    A = base[:i+1]
    return A

def progress(a,interval_s=5.0,msg="%s"):
    """
    Print progress messages while iterating over a sequence a.

    a: iterable
    interval_s: progress will be printed every x seconds
    msg: message format, with %s format string 
    func: alternate display mechanism.  defaults log.info
    """
    # stripped out backup logic
    L=len(a) # may fail?

    t0=time.time()
    for i,elt in enumerate(a):
        t=time.time()
        if t-t0>interval_s:
            print( msg%("%d/%d"%(i,L)) )
            t0=t
        yield elt


def mesh_duplicate_triples(xyz,face_nodes,face_cells,cell_faces):
    # Brute-force scan for repeated triplets
    # Maybe 30 seconds?
    def norm_triple(abc):
        return tuple(np.sort(abc))
    
    triples_to_faces=defaultdict(list)
    for fIdx in progress(range(face_nodes.shape[0]),msg="Find duplicate triples %s"):
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

def mesh_slice_fix_cells(cells_to_sort: cython.int[:],
                         side_xyz: cython.int[:],
                         cell_n_faces: cython.int[:],
                         n_face_orig: cython.int, 
                         cell_mapping: cython.int[:],
                         xyz: cython.float64[:,:],
                         face_nodes: cython.int[:,:],
                         face_cells: cython.int[:,:],
                         cell_faces: cython.int[:,:]):
    """
    assumes that cells_to_sort is already sorted by index.
    """

    #assert np.all(face_nodes[:,:3]>=0)
    #assert np.all(face_cells[:,0]>=0) # all faces must have an owner. Failing here.

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
            fIdx: cython.int
            if signed_fIdx<0:
                fIdx = ~signed_fIdx
            else:
                fIdx = signed_fIdx
            #if verbose:
            #    print(f"  fIdx={fIdx} ({signed_fIdx})")

            #CY f_nodes = f_nodes[f_nodes>=0]
            #CY f_nodes_sides=side_xyz[f_nodes]
            #CY is_neg = f_nodes_sides.mean()<0
            side_sum: cython.int = 0
            fni: cython.int = 0
            for fni in range(face_nodes.shape[1]):
                if face_nodes[fIdx,fni]<0:
                    break
                side_sum += side_xyz[face_nodes[fIdx,fni]]
                
            f_nodes: cython.int[:] = face_nodes[fIdx,:fni]

            is_neg = side_sum<0
            #if f_nodes_sides.min()<0 and f_nodes_sides.max()>0:
            #    assert False

            #is_neg_coords = (np.dot(slice_normal,xyz[f_nodes].mean(axis=0))<slice_offset)
            #if is_neg != is_neg_coords:
            #    print(f"HEY - side_xyz shows {is_neg=} but from the coordinates {is_neg_coords=}")

            #if verbose:
            #    print(f"  relative to slice_offset: ", np.dot(slice_normal,xyz[f_nodes].mean(axis=0))-slice_offset)
            if is_neg:
                #if verbose:
                #    print(f"    Face {fIdx} is_neg, assign to original cell as {signed_fIdx}")
                cell_face_neg[count_neg]=signed_fIdx
                count_neg+=1
            else:
                #if verbose:
                #    print(f"    Face {fIdx} !is_neg, assign to new cell as {signed_fIdx}")
                cell_face_pos[count_pos]=signed_fIdx
                count_pos+=1

            #if verbose:
            #    print("    Nodes: ",f_nodes)
            #    print("    sides: ",side_xyz[f_nodes])
                
            # since original faces can be tangent/coincident with slice plane,
            # not sufficient to check that it's a new face. 
            if not is_neg: 
                nodeA=nodeB=-1

                #CY on_slice_count = (side_xyz[f_nodes]==0).sum()
                on_slice_count: cython.int = 0
                for f_node in f_nodes:
                    if f_node==0: on_slice_count+=1
                    
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
            new_face_node = np.full(FACE_MAX_NODES,-1, dtype=np.int32)

            # Find an unmatched edge
            for edge_i in range(len(new_face_edges)):
                if new_face_edges[edge_i][0]<0: continue # already used
                new_face_node[:2] = new_face_edges[edge_i]
                new_face_edges[edge_i] = [-1,-1]
                matched_count+=1 # first match is free!
                break
            else:
                print("matched_count suggested we still unmatched edges, but couldn't find it")
                assert False

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
            assert new_face_node[nfn_count-1]==new_face_node[0] # FAILS HERE, maybe bad append code
            new_face_node[nfn_count-1]=-1
            assert (new_face_node>=0).sum() >= 3,"Possible duplicate faces, maybe bad triangulation of warped faces"
            
            new_fIdx = face_nodes.shape[0]
            face_nodes = array_append_int32_2d(face_nodes,new_face_node)
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

        #assert np.all(face_cells[:,0]>=0) # all faces must have an owner
        assert min(face_cells[:,0])>=0 # all faces must have an owner

        # Create new cell and update face_cells and cell_face adjacency info
        new_cIdx = len(cell_faces)
        #CY cell_faces[cIdx,:] = cell_face_neg
        for cfi in range(cell_faces.shape[1]):
            cell_faces[cIdx,cfi] = cell_face_neg[cfi]
        print("About to array_append 1")
        cell_faces = array_append_int32_2d(cell_faces,cell_face_pos)
        print("About to array_append 2")
        cell_mapping = array_append_int32_1d(cell_mapping, cell_mapping[cIdx])
        for i in range(new_faces_count):
            print("About to array_append 3")
            face_cells = array_append_int32_2d(face_cells,np.array([cIdx,new_cIdx],dtype=np.int32))

        #assert np.all(face_cells[:,0]>=0) # all faces must have an owner
        assert min(face_cells[:,0])>=0 # all faces must have an owner
            
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
                #CY face_cells[fIdx] = [cNbr,cOwn]
                face_cells[fIdx,0] = cNbr
                face_cells[fIdx,1] = cOwn
                
                #CY f_nodes = face_nodes[fIdx]
                #CY f_nodes = f_nodes[f_nodes>=0]
                #CY face_nodes[fIdx,:len(f_nodes)] = f_nodes[::-1] # flip orientation

                fni: cython.int = 0
                for fni in range(face_nodes.shape[1]):
                    if face_nodes[fIdx,fni]<0:
                        break
                f_nodes = face_nodes[fIdx,:fni][::-1] # flip orientation
                
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
        #assert mesh_cell_is_closed(cIdx,(xyz,face_nodes,face_cells,cell_faces))
        #assert mesh_cell_is_closed(new_cIdx,(xyz,face_nodes,face_cells,cell_faces))
        #/DBG
    return cell_mapping, xyz, face_nodes, face_cells, cell_faces
