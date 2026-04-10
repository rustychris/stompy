
from ... import utils
from rtree import index
import time, os
from matplotlib.path import Path
from scipy import sparse
import numpy as np
from collections import defaultdict
from . import mesh_ops_nb, mesh_ops
from ...spatial import field
import hashlib, pickle


def mesh_face_2d_index(xyz, face_nodes, face_cells, cell_faces, eps=1e-6):
    """
    Flattens input to just xy coordinates.
    eps: pad face bounds to include exact/near-exact hits
    """
    xy=xyz[:,:2]
    face_xy = np.where(face_nodes[:,:,None]>=0,xy[face_nodes],np.nan)
    face_bounds=np.zeros( (face_nodes.shape[0],4), np.float64)
    face_bounds[:,0::2] = np.nanmin(face_xy[:,:],axis=1) - eps
    face_bounds[:,1::2] = np.nanmax(face_xy[:,:],axis=1) + eps

    # About 5s
    idx = index.Index(interleaved=False)
    for fIdx,xxyy in enumerate(face_bounds):
        idx.insert(fIdx,xxyy)
    return idx

def mesh_face_paths(xyz, face_nodes, face_cells, cell_faces):
    face_paths=[]
    for fidx in utils.progress(range(face_nodes.shape[0])):
        nodes=face_nodes[fidx]
        nodes=nodes[nodes>=0]
        face_paths.append( Path( xyz[nodes,:2]) )
    return face_paths

def sample_line(normal, point, face_index, face_paths, face_centers, face_areas, xyz, face_nodes, face_cells, cell_faces):
    eps=1e-12 # absolute units for perpendicular face area vector
    # returns cells, lengths
    assert np.all(normal==np.r_[0,0,1])  # for starters

    # query face_index
    query = np.array( [point[0],point[0],point[1],point[1]] )
    hits = face_index.intersection(query)
    # Could get a list of faces from the cells, compute intersections for all
    # of them.
    # maybe triangulate them all

    fidx_hits=[]
    z_hits=[]
    exit_hits=[]
    enter_hits=[]
    for fidx in hits:
        # Filter down to the faces that actually intersect
        # Note that contains_point can accept a tolerance, and exact hits are taken to be
        # outside the path
        if face_paths[fidx].contains_point(point[:2]):
            # intersect query vector with face's plane, as defined by
            # face_area and face_center.
            # record z coodinate and also relative orientation
            face_area=face_areas[fidx]
            if np.abs(face_area[2])<=eps:
                continue
            face_center=face_centers[fidx]

            # looking for the z coordinate such that
            # P=[point[0], point[1], z]
            # (P - face_center).dot(face_area) == 0
            # (P[0] - face_center[0]) * face_area[0] + (P[1]-face_center[1])*face_area[1] + (z-face_center[2])*face_area[2] = 0
            # z = face_center[2] - 1/face_area[2] * ( (P[0] - face_center[0]) * face_area[0] + (P[1]-face_center[1])*face_area[1])
            z = face_center[2] - 1/face_area[2] * ( (point[0] - face_center[0]) * face_area[0] + (point[1]-face_center[1])*face_area[1])
            fidx_hits.append(fidx)
            z_hits.append(z)
            if face_area[2]>0:
                exit_hits.append(face_cells[fidx,0])
                enter_hits.append(face_cells[fidx,1])
            else:
                exit_hits.append(face_cells[fidx,1])
                enter_hits.append(face_cells[fidx,0])

    hit_cells=[]
    hit_lengths=[]
                
    # Now the fun begins.
    # There's some slop when it comes to exact tests, can end up with a single hit
    # which doesn't make much sense.
    if len(fidx_hits)>1:
        fidx_hits=np.array(fidx_hits)
        z_hits=np.array(z_hits)
        exit_hits =np.array(exit_hits)
        enter_hits=np.array(enter_hits)
        order = np.argsort(z_hits)
        z_hits= z_hits[order]
        exit_hits =exit_hits[order]
        enter_hits=enter_hits[order]
        fidx_hits=fidx_hits[order] # necessary?
        
        # cell_exits[0],cell_enters[-1] should be -1
        mask_invalid=False # additionally mask out mismatched exit/enters
        if exit_hits[0]==-1 and enter_hits[-1]==-1 and np.all(enter_hits[:-1]==exit_hits[1:]):
            pass # all good
        else:
            print("Weird:")
            # Preemptively check for mismatches that are impossible to solve with swapping.
            cell_counts=defaultdict(lambda: 0)
            for cells in [exit_hits,enter_hits]:
                for c in cells:
                    cell_counts[c] += 1
            for c in cell_counts:
                if cell_counts[c]%2!=0:
                    print("  {cell_counts[c]} of enter/exits for c={c}. Drastic backup")
                    mask_invalid=True
                    break

            if not mask_invalid:
                n_hits=len(exit_hits)
                state=-1
                orig_exit_enter = np.c_[exit_hits,enter_hits]

                for i in range(n_hits):
                    if exit_hits[i] != state:
                        print(f"Was in {state} but exited {exit_hits[i]}. Attempt approx repair")
                        # Scan forward for a hit with exit_hit == state, and shift
                        # On rare occasions we get the enter/exit flipped (maybe the normal was misleading),
                        # so check both sides 
                        for j in range(i,n_hits):
                            if exit_hits[j]==state:
                                # will never happen for j=i, but that's okay. This way we can flip hit i
                                # without extra code.
                                print(f"  Rolling {j} back to {i}: shifts z from {z_hits[j]} to {z_hits[i]}")
                            elif enter_hits[j]==state:
                                # flipped -
                                print(f"  Rolling {j} back to {i}: shifts z from {z_hits[j]} to {z_hits[i]} and flip")
                                enter_hits[j],exit_hits[j] = exit_hits[j],enter_hits[j]
                            else:
                                continue
                            exit_hits[i:j+1] = np.roll(exit_hits[i:j+1],1)
                            enter_hits[i:j+1] = np.roll(enter_hits[i:j+1],1)
                            break
                        else:
                            print("  Extra weird: could not find a matching exit")
                            print(orig_exit_enter)
                            import pdb
                            pdb.set_trace()
                    state = enter_hits[i]
        hit_cells=enter_hits[:-1]
        hit_lengths=np.diff(z_hits)

        if mask_invalid:
            valid = enter_hits[:-1]==exit_hits[1:]
            print(f"Resorted to filtering. Dropping {(~valid).sum()} segments, "
                  +f"{hit_lengths[~valid].sum():.3f} of {hit_lengths.sum()} length")
            hit_cells = hit_cells[valid]
            hit_lengths = hit_lengths[valid]
    return hit_cells,hit_lengths


def sample_lines(normal, origins, weights_dok, *mesh_state):
    """
    For each coordinate in origins, sample cell intersections along a line following normal and going through
    origin point. Add corresponding weights to weights_dok.
    """
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
    t=time.time()
    face_centers,face_areas = mesh_ops_nb.mesh_face_center_areas_nb(*mesh_state)
    print(f"Face centers and areas: {time.time()-t:.3f}s")

    t=time.time()
    face_index = mesh_face_2d_index(*mesh_state)
    print(f"Build face index: {time.time()-t:.3f}s") # 8s
    
    t=time.time()
    face_paths = mesh_face_paths(*mesh_state)
    print(f"Face paths: {time.time()-t:.3f}s") # 2s
    
    for row in utils.progress(range(origins.shape[0]),msg="Sample lines %s"):
        for col in range(origins.shape[1]):
            pix = row*origins.shape[1] + col # row-major ordering of pixels
            point=origins[row,col]
            (cIdxs, cell_lengths) = sample_line(normal, point, face_index, face_paths,
                                                face_centers, face_areas, *mesh_state)
            for cIdx,weight in zip(cIdxs, cell_lengths):
                weights_dok[pix,cIdx] = weight


def precalc_raster_weights_proc_by_sampling_internal(fld, xyz, face_nodes, face_cells, cell_faces):
    """
    Try a sampling based approach
    """
    mesh_state = (xyz,face_nodes,face_cells,cell_faces)

    fld_X,fld_Y = fld.XY() # these are pixel centers
    fld_Z=0*fld_X

    fld_XYZ=np.array([fld_X,fld_Y,fld_Z]).transpose([1,2,0])
    
    raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                        len(cell_faces)),
                                       np.float64)
    sample_lines(np.r_[0,0,1], fld_XYZ, raster_weights, *mesh_state)
    
    return raster_weights

# This is what is generally called by depth_average
def precalc_raster_weights_proc_by_sampling(proc, proc_dir, extents, shape, rot, force):
    # Wrapper with easily serializable arguments, handles caching, wraps the internal
    # method above.
    fld = field.SimpleGrid(extents=extents,F=np.zeros(shape))
    
    hash_input=[extents,shape,rot]
    hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
    cache_dir=os.path.join(proc_dir,"cache")
    cache_fn=os.path.join(cache_dir,
                          f"raster_weights-sampling-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}")
    
    if os.path.exists(cache_fn) and not force:
        print(f"Weights for {proc}: will read from cache file {cache_fn}")
        with open(cache_fn,'rb') as fp:
            return pickle.load(fp)

    print(f"Computing weights for {proc}: will write to cache file {cache_fn}")
            
    #mesh_state = self.read_mesh_state(proc)
    mesh_state = mesh_ops.load_mesh_state(proc_dir,precision=15)
    mesh_state = mesh_ops.mesh_rotate(rot,*mesh_state)

    assert np.all(mesh_state[2]<0, axis=1).sum()==0 # right?

    # No clean triplets for sampling
    
    raster_weights = precalc_raster_weights_proc_by_sampling_internal(fld, *mesh_state)
    if cache_fn is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_fn,'wb') as fp:
            pickle.dump(raster_weights, fp, -1)
    return raster_weights
    
