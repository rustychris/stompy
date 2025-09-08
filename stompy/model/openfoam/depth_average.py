# -*- coding: utf-8 -*-
"""
Approximate depth-average for interFoam results

Created on Fri Jan  3 13:25:56 2025

@author: chrhol4587
"""
import numpy as np
import os, time
import hashlib, pickle
from multiprocessing import Pool
import subprocess
from collections import defaultdict
from numba.typed import List
from numba import jit,njit
from scipy.spatial import KDTree


# import matplotlib.pyplot as plt
#from fluidfoam.readof import getVolumes
#from fluidfoam import readmesh
from ... import utils, memoize
from ...spatial import field

from scipy import sparse
from scipy.spatial import KDTree
import pandas as pd
import scipy.spatial as ss # only for calc_volume=True

from fluidfoam.readof import OpenFoamFile
from fluidfoam import readvector, readscalar

from shapely import geometry, ops, convex_hull

from . import mesh_ops, mesh_ops_nb

class PostFoam:
    """
    Extracting 2D information from Openfoam results.
    
    Focus on extracting velocity and water surface elevation for
    comparison to RAS results. Results are extracted on a regular
    grid, ready to write as a geotiff.
    
    # Point to a case
    pf = PostFoam(sim_dir='/path/to/openfoam_case')

    # Set resolution of output
    pf.set_raster_parameters(dx=0.1)

    # Extract depth-average of 3D velocity
    fldU = pfclip.to_raster('U',t)
    # U is a stompy.spatial.field.SimpleGrid,
    # fldU.F ~ [nrows,ncols, {u,v,w} ]
    # fldU.extents ~ [xmin,xmax,ymin,ymax]

    # Extract water surface elevation (0.5 contour of alpha)
    fld_wse = pf.to_raster('wse',t)
    # fld_wse.F ~ [nrows,ncols]
    
    
    Extracting depth-averaged velocity:
     For each pixel in the output raster
         Find intersecting finite volume cells
         Clip each cell to the pixel boundaries
         Estimate volume by convex hull volume
         result is (volume integral of alpha*U)/(volume integral of alpha)
         
    Caveats: Openfoam does not guarantee that cell faces are planar, which
     implies cells are not guaranteed convex. This can lead to some slop in 
     the clipping process, and some bias in the volume calculation.
     
    Extracting water-surface elevation:
     For each processor:
         Load cell centers (text file, written by openfoam writeMeshObj, see below)
         Find all faces that straddle the contour
         Interpolate the point of intersection between cell centers
         Interpolate between intersection points via griddata
         Compute the union of all intersecting faces and clip the interpolated data
            to this region.
    
    Caveats: Relies on cell centers that were written by openfoam. Does not 
     handle faces between processors. This can leave some missing points that will
     get interpolated over.


    Note: The code here used to implicitly transposes y and z in the output. 
    Now that must be handled via rot matrix specified during instantiona
    """

    sim_dir = None
    verbose = False
    # 'clipping': precise, but slow and not robust. Clips each unstructured
    #    cell to pixel boundaries, uses clipped volume for integration
    # 'bbox': cells with bounding box overlapping pixel are counted, with minor
    #    adjustment for how much the bounding box overlaps the pixel
    # 'sampling': samples cells that intersect the center of a pixel
    raster_mode = 'clipping' # or 'clipping', 'bbox', 'sampling'

    # mesh_slice_method=='numba'
    mesh_slice_method='python'

    # Scan each mesh for repeated triples of nodes, try to eliminate by
    # triangulation and merging.
    clean_duplicate_triples = False

    max_n_procs=1024 # just to bound the search, no inf loops
    precision=15
    current_timename = None
    n_tasks=8 # parallelism for cell-clipping step
    
    # command, with path as needed, to openfoam writeMeshObj
    # writeMeshObj="writeMeshObj"

    # transform OF coordinates to z-up coordinates via
    #  output_z_up=np.tensordot(input_point_xyz,R,[-1,0])
    # Default value is to avoid having to change all of Ben's scripts
    rot=np.array( [[1, 0, 0],
                   [0, 0, 1],
                   [0,-1, 0]])

    _local_to_global = None

    def __init__(self,**kw):
        utils.set_keywords(self,kw)

        self.init_mesh()

    def available_times(self,omit_zero=True):
        times=[]
        
        for t in os.listdir(self.proc_dir(0)):
            try:
                float(t) # make sure it's a number, but keep string representation
            except ValueError:
                continue
            if omit_zero and float(t)==0.0: continue
            times.append(t)
            
        # parsing to float then converting back in read_timestep() may lead to
        # roundoff issues, but so far not a problem.
        times.sort(key=float) # sort numerically, represent as string
        return times
    
    def fmt_timename(self,timename):
        if not isinstance(timename,str):
            return "%g"%timename
        return timename

    def map_local_to_global(self):
        global_centers = self.cell_centers(proc=None)
        self.ncell_global = global_centers.shape[0]
        kdt = KDTree(global_centers)
        self._local_to_global = [None]*self.n_procs
        for proc in range(self.n_procs):
            proc_centers = self.cell_centers(proc=proc)
            dists,self._local_to_global[proc] = kdt.query(proc_centers,distance_upper_bound=1e-5)

    def local_to_global(self,proc):
        if self._local_to_global is None:
            self.map_local_to_global()
        return self._local_to_global[proc]
            
    def read_timestep(self,timename):
        timename=self.fmt_timename(timename)
        if timename==self.current_timename:
            return
        self.current_timename=timename
        vels=[]
        alphas=[]
        for proc in range(self.n_procs):
            print(f"Reading output for processor {proc}")
            proc_dir = self.proc_dir(proc)
            vels.append( readvector(proc_dir, timename, 'U') )
            alphas.append( readscalar(proc_dir, timename, 'alpha.water') )

        self.vels = np.concatenate( vels,axis=1 ).transpose()
        self.alphas = np.concatenate(alphas)

    def read_scalar(self,proc,timename,scalar_name):
        if proc is not None:
            return readscalar(self.proc_dir(proc), timename, scalar_name)
        else:
            # need some global cell information
            self.local_to_global(0) # Trigger mapping during DEV
            result = np.full( self.ncell_global, np.nan)
            for lproc in range(self.n_procs):
                proc_scalar = self.read_scalar(lproc,timename,scalar_name)
                l2g = self.local_to_global(lproc)
                result[l2g] = proc_scalar
            return result
        
    def read_n_procs(self):
        for n_procs in range(self.max_n_procs):
            if not os.path.exists(self.proc_dir(n_procs)):
                break
        if n_procs==0:
            raise Exception(f"No processor output found in {self.proc_dir(0)}")
        self.n_procs = n_procs

    def init_mesh(self):
        print("Reading openfoam mesh")

        self.read_n_procs()
        self._bboxes=None
        self._volumes_centers=None
        
    @property
    def bboxes(self):
        # Get bounding boxes for the whole domain
        if self._bboxes is None:
            bboxes=[]
            for proc in range(self.n_procs):
                bboxes.append(self.get_mesh_bbox(proc))
                             
            # Appears there are no ghost cells,
            # so it's valid to just concatenate results from each processor when 
            # only considering cell-centered quantities.
            self._bboxes = np.concatenate(bboxes, axis=0)
        return self._bboxes
    
    @property
    def volumes(self):
        if self._volumes_centers is None:
            self._volumes_centers = self.calc_volumes_centers()
        return self._volumes_centers[0]

    def calc_volumes_centers(self):
        volumes=[]
        centers=[]
        for proc in range(self.n_procs):
            mesh_state = self.read_mesh_state(proc)
            vol,ctr = mesh_ops.mesh_cell_volume_centers(*mesh_state)
            volumes.append(vol)
            centers.append(ctr)

        # Appears there are no ghost cells,
        # so it's valid to just concatenate results from each processor when 
        # only considering cell-centered quantities.
        volumes = np.concatenate(volumes, axis=0)
        centers = np.concatenate(centers, axis=0)
        return (volumes,centers)

    @memoize.imemoize(lru=100)
    def read_mesh_state(self,proc=None):
        if proc is None:
            case_dir = self.sim_dir
        else:
            case_dir = self.proc_dir(proc)
        mesh_state = mesh_ops.load_mesh_state(case_dir,precision=self.precision)
        mesh_state = mesh_ops.mesh_rotate(self.rot,*mesh_state)

        if self.clean_duplicate_triples:
            mesh_state = mesh_ops.mesh_clean_duplicate_triples(*mesh_state)
        return mesh_state # Note that this does make any cached information within depth_average a fn of rot
    def get_mesh_bbox(self,proc):
        assert proc is not None
        cache_fn = os.path.join(self.proc_dir(proc),"cache/mesh_bbox-v00")
        
        if os.path.exists(cache_fn):
            with open(cache_fn,'rb') as fp:
                bbox=pickle.load(fp)
        else:
            print(f"Read mesh for processor {proc}")
            mesh_state = self.read_mesh_state(proc)
            print(f"Calculate bboxes for processor {proc}")
            bbox = mesh_ops_nb.mesh_cell_bboxes_nb(*mesh_state)
            os.makedirs(os.path.dirname(cache_fn))
            with open(cache_fn,'wb') as fp:
                pickle.dump(bbox, fp, protocol=-1)
        return bbox
    
    def read_mesh(self, read_volumes=False):
        # Get bounding boxes for the whole domain
        bboxes=[]
        volumes=[]
        
        for proc in range(self.n_procs):
            print(f"Reading mesh for processor {proc}, volumes={read_volumes}")
            if read_volumes:
                bbox,vol = self.read_cell_bbox(proc=proc,calc_volume=True)
                volumes.append(vol)
            else:
                bbox = self.read_cell_bbox(proc=proc)
            bboxes.append(bbox)
                
        # Appears there are no ghost cells,
        # so it's valid to just concatenate results from each processor when 
        # only considering cell-centered quantities.
        self._bboxes = np.concatenate(bboxes, axis=0)
        if read_volumes:
            self._volumes =np.concatenate(volumes)
        
    def proc_dir(self,proc):
        return os.path.join(self.sim_dir,f'processor{proc}')

    def read_owner(self,proc=None):
        return self.read_mesh_file("owner",proc)
    
    def read_neighbor(self,proc=None):
        return self.read_mesh_file("neighbour",proc=proc)
    def read_facefile(self,proc=None):
        return self.read_mesh_file("faces",proc=proc)
    def read_pointfile(self,proc=None):
        return self.read_mesh_file("points",proc=proc)
        
    def read_mesh_file(self,filetype,proc=None):
        if proc is None:
            case_dir = self.sim_dir
        else:
            case_dir = self.proc_dir(proc)            
        meshpath = os.path.join(case_dir,"constant/polyMesh")   
        if not os.path.exists(meshpath):
            raise ValueError(
                f"No {meshpath} directory found to load mesh",
                " Please verify the directory of your case.",
            )
        return OpenFoamFile(meshpath, name=filetype, verbose=self.verbose)
                                    
    def set_raster_parameters(self,dx,dy=None,xxyy=None):
        if xxyy is None:
            xxyy = [self.bboxes[:,0].min(),self.bboxes[:,1].max(),
                    self.bboxes[:,2].min(),self.bboxes[:,3].max()]
            
        self.raster_xxyy = xxyy
        if dx<0:
            dx=int((xxyy[1] - xxyy[0])/-dx)
        self.raster_dx = dx
        if dy is None:
            dy=dx
        elif dy<0:
            dy=int((xxyy[3] - xxyy[2])/-dy)
        else:
            dy=dx
        self.raster_dy = dy
        
        self.raster_precalc=None # calculate on-demand

        return dict(dx=dx,dy=dy,xxyy=xxyy,nx=int((xxyy[1]-xxyy[0])/dx),ny=int((xxyy[3]-xxyy[2])/dy))

    def precalc_raster_info(self,fld,force=False):
        if self.raster_mode=='clipping':
            self.precalc_raster_info_clipping(fld,force=force)
        elif self.raster_mode=='bbox':
            self.precalc_raster_info_bbox(fld)
        elif self.raster_mode=='sampling':
            self.precalc_raster_info_sampling(fld)
        else:
            raise Exception(f"Bad raster mode {self.raster_mode}")
            
    def precalc_raster_info_bbox(self,fld):
        # weights is a matrix with columns corresponding to openfoam cells
        # and rows corresponding to output pixels in row-major order
        
        raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                            self.bboxes.shape[0]),
                                           np.float64)
            
        fld_x,fld_y = fld.xy()
        Apixel = fld.dx*fld.dy

        for row in utils.progress(range(fld.shape[0])):
            ymin=fld_y[row]-fld.dy/2
            ymax=fld_y[row]+fld.dy/2

            # Updated handling of coordinates:
            # Here the loops are on raster coordinates
            # what are dy and dx used for? upper bound on projected area of OF cells
            
            # No more implicit z->y change of coodinates
            dy = (np.minimum(ymax,self.bboxes[:,3]) - np.maximum(ymin,self.bboxes[:,2])).clip(0)
            
            for col in range(fld.shape[1]):
                pix = row*fld.shape[1] + col # row-major ordering of pixels
                
                xmin=fld_x[col]-fld.dx/2
                xmax=fld_x[col]+fld.dx/2
                dx = (np.minimum(xmax,self.bboxes[:,1]) - np.maximum(xmin,self.bboxes[:,0])).clip(0)
                # dx and dy now reflect the intersection of the pixel and each cell's bounding box
                # for non-rectangular cells this is an overestimate, the corresponding cells will have
                # too much weight over too large an area. 
                # the too much weight part could possibly be improved by including a cell_volume/bbox_volume 
                # porosity term, but not 100% sure.
                A = dx*dy
                # can volume also improve this? Maybe if we also calculated dz.
                #weights = A[mask]/(fld.dx*fld.dy)
                for cell in np.nonzero(A>0.0)[0]:
                    # alpha-weighting is included per timestep
                    raster_weights[pix,cell] = A[cell]/Apixel
        self.raster_precalc = raster_weights
       
    def precalc_raster_info_clipping(self, fld, force=False):
        # weights is a matrix with columns corresponding to openfoam cells
        # and rows corresponding to output pixels in row-major order

        # fallback to serial... had issues with multiprocessing at this level
        #    tasks = [ (self.proc_dir(proc),fld,self.rot) 
        #              for proc in range(self.n_procs) ]
        #    with Pool(self.n_tasks) as pool: 
        #        results = pool.starmap(precalc_raster_weights_proc_by_faces,tasks,15,force)
        results = [self.precalc_raster_weights_proc_by_faces(proc,fld,force=force)
                   for proc in range(self.n_procs)]

        self.raster_precalc = self.merge_proc_raster_weights(results)
        
    # def get_raster_cache_fn(self,label,fld,proc=None,**kw):
    #     hash_input=[fld.extents,fld.F.shape,self.rot]
    #     hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
    #     
    #     cache_dir=os.path.join(self.proc_dir(proc),"cache")
    #     cache_fn=os.path.join(cache_dir,
    #                           f"raster_weights-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}")
        
    def precalc_raster_weights_proc_by_faces(self,proc,fld,force=False):
        hash_input=[fld.extents,fld.F.shape,self.rot]
        hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
        
        cache_dir=os.path.join(self.proc_dir(proc),"cache")
        cache_fn=os.path.join(cache_dir,
                              f"raster_weights-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}")
        print(f"Weights for {proc}: cache file is {cache_fn}")
    
        if os.path.exists(cache_fn) and not force:
            with open(cache_fn,'rb') as fp:
                return pickle.load(fp)
        
        mesh_state = self.read_mesh_state(proc)
        raster_weights = precalc_raster_weights_proc_by_faces(fld, *mesh_state, mesh_slice_method=self.mesh_slice_method)
        if cache_fn is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_fn,'wb') as fp:
                pickle.dump(raster_weights, fp, -1)
        return raster_weights

    def precalc_raster_info_sampling(self, fld, force=False):
        # weights is a matrix with columns corresponding to openfoam cells
        # and rows corresponding to output pixels in row-major order
        results = [self.precalc_raster_weights_proc_by_sampling(proc,fld,force=force)
                   for proc in range(self.n_procs)]

        self.raster_precalc = self.merge_proc_raster_weights(results)

    def precalc_raster_weights_proc_by_sampling(self,proc,fld,force=False):
        hash_input=[fld.extents,fld.F.shape,self.rot]
        hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
        
        cache_dir=os.path.join(self.proc_dir(proc),"cache")
        cache_fn=os.path.join(cache_dir,
                              f"raster_weights-sampling-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}")
        print(f"Weights for {proc}: cache file is {cache_fn}")
    
        if os.path.exists(cache_fn) and not force:
            with open(cache_fn,'rb') as fp:
                return pickle.load(fp)
        
        mesh_state = self.read_mesh_state(proc)
        bbox = self.get_mesh_bbox(proc)
        raster_weights = precalc_raster_weights_proc_by_sampling(fld, bbox, *mesh_state)
        if cache_fn is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_fn,'wb') as fp:
                pickle.dump(raster_weights, fp, -1)
        return raster_weights

    def merge_proc_raster_weights(self,results):
        raster_weights = None

        for proc,proc_raster_weights in enumerate(results):
            print(f"Assembling from processor {proc}")
            # Want to stack these left-to-right
            if raster_weights is None:
                raster_weights = proc_raster_weights
            else:
                raster_weights = sparse.hstack( (raster_weights,proc_raster_weights) )
        return raster_weights
    
    def to_raster(self, variable, timename, force_precalc=False, cache=True):
        n_components=None
        if variable=='U':
            n_components = 3
        fld = field.SimpleGrid.zeros(extents=self.raster_xxyy,
                                     dx=self.raster_dx,dy=self.raster_dy, 
                                     n_components=n_components)

        if cache:
            hash_input=[fld.extents,fld.F.shape,self.rot]
            hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
            cache_dir=os.path.join(self.sim_dir,"cache")
            cache_fn=os.path.join(cache_dir,
                                  f"{variable}-{timename}-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}.tif")
            if os.path.exists(cache_fn):
                return field.GdalGrid(cache_fn)
        
        if variable=='wse':
            fld = self.raster_wse(timename)
        else:
            self.read_timestep(timename)

            if self.raster_precalc is None:
                self.precalc_raster_info(fld,force=force_precalc)

            raster_weights = self.raster_precalc

            fld_x,fld_y = fld.xy()

            # for sanity, process each component in sequence (tho memory inefficient)
            for comp in range(n_components or 1):

                if variable=='U':
                    # this 
                    num = raster_weights.dot(self.alphas*self.vels[:,comp])
                    den = raster_weights.dot(self.alphas)

                    # seems like division by zero should be handled by 'divide', but
                    # so far it trips 'invalid'
                    with np.errstate(divide='ignore',invalid='ignore'):
                        u = num/den
                        u[ np.abs(den)<1e-10 ] = np.nan

                    fld.F[:,:,comp] = u.reshape( (fld.F.shape[0],fld.F.shape[1]) )                
                elif variable=='depth_bad':
                    # this field and inv_depth_bad are not good approximations
                    # and left here only for comparison to other approaches.
                    # measure up from the bed, which is assumed flat
                    # This will not be a good estimate when the bed is not flat,
                    # non-simple, or the domain doesn't fill the pixel.
                    fld.F[:,:] = raster_weights.dot(self.alphas).reshape( fld.F.shape )
                elif variable=='inv_depth_bad':
                    # More likely that the top of the domain is flat than the
                    # bed being flat, but still has errors when the freesurface
                    # intersects the walls.
                    fld.F[:,:] = raster_weights.dot(1.0-self.alphas).reshape( fld.F.shape )
                else:
                    raise Exception(f"Not ready for other fields like {variable}")

        if cache:
            cache_dir = os.path.dirname(cache_fn)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            fld.write_gdal(cache_fn)

        return fld

    def rot_name(self):
        # summarize rotation matrix for cache naming
        # not worrying about the specifics too much
        rot = self.rot
        if np.all( (rot==0) | (np.abs(rot)==1)):
            s=""
            for row in range(3):
                for col,comp in enumerate('xyz'):
                    if rot[row,col]==1:
                        s+=comp
                    elif rot[row,col]==-1:
                        s+=comp.upper()
        else:
            s = hashlib.md5(pickle.dumps(rot)).hexdigest()
        return s
                
    # Contour-based approach for WSE
    # First get it working for one proc, then figure out how to glue them
    # together (or have openfoam do the merge)
    proc=0
    def cell_centers(self,proc=None):
        # Assumes writeMeshObj output is available,
        # e.g. in the case directory, writeMeshObj -constant -time none
        # that puts meshCellCentres_constant.obj in the case directory

        # for global mesh, no guarantee this mesh is in the same order
        # as reading the subdomains and concatenating

        base=f'meshCellCentres_constant_{self.rot_name()}.obj'
        
        if proc is None:
            center_fn = os.path.join(self.sim_dir,base)
        else:
            center_fn = os.path.join(self.proc_dir(proc),base)

        if not os.path.exists(center_fn):
            df = self.compute_cell_centers_py(proc)
            df.to_csv(center_fn,sep=' ',header=False,index=False,float_format="%.15f")

        df = pd.read_csv(center_fn,sep=r'\s+',names=['type','x','y','z'])
        points = df[ ['x','y','z'] ].values
        return points

    def face_centers(self,proc=None):
        # see cell_centers
        base = f'meshFaceCentres_constant_{self.rot_name()}.obj'
        if proc is None:
            center_fn = os.path.join(self.sim_dir,base)
        else:
            center_fn = os.path.join(self.proc_dir(proc),base)

        assert False,"Needs to switch to using mesh_state"
        if not os.path.exists(center_fn):
            if not self.compute_cell_centers(center_fn):
                raise Exception(f"Face center file {center_fn} not found. Generate with writeMeshObj")
        df = pd.read_csv(center_fn,sep=r'\s+',names=['type','x','y','z'])
        points = df[ ['x','y','z'] ].values
        return points
    
    def compute_cell_centers_py(self,proc):
        mesh_state = self.read_mesh_state(proc)

        vols,ctrs = mesh_ops.mesh_cell_volume_centers(*mesh_state)

        df=pd.DataFrame()
        df['t']=["v"] * len(ctrs)
        df['x']=ctrs[:,0]
        df['y']=ctrs[:,1]
        df['z']=ctrs[:,2]
        return df
                  
    def contour_points_proc(self, scalar, contour_value, proc):
        centers = self.cell_centers(proc)

        print(f"proc:{proc} centers x:{centers[:,0].min()} to {centers[:,0].max()}")
        print(f"                    y:{centers[:,1].min()} to {centers[:,1].max()}")
        print(f"                    z:{centers[:,2].min()} to {centers[:,2].max()}")
        
        #owner =self.read_owner(proc)
        #nbr   =self.read_neighbor(proc)
        mesh_state = self.read_mesh_state(proc)
        xyz,face_nodes,face_cells,cell_faces = mesh_state
        internal = face_cells[:,1]>=0
        owner=face_cells[internal,0]
        nbr  =face_cells[internal,1]
        
        # n_face_int = len(nbr.values) # assuming that boundary faces are omitted from nbr
        # per-internal-face values
        
        scalar_owner = scalar[owner]
        scalar_nbr = scalar[nbr]
        
        # (1-frac)*scalar_owner + frac*scalar_nbr = contour_value
        # frac*(scalar_nbr-scalar_owner) = contour_value - scalar_owner
        sel = ( (np.minimum(scalar_owner,scalar_nbr)<=contour_value)
               &(np.maximum(scalar_owner,scalar_nbr)>=contour_value) )
        
        frac = (contour_value-scalar_owner[sel]) / (scalar_nbr[sel] - scalar_owner[sel])
        
        pnts = ( (1-frac[:,None])*centers[owner[sel]]
                +frac[:,None]*centers[nbr[sel]])
        
        # Also need selected faces to clip interpolation
        #facefile = self.read_facefile(proc)
        #pointfile = self.read_pointfile(proc)
        #points_xyz=pointfile.values.reshape([-1,3])
        #points_xyz=np.tensordot(points_xyz,self.rot,[-1,0]) # ROTATE
    
        face_pnts=[]
        internal_faces = np.nonzero(internal)[0]
        # sel is only over the internal faces
        for int_fIdx in np.nonzero(sel)[0]:
            # map back to face index over all faces
            fIdx = internal_faces[int_fIdx]
            # id_pts = facefile.faces[fIdx]["id_pts"][:]
            f_nodes = face_nodes[fIdx]
            f_nodes = f_nodes[f_nodes>=0]
            face_pnts.append( xyz[f_nodes,:] )
        
        return pnts,face_pnts
    
    def contour_points(self, timename, contour_value=0.5, scalar_name='alpha.water'):
        if 0: # per-subdomain
            pntss=[]
            facess=[]
            for proc in range(self.n_procs):
                scalar = readscalar(self.proc_dir(proc), timename, scalar_name)
                pnts,face_pnts = self.contour_points_proc(scalar, 0.5, proc=proc)
                pntss.append(pnts)
                facess += face_pnts
            return np.concatenate(pntss,axis=0), facess
        else: # global - should avoid missing patches
            scalar = self.read_scalar(None,timename,scalar_name)
            pnts,face_pnts = self.contour_points_proc(scalar, 0.5, proc=None)
            return pnts, face_pnts
    
    
    def raster_wse(self,timename):
        timename=self.fmt_timename(timename)
        # interpolate
        pnts,faces = self.contour_points(timename,contour_value=0.5, scalar_name='alpha.water')
        fld_pnts = field.XYZField(X=pnts[:,:2], F=pnts[:,2])

        # In some unfortunate situations can end up with duplicate points. Filter out via KDTree
        tol = 0.001*self.raster_dx

        for finite in range(5): # really shouldn't take multiple iterations
            kdt = KDTree(fld_pnts.X)
            remapped = np.full(fld_pnts.X.shape[0],-1) 
            pairs = kdt.query_pairs(tol) # set of pairs (i,j) i<j
            if pairs:
                for i,j in pairs:
                    # Average i and j, make j point to i
                    # Make both canonical:
                    i_can=i
                    while remapped[i_can]>=0:
                        i_can=remapped[i_can]
                    j_cans=[j]
                    while remapped[j_cans[-1]]>=0:
                        j_cans.append(remapped[j_cans[-1]])
                    if i_can==j_cans[-1]: continue
                    can_dist = utils.mag(fld_pnts.X[i_can] - fld_pnts.X[j_cans[-1]])
                    if can_dist>tol: continue
                    j_can=j_cans[-1]
                    # ideally have counts so this can be weighted.
                    # Moving i_can around causes problems.
                    #fld_pnts.X[i_can] = 0.5*(fld_pnts.X[i_can] + fld_pnts.X[j_can])
                    fld_pnts.F[i_can] = 0.5*(fld_pnts.F[i_can] + fld_pnts.F[j_can])
                    fld_pnts.X[j_can] = np.nan
                    fld_pnts.F[j_can] = np.nan
                    for j_can in j_cans:
                        remapped[j_can] = i_can

                valid = remapped<0
                print(f"Combining { (~valid).sum() } point pairs that were too close together (iteration {1+finite})")
                fld_pnts.X = fld_pnts.X[ valid ]
                fld_pnts.F = fld_pnts.F[ valid ]
            else:
                break

        # clean: openfoam domains tend to have a lot of regular grid, boundaries of which can
        # run into floating point issues. Try to clean out problem triangles.
        fld_wse = fld_pnts.to_grid(bounds=self.raster_xxyy, dx=self.raster_dx,dy=self.raster_dy,
                                   clean=True)
        
        if 1: # trim raster to valid parts of the slice.
            face_polys = []
            for i,pnts in enumerate(faces):
                # convex hull gets around occasional warped, invalid polygons.
                geom = convex_hull(geometry.MultiPoint(pnts[:,:2]))
                if geom.geom_type=='Polygon':
                    # vertical cells end up with a LineString here that infects
                    # the union.
                    face_polys.append(geom)
                    
            face_poly = ops.unary_union( face_polys )
            
            in_bounds = fld_wse.polygon_mask(face_poly)
            fld_wse.F[~in_bounds]=np.nan
        
        return fld_wse


def clip_cell_edges(edges, cut_normal, cut_point):
    cut_dist = cut_normal.dot(cut_point)
    clipped_edges=[]
    new_points=[]
    for edge in edges:
        a_dist = cut_normal.dot(edge[0]) - cut_dist
        b_dist = cut_normal.dot(edge[1]) - cut_dist
        if a_dist<=0 and b_dist<=0:
            clipped_edges.append(edge)
        elif a_dist>0 and b_dist>0:
            continue
        else:
            # the fun part
            if a_dist<=0:
                # Replace b
                frac = (-a_dist)/ (b_dist-a_dist)
                b_pnt=edge[0] + frac*(edge[1] - edge[0])
                edge[1] = b_pnt # mutation!
                new_points.append(b_pnt)
            else:
                # Replace a
                frac = (-b_dist)/ (a_dist-b_dist)
                a_pnt=edge[1] + frac*(edge[0] - edge[1])
                edge[0] = a_pnt # mutation!
                new_points.append(a_pnt)
            clipped_edges.append(edge)

    if new_points:
        if len(new_points)>3: # RH: reorder >3 points so they are ordered in a ring. Unnecessary for <=3
            new_points=np.array(new_points)

            # Orient new points, add edges
            # For current approach, CCW vs CW doesn't matter, they just
            # need to be convex, no intersecting edges
            # project to a 2D plane perpendicular to the cut normal
            # use cross products to find two vectors perpendicular to 
            # cut_normal.
            eye=np.eye(3)
            bases=np.cross(cut_normal, eye)
            # each row of the output is cut_normal x row of eye(3)
            base_x = eye[np.argmax( (bases**2).sum(axis=1) ),:]
            base_y = np.cross(cut_normal, base_x)
            bases=np.array([base_x,base_y]).T
            new_point_proj = (new_points - new_points.mean(axis=0)).dot(bases)
            angles = np.arctan2(new_point_proj[:,1],new_point_proj[:,0])
            order = np.argsort(angles)
            new_points=new_points[order,:]
        for a,b in utils.circular_pairs(new_points):
            clipped_edges.append( [a,b] )
    return clipped_edges

def volume_cell_edges(edges):
    cell_vertices = np.unique(np.array(edges).reshape([-1,3]), axis=0)
    if len(cell_vertices)<4:
        # This can happen with warped input cells. Currently no robust
        # handling, but it appears to be rare. Likely more common is
        # the case where a warped or nonconvex polyhedron silently
        # succeeds but has a volume that is biased high.
        print("Degenerate clipped cell")
        return 0.0
    try:
        return ss.ConvexHull(cell_vertices).volume
    except ss.QhullError:
        print("Degenerate clipped cell (qhull error)")
        return 0.0

# volume_cell_edges is a substantial time sink
# Options for a non-qhull approach:
#   change the representation to track faces in addition to edges,
#     and use the cell volume code.
#   reconstruct faces from the edges, then use the cell volume code
#   write a python convex hull code (!)
# How hard is it to reconstruct the faces?
#  Say it gets processed into nodes and edges that reference the nodes,
#  each each must be adjacent to two faces
#  at each vertex, the adjacent edges can be enumerated. Need to establish
#  the CCW order of the edges when looking from outside the cell towards the
#  inside of the cell.
#  For each edge, get a unit vector in that direction.
#  if the edges are not all coplanar, I think that gives a good normal for
#  a plane on which to project the vectors and sort order. If the edges
#  are close to coplanar it's not going to work. Might happen in some octree
#  cells
# HERE: trying to figure out how to orient edges at a vertex.
#  with that it becomes possible to walk the edges, construct faces, which
#  can then be passed to volume code

    
def clipped_volume(edges, xmin=None, xmax=None, ymin=None, ymax=None,
                   zmin=None, zmax=None):
    # Clip the given cell to the specified ranges. Coordinates are native
    # openfoam, before any mapping of z to y
    edges = np.array(edges) # avoid any potential mutation issues
    if xmin is not None:
        edges = clip_cell_edges(edges,np.r_[-1,0,0], np.r_[xmin,0,0])
    if xmax is not None:
        edges = clip_cell_edges(edges,np.r_[1,0,0], np.r_[xmax,0,0])
    if ymin is not None:
        edges = clip_cell_edges(edges,np.r_[0,-1,0], np.r_[0,ymin,0,0])
    if ymax is not None:
        edges = clip_cell_edges(edges,np.r_[0,1,0], np.r_[0,ymax,0])
    if zmin is not None:
        edges = clip_cell_edges(edges,np.r_[0,0,-1], np.r_[0,0,zmin])
    if zmax is not None:
        edges = clip_cell_edges(edges,np.r_[0,0,1], np.r_[0,0,zmax])
    if len(edges)==0:
        return 0.0
    else:
        return volume_cell_edges(edges)
    
def precalc_raster_weights_proc_by_faces(fld, xyz, face_nodes, face_cells, cell_faces, mesh_slice_method='numba'):
    """
    This version should be much faster. It avoids qhull, slices faces rather than clipping
    cells, and uses a padded array mesh representation that's faster for both python and
    numba
    """
    #  base operation is to slice the mesh
    
    #  Slice by y=n*dy planes and x=n*dx planes
    #  compute volumes and centers
    #  assign to pixel via centers
    #  compile into matrix

    mesh_state = (xyz,face_nodes,face_cells,cell_faces)
    n_cells_orig = len(cell_faces)
    
    fld_x,fld_y = fld.xy() # I think these are pixel centers

    # unclear whether numba version is sketch or not.
    if mesh_slice_method=='numba':
        slicer = mesh_ops_nb.mesh_slice_nb
    else:
        slicer = mesh_ops.mesh_slice 

    if 0: # debugging checks
        print("Checking cells before any operations")
        for cIdx in utils.progress(range(len(mesh_state[3])), func=print):
            if not mesh_ops.mesh_check_cell(cIdx,False,mesh_state):
                import pdb
                pdb.set_trace()
                mesh_ops.mesh_check_cell(cIdx,True,mesh_state)
                
        print("Check complete")
    assert np.all(mesh_state[1][:,:3]>=0)

    #DBG
    # print("Check 265980 and 271032 for a common triplet of nodes")
    # # looking for nodes ..., 116666, 152228, 117240, ...
    # import pdb
    # pdb.set_trace()
    #/DBG
    
    cell_mapping=None
    for col in utils.progress(range(fld.shape[1])):
        if col==0: continue
        xmin=fld_x[col]-fld.dx/2
        xmax=fld_x[col]+fld.dx/2
        cell_mapping, mesh_state = slicer(np.r_[1,0,0], xmin, cell_mapping, *mesh_state)
        print(f"col={col} face_count={mesh_state[1].shape[0]}")
        assert np.all(mesh_state[1][:,:3]>=0) # col 232 is leaving this corrupt.

    if 0: # debugging checks
        print("Checking cells")
        for cIdx in utils.progress(range(len(mesh_state[3])), func=print):
            if not mesh_ops.mesh_check_cell(cIdx,False,mesh_state):
                print(f"cIdx={cIdx} failed check")
                mesh_ops.mesh_check_cell(cIdx,True,mesh_state)
                raise Exception(f"cIdx={cIdx} failed check")
        print("Check complete")
    if 1: # and check that all faces have at least 3 nodes
        assert np.all(mesh_state[1][:,:3]>=0)
    
    # Slice the mesh so all cells are in exactly one pixel
    print("Slicing by row")
    for row in utils.progress(range(fld.shape[0])):
        print("Row:", row)
        if row==0: continue
        ymin=fld_y[row]-fld.dy/2
        ymax=fld_y[row]+fld.dy/2
        # if row==5: 
        #     import pdb
        #     print("DEBUG: about to hit 2-node face bug")
        #     pdb.set_trace()
        cell_mapping, mesh_state = slicer(np.r_[0,1,0], ymin, cell_mapping, *mesh_state)
        print(f"row={row} face_count={mesh_state[1].shape[0]}")

    if 0: # debugging checks
        print("Checking cells after all slicing")
        for cIdx in utils.progress(range(len(mesh_state[3])), func=print):
            assert mesh_ops.mesh_check_cell(cIdx,False,mesh_state)
        print("Check complete")
    if 1: # and check that all faces have at least 3 nodes
        assert np.all(mesh_state[1][:,:3]>=0)

    print("Calculate volume")
    volumes,centers = mesh_ops_nb.mesh_cell_volume_centers_nb(*mesh_state)

    print("Assemble matrix")
    raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                        n_cells_orig),
                                       np.float64)
        
    # check bounds, then compute clipped volume.
    Apixel = fld.dx*fld.dy

    # OPT: would be faster to instead map each cell center to a pixel, loop
    # over cells
    for row in utils.progress(range(fld.shape[0])):
        ymin=fld_y[row]-fld.dy/2
        ymax=fld_y[row]+fld.dy/2
        for col in range(fld.shape[1]):
            pix = row*fld.shape[1] + col # row-major ordering of pixels
            
            xmin=fld_x[col]-fld.dx/2
            xmax=fld_x[col]+fld.dx/2
         
            # No more implicit transposition
            sel = ( (centers[:,0]<xmax)&(centers[:,0]>xmin) &
                    (centers[:,1]<ymax)&(centers[:,1]>ymin) )
            
            for cIdx in np.nonzero(sel)[0]:
                # alpha-weighting is included per timestep
                raster_weights[pix,cell_mapping[cIdx]] = volumes[cIdx]/Apixel
                
    return raster_weights

def precalc_raster_weights_proc_by_sampling(fld, bbox, xyz, face_nodes, face_cells, cell_faces):
    """
    Try a sampling based approach
    """
    mesh_state = (xyz,face_nodes,face_cells,cell_faces)

    fld_X,fld_Y = fld.XY() # these are pixel centers
    fld_Z=0*fld_X

    fld_XYZ=np.array([fld_X,fld_Y,fld_Z]).transpose([1,2,0])
    
    print("Assemble matrix")
    raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                        n_cells_orig),
                                       np.float64)
    mesh_ops.sample_lines(np.r_[0,0,1],fld_XYZ, raster_weights, bbox, *mesh_state)
    
    return raster_weights


def cell_as_edges(cIdx, cell_faces, facefile, xyz):
    # translate a set of faces into a set of edges
    edge_nodes=set()
    for fIdx in cell_faces[cIdx]:
        face_nodes = facefile.faces[fIdx]["id_pts"][:]
        for i in range(len(face_nodes)-1):
            a=face_nodes[i]
            b=face_nodes[i+1]
            if a<b:
                edge_nodes.add( (a,b) )
            else:
                edge_nodes.add( (b,a) )
        a=face_nodes[0]
        if a<b:
            edge_nodes.add( (a,b) )
        else:
            edge_nodes.add( (b,a) )
        
    # Convert the edges from point index to 3D point:
    edges = [xyz[list(edge_node)] for edge_node in edge_nodes]
    return edges

@njit
def calc_face_center(face):
    VSMALL=1e-14
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
            area = np.sqrt(np.sum(area_norm**2)) # utils.mag(area_norm)

            sumN += area_norm;
            sumA += area;
            sumAc += area*centroid3;

        ctr = (1.0/3.0)*sumAc/(sumA + VSMALL)
        area = 0.5*sumN
    return ctr,area

# @njit
def face_center_area_py(face_id_pts, xyz):
    """
    face_id_pts: list( array(point ids for a face), ...)
    """
    VSMALL=1e-14
    # 20 s for one domain w/o numba
    # 4s with numba just in calc_face_center
    # 6s with njit on face_center_area_py, calling calc_face_center (?!)
    # 5.5s with njit on face_center_area_py and inline calc_face_center
    nfaces=len(face_id_pts)
    face_ctr=np.zeros((nfaces,3),np.float64)
    face_area=np.zeros((nfaces,3),np.float64)

    for fIdx in range(nfaces):
        face_nodes = face_id_pts[fIdx]
        face = xyz[face_nodes]

        if 0:
            ctr,area = calc_face_center(face)
        else:
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
                    area = np.sqrt(np.sum(area_norm**2)) # utils.mag(area_norm)

                    sumN += area_norm;
                    sumA += area;
                    sumAc += area*centroid3;

                ctr = (1.0/3.0)*sumAc/(sumA + VSMALL)
                area = 0.5*sumN
        face_ctr[fIdx] = ctr
        face_area[fIdx] = area

    return face_ctr, face_area


def cell_center_volume_py(facefile, xyz, owner, neigh):
    """
    facefile:
    xyz: pointfile.values.reshape([-1,3])
    owner: [Nfaces] index of face's owner cell
    neigh: [NInternalFaces]  index of faces's neighbor cell

    e.g.
    ctrs,vols = cell_center_volume_py(facefile, pointfile.values.reshape([-1,3]), owner.values, neigh.values)
    """
    # replicate cell center calculation from openfoam-master/src/meshTools/primitiveMeshGeometry.C
    faces=[] # [N,{xyz}] array per face

    VSMALL=1e-14
    nfaces = facefile.nfaces
    ncells = 1+max(owner.max(),neigh.max()) # or get it from owner file
    n_internal = neigh.shape[0]        

    if 1: # get face centers
        face_id_pts=[ facefile.faces[fIdx]["id_pts"] for fIdx in range(nfaces)]
        t=time.time()
        if 0:
            # oddly, this is slower than the outer loop being in python
            # probably because even with List, the contents are individual
            # arrays, each having to be checked.
            # solution is to convert to a pure numpy ragged array (data,start,count)
            face_id_pts_numba = List(face_id_pts)
            face_ctr,face_area = face_center_area_py(face_id_pts_numba,xyz)
        else:
            face_ctr,face_area = face_center_area_py(face_id_pts,xyz)
        print("Time for face_center call: ", time.time()-t)

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
    



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #timename = '34'
    sim_dir=r'..\..\..\fishpassage-DWR-7\fishpassage-DWR-7-7cms-local'

    # This takes several minutes as it reads and processes the entire mesh
    pf = PostFoam(sim_dir=sim_dir)
    pf.set_raster_parameters(dx=0.1)


    frame_dir=os.path.join(sim_dir,"figures","depth_avg_U")
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    fig,ax=plt.subplots(num=2,clear=1)
    fig.set_size_inches( (12,4), forward=True)
    fig.subplots_adjust(left=0.03,right=0.97)

    # skip 0 state
    for frame,t in enumerate(pf.available_times()[1:]):
        pf.read_timestep(t)
        fld = pf.to_raster('U')

        fld_mag=field.SimpleGrid(extents=fld.extents,F=np.sqrt((fld.F[...,0]**2+fld.F[...,2]**2)))
        img = fld_mag.plot(ax=ax,cmap='turbo',clim=[0,3])
        plt.colorbar(img,fraction=0.04,label='Depth-average speed (m/s)',pad=0.02)
        X,Y = fld.XY()
        stride=slice(None,None,4)
        ax.quiver( X[stride,stride].ravel(), Y[stride,stride].ravel(), 
                  fld.F[stride,stride,0].ravel(), fld.F[stride,stride,2].ravel(),
                  scale=60, width=0.0015,color='w')
        plt.setp(ax.spines.values(),visible=0)
        fig.savefig(os.path.join(frame_dir,"frame%04d.png"%frame))


