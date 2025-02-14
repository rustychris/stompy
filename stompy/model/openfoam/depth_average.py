# -*- coding: utf-8 -*-
"""
Approximate depth-average for interFoam results

Created on Fri Jan  3 13:25:56 2025

@author: chrhol4587
"""
import numpy as np
import os
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
from ... import utils
from ...spatial import field

from scipy import sparse
from scipy.spatial import KDTree
import pandas as pd
import scipy.spatial as ss # only for calc_volume=True

from fluidfoam.readof import OpenFoamFile
from fluidfoam import readvector, readscalar

from shapely import geometry, ops, convex_hull


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


    Note: The code here implicitly transposes y and z in the output. It expects
    y to be the vertical coordinate, and x/z are in the horizontal plane. The 
    transposition is possibly changing the coordinate system from right-handed to
    left-handed. Caveat emptor.
    """

    sim_dir = None
    verbose = False
    raster_mode = 'clipping' # or 'bbox'
    
    max_n_procs=1024 # just to bound the search, no inf loops
    precision=15
    current_timename = None
    n_tasks=8 # parallelism for cell-clipping step
    
    # command, with path as needed, to openfoam writeMeshObj
    writeMeshObj="writeMeshObj"

    _local_to_global = None

    def __init__(self,**kw):
        """
        
        """
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
        self._volumes=None
    @property
    def bboxes(self):
        if self._bboxes is None:
            self.read_mesh()
        return self._bboxes
    @property
    def volumes(self):
        if self._volumes is None:
            self.read_mesh(read_volumes=True)
        return self._volumes
    
    def read_mesh(self, read_volumes=False):
        # Get bounding boxes for the whole domain
        bboxes=[]
        volumes=[]
        
        for proc in range(self.n_procs):
            print(f"Reading mesh for processor {proc}")
            if read_volumes:
                bbox,vol = self.read_cell_bbox(self.proc_dir(proc))
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
                                    
    def read_cell_bbox(self, proc=None, calc_volume=False):
        """
        Get the axis-aligned extents of each cell. Limited to 
        constant meshes. path can be self.sim_dir for the global mesh,
        or a processor path for a submesh
        
        Calculating volumes increases run time by a factor of 4 or so
        """
        owner = self.read_owner(proc) 
        facefile = self.read_facefile(proc) 
        pointfile = self.read_pointfile(proc)
        neigh = self.read_neighbor(proc)
        
        face = {}
        for i in range(neigh.nb_faces):
            if not neigh.values[i] in face:
                face[neigh.values[i]] = list()
            face[neigh.values[i]].append(facefile.faces[i]["id_pts"][:])
        for i in range(owner.nb_faces):
            if not owner.values[i] in face:
                face[owner.values[i]] = list()
            face[owner.values[i]].append(facefile.faces[i]["id_pts"][:])
    
        if calc_volume:
            VolCell_all = np.empty(owner.nb_cell, dtype=float)
    
        # bounds of each cell, xxyyzz, in original OF reference frame
        cell_bounds = np.full( (owner.nb_cell,6), np.nan )
        
        for i in range(owner.nb_cell):
            pointsCell=[]
            for k in zip(np.unique(np.concatenate(face[i])[:])):
                pointsCell.append([pointfile.values_x[k],pointfile.values_y[k],pointfile.values_z[k]])
    
            # Add 3D elements into the empty array
            pointsCell=np.array(pointsCell)
            if calc_volume:
                # Note that cells are not necessarily convex, and faces
                # are not necessarily planar, so this is not exact.
                VolCell_all[i]=ss.ConvexHull(pointsCell).volume
            for dim in range(3):
                cell_bounds[i,2*dim  ] = np.min(pointsCell[:,dim])
                cell_bounds[i,2*dim+1] = np.max(pointsCell[:,dim])
            
        if calc_volume:
            return cell_bounds, VolCell_all
        else:
            return cell_bounds

    def set_raster_parameters(self,dx,dy=None,xxyy=None):
        if xxyy is None:
            # Note this pulls the z coordinate for bounds which
            # we implicitly convert to y.
            xxyy = [self.bboxes[:,0].min(),self.bboxes[:,1].max(),
                    self.bboxes[:,4].min(),self.bboxes[:,5].max()]
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
        
        if self.bboxes is None:
            raise Exception("set_raster_parameters() must be called after initializing mesh")

        self.raster_precalc=None # calculate on-demand

        return dict(dx=dx,dy=dy,xxyy=xxyy,nx=int((xxyy[1]-xxyy[0])/dx),ny=int((xxyy[3]-xxyy[2])/dy))

    def precalc_raster_info(self,fld):
        if self.raster_mode=='clipping':
            self.precalc_raster_info_clipping(fld)
        elif self.raster_mode=='bbox':
            self.precalc_raster_info_bbox(fld)
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
            # Note implicit z->y change of coodinates
            dy = (np.minimum(ymax,self.bboxes[:,5]) - np.maximum(ymin,self.bboxes[:,4])).clip(0)
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

        raster_weights = None
        tasks = [ (self.proc_dir(proc),fld) 
                 for proc in range(self.n_procs) ]
              
        with Pool(self.n_tasks) as pool: 
            
            results = pool.starmap(precalc_raster_weights_proc,tasks)
            
            for proc,proc_raster_weights in enumerate(results):
                print(f"Assembling from processor {proc}")
                # Want to stack these left-to-right
                if raster_weights is None:
                    raster_weights = proc_raster_weights
                else:
                    raster_weights = sparse.hstack( (raster_weights,proc_raster_weights) )
        self.raster_precalc = raster_weights

    def to_raster(self, variable, timename):
        if variable=='wse':
            return self.raster_wse(timename)
        
        self.read_timestep(timename)
        n_components=None
        if variable=='U':
            n_components = 3
        fld = field.SimpleGrid.zeros(extents=self.raster_xxyy,
                                     dx=self.raster_dx,dy=self.raster_dy, 
                                     n_components=n_components)
        
        if self.raster_precalc is None:
            self.precalc_raster_info(fld)
            
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
                
        return fld

    
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
        if proc is None:
            center_fn = os.path.join(self.sim_dir,'meshCellCentres_constant.obj')
        else:
            center_fn = os.path.join(self.proc_dir(proc),'meshCellCentres_constant.obj')

        if not os.path.exists(center_fn):
            if not self.compute_cell_centers(center_fn):
                print("Fall back to python cell center computation")
                self.compute_cell_centers_py(proc,center_fn)
        df = pd.read_csv(center_fn,sep=r'\s+',names=['type','x','y','z'])
        return df[ ['x','y','z'] ].values

    def face_centers(self,proc=None):
        # see cell_centers
        if proc is None:
            center_fn = os.path.join(self.sim_dir,'meshFaceCentres_constant.obj')
        else:
            center_fn = os.path.join(self.proc_dir(proc),'meshFaceCentres_constant.obj')

        if not os.path.exists(center_fn):
            if not self.compute_cell_centers(center_fn):
                raise Exception(f"Face center file {center_fn} not found. Generate with writeMeshObj")
        df = pd.read_csv(center_fn,sep=r'\s+',names=['type','x','y','z'])
        return df[ ['x','y','z'] ].values
    
    def compute_cell_centers(self,center_fn):
        case = os.path.dirname(center_fn)
        completed = subprocess.run([self.writeMeshObj,"-constant","-time","none"], cwd=case)
        if completed.returncode!=0:
            return False
        if not os.path.exists(center_fn):
            print(f"After writeMeshObj failed to find {center_fn}")
            return False
        return True
    
    def compute_cell_centers_py(self,proc,center_fn):
        owner = self.read_owner(proc) 
        facefile = self.read_facefile(proc) 
        pointfile = self.read_pointfile(proc)
        neigh = self.read_neighbor(proc)

        ctrs,vols = cell_center_volume_py(facefile, pointfile.values.reshape([-1,3]), owner.values, neigh.values)
        df=pd.DataFrame()
        df['t']=["v"] * len(ctrs)
        df['x']=ctrs[:,0]
        df['y']=ctrs[:,1]
        df['z']=ctrs[:,2]
        df.to_csv(center_fn,sep=' ',header=False,index=False,float_format="%.15f")
                  
    def contour_points_proc(self, scalar, contour_value, proc):
        centers = self.cell_centers(proc)    
        owner =self.read_owner(proc)
        nbr   =self.read_neighbor(proc)
        
        n_face_int = len(nbr.values) # assuming that boundary faces are omitted from nbr
        # per-internal-face values
        scalar_owner = scalar[owner.values[:n_face_int]]
        scalar_nbr = scalar[nbr.values[:n_face_int]]
        
        # (1-frac)*scalar_owner + frac*scalar_nbr = contour_value
        # frac*(scalar_nbr-scalar_owner) = contour_value - scalar_owner
        sel = ( (np.minimum(scalar_owner,scalar_nbr)<=contour_value)
               &(np.maximum(scalar_owner,scalar_nbr)>=contour_value) )
        
        frac = (contour_value-scalar_owner[sel]) / (scalar_nbr[sel] - scalar_owner[sel])
        
        pnts = ( (1-frac[:,None])*centers[owner.values[:n_face_int][sel]]
                +frac[:,None]*centers[nbr.values[:n_face_int][sel]])
        
        # Also need selected faces to clip interpolation
        facefile = self.read_facefile(proc)
        pointfile = self.read_pointfile(proc)
        points_xyz=pointfile.values.reshape([-1,3])
    
        face_pnts=[]
        for fIdx in np.nonzero(sel)[0]:    
            id_pts = facefile.faces[fIdx]["id_pts"][:]
            face_pnts.append( points_xyz[id_pts,:] )
        
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
        fld_pnts = field.XYZField(X=pnts[:,::2], F=pnts[:,1])

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
                geom = convex_hull(geometry.MultiPoint(pnts[:,::2]))
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
        if len(new_points)>3:
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
    
def precalc_raster_weights_proc(meshpath, fld, precision=15, force=False):
    hash_input=[fld.extents,fld.F.shape]
    hash_out = hashlib.md5(pickle.dumps(hash_input)).hexdigest()
    cache_dir=os.path.join(meshpath,"cache")
    cache_fn=os.path.join(cache_dir,
                          f"raster_weights-{fld.dx:.3g}x_{fld.dy:.3g}y-{hash_out}")
    print(f"Weights for {meshpath}: cache file is {cache_fn}")
    
    if os.path.exists(cache_fn) and not force:
        with open(cache_fn,'rb') as fp:
            return pickle.load(fp)
    
    fld_x,fld_y = fld.xy()
    Apixel = fld.dx*fld.dy

    # Load proc mesh                    
    verbose=False
    meshpath = os.path.join(meshpath,'constant/polyMesh')
    # owner.nb_faces, boundary, values, nb_cell
    owner = OpenFoamFile(meshpath, name="owner", verbose=verbose)
    facefile = OpenFoamFile(meshpath, name="faces", verbose=verbose)
    pointfile = OpenFoamFile(meshpath,name="points",precision=precision,
        verbose=verbose)
    neigh = OpenFoamFile(meshpath, name="neighbour", verbose=verbose)
    cell_faces=defaultdict(list)

    for fIdx,cIdx in enumerate(neigh.values):
        cell_faces[cIdx].append(fIdx) 
    for fIdx,cIdx in enumerate(owner.values):
        cell_faces[cIdx].append(fIdx)

    n_cell = owner.nb_cell
    cells_as_edges=[None]*n_cell
    
    bboxes=np.zeros((n_cell,6),np.float64)
    for cIdx in range(n_cell):
        edges = cell_as_edges(cIdx, cell_faces, facefile, pointfile)
        cells_as_edges[cIdx] = edges
        pnts = np.array(edges).reshape( (-1,3) )
        xxyyzz = [pnts[:,0].min(),
                  pnts[:,0].max(),
                  pnts[:,1].min(),
                  pnts[:,1].max(),
                  pnts[:,2].min(),
                  pnts[:,2].max()]
        bboxes[cIdx] = xxyyzz # OF native reference frame

    raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                        bboxes.shape[0]),
                                       np.float64)
        
    # check bounds, then compute clipped volume.
    
    for row in utils.progress(range(fld.shape[0])):
        ymin=fld_y[row]-fld.dy/2
        ymax=fld_y[row]+fld.dy/2
        # Implicit transposition - switch openfoam z to raster y
        #dy = (np.minimum(ymax,bboxes[:,5]) - np.maximum(ymin,bboxes[:,4])).clip(0)
        for col in range(fld.shape[1]):
            pix = row*fld.shape[1] + col # row-major ordering of pixels
            
            xmin=fld_x[col]-fld.dx/2
            xmax=fld_x[col]+fld.dx/2
         
            # Implicit transposition - switch openfoam z to raster y
            sel = ( (bboxes[:,0]<xmax)&(bboxes[:,1]>xmin) &
                    (bboxes[:,4]<ymax)&(bboxes[:,5]>ymin) )
            
            #dx = (np.minimum(xmax,bboxes[:,1]) - np.maximum(xmin,bboxes[:,0])).clip(0)
            #A = dx*dy
            
            # can volume also improve this? Maybe if we also calculated dz.
            #weights = A[mask]/(fld.dx*fld.dy)
            for cIdx in np.nonzero(sel)[0]:
                # alpha-weighting is included per timestep
                # Implicit transposition - switch openfoam z to raster y
                vol = clipped_volume(cells_as_edges[cIdx],
                                     xmin=xmin,xmax=xmax,
                                     zmin=ymin,zmax=ymax)
                raster_weights[pix,cIdx] = vol/Apixel
                
    if cache_fn is not None:
        os.makedirs(cache_dir)
        with open(cache_fn,'wb') as fp:
            pickle.dump(raster_weights,fp,-1)
    return raster_weights


def cell_as_edges(cIdx, cell_faces, facefile, pointfile):
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
    xyz=pointfile.values.reshape([-1,3])
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


