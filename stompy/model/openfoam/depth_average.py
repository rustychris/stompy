# -*- coding: utf-8 -*-
"""
Approximate depth-average for interFoam results

Created on Fri Jan  3 13:25:56 2025

@author: chrhol4587
"""
import numpy as np
import os
# import matplotlib.pyplot as plt
#from fluidfoam.readof import getVolumes
#from fluidfoam import readmesh
from ... import utils
from ...spatial import field

from scipy import sparse
# import scipy.spatial as ss

from fluidfoam.readof import OpenFoamFile
from fluidfoam import readvector, readscalar


class PostFoam:
    """
    
    """
    sim_dir = None
    max_n_procs=1024 # just to bound the search, no inf loops
    
    def __init__(self,**kw):
        utils.set_keywords(self,kw)

        self.init_mesh()

    def available_times(self):
        times = [float(t) for t in os.listdir(self.proc_dir(0)) if t!='constant']
        # parsing to float then converting back in read_timestep() may lead to
        # roundoff issues, but so far not a problem.
        times.sort()
        return times
    
    def read_timestep(self,timename):
        if not isinstance(timename,str):
            timename = "%g"%timename
        vels=[]
        alphas=[]
        for proc in range(self.n_procs):
            print(f"Reading output for processor {proc}")
            proc_dir = self.proc_dir(proc)
            vels.append( readvector(proc_dir, timename, 'U') )
            alphas.append( readscalar(proc_dir, timename, 'alpha.water') )

        self.vels = np.concatenate( vels,axis=1 ).transpose()
        self.alphas = np.concatenate(alphas)
        
    def init_mesh(self):
        print("Reading openfoam mesh")        

        for n_procs in range(self.max_n_procs):
            if not os.path.exists(self.proc_dir(n_procs)):
                break
        self.n_procs = n_procs

        # Get bounding boxes for the whole domain
        bboxes=[]
        #volumes=[]
        
        for proc in range(self.n_procs):
            print(f"Reading mesh for processor {proc}")
            #bbox,vol = self.read_cell_bbox(self.proc_dir(proc))
            bbox = self.read_cell_bbox(self.proc_dir(proc))
            bboxes.append(bbox)
            #volumes.append(vol)
                
        # Appears there are no ghost cells,
        # so it's valid to just concatenate results from each processor when 
        # only considering cell-centered quantities.
        self.bboxes = np.concatenate(bboxes, axis=0)
        #self.volumes =np.concatenate(volumes)
        
    def proc_dir(self,proc):
        return os.path.join(self.sim_dir,f'processor{proc}')
                                    
    def read_cell_bbox(self,path,verbose=True, calc_volume=False):
        """
        Get the axis-aligned extents of each cell. Limited to 
        constant meshes. path can be self.sim_dir for the global mesh,
        or a processor path for a submesh
        
        Calculating volumes increases run time by a factor of 4 or so
        """
        precision=15
        
        meshpath = os.path.join(path,"constant/polyMesh")
        if not os.path.exists(meshpath):
            raise ValueError(
                f"No {meshpath} directory found to load mesh",
                " Please verify the directory of your case.",
            )
    
        owner = OpenFoamFile(meshpath, name="owner", verbose=verbose)
        facefile = OpenFoamFile(meshpath, name="faces", verbose=verbose)
        # No support for time-varying mesh
        pointfile = OpenFoamFile(
            meshpath,
            name="points",
            precision=precision,
            verbose=verbose
        )
        neigh = OpenFoamFile(
            meshpath, name="neighbour", verbose=verbose
        )
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
    
        # bounds of each cell, xxyyzz
        cell_bounds = np.full( (owner.nb_cell,6), np.nan )
        
        for i in range(owner.nb_cell):
            pointsCell=[]
            for k in zip(np.unique(np.concatenate(face[i])[:])):
                pointsCell.append([pointfile.values_x[k],pointfile.values_y[k],pointfile.values_z[k]])
    
            # Add 3D elements into the empty array
            pointsCell=np.array(pointsCell)
            if calc_volume:
                VolCell_all[i]=ss.ConvexHull(pointsCell).volume
            for dim in range(3):
                cell_bounds[i,2*dim  ] = np.min(pointsCell[:,dim])
                cell_bounds[i,2*dim+1] = np.max(pointsCell[:,dim])
            
        if calc_volume:
            return cell_bounds, VolCell_all
        else:
            return cell_bounds

    def set_raster_parameters(self,dx,dy=None,xxyy=None):
        self.raster_dx = dx
        self.raster_dy = dy or dx
        if xxyy is None:
            xxyy = [self.bboxes[:,0].min(),self.bboxes[:,1].max(),
                    self.bboxes[:,2].min(),self.bboxes[:,3].max()]
        self.raster_xxyy = xxyy
        
        if self.bboxes is None:
            raise Exception("set_raster_parameters() must be called after initializing mesh")

        self.raster_precalc=None # calculate on-demand

    def precalc_raster_info(self,fld):
        # weights is a matrix with columnscorresponding to openfoam cells
        # and rows corresponding to output pixels in row-major order
        
        raster_weights = sparse.dok_matrix((fld.F.shape[0]*fld.F.shape[1],
                                            self.bboxes.shape[0]),
                                           np.float64)
            
        fld_x,fld_y = fld.xy()
        Apixel = fld.dx*fld.dy
            
        for row in utils.progress(range(fld.shape[0])):
            ymin=fld_y[row]-fld.dy/2
            ymax=fld_y[row]+fld.dy/2
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
       
    def to_raster(self,variable='U'):
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

                u = num/den                
                
                fld.F[:,:,comp] = u.reshape( (fld.F.shape[0],fld.F.shape[1]) )
            else:
                raise Exception("Not ready for other fields")

        return fld

if __name__ == '__main__':
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


