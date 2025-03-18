# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 19:09:21 2025

@author: ChrHol4587
"""

from ...grid import unstructured_grid
from ... import utils, memoize
import h5py
import numpy as np
import datetime
import glob, os

# Would like to replicate what RAS does
# technically that requires the terrain
# terrain is only used for masking, afaik
# should be able to do the rest of the interpolation.
# probably this should be some sort of RasReader class, which
# can be queried for WSE, depth, velocity etc, for specific
# timesteps


def face_interp_ben(wse_j1, wse_j2, z_j1, z_j2, face_min_z):
    '''
    wse_j1: WSE from cell A of the face
    wse_j2: WSE from cell B of the face
    z_j1: min bed elevation from cell A
    z_j2: min bed leevation from cell B
    face_delta_z
    face_min_z: 
        
    # face_z = minimum elevation of the face
    # wse_ben = face_interp_ben(wse_left, wse_right, cell_z[ileft], cell_z[iright], face_dz, face_z[j])
    '''

    face_delta_z = z_j2 - z_j1 # why was this passed in previously?
    
    if wse_j1 >= wse_j2:
        max_wse = wse_j1
        max_z = z_j1
    else:
        max_wse = wse_j2
        max_z = z_j2

    if (wse_j2 - wse_j1) * face_delta_z <= 0:
      face_wse = max_wse
    else:
      if face_min_z >= max_wse:
        face_wse = max_wse
      else:
        face_delta_z = np.abs(face_delta_z)
        h2 = max_wse - max_z
        if h2 >= 2. * face_delta_z:
          face_wse = max_wse
        else:
          min_wse = min(wse_j1, wse_j2)
          low_bound = max(face_min_z, min_wse)
          h1 = low_bound - max_z
          face_wse = low_bound + (h2*h2 - h1*h1) / (2. * face_delta_z)
          if h2 > face_delta_z:
            face_wse = ((2. * face_delta_z - h2) * face_wse + (h2 - face_delta_z) * max_wse) / face_delta_z
          face_wse = min(face_wse, max_wse)
    return face_wse

class RasProject:
    def __init__(self,prj_fn):
        self.prj_fn = prj_fn
        self.prj_dir = os.path.dirname(prj_fn)
        self.prj_name = os.path.basename(prj_fn).replace(".prj","")
        
    def getPlanFile(self, plan_name):
        '''
        reads the plan file from the plan HDF5 file and finds correct corresponding plan file to the short ID
        '''
        plan_files = self.getPlanFiles()
        h=None
        for plf in plan_files:
            if plf.endswith('.tmp.hdf'):
                continue
            try:
                h = h5py.File(os.path.join(self.prj_dir, plf), 'r')
                try:
                    short_title = h['Results/Unsteady'].attrs['Short ID'].decode('utf-8')
                finally:
                    h.close()
                if short_title == plan_name:
                    print(f'Found plan file {plf} for plan: {plan_name}')
                    return plf
            except Exception as e:
                print('Error reading plan file:', plf)
                # print(e)
                continue

        print('Unable to find plan file for plan:', plan_name)
        #print('Please specficy plan file using variable plan_file=... in inputs.')
        return None
    
    def getPlanFiles(self):
        '''
        gets all the plan files in the project directory
        '''
        plan_file_form = f'{self.prj_name}.p*.hdf' # "*.p[!*.]*"
        plan_files = glob.glob(os.path.join(self.prj_dir, plan_file_form))
        return plan_files

    @memoize.imemoize()
    def get_results(self,plan_name,area_name=None):
        plan_file = self.getPlanFile(plan_name)
        if plan_file is None:
            return None
        return RasReader(plan_file, area_name=area_name)

class RasReader:
    # Where to find results within h5 file
    unsteady_base_ras6=('/Results/Unsteady/Output/Output Blocks/'
                        'Base Output/Unsteady Time Series')
    unsteady_base_2025 = '/Results/Output Blocks/Base Output'

    def __init__(self,results_fn, area_name=None):
        self.results_fn = results_fn
        self.area_name = area_name        
        
        self.load_h5()
        self.load_grid()
        
    def short_title(self):
        return self.h5['Results/Unsteady'].attrs['Short ID'].decode('utf-8')
        
    def load_h5(self):
        self.h5 = h5py.File(self.results_fn,'r')
        # Is this RAS6 or RAS2025?
        try:
            self.h5[self.unsteady_base_ras6]
            self.version="RAS6"
        except KeyError:
            self.version="RAS2025"
            
    def close(self):
        self.h5.close()
        self.h5 = None
        
    def terrain_file(self):
        key = f"/Geometry/2D Flow Areas/{self.area_name}"
        return self.h5[key].attrs['Terrain Filename'].decode('ascii')

    def terrain_path(self):
        return os.path.join(os.path.dirname(self.results_fn),self.terrain_file())

    def load_grid(self):
        self.grid=unstructured_grid.UnstructuredGrid.read_ras2d(self.h5,
                                                                twod_area_name=self.area_name,
                                                                subedges='subedges')
        if self.area_name is None:
            self.area_name=self.grid.twod_area_name
        
    @property
    def unsteady_base(self):
        if self.version=='RAS6':
            return self.unsteady_base_ras6
        elif self.version=='RAS2025':
            return self.unsteady_base_2025
        else:
            raise Exception(f"Bad version {self.version}")
            
    @property
    def area_base(self):
        if self.version=='RAS6':        
            return (self.unsteady_base + f'/2D Flow Areas/{self.area_name}/')
        elif self.version=='RAS2025':
            # maybe "Mesh" is actually the twod_area_name? 2025 doesn't have
            # multiple areas yet.
            return self.unsteady_base + '/2D Flow Areas/Mesh/'
        else:
            raise Exception("Bad version: "+self.version)

    def time_relative_days(self):
        return self.h5[self.unsteady_base+'/Time']

    def time_start(self):
        t = self.h5['Plan Data/Plan Information'].attrs['Simulation Start Time'].decode('ascii')
        # '22Aug2022 02:00:00'
        t_datetime = datetime.datetime.strptime(t,"%d%b%Y %H:%M:%S")
        return np.datetime64(t_datetime)

    def times(self):
        t0 = self.time_start()
        offset_days = self.time_relative_days()[:]
        return t0 + np.array(offset_days*86400,np.int64)*np.timedelta64(1,'s')
        
    def cell_wse(self,time_step,trim_virtual=True):
        key=self.area_base+'Water Surface'
        result = self.h5[key][time_step]
        if trim_virtual:
            return result[:self.grid.Ncells()]
        else:
            return result
    
    def face_wse(self,time_step):
        """
        Ben's hard way to interpolate WSE at the face, currently only RAS6
        """
        result = np.full(self.grid.Nedges(),np.nan)
        # for RAS6, keep virtual to simplify code below
        cell_wse = self.cell_wse(time_step)

        if self.version!='RAS6':
            print('Face WSE assumes RAS6')
        cellA=self.grid.edges['cells'][:,0]
        cellB=self.grid.edges['cells'][:,1]
        # See if we can get away with ignoring virtual cells.
        cellB = np.where(cellB>=self.grid.Ncells(),cellA,cellB)
        zminA=self.grid.cells['cell_z_min'][cellA]
        zminB=self.grid.cells['cell_z_min'][cellB]
        wseA=cell_wse[ cellA ]
        wseB=cell_wse[ cellB ]
        face_min_elev=self.grid.edges['edge_z_min']
        for j in range(len(wseA)):
            result[j] = face_interp_ben(wseA[j], wseB[j], zminA[j], zminB[j], face_min_elev[j])
        return result
    

    def face_flow(self,time_step):
        flow_key = self.area_base + "Face Flow"
        if flow_key in self.h5:
            return self.h5[flow_key][time_step,:]
        else:
            raise Exception("Inferring face flow not yet implemented")
    def face_velocity(self,time_step):
        vel_key = self.area_base + "Face Velocity"
        if vel_key in self.h5:
            return self.h5[vel_key][time_step,:]
        else:
            raise Exception("Inferring face velocity not implemented")

    def face_area(self,time_step):
        face_wse = self.face_wse(time_step)
        face_area = np.zeros_like(face_wse)
        for j in utils.progress(range(self.grid.Nedges())):
            face_area[j] = self.face_area_elev_interp(j,face_wse[j])
        return face_area
    
    def face_area_elev_interp(self,j,wse):
        tbl = self.grid.edges['area_table'][j]
        interp_area = np.interp(wse, tbl['z'], tbl['area'])
        # all RAS6 tables start with 0 area, and at least 2 entries.
        if wse>tbl['z'][-1]:
            wet_length = utils.dist_total(self.grid.get_subedge(j,subedges='subedges'))
            interp_area += (wse-tbl['z'][-1])*wet_length
        return interp_area

    def cell_velocity(self,time_step):
        # Cell-centered vector velocity via weighted least squares
        face_areas = self.face_area(time_step)
        face_flows = self.face_flow(time_step)
        normals = self.grid.edges['normal']

        # WLS
        result=np.full( (self.grid.Ncells(),2), np.nan)
        for c in range(self.grid.Ncells()):
            rows=[]
            rhs=[]
            for j in self.grid.cell_to_edges(c):
                rows.append( [face_areas[j]*normals[j,0],
                              face_areas[j]*normals[j,1]]) 
                rhs.append( face_flows[j] )  
            x,res,rank,s = np.linalg.lstsq(rows,rhs,rcond=None)
            result[c,:] = x
        return result

