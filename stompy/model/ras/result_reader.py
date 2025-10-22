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
import xarray as xr

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

    def load_postproc(run):
        # Doesn't actually have the computed flows. Maybe not useful.
        post_proc_fn = os.path.dirname(self.results_fn) + f"/{self.short_title()}/PostProcessing.hdf"
        return h5py.File(post_proc_fn, 'r')

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
        # This is failing with 2025
        plan_info = 'Plan Information'
        compute_start = 'Compute Start Time (DESC)'
        if plan_info in self.h5['Plan Data']:
            # RAS 6.x
            t = self.h5['Plan Data/Plan Information'].attrs['Simulation Start Time'].decode('ascii')
            # '22Aug2022 02:00:00'
            t_datetime = datetime.datetime.strptime(t,"%d%b%Y %H:%M:%S")
        elif compute_start in self.h5['Plan Data'].attrs:
            t = self.h5['Plan Data'].attrs[compute_start].decode('ascii')
            # '1/1/2000 12:00:00 AM'
            t_datetime = datetime.datetime.strptime(t,"%m/%d/%Y %I:%M:%S %p")

        return np.datetime64(t_datetime)

    def times(self):
        t0 = self.time_start()
        offset_days = self.time_relative_days()[:]
        return t0 + np.array(offset_days*86400,np.int64)*np.timedelta64(1,'s')

    @memoize.imemoize(lru=5)
    def cell_wse(self,time_step,trim_virtual=True):
        key=self.area_base+'Water Surface'
        result = self.h5[key][time_step]
        if trim_virtual:
            return result[:self.grid.Ncells()]
        else:
            return result
        
    def cell_min_bed_elevation(self, trim_virtual=True):
        base=f'Geometry/2D Flow Areas/{self.area_name}'
        z = self.h5[base]['Cells Minimum Elevation']
        if trim_virtual:
            z=z[:self.grid.Ncells()]
        return z
    
    def cell_mean_bed_elevation(self, trim_virtual=True):
        base=f'Geometry/2D Flow Areas/{self.area_name}'
        if self.version=='RAS6':
            return self.cell_mean_bed_elevation_ras6(base,trim_virtual)
        elif self.version=='RAS2025':
            return self.cell_mean_bed_elevation_ras2025(base,trim_virtual)
            
    def cell_mean_bed_elevation_ras6(self,base,trim_virtual):
        # Ncell x {start,count}
        cell_vol_elev_info_key="Cells Volume Elevation Info"
        if cell_vol_elev_info_key not in self.h5[base]:
            print("No subgrid bathy for cells found. Flat cells or misinterpreting H5 file.")
            print(f"  (expecting {self.version} formatted file)")
            return self.cell_min_bed_elevation(trim_virtual=trim_virtual)

        cell_vol_elev_info = self.h5[base][cell_vol_elev_info_key][:,:]
        # Ncell x {elevation, volume}
        cell_vol_elev_values = self.h5[base]['Cells Volume Elevation Values'][:,:]
        cell_areas = self.h5[base]['Cells Surface Area'][:]

        # Assumes cells are fully inundated
        last_entries = cell_vol_elev_info[:,0] + cell_vol_elev_info[:,1] - 1
        max_elevs = cell_vol_elev_values[last_entries,0]
        max_vols  = cell_vol_elev_values[last_entries,1]
        # ghost cells throw some 0.0 in
        mean_bed_elevs = np.zeros_like(max_elevs)
        ghost = cell_areas<=0.0
        mean_bed_elevs[~ghost] = max_elevs[~ghost] - max_vols[~ghost]/cell_areas[~ghost]
        if trim_virtual:
            mean_bed_elevs=mean_bed_elevs[:self.grid.Ncells()]
        return mean_bed_elevs

    def cell_mean_bed_elevation_ras2025(self,base,trim_virtual=True):
        cell_table_key="Property Tables/Cell Tables"
        if cell_table_key not in self.h5[base]:
            print("No subgrid bathy for cells found. Flat cells or misinterpreting H5 file.")
            print(f"  (expecting {self.version} formatted file)")
            return self.cell_min_bed_elevation(trim_virtual=trim_virtual)

        cell_elev_vol = self.h5[base][cell_table_key]
        starts = cell_elev_vol.attrs['Start']
        counts = np.zeros_like(starts)
        counts[:-1] = np.diff(starts)
        counts[-1] = len(cell_elev_vol)-starts[-1]
        
        # Ncell x {elevation, volume}
        
        cell_areas = self.grid.cells_area(subedges='subedges')

        # Assumes cells are fully inundated, up to last elev entry.
        last_entries = starts + counts - 1
        max_elevs = cell_elev_vol[last_entries,0]
        max_vols  = cell_elev_vol[last_entries,1]
        # ghost cells throw some 0.0 in
        mean_bed_elevs = np.zeros_like(max_elevs)
        ghost = cell_areas<=0.0
        mean_bed_elevs[~ghost] = max_elevs[~ghost] - max_vols[~ghost]/cell_areas[~ghost]
        if trim_virtual:
            mean_bed_elevs=mean_bed_elevs[:self.grid.Ncells()]
        return mean_bed_elevs
        
    @memoize.imemoize(lru=5)
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
    
    @memoize.imemoize(lru=5)
    def face_flow(self, time_step, structure_adjustment=False):
        flow_key = self.area_base + "Face Flow"
        if flow_key in self.h5:
            flow = self.h5[flow_key][time_step,:]
            if structure_adjustment:
                self.update_structure_flow_velocity(time_step, face_flow=flow)
            return flow
        else:
            return self.face_velocity(time_step, structure_adjustment=structure_adjustment) * self.face_area(time_step)
        
    @memoize.imemoize(lru=5)
    def face_velocity(self, time_step, structure_adjustment=False):
        vel_key = self.area_base + "Face Velocity"
        if vel_key in self.h5:
            vel = self.h5[vel_key][time_step,:]
            if structure_adjustment:
                self.update_structure_flow_velocity(time_step,face_velocity=vel)
            return vel
        else:
            raise Exception("Inferring face velocity not implemented")
        
    @memoize.imemoize(lru=5)
    def face_area(self,time_step):
        face_wse = self.face_wse(time_step)
        face_area = np.zeros_like(face_wse)
        for j in utils.progress(range(self.grid.Nedges())):
            face_area[j] = self.face_area_elev_interp(j,face_wse[j])
        return face_area

    def update_structure_flow_velocity(self, tidx, face_flow=None, face_velocity=None):
        """
        Update face flow and/or area to account for 2D hydraulic connections.
        """
        hyd_conns="2D Hyd Conn"
        face_areas = self.face_area(tidx)
        for conn in self.h5[self.area_base][hyd_conns]:
            # print(conn)
            # Face points - just use the HW side (HW=TW for our gates)
            face_points = self.h5[self.area_base][hyd_conns][conn]['Geometric Info']['Headwater Face Points'][:]
            flows = self.h5[self.area_base][hyd_conns][conn]['HW TW Segments']['Flow'][tidx,:] # has an extra entry
            edges = [self.grid.nodes_to_halfedge(a,b)
                     for a,b in zip(face_points[:-1],face_points[1:])]
            for i,he in enumerate(edges):
                # Guess and check on the sign. I'm assuming that the orientation of HW vs TW is dictated
                # by the order of face face points. Based on results, need the extra negation here.
                Q=-flows[i] * (-1)**he.orient
                if face_flow is not None:
                    face_flow[he.j] = Q
                if face_velocity is not None:
                    face_velocity[he.j] = Q / face_areas[he.j].clip(0.001)
                    
    def structure_faces(self):
        """
        Edge indexes with structures
        """
        hyd_conns="2D Hyd Conn"
        edges=[]
        for conn in self.h5[self.area_base][hyd_conns]:
            face_points = self.h5[self.area_base][hyd_conns][conn]['Geometric Info']['Headwater Face Points'][:]
            edges += [self.grid.nodes_to_halfedge(a,b).j
                      for a,b in zip(face_points[:-1],face_points[1:])]
        return np.array(edges)
    
    def face_area_elev_interp(self,j,wse):
        tbl = self.grid.edges['area_table'][j]
        interp_area = np.interp(wse, tbl['z'], tbl['area'])
        # all RAS6 tables start with 0 area, and at least 2 entries.
        if wse>tbl['z'][-1]:
            wet_length = utils.dist_total(self.grid.get_subedge(j,subedges='subedges'))
            interp_area += (wse-tbl['z'][-1])*wet_length
        return interp_area

    @memoize.imemoize(lru=5)
    def cell_velocity(self, time_step, face_areas=None, face_velocity=None, structure_adjustment=False):
        # Cell-centered vector velocity via weighted least squares
        if face_areas is None:
            face_areas = self.face_area(time_step)
        #if face_flows is None:
        #    face_flows = self.face_flow(time_step, structure_adjustment=structure_adjustment)
        if face_velocity is None:
            face_velocity = self.face_velocity(time_step, structure_adjustment=structure_adjustment)
        normals = self.grid.edges['normal']

        # WLS - mostly follow RAS, though this is not as optimized or vetted
        result=np.full( (self.grid.Ncells(),2), np.nan)
        cell_areas = self.grid.cells_area()
        face_lengths = self.grid.edges_length()
        
        for c in range(self.grid.Ncells()):
            rows=[]
            rhs=[]
            for j in self.grid.cell_to_edges(c):
                v_f = face_velocity[j]
                # Still not working out that well. Quad cell has two closed faces
                # and two with modest flow/velocity. The two with flow are not perfectly
                # aligned, and the flow is noisy, leading to change in volume. WLS does
                # not account for that, and tries to make up the difference with a large
                # lateral velocity.
                # Original approach used face area to weight the rows.
                # But I think face length might be better? 
                w=face_lengths[j]
                # [ n_x*A_f    n_y*A_f ] [u_x] = [ u_f ]
                # [ n_x*A_f    n_y*A_f ] [u_y] =
                
                rows.append( [w*normals[j,0],
                              w*normals[j,1]]) 
                rhs.append( w*v_f )
            x,res,rank,s = np.linalg.lstsq(rows,rhs,rcond=None)
            result[c,:] = x
        return result

    def extract(self,time_steps=None,variables=['Water Surface'],locations=None):
        """
        locations: [ [x,y], ...] or dict { 'label':[x,y], ....}
        """
        ds=xr.Dataset()
        if time_steps is None:
            time_steps = np.arange(len(self.times()))
        ds['time'] = ('time',), self.times()[time_steps].astype('<M8[ns]')

        # flat WSE in each cell
        if locations is None:
            cells = np.arange(self.grid.Ncells())
        else:
            if isinstance(locations,dict):
                loc_xy=np.array(list(locations.values()))
                loc_name=list(locations.keys())
            else:
                loc_xy=np.array(locations)
                loc_name=None
            cells = [self.grid.select_cells_nearest(p) for p in loc_xy]
            ds['x'] = ('sample',), loc_xy[:,0]
            ds['y'] = ('sample',), loc_xy[:,1]
            if loc_name:
                ds['sample']=('sample',),loc_name

        ds['cell']=('sample',),cells

        # initialize
        for v in variables:
            ds[v] = ('time','sample',),np.zeros((ds.sizes['time'],ds.sizes['sample']),np.float64)

        for step_i,time_step in enumerate(time_steps):
            for v in variables:
                if v=='Water Surface':
                    wse = self.cell_wse(time_step)
                    ds[v].values[step_i,:] = wse[cells]
        return ds

    @memoize.imemoize()
    def sa2d_conn_cells(self):
        """
        Returns a list of cell indexes that participate in SA/2D connections.
        """
        gate_cells=[]
        conn_results="Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/SA 2D Area Conn"
        for conn in self.h5[conn_results]:
            hw_cell_path=conn_results+f"/{conn}/HW TW Segments/Headwater Cells"
            tw_cell_path=conn_results+f"/{conn}/HW TW Segments/Tailwater Cells"
            for cell_path in [hw_cell_path, tw_cell_path]:
                cells = self.h5[cell_path][:]
                if cells.dtype.char=='S': # not sure why, but they come in as strings
                    cells=[int(c) for c in cells]
                gate_cells.append(cells)
        gate_cells=np.unique(np.concatenate(gate_cells))
        return gate_cells
    
