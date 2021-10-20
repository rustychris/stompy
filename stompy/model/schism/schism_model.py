from ...grid import multi_ugrid, unstructured_grid
import numpy as np
import shutil
import glob, os
from .. import hydro_model as hm
from ... import utils

from ..delft import io as dio

class Namelist(dio.SectionedConfig):
    inline_comment_prefixes=('!',)
    section_patt=r'^(&[A-Za-z0-9_]+)(\s*(!.*)?)$'
    value_patt = r'^\s*([A-Za-z0-9_\(\)]+)\s*=([^!]*)(!.*)?$'
    blank_patt = r'^\s*(!.*)?$'
    end_section_patt=r'^(/)\s*$'
        
    def is_section(self,s):
        return s[0]=='&'
    def format_section(self,s):
        return '&%s'%s.upper()

class SchismModel(hm.HydroModel,hm.MpiModel):
    def __init__(self,*a,**kw):
        super(SchismModel,self).__init__(*a,**kw)
        self.structures=[]
        self.load_defaults()

    def add_Structure(self,**kw):
        # not wired up to schism yet
        self.structures.append(kw)
        
    def load_defaults(self):
        self.param=Namelist(os.path.join(os.path.dirname(__file__),'data','param.nml'))
        self.sediment=Namelist(os.path.join(os.path.dirname(__file__),'data','sediment.nml'))

    def add_Structure(self,**kwargs):
        self.log.warning("No support yet for writing SCHISM structures")

    def update_config(self):
        """
        Update fields in the param object with data from self.
        """
        if self.param is None:
            self.load_defaults()

        self.param['CORE','rnday'] = (self.run_stop - self.run_start)/np.timedelta64(86400,'s')

        dt_start=utils.to_datetime(self.run_start)
        self.param['OPT','start_year']=dt_start.year
        self.param['OPT','start_month']=dt_start.month
        self.param['OPT','start_day']=dt_start.day
        self.param['OPT','start_hour']=dt_start.hour + dt_start.minute/60. + dt_start.second/3600.
        self.param['OPT','utc_start']=0 # run in UTC
        
        self.param.set_filename(os.path.join(self.run_dir,'param.nml'))
        if self.sediment:
            self.sediment.set_filename(os.path.join(self.run_dir,'sediment.nml'))

        self.update_initial_water_level() # is this the right time?

    def set_grid(self,g):
        super().set_grid(g)
        # Clean slate, and BCs can update as needed.
        g=self.grid
        g.edges['mark']=g.INTERNAL
        g.add_edge_field('bc_id',-np.ones(g.Nedges(),np.int32),on_exists='overwrite')

    _initial_water_level=None
    def initial_water_level(self):
        return self._initial_water_level
    
    def update_initial_water_level(self):
        """
        work in progress. not very clean
        """
        wl=self.infer_initial_water_level()
        if wl is not None:
            self._initial_water_level=wl
        
    def write_config(self):
        # Assumes update_config() already called
        if len(self.structures):
            self.log.warning("No support for structures yet")
            # self.write_structures()
            
        self.write_monitors()
        self.log.info("Writing param.nml to %s"%self.param.filename)
        self.param.write()

        if self.sediment:
            self.log.info("Writing sediment.nml to %s"%self.sediment.filename)
            self.sediment.write()

    def write_extra_files(self):
        # Also create bed_frac, SED_hvar initial conditions.
        g.write_gr3('bed_frac_1.ic',z=1.0) # one sediment class, 100%
        g.write_gr3('SED_hvar_1.ic',z=0.0) # no bed sediments in IC
        g.write_gr3('salt.ic',z=0.0) # fresh
        g.write_gr3('temp.ic',z=20.0) # 20degC
        g.write_gr3('bedthick.ic',z=0.0) # bare rock
        # sets imnp per node. Can probably set to 0.0?
        g.write_gr3('imorphogrid.gr3',z=1.0)

        g.write_gr3('diffmin.gr3',z=1e-8)
        g.write_gr3('diffmax.gr3',z=1e-4)

        with open('tvd.prop','wt') as fp:
            for c in range(g.Ncells()):
                fp.write("%d 1\n"%(c+1))


    def write_forcing(self,overwrite=True):
        """
        Start with empty bctides.in, 
        """
        bc_fn=os.path.join(self.run_dir,'bctides.in')
        if overwrite and os.path.exists(bc_fn):
            os.unlink(bc_fn)
        utils.touch(bc_fn)
        with open(bc_fn,'wt') as fp:
            fp.write("%s\n"%self.run_dir) # just a comment line.

            # Can bctides.in be written in arbitrary order?
            # Not really, so we have to handle it all here.

            fp.write("0 40. ntip\n") # no earth tide potential
            fp.write("0 nbfr\n") # no harmonics for now.

            # Count up the number of open boundaries, both stage and flow
            # mark and count edges, too
            open_bcs=[]
            for bc in self.bcs:
                if isinstance(bc,hm.StageBC) or isinstance(bc,hm.FlowBC):
                    bc_id,bc_edges=self.mark_open_edges(bc.geom)
                    if bc.dredge_depth is not None:
                        self.log.info("Dredging boundary to %.2f"%bc.dredge_depth)
                        self.dredge_boundary(bc.geom,dredge_depth=bc.dredge_depth,
                                             node_field='node_z_bed')
                    open_bcs.append( (bc_id,bc,bc_edges) )
                    assert bc_id+1==len(open_bcs)

            # subordinate BCs that have a parent will be picked up while
            # writing each open boundary

            fp.write("%d nope\n"%len(open_bcs))

            # TODO: build list of tracer names ahead of this.
            stage_count=0

            elev_data=[] # a 1-D data array time series for each type 1 elevation bc
            flow_data=[] # a 1-D data array time series for each type 1 flow bc

            for bc_id,bc,bc_edges in open_bcs:
                #33 3 0 2 2 3 ! turn off velocity bc at ocean. T,S, and sed relax to IC
                # REFACTOR!
                flow_flag=elev_flag=temp_flag=salt_flag=sed0_flag=0
                if isinstance(bc,hm.StageBC):
                    data=bc.data()
                    if 'time' in data.dims:
                        elev_flag=1
                    else:
                        # assume it's a constant
                        elev_flag=2

                    stage_count+=1
                    assert stage_count<=1,"I think there can only be one stage BC"

                if isinstance(bc,hm.FlowBC):
                    data=-bc.data() # Flip to SCHISM convention, negative=inflow
                    if 'time' in data.dims:
                        flow_flag=1
                    else:
                        flow_flag=2

                fp.write(" ".join(["%d"%n for n in
                                   (1+len(bc_edges), # number of nodes
                                    elev_flag,
                                    flow_flag,
                                    temp_flag,
                                    salt_flag,
                                    sed0_flag)]) + "\n")

                if elev_flag==1:
                    elev_data.append(data)
                elif elev_flag==2:
                    fp.write("%.5f ! constant stage\n"%data.item())

                if flow_flag==1:
                    flow_data.append(data)
                elif flow_flag==2:
                    fp.write("%.5f ! constant discharge\n"%data.item())

            # Will have to come back for the others
        if elev_data:
            self.write_th(os.path.join(self.run_dir,'elev.th'),elev_data)
        if flow_data:
            self.write_th(os.path.join(self.run_dir,'flux.th'),flow_data)
            
    def write_th(self,fn,data):
        # data should be a list of data arrays
        # Currently treats all data the same, and will pad out with
        # first/last entry if one of data is shorter than the simulation
        # May at some point have to be more specific to flow vs elevation,
        # different treatment for wind.

        dt_sim=float(self.param['CORE','dt'])
        # min of all timesteps
        data_dt=min( [ np.diff(datum.time.values).min()/np.timedelta64(1,'s')
                       for datum in data] )
        # even multiple of sim_dt. Possible that it just has to be >=dt.
        dt_s=int(np.ceil(max(dt_sim,dt_sim*np.floor(data_dt/dt_sim))))
        dt=np.timedelta64(dt_s,'s')
        nsteps=1+(self.run_stop-self.run_start)/np.timedelta64(dt,'s')

        times=self.run_start + dt*np.arange(nsteps)
        t_s=dt_s*np.arange(nsteps)

        # extra column for times
        result=np.zeros( (len(t_s),1+len(data)), np.float64)
        result[:,0]=t_s

        for di,datum in enumerate(data):
            datum_t_s=(datum.time.values-self.run_start)/np.timedelta64(1,'s')
            result[:,1+di]=np.interp(t_s,
                                     datum_t_s,datum.values)
        with open(fn,'wt') as fp:
            np.savetxt(fp,result)
                    
    def write_roughness_bc(self,bc):
        self.log.warning("Rougness BC not implemented for SCHISM yet")
        
    def write_grid(self):
        self.mark_land_edges()
        # framework operates on elevations in node_z_bed, so force ignore of a 'depth'
        # field in case it exists on self.grid
        self.grid.write_gr3(os.path.join(self.run_dir,'hgrid.gr3'),bc_marks=True,
                            z=-self.grid.nodes['node_z_bed'])
        self.grid.write_gr3(os.path.join(self.run_dir,'drag.gr3'),z=self.grid.nodes['drag'],
                            bc_marks=False)
        self.grid.write_gr3(os.path.join(self.run_dir,'elev.ic'),
                            z=np.maximum(self.grid.nodes['node_z_bed'],
                                         self.initial_water_level()),bc_marks=False)
        
    def mark_open_edges(self,linestring):
        """
        linestring: [N,2] coordinate array or a shapely.geometry.LineString
        returns the bc_id and edge indices
        """
        if not isinstance(linestring,np.ndarray):
            linestring=np.array(linestring.coords)

        g=self.grid
        j_open=g.select_edges_by_polyline(linestring)
        if np.any(g.edges['mark']==g.OPEN):
            new_idx=1+g.edges['bc_id'][ g.edges['mark']==g.OPEN ].max()
        else:
            new_idx=0
            
        g.edges['mark'][j_open]=g.OPEN # no difference here in flow, velocity, or stage BC.
        g.edges['bc_id'][j_open]=new_idx
        return new_idx,j_open
        
    def mark_land_edges(self):
        """
        Once open boundaries have been set, update the remaining
        boundaries to be land.
        """
        g=self.grid
        e2c=g.edge_to_cells()

        # each string starts/ends with the same node
        strings=g.boundary_linestrings(return_nodes=True,sort=True)

        #  Need to walk through the strings, skip over edges that
        #  are already marked open. check whether first/last strings
        #  can be combined.
        #  Add each string as a land boundary.
        land_bc_count=0
        for si,string in enumerate(strings):
            land_strings=[] # this string, broken up into land boundaries
            land_string=[] # string under construction
            for a,b in zip(string[:-1],string[1:]):
                j=g.nodes_to_edge([a,b])
                # Is this edge already part of a bc?
                if g.edges['mark'][j]==0: # nope
                    land_string.append([a,j,b])
                else:
                    # hit an edge that is already BC
                    if land_string:
                        land_strings.append(land_string)
                        land_string=[]
            if land_string: # clean up
                land_strings.append(land_string)
            # Check for joining first and last:
            if len(land_strings)>1 and land_strings[0][0][0]==land_strings[-1][-1][2]:
                land_strings[0]=land_strings.pop(-1) + land_strings[0]
            # And if there is only one and it goes all the way around,
            # break it into two (I think some part of schism
            # can't handle a single continuous boundary)
            if (len(land_strings)==1) and (land_strings[0][0][0]==land_strings[0][-1][-1]):
                ls=land_strings[0]
                L=len(ls)
                land_strings=[ ls[:L//2],ls[L//2:] ]
            # And write marks back to grid
            for land_string in land_strings:
                for a,j,b in land_string:
                    g.edges['mark'][j]=g.LAND
                    g.edges['bc_id'][j]  =land_bc_count
                land_bc_count+=1
        self.log.info("Land bc count: %d"%land_bc_count)

    def write_monitors(self):
        if self.mon_points:
            self.write_monitor_points()
        if self.mon_sections:
            self.log.warning("No support [yet] for sections in SCHISM")
            #self.write_monitor_sections()
            
    monitor_z=None
    def write_monitor_points(self):

        if self.monitor_z is None:
            z_min=self.grid.nodes['node_z_bed'].min()
            z_max=2.0 # ad hoc.  need to be smarter
            self.monitor_z=np.linspace(z_min,z_max,2)

        feats=self.mon_points
        
        count=0
        with open(os.path.join(self.run_dir,'station.in'),'wt') as fp:
            fp.write("1 0 0 0 0 0 1 1 1 !on (1)|off(0) flags for elev, air pressure, windx, windy, T, S, u, v, w\n")
            fp.write(f"{len(feats)*len(self.monitor_z)} !# of stations\n")
            for i in range(len(feats)):
                pnt=np.array(feats[i]['geom'])
                for z in self.monitor_z:
                    fp.write(f"{count+1} {pnt[0]:.3f} {pnt[1]:.3f} {z:.3f}\n")
                    count+=1

    def partition(self):
        self.log.info("Partioning is a no-op in SCHISM")
    def run_simulation(self):
        # Assuming it's not a restart
        outputs=os.path.join(self.run_dir,'outputs')
        if os.path.exists(outputs):
            outputs_old=os.path.join(self.run_dir,'outputs-old')
            if os.path.exists(outputs_old):
                shutil.rmtree(outputs_old)
            shutil.move(outputs,outputs_old)
        os.makedirs(outputs)

        self.run_schism()

    def run_schism(self,mpi='auto',wait=True):
        if mpi=='auto':
            num_procs=self.num_procs
        else:
            num_procs=1

        if num_procs>1:
            try:
                # kludge in different mpiexec
                self.mpi_bin_exe=self.schism_mpi_bin_exe
                self.log.info("Override mpiexec for SCHISM to %s"%self.mpi_bin_exe)
            except AttributeError:
                pass
            
            return self.mpirun([self.schism_bin],working_dir=self.run_dir,wait=wait)
        else:
            self.log.info("Running command: %s"%self.schism_bin)
            return utils.call_with_path(self.schism_bin,self.run_dir)
        
class MultiSchism(multi_ugrid.MultiUgrid):
    def __init__(self,grid_fn,paths,xr_kwargs={}):
        if not isinstance(grid_fn,unstructured_grid.UnstructuredGrid):
            self.grid=unstructured_grid.UnstructuredGrid.read_gr3(grid_fn)
        else:
            self.grid=grid_fn
            
        if isinstance(paths,str):
            paths=glob.glob(paths)
            # more likely to get datasets in order of processor rank
            # with a sort.
            paths.sort()
        self.paths=paths
        self.dss=self.load(**xr_kwargs)
        self.n_ranks=len(self.dss)
        
        # avoid loading grids -- in SCHISM the subdomain grids are
        # not as cleanly independent, but we have explicit mappings

        self.rev_meta={ 'nSCHISM_hgrid_node':'node_dimension',
                        'nSCHISM_hgrid_edge':'edge_dimension',
                        'nSCHISM_hgrid_face':'face_dimension', }

        self.create_global_grid_and_mapping()
        
    def create_global_grid_and_mapping(self):
        self.node_l2g=[] # per rank, 
        self.edge_l2g=[]
        self.cell_l2g=[]

        for rank in range(self.n_ranks):
            ds=self.dss[rank]
            nc_fn=self.paths[rank]
            
            fn=os.path.join(os.path.dirname(nc_fn),'local_to_global_%04d'%rank)
            fp=open(fn,'rt')
            sizes=[int(s) for s in fp.readline().strip().split()]
            comment=fp.readline()
            Ncell=int(fp.readline().strip())
            assert Ncell==ds.dims['nSCHISM_hgrid_face']
            c_map=np.loadtxt(fp,np.int32,max_rows=Ncell) - 1
            Nnode=int(fp.readline().strip())
            assert Nnode==ds.dims['nSCHISM_hgrid_node']
            n_map=np.loadtxt(fp,np.int32,max_rows=Nnode) - 1
            Nedge=int(fp.readline().strip())
            assert Nedge==ds.dims['nSCHISM_hgrid_edge']
            j_map=np.loadtxt(fp,np.int32,max_rows=Nedge) - 1

            # From the source file, these all include the local index.
            # verified that those are sequential, so with no further
            # munging we're done:
            self.node_l2g.append(n_map[:,1])
            self.edge_l2g.append(j_map[:,1])
            self.cell_l2g.append(c_map[:,1])

# with ibc=1, ibtp=0, get an error that "true barotropic models cannot have tracer models"
