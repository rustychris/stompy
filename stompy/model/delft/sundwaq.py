"""
tools related to processing SUNTANS output for DWAQ input.
requires instrument version of SUNTANS code for the flux
integration
"""
import glob
import numpy as np
import os

from ... import utils
from ..suntans import sunreader
from ...io import qnc
from . import waq_scenario
from ..suntans import forcing
from ...spatial import wkb2shp
from ...grid import unstructured_grid

def sun_to_flowgeom(sun,proc,filename,overwrite=True):
    """
    given a SunReader object, write the 2-D grid for the given
    processor out to the a dwaq-compatible xxxx_flowgeom.nc
    file.
    overwrite: if True, silently overwrite existing output file.

    tries to write enough info to recreate the grid
    """
    # Figure out some ownership:
    g=sun.grid(proc) # processor-local grid
    gg=sun.grid() # global grid

    # Cell ownership
    is_local=np.zeros(g.Ncells(),'b1')
    is_local[ sun.proc_nonghost_cells(proc) ] = True

    g2l=sun.map_global_cells_to_local_cells(allow_cache=False,honor_ghosts=True)
    l2g=sun.map_local_to_global(proc)
    my_cell_procs=g2l['proc'][l2g] # map local cell index to home processor
    assert np.all( (my_cell_procs==proc) == is_local )

    # Edge ownership - note that marker 6 edges are not output!
    # and marker 5 edges are given to the lower numbered processor
    edge_marks=g.edges[:,2]
    # edges which will be output - skip closed and super-ghosty edges, but 
    # include shared edges, flow, open boundaries
    edge_sel= (edge_marks != 6) & (edge_marks!=1) 
    bdry_edges=(edge_marks>0)&(edge_marks<4) # non ghost edges which have only 1 cell nbr
    edge_cells=g.edges[edge_sel,3:] # neighbors of edges to be output
    edge_cells[ bdry_edges[edge_sel],1 ] = edge_cells[ bdry_edges[edge_sel], 0]
    assert np.all(edge_cells>=0)
    edge_owners=g2l['proc'][l2g[edge_cells].min(axis=1)]

    cdata=sun.celldata(proc)
    nc=qnc.empty(fn=filename, # os.path.join(dwaq_dir,'%04d_flowgeom.nc'%proc),
                 overwrite=overwrite,
                 # DWAQ requires netcdf3
                 format='NETCDF3_CLASSIC')

    mesh_name='FlowMesh' # for UGRID references

    nc.createDimension('nFlowElem',g.Ncells())
    nc.createDimension('nFlowElemMaxNode',3)
    # other dimensions created on demand.

    # cell centers
    nc['FlowElem_xcc']['nFlowElem']=cdata[:,0]
    nc.FlowElem_xcc.units='m'
    nc.FlowElem_xcc.standard_name='projection_x_coordinate'
    nc.FlowElem_xcc.long_name="Flow element circumcenter x"
    nc.FlowElem_xcc.bounds='FlowElemContour_x' # ?
    nc.FlowElem_xcc.grid_mapping='projected_coordinate_system'

    nc['FlowElem_ycc']['nFlowElem']=cdata[:,1]
    nc.FlowElem_ycc.units='m'
    nc.FlowElem_ycc.standard_name='projection_y_coordinate'
    nc.FlowElem_ycc.long_name="Flow element circumcenter y"
    nc.FlowElem_ycc.bounds='FlowElemContour_y' # ?
    nc.FlowElem_ycc.grid_mapping='projected_coordinate_system'

    nc['FlowElem_zcc']['nFlowElem']=cdata[:,3]
    nc.FlowElem_zcc.long_name="Flow element average bottom level (average of all corners)."
    nc.FlowElem_zcc.positive='down'
    nc.FlowElem_zcc.mesh=mesh_name
    nc.FlowElem_zcc.location='face'

    nc['FlowElem_bac']['nFlowElem']=cdata[:,2]
    nc.FlowElem_bac.long_name="Flow element area"
    nc.FlowElem_bac.units='m2'
    nc.FlowElem_bac.standard_name='cell_area'
    nc.FlowElem_bac.mesh=mesh_name
    nc.FlowElem_bac.location='face'

    nc['FlowElemContour_x']['nFlowElem','nFlowElemContourPts'] = g.points[g.cells[:,:],0]
    nc.FlowElemContour_x.units='m'
    nc.FlowElemContour_x.standard_name="projection_x_coordinate"
    nc.FlowElemContour_x.long_name="List of x-points forming flow element"
    nc.FlowElemContour_x.grid_mapping='projected_coordinate_system'

    nc['FlowElemContour_y']['nFlowElem','nFlowElemContourPts'] = g.points[g.cells[:,:],1]
    nc.FlowElemContour_y.units='m'
    nc.FlowElemContour_y.standard_name="projection_y_coordinate"
    nc.FlowElemContour_y.long_name="List of y-points forming flow element"
    nc.FlowElemContour_y.grid_mapping='projected_coordinate_system'

    # not sure how this differs from zcc, aside from sign.
    nc['FlowElem_bl']['nFlowElem']=-cdata[:,3]
    nc.FlowElem_bl.units='m'
    nc.FlowElem_bl.positive='up'
    nc.FlowElem_bl.standard_name='sea_floor_depth'
    nc.FlowElem_bl.long_name="Bottom level at flow element's circumcenter."
    nc.FlowElem_bl.grid_mapping='projected_coordinate_system'
    nc.FlowElem_bl.mesh=mesh_name
    nc.FlowElem_bl.location='face'

    # should include flow/open boundaries.  just not closed boundaries.

    links=1+g.edges[edge_sel,3:5] # to 1-based
    bdry=links<=0
    nelt=len(nc.FlowElem_xcc)
    # in .poi files, boundaries are negative, but here, they are appended to
    # the regular
    links[bdry] = 1+np.arange(np.sum(bdry))
    nc['FlowLink']['nFlowLink','nFlowLinkPts']=links.astype(np.int32)
    nc.FlowLink.long_name="link/interface between two flow elements"

    nc['FlowLinkType']['nFlowLink']=(2*np.ones(links.shape[0])).astype(np.int32)
    nc.FlowLinkType.long_name="type of flowlink"
    nc.FlowLinkType.valid_range=[1,2]
    nc.FlowLinkType.flag_values=[1,2]
    nc.FlowLinkType.flag_meanings="link_between_1D_flow_elements link_between_2D_flow_elements"

    ec=g.edge_centers()[edge_sel]
    nc['FlowLink_xu']['nFlowLink']=ec[:,0]
    nc.FlowLink_xu.units='m'
    nc.FlowLink_xu.standard_name='projection_x_coordinate'
    nc.FlowLink_xu.long_name='Center coordinate of net link (velocity point).'
    nc.FlowLink_xu.grid_mapping='projected_coordinate_system'

    nc['FlowLink_yu']['nFlowLink']=ec[:,1]
    nc.FlowLink_yu.units='m'
    nc.FlowLink_yu.standard_name='projection_y_coordinate'
    nc.FlowLink_yu.long_name='Center coordinate of net link (velocity point).'
    nc.FlowLink_yu.grid_mapping='projected_coordinate_system'

    # for now, skip lat/lon fields, projection definition..

    if 0:
        # single processor only
        nc['FlowElemDomain']['nFlowElem']=(proc*np.ones(g.Ncells())).astype(np.int16)
        nc['FlowLinkDomain']['nFlowLink']=(proc*np.ones(np.sum(edge_sel))).astype(np.int16)
    else:
        # single or multiple processors
        nc['FlowElemDomain']['nFlowElem']=my_cell_procs.astype(np.int16)
        nc['FlowLinkDomain']['nFlowLink']=edge_owners.astype(np.int16)

    nc.FlowElemDomain.long_name="Domain number of flow element"
    nc.FlowLinkDomain.long_name="Domain number of flow link"
        
    # used to do silly thing with closest_cell() which isn't robust.
    nc['FlowElemGlobalNr']['nFlowElem']=1+l2g
    nc.FlowElemGlobalNr.long_name="Global flow element numbering"

    #---- UGRID-ish metadata and supplementals ----
    mesh=nc.createVariable(mesh_name,'i4')
    mesh.cf_role='mesh_topology'
    mesh.long_name = "Topology data of 2D unstructured mesh" 
    mesh.dimension = 2

    nc['Node_x']['nNode'] = g.points[:,0]
    nc['Node_y']['nNode'] = g.points[:,1]
    mesh.node_coordinates = "Node_x Node_y"

    nc['FlowElemContour_node']['nFlowElem','nFlowElemContourPts'] = g.cells.astype('i4')
    face_nodes=nc.FlowElemContour_node
    face_nodes.cf_role='face_node_connectivity'
    face_nodes.long_name="Maps faces to constituent vertices/nodes"
    face_nodes.start_index=0
    mesh.face_node_connectivity = 'FlowElemContour_node'

    nc['FlowEdge_node']['nFlowEdge','nEdgePts']=g.edges[:,:2].astype('i4')
    edge_nodes=nc.FlowEdge_node
    edge_nodes.cf_role='edge_node_connectivity'
    edge_nodes.long_name="Maps edge to constituent vertices"
    edge_nodes.start_index=0

    mesh.edge_node_connectivity = 'FlowEdge_node' # attribute required if variables will be defined on edges
    # mesh.edge_coordinates = "Mesh2_edge_x Mesh2_edge_y" #  optional attribute (requires edge_node_connectivity)
    mesh.face_coordinates = "FlowElem_xcc FlowElem_ycc" # optional attribute
    # mesh.face_edge_connectivity = "FlowLink" # optional attribute (requires edge_node_connectivity)
    mesh.face_face_connectivity = "FlowLink" # optional attribute

    z_var_name=z_dim_name="n%s_layers"%mesh_name

    # these are a bit fake, as in any given water column the cell with the freesurface
    # and the cell with the bed may be truncated
    z_bookended = np.concatenate( ([0],-sun.z_levels()) )
    nc[z_var_name][z_dim_name] = 0.5*(z_bookended[:-1] + z_bookended[1:])
    layers=nc.variables[z_var_name]
    layers.standard_name = "ocean_zlevel_coordinate" 
    layers.long_name = "elevation at layer midpoints" 
    layers.positive = "up"
    layers.units = "meters"

    # And add a bounds attribute and variable to cover the distribution of cell interfaces
    # note that this doesn't bother with how layers are truncated at the bed or surface
    bounds_name = z_var_name+"_bnds"
    layers.bounds = bounds_name
    bounds=np.concatenate( (z_bookended[:-1,None],
                            z_bookended[1:,None]),axis=1)
    nc[bounds_name][z_dim_name,'d2']=bounds

    # Global attributes:
    nc.setncattr('institution',"San Francisco Estuary Institute")
    nc.setncattr('references',"http://www.deltares.nl")
    nc.setncattr('source',"Python/Delft tools, rustyh@sfei.org")
    nc.setncattr('history',"Converted from SUNTANS run")
    nc.setncattr('Conventions',"CF-1.5:Deltares-0.1")

    nc.close()

    
    
def postprocess(sun=None,sun_dir=None,force=False):
    """
    Take care of any python-side postprocessing of a suntans run.
    Namely, this creates the sun_nnnn_flowgeom.nc files.
    """
    sun = sun or sunreader.SunReader(sun_dir)
    nprocs=sun.num_processors()
    dfm_dwaq_path=os.path.join(sun.datadir,'dwaq')


    for proc in range(nprocs):
        nc_fn=os.path.join(dfm_dwaq_path,
                           "DFM_DELWAQ_sun_%04d"%proc,
                           "sun_%04d_flowgeom.nc"%proc)

        if force or not os.path.exists( nc_fn ):
            sun_to_flowgeom(sun,proc,nc_fn)
    
class SunHydro(waq_scenario.HydroFiles):
    """ specialization for SUNTANS-based hydro.
    """
    def __init__(self,sun,flow_shps,*a,**k):
        self.sun=sun
        self.flow_shps=flow_shps
        super(SunHydro,self).__init__(*a,**k)

    _bc_groups=None
    def group_boundary_elements(self,force=False):
        """ map all element ids (0-based) to either -1 (not a boundary)
        or a nonnegative id corresponding to contiguous boundary elements.

         - why are we grouping elements?  shouldn't this be grouping 
           exchanges?  there is an implicit connection of each boundary
           exchange to its internal segment, and boundaries are grouped
           according to those internal segments.  Effectively cannot have
           two disparate boundaries going into the same element.  Looking
           at the grid, this is mostly the case, with the exception of a pair
           of small tributaries in Marin.
        """
        if force or self._bc_groups is None:
            # This part is the same as in waq_scenario
            g=self.grid()
            if g is None:
                return super(SunHydro,self).group_boundary_elements()

            self.infer_2d_elements()

            poi=self.pointers
            bc_sel = (poi[:,0]<0)
            bc_elts = np.unique(self.seg_to_2d_element[ poi[bc_sel,1]-1 ])

            groups=np.zeros(self.n_2d_elements,self.group_dtype)
            groups['id']-=1

            gforce=forcing.GlobalForcing(sun=self.sun)
            sun_g=self.sun.grid()

            def node_sun_to_g(n):
                return g.select_nodes_nearest(sun_g.points[n])

            # map group id as returned by this method to a dict with items 
            # like which shapefile did it come from, index in that shapefile,
            # and fields from the feature.
            # note that it is possible for two boundary flows to enter the same
            # cell - only the first will be marked, with the second feature
            # skipped in both groups and bc_group_mapping
            # self.bc_group_mapping={} 
            ngroups=0

            for flow_shp in self.flow_shps:
                flows=wkb2shp.shp2geom(flow_shp)
                sun_groups=gforce.add_groups_bulk(defs=flows)

                for feat_id in range(len(flows)):
                    grp=sun_groups[feat_id]
                    if grp.cell_based():
                        sun_cells=grp.cells
                        cells=[]
                        for cell in sun_cells:
                            g_nodes=[node_sun_to_g(n)
                                     for n in sun_g.cells[cell]]
                            cells.append( g.nodes_to_cell(g_nodes) )

                        cells=np.array(cells)
                    else:
                        # for the purposes of bc_groups, figure out the
                        # respective cells
                        cells=[]
                        for sun_e in grp.edges:
                            sun_e_nodes=sun_g.edges[sun_e,:2]
                            e=g.nodes_to_edge(node_sun_to_g(sun_e_nodes[0]),
                                              node_sun_to_g(sun_e_nodes[1]))
                            assert e is not None
                            cells.append(g.edge_to_cells(e))
                        cells=np.array(cells)
                        cells=cells[cells>=0]

                    details=dict(flow_shp=flow_shp,
                                 feat_id=feat_id)
                    for n in flows.dtype.names:
                        details[n]=flows[n][feat_id]

                    # limit this to cells which are not already marked, but *are*
                    # in bc_elts
                    cells=[c for c in cells
                           if (groups['id'][c]<0) and (c in bc_elts) ] 
                    if len(cells):
                        groups['id'][cells] = ngroups
                        groups['name'][cells]=details.get('name','group %d'%ngroups)
                        groups['attrs'][cells] = details
                        # self.bc_group_mapping[ngroups]=details
                        ngroups+=1
                    else:
                        self.log.warning("Feature %d from %s (name=%s) overlaps another flow or wasn't" 
                                         " found as a boundary, "
                                         " and will be skipped"%(feat_id,flow_shp,
                                                                 details.get('name','n/a')))

            # anything not marked already then gets grouped by adjacency and marked
            # the same way as before - see waq_scenario.py for more comments
            def adjacent_cells(g,c,candidates):
                a=list(g.cell_to_adjacent_boundary_cells(c))
                b=list(g.cell_to_cells(c))
                nbrs=filter(lambda cc: cc in candidates,a+b)
                return np.unique(nbrs)
            def trav(c,mark):
                groups['id'][c]=mark
                groups['name'][c]="group %d"%mark
                for nbr in adjacent_cells(g,c,bc_elts):
                    if groups['id'][nbr]<0:
                        trav(nbr,mark)

            ngroups=1+groups['id'].max()

            for bc_elt in bc_elts:
                if groups['id'][bc_elt]<0:
                    # This is the part where if there are other cells 
                    # which are part of the same forcing group, they should
                    # all get this value
                    trav(bc_elt,ngroups)
                    ngroups+=1
            self._bc_groups=groups
        return self._bc_groups

    _bc_lgroups=None
    def group_boundary_links(self,force=False):
        """ 
        map all link ids (0-based) to either -1 (not a boundary)
        or a nonnegative id corresponding to contiguous boundary elements.

         - why are we grouping elements?  shouldn't this be grouping 
           exchanges?  there is an implicit connection of each boundary
           exchange to its internal segment, and boundaries are grouped
           according to those internal segments.  Effectively cannot have
           two disparate boundaries going into the same element.  Looking
           at the grid, this is mostly the case, with the exception of a pair
           of small tributaries in Marin.
        """
        if force or self._bc_lgroups is None:
            # This part is the same as in waq_scenario
            g=self.grid()
            if g is None:
                return super(SunHydro,self).group_boundary_links()

            self.infer_2d_links()

            poi0=self.pointers-1

            bc_sel = (poi0[:,0]<0)
            bc_elts = np.unique(self.seg_to_2d_element[ poi0[bc_sel,1] ])
            bc_links=np.nonzero( self.links[:,0]<0 )[0]

            lgroups=np.zeros(self.n_2d_links,self.link_group_dtype)
            lgroups['id']-=1

            gforce=forcing.GlobalForcing(sun=self.sun)
            sun_g=self.sun.grid()

            def node_sun_to_g(n):
                return g.select_nodes_nearest(sun_g.points[n])

            # map group id as returned by this method to a dict with items 
            # like which shapefile did it come from, index in that shapefile,
            # and fields from the feature.
            # note that it is possible for two boundary flows to enter the same
            # cell - only the first will be marked, with the second feature
            # skipped in both groups and bc_group_mapping
            # self.bc_group_mapping={} 
            ngroups=0

            for flow_shp in self.flow_shps:
                flows=wkb2shp.shp2geom(flow_shp)
                sun_groups=gforce.add_groups_bulk(defs=flows)

                for feat_id in range(len(flows)):
                    grp=sun_groups[feat_id]
                    if grp.cell_based():
                        sun_cells=grp.cells
                        cells=[]
                        for cell in sun_cells:
                            g_nodes=[node_sun_to_g(n)
                                     for n in sun_g.cells[cell]]
                            cells.append( g.nodes_to_cell(g_nodes) )
                        cells=np.array(cells)
                    else:
                        # Here is where it gets tricky -
                        # gforce tells us something about the cells or edges which are
                        # forced in the original hydro.
                        # we'd like to associate those with links.
                        # since links have little geometry, we just have to hope that
                        # there isn't more than one choice.
                        cells=[]
                        for sun_e in grp.edges:
                            sun_e_nodes=sun_g.edges[sun_e,:2]
                            e=g.nodes_to_edge(node_sun_to_g(sun_e_nodes[0]),
                                              node_sun_to_g(sun_e_nodes[1]))
                            assert e is not None
                            cells.append(g.edge_to_cells(e))
                        cells=np.array(cells)
                        cells=cells[cells>=0]

                    details=dict(flow_shp=flow_shp,
                                 feat_id=feat_id)
                    for n in flows.dtype.names:
                        details[n]=flows[n][feat_id]

                    # which boundary links are associated with these cells?
                    links=[l for l in bc_links
                           if self.links[l,1] in cells]
                    # are any of these already marked?
                    clean_links=[]
                    for l in links:
                        if lgroups['id'][l]<0:
                            clean_links.append(l)
                        else:
                            self.log.warning("Feature %d from %s (name=%s) maps to a boundary link"
                                             " which is already associated with another flow"
                                             " and will be skipped"%(feat_id,flow_shp,
                                                                     details.get('name','n/a')))
                    if len(clean_links):
                        lgroups['id'][clean_links] = ngroups
                        lgroups['name'][clean_links]=details.get('name','group %d'%ngroups)
                        lgroups['attrs'][clean_links] = details
                        ngroups+=1
                    elif not links: 
                        self.log.warning("Feature %d from %s (name=%s) wasn't" 
                                         " found as a boundary, "
                                         " and will be skipped"%(feat_id,flow_shp,
                                                                 details.get('name','n/a')))

            # anything not marked already then gets grouped by itself.

            n_stragglers=0
            for bc_link in bc_links:
                if lgroups['id'][bc_link]<0:
                    lgroups['id'][bc_link]=ngroups
                    lgroups['name'][bc_link]="group %d"%ngroups
                    lgroups['attrs'][bc_link]={}
                    n_stragglers+=1
                    ngroups+=1
            if n_stragglers>0:
                self.log.warning("%d boundary links had no flow feature"%n_stragglers)
            self._bc_lgroups=lgroups
        return self._bc_lgroups

    
    # smarter enumeration of available parameters:
    def add_parameters(self,hyd):
        super(SunHydro,self).add_parameters(hyd)

        # SURF, bottomdept we can leave alone - constant in time
        # so not too expensive to handle the usual way.
        
        # getting these filenames is the last thing:
        for label,pname in [('vert-diffusion-file','VertDisper'),
                            ('salinity-file','salinity'),
                            ('temperature-file','temperature')]:
            fn=self.get_path(label)
            if not os.path.exists(fn):
                self.log.info("SunDWAQ: seg function %s not found"%fn)
                continue
            self.log.info("SunDWAQ: seg_fn %s will use %s, symlink=%s"%(pname,fn,
                                                                              self.enable_write_symlink))
            hyd[pname]=waq_scenario.ParameterSpatioTemporal(seg_func_file=fn,
                                                            enable_write_symlink=self.enable_write_symlink)
        
        return hyd


# path less clear for dealing with multiprocessor runs.
#  option A: make a list of SunHydro's, pass those to a version of a MultiAggregator.
#    - where would it match up boundary elements with names?  either SunHydro does this
#      per subdomain, carefully, and then MultiAggregator combines, or some of that logic
#      is brought into MultiAggregator, operating on the global grid.
#    - this option is nice because we don't necessarily have to make a full global dataset.
#      can slice out a bit as needed.
#  option B: compile the run into a global run, which can then be treated like a single
#      processor suntans run.  painful, but one-time pain.
#  for the moment, go with option B, as it is basically implemented (see .../splicer.py)


class SunSplicer(object):
    """ mixin for a domain which adds some commands for waq output.
    splicing multi-processor output together, post-processing a grid
    """
    # could add agg_shp as possible input here...
    def cmd_merge(self,sun_dir,dense=False):
        self.log.info("Postprocess to get per-processor flowgeom.nc")
        self.cmd_suntans_post(sun_dir)
        
        self.log.info("Will aggregate to global grid - reading that now")
        agg_grid=unstructured_grid.SuntansGrid(sun_dir)

        sun=sunreader.SunReader(sun_dir)
        run_prefix="sun"
        dwaq_path=os.path.join(sun.datadir,'dwaq')

        agg=waq_scenario.HydroMultiAggregator(agg_shp=agg_grid,
                                              sparse_layers=(not dense),
                                              run_prefix="sun",
                                              merge_only=True,
                                              path=os.path.join(sun_dir,'dwaq'),
                                              skip_load_basic=True)
        if 1:
            self.log.info('Precomputing elt-to-elt distances')
            cc=agg_grid.cells_center()
            ec=agg_grid.edges_center()
            edge_segs=cc[ agg_grid.edges['cells'] ] - ec[:,None,:]
            lengths=utils.dist(edge_segs)

            lookup={}
            for j,(c1,c2) in enumerate(agg_grid.edges['cells']):
                if (c1>=0) & (c2>=0):
                    lookup[ (c1,c2) ]=(lengths[j,0],lengths[j,1])
            agg.lookup=lookup

        if 1:
            # for the 1:1 merge case, incoming areas should be fine,
            # so no need to waste time on making the areas more constant
            # than constant
            agg.exch_z_area_constant = False
        
        agg.load_basic()
        
        if not dense:
            self.log.info("Creating sparse merged hydrodynamics")
            output_base_path=os.path.join(dwaq_path,"global")
        else:
            self.log.info("Creating dense merged hydrodynamics")
            output_base_path=os.path.join(dwaq_path,"global-dense")

        class SpliceScenario(waq_scenario.Scenario):
            base_path=output_base_path
            name="spliced"

        self.log.info("Creating Scenario to drive splicing")
        scen=SpliceScenario(hydro=agg)
        scen.start_time = scen.time0+scen.scu*scen.hydro.t_secs[0]
        scen.stop_time  = scen.time0+scen.scu*scen.hydro.t_secs[-1]

        # going to take a while!
        self.log.info("Writing hydrodynamics")
        scen.write_hydro()

        # not exactly tested inside this context
        self.log.info("Writing boundary identifiers")
        self.cmd_label_merged_boundaries(sun_dir,output_dir=output_base_path)

    def cmd_label_merged_boundaries(self,sun_dir,output_dir=None):
        """
        assumes merge has mostly completed - and then writes out a boundary-links.csv
        file for the global grid.
        """
        sun=sunreader.SunReader(sun_dir)
        if output_dir is None:
            # defaults location of dense output
            output_dir=os.path.join(sun.datadir,'dwaq',"global-dense")
            
        hyd_fn=glob.glob(os.path.join(output_dir,'*.hyd'))[0]
                         
        hydro=SunHydro(sun=sun,hyd_path=hyd_fn,flow_shps=[self.flows_shp])

        class SpliceScenario(waq_scenario.Scenario):
            base_path=output_dir
            name="spliced"

        scen=SpliceScenario(hydro=hydro)

        self.log.info("Writing labels")
        scen.hydro.write_boundary_links()
        
    def cmd_suntans_post(self,sun_dir):
        sun=sunreader.SunReader(sun_dir)
        postprocess(sun=sun,force=False)

    def cmd_write_agg_shp(self,sun_dir,shp=None):
        if shp is None:
            shp="grid-cells.shp"

        if not os.path.exists(agg_shp):
            sun=sunreader.SunReader(sun_dir)
            g=sun.grid()
            fields=np.zeros(g.Ncells(),dtype=[('name',object)])
            fields['name'] =["%d"%c for c in range(g.Ncells())] 
            g.write_cells_shp(agg_shp,overwrite=True,fields=fields)
