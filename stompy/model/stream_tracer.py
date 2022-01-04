import numpy as np
import xarray as xr
from stompy.grid import unstructured_grid
from stompy import utils
from shapely import geometry
# Then a Perot-like calculation on each cell in the dual
def U_perot(g,Q,V):
    cc=g.cells_center()
    ec=g.edges_center()
    normals=g.edges_normals()

    e2c=g.edge_to_cells()
    Uc=np.zeros((g.Ncells(),2),np.float64)
    dist_edge_face=np.nan*np.zeros( (g.Ncells(),g.max_sides), np.float64)

    for c in np.arange(g.Ncells()):
        js=g.cell_to_edges(c)
        for nf,j in enumerate(js):
            # normal points from cell 0 to cell 1
            if e2c[j,0]==c: # normal points away from c
                csgn=1
            else:
                csgn=-1
            dist_edge_face[c,nf]=np.dot( (ec[j]-cc[c]), normals[j] ) * csgn
            # Uc ~ m3/s * m
            Uc[c,:] += Q[j]*normals[j]*dist_edge_face[c,nf]
    Uc /= np.maximum(V,0.01)[:,None]
    return Uc

def rotated_hydro(hydro):
    """
    hydro: xarray Dataset with a grid and edge-centered fluxes in Q.
    returns a new Dataset with the dual grid, 90deg rotated edge velocities,
    and cell-centered vector velocities.
    """
    g=unstructured_grid.UnstructuredGrid.from_ugrid(hydro)
    
    # using centroids yields edges that aren't orthogonal, but the centers
    # are nicely centered in the cells.
    # As long as the grid is nice and orthogonal, it should be okay
    # use true circumcenters.
    gd=g.create_dual(center='circumcenter',create_cells=True,remove_1d=True)

    # Need to get the rotated Q
    en=g.edges_normals() # edge normals that go with Q on the original grid
    enr=utils.rot(np.pi/2,en) 
    dual_en=gd.edges_normals()
    j_orig=gd.edges['dual_edge']
    edge_sign_to_dual=np.round( (enr[j_orig]*dual_en).sum(axis=1) )
    Qdual=hydro.Q.values[j_orig]*edge_sign_to_dual

    Urot_dual=U_perot(gd,Qdual,gd.cells_area())

    ds=gd.write_to_xarray()
    ds['Q']=('edge',),Qdual
    ds['u']=('face','xy'),Urot_dual
    return ds

def steady_streamline_oneway(g,Uc,x0,max_t=3600,max_steps=1000,max_dist=None,
                             u_min=1e-3,bidir=False):
    """
    Trace a streamline downstream 
    g: unstructured grid
    Uc: cell centered velocity vectors

    bidir: interpret velocities as principal directions, i.e. unique only down to
    a sign.  when enabled, on each crossing into a new cell, the sign of the
    velocity is resolved to be consistent with the last cell.

    returns Dataset with positions x, cells, times
    """
    # trace some streamlines
    x0=np.asarray(x0)
        
    c=g.select_cells_nearest(x0,inside=True)
    if c is None: # can happen with the dual grid
        c=-1
           
    t=0.0 # steady field, start the counter at 0.0
    x=x0.copy()
    times=[t]
    pnts=[x.copy()]
    cells=[c]
    dist=0.0
    stop_condition="none"

    # This part is taking a lot of time -- allow precomputed values in the grid
    try:
        edge_norm=g.edges['normal']
    except ValueError:
        edge_norm=g.edges_normals()
    try:
        edge_ctr=g.edges['center']
    except ValueError:
        edge_ctr=g.edges_center()

    e2c=g.edges['cells']

    def is_convergent(j,ca,Ua,cb,Ub):
        # True if a point on edge j will remain on edge j
        # due to cell velocities converging towards it
        nc1,nc2=e2c[j,:]
        if nc1<0:
            conv_c1=True
        elif nc1==ca:
            conv_c1=np.dot(Ua,edge_norm[j])>=0.0
        elif nc1==cb:
            conv_c1=np.dot(Ub,edge_norm[j])>=0.0
        else:
            raise Exception("BUG!")
        
        if nc2<0:
            conv_c2=True
        elif nc2==ca:
            conv_c2=np.dot(Ua,edge_norm[j])<=0.0
        elif nc2==cb:
            conv_c2=np.dot(Ub,edge_norm[j])<=0.0
        else:
            raise Exception("BUG!")
            
        return (conv_c1 and conv_c2)

    c_U=Uc[c,:] # track the velocity for the current cell c
        
    while (t<max_t) and (c>=0):
        dt_max_edge=np.inf # longest time step we're allowed based on hitting an edge
        j_cross=None
        c_cross=None # the cell that would be entered

        for j in g.cell_to_edges(c):
            if g.edges['cells'][j,1]==c: # ~checked
                # normals point from cell 0 to cell 1
                csgn=-1
            else:
                csgn=1

            out_normal=csgn*edge_norm[j] # normal of edge j pointing away from cell c

            d_xy_n = edge_ctr[j] - x # vector from xy to a point on the edge

            # perpendicular distance
            dp_xy_n=d_xy_n[0] * out_normal[0] + d_xy_n[1]*out_normal[1]

            if dp_xy_n<0.0: # roundoff error
                dp_xy_n=0.0

            closing=(c_U[0]*out_normal[0] + c_U[1]*out_normal[1])
            if closing<=0.0: continue # moving away from or parallel to that edge

            # what cell would we be entering?
            if e2c[j,0]==c:
                nbr_c=e2c[j,1]
            elif e2c[j,1]==c:
                nbr_c=e2c[j,0]
            else:
                assert False
                
            #if (dp_xy_n==0.0):
            #    print("On edge j=%d, dp_xy_n is zero, and closing is %f"%(j,closing))

            dt_j=dp_xy_n/closing
            if dt_j<dt_max_edge: # dt_j>0 redundant with dp_xy_n==0.0
                j_cross=j
                c_cross=nbr_c
                dt_max_edge=dt_j

        if (j_cross is not None) and (c_cross>=0):
            c_cross_U=Uc[c_cross,:]
            if bidir and (np.dot(c_U,c_cross_U)<0):
                # print("Flip cells %d -- %d"%(c,c_cross))
                c_cross_U=-c_cross_U # don't modify Uc!
        else:
            c_cross_U=None # catch errors by unsetting this
                
        # Special case for sliding along an edge
        # note that we only want to do this when the edge is convergent.
        if dt_max_edge==0.0 and j_cross is not None and is_convergent(j_cross,c,c_U,c_cross,c_cross_U):
            # print("Moving along edge")
            edge_tan=np.array([-edge_norm[j_cross,1],
                               edge_norm[j_cross,0]])
            Utan=np.dot(c_U,edge_tan)
            # so edge_norm points from left to right.
            # and edge_tan points from n1 to n2
            #  n2
            #  | ---> norm
            #  |
            #  n1
            #  
            if Utan>0: #moving toward n2
                to_node=g.edges['nodes'][j_cross,1]
            else:  # moving toward n1
                to_node=g.edges['nodes'][j_cross,0]
            dist=utils.mag( x - g.nodes['x'][to_node] )
            dt_max_edge=dist/np.abs(Utan)
            t_max_edge=t+dt_max_edge
            
            if t_max_edge>max_t:
                # didn't make it to the node
                dt=max_t-t
                t=max_t
                delta=Utan*edge_tan*dt
                x+=delta
            else:
                # move all the way and exactly to the node
                x=g.nodes['x'][to_node]
                t=t+dt_max_edge
                # And get off of this edge
                for j_node in g.node_to_edges(to_node):
                    if j_node==j_cross: continue
                    if c==e2c[j_node,0]:
                        j_cross=j_node
                        c_cross=e2c[j_node,1]
                        break
                    elif c==e2c[j_node,1]:
                        j_cross=j_node
                        c_cross=e2c[j_node,0]
                        break
                else:
                    raise Exception("Couldn't find a good edge after going through node")
                # update c_cross_U since we changed c_cross
                c_cross_U=Uc[c_cross,:]
                if bidir and (np.dot(c_U,c_cross_U)<0):
                    c_cross_U=-c_cross_U # don't modify Uc!
        else:
            t_max_edge=t+dt_max_edge
            if t_max_edge>max_t:
                # don't make it to the edge
                dt=max_t-t
                t=max_t
                j_cross=None
            else:
                # this step will take us to the edge j_cross
                dt=dt_max_edge
                t=t_max_edge
            
            # Take the step
            x += c_U*dt
            
        dist += utils.dist(x,pnts[-1])
        pnts.append(x.copy())
        cells.append(c)
        times.append(t)

        if j_cross is not None: # crossing an edge
            if c_cross<0: # leaving the domain
                stop_condition="leave_domain"
                break

            c=c_cross
            c_U=c_cross_U
            c_cross=None # catch errors
            c_cross_U=None

            # with roundoff, good to make sure that we are properly on the
            # line segment of j_cross
            nodes=g.nodes['x'][ g.edges['nodes'][j_cross] ]
            
            tangent=nodes[1]-nodes[0]
            edgelen=utils.mag(tangent)
            tangent /= edgelen
            
            alpha=np.dot( x-nodes[0], tangent ) / edgelen
            eps=1e-4
            if alpha<eps: 
                # print('alpha correction %f => %f'%(alpha,eps))
                alpha=1e-4
            elif alpha>1-eps:
                # print('alpha correction %f => %f'%(alpha,1-eps))
                alpha=1-eps
            x=(1-alpha)*nodes[0] + alpha*nodes[1]
            pnts[-1]=x.copy()
            
            umag=utils.mag(Uc[c])
            if umag<=u_min:
                # should only happen with rotate velocities
                # means we hit shore.
                break
        if len(pnts)>=max_steps:
            stop_condition="max_steps"
            break
        if max_dist and (dist>=max_dist):
            stop_condition="max_dist"
            break
    if t>=max_t:
        stop_condition="max_t"

    ds=xr.Dataset()
    ds['time']=('time',),np.array(times)
    ds['x']=('time','xy'),np.array(pnts)
    ds['cell']=('time',),np.array(cells)
    ds['stop_condition']=(),stop_condition
    
    return ds
    
def steady_streamline_twoways(g,Uc,x0,**kw):
    """
    Trace upstream and downstream with velocities Uc, concatenate
    the results and return dataset.
    Note that there may be repeated points in the trajectory -- this is
    currently left in place to help with debugging, since the points may
    be repeated but reflect different cells or stopping conditions.
    """
    ds_fwd=steady_streamline_oneway(g,Uc,x0,**kw)
    ds_rev=steady_streamline_oneway(g,-Uc,x0,**kw)
    ds_rev.time.values[...] *= -1
    ds=xr.concat( [ds_rev.isel(time=slice(None,None,-1)), ds_fwd],
                      dim='time' )
    del ds['stop_condition']
    ds['stop_condition']= ('leg',), [ds_rev.stop_condition.values,
                                     ds_fwd.stop_condition.values]
    ds['root']=(),len(ds_rev.time.values)
    return ds

def prepare_grid(g):
    g.add_edge_field('normal',g.edges_normals(),on_exists='overwrite')
    g.add_edge_field('center',g.edges_center(),on_exists='overwrite')
    

class StreamDistance(object):
    def __init__(self,g,U,source_ds,
                 alongs=None,acrosses=None,
                 bidir=False,
                 g_rot=None,U_rot=None,
                 along_args={},across_args={}):
        self.g=g
        prepare_grid(g)
        
        self.U=U
        self.source_ds=source_ds
        self.g_rot=g_rot or self.g

        if self.g_rot != self.g:
            prepare_grid(self.g_rot)
            
        if U_rot is None:
            assert self.g==self.g_rot
            U_rot=utils.rot(np.pi/2,self.U)
        self.U_rot=U_rot

        self.bidir=bidir
        self.along_args=dict(bidir=bidir,max_t=20*3600,max_dist=500)
        self.along_args.update(along_args)
        self.across_args=dict(bidir=bidir,max_t=20*3600,max_dist=100)
        self.across_args.update(across_args)

        if alongs:
            self.alongs=alongs
        else:
            self.calc_alongs()
        if acrosses:    
            self.acrosses=acrosses
        else:
            self.calc_acrosses()
        
    def calc_alongs(self):
        alongs=[]
        for i in utils.progress(range(self.source_ds.dims['sample'])):
            x0=self.source_ds.x.values[i,:]
            alongs.append(self.trace_along(x0))
        self.alongs=alongs
    def calc_acrosses(self):
        acrosses=[]
        for i in utils.progress(range(self.source_ds.dims['sample'])):
            x0=self.source_ds.x.values[i,:]
            acrosses.append(self.trace_across(x0))
        self.acrosses=acrosses
    
    def trace_along(self,x):
        # bidir, max_t, max_dist should be instance attributes
        return steady_streamline_twoways(self.g,self.U,x,**self.along_args)
    def trace_across(self,x):
        # bidir, max_t, max_dist should be instance attributes
        return steady_streamline_twoways(self.g_rot,self.U_rot,x,**self.across_args)

    def stream_distance(self,x_target,sample,x_along=None,x_across=None,plot=False):
        if x_along is None:
            x_along=self.trace_along(x_target)
        if x_across is None:
            x_across=self.trace_across(x_target)

        s_along=self.alongs[sample]
        s_across=self.acrosses[sample]
        if plot: # plot the pieces
            import matplotlib.pyplot as plt
            plt.figure(1).clf()
            fig,ax=plt.subplots(1,1,num=1)

            self.g.plot_edges(ax=ax,color='k',lw=0.3)

            ax.plot( [x_target[0]],[x_target[1]],'ro')
            ax.plot( self.source_ds.x.values[sample,0],
                     self.source_ds.x.values[sample,1],
                     'go')

            ax.plot(x_along.x.values[:,0],x_along.x.values[:,1],'b-',label='x along')
            ax.plot(x_across.x.values[:,0],x_across.x.values[:,1],'k-',label='x across')
            ax.plot(s_along.x.values[:,0],s_along.x.values[:,1],'r-',label='s along')
            ax.plot(s_across.x.values[:,0],s_across.x.values[:,1],'-',color='orange',label='s across')

        x_along_g=geometry.LineString(x_along.x.values)
        x_across_g=geometry.LineString(x_across.x.values)
        s_along_g=geometry.LineString(s_along.x.values)
        s_across_g=geometry.LineString(s_across.x.values)

        x_pnt=geometry.Point(x_target)
        s_pnt=geometry.Point(self.source_ds.x.values[sample])

        xs1=x_along_g.intersection(s_across_g)
        xs2=s_along_g.intersection(x_across_g)

        if xs1.is_empty or xs2.is_empty:
            return np.array([np.nan,np.nan])

        if xs1.type!='Point':
            #raise Exception("xs1: expected point but got %s"%xs1.wkt)
            print("x_target=%s  sample=%s  expected point but got %s"%(str(x_target),sample,xs1.wkt))
            return np.array([np.nan,np.nan])
        if xs2.type!='Point':
            #raise Exception("xs2: expected point but got %s"%xs2.wkt)
            print("x_target=%s  sample=%s  expected point but got %s"%(str(x_target),sample,xs2.wkt))
            return np.array([np.nan,np.nan])

        dist_along1=x_along_g.project(x_pnt) - x_along_g.project(xs1)
        dist_along2=-(s_along_g.project(s_pnt) - s_along_g.project(xs2))
        dist_across1=s_across_g.project(s_pnt) - s_across_g.project(xs1)
        dist_across2=-(x_across_g.project(x_pnt) - x_across_g.project(xs2))

        if self.bidir:
            # those dist values are signed, and no guarantees that the signs
            # are the same. e.g. x_along and s_along may have the opposite
            # orientation when bidir is true.
            # use the sign at the target -- that way the signed distances are
            # consistent across multiple samples with the same target
            if dist_along1*dist_along2<0:
                dist_along2*=-1
            if dist_across1*dist_across2<0:
                dist_across2*=-1

        return np.array( [0.5*(dist_along1+dist_along2),
                          0.5*(dist_across1+dist_across2)])
    
