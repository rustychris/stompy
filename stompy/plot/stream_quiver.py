"""
A quiver plot for flow defined an unstructured grid.

Each arrow follows the local streamlines, and the spacing
between arrows is maintained to avoid overlap.

It's slow.
"""
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
from shapely import geometry

from .. import utils
from ..grid import exact_delaunay
from ..model import stream_tracer

class StreamlineQuiver(object):
    max_short_traces=100 # abort loop when this many traces have come up short.
    short_traces=0 # count of short traces so far.
    streamline_count=1000
    min_clearance=6.0 # streamlines are truncated when this close to each other
    seed_clearance = 12.0 # streamlines are started when the circumradius >= this
    cmap='jet'
    clim=[0,1.5]
    max_t=60.0
    max_dist=60.
    size=1.0

    # don't start traces outside this xxyy bounding box.
    clip=None

    # If True, trace long streamlines, run them out in a deterministic direction,
    # and try to keep just the part that abuts an obstactle
    pack=False
    
    def __init__(self,g,U,**kw):
        utils.set_keywords(self,kw)
        self.g=g
        self.U=U
        self.island_points=[] # log weird island points for debugging
        self.Umag=utils.mag(U)
        self.boundary=g.boundary_polygon()
        self.init_tri()
        self.calculate_streamlines()

    NOT_STREAM=0
    STREAM=1
    TRUNC=2
    def init_tri(self):
        self.tri=tri=exact_delaunay.Triangulation()
        tri.add_cell_field('outside',np.zeros(0,np.bool8))
        tri.add_node_field('tip',np.zeros(0,np.bool8))
        # NOT_STREAM=0: not part of a streamline
        # STREAM=1: streamline without truncation
        # TRUNC=2: streamline that got truncated.
        tri.add_node_field('stream_code',np.zeros(0,np.int32))
        
        tri.cell_defaults['_center']=np.nan
        tri.cell_defaults['_area']=np.nan
        tri.cell_defaults['outside']=False
        tri.node_defaults['tip']=False

        bound_cycle=self.g.boundary_cycle()
        tri.bulk_init( self.g.nodes['x'][bound_cycle] )

        for a,b in utils.circular_pairs(np.arange(len(bound_cycle))):
            tri.add_constraint(a,b)

        tri.cells['_area']=np.nan
        centers=tri.cells_centroid()
        for c in tri.valid_cell_iter():
            if not self.boundary.intersects(geometry.Point(centers[c])):
                tri.cells['outside'][c]=True
            else:
                tri.cells['outside'][c]=False
        return tri

    def calculate_streamlines(self,count=None):
        if count is None:
            count=self.streamline_count
            
        for i in range(self.streamline_count):
            self.process_one_streamline()
            if self.short_traces>self.max_short_traces:
                break

    def process_one_streamline(self):
        xy=self.pick_starting_point()
        if xy is None:
            print("Stopping on seed clearance")
            # should refactor stopping criteria
            self.short_traces=self.max_short_traces + 1
            return

        if self.pack:
            xy=self.pack_starting_point(xy)
            trace=stream_tracer.steady_streamline_oneway(self.g,self.U,xy,
                                                         max_t=self.max_t,max_dist=self.max_dist)
        else:
            # max_t=20.0 was decent.  
            trace=stream_tracer.steady_streamline_twoways(self.g,self.U,xy,
                                                          max_t=self.max_t,max_dist=self.max_dist)
        n_nodes=self.add_trace_to_tri(trace)
        if n_nodes==1:
            print(".",end="")
            self.short_traces+=1

    def pack_starting_point(self,xy):
        """
        Pack the given starting point as far upstream as possible, based 
        on the seed_clearance.
        """
        trace=stream_tracer.steady_streamline_oneway(self.g,-self.U,xy,
                                                     max_t=100*self.max_t,
                                                     max_dist=100*self.max_dist)

        new_xy_t_idx=0
        hint={}
        for t_idx in range(len(trace.time)):
            xy_i=trace.x.values[t_idx]
            # quickly test clearance of this point:
            rad,hint=self.tri.point_clearance(xy_i,hint=hint)
            if rad < self.seed_clearance:
                break
            else:
                new_xy_t_idx=t_idx

        new_xy=trace.x.values[new_xy_t_idx]
        return new_xy

    def add_trace_to_tri(self,trace,min_clearance=None):
        """
        trace: a trace Dataset as return from stream_tracer
        """
        if min_clearance is None:
            min_clearance=self.min_clearance
        
        if 'root' in trace:
            trace_root=trace.root.item()
        else:
            trace_root=0

        xys=trace.x.values
        if self.pack:
            stops=[None,trace.stop_condition.item()]
        else:
            stops=trace.stop_conditions.values
            
        if stops[0]=='leave_domain':
            xys=xys[1:]
            trace_root=max(0,trace_root-1)
        if stops[-1]=='leave_domain':
            xys=xys[:-1]

        # Keep this in the order of the linestring
        recent=[]

        nroot=self.tri.add_node(x=xys[trace_root])
        recent.append(nroot)

        clearance=self.neighbor_clearance(nroot,recent)
        if clearance<min_clearance:
            print(".",end="")
            self.tri.nodes['stream_code'][recent]=self.TRUNC
            return len(recent)

        stream_code=self.STREAM

        if self.pack:
            # Starting point is fine, only need to check as we go downstream
            incrs=[1]
        else:
            # Check both ways.
            incrs=[1,-1]
        
        for incr in incrs:
            xy_leg=xys[trace_root+incr::incr]
            na=nroot
            for xy in xy_leg:
                if np.all(xy==self.tri.nodes['x'][na]):
                    # root is repeated. could happen in other cases, too.
                    continue
                try:
                    nb=self.tri.add_node(x=xy)
                except self.tri.DuplicateNode:
                    # Essentially a degenerate case of neighbor_clearance
                    # going to 0.
                    import pdb
                    pdb.set_trace()
            
                    print("x",end="")
                    stream_code=self.TRUNC
                    break
                recent.append(nb)
                clearance=self.neighbor_clearance(nb,recent)
                if clearance<min_clearance:
                    # if it's too close, don't add an edge
                    print("-",end="")
                    stream_code=self.TRUNC
                    break
                try:
                    self.tri.add_constraint(na,nb)
                except self.tri.IntersectingConstraints:
                    print('!') # shouldn't happen..
                    break
                na=nb
            if incr>0:
                self.tri.nodes['tip'][na]=True
            recent=recent[::-1] # Second iteration goes reverse to the first.
        self.tri.nodes['stream_code'][recent]=stream_code
        return len(recent)

    def pick_starting_point(self):
        """
        Pick a starting point based on the triangle with the largest circumradius.
        Complicated by the presence of constrained edges at the boundary of the
        grid. Using constrained centers helps to some degree.
        """
        while 1:
            centers=self.tri.constrained_centers() 

            radii=utils.dist(centers - self.tri.nodes['x'][self.tri.cells['nodes'][:,0]])
            radii[ self.tri.cells['outside'] | self.tri.cells['deleted']] = 0.0
            if self.clip is not None:
                clipped=( (centers[:,0]<self.clip[0])
                          | (centers[:,0]>self.clip[1])
                          | (centers[:,1]<self.clip[2])
                          | (centers[:,1]>self.clip[3]) )
                radii[clipped]=0.0
            radii[ ~np.isfinite(radii)]=0.0
            best=np.argmax(radii)

            if radii[best]<self.seed_clearance:
                return None
            
            xy=centers[best]
            print("*",end="") # xy)

            if not self.boundary.intersects( geometry.Point(xy) ):
                # Either constrained_centers() did a bad job and the point isn't
                # in the cell,
                cpoly=self.tri.cell_polygon(best)
                if not cpoly.intersects( geometry.Point(xy) ):
                    print("Constrained center %s fell outside cell %d.  Lie and mark cell 'outside'"%(str(xy),best))
                    self.tri.cells['outside'][best]=True
                    continue
                else:
                    # Assume this is an island.
                    print("Island")
                    self.island_points.append( xy )
                    self.tri.cells['outside'][best]=True
                    continue
            break
        return xy

    # would have been keeping track of the recent nodes as they were 
    # created.
    def neighbor_clearance(self,n,recent=[]):
        # This is a bit tricky, as we could get into a spiral, and have only 
        # neighbors that are on our own streamline.
        nbrs=self.tri.node_to_nodes(n)
        nbr_dists=utils.dist( self.tri.nodes['x'][n] - self.tri.nodes['x'][nbrs])

        # Only worry about recent nodes when the distance is some factor
        # smaller than the along-path distance.
        min_dist=np.inf

        # node indices from n and moving away
        nodes=np.r_[ n, recent[::-1]]
        recent_path=self.tri.nodes['x'][nodes]
        recent_dist=utils.dist_along(recent_path)
        path_dists={ rn:rd for rn,rd in zip(nodes,recent_dist)}
        for nbr,dist in zip(self.tri.node_to_nodes(n),nbr_dists):
            if nbr in recent:
                # What is the along path distance?
                path_dist=path_dists[nbr]
                # if the straightline distance is not much smaller
                # than the along-path distance, then we're probably
                # just seeing ourselves, and not grounds for clearance
                # issues
                if dist > 0.5 *path_dist:
                    continue
                # otherwise, path may have looped back, and we should bail.
            min_dist=min(min_dist,dist)

        return min_dist
    
    def fig_constrained(self,num=None):
        fig,ax=plt.subplots(num=num)
        sel=~(self.tri.cells['outside'] | self.tri.cells['deleted'] )
        self.tri.plot_edges(color='k',lw=0.3,mask=self.tri.edges['constrained'])
        # tri.plot_edges(color='0.7',lw=0.3,mask=~tri.edges['constrained'])
        #ax.plot(centers[sel,0],centers[sel,1],'r.')
        # tri.plot_nodes(mask=tri.nodes['tip'])
        ax.axis('off')
        ax.set_position([0,0,1,1])
        return fig,ax
    def segments_and_speeds(self,include_truncated=True):
        """
        Extract segments starting from nodes marked as tip.

        include_truncate: include segments that are not as long as their
        trace on account of truncation due to spacing.
        """
        strings=self.tri.extract_linear_strings(edge_select=self.tri.edges['constrained'])

        # Order them ending with the tip, and only strings that include
        # a tip (gets rid of boundary)
        segs=[]
        for string in strings:
            node_tips=self.tri.nodes['tip'][string]
            if np.all( ~node_tips): continue
            if not include_truncated and np.any(self.tri.nodes['stream_code'][string]==self.TRUNC):
                continue
            xy=self.tri.nodes['x'][string]
            if node_tips[0]:
                xy=xy[::-1]
            elif node_tips[-1]:
                pass
            else:
                print("Weird - there's a tip but it's not at the tip")
            segs.append(xy)

        tip_cells=[self.g.select_cells_nearest(seg[-1],inside=True) for seg in segs]
        speeds=self.Umag[tip_cells]
        return segs,speeds

    sym=2.0 * np.array( [ [1.5,    0],
                    [-0.5, 1],
                    [0,    0],
                    [-0.5, -1]])
    diam=np.array([ [0.5,0],
                    [0, 0.5],
                    [-0.5,0],
                   [0,-0.5]])

    def manual_arrows(self,x,y,u,v,speeds,size=1.0): 
        # manual arrow heads.
        angles=np.arctan2(v,u)

        polys=[ utils.rot(angle,self.sym) for angle in angles]
        polys=np.array(polys)
        polys[speeds<0.1,:]=self.diam
        polys *= size
        polys[...,0] += x[:,None]
        polys[...,1] += y[:,None]
        pcoll=collections.PolyCollection(polys)
        return pcoll

    def plot_quiver(self,ax=None,lw=0.8,include_truncated=True):
        """
        Add the quiver plot to the given axes.
        The quiver is split into two collections: a line (shaft) and
        a polygon (arrow head).

        This method should evolve to have a calling convention closer to
        the MPL quiver function in terms of color, clim, cmap.  For now
        it uses self.clim, self.cmap, and defines color using the speed,
        which in turn is defined at the downstream tip of the arrow.
        """
        if ax is None:
            ax=plt.gca()

        segs,speeds=self.segments_and_speeds(include_truncated=include_truncated)

        result={}
        result['lcoll']=collections.LineCollection(segs,
                                                   array=speeds,
                                                   clim=self.clim,cmap=self.cmap,
                                                   lw=lw)
        ax.add_collection(result['lcoll'])

        # Need end points, end velocity for each segments
        xyuvs=[]
        for seg,speed in zip(segs,speeds):
            seg=seg[np.isfinite(seg[:,0])]
            uv=speed*utils.to_unit(seg[-1]-seg[-2])
            xyuvs.append( [seg[-1,0],seg[-1,1],uv[0],uv[1]])
        xyuvs=np.array(xyuvs)

        pcoll=self.manual_arrows(xyuvs[:,0],xyuvs[:,1],
                                 xyuvs[:,2],xyuvs[:,3],
                                 speeds,size=self.size)
        pcoll.set_array(speeds)
        pcoll.set_cmap(self.cmap)
        pcoll.set_clim(self.clim)
        pcoll.set_lw(0)

        ax.add_collection(pcoll)
        result['pcoll']=pcoll
        return result
