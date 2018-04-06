from stompy.grid import orthogonalize, front, unstructured_grid, exact_delaunay,shadow_cdt

from stompy.model.delft import dfm_grid
from stompy import utils

import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import collections, patches
import six

## 
six.moves.reload_module(unstructured_grid)
six.moves.reload_module(exact_delaunay)
six.moves.reload_module(dfm_grid)

## 

g=dfm_grid.DFMGrid("data/lsb_combined_v14_net.nc")
# Trim that down to speed up things
clip=(577006.59313042194, 579887.89496937161, 4143066.3785693897, 4145213.4131655102)

node_to_del=np.nonzero(~utils.within_2d(g.nodes['x'],clip))[0]

for n in node_to_del:
    g.delete_node_cascade(n)

g.renumber() 

##

zoom=(578260, 579037, 4143970., 4144573)
point_in_poly=(578895, 4144375)
point_on_edge=(579003., 4144517.)
j_init=g.select_edges_nearest(point_on_edge)

dim_par=10.0 # 10m wide for edges "parallel" to j
dim_perp=20.0 # 20m long for edges "perpendicular" to j.

g.edge_to_cells()


## 
#  plt.figure(1).clf()
#  
#  fig,ax=plt.subplots(num=1)
#  
#  g.plot_edges(ax=ax,clip=zoom)
#  
#  ax.text(point_in_poly[0],point_in_poly[1],'START')
#  
#  g.plot_edges(mask=[j_init],color='r',lw=2)
#  
#  if 0:
#      coll=g.plot_halfedges(values=af.grid.edges['cells'],clip=zoom,
#                            ax=ax)
#      coll.set_clim([-3,0])
#      coll.set_lw(0)
#  
#  # The idea is to start with an edge adjacent to an unfilled
#  # area (and with an existing cell on the other side).
#  
#  # This is the starting point for paving
#  # paving will only use rectangles.
#  
#  
#  ## 
#  af.plot_summary(label_nodes=False,clip=zoom)
#  ax.axis(zoom)

## 

six.moves.reload_module(shadow_cdt)
six.moves.reload_module(front)

af=front.AdvancingQuads(grid=g.copy(),scale=dim_par,perp_scale=dim_perp)
af.grid.edges['para']=0 # avoid issues during dev

af.add_existing_curve_surrounding(point_in_poly)
af.orient_quad_edge(j_init,af.PARA)

# zoom=ax.axis()
# zoom=(578973.9386167785, 579021.8038241524, 4144502.1423333557, 4144537.809633044)
zoom=(578831.3252777942, 579048.5730854167, 4144375.5031878487, 4144537.3878444955)

af.loop(7)

##

# At this point it has gone around a bend and started with the perpetual
# trapezoids.

# should compare quad's resample_neighbors with triangle's, which has
# much more recent development.


# Fixed the HINT->SLIDE update problem

# Problem 1: going around a bend it starts making trapezoids, 
#   never really recovers.  A more nuanced cost function with angles
#   would help here.
#  HERE: try pulling in some of the new cost function for triangles.

# Enhancement: there is an obvious place to put a triangle around a bend.
#   will get there eventually.

# other stopping strategies (triangle?) may become necessary..


# thoughts on how to optimize out the trapezoids:
#  - more weight on the lengths could help, but it's nuanced since 
#    some of the edges can only be made longer by creating a trapezoid.
#  - 


# # The existing quad fn:
# def one_point_quad_cost(x,edge_scales,quads,para_scale,perp_scale):
#     # orthogonality cost:
#     ortho_cost=0.0
# 
#     base_scale=np.sqrt( para_scale**2 + perp_scale**2 )
#     
#     quads[:,0,:] = x # update the first point of each quad
# 
#     for quad in quads:
#         cc=utils.poly_circumcenter(quad)
#         dists=utils.mag(quad-cc)
#         err=np.std(dists) / base_scale
# 
#         ortho_cost += 10*err # ad hoc hoc hoc
# 
#     # length cost:
#     scale_cost=0.0
# 
#     dists=utils.mag(x - edge_scales[:,:2])
#     errs=(dists - edge_scales[:,2]) / edge_scales[:,2]
#     scale_cost = (2*errs**2).sum()
# 
#     return ortho_cost+scale_cost

# Trying something more like the triangle CC cost
def one_point_quad_cost(x,edge_scales,quads,para_scale,perp_scale):
    # orthogonality cost:
    ortho_cost=0.0

    base_scale=np.sqrt( para_scale**2 + perp_scale**2 )
    
    quads[:,0,:] = x # update the first point of each quad

    for quad in quads:
        cc=utils.poly_circumcenter(quad)
        dists=utils.mag(quad-cc)
        err=np.std(dists) / base_scale

        ortho_cost += 10*err # ad hoc hoc hoc

    # length cost:
    scale_cost=0.0

    dists=utils.mag(x - edge_scales[:,:2])
    errs=(dists - edge_scales[:,2]) / edge_scales[:,2]
    scale_cost = (2*errs**2).sum()

    return ortho_cost+scale_cost


ccs=[]
radii=[]

def one_point_quad_cost(x0,edge_scales,quads,para_scale,perp_scale):
    """
    edge_scales: not used yet.
    quads: [N,4,2] coordinates for the quads.  the node being moved appears
      first for each.
    """
    quads[:,0,:]=x0
    cc_cost=0

    del ccs[:]
    del radii[:]
    
    for quad_i in range(quads.shape[0]):
        # This leads to really bad orthogonality
        #tri_cc=utils.poly_circumcenter(quads[quad_i,:,:])
        # much better
        tri_cc=utils.poly_circumcenter(quads[quad_i,:3,:])
        ccs.append(tri_cc)
        
        this_quad_cc_cost=0
        radius=0
        
        for side in range(4):
            sidep1=(side+1)%4

            p1=quads[quad_i,side]
            p2=quads[quad_i,sidep1]
            
            deltaAB = tri_cc - p1

            radius += utils.dist(deltaAB)

            AB=p2-p1
            magAB=math.sqrt( AB[0]*AB[0] + AB[1]*AB[1])
            vecAB=AB/magAB
            leftAB=vecAB[0]*deltaAB[1] - vecAB[1]*deltaAB[0] 

            cc_fac=-4. # not bad
            # clip to 100, to avoid overflow in math.exp
            this_edge_cc_cost = math.exp(min(100,cc_fac*leftAB/magAB))
            this_quad_cc_cost += this_edge_cc_cost

        radii.append(radius / 4.)
        
        cc_cost+=this_quad_cc_cost

    dists=utils.dist( x0 - edge_scales[:,:2])
    tgts=edge_scales[:,2]
    scale_cost=np.sum( (dists-tgts)**2/tgts**2 )

    # With even weighting between these, some edges are pushed long rather than
    # having nice angles.
    # 3 is a shot in the dark.
    # 50 is more effective at avoiding a non-orthogonal cell
    # 50 was good for triangles.
    # trying more on the scale here
    return 5*cc_cost+scale_cost


def cost_function(self,n):
    local_para = self.para_scale
    local_perp = self.perp_scale

    my_cells = self.grid.node_to_cells(n)

    if len(my_cells) == 0:
        return None

    cell_nodes=self.grid.cells['nodes'][my_cells]
    # except that for the moment I'm only going to worry about the
    # quads:
    sel_quad=(cell_nodes[:,3]>=0)
    if self.grid.max_sides>4:
        sel_quad &=(cell_nodes[:,4]<0)
    cell_nodes=cell_nodes[sel_quad]

    # For each quad, rotate our node to be at the front of the list:
    quad_nodes=[np.roll(quad,-list(quad).index(n))
                for quad in cell_nodes[:,:4]]
    quad_nodes=np.array(quad_nodes)
    quads=self.grid.nodes['x'][quad_nodes]

    if 0:
        # for the moment, don't worry about reestablishing scale, just
        # focus on orthogonality
        edge_scales=np.zeros( [0,3], 'f8')
    else:
        edge_scales=[]
        for j in self.grid.node_to_edges(n):
            a,b=self.grid.edges['nodes'][j]
            if a==n:
                nbr=b
            else:
                nbr=a
            if self.grid.nodes['fixed'][nbr]==self.HINT:
                # ignore
                continue
            if self.grid.edges['para'][j]:
                tgt=local_para
            else:
                tgt=local_perp
            edge_scales.append( [self.grid.nodes['x'][nbr,0],
                                 self.grid.nodes['x'][nbr,1],
                                 tgt] )
        edge_scales=np.array(edge_scales)

    def cost(x,edge_scales=edge_scales,quads=quads,
             local_para=local_para,local_perp=local_perp):
        return one_point_quad_cost(x,edge_scales,quads,local_para,local_perp)

    return cost

##

# This a reasonable list of nodes to be jiggled:
node=[159,160,158,157,156,155]

for it in range(4):
    for n in [160,159]: # node:
        # the body of relax_slide_node()
        self=af
        cost_free=cost_function(self,n)
        x0=self.grid.nodes['x'][n]
        ax.plot( [x0[0]], [x0[1]],'ro')
        plt.pause(0.01)
        f0=self.grid.nodes['ring_f'][n]
        ring=self.grid.nodes['oring'][n]-1

        assert np.isfinite(f0)
        assert ring>=0

        local_length=self.scale( x0 )

        slide_limits=self.find_slide_limits(n,3*local_length)

        # used to just be f, but I think it's more appropriate to
        # be f[0]
        def cost_slide(f):
            # lazy bounded optimization
            f=f[0]
            fclip=np.clip(f,*slide_limits)
            err=(f-fclip)**2
            return err+cost_free( self.curves[ring](fclip) )

        new_f = opt.fmin(cost_slide,
                         [f0],
                         xtol=local_length*1e-4,
                         disp=0)

        new_x=self.curves[ring](new_f[0])

        ax.lines=[]
        ax.patches=[]
        ax.plot( [new_x[0]],[new_x[1]],'go')
        ccs_a=np.array(ccs)
        ax.plot( ccs_a[:,0], ccs_a[:,1],'ro')

        for cc,r in zip(ccs,radii):
            patch=patches.Circle(cc,r,facecolor='none',edgecolor='m')
            ax.add_patch(patch)

        plt.pause(0.01)

        if 1:
            # execute that move:
            if not self.curves[ring].is_forward(slide_limits[0],
                                                new_f,
                                                slide_limits[1]):
                print "Bad slide"
            else:
                cp=self.grid.checkpoint()
                try:
                    if new_f[0]!=f0:
                        self.slide_node(n,new_f[0]-f0)
                except self.cdt.IntersectingConstraints as exc:
                    self.grid.revert(cp)
                    self.log.info("Relaxation caused intersection, reverting")


            if 1:
                zoom=(578870.9475716517, 578978.673235342, 4144424.1443142556, 4144504.4173088106)

                plt.figure(1).clf()
                fig,ax=plt.subplots(num=1)

                af.grid.plot_edges(color='k',lw=0.5)
                af.grid.plot_nodes(labeler='id',clip=zoom)
                ax.axis(zoom)
                plt.pause(0.1)

## 

zoom=(578870.9475716517, 578978.673235342, 4144424.1443142556, 4144504.4173088106)

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

af.grid.plot_edges(color='k',lw=0.5)
af.grid.plot_nodes(labeler='id',clip=zoom)
ax.axis(zoom)


# Findings:
# A. Using poly_circumcenter(), and just optimizing point-line distances to the circumcenter
#    is insufficient.  This does nothing for orthogonality, and
#    in fact makes orthogonality much worse as a way to trick the
#    apparent the CC into the middle of the cell.
# B. Using circumcenter, excluding the node to be moved, is not bad.
