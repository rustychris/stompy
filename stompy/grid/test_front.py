import matplotlib.pyplot as plt
import numpy as np
import field
from scipy import optimize as opt
import unstructured_grid
reload(unstructured_grid)
import exact_delaunay
reload(exact_delaunay)
import front
reload(front)

#-# Curve -

def hex_curve():
    hexagon = np.array( [[0,1],
                         [1,0],
                         [3,0],
                         [4,1],
                         [3,2],
                         [1,2]] )
    return front.Curve(10*hexagon)

def test_curve_eval():
    crv=hex_curve()
    f=np.linspace(0,2*crv.total_distance(),25)
    crvX=crv(f)
    
    if plt:
        plt.clf()
        crv.plot()

        f=np.linspace(0,crv.total_distance(),25)
        crvX=crv(f)
        plt.plot(crvX[:,0],crvX[:,1],'ro')

def test_distance_away():
    crv=hex_curve()

    if plt:
        plt.clf()
        crv.plot()
        plt.axis('equal')
        
    rtol=0.05

    for f00,tgt,style in [ (0,10,'g-'),
                           (3.4,20,'r-'),
                           (3.4,-20,'r--') ]:
        for f0 in np.linspace(f00,crv.distances[-1],20):
            x0=crv(f0)
            f,x =crv.distance_away(f0,tgt,rtol=rtol)
            d=utils.dist(x-x0)
            assert np.abs( (d-np.abs(tgt))/tgt) <= rtol
            if plt:
                plt.plot( [x0[0],x[0]],
                          [x0[1],x[1]],style)

    try:
        f,x=crv.distance_away(0.0,50,rtol=0.05)
        raise Exception("That was supposed to fail!")
    except crv.CurveException:
        #print "Okay"
        pass

def test_is_forward():
    crv=hex_curve()
    assert crv.is_forward(5,6,50)
    assert crv.is_reverse(5,-5,10)



#-# 
def test_curve_upsample():
    boundary=hex_curve()
    scale=field.ConstantField(3)

    pnts,dists = boundary.upsample(scale,return_sources=True)

    if plt:
        plt.clf()
        line=boundary.plot()
        plt.setp(line,lw=0.5,color='0.5')

        #f=np.linspace(0,crv.total_distance(),25)
        #crvX=crv(f)
        plt.scatter(pnts[:,0],pnts[:,1],30,dists,lw=0)
    
def test_basic_setup():
    boundary=hex_curve()
    af=front.AdvancingFront()
    scale=field.ConstantField(3)

    af.add_curve(boundary)
    af.set_edge_scale(scale)

    # create boundary edges based on scale and curves:
    af.initialize_boundaries()

    if plt:
        plt.clf()
        g=af.grid
        g.plot_edges()
        g.plot_nodes()

        # 
        coll=g.plot_halfedges(values=g.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')
        

    return af


# Going to try more of a half-edge approach, rather than explicitly
# tracking the unpaved rings.
# hoping that a half-edge interface is sufficient for the paver, and
# could be supported by multiple representations internally.

# for starters, don't worry about caching/speed/etc.
# okay to start from scratch each time.

# the product here is a list of the N best internal angles for
# filling with a triangle(s)

def test_halfedge_traverse():
    af=test_basic_setup()
    J,Orient = np.nonzero( (af.grid.edges['cells'][:,:]==self.grid.UNMESHED) )

    # he=he0=HalfEdge(af.grid,J[0],Orient[0])
    he=he0=af.grid.halfedge(J[0],Orient[0])

    for i in range(af.grid.Nedges()*2):
        he=he.fwd()
        if he == he0:
            break
    else:
        assert False
    assert i==33 # pretty sure about that number...

    he=he0=af.grid.halfedge(J[0],Orient[0])

    for i in range(af.grid.Nedges()*2):
        he=he.rev()
        if he == he0:
            break
    else:
        assert False
    assert i==33 # pretty sure about that number...


    assert he.fwd().rev() == he
    assert he.rev().fwd() == he
    #-# 

def test_merge_edges():
    af=test_basic_setup()

    new_j=af.grid.merge_edges(node=0)
    
    he0=he=af.grid.halfedge(new_j,0)
    c0_left = af.grid.edges['cells'][he.j,he.orient]
    c0_right = af.grid.edges['cells'][he.j,1-he.orient]

    while True:
        he=he.fwd()
        c_left = af.grid.edges['cells'][he.j,he.orient]
        c_right = af.grid.edges['cells'][he.j,1-he.orient]
        assert c_left==c0_left
        assert c_right==c0_right
        
        if he==he0:
            break


    if plt:
        plt.clf()
        af.grid.plot_edges()

        coll=af.grid.plot_halfedges(values=af.grid.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')

# when resample nodes on a sliding boundary, want to calculate the available
# span, and if it's small, start distributing the nodes evenly.
# where small is defined by local_scale * max_span_factor


def test_resample():
    af=test_basic_setup()
    a=0
    b=af.grid.node_to_nodes(a)[0]
    he=af.grid.nodes_to_halfedge(a,b)
    anchor=he.node_rev()
    n=he.node_fwd()
    n2=he.rev().node_rev()
    af.resample(n=n,anchor=anchor,scale=25,direction=1)
    af.resample(n=n2,anchor=anchor,scale=25,direction=-1)

    if plt:
        plt.clf()
        af.grid.plot_edges()

        coll=af.grid.plot_halfedges(values=af.grid.edges['cells'])
        coll.set_lw(0)
        coll.set_cmap('winter')
    
    
#-#     



def test_resample_neighbors():
    af=test_basic_setup()
    
    if plt:
        plt.clf()
        af.grid.plot_nodes(color='r')
    
    site=af.choose_site()
            
    af.resample_neighbors(site)

    if plt:
        af.grid.plot_edges()

        af.grid.plot_nodes(color='g')
        # hmm - some stray long edges, where it should be collinear
        # ahh - somehow node 23 is 3.5e-15 above the others.
        # not sure why it happened, but for the moment not a show stopper.
        # in fact probably a good test of the robust predicates
        af.cdt.plot_edges(values=af.cdt.edges['constrained'],lw=3,alpha=0.5)

        plt.axis( [34.91, 42.182, 7.300, 12.97] )
    return af
        
af=test_resample_neighbors()

af.grid.plot_nodes(labeler=lambda n,nr: str(n))

## 
af.grid.modify_node(23,x=[af.grid.nodes['x'][23,0],20.0] )
plt.clf()
af.cdt.plot_edges(values=af.cdt.edges['constrained'],lw=3,alpha=0.5)

## 

# enumerate the strategies for a site:
# paver preemptively resamples the neighbors
# conceivable that one action might want to resample the neighbors
# in a slightly different way than another action.
# but the idea of having them spaced at the local scale when possible
# is general enough to do it preemptively.

# strategies:
#  try this as a separate class for each strategy, but they are all singletons


class Strategy(object):
    def metric(self,site,scale_factor):
        assert False
    def execute(self,site):
        """
        Apply this strategy to the given Site.
        Returns a dict with nodes,cells which were modified 
        """
        assert False

class WallStrategy(Strategy):
    """ 
    Add two edges and a new triangle to the forward side of the
    site.
    """
    def metric(self,site):
        # rough translation from paver
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Wall can be applied in a wide variety of situations
        return 1.0 * (theta > 60*pi/180.) 

    def execute(self,site):
        na,nb,nc= site.abc
        grid=site.grid
        b,c = grid.nodes['x'][ [nb,nc] ]
        bc=c-b
        new_x = b + utils.rot(np.pi/3,bc)
        nd=grid.add_node(x=new_x,fixed=site.af.FREE)
        new_c=grid.add_cell_and_edges( [nb,nc,nd] )

        return {'nodes': [nd],
                'cells': [new_c] }


class CutoffStrategy(Strategy):
    def metric(self,site):
        theta=site.internal_angle
        scale_factor = site.edge_length / site.local_length

        # Cutoff wants a small-ish internal angle
        # If the sites edges are long, scale_factor > 1
        # and we'd like to be making smaller edges, so ideal angle gets smaller
        # 
        ideal=60 + (1-scale_factor)*30
        return np.abs(theta - ideal*pi/180.)
    def execute(self,site):
        c=site.grid.add_cell_and_edges( site.abc )
        return {'cells':[c] }

# copied from paver verbatim
def one_point_cost(pnt,edges,target_length=5.0):
    # pnt is intended to complete a triangle with each
    # pair of points in edges, and should be to the left
    # of each edge
    penalty = 0
    
    max_angle = 85.0*pi/180.

    # all_edges[triangle_i,{ab,bc,ca},{x,y}]
    all_edges = zeros( (edges.shape[0], 3 ,2), float64 )
    
    # get the edges:
    all_edges[:,0,:] = edges[:,0] - pnt  # ab
    all_edges[:,1,:] = edges[:,1] - edges[:,0] # bc
    all_edges[:,2,:] = pnt - edges[:,1] # ca

    i = arange(3)
    im1 = (i-1)%3
    
    #--# cost based on angle:
    abs_angles = arctan2( all_edges[:,:,1], all_edges[:,:,0] )
    all_angles = (pi - (abs_angles[:,i] - abs_angles[:,im1]) % (2*pi)) % (2*pi)
        
    # a_angles = (pi - (ab_angles - ca_angles) % (2*pi)) % (2*pi)
    # b_angles = (pi - (bc_angles - ab_angles) % (2*pi)) % (2*pi)
    # c_angles = (pi - (ca_angles - bc_angles) % (2*pi)) % (2*pi)

    if 1:
        # 60 is what it's been for a while, but I think in one situation
        # this put too much weight on small angles.
        # tried considering just large angles, but that quickly blew up.
        # even just changing this to 50 still blows up.
        #  how about a small tweak - s/60/58/ ??
        worst_angle = abs(all_angles - 60*pi/180.).max() 
        alpha = worst_angle /(max_angle - 60*pi/180.0)

        # 10**alpha: edges got very short...
        # 5**alpha - 1: closer, but overall still short edges.
        # alpha**5: angles look kind of bad
        angle_penalty = 10*alpha**5

        # Seems like it doesn't try hard enough to get rid of almost bad angles.
        # in one case, there is a small angle of 33.86 degrees, and another angle
        # of 84.26 degrees. so the cost function only cares about the small angle
        # because it is slightly more deviant from 60deg, but we may be in a cell
        # where the only freedom we have is to change the larger angles.

        # so add this in:
        if 1:
            # extra exponential penalty for nearly bad triangles:
            # These values mean that 3 degrees before the triangle is invalid
            # the exponential cuts in and will add a factor of e by the time the
            # triangles is invalid.

            scale_rad = 3.0*pi/180. # radians - e-folding scale of the cost
            # max_angle - 2.0*scale_rad works..
            thresh = max_angle - 1.0*scale_rad # angle at which the exponential 'cuts in'
            big_angle_penalty = exp( (all_angles.max() - thresh) / scale_rad)
    else:
        alphas = (all_angles - 60*pi/180.) / (max_angle - 60*pi/180.)
        alphas = 10*alphas**4
        angle_penalty = alphas.sum()
    
    penalty += angle_penalty + big_angle_penalty

    #--# Length penalties:
    ab_lens = (all_edges[:,0,:]**2).sum(axis=1)
    ca_lens = (all_edges[:,2,:]**2).sum(axis=1)

    if 1:  # the usual..
        min_len = min(ab_lens.min(),ca_lens.min())
        max_len = max(ab_lens.min(),ca_lens.min())

        undershoot = target_length**2 / min_len
        overshoot  = max_len / target_length**2

        length_penalty = 0

        length_factor = 2
        length_penalty += length_factor*(max(undershoot,1) - 1)
        length_penalty += length_factor*(max(overshoot,1) - 1)
    elif 1:
        # Try an exponential
        rel_len_ab = ab_lens / target_length**2
        rel_len_ca = ca_lens / target_length**2

        # So we want to severely penalize things that are more than double
        # the target length or less than half the target length.
        # just a wild guess here, that maybe the threshold needs to be larger.
        # well, how about the penalty kicks in at 3x
        thresh = 9.0 # 2.5*2.5
        length_penalty = exp( rel_len_ab - thresh ).sum() + exp( 1.0/rel_len_ab - thresh).sum()
        length_penalty += exp( rel_len_ca - thresh ).sum() + exp( 1.0/rel_len_ca - thresh).sum()

    else:
        rel_errs_ab = (ab_lens - target_length**2) / target_length**2
        rel_errs_ca = (ca_lens - target_length**2) / target_length**2

        length_penalty = ( (rel_errs_ab**2).sum() + (rel_errs_ca**2).sum() )
        
    penalty += length_penalty

    return penalty
    
Wall=WallStrategy()
Cutoff=CutoffStrategy()

def cost_function(self,n):
    local_length = self.scale( self.grid.nodes['x'][n] )
    my_cells = self.grid.node_to_cells(n)

    if len(my_cells) == 0:
        return None

    cell_nodes = [self.grid.cell_to_nodes(c)
                  for c in my_cells ]
    
    # for the moment, can only deal with triangles
    cell_nodes=np.array(cell_nodes)
    
    # pack our neighbors from the cell list into an edge
    # list that respects the CCW condition that pnt must be on the
    # left of each segment
    for j in range(len(cell_nodes)):
        if cell_nodes[j,0] == n:
            cell_nodes[j,:2] = cell_nodes[j,1:]
        elif cell_nodes[j,1] == n:
            cell_nodes[j,1] = cell_nodes[j,0]
            cell_nodes[j,0] = cell_nodes[j,2]

    edges = cell_nodes[:,:2]
    edge_points = self.grid.nodes['x'][edges]

    def cost(x):
        return one_point_cost(x,edge_points,target_length=local_length)
    
    return cost


def optimize(self,edits):
    nodes = edits.get('nodes',[])
    for c in edits.get('cells',[]):
        for n in self.grid.cell_to_nodes(c):
            if n not in nodes:
                nodes.append(n)
    for n in nodes:
        relax_node(self,n)
    
def relax_node(self,n):
    if self.grid.nodes['fixed'][n] == self.FREE:
        relax_free_node(self,n)
    else:
        print "Not ready for relaxing non-free nodes"
        
def relax_free_node(self,n0):
    cost=cost_function(self,n)
    if cost is None:
        return 
    x0=self.grid.nodes['x'][n]
    local_length=self.scale( x0 )
    new_x = opt.fmin(cost,
                     x0,
                     xtol=local_length*1e-4,
                     disp=0)
    dx=utils.dist( new_x - x0 )
    print dx
    self.grid.modify_node(n,x=new_x)


af=test_basic_setup()

site=af.choose_site()
af.resample_neighbors(site)
edits = Wall.execute(site)
optimize(af,edits)

## 
site2=af.choose_site()
af.resample_neighbors(site2)
edits=Cutoff.execute(site2)
optimize(af,edits)

##

# HERE:
#   So it can do a wall, a cutoff, and optimize them
#   correctly.
#   Need to think about how these pieces are going to work together
#   And probably a good time to (a) move most of the code above into front.py
#   and (b) start adding the rollback, graph search side of things.
#   CDT is included now, though without any real integration - no checks yet
#   for colliding edges or collinear nodes.


site3=af.choose_site()

if plt:
    plt.clf()
    af.grid.plot_edges()

    coll=af.grid.plot_halfedges(values=af.grid.edges['cells'])
    coll.set_lw(0)
    coll.set_cmap('winter')
    site3.plot()
