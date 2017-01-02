import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import field
from scipy import optimize as opt
import utils

import unstructured_grid
reload(unstructured_grid)
import exact_delaunay
reload(exact_delaunay)
import front
reload(front)
## 
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
        
# af=test_resample_neighbors()

# enumerate the strategies for a site:
# paver preemptively resamples the neighbors
# conceivable that one action might want to resample the neighbors
# in a slightly different way than another action.
# but the idea of having them spaced at the local scale when possible
# is general enough to do it preemptively.

# strategies:
#  try this as a separate class for each strategy, but they are all singletons


def test_actions():
    af=test_basic_setup()

    site=af.choose_site()
    af.resample_neighbors(site)
    actions=site.actions()
    metrics=[a.metric(site) for a in actions]
    best=np.argmin(metrics)
    edits=actions[best].execute(site)
    af.optimize_edits(edits)

# #

af=test_basic_setup()
check0=af.grid.checkpoint()

##

# Back-tracking
# The idea is that with sufficient roll-back, it can build a 
# decision tree and optimize between strategies.
# There are at least two ways in which this can be used:
#   optimizing: try multiple strategies, possibly even multiple
#     steps of each, evaluate the quality of the result, and then
#     go with the best one.
#   recovering: when a strategy fails, step back one or more steps
#     and try other options.


# This process should be managed in a decision tree, where each node
# of the tree represents a state, any metrics associated with that
# state, and the set of choices for what to do next.

# The decisions are (i) which site to pursue, and (ii) which strategy
# to apply at the site.

# There has to be a way to "commit" parts of the tree, moving the root
# of the tree down.

# Assuming that we deal with only one copy of the grid, then at most one
# node in the tree reflects the actual state of the grid.

# As long as we're careful about how checkpoints store data (i.e. no
# shared state), then chunks of the op_stack can be stored and used
# for quicker fast-forwarding.

# There is a distinction between decisions which have been tried (and
# so they can have a metric for how it turned out, and some operations for
# fast-forwarding the actions), versus decisions which have been posed
# but not tried.
# Maybe it's up to the parent node to hold the set of decisions, and as
# they are tried, then we populate the child nodes.


# the tree should be held by af.
# 

af=test_basic_setup()
af.log.setLevel(logging.INFO)

af.cdt.post_check=False
# af.plot_summary()


class DTNode(object):
    parent=None 
    af=None # AdvancingFront object
    cp=None # checkpoint
    ops_parent=None # chunk of op_stack to get from parent to here.
    children=None # filled in by subclass
    
    def __init__(self,af,parent=None):
        self.af=af
        self.parent=parent
        self.cp=af.grid.checkpoint()

class DTChooseSite(DTNode):
    def __init__(self,af,parent=None):
        super(DTChooseSite,self).__init__(af=af,parent=parent)
        self.sites=af.enumerate_sites()
        self.children=[None]*len(self.sites)
    def try_child(self,i):
        """ Assumes that af state is currently at this node,
        try the decision of the ith child, create the new DTNode
        for that, and shift af state to be that child.
        """
        site=self.sites[i]
        af.advance_at_site(site)
        self.children[i] = DTChooseSite(af=self.af,parent=self)
        af.current=self.children[i]
    def revert_to_parent(self):
        if self.parent is None:
            return False
        return self.parent.revert_to_here()
    def revert_to_here(self):
        self.af.grid.revert(self.cp)
        self.af.current=self
        

class DTChooseStrategy(DTNode):
    pass


self=af
af.root=DTChooseSite(self)
af.current=af.root
af.plot_summary() ; plt.pause(1.0)

af.current.try_child(0)
# Could then pull out some metrics on how well that decision went..
af.plot_summary() ; plt.pause(1.0)

af.current.revert_to_parent() # now back up
af.plot_summary() ; plt.pause(1.0)

af.current.try_child(1) # try a different child
af.plot_summary()


## 
# on sfei desktop, it's 41 cells/s.
# any chance numba can help out here? too much of a pain for now.

af=test_basic_setup()
af.log.setLevel(logging.INFO)

af.cdt.post_check=False

t_start=time.time()
# # 
af.loop()  
elapsed=time.time() - t_start
print "Elapsed: %.2fs, or %f cells/s"%(elapsed,af.grid.Ncells()/elapsed)

plt.figure(1).clf()
af.plot_summary(label_nodes=False)

##

# I think the best plan of attack is to roughly replicate the way paver
# worked, then extend with the graph search

#   Need to think about how these pieces are going to work together
#   And probably a good time to (a) start adding the rollback, graph
#   search side of things.
#   CDT is included now, and can trigger an alternate strategy when
#   edges intersect.  No non-local connections, though.

site3=af.choose_site()

if plt:
    plt.clf()
    af.grid.plot_edges()

    coll=af.grid.plot_halfedges(values=af.grid.edges['cells'])
    coll.set_lw(0)
    coll.set_cmap('winter')
    site3.plot()


###

# Does numba do anything now?
# takes some work -
# and it is having trouble with being called in the true context.
# passing it basically the same values, but manually on the command
# line appears to work just fine.
# could be that this doesn't call it in quite the same way?
# or that there is some nuance in the agruments, like not being
# contiguous, C-style, etc.?
