import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import field
import pdb
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

    if 0 and plt:
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

##

af2=test_basic_setup()
af2.log.setLevel(logging.INFO)

af2.cdt.post_check=False
af2.loop()
plt.figure(2).clf()
fig,ax=plt.subplots(num=2)
af2.plot_summary(ax=ax)
ax.set_title('loop()')

## 

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
af=test_basic_setup()
af.log.setLevel(logging.INFO)

af.cdt.post_check=False

class DTNode(object):
    parent=None 
    af=None # AdvancingFront object
    cp=None # checkpoint
    ops_parent=None # chunk of op_stack to get from parent to here.
    options=None # node-specific list of data for child options
    
    children=None # filled in by subclass [DTNode, ... ]
    child_prior=None # est. cost for child
    child_post =None # actual cost for child
    
    def __init__(self,af,parent=None):
        self.af=af
        self.parent=parent
        # in cases where init of the node makes some changes,
        # this should be updated
        self.cp=af.grid.checkpoint() 
        self.active_child=None
    def set_options(self,options,priors):
        self.options=options
        self.child_prior=priors
        
        N=len(options)
        self.children=[None] * N
        self.child_post =[None]*N
        self.child_order=np.argsort(self.child_prior) 
        
    def revert_to_parent(self):
        if self.parent is None:
            return False
        return self.parent.revert_to_here()
    def revert_to_here(self):
        self.af.grid.revert(self.cp)
        self.af.current=self
        self.active_child=None

    def try_child(self,i):
        assert False # implemented by subclass
        
    def best_child(self,count=0,cb=None):
        """
        Try all, (or up to count) children, 
        use the best one based on post scores.
        If no children succeeded, return False, otherwise True
        """
        if count:
            count=min(count,len(self.options))
        else:
            count=len(self.options)

        best=None
        for i in range(count):
            print("best_child: trying %d / %d"%(i,count))
            
            if self.try_child(i):
                if cb: cb()
                if best is None:
                    best=i
                elif self.child_post[i] < self.child_post[best]:
                    best=i
                if i<count-1: 
                    self.revert_to_here()
            else:
                print("best_child: option %d did not succeed"%i)
        if best is None:
            # no children worked out -
            print("best_child: no children worked")
            return False
        
        # wait to see if the best was the last, in which case
        # can save an undo/redo
        if best!=count-1:
            self.revert_to_here()
            self.try_child(best)
        return True

class DTChooseSite(DTNode):
    def __init__(self,af,parent=None):
        super(DTChooseSite,self).__init__(af=af,parent=parent)
        sites=af.enumerate_sites()
        
        priors=[ site.metric()
                 for site in sites ]
        
        self.set_options(sites,priors)
        
    def try_child(self,i):
        """ 
        Assumes that af state is currently at this node,
        try the decision of the ith child, create the new DTNode
        for that, and shift af state to be that child.

        Returns true if successful.  On failure (topological violation?)
        return false, and state should be unchanged.
        """
        assert self.af.current==self
        
        site=self.options[self.child_order[i]]

        self.children[i] = DTChooseStrategy(af=self.af,parent=self,site=site)
        # nothing to update for posterior
        self.child_post[i] = self.child_prior[i]
        
        af.current=self.children[i]
        self.active_child=i
        return True
    
    def best_child(self,count=0,cb=None):
        """
        For choosing a site, prior is same as posterior
        """
        if count:
            count=min(count,len(self.options))
        else:
            count=len(self.options)

        best=None
        for i in range(count):
            print("best_child: trying %d / %d"%(i,count))
            if self.try_child(i):
                if cb: cb()
                # no need to go further
                return True
        return False
        
class DTChooseStrategy(DTNode):
    def __init__(self,af,parent,site):
        super(DTChooseStrategy,self).__init__(af=af,parent=parent)
        self.site=site

        self.af.resample_neighbors(site)
        self.cp=af.grid.checkpoint() 

        actions=site.actions()
        priors=[a.metric(site)
                for a in actions]
        self.set_options(actions,priors)

    def try_child(self,i):
        try:
            edits=self.options[self.child_order[i]].execute(self.site)
            self.af.optimize_edits(edits)
            # could commit?
        except self.af.cdt.IntersectingConstraints as exc:
            self.af.log.error("Intersecting constraints - rolling back")
            self.af.grid.revert(self.cp)
            return False
        except self.af.StrategyFailed as exc:
            self.af.log.error("Strategy failed - rolling back")
            self.af.grid.revert(self.cp)
            return False
        
        self.children[i] = DTChooseSite(af=self.af,parent=self)
        self.active_edits=edits # not sure this is the right place to store this
        self.af.current=self.children[i]

        nodes=[]
        for c in edits['cells']:
            nodes += list(self.af.grid.cell_to_nodes(c))
        for n in edits.get('nodes',[]):
            nodes.append(n)
        nodes=list(set(nodes))
        cost = np.max( [self.af.eval_cost(n)
                        for n in nodes] )
        self.child_post[i]=cost
        return True

if 0: # manual testing
    af.root=DTChooseSite(af)
    af.current=af.root
    # af.plot_summary() ; plt.pause(1.0)

    def cb():
        print("tried...")
        af.plot_summary()
        fig.canvas.draw()
    af.current.best_child()
    af.current.best_child(cb=cb)
    af.current.best_child()
    # This is leaving things in a weird place
    af.current.best_child(cb=cb)

af.plot_summary()    


## 
# Single step lookahead:

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)

af=test_basic_setup()
af.log.setLevel(logging.INFO)
af.cdt.post_check=False
af.current=af.root=DTChooseSite(af)

# the results are not that great, seems that Wall
# wins out as much as possible, since it is least constrained
# then it ends up using too many joins, where it should have
# just used a bisect.
# also seems like the optimization is favoring angles over 
# scale too much, such that the results stray too far from
# a constant scale, and then it can't recover

# It now runs to completion, but the quality is low.

while 1:
    if not af.current.children:
        break # we're done?

    def cb():
        af.plot_summary(label_nodes=False)
        try:
            af.current.site.plot()
        except: # AttributeError:
            pass
        # fig.canvas.draw()
        plt.pause(0.01)
    
    if not af.current.best_child(): # cb=cb
        assert False
    cb()
    break

## 


##     
# Basic, no lookahead:
af.current=af.root=DTChooseSite(af)
while 1:
    if not af.current.children:
        break # we're done?
    
    for child_i in range(len(af.current.children)):
        if af.current.try_child(child_i):
            # Accept the first child which returns true
            break
    else:
        assert False # none of the children worked out
af.plot_summary(ax=ax)


# 4. Add in test metrics to evaluate the result of each step.
#    maybe use edits, which already tracks what parts of the grid have changed
# 5. Implement one-lookahead (ChooseSite won't do anything smart here...)
# 6. Implement n-lookahead

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
