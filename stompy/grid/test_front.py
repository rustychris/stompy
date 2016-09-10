import matplotlib.pyplot as plt
import numpy as np
import field
import unstructured_grid
reload(unstructured_grid)
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

test_halfedge_traverse()

af=test_basic_setup()

site=af.choose_site()
site.plot()

## 

# enumerate the strategies for a site:
# paver preemptively resamples the neighbors
# conceivable that one action might want to resample the neighbors
# in a slightly different way than another action.
# but the idea of having them spaced at the local scale when possible
# is general enough to do it preemptively.

def free_span(he,max_span,direction):
    span=0.0
    if direction==1:
        trav=he.node_fwd()
        last=anchor=he.node_rev()
    else:
        trav=he.node_rev()
        last=anchor=he.node_fwd()

    def pred(n):
        return ( (self.grid.nodes['fixed'][n]== self.SLIDE) and
                 len(self.grid.node_to_edges(n))<=2 )

    while pred(trav) and (trav != anchor) and (span<max_span):
        span += utils.dist( self.grid.nodes['x'][last] -
                             self.grid.nodes['x'][trav] )
        if direction==1:
            he=he.fwd()
            last,trav = trav,he.node_fwd()
        elif direction==-1:
            he=he.rev()
            last,trav = trav,he.node_rev()
        else:
            assert False
    return span
    
# when resample nodes on a sliding boundary, want to calculate the available
# span, and if it's small, start distributing the nodes evenly.
# where small is defined by local_scale * max_span_factor
max_span_factor=4     
def resample(n,anchor,scale,direction):
    self.log.debug("resample %d to be  %g away from %d in the %s direction"%(n,scale,anchor,
                                                                             direction) )
    if direction==1: # anchor to n is t
        he=self.grid.nodes_to_halfedge(anchor,n)
    elif direction==-1:
        he=self.grid.nodes_to_halfedge(n,anchor)

    span_length = free_span(he,max_span_factor*scale,direction)
    self.log.debug("free span from the anchor is %g"%slidable_length)

    if span_length < max_span_factor*scale:
        n_segments = max(1,round(span_length / scale))
        target_span = span_length / n_segments
    else:
        target_span=scale

    # HERE - move n, deleting anybody in its way, such that the distance between
    # n and anchor is approximately target_span
        
self=af

local_length = self.scale( points.mean(axis=0) )

# def resample_neighbors(self,site):
a,b,c = site.abc
for n,direction in [ (a,-1),
                     (c,1) ]:
    if ( (self.grid.nodes['fixed'][n] == self.SLIDE) and
         len(self.grid.node_to_edges(n))<=2 ):
        resample(n=n,anchor=b,scale=local_length,direction=direction)



## 
def wall():
    theta=site.internal_angle()
    points=site.points()
edge_length = utils.dist( np.diff(points,axis=0) ).mean()
scale_factor = edge_length / local_length

