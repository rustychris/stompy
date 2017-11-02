import os
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import matplotlib.pyplot as plt

from stompy.grid import paver
from stompy.spatial.linestring_utils import upsample_linearring,resample_linearring
from stompy.grid import paver
from stompy.spatial import field,constrained_delaunay,wkb2shp

##
from stompy.grid import exact_delaunay
from stompy.grid import live_dt
from stompy.grid import paver

reload(exact_delaunay)
reload(live_dt)
reload(paver)

##
def test_basic():
    # Define a polygon
    boundary=np.array([[0,0],[1000,0],[1000,1000],[0,1000]])
    island  =np.array([[200,200],[600,200],[200,600]])

    rings=[boundary,island]

    # And the scale:
    scale=field.ConstantField(50)

    p=paver.Paving(rings=rings,density=scale)

    p.pave_all()

## 
def test_basic_apollo():
    # Define a polygon
    boundary=np.array([[0,0],[1000,0],[1000,1000],[0,1000]])
    island  =np.array([[200,200],[600,200],[200,600]])

    rings=[boundary,island]

    # And the scale:
    scale=field.PyApolloniusField()
    scale.insert([50,50],20)

    p=paver.Paving(rings=rings,density=scale)

    p.pave_all()
    return p



##     
# A circle - r = 100, C=628, n_points = 628
def test_circle():
    r = 100
    thetas = np.linspace(0,2*np.pi,200)[:-1]
    circle = np.zeros((len(thetas),2),np.float64)
    circle[:,0] = r*np.cos(thetas)
    circle[:,1] = r*np.sin(thetas)
    class CircleDensityField(field.Field):
        # horizontally varying, from 5 to 20
        def value(self,X):
            X = np.array(X)
            return 5 + 15 * (X[...,0] + 100) / 200.0
    density = CircleDensityField()
    p=paver.Paving(circle,density,label='circle')
    p.pave_all()

def test_long_channel():
    l = 2000
    w = 50
    long_channel = np.array([[0,0],
                             [l,0],
                             [l,w],
                             [0,w]], np.float64 )

    density = field.ConstantField( 19.245 )
    p=paver.Paving(long_channel,density)
    p.pave_all()

def test_long_channel_rigid():
    l = 2000
    w = 50
    long_channel = np.array([[0,0],
                             [l,0],
                             [l,w],
                             [0,w]], np.float64 )

    density = field.ConstantField( 19.245 )
    p=paver.Paving(long_channel,density,initial_node_status=paver.Paving.RIGID)
    p.pave_all()



def test_narrow_channel():
    l = 1000
    w = 50
    long_channel = np.array([[0,0],
                             [l,0.375*w],
                             [l,0.625*w],
                             [0,w]], np.float64 )

    density = field.ConstantField( w/np.sin(60*np.pi/180.) / 4 )
    p=paver.Paving(long_channel,density)
    p.pave_all()
    
def test_small_island():
    l = 100
    square = np.array([[0,0],
                       [l,0],
                       [l,l],
                       [0,l]], np.float64 )

    r=10
    theta = np.linspace(0,2*np.pi,30)
    circle = r/np.sqrt(2) * np.swapaxes( np.array([np.cos(theta), np.sin(theta)]), 0,1)
    island1 = circle + np.array([45,45])
    island2 = circle + np.array([65,65])
    island3 = circle + np.array([20,80])
    rings = [square,island1,island2,island3]

    density = field.ConstantField( 10 )
    p=paver.Paving(rings,density)
    p.pave_all()

def test_tight_peanut():
    r = 100
    thetas = np.linspace(0,2*np.pi,300)
    peanut = np.zeros( (len(thetas),2), np.float64)
    x = r*np.cos(thetas)
    y = r*np.sin(thetas) * (0.9/10000 * x*x + 0.05)
    peanut[:,0] = x
    peanut[:,1] = y
    density = field.ConstantField( 6.0 )
    p=paver.Paving(peanut,density,label='tight_peanut')
    p.pave_all()

def test_tight_with_island():
    # build a peanut first:
    r = 100
    thetas = np.linspace(0,2*np.pi,250)
    peanut = np.zeros( (len(thetas),2), np.float64)
    x = r*np.cos(thetas)
    y = r*np.sin(thetas) * (0.9/10000 * x*x + 0.05)
    peanut[:,0] = x
    peanut[:,1] = y

    # put two holes into it
    thetas = np.linspace(0,2*np.pi,30)

    hole1 = np.zeros( (len(thetas),2), np.float64)
    hole1[:,0] = 10*np.cos(thetas) - 75
    hole1[:,1] = 10*np.sin(thetas)

    hole2 = np.zeros( (len(thetas),2), np.float64)
    hole2[:,0] = 20*np.cos(thetas) + 75
    hole2[:,1] = 20*np.sin(thetas)

    rings = [peanut,hole1,hole2]

    density = field.ConstantField( 6.0 )
    p=paver.Paving(rings,density,label='tight_with_island')
    p.pave_all()

def test_peninsula():
    r = 100
    thetas = np.linspace(0,2*np.pi,1000)
    pen = np.zeros( (len(thetas),2), np.float64)

    pen[:,0] = r*(0.2+ np.abs(np.sin(2*thetas))**0.2)*np.cos(thetas)
    pen[:,1] = r*(0.2+ np.abs(np.sin(2*thetas))**0.2)*np.sin(thetas)

    density = field.ConstantField( 10.0 )
    pen2 = upsample_linearring(pen,density)
    
    p=paver.Paving(pen2,density,label='peninsula')
    p.pave_all()


def test_peanut():
    # like a figure 8, or a peanut
    r = 100
    thetas = np.linspace(0,2*np.pi,1000)
    peanut = np.zeros( (len(thetas),2), np.float64)

    peanut[:,0] = r*(0.5+0.3*np.cos(2*thetas))*np.cos(thetas)
    peanut[:,1] = r*(0.5+0.3*np.cos(2*thetas))*np.sin(thetas)

    min_pnt = peanut.min(axis=0)
    max_pnt = peanut.max(axis=0)
    d_data = np.array([ [min_pnt[0],min_pnt[1], 1.5],
                        [min_pnt[0],max_pnt[1], 1.5],
                        [max_pnt[0],min_pnt[1], 8],
                        [max_pnt[0],max_pnt[1], 8]])
    density = field.XYZField(X=d_data[:,:2],F=d_data[:,2])

    p=paver.Paving(peanut,density)
    p.pave_all()

def test_cul_de_sac():
    r=5
    theta = np.linspace(-np.pi/2,np.pi/2,20)
    cap = r * np.swapaxes( np.array([np.cos(theta), np.sin(theta)]), 0,1)
    box = np.array([ [-3*r,r],
                     [-4*r,-r] ])
    ring = np.concatenate((box,cap))

    density = field.ConstantField(2*r/(np.sqrt(3)/2))
    p=paver.Paving(ring,density,label='cul_de_sac')
    p.pave_all()

def test_bow():
    x = np.linspace(-100,100,50)
    # with /1000 it seems to do okay
    # with /500 it still looks okay
    y = x**2 / 250.0
    bow = np.swapaxes( np.concatenate( (x[None,:],y[None,:]) ), 0,1)
    height = np.array([0,20])
    ring = np.concatenate( (bow+height,bow[::-1]-height) )
    density = field.ConstantField(2)
    p=paver.Paving(ring,density,label='bow')
    p.pave_all()

def test_ngon(nsides=7):
    # hexagon works ok, though a bit of perturbation
    # septagon starts to show expansion issues, but never pronounced
    # octagon - works fine.
    theta = np.linspace(0,2*np.pi,nsides+1)[:-1]

    r=100
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    poly = np.swapaxes( np.concatenate( (x[None,:],y[None,:]) ), 0,1)
    
    density = field.ConstantField(6)
    p=paver.Paving(poly,density,label='ngon%02d'%nsides)
    p.pave_all()

def test_expansion():
    # 40: too close to a 120deg angle - always bisect on centerline
    # 30: rows alternate with wall and bisect seams
    # 35: starts to diverge, but recovers.
    # 37: too close to 120.
    d = 36
    pnts = np.array([[0.,0.],
                     [100,-d],
                     [200,0],
                     [200,100],
                     [100,100+d],
                     [0,100]])

    density = field.ConstantField(6)
    p=paver.Paving([pnts],density,label='expansion')
    p.pave_all()

def test_embedded_channel():
    # trying out degenerate internal lines - the trick may be mostly in
    # how to specify them.
    # make a large rectangle, with a sinuous channel in the middle
    L = 500.0
    W = 300.0
    
    rect = np.array([[0,0],
                  [L,0],
                  [L,W],
                  [0,W]])

    x = np.linspace(0.1*L,0.9*L,50)
    y = W/2 + 0.1*W*np.cos(4*np.pi*x/L)
    shore = np.swapaxes( np.concatenate( (x[None,:],y[None,:]) ), 0,1)
    
    density = field.ConstantField(10)
    
    # this will probably get moved into Paver itself.
    # Note closed_ring=0 !
    shore = resample_linearring(shore,density,closed_ring=0)

    south_shore = shore - np.array([0,0.1*W])
    north_shore = shore + np.array([0,0.1*W])

    p=paver.Paving([rect],density,degenerates=[north_shore,south_shore])
    p.pave_all()

# dumbarton...
def test_dumbarton():
    shp=os.path.join( os.path.dirname(__file__), 'data','dumbarton.shp')
    features=wkb2shp.shp2geom(shp)
    geom = features['geom'][0]
    dumbarton = np.array(geom.exterior)
    density = field.ConstantField(250.0)
    p=paver.Paving(dumbarton, density,label='dumbarton')
    p.pave_all()

# #def log_spiral_channel():
# t = linspace(1.0,12*pi,200)
# a = 1 ; b = 0.1
# x = a*exp(b*t)*cos(t)
# y = a*exp(b*t)*sin(t)
# # each 2*pi, the radius gets bigger by exp(2pi*b)
# x2 = a*exp(b*t-b*pi)*cos(t)
# y2 = a*exp(b*t-b*pi)*sin(t)
# cla(); plot(x,y,'b',x2,y2,'r')

##

# This is going to require a fair bit of porting --

# hmm - maybe better just to have a sinusoid channel, then perturb it
# and put some islands in there.  having a wide range of scales looks
# nice but isn't going to be a great test.
def gen_sine_sine():
    t = np.linspace(1.0,12*np.pi,400)
    x1 = 100*t
    y1 = 200*np.sin(t)
    # each 2*pi, the radius gets bigger by exp(2pi*b)
    x2 = x1
    y2 = y1+50
    # now perturb both sides, but keep amplitude < 20
    y1 = y1 + 20*np.sin(10*t)
    y2 = y2 + 10*np.cos(5*t)
    
    x = np.concatenate( (x1,x2[::-1]) )
    y = np.concatenate( (y1,y2[::-1]) )

    shore = np.swapaxes( np.concatenate( (x[None,:],y[None,:]) ), 0,1)
    rings = [shore]

    # and make some islands:
    north_island_shore = 0.4*y1 + 0.6*y2
    south_island_shore = 0.6*y1 + 0.4*y2

    Nislands = 20
    # islands same length as space between islands, so divide
    # island shorelines into 2*Nislands blocks
    for i in range(Nislands):
        i_start = int( (2*i+0.5)*len(t)/(2*Nislands) )
        i_stop =  int( (2*i+1.5)*len(t)/(2*Nislands) )
        
        north_y = north_island_shore[i_start:i_stop]
        south_y = south_island_shore[i_start:i_stop]
        north_x = x1[i_start:i_stop]
        south_x = x2[i_start:i_stop]
        
        x = np.concatenate( (north_x,south_x[::-1]) )
        y = np.concatenate( (north_y,south_y[::-1]) )
        island = np.swapaxes( np.concatenate( (x[None,:],y[None,:]) ), 0,1)

        rings.append(island)

    density = field.ConstantField(25.0)
    min_density = field.ConstantField(2.0)
    p = paver.Paving(rings,density=density,min_density=min_density)
    
    print("Smoothing to nominal 1.0m")
    # mostly just to make sure that long segments are
    # sampled well relative to the local feature scale.
    p.smooth() 

    print("Adjusting other densities to local feature size")
    p.telescope_rate=1.1
    p.adjust_density_by_apollonius()

    return p

def test_sine_sine():
    p=gen_sine_sine()
    p.pave_all()


if 0:
    # debugging the issue with sine_sine()
    # fails deep inside here, step 512
    # lots of crap coming from this one, too.
    # at some point, dt_incident_constraints reports only 1 constraint,
    # but it should have two, which happens because of a bad slide.

    # Tricky to guard against -
    # Several avenues to fix this:
    #   1. Make the resample_neighbors code (which I'm pretty sure is the culprit)
    #      more cautious and willing to accept a local maximum in distance instead
    #      of shoving a node far away.  This is a nice but incomplete solution.
    #   2. The resample code, which I think is responsible for adding the new node
    #      that screwed it all up, should check for self-intersections
    #      this is probably the appropriate thing to do.


    # test_sine_sine()

    p=gen_sine_sine()

    p.pave_all(n_steps=512)

    ## 
    p.verbose=3

    p.pave_all(n_steps=513)
    ##
    zoom=plt.axis()

    plt.figure(1).clf()
    p.plot()
    p.plot_boundary()
    plt.axis('equal')
    plt.axis(zoom)
    ##

    # Step 510 really takes the end off an island
    # yep.
    p.pave_all(n_steps=512)

    ##

    # node is 3626
    # to_remove: an edge with nodes 5374, 3626
    # pnt2edges: [3626, 5915]
    # part of the problem is that there is some sliding around
    # at the beginning of step 512 that really wreaks havoc on
    # what was already a dicey node.
    p.plot_nodes([3626,5374])
