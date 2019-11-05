import numpy as np
import matplotlib.pyplot as plt
from stompy.grid import front, unstructured_grid
from stompy.plot import plot_wkb
from stompy.spatial import linestring_utils
from stompy import filters
from shapely import geometry

##

# define the domain


s=np.linspace(0,6000,200) # along-channel coordinate
amp=700 # amplitude of meanders
lamb=4000 # wave-length of meanders
width=500 # mean channel width
noise_w=50 # amplitude of noise to add to the channel banks
noise_l=1500 # length-scale of noise

centerline=np.c_[ s, amp*np.cos(2*np.pi*s/lamb)]
pline=geometry.LineString(centerline)
channel=pline.buffer(width/2)
ring=np.array(channel.exterior)
ring_norm=linestring_utils.left_normals(ring)

noise=(np.random.random(len(ring_norm))-0.5)
winsize=int( noise_l/( channel.exterior.length/len(ring_norm) ) )
noise[:winsize]=0 # so the ends still match up
noise[-winsize:]=0
noise_lp=filters.lowpass_fir(noise,winsize)
noise_lp *= noise_w/np.sqrt(np.mean(noise_lp**2))

# domain boundary including the random noise
ring_noise=ring+noise_lp[:,None]*ring_norm

# Create the curvilinear section
thalweg=centerline[50:110]

plt.figure(1).clf()
plt.plot(centerline[:,0],
         centerline[:,1],
         'k-',zorder=2)
plt.axis('equal')

plot_wkb.plot_wkb(channel,zorder=-2)
plt.plot(ring_noise[:,0],
         ring_noise[:,1],
         'm-')

plt.plot(thalweg[:,0],thalweg[:,1],'r--',lw=3)

##

# First, just the curvilinear section:
g=unstructured_grid.UnstructuredGrid(max_sides=4)

thalweg_resamp=linestring_utils.resample_linearring(thalweg,50,closed_ring=0)

g.add_rectilinear_on_line(thalweg_resamp,
                          profile=lambda x,s,perp: np.linspace(-200,200,20),
                          add_streamwise=False)
g.plot_edges(zorder=5,color='y')
