"""
Encapsulating some of the 2D particle tracking from
currents-figures-netcdf-animation-v03.ipynb
"""
from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import six
import logging

import matplotlib.dates as mdates
from shapely import geometry

import xarray as xr
from stompy import utils
from stompy.plot import (plot_utils, cmap, plot_wkb)
from stompy.grid import unstructured_grid
from stompy.spatial import (wkb2shp,field)

##
class UgridParticles(object):
    """
    time: all timestamps are stored as 64-bit double, in seconds
    since the epoch.

    This is to hedge between the various desires on time - data is
    rarely at a resolution finer than 1s.  A unix timestamp in 64-bit
    double is exact to the second for over 280 million years.
    """
    # x: horizontal location
    # c: cell containing particle
    # j_last: last edge the particle crossed
    part_dtype = [ ('x',(np.float64,2)),
                   ('c',np.int32),
                   ('u',(np.float64,2)),
                   ('j_last',np.int32) ]

    def __init__(self,ncs,grid=None,**kw):
        self.__dict__.update(kw)
        self.log=logging.getLogger(self.__class__.__name__)
        self.ncs=ncs
        self.scan_ncs()

        self.load_grid(grid=grid)

        self.init_particles()
        self.t_unix=None
        self.current_nc_idx=-1

        self.U=None

    def scan_ncs(self):
        self.nc_catalog=np.zeros( len(self.ncs),
                                  dtype=[('start_t','f8'),
                                         ('end_t','f8')] )
        for nc_i,nc in enumerate(self.ncs):
            self.nc_catalog['start_t'][nc_i] = utils.to_unix(nc.time[0])
            self.nc_catalog['end_t'][nc_i]   = utils.to_unix(nc.time[-1])

    def set_time(self,t):
        self.t_unix=t
        self.update_velocity()

    def load_grid(self,grid=None):
        if grid is None:
            self.g=unstructured_grid.UnstructuredGrid.from_ugrid(self.ncs[0])
        else:
            self.g=grid
        self.g.edge_to_cells()

        self.edge_norm=self.g.edges_normals()

    def set_current_nc(self,nc_i):
        self.current_nc_idx=nc_i
        self.current_nc=self.ncs[self.current_nc_idx]
        self.nc_t_unix=utils.to_unix(self.current_nc.time.values)
        self.nc_time_i=-999

    def update_velocity(self):
        changed=False
        nc_i = np.searchsorted(self.nc_catalog['start_t'],self.t_unix,side='right') - 1

        if nc_i != self.current_nc_idx:
            self.set_current_nc(nc_i)
            changed=True

        new_time_i=np.searchsorted(self.nc_t_unix, self.t_unix,side='right')-1
        if new_time_i!=self.nc_time_i:
            changed=True
        self.nc_time_i=new_time_i

        # find out how long this velocity is good for
        next_step=np.inf
        if self.nc_time_i+1 < len(self.nc_t_unix):
            next_step=self.nc_t_unix[self.nc_time_i+1]
        if self.current_nc_idx+1<len(self.ncs):
            next_step=min(next_step, 
                          self.nc_catalog['start_t'][self.current_nc_idx+1] )

        assert np.isfinite(next_step) # did we integrate off the end of the avail. data?

        self.velocity_valid_time=[self.nc_t_unix[self.nc_time_i],
                                  next_step ]
        # This is split off here because it's a point where basic
        # and KSG differ
        self.update_particle_velocity_for_new_step()

    def update_particle_velocity_for_new_step(self):
        # face, layer, time.
        # assumes 2D here.
        u=self.current_nc.cell_east_velocity.values[:,0,self.nc_time_i]
        v=self.current_nc.cell_north_velocity.values[:,0,self.nc_time_i]

        self.U=np.array( [u,v] ).T # again assume 2D

        # A little dicey - this overwrites any memory of convergent edges.
        # so every input interval, it's going to forget
        self.P['u']=self.U[ self.P['c'] ]

    def init_particles(self):
        self.P=np.zeros( 0, self.part_dtype )

    def add_particles(self,**kw):
        fields=list(kw.keys())

        Nold=len(self.P)
        Nnew=np.atleast_1d( kw[fields[0]] ).shape[0]

        recs=np.zeros( Nnew, dtype=self.part_dtype)

        slc=slice(Nold,Nold+Nnew)

        self.P=utils.array_concatenate( [self.P,recs] )

        for k,v in six.iteritems(kw):
            self.P[k][slc]=v

        # figure out which cell they are in
        for i in range(Nold,Nold+Nnew):
            c=self.g.select_cells_nearest(self.P['x'][i],inside=True)
            self.P['c'][i]=c
            self.P['j_last'][i]=-999

            # if the velocity fields were continuous, then we could
            # skip this part, since we wouldn't really need to store
            # velocity.
            self.P['u'][i] = np.nan # signal that it needs to be set

    record_dense=False

    def integrate(self,output_times_unix):
        next_out_idx=0
        next_out_time=output_times_unix[next_out_idx]

        self.output=[]
        self.append_state(self.output)
        if self.record_dense:
            self.dense=[]
            self.append_state(self.dense)

        next_vel_time=self.velocity_valid_time[1]

        assert self.t_unix>=self.velocity_valid_time[0]
        assert self.t_unix<=self.velocity_valid_time[1]
        assert next_out_time>=self.t_unix

        while next_out_time is not None: # main loop
            # the max time step we can take is the minimum of
            # (i) time to next output interval
            # (ii) time to next update of input velocities
            # (iii) time to cross into a new cell
            #[(iv) eventually, time until update behavior]

            t_next=min(next_out_time,next_vel_time)

            self.move_particles(t_next)

            self.t_unix=t_next
            if t_next==next_out_time:
                self.log.info('Output %d / %d'%(next_out_idx,len(output_times_unix)))
                self.append_state(self.output)
                next_out_idx+=1
                if next_out_idx<len(output_times_unix):
                    next_out_time=output_times_unix[next_out_idx]
                else:
                    next_out_time=None
            if t_next==next_vel_time:
                self.update_velocity()
                next_vel_time=self.velocity_valid_time[1]
                self.P['j_last']=-999 # okay to cross back if the velocities changed.


    def append_state(self,A):
        A.append( (self.P['x'].copy(), self.t_unix ))

    def move_particles(self,stop_t):
        """
        Advance each particle to the correct state at stop_t.
        Assumes that no input (updating velocities) or output
        is needed between self.t_unix and stop_t.

        Caller is responsible for updating self.t_unix
        """
        g=self.g

        for i,p in enumerate(self.P):
            # advance each particle to the correct state at stop_t
            part_t=self.t_unix

            if np.isnan( p['u'][0] ):
                # probably first time this particle has been moved.
                self.P['u'][i]=self.U[ self.P['c'][i] ]

            while part_t<stop_t:
                dt_max_edge=np.inf
                j_cross=None
                j_cross_normal=None

                for j in g.cell_to_edges(p['c']):
                    if j==p['j_last']:
                        continue # don't cross back
                    normal=self.edge_norm[j]
                    if g.edges['cells'][j,1]==p['c']: # ~checked
                        normal=-normal
                    # vector from xy to a point on the edge
                    d_xy_n = g.nodes['x'][g.edges['nodes'][j,0]] - p['x']
                    # perpendicular distance
                    dp_xy_n=d_xy_n[0] * normal[0] + d_xy_n[1]*normal[1]
                    assert dp_xy_n>=0 #otherwise sgn probably wrong above

                    #closing=u*normal[0] + v*normal[1]
                    closing=self.P['u'][i,0]*normal[0] + self.P['u'][i,1]*normal[1]

                    if closing<0: 
                        continue
                    else:
                        dt_j=dp_xy_n/closing
                        if dt_j>0 and dt_j<dt_max_edge:
                            j_cross=j
                            dt_max_edge=dt_j
                            j_cross_normal=normal

                t_max_edge=part_t+dt_max_edge
                if t_max_edge>stop_t:
                    # don't make it to the edge
                    dt=stop_t-part_t
                    part_t=stop_t
                    j_cross=None
                else:
                    dt=dt_max_edge
                    part_t=t_max_edge

                # Take the step
                delta=self.P['u'][i]*dt
                # see if we're stuck
                if utils.mag(delta) / (utils.mag(delta) + utils.mag(self.P['x'][i])) < 1e-14:
                    print("Steps are too small")
                    part_t=stop_t
                    continue

                self.P['x'][i] += delta

                if j_cross is not None:
                    # cross edge j, update time.  careful that j isn't boundary
                    # or start sliding on boundary.
                    # print "Cross edge"
                    cells=g.edges['cells'][j_cross]
                    if cells[0]==p['c']:
                        new_c=cells[1]
                    elif cells[1]==p['c']:
                        new_c=cells[0]
                    else:
                        assert False

                    # More scrutiny on the edge crossing -
                    # would it take us out of the domain? then bounce and frown.
                    # would it take us to a convergent edge? then bounce and frown.
                    bounce=False

                    if new_c<0:
                        bounce=True
                    else:
                        recross= ( self.U[new_c,0]*j_cross_normal[0] + 
                                   self.U[new_c,1]*j_cross_normal[1] )
                        if recross<=0: 
                            bounce=True

                    if bounce:
                        closing= (self.P['u'][i,0]*j_cross_normal[0] + 
                                  self.P['u'][i,1]*j_cross_normal[1] )
                        # slightly over-compensate, pushing away from problematic
                        # edge
                        print("BOUNCE")

                        self.P['u'][i] -= 1.1 * j_cross_normal*closing
                        self.P['j_last'][i]=j_cross
                    else:
                        self.P['c'][i]=new_c
                        self.P['j_last'][i]=j_cross
                        self.P['u'][i] = self.U[new_c]
                        # HERE: need to check whether the new cell's
                        # velocity is going to push us back, in which case
                        # we should instead scoot along the tangent and
                        # hide our faces.
                        # actually, better solution is to handle *before*
                        # the particle hits the edge.  If we handle it after,
                        # two particles converging on this edge will cross
                        # paths, and that really doesn't pass the sniff test.
                        # this is not good, but will let the sim complete

                    if self.record_dense:
                        self.append_state(self.dense)

    def save_tracks(self,fn,overwrite=True):
        if os.path.exists(fn):
            if overwrite:
                os.unlink(fn)
            else:
                raise Exception("Output %s already exists"%fn)

        # archive the grid for good measure:
        self.g.write_ugrid(fn)
        ds=xr.open_dataset(fn).copy(deep=True)

        ds['particle']=( ('particle',), np.arange(len(self.P)) )
        times=np.array([out[1] for out in self.output])

        ds['time'] = ( ('time',), times )
        ds.time.attrs['units']="seconds since 1970-01-01T00:00:00"

        X=np.array( [out[0] for out in self.output] )

        ds['x'] = ( ('time','particle'), X[:,:,0] )
        ds['y'] = ( ('time','particle'), X[:,:,1] )

        os.unlink(fn)
        ds.to_netcdf(fn)
        return ds

#    Edges and incompatible velocities
#    ---------------------------------
#
#    With a discontinuous velocity field, sooner or later there
#    is a particle which is one cell, pushed to the edge, and
#    enters a new cell which also wants to push the particle to
#    the edge.
#
#    Solutions:
#     1. Fix the velocity field.  The 0th order in space (constant
#        velocity within a cell) approach cannot be fixed in this way.
#        Going to a reconstruction like in Gerard's paper would be a
#        solution, though it doesn't extend beyond triangles.  Postma
#        I think would be sufficient to get triangles and quads, but
#        still no good on pentagons, hexes, etc.
#     2. Adjust the velocity when this happens.  The adjustment might
#        be to take the average of the two cells, reach back and use
#        an edge velocity to figure it out, force a tangent trajectory,
#        or other approaches.  
#    
#        Do any of these approaches not break the reversibility of 
#        the method?  Seems that any approach which "reacts" to finding 
#        a convergent edge is not reversible, since that edge would be 
#        divergent in reverse.  
#        
#        I think for now I have to give up on reversibility, but at least
#        preserve invariance w.r.t. to output time stepping.  I.e. changing the
#        output timestep shouldn't alter any results.


# Method of Ketefian, Gross, Stelling
# - need a 2x2 matrix Ai and vector Bi per cell
#   u is linear in x,y
#   they include closed solutions for a trajectory

# The full analytic solution is pretty messy, but the bulk of the mess
# is due to the analytical integration of the linear velocity field.
# an intermediate step would be to go for the velocity field reconstruction,
# but go back to a numerical integration.
