"""
Venturing into generic code to match two datasets.  

Not remotely generic at this point, and makes some assumptions
about dimensions, depth, time, etc.

"""
import numpy as np
import xarray as xr
from scipy.spatial import kdtree

from .. import utils


class MatchVarsCruise(object):
    def __init__(self,varA,varB,B_type):
        """
        Building on the development in ~/notebooks/nitrogen_budgets/sfbay_din/

        Callable instance which takes a variable with the same shape/dimenions as
        varB, and returns a variable of the shape/dims of varA.
        """
        new_coords={}    

        #---- Time!

        mapBtime_to_A = np.searchsorted( varB.time.values,
                                         varA.time.values )
        if 1:
            mapBtime_to_A=np.ma.array( mapBtime_to_A,
                                       mask=utils.isnat(varA.time.values) )

        new_coords['time']= xr.DataArray( varB.time.values[mapBtime_to_A], dims=varA.time.dims)

        #---- Stations:
        A_xy=np.array( [varA.x, varA.y] ).T 
        if B_type=='hist':
            B_xy=np.array( [varB.element_x,varB.element_y] ).T
        elif B_type=='map':
            B_xy=np.array( [varB.FlowElem_xcc,varB.FlowElem_ycc] ).T

        # in the case of hist files, some history output isn't tied to a spatial element
        # (element ids come from a convention in waq_scenario), and those elements will
        # have nan coordinates
        valid=np.isfinite(B_xy[:,0])
        kdt=kdtree.KDTree(B_xy[valid])
        dists,valid_indices = kdt.query(A_xy)
        # print("Distance from observed locations to model output: ",dists)
        all_indices= np.arange(B_xy.shape[0])[valid][valid_indices]
        mapBstn_to_A=all_indices

        new_coords['x'] = xr.DataArray(B_xy[mapBstn_to_A,0],dims=['Distance_from_station_36'])
        new_coords['y'] = xr.DataArray(B_xy[mapBstn_to_A,1],dims=['Distance_from_station_36'])

        #---- Layers:

        A_depth=varA.depth # fully 3D, all values present
        B_depth=varB.localdepth # fully 3D, lots missing

        mapBdepth_to_A=np.zeros(varA.shape,'i4')
        mask=np.zeros(varA.shape,'b1')

        # This set of loops is painful slow
        for idx0 in range(varA.shape[0]):
            for idx1 in range(varA.shape[1]):
                # gets tricky with generalizing here
                # varA.time.dims => ('date', 'Distance_from_station_36', 'prof_sample')
                # varA.Distance_from_station_36.dims => 'Distance_from_station_36'

                for idx2 in range(varA.shape[2]):
                    Bidx0=mapBtime_to_A[idx0,idx1,idx2]
                    masked=mapBtime_to_A.mask[idx0,idx1,idx2]
                    Bidx1=mapBstn_to_A[idx1] # could be moved out

                    if masked:
                        mask[idx0,idx1,idx2]=True
                        continue

                    this_A_depth  =A_depth[idx0,idx1,idx2]
                    these_B_depths=B_depth[Bidx0,Bidx1,:]
                    valid=np.isfinite(these_B_depths)
                    idx_valid=np.searchsorted(these_B_depths[valid],this_A_depth)
                    idx_valid=idx_valid.clip(0,len(these_B_depths)-1)
                    idx=np.arange(len(these_B_depths))[idx_valid]

                    mapBdepth_to_A[idx0,idx1,idx2]=idx

        mapBdepth_to_A=np.ma.array(mapBdepth_to_A,mask=mask)

        #---- Extract depth for a coordinate

        new_depths=B_depth.values[mapBtime_to_A,
                                  mapBstn_to_A[None,:,None],
                                  mapBdepth_to_A]
        
        new_coords['depth']= xr.DataArray( np.ma.array(new_depths,mask=mask),
                                           dims=varA.dims)
        # save the mapping info
        self.mask=mask
        self.new_coords=new_coords
        self.map_time=mapBtime_to_A
        self.map_station=mapBstn_to_A
        self.map_depth=mapBdepth_to_A
        self.varA=varA

    # important to pass these in as default args to establish a robust
    # binding
    def __call__(self,varB):
        #---- Extract the actual analyte
        newBvalues=varB.values[ self.map_time,
                                self.map_station[None,:,None],
                                self.map_depth ]
        newBvalues=np.ma.array(newBvalues,mask=self.mask)

        # the primary dimensions are copied from A
        Bcoords=[ (d,self.varA[d]) 
                  for d in self.varA.dims ]
        newB=xr.DataArray(newBvalues,
                          dims=self.varA.dims,
                          coords=Bcoords)
        # additionaly coordinate information reflects the original times/locations
        # of the data
        newB=newB.assign_coords(**self.new_coords)

        return newB
