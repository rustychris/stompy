"""
a variation on contour plots that understands transect data

This is deprecated in favor of plot_utils.tricontourf(), but included
here because of old references from suntans plotting code.

It probably is not functional, as it has not been tested since being
brought into the tree from an old repository.
"""

from __future__ import print_function

# The goal is to show contours even at water columns where
# the neighboring columns are shallower.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection,LineCollection


def contourf_t(X,Y,Z,V=None,ax=None,**kwargs):
    ax=ax or plt.gca()
    if V is not None:
        cset = ax.contourf(X,Y,Z,V,**kwargs)
    else:
        cset = ax.contourf(X,Y,Z,**kwargs)

    # look at one layer:
    for i in range(len(cset.levels)-1):
        low = cset.levels[i]
        high = cset.levels[i+1]
        coll = cset.collections[i]

        my_paths = []

        # go over the transect data looking for triangles at the bottom that should have
        # part of this contour:

        for col in range(Z.shape[0]-1):
            # find last valid cell for col and col+1
            d1 = np.where( ~Z.mask[col,:]   )[0]
            d2 = np.where( ~Z.mask[col+1,:] )[0]
            if len(d1) > 0 and len(d2) > 0:
                d1 = d1[-1]
                d2 = d2[-1]
            else:
                continue

            c1 = col
            c2 = col+1

            # rearrange so that d1 is the deeper column
            if d1 > d2:
                pass
            elif d2 > d1:
                d2,d1 = d1,d2
                c2,c1 = c1,c2
            else:
                continue

            # Iterate over all triangles joining the two columns
            # here c always refers to the bottom-most point in the
            # shallow column, a refers to the upper point in the
            # deep column and b to the deeper point
            c = d2
            for a in range(d2,d1):
                b = a+1
                # values at the vertices a,b,c
                vals = np.array( [ Z[c1,a],Z[c1,b],Z[c2,c]] )
                # x,y (really dist,depth) locations of the vertices a,b,c
                locs = np.array( [[X[c1,a],Y[c1,a] ],
                                  [X[c1,b],Y[c1,b] ],
                                  [X[c2,c],Y[c2,c] ]] )

                # print "Would be looking at the triangle with values:",vals
                if np.all(vals < low) or np.all(vals > high):
                    continue

                # iterate over edges and output the polygon that falls within the range:
                verts = []
                for i in range(3):
                    ip1 = (i+1)%3

                    if vals[i] < low:
                        cmp_i = -1
                    elif vals[i] > high:
                        cmp_i = 1
                    else:
                        cmp_i = 0

                    if vals[ip1] < low:
                        cmp_ip1 = -1
                    elif vals[ip1] > high:
                        cmp_ip1 = 1
                    else:
                        cmp_ip1 = 0

                    # 9 total possibilities...

                    # this edge is completely outside interval
                    if cmp_i * cmp_ip1 == 1:
                        continue

                    # 7 possibilities left

                    # do we include vertex i?
                    if cmp_i == 0:
                        verts.append( locs[i] )

                        if cmp_ip1 == 0:
                            pass
                        else:
                            if cmp_ip1 == 1:
                                # find point where val goes to high
                                alpha = (high - vals[i]) / (vals[ip1] - vals[i])
                            else: # cmp_ip1 == -1
                                alpha = (low - vals[i]) / (vals[ip1] - vals[i])
                            verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )

                    elif cmp_i == -1:# 4 possibilities left
                        # look for the intersection with low
                        alpha = (low - vals[i]) / (vals[ip1] - vals[i])
                        verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )

                        if cmp_ip1 == 1:
                            # and another intersection with high
                            alpha = (high - vals[i]) / (vals[ip1] - vals[i])
                            verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )
                        else:
                            pass # ip1 will get included in next loop
                    elif cmp_i == 1:
                        # look for intersection with high
                        alpha = (high - vals[i]) / (vals[ip1] - vals[i])
                        verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )
                        if cmp_ip1 == -1:
                            # and intersection with low
                            alpha = (low - vals[i]) / (vals[ip1] - vals[i])
                            verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )
                    else:
                        raise Exception("how did you get here?")


                    my_paths.append( np.array(verts) )
        # package up my_paths into a poly collection:
        pcoll = PolyCollection(my_paths,facecolors=coll.get_facecolor(),
                               edgecolors=coll.get_facecolor())

        ax.add_collection(pcoll)
    plt.draw()
    return cset


def contour_t(X,Y,Z,V=None,ax=None,**kwargs):
    """ draws contour lines for transect-like data
    """
    ax=ax or plt.gca()
    if V is not None:
        cset = ax.contour(X,Y,Z,V,**kwargs)
    else:
        cset = ax.contour(X,Y,Z,**kwargs)
    
    # look at one layer:
    for i in range(len(cset.levels)):
        thresh = cset.levels[i]
        coll = cset.collections[i]

        my_paths = []

        # go over the transect data looking for triangles at the bottom that should have
        # part of this contour:

        for col in range(Z.shape[0]-1):
            # find last valid cell for col and col+1
            d1 = np.where( ~Z.mask[col,:]   )[0][-1]
            d2 = np.where( ~Z.mask[col+1,:] )[0][-1]

            c1 = col
            c2 = col+1


            # rearrange so that d1 is the deeper column
            if d1 > d2:
                pass
            elif d2 > d1:
                d2,d1 = d1,d2
                c2,c1 = c1,c2
            else:
                continue

            # Iterate over all triangles joining the two columns
            # here c always refers to the bottom-most point in the
            # shallow column, a refers to the upper point in the
            # deep column and b to the deeper point
            c = d2
            for a in range(d2,d1):
                b = a+1
                # values at the vertices a,b,c
                vals = np.array( [ Z[c1,a],Z[c1,b],Z[c2,c]] )
                # x,y (really dist,depth) locations of the vertices a,b,c
                locs = np.array( [[X[c1,a],Y[c1,a] ],
                                  [X[c1,b],Y[c1,b] ],
                                  [X[c2,c],Y[c2,c] ]] )

                # iterate over edges and output the line that falls within the range:
                verts = [] 
                
                for i in range(3):
                    ip1 = (i+1)%3

                    cmp_i = cmp(vals[i],thresh)
                    cmp_ip1 = cmp(vals[ip1],thresh)
                    cmp_ip2 = cmp(vals[(i+2)%3],thresh)

                    # 9 total possibilities... (ignoring cmp_ip2)

                    # this edge doesn't intersect the contour at all
                    if cmp_i * cmp_ip1 == 1:
                        continue

                    # do we include vertex i?
                    if cmp_i == 0:
                        if cmp_ip1 == 0:
                            # this edge is part of the contour, but should we
                            # include it?  only when the third vertex is
                            # above the contour (an attempt to avoid double-counting
                            # this edge)
                            if cmp_ip2 == 1:
                                verts.append( locs[i] )
                                verts.append( locs[ip1] )
                                break
                        else:
                            # if our neighbors are on opposite sides of the contour,
                            # include this point
                            if cmp_ip1 * cmp_ip2 == -1:
                                verts.append( locs[i] )
                            elif cmp_ip1 * cmp_ip2 == 1:
                                # only this vertex is on the contour, so draw nothing
                                break
                            else:
                                # ip2-ip is tangent to the contour, it will get
                                # handled on a different iteration of this loop
                                continue
                    elif cmp_i * cmp_ip1 == -1:
                        # this edge crosses the contour
                        alpha = (thresh - vals[i]) / (vals[ip1] - vals[i])
                        verts.append( (1-alpha)*locs[i] + alpha*locs[ip1] )
                        continue
                    elif cmp_ip1 == 0:
                        # this should only happen when cmp_ip1=0, in which case
                        # we are definitely not on the contour
                        continue
                    else:
                        raise Exception("how did you get here?")
                    
                if len(verts) == 2:
                    my_paths.append( verts )
                elif len(verts) != 0:
                    print("Verts:",verts)
                    raise Exception("How did you get that number of verts?")
                
        # package up my_paths into a poly collection:
        lcoll = LineCollection(my_paths,colors=coll.get_edgecolor(),
                               linestyles=coll.get_linestyle() )

        ax.add_collection(lcoll)
    plt.draw()
    return cset

# # call it just like regular contourf
# if __name__ == '__main__':
#     # Get some sample data:
#     import polaris
# 
#     pc = polaris.PolarisCruise(juliandate=2008323)
#     pc.fetch()
#     t = pc.to_transect()
# 
#     D,X = meshgrid(t.depths,t.dists)
#     V = linspace(0,32,9)
#     args = []
# 
#     clf()
#     contourf_t(X,D,transpose(t.scalar),V)
#     contour_t(X,D,transpose(t.scalar),linspace(0,32,33),colors='k')
 

# if 0: # reimplemented via tricontour:
#     def contourf_t(X,Y,Z,V=None,ax=None,**kwargs):
#         X,Y=np.meshgrid(x,-trans_nc.PrDM[:])
#         Z=ma.masked_invalid(trans_nc.variables[field][:].T)
#         V=25
#         #
#         ax=ax or gca()
#         if V is not None:
#             cset = ax.contourf(X,Y,Z,V)
#         else:
#             cset = ax.contourf(X,Y,Z)
# 
#         # X,Y,Z are all (vertical,horizontal)
#         # Z is masked
# 
#         # build up triangles to hand off to tricontour
#         # assumes that the missing data is at the end
#         tris=[]
# 
#         # raveling gives along line (cols) first, then depth (rows)
#         for left in range(X.shape[1]-1):
#             right=left+1
#             left_last=np.nonzero(~Z.mask[:,left])[0].max()
#             right_last=np.nonzero(~Z.mask[:,right])[0].max()
#             cols=X.shape[1]
#             for left_side in range(right_last,left_last):
#                 # add a triangle 
#                 tris.append( [left_side*cols+left,
#                               (left_side-1)*cols+left,
#                               right_last*cols+right] )
#             for right_side in range(left_last,right_last):
#                 # add a triangle 
#                 tris.append( [right_side*cols+right,
#                               (right_side-1)*cols+right,
#                               left_last*cols+left] )
# 
#         tris=np.array(tris)
# 
#         ax.tricontourf(X.ravel(),
#                        Y.ravel(),
#                        tris,
#                        Z.ravel(),
#                        cset.levels )
# 
#         return cset

