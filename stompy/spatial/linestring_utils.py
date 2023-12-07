# import field
from numpy import *
from numpy.linalg import norm
import logging as log
import numpy as np

def as_density(d):
    #if not isinstance(density,field.Field):
    #    density = field.ConstantField(density)

    # try for duck typing here
    try:
        d([0,0])
    except TypeError:
        orig_density = d
        d = lambda X,orig_density=orig_density: orig_density * ones(X.shape[:-1])
    return d

def upsample_linearring(points,density,closed_ring=1,return_sources=False):
    new_segments = []

    sources = []

    density = as_density(density)
    points=np.asarray(points)

    for i in range(len(points)):
        A = points[i]

        if i+1 == len(points) and not closed_ring:
            new_segments.append( [A] )
            sources.append( [i] )
            break

        B = points[(i+1)%len(points)]

        l = norm(B-A)
        # print "Edge length is ",l

        scale = density( 0.5*(A+B) )
        # print "Scale is ",scale

        npoints = max( [1, int(round( l/scale ))] )
        # print "N points ",npoints

        alphas = arange(npoints) / float(npoints)

        new_segment = (1.0-alphas[:,newaxis])*A + alphas[:,newaxis]*B
        new_segments.append(new_segment)
        sources.append(i + alphas)

    new_points = concatenate( new_segments )

    if return_sources:
        sources = concatenate(sources)
        # print "upsample: %d points, %d alpha values"%( len(new_points), len(sources))
        return new_points,sources
    else:
        return new_points


def downsample_linearring(points,density,factor=None,closed_ring=1):
    """ Makes sure that points aren't *too* close together
    Allow them to be 0.3*density apart, but any edges shorter than that will
    lose one of their endpoints.
    """
    if factor is not None:
        density = density * factor # should give us a BinOpField
    density = as_density(density)

    valid = ones( len(points), 'bool8')

    # go with a slower but safer loop here -
    last_valid=0
    for i in range(1,len(points)):
        scale = density( 0.5*(points[last_valid] + points[i]) )

        if norm( points[last_valid]-points[i] ) < scale:
            if i==len(points)-1:
                # special case to avoid moving the last vertex
                valid[last_valid] = False
                last_valid = i
            else:
                valid[i] = False
        else:
            last_valid = i

    return points[valid]

def resample_linearring(points,density,closed_ring=1,return_sources=False):
    """  similar to upsample, but does not try to include
    the original points, and can handle a density that changes
    even within one segment of the input
    """
    if isinstance(density,np.ndarray):
        density_mode='precalc'
    else:
        density_mode='dynamic'
        density = as_density(density)

    if closed_ring:
        points = concatenate( (points, [points[0]]) )

    # distance_left[i] is the distance from points[i] to the end of
    # the line, along the input path.
    lengths = sqrt( ((points[1:] - points[:-1])**2).sum(axis=1) )
    distance_left = cumsum( lengths[::-1] )[::-1]

    new_points = []
    new_points.append( points[0] )

    # x=sources[i] means that the ith point is between points[floor(x)]
    # and points[floor(x)+1], with the fractional step between them
    #  given by x%1.0
    sources = [0.0]

    # indexes the ending point of the segment we're currently sampling
    # the starting point is just new_points[-1]
    i=1

    # print "points.shape ",points.shape
    while 1:
        last_point = new_points[-1]
        last_source = sources[-1]

        if i < len(distance_left): 
            total_distance_left = norm(points[i] - last_point) + distance_left[i] 
        else:
            total_distance_left = norm(points[i] - last_point)

        if density_mode=='dynamic':
            scale=density(last_point)
        elif density_mode=='precalc':
            # density is precalculated at the input points
            alpha=last_source%1.0
            scale=(1-alpha)*density[i-1] + alpha*density[i]
        npoints_at_scale = round( total_distance_left/scale )

        if (npoints_at_scale <= 1):
            if not closed_ring:
                new_points.append( points[-1] )
                sources.append(len(points)-1)
            break

        this_step_length = total_distance_left / npoints_at_scale
        #print("scale = %g   this_step_length = %g "%(scale,this_step_length))

        # at this point this_step_length refers to how far we must go
        # from new_points[i], along the boundary.

        while norm( points[i] - last_point ) < this_step_length:
            # print "i=",i
            this_step_length -= norm( points[i] - last_point )
            last_point = points[i]
            last_source = float(i)
            i += 1


        seg_length = norm(points[i] - points[i-1])
        # along this whole segment, we might be starting in the middle
        # from a last_point that was on this same segment, in which
        # case add our alpha to the last alpha
        # print "[%d,%d] length=%g   step_length=%g "%(i-1,i,seg_length,this_step_length)

        alpha = this_step_length / seg_length
        # print "My alpha", alpha
        last_alpha = (last_source  - floor(last_source))
        # print "Alpha from last step:",last_alpha
        alpha = alpha + last_alpha

        new_points.append( (1-alpha)*points[i-1] + alpha * points[i] )
        frac = norm(new_points[-1] - points[i-1])/ norm(points[i] - points[i-1])

        # print "frac=%g   alpha = %g"%(frac,alpha)
        sources.append( (i-1) + frac )

    new_points = array( new_points )

    if return_sources:
        sources = array( sources )
        return new_points,sources
    else:
        return new_points


def distance_along(linestring):
    # linestring: [N,2]
    diffs=np.sqrt( np.sum( np.diff(linestring,axis=0)**2, axis=1) )

    return np.concatenate( ( [0],
                             np.cumsum(diffs) ) )

def left_normals(linestring):
    """
    For each point in the [N,2] linestring find the left-pointing
    unit normal vector, returned in a [N,2] array.
    """
    # central differences:
    ctr_diffs=linestring[2:,:] - linestring[:-2,:]
    # one-sided at ends
    a_diffs=linestring[1:2,:] - linestring[:1,:]
    z_diffs=linestring[-1:,:] - linestring[-2:-1,:]

    vecs=np.r_[a_diffs,ctr_diffs,z_diffs]
    left_vecs=np.c_[ -vecs[:,1], vecs[:,0] ]
    mags=np.sqrt( (left_vecs**2).sum(axis=1) )
    bad=mags==0
    if np.any(bad):
        mags[bad]=1.
        log.warning("left_normals: repeated points in linestring")
        raise Exception("Why is this happening")
    normals=left_vecs/mags[:,None]
    normals[bad,:]=np.nan
    return normals


