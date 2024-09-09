from __future__ import print_function

from optparse import OptionParser

# try to connect features that really ought to be connected:
import numpy as np

import shapely.wkb,shapely.geometry
try:
    from osgeo import ogr
except ImportError:
    import ogr
import sys
import os.path
import six

from numpy.linalg import norm

from . import wkb2shp
from .. import utils

import logging
logging.basicConfig(level=logging.INFO)
log=logging.getLogger('join_features')

try:
    from shapely.prepared import prep as prepare_geometry
except ImportError:
    prepare_geometry=lambda x:x
    log.warning("Prepared geometries not available - tests will be slow")

try:
    from shapely.strtree import STRtree
except ImportError:
    # this will make complex inputs horrifically slow, but might
    # complete
    class STRtree(object):
        def __init__(self,geoms):
            self.geoms=geoms
        def query(self,g):
            return self.geoms

def progress_printer(str,steps_done=None,steps_total=None):
    if steps_done is not None and steps_total is not None:
        log.info( "%s -- %d%%"%(str,100.0*steps_done/steps_total) )
    elif steps_done:
        log.info( "%s -- %d"%(str,steps_done) )
    else:
        log.info(str)

progress_message = progress_printer

# in a few cases (<0.01% ??) the prepared geometry will report a different answer
# than the real geometry.  But the prepared geomtries are an order of magnitude
# faster...
trust_prepared = True

def merge_lines(layer=None,segments=None):
    """ Given an ogr LineString layer, merge linestrings by matching
    endpoints, and return a list of arrays of points.

    if layer is given, it should be an ogr LineString layer
    if segments is given, it should be a list of numpy arrays, where
    each array is [N,2] giving points along a path.

    this version only handles *exact* matches between endpoints
    """

    # a hash that maps x,y values to feature ids
    endpoints = {}
    features = {} # map feature id to a list of points
    # map old feature ids to new ones:
    remapper = {}

    progress_message("Reading features")


    def seg_iter():
        if not layer:
            for fid,seg in enumerate(segments):
                yield fid,seg
        else:
            layer.ResetReading()
            while 1:
                feat = layer.GetNextFeature()
                if not feat:
                    return
                fid = feat.GetFID()
                geo = feat.GetGeometryRef() # should be a linestring
                if geo is None:
                    log.warning("Missing geometry - will skip")
                    continue

                geom=shapely.wkb.loads(geo.ExportToWkb())
                
                if geom.type == 'MultiLineString':
                    geolist=geom.geoms
                else:
                    geolist=[geom]

                for sub_idx,one_geom in enumerate(geolist):
                    if one_geom.type != 'LineString':
                        raise Exception("All (sub)features must be linestrings (fid=%s, %s)"%(fid,one_geom.type))

                    # read the points into a numpy array:
                    points = np.array(one_geom.coords)
                    # use tuples to keep sub-features distinct
                    yield (fid,sub_idx),points
        
    for fid,points in seg_iter():
        features[fid] = points

        start_point = tuple(points[0])
        end_point   = tuple(points[-1])

        if start_point == end_point:
            continue

        if start_point not in endpoints:
            endpoints[start_point] = []
        endpoints[start_point].append(fid)
        if end_point not in endpoints:
            endpoints[end_point] = []
        endpoints[end_point].append(fid)

        remapper[fid] = fid

    # check on how many things match up:
    # almost every point has exactly two features - perfect!

    progress_message("%i possible matched features"%len(endpoints))

    # toss out endpoints that don't have exactly two matches:
    endpoint_list = []
    for k in endpoints:
        if len(endpoints[k]) == 2:
            endpoint_list.append(endpoints[k])

    total_pairs = len(endpoint_list)
    pairs_processed = 0
    
    # iterate over the end points, merging all exact matches:
    for matched_pair in endpoint_list:
        fidA,fidB = [remapper[fid] for fid in matched_pair]

        if fidA==fidB:
            continue

        pairs_processed += 1
        if pairs_processed%1000==0:
            progress_message("Merge lines exact",pairs_processed,total_pairs)

        # 
        coordsA = features[fidA]
        coordsB = features[fidB]

        # figure out how they go together, and figure out what point needs
        # to be redirected from featB:
        # also be sure to skip the repeated poin
        if all(coordsA[0]==coordsB[0]):
            coordsC = np.concatenate((coordsA[::-1],coordsB[1:]))
            redirect = coordsB[-1]
        elif all(coordsA[-1]==coordsB[0]):
            coordsC = np.concatenate((coordsA,coordsB[1:]))
            redirect = coordsB[-1]
        elif all(coordsA[0]==coordsB[-1]):
            coordsC = np.concatenate((coordsB,coordsA[1:]))
            redirect = coordsB[0]
        elif all(coordsA[-1]==coordsB[-1]):
            coordsC = np.concatenate((coordsA[:-1],coordsB[::-1]))
            redirect = coordsB[0]
        else:
            log.error( "No match:" )
            log.error( "%s %s"%( fidA,fidB) )
            log.error( "%s"%( coordsA[0])   )
            log.error( "%s"%( coordsA[-1])  )
            log.error( "%s"%( coordsB[0] )  )
            log.error( "%s"%( coordsB[-1] ) )
            raise Exception("hash says we have a match, but no good match found")

        # replace the geometry of featA
        features[fidA] = coordsC
        for k in remapper.keys():
            if remapper[k] == fidB:
                remapper[k] = fidA

        # and delete featB
        del features[fidB]
    progress_message("merge completed")
    # cast to list for python 3
    return list(features.values())

def tolerant_merge_lines(features,tolerance):
    """ expects features to be formatted like the output of merge_lines,
    i.e. a list of numpy arrays
    """

    NO_MATCH   =0
    FIRST_FIRST=1
    FIRST_LAST =2
    LAST_FIRST =3
    LAST_LAST  =4
    INIT_MATCH =5 # dummy value to kick-start the loop

    closed_already = [ all(feat[0]==feat[-1]) for feat in features]

    def check_match(pntsA,pntsB):
        if norm(pntsA[0]-pntsB[0]) <= tolerance:
            return FIRST_FIRST
        elif norm(pntsA[0]-pntsB[-1]) <= tolerance:
            return FIRST_LAST
        elif norm(pntsA[-1]-pntsB[0]) <= tolerance:
            return LAST_FIRST
        elif norm(pntsA[-1]-pntsB[-1]) <= tolerance:
            return LAST_LAST
        else:
            return NO_MATCH

    # how to do the matching:
    #  nested loops?  match the i-th feature against each jth other feature
    #    if they match, merge j onto i, set j-th to None, and start scanning
    #    again to match more features against i-th
    for i in range(len(features)):
        if features[i] is None:
            continue
        if closed_already[i]:
            continue

        progress_message("Merge lines tolerant",i,len(features))

        # once we've tried to match the i-th feature against everybody
        # after i, there's no reason to look at it again, so the inner
        # loop starts at i+1

        match = INIT_MATCH
        while match:
            match = NO_MATCH
            # check each subsequent feature
            for j in range(i+1,len(features)):
                if features[j] is None:
                    continue # check next j-th
                if closed_already[j]:
                    continue

                match = check_match(features[i],
                                    features[j])

                # When merging, drop one point from the merge location
                # otherwise if they are very close we'll end up with numerical issues
                # related to repeated points.
                if match==FIRST_FIRST:
                    features[i] = np.concatenate((features[i][::-1],features[j][1:]))
                elif match==FIRST_LAST:
                    features[i] = np.concatenate((features[j],features[i][1:]))
                elif match==LAST_FIRST:
                    features[i] = np.concatenate((features[i],features[j][1:]))
                elif match==LAST_LAST:
                    features[i] = np.concatenate((features[i][:-1],features[j][::-1]))

                # if we get a match, we just merged the features and can
                # remove the j-th feature.
                if match != NO_MATCH:
                    features[j] = None
                    # at this point, though, our i-th feature has changed and
                    # requires that we re-process matches against it, so
                    # with match set non-zero, escape out of the j-loop
                    # and the while loop will restart the j-loop.
                    break
            # if we fall out of this loop and didn't have a match, we're done
            # with the i-th feature, so let the next iteration of the i-loop
            # run

    # this just eliminates None elements
    features = [f for f in features if f is not None]

    # Make an additional loop to see if there are rings that we need to close:
    for feat in features:
        delta = norm(feat[0] - feat[-1])
        if delta > 0.0 and delta <= tolerance:
            log.info("tolerant_merge: joining a loop - dist = %f"%delta)
            feat[-1] = feat[0]

    return features

# how many of the features are closed, and return the one that isn't
# since it will define the exterior ring in the output
# if all the rings are closed, return the ring with the greatest area
#  and closed_p=True

def clean_degenerate_rings(point_lists,degen_shpname='degenerate_rings.shp'):
    """ Given a list of lists of points - filter out point lists
    which represent degenerate rings, writing the invalid rings
    to a shapefile degen_shpname, and returning a list of only
    the valid rings.  Unclosed linestrings are passed through.

    set degen_shpname to None to disable that output.
    """
    degen_lines = []
    valid_lists = []
    for i in range(len(point_lists)):
        point_list = point_lists[i]
        if all(point_list[0]!=point_list[-1]):
            valid_lists.append(point_list)
        else: # closed - check it's area
            poly = shapely.geometry.Polygon(point_list)
            try:
                a=poly.area
                valid_lists.append(point_list)
            except ValueError:
                log.error( "degenerate feature: %s"%i )
                degen_line = shapely.geometry.LineString(point_list)
                degen_lines.append(degen_line)

    if degen_shpname is not None and len(degen_lines)>0:
        wkb2shp.wkb2shp(degen_shpname,degen_lines,srs_text='EPSG:26910',overwrite=True)

    return valid_lists


def find_exterior_ring(point_lists):
    open_strings = []
    max_area = 0
    max_area_id = None

    for i in range(len(point_lists)):
        point_list = point_lists[i]
        if all(point_list[0]!=point_list[-1]):
            open_strings.append(i)
        else: # closed - check it's area
            poly = shapely.geometry.Polygon(point_list)
            a = poly.area
            if a > max_area:
                max_area = a
                max_area_id = i

    if len(open_strings) > 1:
        log.error( "Wanted exactly 0 or 1 open strings, got %i"%len(open_strings) )
        for i in open_strings:
            log.error("  Open string: %s"%( point_lists[i] ) )
        raise Exception("Can't figure out who is the exterior ring")

    if len(open_strings) == 1:
        log.error("Choosing exterior ring based on it being the only open ring")
        log.error( "Endpoints: %s"%( point_lists[open_strings[0]][0],point_lists[open_strings[0]][-1] ) )
        return open_strings[0],False
    else:
        log.info( "No open linestrings, resorting to choosing exterior ring by area" )
        print("Selected exterior ring with area %.0f"%max_area)
        return max_area_id,True

def arc_to_close_line(points,n_arc_points=40):
    """ Given a list of points, return an arc that closes the linestring,
    and faces away from the centroid of the points
    """

    # Find the centroid of the original points.
    geo = shapely.geometry.Polygon(points)
    centroid = np.array(geo.centroid)

    # for now, assume a 180 degree arc:
    arc_center = (points[0]+points[-1])/2.0

    # the arc will get appended to the linestring, so find the initial vector from
    # the last point in the linestring:
    start_vector = points[-1] - arc_center

    arc_center_to_centroid = centroid - arc_center

    # if we are going CCW, then
    if cross(arc_center_to_centroid, start_vector) > 0:
        arc_dir = +1
    else:
        arc_dir = -1

    # how many steps in the arc? ultimately could be tied to the desired spatial
    # resolution, or at least that great (since it will get filtered down to the
    # desired resolution, but not filtered up)
    angles = np.linspace(0,arc_dir*np.pi,n_arc_points)
    arc_points = np.zeros((n_arc_points,2),np.float64)

    # rotate the start vector
    for i in range(n_arc_points):
        angle = angles[i]
        xx = np.cos(angle)
        xy = -np.sin(angle)
        yx = np.sin(angle)
        yy = np.cos(angle)

        new_x = start_vector[0]*xx + start_vector[1]*xy
        new_y = start_vector[0]*yx + start_vector[1]*yy

        arc_points[i] = arc_center + [new_x,new_y]

    return arc_points


def lines_to_polygons_slow(new_features,close_arc=False,single_feature=True,force_orientation=True):
    """
    single_feature: False is not yet implemented!
    returns a list of Polygons and a list of features which were not part of a polygon
    force_orientation: ensure that interior rings have negative signed area
    """
    assert single_feature

    ### Remove non-polygons - still not smart enough to handle duplicate points
    new_features = [f for f in new_features if len(f) > 2]

    ### Find exterior ring
    log.info("Finding exterior ring from %d linestrings"%len(new_features))

    new_features = clean_degenerate_rings(new_features)

    exterior_id,closed_p = find_exterior_ring(new_features)

    if close_arc and not closed_p:
        ### Add an arc to close the exterior ring:
        # really this out to test whether or not it's necessary
        # to add the arc.
        closing_arc = arc_to_close_line(new_features[exterior_id])

        new_features[exterior_id] = np.concatenate((new_features[exterior_id],closing_arc))

    exterior = new_features[exterior_id]
    interiors = [new_features[i] for i in range(len(new_features)) if i!=exterior_id]

    ### Remove features that are not contained by the exterior ring:
    ext_poly = shapely.geometry.Polygon(exterior)

    if prepared is not None:
        prep_ext_poly = prepared.prep(ext_poly)
    else:
        prep_ext_poly = None

    new_interiors = []
    extras = [] # features which were not inside exterior, but otherwise valid
    for i in range(len(interiors)):
        interior = interiors[i]
        if i%300==0:
            progress_message("Checking for orphan interior features",i,len(interiors))
        if force_orientation and (utils.signed_area(interior) > 0):
            interior=interior[::-1]
        int_poly = shapely.geometry.Polygon(interior)

        # spaghetti logic
        if prep_ext_poly is None or prep_ext_poly.contains(int_poly):
            if prep_ext_poly and trust_prepared:
                new_interiors.append(interior)
            else:                    
                if ext_poly.contains(int_poly):
                    new_interiors.append(interior)
                else:
                    if prep_ext_poly is not None:
                        log.warning( "A feature got through the prepared query, but the real query says it's outside the exterior")
                    else:
                        log.debug("Removing a feature that was outside the exterior ring" )
                    extras.append(interior)
        else:
            log.debug("Removing a feature that the fast query said was outside the exterior ring")
            extras.append(interior)

    # create a single polygon feature from all of the rings:
    poly_geom = shapely.geometry.Polygon(exterior,new_interiors)
    return [poly_geom],extras

# updated version, hopefully faster in the usual case of no open loops, but
# multiple polygons
def lines_to_polygons(new_features,close_arc=False,single_feature=True,force_orientation=True,
                      return_open=False,min_area=0.0):
    """
    returns a list of Polygons and a list of features which were not part of a polygon
    force_orientation: ensure that interior rings have negative signed area
    return_open: if True, allow open linestrings, but they will be returned in a 3rd item.
    min_area: prune polygons with area below this threshold
    """
    ### Remove non-polygons - still not smart enough to handle duplicate points
    new_features = [f for f in new_features if len(f) > 2]
    new_features = clean_degenerate_rings(new_features)

    open_strings=[]
    simple_polys=[]
    for i,point_list in enumerate(new_features):
        if np.any(point_list[0]!=point_list[-1]):
            open_strings.append(i)
        else: # closed - check it's area
            simple_polys.append(shapely.geometry.Polygon(point_list))

    log.info("%d open strings, %d simple polygons"%(len(open_strings),
                                                    len(simple_polys)))

    if len(open_strings):
        if not return_open:
            log.error("New version of lines_to_polygons is faster but intolerant.  Cannot handle ")
            log.error("%d open strings"%len(open_strings))
            log.error("First open string starts at %s"%(new_features[open_strings[0]][0]))
            raise Exception("No longer can handle open line strings")

    polys=[] # output polygons

    areas=np.array([p.area for p in simple_polys])

    if min_area>0:
        select=areas>=min_area
        simple_polys=[p for p,a in zip(simple_polys,areas) if a>=min_area]
        areas=areas[select]
    
    # sort big to small
    ordering=np.argsort(-areas)
    simple_polys=[simple_polys[i] for i in ordering]
    areas=areas[ordering]

    log.info("Building index")
    # Because the index only hands back the poly, not an index.
    for i,p in enumerate(simple_polys):
        p.join_id=i

    index=STRtree(simple_polys)
    log.info("done building index")

    poly_geoms=[] # accumulate results

    assigned_p=[False]*len(simple_polys)
    unassigned_idxs=list(range(len(simple_polys)))

    while len(unassigned_idxs):
        ext_idx=unassigned_idxs.pop(0) # get first/biggest one
        if assigned_p[ext_idx]:
            continue # was included as a sub-feature already

        assigned_p[ext_idx]=True
        ext_poly=simple_polys[ext_idx]

        ### Find exterior ring
        log.info("Examining largest poly left with area=%f, %d potential interiors"%
                 (ext_poly.area,len(unassigned_idxs)))
        prep_ext_poly = prepare_geometry(ext_poly)

        hits=index.query(ext_poly)
        hit_indexes=[p.join_id for p in hits]
        # this keeps us comparing large->small, needed to avoid
        # confusing islands in lake with lakes
        hit_indexes.sort()

        for i in utils.progress(hit_indexes):
            if assigned_p[i]:
                continue
            int_poly=simple_polys[i]
            if prep_ext_poly.contains(int_poly):
                # include that poly as an interior
                ext_poly=shapely.geometry.Polygon(ext_poly.exterior,
                                                  list(ext_poly.interiors)+[int_poly.exterior])
                prep_ext_poly=prepare_geometry(ext_poly)
                assigned_p[i]=True # lazily remove from unassigned_idxs at top of loop

        poly_geoms.append(ext_poly)

        if single_feature:
            break

    extras=[p for p,is_assigned in zip(simple_polys,assigned_p) if not is_assigned]
    if return_open:
        return poly_geoms,extras,open_strings
    else:        
        return poly_geoms,extras


####### Running the actual steps ########

def vector_mag(vectors):
    """
    vectors: xy vectors, shape [...,2]
    return magnitude (L2 norm)
    """
    # equivalent to np.linalg.norm(axis=-1), but older numpy doesn't
    # allow axis keyword
    return np.sqrt(np.sum(vectors**2,axis=-1))


def process_layer(orig_layer,output_name,tolerance=0.0,
                  create_polygons=False,close_arc=False,
                  single_feature=True,
                  remove_duplicates=True):
    """
    remove_duplicates: if true, exactly duplicated nodes along a single path will be removed, i.e.
      the linestring A-B-B-C will become A-B-C.
    single_feature: only save the biggest feature
    """
    orig_srs=None
    orig_srs_text=None

    if isinstance(orig_layer,str):
        ods = ogr.Open(orig_layer)
        orig_layer = ods.GetLayer(0)

    try:
        orig_srs=orig_layer.GetSpatialRef()
        orig_srs_text=orig_srs.ExportToWkt()
    except Exception as exc:
        log.error("Attempted to get spatial ref: %s"%exc)


    ### The actual geometry processing: ###
    ### <processing>
    new_features = merge_lines(orig_layer)

    if remove_duplicates:
        log.info("Checking the merged features for duplicate points" )
        # possibly important here to have the duplicate test more stringent than
        # the tolerant_merge_lines.
        # also have to be careful about a string of points closely spaced - don't
        # want to remove all of them, just enough to keep the minimal spacing above
        # tolerance.
        short_tol = 0.5*tolerance

        for fi in range(len(new_features)):
            pnts = new_features[fi]
            valid = np.ones( len(pnts), np.bool_)
            # go with a slower but safer loop here -
            last_valid=0
            for i in range(1,len(pnts)):
                if vector_mag( pnts[last_valid]-pnts[i] ) < short_tol:
                    if i==len(pnts)-1:
                        # special case to avoid moving the last vertex
                        valid[last_valid] = False
                        last_valid = i
                    else:
                        valid[i] = False
                else:
                    last_valid = i

            # print "Ring %d: # invalid=%d / %d"%(i,sum(~valid),len(new_features[i]))
            new_features[fi] = new_features[fi][valid,:]

    if tolerance > 0.0:
        new_features = tolerant_merge_lines(new_features,tolerance)

    ### </processing>

    ### <output>
    if create_polygons:
        if single_feature:
            geoms,extras = lines_to_polygons(new_features,close_arc=close_arc,single_feature=True)
        else:
            # geoms=[]
            # unmatched=new_features
            # while len(unmatched):
            #     one_poly,unmatched=lines_to_polygons(unmatched,close_arc=close_arc,single_feature=True)
            #     geoms.append(one_poly[0])
            geoms,extras=lines_to_polygons(new_features,close_arc=close_arc,single_feature=False)
    else:
        # Line output
        geoms = [shapely.geometry.LineString(pnts) for pnts in new_features]

    # Write it all out to a shapefile:
    progress_message("Writing output")

    kws={}
    if orig_srs_text is not None:
        kws['srs_text']=orig_srs_text
        
    wkb2shp.wkb2shp(output_name,geoms,
                    overwrite=True,**kws)

    return output_name

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] input.shp output.shp")
    parser.add_option("-p", "--poly",
                      help="create polygons from closed linestrings",
                      action="store_true",
                      dest='create_polygons',default=False)
    parser.add_option("-a", "--arc", dest="close_arc", default=False,
                      action="store_true",
                      help="close the largest open linestring with a circular arc")
    parser.add_option("-t","--tolerance", dest="tolerance", type="float", default=0.0,
                      metavar="DISTANCE",
                      help="Tolerance for joining two endpoints, in geographic units")
    parser.add_option("-m","--multiple", dest="single_feature", default=True,
                      action="store_false",metavar="SINGLE_FEATURE")

    (options, args) = parser.parse_args()
    input_shp,output_shp = args

    ods = ogr.Open(input_shp)
    layer = ods.GetLayer(0)

    process_layer(layer,output_shp,
                  create_polygons=options.create_polygons,close_arc=options.close_arc,
                  tolerance=options.tolerance,single_feature=options.single_feature)

