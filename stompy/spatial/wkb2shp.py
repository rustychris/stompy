"""
Functions for reading and writing shapefiles.

These are simple wrappers around OGR and Shapely
libraries.

"""
from __future__ import print_function
import six

try:
    from osgeo import ogr,osr
except ImportError:
    import ogr,osr

import glob,os,re

from shapely import wkb,wkt
from shapely.geometry import Polygon,LineString,Point,MultiPolygon,MultiLineString,MultiPoint
import shapely.geos
from shapely import ops # for transform()
from .geom_types import ogr2text,text2ogr
from . import proj_utils
import uuid

import numpy as np

def wkb2shp(shp_name,
            input_wkbs,
            srs_text='EPSG:26910',
            field_gen = lambda f: {},
            fields = None,
            overwrite=False,
            geom_type=None,
            driver=None,
            layer_name=None,
            unpack_multi=True):
    """
    Save data to a shapefile.

    shp_name: filename.shp for writing the result
    or 'MEMORY' to return an in-memory ogr layer.

      input_wkbs: list of shapely geometry objects for each feature.  They must all
                  be the same geometry type (no mixing lines and polygons, etc.)

    There are three ways of specifying fields:
       field_gen: a function which will be called once for each feature, with
                  the geometry as its argument, and returns a dict of fields.
       fields: a numpy structure array with named fields, or
       fields: a dict of field names, with each dictionary entry holding a sequence
         of field values.

    srs_text: sets the projection information when writing the shapefile.  Expects
    a string, for example 'EPSG:3095'  or 'WGS84'.

    driver: Directly specify an alternative driver, such as GPKG. if None, assumed
      shapefile, unless shp_name is 'memory' in which case create an in-Memory
      layer.
      for GPKG, the optional layer_name argument can be used to name the layer
      which would default to the basename of the shp_name otherwise
    """
    if layer_name is None:
        layer_name=shp_name
        
    if driver=='GPKG':
        drv = ogr.GetDriverByName('GPKG')
        new_ds = drv.CreateDataSource(shp_name)
    if shp_name.lower()=='memory':
        drv = ogr.GetDriverByName('Memory')
        new_ds = drv.CreateDataSource("mem_" + uuid.uuid1().hex)
    else:
        if os.path.exists(shp_name):
            if shp_name[-4:] == '.shp':
                if overwrite:
                    # remove any matching files:
                    print("Removing the old to make way for the new")
                    os.unlink(shp_name)
                else:
                    raise Exception("Shapefile exists, but overwrite is False")

        # open the output shapefile:
        drv = ogr.GetDriverByName('ESRI Shapefile')
        new_ds = drv.CreateDataSource(shp_name)

    if isinstance(srs_text,osr.SpatialReference):
        srs = srs_text
    else:
        srs = osr.SpatialReference()
        srs.SetFromUserInput(srs_text)

    ## Depending on the inputs, populate
    #  geoms - a list or array of shapely geometries
    #  field_names - ordered list of field names
    #  field_values - list of lists of field values

    # Case 1: all the data is packed into a numpy struct array
    geoms = input_wkbs

    if fields is not None and type(fields) == list: # sub case - fields is a list of dicts
        field_iter = iter(fields)
        field_gen = lambda x: six.next(field_iter)
        fields = None

    if fields is not None and isinstance(fields,dict):
        field_names=list(fields.keys())
        N=len(fields[field_names[0]])

        field_values=[None]*N

        for n in range(N):
            row=[fields[fname][n] for fname in field_names] 
            field_values[n]=row

    elif fields is not None and isinstance(fields,np.ndarray):
        dt = fields.dtype

        # Special case to round-trip shp2geom->wkb2shp
        # Not working yet
        if input_wkbs is None:
            input_wkbs=fields['geom']

        # Note that each field may itself have some shape - so we need to enumerate those
        # dimensions, too.
        field_names = []
        for name in dt.names:
            if name=='geom': continue
            # ndindex iterates over tuples which index successive elements of the field
            for index in np.ndindex( dt[name].shape ):
                name_idx = name + "_".join([str(i) for i in index])
                field_names.append(name_idx)

        field_values = []
        for i in range(len(fields)):
            fields_onerow = []
            for name in dt.names:
                if name=='geom': continue
                for index in np.ndindex( dt[name].shape ):
                    if index != ():
                        fields_onerow.append( fields[i][name][index] )
                    else:
                        fields_onerow.append( fields[i][name] )
            field_values.append( fields_onerow )
    else:
        # Case 2: geometries and a field generator are specified
        field_dicts = []
        for g in geoms:
            field_dicts.append( field_gen(g) )
        # py3k: .keys() is a dict_keys object, not 100% compatible with a list.
        field_names = list(field_dicts[0].keys())
        field_values = []
        for i in range(len(input_wkbs)):
            field_values.append( [field_dicts[i][k] for k in field_names] )

    for n in field_names:
        if len(n)>10:
            raise Exception("Cannot have field names longer than 10 characters: %s"%
                            n)

    if geom_type is None:
        # find it by querying the features - minor bug - this only 
        # works when shapely geometries were passed in.
        types = np.array( [text2ogr[g.geom_type] for g in geoms] )
        geom_type = int(types.max())
        # print("Chose geometry type to be %s"%ogr2text[geom_type])

    new_layer = new_ds.CreateLayer(layer_name,
                                   srs=srs,
                                   geom_type=geom_type)

    # setup the feature definition:

    # create fields based on the field key/value pairs
    # return for the first wkb file
    casters = []
    for field_i,key in enumerate(field_names):
        val = field_values[0][field_i]
        if type(val) == int or isinstance(val,np.integer):
            field_def = ogr.FieldDefn(key,ogr.OFTInteger)
            casters.append( int )
        elif isinstance(val,np.floating):
            # above: use np.float, as it seems to be more compatible with
            # 32-bit and 64-bit floats.

            # This is an old bug - seems to work without this in the modern
            # era, and in turn, asscalar does *not* work with list of lists
            # # a numpy array of float32 ends up with
            # # a type here of <type 'numpy.float32'>,
            # # which doesn't match isinstance(...,float)
            # # asscalar helps out with that
            # 2018-07: np.floating is maybe the proper solution.
            print( "float valued key is %s"%key)
            field_def = ogr.FieldDefn(key,ogr.OFTReal)
            field_def.SetWidth(64)
            field_def.SetPrecision(10)
            casters.append( float )
        else:
            field_def = ogr.FieldDefn(key,ogr.OFTString)
            casters.append( str )
        # print "Field name is %s"%key
        new_layer.CreateField( field_def )
    
    for i,geom in enumerate(geoms):
        feature_fields = field_values[i]        

        # print "Processing: ",feature_fields

        if type(geom) == str:
            fp = open(wkb_file,'rb')
            geom_wkbs = [fp.read()]
            fp.close()
        elif type(geom) in (Polygon,LineString,Point):
            geom_wkbs = [geom.wkb]
        elif type(geom) in (MultiPolygon,MultiLineString,MultiPoint):
            if unpack_multi:
                geom_wkbs = [g.wkb for g in geom.geoms]
            else:
                geom_wkbs = [geom.wkb]
    
        for geom_wkb in geom_wkbs:
            feat_geom = ogr.CreateGeometryFromWkb(geom_wkb)
            feat = ogr.Feature( new_layer.GetLayerDefn() )
            feat.SetGeometryDirectly(feat_geom)
            for i,val in enumerate(feature_fields):
                feat.SetField(str(field_names[i]),casters[i](feature_fields[i]))

            new_layer.CreateFeature(feat)
            feat.Destroy()

    if shp_name.lower()!="memory":
        new_layer.SyncToDisk()
    else:
        return new_ds


# kind of the reverse of the above
def shp2geom(shp_fn,use_wkt=False,target_srs=None,
             source_srs=None,return_srs=False,
             query=None,layer_patt=None,
             fold_to_lower=False):
    """
    Read a shapefile into memory as a numpy array.
    Data is returned as a record array, with geometry as a shapely
    geometry object in the 'geom' field.

    target_srs: input suitable for osgeo.osr.SetFromUserInput(), or an
    existing osr.SpatialReference, to specify
    a projection to which the data should be projected.  If this is specified
    but the shapefile does not specify a projection, and source_srs is not given,
    then an exception is raised.  source_srs will override the projection in 
    the shapefile if specified.

    fold_to_lower: fold field names to lower case.

    return_srs: return a tuple, second item being the text representation of the project, or
     None if no projection information was found.
    """
    ods = ogr.Open(shp_fn)
    if ods is None:
        raise ValueError("File '%s' corrupt or not found"%shp_fn)
    if layer_patt is not None:
        import re
        names=[]
        for layer_idx in range(ods.GetLayerCount()):
            layer=ods.GetLayerByIndex(layer_idx)
            names.append(layer.GetName())
            if re.match(layer_patt,names[-1]):
                break
        else:
            print("Layers were: ")
            for l in names:
                print("  "+l)
            raise Exception("Pattern %s matched no layers"%layer_patt)
    else:
        layer = ods.GetLayer(0)

    if query is not None:
        layer.SetAttributeFilter(query)

    if target_srs is not None: # potentially transform on the fly
        if source_srs is None:
            source_srs=layer.GetSpatialRef()
        if source_srs is None:
            raise Exception("Reprojection requested, but no source reference available")

        mapper=proj_utils.mapper(source_srs,target_srs)
        # have to massage it a bit to suit shapely's calling convention
        def xform(x,y,z=None): # x,y,z may be scalar or array
            # ugly code... annoying code...
            if z is None:
                xy=np.moveaxis( np.array([x,y]), 0, -1 )
                xyp=mapper(xy)
                return xyp[...,0], xyp[...,1]
            else:
                xyz=np.moveaxis( np.array([x,y,z]), 0, -1 )
                xyzp=mapper(xyz)
                return xyzp[...,0],xyzp[...,1],xyzp[...,2]
        def geom_xform(g):
            return ops.transform(xform,g)
    else:
        target_srs=layer.GetSpatialRef()
        def geom_xform(g):
            return g

    feat = layer.GetNextFeature()

    defn = feat.GetDefnRef()
    fld_count = defn.GetFieldCount()

    fields = []
    
    for i in range(fld_count):
        fdef =defn.GetFieldDefn(i)
        name = fdef.name
        if fold_to_lower:
            name=name.lower()
        ogr_type = fdef.GetTypeName()
        if ogr_type == 'String':
            np_type = object
            getter = lambda f,i=i: f.GetFieldAsString(i)
        elif ogr_type =='Integer':
            np_type = np.int32
            getter = lambda f,i=i: f.GetFieldAsInteger(i)
        elif ogr_type == 'Date':
            np_type = '<M8[s]' # np.datetime64
            # this handles null and real dates, whereas GetFieldAsDateTime
            # would take more finagling to deal with nulls.
            getter = lambda f,i=i: np.datetime64(f.GetFieldAsString(i).replace('/','-'))
        else:
            np_type = np.float64
            getter = lambda f,i=i: f.GetFieldAsDouble(i)
        fields.append( (i,name,np_type,getter) )

    # And one for the geometry
    def rdr(f):
        # The try..except block is from olden days when OGR was not stable
        # to weird geometries.

        geo_ref=f.GetGeometryRef()
        if geo_ref is None:
            # this is possible, for example, in QGIS delete all nodes of a
            # line, but don't delete the actual feature.
            return None
        #try:
        if use_wkt:
            geo=wkt.loads( f.GetGeometryRef().ExportToWkt() )
        else:
            data=f.GetGeometryRef().ExportToWkb()
            # API changed somewhere to return mutable bytearray, but
            # wkb only accepts bytes.
            if isinstance(data,bytearray):
                data=bytes(data)
            geo=wkb.loads( data )
        geo=geom_xform(geo)
        #except:
        #    geo=None
        return geo
    fields.append( (None,'geom',object,rdr) )
    
    layer_dtype = [ (name,np_type) for i,name,np_type,getter in fields]

    recs = []

    layer.ResetReading()

    while 1:
        feat = layer.GetNextFeature()
        if feat is None:
            break
        try:
            field_vals = [getter(feat) for i,name,np_type,getter in fields]
        except shapely.geos.WKBReadingError as exc:
            # Used to just be shapely.geos.ReadingError
            print("Failed to load geometry for feature")
            print(exc)
            continue
        field_array = tuple(field_vals)
        recs.append(field_array)

    recs = np.array( recs, dtype=layer_dtype)

    if return_srs:
        return recs, target_srs.ExportToWkt()
    else:
        return recs

    

    
