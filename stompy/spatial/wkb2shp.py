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
from .geom_types import ogr2text,text2ogr
import uuid

import numpy as np

def wkb2shp(shp_name,
            input_wkbs,
            srs_text='EPSG:26910',
            field_gen = lambda f: {},
            fields = None,
            overwrite=False,
            geom_type=None):
    """ shp_name: filename.shp for writing the result
    or 'MEMORY' to return an in-memory ogr layer.

      input_wkbs: shapely geometry objects for each feature.  They must all
                  be the same geometry type (no mixing lines and polygons, etc.)
    Three ways of specifying fields:
       field_gen: a function which will be called once for each feature, with
                  the geometry as its argument, and returns a dict of fields.
       fields: a numpy array with named fields, or
       fields: a dict of field names
    """
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
        field_names=fields.keys()
        N=len(fields[field_names[0]])

        field_values=[None]*N

        for n in range(N):
            row=[fields[fname][n] for fname in field_names] 
            field_values[n]=row
    
    elif fields is not None and isinstance(fields,np.ndarray):
        dt = fields.dtype
        
        # Note that each field may itself have some shape - so we need to enumerate those
        # dimensions, too.
        field_names = []
        for name in dt.names:
            # ndindex iterates over tuples which index successive elements of the field
            for index in np.ndindex( dt[name].shape ):
                name_idx = name + "_".join([str(i) for i in index])
                field_names.append(name_idx)
        
        field_values = []
        for i in range(len(fields)):
            fields_onerow = []
            for name in dt.names:
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
            raise Exception("Cannot have field names longer than 10 characters")
            
    if geom_type is None:
        # find it by querying the features - minor bug - this only 
        # works when shapely geometries were passed in.
        types = np.array( [text2ogr[g.type] for g in geoms] )
        geom_type = int(types.max())
        # print "Chose geometry type to be %s"%ogr2text[geom_type]

    new_layer = new_ds.CreateLayer(shp_name,
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
        elif isinstance(val,float):
            # This is an old bug - seems to work without this in the modern
            # era, and in turn, asscalar does *not* work with list of lists
            # # a numpy array of float32 ends up with
            # # a type here of <type 'numpy.float32'>,
            # # which doesn't match isinstance(...,float)
            # # asscalar helps out with that
            print( "float valued key is %s"%key)
            field_def = ogr.FieldDefn(key,ogr.OFTReal)
            field_def.SetWidth(10)
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
            geom_wkbs = [g.wkb for g in geom.geoms]

        for geom_wkb in geom_wkbs:
            feat_geom = ogr.CreateGeometryFromWkb(geom_wkb)
            feat = ogr.Feature( new_layer.GetLayerDefn() )
            feat.SetGeometryDirectly(feat_geom)
            for i,val in enumerate(feature_fields):
                feat.SetField(str(field_names[i]),casters[i](feature_fields[i]))

            new_layer.CreateFeature(feat)
            feat.Destroy()

    if shp_name!="Memory":
        new_layer.SyncToDisk()
    else:
        return new_ds


# kind of the reverse of the above
def shp2geom(shp_fn,use_wkt=False):
    ods = ogr.Open(shp_fn)
    if ods is None:
        raise ValueError("File '%s' corrupt or not found"%shp_fn)
    layer = ods.GetLayer(0)

    feat = layer.GetNextFeature()

    defn = feat.GetDefnRef()
    fld_count = defn.GetFieldCount()

    fields = []
    
    for i in range(fld_count):
        fdef =defn.GetFieldDefn(i)
        name = fdef.name 
        ogr_type = fdef.GetTypeName()
        if ogr_type == 'String':
            np_type = object
            getter = lambda f,i=i: f.GetFieldAsString(i)
        elif ogr_type =='Integer':
            np_type = np.int32
            getter = lambda f,i=i: f.GetFieldAsInteger(i)
        else:
            np_type = np.float64
            getter = lambda f,i=i: f.GetFieldAsDouble(i)
        fields.append( (i,name,np_type,getter) )

    # And one for the geometry
    def rdr(f):
        try:
            if use_wkt:
                geo=wkt.loads( f.GetGeometryRef().ExportToWkt() )
            else:
                geo=wkb.loads( f.GetGeometryRef().ExportToWkb() )
        except:
            geo=None
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
        except shapely.geos.ReadingError:
            print("Failed to load geometry for feature")
            continue
        field_array = tuple(field_vals)
        recs.append(field_array)

    recs = np.array( recs, dtype=layer_dtype)
    return recs

    

    
