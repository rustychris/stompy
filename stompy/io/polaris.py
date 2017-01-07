from __future__ import print_function

import urllib,re,time,datetime,pytz
from numpy import *
from numpy import ma # masked arrays

import datetime

try:
    from osgeo import osr
except ImportError:
    import osr

from cache_handler import urlopen

from safe_pylab import *

from matplotlib import ticker

import field
import transect

def ll2utm(lonlat):
    if ll2utm.xform is None: # Lazy loading of xform
        nad83 = osr.SpatialReference()
        nad83.SetFromUserInput('NAD83')
        utmz10 = osr.SpatialReference()
        utmz10.SetFromUserInput('EPSG:26910')
        ll2utm.xform = osr.CoordinateTransformation(nad83,utmz10)

    utm = zeros_like(lonlat)

    for i in range(len(lonlat)):
        utm[i] = ll2utm.xform.TransformPoint(*lonlat[i])[:2]

    return utm
ll2utm.xform = None

cruise_jdates = array([2004252, 2004258, 2004308, 2004349, 2005341,
                       2005326, 2005285, 2005269, 2005250, 2005298,
                       2005312, 2005241, 2005172, 2005081, 2005123,
                       2005102, 2005054, 2005108, 2005033, 2005011,
                       2006045, 2006052, 2006010, 2005130, 2005077,
                       2006066, 2006080, 2006074, 2006095, 2006111,
                       1995114, 2006129, 2006255, 2006272, 2006243,
                       2006227, 2006214, 2006192, 2006290, 2006318,
                       2006346, 2007009, 2007040, 2007037, 
                       # 2005221, # oddly missing from database, but
                       # the cruise happened.  ??
                       2007087, 2007100, 2007093, 2007071, 2007065,
                       2007113, 2007172, 2007204, 2007234, 2007254,
                       2007264, 2007292, 2007318, 2007296, 2007345,
                       2008010, 2008032, 2008043, 2008060, 2008091,
                       2008071, 2008105, 2008119, 2008127, 2008150,
                       2008169, 2008192, 2008197, 2008225, 2008232,
                       2008253, 2008260, 2008281, 2008289, 2008310,
                       2008323, 2008351, 2009013, 2009035, 2009041,
                       2009048, 2009064, 2009079, 2009071, 2009097,
                       2009106, 2009125, 2009139, 2009167, 2009174,
                       2009196, 2009202, 2009225, 2009239, 2009265,
                       2009271, 2009302, 2009322, 2009337, 2010005,
                       2010026, 2010040, 2010054, 2010068, 2010083,
                       2010089, 2010097, 2010103, 2010113, 2010127,
                       2010140, 2010158, 2010166, 2010187, 2010194,
                       2010215, 2010229, 2010244, 2010257, 2010274,
                       2010288, 2010299, 2010319, 2010350,
                       2011011, 2011040, 2011070, 2011088, 2011098,
                       2011102, 2011118, 2011131,
                       2012090, 2012102, 2012111,
                       2012118, 2012136, 2012144, 2012164, 2012071,
                       2012193, 2012199, 2012220, 2012240, 2012251,
                       2012255, 2012277, 2012284, 2012311, 2012339
                       ])


cruise_jdates.sort()



def day_of_year(t):
    return int(1 + date2num(t) - date2num( datetime.datetime(t.year,1,1)))

    
def cruise_jdates_in_range(start_date,end_date):
    """ Return a list of julian dates for Polaris cruises that
    happened in the given period.  Expects python datetime instances.
    """
    start_jdate = start_date.year * 1000 + day_of_year(start_date)
    end_jdate = end_date.year * 1000 + day_of_year(end_date)

    if start_jdate > cruise_jdates[-1] or end_jdate < cruise_jdates[0]:
        # raise Exception("The requested dates fall beyond the range of cruises polaris.py knows about")
        return []

    start_i = searchsorted(cruise_jdates,start_jdate)
    end_i = searchsorted(cruise_jdates,end_jdate)
    
    return cruise_jdates[start_i:end_i]

def cruise_nearest_jdate(target):
    right_i = searchsorted(cruise_jdates,target)
    left_i = right_i - 1

    def julian_to_date(j):
        year = int(j/1000)
        doy = int(j - year)
        return datetime.date(year,1,1) + datetime.timedelta(doy - 1) # pretty sure that's one-based.

    target_d = julian_to_date(target)
    if left_i < 0:
        left_i = right_i
    if right_i >= len(cruise_jdates):
        right_i -= 1
        
    left_d = (julian_to_date(cruise_jdates[left_i]) - target_d).days
    right_d = (julian_to_date(cruise_jdates[right_i]) - target_d).days

    if abs(left_d) < abs(right_d):
        return cruise_jdates[left_i]
    else:
        return cruise_jdates[right_i]


class PolarisCruise(object):
    form_url = "http://sfbay.wr.usgs.gov/cgi-bin/sfbay/dataquery/query16.pl"

    def query_values(self,juliandate,year,month):
        qvals =[('col','realdate'),  # mm/dd/yyyy
                ('col','stat'),      # Station number
                ('col','dist'),      # distance from station 36
                ('col','depth'),     # depth
                ('col','salin'),     # salinity
                ('col','temp'),      # temperature
                ('col','time')]      # time HHMM, 24 hour
        if year is not None and month is not None:
            # and select box:
            qvals += [('p11','on'),
                      ('type1','month'),
                      ('comp1','eq'),
                      ('value1',str(month)), # my choice...
                      ('p12','on'),
                      ('conj2','AND'),
                      ('p21','on'),
                      ('type2','year'),
                      ('comp2','eq'),
                      ('value2',str(year)),
                      ('p22','on'),
                      ('conj3','AND'),
                      ('p31',''),
                      ('type3','---'),
                      ('comp3','gt'),
                      ('value3',''),
                      ('p32','')]
        elif juliandate is not None:
            # and select box:
            qvals += [('p11','on'),
                      ('type1','jdate'),
                      ('comp1','eq'),
                      ('value1',str(juliandate)), # my choice...
                      ('p12','on'),
                      ('conj2','AND'),
                      ('p21',''),
                      ('type2','---'),
                      ('comp2','eq'),
                      ('value2',''),
                      ('p22',''),
                      ('conj3','AND'),
                      ('p31',''),
                      ('type3','---'),
                      ('comp3','gt'),
                      ('value3',''),
                      ('p32','')]
            
        qvals += [('sort1','---'),
                  ('asc1','on'),
                  ('sort2','---'),
                  ('asc2','on'),
                  ('sort3','---'),
                  ('asc3','on'),
                  ('out','comma'),
                  ('parm','on'),
                  ('minrow','0'),
                  ('maxrow','1000'), # Come back and change this
                  ('ftype','expert'),
                  # added after looking through the actual data submitted:
                  ('dstart','1990')   
                  ]
        return qvals
    # data below from http://sfbay.wr.usgs.gov/access/wqdata/overview/wherewhen/where.html
    # station_no  name                     lat_n          lon_w          depth_mllw_m
    station_data = [
        (657        , 'Rio Vista'               , 38+( 8.9)/60.0      , 121+( 41.3)/60.0    ,  10.1  ),
        (655        , 'N. of Three Mile Slough' , 38+( 7.3)/60.0      , 121+( 42.1)/60.0    ,  10.1  ),
        (653        , 'Mid-Decker Island'       , 38+( 6.3)/60.0      , 121+( 43.2)/60.0    ,  10.1  ),
        (649        , 'Sacramento River'        , 38+( 3.7)/60.0      , 121+( 48.0)/60.0    ,  10.1  ),
        (2          , 'Chain Island'            , 38+( 3.8)/60.0      , 121+( 51.3)/60.0    ,  11.3  ),
        (3          , 'Pittsburg'               , 38+( 3.0)/60.0      , 121+( 52.7)/60.0    ,  11.3  ),
        (4          , 'Simmons Point'           , 38+( 2.9)/60.0      , 121+( 56.1)/60.0    ,  11.6  ),
        (5          , 'Middle Ground'           , 38+( 3.6)/60.0      , 121+( 58.8)/60.0    ,   9.8  ),
        (6          , 'Roe Island'              , 38+( 3.9)/60.0      , 122+( 2.1)/60.0     ,  10.1  ),
        (7          , 'Avon Pier'               , 38+( 2.9)/60.0      , 122+( 5.8)/60.0     ,   11.6 ),
        (411        , 'Garnet Sill'             , 38+( 5.8)/60.0      , 122+( 3.5)/60.0     ,   0    ),    
        (407        , 'Reserve Fleet'           , 38+( 4.3)/60.0     , 122+( 5.6)/60.0     ,   0    ),   
        (405        , 'Reserve Fleet'           , 38+( 2.9)/60.0     , 122+( 7.4)/60.0     ,   0    ),  
        (8          , 'Martinez'                , 38+( 1.8)/60.0      , 122+( 9.1)/60.0     ,   14.3 ),
        (9          , 'Benicia'                 , 38+( 3.0)/60.0      , 122+( 10.4)/60.0    ,   34.4 ),
        (10         , 'Crockett'                , 38+( 3.6)/60.0      , 122+( 12.5)/60.0    ,   17.7 ),
        (11         , 'Mare Island'             , 38+( 3.7)/60.0      , 122+( 15.8)/60.0    ,   15.5 ),
        (12         , 'Pinole Shoal'            , 38+( 3.1)/60.0      , 122+( 18.7)/60.0    ,   8.8  ),
        (12.5       , 'Pinole Point (ns)'       , 38+( 2.4)/60.0      , 122+( 18.9)/60.0    ,   6.7  ),
        (13         , 'N. of Pinole Point'      , 38+( 1.7)/60.0      , 122+( 22.2)/60.0    ,   9.8  ),
        (14         , '"Echo" Buoy'             , 38+( 0.4)/60.0      , 122+( 24.3)/60.0    ,   13.1 ),
        (15         , 'Point San Pablo'         , 37+( 58.5)/60.0     , 122+( 26.2)/60.0    ,   22.9 ),
        (16         , '"Charlie" Buoy'          , 37+( 54.9)/60.0     , 122+( 26.8)/60.0    ,   43.0 ),
        (17         , 'Raccoon Strait'          , 37+( 52.9)/60.0     , 122+( 25.6)/60.0    ,   32.0 ),
        (18         , 'Point Blunt'             , 37+( 50.8)/60.0     , 122+( 25.3)/60.0    ,   43.0 ),
        (19         , 'Golden Gate'             , 37+( 49.1)/60.0     , 122+( 28.3)/60.0    ,   91.0 ),
        (20         , 'Blossom Rock'            , 37+( 49.2)/60.0     , 122+( 23.6)/60.0    ,   18.2 ),
        (21         , 'Bay Bridge'              , 37+( 47.3)/60.0     , 122+( 21.5)/60.0    ,   17.4 ),
        (22         , 'Potrero Point'           , 37+( 45.9)/60.0     , 122+( 21.5)/60.0    ,   18.0 ),
        (23         , 'Hunters Point'           , 37+( 43.7)/60.0     , 122+( 20.2)/60.0    ,   20.1 ),
        (24         , 'Candlestick Point'       , 37+( 41.9)/60.0     , 122+( 20.3)/60.0    ,   11.0 ),
        (25         , 'Oyster Point'            , 37+( 40.2)/60.0     , 122+( 19.5)/60.0    ,   8.8  ),
        (26         , 'San Bruno Shoal'         , 37+( 38.1)/60.0     , 122+( 18.8)/60.0    ,   9.8  ),
        (27         , 'San Francisco Airport'   , 37+( 37.1)/60.0     , 122+( 17.5)/60.0    ,   13.0 ),
        (28         , 'N. of San Mateo Bridge'  , 37+( 36.1)/60.0     , 122+( 16.2)/60.0    ,   16.2 ),
        (28.5       , 'Geo Probe (ns)'          , 37+( 35.8)/60.0     , 122+( 14.1)/60.0    ,   15.7 ),
        (29         , 'S. of San Mateo Bridge'  , 37+( 34.8)/60.0     , 122+( 14.7)/60.0    ,   14.6 ),
        (29.5       , 'Steinberger Slough (ns)' , 37+( 34.1)/60.0     , 122+( 13.1)/60.0    ,   14.6 ),   
        (30         , 'Redwood Creek'           , 37+( 33.3)/60.0     , 122+( 11.4)/60.0    ,   12.8 ),
        (31         , 'Coyote Hills'            , 37+( 31.7)/60.0     , 122+( 9.5)/60.0     ,   13.7 ),
        (32         , 'Ravenswood Point'        , 37+( 31.1)/60.0     , 122+( 8.0)/60.0     ,   12.8 ),
        (33         , 'Dumbarton Bridge'        , 37+( 30.5)/60.0     , 122+( 7.3)/60.0     ,   11.6 ),
        (34         , 'Newark Slough'           , 37+( 29.7)/60.0     , 122+( 5.6)/60.0     ,   7.9  ),
        (35         , 'Mowry Slough'            , 37+( 28.8)/60.0     , 122+( 4.8)/60.0     ,   8.5  ),
        (36         , 'Calaveras Point'         , 37+( 28.3)/60.0     , 122+( 3.9)/60.0     ,   7.9  )
        ]

    def __init__(self,juliandate=None,year=None,month=None,nearest=True):
        """ juliandate: date string of Polaris cruise in YYYYDDD format.
        if nearest is True, then adjust the juliandate to fall on the nearest
        cruise in the database - this only works when juliandate is specified.
        """
        self.req_juliandate = juliandate
        self.req_year = year
        self.req_month = month

        if nearest and juliandate:
            self.req_juliandate = cruise_nearest_jdate(juliandate)
            if self.req_juliandate != juliandate:
                print("Nearest Polaris cruise is ",self.req_juliandate)

        # initialize the station data:
        self.station_data = array(self.station_data,
                                  [('station_no',float64),
                                   ('name',object),
                                   ('lat',float64),
                                   ('lon',float64),
                                   ('depth',float64)])
        # switch to more conventional east-positive longitide
        self.station_data['lon'] *= -1
        
        self.station_data.sort() # sorts on station number for lookups later on
        self.utc_offset = None # Don't figure timezone until we get the data

    def station_locs_utm(self):
        lats = self.station_data['lat']
        lons = self.station_data['lon']
        lonlat = hstack( ( lons[:,newaxis], lats[:,newaxis]) )
        utm = ll2utm(lonlat)

        return utm
        
    def start_date(self):
        return num2date(self.times[0])
    
    def download(self):
        query = self.query_values(year = self.req_year,
                                  month = self.req_month,
                                  juliandate = self.req_juliandate)
        formdata=six.moves.urllib.parse.urlencode(query)
        # formdata = urllib.urlencode(query)        

        # This will use a POST - the way the original form is submitted
        # fp = urllib.urlopen(self.form_url,data=formdata)
        # Try a GET - it is easier to cache
        self.request_url = self.form_url + "?" + formdata
        
        fp = urlopen(self.request_url)
        self.fp = fp
        txt = fp.read()
        fp.close()

        if len(txt.strip())==0:
            print( "No data in downloaded polaris page")
            print("cache location: ",fp.cache_filenames)

        try:
            txt=txt.decode()
        except AttributeError:
            pass
        
        return txt

    def parse(self,txt):
        # the interesting stuff is inside the one <pre> tag, though
        # we also have to filter out HTML comments, and grab the first row
        # as column names and the second row as units
        txt.find('<pre>')
        pre_section = txt[txt.find('<pre>')+5 : txt.find('</pre>')]

        # filter out html script sections
        patt = re.compile(r'<script>.*?</script>',re.S)
        no_script = re.sub(patt,'',pre_section)
        lines = no_script.split("\n")
        columns = lines[0].split(',')
        units = lines[1].split(',')
        records=lines[2:-1] # ignore empty last line

        headers = [s.lower() for s in columns]

        if len(records) > 0:
            ldata = []

            for r in records:
                fields = r.split(',')
                row = []

                for h,s in zip(headers,fields):
                    if h == 'date':
                        row.append( date2num( datetime.datetime.strptime(s.strip(),'%m/%d/%Y') ) )
                    else:
                        try:
                            row.append(float(s))
                        except ValueError:
                            print( "Issues converting entry for col %s: '%s'"%(h,s) )
                            print( "  from record '%s'"%r)
                            row.append(nan)
                ldata.append(row)

            self.data = array(ldata)
        else:
            print( "NO data!!")
            self.data = zeros( (0,len(headers)), float64 )
        
        self.columns = {}
        
        for i in range(len(headers)):
            self.columns[ headers[i] ] = self.data[:,i]

        return self.columns

    def utc_to_local_days(self):
        # If we have a date, then figure out timezone now
        if 'date' in self.columns:
            # Add the half so that we are in the middle of the day, rather than midnight
            # before it.
            base_date = num2date( self.columns['date'][0] ) + datetime.timedelta(0.5)
            pacific = pytz.timezone('US/Pacific')

            # oddities of the python tzinfo means we create the date then ask
            # pytz to make it 'correct'.
            base_date_py = datetime.datetime(base_date.year,
                                             base_date.month,
                                             base_date.day,
                                             base_date.hour)
            # Note that this will fail if given an ambiguous or impossible date.
            base_date_py = pacific.localize( base_date_py )

            # So during the winter this is -8 hours, and -7 hours in the summer
            off = base_date_py.utcoffset()

            # this value can be *added* to *UTC* to get local times
            return off.days + off.seconds/(24*3600.)
        else:
            raise Exception("No date field, so we can't figure out the timezone offset")
            

    def fetch(self):
        """ download or read from cache the desired data and
        return a dict with arrays for each field
        """
        txt = self.download()
        try:
            self.parse(txt)
        except Exception as exc:
            print("Error while parsing polaris data:")
            print("url was: %s"%self.request_url)
            print("="*80)
            print(txt)
            print("="*80)
            raise
        self.calc_grid_index()
        return self.columns

    grid_index = 'call calc_grid_index!'
    
    def calc_grid_index(self):
        """ Populates self.grid_index with a 2-D integer array
        that maps rows in the original data to cells in a grid
        version of the data.  grid indices are depth,station
        (matrix convention, row x col)
        """
        all_depths = self.columns['depth']
        all_distances = self.columns['distance from 36']
        
        self.depths = unique(all_depths)
        self.depths.sort()

        self.distances = unique(all_distances)
        self.distances.sort()

        # then work backwards to get station indexes and times for each distance:
        # 
        self.stations = zeros( self.distances.shape, float64 )
        
        self.times = zeros( self.distances.shape, float64 )
        local_time_offset = self.utc_to_local_days()
        
        for i in range(len(self.stations)):
            # what station number goes with this distance?
            record = where(all_distances == self.distances[i])[0][0]
            
            self.stations[i] = self.columns['station number'][record]

            # so columns['date'] has absdays in it, and 'time' has
            # 4-digit 24 hour local time.  So the key is converting the
            # local time back to UTC
            hhmm = self.columns['time'][record]
            frac_days = (floor(hhmm/100.) + (hhmm%100.)/60.)/24.
            self.times[i] = self.columns['date'][record] + frac_days - local_time_offset

        # Need to index stations by their distance from 36, but still be able
        # to refer to the station number so we can get the lat/lon later on
        
        self.grid_index = -1 * ones( (len(self.depths),len(self.distances)),int32 )

        for i in range(len(self.data)):
            depth_i = searchsorted(self.depths,all_depths[i])
            dist_i = searchsorted(self.distances,all_distances[i])

            self.grid_index[depth_i,dist_i] = i

        # and put together an array of longitudinal distances, starting from the
        # first station

        # get indices into station_data
        idx = searchsorted(self.station_data['station_no'],self.stations)
        
        lats = self.station_data['lat'][idx]
        lons = self.station_data['lon'][idx]
        self.lonlat = hstack( ( lons[:,newaxis], lats[:,newaxis]) )
        self.utm = ll2utm(self.lonlat)
        
        d = sqrt( (diff(self.utm,axis=0)**2).sum(axis=1) )
        self.x = concatenate( ([0.0], cumsum( d )) )
        self.y = - self.depths

        
    def as_grid(self,field):
        d = self.columns[field]
        d = concatenate( (d,[nan]) )

        g = d[self.grid_index]
        g = ma.array(g,mask=isnan(g))
        return g
    def contourf(self,field,**kwargs):
        gd = self.as_grid(field)
        Y,X = meshgrid(self.y,self.x)
        contourf(X,Y,transpose(gd),**kwargs)

    def to_transect(self,v='salinity',clip_dist=None):
        """ A hopefully neat package for dealing with transect-like
        data
        if clip_dist is specified, throw out data for points that are beyond this distance.
        """
        scalar = self.as_grid(v)
        d = num2date( self.columns['date'][0] )
        
        desc = 'Polaris Cruise, %s: %s'%(d.strftime('%m/%d/%Y'),v)
        if clip_dist is not None:
            valid = self.x <= clip_dist
        else:
            valid = ones( self.x.shape,'bool8')

        # In some cases, a location is missing data entirely -
        # punt, and just remove that water column.
        valid_data = any( ~scalar.mask, axis=0 )
        
        if any( ~valid_data ):
            print("Some Polaris data is missing")
            valid = valid & valid_data

        # assumption on how to handle the bed and surface:
        #  the surface scalar is extended to the surface, while
        #  the bed is taken to be at the location of the last measurement
        if self.y[0] < 0:
            # extend the surface measurement all the way to the surface
            i = concatenate( ( [0],arange(scalar.shape[0])) )
            scalar = scalar[i,:] # concatenate( ( scalar[0:1,:], scalar[:,:]), axis=0)
            y = concatenate( ([0],self.y) )
        else:
            y = self.y
        
        return transect.Transect(self.utm[valid,:],
                                 self.times[valid],
                                 y,scalar[:,valid],
                                 desc=desc,
                                 dists=self.x[valid])

    # Upsampling, in preparation for extrapolation
    def z_level_field(self,dx=5000,v='salinity'):
        """ dx: nominal horizontal spacing
             v: which variable to use

        Interpolate along the transect to get approx. the spacing given by
        dx.  Assumes that the transect is linear, i.e. no loops or
        backtracking

        Returns a ZLevelField
        """
        d = self.fetch()

        # build up new sets of columns
        new_g_list = []
        new_x_list = []
        new_utm_list = []

        X,Y = meshgrid( self.x, self.y ) # that's distance along transect and depth
        Z = self.as_grid(v)

        for station in range(X.shape[1] - 1):
            # This x is distance along transect - utm x,y will have to be dealt with separately
            x1 = self.x[station]
            x2 = self.x[station+1]
            u1 = self.utm[station]
            u2 = self.utm[station+1]

            z1 = Z[:,station]
            z2 = Z[:,station+1]


            # Beginning of transect, include the original data
            if station == 0:
                new_g_list.append(z1[:,newaxis])
                new_x_list.append([x1])
                new_utm_list.append([u1])

            # Choose points in the interval
            nsteps = int(ceil( (x2-x1)/dx ))

            if nsteps > 1:
                alpha = linspace(0,1,nsteps+1)[1:-1]

                new_x = (1-alpha)*x1 + alpha*x2
                new_utm = (1-alpha[:,newaxis])*u1 + alpha[:,newaxis]*u2

                alpha = alpha[newaxis,:]
                new_Z = (1-alpha)*z1[:,newaxis] + alpha*z2[:,newaxis]

                new_g_list.append(new_Z)
                new_x_list.append(new_x)
                new_utm_list.append(new_utm)

            # Include original data at the other end of the interval
            new_g_list.append(z2[:,newaxis])
            new_x_list.append([x2])
            new_utm_list.append([u2])


        # put them all back together:    
        new_x = concatenate( new_x_list)
        new_y = self.y
        new_G = concatenate( new_g_list,axis=1)
        new_G = ma.array(new_G,mask=isnan(new_G))
        new_utm = concatenate( new_utm_list )

        return field.ZLevelField(new_utm,new_y,new_G)
    
    def geographic_xticks(self,ax):
        nums = self.columns['station number']
        dists = self.columns['distance from 36']

        # so I want to remove any duplicates
        nums,num_index = unique(nums,return_index=1)
        dists = dists[num_index]

        db_i = [ nonzero( self.station_data['station_no'] == num )[0][0] for num in nums]
        names = self.station_data['name'][db_i]

        # ax.xaxis.set_tick_major_locator( ticker.FixedLocator(dists) )
        ax.xaxis.set_ticks( 1000*dists )
        ax.xaxis.set_ticklabels( names )
        


if __name__ == '__main__':
    pc = PolarisCruise(juliandate=2008232)

    zlf = pc.z_level_field()

    # and plot the new data:

    clf()
    subplot(211)
    zlf.plot_transect()

    subplot(212)
    import trigrid
    g = trigrid.TriGrid(suntans_path='/home/rusty/classes/research/suntans/runs/long-delta-250m/rundata')
    g.plot(all_cells=0)
    zlf.plot_surface()


    print("Extrapolated value: ",zlf.extrapolate(550000,4.2e6,0))




# # For reference, here are recent cruises that include the Bay Bridge
# Date          Julian Date
# 9/8/2004	2004252	
# 9/14/2004	2004258	
# 11/3/2004	2004308	
# 12/14/2004	2004349	
# 12/7/2005	2005341	
# 11/22/2005	2005326	
# 10/12/2005	2005285	
# 9/26/2005	2005269	
# 9/7/2005	2005250	
# 10/25/2005	2005298	
# 11/8/2005	2005312	
# 8/29/2005	2005241	
# 6/21/2005	2005172	
# 3/22/2005	2005081	
# 5/3/2005	2005123	
# 4/12/2005	2005102	
# 2/23/2005	2005054	
# 4/18/2005	2005108	
# 2/2/2005	2005033	
# 1/11/2005	2005011	
# 2/14/2006	2006045	
# 2/21/2006	2006052	
# 1/10/2006	2006010	
# 5/10/2005	2005130	
# 3/18/2005	2005077	
# 3/7/2006	2006066	
# 3/21/2006	2006080	
# 3/15/2006	2006074	
# 4/5/2006	2006095	
# 4/21/2006	2006111	
# 4/24/1995	1995114	
# 5/9/2006	2006129	
# 9/12/2006	2006255	
# 9/29/2006	2006272	
# 8/31/2006	2006243	
# 8/15/2006	2006227	
# 8/2/2006	2006214	
# 7/11/2006	2006192	
# 10/17/2006	2006290	
# 11/14/2006	2006318	
# 12/12/2006	2006346	
# 1/9/2007	2007009	
# 2/9/2007	2007040	
# 2/6/2007	2007037	
# 8/9/2005	2005221	
# 3/28/2007	2007087	
# 4/10/2007	2007100	
# 4/3/2007	2007093	
# 3/12/2007	2007071	
# 3/6/2007	2007065	
# 4/23/2007	2007113	
# 6/21/2007	2007172	
# 7/23/2007	2007204	
# 8/22/2007	2007234	
# 9/11/2007	2007254	
# 9/21/2007	2007264	
# 10/19/2007	2007292	
# 11/14/2007	2007318	
# 10/23/2007	2007296	
# 12/11/2007	2007345	
# 1/10/2008	2008010	
# 2/1/2008	2008032	
# 2/12/2008	2008043	
# 2/29/2008	2008060	
# 3/31/2008	2008091	
# 3/11/2008	2008071	
# 4/14/2008	2008105	
# 4/28/2008	2008119	
# 5/6/2008	2008127	
# 5/29/2008	2008150	
# 6/17/2008	2008169	
# 7/10/2008	2008192	
# 7/15/2008	2008197	
# 8/12/2008	2008225	
# 8/19/2008	2008232	
# 9/9/2008	2008253	
# 9/16/2008	2008260	
# 10/7/2008	2008281	
# 10/15/2008	2008289	
# 11/5/2008	2008310	
# 11/18/2008	2008323	
# 12/16/2008	2008351	
# 1/13/2009	2009013	
# 2/4/2009	2009035	
# 2/10/2009	2009041	
# 2/17/2009	2009048	
# 3/5/2009	2009064	
# 3/20/2009	2009079	
# 3/12/2009	2009071	
# 4/7/2009	2009097	
# 4/16/2009	2009106	
# 5/5/2009	2009125	
# 5/19/2009	2009139	
# 6/16/2009	2009167	
# 6/23/2009	2009174	
# 7/15/2009	2009196	
# 7/21/2009	2009202	
# 8/13/2009	2009225	


if 0:
    # Write the locations of the cruise to a shapefile as points
    # could also do a transect, but we'd have to pick a specific line up of the stations
    import wkb2shp
    from shapely import geometry

    pc = PolarisCruise()
    stations = pc.station_locs_utm()

    geoms = [ geometry.Point( stations[i,0], stations[i,1] ) for i in range(len(stations)) ]

    i_iter = iter( range(len(stations)) )

    def field_gen(g):
        i = i_iter.next()

        s = pc.station_data[i]
        return {'name':s['name'],
                'station_no':s['station_no'],
                'depth':s['depth'],
                'source':'polaris_cruise'
                }

    wkb2shp.wkb2shp('/home/rusty/classes/research/suntans/runs/polaris_points.shp',
                    geoms,field_gen=field_gen,overwrite=True)

