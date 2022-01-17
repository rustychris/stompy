"""
Bundle up common projection operations
"""
from __future__ import print_function

from osgeo import osr
from numpy import *
import numpy as np

def to_srs(s):
    if not isinstance(s,osr.SpatialReference):
        srs = osr.SpatialReference() ; srs.SetFromUserInput(s)
        # recent osr uses lat/long ordering, but that breaks a lot
        # of code.  This call says use long/lat, and assume that
        # if it doesn't exist it's an old version and doesn't matter.
        try:
            srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        except AttributeError:
            pass
        return srs
    else:
        return s
    
def xform(src,dest):
    src = to_srs(src)
    dest = to_srs(dest)
    return osr.CoordinateTransformation(src,dest)

def mapper(src,dest):
    """
    use it like this:
      m = mapper("WGS84","EPSG:26910")
      print m(array([-122,35]))

    """
    trans = xform(src,dest)
    def process(X,Y=None):
        if Y is not None:
            X=np.stack([X,Y],axis=-1)
        X = asarray(X,float64)
        out = zeros(X.shape,float64)
        inx = X[...,0]
        iny = X[...,1]

        # careful - these have to remain as views
        outx = out[...,0]
        outy = out[...,1]

        # for idx in range(len(inx)):
        for idx in ndindex(inx.shape):
            outx[idx],outy[idx],dummy = trans.TransformPoint(inx[idx],iny[idx],0)
        return out
    return process


# maps lat/lon to utm
def ll_to_utm(LL,center_lon=None):

    # Based on deg2utm
    # Author: 
    #   Rafael Palacios
    #   Universidad Pontificia Comillas
    #   Madrid, Spain
    # Version: Apr/06, Jun/06, Aug/06, Aug/06
 
    lo=LL[...,0]
    la=LL[...,1]

    center_lon = center_lon or lo.mean()

    # choose a utm zone based on center_lon
    Huso = np.fix( ( center_lon / 6.0 ) + 31)
    S = Huso*6 - 183

    # allocate output
    X=np.zeros_like(LL)

    sa = 6378137.000000 ; sb = 6356752.314245;
         
    e2 = np.sqrt( sa**2 - sb**2 ) / sb
    e2cuadrada = e2**2
    c = sa**2 / sb
    alfa = (3./4) * e2cuadrada
    beta = (5./3) * alfa**2
    gama = (35./27 ) * alfa**3

    # Main Loop
    #
    # for i=1:n1

    lat = la * ( np.pi / 180 )
    lon = lo * ( np.pi / 180 )

    deltaS = lon - S*(np.pi/180) 

    a = np.cos(lat) * np.sin(deltaS)
    epsilon = 0.5 * np.log( (1+a) / (1-a) )
    nu = np.arctan( np.tan(lat) / cos(deltaS) ) - lat
    v = ( c / ( 1 + ( e2cuadrada * np.cos(lat)**2 ) )**0.5 ) * 0.9996
    ta = e2cuadrada/2 * epsilon**2 * np.cos(lat)**2
    a1 = np.sin( 2*lat )
    a2 = a1*np.cos(lat)**2
    j2 = lat + a1/2
    j4 = (3*j2 + a2)/4
    j6 = ( (5*j4) + ( a2*( np.cos(lat) ) ** 2) ) / 3
    Bm = 0.9996 * c * ( lat - alfa * j2 + beta * j4 - gama * j6 )
    xx = epsilon * v * ( 1 + ( ta / 3. ) ) + 500000
    yy = nu * v * ( 1 + ta ) + Bm

    yy[yy<0] += 9999999

    X[...,0]=xx
    X[...,1]=yy
    print("UTM zone ",Huso)
    return X

    
def reproject_bounds(src_bounds,src_projection,tgt_projection,mode='outside'):
    """
    Transform a coordinate-aligned bounding box into a new coordinate system.
    src_bounds: xxyy sequence for source bounding box
    src_projection, tgt_projection: proj.4 compatible text strings giving coordinate references.
    mode: 'outside' the returned box will be larger than the src_bounds
          'inside': not yet supported and not well-defined. This would return a projected 
                    bounding box that is inscribed in the true bounding polygon. A naive 
                    implementation may return an empty region.
    """
    x0,x1,y0,y1 = src_bounds
    # Recent proj seems to have a project bounds method, but a quick check of my local
    # install does not have or expose this, so do a goofy manual approach
    N=10 # how many points to use when discretizing the boundary
    x=np.linspace(x0,x1,N)
    y=np.linspace(y0,y1,N)
    XY=np.zeros((4*N,2),np.float64)
    
    XY[:N,0]=x
    XY[:N,1]=y0
    
    XY[N:2*N,0]=x
    XY[N:2*N,1]=y1
    
    XY[2*N:3*N,0]=x0
    XY[2*N:3*N,1]=y
    
    XY[3*N:4*N,0]=x1
    XY[3*N:4*N,1]=y
    
    tgtXY=mapper(src_projection,tgt_projection)(XY)
    if mode=='outside':
        return [tgtXY[:,0].min(),tgtXY[:,0].max(),
                tgtXY[:,1].min(),tgtXY[:,1].max()]
    else:
        raise Exception("Only mode='outside' is implemented")
    
