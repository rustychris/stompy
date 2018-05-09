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
    def process(X):
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

    
