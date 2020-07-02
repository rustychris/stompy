import re
import os
import mmap
import numpy as np
import xml.etree.ElementTree as ET
from .. import utils

def read_tin_xml_ET(fn):
    """
    Parse a TIN from a LandXML file.
    Returns two arrays:
      P: points [:,3] doubles
      F: faces [:,3]  integers
    
    This version uses a proper XML parser, for robust but slow parsing.

    Only handles a single TIN per file
    """
    Ps=np.nan*np.zeros( (1,3), np.float64)
    Fs=-1*np.ones( (1,3), np.int32)
    Fcount=0

    parser = ET.XMLPullParser(['start', 'end'])

    tag_types={}
    blksize=10000
    n_blks=1+int(os.stat(fn).st_size / blksize)
    with open(fn,'rt') as fp:
        for _ in utils.progress(range(n_blks)):
            buff=fp.read(blksize)
            if len(buff)==0:
                break
            parser.feed(buff)
            for event, elem in parser.read_events():
                if elem.tag not in tag_types:
                    print(elem.tag, 'text=', elem.text)
                    tag_types[elem.tag]=elem
                if elem.text is None: continue
                if elem.tag=="{http://www.landxml.org/schema/LandXML-1.2}P":
                    pid=int(elem.attrib['id'])
                    P=[float(s) for s in elem.text.split()]
                    # Appears that these are written lat/long order, but I prefer
                    # to keep everything x/y
                    P[0],P[1] = P[1],P[0]
                    while len(Ps)<pid+1:
                        Ps=np.concatenate( [Ps,np.nan*Ps], axis=0)
                    Ps[pid]=P
                elif elem.tag=="{http://www.landxml.org/schema/LandXML-1.2}F":
                    F=[int(s) for s in elem.text.split()]
                    fid=Fcount
                    while fid+1>len(Fs):
                        Fs=np.concatenate( [Fs,-1*Fs], axis=0)
                    Fs[fid]=F
                    Fcount+=1
    
    return Ps,Fs

def read_tin_xml_mmap(fn):
    """
    Parse a TIN from a LandXML file.
    Returns two arrays:
      P: points [:,3] doubles
      F: faces [:,3]  integers

    This version is faster than the ET version, but may not be 
    as portable since it relies on memory mapping the file.

    Only handles a single TIN per file
    """
    Ps=np.nan*np.zeros( (1,3), np.float64)
    Fs=-1*np.ones( (1,3), np.int32)
    Pcount=0
    Fcount=0

    nbytes=os.stat(fn).st_size
    
    with open(fn,'rb') as fp:
        mm=mmap.mmap(fp.fileno(),length=0,prot=mmap.PROT_READ)
        miter=re.finditer( (b'(<P id="(\d+)">([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+)</P>)'
                            b'|(<F[^>]*>(\d+) (\d+) (\d+)</F>)'),
                           mm)
        for g in miter:
            if (Pcount+Fcount)%100000==0:
                print("%5.1f%% of %d bytes"%(100*g.span()[-1]/float(nbytes),nbytes))
                
            if g.group(1) is not None:
                pid=int(g.group(2))
                # With a sample size of 1, the x/y are reversed (i.e. they are in
                # lat/long ordering, not x/y ordering.
                # flip around here.
                P=[float(g.group(4)),
                   float(g.group(3)),
                   float(g.group(5))]
                while len(Ps)<pid+1:
                    Ps=np.concatenate( [Ps,np.nan*Ps], axis=0)
                Ps[pid]=P
                Pcount+=1
            elif g.group(6) is not None:
                F=[int(g.group(7)), int(g.group(8)), int(g.group(9))]
                fid=Fcount
                while fid+1>len(Fs):
                    Fs=np.concatenate( [Fs,0*Fs-1], axis=0)
                Fs[fid]=F
                Fcount+=1
    
        mm.close()
    Fs=Fs[:Fcount]

    # Renumber to get just the points that were defined.
    Pvalid=np.isfinite(Ps[:,0])
    Ps_valid=Ps[Pvalid,:]
    remap=np.zeros(Ps.shape[0],np.int32)-999
    remap[Pvalid]=np.arange(len(Ps_valid))
    Fs_valid=remap[Fs]

    missing=np.nonzero( Fs_valid.min(axis=1)<0 )[0]
    if len(missing):
        missing_points=np.unique( Fs[ Fs_valid<0] )
        # 2691390, 2693678, 2693687
        import pdb
        pdb.set_trace()
    
    return Ps_valid,Fs_valid
