import os
import numpy as np
import pandas as pd
import re
import xarray as xr
import six

import logging

log=logging.getLogger('delft.io')

from . import waq_scenario as waq
from ... import utils

def parse_his_file(fn):
    """
    you probably want mon_his_file_dataframe() or bal_his_file_dataframe()
    --
    parse mixed ascii/binary history files as output by delwaq.
    applies to both monitoring output and balance output.
        
    returns tuple:
      sim_descs - descriptive text from inp file.
      time0 - text line giving time origin and units
      regions - names of regions with numeric index (1-based as read from file)
      fields - names of fields, separate into substance and process
      frames - actual data, and timestamps
    """
    fp=open(fn,'rb') 

    sim_descs=np.fromfile(fp,'S40',4)
    time0=sim_descs[3]

    n_fields,n_regions=np.fromfile(fp,'i4',2)

    fdtype=np.dtype( [ ('sub','S10'),
                       ('proc','S10') ] )

    fields=np.fromfile( fp, fdtype, n_fields)

    regions=np.fromfile(fp,
                        [('num','i4'),('name','S20')],
                        n_regions)

    # assume that data is 'f4'
    # following other Delft output, probably each frame is prepended by
    # 'i4' time index
    frame_dtype=np.dtype( [('tsec','i4'),
                           ('data','f4',(n_regions,n_fields))] )
    frames=np.fromfile(fp,frame_dtype)

    return sim_descs,time0,regions,fields,frames

def bal_his_file_dataframe(fn):
    sim_descs,time0,regions,fields,frames = parse_his_file(fn)

    n_regions=len(regions)
    n_fields=len(fields)
    cols=[]
    tuples=[]

    for ri,region in enumerate(regions):
        for fi,field in enumerate(fields):
            tuples.append( (region['name'].strip(),
                            field['sub'].strip(),
                            field['proc'].strip()) )

    col_index=pd.MultiIndex.from_tuples(tuples,names=('region','sub','proc'))
    df=pd.DataFrame(data=frames['data'].reshape( (-1,n_regions*n_fields) ),
                    index=frames['tsec'],
                    columns=col_index)
    return df


def bal_his_file_xarray(fn,region_exclude=None,region_include=None):
    """
    Read a delwaq balance file, return the result as an xarray.
    region_exclude: regular expression for region names to omit from the result
    region_include: regular expression for region names to include.  

    Defaults to returning all regions.
    """
    sim_descs,time_meta,regions,fields,frames = parse_his_file(fn)

    def decstrip(s):
        try:
            s=s.decode() # in case binary
        except AttributeError:
            pass
        return s.strip()

    ds=xr.Dataset()

    ds['descs']=( ('n_desc',), [decstrip(s) for s in sim_descs])

    time0,time_unit = parse_time0(time_meta)
    times=time0 + time_unit*frames['tsec']
    ds['time']=( ('time',), times)
    ds['tsec']=( ('time',), frames['tsec'])

    region_names=[decstrip(s) for s in regions['name']]
    subs=[decstrip(s) for s in np.unique(fields['sub'])]
    procs=[decstrip(s) for s in np.unique(fields['proc'])]

    if region_include:
        region_mask=np.array( [bool(re.match(region_include,region))
                               for region in region_names] )
    else:
        region_mask=np.ones(len(region_names),np.bool8)

    if region_exclude:
        skip=[bool(re.match(region_exclude,region))
              for region in region_names]
        region_mask &= ~np.array(skip)
    
    sub_proc=[ "%s,%s"%(decstrip(s),decstrip(p))
               for s,p in fields]

    region_idxs=np.nonzero(region_mask)[0]
    ds['region']=( ('region',), [region_names[i] for i in region_idxs] )
    ds['sub']  =( ('sub',), subs)
    ds['proc'] =( ('proc',), procs)
    ds['field']=( ('field',), sub_proc)

    ds['bal']=( ('time','region','field'),
                frames['data'][:,region_mask,:] )
    return ds



def mon_his_file_dataframe(fn):
    df=bal_his_file_dataframe(fn)
    df.columns=df.columns.droplevel(2) # drop process level
    return df


def inp_tok(fp):
    # tokenizer for parsing rules of delwaq inp file.
    # parses either single-quoted strings, or space-delimited literals.
    for line in fp:
        if ';' in line:
            line=line[ : line.index(';')]
        # pattern had been
        # r'\s*((\'[^\']+\')|([/:-a-zA-Z_#0-9\.]+))'
        # but that has a bad dash before a, and doesn't permit +, either.
        matches=re.findall(r'\s*((\'[^\']+\')|([-/:a-zA-Z_#+0-9\.]+))', line)
        for m in matches:
            yield m[0]


            
def parse_inp_monitor_locations(inp_file):
    """
    returns areas[name]=>[seg1,...] , transects[name]=>[+-exch1, ...]
    ONE-BASED return values.
    """
    with open(inp_file,'rt') as fp:
        tokr=inp_tok(fp)

        while next(tokr)!='#1':
            continue
        for _ in range(4):  # clock/date formats, integration float
            next(tokr)
        for t in tokr:
            if re.match(r'[-_a-zA-Z]+',t):
                continue
            break
        # t is now start timestep
        for _ in range(3):
            next(tokr) # stop, time step time step
        areas={}
        if int(next(tokr)) == 1: # monitoring points used
            nmon = int(next(tokr))
            for imon in range(nmon):
                name, segcount=next(tokr),int(next(tokr))
                segs=[int(next(tokr)) for iseg in range(segcount)]
                areas[name.strip("'")]=segs
        transects={} # name => list of signed, 1-based exchanges
        if int(next(tokr)) == 1: # transects used
            ntrans=int(next(tokr))
            for itrans in range(ntrans):
                name,style,ecount = next(tokr),next(tokr),int(next(tokr))
                exchs=[int(next(tokr)) for _ in range(ecount)]
                transects[name.strip("'")]=exchs
    return areas,transects

def parse_inp_transects(inp_file):
    # with open(inp_file,'rt') as fp:
    #     tokr=inp_tok(fp)
    # 
    #     while next(tokr)!='#1':
    #         continue
    #     for _ in range(4):  # clock/date formats, integration float
    #         next(tokr)
    #     for t in tokr:
    #         if re.match(r'[-_a-zA-Z]+',t):
    #             continue
    #         break
    #     # t is now start timestep
    #     for _ in range(3):
    #         next(tokr) # stop, time step time step
    #     if int(next(tokr)) == 1: # monitoring points used
    #         nmon = int(next(tokr))
    #         for imon in range(nmon):
    #             name, segcount=next(tokr),int(next(tokr))
    #             for iseg in range(segcount):
    #                 next(tokr)
    #     transects={} # name => list of signed, 1-based exchanges
    #     if int(next(tokr)) == 1: # transects used
    #         ntrans=int(next(tokr))
    #         for itrans in range(ntrans):
    #             name,style,ecount = next(tokr),next(tokr),int(next(tokr))
    #             exchs=[int(next(tokr)) for _ in range(ecount)]
    #             transects[name.strip("'")]=exchs
    
    areas,transects=parse_inp_monitor_locations(inp_file)
    
    return transects

def parse_time0(time0):
    """ return a np.datetime64 for the time stamp, and the time unit in seconds
    (almost always equal to 1 second)
    input format is: b'T0: 2012/08/07-00:00:00  (scu=       1s)'
    """
    try:
        time0=time0.decode()
    except AttributeError:
        pass

    m=re.match(r'T0:\s+(\S+)\s+\(scu=\s*(\d+)(\w+)\)',time0)
    dt = m.group(1)
    # make it clear it's UTC:
    dt=dt.replace('-','T').replace('/','-') + "Z"
    origin=np.datetime64(dt)
    unit=np.timedelta64(int(m.group(2)),m.group(3)) 

    return (origin, unit)


# just a start.  And really this stuff should be rolled into the Scenario
# class, so it builds up a Scenario
def parse_boundary_conditions(inp_file):
    with open(inp_file,'rt') as fp:
        tokr=inp_tok(fp)

        while next(tokr)!='#4':
            continue

        bcs=[]
        while 1:
            tok = next(tokr)
            if tok[0] in "-0123456789":
                n_thatcher = int(tok)
                break
            else:
                bc_id=str_or_num
                bc_typ=next(tokr)
                bc_grp=next(tokr)
                bcs.append( (bc_id,bc_typ,bc_grp) )


def read_pli(fn):
    """
    Parse a polyline file a la DFM inputs.
    Return a list of features:
    [  (feature_label, N*M values), ... ]
    where the first two columns are typically x and y, but there may be
    more columns depending on the usage.
    """
    with open(fn,'rt') as fp:
        toker=inp_tok(fp)
        token=lambda: six.next(toker)

        features=[]
        while True:
            try:
                label=token()
            except StopIteration:
                break
            nrows=int(token())
            ncols=int(token())
            geometry=[]
            for row in range(nrows):
                rec=[float(token()) for c in range(ncols)]
                geometry.append(rec)
            features.append( (label, np.array(geometry) ) )
    return features


def read_map(fn,hyd,use_memmap=True,include_grid=True):
    """
    Read binary D-Water Quality map output, returning an xarray dataset.

    fn: path to .map file
    hyd: path to .hyd file describing the hydrodynamics.
    use_memmap: use memory mapping for file access.  Currently
      this must be enabled.

    include_grid: the returned dataset also includes grid geometry, suitable
       for unstructured_grid.from_ugrid(ds)

    note that missing values at this time are not handled - they'll remain as
    the delwaq standard -999.0.
    """
    if not isinstance(hyd,waq.Hydro):
        hyd=waq.HydroFiles(hyd)

    nbytes=os.stat(fn).st_size # 420106552 

    with open(fn,'rb') as fp:

        # header line of 160 characters
        txt_header=fp.read(160)
        # print "Text header: ",txt_header

        # 4 bytes, maybe a little-endian int.  0x0e, that's 14, number of substances
        n_subs=np.fromfile(fp,np.int32,1)[0]
        # print "# substances: %d"%n_subs

        n_segs=np.fromfile(fp,np.int32,1)[0]
        # print "Nsegs: %d"%n_segs

        substance_names=np.fromfile(fp,'S20',n_subs)

        # not sure if there is a quicker way to get the number of layers
        hyd.infer_2d_elements()
        n_layers=1+hyd.seg_k.max()

        g=hyd.grid() # ignore message about ugrid.

        assert g.Ncells()*n_layers == n_segs

        # I'm hoping that now we get 4 byte timestamps in reference seconds,
        # and then n_subs,n_segs chunks.
        # looks that way.
        data_start=fp.tell()

    bytes_left=nbytes-data_start 
    framesize=(4+4*n_subs*n_segs)
    nframes,extra=divmod(bytes_left,framesize)
    if extra!=0:
        log.warning("Reading map file %s: bad length %d extra bytes (or %d missing)"%(
            fn,extra,framesize-extra))

    # Specify nframes in cases where the filesizes don't quite match up.
    mapped=np.memmap(fn,[ ('tsecs','i4'),
                          ('data','f4',(n_layers,hyd.n_2d_elements,n_subs))] ,
                     mode='r',
                     shape=(nframes,),
                     offset=data_start)

    ds=xr.Dataset()

    ds.attrs['header']=txt_header
    
    ds['sub']= ( ('sub',), [s.strip() for s in substance_names] )

    times=utils.to_dt64(hyd.time0) + np.timedelta64(1,'s') * mapped['tsecs']

    ds['time']=( ('time',), times)
    ds['t_sec']=( ('time',), mapped['tsecs'] )

    for idx,name in enumerate(ds.sub.values):
        ds[name]= ( ('time','layer','face'), 
                    mapped['data'][...,idx] )

    if include_grid:
        # not sure why this doesn't work.
        g.write_to_xarray(ds=ds)

    return ds
