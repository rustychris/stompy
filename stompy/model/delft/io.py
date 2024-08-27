import os
import glob
import subprocess

import copy
import datetime
import io # python io.
import numpy as np
import pandas as pd
import re
import xarray as xr
import six
from collections import defaultdict
from shapely import geometry

from ...grid import unstructured_grid

import logging

log=logging.getLogger('delft.io')

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


def his_file_xarray(fn,region_exclude=None,region_include=None):
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
        region_mask=np.ones(len(region_names),np.bool_)

    if region_exclude:
        skip=[bool(re.match(region_exclude,region))
              for region in region_names]
        region_mask &= ~np.array(skip)

    sub_proc=[]
    for s,p in fields:
        if decstrip(p):
            sub_proc.append("%s,%s"%(decstrip(s),decstrip(p)))
        else:
            sub_proc.append(decstrip(s))

    region_idxs=np.nonzero(region_mask)[0]
    ds['region']=( ('region',), [region_names[i] for i in region_idxs] )
    ds['sub']  =( ('sub',), subs)
    ds['proc'] =( ('proc',), procs)
    ds['field']=( ('field',), sub_proc)

    ds['bal']=( ('time','region','field'),
                frames['data'][:,region_mask,:] )
    return ds

# older name - xarray version doesn't discriminate between balance
# and monitoring output
bal_his_file_xarray=his_file_xarray

def mon_his_file_dataframe(fn):
    df=bal_his_file_dataframe(fn)
    df.columns=df.columns.droplevel(2) # drop process level
    return df


def inp_tok(fp,comment=';'):
    # tokenizer for parsing rules of delwaq inp file.
    # parses either single-quoted strings, or space-delimited literals.
    for line in fp:
        if comment in line:
            line=line[ : line.index(comment)]
        # pattern had been
        # r'\s*((\'[^\']+\')|([/:-a-zA-Z_#0-9\.]+))'
        # but that has a bad dash before a, and doesn't permit +, either.
        matches=re.findall(r'\s*((\'[^\']+\')|([-/:a-zA-Z_#+0-9\.]+))', line)
        for m in matches:
            yield m[0]


def inp_tok_include(fp,fn,**kw):
    """
    Wrap inp_tok and handle INCLUDE tokens transparently.
    Note that also requires the filename
    """
    tokr=inp_tok(fp,**kw)

    while 1:
        tok=next(tokr)
        if tok.upper()!='INCLUDE':
            yield tok
        else:
            inc_fn=next(tokr)
            if inc_fn[0] in ["'",'"']:
                inc_fn=inc_fn.strip(inc_fn[0])
            inc_path=os.path.join( os.path.dirname(fn),
                                   inc_fn )
            # print("Will include %s"%inc_path)

            with open(inc_path,'rt') as inc_fp:
                inc_tokr=inp_tok_include(inc_fp,inc_path,**kw)
                for tok in inc_tokr:
                    yield tok
            # print("Done with include")


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
    """
    Parse section 5 of DWAQ input file.
    Returns bcs,items
    bcs: BC links
    items: match data and bc links.
     - strings are folded to lowercase
    """
    def dequote(s):
        s=s.strip()
        if s[0] in ['"',"'"]:
            s=s.strip(s[0])
        return s
    with open(inp_file,'rt') as fp:
        tokr=inp_tok_include(fp,inp_file)

        while next(tokr)!='#4':
            continue

        bcs=[]
        while 1:
            tok = next(tokr)
            if tok[0] in "-0123456789":
                thatcher = int(tok)
                break
            else:
                bc_id=dequote(tok)
                bc_typ=dequote(next(tokr))
                bc_grp=dequote(next(tokr))
                bcs.append( (bc_id,bc_typ,bc_grp) )

        if thatcher==0: # no lags
            pass
        else:
            assert False,"Parsing Thatcher-Harleman lags not yet implemented"

        # The actual items are not yet implemented -- this is where
        # the inp file would assign concentrations or fluxes to
        # specific boundary exchanges are groups defined above
        bc_items=[]

        tok=next(tokr)
        while 1: # iterate over BC blocks
            defs=[]
            while 1: # iterate over the 3 subparts of a block
                if tok.upper()=='ITEM':
                    # Read names of BC items, which could also be integers
                    item_block=[]
                    while 1:
                        tok=next(tokr)
                        if tok[0] in ["'",'"']:
                            item_block.append(dequote(tok).lower())
                            continue
                        elif tok[0] in "0123456789":
                            item_block.append(int(tok))
                        else:
                            break
                    defs.append( ('item',item_block) )
                elif tok.upper()=='CONCENTRATION':
                    # Read the names of scalars
                    # Read names of BC items, which could also be integers
                    conc_block=[]
                    while 1:
                        tok=next(tokr)
                        if tok.upper() not in ['DATA','ITEM']:
                            conc_block.append(dequote(tok).lower())
                            continue
                        else:
                            break
                    defs.append( ('concentration',conc_block) )
                elif tok.upper()=='DATA':
                    matrix=np.zeros( ( len(defs[0][1]),
                                       len(defs[1][1]) ), 'f8')
                    for row_i,row in enumerate(defs[0][1]):
                        for col_i,col in enumerate(defs[1][1]):
                            matrix[row_i,col_i]=float(next(tokr))
                    defs.append( ('data',matrix) )
                    tok=next(tokr)
                else:
                    break # must not have been a BC block
            if len(defs)==0:
                assert tok=='#5'
                break # great - not a block
            elif len(defs)==3:
                bc_items.append(defs)
            else:
                assert False,"Incomplete BC block"

    return bcs,bc_items

def pli_to_shp(pli_fn,shp_fn,overwrite=False):
    from shapely import geometry
    from ...spatial import wkb2shp

    feats=read_pli(pli_fn)
    def clean_pnts(pnts):
        if pnts.shape[0]==1:
            pnts=np.concatenate( [pnts,pnts])
        return pnts
    geoms=[ geometry.LineString(clean_pnts(feat[1]))
            for feat in feats ]
    names=[ feat[0] for feat in feats ]
    wkb2shp.wkb2shp(shp_fn,geoms,fields=dict(name=names),
                    overwrite=overwrite)

def read_pli(fn,one_per_line=True):
    """
    Parse a polyline file a la DFM inputs.
    Return a list of features:
    [  (feature_label, N*M values, N labels), ... ]
    where the first two columns are typically x and y, but there may be
    more columns depending on the usage.  If no labels are in the file,
    the list of labels will be all empty strings.

    Generally assumes that the file is honest about the number of fields,
    but some files (like boundary condition pli) will add a text label for 
    each node.

    one_per_line: for files which add a label to each node but say nothing of 
      this in the number of fields, one_per_line=True will assume that each line
      of the text file has exactly one node, and any extra text becomes the label.
    """
    features=[]

    with open(fn,'rt') as fp:
        if not one_per_line:
            toker=inp_tok(fp)
            token=lambda: six.next(toker)

            while True:
                try:
                    label=token()
                except StopIteration:
                    break
                nrows=int(token())
                ncols=int(token())
                geometry=[]
                node_labels=[]
                for row in range(nrows):
                    rec=[float(token()) for c in range(ncols)]
                    geometry.append(rec)
                    node_labels.append("") 
                features.append( (label, np.array(geometry), node_labels) )
        else: # line-oriented approach which can handle unannounced node labels
            def getline():
                while True:
                    l=fp.readline()
                    if l=="": return l # EOF
                    # lazy comment handling
                    l=l.split('#')[0]
                    l=l.split('*')[0]
                    l=l.strip()
                    if l!="":
                        return l
                
            while True:
                label=getline()
                if label=="":
                    break

                nrows,ncols = [int(s) for s in getline().split()]
                geometry=[]
                node_labels=[]
                for row in range(nrows):
                    values=getline().split(None,ncols+1)
                    geometry.append( [float(s) for s in values[:ncols]] )
                    if len(values)>ncols:
                        node_labels.append(values[ncols])
                    else:
                        node_labels.append("")
                features.append( (label, np.array(geometry), node_labels) )
    return features

def write_pli(file_like,pli_data):
    """
    Reverse of read_pli.
    file_like: a string giving the name of a file to be opened (clobbering
    an existing file), or a file-like object.
    pli_data: [ (label, N*M values, [optional N labels]), ... ]
    typically first two values of each row are x and y, and the rest depend on intended
    usage of the file
    """
    if hasattr(file_like,'write'):
        fp=file_like
        do_close=False
    else:
        fp=open(file_like,'wt')
        do_close=True

    try:
        for feature in pli_data:
            label,data = feature[:2]
            data=np.asanyarray(data)
            if len(feature)==3:
                node_labels=feature[2]
            else:
                node_labels=[""]*len(data)

            fp.write("%s\n"%label)
            fp.write("     %d     %d\n"%data.shape)
            if len(data) != len(node_labels):
                raise Exception("%d nodes, but there are %d node labels"%(len(data),
                                                                          len(node_labels)))
            # .strip to trim leading white space
            block="\n".join( [ "  ".join(["%15s"%d for d in row]).strip() + "   " + node_label
                               for row,node_label in zip(data,node_labels)] )
            fp.write(block)
            fp.write("\n")
    finally:
        if do_close:
            fp.close()

def grid_to_pli_data(g,node_fields,labeler=None):
    """
    UnstructuredGrid => PLI translation
    translate the edges of g into a list of features as returned
    by read_pli()
    features are extracted as contiguous linestrings, as long as possible.
    node_fields is a list giving a subset of the grid's node fields to
    be written out, in addition to x and y.
    labeler: leave as None to get Lnnn style labels.  Otherwise, a function
       which takes the index, and returns a string for the label.
    """
    strings=g.extract_linear_strings()

    features=[]

    labeler=labeler or (lambda i: "L%04d"%i)

    for feat_i,nodes in enumerate(strings):
        label=labeler(feat_i)

        cols=[ g.nodes['x'][nodes,0], g.nodes['x'][nodes,1] ]

        for fld in node_fields:
            cols.append( g.nodes[fld][nodes] )
        feature=np.array( cols ).T
        features.append( (label,feature) )
    return features

def add_suffix_to_feature(feat,suffix):
    """
    Utility method, takes a feature as returned by read_pli
    (name,
     [ [x0,y0],[x1,y1],...],
     { [node_label0,node_label1,...] }  # optional
    )

    and adds a suffix to the name of the feature and the
    names of nodes if they exist
    """
    name=feat[0]
    suffize=lambda s: s.replace(name,name+suffix)
    feat_suffix=[suffize(feat[0]), feat[1]] # points stay the same
    if len(feat)==3: # includes names for nodes
        feat_suffix.append( [suffize(s) for s in feat[2]] )
    return feat_suffix


def pli_to_grid_edges(g,levees):
    """
    g: UnstructuredGrid
    levees: polylines in the format returned by stompy.model.delft.io.read_pli,
    i.e. a list of features
    [ 
      [ 'name', 
        [ [x,y,z,...],...], 
        ['node0',...]
      ], ... 
    ]

    returns an array of length g.Nedges(), with z values from those features
    mapped onto edges. when multiple z values map to the same grid edge, the 
    minimum value is used.
    grid edges which do not have a levee edge get nan.
    """
    poly=g.boundary_polygon()

    # The dual additionally allows picking out edges 
    gd=g.create_dual(center='centroid',create_cells=False,remove_disconnected=False,
                     remove_1d=False)

    levee_de=np.nan*np.zeros(g.Nedges())

    for levee in utils.progress(levees,msg="Levees: %s"):
        # levee: [name, Nx{x,y,z,l,r}, {labels}]
        xyz=levee[1][:,:3]
        # having shapely check the intersection is 100x
        # faster than using select_cells_nearest(inside=True)
        ls=geometry.LineString(xyz[:,:2])
        if not poly.intersects(ls): continue

        # clip the edges to get some speedup
        xxyy=[xyz[:,0].min(),
              xyz[:,0].max(),
              xyz[:,1].min(),
              xyz[:,1].max()]
        edge_mask=gd.edge_clip_mask(xxyy,ends=True)

        # edges that make up the snapped line
        gde=gd.select_edges_intersecting(ls,mask=edge_mask)
        gde=np.nonzero(gde)[0]
        if len(gde)==0:
            continue
        # map the dual edge indexes back to the original grid
        ge=gd.edges['dual_edge'][gde]

        # print("Got a live one!")

        # check for closed ring:
        closed=np.all( xyz[-1,:2]==xyz[0,:2] )
        dists=utils.dist_along(xyz[:,:2])

        for j in ge:
            n1,n2=g.edges['nodes'][j]
            l1=np.argmin( utils.dist(g.nodes['x'][n1] - xyz[:,:2] ) )
            l2=np.argmin( utils.dist(g.nodes['x'][n2] - xyz[:,:2] ) )
            if l1>l2:
                l1,l2=l2,l1
            zs=xyz[l1:l2+1,2]

            if closed:
                # allow for possibility that the levee is a closed ring
                # and this grid edge actually straddles the end,
                dist_fwd=dists[l2]-dists[l1]
                dist_rev=dists[-1] - dist_fwd
                if dist_rev<dist_fwd:
                    print("wraparound")
                    zs=np.concatenate( [xyz[l2:,2],
                                        xyz[:l1,2]] )

            z=zs.min() 
            levee_de[j]=z
    return levee_de


def create_restart(res_fn, map_fn, hyd, state_vars = None, map_net_cdf = False, extr_time = None,
                   start_time = None):
    """ 
    Create a restart file using an existing map file and a user defined 
    time. 
    
    res_fn: path/file name of the restart file 
    map_fn: provide either a binary or net CDF file. Default is the raw binary output. 
    hyd: waq_scenario.Hydro() object
    state_vars: an array that provides the state variables used in the original 
    model run. Name must match the information written to map file
    map_net_cdf: set this flag to True if providing net CDF output that has 
        already been created by read_map 
    extr_time: a date string that will be converted internally to a 
        numpy datetime object.If not provided, will use the last time step of the 
        mapfile 
    start_time: a date string that will be converted internally to a numpy
        datetime object. If not provided will be set as hydro.

    [author: Pradeep Mugunthan]
    """
    # Identify whether read_map call is needed 
    if not map_net_cdf:
        ds = read_map(map_fn, hyd)
    else:
        ds = map_fn
        assert isinstance(ds,xr.Dataset),"Maybe this should check for a string and xr.open_dataset it"
        assert 'layer' in ds.dims,"Provided dataset must have 'layer' and 'face' dimensions"
        assert 'face' in ds.dims,"Provided dataset must have 'layer' and 'face' dimensions"
    
    # infer n_subs and n_segs
    if state_vars is None: 
        state_vars = ds.sub.values

    n_subs = len(state_vars)
    
    n_segs = np.int32(len(ds.layer)*len(ds.face))
    
    # convert time to dt
    if 'time' in ds.dims:
        if extr_time is not None:
            dt = np.datetime64(extr_time)
        else:
            dt = max(ds.time.values)
            extr_time = pd.to_datetime(dt).strftime('%Y-%m-%d %H:%M')
        ds_t = ds.sel(time = dt)
    else:
        # assume ds already represents a single snapshot.
        # I don't think the time values really matters.
        dt = np.datetime64(hyd.time0)
        ds_t = ds
        pass
        
    dt_st = np.datetime64(hyd.time0)
    if start_time is not None:
        dt_st = np.datetime64(start_time)
     
    
    # construct output arrays 
    # now = pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')
    # txt_header = np.array('Restart file starting at {:s} ' \
    #                       'Written by stompy.model.delft.io.create_restart '\
    #                       'Written at {:s}'.format(extr_time, now),dtype='S160')

    txt_header = np.array(ds.header,dtype = 'S160')
    
    out_arr = np.zeros((n_segs,n_subs), dtype = 'f4' )  
    subs = np.array(['']*n_subs,dtype='S20')
    time = np.int32(pd.to_timedelta(dt-dt_st).total_seconds())

    for count,name in enumerate(state_vars):
        subs[count] = name
        # RH: Have to be careful about order. I don't want to create a bug
        # here, but also don't want to get inputs transposed.
        # assert the expected order. Ideally this would be 
        # out_arr[:,count] = ds_t[name].transpose('layer','face').values.flatten()
        # but I'm wary of making this change and creating a bug.
        assert ds_t[name].dims == ('layer','face'),"Dimension order issue"
        out_arr[:,count] = ds_t[name].values.flatten()

        # print(f"create_restart(): {name} has range {ds_t[name].values.min()} to {ds_t[name].values.max()}")
    
    # write out file 
    with open(res_fn,'wb') as fp:
        # header 
        fp.write(txt_header)
        
        # nosubs, noseg
        fp.write(np.array(n_subs, 'i4').tobytes())
        fp.write(np.array(n_segs, 'i4').tobytes())
        
        # parameter names 
        fp.write(subs) # is tobytes() implied here?
        
        # output time and substance values 
        fp.write(np.array(time, 'i4').tobytes())
        fp.write(out_arr.astype('f4'))
        
        fp.close() # PM - this shoudn't be necessary - but am having problems with file remanining open even after return
    
    return 


def read_map(fn,hyd=None,use_memmap=True,include_grid=True,return_grid=False,n_layers='hydro'):
    """
    Read binary D-Water Quality map output, returning an xarray dataset.

    fn: path to .map file
    hyd: waq_scenario.Hydro() object.  In the past this could be a path,
       but to avoid an apparent circular import, this must now be a
       Hydro object.
    use_memmap: use memory mapping for file access.  Currently
      this must be enabled.

    include_grid: the returned dataset also includes grid geometry, suitable
       for unstructured_grid.from_ugrid(ds).
       WARNING: there is currently a bug which causes this grid to have errors.
       probably a one-off error of some sort.

    n_layers: generally can be inferred from the hydro ('hydro'), but for delwaqg output this must
    be specified as an integer, or can be trusted from the map file by passing 'auto'

    note that missing values at this time are not handled - they'll remain as
    the delwaq standard -999.0.
    """

    # pycharm does not like the circular import, even when it's inside
    # a function like this, so until this all gets refactored, disallow
    # this feature
    assert hyd is not None,"Inferring hyd is disabled because of circular imports"
    assert not isinstance(hyd,six.string_types),"Must pass in Hydro() object, not path"

    # from . import waq_scenario as waq
    #
    # if not isinstance(hyd,waq.Hydro):
    #     if hyd==None:
    #         hyds=glob.glob( os.path.join(os.path.dirname(fn),"*.hyd"))
    #         assert len(hyds)==1,"hyd=auto only works when there is exactly 1 (not %d) hyd files"%(len(hyds))
    #         hyd=hyds[0]
    #     hyd=waq.HydroFiles(hyd)

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

        g=hyd.grid() # ignore message about ugrid.

        # not sure if there is a quicker way to get the number of layers
        if n_layers=='hydro':
            hyd.infer_2d_elements()
            n_layers=1+hyd.seg_k.max()
        elif n_layers=='auto':
            # This is quicker, but gives up an extra sanity check
            n_layers=int(n_segs / g.Ncells())

        assert g.Ncells()*n_layers == n_segs

        # I'm hoping that now we get 4 byte timestamps in reference seconds,
        # and then n_subs,n_segs chunks.
        # looks that way.
        data_start=fp.tell()

    bytes_left=nbytes-data_start
    framesize=(4+4*n_subs*n_segs)
    nframes,extra=divmod(bytes_left,framesize)
    if extra!=0:
        log.warning("Reading map file %s: %d or %d frames? bad length %d extra bytes (or %d missing)"%(
            fn,nframes,nframes+1,extra,framesize-extra))

    if hyd.n_2d_elements<=0:
        hyd.infer_2d_elements() # or just use g.Ncells()...
        
    # Specify nframes in cases where the filesizes don't quite match up.
    mapped=np.memmap(fn,[ ('tsecs','i4'),
                          ('data','f4',(n_layers,hyd.n_2d_elements,n_subs))] ,
                     mode='r',
                     shape=(nframes,),
                     offset=data_start)

    ds=xr.Dataset()

    try:
        txt_header=txt_header.decode()
    except AttributeError:
        pass # Hopefully header is already a string
    ds.attrs['header']=txt_header

    # a little python 2/3 misery
    try:
        substance_names=[s.decode() for s in substance_names]
    except AttributeError:
        pass

    ds['sub']= ( ('sub',), [s.strip() for s in substance_names] )

    times=utils.to_dt64(hyd.time0) + np.timedelta64(1,'s') * mapped['tsecs']

    # force ns to satisfy transient xarray warning.
    ds['time']=( ('time',), times.astype('M8[ns]') )
    ds['t_sec']=( ('time',), mapped['tsecs'] )

    for idx,name in enumerate(ds.sub.values):
        ds[name]= ( ('time','layer','face'), 
                    mapped['data'][...,idx] )
        ds[name].attrs['_FillValue']=-999

    if include_grid:
        # not sure why this doesn't work.
        g.write_to_xarray(ds=ds)

    if return_grid:
        return ds,g
    else:
        return ds

def map_add_z_coordinate(map_ds,total_depth='TotalDepth',coord_type='sigma',
                         layer_dim='layer'):
    """
    For an xarray representation of dwaq output, where the total depth
    has been recorded, add an inferred vertical coordinate in the dataset.
    This is necessary to allow the ugrid visit reader to understand
    the file.
    Currently only sigma coordinates, assumed to be evenly spaced, are
    supported.

    total_depth: Name of the xarray variable in map_ds holding total water column
    depth for each segment.
    coord_type: type of coordinate, currently must be "sigma".
    layer_dim: name of the vertical dimension in the data.

    Makes an arbitrary assumption that the first output time step is roughly
    mean sea level.  Obviously wrong, but a starting point.

    Given the ordering of layers in dwaq output, the sigma coordinate created 
    here is decreasing from 1 to 0.

    Modifies map_ds in place, also returning it.
    """
    assert coord_type=='sigma'

    bedlevel=-map_ds[total_depth].isel(**{layer_dim:0,'time':0,'drop':True})
    dry=(bedlevel==999)
    bedlevel[dry]=0.0
    map_ds['bedlevel']=bedlevel
    map_ds.bedlevel.attrs['units']='m'
    map_ds.bedlevel.attrs['positive']='up'
    map_ds.bedlevel.attrs['long_name']='Bed elevation relative to initial water level'

    tdepth=map_ds[total_depth].isel(**{layer_dim:0,'drop':True})
    eta=tdepth + map_ds.bedlevel
    eta.values[ tdepth.values==-999 ] = 0.0
    map_ds['eta']=eta
    map_ds.eta.attrs['units']='m'
    map_ds.eta.attrs['positive']='up'
    map_ds.eta.attrs['long_name']='Sea surface elevation relative initial time step'

    Nlayers=len(map_ds[layer_dim])
    # This is where sigma is made to be decreasing to capture the order of
    # layers in DWAQ output.
    map_ds['sigma']=(layer_dim,), (0.5+np.arange(Nlayers))[::-1] / float(Nlayers)
    map_ds.sigma.attrs['standard_name']="ocean_sigma_coordinate"
    map_ds.sigma.attrs['positive']='up'
    map_ds.sigma.attrs['units']=""
    map_ds.sigma.attrs['formula_terms']="sigma: sigma eta: eta  bedlevel: bedlevel"

    return map_ds

def dfm_wind_to_nc(wind_u_fn,wind_v_fn,nc_fn):
    """
    Transcribe DFM 'arcinfo' style gridded wind to
    CF compliant netcdf file (ready for import to erddap)

    Note that the order of rows in the DFM format is weird, and
    required bug fixes to this code 2017-12-21.  While dy is
    specified positive, the rows of data are written from north to
    south.  The DFM text file specifies coordinates for a llcorner
    and a dy, but that llcorner corresponds to the first column of
    the *last* row of data written out.

    wind_u_fn:
      path to the amu file for eastward wind
    wind_v_fn:
      path to the amv file for northward wind
    nc_fn:
      path to the netcdf file which will be created.
    """
    fp_u=open(wind_u_fn,'rt')
    fp_v=open(wind_v_fn,'rt')

    # read the header, gathering parameters in a dict.
    def parse_header(fp):
        params={}
        while 1:
            line=fp.readline().strip()
            if line.startswith('### START OF HEADER'):
                break
        for line in fp:
            line=line.strip()
            if line.startswith('#'):
                if line.startswith('### END OF HEADER'):
                    break
                continue # comment lines
            try:
                key,value = line.split('=',2)
            except ValueError:
                print("Failed to split key=value for '%s'"%line)
                raise
            key=key.strip()
            value=value.strip()

            # some hardcoded data type conversion:
            if key in ['n_rows','n_cols']:
                value=int(value)
            elif key in ['dx','dy','x_llcorner','y_llcorner','NODATA_value']:
                value=float(value)
            params[key]=value
        return params

    fp_u.seek(0)
    fp_v.seek(0)
    u_header=parse_header(fp_u)
    v_header=parse_header(fp_v)

    # make sure they match up
    for k in u_header.keys():
        if k in ['quantity1']:
            continue
        assert u_header[k] == v_header[k]

    # use netCDF4 directly, so we can stream it to disk
    import netCDF4

    os.path.exists(nc_fn) and os.unlink(nc_fn)
    nc=netCDF4.Dataset(nc_fn,'w') # don't worry about netcdf versions quite yet

    xdim='x'
    ydim='y'
    tdim='time'

    nc.createDimension(xdim,u_header['n_cols'])
    nc.createDimension(ydim,u_header['n_rows'])
    nc.createDimension(tdim,None) # unlimited

    # assign some attributes while we're at it
    for k in ['FileVersion','Filetype','dx','dy','grid_unit','unit1','x_llcorner','y_llcorner']:
        if k in u_header:
            setattr(nc,k,u_header[k])

    # cf conventions suggest this order of dimensions
    u_var = nc.createVariable('wind_u',np.float32,[tdim,ydim,xdim],
                              fill_value=u_header['NODATA_value'])
    v_var = nc.createVariable('wind_v',np.float32,[tdim,ydim,xdim],
                              fill_value=v_header['NODATA_value'])
    t_var = nc.createVariable('time',np.float64,[tdim])


    # install some metadata

    # parse the times into unix epochs for consistency
    t_var.units='seconds since 1970-01-01T00:00:00Z'
    t_var.calendar = "proleptic_gregorian"

    # Going to assume that we're working out of the same UTM 10:
    utm_var = nc.createVariable('UTM10',np.int32,[])
    utm_var.grid_mapping_name = "universal_transverse_mercator" 
    utm_var.utm_zone_number = 10
    utm_var.semi_major_axis = 6378137
    utm_var.inverse_flattening = 298.257 
    utm_var._CoordinateTransformType = "Projection" 
    utm_var._CoordinateAxisTypes = "GeoX GeoY" 
    utm_var.crs_wkt = """PROJCS["NAD83 / UTM zone 10N",
        GEOGCS["NAD83",
            DATUM["North_American_Datum_1983",
                SPHEROID["GRS 1980",6378137,298.257222101,
                    AUTHORITY["EPSG","7019"]],
                TOWGS84[0,0,0,0,0,0,0],
                AUTHORITY["EPSG","6269"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4269"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",-123],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","26910"]]
    """

    y_var=nc.createVariable('y',np.float64,[ydim])
    y_var.units='m'
    y_var.long_name="y coordinate of projection"
    y_var.standard_name="projection_y_coordinate"
    y_var[:]=u_header['y_llcorner'] + u_header['dy']*np.arange(u_header['n_rows'])

    x_var=nc.createVariable('x',np.float64,[xdim])
    x_var.units='m'
    x_var.long_name="x coordinate of projection"
    x_var.standard_name="projection_x_coordinate"
    x_var[:]=u_header['x_llcorner'] + u_header['dx']*np.arange(u_header['n_cols'])

    u_var.units='m s-1'
    u_var.grid_mapping='transverse_mercator'
    u_var.long_name='eastward wind from F Ludwig method'
    u_var.standard_name='eastward_wind'

    v_var.units='m s-1'
    v_var.grid_mapping='transverse_mercator'
    v_var.long_name='northward wind from F Ludwig method'
    v_var.standard_name='northward_wind'

    def read_frame(fp,header):
        # Assumes that the TIME line is alone,
        # but the pixel data isn't restricted to a particular number of elements per line.
        time_line=fp.readline()
        if not time_line:
            return None,None

        assert time_line.startswith('TIME')
        _,time_string=time_line.split('=',2)
        count=0
        items=[]
        expected=header['n_cols'] * header['n_rows']
        for line in fp:
            this_data=np.fromstring(line,sep=' ',dtype=np.float32)
            count+=len(this_data)
            items.append(this_data)
            if count==expected:
                break
            assert count<expected

        block=np.concatenate( items ).reshape( header['n_rows'],header['n_cols'] )
        # 2017-12-21: flip so that array index matches coordinate index.
        block=block[::-1,:]
        time=utils.to_dt64(time_string)
        return time,block

    frame_i=0
    while 1:
        u_time,u_block = read_frame(fp_u,u_header)
        v_time,v_block = read_frame(fp_v,v_header)
        if u_time is None or v_time is None:
            break

        assert u_time==v_time

        if frame_i%200==0:
            print("%d frames, %s most recent"%(frame_i,u_time))
        u_var[frame_i,:,:] = u_block
        v_var[frame_i,:,:] = v_block
        t_var[frame_i] = u_time

        # t... come back to it.
        frame_i+=1

    nc.close()


def dataset_to_dfm_wind(ds,period_start,period_stop,target_filename_base,
                        extra_header=None,min_records=1,
                        wind_u='wind_u',wind_v='wind_v',pres='pres'):
    """
    Write wind in an xarray dataset to a pair of gridded meteo files for DFM.

    ds:
      xarray dataset.  Currently fairly brittle assumptions on the format of
      this dataset, already in the proper coordinates system, coordinates of x and
      y, and the wind variables named wind_u and wind_v.
    period_start,period_stop:
      include data from the dataset on or after period_start, and up to period_stop,
    inclusive
    target_filename_base:
      the path and filename for output, without the .amu and .amv extensions.
    extra_header:
      extra text to place in the header.  This is included as is, with the exception that
      a newline will be added if it's missing

    returns the number of available records overlapping the requested period.
    If that number is less than min_records, no output is written.
    """
    time_idx_start = np.searchsorted(ds.time,period_start,side='left')
    # make stop inclusive by using side='right'
    time_idx_stop  = np.searchsorted(ds.time,period_stop,side='right')

    record_count=time_idx_stop-time_idx_start
    if record_count<min_records:
        return record_count

    # Sanity checks that there was actually some overlapping data.
    # maybe with min_records, this can be relaxed?  Unsure of use case there.
    assert time_idx_start+1<len(ds.time)
    assert time_idx_stop>0
    assert time_idx_start<time_idx_stop

    nodata=-999

    if extra_header is None:
        extra_header=""
    else:
        extra_header=extra_header.rstrip()+"\n"
        
    header_template="""### START OF HEADER
# Created with %(creator)s
%(extra_header)sFileVersion = 1.03
Filetype = meteo_on_equidistant_grid
NODATA_value = %(nodata)g
n_cols = %(n_cols)d
n_rows = %(n_rows)d
grid_unit = m
x_llcorner = %(x_llcorner)g
y_llcorner = %(y_llcorner)g
dx = %(dx)g
dy = %(dy)g
n_quantity = 1
quantity1 = %(quantity)s
unit1 = m s-1
### END OF HEADER
"""

    fp_u=open(target_filename_base+".amu",'wt')
    fp_v=open(target_filename_base+".amv",'wt')

    if pres in ds:
        fp_p=open(target_filename_base+".amp",'wt')
    else:
        fp_p=None

    base_fields=dict(creator="stompy",nodata=nodata,
                     n_cols=len(ds.x),n_rows=len(ds.y),
                     dx=np.median(np.diff(ds.x)),
                     dy=np.median(np.diff(ds.y)),
                     x_llcorner=ds.x[0],
                     y_llcorner=ds.y[0],
                     extra_header=extra_header)

    for fp,quant in [ (fp_u,'x_wind'),
                      (fp_v,'y_wind'),
                      (fp_p,'pres') ]:
        if fp is None:
            continue
        # Write the headers:
        fields=dict(quantity=quant)
        fields.update(base_fields)
        header=header_template%fields
        fp.write(header)

    count=0
    for time_idx in range(time_idx_start, time_idx_stop):
        count+=1
        if (time_idx-time_idx_start) % 96 == 0:
            log.info("Written %d/%d time steps"%( time_idx-time_idx_start,time_idx_stop-time_idx_start))
        u=ds['wind_u'].isel(time=time_idx)
        v=ds['wind_v'].isel(time=time_idx)
        if pres in ds:
            p=ds[pres].isel(time=time_idx)
        else:
            p=None

        t=ds.time.isel(time=time_idx)

        # write a time line formatted like this:
        # TIME=00000.000000 hours since 2012-08-01 00:00:00 +00:00
        time_line="TIME=%f seconds since 1970-01-01 00:00:00 +00:00"%utils.to_unix(t.values)

        for fp,data in [ (fp_u,u),
                         (fp_v,v),
                         (fp_p,p)]:
            if fp is None:
                continue
            # double check order.
            fp.write(time_line) ; fp.write("\n")
            # 2017-12-21: flip order of rows to suit DFM convention
            for row in data.values[::-1]:
                fp.write(" ".join(["%g"%rowcol for rowcol in row]))
                fp.write("\n")

    log.info("Wrote %d time steps"%count)

    fp_u.close()
    fp_v.close()
    if fp_p is not None:
        fp_p.close()
    return record_count

class SectionedConfig(object):
    """
    Handles reading and writing of config-file like formatted files.
    Follows some of the API of the standard python configparser
    """
    inline_comment_prefixes=('#',';')

    def __init__(self,filename=None,text=None):
        """
        filename: path to file to open and parse
        text: a string containing the entire file to parse
        """
        # This isn't being used anywhere -- delete?  and below.
        self.rows=[]    # full text of each line
        self.filename=filename
        self.base_path=None
        # For flags which do not get written into the config file.
        self.flags=defaultdict(lambda:None)

        if self.filename is not None:
            self.read(self.filename)
            self.base_path=os.path.dirname(self.filename)

        if text is not None:
            fp = StringIO(text)
            self.read(fp,'text')

    def set_filename(self,fn):
        """
        Updates self.filename and base_path, in anticipation of the file
        being written to a new location (this is so new file paths can be
        extrapolated before having to write this out)
        """
        self.filename=fn
        self.base_path=os.path.dirname(self.filename)

    def copy(self):
        return copy.deepcopy(self)

    def read(self, filename, label=None):
        if six.PY2:
            file_base = file
        else:
            file_base = io.IOBase

        if isinstance(filename, file_base):
            label = label or 'n/a'
            fp=filename
            filename=None
        else:
            fp = open(filename,'rt')
            label=label or filename

        # This isn't being used anywhere -- delete?
        # self.sources.append(label)

        for line in fp:
            # save original text so we can write out a new mdu with
            # only minor changes
            # the rstrip()s leave trailing whitespace, but strip newline or CR/LF
            self.rows.append(line.rstrip("\n").rstrip("\r"))
        if filename:
            fp.close()

    def entries(self):
        """ 
        Generator which iterates over rows, parsing them into index, section, key, value, comment.

        key is always present, but might indicate a section by including square
        brackets.
        value may be a string, or None.  Strings will be trimmed
        comment may be a string, or None.  It includes the leading comment character.
        Indices are not stable across set_value(), since new entries may get inserted into
        the middle of a section and shift other rows down.
        """
        section=None
        for idx,row in enumerate(self.rows):
            parsed=self.parse_row(row)

            if parsed[0] is None: # blank line
                continue # don't send back blank rows

            if self.is_section(parsed[0]):
                section=parsed[0]

            yield [idx,section] + list(parsed)

    def is_section(self,s):
        """
        Test whether the first item in the tuple returned by parse_row is the
        start of a section.
        """
        return s[0]=='['
        
    # experimental interface for files with duplicate sections
    def section_dicts(self):
        """
        Iterator over sections, returning a dictionary over each
        """
        sec=None
        for idx,section,key,value,comment in self.entries():
            if key==section: # new section
                if sec is not None:
                    yield sec
                # TODO: Make this a case-insensitive dict
                sec={'_section':section}
            else:
                sec[key]=value
        if sec is not None:
            yield sec

    # Start of section lines are assumed to have two groups:
    #  section name and option comment
    section_patt=r'^(\[[A-Za-z0-9 ]+\])([#;].*)?$'
    # value lines have key, value, and optional comment
    value_patt = r'^([A-Za-z0-9_ ]+)\s*=([^#;]*)([#;].*)?$'
    
    # 2019-12-31: appears that some mdu's written by delta shell have
    # lines near the top that are just an asterisk.  Assume
    # those are comments
    # blank lines can have a comment
    blank_patt = r'^\s*([\*#;].*)?$'
    # End of section lines may not exist, or subclasses may define
    # with an option comment.
    end_section_patt=None
    
    def parse_row(self,row):
        m_sec = re.match(self.section_patt, row)
        if m_sec is not None:
            return m_sec.group(1), None, m_sec.group(2)

        m_val = re.match(self.value_patt, row)
        if m_val is not None:
            return m_val.group(1).strip(), m_val.group(2).strip(), m_val.group(3)

        m_cmt = re.match(self.blank_patt, row)
        if m_cmt is not None:
            return None,None,m_cmt.group(1)

        if self.end_section_patt is not None:
            m_end_sec = re.match(self.end_section_patt, row)
            if m_end_sec is not None:
                # Gets return same as a comment
                return None,None,m_end_sec.group(1)

        print("Failed to parse row:")
        print(row)

    def format_section(self,s):
        return '[%s]'%s.lower()
    
    def get_value(self,sec_key):
        """
        return the string-valued settings for a given key.
        if they key is not found, returns None.
        If the key is present but with no value, returns the empty string
        """
        section=self.format_section(sec_key[0])
        key = sec_key[1].lower()

        for row_idx,row_sec,row_key,row_value,row_comment in self.entries():
            if (row_key.lower() == key) and (section.lower() == row_sec.lower()):
                return row_value
        else:
            return None
        
    def set_value(self,sec_key,value):
        # set value and optionally comment.
        # sec_key: tuple of section and key (section without brackets)
        # value: either the value (a string, or something that can be converted via str())
        #   or a tuple of value and comment, without the leading comment character
        section=self.format_section(sec_key[0])
        key=sec_key[1]

        last_row_of_section={} # map [lower_section] to the index of the last entry in that section

        if isinstance(value,tuple):
            value,comment=value
            comment=self.inline_comment_prefixes[0] + ' ' + comment
        else:
            comment=None

        value=self.val_to_str(value)

        def fmt(key,value,comment):
            return "%-18s= %-20s %s"%(key,value,comment or "")

        for row_idx,row_sec,row_key,row_value,row_comment in self.entries():
            last_row_of_section[row_sec]=row_idx

            if (row_key.lower() == key.lower()) and (section.lower() == row_sec.lower()):
                self.rows[row_idx] = fmt(row_key,value,comment or row_comment)
                return

        row_text=fmt(key,value,comment)
        if section in last_row_of_section:
            # the section exists
            last_idx=last_row_of_section[section]
            self.rows.insert(last_idx+1,row_text)
        else: # have to append the new section
            self.rows.append(section)
            self.rows.append(row_text)

    def __setitem__(self,sec_key,value): # self[sec_key]=value
        self.set_value(sec_key,value)
    def __getitem__(self,sec_key):       # self[sec_key]
        return self.get_value(sec_key)

    def __contains__(self,sec_key):
        """
        Return true if the given section or (section,key) tuple 
        exists.  Note that [currently] if the entry exists but is
        empty, this still returns True.
        """
        if isinstance(sec_key,tuple):
            section=self.format_section(sec_key[0])
            key = sec_key[1].lower()
        else:
            section=self.format_section(sec_key)
            key=None
            
        for row_idx,row_sec,row_key,row_value,row_comment in self.entries():
            if ( ((key is None) or (row_key.lower() == key))
                 and (section.lower() == row_sec.lower())):
                return True
        else:
            return False

    def __delitem__(self,sec_key):
        if isinstance(sec_key,tuple):
            section=self.format_section(sec_key[0])
            key = sec_key[1].lower()
        else:
            section=self.format_section(sec_key)
            key=None

        new_rows=[]
        row_sec=None
        
        for row_idx,row in enumerate(self.rows):
            row_key=None
            parsed=self.parse_row(row)

            if parsed[0] is None: # blank line
                new_rows.append(row)
                continue # don't send back blank rows

            if self.is_section(parsed[0]):
                row_sec=parsed[0]

                if (row_sec == section) and (key is None):
                    # delete this row.
                    continue
                else:
                    new_rows.append(row)
                    continue
            else:
                row_key=parsed[0]
                if ( ((key is None) or (row_key.lower() == key))
                     and (section.lower() == row_sec.lower())):
                    # delete this row
                    continue
                else:
                    new_rows.append(row)
        self.rows=new_rows
        
    def filepath(self,sec_key):
        """
        Lookup a filename via a ('section','name') tuple, and
        return the full filename including base path.
        if the key does not exist or is empty, return None.
        """
        val=self.get_value(sec_key)
        if not val:
            return None
        if self.base_path:
            return os.path.join(self.base_path,val)
        else:
            return val

    def val_to_str(self,value):
        # make sure that floats are formatted with plenty of digits:
        # and handle annoyance of standard Python types vs. numpy types
        # But None stays None, as it gets handled specially elsewhere
        if value is None:
            return None
        if isinstance(value,float) or isinstance(value,np.floating):
            return "%.12g"%value
        else:
            return str(value)

    def write(self,filename=None):
        """
        Write this config out to a text file.
        filename: defaults to self.filename
        check_changed: if True, and the file already exists and is not materially different,
          then do nothing.  Good for avoiding unnecessary changes to mtimes.
        backup: if true, copy any existing file to <filename>.bak
        """
        if filename is None:
            filename=self.filename
        with open(filename,'wt') as fp:
            for line in self.rows:
                fp.write(line)
                fp.write("\n")

class MDUFile(SectionedConfig):
    """
    Read/write MDU files, with an interface similar to python's
    configparser, but better support for discerning and retaining
    comments
    """
    @property
    def name(self):
        """
        base name of mdu filename, w/o extension, which is used in various other filenames.
        """
        base=os.path.basename(self.filename)
        assert base.endswith('.mdu'),"Not sure what dfm does in this case"
        return base[:-4]
    def output_dir(self):
        """
        path to the folder holding DFM output based on MDU filename
        and contents.
        """
        output_dir=self['Output','OutputDir']
        if output_dir in (None,""):
            output_dir="DFM_OUTPUT_%s"%self.name

        return os.path.join(self.base_path,output_dir)

    def time_range(self):
        """
        return tuple of t_ref,t_start,t_stop
        as np.datetime64
        """
        t_ref=utils.to_dt64( datetime.datetime.strptime(self['time','RefDate'],'%Y%m%d') )
        dt=self.t_unit_td64()
        # float() is a bit dicey. Some older np doesn't like mixing floats
        # and datetimes.
        t_start = t_ref+float(self['time','tstart'])*dt
        t_stop = t_ref+float(self['time','tstop'])*dt
        return t_ref,t_start,t_stop

    def t_unit_td64(self,default='S'):
        """ Return Tunit as timedelta64.  If none is set, set to default
        """
        t_unit=self['time','Tunit']
        if t_unit is None:  # or does the above throw an error?
            self['time','Tunit']=t_unit=default

        if t_unit.lower() == 'm':
            dt=np.timedelta64(60,'s')
        elif t_unit.lower() == 's':
            dt=np.timedelta64(1,'s')
        elif t_unit.lower() == 'h':
            dt=np.timedelta64(3600,'s')
        else:
            raise Exception("Bad time unit %s"%t_unit)

        return dt

    def set_time_range(self,start,stop,ref_date=None):
        if ref_date is not None:
            # Make sure ref date is integer number of days
            assert ref_date==ref_date.astype('M8[D]')
        else:
            # Default to truncating the start date
            ref_date = start.astype('M8[D]')

        self['time','RefDate'] = utils.to_datetime(ref_date).strftime('%Y%m%d')

        dt=self.t_unit_td64()
        self['time','TStart'] = int( (start - ref_date)/ dt )
        self['time','TStop'] = int( (stop - ref_date) / dt )

    def partition(self,nprocs,dfm_bin_dir=None,mpi_bin_dir=None):
        if nprocs<=1:
            return

        # As of r52184, explicitly built with metis support, partitioning can be done automatically
        # from here.
        if mpi_bin_dir is None:
            mpi_bin_dir=dfm_bin_dir

        dflowfm="dflowfm"
        gen_parallel="generate_parallel_mdu.sh"
        if dfm_bin_dir is not None:
            dflowfm=os.path.join(dfm_bin_dir,dflowfm)
            gen_parallel=os.path.join(dfm_bin_dir,gen_parallel)

        mpiexec="mpiexec"
        if mpi_bin_dir is not None:
            mpiexec=os.path.join(mpi_bin_dir,mpiexec)

        cmd="%s -n %d %s --partition:ndomains=%d %s"%(mpiexec,nprocs,dflowfm,nprocs,
                                                      self['geometry','NetFile'])
        pwd=os.getcwd()
        try:
            os.chdir(self.base_path)
            res=subprocess.call(cmd,shell=True)
        finally:
            os.chdir(pwd)

        # similar, but for the mdu:
        cmd="%s %s %d 6"%(gen_parallel,os.path.basename(self.filename),nprocs)
        try:
            os.chdir(self.base_path)
            res=subprocess.call(cmd,shell=True)
        finally:
            os.chdir(pwd)

    def get_bool(self,sec_key,key=None):
        """
        missing, or numeric equal to 0 => False
        present and nonzero numeric => True
        """
        if key is not None:
            sec_key=(sec_key,key)
        value=self[sec_key]
        if value is None:
            value=0.0
        else:
            value=float(value)
        return value!=0.0
            

def exp_z_layers(mdu,zmin=None,zmax=None):
    """
    This will probably change, not very flexible now.
    For singly exponential z-layers, return zslay, positive up, starting
    from the bed.  first element is the bed itself.

    zmin: deepest depth, positive up.  Defaults to ds.NetNode_z.min(),
       read from the net file specified in mdu.
    zmax: top of water column.  Defaults to WaterLevIni in mdu.
    """

    if zmax is None:
        zmax=float(mdu['geometry','WaterLevIni'] )
    if zmin is None:
        ds=xr.open_dataset(mdu.filepath(['geometry','NetFile']))
        zmin=float(ds.NetNode_z.min())
        ds.close()
        
    kmx=int(mdu['geometry','kmx'])
    coefs=[float(s) for s in mdu['geometry','StretchCoef'].split()] # 0.002 0.02 0.8

    ztot=zmax-zmin

    # From unstruc.F90:
    dzslay=np.zeros(kmx,'f8')
    zslay=np.zeros(kmx+1,'f8')

    gfi = 1.0 / coefs[1] # this shouldn't do anything
    gf  = coefs[2]
    mx=kmx
    k1=int( coefs[0]*kmx) 

    gfk = gfi**k1

    if gfk == 1.0:
        gfi = 1.0
        dzslay[0] = 1.0 / mx
    else:
        dzslay[0] = ( 1.0 - gfi ) / ( 1.0 - gfk )* coefs[0]

    for k in range(1,k1):
        dzslay[k] = dzslay[k-1] * gfi

    gfk = gf**(kmx-k1)
    if k1 < kmx:
        if gfk == 1.0:
            gf = 1.0
            dzslay[k1] = 1.0 / mx
        else:
            dzslay[k1] = ( 1.0 - gf ) / ( 1.0 - gfk ) * ( 1.0 - coefs[0] )

        for k in range(k1+1,mx):
            dzslay[k] = dzslay[k-1] * gf

    zslay[0] = zmin
    for k in range(mx):
        zslay[k+1] = zslay[k] + dzslay[k] * (zmax-zmin)


    return zslay
        

def read_bnd(fn):
    """
    Parse DWAQ-style boundary data file
    
    Returns a list [ ['boundary_name',array([ boundary_link_idx,[[x0,y0],[x1,y1]] ])], ...]
    """
    with open(fn,'rt') as fp:
        toker=inp_tok(fp)
        token=lambda: six.next(toker)

        N_groups=int(token())
        groups=[]
        for group_idx in range(N_groups):
            group_name=token()
            N_link=int(token())
            links=np.zeros( N_link,
                            dtype=[ ('link','i4'),
                                    ('x','f8',(2,2)) ] )
            for i in range(N_link):
                links['link'][i]=int(token())
                links['x'][i,0,0]=float(token())
                links['x'][i,0,1]=float(token())
                links['x'][i,1,0]=float(token())
                links['x'][i,1,1]=float(token())
            groups.append( [group_name,links] )
    return groups

def write_bnd(bnd,fn):
    with open(fn,'wt') as fp:
        fp.write("%10d\n"%len(bnd))
        for name,segs in bnd:
            fp.write("%s\n"%name)
            fp.write("%10d\n"%len(segs))
            for seg in segs:
                x=seg['x']
                fp.write("%10d  %.7f  %.7f   %.7f  %.7f\n"%(seg['link'],
                                                            x[0,0],x[0,1],x[1,0],x[1,1]))


def read_dfm_tim(fn, ref_time, time_unit='M', columns=None):
    """
    Parse a tim file to xarray Dataset.  Must pass in the reference
    time (datetime64, or convertable to that via utils.to_dt64())
    and the time_unit ('s' or 'm')

    time_unit: 'S' for seconds, 'M' for minutes.  Relative to model reference
    time.  Probably ought to be 'M' always.

    returns Dataset with 'time' dimension, and data columns labeled according
    to columns (list of strings naming the columns after the time stamp).
    Defaults to val1, val2,... and if specified columns is not long enough
    # valN, valN+1, etc. will be appended.
    """
    if time_unit.lower()=='m':
        dt=np.timedelta64(60,'s')
    elif time_unit.lower()=='s':
        dt=np.timedelta64(1,'s')
    else:
        raise Exception("Bad time unit %s"%time_unit)

    if not isinstance(ref_time,np.datetime64):
        ref_time=utils.to_dt64(ref_time)

    raw_data=np.loadtxt(fn)
    t=ref_time + dt*raw_data[:,0]

    if columns is None:
        columns=[]
        
    # fill in defaults to make sure it's long enough.
    for i in range(len(columns), raw_data.shape[1]-1):
        columns.append('val%d'%(i+1))
        
    ds=xr.Dataset()
    # 2023-05-16: force conversion to nanosecond to avoid casting warning.
    ds['time']=('time',),t.astype('M8[ns]')
    for col_i in range(1,raw_data.shape[1]):
        ds[columns[col_i-1]]=('time',),raw_data[:,col_i]

    ds.attrs['source']=fn

    return ds

def read_dfm_bc(fn):
    """
    Parse DFM new-style BC file, returning a hash of
    Name => xarray dataset.
    """
    bcs={} # indexed by Name

    import re
    #with open(fn,'rt') as fp:w
    fp=open(fn,'rt') # during DEV
    if 1:
        # pre-read a line, and always keep the next line
        # in this variable for some low-budget lookahead
        line=fp.readline()

        while 1: # looping over datasets
            ds=xr.Dataset()
            ds.attrs['source']=fn

            # Eat blank lines
            while line and (line.strip()==""):
                line=fp.readline()

            if not line:
                break # end of file.  could be empty file.

            assert line.strip().lower()=='[forcing]',"Expected [forcing], got %s in %s"%(line,fn)
            line=fp.readline()

            quantities=[]
            curr_quantity={} # hash of quantity, unit
            def push_quantity():
                # Sanity check, make sure we have the bare minimum to define
                # a quantity.
                assert 'quantity' in curr_quantity
                quantities.append(dict(curr_quantity))
                curr_quantity.clear()

            while 1: # reading key-value pairs
                m=re.match(r'^([^=]+)=([^=]*)$',line)
                if m is not None: # key-value pair
                    key=m.group(1).strip().lower()
                    value=m.group(2).strip()
                    if key in ['quantity','unit']:
                        if key in curr_quantity:
                            # already seen this key, so must be a new quantity.
                            # push the old onto the list
                            push_quantity()
                        # record the new
                        curr_quantity[key]=value
                    else:
                        ds.attrs[key]=value
                    line=fp.readline()
                else:
                    # Not a key-value line.  Must be data
                    if curr_quantity:
                        push_quantity()
                    break # ready for data
            # Reading data
            columns=[ list() for q in quantities ]
            while 1:
                if line.strip()=="": # eat blank lines
                    pass
                elif line.strip().lower()=="[forcing]": # start of next block
                    break
                else:
                    items=line.strip().split()
                    assert len(items)==len(quantities)
                    for v,col in zip(items,columns):
                        col.append(float(v))
                line=fp.readline()
                if not line: # end by way of end-of-file
                    break
            for q,col in zip(quantities,columns):
                var_name=q['quantity']
                ds[var_name]=('time',),np.array(col)
                for k in q.keys():
                    if k!='quantity':
                        # everything else becomes an attribute
                        ds[var_name].attrs[k] = q[k]
            if 'time' in ds:
                if ('unit' in ds['time'].attrs) and ('units' not in ds['time'].attrs):
                    ds['time'].attrs['units']=ds['time'].attrs['unit']

                # This actually should line up with CF conventions
                # pretty well.  Give xarray a chance to parse the
                # dates
                try:
                    ds=xr.decode_cf(ds)
                except TypeError:
                    # Really no idea what kinds of exceptions may crop up there.
                    log.debug("While decoding time data",exc_info=True)
                    # fall through, which will leave original ds intact.
            bcs[ds.attrs['name']]=ds
    return bcs


    
    
def read_meteo_on_curvilinear(fn,run_dir,flip_grid_rows=True):
    # The orientation of the data is confusing
    # Using hac_linear_wind_2022_bloom.tem and meteo_coarse.grd,
    # The grid places node=0 == curvilinear node [0,0] at the physical lower left
    # corner.
    # 
    # The data includes:
    # first_data_value =    grid_llcorner
    #
    # Comparing these inputs to Tair in the DFM output (i.e. on the computational
    # grid for which orientation is not ambiguous), I get correct alignment when
    # plotting the gridded data with imshow(origin='upper').

    # On the day in question, the first temperature value in the .tem file is 15.5
    # this comes through as the [0,0] value in the dataset parsed below.
    # This value appears to be in the upper-left corner of the domain.

    # Confusing. Either (a) DFM ignores some of the metadata, and the behavior is
    # just arbitary, or (b) DFM expects the curvilinear mesh to place its origin at the
    # upper left, so 'grid_llcorner' really means "Data rows are flipped relative to grid"

    # flip_grid_rows: flip the grid vertically to make it agree with the data. 
    # Rearrange the grid to match the data, and copy pixel x,y values to the dataset
    # To that end,
    header={}
    with open(fn,'rt') as fp:
        l=fp.readline()
        assert l.startswith('###') # START OF HEADER
        
        while 1:
            line=fp.readline()
            if line=='':
                print(f"Early EOF reading {fn}")
                return None

            if line.startswith('###'): # END OF HEADER
                break

            line=line.split('#')[0].strip()
            if line=='': continue
            lval,rval = line.split('=')
            header[lval.strip()] = rval.strip()

        quantities=[]
        for i in range(10):
            hdr_name=f'quantity{i+1}'
            hdr_unit=f'unit{i+1}'
            if hdr_name in header:
                quantities.append( dict(name=header[hdr_name], units=header[hdr_unit]))
            else:
                break

        assert header['filetype'] == 'meteo_on_curvilinear_grid'

        grid_fn=os.path.join(run_dir,header['grid_file'])
        grid = unstructured_grid.UnstructuredGrid.read_delft_curvilinear(grid_fn,dep_fn=None,enc_fn=None)

        # Data is on nodes, so drop cells to avoid confusion.
        for c in grid.valid_cell_iter():
            grid.delete_cell(c)

        shape = grid.rowcol_to_node.shape # TRANSPOSITION WOES
            
        if flip_grid_rows:
            n_row,n_col = shape
            grid.rowcol_to_node = grid.rowcol_to_node[::-1,:]
            grid.nodes['row'] = n_row - grid.nodes['row'] - 1

        values_per_frame = grid.Nnodes()
        frames=[]
        while True:
            # Read a frame:
            ds=xr.Dataset()
            line=fp.readline()
            if line=='': break
            ds['time_string'] = line.strip()
            time_val=line.split('=')[1].strip()
            val_units=time_val.split(' ',1)
            ds['time']=float(val_units[0])
            ds['time'].attrs['units'] = val_units[1]
            ds['time'].attrs['calendar'] = "proleptic_gregorian"
            ds=xr.decode_cf(ds)
            
            for quantity in quantities:
                raveled=[]
                data_count=0
                while data_count < values_per_frame:
                    line=fp.readline()
                    assert line!=""
                    values = [float(s) for s in line.strip().split()]
                    raveled.append(values)
                    data_count+=len(values)
                assert data_count==values_per_frame
                data=np.concatenate(raveled).reshape(shape)
                ds[quantity['name']] = ('row','col'), data
            frames.append(ds)

        ds=xr.concat(frames, dim='time')
        for key in header:
            ds.attrs[key] = header[key]

        xy = grid.nodes['x'][grid.rowcol_to_node]
        ds['x'] = ('row','col'),xy[:,:,0]
        ds['y'] = ('row','col'),xy[:,:,1]        
        ds['quantity'] = ('quantity',), [q['name'] for q in quantities]
            
    return ds # ,grid
            
                
def rewrite_meteo_on_curvilinear(orig_fn,new_fn,ds):
    """
    Copy a curvilinear met file, but replace the data.
    orig_fn: path to curvilinear met file, as passed to read_meteo_on_curvilinear()
    new_fn: path to the new file to create
    ds: a dataset as returned by read_meteo_on_curvilinear()
    """
    header_count=0 # how many times we've seen '###'
    with open(orig_fn,'rt') as fp_old:
        with open(new_fn,'wt') as fp_new:
            while 1:
                l=fp_old.readline()
                assert l!="","Premature end of file"
                if l.startswith('###'): header_count+=1
                fp_new.write(l)

                if header_count==2:
                    break

            quantities=ds.quantity.values
            for tidx,time_string in utils.progress(enumerate(ds.time_string.values)):
                fp_new.write(time_string)
                fp_new.write("\n")
                for quantity in quantities:
                    data=ds[quantity].isel(time=tidx).values
                    s=(np.array2string(data,max_line_width=2000,separator=' ', threshold=1e9, formatter={'float': lambda f: f"{f:.1f}"})
                       .replace(' [','')
                       .replace('[','')
                       .replace(']','') )
                    fp_new.write(s)
                    fp_new.write("\n")


