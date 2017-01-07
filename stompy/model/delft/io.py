import numpy as np
import pandas as pd
import re

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
        matches=re.findall(r'\s*((\'[^\']+\')|([/:-a-zA-Z_#0-9\.]+))', line)
        for m in matches:
            yield m[0]

def parse_inp_transects(inp_file):
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
        if int(next(tokr)) == 1: # monitoring points used
            nmon = int(next(tokr))
            for imon in range(nmon):
                name, segcount=next(tokr),int(next(tokr))
                for iseg in range(segcount):
                    next(tokr)
        transects={} # name => list of signed, 1-based exchanges
        if int(next(tokr)) == 1: # transects used
            ntrans=int(next(tokr))
            for itrans in range(ntrans):
                name,style,ecount = next(tokr),next(tokr),int(next(tokr))
                exchs=[int(next(tokr)) for _ in range(ecount)]
                transects[name.strip("'")]=exchs
    return transects

