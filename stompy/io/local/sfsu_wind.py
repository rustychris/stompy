"""
Methods related to wind data as formatted and provided by
SFSU via www.met.sfsu.edu/~sinton/winds/
"""
from __future__ import print_function
import datetime
import pandas as pd
import numpy as np
import xarray as xr
import glob

from stompy import utils

class Tok(object):
    def __init__(self,fp):
        self.fp=fp
        self.line=None
    def tok(self,mode='atom',sep=None):
        while 1:
            if self.line is None:
                self.line=self.fp.readline()
                if self.line=="":
                    self.line=None
                    return None
            if self.line.strip()=="":
                self.line=None
                continue

            if mode=='line':
                val=self.line.rstrip() # drop the newline
                self.line=None
            elif mode=='atom':
                parts = self.line.split(sep,1)
                if len(parts)>1:
                    val,self.line=parts
                else:
                    val=parts[0]
                    self.line=None
            else:
                assert("Mode must be line or atom")
            val=val.strip()
            if val=="":
                val=None
            if val is not None:
                return val
    def __call__(self,*a,**k):
        return self.tok(*a,**k)
        

# def read_daily(fn):
def read_timestamp(tok):
    """ read an 'hour month day year' timestamp
    """
    while 1:
        line=tok('line')
        if line is None:
            return None
        try:
            vals=[int(s) for s in line.split()]
            hour,month,day,year=vals
            break
        except IndexError:
            pass
        except ValueError:
            pass
    return utils.to_dt64(datetime.datetime(year,month,day,hour,0))
    
def read_wind_rec(tok):
    x=float(tok())
    y=float(tok())
    press=tok()
    if press=="99.9":
        press=np.nan
    else:
        press=float(press)
    temp=float(tok())
    if temp==-99:
        temp=np.nan
    direction=float(tok())
    speed=float(tok())
    name=tok('line')
    return dict(x=x,y=y,press=press,temp=temp,direction=direction,speed=speed,name=name)
    

def read_one_time(tok):
    timestamp0=read_timestamp(tok)
    if timestamp0 is None:
        return None

    unknown=tok('line') # 6 0 0 0 0
    unknown=tok('line') # 6 0 0 0 0

    count=int(tok())


    wind_recs=[]
    for i in range(count):
        wind_recs.append(read_wind_rec(tok))
    winds=pd.DataFrame(wind_recs)

    # Now it gets very ad hoc
    x_prof=float(tok())
    y_prof=float(tok())
    n_prof=int(tok())
    prof=[]
    for i in range(n_prof):
        prof.append( dict( elev_ft=float(tok()), direc=float(tok()), speed_kts=float(tok())) )
    prof1=pd.DataFrame(prof)

    # Sounding
    sounding_name=tok('line') # Sounding Title: OC101400.OAK
    x_sound=float(tok())
    y_sound=float(tok())
    n_sound=int(tok())
    cols=int(tok())
    col_names=[tok(sep='  ').replace('\xb0','deg') for i in range(cols)]
    sounding=[]
    for i in range(n_sound):
        rec={}
        for col_name in col_names:
            rec[col_name]=float(tok(sep='  '))
        sounding.append(rec)
    sounding=pd.DataFrame(sounding)

    # The second wind profile is not always present, and seems to be the
    # same data as the first
    if 0:
        # Second wind profile
        tok('line') # ht (ft)  DD   FF (kt)
        n_prof2=int(tok())
        prof2=[]
        for i in range(n_prof2):
            prof2.append( dict( elev_ft=float(tok()),
                                direc=float(tok()),
                                speed_kts=float(tok()) ) )
        prof2=pd.DataFrame(prof2)
        # in theory we end up exactly at the full timestamp line
        full_timestamp_str=tok('line')
    else:
        # otherwise scan for the full timestamp line
        while 1:
            line=tok('line')
            if ' GMT ' in line:
                full_timestamp_str=line
                break
        prof2=None
        
    hhmmss=full_timestamp_str.split()[3]
    hour,minute,second=[ int(s) for s in hhmmss.split(':') ]
    full_timestamp=timestamp0+(hour*3600 + minute*60 + second)*np.timedelta64(1,'s')
    
    comment1=tok('line')
    comment2=tok('line')
    # The number and meaning of these trailing records is unclear
    # have the start of this method scan for a timestamp instead of peeking ahead here.
    # comment3=tok(),tok('line')

    rec=dict(winds=winds,
             prof1=prof1,
             prof2=prof2,
             sounding=sounding,
             timestamp0=timestamp0,
             timestamp=full_timestamp,
             full_timestamp_str=full_timestamp_str)
    return rec


def read_wnd(fn):
    fp=open(fn,'rt')

    tok=Tok(fp)

    recs=[]
    while 1:
        rec=read_one_time(tok)
        if rec is None:
            break
        recs.append(rec)

    return recs



all_wind_recs=[]
for wnd_fn in glob.glob('data/*.wnd'):
    print(wnd_fn)
    recs=read_wnd(wnd_fn)

    # For the current purposes, extract just the surface wind data
    daily_recs=[]
    for rec in recs:
        winds=rec['winds']
        winds['time']=rec['timestamp']
        daily_recs.append(winds)

    daily_recs=pd.concat(daily_recs)
    all_wind_recs.append(daily_recs)
all_winds=pd.concat(all_wind_recs)

locs=all_winds.groupby('name').first()
summary=locs[ ['x','y'] ]

## 

# Make a summary shapefile of that:
from shapely import geometry
from stompy.spatial import wkb2shp

pnts=[ geometry.Point( rec.x * 1000, rec.y*1000 )
       for idx,rec in summary.iterrows() ]

fields=np.zeros(len(pnts),[ ('name','O'),
                            ('x','f8'),
                            ('y','f8') ])

for i,(idx,rec) in enumerate(summary.iterrows()):
    fields['name'][i]=idx
    fields['x'][i]=rec.x
    fields['y'][i]=rec.y

wkb2shp.wkb2shp('wind_locations.shp',pnts,
                overwrite=True,
                fields=fields)
