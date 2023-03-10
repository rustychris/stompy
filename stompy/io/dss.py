# Emulate reading record via DSSUTL2, but using
# command line interface to DSSVUE.
import datetime as dt
import subprocess
import datetime
import os
import numpy as np
import pandas as pd
from .. import memoize, utils

class DssReader(object):
    '''
    Class to handle reading DSS files with only access to HEC DSS Vue binary.
    '''
    bad_data_value = -999999.

    script_fn="hec_script.py"
    bad_data=[]
    csv_out="output.csv"

    # path to hec-dssvue.sh on Linux
    hec_dssvue_default="hec-dssvue.sh"
    hec_dssvue=None # fallback to DSSVUE from environment, and hec_dssvue_default failing that.

    vue_script=r"""
# HEC-DSSVUE script to emulate dssutl2.exe
import java
import datetime
from hec.heclib.dss import HecDss

hecfile=HecDss.open("%(dss_file_name)s",1) # must exist
data=hecfile.get("%(dss_path)s",True) # True to read full time series

with open("%(csv_out)s",'w') as fp:
    fp.write("time,value\n")
    times = data.getTimes()
    for i,V in enumerate(data.values):
        t = times.element(i)
        # Note that hour is sometimes 24!
        tstr="%%d-%%02d-%%02d %%02d:%%02d"%%( t.year(),t.month(),t.day(),
                                        t.hour(),t.minute() )
        fp.write(tstr+",%%.6f\n"%%V)
    
hecfile.done()
"""
    
    def __init__(self, dss_file_name, **kwargs):
        self.__dict__.update(kwargs)
        self.dss_file_name = dss_file_name

    def read_DSS_record(self, record_name, start_time=None, end_time=None):
        df=self.worker(record_name,start_time,end_time)
        if len(df)==0:
            log.warning("DSS path %s had no records"%(record_name))
            return [[],[]]

        date_sel=df.time.notnull()
        if start_time is not None:
            date_sel&=(df['time'].values>=np.datetime64(start_time))
        if end_time is not None:
            date_sel&=(df['time'].values<=np.datetime64(end_time))
            
        df=df[date_sel]

        bad=df['value'].isin(self.bad_data)
        df.loc[bad,'value']=self.bad_data_value
        
        return df

    def worker(self,record_name,tstart,tend):
        args=dict(dss_file_name=self.dss_file_name,
                  dss_path=record_name,
                  csv_out=self.csv_out)

        with open(self.script_fn,'wt') as fp:
            fp.write(self.vue_script%args)

        if os.path.exists(args['csv_out']):
            os.unlink(args['csv_out'])

        #subprocess.call([self.hec_dssvue,self.script_fn])
        self.stdout=subprocess.check_output([self.get_hec_dssvue(),self.script_fn],
                                            stderr=subprocess.STDOUT)
        df=pd.read_csv(args['csv_out'])

        # Come back to fix up dates:
        def parse_date(s):
            Y=int(s[:4])
            m=int(s[5:7])
            d=int(s[8:10])
            H=int(s[11:13])
            M=int(s[14:16])

            t=datetime.datetime(Y,m,d,0,M) + datetime.timedelta(hours=H)
            return t

        df['time']=df['time'].apply(parse_date)

        return df
    
    def get_hec_dssvue(self):
        if self.hec_dssvue is not None:
            return self.hec_dssvue
        if 'DSSVUE' in os.environ:
            return os.environ['DSSVUE']
        return self.hec_dssvue_default
    
    # placeholder for future interface to reuse instance
    def open(self):
        pass
    def close(self):
        pass


def read_records(filename, dss_path,
                 start_time=None, end_time=None,
                 cache_dir=None,hec_dssvue=None):
    if cache_dir is not None:
        # put full information in the key
        key=memoize.memoize_key(filename,dss_path,start_time,end_time)
        # and make at least the filename legible in the cache file
        # so it's easier to clear out cache files manually.
        cache_file=os.path.join(cache_dir,
                                f"dss-{os.path.basename(filename)}-{key}")

        if not utils.is_stale(cache_file,filename):
            with open(cache_file,'rb') as fp:
                return pickle.load(fp)
    else:
        cache_file=None

    reader=DssReader(filename,hec_dssvue=hec_dssvue)
    
    data=reader.read_DSS_record(dss_path)
    if cache_file is not None:
        with open(cache_file,'wb') as fp:
            pickle.dump(data,fp)
    return data

            
