"""
Tools for reading RDB files, the text-based format often used in USGS 
data.  See stompy/test/data for examples of this type of data.
"""

from __future__ import print_function

from functools import reduce

# the 2to3 stuff added some other cruft which I think actually makes it all worse...

# python import routines for rdb (USGS tab delimited) format

import re,string,time
import six
import io
import pytz

import datetime
import numpy as np
import xarray as xr
from matplotlib.dates import date2num,num2date

try:
    import cPickle as pickle
except ImportError:
    import pickle
    
from .rdb_datadescriptors import dd_to_synonyms

from .. import utils
from . import rdb_codes

class Rdb(object):
    record_count = 0
    
    aggregate=True # combine fields that don't change into a scalar
    
    def __init__(self,
                 text=None,
                 source_file=None,
                 fp=None,
                 **kw):
        utils.set_keywords(self,kw)
        self.source_filename = source_file
        if text is not None:
            try:
                text=text.decode()
            except AttributeError:
                pass
        self.text = text
        self.fp = fp

        if self.fp is None:
            if self.text is not None:
                self.fp = io.StringIO(self.text)
            elif self.source_filename:
                self.fp = open(self.source_filename,'rt')
        else:
            # coerce fp to be text, not binary
            try:
                self.fp.encoding
            except AttributeError:
                self.fp=io.StringIO(fp.read().decode())
        self.parse_source_file()

    def float_or_nan(self,s):
        # Not sure what Eqp or Mnt mean, but will assume that's
        # missing data.
        if s in (None,'','Eqp','Mnt','***'):
            return np.nan
        else:
            return float(s)

    def data(self):
        """ assuming that only one data type was requested, try to figure out which
        column it is, and return that data
        for single-valued columns, this will expand the data out to be the right
        length
        """
        for k in list(self.keys()):
            if re.match('^[0-9_]+$',k):
                d = self[k]
                try:
                    len(d)
                    return d
                except TypeError:
                    return d*ones(self.record_count)
            
        print(list(self.keys()))
        raise Exception("None of the keys looked good")
            
    def parse_date(self,s):
        """ parse a date like '2008-01-13 00:31' into a float representing
            absolute days since 0ad """

        d=None
        for fmt in ["%Y-%m-%d %H:%M","%Y-%m-%d"]:
            try:
                d = datetime.datetime.strptime(s,fmt)
            except ValueError:
                pass
        if d:
            return date2num(d)
        else:
            return None
        
    def parse_source_file(self):
        # eat the comments
        line = None

        preamble_lines=[]
        
        for line in self.fp:
            if line[0]!='#':
                break
            preamble_lines.append(line)

        self.preamble="".join(preamble_lines)
        
        if line is None:
            print("No data in source!")
            print("text:",self.text)
            print("source_filename:",self.source_filename)
            raise Exception("Empty RDB data?!")

        headers = line.strip().split("\t")
        try:
            specs = next(self.fp).strip().split("\t")
        except StopIteration:
            raise Exception("Possible bad request: %s"%line.strip())

        # import the data into an array
        columns = [[] for each in headers]

        # split rows into columns
        self.record_count = 0
        for line in self.fp:
            data = line.strip("\r\n").split("\t")
            for column,datum in zip(columns,data):
                column.append(datum)
            self.record_count += 1

        self.fp.close()

        # fix up columns
        for i in range(len(headers)):
            if specs[i][-1] == 'n':
                columns[i] = np.array(list(map(self.float_or_nan,columns[i])))
            elif specs[i][-1] == 'd':
                # dates get mapped to days since AD 0
                columns[i] = np.array(list(map(self.parse_date,columns[i])))
            elif specs[i][-1] == 's':
                columns[i] = np.array(columns[i],dtype=object)

            if self.aggregate:
                # check for single-valued lists -
                # The logic here is a bit odd - since we often get columns
                # that have the same value (e.g. station_id) for every record,
                # it's convenient to detect that and replace them with a constant.
                # however, if there is just one record, it screws up code that
                # isn't expecting a single value...  
                if self.record_count > 1:
                    try:
                        for elt in columns[i]:
                            if elt != columns[i][0]:
                                raise StopIteration
                        columns[i] = columns[i][0]
                    except StopIteration:
                        pass

        # merge the columns into a hash
        self.compiled={}
        for header,column in zip(headers,columns):
            # main entry for given header title
            self.compiled[header] = column
            # other entries
            for syn in dd_to_synonyms(header):
                self.compiled[syn] = column

    # Dict interface:
    def __getitem__(self,key):
        if type(key) == tuple:
            # split into label and index
            if type(key[0]) == str:
                label,idx = key
            else:
                idx,label = key
            val = self.compiled[label]
            if type(val) in (list,ndarray):
                val = val[idx]
            return val
        else:
            return self.compiled[key]
    def keys(self):
        return list(self.compiled.keys())

    # Automatic removal of missing data:
    def series(self,*keys):
        """ return a tuple of vectors for the given keys,
        but only when all values have valid data
        """
        columns = [self.compiled[key] for key in keys]
        # create a mask for each:
        masks = [1-isnan(column) for column in columns]
        valid = reduce(lambda x,y: x&y, masks)

        return [compress(valid,column) for column in columns]




def rdb_to_dataset(filename=None,text=None,to_utc=True):
    """
    Read an rdb file and return an xarray dataset.
    if to_utc is set, look for a tz_cd attribute, and adjust times
    to UTC if tz_cd is present.

    If no data was found, return None
    """

    if filename is not None:
        usgs_data=Rdb(source_file=filename)
    else:
        usgs_data=Rdb(text=text)

    if len(usgs_data['datetime'])==0:
        return None
    
    # Convert that xarray for consistency
    ds=xr.Dataset()
    
    ds['time']=( ('time',), utils.to_dt64(usgs_data['datetime']) )
    ds['datenum']=( ('time',), usgs_data['datetime'])

    ds.attrs['preamble']=usgs_data.preamble

    for key in usgs_data.keys():
        if key=='datetime':
            continue
        data=usgs_data[key]

        # attempt to find better name for the columns
        varname=key # default to original name

        # first number is timeseries code, second is parameter,
        # and third, optional, is statistic code
        m=re.match(r'(\d+)_(\d+)(_(\d+))?(_cd)?$',key)
        meta={}
        parameter=None
        if m:
            meta['ts_code']=m.group(1)
            meta['parm_code']=m.group(2)
            if m.group(4):
                meta['stat_code']=m.group(4)
            # print("ts: %s  parm: %s  stat: %s"%(meta['ts_code'],
            #                                     meta['parm_code'],
            #                                     meta['stat_code']))
            parameter=rdb_codes.parm_code_lookup(meta['parm_code'])
            if parameter is not None:
                # parm_nm is often really long!
                # srsname, when present, is shorter
                # not great -- srsname is sometimes misleading
                # like 'stream_flow_mean_daily' when really it's instantaneous
                # tidal flow.
                srsname=parameter['srsname']
                varname=None
                if srsname:
                    # sometimes this is nan, though!
                    try:
                        varname=srsname.lower() # force string-like check
                    except AttributeError:
                        pass
                if varname is None:
                    varname=parameter['parameter_nm']
                varname=varname.lower().replace(' ','_').replace(',','').replace('.','')

                meta['units']=parameter['parameter_units']

                # But possible that one station has multiple instances of the same
                # parameter.  In this case,
                count=0
                base_varname=varname
                while varname in ds:
                    count+=1
                    varname=base_varname+"_%02d"%count

            if m.group(4):
                statistic=rdb_codes.stat_code_lookup(meta['stat_code'])
                if statistic is not None:
                    meta['statistic']=statistic['name']

            if m.group(5) is not None:
                # TODO: save QA codes
                continue

        if (not isinstance(data, np.ndarray)) and (parameter is not None):
            # In the past, if it had no dimension, I assumed it was
            # an attribute.  But maybe it's better to see whether
            # it was found as a parameter.  In that case, it's
            # probably a real data point but was collapsed by
            # the rdb code because it had no variation.
            # Depending, this may need to be smarter about datatype
            data=data*np.ones(ds.dims['time'])
            
        if isinstance(data, np.ndarray):
            if len(data)==len(ds.time):
                ds[varname] = ('time',), data
            else:
                print("What to do with %s"%key)

            for k in meta:
                ds[varname].attrs[k]=meta[k]
        else:
            # probably should be an attribute
            ds.attrs[varname]=data

    # if there is a tz_cd attribute, use that to get timestamps
    # back to UTC.
    if 'tz_cd' in usgs_data.keys() and to_utc:
        # tz_cd, in a sample size of 1, is something like PST.
        tz_target=pytz.utc
        #if timezone changes, tz_src is an array of strings

        def tz_to_offset(tz_src):
            if tz_src == 'PST':
                return -8
            elif tz_src=='PDT':
                return -7
            elif tz_src=='EST':
                return -5
            elif tz_src=='EDT':
                return -4
            else:
                raise Exception("Not sure how to interpret time zone %s"%tz_src)

        if isinstance(usgs_data['tz_cd'],np.ndarray):
            offset_hours=np.array([tz_to_offset(tz_src) for tz_src in usgs_data['tz_cd']])
            ds.attrs['tz_cd_original']=",".join( np.unique(ds.tz_cd) )
        else:
            offset_hours=tz_to_offset(usgs_data['tz_cd'])
            ds.attrs['tz_cd_original']=ds.tz_cd

        tz_src=usgs_data['tz_cd']
        # assign creates a *new* dataset. Be sure to get the result.
        ds=ds.assign(time=lambda x: x.time - offset_hours * np.timedelta64(1,'h'))
        ds=ds.assign(datenum=lambda x: x.datenum - offset_hours/24.)
        
        ds.attrs['tz_cd']='UTC'

    return ds

